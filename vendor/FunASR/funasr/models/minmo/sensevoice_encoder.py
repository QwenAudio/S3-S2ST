import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Iterable, Optional

from funasr.register import tables
from funasr.models.transformer.utils.nets_utils import make_pad_mask


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(
            self.inv_freq
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttentionFSMNSdpaRoPE(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()

        self.attention_heads = attention_heads
        self.query = Linear(linear_units, linear_units)
        self.key = Linear(linear_units, linear_units, bias=False)
        self.value = Linear(linear_units, linear_units)
        self.out = Linear(linear_units, linear_units)
        self.rotary_emb = RotaryEmbedding(
            linear_units // attention_heads,
            max_position_embeddings=kwargs.get("max_position_embeddings", 2048),
            base=kwargs.get("rope_theta", 10000),
        )

        self.fsmn_block = nn.Conv1d(
            linear_units,
            linear_units,
            kwargs.get("kernel_size", 15),
            stride=1,
            padding=0,
            groups=linear_units,
            bias=False,
        )
        # padding
        left_padding = (kwargs.get("kernel_size", 15) - 1) // 2
        left_padding = left_padding + kwargs.get("sanm_shfit", 0)
        right_padding = kwargs.get("kernel_size", 15) - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)
        self.dropout = torch.nn.Dropout(kwargs.get("dropout_rate", 0.0))

    def fsmn(self, inputs, mask):
        b, t, d = inputs.size()  # b, t, d
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))  # b, t, 1
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2) + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        memory = self.fsmn(v, mask=mask)

        wv, qk = self.qkv_attention(q, k, v, mask, **kwargs)
        return self.out(wv) + memory, qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):
        is_causal = kwargs.get("is_causal", False)
        b, t, d = q.shape
        scale = (d // self.attention_heads) ** -0.5
        q = q.view(*q.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.attention_heads, -1).permute(0, 2, 1, 3)

        position_ids = kwargs.get("position_ids", None)
        kv_seq_len = v.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if mask is not None:
            mask = mask.unsqueeze(1).to(torch.bool)  # (batch, 1, 1, t)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )
        if mask is not None:
            attn_output = attn_output.masked_fill(mask.transpose(2, 3).logical_not(), 0.0)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(start_dim=2)
        return attn_output, None


att_type_dict = {
    "self_att_fsmn_sdpa": MultiHeadAttentionFSMNSdpaRoPE,
}


class EncoderLayerSANMLarge(nn.Module):
    def __init__(self, linear_units: int, attention_heads: int, **kwargs):
        super().__init__()

        att_type = kwargs.get("att_type", "self_att_fsmn_sdpa")
        self.attn = att_type_dict[att_type](linear_units, attention_heads, **kwargs)
        self.attn_ln = LayerNorm(linear_units)

        n_mlp = linear_units * 4
        self.mlp = nn.Sequential(
            Linear(linear_units, n_mlp), nn.GELU(), Linear(n_mlp, linear_units)
        )
        self.mlp_ln = LayerNorm(linear_units)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ):

        x = x + self.attn(self.attn_ln(x), mask=mask, **kwargs)[0]

        x = x + self.mlp(self.mlp_ln(x))
        return x


@tables.register("encoder_classes", "SenseVoiceLargeEncoder")
class SenseVoiceLargeEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        linear_units: int,
        attention_heads: int,
        num_blocks: int,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = Conv1d(input_size, linear_units, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv1d(linear_units, linear_units, kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList(
            [
                EncoderLayerSANMLarge(linear_units, attention_heads, **kwargs)
                for _ in range(num_blocks)
            ]
        )
        self.ln_post = LayerNorm(linear_units)
        self.use_padmask = kwargs.get("use_padmask", True)
        self.downsample_rate = kwargs.get("downsample_rate", 4)
        self.linear_units = linear_units
        self.input_size = input_size

        from funasr.utils.torch_function import sequence_mask
        self.make_pad_mask = sequence_mask(flip=False)

    def output_size(self):
        return self.linear_units

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor = None,
        **kwargs,
    ):
        use_padmask = self.use_padmask
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = n_frames

        if ilens is not None:
            if self.downsample_rate == 4:
                olens = (
                    1
                    + (ilens - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0])
                    // self.conv1.stride[0]
                )
            else:
                olens = ilens
            olens = (
                1
                + (olens - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0])
                // self.conv2.stride[0]
            )
            # olens = torch.clamp(olens, max=max_pos)
            olens[olens > max_pos] = max_pos
        else:
            olens = None

        if use_padmask and olens is not None:
            # padding_mask = (~make_pad_mask(olens)[:, None, :]).to(torch.bool).to(x.device)
            padding_mask = self.make_pad_mask(olens)[:, None, :].to(torch.bool).to(x.device)
        else:
            padding_mask = None

        device = x.device
        seq_length = x.shape[1]
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        for layer, block in enumerate(self.blocks):
            x = block(x, mask=padding_mask, position_ids=position_ids)

        x = self.ln_post(x)

        if ilens is None:
            return x
        else:
            return x, olens



@tables.register("encoder_classes", "SenseVoiceLargeEncoderExport")
class SenseVoiceLargeEncoderExport(nn.Module):
    def __init__(
        self,
        model,
        model_name="encoder",
        **kwargs,
    ):
        super().__init__()
        self.model = model
        from funasr.utils.torch_function import sequence_mask
        self.make_pad_mask = sequence_mask(flip=False)


    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]
        mask_4d_bhlt = mask_4d_bhlt * -10000.0

        return mask_3d_btd, mask_4d_bhlt

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor = None,
    ):
        use_padmask = self.model.use_padmask
        x = F.gelu(self.model.conv1(x))
        x = F.gelu(self.model.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = n_frames

        if ilens is not None:
            if self.model.downsample_rate == 4:
                olens = (
                    1
                    + (ilens - self.model.conv1.kernel_size[0] + 2 * self.model.conv1.padding[0])
                    // self.model.conv1.stride[0]
                )
            else:
                olens = ilens
            olens = (
                1
                + (olens - self.model.conv2.kernel_size[0] + 2 * self.model.conv2.padding[0])
                // self.model.conv2.stride[0]
            )
            # olens = torch.clamp(olens, max=max_pos)
            olens[olens > max_pos] = max_pos
        else:
            olens = None

        if use_padmask and olens is not None:
            # padding_mask = (~make_pad_mask(olens)[:, None, :]).to(torch.bool).to(x.device)
            padding_mask = self.make_pad_mask(olens)[:, None, :].to(torch.bool).to(x.device)
        else:
            padding_mask = None

        device = x.device
        seq_length = x.shape[1]
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        for layer, block in enumerate(self.model.blocks):
            x = block(x, mask=padding_mask, position_ids=position_ids)

        x = self.model.ln_post(x)

        return x, olens


    def get_output_size(self):
        return self.model.linear_units

    def get_dummy_inputs(self, **kwargs):

        if kwargs.get("data_in", None)[0] is not None:
            from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
            data_src = load_audio_text_image_video(kwargs.get("data_in", None), fs=kwargs["frontend"].fs)

            x, ilens = extract_fbank(
                data_src,
                data_type="sound",
                frontend=kwargs["frontend"],
                is_final=True,
            )  # speech: [b, T, d]
        else:
            x = torch.randn(1, 40, self.model.input_size)
            ilens = torch.tensor([40])
        return x.to(kwargs.get("device")), ilens.to(kwargs.get("device"))

    def get_input_names(self):
        return ["x", "ilens"]

    def get_output_names(self):
        return ["encoder_out", "olens"]

    def get_dynamic_axes(self):
        return {
            "x": {2: "x_length"},
            "ilens": {0: "batch_size"},
            "encoder_out": {1: "enc_out_length"},
            "olens": {0: "batch_size"},
        }
