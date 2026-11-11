import math
import typing as tp
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import yaml
from torch import Tensor, nn, pow, sin
from torch.nn import Parameter
from torch.nn import functional as F


@dataclass
class ModelArgs:
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    dim: int = 512
    intermediate_size: int = 1536
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    channels_first: bool = True
    pos_embed_type: str = "rope"
    max_relative_position: int = 128

    def find_multiple(self, n: int, k: int) -> int:
        if n % k == 0:
            return n
        return n + k - (n % k)

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = self.find_multiple(n_hidden, 256)
        assert self.pos_embed_type in [
            "rope",
            "conformer",
        ], "pos_embed_type must be either 'rope' or 'conformer'"


class Transformer(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if config.pos_embed_type == "rope":
            freqs_cis = precompute_freqs_cis(self.config.block_size,
                                             self.config.head_dim,
                                             self.config.rope_base)
            self.register_buffer("freqs_cis", freqs_cis)
        else:
            self.register_buffer("freqs_cis", None)

        causal_mask = torch.tril(
            torch.ones(self.config.block_size,
                       self.config.block_size,
                       dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask)

        self.max_batch_size = -1
        self.max_seq_length = -1
        self.use_kv_cache = False

    def forward(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.config.pos_embed_type == "rope":
            assert (
                self.freqs_cis is not None
            ), "RoPE frequencies must be initialized for RoPE positional embedding"
            freqs_cis = self.freqs_cis[input_pos]
        else:
            freqs_cis = None

        if mask is None:
            if not self.training and self.use_kv_cache:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., :input_pos.max() + 1]
            else:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., input_pos]

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_layer_scale = LayerScale(config.dim, inplace=True)
        self.ffn_layer_scale = LayerScale(config.dim, inplace=True)

    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
    ) -> Tensor:
        h = x + self.attention_layer_scale(
            self.attention(self.attention_norm(x), freqs_cis, mask, input_pos))
        out = h + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(h)))
        return out


class Attention(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head +
                          2 * config.n_local_heads) * config.head_dim

        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head,
                            config.dim,
                            bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.attn_dropout_rate = config.attn_dropout_rate
        self.pos_embed_type = config.pos_embed_type

        if self.pos_embed_type == "conformer":
            self.max_relative_position = config.max_relative_position
            num_pos_embeddings = 2 * config.max_relative_position + 1
            self.rel_pos_embeddings = nn.Parameter(
                torch.zeros(num_pos_embeddings, self.head_dim))
            nn.init.normal_(self.rel_pos_embeddings, mean=0.0, std=0.02)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
        context_seqlen = seqlen

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        if self.pos_embed_type == "rope":
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout_rate if self.training else 0.0,
            attn_mask=mask,
        )

        y = (y.transpose(1, 2).contiguous().view(bsz, seqlen,
                                                 self.head_dim * self.n_head))
        y = self.wo(y)
        return y


class FeedForward(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(
            torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerScale(nn.Module):

    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-2,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class WindowLimitedTransformer(Transformer):

    def __init__(
        self,
        config: ModelArgs,
        input_dim: int = 512,
        window_size: Optional[int] = None,
        look_ahead_conv: nn.Module = None,
    ):
        super().__init__(config)
        self.window_size = window_size

        self.channels_first = config.channels_first
        self.look_ahead_conv = (look_ahead_conv if look_ahead_conv is not None
                                else nn.Identity())
        self.input_proj = (nn.Linear(input_dim, config.dim)
                           if input_dim != config.dim else nn.Identity())
        self.output_proj = (nn.Linear(config.dim, input_dim)
                            if input_dim != config.dim else nn.Identity())

    def make_window_limited_mask(
        self,
        max_length: int,
        x_lens: Optional[Tensor] = None,
    ) -> Tensor:
        mask = torch.tril(torch.ones(max_length, max_length))
        row_indices = torch.arange(max_length).view(-1, 1)
        window_size = self.window_size or max_length
        valid_range = (row_indices - window_size + 1).clamp(min=0)
        column_indices = torch.arange(max_length)
        mask = (column_indices >= valid_range) & mask.bool()

        mask = mask.bool()[None, None]
        return mask

    def forward(
        self,
        x: Tensor,
        x_lens: Optional[Tensor] = None,
    ) -> Tensor:
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.look_ahead_conv(x)
        input_pos = torch.arange(x.shape[1], device=x.device)

        max_length = x.shape[1]
        if self.window_size is not None:
            mask = self.make_window_limited_mask(max_length, x_lens)
        else:
            mask = self.make_mask(max_length, x_lens)
        mask = mask.to(x.device)
        x = super().forward(x, input_pos, mask)
        x = self.output_proj(x)
        if self.channels_first:
            x = x.transpose(1, 2)
        return x


def precompute_freqs_cis(seq_len: int,
                         n_elem: int,
                         base: int = 10000,
                         dtype: torch.dtype = torch.float32) -> Tensor:
    freqs = 1.0 / (base**(torch.arange(0, n_elem, 2)[:(n_elem // 2)].float() /
                          n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] -
            xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] +
            xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left,
                                                      padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(x: torch.Tensor,
                                 kernel_size: int,
                                 stride: int,
                                 padding_total: int = 0) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size -
                                                         padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zeros",
    value: float = 0.0,
):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left,
                                                      padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class CausalConvNet(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
        padding=None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def forward(self, x):
        pad = self.padding
        extra_padding = get_extra_padding_for_conv1d(x, self.kernel_size,
                                                     self.stride, pad)
        x = pad1d(x, (pad, extra_padding), mode="constant", value=0)
        return self.conv(x).contiguous()


class CausalTransConvNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 stride=1,
                 padding=None):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride,
                                       dilation=dilation)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()


class ConvNeXtBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        self.dwconv = CausalConvNet(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True)
                      if layer_scale_init_value > 0 else None)

    def forward(self, x, apply_residual: bool = True):
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)

        if apply_residual:
            x = input + x

        return x


class Snake1d(nn.Module):

    def __init__(self,
                 in_features,
                 alpha=1.0,
                 alpha_trainable=True,
                 alpha_logscale=True):
        super().__init__()
        self.in_features = in_features

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class ResidualUnit(nn.Module):

    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.ModuleList([
            Snake1d(dim),
            CausalConvNet(dim,
                          dim,
                          kernel_size=7,
                          dilation=dilation,
                          padding=pad),
            Snake1d(dim),
            CausalConvNet(dim, dim, kernel_size=1),
        ])
        self.causal = causal

    def forward(self, x):

        x_ori = x
        for m in self.block:
            x = m(x)
        pad = x_ori.shape[-1] - x.shape[-1]
        if pad > 0:
            if self.causal:
                x = x[..., :-pad]
            else:
                x = x[..., pad // 2:-pad // 2]
        return x + x_ori


class DecoderBlock(nn.Module):

    def __init__(self,
                 input_dim: int = 16,
                 output_dim: int = 8,
                 stride: int = 1,
                 causal: bool = False):
        super().__init__()

        self.block = nn.ModuleList([
            Snake1d(input_dim),
            CausalTransConvNet(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1, causal=causal),
            ResidualUnit(output_dim, dilation=3, causal=causal),
            ResidualUnit(output_dim, dilation=9, causal=causal),
        ])

    def forward(self, x):
        for m in self.block:
            x = m(x)
        return x


class Decoder(nn.Module):

    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        causal: bool = False,
        n_transformer_layers: list | None = None,
    ):
        super().__init__()
        layers = [
            CausalConvNet(input_channel, channels, kernel_size=7, padding=3)
        ]

        if n_transformer_layers is None:
            n_transformer_layers = [0, 0, 0, 0]

        # Add upsampling + MRF blocks
        for i, (stride,
                n_t_layer) in enumerate(zip(rates, n_transformer_layers)):
            input_dim = channels // 2**i
            output_dim = channels // 2**(i + 1)
            layers += [
                DecoderBlock(input_dim, output_dim, stride, causal=causal)
            ]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            CausalConvNet(output_dim, d_out, kernel_size=7, padding=3),
        ]

        self.model = nn.ModuleList(layers)

    def forward(self, x):
        for m in self.model:
            x = m(x)
        return torch.clamp(x, min=-1, max=1)


class DAC(nn.Module):

    def __init__(
        self,
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] | None = None,
        pre_transformer: torch.nn.Module = None,
        causal: bool = True,
        decoder_transformer_layers: List[int] | None = None,
        transformer_general_config=None,
        codebook_size=2048,
        codebook_nums=16,
    ):
        super().__init__()

        if decoder_rates is None:
            decoder_rates = [8, 8, 4, 2]

        if decoder_transformer_layers is None:
            decoder_transformer_layers = [0, 0, 0, 0]

        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.decoder_hop = np.prod(decoder_rates)
        self.latent_dim = latent_dim

        self.code_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, latent_dim)
            for _ in range(codebook_nums)
        ])
        self.pre_transformer = pre_transformer
        self.upsample = nn.Sequential(*[
            nn.Sequential(
                CausalTransConvNet(
                    latent_dim,
                    latent_dim,
                    kernel_size=factor,
                    stride=factor,
                ),
                ConvNeXtBlock(dim=latent_dim),
            ) for _, factor in enumerate([2, 2])
        ])

        self.upsample = nn.ModuleList([
            nn.ModuleList([
                CausalTransConvNet(
                    latent_dim,
                    latent_dim,
                    kernel_size=factor,
                    stride=factor,
                ),
                ConvNeXtBlock(dim=latent_dim),
            ]) for idx, factor in reversed(list(enumerate([2, 2])))
        ])
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            causal=causal,
            n_transformer_layers=decoder_transformer_layers,
        )

    @torch.inference_mode()
    @torch.compile
    def decode(self, codes: torch.Tensor):
        code_list = codes.unbind(dim=-1)
        code_embs = [
            emb(code) for emb, code in zip(self.code_embeddings, code_list)
        ]

        code_embs = torch.stack(code_embs, dim=0).mean(dim=0)

        x = self.pre_transformer(code_embs.transpose(1, 2))
        for block in self.upsample:
            for m in block:
                x = m(x)
        rec = self.decoder(x)

        return rec


class HParams:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def hparams_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return HParams(**fields)


def get_hparams_from_file(config_path):
    yaml.add_constructor('utils.utils.HParams', hparams_constructor)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    hparams = HParams(**config)
    return hparams


def load_qwen3_omni_dac_from_config(config_path=None,
                                    checkpoint_path=None,
                                    device="cuda",
                                    dtype=torch.bfloat16):

    hps = get_hparams_from_file(config_path).model

    tranformer_config_quantizer = ModelArgs(
        block_size=hps.post_module.args.block_size,
        n_layer=hps.post_module.args.n_layer,
        n_head=hps.post_module.args.n_head,
        dim=hps.post_module.args.dim,
        intermediate_size=hps.post_module.args.intermediate_size,
        n_local_heads=hps.post_module.args.n_local_heads,
        head_dim=hps.post_module.args.head_dim,
        rope_base=hps.post_module.args.rope_base,
        norm_eps=hps.post_module.args.norm_eps,
        dropout_rate=hps.post_module.args.dropout_rate,
        attn_dropout_rate=hps.post_module.args.attn_dropout_rate,
        channels_first=hps.post_module.args.channels_first,
    )

    post_module = WindowLimitedTransformer(
        config=tranformer_config_quantizer,
        input_dim=hps.post_module.input_dim,
        window_size=hps.post_module.window_size,
    )

    model = DAC(latent_dim=hps.dac.latet_dim,
                decoder_dim=hps.dac.decoder_dim,
                decoder_rates=hps.dac.decoder_rates,
                pre_transformer=post_module,
                causal=hps.dac.causal,
                decoder_transformer_layers=hps.dac.decoder_transformer_layers,
                transformer_general_config=None,
                codebook_nums=hps.dac.codebook_nums,
                codebook_size=hps.dac.codebook_size)

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path,
                                map_location=device,
                                mmap=True,
                                weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        if any("generator" in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items() if "generator." in k
            }

        result = model.load_state_dict(state_dict, strict=False, assign=True)
        print(f"Loaded model: {result}")

    model.eval()
    model.to(device)
    model.to(dtype)

    return model
