import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import ConformerBlock
from diffusers.models.activations import get_activation
from einops import pack, rearrange, repeat
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from .transformer import BasicTransformerBlock
from funasr.utils.hinter import hint_once
from funasr.models.transformer.utils.mask import subsequent_mask, causal_block_mask
import time
import logging


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PaddedConv1d(torch.nn.Conv1d):
    def __init__(
            self, in_channels, out_channels,
            kernel_size, stride: int = 1,
            padding: tuple = (0, 0), dilation: int = 1,
            groups: int = 1, bias: bool = True,
            padding_mode: str = 'zeros',
            device=None, dtype=None
    ):
        super(PaddedConv1d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride,
            padding=0, dilation=dilation,
            groups=groups, bias=bias,
            padding_mode=padding_mode,
            device=device, dtype=dtype
        )
        self.custom_padding = padding

    def forward(self, x):
        x = F.pad(x, self.custom_padding)
        x = super(PaddedConv1d, self).forward(x)
        return x


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True, device=None, dtype=None, channel_first=False):
        super(LayerNorm, self).__init__(dim, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)
        self.channel_first = channel_first

    def forward(self, x):
        if self.channel_first:
            x = x.transpose(1, 2)
        x = super(LayerNorm, self).forward(x)
        if self.channel_first:
            x = x.transpose(1, 2)
        return x


class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8, causal=False, use_layer_norm=None):
        super().__init__()
        if use_layer_norm is None:
            use_layer_norm = causal
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1) if not causal else
            PaddedConv1d(dim, dim_out, 3, padding=(2, 0)),

            torch.nn.GroupNorm(groups, dim_out) if not use_layer_norm else
            LayerNorm(dim_out, channel_first=True),

            nn.Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8, causal=False, use_layer_norm=None):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block1D(dim, dim_out, groups=groups, causal=causal, use_layer_norm=use_layer_norm)
        self.block2 = Block1D(dim_out, dim_out, groups=groups, causal=causal, use_layer_norm=use_layer_norm)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Downsample1D(nn.Module):
    def __init__(self, dim, channel_first=True, padding=1):
        super().__init__()
        self.channel_first = channel_first
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, padding)

    def forward(self, x):
        if not self.channel_first:
            x = x.transpose(1, 2).contiguous()

        out = self.conv(x)

        if not self.channel_first:
            out = out.transpose(1, 2).contiguous()
        return out


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=True,
                 out_channels=None, name="conv", channel_first=True, stride=2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.channel_first = channel_first
        self.stride = stride

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, stride*2, stride, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        if not self.channel_first:
            inputs = inputs.transpose(1, 2).contiguous()
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=self.stride, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        if not self.channel_first:
            outputs = outputs.transpose(1, 2).contiguous()
        return outputs


class ConformerWrapper(ConformerBlock):
    def __init__(  # pylint: disable=useless-super-delegation
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0,
        ff_dropout=0,
        conv_dropout=0,
        conv_causal=False,
    ):
        super().__init__(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            conv_causal=conv_causal,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        **kwargs
    ):
        return super().forward(x=hidden_states, mask=attention_mask.bool())


class TransformerDecoderWrapper(nn.Module):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int = 8,
            attention_head_dim: int = 64,
            dropout: float = 0.1,
            activation_fn: str = "snakebeta",
            cond_dim: int = 80,
            concat_after: bool = False,
    ):
        super().__init__()
        attn_dim = num_attention_heads * attention_head_dim
        self.input_proj = torch.nn.Linear(dim, attn_dim)
        self.cond_proj = torch.nn.Linear(cond_dim, attn_dim)
        self.output_proj = torch.nn.Linear(attn_dim, dim)
        from funasr.models.transformer.decoder import (
            DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward
        )
        self.decoder_layer = DecoderLayer(
            attn_dim,
            MultiHeadedAttention(
                num_attention_heads, attn_dim, dropout
            ),
            MultiHeadedAttention(
                num_attention_heads, attn_dim, dropout
            ),
            PositionwiseFeedForward(attn_dim, attn_dim * 4, dropout),
            dropout,
            normalize_before=True,
            concat_after=concat_after,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            timestep: torch.Tensor=None,
            prompt: torch.Tensor = None,
            prompt_lengths: torch.Tensor = None,
            **kwargs
    ):
        # make masks and forward attention layer
        x_mask = attention_mask[:, :1].transpose(1, 2).contiguous()
        prompt_mask = (~make_pad_mask(prompt_lengths)[:, :, None]).to(hidden_states.device)
        x = self.input_proj(hidden_states) * x_mask
        prompt = self.cond_proj(prompt) * prompt_mask
        x = self.decoder_layer(x, x_mask.transpose(1, 2).contiguous(), prompt, prompt_mask.transpose(1, 2).contiguous())[0]
        x = self.output_proj(x) * x_mask
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
        conditions: dict = None,
        concat_after: bool = False,
        **kwargs
    ):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        super().__init__()
        channels = tuple(channels)
        self.channels = channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conditions = conditions
        self.pos_dropout_rate = kwargs.get("pos_dropout_rate", 0.0)
        self.attention_head_dim = attention_head_dim
        self.num_heads = num_heads
        self.attn_dim = attention_head_dim * num_heads
        self.add_pos_to_blocks = kwargs.get("add_pos_to_blocks", [])
        self.positional_encoding = self.get_positional_encoding(
            kwargs.get("positional_encoding", None)
        )
        self.causal = causal = kwargs.get("causal", False)

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        self.time_embed_dim = time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )
        self.add_text_to_time_embed = kwargs.get("add_text_to_time_embed", False)
        if self.add_text_to_time_embed:
            from diffusers.models.embeddings import TextTimeEmbedding
            self.text_time_embed = TextTimeEmbedding(
                encoder_dim=self.conditions["prompt"]["dim"],
                time_embed_dim=time_embed_dim,
                num_heads=num_heads * 4,  # using many heads to compress text sequence information
            )
        self.down_block_type = down_block_type
        self.mid_block_type = mid_block_type
        self.up_block_type = up_block_type

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            if conditions is not None and "xvec" in conditions:
                input_channel = input_channel + conditions["xvec"]["dim"]
            resnet = ResnetBlock1D(
                dim=input_channel, dim_out=output_channel,
                time_emb_dim=time_embed_dim, causal=causal
            )
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        down_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        concat_after=concat_after,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else (
                    nn.Conv1d(output_channel, output_channel, 3, padding=1) if not causal else
                    PaddedConv1d(output_channel, output_channel, 3, padding=(2, 0))
                )
            )
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]

            if conditions is not None and "xvec" in conditions:
                input_channel = input_channel + conditions["xvec"]["dim"]
            resnet = ResnetBlock1D(
                dim=input_channel, dim_out=output_channel,
                time_emb_dim=time_embed_dim, causal=causal
            )

            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        concat_after=concat_after,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2

            if conditions is not None and "xvec" in conditions:
                input_channel = input_channel + conditions["xvec"]["dim"]
            resnet = ResnetBlock1D(
                dim=input_channel, dim_out=output_channel,
                time_emb_dim=time_embed_dim, causal=causal
            )
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        up_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        concat_after=concat_after,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True) if not is_last else (
                    nn.Conv1d(output_channel, output_channel, 3, padding=1) if not causal else
                    PaddedConv1d(output_channel, output_channel, 3, padding=(2, 0))
                )
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1], causal=causal)
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

    def get_positional_encoding(self, pe_type: Optional[str] = None):
        if pe_type is None:
            return None
        elif pe_type == "abs_pos":
            from funasr.models.transformer.embedding import PositionalEncoding
            return PositionalEncoding(self.channels[-1], dropout_rate=self.pos_dropout_rate)
        elif pe_type == "scaled_abs_pos":
            from funasr.models.transformer.embedding import ScaledPositionalEncoding
            return ScaledPositionalEncoding(self.channels[-1], dropout_rate=self.pos_dropout_rate)
        elif pe_type == "learnable_fourier_pos":
            from funasr.models.transformer.embedding import LearnableFourierPosEnc
            return LearnableFourierPosEnc(self.channels[-1], dropout_rate=self.pos_dropout_rate)
        else:
            raise NotImplementedError(f"Unknown positional encoding type {pe_type}")

    def get_block(self, block_type, dim, attention_head_dim, num_heads, dropout, act_fn, **kwargs):
        if block_type == "conformer":
            block = ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )
        elif block_type == "transformer":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
        elif block_type == "transformer_decoder":
            block = TransformerDecoderWrapper(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
                cond_dim=self.conditions["prompt"]["dim"],
                concat_after=kwargs.get("concat_after", False)
            )
        elif block_type == "diffuser_transformer_decoder":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                cross_attention_dim=self.conditions["prompt"]["dim"],
                activation_fn=act_fn,
                norm_type="ada_norm",
                num_embeds_ada_norm=self.time_embed_dim,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")

        return block

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None, return_hidden=False):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        t = self.time_embeddings(t)
        t = self.time_mlp(t)
        if self.add_text_to_time_embed:
            text_emb = self.text_time_embed(cond["prompt"][0])
            t = t + text_emb
            hint_once("add text embedding to time embedding", "add_text_to_time_embed")

        if self.positional_encoding is not None:
            cond["prompt"][0] = self.positional_encoding(cond["prompt"][0])
            hint_once("add positional encoding to text", "add_pos_enc_to_text")

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]

        hiddens = []
        masks = [mask]
        for i, (resnet, transformer_blocks, downsample) in enumerate(self.down_blocks):
            mask_down = masks[-1]
            if (cond is not None and "xvec" in cond and
                self.conditions is not None and "xvec" in self.conditions):
                xvec = repeat(cond["xvec"], "b c -> b c t", t=x.shape[-1])
                x = pack([x, xvec], "b * t")[0]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if self.positional_encoding is not None and "down_block" in self.add_pos_to_blocks and i == 0:
                x = self.positional_encoding(x)
                hint_once(f"add pos encoding to down block {i}", f"add_pos_enc_to_down_block_{i}")
            if self.down_block_type == "transformer":
                attn_mask = torch.matmul(mask_down.transpose(1, 2).contiguous(), mask_down)
                if self.causal:
                    hint_once("use causal attention mask for down blocks", "down causal mask")
                    ilen, device = mask_down.shape[-1], mask_down.device
                    causal_mask = torch.tril(torch.ones((1, ilen, ilen), device=device)).to(torch.bool)
                    attn_mask = attn_mask * causal_mask
            elif self.down_block_type in ["diffuser_transformer_decoder", "ditto_transformer_decoder"]:
                attn_mask = torch.matmul(mask_down.transpose(1, 2).contiguous(), mask_down)
                enc_mask = (~make_pad_mask(cond["prompt"][1])).to(attn_mask).unsqueeze(1)
                cross_attn_mask = torch.matmul(mask_down.transpose(1, 2).contiguous(), enc_mask)
            elif self.down_block_type == "transformer_decoder":
                attn_mask = mask_down
            else:
                attn_mask = mask_down.squeeze(1)
            for transformer_block in transformer_blocks:
                cond_kwargs = {}
                if self.down_block_type == "transformer_decoder":
                    cond_kwargs["prompt"], cond_kwargs["prompt_lengths"] = cond["prompt"]
                elif self.down_block_type == "diffuser_transformer_decoder":
                    cond_kwargs["encoder_hidden_states"] = cond["prompt"][0]
                    cond_kwargs["encoder_attention_mask"] = cross_attn_mask
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                    **cond_kwargs
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            # mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for i, (resnet, transformer_blocks) in enumerate(self.mid_blocks):
            if (cond is not None and "xvec" in cond and
                self.conditions is not None and "xvec" in self.conditions):
                xvec = repeat(cond["xvec"], "b c -> b c t", t=x.shape[-1])
                x = pack([x, xvec], "b * t")[0]
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if self.positional_encoding is not None and "mid_block" in self.add_pos_to_blocks and i == 0:
                x = self.positional_encoding(x)
                hint_once(f"add pos encoding to mid block {i}", f"add_pos_enc_to_mid_block_{i}")
            if self.mid_block_type == "transformer":
                attn_mask = torch.matmul(mask_mid.transpose(1, 2).contiguous(), mask_mid)
                if self.causal:
                    hint_once("use causal attention mask for mid blocks", "mid causal mask")
                    ilen, device = mask_mid.shape[-1], mask_mid.device
                    causal_mask = torch.tril(torch.ones((1, ilen, ilen), device=device)).to(torch.bool)
                    attn_mask = attn_mask * causal_mask
            elif self.mid_block_type == "transformer_decoder":
                attn_mask = mask_mid
            elif self.mid_block_type in ["diffuser_transformer_decoder", "ditto_transformer_decoder"]:
                attn_mask = torch.matmul(mask_mid.transpose(1, 2).contiguous(), mask_mid)
                enc_mask = (~make_pad_mask(cond["prompt"][1])).to(attn_mask).unsqueeze(1)
                cross_attn_mask = torch.matmul(mask_mid.transpose(1, 2).contiguous(), enc_mask)
            else:
                attn_mask = mask_mid.squeeze(1)
            for transformer_block in transformer_blocks:
                cond_kwargs = {}
                if self.mid_block_type == "transformer_decoder":
                    cond_kwargs["prompt"], cond_kwargs["prompt_lengths"] = cond["prompt"]
                elif self.mid_block_type in ["diffuser_transformer_decoder", "ditto_transformer_decoder"]:
                    cond_kwargs["encoder_hidden_states"] = cond["prompt"][0]
                    cond_kwargs["encoder_attention_mask"] = cross_attn_mask
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                    **cond_kwargs
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            # mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        for i, (resnet, transformer_blocks, upsample) in enumerate(self.up_blocks):
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            if (cond is not None and "xvec" in cond and
                self.conditions is not None and "xvec" in self.conditions):
                xvec = repeat(cond["xvec"], "b c -> b c t", t=x.shape[-1])
                x = pack([x, xvec], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if self.positional_encoding is not None and "up_block" in self.add_pos_to_blocks and i == 0:
                x = self.positional_encoding(x)
                hint_once(f"add pos encoding to up block {i}", f"add_pos_enc_to_up_block_{i}")
            if self.up_block_type == "transformer":
                attn_mask = torch.matmul(mask_up.transpose(1, 2).contiguous(), mask_up)
                if self.causal:
                    hint_once("use causal attention mask for up blocks", "up causal mask")
                    ilen, device = mask_up.shape[-1], mask_up.device
                    causal_mask = torch.tril(torch.ones((1, ilen, ilen), device=device)).to(torch.bool)
                    attn_mask = attn_mask * causal_mask
            elif self.up_block_type == "transformer_decoder":
                attn_mask = mask_up
            elif self.up_block_type in ["diffuser_transformer_decoder", "ditto_transformer_decoder"]:
                attn_mask = torch.matmul(mask_up.transpose(1, 2).contiguous(), mask_up)
                enc_mask = (~make_pad_mask(cond["prompt"][1])).to(attn_mask).unsqueeze(1)
                cross_attn_mask = torch.matmul(mask_up.transpose(1, 2).contiguous(), enc_mask)
            else:
                attn_mask = mask_up.squeeze(1)
            for transformer_block in transformer_blocks:
                cond_kwargs = {}
                if self.up_block_type == "transformer_decoder":
                    cond_kwargs["prompt"], cond_kwargs["prompt_lengths"] = cond["prompt"]
                elif self.up_block_type in ["diffuser_transformer_decoder", "ditto_transformer_decoder"]:
                    cond_kwargs["encoder_hidden_states"] = cond["prompt"][0]
                    cond_kwargs["encoder_attention_mask"] = cross_attn_mask
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                    **cond_kwargs
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            # mask_up = rearrange(mask_up, "b t -> b 1 t")
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)

        if return_hidden:
            return output * mask, None

        return output * mask


class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
        **kwargs,
    ):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal = kwargs.get("causal", False)
        self.attn_mask_type = kwargs.get("attn_mask_type", "diffusers")
        self.use_layer_norm = kwargs.get("use_layer_norm", None)
        self.causal_mask_type = kwargs.get("causal_mask_type", None)

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )
        self.down_block_type = down_block_type
        self.mid_block_type = mid_block_type
        self.up_block_type = up_block_type

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(
                dim=input_channel, dim_out=output_channel,
                time_emb_dim=time_embed_dim, causal=causal,
                use_layer_norm=self.use_layer_norm,
            )
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        down_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else (
                    nn.Conv1d(output_channel, output_channel, 3, padding=1) if not causal else
                    PaddedConv1d(output_channel, output_channel, 3, padding=(2, 0))
                )
            )
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]
            resnet = ResnetBlock1D(
                dim=input_channel, dim_out=output_channel,
                time_emb_dim=time_embed_dim, causal=causal,
                use_layer_norm=self.use_layer_norm,
            )

            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = ResnetBlock1D(
                dim=input_channel, dim_out=output_channel,
                time_emb_dim=time_embed_dim, causal=causal,
                use_layer_norm=self.use_layer_norm,
            )
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        up_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True) if not is_last else (
                    nn.Conv1d(output_channel, output_channel, 3, padding=1) if not causal else
                    PaddedConv1d(output_channel, output_channel, 3, padding=(2, 0))
                )
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1], causal=causal)
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

    def get_block(self, block_type, dim, attention_head_dim, num_heads, dropout, act_fn, **kwargs):
        if block_type == "conformer":
            block = ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )
        elif block_type == "transformer":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")

        return block

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_mix_causal_mask(self, attn_mask, rand=None, ratio=None):
        b, _, t = attn_mask.shape
        if rand is None:
            rand = torch.rand((b, 1, 1), device=attn_mask.device, dtype=torch.float32)
        mixed_mask = attn_mask.clone()
        for item in self.causal_mask_type:
            prob_min, prob_max = item["prob_min"], item["prob_max"]
            _ratio = 1
            if "ratio" in item:
                _ratio = item["ratio"]
            if ratio is not None:
                _ratio = ratio
            block_size = item["block_size"] * _ratio
            if block_size <= 0:
                causal_mask = attn_mask
            else:
                causal_mask = causal_block_mask(t, block_size, attn_mask.device, torch.float32).unsqueeze(0)
            flag = (prob_min <= rand) & (rand < prob_max)
            mixed_mask = mixed_mask * (~flag) + (causal_mask * attn_mask) * flag

        return mixed_mask

    def forward(self, x, mask, mu, t, spks=None, cond=None, return_hidden=False, mask_rand=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        t = self.time_embeddings(t).to(mu)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        conv_time, transformer_time = 0, 0
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            _time = time.time()
            x = resnet(x, mask_down, t)
            conv_time += time.time() - _time
            x = rearrange(x, "b c t -> b t c").contiguous()
            # mask_down = rearrange(mask_down, "b 1 t -> b t")
            if self.down_block_type == "transformer":
                attn_mask = torch.matmul(mask_down.transpose(1, 2).contiguous(), mask_down)
                if self.causal:
                    hint_once(f"use causal attention mask {self.attn_mask_type} for down blocks", "down causal mask")
                    if self.causal_mask_type is None:
                        ilen, device = mask_down.shape[-1], mask_down.device
                        causal_mask = torch.tril(torch.ones((1, ilen, ilen), device=device)).to(torch.bool)
                    else:
                        causal_mask = self.build_mix_causal_mask(attn_mask, mask_rand)
                    attn_mask = attn_mask * causal_mask
                    if self.attn_mask_type == "diffusers":
                        attn_mask = (1 - attn_mask) * torch.finfo(x.dtype).min
            else:
                attn_mask = mask_down.squeeze(1)
            _time = time.time()
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            transformer_time += time.time() - _time
            x = rearrange(x, "b t c -> b c t").contiguous()
            # mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x)  # Save hidden states for skip connections
            _time = time.time()
            x = downsample(x * mask_down)
            conv_time += time.time() - _time
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        if self.mid_block_type == "transformer":
            hint_once(f"use causal attention mask {self.attn_mask_type} for mid blocks", "mid causal mask")
            attn_mask = torch.matmul(mask_mid.transpose(1, 2).contiguous(), mask_mid)
            if self.causal:
                if self.causal_mask_type is None:
                    ilen, device = mask_mid.shape[-1], mask_mid.device
                    causal_mask = torch.tril(torch.ones((1, ilen, ilen), device=device)).to(torch.bool)
                else:
                    causal_mask = self.build_mix_causal_mask(attn_mask, mask_rand)
                attn_mask = attn_mask * causal_mask
                if self.attn_mask_type == "diffusers":
                    attn_mask = (1 - attn_mask) * torch.finfo(x.dtype).min
        else:
            attn_mask = mask_mid.squeeze(1)
        for resnet, transformer_blocks in self.mid_blocks:
            _time = time.time()
            x = resnet(x, mask_mid, t)
            conv_time += time.time() - _time
            x = rearrange(x, "b c t -> b t c").contiguous()
            # mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            _time = time.time()
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            transformer_time += time.time() - _time
            x = rearrange(x, "b t c -> b c t").contiguous()
            # mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            _time = time.time()
            x = resnet(x, mask_up, t)
            conv_time += time.time() - _time
            x = rearrange(x, "b c t -> b t c").contiguous()
            # mask_up = rearrange(mask_up, "b 1 t -> b t")
            if self.up_block_type == "transformer":
                attn_mask = torch.matmul(mask_up.transpose(1, 2).contiguous(), mask_up)
                if self.causal:
                    hint_once(f"use causal attention mask {self.attn_mask_type} for up blocks", "up causal mask")
                    if self.causal_mask_type is None:
                        ilen, device = mask_up.shape[-1], mask_up.device
                        causal_mask = torch.tril(torch.ones((1, ilen, ilen), device=device)).to(torch.bool)
                    else:
                        causal_mask = self.build_mix_causal_mask(attn_mask, mask_rand)
                    attn_mask = attn_mask * causal_mask
                    if self.attn_mask_type == "diffusers":
                        attn_mask = (1 - attn_mask) * torch.finfo(x.dtype).min
            else:
                attn_mask = mask_up.squeeze(1)
            _time = time.time()
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            transformer_time += time.time() - _time
            x = rearrange(x, "b t c -> b c t").contiguous()
            # mask_up = rearrange(mask_up, "b t -> b 1 t")
            _time = time.time()
            x = upsample(x * mask_up)
            conv_time += time.time() - _time

        _time = time.time()
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        if not self.training:
            logging.info(f"conv time: {conv_time*1000.0:.2f}ms, transformer time: {transformer_time*1000.0:.2f}ms, "
                         f"out time: {time.time() - _time:.2f}ms")

        if return_hidden:
            return output * mask, None

        return output * mask


class DiTDecoder(Decoder):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
        conditions: dict = None,
        concat_after: bool = False,
        **kwargs
    ):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        super().__init__(
            in_channels, out_channels, channels, dropout, attention_head_dim,
            n_blocks, num_mid_blocks, num_heads, act_fn, down_block_type, mid_block_type,
            up_block_type, conditions, concat_after, **kwargs
        )
        channels = tuple(channels)
        self.channels = channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conditions = conditions
        self.pos_dropout_rate = kwargs.get("pos_dropout_rate", 0.0)
        self.attention_head_dim = attention_head_dim
        self.num_heads = num_heads
        self.attn_dim = attention_head_dim * num_heads
        self.positional_encoding = self.get_positional_encoding(
            kwargs.get("positional_encoding", None)
        )

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        self.time_embed_dim = time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )
        self.add_text_to_time_embed = kwargs.get("add_text_to_time_embed", False)
        if self.add_text_to_time_embed:
            from diffusers.models.embeddings import TextTimeEmbedding
            self.text_time_embed = TextTimeEmbedding(
                encoder_dim=self.conditions["prompt"]["dim"],
                time_embed_dim=time_embed_dim,
                num_heads=num_heads * 4,  # using many heads to compress text sequence information
            )
        xvec_dim = None
        if conditions is not None and "xvec" in conditions:
            xvec_dim = conditions["xvec"]["dim"]
        self.add_xvec_to_time_embed = kwargs.get("add_xvec_to_time_embed", False)
        if self.add_xvec_to_time_embed:
            assert xvec_dim is not None
            self.xvec_to_time_proj = nn.Linear(
                xvec_dim, time_embed_dim
            )
        self.down_block_type = down_block_type
        self.mid_block_type = mid_block_type
        self.up_block_type = up_block_type

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnet, downsample]))

        for i in range(num_mid_blocks):
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        concat_after=concat_after,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2

            resnet = ResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

    def forward(self, x, mask, mu, t, spks=None, cond=None, return_hidden=False):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        t = self.time_embeddings(t)
        t = self.time_mlp(t)
        if self.add_text_to_time_embed:
            text_emb = self.text_time_embed(cond["prompt"][0])
            t = t + text_emb
            hint_once("add text embedding to time embedding", "add_text_to_time_embed")
        if self.add_xvec_to_time_embed:
            assert spks is not None
            xvec_emb = self.xvec_to_time_proj(spks)
            t = t + xvec_emb
            hint_once("add xvec embedding to time embedding", "add_xvec_to_time_embed")

        if self.positional_encoding is not None:
            cond["prompt"][0] = self.positional_encoding(cond["prompt"][0])
            hint_once("add positional encoding to text", "add_pos_enc_to_text")

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]

        rt_vals = {"inputs": (x, mask[:, 0].sum(dim=-1).long())}

        hiddens = []
        masks = [mask]
        # down-sampling blocks
        for i, (resnet, downsample) in enumerate(self.down_blocks):
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
            rt_vals[f"down_conv_{i}"] = (x, mask_down[:, 0].sum(dim=-1).long())

        # Diffusion Transformer
        masks = masks[:-1]
        mask_mid = masks[-1]
        x = rearrange(x, "b c t -> b t c").contiguous()
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
            hint_once(f"add pos encoding to mid block", f"add_pos_enc_to_mid_block")
        if self.mid_block_type == "transformer":
            attn_mask = torch.matmul(mask_mid.transpose(1, 2).contiguous(), mask_mid)
        elif self.mid_block_type == "transformer_decoder":
            attn_mask = mask_mid
        elif self.mid_block_type in ["diffuser_transformer_decoder", "ditto_transformer_decoder"]:
            attn_mask = torch.matmul(mask_mid.transpose(1, 2).contiguous(), mask_mid)
            enc_mask = (~make_pad_mask(cond["prompt"][1])).to(attn_mask).unsqueeze(1)
            cross_attn_mask = torch.matmul(mask_mid.transpose(1, 2).contiguous(), enc_mask)
        else:
            attn_mask = mask_mid.squeeze(1)
        cond_kwargs = {}
        if self.mid_block_type == "transformer_decoder":
            cond_kwargs["prompt"], cond_kwargs["prompt_lengths"] = cond["prompt"]
        elif self.mid_block_type in ["diffuser_transformer_decoder", "ditto_transformer_decoder"]:
            cond_kwargs["encoder_hidden_states"] = cond["prompt"][0]
            cond_kwargs["encoder_attention_mask"] = cross_attn_mask
        for i, (transformer_blocks, ) in enumerate(self.mid_blocks):
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                    **cond_kwargs
                )
            rt_vals[f"mid_attn_{i}"] = (x, mask_mid[:, 0].sum(dim=-1).long())
        x = rearrange(x, "b t c -> b c t").contiguous()

        # up-sampling blocks
        for i, (resnet, upsample) in enumerate(self.up_blocks):
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = upsample(x * mask_up)
            rt_vals[f"up_conv_{i}"] = (x, mask_up[:, 0].sum(dim=-1).long())

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        rt_vals[f"final"] = (output, mask_up[:, 0].sum(dim=-1).long())

        if return_hidden:
            return output * mask, rt_vals
        return output * mask


class AdaMLPBlock(nn.Module):
    def __init__(
            self,
            channels,
            act_fn="relu",
    ):
        super().__init__()
        from .transformer import get_norm_layer
        self.ln = get_norm_layer("ada_norm", channels, channels)
        self.linear1 = nn.Linear(channels, channels)
        self.act_fn = get_activation(act_fn)
        self.linear2 = nn.Linear(channels, channels)

    def forward(self, x, t):
        o = self.act_fn(self.linear1(self.ln(x, t)))
        o = self.linear2(o) + x
        return o


class SimpleMLP(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            channels=512,
            n_blocks=1,
            act_fn="silu",
    ):
        super().__init__()
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        self.time_embed_dim = time_embed_dim = channels * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            out_dim=channels,
            act_fn="silu",
        )
        self.in_proj = nn.Linear(in_channels, channels)
        self.mlp_layers = nn.ModuleList()
        for i in range(n_blocks):
            self.mlp_layers.append(AdaMLPBlock(channels, act_fn=act_fn))
        self.out_proj = nn.Linear(channels, out_channels)
        self.cond_proj = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.LayerNorm(channels),
        )

    def forward(self, x, mask, t, cond):
        t = self.time_embeddings(t)
        t = self.time_mlp(t)
        t = t + self.cond_proj(cond)

        x = self.in_proj(x)
        for layer in self.mlp_layers:
            x = layer(x, t)
        x = self.out_proj(x)
        x = x * mask

        return x
