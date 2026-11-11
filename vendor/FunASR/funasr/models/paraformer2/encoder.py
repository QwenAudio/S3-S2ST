#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from funasr.models.ctc.ctc import CTC
from funasr.models.rwkv_bat.rwkv import RWKV
from funasr.models.sanm.attention import MultiHeadedAttention, MultiHeadedAttentionSANM
from funasr.models.scama.chunk_utilis import overlap_chunk
from funasr.models.transformer.embedding import SinusoidalPositionEncoder, StreamSinusoidalPositionEncoder
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.transformer.positionwise_feed_forward import PositionwiseFeedForward  # noqa: H301
from funasr.models.transformer.utils.multi_layer_conv import Conv1dLinear, MultiLayeredConv1d
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.utils.repeat import repeat
from funasr.models.transformer.utils.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)
from funasr.register import tables


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder


@tables.register("encoder_classes", "SANMRWKVTPEncoderChunkOpt")
class SANMRWKVTPEncoderChunkOpt(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition
    https://arxiv.org/abs/2006.01713

    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        num_rwkv_blocks: int = 6,
        tp_blocks: int = 0,
        context_size: int = 1024,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        selfattention_layer_type: str = "sanm",
        chunk_model1: bool = True,
        chunk_model2: bool = True,
        chunk_size: Union[int, Sequence[int]] = (16,),
        stride: Union[int, Sequence[int]] = (10,),
        pad_left: Union[int, Sequence[int]] = (0,),
        encoder_att_look_back_factor: Union[int, Sequence[int]] = (1,),
        decoder_att_look_back_factor: Union[int, Sequence[int]] = (1,),
        chunk_size2: Union[int, Sequence[int]] = (16,),
        stride2: Union[int, Sequence[int]] = (10,),
        pad_left2: Union[int, Sequence[int]] = (0,),
        encoder_att_look_back_factor2: Union[int, Sequence[int]] = (1,),
        decoder_att_look_back_factor2: Union[int, Sequence[int]] = (1,),
        time_reduction_factor: int = 1,
        chunk_out: bool = True,
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                eval(pos_enc_class)(output_size, positional_dropout_rate),
            )
        elif input_layer == "linear_no_pos":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                eval(pos_enc_class)(output_size, positional_dropout_rate, use_pos=False),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                eval(pos_enc_class)(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        elif input_layer == "pe":
            self.embed = SinusoidalPositionEncoder()
        elif input_layer == "pe_online":
            self.embed = StreamSinusoidalPositionEncoder()
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "sanm":
            encoder_selfattn_layer = MultiHeadedAttentionSANM
            encoder_selfattn_layer_args0 = (
                attention_heads,
                input_size,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
            )

            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
            )
        self.encoders0 = repeat(
            1,
            lambda lnum: EncoderLayerSANM(
                input_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

        self.encoders = repeat(
            num_blocks - 1,
            lambda lnum: EncoderLayerSANM(
                output_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate,
            ),
        )

        self.rwkv_encoders = torch.nn.ModuleList(
            [
                RWKV(
                    output_size,
                    linear_units,
                    output_size,
                    context_size,
                    block_id,
                    num_rwkv_blocks,
                    att_dropout_rate=attention_dropout_rate,
                    ffn_dropout_rate=dropout_rate,
                    dropout_rate=dropout_rate,
                )
                for block_id in range(num_rwkv_blocks)
            ]
        )

        self.tp_encoders = repeat(
            tp_blocks,
            lambda lnum: EncoderLayerSANM(
                output_size,
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)
        self.num_rwkv_blocks = num_rwkv_blocks
        if self.num_rwkv_blocks > 0:
            self.rwkv_norm = LayerNorm(output_size)
        self.tp_blocks = tp_blocks
        if self.tp_blocks > 0:
            self.tp_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        shfit_fsmn = (kernel_size - 1) // 2
        if chunk_model1:
            self.overlap_chunk_cls = overlap_chunk(
                chunk_size=chunk_size,
                stride=stride,
                pad_left=pad_left,
                shfit_fsmn=shfit_fsmn,
                encoder_att_look_back_factor=encoder_att_look_back_factor,
                decoder_att_look_back_factor=decoder_att_look_back_factor,
            )
        else:
            self.overlap_chunk_cls = None
        if self.tp_blocks > 0 and chunk_model2:
            self.overlap_chunk_cls2 = overlap_chunk(
                chunk_size=chunk_size2,
                stride=stride2,
                pad_left=pad_left2,
                shfit_fsmn=shfit_fsmn,
                encoder_att_look_back_factor=encoder_att_look_back_factor2,
                decoder_att_look_back_factor=decoder_att_look_back_factor2,
            )
        else:
            self.overlap_chunk_cls2 = None
        self.time_reduction_factor = time_reduction_factor
        self.chunk_out = chunk_out

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
        ind_1st: int = 0,
        ind_2st: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
                xs_pad: input tensor (B, L, D)
                ilens: input length (B)
                prev_states: Not to be used now.
        Returns:
                position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        xs_pad *= self.output_size() ** 0.5
        if self.embed is None:
            xs_pad = xs_pad
        elif (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        # forward encoder1
        mask_shfit_chunk, mask_att_chunk_encoder = None, None
        if self.overlap_chunk_cls is not None:
            ilens = masks.squeeze(1).sum(1)
            ilens_clone = ilens.clone()
            chunk_outs = self.overlap_chunk_cls.gen_chunk_mask(ilens, ind_1st)
            xs_pad, ilens = self.overlap_chunk_cls.split_chunk(xs_pad, ilens, chunk_outs=chunk_outs)
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
            mask_shfit_chunk = self.overlap_chunk_cls.get_mask_shfit_chunk(
                chunk_outs, xs_pad.device, xs_pad.size(0), dtype=xs_pad.dtype
            )
            mask_att_chunk_encoder = self.overlap_chunk_cls.get_mask_att_chunk_encoder(
                chunk_outs, xs_pad.device, xs_pad.size(0), dtype=xs_pad.dtype
            )

        encoder_outs = self.encoders0(xs_pad, masks, None, mask_shfit_chunk, mask_att_chunk_encoder)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]
        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            encoder_outs = self.encoders(xs_pad, masks, None, mask_shfit_chunk, mask_att_chunk_encoder)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                encoder_outs = encoder_layer(xs_pad, masks, None, mask_shfit_chunk, mask_att_chunk_encoder)
                xs_pad, masks = encoder_outs[0], encoder_outs[1]
                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)
                        xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if hasattr(self, "overlap_chunk_cls") and self.overlap_chunk_cls is not None:
            xs_pad, olens = self.overlap_chunk_cls.remove_chunk(xs_pad, olens, chunk_outs=None)

        if self.num_rwkv_blocks > 0:
            for block in self.rwkv_encoders:
                xs_pad, _ = block(xs_pad)
            xs_pad = self.rwkv_norm(xs_pad)
        enc_out_1st = xs_pad

        # forward encoder2
        ilens = olens
        mask_shfit_chunk2, mask_att_chunk_encoder2 = None, None
        if self.overlap_chunk_cls2 is not None:
            chunk_outs = self.overlap_chunk_cls2.gen_chunk_mask(ilens, ind_2st)
            xs_pad, ilens = self.overlap_chunk_cls2.split_chunk(xs_pad, ilens, chunk_outs=chunk_outs)
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
            mask_shfit_chunk2 = self.overlap_chunk_cls2.get_mask_shfit_chunk(
                chunk_outs, xs_pad.device, xs_pad.size(0), dtype=xs_pad.dtype
            )
            mask_att_chunk_encoder2 = self.overlap_chunk_cls2.get_mask_att_chunk_encoder(
                chunk_outs, xs_pad.device, xs_pad.size(0), dtype=xs_pad.dtype
            )

        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            encoder_outs = encoder_layer(xs_pad, masks, None, mask_shfit_chunk2, mask_att_chunk_encoder2)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        if self.tp_blocks > 0:
            xs_pad = self.tp_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if hasattr(self, "overlap_chunk_cls2") and self.overlap_chunk_cls2 is not None:
            xs_pad, olens = self.overlap_chunk_cls2.remove_chunk(xs_pad, olens, chunk_outs=None)

        enc_out_2st = xs_pad

        if self.chunk_out:
            if hasattr(self, "overlap_chunk_cls") and self.overlap_chunk_cls is not None:
                enc_out_1st, _ = self.overlap_chunk_cls.split_chunk(enc_out_1st, ilens_clone, chunk_outs=None)

            if hasattr(self, "overlap_chunk_cls2") and self.overlap_chunk_cls2 is not None:
                enc_out_2st, _ = self.overlap_chunk_cls2.split_chunk(enc_out_2st, ilens_clone, chunk_outs=None)

        if self.time_reduction_factor > 1:
            enc_out_1st = enc_out_1st[:, :: self.time_reduction_factor, :]
            enc_out_2st = enc_out_2st[:, :: self.time_reduction_factor, :]
            olens = torch.floor_divide(olens - 1, self.time_reduction_factor) + 1

        if len(intermediate_outs) > 0:
            return (enc_out_1st, intermediate_outs), olens, enc_out_2st

        return enc_out_1st, olens, enc_out_2st
