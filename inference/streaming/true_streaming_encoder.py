"""Block-causal SenseVoice inference adapter kept inside streaming/eval."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.models.omnidrca2_encoder import apply_rotary_pos_emb


def enable_true_streaming_encoder(encoder: nn.Module, chunk_ms: float = 640.0) -> None:
    """Make every encoder output depend only on its current/previous chunks.

    The forward still batches the complete fbank for efficient evaluation, but
    block-causal attention and history-only FSMN make the output for each closed
    640-ms chunk identical to an online encoder with the same accumulated state.
    """
    if chunk_ms <= 0:
        raise ValueError(f"chunk_ms must be positive: {chunk_ms}")
    conv_downsample = encoder.conv1.stride[0] * encoder.conv2.stride[0]
    chunk_frames = max(1, round(chunk_ms / (10.0 * conv_downsample)))

    def qkv_attention(attn, q, k, v, mask=None, **kwargs):
        batch, time, dim = q.shape
        scale = (dim // attn.attention_heads) ** -0.5
        q = q.view(batch, time, attn.attention_heads, -1).permute(0, 2, 1, 3)
        k = k.view(batch, time, attn.attention_heads, -1).permute(0, 2, 1, 3)
        v = v.view(batch, time, attn.attention_heads, -1).permute(0, 2, 1, 3)
        position_ids = kwargs.get("position_ids")
        cos, sin = attn.rotary_emb(v, seq_len=v.shape[-2])
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            attention_mask = mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).to(torch.bool)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=False, scale=scale,
        )
        if mask is not None:
            valid_queries = mask.to(torch.bool).unsqueeze(-1)
            output = output.masked_fill(valid_queries.logical_not(), 0.0)
        return output.transpose(1, 2).flatten(start_dim=2), None

    for block in encoder.blocks:
        attn = block.attn
        kernel = int(attn.fsmn_block.kernel_size[0])
        attn.pad_fn = nn.ConstantPad1d((kernel - 1, 0), 0.0)
        attn.qkv_attention = MethodType(qkv_attention, attn)

    def streaming_forward(module, x, ilens=None, **_kwargs):
        # Match reference training implementation: both strided
        # convolutions use left-only causal padding.  The 2.5-ms waveform
        # lookahead remains necessary for the centered Whisper STFT boundary.
        x = F.pad(x, (module.conv1.kernel_size[0] - 1, 0))
        x = F.gelu(F.conv1d(
            x, module.conv1.weight, module.conv1.bias,
            stride=module.conv1.stride, padding=0,
            dilation=module.conv1.dilation, groups=module.conv1.groups,
        ))
        x = F.pad(x, (module.conv2.kernel_size[0] - 1, 0))
        x = F.gelu(F.conv1d(
            x, module.conv2.weight, module.conv2.bias,
            stride=module.conv2.stride, padding=0,
            dilation=module.conv2.dilation, groups=module.conv2.groups,
        )).permute(0, 2, 1)
        time = x.size(1)
        if ilens is None:
            olens = torch.full((x.size(0),), time, dtype=torch.long, device=x.device)
        else:
            olens = 1 + (
                ilens - module.conv1.kernel_size[0] + 2 * module.conv1.padding[0]
            ) // module.conv1.stride[0]
            olens = 1 + (
                olens - module.conv2.kernel_size[0] + 2 * module.conv2.padding[0]
            ) // module.conv2.stride[0]
            olens = torch.clamp(olens, max=time)
        positions = torch.arange(time, device=x.device)
        valid = positions.unsqueeze(0) < olens.unsqueeze(1)
        padding_mask = valid.unsqueeze(1)
        chunk_ids = torch.div(positions, chunk_frames, rounding_mode="floor")
        block_causal = chunk_ids.unsqueeze(0) <= chunk_ids.unsqueeze(1)
        attention_mask = block_causal.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)
        position_ids = positions.unsqueeze(0)
        for block in module.blocks:
            x = block(
                x, mask=padding_mask, position_ids=position_ids,
                attention_mask=attention_mask,
            )
        x = module.ln_post(x)
        return x if ilens is None else (x, olens)

    encoder.forward = MethodType(streaming_forward, encoder)
    encoder.streaming_chunk_ms = chunk_ms
    encoder.streaming_chunk_frames = chunk_frames


def enable_true_streaming_adaptor(
    adaptor: nn.Module,
    encoder_chunk_frames: int,
) -> None:
    """Make a full adaptor pass identical to closed-chunk prefix inference."""
    downsample = int(getattr(adaptor, "k", 0))
    if downsample <= 0:
        raise ValueError(f"invalid adaptor downsample rate: {downsample}")
    if encoder_chunk_frames % downsample:
        raise ValueError(
            "encoder chunk must align with adaptor downsampling: "
            f"{encoder_chunk_frames} % {downsample} != 0"
        )
    adaptor.streaming_chunk_size = encoder_chunk_frames // downsample
