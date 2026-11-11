"""Inference-only replica of the isolated block640-lookahead160 encoder.

This module is separate from true_streaming_encoder.py so the historical Hejia
evaluation path remains byte-for-byte unchanged.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.models.omnidrca2_encoder import apply_rotary_pos_emb


def _block_streaming_fsmn(
    inputs: torch.Tensor,
    mask: torch.Tensor | None,
    conv: nn.Conv1d,
    chunk_frames: int,
    right_context_frames: int,
) -> torch.Tensor:
    """Match training's tap-preserving FSMN with an exact per-block cutoff."""
    batch, time, dim = inputs.shape
    kernel = int(conv.kernel_size[0])
    if kernel % 2 != 1:
        raise ValueError(f"block FSMN requires an odd kernel, got {kernel}")
    half = kernel // 2
    if not 0 <= right_context_frames <= half:
        raise ValueError(
            "FSMN right context exceeds the offline taps: "
            f"right={right_context_frames} half={half}"
        )

    valid = None
    if mask is not None:
        valid = mask.reshape(batch, -1, 1).to(dtype=inputs.dtype)
        inputs = inputs * valid

    num_chunks = (time + chunk_frames - 1) // chunk_frames
    padded_time = num_chunks * chunk_frames
    values = inputs.transpose(1, 2)
    if padded_time != time:
        values = F.pad(values, (0, padded_time - time))

    padded = F.pad(values, (half, half))
    context = 2 * half + chunk_frames
    windows = padded.unfold(-1, context, chunk_frames)
    windows = windows.permute(0, 2, 1, 3).contiguous()
    boundary_gate = torch.ones(
        context, dtype=windows.dtype, device=windows.device
    )
    boundary_gate[half + chunk_frames + right_context_frames :] = 0
    windows = windows * boundary_gate.view(1, 1, 1, context)
    windows = windows.reshape(batch * num_chunks, dim, context)

    output = F.conv1d(
        windows,
        conv.weight,
        conv.bias,
        stride=1,
        padding=0,
        dilation=conv.dilation,
        groups=conv.groups,
    )
    output = output.reshape(batch, num_chunks, dim, chunk_frames)
    output = output.permute(0, 2, 1, 3).reshape(batch, dim, padded_time)
    output = output[..., :time].transpose(1, 2) + inputs
    if valid is not None:
        output = output * valid
    return output


def _block_attention_allow_mask(
    lengths: torch.Tensor,
    sequence_length: int,
    chunk_frames: int,
    lookahead_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the exact [B,Q,K] allow-mask used by training."""
    positions = torch.arange(sequence_length, device=device)
    chunk_ids = torch.div(positions, chunk_frames, rounding_mode="floor")
    chunk_end = (chunk_ids + 1) * chunk_frames - 1
    visible = positions.unsqueeze(0) <= (
        chunk_end + lookahead_frames
    ).unsqueeze(1)
    valid = positions.unsqueeze(0) < lengths.to(device=device).unsqueeze(1)
    return visible.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)


def _streaming_qkv_attention(attn, q, k, v, mask=None, **kwargs):
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
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    if mask is not None:
        valid_queries = mask.to(torch.bool).unsqueeze(-1)
        output = output.masked_fill(valid_queries.logical_not(), 0.0)
    return output.transpose(1, 2).flatten(start_dim=2), None


def enable_block640_lookahead160_encoder(
    encoder: nn.Module,
    chunk_ms: float = 640.0,
    lookahead_ms: float = 160.0,
) -> None:
    """Patch vLLM's encoder to match the isolated training implementation."""
    if chunk_ms <= 0 or lookahead_ms < 0:
        raise ValueError(
            f"invalid block/lookahead: chunk={chunk_ms}, lookahead={lookahead_ms}"
        )
    conv_downsample = encoder.conv1.stride[0] * encoder.conv2.stride[0]
    chunk_frames = max(1, round(chunk_ms / (10.0 * conv_downsample)))
    lookahead_frames = round(lookahead_ms / (10.0 * conv_downsample))
    if chunk_frames != 16 or lookahead_frames != 4:
        raise ValueError(
            "block640-lookahead160 requires 16+4 encoder frames, got "
            f"{chunk_frames}+{lookahead_frames}"
        )

    for layer, block in enumerate(encoder.blocks):
        attn = block.attn
        attn.qkv_attention = MethodType(_streaming_qkv_attention, attn)
        right_context = lookahead_frames if layer == 0 else 0

        def block_fsmn(this, inputs, mask=None, *, _right=right_context):
            output = _block_streaming_fsmn(
                inputs,
                mask,
                this.fsmn_block,
                chunk_frames,
                _right,
            )
            return this.dropout(output)

        attn.fsmn = MethodType(block_fsmn, attn)

    def streaming_forward(module, x, ilens=None, **_kwargs):
        # Preserve the offline convolution and its pretrained centered taps.
        x = F.gelu(module.conv1(x))
        x = F.gelu(module.conv2(x)).permute(0, 2, 1)
        time = x.size(1)
        if ilens is None:
            olens = torch.full(
                (x.size(0),), time, dtype=torch.long, device=x.device
            )
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
        position_ids = positions.unsqueeze(0)

        for layer, block in enumerate(module.blocks):
            layer_lookahead = lookahead_frames if layer == 0 else 0
            attention_mask = _block_attention_allow_mask(
                olens,
                time,
                chunk_frames,
                layer_lookahead,
                x.device,
            )
            x = block(
                x,
                mask=padding_mask,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        x = module.ln_post(x)
        return x if ilens is None else (x, olens)

    encoder.forward = MethodType(streaming_forward, encoder)
    encoder.streaming_chunk_ms = chunk_ms
    encoder.streaming_chunk_frames = chunk_frames
    encoder.streaming_lookahead_ms = lookahead_ms
    encoder.streaming_lookahead_frames = lookahead_frames
    print(
        "[BLOCK640 LOOKAHEAD160 INFERENCE] "
        "offline-conv + tap-preserving-FSMN; "
        "chunk=640ms; layer0-lookahead=160ms; later-layer-lookahead=0",
        flush=True,
    )
