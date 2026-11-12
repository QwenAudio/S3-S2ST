"""Packed-sequence-safe causal convolution for the 4B GatedDeltaNet.

The imported reference stack falls back to one depthwise ``F.conv1d`` over the
entire packed bin when causal-conv1d is unavailable.  That convolution ignores
``cu_seqlens`` and leaks the last ``kernel_size - 1`` states of one example into
the next.  This local implementation follows causal-conv1d's ``seq_idx``
contract and keeps the external reference checkout read-only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def packed_safe_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply depthwise causal convolution with an exact reset at every seq_idx."""
    if x.ndim != 3:
        raise ValueError(f"causal conv expects [batch, channel, time], got {x.shape}")
    if weight.ndim != 2 or weight.shape[0] != x.shape[1]:
        raise ValueError(
            "causal conv weight must be [channel, kernel], got "
            f"x={tuple(x.shape)} weight={tuple(weight.shape)}"
        )
    time = x.shape[-1]
    kernel = weight.shape[-1]
    if kernel <= 0:
        raise ValueError(f"causal conv kernel must be positive, got {kernel}")

    if seq_idx is None:
        output = F.conv1d(
            x,
            weight.unsqueeze(1),
            bias=bias,
            stride=1,
            padding=kernel - 1,
            groups=x.shape[1],
        )[..., :time]
    else:
        if seq_idx.ndim == 1:
            seq_idx = seq_idx.unsqueeze(0)
        if seq_idx.ndim != 2 or seq_idx.shape[-1] != time:
            raise ValueError(
                f"seq_idx must be [batch,time], got {tuple(seq_idx.shape)}"
            )
        if seq_idx.shape[0] == 1 and x.shape[0] != 1:
            seq_idx = seq_idx.expand(x.shape[0], -1)
        if seq_idx.shape[0] != x.shape[0]:
            raise ValueError(
                f"seq_idx batch mismatch: {seq_idx.shape[0]} != {x.shape[0]}"
            )

        # The Qwen3.5 kernel is four taps.  Accumulating shifted inputs avoids
        # a Python loop over packed examples and remains fully differentiable.
        accumulation_dtype = (
            torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        )
        output = torch.zeros_like(x, dtype=accumulation_dtype)
        for kernel_index in range(kernel):
            lag = kernel - 1 - kernel_index
            if lag:
                shifted = F.pad(x, (lag, 0))[..., :time]
                valid = torch.zeros_like(seq_idx, dtype=torch.bool)
                valid[:, lag:] = seq_idx[:, lag:] == seq_idx[:, :-lag]
                shifted = shifted * valid.unsqueeze(1).to(dtype=x.dtype)
            else:
                shifted = x
            output = output + shifted.to(accumulation_dtype) * weight[
                :, kernel_index
            ].to(accumulation_dtype).view(1, -1, 1)
        if bias is not None:
            output = output + bias.to(accumulation_dtype).view(1, -1, 1)
        output = output.to(dtype=x.dtype)

    if activation in (None, "identity"):
        return output
    if activation in ("silu", "swish"):
        return F.silu(output)
    raise ValueError(f"unsupported causal conv activation: {activation!r}")


def install_packed_safe_causal_conv1d() -> None:
    """Install before GatedDeltaNet construction if the CUDA op is unavailable."""
    from megatron.core.transformer import gated_delta_net

    if gated_delta_net.causal_conv1d_fn is not None:
        print("[STREAMING] GatedDeltaNet causal conv=native causal-conv1d", flush=True)
        return
    gated_delta_net.causal_conv1d_fn = packed_safe_causal_conv1d
    print(
        "[STREAMING] GatedDeltaNet causal conv=torch packed-safe "
        "(exact cu_seqlens resets)",
        flush=True,
    )
