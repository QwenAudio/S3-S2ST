"""READ/WRITE gate attached to the Thinker final hidden states.

The audio adaptor emits 12.5 tokens/s, therefore one decision every 640 ms is
exactly one decision every 8 audio positions.  No vocabulary token is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


READ = 0
WRITE = 1


class ReadWriteGate(nn.Module):
    def __init__(self, hidden_size: int = 2560, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 2),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        reference = next(self.parameters())
        return self.net(hidden.to(device=reference.device, dtype=reference.dtype)).float()


@dataclass
class GateBatch:
    hidden: Tensor
    labels: Tensor
    batch_index: Tensor
    token_index: Tensor


def make_gate_targets(
    hidden_states: Tensor,
    feature_beg: Tensor,
    feature_type: Tensor,
    fake_token_len: Tensor,
    interval_tokens: int = 8,
) -> Optional[GateBatch]:
    """Select 640-ms checkpoints from interleaved audio spans.

    ``hidden_states`` is decoder output [sequence, batch, hidden].  Every audio
    feature is one READ...WRITE interval: internal grid points are READ and its
    last complete 8-token grid point is WRITE.  The data builder quantizes turn
    boundaries to 640 ms, so a normal feature has a multiple of 8 adaptor tokens.
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"expected [s,b,h], got {tuple(hidden_states.shape)}")
    seq_len, batch_size, _ = hidden_states.shape
    selected_h, selected_y, selected_b, selected_t = [], [], [], []
    for batch_idx in range(batch_size):
        spans = []
        for feature_idx in range(feature_beg.shape[1]):
            if int(feature_type[batch_idx, feature_idx]) != 1:
                continue
            begin = int(feature_beg[batch_idx, feature_idx])
            length = int(fake_token_len[batch_idx, feature_idx])
            if begin <= 0 or length < interval_tokens:
                continue
            points = list(range(begin + interval_tokens - 1, min(begin + length, seq_len), interval_tokens))
            if points:
                spans.append((points, length % interval_tokens == 0))
        for points, ends_on_grid in spans:
            for point_idx, token_idx in enumerate(points):
                selected_h.append(hidden_states[token_idx, batch_idx])
                selected_y.append(
                    WRITE if ends_on_grid and point_idx == len(points) - 1 else READ
                )
                selected_b.append(batch_idx)
                selected_t.append(token_idx)
    if not selected_h:
        return None
    device = hidden_states.device
    return GateBatch(
        hidden=torch.stack(selected_h),
        labels=torch.tensor(selected_y, dtype=torch.long, device=device),
        batch_index=torch.tensor(selected_b, dtype=torch.long, device=device),
        token_index=torch.tensor(selected_t, dtype=torch.long, device=device),
    )


def balanced_gate_loss(logits: Tensor, labels: Tensor, write_weight: float = 3.33) -> Tensor:
    """Class-weighted CE using the global READ/WRITE prior.

    Streaming valid has WRITE prior ~=0.231, so an unweighted classifier can
    reach ~=0.769 accuracy by predicting READ everywhere.  A fixed global
    weight is more stable than recomputing inverse frequency per packed batch.
    """
    if write_weight <= 0:
        raise ValueError(f"write_weight must be positive, got {write_weight}")
    weight = torch.tensor([1.0, write_weight], dtype=logits.dtype, device=logits.device)
    return F.cross_entropy(logits, labels, weight=weight)


def install_decoder_capture(model: nn.Module) -> None:
    """Capture final normalized Thinker states without editing offline sources."""
    if getattr(model, "_streaming_capture_handle", None) is not None:
        return

    def capture(_module, _inputs, output):
        while isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, Tensor):
            raise TypeError(f"decoder hook expected Tensor, got {type(output)!r}")
        model._streaming_hidden_states = output

    model._streaming_capture_handle = model.decoder.register_forward_hook(capture)


def attach_streaming_gate(model: nn.Module, bottleneck: int = 256, dropout: float = 0.1) -> nn.Module:
    hidden_size = int(model.config.hidden_size)
    model.streaming_gate = ReadWriteGate(hidden_size, bottleneck, dropout)
    install_decoder_capture(model)
    return model
