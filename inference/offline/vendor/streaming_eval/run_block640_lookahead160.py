#!/usr/bin/env python3
"""True-streaming global-Talker S2ST entry for block640-lookahead160.

All project-specific Python sources are colocated in this directory.  This
wrapper installs the exact 640+160 frontend schedule before importing the
global-Talker evaluator.
"""

from __future__ import annotations

import numpy as np
import torch

import streaming_core as s2tt
from block640_lookahead160_encoder import (
    enable_block640_lookahead160_encoder,
)


ENCODER_LOOKAHEAD_MS = 160.0
STFT_BOUNDARY_LOOKAHEAD_MS = 2.5
TOTAL_WAVEFORM_LOOKAHEAD_MS = (
    ENCODER_LOOKAHEAD_MS + STFT_BOUNDARY_LOOKAHEAD_MS
)


def _enable_encoder(encoder, chunk_ms=640.0):
    enable_block640_lookahead160_encoder(
        encoder,
        chunk_ms=chunk_ms,
        lookahead_ms=ENCODER_LOOKAHEAD_MS,
    )


@torch.inference_mode()
def _stream_prefixes(self, wav: np.ndarray, chunk_samples: int, lookahead_samples: int):
    """Encode 160-ms future but emit only tokens belonging to closed blocks."""
    if not self.true_streaming:
        raise RuntimeError("stream_prefixes requires --true-streaming-encoder")
    if chunk_samples <= 0:
        raise ValueError(f"chunk_samples must be positive: {chunk_samples}")
    if lookahead_samples < 0:
        raise ValueError(f"lookahead_samples must be non-negative: {lookahead_samples}")

    chunk_tokens = int(getattr(self.adaptor, "streaming_chunk_size", 0))
    if chunk_tokens != 8:
        raise RuntimeError(
            "training contract requires an 8-token block-causal adaptor, "
            f"got {chunk_tokens}"
        )

    emitted = 0
    feature_prefix_samples = chunk_samples
    while True:
        if feature_prefix_samples >= len(wav):
            feature_prefix_samples = len(wav)
            available_prefix_samples = len(wav)
            end_of_audio = True
        else:
            available_prefix_samples = min(
                feature_prefix_samples + lookahead_samples, len(wav)
            )
            end_of_audio = available_prefix_samples == len(wav)

        prefix = np.ascontiguousarray(wav[:available_prefix_samples])
        prefix_embeds = self(prefix)
        if end_of_audio:
            available = len(prefix_embeds)
        else:
            closed_chunks = feature_prefix_samples // chunk_samples
            closed_token_limit = closed_chunks * chunk_tokens
            available = min(len(prefix_embeds), closed_token_limit)
            prefix_embeds = prefix_embeds[:available]

        if available < emitted:
            raise RuntimeError(
                f"streaming frontend length regressed: {available} < {emitted}"
            )
        new_embeds = prefix_embeds[emitted:available].contiguous()
        yield (
            feature_prefix_samples,
            available_prefix_samples,
            new_embeds,
            available,
            end_of_audio,
        )
        emitted = available
        if end_of_audio:
            break
        feature_prefix_samples += chunk_samples


s2tt.enable_true_streaming_encoder = _enable_encoder
s2tt.FullRecordEncoder.stream_prefixes = _stream_prefixes
s2tt.STREAMING_LOOKAHEAD_MS = TOTAL_WAVEFORM_LOOKAHEAD_MS

# Import only after installing the frontend patch. The S2ST evaluator holds a
# reference to this same ``run_streaming_s2tt`` module object.
import infer_deferred_s2st as s2st  # noqa: E402


if __name__ == "__main__":
    print(
        "[BLOCK640 LOOKAHEAD160 GLOBAL-TIMELINE S2ST CONTRACT] "
        "waveform availability = closed 640ms + 160ms encoder context "
        "+ 2.5ms centered-STFT boundary; only closed-block adaptor tokens "
        "are emitted; one persistent Talker receives one real Thinker hidden "
        "per position plus valid codec, injected BOS, or a zero codec branch; "
        "gap logits are discarded; Thinker padding starts only after the "
        "source-complete final WRITE. Boundary policy is selected explicitly: "
        "historical unified_eos treats 59050 as per-WRITE AUDIO_END, while "
        "dual_bos_pause_eos uses 59051 between WRITEs and 59050 only at end",
        flush=True,
    )
    s2st.main()
