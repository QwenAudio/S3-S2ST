"""Split full-record encoder output back into interleaved feature spans."""

from __future__ import annotations

import bisect
import types

import torch
from torch import nn


def split_full_record_embeddings(
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    feature_beg: torch.Tensor,
    feature_type: torch.Tensor,
    fake_token_len: torch.Tensor,
    cu_seqlens: torch.Tensor,
):
    """Return the per-feature batch expected by the unchanged offline model."""
    if feature_beg.shape[0] != 1:
        raise ValueError("full-record streaming adapter currently requires micro batch 1")
    boundaries = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    features = []
    for feature_index in range(feature_beg.shape[1]):
        begin = int(feature_beg[0, feature_index])
        length = int(fake_token_len[0, feature_index])
        if begin <= 0 or length <= 0 or int(feature_type[0, feature_index]) != 1:
            continue
        interval_index = bisect.bisect_right(boundaries, begin) - 1
        if interval_index < 0 or interval_index + 1 >= len(boundaries):
            raise ValueError(
                f"feature begin {begin} is outside packed boundaries {boundaries}"
            )
        features.append((interval_index, length))
    source_intervals = []
    for interval_index, _length in features:
        if not source_intervals or source_intervals[-1] != interval_index:
            source_intervals.append(interval_index)
    if len(source_intervals) != encoder_out.shape[0]:
        raise ValueError(
            "full-record source/encoder batch mismatch: "
            f"source_intervals={source_intervals}, encoder={tuple(encoder_out.shape)}"
        )
    row_for_interval = {
        interval_index: row for row, interval_index in enumerate(source_intervals)
    }
    offsets = {interval_index: 0 for interval_index in source_intervals}
    pieces = []
    piece_lengths = []
    for interval_index, length in features:
        row = row_for_interval[interval_index]
        offset = offsets[interval_index]
        available = int(encoder_out_lens[row])
        end = min(offset + length, available)
        piece = encoder_out[row, offset:end]
        if piece.shape[0] < length:
            missing = length - piece.shape[0]
            if missing > 2:
                raise ValueError(
                    "full-record encoder output is too short: "
                    f"row={row} offset={offset} requested={length} available={available}"
                )
            piece = torch.cat(
                [piece, piece.new_zeros((missing, piece.shape[-1]))], dim=0
            )
        pieces.append(piece)
        piece_lengths.append(length)
        offsets[interval_index] += length
    if not pieces:
        raise ValueError("full-record streaming batch has no audio features")
    packed = nn.utils.rnn.pad_sequence(pieces, batch_first=True)
    lengths = encoder_out_lens.new_tensor(piece_lengths)
    return packed, lengths


def install_full_record_encoder_split(model: nn.Module) -> None:
    if getattr(model, "_streaming_full_record_encoder_installed", False):
        return
    original_forward = model.forward_encoder_and_adaptor

    def forward_and_split(this, speech, speech_lengths):
        encoder_out, encoder_out_lens = original_forward(speech, speech_lengths)
        layout = getattr(this, "_streaming_full_record_layout", None)
        if layout is None:
            raise RuntimeError("full-record audio layout was not set before model forward")
        return split_full_record_embeddings(
            encoder_out,
            encoder_out_lens,
            layout["feature_beg"],
            layout["feature_type"],
            layout["fake_token_len"],
            layout["cu_seqlens"],
        )

    model.forward_encoder_and_adaptor = types.MethodType(forward_and_split, model)
    model._streaming_full_record_encoder_installed = True


def set_full_record_layout(model: nn.Module, batch) -> None:
    required = ("feature_beg", "feature_type", "fake_token_len", "cu_seqlens")
    missing = [key for key in required if batch.get(key) is None]
    if missing:
        raise RuntimeError(f"full-record streaming batch is missing {missing}")
    model._streaming_full_record_layout = {
        key: batch[key] for key in required
    }

