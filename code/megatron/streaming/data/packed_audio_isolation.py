"""Streaming-only repair for FunASR Mega packed audio decode alignment."""

from __future__ import annotations

import torch


def merge_isolated_packed_samples(samples):
    """Rebuild one packed bin from independently decoded source samples."""
    samples = [sample for sample in samples if sample is not None]
    if not samples:
        return None

    seq_keys = (
        "input_ids",
        "labels_ids",
        "input_ids_audio",
        "labels_ids_audio",
    )
    merged = {key: [] for key in seq_keys}
    merged.update(
        attention_mask=[],
        feature_beg=[],
        feature_type=[],
        fake_token_len=[],
        speech=[],
        speech_lengths=[],
    )
    cu_seqlens = [0]
    offset = 0
    for sample_index, sample in enumerate(samples, start=1):
        seq_len = int(sample["input_ids"].numel())
        for key in seq_keys:
            merged[key].append(sample[key])
        merged["attention_mask"].append(
            torch.full_like(sample["attention_mask"], sample_index)
        )
        merged["feature_beg"].append(sample["feature_beg"] + offset)
        merged["feature_type"].append(sample["feature_type"])
        merged["fake_token_len"].append(sample["fake_token_len"])
        merged["speech"].extend(sample.get("speech", []))
        merged["speech_lengths"].extend(sample.get("speech_lengths", []))
        offset += seq_len
        cu_seqlens.append(offset)

    for key in seq_keys + ("attention_mask",):
        merged[key] = torch.cat(merged[key], dim=0)
    for key in ("feature_beg", "feature_type", "fake_token_len"):
        values = merged[key]
        if values:
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = samples[0][key].new_empty((0,))
    merged["cu_seqlens"] = torch.tensor(cu_seqlens, dtype=torch.int32)
    merged["item"] = None
    merged["speech_token_id_offset"] = samples[0]["speech_token_id_offset"]
    return merged
