#!/usr/bin/env python3
"""Audit the real S2ST training dataset/collator without model training."""

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path


def audit_shard(job):
    shard_rank, shard_path, count, config_path = job
    import torch
    import yaml
    from transformers import AutoTokenizer
    from funasr.frontends.whisper_frontend import WhisperFrontend
    from funasr.datasets.minmo_datasets.datasets import MinMo_S2S_Channels_SFT_Pack_Mega

    cfg = yaml.safe_load(Path(config_path).read_text())
    dataset_conf = dict(cfg["dataset_conf"])
    index_ds = dataset_conf.pop("index_ds")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer_conf"]["init_param_path"], trust_remote_code=True
    )
    frontend = WhisperFrontend(**cfg["frontend_conf"])
    dataset = MinMo_S2S_Channels_SFT_Pack_Mega(
        shard_path,
        index_ds=index_ds,
        frontend=frontend,
        tokenizer=tokenizer,
        **dataset_conf,
    )
    limit = min(count, len(dataset))
    eos_id = dataset.codec_eos_token_id
    max_len = dataset.max_token_length
    rows = []
    aggregate = {
        "requested_packs": limit,
        "loaded_packs": 0,
        "none_packs": 0,
        "utterances": 0,
        "raw_seq_over_max": 0,
        "packs_eos_lost_by_truncation": 0,
        "eos_before_truncation": 0,
        "eos_after_truncation": 0,
        "packs_placeholder_actual_mismatch_before_truncation": 0,
        "packs_placeholder_actual_mismatch_after_truncation": 0,
        "utterances_length_mismatch": 0,
        "utterances_abs_diff_gt1": 0,
        "placeholder_before_truncation": 0,
        "placeholder_after_truncation": 0,
        "actual_codec_tokens": 0,
    }
    for index in range(limit):
        sample = dataset[index]
        if sample is None:
            aggregate["none_packs"] += 1
            continue
        aggregate["loaded_packs"] += 1
        labels_audio = sample["labels_ids_audio"]
        raw_len = int(labels_audio.numel())
        eos_before = int((labels_audio == eos_id).sum().item())
        placeholder_before = int((labels_audio == -50).sum().item())
        labels_truncated = labels_audio[:max_len]
        eos_after = int((labels_truncated == eos_id).sum().item())
        placeholder_after = int((labels_truncated == -50).sum().item())

        actual_lengths = []
        for length in sample.get("segment_fbank_lengths", []):
            value = length[0]
            value = int(value.item()) if hasattr(value, "item") else int(value)
            actual_lengths.append(math.ceil(value / 8))
        segments = list(sample.get("item", {}).get("segments_dict", {}).values())
        expected_lengths = [
            int(round((float(x["end_time"]) - float(x["start_time"])) * 12.5))
            for x in segments
            if x.get("onlie_discrete_speech_token")
        ]
        actual_total = sum(actual_lengths)
        diffs = [a - e for a, e in zip(actual_lengths, expected_lengths)]
        pair_count = min(len(actual_lengths), len(expected_lengths))
        utterance_mismatch = sum(d != 0 for d in diffs)
        utterance_abs_gt1 = sum(abs(d) > 1 for d in diffs)

        over = raw_len > max_len
        eos_lost = eos_before - eos_after
        mismatch_before = placeholder_before != actual_total
        mismatch_after = placeholder_after != actual_total
        aggregate["utterances"] += pair_count
        aggregate["raw_seq_over_max"] += int(over)
        aggregate["packs_eos_lost_by_truncation"] += int(eos_lost > 0)
        aggregate["eos_before_truncation"] += eos_before
        aggregate["eos_after_truncation"] += eos_after
        aggregate["packs_placeholder_actual_mismatch_before_truncation"] += int(mismatch_before)
        aggregate["packs_placeholder_actual_mismatch_after_truncation"] += int(mismatch_after)
        aggregate["utterances_length_mismatch"] += utterance_mismatch
        aggregate["utterances_abs_diff_gt1"] += utterance_abs_gt1
        aggregate["placeholder_before_truncation"] += placeholder_before
        aggregate["placeholder_after_truncation"] += placeholder_after
        aggregate["actual_codec_tokens"] += actual_total
        if over or mismatch_before or eos_lost or len(actual_lengths) != len(expected_lengths):
            rows.append({
                "shard_rank": shard_rank,
                "shard": shard_path,
                "index": index,
                "dataset_index_cur": sample.get("speech_info", {}).get("index_cur"),
                "raw_seq_len": raw_len,
                "max_len": max_len,
                "eos_before": eos_before,
                "eos_after": eos_after,
                "placeholder_before": placeholder_before,
                "placeholder_after": placeholder_after,
                "actual_total": actual_total,
                "expected_segments": len(expected_lengths),
                "actual_segments": len(actual_lengths),
                "utterance_mismatch": utterance_mismatch,
                "max_abs_diff": max((abs(x) for x in diffs), default=0),
                "diffs": diffs,
            })
        if (index + 1) % 50 == 0:
            print(f"AUDIT_PROGRESS shard={shard_rank} packs={index + 1}/{limit}", flush=True)
    return {"shard_rank": shard_rank, "shard": shard_path, "aggregate": aggregate, "anomalies": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--packs-per-shard", type=int, default=256)
    args = parser.parse_args()
    shard_paths = [x.strip() for x in Path(args.data_list).read_text().splitlines() if x.strip()]
    if args.num_shards > len(shard_paths):
        raise ValueError("num_shards exceeds available shards")
    positions = [round(i * (len(shard_paths) - 1) / max(args.num_shards - 1, 1)) for i in range(args.num_shards)]
    jobs = [(i, shard_paths[pos], args.packs_per_shard, args.config) for i, pos in enumerate(positions)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.num_shards) as pool:
        results = pool.map(audit_shard, jobs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = {}
    for result in results:
        for key, value in result["aggregate"].items():
            total[key] = total.get(key, 0) + value
    report = {"config": args.config, "data_list": args.data_list, "jobs": jobs, "total": total, "shards": results}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "anomalies.jsonl").open("w") as f:
        for result in results:
            for row in result["anomalies"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("AUDIT_SUMMARY " + json.dumps(total, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
