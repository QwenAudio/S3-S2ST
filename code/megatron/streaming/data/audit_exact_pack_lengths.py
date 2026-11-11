#!/usr/bin/env python3
"""Audit exact streaming length metadata and packed-bin utilization."""

import argparse
import json
import math
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-list", required=True)
    parser.add_argument("--cutoff-len", type=int, default=8192)
    parser.add_argument("--underfilled-threshold", type=int, default=7000)
    parser.add_argument("--output")
    args = parser.parse_args()

    packed_list = Path(args.packed_list).resolve()
    shards = [Path(line.strip()) for line in packed_list.read_text().splitlines() if line.strip()]
    if not shards:
        raise ValueError("packed data.list is empty")
    summary_path = packed_list.parent / "packing_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("length_accounting") != "exact_streaming_runtime_v1":
        raise ValueError(f"missing exact length marker in {summary_path}")

    bin_lengths = []
    conversations = mismatches = over_cutoff = 0
    underfilled_by_shard = {}
    for shard in shards:
        shard_underfilled = 0
        with shard.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                packed = json.loads(line)
                bin_length = 0
                for conversation in packed:
                    conversations += 1
                    message_text = sum(int(item["text_length"]) for item in conversation["messages"])
                    message_speech = sum(int(item["speech_length"]) for item in conversation["messages"])
                    if (
                        int(conversation["text_length"]) != message_text
                        or int(conversation["speech_length"]) != message_speech
                    ):
                        mismatches += 1
                    bin_length += message_text + message_speech
                bin_lengths.append(bin_length)
                over_cutoff += int(bin_length > args.cutoff_len)
                shard_underfilled += int(bin_length < args.underfilled_threshold)
        underfilled_by_shard[str(shard)] = shard_underfilled

    underfilled = sum(underfilled_by_shard.values())
    # Greedy knapsack normally leaves at most one low-fill residual bin per
    # shard.  Keep a tiny allowance for length-distribution edge cases.
    allowed_underfilled = len(shards) + math.ceil(len(bin_lengths) * 0.001)
    report = {
        "length_accounting": summary["length_accounting"],
        "shards": len(shards),
        "bins": len(bin_lengths),
        "conversations": conversations,
        "conversation_length_mismatches": mismatches,
        "over_cutoff_bins": over_cutoff,
        "underfilled_threshold": args.underfilled_threshold,
        "underfilled_bins": underfilled,
        "allowed_underfilled_bins": allowed_underfilled,
        "bin_length": {
            "min": min(bin_lengths),
            "p10": percentile(bin_lengths, 0.10),
            "p50": percentile(bin_lengths, 0.50),
            "p90": percentile(bin_lengths, 0.90),
            "max": max(bin_lengths),
        },
        "underfilled_by_shard": {
            shard: count for shard, count in underfilled_by_shard.items() if count
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if mismatches or over_cutoff or underfilled > allowed_underfilled:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
