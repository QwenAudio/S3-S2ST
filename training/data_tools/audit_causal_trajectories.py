#!/usr/bin/env python3
"""Audit unpacked causal-clean streaming records and emit readable samples."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


PROMPTS = {
    "en2zh": "Translate into Chinese:",
    "zh2en": "Translate into English:",
}
LANGS = {
    "en2zh": ("EN", "ZH"),
    "zh2en": ("ZH", "EN"),
}


def read_list(path: Path) -> list[Path]:
    shards = [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not shards:
        raise ValueError(f"empty data.list: {path}")
    for shard in shards:
        if not shard.is_file() or shard.stat().st_size == 0:
            raise FileNotFoundError(shard)
    return shards


def validate_record(record: dict, direction: str) -> dict:
    messages = record.get("messages") or []
    if len(messages) < 5 or len(messages) % 2 == 0:
        raise ValueError("record must contain system and at least two user/assistant turns")
    expected_roles = ["system"] + [
        role
        for _ in range((len(messages) - 1) // 2)
        for role in ("user", "assistant")
    ]
    if [message.get("role") for message in messages] != expected_roles:
        raise ValueError("role order mismatch")
    source_lang, target_lang = LANGS[direction]
    if record.get("src_lang") != source_lang or record.get("tgt_lang") != target_lang:
        raise ValueError("language metadata mismatch")
    streaming = record.get("streaming") or {}
    if streaming.get("direction") != direction:
        raise ValueError("direction metadata mismatch")
    if int(streaming.get("decision_ms", 0)) != 640:
        raise ValueError("decision_ms is not 640")
    if int(streaming.get("interval_audio_tokens", 0)) != 8:
        raise ValueError("interval_audio_tokens is not 8")
    causal = streaming.get("causal_clean") or {}
    if causal.get("version") != "causal_clean_v1":
        raise ValueError("missing causal_clean_v1 metadata")
    if int(causal.get("boundary_delay_ms", -1)) < 0:
        raise ValueError("invalid boundary delay")
    users = messages[1::2]
    assistants = messages[2::2]
    first_text = str(users[0].get("text") or users[0].get("content") or "")
    if not first_text.startswith(PROMPTS[direction]):
        raise ValueError("direction prompt mismatch")
    previous_end = None
    total_duration = 0.0
    for index, user in enumerate(users):
        start = float(user["start_time"])
        end = float(user["end_time"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("invalid audio interval")
        if previous_end is not None and abs(start - previous_end) > 2e-4:
            raise ValueError("noncontiguous audio intervals")
        duration = end - start
        if index + 1 < len(users):
            grid_units = duration / 0.640
            if abs(grid_units - round(grid_units)) > 2e-3:
                raise ValueError("non-final interval is not on 640-ms grid")
        if not str(assistants[index].get("text") or "").strip():
            raise ValueError("empty target chunk")
        total_duration += duration
        previous_end = end
    return {
        "turns": len(users),
        "duration_seconds": total_duration,
        "max_target_lead": float(causal.get("max_target_lead", 0.0)),
        "asr_pass_votes": int(causal.get("asr_pass_votes", 0)),
        "alignment_score": float(causal.get("alignment_score", 0.0)),
        "boundary_delay_ms": int(causal.get("boundary_delay_ms", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--direction", choices=sorted(PROMPTS), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=8)
    args = parser.parse_args()

    counts = collections.Counter()
    sums = collections.Counter()
    sample_rows = []
    for shard in read_list(args.input_list):
        with shard.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"blank line: {shard}:{line_number}")
                record = json.loads(line)
                metrics = validate_record(record, args.direction)
                counts["records"] += 1
                counts[f"turns_{metrics['turns']}"] += 1
                sums["turns"] += metrics["turns"]
                sums["duration_seconds"] += metrics["duration_seconds"]
                sums["max_target_lead"] += metrics["max_target_lead"]
                sums["asr_pass_votes"] += metrics["asr_pass_votes"]
                if len(sample_rows) < args.sample_count:
                    sample_rows.append(
                        {
                            "key": record.get("key"),
                            "direction": args.direction,
                            "metrics": metrics,
                            "turns": [
                                {
                                    "audio_start": user["start_time"],
                                    "audio_end": user["end_time"],
                                    "target": assistant.get("text"),
                                }
                                for user, assistant in zip(
                                    record["messages"][1::2],
                                    record["messages"][2::2],
                                )
                            ],
                        }
                    )
                if args.max_records and counts["records"] >= args.max_records:
                    break
        if args.max_records and counts["records"] >= args.max_records:
            break

    total = counts["records"]
    if total == 0:
        raise ValueError("no records audited")
    report = {
        "direction": args.direction,
        "records": total,
        "mean_turns": sums["turns"] / total,
        "mean_duration_seconds": sums["duration_seconds"] / total,
        "mean_max_target_lead": sums["max_target_lead"] / total,
        "mean_asr_pass_votes": sums["asr_pass_votes"] / total,
        "turn_histogram": {
            key.removeprefix("turns_"): value
            for key, value in sorted(counts.items())
            if key.startswith("turns_")
        },
        "validation": [
            "direction prompt and language metadata",
            "causal_clean_v1 metadata",
            "contiguous source audio intervals",
            "640-ms non-final interval grid",
            "non-empty target chunks",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.samples.open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
