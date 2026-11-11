#!/usr/bin/env python3
"""Audit that EN->ZH mappings use real English input and TTS Chinese target."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=1_881_337)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    counts: collections.Counter[str] = collections.Counter()
    pair_ids: set[str] = set()
    invalid_examples: list[dict[str, object]] = []

    required = {
        "direction_en2zh": lambda row: str(row.get("direction", "")).upper() == "EN2ZH",
        "source_is_original": lambda row: row.get("source_is_original") is True,
        "target_is_tts": lambda row: row.get("target_is_tts") is True,
        "en_audio_path_present": lambda row: bool(row.get("en_audio_path")),
        "zh_audio_path_present": lambda row: bool(row.get("zh_audio_path")),
        "en_segment_id_present": lambda row: bool(row.get("en_segment_wav_id")),
        "zh_segment_id_present": lambda row: bool(row.get("zh_segment_wav_id")),
        "en_text_present": lambda row: bool(str(row.get("en_text") or "").strip()),
        "zh_text_present": lambda row: bool(str(row.get("zh_text") or "").strip()),
    }

    with args.mapping.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            counts["rows"] += 1
            pair_id = str(row.get("pair_id") or "")
            if not pair_id or pair_id in pair_ids:
                counts["unique_pair_id"] += 0
                if len(invalid_examples) < args.max_examples:
                    invalid_examples.append(
                        {"line": line_no, "pair_id": pair_id, "failed": ["unique_pair_id"]}
                    )
            else:
                pair_ids.add(pair_id)
                counts["unique_pair_id"] += 1

            failed = []
            for name, check in required.items():
                if check(row):
                    counts[name] += 1
                else:
                    failed.append(name)
            if failed and len(invalid_examples) < args.max_examples:
                invalid_examples.append({"line": line_no, "pair_id": pair_id, "failed": failed})

    expected = args.expected_pairs
    passed = counts["rows"] == expected and counts["unique_pair_id"] == expected
    passed = passed and all(counts[name] == expected for name in required)
    result = {
        "mapping": str(args.mapping.resolve()),
        "expected_pairs": expected,
        "rows": counts["rows"],
        "unique_pair_ids": counts["unique_pair_id"],
        "checks": {name: counts[name] for name in required},
        "semantics": {
            "model_input": "en_audio_path (original/real English speech)",
            "s2tt_supervision": "zh_text (translation associated with the TTS target)",
            "future_s2st_target": "zh_audio_path (TTS Chinese speech)",
        },
        "invalid_examples": invalid_examples,
        "passed": passed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
