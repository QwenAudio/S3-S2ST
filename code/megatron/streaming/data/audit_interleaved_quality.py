#!/usr/bin/env python3
"""Audit unpacked streaming S2TT JSONL without reading audio.

The report is intentionally conservative: it separates hard structural errors
from suspicious-content counters and emits deterministic reservoir samples for
choosing cleaning thresholds.  It never mutates the input dataset.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import random
import re
from pathlib import Path


QUANTILE_FIELDS = (
    "turns",
    "total_duration",
    "min_chunk_duration",
    "max_chunk_duration",
    "total_speech_tokens",
    "target_chars",
    "target_language_ratio",
    "target_units_per_second",
)
CONTROL_RE = re.compile(r"<\|[^>]+\|>")
URL_RE = re.compile(r"(?:https?://|www\.)", re.I)
LATIN_RE = re.compile(r"[A-Za-z]")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
REPEAT_RE = re.compile(r"(.)\1{15,}")


def _quantiles(values):
    if not values:
        return {}
    values = sorted(values)
    out = {}
    for q in (0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999, 1):
        index = round(q * (len(values) - 1))
        out[f"p{q * 100:g}"] = values[index]
    return out


def _reservoir_add(samples, field, value, rng, seen, limit):
    seen[field] += 1
    bucket = samples[field]
    if len(bucket) < limit:
        bucket.append(value)
        return
    index = rng.randrange(seen[field])
    if index < limit:
        bucket[index] = value


def _extract_uri(text):
    start = text.find("<|startofspeech|>!")
    end = text.find("<|endofspeech|>", start + 1)
    if start < 0 or end < 0:
        return None
    return text[start + len("<|startofspeech|>!") : end]


def _audit_shard(job):
    shard_index, path_string, direction, reservoir_per_shard = job
    path = Path(path_string)
    counters = collections.Counter()
    samples = {field: [] for field in QUANTILE_FIELDS}
    sample_seen = collections.Counter()
    examples = collections.defaultdict(list)
    rng = random.Random(0x51A2 + shard_index)

    def flag(reason, key):
        counters[reason] += 1
        if len(examples[reason]) < 3:
            examples[reason].append(key)

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            counters["records"] += 1
            try:
                record = json.loads(line)
            except Exception:
                flag("hard:invalid_json", f"{path}:{line_number}")
                continue
            key = str(record.get("key") or f"{path}:{line_number}")
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                flag("hard:missing_messages", key)
                continue
            roles = [message.get("role") for message in messages]
            expected = ["system"] + [role for _ in range((len(messages) - 1) // 2) for role in ("user", "assistant")]
            if len(messages) % 2 == 0 or roles != expected:
                flag("hard:invalid_role_order", key)
                continue
            users = messages[1::2]
            assistants = messages[2::2]
            turns = len(users)
            durations = []
            speech_tokens = []
            uris = []
            segment_ids = []
            previous_end = None
            hard_failed = False
            for user in users:
                try:
                    start = float(user["start_time"])
                    end = float(user["end_time"])
                    speech_length = int(user["speech_length"])
                except Exception:
                    flag("hard:invalid_user_fields", key)
                    hard_failed = True
                    break
                duration = end - start
                durations.append(duration)
                speech_tokens.append(speech_length)
                uri = _extract_uri(str(user.get("text", "")))
                if not uri:
                    flag("hard:missing_audio_uri", key)
                    hard_failed = True
                    break
                uris.append(uri)
                segment_ids.append(str(user.get("segment_id", "")))
                if duration <= 0 or not math.isfinite(duration):
                    flag("hard:nonpositive_duration", key)
                    hard_failed = True
                expected_tokens = max(1, int(math.ceil(duration * 12.5 - 1e-6)))
                if speech_length <= 0:
                    flag("hard:nonpositive_speech_length", key)
                    hard_failed = True
                elif abs(speech_length - expected_tokens) > 1:
                    flag("hard:speech_length_mismatch", key)
                    hard_failed = True
                if previous_end is not None and abs(start - previous_end) > 2e-4:
                    flag("hard:noncontiguous_timing", key)
                    hard_failed = True
                previous_end = end
            if hard_failed:
                continue
            if len(set(uris)) != 1:
                flag("hard:multiple_audio_uris", key)
                continue
            if len(set(segment_ids)) != 1 or not segment_ids[0]:
                flag("hard:multiple_or_empty_segment_ids", key)
                continue
            expected_direction = str((record.get("streaming") or {}).get("direction", ""))
            if expected_direction != direction:
                flag("hard:direction_mismatch", key)
                continue

            target = "".join(str(message.get("text", "")) for message in assistants)
            if not target.strip():
                flag("hard:empty_target", key)
                continue
            if CONTROL_RE.search(target):
                flag("hard:control_token_in_target", key)
                continue

            total_duration = sum(durations)
            total_tokens = sum(speech_tokens)
            latin = len(LATIN_RE.findall(target))
            cjk = len(CJK_RE.findall(target))
            language_denominator = latin + cjk
            language_numerator = latin if direction == "zh2en" else cjk
            language_ratio = language_numerator / max(1, language_denominator)
            units = len(WORD_RE.findall(target)) if direction == "zh2en" else cjk
            rate = units / max(total_duration, 1e-9)
            values = {
                "turns": turns,
                "total_duration": total_duration,
                "min_chunk_duration": min(durations),
                "max_chunk_duration": max(durations),
                "total_speech_tokens": total_tokens,
                "target_chars": len(target),
                "target_language_ratio": language_ratio,
                "target_units_per_second": rate,
            }
            for field, value in values.items():
                _reservoir_add(samples, field, value, rng, sample_seen, reservoir_per_shard)

            counters["valid_structure"] += 1
            if durations[-1] < 0.32:
                flag("suspicious:final_chunk_lt_320ms", key)
            if min(durations) < 0.08:
                flag("suspicious:any_chunk_lt_80ms", key)
            if max(durations) > 10.0:
                flag("suspicious:any_chunk_gt_10s", key)
            if total_duration < 0.64:
                flag("suspicious:total_duration_lt_640ms", key)
            if total_duration > 60.0:
                flag("suspicious:total_duration_gt_60s", key)
            if turns > 64:
                flag("suspicious:turns_gt_64", key)
            if language_ratio < 0.5:
                flag("suspicious:target_language_ratio_lt_0.5", key)
            if rate < 0.05:
                flag("suspicious:target_rate_lt_0.05", key)
            max_rate = 8.0 if direction == "zh2en" else 15.0
            if rate > max_rate:
                flag("suspicious:target_rate_extreme", key)
            if URL_RE.search(target):
                flag("suspicious:url_in_target", key)
            if REPEAT_RE.search(target):
                flag("suspicious:character_run_ge_16", key)
            if any(not str(message.get("text", "")).strip() for message in assistants):
                flag("suspicious:empty_target_chunk", key)

    return {
        "path": str(path),
        "counters": dict(counters),
        "samples": samples,
        "sample_seen": dict(sample_seen),
        "examples": dict(examples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--direction", choices=("zh2en", "en2zh"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reservoir-per-shard", type=int, default=2000)
    args = parser.parse_args()

    paths = [line.strip() for line in args.input_list.read_text().splitlines() if line.strip()]
    jobs = [(index, path, args.direction, args.reservoir_per_shard) for index, path in enumerate(paths)]
    counters = collections.Counter()
    combined_samples = {field: [] for field in QUANTILE_FIELDS}
    examples = collections.defaultdict(list)
    with mp.Pool(processes=min(args.workers, len(jobs))) as pool:
        for index, result in enumerate(pool.imap_unordered(_audit_shard, jobs), 1):
            counters.update(result["counters"])
            for field in QUANTILE_FIELDS:
                combined_samples[field].extend(result["samples"][field])
            for reason, keys in result["examples"].items():
                examples[reason].extend(keys[: max(0, 10 - len(examples[reason]))])
            if index % 16 == 0 or index == len(jobs):
                print(f"audited {index}/{len(jobs)} shards", flush=True)

    report = {
        "direction": args.direction,
        "input_list": str(args.input_list.resolve()),
        "shards": len(paths),
        "counters": dict(counters.most_common()),
        "quantiles_from_deterministic_reservoir": {
            field: _quantiles(values) for field, values in combined_samples.items()
        },
        "reservoir_sizes": {field: len(values) for field, values in combined_samples.items()},
        "examples": dict(examples),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
