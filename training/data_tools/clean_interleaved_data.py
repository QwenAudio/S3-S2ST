#!/usr/bin/env python3
"""Conservatively clean unpacked streaming S2TT records before exact packing."""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import re
from pathlib import Path


GRID_SECONDS = 0.640
CONTROL_RE = re.compile(r"<\|[^>]+\|>")
URL_RE = re.compile(r"(?:https?://|www\.)", re.I)
LATIN_RE = re.compile(r"[A-Za-z]")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
REPEAT_RE = re.compile(r"(.)\1{15,}")


def speech_token_length(seconds):
    frames = max(1, int(round(float(seconds) * 100)))
    length = 1 + (frames - 3 + 2) // 2
    length = 1 + (length - 3 + 2) // 2
    return max(1, (length - 1) // 2 + 1)


def _audio_uri(text):
    prefix = "<|startofspeech|>!"
    suffix = "<|endofspeech|>"
    start = text.find(prefix)
    end = text.find(suffix, start + len(prefix))
    return None if start < 0 or end < 0 else text[start + len(prefix) : end]


def _set_message_text(message, text):
    message["text"] = text
    message["content"] = text


def clean_record(record, direction, min_total_duration=0.64,
                 max_total_duration=60.0, max_chunk_duration=10.0,
                 merge_final_below=0.32, min_language_ratio=0.5,
                 min_target_rate=0.05, max_target_rate=None):
    """Return ``(record, actions)`` or raise ``ValueError(reason)``."""
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 == 0:
        raise ValueError("invalid_messages")
    expected_roles = ["system"] + [
        role for _ in range((len(messages) - 1) // 2)
        for role in ("user", "assistant")
    ]
    if [message.get("role") for message in messages] != expected_roles:
        raise ValueError("invalid_role_order")
    if str((record.get("streaming") or {}).get("direction", "")) != direction:
        raise ValueError("direction_mismatch")

    users = messages[1::2]
    uris, segment_ids = [], []
    previous_end = None
    for user in users:
        try:
            start = float(user["start_time"])
            end = float(user["end_time"])
            actual_length = int(user["speech_length"])
        except Exception as exc:
            raise ValueError("invalid_user_fields") from exc
        duration = end - start
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("nonpositive_duration")
        if previous_end is not None and abs(start - previous_end) > 2e-4:
            raise ValueError("noncontiguous_timing")
        if abs(actual_length - speech_token_length(duration)) > 1:
            raise ValueError("speech_length_mismatch")
        uri = _audio_uri(str(user.get("text", "")))
        if not uri:
            raise ValueError("missing_audio_uri")
        uris.append(uri)
        segment_ids.append(str(user.get("segment_id", "")))
        previous_end = end
    if len(set(uris)) != 1:
        raise ValueError("multiple_audio_uris")
    if len(set(segment_ids)) != 1 or not segment_ids[0]:
        raise ValueError("multiple_or_empty_segment_ids")

    actions = []
    final_duration = float(users[-1]["end_time"]) - float(users[-1]["start_time"])
    if len(users) > 1 and final_duration < merge_final_below:
        previous_user = messages[-4]
        previous_assistant = messages[-3]
        final_user = messages[-2]
        final_assistant = messages[-1]
        previous_user["end_time"] = final_user["end_time"]
        merged_duration = float(previous_user["end_time"]) - float(previous_user["start_time"])
        previous_user["speech_length"] = speech_token_length(merged_duration)
        merged_target = str(previous_assistant.get("text", "")) + str(final_assistant.get("text", ""))
        _set_message_text(previous_assistant, merged_target)
        del messages[-2:]
        actions.append("merged_final_chunk_lt_320ms")

    users = messages[1::2]
    assistants = messages[2::2]
    durations = [float(user["end_time"]) - float(user["start_time"]) for user in users]
    total_duration = sum(durations)
    if total_duration < min_total_duration:
        raise ValueError("total_duration_lt_640ms")
    if total_duration > max_total_duration:
        raise ValueError("total_duration_gt_60s")
    if max(durations) > max_chunk_duration:
        raise ValueError("chunk_duration_gt_10s")

    target = "".join(str(message.get("text", "")) for message in assistants)
    if not target.strip():
        raise ValueError("empty_target")
    if CONTROL_RE.search(target):
        raise ValueError("control_token_in_target")
    if URL_RE.search(target):
        raise ValueError("url_in_target")
    if REPEAT_RE.search(target):
        raise ValueError("character_run_ge_16")
    latin = len(LATIN_RE.findall(target))
    cjk = len(CJK_RE.findall(target))
    target_language = latin if direction == "zh2en" else cjk
    if target_language / max(1, latin + cjk) < min_language_ratio:
        raise ValueError("target_language_ratio_lt_0.5")
    target_units = len(WORD_RE.findall(target)) if direction == "zh2en" else cjk
    rate = target_units / total_duration
    if rate < min_target_rate:
        raise ValueError("target_rate_lt_0.05")
    if max_target_rate is None:
        max_target_rate = 8.0 if direction == "zh2en" else 15.0
    if rate > max_target_rate:
        raise ValueError("target_rate_extreme")
    return record, actions


def _clean_shard(job):
    index, source_string, destination_string, reject_string, direction, options = job
    source = Path(source_string)
    destination = Path(destination_string)
    reject_path = Path(reject_string)
    destination_tmp = destination.with_suffix(destination.suffix + ".tmp")
    reject_tmp = reject_path.with_suffix(reject_path.suffix + ".tmp")
    counters = collections.Counter()
    reject_examples = collections.defaultdict(list)
    with source.open(encoding="utf-8") as reader, \
            destination_tmp.open("w", encoding="utf-8") as writer, \
            reject_tmp.open("w", encoding="utf-8") as reject_writer:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            counters["input"] += 1
            try:
                record = json.loads(line)
                record, actions = clean_record(record, direction, **options)
            except Exception as exc:
                reason = str(exc)
                counters[f"rejected:{reason}"] += 1
                key = None
                try:
                    key = record.get("key")
                except Exception:
                    pass
                if len(reject_examples[reason]) < 3:
                    reject_examples[reason].append(key or f"{source}:{line_number}")
                reject_writer.write(json.dumps({
                    "source": str(source), "line": line_number,
                    "key": key, "reason": reason,
                }, ensure_ascii=False) + "\n")
                continue
            for action in actions:
                counters[f"modified:{action}"] += 1
            writer.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counters["kept"] += 1
    os.replace(destination_tmp, destination)
    os.replace(reject_tmp, reject_path)
    return index, dict(counters), dict(reject_examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direction", choices=("zh2en", "en2zh"), required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-total-duration", type=float, default=0.64)
    parser.add_argument("--max-total-duration", type=float, default=60.0)
    parser.add_argument("--max-chunk-duration", type=float, default=10.0)
    parser.add_argument("--merge-final-below", type=float, default=0.32)
    parser.add_argument("--min-language-ratio", type=float, default=0.5)
    parser.add_argument("--min-target-rate", type=float, default=0.05)
    parser.add_argument("--max-target-rate", type=float, default=None)
    args = parser.parse_args()

    sources = [Path(line.strip()) for line in args.input_list.read_text().splitlines() if line.strip()]
    if not sources:
        raise ValueError("empty input list")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    data_dir = args.output_dir / "data"
    reject_dir = args.output_dir / "rejects"
    data_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "min_total_duration": args.min_total_duration,
        "max_total_duration": args.max_total_duration,
        "max_chunk_duration": args.max_chunk_duration,
        "merge_final_below": args.merge_final_below,
        "min_language_ratio": args.min_language_ratio,
        "min_target_rate": args.min_target_rate,
        "max_target_rate": args.max_target_rate,
    }
    jobs = []
    destinations = []
    rejects = []
    for index, source in enumerate(sources):
        destination = (data_dir / f"data.{index:05d}.jsonl").resolve()
        reject_path = (reject_dir / f"rejects.{index:05d}.jsonl").resolve()
        destinations.append(destination)
        rejects.append(reject_path)
        jobs.append((index, str(source), str(destination), str(reject_path), args.direction, options))

    counters = collections.Counter()
    examples = collections.defaultdict(list)
    with mp.Pool(processes=min(args.workers, len(jobs))) as pool:
        for completed, (_index, shard_counters, shard_examples) in enumerate(
                pool.imap_unordered(_clean_shard, jobs), 1):
            counters.update(shard_counters)
            for reason, keys in shard_examples.items():
                examples[reason].extend(keys[: max(0, 10 - len(examples[reason]))])
            if completed % 16 == 0 or completed == len(jobs):
                print(f"cleaned {completed}/{len(jobs)} shards", flush=True)

    kept_paths = [path for path in destinations if path.stat().st_size]
    if not kept_paths:
        raise RuntimeError("cleaning removed every record")
    list_tmp = args.output_dir / "data.list.tmp"
    with list_tmp.open("w", encoding="utf-8") as stream:
        for path in kept_paths:
            stream.write(str(path) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(list_tmp, args.output_dir / "data.list")
    summary = {
        "cleaning_version": "streaming_s2tt_conservative_v1",
        "direction": args.direction,
        "input_list": str(args.input_list.resolve()),
        "output_data_list": str((args.output_dir / "data.list").resolve()),
        "options": options,
        "counters": dict(counters.most_common()),
        "kept_fraction": counters["kept"] / max(1, counters["input"]),
        "reject_examples": dict(examples),
        "reject_shards": [str(path) for path in rejects],
    }
    (args.output_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
