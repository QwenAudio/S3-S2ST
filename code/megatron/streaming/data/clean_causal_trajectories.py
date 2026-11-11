#!/usr/bin/env python3
"""Build a low-cost, direction-specific causal streaming S2TT dataset.

This first-pass cleaner intentionally avoids a new neural alignment dependency.
It joins the original bilingual mapping, LLM chunk labels, and source forced
alignment, then filters with:
  * LLM monotonicity/review metadata;
  * agreement between the source transcript and independent ASR engines;
  * timestamp/alignment sanity;
  * a direction-specific target-prefix lead proxy;
  * the existing structural streaming cleaner.

Non-final WRITE boundaries can be delayed by one 640-ms grid as a conservative
causal relabeling. The original artifacts are never modified.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streaming.data.build_interleaved_data import (  # noqa: E402
    GRID_SECONDS,
    ShardedWriter,
    make_absolute,
    mapping_pair_id,
    normalized,
    prefix_end_times,
    progress,
    rows,
    speech_token_length,
    split_name,
    timestamp_entries,
)
from streaming.data.clean_interleaved_data import clean_record  # noqa: E402


VERSION = "causal_clean_v1"
ASR_ENGINES = (
    "asr_fireredasr",
    "asr_sensevoice_small",
    "asr_whisper",
    "asr_qwen3asr",
)
EN_UNITS = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
ZH_OR_ALNUM = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z0-9]", re.IGNORECASE)


def source_target_keys(direction: str) -> tuple[str, str]:
    return ("en", "zh") if direction == "en2zh" else ("zh", "en")


def text_units(text: str, language: str) -> list[str]:
    value = str(text).casefold()
    if language == "en":
        return EN_UNITS.findall(value)
    return ZH_OR_ALNUM.findall(value)


def edit_distance(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row_index, left_value in enumerate(left, 1):
        current = [row_index]
        for column_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, language: str) -> float:
    ref = text_units(reference, language)
    hyp = text_units(hypothesis, language)
    if not ref:
        return math.inf
    return edit_distance(ref, hyp) / len(ref)


def extract_asr_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("text_org", "text", "text_punc"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def segment_duration(mapping: dict, source_key: str) -> tuple[float, float]:
    start = float(mapping.get(f"{source_key}_start_time", 0.0))
    end = float(mapping.get(f"{source_key}_end_time", -1.0))
    if end < 0:
        duration = float(mapping.get("source_duration_seconds") or 0.0)
        if duration <= 0:
            raise ValueError("source_duration_unavailable")
        end = start + duration
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("nonpositive_source_duration")
    return start, end


def label_quality(label: dict, min_confidence: float) -> dict:
    quality = label.get("quality") or {}
    try:
        monotonicity = int(quality.get("monotonicity", 999))
    except (TypeError, ValueError):
        monotonicity = 999
    try:
        confidence = float(quality.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        bad_boundaries = int(quality.get("review_bad_boundaries", 0) or 0)
    except (TypeError, ValueError):
        bad_boundaries = 999
    review_applied = bool(quality.get("review_applied", False))
    review_failed = bool(quality.get("review_failed", False))
    needs_review = bool(quality.get("needs_review", False))
    if review_failed:
        raise ValueError("llm_review_failed")
    if monotonicity != 0:
        raise ValueError("llm_nonmonotonic")
    if bad_boundaries:
        raise ValueError("llm_bad_boundaries")
    reviewed_ok = review_applied and not review_failed
    if not reviewed_ok and confidence < min_confidence:
        raise ValueError("llm_low_confidence")
    return {
        "monotonicity": monotonicity,
        "confidence": confidence,
        "review_applied": review_applied,
        "needs_review": needs_review,
    }


def chunk_parts(mapping: dict, label: dict, direction: str) -> tuple[list[str], list[str]]:
    source_key, target_key = source_target_keys(direction)
    chunks = label.get("chunks") or []
    if not isinstance(chunks, list) or len(chunks) < 2:
        raise ValueError("fewer_than_two_chunks")
    if any(not isinstance(chunk, (list, tuple)) or len(chunk) < 2 for chunk in chunks):
        raise ValueError("malformed_llm_chunks")
    source_parts = (
        [str(chunk[1]) for chunk in chunks]
        if direction == "en2zh"
        else [str(chunk[0]) for chunk in chunks]
    )
    target_parts = (
        [str(chunk[0]) for chunk in chunks]
        if direction == "en2zh"
        else [str(chunk[1]) for chunk in chunks]
    )
    if "".join(source_parts) != str(mapping[f"{source_key}_text"]):
        raise ValueError("source_chunks_do_not_reconstruct_mapping")
    if "".join(target_parts) != str(mapping[f"{target_key}_text"]):
        raise ValueError("target_chunks_do_not_reconstruct_mapping")
    if any(not normalized(part) for part in source_parts):
        raise ValueError("empty_source_chunk")
    if any(not normalized(part) for part in target_parts):
        raise ValueError("empty_target_chunk")
    return source_parts, target_parts


def target_lead_proxy(
    source_parts: list[str],
    target_parts: list[str],
    source_language: str,
    target_language: str,
) -> float:
    source_sizes = [len(text_units(part, source_language)) for part in source_parts]
    target_sizes = [len(text_units(part, target_language)) for part in target_parts]
    source_total = sum(source_sizes)
    target_total = sum(target_sizes)
    if source_total <= 0 or target_total <= 0:
        raise ValueError("empty_language_units")
    source_prefix = target_prefix = 0
    leads = []
    for index in range(len(source_parts) - 1):
        source_prefix += source_sizes[index]
        target_prefix += target_sizes[index]
        leads.append(target_prefix / target_total - source_prefix / source_total)
    return max(leads, default=0.0)


def alignment_quality(
    mapping: dict,
    segment: dict,
    source_key: str,
    asr_error_threshold: float,
    min_asr_votes: int,
    min_alignment_score: float,
    max_zero_duration_ratio: float,
    min_speech_span_ratio: float,
) -> dict:
    reference = str(mapping[f"{source_key}_text"])
    gemini = segment.get("asr_gemini") or {}
    alignment = gemini.get("alignments") or {}
    try:
        alignment_score = float(alignment.get("score"))
    except (TypeError, ValueError):
        raise ValueError("missing_alignment_score")
    if alignment_score < min_alignment_score:
        raise ValueError("low_alignment_score")

    entries = timestamp_entries(segment)
    if not entries:
        raise ValueError("missing_source_timestamps")
    joined = "".join(text for text, _start, _end in entries)
    if normalized(joined) != normalized(reference):
        raise ValueError("timestamps_do_not_reconstruct_source")

    start, end = segment_duration(mapping, source_key)
    absolute = make_absolute(entries, start, end)
    previous_start = -math.inf
    zero_duration = 0
    max_duration = 0.0
    for _text, token_start, token_end in absolute:
        if not math.isfinite(token_start) or not math.isfinite(token_end):
            raise ValueError("nonfinite_timestamp")
        if token_start + 1e-4 < previous_start:
            raise ValueError("nonmonotonic_timestamp_start")
        if token_end + 1e-4 < token_start:
            raise ValueError("negative_token_duration")
        duration = max(0.0, token_end - token_start)
        zero_duration += duration <= 1e-4
        max_duration = max(max_duration, duration)
        previous_start = token_start
    zero_ratio = zero_duration / len(absolute)
    if zero_ratio > max_zero_duration_ratio:
        raise ValueError("too_many_zero_duration_tokens")
    if max_duration > 2.0:
        raise ValueError("token_duration_gt_2s")
    speech_span = max(0.0, absolute[-1][2] - absolute[0][1])
    span_ratio = speech_span / (end - start)
    if span_ratio < min_speech_span_ratio:
        raise ValueError("source_speech_span_too_short")

    gemini_text = extract_asr_text(gemini)
    gemini_error = error_rate(reference, gemini_text, source_key)
    if gemini_error > min(0.05, asr_error_threshold):
        raise ValueError("gemini_transcript_mismatch")

    engine_errors: dict[str, float] = {}
    passing_normalized: set[str] = set()
    available_normalized: set[str] = set()
    for engine in ASR_ENGINES:
        hypothesis = extract_asr_text(segment.get(engine))
        if not hypothesis:
            continue
        normalized_hypothesis = "".join(text_units(hypothesis, source_key))
        if not normalized_hypothesis:
            continue
        available_normalized.add(normalized_hypothesis)
        value = error_rate(reference, hypothesis, source_key)
        engine_errors[engine] = value
        if value <= asr_error_threshold:
            passing_normalized.add(normalized_hypothesis)
    if len(available_normalized) < min_asr_votes:
        raise ValueError("insufficient_independent_asr")
    if len(passing_normalized) < min_asr_votes:
        raise ValueError("source_asr_disagreement")

    return {
        "entries": entries,
        "alignment_score": alignment_score,
        "zero_duration_ratio": zero_ratio,
        "speech_span_ratio": span_ratio,
        "asr_available": len(available_normalized),
        "asr_pass_votes": len(passing_normalized),
        "asr_median_error": statistics.median(engine_errors.values()),
        "asr_errors": engine_errors,
    }


def delayed_quantized_boundaries(
    end_times: list[float],
    segment_start: float,
    segment_end: float,
    boundary_delay_ms: int,
) -> list[float]:
    delay = boundary_delay_ms / 1000.0
    quantized = [
        min(
            segment_end,
            segment_start
            + math.ceil(max(0.001, value - segment_start) / GRID_SECONDS)
            * GRID_SECONDS,
        )
        for value in end_times
    ]
    for index in range(len(quantized) - 1):
        quantized[index] = min(segment_end, quantized[index] + delay)
    quantized[-1] = segment_end
    return quantized


def build_causal_record(
    mapping: dict,
    label: dict,
    entries: list[tuple[str, float, float]],
    direction: str,
    boundary_delay_ms: int,
    quality_metadata: dict,
) -> dict:
    source_key, target_key = source_target_keys(direction)
    source_parts, target_parts = chunk_parts(mapping, label, direction)
    segment_start, segment_end = segment_duration(mapping, source_key)
    ends = prefix_end_times(
        source_parts,
        make_absolute(entries, segment_start, segment_end),
    )
    boundaries = delayed_quantized_boundaries(
        ends, segment_start, segment_end, boundary_delay_ms
    )
    collapsed: list[tuple[float, str]] = []
    for boundary, target in zip(boundaries, target_parts):
        if collapsed and boundary <= collapsed[-1][0] + 1e-6:
            collapsed[-1] = (collapsed[-1][0], collapsed[-1][1] + target)
        else:
            collapsed.append((boundary, target))
    if len(collapsed) < 2:
        raise ValueError("delay_collapsed_to_one_turn")

    pair_id = mapping_pair_id(mapping)
    segment_id = str(mapping[f"{source_key}_segment_wav_id"])
    audio_path = str(mapping[f"{source_key}_audio_path"])
    prompt = "Translate into Chinese:" if direction == "en2zh" else "Translate into English:"
    system_text = "You are a helpful assistant."
    messages = [
        {
            "role": "system",
            "text": system_text,
            "content": system_text,
            "mode": "tc",
            "audio": "",
            "speech_length": 0,
        }
    ]
    previous = segment_start
    for turn, (boundary, target) in enumerate(collapsed):
        duration = boundary - previous
        final = turn + 1 == len(collapsed)
        if duration <= 0:
            raise ValueError("nonincreasing_delayed_boundary")
        if duration < GRID_SECONDS * 0.90 and not final:
            raise ValueError("sub640ms_nonfinal_audio_turn")
        user_text = (
            (prompt if turn == 0 else "")
            + f"<|startofspeech|>!{audio_path}<|endofspeech|>"
        )
        messages.append(
            {
                "role": "user",
                "text": user_text,
                "content": user_text,
                "mode": "tc",
                "audio": "",
                "speech_length": speech_token_length(duration),
                "start_time": round(previous, 6),
                "end_time": round(boundary, 6),
                "segment_id": segment_id,
                "slice_mode": "absolute",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "text": target,
                "content": target,
                "mode": "t",
                "audio": "",
                "speech_length": 0,
            }
        )
        previous = boundary

    record = {
        "key": pair_id,
        "messages": messages,
        "task": "streaming_s2tt",
        "src_lang": source_key.upper(),
        "tgt_lang": target_key.upper(),
        "streaming": {
            "decision_ms": 640,
            "interval_audio_tokens": 8,
            "end_token": "<|im_end|>",
            "direction": direction,
            "source_segment_id": segment_id,
            "causal_clean": {
                "version": VERSION,
                "boundary_delay_ms": boundary_delay_ms,
                **quality_metadata,
            },
        },
    }
    clean_record(record, direction)
    if len(record["messages"]) < 5:
        raise ValueError("structural_clean_collapsed_to_one_turn")
    return record


def ingest_mapping(path: Path, direction: str, connection: sqlite3.Connection, limit: int) -> tuple[int, set[str], set[str]]:
    source_key, _ = source_target_keys(direction)
    connection.execute(
        "CREATE TABLE mapping(pair_id TEXT PRIMARY KEY, source_sid TEXT NOT NULL, payload TEXT NOT NULL)"
    )
    batch = []
    source_ids: set[str] = set()
    pair_ids: set[str] = set()
    count = 0
    for row in rows(path):
        pair_id = mapping_pair_id(row)
        source_id = str(row[f"{source_key}_segment_wav_id"])
        batch.append(
            (
                pair_id,
                source_id,
                json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            )
        )
        source_ids.add(source_id)
        pair_ids.add(pair_id)
        count += 1
        progress("mapping", count)
        if len(batch) >= 10000:
            connection.executemany("INSERT OR REPLACE INTO mapping VALUES (?,?,?)", batch)
            batch.clear()
        if limit and count >= limit:
            break
    if batch:
        connection.executemany("INSERT OR REPLACE INTO mapping VALUES (?,?,?)", batch)
    connection.execute("CREATE INDEX mapping_source_sid ON mapping(source_sid)")
    connection.commit()
    return count, source_ids, pair_ids


def ingest_labels(path: Path, needed: set[str], connection: sqlite3.Connection) -> int:
    connection.execute("CREATE TABLE labels(pair_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    batch = []
    count = 0
    for row in rows(path):
        pair_id = str(row.get("pair_id") or "")
        if pair_id not in needed:
            continue
        batch.append(
            (pair_id, json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        )
        count += 1
        if len(batch) >= 10000:
            connection.executemany("INSERT OR REPLACE INTO labels VALUES (?,?)", batch)
            batch.clear()
        if count >= len(needed):
            break
    if batch:
        connection.executemany("INSERT OR REPLACE INTO labels VALUES (?,?)", batch)
    connection.commit()
    return count


def ingest_alignments(path: Path, needed: set[str], connection: sqlite3.Connection) -> tuple[int, int]:
    connection.execute("CREATE TABLE align(segment_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    batch = []
    found: set[str] = set()
    without_timestamps = 0
    for row in rows(path):
        for segment in row.get("segments") or []:
            segment_id = str(segment.get("segment_wav_id") or "")
            if segment_id not in needed:
                continue
            if timestamp_entries(segment):
                batch.append(
                    (
                        segment_id,
                        json.dumps(segment, ensure_ascii=False, separators=(",", ":")),
                    )
                )
                found.add(segment_id)
            else:
                without_timestamps += 1
        if len(batch) >= 10000:
            connection.executemany("INSERT OR REPLACE INTO align VALUES (?,?)", batch)
            batch.clear()
        if len(found) >= len(needed):
            break
    if batch:
        connection.executemany("INSERT OR REPLACE INTO align VALUES (?,?)", batch)
    connection.commit()
    return len(found), without_timestamps


def safe_unlink_sqlite(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            candidate.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["en2zh", "zh2en"], required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--text-labels", type=Path, required=True)
    parser.add_argument("--word-alignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--train-shards", type=int, default=128)
    parser.add_argument("--valid-shards", type=int, default=4)
    parser.add_argument("--valid-basis-points", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-label-confidence", type=float, default=90.0)
    parser.add_argument("--min-alignment-score", type=float, default=95.0)
    parser.add_argument("--min-asr-votes", type=int, default=2)
    parser.add_argument("--asr-error-threshold", type=float)
    parser.add_argument("--max-zero-duration-ratio", type=float, default=0.05)
    parser.add_argument("--min-speech-span-ratio", type=float, default=0.35)
    parser.add_argument("--max-target-lead", type=float)
    parser.add_argument("--boundary-delay-ms", type=int, default=640)
    args = parser.parse_args()

    if args.boundary_delay_ms < 0 or args.boundary_delay_ms % 640:
        raise ValueError("--boundary-delay-ms must be a non-negative multiple of 640")
    if args.train_shards <= 0 or args.valid_shards <= 0:
        raise ValueError("shard counts must be positive")
    if not 0 < args.valid_basis_points < 10000:
        raise ValueError("--valid-basis-points must be in (0,10000)")
    if args.asr_error_threshold is None:
        args.asr_error_threshold = 0.25 if args.direction == "en2zh" else 0.12
    if args.max_target_lead is None:
        args.max_target_lead = 0.18 if args.direction == "en2zh" else 0.15

    for path in (args.mapping, args.text_labels, args.word_alignments):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.sqlite.parent.mkdir(parents=True, exist_ok=True)
    safe_unlink_sqlite(args.sqlite)

    connection = sqlite3.connect(args.sqlite)
    connection.execute("PRAGMA journal_mode=WAL")
    mapping_count, source_ids, pair_ids = ingest_mapping(
        args.mapping, args.direction, connection, args.limit
    )
    label_count = ingest_labels(args.text_labels, pair_ids, connection)
    del pair_ids
    aligned_count, without_timestamps = ingest_alignments(
        args.word_alignments, source_ids, connection
    )
    del source_ids

    writer = ShardedWriter(args.output_dir, args.train_shards, args.valid_shards)
    reject_handle = (args.output_dir / "rejects.jsonl").open("w", encoding="utf-8")
    rejected: collections.Counter[str] = collections.Counter()
    aggregate = collections.Counter()
    processed = written = 0
    query = (
        "SELECT m.payload,l.payload,a.payload FROM mapping m "
        "LEFT JOIN labels l USING(pair_id) "
        "LEFT JOIN align a ON a.segment_id=m.source_sid ORDER BY m.rowid"
    )
    source_key, target_key = source_target_keys(args.direction)
    try:
        for mapping_json, label_json, segment_json in connection.execute(query):
            processed += 1
            progress("causal-clean", processed)
            mapping = json.loads(mapping_json)
            pair_id = mapping_pair_id(mapping)
            try:
                if label_json is None:
                    raise ValueError("missing_text_label")
                if segment_json is None:
                    raise ValueError("missing_source_alignment")
                label = json.loads(label_json)
                segment = json.loads(segment_json)
                label_metrics = label_quality(label, args.min_label_confidence)
                source_parts, target_parts = chunk_parts(mapping, label, args.direction)
                lead = target_lead_proxy(
                    source_parts, target_parts, source_key, target_key
                )
                if lead > args.max_target_lead:
                    raise ValueError("target_prefix_leads_source")
                alignment_metrics = alignment_quality(
                    mapping,
                    segment,
                    source_key,
                    args.asr_error_threshold,
                    args.min_asr_votes,
                    args.min_alignment_score,
                    args.max_zero_duration_ratio,
                    args.min_speech_span_ratio,
                )
                quality_metadata = {
                    "label_confidence": label_metrics["confidence"],
                    "label_review_applied": label_metrics["review_applied"],
                    "max_target_lead": round(lead, 6),
                    "alignment_score": alignment_metrics["alignment_score"],
                    "asr_pass_votes": alignment_metrics["asr_pass_votes"],
                    "asr_median_error": round(alignment_metrics["asr_median_error"], 6),
                }
                record = build_causal_record(
                    mapping,
                    label,
                    alignment_metrics["entries"],
                    args.direction,
                    args.boundary_delay_ms,
                    quality_metadata,
                )
                split = split_name(pair_id, args.valid_basis_points)
                writer.write(split, pair_id, record)
                written += 1
                aggregate["total_turns"] += (len(record["messages"]) - 1) // 2
                aggregate["asr_pass_votes"] += alignment_metrics["asr_pass_votes"]
            except Exception as exc:
                reason = str(exc)
                rejected[reason] += 1
                reject_handle.write(
                    json.dumps(
                        {"pair_id": pair_id, "reason": reason},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    finally:
        reject_handle.close()
        writer.close()
        connection.close()

    summary = {
        "version": VERSION,
        "direction": args.direction,
        "mapping_rows": mapping_count,
        "text_label_rows": label_count,
        "source_alignments_with_timestamps": aligned_count,
        "source_alignment_segments_without_timestamps": without_timestamps,
        "processed_pairs": processed,
        "written_pairs": written,
        "retention_rate": written / processed if processed else 0.0,
        "split_counts": dict(writer.counts),
        "rejected_pairs": processed - written,
        "reject_reasons": dict(rejected.most_common()),
        "mean_turns": aggregate["total_turns"] / written if written else 0.0,
        "mean_asr_pass_votes": aggregate["asr_pass_votes"] / written if written else 0.0,
        "thresholds": {
            "min_label_confidence": args.min_label_confidence,
            "min_alignment_score": args.min_alignment_score,
            "min_asr_votes": args.min_asr_votes,
            "asr_error_threshold": args.asr_error_threshold,
            "max_zero_duration_ratio": args.max_zero_duration_ratio,
            "min_speech_span_ratio": args.min_speech_span_ratio,
            "max_target_lead": args.max_target_lead,
            "boundary_delay_ms": args.boundary_delay_ms,
        },
        "split_policy": {
            "valid_basis_points": args.valid_basis_points,
            "stable_hash": "sha1(pair_id)",
        },
        "sqlite": str(args.sqlite.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    (args.output_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
