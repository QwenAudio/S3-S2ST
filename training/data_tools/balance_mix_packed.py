#!/usr/bin/env python3
"""Build a deterministic, approximately balanced mixture of packed S2TT sets.

The training sampler consumes one packed JSONL line as one bin. By default all
bins are retained and randomly interleaved; exact 50:50 downsampling is optional.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterator


PROMPTS = {
    "zh2en": "Translate into English:",
    "en2zh": "Translate into Chinese:",
}


def read_data_list(path: Path) -> list[Path]:
    paths = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not paths:
        raise ValueError(f"empty data.list: {path}")
    missing = [str(item) for item in paths if not item.is_file() or item.stat().st_size == 0]
    if missing:
        raise ValueError(f"missing/empty packed shards from {path}: {missing[:5]}")
    return paths


def packed_bin_count(data_list: Path) -> int:
    summary = data_list.parent / "packing_summary.json"
    if summary.is_file():
        value = int(json.loads(summary.read_text(encoding="utf-8"))["packed_bins"])
        if value <= 0:
            raise ValueError(f"invalid packed_bins in {summary}: {value}")
        return value
    return sum(1 for path in read_data_list(data_list) for _ in path.open(encoding="utf-8"))


def sampled_indices(total: int, target: int, seed: int) -> set[int] | None:
    if target > total:
        raise ValueError(f"cannot select {target} from {total}")
    if target == total:
        return None
    return set(random.Random(seed).sample(range(total), target))


def detect_direction(value: object) -> str:
    if not isinstance(value, list) or not value:
        raise ValueError("packed line must be a non-empty JSON list")
    directions: set[str] = set()
    for record in value:
        if not isinstance(record, dict):
            raise ValueError("packed record must be an object")
        found = None
        for message in record.get("messages") or []:
            if message.get("role") != "user":
                continue
            text = str(message.get("text") or message.get("content") or "")
            for direction, prompt in PROMPTS.items():
                if text.startswith(prompt):
                    found = direction
                    break
            if found:
                break
        if found is None:
            raise ValueError("packed record has no direction prompt")
        directions.add(found)
    if len(directions) != 1:
        raise ValueError(f"packed bin mixes unexpected directions: {sorted(directions)}")
    return directions.pop()


def selected_lines(
    paths: list[Path],
    selected: set[int] | None,
    expected_total: int,
    expected_direction: str,
) -> Iterator[str]:
    seen = yielded = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                index = seen
                seen += 1
                if selected is not None and index not in selected:
                    continue
                if not line.strip():
                    raise ValueError(f"blank packed line: {path}:{line_no}")
                actual = detect_direction(json.loads(line))
                if actual != expected_direction:
                    raise ValueError(
                        f"direction mismatch at {path}:{line_no}: expected {expected_direction}, got {actual}"
                    )
                yielded += 1
                yield line if line.endswith("\n") else line + "\n"
    if seen != expected_total:
        raise ValueError(f"packing summary mismatch: declared {expected_total}, read {seen}")
    expected_yielded = expected_total if selected is None else len(selected)
    if yielded != expected_yielded:
        raise ValueError(f"selection mismatch: expected {expected_yielded}, yielded {yielded}")


def mix_packed(
    left_list: Path,
    right_list: Path,
    output_dir: Path,
    left_direction: str = "zh2en",
    right_direction: str = "en2zh",
    output_shards: int = 128,
    seed: int = 20260811,
    strategy: str = "all",
) -> dict:
    if left_direction == right_direction:
        raise ValueError("the two inputs must have different directions")
    if left_direction not in PROMPTS or right_direction not in PROMPTS:
        raise ValueError("directions must be zh2en/en2zh")
    if output_shards <= 0:
        raise ValueError("output_shards must be positive")
    if strategy not in {"all", "exact"}:
        raise ValueError("strategy must be all or exact")

    left_list = left_list.resolve()
    right_list = right_list.resolve()
    output_dir = output_dir.resolve()
    left_paths = read_data_list(left_list)
    right_paths = read_data_list(right_list)
    left_total = packed_bin_count(left_list)
    right_total = packed_bin_count(right_list)
    if min(left_total, right_total) <= 0:
        raise ValueError("no packed bins to mix")
    if strategy == "exact":
        left_target = right_target = min(left_total, right_total)
    else:
        left_target, right_target = left_total, right_total

    existing = list(output_dir.glob("data.*.jsonl")) if output_dir.exists() else []
    if existing or (output_dir / "data.list").exists():
        raise FileExistsError(f"refusing to overwrite existing mixed data in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_total = left_target + right_target
    shard_count = min(output_shards, selected_total)
    tmp_paths = [output_dir / f"data.{index:05d}.jsonl.tmp" for index in range(shard_count)]
    final_paths = [output_dir / f"data.{index:05d}.jsonl" for index in range(shard_count)]

    left_selected = sampled_indices(left_total, left_target, seed)
    right_selected = sampled_indices(right_total, right_target, seed + 1)
    left_iter = selected_lines(
        left_paths, left_selected, left_total, left_direction
    )
    right_iter = selected_lines(
        right_paths, right_selected, right_total, right_direction
    )
    order_rng = random.Random(seed + 2)
    handles = [path.open("w", encoding="utf-8") for path in tmp_paths]
    try:
        left_remaining, right_remaining = left_target, right_target
        output_index = 0
        while left_remaining or right_remaining:
            choose_left = right_remaining == 0 or (
                left_remaining > 0
                and order_rng.randrange(left_remaining + right_remaining) < left_remaining
            )
            line = next(left_iter if choose_left else right_iter)
            handles[output_index % shard_count].write(line)
            output_index += 1
            if choose_left:
                left_remaining -= 1
            else:
                right_remaining -= 1
        try:
            next(left_iter)
            raise ValueError("left iterator produced too many selected bins")
        except StopIteration:
            pass
        try:
            next(right_iter)
            raise ValueError("right iterator produced too many selected bins")
        except StopIteration:
            pass
    finally:
        for handle in handles:
            handle.close()

    for tmp, final in zip(tmp_paths, final_paths):
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError(f"empty mixed shard: {tmp}")
        os.replace(tmp, final)

    data_list_tmp = output_dir / "data.list.tmp"
    with data_list_tmp.open("w", encoding="utf-8") as handle:
        for path in final_paths:
            handle.write(str(path) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(data_list_tmp, output_dir / "data.list")

    summary = {
        "strategy": ("packed_bin_keep_all_random_interleave" if strategy == "all"
                     else "packed_bin_exact_50_50_without_replacement"),
        "seed": seed,
        "left": {
            "direction": left_direction,
            "data_list": str(left_list),
            "available_bins": left_total,
            "selected_bins": left_target,
            "sampling_fraction": left_target / left_total,
        },
        "right": {
            "direction": right_direction,
            "data_list": str(right_list),
            "available_bins": right_total,
            "selected_bins": right_target,
            "sampling_fraction": right_target / right_total,
        },
        "mixed_bins": selected_total,
        "direction_fraction": {
            left_direction: left_target / selected_total,
            right_direction: right_target / selected_total,
        },
        "output_shards": shard_count,
        "output_data_list": str(output_dir / "data.list"),
        "validation": "every selected packed line parsed; every record prompt matched its declared direction",
    }
    (output_dir / "mixing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-list", type=Path, required=True)
    parser.add_argument("--right-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-direction", choices=sorted(PROMPTS), default="zh2en")
    parser.add_argument("--right-direction", choices=sorted(PROMPTS), default="en2zh")
    parser.add_argument("--output-shards", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--strategy", choices=["all", "exact"], default="all")
    args = parser.parse_args()
    summary = mix_packed(
        args.left_list,
        args.right_list,
        args.output_dir,
        args.left_direction,
        args.right_direction,
        args.output_shards,
        args.seed,
        args.strategy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
