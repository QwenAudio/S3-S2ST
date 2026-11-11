#!/usr/bin/env python3
"""Deterministically interleave balanced S2ST rows into mixed-direction shards.

The input may be grouped by translation direction. Rows are distributed in
round-robin order across output shards; because the global counts are exactly
balanced, every output shard receives nearly identical counts from both
directions. The normal S2ST packer subsequently shuffles each mixed shard
before forming packed bins, matching Hejia's shuffle/split/pack ordering.
"""

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path


SRC_RE = re.compile(br'"src_lang"\s*:\s*"([^"]+)"')
TGT_RE = re.compile(br'"tgt_lang"\s*:\s*"([^"]+)"')
EXPECTED = {"EN-ZH", "ZH-EN"}


def pair_of(line: bytes) -> str:
    src = SRC_RE.search(line)
    tgt = TGT_RE.search(line)
    if src is None or tgt is None:
        raise ValueError("missing src_lang/tgt_lang")
    return f"{src.group(1).decode().upper()}-{tgt.group(1).decode().upper()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-shards", type=int, default=256)
    parser.add_argument("--progress-interval", type=int, default=1_000_000)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    sources = [
        Path(line.strip())
        for line in open(args.data_list, encoding="utf-8")
        if line.strip()
    ]
    if not sources:
        raise ValueError("input data.list is empty")
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input shards: {missing[:5]}")

    output = Path(args.output).resolve()
    building = output.with_name(output.name + ".building")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)

    destinations = [building / f"data.{i:05d}.jsonl" for i in range(args.num_shards)]
    writers = [open(path, "wb", buffering=1024 * 1024) for path in destinations]
    totals = Counter()
    per_shard = [Counter() for _ in destinations]
    row_index = 0
    try:
        for source_index, source in enumerate(sources, 1):
            with source.open("rb") as reader:
                for line_number, line in enumerate(reader, 1):
                    if not line.strip():
                        continue
                    try:
                        pair = pair_of(line)
                    except Exception as exc:
                        raise ValueError(f"{source}:{line_number}: {exc}") from exc
                    if pair not in EXPECTED:
                        raise ValueError(f"unexpected direction {pair} at {source}:{line_number}")
                    shard = row_index % args.num_shards
                    writers[shard].write(line)
                    totals[pair] += 1
                    per_shard[shard][pair] += 1
                    row_index += 1
                    if row_index % args.progress_interval == 0:
                        print(
                            f"mixed rows={row_index} files={source_index}/{len(sources)} "
                            f"counts={dict(totals)}",
                            flush=True,
                        )
    finally:
        for writer in writers:
            writer.close()

    if set(totals) != EXPECTED or len(set(totals.values())) != 1:
        raise RuntimeError(f"input/output directions are not exactly balanced: {dict(totals)}")
    for index, counts in enumerate(per_shard):
        if set(counts) != EXPECTED or abs(counts["EN-ZH"] - counts["ZH-EN"]) > 1:
            raise RuntimeError(f"mixed shard {index} is not balanced: {dict(counts)}")

    data_list = building / "data.list"
    with data_list.open("w", encoding="utf-8") as writer:
        # Write final paths, not temporary .building paths.
        for path in destinations:
            writer.write(str(output / path.name) + "\n")
        writer.flush()
        os.fsync(writer.fileno())
    stats = {
        "source_data_list": str(Path(args.data_list).resolve()),
        "method": "deterministic global round-robin mix before per-shard shuffle/packing",
        "num_input_shards": len(sources),
        "num_output_shards": args.num_shards,
        "output_counts": dict(sorted(totals.items())),
        "per_shard_min": {
            pair: min(counts[pair] for counts in per_shard) for pair in sorted(EXPECTED)
        },
        "per_shard_max": {
            pair: max(counts[pair] for counts in per_shard) for pair in sorted(EXPECTED)
        },
    }
    with (building / "mix_stats.json").open("w", encoding="utf-8") as writer:
        json.dump(stats, writer, indent=2, ensure_ascii=False)
        writer.write("\n")
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(building, output)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
