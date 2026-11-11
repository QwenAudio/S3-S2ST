#!/usr/bin/env python3
"""Create an exact-count EN-ZH / ZH-EN sample without modifying source data."""

import argparse
import concurrent.futures
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


SRC_RE = re.compile(br'"src_lang"\s*:\s*"([^"]+)"')
TGT_RE = re.compile(br'"tgt_lang"\s*:\s*"([^"]+)"')
EXPECTED = {"EN-ZH", "ZH-EN"}


def pair_of(line):
    src = SRC_RE.search(line)
    tgt = TGT_RE.search(line)
    if src is None or tgt is None:
        raise ValueError("missing src_lang/tgt_lang")
    return f"{src.group(1).decode().upper()}-{tgt.group(1).decode().upper()}"


def stat_file(path):
    counts = Counter()
    with open(path, "rb") as reader:
        for line_no, line in enumerate(reader, 1):
            try:
                counts[pair_of(line)] += 1
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return path, dict(counts)


def sample_file(task):
    index, source, destination, input_counts, keep_counts = task
    if sum(keep_counts.values()) == 0:
        return index, {}, None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    seen = Counter()
    written = Counter()
    with open(source, "rb") as reader, open(temporary, "wb") as writer:
        for line_no, line in enumerate(reader, 1):
            pair = pair_of(line)
            before = seen[pair]
            seen[pair] += 1
            keep = keep_counts.get(pair, 0)
            total = input_counts[pair]
            if (seen[pair] * keep) // total > (before * keep) // total:
                writer.write(line)
                written[pair] += 1
        writer.flush()
        os.fsync(writer.fileno())
    if dict(seen) != input_counts or any(
        written[pair] != keep for pair, keep in keep_counts.items()
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"sampling invariant failed for {source}: seen={dict(seen)}, "
            f"expected={input_counts}, written={dict(written)}, keep={keep_counts}"
        )
    os.replace(temporary, destination)
    return index, dict(written), destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    sources = [line.strip() for line in open(args.data_list, encoding="utf-8") if line.strip()]
    if not sources:
        raise ValueError("source data.list is empty")

    file_counts = []
    totals = Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (path, counts) in enumerate(pool.map(stat_file, sources), 1):
            file_counts.append(counts)
            totals.update(counts)
            if done % 100 == 0:
                print(f"stats {done}/{len(sources)}", flush=True)

    if set(totals) != EXPECTED:
        raise ValueError(f"expected exactly {sorted(EXPECTED)}, found {dict(totals)}")
    target = min(totals.values())

    # Allocate each direction's exact target count proportionally across files.
    cumulative = Counter()
    previous_kept = Counter()
    allocations = []
    for counts in file_counts:
        keep = {}
        for pair, count in counts.items():
            cumulative[pair] += count
            current_kept = cumulative[pair] * target // totals[pair]
            keep[pair] = current_kept - previous_kept[pair]
            previous_kept[pair] = current_kept
        allocations.append(keep)
    if any(previous_kept[pair] != target for pair in EXPECTED):
        raise RuntimeError(f"allocation did not reach target={target}: {dict(previous_kept)}")

    output = Path(args.output).resolve()
    jsonl_dir = output / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    width = max(5, len(str(len(sources))))
    for index, (source, counts, keep) in enumerate(zip(sources, file_counts, allocations)):
        destination = jsonl_dir / f"data.{index:0{width}d}.jsonl"
        tasks.append((index, source, destination, counts, keep))

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(sample_file, task) for task in tasks]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            if done % 100 == 0:
                print(f"sample {done}/{len(futures)}", flush=True)

    output_counts = Counter()
    destinations = []
    for index, counts, destination in sorted(results):
        output_counts.update(counts)
        if destination is not None:
            destinations.append(destination)
    if any(output_counts[pair] != target for pair in EXPECTED):
        raise RuntimeError(f"output is not balanced: {dict(output_counts)}, target={target}")

    list_tmp = output / "data.list.tmp"
    with list_tmp.open("w", encoding="utf-8") as writer:
        for destination in destinations:
            writer.write(str(destination) + "\n")
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(list_tmp, output / "data.list")

    stats = {
        "source_data_list": str(Path(args.data_list).resolve()),
        "input_counts": dict(sorted(totals.items())),
        "target_per_direction": target,
        "output_counts": dict(sorted(output_counts.items())),
        "output_shards": len(destinations),
        "selection": "deterministic proportional systematic sampling",
    }
    stats_tmp = output / "balance_stats.json.tmp"
    with stats_tmp.open("w", encoding="utf-8") as writer:
        json.dump(stats, writer, indent=2, ensure_ascii=False)
        writer.write("\n")
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(stats_tmp, output / "balance_stats.json")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
