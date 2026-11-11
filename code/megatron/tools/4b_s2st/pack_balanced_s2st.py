#!/usr/bin/env python3
"""Pack a data.list with the 0803 FunASR S2S packer and a 4B tokenizer."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--funasr-root", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--cutoff-len", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    import ray
    from tqdm import tqdm

    funasr_root = str(Path(args.funasr_root).resolve())
    tokenizer_model = str(Path(args.tokenizer_model).resolve())
    sources = [Path(line.strip()) for line in open(args.data_list, encoding="utf-8") if line.strip()]
    if not sources:
        raise ValueError("balanced data.list is empty")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    @ray.remote(num_cpus=1)
    class Actor:
        def __init__(self, worker_index):
            import sys
            sys.path.insert(0, funasr_root)
            from offline_pack.format_converter import Converter
            from offline_pack.sft_packer import SFTDataPacker

            self.worker_index = worker_index
            self.converter = Converter(
                tokenizer_model_path=tokenizer_model,
                dataset_type="MinMo_S2S_Channels_SFT_Pack_Mega",
                packing_mode="s2s",
            )
            self.packer = SFTDataPacker(cutoff_len=args.cutoff_len)

        def pack(self, index, source, destination):
            import random
            random.seed(args.seed + index)
            stats = self.packer.pack_from_jsonl(
                input_file=source,
                output_file=destination,
                format_converter=self.converter,
                verbose=False,
                shuffle=True,
            )
            return index, destination, stats

    ray.init(log_to_driver=False)
    actors = [Actor.remote(index) for index in range(args.num_workers)]
    futures = []
    for index, source in enumerate(sources):
        destination = output / f"data.{index:05d}.jsonl"
        futures.append(
            actors[index % len(actors)].pack.remote(index, str(source), str(destination))
        )

    results = []
    try:
        with tqdm(total=len(futures), desc="packing") as progress:
            while futures:
                done, futures = ray.wait(futures, num_returns=1, timeout=15.0)
                if done:
                    results.extend(ray.get(done))
                    progress.update(len(done))
    finally:
        ray.shutdown()

    destinations = []
    total_original = 0
    total_bins = 0
    for index, destination, stats in sorted(results):
        path = Path(destination)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"packer produced an empty shard: {path}")
        destinations.append(path)
        total_original += int(stats["total_samples"])
        total_bins += int(stats["packed_bins"])

    list_tmp = output / "data.list.tmp"
    with list_tmp.open("w", encoding="utf-8") as writer:
        for destination in destinations:
            writer.write(str(destination) + "\n")
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(list_tmp, output / "data.list")
    summary = output / "packing_stats.txt"
    summary.write_text(
        f"source_shards={len(sources)}\npacked_shards={len(destinations)}\n"
        f"original_samples={total_original}\npacked_bins={total_bins}\n"
        f"cutoff_len={args.cutoff_len}\nseed={args.seed}\n",
        encoding="utf-8",
    )
    print(summary.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
