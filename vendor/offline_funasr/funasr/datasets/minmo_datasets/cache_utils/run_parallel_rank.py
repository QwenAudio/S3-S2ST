#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import logging
import os
from typing import List, Dict, Any

from funasr.datasets.minmo_datasets.cache_utils.pre_cache_audio_data import process_jsonl
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _read_lines(path: str) -> List[str]:
    files: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            files.append(p)
    return files


def _process_one(jsonl_path: str, output_base: str, temp_base: str,
                 num_workers: int, slice_suffix: str, slice_workers: int, mode: str) -> Dict[str, int]:
    return process_jsonl(
        jsonl_path=jsonl_path,
        output_base=output_base,
        temp_base=temp_base,
        num_workers=num_workers,
        slice_suffix=slice_suffix,
        slice_workers=slice_workers,
        mode=mode,
    )


def main():
    parser = argparse.ArgumentParser(description="Process multiple JSONL files by rank with multiprocessing")
    parser.add_argument("file_list", type=str, help="Path to a text file containing JSONL paths (one per line)")
    parser.add_argument("output_base", type=str, help="Output base directory for final slices")
    parser.add_argument("temp_base", type=str, help="Temporary base directory for downloads and intermediate slices")
    parser.add_argument("--num-workers", type=int, default=8, help="Per-file process workers for slicing pipeline")
    parser.add_argument("--slice-workers", type=int, default=10, help="Per-line thread workers for slicing")
    parser.add_argument("--slice-suffix", type=str, default="wav", choices=["wav", "flac"], help="Slice audio format")
    parser.add_argument("--file-proc-workers", type=int, default=4, help="Number of files to process in parallel (processes)")
    parser.add_argument("--mode", type=str, default="slices", choices=["slices", "segments"], help="Whether to cut from messages or segments")

    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    all_files = _read_lines(args.file_list)
    assigned = [p for idx, p in enumerate(all_files) if idx % world_size == rank]

    logger.info(f"WORLD_SIZE={world_size} RANK={rank} => assigned {len(assigned)} / {len(all_files)} files")

    if not assigned:
        logger.info("No files assigned to this rank. Exiting.")
        return

    # Ensure bases exist
    os.makedirs(args.output_base, exist_ok=True)
    os.makedirs(args.temp_base, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.file_proc_workers)) as pool:
        futures = []
        for path in assigned:
            futures.append(pool.submit(
                _process_one,
                path,
                args.output_base,
                args.temp_base,
                args.num_workers,
                args.slice_suffix,
                args.slice_workers,
                args.mode,
            ))

        total_ok = 0
        total_skipped = 0
        total_failed = 0

        with tqdm(total=len(futures), desc="Files", dynamic_ncols=True) as pbar:
            for fut in concurrent.futures.as_completed(futures):
                try:
                    counters = fut.result()
                    total_ok += counters.get("ok", 0)
                    total_skipped += counters.get("skipped", 0)
                    total_failed += counters.get("failed", 0)
                except Exception as e:
                    logger.error(f"File processing failed: {e}")
                    total_failed += 1
                pbar.update(1)
                pbar.set_postfix(ok=total_ok, skipped=total_skipped, failed=total_failed, refresh=False)

    logger.info(f"All assigned files done. ok={total_ok} skipped={total_skipped} failed={total_failed}")


if __name__ == "__main__":
    main()


