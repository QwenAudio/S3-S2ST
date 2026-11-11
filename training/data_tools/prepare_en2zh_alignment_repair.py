#!/usr/bin/env python3
"""Prepare a lossless repair canary for EN source segments with non-zero offsets.

This only writes manifests.  ``materialize_alignment_clips.py`` performs the
explicit OSS download -> ffmpeg clip -> OSS upload step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--dest-prefix", required=True, help="oss://bucket/prefix without trailing slash")
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()
    args.jobs.parent.mkdir(parents=True, exist_ok=True)
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.mapping.open(encoding="utf-8") as src, args.jobs.open("w", encoding="utf-8") as jobs, args.submission.open("w", encoding="utf-8") as submission:
        for line in src:
            row = json.loads(line)
            start, end = float(row["en_start_time"]), float(row["en_end_time"])
            if start <= 0 or end <= start:
                continue
            digest = hashlib.sha1(f'{row["en_audio_path"]}\0{start:.3f}\0{end:.3f}'.encode()).hexdigest()
            dest = f'{args.dest_prefix.rstrip("/")}/{digest}.wav'
            job = {
                "pair_id": row["pair_id"],
                "segment_wav_id": row["en_segment_wav_id"],
                "source_uri": row["en_audio_path"],
                "start_time": start,
                "end_time": end,
                "dest_uri": dest,
            }
            fa = {
                "raw_audio_path": dest,
                "wav_id": row["en_wav_id"],
                "segments": [{
                    "start_time": 0,
                    "end_time": -1,
                    "segment_wav_id": row["en_segment_wav_id"],
                    "asr_gemini": row["en_text"],
                }],
                "info": {"audio_role": "original_source", "pair_id": row["pair_id"], "repair": "clip_then_align_v2"},
            }
            jobs.write(json.dumps(job, ensure_ascii=False, separators=(",", ":")) + "\n")
            submission.write(json.dumps(fa, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    print(json.dumps({"repair_rows": count, "jobs": str(args.jobs), "submission": str(args.submission)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
