#!/usr/bin/env python3
"""Materialize a small EN alignment-repair canary using ossutil and ffmpeg."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def run(command: list[str]):
    subprocess.run(command, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--ossutil", default="ossutil")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()
    completed = 0
    with args.jobs.open(encoding="utf-8") as handle, tempfile.TemporaryDirectory(prefix="stream_align_repair_") as tmp:
        tmp = Path(tmp)
        for line in handle:
            if args.limit and completed >= args.limit:
                break
            job = json.loads(line)
            source = tmp / "source"
            clipped = tmp / "clip.wav"
            run([args.ossutil, "cp", "-f", job["source_uri"], str(source)])
            run([
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-ss", str(job["start_time"]), "-to", str(job["end_time"]),
                "-i", str(source), "-ac", "1", "-ar", "16000", str(clipped),
            ])
            run([args.ossutil, "cp", "-f", str(clipped), job["dest_uri"]])
            source.unlink(missing_ok=True)
            clipped.unlink(missing_ok=True)
            completed += 1
    print(json.dumps({"materialized": completed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
