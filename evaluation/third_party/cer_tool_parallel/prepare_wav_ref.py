#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare wav|reference list for WER/CER scoring.")
    parser.add_argument("text", help="Reference text file. Each row: '<utt> <reference text>'.")
    parser.add_argument("wav_dir", help="Directory containing generated '<utt>.wav' files.")
    parser.add_argument("output", help="Output list. Each row: '<wav_path>|<reference text>'.")
    return parser.parse_args()


def main():
    args = parse_args()
    wav_dir = Path(args.wav_dir)
    kept = 0
    skipped = 0
    with open(args.text, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                utt, text = line.split(maxsplit=1)
            except ValueError:
                utt, text = line, ""
            wav_path = wav_dir / f"{utt}.wav"
            if not wav_path.exists():
                skipped += 1
                continue
            fout.write(f"{wav_path}|{text}\n")
            kept += 1
    print(f"prepared {kept} rows, skipped {skipped} rows")


if __name__ == "__main__":
    main()
