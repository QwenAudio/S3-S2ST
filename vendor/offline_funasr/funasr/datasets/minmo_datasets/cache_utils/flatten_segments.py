#!/usr/bin/env python3
import sys
import os
import argparse

try:
    import ujson as json
except ImportError:
    import json


def flatten_segments(input_file, max_lines):
    base_name = os.path.basename(input_file)
    output_base = input_file + ".segments"

    line_count = 0  # lines written to current output file
    file_index = 0
    output_fp = None
    current_filename = None
    file_stats = []  # (filename, line_count) per output file
    total_segments = 0
    input_lines_with_segments = 0

    def open_new_file(index):
        nonlocal output_fp, line_count, current_filename
        if output_fp:
            output_fp.close()
            file_stats.append((current_filename, line_count))
            line_count = 0
        current_filename = f"{output_base}.{index:05d}.jsonl"
        output_fp = open(current_filename, "w", encoding="utf-8")
        return output_fp

    with open(input_file, "r", encoding="utf-8") as infile:
        for input_line in infile:
            input_line = input_line.strip()
            if not input_line:
                continue
            try:
                obj = json.loads(input_line)
            except Exception as e:
                sys.stderr.write(f"Error parsing line: {e}\n")
                continue

            segments = obj.get("segments", [])
            if not isinstance(segments, list):
                sys.stderr.write("Warning: segments is not a list\n")
                continue
            if not segments:
                continue

            input_lines_with_segments += 1
            n_seg = len(segments)
            # Keep all segments from this input line in the same output file
            if output_fp is None:
                open_new_file(file_index)
                file_index += 1
            elif line_count >= max_lines:
                open_new_file(file_index)
                file_index += 1
            # else: may exceed max_lines for this file if n_seg is large

            raw_audio_path = obj.get("raw_audio_path")

            for seg in segments:
                if isinstance(seg, dict) and raw_audio_path is not None:
                    out_seg = {"raw_audio_path": raw_audio_path, **seg}
                else:
                    out_seg = seg
                output_fp.write(json.dumps(out_seg, ensure_ascii=False) + "\n")
                line_count += 1
            total_segments += n_seg

    if output_fp:
        output_fp.close()
        file_stats.append((current_filename, line_count))

    # Print output summary
    n_files = len(file_stats)
    lines_per_file = [c for _, c in file_stats]
    sys.stdout.write("=== flatten_segments output summary ===\n")
    sys.stdout.write(f"Input: {input_file}\n")
    sys.stdout.write(f"Input lines with segments: {input_lines_with_segments}\n")
    sys.stdout.write(f"Total segments: {total_segments}\n")
    sys.stdout.write(f"Output files: {n_files}\n")
    sys.stdout.write(f"Max lines per output (requested): {max_lines}\n")
    if lines_per_file:
        sys.stdout.write(
            f"Lines per output file: min={min(lines_per_file)}, max={max(lines_per_file)}, avg={sum(lines_per_file) / len(lines_per_file):.1f}\n"
        )
        for fname, cnt in file_stats:
            sys.stdout.write(f"  {fname}: {cnt} lines\n")
    sys.stdout.write("=======================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Flatten 'segments' from a JSONL file into multiple JSONL output files."
    )
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument(
        "--max-lines", type=int, default=40000, help="Maximum lines per output file"
    )
    args = parser.parse_args()

    flatten_segments(args.input_file, args.max_lines)


if __name__ == "__main__":
    main()
