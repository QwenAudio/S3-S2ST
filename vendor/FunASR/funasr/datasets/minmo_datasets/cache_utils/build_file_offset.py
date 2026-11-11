#!/usr/bin/env python3
"""
Standalone script to build .offset file for a jsonl file.
This script processes a jsonl file and writes block offsets to a .offset file.
"""
import os
import sys
import json
import argparse

# Import process_file_index from data_project_manager
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from data_project_manager import process_file_index

def main():
    parser = argparse.ArgumentParser(description="Build .offset file for a jsonl file")
    parser.add_argument('input_file', help='Input jsonl file path')
    parser.add_argument('--stride-lines', type=int, default=1000, help='Number of lines per block (default: 1000)')
    parser.add_argument('--output-file', type=str, default=None, help='Output .offset file path (default: input_file.offset)')
    args = parser.parse_args()
    
    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = input_file + '.offset'
    
    print(f"Processing {input_file} with stride_lines={args.stride_lines}")
    print(f"Writing offsets to {output_file}")
    
    # Process the file to get offsets
    file_path, offsets = process_file_index(input_file, args.stride_lines, use_offset_cache=False)
    
    # Write offsets to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(offsets, f, indent=2)
    
    print(f"Successfully wrote {len(offsets)} blocks to {output_file}")

if __name__ == "__main__":
    main()
