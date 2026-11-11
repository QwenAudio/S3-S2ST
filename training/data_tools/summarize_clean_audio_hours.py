#!/usr/bin/env python3
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "clean_exact_v1"
TIME_RE = re.compile(
    br'"start_time":\s*([-+0-9.eE]+),\s*"end_time":\s*([-+0-9.eE]+)'
)
TOKEN_RE = re.compile(br'"speech_length":\s*(\d+)')

def scan(path):
    rows = tokens = intervals = 0
    seconds = 0.0
    with open(path, "rb") as source:
        for line in source:
            rows += 1
            tokens += sum(int(x) for x in TOKEN_RE.findall(line))
            for start, end in TIME_RE.findall(line):
                duration = float(end) - float(start)
                if duration <= 0:
                    raise ValueError(f"non-positive duration in {path}")
                seconds += duration
                intervals += 1
    return rows, tokens, intervals, seconds

started = time.time()
result = {"definition": "sum of source audio user-turn intervals after clean_exact_v1", "directions": {}}
for direction in ("zh2en", "en2zh"):
    result["directions"][direction] = {}
    for split in ("train", "valid"):
        data_list = ROOT / direction / "unpacked" / split / "data.list"
        paths = [Path(x) for x in data_list.read_text().splitlines() if x]
        with ThreadPoolExecutor(max_workers=32) as pool:
            values = list(pool.map(scan, paths))
        rows = sum(x[0] for x in values)
        tokens = sum(x[1] for x in values)
        intervals = sum(x[2] for x in values)
        seconds = sum(x[3] for x in values)
        result["directions"][direction][split] = {
            "records": rows,
            "audio_intervals": intervals,
            "duration_seconds_from_timestamps": seconds,
            "duration_hours_from_timestamps": seconds / 3600,
            "speech_tokens": tokens,
            "duration_hours_from_speech_tokens": tokens / 12.5 / 3600,
        }
result["total_hours_from_timestamps"] = sum(
    value["duration_hours_from_timestamps"]
    for splits in result["directions"].values()
    for value in splits.values()
)
result["total_hours_from_speech_tokens"] = sum(
    value["duration_hours_from_speech_tokens"]
    for splits in result["directions"].values()
    for value in splits.values()
)
result["elapsed_seconds"] = time.time() - started
output = ROOT / "audio_hours_summary.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
