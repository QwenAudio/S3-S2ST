#!/usr/bin/env python3
"""Collapse streaming S2TT turns to a one-turn offline-style diagnostic."""

import argparse
import json
import os
import re
from pathlib import Path


_AUDIO = re.compile(r"<\|startofspeech\|>(.*?)<\|endofspeech\|>")


def speech_token_length(seconds):
    """Match the encoder's two conv2d subsamplers plus adaptor conv stride."""
    frames = max(1, int(round(float(seconds) * 100)))
    length = 1 + (frames - 3 + 2) // 2
    length = 1 + (length - 3 + 2) // 2
    return max(1, (length - 1) // 2 + 1)


def collapse(sample):
    messages = sample["messages"]
    systems = [item for item in messages if item["role"] == "system"]
    users = [item for item in messages if item["role"] == "user"]
    assistants = [item for item in messages if item["role"] == "assistant"]
    if len(systems) != 1 or not users or len(users) != len(assistants):
        raise ValueError("expected one system and alternating non-empty user/assistant turns")

    audio_values = []
    for user in users:
        match = _AUDIO.search(user.get("text", ""))
        if match is None:
            raise ValueError("user turn has no speech marker")
        audio_values.append(match.group(1))
    if len(set(audio_values)) != 1:
        raise ValueError("streaming turns do not reference one source audio")
    for previous, current in zip(users, users[1:]):
        if abs(float(previous["end_time"]) - float(current["start_time"])) > 1e-5:
            raise ValueError("streaming audio turns are not contiguous")

    first_text = users[0].get("text", "")
    prompt = first_text.split("<|startofspeech|>", 1)[0]
    user_text = f"{prompt}<|startofspeech|>{audio_values[0]}<|endofspeech|>"
    assistant_text = "".join(item.get("text", "") for item in assistants)
    system_text = systems[0].get("text", "")
    start_time = float(users[0]["start_time"])
    end_time = float(users[-1]["end_time"])
    collapsed_messages = [
        {
            "role": "system",
            "text": system_text,
            "content": system_text,
            "mode": "tc",
            "audio": "",
            "speech_length": 0,
        },
        {
            "role": "user",
            "text": user_text,
            "content": user_text,
            "mode": "tc",
            "audio": "",
            # The offline encoder sees one continuous slice, so convolution
            # rounding must be applied once to the full duration. Summing the
            # per-chunk lengths over-counts boundaries and is not an exact A/B.
            "speech_length": speech_token_length(end_time - start_time),
            "start_time": start_time,
            "end_time": end_time,
            "segment_id": users[0].get("segment_id"),
            "slice_mode": users[0].get("slice_mode", "absolute"),
        },
        {
            "role": "assistant",
            "text": assistant_text,
            "content": assistant_text,
            "mode": "t",
            "audio": "",
            "speech_length": 0,
        },
    ]
    return {
        "key": sample.get("key"),
        "messages": collapsed_messages,
        "task": "s2tt_offline_style_diagnostic",
        "src_lang": sample.get("src_lang"),
        "tgt_lang": sample.get("tgt_lang"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--packed-input",
        action="store_true",
        help="flatten each packed JSON array before collapsing conversations",
    )
    args = parser.parse_args()

    sources = [Path(line.strip()) for line in Path(args.input_list).read_text().splitlines() if line.strip()]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = []
    total = 0
    for shard_index, source in enumerate(sources):
        destination = output_dir / f"data.{shard_index:05d}.jsonl"
        with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
            for line in reader:
                value = json.loads(line)
                samples = value if args.packed_input else [value]
                if args.packed_input and not isinstance(value, list):
                    raise ValueError(f"expected packed JSON array in {source}")
                for sample in samples:
                    if args.limit and total >= args.limit:
                        break
                    writer.write(json.dumps(collapse(sample), ensure_ascii=False) + "\n")
                    total += 1
                if args.limit and total >= args.limit:
                    break
        if destination.stat().st_size:
            destinations.append(destination)
        if args.limit and total >= args.limit:
            break
    temporary = output_dir / "data.list.tmp"
    with temporary.open("w", encoding="utf-8") as writer:
        for destination in destinations:
            writer.write(str(destination) + "\n")
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, output_dir / "data.list")
    print(json.dumps({"samples": total, "shards": len(destinations)}, indent=2))


if __name__ == "__main__":
    main()
