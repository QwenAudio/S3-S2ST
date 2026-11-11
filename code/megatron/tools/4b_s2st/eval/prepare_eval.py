#!/usr/bin/env python3
"""Prepare talker-fidelity WER refs, voice manifests, and COMET input."""

import argparse
import hashlib
import json
import re
from pathlib import Path


SPEECH_RE = re.compile(r"<\|startofspeech\|>!?(.*?)<\|endofspeech\|>")


def safe_key(original_key: str, basename_limit: int = 180) -> str:
    original_key = original_key.replace(" ", "")
    basename = original_key.rsplit("/", 1)[-1]
    basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename).strip("._") or "audio"
    digest = hashlib.sha1(original_key.encode("utf-8")).hexdigest()[:16]
    return f"{basename[:basename_limit]}_{digest}"


def read_keyed(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1) if "\t" in line else line.split(maxsplit=1)
        if len(parts) == 2:
            rows[parts[0]] = parts[1]
    return rows


def cache_path(path: str) -> str:
    return (path.replace("oss://example-storage-1/", "/fun_resource_sh_cache/", 1)
                .replace("oss://example-storage-4/", "/nls_spider_sh_cache/", 1))


def source_records(jsonl: Path):
    records = {}
    for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        messages = obj.get("messages", [])
        user = messages[1] if len(messages) > 1 else {}
        content = user.get("content", user.get("text", ""))
        match = SPEECH_RE.search(content)
        if not match:
            continue
        wav = match.group(1).replace(" ", "")
        original = wav
        try:
            original = f"{wav}_{float(user['start_time'])}_{float(user['end_time'])}"
        except (KeyError, TypeError, ValueError):
            pass
        records[safe_key(original)] = cache_path(wav)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--recog-dir", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    src = read_keyed(args.recog_dir / "source")
    ref = read_keyed(args.recog_dir / "label")
    hyp = read_keyed(args.recog_dir / "text")
    keys = [key for key in ref if key in hyp]

    with (args.output_dir / "wer_ref.txt").open("w", encoding="utf-8") as fout:
        for key in keys:
            # WER/CER measures whether the synthesized speech follows the
            # thinker output. Translation quality against the dataset label is
            # reported separately by BLEU/COMET and ASR-BLEU.
            fout.write(f"{key} {hyp[key]}\n")

    with (args.output_dir / "comet_input.jsonl").open("w", encoding="utf-8") as fout:
        for key in keys:
            if key in src:
                fout.write(json.dumps({"id": key, "src": src[key], "mt": hyp[key],
                                       "ref": ref[key]}, ensure_ascii=False) + "\n")

    source = source_records(args.jsonl)
    prompt_lines, hyp_lines = [], []
    missing_source = missing_hyp = 0
    for key in keys:
        wav = args.wav_dir / f"{key}.wav"
        if key not in source:
            missing_source += 1
            continue
        if not wav.is_file():
            missing_hyp += 1
            continue
        prompt_lines.append(f"{key} {source[key]}\n")
        hyp_lines.append(f"{key} {wav}\n")
    (args.output_dir / "prompt_wav.scp").write_text("".join(prompt_lines), encoding="utf-8")
    (args.output_dir / "hyp_wav.scp").write_text("".join(hyp_lines), encoding="utf-8")
    summary = {"matched_text": len(keys), "comet_rows": sum(k in src for k in keys),
               "voice_pairs": len(hyp_lines), "missing_source": missing_source,
               "missing_generated_wav": missing_hyp}
    (args.output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
