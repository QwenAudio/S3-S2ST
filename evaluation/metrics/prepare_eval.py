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


from metric_inputs import read_keyed, full_keys


def cache_path(path: str) -> str:
    return (path.replace("oss://example-storage-1/", "/fun_resource_sh_cache/", 1)
                .replace("oss://example-storage-4/", "/nls_spider_sh_cache/", 1))


def source_records(jsonl: Path):
    """Return source-audio paths indexed by both historical and robust keys.

    Some legacy collectors derive the utterance hash after replacing an OSS URI
    with its local cache path, while the evaluation JSON keeps the original
    URI.  The basename is stable in both representations, and is unique for
    the supported evaluation manifests, so retain it as a compatibility index.
    """
    records = {}
    by_basename = {}
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
        resolved_wav = cache_path(wav)
        records[safe_key(original)] = resolved_wav
        explicit_key = obj.get('_eval_key')
        if explicit_key is not None:
            assert isinstance(explicit_key, str) and explicit_key
            assert explicit_key not in records or records[explicit_key] == resolved_wav
            records[explicit_key] = resolved_wav
        basename = Path(wav).name
        previous = by_basename.get(basename)
        if previous is None:
            by_basename[basename] = resolved_wav
        elif previous != resolved_wav:
            # Do not guess if a manifest reuses a basename for distinct audio.
            by_basename[basename] = None
    return records, by_basename


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
    keys = full_keys(ref, hyp)
    full_keys(ref, src)
    empty_path = args.recog_dir / "empty_audio_keys.json"
    empty_audio = set(json.loads(empty_path.read_text())) if empty_path.exists() else set()

    with (args.output_dir / "wer_ref.txt").open("w", encoding="utf-8") as fout:
        for key in keys:
            # WER/CER measures whether the synthesized speech follows the
            # thinker output. Translation quality against the dataset label is
            # reported separately by BLEU/COMET and ASR-BLEU.
            fout.write(f"{key} {hyp[key]}\n")

    # Text COMET measures the Thinker translation.  ASR-COMET is prepared
    # after decoded WAV ASR is available, by prepare_asr_comet.py.
    with (args.output_dir / "comet_input.jsonl").open("w", encoding="utf-8") as fout:
        for key in keys:
            if key in src:
                fout.write(json.dumps({"id": key, "src": src[key], "mt": hyp[key],
                                       "ref": ref[key]}, ensure_ascii=False) + "\n")

    source, source_by_basename = source_records(args.jsonl)
    prompt_lines, hyp_lines = [], []
    missing_source = missing_hyp = 0
    for key in keys:
        if key in empty_audio:
            continue
        wav = args.wav_dir / f"{key}.wav"
        source_wav = source.get(key)
        if source_wav is None:
            # ``safe_key`` emits "<original-basename>_<sha1>".  This fallback
            # bridges cache-vs-OSS hash derivations without weakening pairing.
            source_wav = source_by_basename.get(key.rsplit("_", 1)[0])
        if source_wav is None:
            missing_source += 1
            continue
        if not wav.is_file():
            missing_hyp += 1
            continue
        prompt_lines.append(f"{key} {source_wav}\n")
        hyp_lines.append(f"{key} {wav}\n")
    (args.output_dir / "prompt_wav.scp").write_text("".join(prompt_lines), encoding="utf-8")
    (args.output_dir / "hyp_wav.scp").write_text("".join(hyp_lines), encoding="utf-8")
    summary = {"matched_text": len(keys), "comet_rows": sum(k in src for k in keys),
               "voice_pairs": len(hyp_lines), "missing_source": missing_source,
               "missing_generated_wav": missing_hyp, "empty_audio_count": len(empty_audio), "audio_quality_coverage": len(hyp_lines)/len(keys) if keys else 0}
    (args.output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
