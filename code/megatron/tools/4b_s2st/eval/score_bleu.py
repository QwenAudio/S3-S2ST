#!/usr/bin/env python3
"""Score keyed text files or Hejia ASR TSV with sacreBLEU."""

import argparse
import json
from pathlib import Path

import sacrebleu


def read_keyed(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1) if "\t" in line else line.split(maxsplit=1)
        if len(parts) == 2:
            rows[parts[0]] = parts[1]
    return rows


def read_asr_tsv(path: Path):
    refs, hyps = {}, {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        key = Path(parts[0]).stem
        refs[key] = parts[2]
        hyps[key] = parts[3]
    return refs, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, help="Keyed reference: <utt><TAB><text>.")
    ap.add_argument("--hyp", type=Path, help="Keyed hypothesis: <utt><TAB><text>.")
    ap.add_argument("--asr-tsv", type=Path,
                    help="Hejia run_wer.py TSV; column 4 is the ASR hypothesis. "
                         "With --ref, use that keyed file as the BLEU reference.")
    ap.add_argument("--language", choices=("zh", "en"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.asr_tsv:
        tsv_refs, hyps = read_asr_tsv(args.asr_tsv)
        refs = read_keyed(args.ref) if args.ref else tsv_refs
    elif args.ref and args.hyp:
        refs, hyps = read_keyed(args.ref), read_keyed(args.hyp)
    else:
        ap.error("use --asr-tsv or both --ref and --hyp")

    keys = [key for key in refs if key in hyps]
    if not keys:
        raise SystemExit("No matched reference/hypothesis rows")
    tokenize = "zh" if args.language == "zh" else "13a"
    ref_text = [refs[key] for key in keys]
    hyp_text = [hyps[key] for key in keys]
    score = sacrebleu.corpus_bleu(hyp_text, [ref_text], tokenize=tokenize)
    result = {
        "metric": "ASR-BLEU" if args.asr_tsv else "BLEU",
        "language": args.language,
        "num_samples": len(keys),
        "score": score.score,
        "signature": str(score),
        "ref_file": str(args.ref or args.asr_tsv),
        "hyp_file": str(args.asr_tsv or args.hyp),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
