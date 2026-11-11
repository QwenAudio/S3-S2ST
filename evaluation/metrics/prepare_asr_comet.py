#!/usr/bin/env python3
"""Build XCOMET input for S2ST generated-speech ASR.

This intentionally uses the same source/reference contract as text COMET:
src is the source sentence, mt is the ASR transcript of generated speech, and
ref is the dataset translation reference.  It is therefore an S2ST quality
metric, unlike WER/CER which compares ASR against Thinker text (talker fidelity).
"""
import argparse
import json
import re
from pathlib import Path


from metric_inputs import read_keyed as keyed, read_asr_tsv, full_keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recog-dir", type=Path, required=True)
    ap.add_argument("--asr-tsv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-language", choices=("en", "zh"), required=True,
                    help="Target language used to normalize ASR text for COMET.")
    args = ap.parse_args()
    source, reference = keyed(args.recog_dir / "source"), keyed(args.recog_dir / "label")
    _, hypotheses = read_asr_tsv(args.asr_tsv)
    keys = full_keys(reference, hypotheses)
    full_keys(reference, source)
    rows = []
    for key in keys:
        asr_text = hypotheses[key].strip()
        # The Chinese ASR backend emits one Han character per whitespace token.
        # CER accepts this convention, but XCOMET expects ordinary Chinese text
        # and severely underrates otherwise correct hypotheses.  Use exactly the
        # text form that humans/readers and the Chinese text metrics consume.
        if args.target_language == "zh":
            asr_text = re.sub(r"\s+", "", asr_text)
        rows.append({"id": key, "src": source[key], "mt": asr_text, "ref": reference[key]})
    if not rows:
        raise SystemExit("No aligned ASR-COMET rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    print(json.dumps({"asr_comet_rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
