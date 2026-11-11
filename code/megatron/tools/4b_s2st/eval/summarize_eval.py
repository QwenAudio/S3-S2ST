#!/usr/bin/env python3
"""Collect the principal 9pair S2ST metrics into one TSV/JSON report."""

import json
import re
import sys
from pathlib import Path


def json_score(path, key="score"):
    try:
        return json.loads(path.read_text(encoding="utf-8"))[key]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def main():
    root = Path(sys.argv[1])
    rows = []
    for result in sorted(root.glob("*.out")):
        metrics = result / "metrics"
        wav_root = result / "1best_recog" / "s2st_wav"
        row = {
            "dataset": result.name.removesuffix(".out"),
            "text_bleu": json_score(metrics / "text_bleu.json"),
            "comet": json_score(metrics / "comet.json", "system_score"),
            "asr_bleu": json_score(metrics / "asr_bleu.json"),
            "wer_cer": None,
            "spk_sim": None,
            "dnsmos": None,
        }
        try:
            last = (wav_root / "wav_res_ref_text.wer").read_text(
                encoding="utf-8", errors="replace").splitlines()[-1]
            match = re.match(r"(?:WER|CER):\s*([0-9.]+)", last)
            row["wer_cer"] = float(match.group(1)) if match else None
        except (OSError, IndexError):
            pass
        try:
            line = (metrics / "voice" / "spk_simi_scores.eres2net.txt").read_text(
                encoding="utf-8").splitlines()[-1].split()
            # Voice eval writes: "avg <speaker_similarity> <auxiliary_score>".
            # Speaker similarity is the bounded 0-1 value in column 2.
            row["spk_sim"] = float(line[1]) if line and line[0] == "avg" else None
        except (OSError, IndexError, ValueError):
            pass
        try:
            row["dnsmos"] = float((metrics / "voice" / "dnsmos_mean.txt").read_text().strip())
        except (OSError, ValueError):
            pass
        rows.append(row)

    columns = ["dataset", "text_bleu", "comet", "wer_cer", "asr_bleu", "spk_sim", "dnsmos"]
    (root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    with (root / "summary.tsv").open("w", encoding="utf-8") as fout:
        fout.write("\t".join(columns) + "\n")
        for row in rows:
            fout.write("\t".join("" if row[c] is None else str(row[c]) for c in columns) + "\n")
    print(root / "summary.tsv")


if __name__ == "__main__":
    main()
