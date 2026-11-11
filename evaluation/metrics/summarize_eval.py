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
            "asr_comet": json_score(metrics / "asr_comet.json", "system_score"),
            "wer_cer": None,
            "spk_sim": None,
            "utmos": json_score(metrics / "utmos.json"),
        }
        corpus = json.loads((metrics / "wer_corpus.json").read_text())
        if corpus.get("error_rate_aggregation") != "corpus: sum(S+D+I)/sum(reference tokens)":
            raise ValueError(f"Not an official corpus result: {metrics}")
        row["wer_cer"] = corpus["score"]
        try:
            line = (metrics / "voice" / "spk_simi_scores.eres2net.txt").read_text(
                encoding="utf-8").splitlines()[-1].split()
            # Voice eval writes: "avg <prompt-ref cosine*100>\n+            # <prompt-hyp cosine*100>".  This evaluation does not provide a
            # reference-WAV manifest, so column 2 is necessarily 0.00; the
            # actual generated-speech similarity is the third column.  Convert
            # the logged percentage back to its standard bounded cosine scale.
            row["spk_sim"] = float(line[2]) / 100.0 if line and line[0] == "avg" else None
        except (OSError, IndexError, ValueError):
            pass
        rows.append(row)

    columns = ["dataset", "text_bleu", "comet", "wer_cer", "asr_bleu", "asr_comet", "spk_sim", "utmos"]
    (root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    with (root / "summary.tsv").open("w", encoding="utf-8") as fout:
        fout.write("\t".join(columns) + "\n")
        for row in rows:
            fout.write("\t".join("" if row[c] is None else str(row[c]) for c in columns) + "\n")
    print(root / "summary.tsv")


if __name__ == "__main__":
    main()
