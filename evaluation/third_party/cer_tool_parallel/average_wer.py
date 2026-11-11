#!/usr/bin/env python3
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Average per-utterance WER/CER scores.")
    parser.add_argument("input", help="Merged per-utterance TSV from run_wer.py.")
    parser.add_argument("output", help="Final report path.")
    parser.add_argument("--metric-name", default="WER", help="Metric label for final line, WER or CER.")
    return parser.parse_args()


def main():
    args = parse_args()
    wers = []
    wers_below50 = []
    wers_clip = []
    inses = []
    deles = []
    subses = []
    n_higher_than_50 = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        fout.write("utt\twav_res\tres_wer\ttext_ref\ttext_res\tres_wer_ins\tres_wer_del\tres_wer_sub\n")
        with open(args.input, "r", encoding="utf-8") as fin:
            for line in fin:
                wav_path, wer, text_ref, text_res, inse, dele, subs = line.rstrip("\n").split("\t")
                wer = float(wer)
                if wer > 0.5:
                    n_higher_than_50 += 1
                else:
                    wers_below50.append(wer)
                wers.append(wer)
                wers_clip.append(min(wer, 1.0))
                inses.append(float(inse))
                deles.append(float(dele))
                subses.append(float(subs))
                fout.write(line)

        if not wers:
            raise RuntimeError("No valid utterance scores found.")

        metric = args.metric_name.upper()
        wer = round(float(np.mean(wers)) * 100, 3)
        wer_below50 = round(float(np.mean(wers_below50)) * 100, 3) if wers_below50 else 0.0
        wer_clip = round(float(np.mean(wers_clip)) * 100, 3)
        subs = round(float(np.mean(subses)) * 100, 3)
        dele = round(float(np.mean(deles)) * 100, 3)
        inse = round(float(np.mean(inses)) * 100, 3)
        fout.write(f"{metric}: {wer}, {metric}_L50: {wer_below50}, {metric}_CLIP: {wer_clip}, SUBS: {subs}, DEL: {dele}, INS: {inse}\n")

    print(f"{metric}: {wer}%")
    print(n_higher_than_50)


if __name__ == "__main__":
    main()
