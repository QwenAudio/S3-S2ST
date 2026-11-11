#!/usr/bin/env python3
"""Standalone version of seed-tts-eval/run_wer.py with bundled ASR models."""

import argparse
import string
from pathlib import Path

import torch
from tqdm import tqdm
from zhon.hanzi import punctuation

try:
    from jiwer import compute_measures
except ImportError:
    compute_measures = None
    from jiwer import process_words

import zhconv

PUNCTUATION_ALL = punctuation + string.punctuation
TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_ZH_MODEL = str(TOOL_DIR / "models" / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
DEFAULT_WHISPER_MODEL = str(TOOL_DIR / "models" / "whisper-large-v3" / "large-v3.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ASR and compute WER/CER from wav|reference list.")
    parser.add_argument("wav_ref_list", help="Input list. Each row: '<wav_path>|<reference text>'.")
    parser.add_argument("output", help="Per-utterance score TSV.")
    parser.add_argument("lang", help="Language/subset name, e.g. zh, hard_zh, en.")
    parser.add_argument("--zh-model", default=DEFAULT_ZH_MODEL, help="FunASR Chinese ASR model path/id.")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help="Whisper model .pt path/name for non-zh languages.")
    parser.add_argument("--batch-size-s", type=int, default=300, help="FunASR batch_size_s.")
    return parser.parse_args()


def lang_code(lang):
    if lang == "hard_zh" or lang.endswith("zh") or "zh" in lang:
        return "zh"
    return lang[-2:]


def is_char_level(lang):
    return lang_code(lang) in {"zh", "ja", "ko", "th"}


def load_en_model(model_path):
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_path).to(device)
    model.eval()
    return model


def load_zh_model(model_path):
    from funasr import AutoModel

    return AutoModel(model=model_path, disable_update=True)


def compute_error_rate(truth, hypo):
    ref_list = truth.split(" ")
    if not ref_list or ref_list == [""]:
        return 0.0, 0.0, 0.0, 0.0
    if compute_measures is not None:
        measures = compute_measures(truth, hypo)
        wer = measures["wer"]
        subs = measures["substitutions"] / len(ref_list)
        dele = measures["deletions"] / len(ref_list)
        inse = measures["insertions"] / len(ref_list)
    else:
        measures = process_words(truth, hypo)
        wer = measures.wer
        subs = measures.substitutions / len(ref_list)
        dele = measures.deletions / len(ref_list)
        inse = measures.insertions / len(ref_list)
    return wer, subs, dele, inse


def process_one(hypo, truth, char_level):
    raw_truth = truth
    raw_hypo = hypo

    for x in PUNCTUATION_ALL:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    if char_level:
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    else:
        truth = truth.lower()
        hypo = hypo.lower()

    wer, subs, dele, inse = compute_error_rate(truth, hypo)
    return raw_truth, raw_hypo, wer, subs, dele, inse


def iter_rows(path):
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 2:
                wav_res_path, text_ref = parts
            elif len(parts) == 3:
                wav_res_path, _, text_ref = parts
            elif len(parts) == 4:
                wav_res_path, _, text_ref, _ = parts
            else:
                raise NotImplementedError(f"Unsupported input row: {line}")
            if Path(wav_res_path).exists():
                yield wav_res_path, text_ref


def main():
    args = parse_args()
    code = lang_code(args.lang)
    char_level = is_char_level(args.lang)

    if code == "zh":
        model = load_zh_model(args.zh_model)
    else:
        model = load_en_model(args.whisper_model)

    with open(args.output, "w", encoding="utf-8") as fout:
        for wav_res_path, text_ref in tqdm(list(iter_rows(args.wav_ref_list))):
            try:
                if code == "zh":
                    res = model.generate(input=wav_res_path, batch_size_s=args.batch_size_s)
                    transcription = res[0]["text"]
                else:
                    result = model.transcribe(wav_res_path, language=code)
                    transcription = result["text"].strip()
            except Exception as exc:
                print(f"skip {wav_res_path}: {exc}", flush=True)
                continue

            if "zh" in args.lang:
                transcription = zhconv.convert(transcription, "zh-cn")

            raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, text_ref, char_level)
            fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
            fout.flush()


if __name__ == "__main__":
    main()
