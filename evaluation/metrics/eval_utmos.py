#!/usr/bin/env python3
"""Pinned UTMOS-22 evaluation for generated S2ST WAVs.

Uses the official SpeechMOS torch-hub entry pinned to v1.2.0.  Set
TORCH_HOME to a shared cache on air-gapped clusters after pre-populating it.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import torch


def score(model, wav: str, device: str) -> float:
    import torchaudio
    audio, sr = torchaudio.load(wav)
    # UTMOS-22 expects (batch, samples); downmix deterministic multi-channel WAV.
    audio = audio.mean(dim=0, keepdim=True).to(device)
    with torch.inference_mode():
        value = model(audio, sr)
    return float(value.item() if hasattr(value, "item") else value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-scp", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--device", default="cuda:0")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--repo", type=Path, default=root / "third_party" / "SpeechMOS")
    ap.add_argument("--cache-dir", type=Path, default=root / "third_party" / "torch_cache")
    args = ap.parse_args()
    entries = [line.split(maxsplit=1) for line in args.wav_scp.read_text().splitlines() if line.strip()]
    if not entries:
        raise SystemExit("No WAVs for UTMOS")
    if not (args.repo / "hubconf.py").is_file():
        raise SystemExit(f"missing pinned SpeechMOS checkout: {args.repo}; run setup_utmos.sh")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(args.cache_dir)
    model = torch.hub.load(str(args.repo), "utmos22_strong", source="local", trust_repo=True)
    if hasattr(model, "to"):
        model = model.to(args.device)
    if hasattr(model, "eval"):
        model.eval()
    scores = [{"key": key, "utmos": score(model, wav, args.device)} for key, wav in entries]
    payload = {"metric": "UTMOS-22_strong", "source": "tarepan/SpeechMOS:v1.2.0 (local pinned checkout)",
               "num_samples": len(scores), "score": sum(x["utmos"] for x in scores) / len(scores),
               "segment_scores": scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["key", "utmos"])
            writer.writeheader(); writer.writerows(scores)
    print(json.dumps({k: v for k, v in payload.items() if k != "segment_scores"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
