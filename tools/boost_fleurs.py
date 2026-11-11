#!/usr/bin/env python3
"""Offline, constant-per-file gain. Requires: pip install numpy soundfile."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def boost(audio, target_rms=0.0743324953, peak_limit=0.95):
    """Use whole-file RMS, boost only; no AGC, compression or resampling."""
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains NaN/Inf")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    gain = max(1.0, min(target_rms / rms, peak_limit / peak)) if rms else 1.0
    result = audio * gain
    return result, {
        "input_rms": rms,
        "output_rms": rms * gain,
        "input_peak": peak,
        "output_peak": peak * gain,
        "gain": gain,
        "gain_db": float(20 * np.log10(gain)),
        "peak_limited": bool(rms and target_rms / rms > peak_limit / peak and rms < target_rms),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Audio file or directory (recursive)")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="New output directory; source files are never overwritten")
    parser.add_argument("--target-rms", type=float, default=0.0743324953,
                        help="Whole-file RMS target (default: 0.0743324953)")
    parser.add_argument("--peak-limit", type=float, default=0.95,
                        help="Peak ceiling for added gain; default: 0.95")
    args = parser.parse_args()
    if not np.isfinite(args.target_rms) or not 0 < args.target_rms <= 1:
        parser.error("--target-rms must be in (0, 1]")
    if not np.isfinite(args.peak_limit) or not 0 < args.peak_limit <= 1:
        parser.error("--peak-limit must be in (0, 1]")
    src, dst = args.input.resolve(), args.output_dir.resolve()
    if not src.exists():
        parser.error(f"Input does not exist: {src}")
    if dst.exists():
        parser.error("Output directory already exists; choose a new directory")
    if src.is_dir() and dst.is_relative_to(src):
        parser.error("Output directory must be outside the input directory")
    extensions = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in extensions) if src.is_dir() else [src]
    if not files:
        parser.error("No supported audio files found")
    base = src if src.is_dir() else src.parent
    # Preserve the original extension in the output name to avoid stem collisions.
    outputs = [dst / (str(p.relative_to(base)) + ".wav") for p in files]
    dst.mkdir(parents=True)
    failures = 0
    with (dst / "gain_manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for i, (path, out) in enumerate(zip(files, outputs), 1):
            row = {"input": str(path), "output": str(out)}
            try:
                audio, sr = sf.read(path, dtype="float64", always_2d=True)
                result, stats = boost(audio, args.target_rms, args.peak_limit)
                out.parent.mkdir(parents=True, exist_ok=True)
                # Float WAV avoids introducing integer quantization or clipping.
                sf.write(out, result, sr, subtype="FLOAT")
                row.update(stats, sample_rate=sr, channels=audio.shape[1], status="ok")
            except Exception as exc:
                failures += 1
                row.update(status="error", error=str(exc))
                print(f"ERROR {path}: {exc}", flush=True)
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest.flush()
            if i % 50 == 0 or i == len(files):
                print(f"Processed {i}/{len(files)}, failures={failures}", flush=True)
    print(f"Output: {dst}\nManifest: {dst / 'gain_manifest.jsonl'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
