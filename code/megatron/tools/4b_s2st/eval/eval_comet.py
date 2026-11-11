#!/usr/bin/env python3
"""XCOMET-XL scoring, adapted from Hejia's S2TT evaluation script."""

import argparse
import json
import os
from pathlib import Path

from comet import load_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--model", default=os.environ.get(
        "COMET_MODEL", "/path/to/model.ckpt"))
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise SystemExit("No COMET rows")
    model = load_from_checkpoint(args.model, reload_hparams=True)
    gpu_env = os.environ.get("COMET_NUM_GPUS")
    gpus = int(gpu_env) if gpu_env else 1
    batch_size = int(os.environ.get("COMET_BATCH_SIZE", "8"))
    result = model.predict(rows, batch_size=batch_size, gpus=gpus)
    payload = {"metric": "XCOMET-XL", "num_samples": len(rows),
               "system_score": float(result.system_score),
               "segment_scores": [float(x) for x in result.scores]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "segment_scores"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
