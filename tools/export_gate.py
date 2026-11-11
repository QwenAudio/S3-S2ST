#!/usr/bin/env python3
"""Export the streaming READ/WRITE gate from a torch.distributed checkpoint."""

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    reader = dcp.FileSystemReader(str(args.checkpoint))
    metadata = reader.read_metadata()
    gate = {}
    for key, value in metadata.state_dict_metadata.items():
        if key.startswith("streaming_gate."):
            gate[key.removeprefix("streaming_gate.")] = torch.empty(
                tuple(value.size), dtype=value.properties.dtype
            )
    if not gate:
        raise RuntimeError(f"no streaming_gate tensors in {args.checkpoint}")
    flat = {f"streaming_gate.{key}": value for key, value in gate.items()}
    dcp.load(flat, checkpoint_id=str(args.checkpoint))
    gate = {
        key.removeprefix("streaming_gate."): value.cpu()
        for key, value in flat.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": gate, "read_class": 0, "write_class": 1}, args.output)
    print(f"exported {len(gate)} tensors to {args.output}")


if __name__ == "__main__":
    main()
