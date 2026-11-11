"""Overlay the trained offline CosyVoice Talker branch from a Zarr checkpoint.

This reader intentionally supports only the uncompressed Zarr-v2 arrays written
by the existing Megatron S2ST experiment.  It keeps checkpoint composition local
to streaming S2ST: the main checkpoint remains the true-streaming S2TT model,
while bridge/Talker/codec-I/O parameters come from the proven offline S2ST run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


PREFIXES = (
    "cosyvoice_llm.",
    "cosyvoice_llm_input_proj.",
    "cosyvoice_codec_embed.",
    "cosyvoice_llm_codec_head.",
)


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "<f4": torch.float32,
        "<f2": torch.float16,
        "<i8": torch.int64,
        "<i4": torch.int32,
        "|u1": torch.uint8,
    }
    if name not in mapping:
        raise TypeError(f"unsupported Zarr dtype: {name}")
    return mapping[name]


def load_uncompressed_zarr_array(path: Path) -> torch.Tensor:
    metadata = json.loads((path / ".zarray").read_text(encoding="utf-8"))
    if metadata.get("zarr_format") != 2 or metadata.get("compressor") is not None:
        raise ValueError(f"expected uncompressed Zarr-v2 array: {path}")
    if metadata.get("order") != "C":
        raise ValueError(f"only C-order Zarr arrays are supported: {path}")
    shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(int(value) for value in metadata["chunks"])
    dtype = _dtype(metadata["dtype"])
    output = torch.empty(shape, dtype=dtype)
    chunk_files = sorted(
        entry for entry in path.iterdir() if entry.is_file() and not entry.name.startswith(".")
    )
    if not chunk_files:
        raise FileNotFoundError(f"Zarr array contains no chunks: {path}")
    for chunk_file in chunk_files:
        indices = tuple(int(value) for value in chunk_file.name.split("."))
        if len(indices) != len(shape):
            raise ValueError(f"invalid Zarr chunk name {chunk_file.name!r}: {path}")
        starts = tuple(index * chunk for index, chunk in zip(indices, chunks))
        chunk_shape = tuple(
            min(chunk, size - start)
            for start, chunk, size in zip(starts, chunks, shape)
        )
        raw = torch.frombuffer(bytearray(chunk_file.read_bytes()), dtype=dtype).clone()
        expected = 1
        for size in chunk_shape:
            expected *= size
        if raw.numel() != expected:
            raise ValueError(
                f"Zarr chunk size mismatch: {chunk_file}; "
                f"values={raw.numel()} expected={expected}"
            )
        slices = tuple(
            slice(start, start + size) for start, size in zip(starts, chunk_shape)
        )
        output[slices] = raw.reshape(chunk_shape)
    return output


def _select_tensor_parallel_shard(
    name: str, value: torch.Tensor, parameter: torch.nn.Parameter
) -> torch.Tensor:
    """Select this rank's slice when the Zarr checkpoint stores a global tensor."""
    if tuple(value.shape) == tuple(parameter.shape):
        return value

    # The proven Talker was saved with TP=2. Its vocabulary dimension was
    # padded from the real 59,249 entries to 59,250 so it divided evenly over
    # the two ranks. At TP=1 the model correctly allocates only the real
    # vocabulary. Drop only those trailing, never-addressable padding rows;
    # all non-vocabulary shape differences remain fatal below.
    if (
        name in {
            "cosyvoice_codec_embed.weight",
            "cosyvoice_llm_codec_head.weight",
        }
        and value.ndim == parameter.ndim
        and value.shape[1:] == parameter.shape[1:]
        and value.shape[0] > parameter.shape[0]
        and value.shape[0] - parameter.shape[0] < 8
    ):
        return value.narrow(0, 0, parameter.shape[0])

    from megatron.core import parallel_state

    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    candidates = []
    if value.ndim == parameter.ndim and tp_size > 1:
        for dim, (global_size, local_size) in enumerate(
            zip(value.shape, parameter.shape)
        ):
            other_dims_match = all(
                value.shape[index] == parameter.shape[index]
                for index in range(value.ndim)
                if index != dim
            )
            if global_size == local_size * tp_size and other_dims_match:
                candidates.append(dim)

    if len(candidates) != 1:
        raise RuntimeError(
            f"Talker overlay shape mismatch for {name}: "
            f"checkpoint={tuple(value.shape)} model={tuple(parameter.shape)} "
            f"tp_size={tp_size} shard_candidates={candidates}"
        )

    dim = candidates[0]
    start = tp_rank * parameter.shape[dim]
    return value.narrow(dim, start, parameter.shape[dim])


def register_cosyvoice_zarr_overlay(model) -> None:
    if os.environ.get("STREAMING_S2ST_SKIP_TALKER_OVERLAY", "0").lower() in {
        "1",
        "true",
        "yes",
    }:
        print(
            "[STREAMING S2ST] skipped offline Talker overlay; preserving the "
            "Talker weights loaded from INIT_CKPT_DIR",
            flush=True,
        )
        return
    checkpoint = Path(os.environ["STREAMING_S2ST_TALKER_CKPT"]).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing offline S2ST checkpoint: {checkpoint}")
    state = {"loaded": False}

    def overlay(module, _incompatible_keys):
        if state["loaded"]:
            return
        targets = {
            name: parameter
            for name, parameter in module.named_parameters()
            if name.startswith(PREFIXES)
        }
        if not targets:
            raise RuntimeError("model has no CosyVoice Talker parameters to overlay")
        missing = []
        loaded_values = 0
        for name, parameter in targets.items():
            array_path = checkpoint / name
            if not (array_path / ".zarray").is_file():
                missing.append(name)
                continue
            value = load_uncompressed_zarr_array(array_path)
            value = _select_tensor_parallel_shard(name, value, parameter)
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            loaded_values += value.numel()
        if missing:
            raise RuntimeError(
                f"offline Talker checkpoint misses {len(missing)} parameter(s): "
                + repr(missing[:12])
            )
        if os.environ.get("STREAMING_S2ST_INJECTED_DUAL_BOS", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            continue_bos = int(
                os.environ.get("STREAMING_S2ST_CODEC_BOS_TOKEN_ID", "59049")
            )
            final_bos = int(
                os.environ.get(
                    "STREAMING_S2ST_CODEC_FINAL_BOS_TOKEN_ID", "59052"
                )
            )
            codec_embedding = targets.get("cosyvoice_codec_embed.weight")
            if codec_embedding is None:
                raise RuntimeError("missing cosyvoice_codec_embed.weight for dual BOS")
            if not (
                0 <= continue_bos < codec_embedding.shape[0]
                and 0 <= final_bos < codec_embedding.shape[0]
            ):
                raise RuntimeError(
                    "dual BOS ids outside codec embedding: "
                    f"continue={continue_bos} final={final_bos} "
                    f"rows={codec_embedding.shape[0]}"
                )
            # FINAL_BOS begins exactly as the proven offline BOS.  Only the
            # input embedding is cloned: BOS is never a training target, so
            # duplicating its output-head row would only add unwanted softmax
            # probability mass.
            codec_embedding.data[final_bos].copy_(
                codec_embedding.data[continue_bos]
            )
        state["loaded"] = True
        print(
            "[STREAMING S2ST] overlaid "
            f"{len(targets)} Talker/bridge/codec-I/O tensors "
            f"({loaded_values:,} values) from {checkpoint}",
            flush=True,
        )
        if os.environ.get("STREAMING_S2ST_INJECTED_DUAL_BOS", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            print(
                "[STREAMING S2ST] initialized injected FINAL_BOS embedding "
                f"row {final_bos} from CONTINUE_BOS row {continue_bos}",
                flush=True,
            )

    model.register_load_state_dict_post_hook(overlay)
