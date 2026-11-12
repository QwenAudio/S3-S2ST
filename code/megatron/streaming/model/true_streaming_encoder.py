"""Streaming-only integration for reference 640-ms SenseVoice encoder.

The offline source tree stays read-only.  When enabled, this module installs the
vendored block-causal encoder implementation before ``pretrain_fun_base`` imports
the multimodal model, and overlays only ``audio_encoder.*`` weights after the
parent checkpoint has been restored.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch


_MODULE_NAME = "megatron.core.models.multimodal.audio_encoder"
_ADAPTOR_MODULE_NAME = "megatron.core.models.multimodal.audio_adaptor"


def enabled() -> bool:
    return os.environ.get("STREAMING_TRUE_ENCODER", "0").lower() in {
        "1", "true", "yes",
    }


def adaptor_enabled() -> bool:
    """Keep the new causal adaptor opt-in so historical runs stay reproducible."""
    return os.environ.get("STREAMING_TRUE_ADAPTOR", "0").lower() in {
        "1", "true", "yes",
    }


def _patch_true_streaming_adaptor(chunk_tokens: int) -> None:
    """Patch the selected offline stack's adaptor without editing it in place."""
    import importlib

    module = importlib.import_module(_ADAPTOR_MODULE_NAME)
    adaptor_cls = getattr(module, "AudioAdaptor", None)
    if adaptor_cls is not None and hasattr(adaptor_cls, "_streaming_attention_mask"):
        # reference stack has the production block-causal adaptor.
        # Rely on it directly instead of wrapping every attention call twice.
        print(
            "[TRUE STREAMING ADAPTOR] "
            f"implementation={module.__file__} native block-causal "
            f"chunk={chunk_tokens} adaptor tokens",
            flush=True,
        )
        return
    attention_cls = module.MultiHeadedAttention
    installed = getattr(attention_cls, "_streaming_chunk_tokens", None)
    if installed is not None:
        if installed != chunk_tokens:
            raise RuntimeError(
                "true-streaming adaptor already installed with a different chunk: "
                f"{installed} != {chunk_tokens}"
            )
        return

    original_forward = attention_cls.forward

    def block_causal_forward(this, query, key, value, mask=None):
        query_time = query.shape[1]
        key_time = key.shape[1]
        if query_time != key_time:
            raise ValueError(
                "streaming audio adaptor expects self-attention, got "
                f"query={query_time} key={key_time}"
            )

        # The first adaptor block receives [B,1,T] padding validity.  Later
        # blocks receive the already expanded [B,T,T] block-causal mask.
        if (
            mask is not None
            and mask.ndim == 3
            and mask.shape[-2:] == (query_time, key_time)
        ):
            streaming_mask = mask.to(dtype=torch.bool)
        else:
            if mask is None:
                valid = torch.ones(
                    query.shape[0], key_time,
                    device=query.device, dtype=torch.bool,
                )
            elif mask.ndim == 3 and mask.shape[1] == 1:
                valid = mask[:, 0, :].to(dtype=torch.bool)
            elif mask.ndim == 2:
                valid = mask.to(dtype=torch.bool)
            else:
                raise ValueError(
                    f"unexpected audio adaptor mask shape: {tuple(mask.shape)}"
                )
            positions = torch.arange(key_time, device=query.device)
            chunk_ids = torch.div(
                positions, chunk_tokens, rounding_mode="floor"
            )
            block_causal = chunk_ids.unsqueeze(0) <= chunk_ids.unsqueeze(1)
            streaming_mask = (
                block_causal.unsqueeze(0)
                & valid.unsqueeze(1)
                & valid.unsqueeze(2)
            )
        return original_forward(this, query, key, value, streaming_mask)

    attention_cls.forward = block_causal_forward
    attention_cls._streaming_chunk_tokens = chunk_tokens
    print(
        "[TRUE STREAMING ADAPTOR] "
        f"implementation={module.__file__} "
        f"block-causal chunk={chunk_tokens} adaptor tokens",
        flush=True,
    )


def install_true_streaming_encoder_module(streaming_root: Path) -> None:
    """Replace only the imported audio-encoder module with the vendored version."""
    if not enabled():
        return

    chunk_ms = float(os.environ.get("STREAMING_TRUE_ENCODER_CHUNK_MS", "640"))
    if chunk_ms <= 0:
        raise ValueError(f"STREAMING_TRUE_ENCODER_CHUNK_MS must be positive: {chunk_ms}")

    # Import the parent package and config from the selected offline stack first;
    # multimodal/__init__.py does not import the encoder itself.
    from megatron.core.transformer.transformer_config import AudioConfig
    import megatron.core.models.multimodal  # noqa: F401

    # Older offline stacks predate these fields, so retain class-level defaults
    # only for those stacks.  Newer reference dataclasses have instance fields whose
    # generated __init__ defaults cannot be changed by assigning the class
    # attribute after class creation; the launcher must pass the real CLI flag.
    dataclass_fields = getattr(AudioConfig, "__dataclass_fields__", {})
    if "encoder_streaming" not in dataclass_fields:
        AudioConfig.encoder_streaming = True
    if "encoder_streaming_chunk_ms" not in dataclass_fields:
        AudioConfig.encoder_streaming_chunk_ms = chunk_ms
    if "adaptor_streaming_chunk_size" not in dataclass_fields:
        AudioConfig.adaptor_streaming_chunk_size = 0

    # Patch the factory before pretrain_fun_base imports it by name.  This does
    # not silently force streaming: it fails closed if the generated launch
    # command omitted --audio-encoder-streaming or supplied a different chunk.
    from megatron.training import arguments as training_arguments

    original_factory = training_arguments.core_audio_model_config_from_args
    installed_chunk = getattr(original_factory, "_true_streaming_chunk_ms", None)
    if installed_chunk is not None and installed_chunk != chunk_ms:
        raise RuntimeError(
            "audio config factory already installed with a different streaming "
            f"chunk: {installed_chunk} != {chunk_ms}"
        )
    if installed_chunk is None:
        def checked_audio_config_from_args(args):
            config = original_factory(args)
            if not bool(getattr(config, "encoder_streaming", False)):
                raise RuntimeError(
                    "STREAMING_TRUE_ENCODER=1 requires the generated training "
                    "command to include --audio-encoder-streaming"
                )
            actual_chunk = float(getattr(config, "encoder_streaming_chunk_ms", -1))
            if abs(actual_chunk - chunk_ms) > 1e-6:
                raise RuntimeError(
                    "training audio encoder chunk disagrees with the streaming "
                    f"runtime: config={actual_chunk} ms runtime={chunk_ms} ms"
                )
            return config

        checked_audio_config_from_args._true_streaming_chunk_ms = chunk_ms
        training_arguments.core_audio_model_config_from_args = checked_audio_config_from_args
    if adaptor_enabled():
        # SenseVoice emits one encoder frame per 40 ms and the adaptor joins
        # pairs of frames, so a 640-ms encoder chunk contains 8 adaptor tokens.
        default_chunk_tokens = max(1, round(chunk_ms / 80.0))
        chunk_tokens = int(os.environ.get(
            "STREAMING_TRUE_ADAPTOR_CHUNK_TOKENS", str(default_chunk_tokens)
        ))
        if chunk_tokens <= 0:
            raise ValueError(
                "STREAMING_TRUE_ADAPTOR_CHUNK_TOKENS must be positive: "
                f"{chunk_tokens}"
            )
        if "adaptor_streaming_chunk_size" not in dataclass_fields:
            AudioConfig.adaptor_streaming_chunk_size = chunk_tokens
        _patch_true_streaming_adaptor(chunk_tokens)

    source = (
        streaming_root
        / "vendor"
        / "streaming_audio_encoder"
        / "audio_encoder.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"missing vendored true-streaming encoder: {source}")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load true-streaming encoder module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    print(
        f"[TRUE STREAMING ENCODER] implementation={source} chunk_ms={chunk_ms:g}",
        flush=True,
    )


def register_encoder_weight_overlay(model, streaming_root: Path) -> None:
    """Load reference weights after the user's full Thinker/MLP checkpoint."""
    if not enabled():
        return
    if os.environ.get("STREAMING_TRUE_ENCODER_OVERLAY", "1").lower() not in {
        "1", "true", "yes",
    }:
        print(
            "[TRUE STREAMING ENCODER] preserving encoder weights from resumed checkpoint",
            flush=True,
        )
        return
    weight_path = Path(
        os.environ.get(
            "STREAMING_TRUE_ENCODER_CKPT",
            str(
                streaming_root
                / "assets"
                / "true_streaming_encoder"
                / "streaming_encoder"
                / "audio_encoder.pt"
            ),
        )
    ).resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f"missing true-streaming encoder checkpoint: {weight_path}")

    state = {"loaded": False}

    def overlay(module, _incompatible_keys):
        if state["loaded"]:
            return
        raw = torch.load(weight_path, map_location="cpu", weights_only=False)
        for container_key in ("state_dict", "model"):
            if isinstance(raw, dict) and isinstance(raw.get(container_key), dict):
                raw = raw[container_key]
        if not isinstance(raw, dict):
            raise TypeError(f"unexpected encoder checkpoint type: {type(raw)!r}")
        prefix = "audio_encoder."
        encoder_state = {
            key[len(prefix):]: value
            for key, value in raw.items()
            if key.startswith(prefix)
        }
        if len(encoder_state) != len(raw):
            unexpected = sorted(key for key in raw if not key.startswith(prefix))[:8]
            raise RuntimeError(
                "encoder-only checkpoint contains non-encoder keys: " + repr(unexpected)
            )
        # HF/vLLM exports Conv1d as a direct nn.Conv1d, while the Megatron
        # SenseVoice module wraps the same tensor in Conv1d.conv.  This is the
        # only naming translation; values and shapes remain unchanged.
        for layer in ("conv1", "conv2"):
            for suffix in ("weight", "bias"):
                exported = f"{layer}.{suffix}"
                megatron = f"{layer}.conv.{suffix}"
                if exported in encoder_state:
                    encoder_state[megatron] = encoder_state.pop(exported)
        target = module.audio_encoder
        expected = target.state_dict()
        missing = sorted(set(expected) - set(encoder_state))
        extra = sorted(set(encoder_state) - set(expected))
        mismatched = sorted(
            key for key in set(expected) & set(encoder_state)
            if tuple(expected[key].shape) != tuple(encoder_state[key].shape)
        )
        if missing or extra or mismatched:
            raise RuntimeError(
                "true-streaming encoder is incompatible: "
                f"missing={missing[:8]} extra={extra[:8]} shape={mismatched[:8]}"
            )
        target.load_state_dict(encoder_state, strict=True)
        state["loaded"] = True
        print(
            f"[TRUE STREAMING ENCODER] overlaid {len(encoder_state)} tensors from {weight_path}",
            flush=True,
        )

    model.register_load_state_dict_post_hook(overlay)
