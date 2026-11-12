#!/usr/bin/env python3
"""Streaming S2TT entrypoint; imports offline code read-only.

Current implementation deliberately supports PP=1/CP=1.  It attaches a small
head to the final Thinker hidden state and derives gate labels from interleaved
audio feature spans, so the existing FunASR collator needs no modification.
"""

from __future__ import annotations

import os
import copy
import inspect
import sys
import warnings
from functools import partial
from pathlib import Path

import torch

# Keep the streaming head/entrypoint in this checkout, while allowing the
# proven offline training stack to be imported read-only.  This is useful when
# the surrounding checkout contains later S2ST model changes that should not
# alter the first-stage S2TT baseline.
_streaming_repo_root = Path(__file__).resolve().parents[1]
_offline_code_root = Path(
    os.environ.get("STREAMING_OFFLINE_CODE_ROOT", str(_streaming_repo_root))
).resolve()
sys.path.insert(0, str(_streaming_repo_root))
if _offline_code_root != _streaming_repo_root:
    # The historical tree imports optional FlashMoE unconditionally.  The
    # dense 4B path never uses it; a fail-closed import shim lives here so the
    # proven dense stack remains runnable on hosts without that package.
    sys.path.insert(0, str(_streaming_repo_root / "streaming" / "compat"))
    sys.path.insert(0, str(_offline_code_root))
from streaming.model.true_streaming_encoder import (
    adaptor_enabled,
    enabled as true_streaming_encoder_enabled,
    install_true_streaming_encoder_module,
    register_encoder_weight_overlay,
)

# This must run before pretrain_fun_base imports FunBaseModelChannels, whose
# module-level import binds the audio encoder class.
install_true_streaming_encoder_module(_streaming_repo_root / "streaming")
import pretrain_fun_base as offline
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain, print_rank_0
from megatron.training.utils import unwrap_model
from streaming.model.read_write_gate import (
    attach_streaming_gate,
    balanced_gate_loss,
    make_gate_targets,
)
from streaming.data.full_record_audio import install_full_record_audio_loader
from streaming.model.full_record_audio import (
    install_full_record_encoder_split,
    set_full_record_layout,
)


_native_offline_entry = os.environ.get(
    "STREAMING_DEBUG_NATIVE_OFFLINE_ENTRY", "0"
).lower() in {"1", "true", "yes"}
if not _native_offline_entry:
    install_full_record_audio_loader()


def native_offline_streaming_model_provider(pre_process=True, post_process=True):
    """Keep the offline model path while replacing/initializing only the encoder."""
    model = offline.model_provider(pre_process=pre_process, post_process=post_process)
    if true_streaming_encoder_enabled() and pre_process:
        encoder = getattr(model, "audio_encoder", None)
        adaptor = getattr(model, "audio_adaptor", None)
        encoder_source = (
            Path(inspect.getsourcefile(encoder.__class__)).resolve()
            if encoder is not None and inspect.getsourcefile(encoder.__class__)
            else None
        )
        expected_encoder_source = (
            _streaming_repo_root
            / "streaming/vendor/streaming_audio_encoder/audio_encoder.py"
        ).resolve()
        gated_delta_layers = [
            module for module in model.modules()
            if module.__class__.__name__ == "GatedDeltaNet"
        ]
        checks = {
            "model.encoder_streaming": bool(getattr(model, "encoder_streaming", False)),
            "audio_encoder.exists": encoder is not None,
            "audio_encoder.vendored_20260820_implementation": (
                encoder_source == expected_encoder_source
            ),
            "audio_encoder.streaming": bool(getattr(encoder, "streaming", False)),
            "audio_encoder.streaming_chunk_frames=16": (
                getattr(encoder, "streaming_chunk_frames", None) == 16
            ),
            "thinker.gated_delta_layers.exist": bool(gated_delta_layers),
            "thinker.gated_delta_layers.packed_safe_conv": all(
                getattr(layer, "_causal_conv1d_fn", None) is not None
                for layer in gated_delta_layers
            ),
        }
        if adaptor_enabled():
            checks.update({
                "audio_adaptor.exists": adaptor is not None,
                "audio_adaptor.streaming": bool(getattr(adaptor, "streaming", False)),
                "audio_adaptor.streaming_encoder_chunk_frames=16": (
                    getattr(adaptor, "streaming_encoder_chunk_frames", None) == 16
                ),
            })
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "true-streaming frontend model validation failed: " + ", ".join(failed)
            )
        print(
            "[TRUE STREAMING FRONTEND VERIFIED] encoder=causal-conv+block-causal-attn "
            "chunk=640ms adaptor=block-causal chunk=8 tokens "
            f"gated_delta_layers={len(gated_delta_layers)} packed-safe-conv=yes",
            flush=True,
        )
    register_encoder_weight_overlay(model, _streaming_repo_root / "streaming")
    return model


def _configure_gated_delta_rule_kernel() -> None:
    """Select an FLA kernel without editing the proven offline source tree."""
    from megatron.core.transformer import gated_delta_net
    from streaming.model.packed_causal_conv import install_packed_safe_causal_conv1d

    install_packed_safe_causal_conv1d()
    implementation = os.environ.get("STREAMING_GDR_IMPL", "chunk").strip().lower()
    if implementation == "chunk":
        if gated_delta_net.chunk_gated_delta_rule is None:
            raise RuntimeError(
                "STREAMING_GDR_IMPL=chunk requires FLA; use torch_packed only "
                "for a slow diagnostic fallback"
            )
        print(
            "[STREAMING] GatedDeltaNet recurrence=FLA chunk "
            "(cu_seqlens packed resets)",
            flush=True,
        )
        return
    if implementation not in {"fused_recurrent", "torch_packed"}:
        raise ValueError(
            "STREAMING_GDR_IMPL must be chunk, fused_recurrent, or torch_packed, got "
            f"{implementation!r}"
        )
    if implementation == "torch_packed":
        reference = gated_delta_net._torch_chunk_gated_delta_rule

        def torch_packed_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=64,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
            **unused,
        ):
            """Run the reference recurrence with an exact reset per packed item.

            The historical reference function predates ``cu_seqlens``.  Calling
            it once for an entire packed bin leaks recurrence state across
            examples.  This streaming-only diagnostic adapter slices the
            flattened batch at every packed boundary and therefore provides a
            numerically independent backward implementation for checking FLA.
            """
            if head_first:
                raise ValueError("torch_packed does not support head_first=True")
            if initial_state is not None:
                raise ValueError("torch_packed diagnostic expects initial_state=None")

            common = dict(
                chunk_size=chunk_size,
                initial_state=None,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            if cu_seqlens is None:
                return reference(query, key, value, g, beta, **common)
            if query.shape[0] != 1:
                raise ValueError(
                    "torch_packed expects flattened packed batch size 1, got "
                    f"{query.shape[0]}"
                )

            boundaries = [int(item) for item in cu_seqlens.detach().cpu().tolist()]
            outputs = []
            final_states = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end <= start:
                    continue
                output, final_state = reference(
                    query[:, start:end],
                    key[:, start:end],
                    value[:, start:end],
                    g[:, start:end],
                    beta[:, start:end],
                    **common,
                )
                outputs.append(output)
                if output_final_state:
                    final_states.append(final_state)
            if not outputs:
                raise ValueError("torch_packed received no non-empty sequences")
            output = torch.cat(outputs, dim=1)
            final_state = (
                torch.cat(final_states, dim=0) if output_final_state else None
            )
            return output, final_state

        # GatedDeltaNet.__init__ snapshots this module-level symbol.
        gated_delta_net.chunk_gated_delta_rule = torch_packed_gated_delta_rule
        print(
            "[STREAMING] GatedDeltaNet kernel=torch_packed "
            "(reference diagnostic with packed resets)",
            flush=True,
        )
        return

    fused = gated_delta_net.fused_recurrent_gated_delta_rule
    if fused is None:
        raise RuntimeError(
            "STREAMING_GDR_IMPL=fused_recurrent requires the FLA fused recurrent op"
        )
    # GatedDeltaNet.__init__ snapshots the module-level chunk function into
    # each layer, so replace it before model construction.  Both functions use
    # the same q/k/v/g/beta contract and support packed cu_seqlens.
    gated_delta_net.chunk_gated_delta_rule = fused
    print("[STREAMING] GatedDeltaNet kernel=fused_recurrent", flush=True)


_configure_gated_delta_rule_kernel()


_THINKER_PARAMETER_PREFIXES = ("embedding.", "decoder.", "output_layer.")
_debug_fixed_batch = None
_debug_forward_count = 0


def _clone_validation_batch(value):
    """Clone a cached CPU validation batch before every replay."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_validation_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_validation_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_validation_batch(item) for item in value)
    return copy.deepcopy(value)


def _fixed_validation_iterator(train_iterator):
    """Replay one deterministic validation batch for comparable probe loss."""
    _, raw_validation = train_iterator.dataloader.build_iter(
        0,
        data_split_i=int(os.environ.get("STREAMING_FIXED_VALIDATION_SPLIT", "0")),
        start_step=int(os.environ.get("STREAMING_FIXED_VALIDATION_START_STEP", "0")),
    )
    if raw_validation is None:
        raise RuntimeError(
            "STREAMING_FIXED_VALIDATION=1 but valid_data_set_list produced no iterator"
        )
    converted = offline.val_iterator(raw_validation, **train_iterator.kwargs)
    try:
        fixed_batch = next(converted)
    except StopIteration as exc:
        raise RuntimeError(
            "STREAMING_FIXED_VALIDATION=1 but the validation iterator is empty"
        ) from exc
    print_rank_0(
        "[STREAMING_DIAG] fixed validation replays one global batch on every eval"
    )
    while True:
        yield _clone_validation_batch(fixed_batch)


def build_streaming_data_iterators(train_valid_test_num_samples):
    """Keep the offline loader unchanged, optionally adding a fixed probe."""
    train_iterator, valid_iterator, test_iterator = offline.build_fun_data_iterators(
        train_valid_test_num_samples
    )
    fixed_validation = os.environ.get(
        "STREAMING_FIXED_VALIDATION", "0"
    ).lower() in {"1", "true", "yes"}
    if fixed_validation and train_iterator is not None:
        valid_iterator = _fixed_validation_iterator(train_iterator)
    return train_iterator, valid_iterator, test_iterator


def _install_nonfinite_gradient_debug_hooks(model) -> None:
    """Report the exact parameters that first receive NaN/Inf gradients.

    This is intentionally opt-in because checking every parameter introduces
    CUDA synchronizations.  It stays in the streaming entrypoint and therefore
    cannot change the offline training path.
    """
    enabled = os.environ.get(
        "STREAMING_DEBUG_NONFINITE_GRADS", "0"
    ).lower() in {"1", "true", "yes"}
    if not enabled:
        return

    prefixes = tuple(
        value.strip()
        for value in os.environ.get(
            "STREAMING_DEBUG_NONFINITE_GRAD_PREFIXES", ""
        ).split(",")
        if value.strip()
    )
    name_contains = tuple(
        value.strip()
        for value in os.environ.get(
            "STREAMING_DEBUG_GRAD_NAME_CONTAINS", ""
        ).split(",")
        if value.strip()
    )
    large_grad_threshold = float(
        os.environ.get("STREAMING_DEBUG_GRAD_ABS_MAX_THRESHOLD", "0")
    )
    max_reports = int(os.environ.get("STREAMING_DEBUG_NONFINITE_GRAD_MAX_REPORTS", "64"))
    state = {"reports": 0}

    def make_hook(parameter_name):
        def report_nonfinite(gradient):
            if gradient is None or state["reports"] >= max_reports:
                return gradient
            finite = torch.isfinite(gradient)
            all_finite = bool(finite.all())
            max_abs = float(gradient.detach().abs().max().item()) if all_finite else float("inf")
            if all_finite and not (
                large_grad_threshold > 0 and max_abs >= large_grad_threshold
            ):
                return gradient
            state["reports"] += 1
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
            nan_count = int(torch.isnan(gradient).sum().item()) if not all_finite else 0
            posinf_count = int(torch.isposinf(gradient).sum().item()) if not all_finite else 0
            neginf_count = int(torch.isneginf(gradient).sum().item()) if not all_finite else 0
            print(
                "[STREAMING DEBUG] "
                f"{'NONFINITE_GRAD' if not all_finite else 'LARGE_GRAD'} "
                f"rank={rank} forward_count={_debug_forward_count} "
                f"parameter={parameter_name} "
                f"shape={tuple(gradient.shape)} dtype={gradient.dtype} "
                f"max_abs={max_abs:.6e} nan={nan_count} "
                f"posinf={posinf_count} neginf={neginf_count}",
                flush=True,
            )
            return gradient

        return report_nonfinite

    installed = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if prefixes and not name.startswith(prefixes):
            continue
        if name_contains and not any(value in name for value in name_contains):
            continue
        parameter.register_hook(make_hook(name))
        installed += 1
    print_rank_0(
        "[STREAMING DEBUG] installed nonfinite-gradient hooks on "
        f"{installed} trainable parameters; prefixes={prefixes or ('<all>',)} "
        f"name_contains={name_contains or ('<all>',)} "
        f"large_grad_threshold={large_grad_threshold:g}"
    )


def _install_gdr_gradient_trace(model) -> None:
    """Trace the exact gradient entering and leaving every GatedDeltaNet op.

    This wraps only the per-instance recurrence callable after model creation;
    it does not edit the offline model or change any tensor values.  The trace
    is intended for one deterministic diagnostic step.
    """
    if os.environ.get("STREAMING_DEBUG_TRACE_GDR", "0").lower() not in {
        "1", "true", "yes"
    }:
        return

    import torch.distributed as dist

    target_rank = int(os.environ.get("STREAMING_DEBUG_TRACE_GDR_RANK", "0"))
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    if rank != target_rank:
        return
    output_path = Path(
        os.environ.get(
            "STREAMING_DEBUG_TRACE_GDR_PATH",
            str(Path(os.environ.get("EXP_DIR", ".")) / f"gdr_grad_trace_rank{rank}.log"),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    state = {"forward": 0}

    def tensor_stats(value):
        detached = value.detach().float()
        finite = torch.isfinite(detached)
        all_finite = bool(finite.all())
        finite_value = detached[finite]
        if finite_value.numel() == 0:
            return "finite=0 abs_max=inf norm=nan rms=nan"
        norm = float(finite_value.norm())
        rms = norm / max(finite_value.numel() ** 0.5, 1.0)
        abs_max = float(finite_value.abs().max())
        safe_abs = detached.abs().masked_fill(~finite, float("-inf"))
        flat_index = int(safe_abs.reshape(-1).argmax())
        coordinates = []
        remainder = flat_index
        for size in reversed(detached.shape):
            coordinates.append(remainder % size)
            remainder //= size
        coordinates.reverse()
        return (
            f"finite={int(all_finite)} abs_max={abs_max:.9e} "
            f"norm={norm:.9e} rms={rms:.9e} "
            f"max_index={tuple(coordinates)} nonfinite={int((~finite).sum())}"
        )

    def emit(message):
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    def register_tensor(tensor, *, layer, call, name):
        if not torch.is_tensor(tensor) or not tensor.requires_grad:
            return

        def report(gradient):
            emit(
                "[STREAMING GDR TRACE] "
                f"rank={rank} forward={state['forward']} layer={layer} "
                f"call={call} tensor={name} grad_{tensor_stats(gradient)}"
            )
            return gradient

        tensor.register_hook(report)

    installed = 0
    for module_name, module in model.named_modules():
        if not hasattr(module, "_chunk_fn") or not hasattr(module, "layer_number"):
            continue
        original = module._chunk_fn
        layer = int(module.layer_number) - 1
        call_state = {"count": 0}

        def traced_chunk(
            query, key, value, *args,
            _original=original, _layer=layer, _call_state=call_state, **kwargs
        ):
            _call_state["count"] += 1
            call = _call_state["count"]
            register_tensor(query, layer=_layer, call=call, name="q")
            register_tensor(key, layer=_layer, call=call, name="k")
            register_tensor(value, layer=_layer, call=call, name="v")
            register_tensor(kwargs.get("g"), layer=_layer, call=call, name="g")
            register_tensor(kwargs.get("beta"), layer=_layer, call=call, name="beta")
            result = _original(query, key, value, *args, **kwargs)
            output = result[0] if isinstance(result, (tuple, list)) else result
            register_tensor(output, layer=_layer, call=call, name="output")
            return result

        module._chunk_fn = traced_chunk
        installed += 1

    # Count outer model forwards so selective recompute calls can be separated.
    def mark_forward(_module, _inputs):
        state["forward"] += 1

    model.register_forward_pre_hook(mark_forward)
    print_rank_0(
        f"[STREAMING DEBUG] tracing GatedDeltaNet tensors for {installed} layers "
        f"on rank={rank}, path={output_path}"
    )


def _maybe_report_main_grads(model) -> None:
    """Inspect FP32 optimizer-facing main_grad after the previous microbatch."""
    enabled = os.environ.get(
        "STREAMING_DEBUG_MAIN_GRADS", "0"
    ).lower() in {"1", "true", "yes"}
    if not enabled or _debug_forward_count < 1:
        return
    prefixes = tuple(
        value.strip()
        for value in os.environ.get(
            "STREAMING_DEBUG_NONFINITE_GRAD_PREFIXES", ""
        ).split(",")
        if value.strip()
    )
    name_contains = tuple(
        value.strip()
        for value in os.environ.get(
            "STREAMING_DEBUG_GRAD_NAME_CONTAINS", ""
        ).split(",")
        if value.strip()
    )
    threshold = float(os.environ.get("STREAMING_DEBUG_GRAD_ABS_MAX_THRESHOLD", "0"))
    core_model = unwrap_model(model)
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    checked = missing = 0
    for name, parameter in core_model.named_parameters():
        if not parameter.requires_grad:
            continue
        if prefixes and not name.startswith(prefixes):
            continue
        if name_contains and not any(value in name for value in name_contains):
            continue
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            missing += 1
            continue
        checked += 1
        finite = torch.isfinite(gradient)
        all_finite = bool(finite.all())
        max_abs = float(gradient.detach().abs().max().item()) if all_finite else float("inf")
        if not all_finite or (threshold > 0 and max_abs >= threshold):
            print(
                "[STREAMING DEBUG] "
                f"{'NONFINITE_MAIN_GRAD' if not all_finite else 'LARGE_MAIN_GRAD'} "
                f"rank={rank} previous_forward_count={_debug_forward_count} "
                f"parameter={name} dtype={gradient.dtype} max_abs={max_abs:.6e}",
                flush=True,
            )
    if not getattr(core_model, "_streaming_main_grad_audited", False):
        print_rank_0(
            "[STREAMING DEBUG] optimizer-facing main_grad audit "
            f"checked={checked} missing={missing}"
        )
        core_model._streaming_main_grad_audited = True


def _maybe_dump_debug_batch(batch) -> None:
    """Persist metadata for selected optimizer iterations on TP source ranks."""
    global _debug_forward_count
    _debug_forward_count += 1
    raw_targets = os.environ.get("STREAMING_DEBUG_DUMP_BATCH_ITERATIONS", "").strip()
    raw_forward_targets = os.environ.get(
        "STREAMING_DEBUG_DUMP_BATCH_FORWARD_COUNTS", ""
    ).strip()
    if not raw_targets and not raw_forward_targets:
        return
    targets = {int(value.strip()) for value in raw_targets.split(",") if value.strip()}
    forward_targets = {
        int(value.strip())
        for value in raw_forward_targets.split(",")
        if value.strip()
    }
    args = get_args()
    iteration = int(getattr(args, "iteration", -1))
    if iteration not in targets and _debug_forward_count not in forward_targets:
        return

    from megatron.core import parallel_state

    if parallel_state.get_tensor_model_parallel_rank() != 0:
        return
    import torch.distributed as dist

    global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    dp_rank = parallel_state.get_data_parallel_rank()
    max_chars = int(os.environ.get("STREAMING_DEBUG_DUMP_BATCH_MAX_CHARS", "200000"))
    fields = (
        "items",
        "speech_info",
        "speech_lengths",
        "feature_beg",
        "feature_type",
        "fake_token_len",
        "sequence_chunks",
        "cu_seqlens",
        "packed_seq_params",
    )
    output_dir = Path(
        os.environ.get(
            "STREAMING_DEBUG_OUTPUT_DIR",
            os.environ.get("EXP_DIR", str(getattr(args, "save", "."))),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"debug_batches_rank{global_rank}.log"
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"iteration={iteration} forward_count={_debug_forward_count} "
            f"global_rank={global_rank} dp_rank={dp_rank}\n"
        )
        for field in fields:
            value_repr = repr(batch.get(field))
            stream.write(f"{field}={value_repr[:max_chars]}\n")
        stream.write("---\n")
    print(
        "[STREAMING DEBUG] dumped batch metadata "
        f"iteration={iteration} forward_count={_debug_forward_count} "
        f"global_rank={global_rank} dp_rank={dp_rank} path={output_path}",
        flush=True,
    )


def _trace_loss_boundary(batch, output_tensor) -> None:
    """Record CE-output gradients and mask normalization at the LM boundary."""
    if os.environ.get("STREAMING_DEBUG_TRACE_LOSS", "0").lower() not in {
        "1", "true", "yes"
    }:
        return
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    target_rank = int(os.environ.get("STREAMING_DEBUG_TRACE_GDR_RANK", "0"))
    if rank != target_rank:
        return
    loss_mask = batch.get("loss_mask")
    loss_mask_audio = batch.get("loss_mask_audio")
    # FunBaseModelChannels returns ``((text_losses, audio_losses), extras)``
    # before the pipeline schedule peels off ``extras``.
    primary_output = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
    text_losses = (
        primary_output[0] if isinstance(primary_output, tuple) else primary_output
    )
    output_path = Path(
        os.environ.get(
            "STREAMING_DEBUG_TRACE_GDR_PATH",
            str(Path(os.environ.get("EXP_DIR", ".")) / f"gdr_grad_trace_rank{rank}.log"),
        )
    )

    def stats(value):
        value = value.detach().float()
        finite = torch.isfinite(value)
        finite_value = value[finite]
        return (
            f"shape={tuple(value.shape)} finite={int(bool(finite.all()))} "
            f"min={float(finite_value.min()):.9e} max={float(finite_value.max()):.9e} "
            f"mean={float(finite_value.mean()):.9e} norm={float(finite_value.norm()):.9e}"
        )

    mask_sum = float(loss_mask.detach().float().sum()) if loss_mask is not None else -1.0
    audio_mask_sum = (
        float(loss_mask_audio.detach().float().sum())
        if loss_mask_audio is not None else -1.0
    )
    message = (
        "[STREAMING LOSS TRACE] "
        f"rank={rank} mask_sum={mask_sum:.9g} audio_mask_sum={audio_mask_sum:.9g} "
        f"text_losses_{stats(text_losses)}"
    )
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")
    print(message, flush=True)

    def report(gradient):
        grad_message = (
            f"[STREAMING LOSS TRACE] rank={rank} text_losses_grad_{stats(gradient)}"
        )
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(grad_message + "\n")
        print(grad_message, flush=True)
        return gradient

    if text_losses.requires_grad:
        text_losses.register_hook(report)


def _patch_torch_dist_writer_compat() -> None:
    """Adapt this Megatron checkout to the PyTorch 2.7 writer signature.

    Megatron's async filesystem writer calls the private PyTorch helper with
    four arguments.  PyTorch 2.7 prepended a storage-transform object.  Keep
    the compatibility shim local to the streaming entrypoint so offline code
    and the vendored Megatron tree remain untouched.
    """
    from megatron.core.dist_checkpointing.strategies import filesystem_async
    from torch.distributed.checkpoint.filesystem import _write_item as torch_write_item

    if len(inspect.signature(torch_write_item).parameters) != 5:
        return
    # `_StorageWriterTransforms` is a private symbol introduced together with
    # the five-argument writer in PyTorch 2.7.  Import it only after detecting
    # that signature so the proven PyTorch 2.6 offline environment keeps using
    # its native four-argument implementation.
    from torch.distributed.checkpoint.filesystem import _StorageWriterTransforms
    if getattr(filesystem_async._write_item, "_streaming_compat", False):
        return
    transforms = _StorageWriterTransforms()

    def write_item_compat(stream, data, write_item, storage_key):
        return torch_write_item(transforms, stream, data, write_item, storage_key)

    write_item_compat._streaming_compat = True
    filesystem_async._write_item = write_item_compat


def _synchronize_gate_tensor_parallel(model) -> int:
    """Keep the small replicated gate identical when Thinker uses TP>1."""
    import torch.distributed as dist
    from megatron.core import parallel_state

    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_world_size == 1:
        return tp_world_size
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_src_rank = parallel_state.get_tensor_model_parallel_src_rank()

    def average_tp_gradient(gradient):
        dist.all_reduce(gradient, group=tp_group)
        gradient.mul_(1.0 / tp_world_size)
        return gradient

    for parameter in model.streaming_gate.parameters():
        parameter.register_hook(average_tp_gradient)

    # model_provider returns CPU parameters; NCCL cannot broadcast them yet.
    # Delay the one-time replica initialization until Megatron has moved the
    # model to CUDA and the gate is about to run for the first time.
    sync_handle = None

    def sync_on_first_forward(module, _inputs):
        nonlocal sync_handle
        for parameter in module.parameters():
            if parameter.device.type != "cuda":
                raise RuntimeError("streaming gate reached forward before CUDA placement")
            dist.broadcast(parameter.data, src=tp_src_rank, group=tp_group)
        sync_handle.remove()
        sync_handle = None

    sync_handle = model.streaming_gate.register_forward_pre_hook(sync_on_first_forward)
    return tp_world_size


def _gather_sequence_parallel_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """Restore full-sequence decoder states before applying global gate positions.

    The gate and its loss are replicated after the gather, so backward must
    split (not reduce-scatter) the hidden gradient back to each TP rank.
    Gate parameter gradients are synchronized separately above.
    """
    args = get_args()
    if not args.sequence_parallel or args.tensor_model_parallel_size == 1:
        return hidden
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
    )

    return gather_from_sequence_parallel_region(
        hidden,
        tensor_parallel_output_grad=False,
    )


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    if args.pipeline_model_parallel_size != 1 or args.context_parallel_size != 1:
        raise ValueError("streaming gate v1 requires PP=1 and CP=1")
    model = offline.model_provider(pre_process=pre_process, post_process=post_process)
    register_encoder_weight_overlay(model, _streaming_repo_root / "streaming")
    install_full_record_encoder_split(model)
    if os.environ.get("STREAMING_DEBUG_DUMP_GDR_PARAMS", "0").lower() in {
        "1", "true", "yes"
    }:
        def report_gdr_parameters(module, _incompatible_keys):
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
            for name, parameter in module.named_parameters():
                if not (name.endswith("self_attention.A_log") or name.endswith("self_attention.dt_bias")):
                    continue
                value = parameter.detach().float()
                print(
                    "[STREAMING DEBUG] GDR_PARAM_AFTER_LOAD "
                    f"rank={rank} name={name} shape={tuple(value.shape)} "
                    f"min={float(value.min()):.9g} max={float(value.max()):.9g} "
                    f"mean={float(value.mean()):.9g} norm={float(value.norm()):.9g}",
                    flush=True,
                )

        # Model construction happens before the distributed checkpoint load.
        # A post-load hook reports the actual checkpoint values without
        # changing checkpoint or offline loader behavior.
        model.register_load_state_dict_post_hook(report_gdr_parameters)
    disable_gate = os.environ.get("STREAMING_DEBUG_DISABLE_GATE", "0").lower() in {
        "1", "true", "yes"
    }
    preserve_offline_trainable = disable_gate and os.environ.get(
        "STREAMING_DEBUG_PRESERVE_OFFLINE_TRAINABLE", "0"
    ).lower() in {"1", "true", "yes"}
    train_audio_adaptor = os.environ.get(
        "STREAMING_DEBUG_TRAIN_AUDIO_ADAPTOR", "0"
    ).lower() in {"1", "true", "yes"}
    train_frontend = os.environ.get(
        "STREAMING_TRAIN_FRONTEND", "0"
    ).lower() in {"1", "true", "yes"}
    train_audio_adaptor = train_audio_adaptor or train_frontend
    if not disable_gate:
        attach_streaming_gate(
            model,
            bottleneck=int(os.environ.get("STREAM_GATE_BOTTLENECK", "256")),
            dropout=float(os.environ.get("STREAM_GATE_DROPOUT", "0.1")),
        )
    stage = os.environ.get("STREAMING_TRAIN_STAGE", "gate").lower()
    if stage not in {"gate", "joint"}:
        raise ValueError("STREAMING_TRAIN_STAGE must be gate or joint")
    if not preserve_offline_trainable:
        for name, parameter in model.named_parameters():
            if not disable_gate and name.startswith("streaming_gate."):
                parameter.requires_grad = True
                parameter.lr_scale = float(os.environ.get("STREAM_GATE_LR_SCALE", "20"))
            elif stage == "joint" and name.startswith(_THINKER_PARAMETER_PREFIXES):
                parameter.requires_grad = True
            elif (
                stage == "joint"
                and train_audio_adaptor
                and name.startswith("audio_adaptor.")
            ):
                parameter.requires_grad = True
            elif (
                stage == "joint"
                and train_frontend
                and name.startswith("audio_encoder.")
            ):
                parameter.requires_grad = True
            else:
                # This also covers the speech encoder/adaptor, speech tokenizer,
                # audio/Talker heads and all CosyVoice components.
                parameter.requires_grad = False

    if preserve_offline_trainable:
        trainable = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        print_rank_0(
            "[STREAMING DEBUG] preserving the offline model's original "
            f"requires_grad flags; trainable_parameters={trainable:,}"
        )
        _install_nonfinite_gradient_debug_hooks(model)
        _install_gdr_gradient_trace(model)
        return model

    allowed_prefixes = () if disable_gate else ("streaming_gate.",)
    if stage == "joint":
        allowed_prefixes += _THINKER_PARAMETER_PREFIXES
        if train_audio_adaptor:
            allowed_prefixes += ("audio_adaptor.",)
        if train_frontend:
            allowed_prefixes += ("audio_encoder.",)
    unexpected = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise RuntimeError(
            f"{stage} stage has parameters outside the S2TT scope: {unexpected[:10]}"
        )

    trainable_by_group = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = next(prefix.rstrip(".") for prefix in allowed_prefixes if name.startswith(prefix))
        trainable_by_group[group] = trainable_by_group.get(group, 0) + parameter.numel()
    required_groups = set() if disable_gate else {"streaming_gate"}
    if stage == "joint":
        # With tied word embeddings, named_parameters() registers the shared
        # tensor under embedding.* only, so output_layer.* is optional.
        required_groups.update({"embedding", "decoder"})
    missing_groups = sorted(required_groups - trainable_by_group.keys())
    if missing_groups:
        raise RuntimeError(
            f"{stage} stage is missing required trainable groups: {missing_groups}"
        )

    trainable = sum(trainable_by_group.values())
    group_summary = ", ".join(
        f"{name}={count:,}" for name, count in sorted(trainable_by_group.items())
    )
    print_rank_0(
        f"[STREAMING] stage={stage}, trainable_parameters={trainable:,} "
        f"({group_summary})"
    )
    if disable_gate:
        print_rank_0(
            "[STREAMING DEBUG] gate module/hook disabled; using native offline LM path"
        )
    else:
        write_weight = float(os.environ.get("STREAM_GATE_WRITE_WEIGHT", "3.33"))
        if write_weight <= 0:
            raise ValueError(f"STREAM_GATE_WRITE_WEIGHT must be positive, got {write_weight}")
        print_rank_0(
            "[STREAMING] gate_loss=class_weighted_cross_entropy, "
            f"class_weights=(READ=1.0, WRITE={write_weight:g})"
        )
        gate_tp_replicas = _synchronize_gate_tensor_parallel(model)
        print_rank_0(
            f"[STREAMING] gate tensor-parallel replicas={gate_tp_replicas}"
        )
    _install_nonfinite_gradient_debug_hooks(model)
    _install_gdr_gradient_trace(model)
    return model


def streaming_loss_func(model, loss_mask, loss_mask_audio, output_tensor, *extra_outputs):
    lm_loss, metrics = offline.loss_func(loss_mask, loss_mask_audio, output_tensor, *extra_outputs)
    gate_loss = model._streaming_gate_loss
    weight = float(os.environ.get("STREAM_GATE_LOSS_WEIGHT", "1.0"))
    # A zero diagnostic weight must disconnect the gate graph completely.  It
    # is used by the fixed-bin LM-only optimizer-direction test.
    total = lm_loss if weight == 0.0 else lm_loss + weight * gate_loss
    gate_metrics = offline.average_losses_across_data_parallel_group([gate_loss.detach()])

    import torch.distributed as dist
    from megatron.core import parallel_state

    # Aggregate counts before forming ratios.  Averaging per-rank accuracy is
    # biased when packed batches contain different numbers of decision points.
    confusion = model._streaming_gate_confusion.detach().clone()
    dist.all_reduce(confusion, group=parallel_state.get_data_parallel_group())
    tp, tn, fp, fn = confusion.unbind()
    total_count = (tp + tn + fp + fn).clamp_min(1.0)
    pred_write = (tp + fp) / total_count
    true_write = (tp + fn) / total_count
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    true_read_recall = tn / (tn + fp).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)

    metrics["stream_gate_loss"] = gate_metrics[0]
    metrics["stream_gate_accuracy"] = (tp + tn) / total_count
    metrics["stream_gate_balanced_accuracy"] = 0.5 * (recall + true_read_recall)
    metrics["stream_gate_precision"] = precision
    metrics["stream_gate_recall"] = recall
    metrics["stream_gate_f1"] = f1
    metrics["stream_gate_write_rate"] = pred_write
    metrics["stream_gate_true_write_rate"] = true_write
    return total, metrics


def forward_step(data_iterator, model):
    global _debug_fixed_batch
    _maybe_report_main_grads(model)
    repeat_first_batch = os.environ.get(
        "STREAMING_DEBUG_REPEAT_FIRST_BATCH", "0"
    ).lower() in {"1", "true", "yes"}
    if repeat_first_batch and _debug_fixed_batch is not None:
        batch = _debug_fixed_batch
    else:
        batch = offline.get_batch(data_iterator)
        if repeat_first_batch:
            # Inputs do not require gradients and the model treats the batch as
            # read-only, so retaining the post-collation CUDA tensors is enough
            # to make every later optimizer step see the exact same inputs.
            _debug_fixed_batch = batch
            print_rank_0(
                "[STREAMING DEBUG] cached first post-collation batch; "
                "all later steps will reuse the same tensors"
            )
    _maybe_dump_debug_batch(batch)
    core_model = unwrap_model(model)
    set_full_record_layout(core_model, batch)
    disable_gate = os.environ.get("STREAMING_DEBUG_DISABLE_GATE", "0").lower() in {
        "1", "true", "yes"
    }
    if disable_gate:
        output_tensor = model(**batch)
        _trace_loss_boundary(batch, output_tensor)
        return output_tensor, partial(
            offline.loss_func,
            batch.get("loss_mask"),
            batch.get("loss_mask_audio"),
        )

    # Megatron passes the DDP/Float16 wrapper here, while the capture hook and
    # streaming head are installed on the underlying FunBaseModel.
    core_model._streaming_hidden_states = None
    output_tensor = model(**batch)
    _trace_loss_boundary(batch, output_tensor)
    hidden = getattr(core_model, "_streaming_hidden_states", None)
    if hidden is None:
        raise RuntimeError("Thinker decoder hook did not capture hidden states")
    hidden = _gather_sequence_parallel_hidden(hidden)
    gate_backprop_to_thinker = os.environ.get(
        "STREAM_GATE_BACKPROP_TO_THINKER", "1"
    ).lower() not in {"0", "false", "no"}
    gate_thinker_grad_scale = (
        float(os.environ.get("STREAM_GATE_THINKER_GRAD_SCALE", "1.0"))
        if gate_backprop_to_thinker
        else 0.0
    )
    if gate_thinker_grad_scale < 0:
        raise ValueError(
            "STREAM_GATE_THINKER_GRAD_SCALE must be non-negative, got "
            f"{gate_thinker_grad_scale}"
        )
    batch_size = int(batch["feature_beg"].shape[0])
    tokens = batch["input_ids"]
    if hidden.shape[1] != batch_size:
        raise RuntimeError(
            f"gate hidden batch mismatch: hidden={tuple(hidden.shape)}, "
            f"feature_beg={tuple(batch['feature_beg'].shape)}"
        )
    token_seq_len = int(tokens.shape[1] if tokens.shape[0] == batch_size else tokens.shape[0])
    if hidden.shape[0] != token_seq_len:
        raise RuntimeError(
            f"gate hidden must cover the full sequence: hidden={tuple(hidden.shape)}, "
            f"tokens={tuple(tokens.shape)}, sequence_parallel={get_args().sequence_parallel}"
        )
    if not getattr(core_model, "_streaming_hidden_shape_audited", False):
        print_rank_0(
            f"[STREAMING] gate full hidden shape={tuple(hidden.shape)}, "
            f"sequence_parallel={get_args().sequence_parallel}, "
            f"backprop_to_thinker={gate_backprop_to_thinker}, "
            f"thinker_grad_scale={gate_thinker_grad_scale:g}"
        )
        core_model._streaming_hidden_shape_audited = True
    # Keep the gate-head gradient at full strength while independently scaling
    # only the strategy gradient that enters the Thinker.  The forward value is
    # exactly ``hidden`` for every scale.
    hidden_detached = hidden.detach()
    gate_input_hidden = hidden_detached + gate_thinker_grad_scale * (
        hidden - hidden_detached
    )
    gate_batch = make_gate_targets(
        gate_input_hidden,
        batch["feature_beg"],
        batch["feature_type"],
        batch["fake_token_len"],
        interval_tokens=int(os.environ.get("STREAM_GATE_INTERVAL_TOKENS", "8")),
    )
    if gate_batch is None:
        raise RuntimeError("batch contains no complete 640-ms audio decision interval")
    logits = core_model.streaming_gate(gate_batch.hidden)
    core_model._streaming_gate_loss = balanced_gate_loss(
        logits,
        gate_batch.labels,
        write_weight=float(os.environ.get("STREAM_GATE_WRITE_WEIGHT", "3.33")),
    )
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        labels = gate_batch.labels
        core_model._streaming_gate_confusion = torch.stack(
            [
                ((pred == 1) & (labels == 1)).sum(),
                ((pred == 0) & (labels == 0)).sum(),
                ((pred == 1) & (labels == 0)).sum(),
                ((pred == 0) & (labels == 1)).sum(),
            ]
        ).float()
    return output_tensor, partial(
        streaming_loss_func,
        core_model,
        batch.get("loss_mask"),
        batch.get("loss_mask_audio"),
    )


if __name__ == "__main__":
    _patch_torch_dist_writer_compat()
    warnings.filterwarnings(
        "ignore",
        message="The given NumPy array is not writable.*",
        category=UserWarning,
    )
    if _native_offline_entry:
        print(
            "[STREAMING DEBUG] using native offline model/data/forward entry",
            flush=True,
        )
        selected_data_provider = offline.build_fun_data_iterators
        selected_model_provider = native_offline_streaming_model_provider
        selected_forward_step = offline.forward_step
    else:
        selected_data_provider = build_streaming_data_iterators
        selected_model_provider = model_provider
        selected_forward_step = forward_step
    selected_data_provider.is_distributed = True
    pretrain(
        selected_data_provider,
        selected_model_provider,
        ModelType.encoder_or_decoder,
        selected_forward_step,
        args_defaults={"tokenizer_type": "QWenVLTokenizer"},
        use_train_valid_test_dataset_provider_as_data_iterators=True,
        scale_lr_cond=offline.scale_lr_cond,
    )
