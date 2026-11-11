"""Optional frozen-Thinker layer mixing before the existing Talker bridge.

Layer numbers are one-based. Intermediate taps are block outputs; the last tap
is the existing final-normalized decoder output, so last-only initialization
exactly preserves the original conditioning. No extra normalization/softmax.
"""
import os
import types

import torch
from torch import nn


class HadamardLayerFusion(nn.Module):
    def __init__(self, layer_numbers, hidden_size):
        super().__init__()
        self.layer_numbers = tuple(layer_numbers)
        if not self.layer_numbers or list(self.layer_numbers) != sorted(set(self.layer_numbers)):
            raise ValueError("Layer numbers must be nonempty, unique and increasing")
        self.weights = nn.Parameter(torch.zeros(len(self.layer_numbers), hidden_size))
        with torch.no_grad():
            self.weights[-1].fill_(1.0)

    def forward(self, hidden_states):
        if len(hidden_states) != len(self.layer_numbers):
            raise RuntimeError("Missing Thinker layer output")
        shape = hidden_states[-1].shape
        if any(h.shape != shape for h in hidden_states) or shape[-1] != self.weights.shape[-1]:
            raise RuntimeError("Thinker layer shape mismatch")
        # Accumulate in float32; the original bridge still receives its usual dtype.
        mixed = hidden_states[0].float() * self.weights[0].float()
        for index, hidden in enumerate(hidden_states[1:], 1):
            mixed = mixed + hidden.float() * self.weights[index].float()
        return mixed.to(hidden_states[-1].dtype)


def install_thinker_layer_fusion(model, enabled=None, stride=None):
    if enabled is None:
        enabled = os.environ.get("STREAMING_S2ST_THINKER_LAYER_FUSION", "0").lower() in {"1", "true", "yes"}
    if not enabled:
        return None
    if hasattr(model, "thinker_layer_fusion"):
        raise RuntimeError("Thinker layer fusion is already installed")
    cfg = model.config
    if any(getattr(cfg, name, 1) != 1 for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size")):
        raise ValueError("Layer fusion currently requires TP=PP=CP=1")
    if getattr(cfg, "recompute_granularity", None) == "full":
        raise ValueError("Layer capture supports the baseline selective recompute, not full recompute")
    if any(p.requires_grad for p in model.decoder.parameters()):
        raise ValueError("This Talker-first layer fusion requires a frozen Thinker")
    if getattr(cfg, "cosyvoice_codex_use_text_token_input", False):
        raise ValueError("Layer fusion requires hidden-state conditioning")
    layers = model.decoder.layers
    stride = int(stride or os.environ.get("STREAMING_S2ST_THINKER_LAYER_STRIDE", "4"))
    if stride <= 0 or len(layers) % stride:
        raise ValueError("Stride must divide the Thinker layer count")
    selected = tuple(range(stride, len(layers) + 1, stride))
    for index, layer in enumerate(layers, 1):
        if getattr(layer, "layer_number", index) != index:
            raise ValueError("Unexpected Thinker layer numbering")
    fusion = HadamardLayerFusion(selected, cfg.hidden_size)
    bridge = model.cosyvoice_llm_input_proj
    fusion.to(device=bridge.weight.device, dtype=bridge.weight.dtype)
    model.add_module("thinker_layer_fusion", fusion)
    captured = {}
    active = [False]
    used = [False]

    def capture(index):
        def hook(module, inputs, output):
            if not active[0]:
                return
            if index in captured:
                raise RuntimeError("Thinker layer executed twice within one forward")
            hidden = output[0] if isinstance(output, tuple) else output
            # Frozen Thinker: retain only these activations, no backbone graph.
            captured[index] = hidden.detach().clone()
        return hook

    for index in selected[:-1]:
        layers[index - 1].register_forward_hook(capture(index))

    def replace_bridge_input(module, inputs):
        if not active[0] or used[0] or len(inputs) != 1:
            raise RuntimeError("Unexpected Thinker-to-Talker projection invocation")
        if set(captured) != set(selected[:-1]):
            raise RuntimeError("Incomplete Thinker layer capture")
        last = inputs[0]
        hidden = [captured[index] for index in selected[:-1]] + [last.detach()]
        mixed = fusion(hidden)
        used[0] = True
        if not getattr(fusion, "_verified_first_forward", False):
            # A fresh adapter must be exactly equivalent before the first update.
            identity = bool(torch.all(fusion.weights[:-1] == 0) and torch.all(fusion.weights[-1] == 1))
            if identity and not torch.equal(mixed, last):
                raise RuntimeError("Last-only fusion initialization changed conditioning")
            fusion._verified_first_forward = True
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"[THINKER LAYER FUSION] layers={selected} parameters={fusion.weights.numel()} identity_init={identity} shape={tuple(mixed.shape)}", flush=True)
        return (mixed,)

    bridge.register_forward_pre_hook(replace_bridge_input)
    original_forward = model.forward

    def forward_with_capture(self, *args, **kwargs):
        if active[0]:
            raise RuntimeError("Reentrant forward is unsupported for layer fusion")
        captured.clear()
        active[0] = True
        used[0] = False
        try:
            result = original_forward(*args, **kwargs)
            if not used[0]:
                raise RuntimeError("Forward bypassed the layer fusion bridge")
            return result
        finally:
            active[0] = False
            captured.clear()

    model.forward = types.MethodType(forward_with_capture, model)
    return fusion
