"""Run-local Qwen3.5 fusion; preserves final hidden for text/gate logits."""
import json
from functools import wraps
import os
from pathlib import Path
import torch

def reconstruct_block_output(output):
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError('Expected Qwen3.5 (hidden, residual) layer output')
    hidden, residual = output
    return hidden if residual is None else hidden + residual

def mix(hidden, weights):
    result = hidden[0].float() * weights[0].float()
    for i, value in enumerate(hidden[1:], 1):
        result = result + value.float() * weights[i].float()
    return result.to(hidden[-1].dtype)

def install(Model):
    if getattr(Model, '_layer_fusion_installed', False):
        return
    original_init, original_forward = Model.__init__, Model.forward

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        root = Path(os.environ['OMNIDRCA2_MODEL_DIR'])
        metadata = json.loads((root / 'thinker_layer_fusion.json').read_text())
        weights = torch.load(root / 'thinker_layer_fusion.pt', map_location='cpu', weights_only=True)['weights']
        layers = metadata['layers']
        backbone = self.language_model.model
        if (backbone.start_layer != 0 or backbone.end_layer != layers[-1]
            or len(backbone.layers) != layers[-1]):
            raise ValueError('Fusion requires an unsharded complete Thinker')
        if tuple(weights.shape) != (len(layers), metadata['hidden_size']):
            raise ValueError('Invalid fusion weight shape')
        # Plain tensor: sidecar weights must not enter vLLM HF weight-loader expectations.
        self._fusion_weights = weights
        self._fusion_layers = layers
        self._fusion_captured = {}
        self._fusion_active = False
        for number in layers[:-1]:
            def capture(module, inputs, output, number=number):
                if self._fusion_active:
                    if number in self._fusion_captured:
                        raise RuntimeError('Duplicate layer capture')
                    self._fusion_captured[number] = reconstruct_block_output(output).detach().clone()
            backbone.layers[number - 1].register_forward_hook(capture)
        original_projection = self.thinker_to_talker_projection
        def projection(last, accept):
            if set(self._fusion_captured) != set(layers[:-1]):
                raise RuntimeError('Missing intermediate Thinker layer output')
            self._fusion_weights = self._fusion_weights.to(last.device)
            states = [self._fusion_captured[i] for i in layers[:-1]] + [last]
            if any(x.shape != last.shape for x in states):
                raise RuntimeError('Layer capture shape mismatch')
            combined = mix(states, self._fusion_weights)
            _, projected = original_projection(combined, accept)
            # Never substitute the mixed state into language-model/gate logits.
            return last, projected
        self.thinker_to_talker_projection = projection
        print('[INFERENCE LAYER FUSION] layers=' + str(layers), flush=True)

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        if self._fusion_active:
            raise RuntimeError('Reentrant fusion forward')
        self._fusion_captured.clear()
        self._fusion_active = True
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            self._fusion_active = False
            self._fusion_captured.clear()

    Model.__init__, Model.forward = init, forward
    Model._layer_fusion_installed = True
