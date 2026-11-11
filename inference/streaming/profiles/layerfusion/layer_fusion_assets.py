"""Strict, portable inference metadata for Thinker layer fusion."""
import hashlib
import json
from pathlib import Path

CONVENTION = 'intermediate_post_block_final_post_norm'

def resolve(model, mode='auto', profile='auto'):
    model = Path(model)
    mode = {'1': 'on', '0': 'off'}.get(mode, mode)
    config_path = model / 'thinker_layer_fusion.json'
    weights_path = model / 'thinker_layer_fusion.pt'
    hf = json.loads((model / 'config.json').read_text())
    declared = bool(hf.get('thinker_layer_fusion'))
    present = config_path.exists() or weights_path.exists() or declared
    if mode not in ('auto', 'on', 'off'):
        raise ValueError('Fusion mode must be auto/on/off')
    if present and not (config_path.is_file() and weights_path.is_file()):
        raise ValueError('Incomplete layer fusion assets; reconvert the checkpoint')
    if mode == 'on' and not present:
        raise ValueError('Fusion requested but checkpoint has no fusion weights')
    metadata = json.loads(config_path.read_text()) if present else None
    if metadata:
        text = hf.get('text_config', hf)
        layers = metadata['layers']
        if (metadata.get('version') != 1 or metadata.get('hidden_convention') != CONVENTION
            or layers != list(range(4, text['num_hidden_layers'] + 1, 4))
            or metadata['hidden_size'] != text['hidden_size']):
            raise ValueError('Unsupported or incompatible layer fusion metadata')
        checksum = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if checksum != metadata['weights_sha256']:
            raise ValueError('Layer fusion weight checksum mismatch')
        import torch
        weights = torch.load(weights_path, map_location='cpu', weights_only=True)['weights']
        if tuple(weights.shape) != (len(layers), metadata['hidden_size']) or not torch.isfinite(weights).all():
            raise ValueError('Invalid fusion weights')
        if declared and hf['thinker_layer_fusion'] != metadata:
            raise ValueError('HF config and layer fusion metadata disagree')
    enabled = present and mode != 'off'
    expected_profile = metadata.get('talker_profile', 'segmentpad') if metadata else 'continuous'
    if profile == 'auto':
        profile = expected_profile
    if profile not in ('continuous', 'segmentpad'):
        raise ValueError('Unknown Talker profile')
    if enabled and profile != expected_profile:
        raise ValueError('Talker profile differs from fusion training metadata')
    return {'enabled': enabled, 'requested': mode, 'talker_profile': profile,
            'metadata': metadata, 'weights': str(weights_path) if present else None}
