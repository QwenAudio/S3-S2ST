"""Export fusion tensors without constructing/loading the full training model."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from layer_fusion_assets import CONVENTION

def export(checkpoint, output, config=None):
    checkpoint, output = Path(checkpoint), Path(output)
    reader = dcp.FileSystemReader(str(checkpoint))
    tensors = reader.read_metadata().state_dict_metadata
    key = 'thinker_layer_fusion.weights'
    if key not in tensors:
        declaration = Path(config) if config else checkpoint.parent / 'layer_fusion_config.json'
        if declaration.exists() and json.loads(declaration.read_text()).get('enabled'):
            raise ValueError('Training declares fusion but checkpoint tensor is missing')
        print('LAYER_FUSION_ABSENT', checkpoint)
        return False
    source = Path(config) if config else checkpoint.parent / 'layer_fusion_config.json'
    training = json.loads(source.read_text())
    layers, hidden = training['layers'], training['hidden_size']
    expected = 'intermediate post-block outputs; final layer uses existing post-final-norm output'
    if training.get('hidden_convention') not in (expected, CONVENTION):
        raise ValueError('Unsupported training hidden convention')
    if not training['enabled'] or layers != list(range(4, layers[-1] + 1, 4)):
        raise ValueError('Unexpected training layer selection')
    meta = tensors[key]
    if tuple(meta.size) != (len(layers), hidden):
        raise ValueError('Fusion tensor shape disagrees with training metadata')
    state = {key: torch.empty(tuple(meta.size), dtype=meta.properties.dtype)}
    dcp.load(state, checkpoint_id=str(checkpoint))
    weights = state[key].cpu()
    if not torch.isfinite(weights).all():
        raise ValueError('Nonfinite layer fusion weights')
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'thinker_layer_fusion.pt'
    torch.save({'weights': weights}, target)
    metadata = {'version': 1, 'layers': layers, 'hidden_size': hidden,
                'hidden_convention': CONVENTION, 'accumulation_dtype': 'float32',
                'talker_profile': 'segmentpad', 'parameter_key': key,
                'weights_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'source_checkpoint': str(checkpoint.resolve())}
    (output / 'thinker_layer_fusion.json').write_text(json.dumps(metadata, indent=2))
    hf_path = output / 'config.json'
    hf = json.loads(hf_path.read_text())
    text = hf.get('text_config', hf)
    if text['num_hidden_layers'] != layers[-1] or text['hidden_size'] != hidden:
        raise ValueError('Converted HF dimensions disagree with fusion checkpoint')
    hf['thinker_layer_fusion'] = metadata
    hf_path.write_text(json.dumps(hf, indent=2))
    print('LAYER_FUSION_EXPORTED', list(weights.shape), target)
    return True

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint'); ap.add_argument('output')
    ap.add_argument('--training-config')
    args = ap.parse_args()
    export(args.checkpoint, args.output, args.training_config)
