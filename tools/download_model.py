#!/usr/bin/env python3
"""Download and install an S3-S2ST model release from Hugging Face or local files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def install(source, mode, root=ROOT, verify_only=False):
    source = Path(source).resolve()
    manifest = json.loads((source / 'release.json').read_text())
    if manifest['format_version'] != 1 or manifest['mode'] != mode:
        raise ValueError('The release format or model mode does not match')
    records = manifest['files']
    paths = {r['path'] for r in records}
    required = {'config.json', 'talker_config/config.json', 'thinker_layer_fusion.pt',
                'assets/speaker/campplus.onnx', 'assets/flow_matching/model.pt',
                'assets/vocoder/model.pt', 'assets/vocoder/config.yaml',
                'assets/frontend/mel_filters.npz', 'assets/frontend/fm_mel_filters.npz'}
    if mode == 'streaming':
        required.add('streaming_gate.pt')
    if not required <= paths or len(paths) != len(records):
        raise ValueError('Incomplete release manifest or duplicate file names')
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    if not set(index['weight_map'].values()) <= paths:
        raise ValueError('Missing model weight shards')
    destinations = []
    for record in records:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid release file path')
        path = (source / relative).resolve()
        path.relative_to(source)
        if not path.is_file() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'Checksum mismatch: {relative}')
        if relative.parts[0] == 'assets':
            target = root / 'models' / Path(*relative.parts[1:])
            if relative.parts[1] not in {'speaker', 'flow_matching', 'vocoder', 'frontend'}:
                raise ValueError('Unknown shared asset directory')
        else:
            target = root / 'models' / mode / relative
        target.resolve().relative_to((root / "models").resolve())
        destinations.append((path, target, record))
    print(f'Verified {mode}: {len(records)} files', flush=True)
    if verify_only:
        return
    # Validate existing files before copying so installing the second model can reuse shared assets.
    pending = []
    for source_file, target, record in destinations:
        if target.exists():
            if target.is_symlink() or not target.is_file() or digest(target) != record['sha256']:
                raise FileExistsError(f'Different file already exists: {target}')
        else:
            pending.append((source_file, target))
    for source_file, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.resolve().relative_to(root.resolve())
        temporary = target.with_name(target.name + '.part')
        shutil.copyfile(source_file, temporary)
        temporary.replace(target)
    for name in ['README.md', 'release.json', 'SHA256SUMS']:
        target = root / 'models' / mode / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    print(f'Installed model: {root / "models" / mode}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['streaming', 'offline'], required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--repo-id', help='Hugging Face model repository, e.g. organization/model')
    group.add_argument('--local-dir', type=Path, help='Already downloaded model repository')
    parser.add_argument('--revision', default='main', help='HF branch or commit; pin a commit for reproducibility')
    parser.add_argument('--subfolder', choices=['streaming', 'offline'],
                        help='Model subdirectory when both variants share one Hub repository')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.subfolder and args.subfolder != args.mode:
        parser.error('--subfolder must match --mode')
    source = args.local_dir
    if args.repo_id:
        os.environ['HF_HUB_OFFLINE'] = '0'
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(
            repo_id=args.repo_id, revision=args.revision,
            allow_patterns=[args.subfolder + '/**'] if args.subfolder else None,
            local_dir=ROOT / 'cache' / 'model_downloads' / args.mode))
    if args.subfolder and (args.repo_id or not (source / 'release.json').is_file()):
        source = source / args.subfolder
    elif not (source / 'release.json').is_file() and (source / args.mode / 'release.json').is_file():
        source = source / args.mode
    install(source, args.mode, verify_only=args.verify_only)


if __name__ == '__main__':
    main()
