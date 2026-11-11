#!/usr/bin/env python3
"""Rebuild paper FLEURS manifests from pinned official metadata; no corpus in Git."""
import argparse, hashlib, json, subprocess, sys, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def fetch(url, path, digest):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print('Downloading', url, flush=True)
        temporary = path.with_suffix(path.suffix + '.part')
        try:
            with urllib.request.urlopen(url, timeout=60) as src, temporary.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src, dst)
            temporary.rename(path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f'Download failed: {url}. Download on a connected machine and copy to {path}.') from exc
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != digest:
        raise ValueError(f'SHA256 mismatch: {path}')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata-dir', type=Path)
    ap.add_argument('--archive-dir', type=Path)
    ap.add_argument('--manifests-only', action='store_true')
    a = ap.parse_args()
    cfg = json.loads((ROOT / 'configs/fleurs_subset.json').read_text())
    cache = a.archive_dir or ROOT / 'cache/downloads/fleurs_hf'
    metadata = a.metadata_dir or cache
    base = f"https://huggingface.co/datasets/google/fleurs/resolve/{cfg['revision']}/data"
    tables = {}
    for lang, spec in cfg['metadata'].items():
        path = metadata / spec['filename']
        fetch(f'{base}/{lang}/test.tsv', path, spec['sha256'])
        # TSV quotes belong to the transcript; CSV quote processing would change references.
        rows = [line.split('\t') for line in path.read_text().splitlines()]
        tables[lang] = {r[1]: r for r in rows}
    out = ROOT / 'data/fleurs'
    out.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for direction, src, tgt in [('zh2en','cmn_hans_cn','en_us'), ('en2zh','en_us','cmn_hans_cn')]:
        rows = []
        for item in cfg['selection'][direction]:
            source = tables[src][item['source']]
            target = tables[tgt][item['target']]
            assert source[0] == target[0]
            wave = out / 'audio' / src / item['source']
            rows.append(dict(messages=[dict(role='system',content='You are a helpful assistant.'),dict(role='user',content='Translate into '+('English' if direction=='zh2en' else 'Chinese')+':<|startofspeech|>!'+str(wave)+'<|endofspeech|>'),dict(role='assistant',content=target[2])],transcript=source[2],translation=target[2],dataset='fleurs',direction=direction,duration_s=int(source[5])/16000))
        manifests[direction] = rows
        write(out / f'test_{direction}.jsonl', rows)
    write(out / 'test_all.jsonl', manifests['zh2en'] + manifests['en2zh'])
    write(out / 'warmup.jsonl', [manifests['zh2en'][0], manifests['en2zh'][0]])
    write(out / 'smoke8.jsonl', manifests['zh2en'][:4] + manifests['en2zh'][:4])
    print('Manifests prepared: zh2en=945, en2zh=646', flush=True)
    if not a.manifests_only:
        for spec in json.loads((ROOT / 'configs/fleurs_archive_verified.json').read_text()):
            fetch(f"{base}/{spec['language']}/audio/test.tar.gz", cache / spec['filename'], spec['sha256'])
        subprocess.run([sys.executable,str(ROOT/'tools/prepare_fleurs_archives.py'),'--archive-dir',str(cache)], check=True)

def write(path, rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

if __name__ == '__main__':
    main()
