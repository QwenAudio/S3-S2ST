"""Full clean-v2 -> deterministic source-disjoint dev + offline packs.
Never changes source shards. No audio decoding or re-labelling.
"""
_S3_ROOT = next((p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'env.sh').is_file() and (p / 'tools').is_dir()))
import argparse, concurrent.futures as cf, hashlib, json, os, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/path/to/data.list')
FUNASR = '__S3_ROOT__/vendor/FunASR'.replace('__S3_ROOT__', str(_S3_ROOT))
TOKENIZER = '__S3_ROOT__/models/offline'.replace('__S3_ROOT__', str(_S3_ROOT))

def worker(job):
    idx, source, dest, limit = job
    sys.path.insert(0, FUNASR)
    from offline_pack.format_converter import Converter
    conv = Converter(TOKENIZER, 'MinMo_S2S_Channels_SFT_Pack_Mega', packing_mode='s2s')
    dest = Path(dest)
    stats = dict(rows=0, train=0, dev=0, overlong=0, train_bins=0, dev_bins=0)
    paths = {s: dest / s / f'{idx:04d}.jsonl' for s in ['train', 'dev']}
    buffers = {s: [] for s in paths}
    lengths = {s: 0 for s in paths}
    writers = {s: p.with_suffix('.tmp').open('w') for s, p in paths.items()}

    def flush(s):
        if buffers[s]:
            writers[s].write(json.dumps(buffers[s], ensure_ascii=False, separators=(',', ':')) + '\n')
            stats[s + '_bins'] += 1
            buffers[s] = []
            lengths[s] = 0
    with open(source) as f, (dest / 'dev_raw' / f'{idx:04d}.jsonl').open('w') as dev:
        for line in f:
            if limit and stats['rows'] >= limit:
                break
            row = json.loads(line)
            stats['rows'] += 1
            user = next((m for m in row['messages'] if m['role'] == 'user'))
            m = re.search('<\\|startofspeech\\|>!?(.*?)<\\|endofspeech\\|>', user.get('content', user.get('text', '')))
            if not m:
                raise ValueError(f'Missing source at {source}:{stats['rows']}')
            key = m[1].replace('oss://example-storage-4/', '').replace('/nls_spider_sh_cache/', '').replace('oss://example-storage-1/', '').replace('/fun_resource_sh_cache/', '')
            split = 'dev' if int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % 10000 < 2 else 'train'
            item = conv.convert_format(row)
            n = int(item['text_length']) + int(item['speech_length']) + 16
            if n > 8191:
                stats['overlong'] += 1
                continue
            if lengths[split] + n > 8191:
                flush(split)
            buffers[split].append(item)
            lengths[split] += n
            stats[split] += 1
            if split == 'dev':
                dev.write(json.dumps(row, ensure_ascii=False) + '\n')
    for s in paths:
        flush(s)
        writers[s].close()
        writers[s] = None
        paths[s].with_suffix('.tmp').replace(paths[s])
    stats.update(index=idx, source=source, source_size=Path(source).stat().st_size)
    (dest / 'audit' / f'{idx:04d}.json').write_text(json.dumps(stats))
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    for d in ['train', 'dev', 'dev_raw', 'audit', 'mixed']:
        (out / d).mkdir()
    sources = SOURCE.read_text().splitlines()
    if a.smoke:
        sources = [sources[i] for i in [0, len(sources) // 2, -1]]
    jobs = [(i, s, str(out), 256 if a.smoke else None) for i, s in enumerate(sources)]
    stats = []
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for s in pool.map(worker, jobs):
            stats.append(s)
            print(json.dumps(s), flush=True)
    rng = random.Random(20260918)
    active = [(out / 'train' / f'{s['index']:04d}.jsonl').open() for s in stats if s['train_bins']]
    bins = [s['train_bins'] for s in stats if s['train_bins']]
    writers = [(out / 'mixed' / f'{i:03d}.jsonl').open('w') for i in range(32 if a.smoke else 256)]
    count = 0
    while active:
        i = rng.choices(range(len(active)), weights=bins, k=1)[0]
        line = active[i].readline()
        if not line:
            raise RuntimeError('Packed count mismatch')
        writers[count % len(writers)].write(line)
        count += 1
        bins[i] -= 1
        if bins[i] == 0:
            assert not active[i].readline()
            active[i].close()
            active.pop(i)
            bins.pop(i)
    for f in writers:
        f.close()
    train = [p for p in sorted((out / 'mixed').glob('*.jsonl')) if p.stat().st_size]
    valid = [out / 'dev' / f'{s['index']:04d}.jsonl' for s in stats if s['dev_bins']]
    if not valid and a.smoke:
        valid = train[:1]
    if not valid:
        raise RuntimeError('No independent dev records')
    for name, paths in [('train.list', train), ('valid.list', valid)]:
        (out / (name + '.tmp')).write_text(''.join((str(p.resolve()) + '\n' for p in paths)))
        (out / (name + '.tmp')).replace(out / name)
    totals = {k: sum((s[k] for s in stats)) for k in ['rows', 'train', 'dev', 'overlong', 'train_bins', 'dev_bins']}
    totals.update(source_list=str(SOURCE), source_list_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), smoke_only=a.smoke, epoch_steps_gbs64=(count + 63) // 64)
    (out / 'summary.json').write_text(json.dumps(totals, indent=2))
    (out / 'COMPLETE').write_text('ok\n')
    print(json.dumps(totals), flush=True)
if __name__ == '__main__':
    main()
