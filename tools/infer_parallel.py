#!/usr/bin/env python3
"""Paper S2ST: offline gain -> 8 GPUs x N independent workers -> full metrics.
Run with python3; boost_fleurs.py must sit alongside this script. Requires numpy/soundfile.
Defaults preserve fixed05/FM8/source pacing, eight-layer fusion and paper sampling.
Concurrent wall-clock latency is NOT isolated single-request paper latency.
"""
import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
PAPER = ROOT
MODEL = ROOT / 'models/streaming'


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))


def audio_path(row):
    msg = next(x for x in row['messages'] if x['role'] == 'user')
    return re.search(r'<\|startofspeech\|>!?(.*?)<\|endofspeech\|>', msg['content']).group(1)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Paper launcher changed; refusing ambiguous patch: ' + old[:100])
    return text.replace(old, new)


def prepare_launcher(root, args):
    files = [ROOT/'tools/infer.py', ROOT/'inference/runtime/s3_parallel_entry.py', Path(__file__), ROOT/'tools/boost_fleurs.py']
    dump(root/'provenance/launcher.json', dict(files={str(p.resolve().relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, kv_gib=args.kv_gib))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True, help='New output directory inside this repository')
    ap.add_argument('--input', type=Path, nargs='+', help='Paper-format JSONL; default: the exact 1591 paper FLEURS examples')
    ap.add_argument('--model-hf', type=Path, default=MODEL)
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--workers-per-gpu', type=int, default=2)
    ap.add_argument('--kv-gib', type=float, default=1.0, help='Explicit KV cache per engine')
    ap.add_argument('--target-rms', type=float, default=.0743324953)
    ap.add_argument('--peak-limit', type=float, default=.95)
    ap.add_argument('--limit', type=int, help='Small verification run only; full run leaves this unset')
    ap.add_argument('--skip-metrics', action='store_true')
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    assert len(set(gpus)) == len(gpus) and gpus
    assert 1 <= args.workers_per_gpu <= 4 and args.kv_gib >= .5
    assert 0 < args.target_rms <= 1 and 0 < args.peak_limit <= 1
    root = args.output.resolve(); root.relative_to(BASE.resolve())
    root.mkdir(parents=True, exist_ok=False)
    for sub in ['scripts','inputs','audio','runs','logs','provenance']:
        (root / sub).mkdir()
    shutil.copy2(__file__, root / 'scripts/driver.py')
    shutil.copy2(Path(__file__).with_name('boost_fleurs.py'), root / 'scripts/boost_fleurs.py')
    import soundfile as sf
    from boost_fleurs import boost
    sources = args.input or [ROOT/'data/fleurs/test_all.jsonl']
    rows = [r for path in sources for r in read(path)]
    if args.limit:
        # Interleave directions so small concurrency checks cover both languages.
        groups = [[r for r in rows if r['direction']==d] for d in ['en2zh','zh2en']]
        rows = [r for pair in __import__('itertools').zip_longest(*groups) for r in pair if r is not None][:args.limit]
    assert rows and len({audio_path(r) for r in rows}) == len(rows)
    gain_records = []
    for i, row in enumerate(rows):
        original = audio_path(row); x, sr = sf.read(original, dtype='float64', always_2d=True)
        y, stats = boost(x, args.target_rms, args.peak_limit)
        out = root / 'audio' / row['direction'] / f'{i:06d}_{Path(original).stem}.wav'
        out.parent.mkdir(exist_ok=True)
        sf.write(out, y, sr, format='WAV', subtype='FLOAT')
        user = next(m for m in row['messages'] if m['role']=='user')
        user['content'] = user['content'].replace(original, str(out))
        row['_gain_original_audio'] = original; row['_parallel_index'] = i
        gain_records.append(dict(input=original, output=str(out), **stats))
    jsonl(root/'inputs/all.jsonl', rows); jsonl(root/'provenance/gain.jsonl', gain_records)
    prepare_launcher(root, args)
    workers = min(len(rows), len(gpus)*args.workers_per_gpu)
    # Longest-first duration balancing reduces the long tail; one persistent model per shard.
    shards = [[] for _ in range(workers)]; loads = [0.]*workers
    for row in sorted(rows, key=lambda r:float(r.get('duration_s',1)), reverse=True):
        slot = min(range(workers), key=lambda n:loads[n])
        shards[slot].append(row); loads[slot] += float(row.get('duration_s',1))
    tag = 'chainopt_fg_' + time.strftime('%m%d_%H%M%S') + '_' + str(os.getpid())
    jobs=[]
    for slot, shard in enumerate(shards):
        shard.sort(key=lambda r:r['_parallel_index'])
        inp=root/'inputs'/f'worker{slot:02d}.jsonl';jsonl(inp,shard)
        jobs.append(dict(slot=slot,gpu=gpus[slot%len(gpus)],name=f'{tag}_w{slot:02d}',input=str(inp),rows=len(shard),audio_seconds=loads[slot]))
    dump(root/'jobs.json',jobs)
    dump(root/'config.json',dict(model=str(args.model_hf),gpus=gpus,workers_per_gpu=args.workers_per_gpu,samples=len(rows),threshold=.5,fm_chunk=8,paced=True,latency_scope='concurrent throughput run; not isolated paper latency',target_rms=args.target_rms,peak_limit=args.peak_limit))
    if args.prepare_only:
        dump(root/'status.json',dict(stage='prepared'));return
    (PAPER / 'provenance').mkdir(parents=True, exist_ok=True)
    leases=[];children={};guard=threading.Lock();aborted=threading.Event()
    def state(stage, **extra):dump(root/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**extra))
    def shutdown():
        aborted.set()
        with guard:
            for proc in children.values():
                if proc.poll() is None:
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
    def stop(signum, frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    try:
        for gpu in gpus:
            f=(PAPER/'provenance'/f'{socket.gethostname()}_gpu{gpu}.lock').open('a')
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(f)
            used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits']))
            assert used<100, f'GPU {gpu} busy ({used} MiB); do not overlap unrelated runs'
        env=dict(os.environ,FLEURS_POOL_OWNS_GPUS='1',FLEURS_KV_CACHE_BYTES=str(int(args.kv_gib*1024**3)),S3_USE_LOCAL_CACHE='0')
        state('inference',workers=workers,samples=len(rows))
        def run(j):
            out=PAPER/'runs'/j['name'];(root/'runs'/j['name']).symlink_to(out)
            cmd=['bash',str(ROOT/'scripts/infer.sh'),'--mode','streaming','--preset','fixed05','--model',str(args.model_hf),'--input',j['input'],'--output',str(out),'--gpu',str(j['gpu']),'--pool-worker','--kv-cache-gib',str(args.kv_gib)]
            # Stagger workers sharing one GPU to reduce simultaneous initialization spikes.
            time.sleep(20*(j['slot']//len(gpus)))
            with (root/'logs'/f"worker{j['slot']:02d}.log").open('w') as log:
                with guard:
                    if aborted.is_set():raise RuntimeError('Pool aborted')
                    worker_env=dict(env,VLLM_PORT=str(20000+j['slot']*200),MASTER_PORT=str(20100+j['slot']*200))
                    proc=subprocess.Popen(cmd,env=worker_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    children[j['slot']]=proc
                if proc.wait():raise RuntimeError(f"worker {j['slot']} failed; see {out}/inference.log")
            result=read(out/'raw.jsonl')
            assert len(result)==j['rows'] and [r['index'] for r in result]==list(range(j['rows']))
            assert not any(r.get('error') for r in result)
            assert '[INFERENCE LAYER FUSION] layers=[4, 8, 12, 16, 20, 24, 28, 32]' in (out/'inference.log').read_text()
            for r in result:
                pcm=r['live_pcm'];assert pcm['samples']>0 and not pcm.get('truncated',False)
                assert pcm['speaker_protocol']=='all_consumed_prefix_each_nonempty_write_no_crop'
                assert all(abs(g['threshold']-.5)<1e-8 for g in pcm['gates'])
            print('WORKER_DONE',j['slot'],len(result),flush=True)
            return out
        pool=cf.ThreadPoolExecutor(workers)
        futures={pool.submit(run,j):j for j in jobs}
        try:
            while futures:
                done,_=cf.wait(futures,timeout=30,return_when=cf.FIRST_COMPLETED)
                for f in done:f.result();del futures[f]
                progress={}
                for j in jobs:
                    path=PAPER/'runs'/j['name']/'raw.jsonl'
                    progress[str(j['slot'])]=path.read_bytes().count(b'\n') if path.exists() else 0
                state('inference',workers=workers,completed=sum(progress.values()),samples=len(rows),pending_workers=len(futures),per_worker=progress)
        except BaseException:
            shutdown()
            for f in futures:f.cancel()
            raise
        finally:pool.shutdown(wait=True,cancel_futures=True)
        outputs={}
        for j in jobs:
            for inp,out in zip(read(j['input']),read(PAPER/'runs'/j['name']/'raw.jsonl')):
                idx=inp['_parallel_index'];assert idx not in outputs
                assert out['source']==inp['transcript'] and out['reference']==inp['translation']
                out['worker_index']=out['index'];out['index']=idx;outputs[idx]=out
        assert set(outputs)==set(range(len(rows)))
        jsonl(root/'raw.jsonl',[outputs[i] for i in range(len(rows))])
        dump(root/'COVERAGE.json',dict(samples=len(rows),missing=0,duplicates=0,errors=0))
        for lease in leases:lease.close()
        leases=[]
        if not args.skip_metrics:
            state('metrics',samples=len(rows))
            def score(d):
                selected=[i for i,r in enumerate(rows) if r['direction']==d]
                if not selected:return None
                name=tag+'_'+d;out=PAPER/'runs'/name;out.mkdir();(root/'runs'/d).symlink_to(out)
                rs=[dict(outputs[i],index=n) for n,i in enumerate(selected)]
                jsonl(out/'input.jsonl',[rows[i] for i in selected]);jsonl(out/'raw.jsonl',rs)
                (out/'STATUS').write_text('inference_complete')
                gpu=gpus[0 if d=='en2zh' or len(gpus)==1 else 1]
                # Other jobs may acquire the GPU between decoding and scoring.
                while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'])) < 49152:
                    time.sleep(30)
                with (root/'logs'/f'{d}_score.log').open('w') as log:subprocess.run(['bash',str(ROOT/'scripts/evaluate.sh'),'--run',name,'--gpu',str(gpu),'--full'],stdout=log,stderr=subprocess.STDOUT,check=True)
                return read_summary(out)
            def read_summary(out):
                result=json.loads((out/'quality/summary.json').read_text())
                for row in result:
                    for metric in ['text_bleu','comet','asr_bleu','asr_comet','wer_cer','spk_sim','utmos']:
                        assert isinstance(row[metric],(int,float)), (metric,row)
                return result
            with cf.ThreadPoolExecutor(min(2,len(gpus))) as pool:
                scores=list(pool.map(score,['en2zh','zh2en']))
            dump(root/'summary.json',{d:s for d,s in zip(['en2zh','zh2en'],scores) if s is not None})
        state('complete',samples=len(rows),metrics=not args.skip_metrics)
    except BaseException:
        shutdown();state('failed',error=traceback.format_exc());raise
    finally:
        for lease in leases:lease.close()


if __name__=='__main__':main()
