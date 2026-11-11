#!/usr/bin/env python3
import argparse,os,sys,json,hashlib,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description='Paper inference with local model assets')
 p.add_argument('--model',type=Path);p.add_argument('--mode',choices=['streaming','offline'],default='streaming');p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
 p.add_argument('--preset',choices=['fixed05','fixed01','dynamic099'],default='dynamic099');p.add_argument('--threshold',type=float);p.add_argument('--limit',type=int);p.add_argument('--warmup',type=Path,default=ROOT/'data/fleurs/warmup.jsonl');p.add_argument('--no-warmup',action='store_true');p.add_argument('--dry-run',action='store_true')
 p.add_argument('--kv-cache-gib',type=float,help='Explicit per-engine KV memory for concurrent decoding')
 p.add_argument('--pool-worker',action='store_true',help=argparse.SUPPRESS)
 a=p.parse_args();out=a.output.resolve();out.relative_to(ROOT);out.mkdir(parents=True,exist_ok=True)
 lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'raw.jsonl').exists():raise SystemExit('Use a new output directory; raw.jsonl already exists')
 rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
 if a.pool_worker and os.environ.get('FLEURS_POOL_OWNS_GPUS')!='1':raise SystemExit('Use scripts/infer_parallel.sh for pool workers')
 if a.kv_cache_gib is not None and a.kv_cache_gib < .5:raise SystemExit('KV cache must be at least 0.5 GiB')
 if a.limit:rows=rows[:a.limit]
 if not rows:raise SystemExit('Empty input')
 for row in rows:
  import re
  user=next(m for m in row['messages'] if m['role']=='user');content=user.get('content',user.get('text',''))
  source=Path(re.search(r'<\|startofspeech\|>!?(.*?)<\|endofspeech\|>',content).group(1)).resolve()
  source.relative_to(ROOT)
  if not source.is_file():raise SystemExit('Source audio must be present inside the project: '+str(source))
 (out/'input.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 model=(a.model or ROOT/'models'/a.mode).resolve();model.relative_to(ROOT);env=os.environ.copy();env['OMNIDRCA2_MODEL_DIR']=str(model)
 # CPFS atomic cache replacement is not safe across simultaneous engines/runs.
 # Keep independent run-local caches, without using directories outside project.
 for key,sub in {'TRITON_CACHE_DIR':'triton','TORCHINDUCTOR_CACHE_DIR':'inductor','VLLM_CACHE_ROOT':'vllm','CUDA_CACHE_PATH':'cuda','TMPDIR':'tmp','TORCH_EXTENSIONS_DIR':'torch_extensions'}.items():
  path=out/'cache'/sub;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
 env['TORCHINDUCTOR_COMPILE_THREADS']='1'
 env['MAX_JOBS']='1' if a.pool_worker else '2'
 if a.pool_worker:env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 if a.kv_cache_gib is not None:env['FLEURS_KV_CACHE_BYTES']=str(int(a.kv_cache_gib*1024**3))

 if a.mode=='offline':
  cmd=[sys.executable,str(ROOT/'inference/offline/scripts/infer_offline.py'),'--model',str(model),'--input',str(out/'input.jsonl'),'--output',str(out),'--single-gpu','--encoder-device',f'cuda:{a.gpu}','--seed','0']
  for n in ['text','thinker','talker']:cmd += [f'--{n}-gpu',str(a.gpu)]
  if not a.no_warmup:cmd += ['--warmup-input',str(a.warmup.resolve())]
  env['PYTHONPATH']=str(ROOT/'inference/offline/scripts')+':'+str(ROOT/'inference/offline/vendor/streaming_eval')+':'+env.get('PYTHONPATH','')
 else:
  preset={'fixed05':(.5,.5,1.),'fixed01':(.1,.1,1.),'dynamic099':(.1,.99,.5)}[a.preset];base,cap,power=preset
  if a.threshold is not None:
   assert 0<a.threshold<1;base=cap=a.threshold;power=1.
  policy=dict(deliver_read_hidden=True,fm_chunk=8,diagnostic=True,base=base,cap=cap,saturation=1.,power=power,paced=True,fm_config=str(ROOT/'configs/fm_config.yaml'),vocoder_checkpoint=str(ROOT/'models/vocoder/model.pt' if (ROOT/'models/vocoder/model.pt').is_file() else ROOT/'models/vocoder/epoch_75_whole.pt'),vocoder_config=str(ROOT/'models/vocoder/config.yaml'))
  if not a.no_warmup:policy['warmup']=str(a.warmup.resolve())
  (out/'policy.json').write_text(json.dumps(policy,indent=2));(out/'settings.json').write_text(json.dumps(dict(code=str(ROOT/'inference/streaming'),model=str(model),gate=str(model/'streaming_gate.pt'))))
  (out/'sampling.json').write_bytes((ROOT/'configs/talker_k5.json').read_bytes());(out/'evidence').mkdir(exist_ok=True)
  files=[* (ROOT/'inference/streaming').rglob('*.py'),* (ROOT/'inference/runtime').glob('*.py')]
  (out/'evidence/source_hashes.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}))
  env.update(TUNE_ROOT=str(out),TUNE_CONFIG=str(out/'sampling.json'),SINGLE_GPU_PHYSICAL_ID=str(a.gpu),SINGLE_CAPTURE_AUDIT=str(out/'hidden_audit.jsonl'),S3_POLICY_CONFIG=str(out/'policy.json'),CODEX_TALKER_MAX_CODEC_TEXT_RATIO='0',CODEX_AUDIO_BLANK_TOKEN_COUNT='5',CODEX_ALIGNMENT_MASK_TOKEN_ID='-60')
  cmd=[sys.executable,str(ROOT/'inference/runtime'/('s3_parallel_entry.py' if a.kv_cache_gib is not None else 's3_syncgraphs_entry.py')),'--model',str(model),'--gate',str(model/'streaming_gate.pt'),'--input',str(out/'input.jsonl'),'--output',str(out/'raw.jsonl'),'--codec-dir',str(out/'codec'),'--threshold','.5','--encoder-device',f'cuda:{a.gpu}','--seed','0','--codec-temperature','.8','--codec-top-p','.8','--codec-top-k','5','--no-codec-history-replay','--codec-envelope','--codec-boundary-policy','unified_eos','--no-final-eos-only','--codec-final-bos-id','59049','--codec-blank-count','5','--codec-alignment-mask-id','-60','--codec-bos-id','59049','--codec-pause-id','59051','--codec-eos-id','59050']
  for role in ['gate','thinker','text','talker']:cmd += [f'--{role}-devices',json.dumps([a.gpu]),f'--{role}-gpu-memory-utilization',('.05' if role=='talker' else '.18') if a.pool_worker else ('.15' if role=='talker' else '.35')]
 (out/'COMMAND.json').write_text(json.dumps(dict(command=cmd,mode=a.mode,preset=a.preset,rows=len(rows),seed=0,dry_run=a.dry_run),indent=2))
 if a.dry_run:print(json.dumps(cmd));return
 used=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip())
 if used>100 and not a.pool_worker:raise SystemExit(f'GPU {a.gpu} is busy ({used} MiB); use an idle GPU')
 (out/'STATUS').write_text('running')
 with (out/'inference.log').open('w') as f:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:(out/'STATUS').write_text('failed');raise SystemExit(r.returncode)
 outputs=[json.loads(x) for x in (out/'raw.jsonl').read_text().splitlines()]
 assert len(outputs)==len(rows) and [x['index'] for x in outputs]==list(range(len(rows)))
 failures=[dict(index=x['index'],error=x['error']) for x in outputs if x.get('error')]
 (out/'COVERAGE.json').write_text(json.dumps(dict(expected=len(rows),rows=len(outputs),failures=failures),indent=2))
 (out/'STATUS').write_text('failed' if failures else 'inference_complete')
 if failures:raise SystemExit('Inference failures: inspect COVERAGE.json')
 print(out)
if __name__=='__main__':main()
