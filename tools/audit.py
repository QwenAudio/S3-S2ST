#!/usr/bin/env python3
import argparse,json,hashlib,ast,subprocess,sys,re,warnings
warnings.filterwarnings("ignore",category=SyntaxWarning)
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--verify-hashes',action='store_true');a=p.parse_args();errors=[];pending_audio=[];pending_assets=[]
required=['models/streaming/config.json','models/offline/config.json','models/streaming/streaming_gate.pt','models/speaker/campplus.onnx','models/flow_matching/epoch_2_step_780000.pt','models/vocoder/epoch_75_whole.pt','models/frontend/mel_filters.npz','models/xcomet/checkpoints/model.ckpt','models/xlm-roberta-xl/config.json','checkpoints/streaming/iter_0004000','checkpoints/offline/iter_0010000','vendor/vllm/engine/omnidrca2_engine.py','vendor/vllm/engine/shared_hidden_buffer.py']
required += ['code/megatron/megatron/core/models/gpt/gpt_model.py', 'vendor/FunASR/funasr/models/cosyvoice/flow_matching_model.py', 'vendor/FunASR/funasr/models/cosyvoice/causal_bigvgan.py', 'evaluation/voice_eval/thirdparty/3D-Speaker/speakerlab/models/eres2net/ERes2Net.py']
for n in required:
 if not (ROOT/n).exists() or ((ROOT/n).is_file() and (ROOT/n).stat().st_size==0):
  if n=='models/xcomet/checkpoints/model.ckpt':pending_assets.append(n)
  else:errors.append('missing '+n)
for d in ['inference','tools','training','evaluation']:
 for f in (ROOT/d).rglob('*.py'):
  try:ast.parse(f.read_text())
  except (SyntaxError,UnicodeError) as e:errors.append(str(f.relative_to(ROOT))+': '+str(e))
for f in ROOT.rglob('*'):
 if f.relative_to(ROOT).parts[0] in {'envs','cache','.git'}:continue
 if f.is_symlink():
  try:f.resolve().relative_to(ROOT)
  except ValueError:errors.append('external symlink '+str(f.relative_to(ROOT)))
for f in (ROOT/'scripts').glob('*.sh'):
 r=subprocess.run(['bash','-n',str(f)],capture_output=True,text=True)
 if r.returncode:errors.append(r.stderr)
for direction,expected in [('zh2en',945),('en2zh',646)]:
 f=ROOT/'data/fleurs'/f'test_{direction}.jsonl'
 rows=[json.loads(x) for x in f.read_text().splitlines()]
 if len(rows)!=expected:errors.append('FLEURS wrong count '+direction)
 for row in rows:
  c=next(x for x in row['messages'] if x['role']=='user')['content'];q=Path(re.search(r'<\|startofspeech\|>!?(.*?)<\|endofspeech\|>',c).group(1))
  try:q.resolve().relative_to(ROOT)
  except ValueError:errors.append('external audio '+str(q));continue
  if not q.is_file():pending_audio.append(str(q.resolve().relative_to(ROOT)))
checked=0
if a.verify_hashes:
 for line in (ROOT/'provenance/SHA256SUMS').read_text().splitlines():
  expected,name=line.split('  ',1);f=ROOT/name;h=hashlib.sha256()
  with f.open('rb') as inp:
   for block in iter(lambda:inp.read(8*1024*1024),b''):h.update(block)
  if h.hexdigest()!=expected:errors.append('hash mismatch '+name)
  checked+=1
report=dict(status='PASS' if not errors else 'FAIL',pending_public_audio_downloads=len(pending_audio),pending_public_assets=pending_assets,public_benchmark_ready=not pending_audio and not pending_assets,errors=errors,hashes_checked=checked,gpu_inference_tested=(ROOT/'provenance/runtime_validation.json').exists(),private_training_data_included=False)
(ROOT/'provenance/AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));raise SystemExit(bool(errors))
