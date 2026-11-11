"""Fresh-base Talker -> E2E training, resumable at completed checkpoints/evals."""
import fcntl,json,os,subprocess,traceback
from pathlib import Path
from select_checkpoint import summarize,decide
ROOT=Path(__file__).resolve().parents[1]
PY='/usr/local/bin/python'
def run(args,log,env=None):
 e=os.environ.copy();e.update(env or {})
 with log.open('a') as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,env=e,check=True)
def status(s):
 (ROOT/'PIPELINE_STATUS').write_text(s+'\n');print(s,flush=True)
def evaluate(hf,path):
 for lang in ['en','zh']:
  dest=path/lang
  if (dest/'COMPLETE').exists():continue
  run([PY,ROOT/'scripts/evaluate.py','--model',hf,'--input',ROOT/'data'/f'dev_{lang}.jsonl','--output',dest,'--language',lang],ROOT/'audit'/f'{path.name}_{lang}.log')
 return summarize(path)
def main():
 lock=(ROOT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (ROOT/'PIPELINE_FAILED').unlink(missing_ok=True)
 assert (ROOT/'audit/CODE_VALIDATED.json').exists()
 assert (ROOT/'data/full_v2/COMPLETE').exists()
 origin=json.loads((ROOT/'audit/origin.json').read_text())
 source=Path(origin['reference_checkpoint']);baseline=None;talker_hf=None
 for stage,cap in [('talker',10000),('e2e',5000)]:
  exp=ROOT/'runs'/stage;exp.mkdir(exist_ok=True)
  records=[] if stage=='talker' else [dict(step=0,checkpoint=str(source),hf=str(talker_hf),metrics=baseline)]
  for step in range(5000 if stage=='talker' else 500,cap+1,500):
   tracker=exp/'latest_checkpointed_iteration.txt'
   current=int(tracker.read_text()) if tracker.exists() else 0
   if current<step:
    status(f'{stage}: training {current} -> {step} (cap {cap})')
    run(['bash',ROOT/'scripts/train.sh',stage,source],ROOT/'audit'/f'{stage}_pipeline.log',dict(TRAIN_ITERS=str(cap),EXIT_INTERVAL=str(step),RESUME_TRAINING='1' if current else '0',LR_DECAY_ITERS=str(cap),SAVE_INTERVAL='500'))
    assert int(tracker.read_text())==step, 'Unexpected training stop iteration'
   ckpt=exp/f'iter_{step:07d}';hf=Path(str(ckpt)+'_HF')
   if not hf.exists():
    status(f'{stage}: converting {step}')
    run(['bash',ROOT/'config/convert_tp1_s2st_checkpoint_to_hf.sh',ckpt,hf],ROOT/'audit'/f'{stage}_{step}_convert.log')
   status(f'{stage}: evaluating {step}')
   try:
    scores=evaluate(hf,ROOT/'eval'/f'{stage}_{step:07d}')
   except Exception:
    failure=dict(stage=stage,step=step,error=traceback.format_exc())
    with (ROOT/'audit/evaluation_failures.jsonl').open('a') as f:f.write(json.dumps(failure)+'\n')
    status(f'{stage}: evaluation {step} failed; checkpoint retained; training will continue')
    continue
   records.append(dict(step=step,checkpoint=str(ckpt),hf=str(hf),metrics=scores))
   (exp/'selection_history.json').write_text(json.dumps(records,indent=2))
   decision=decide(records,stage);(exp/'selection.json').write_text(json.dumps(decision,indent=2))
   if decision['stop']:break
  if not records or (stage=='e2e' and len(records)==1):raise RuntimeError(f'{stage}: training budget completed but no valid evaluation; inspect audit/evaluation_failures.jsonl')
  best=decide(records,stage)['best'];source=Path(best['checkpoint']);selected_hf=Path(best['hf']);baseline=best['metrics']
  (exp/'SELECTED_CHECKPOINT').write_text(str(source)+'\n')
  if stage=='talker':talker_hf=selected_hf
 tests=Path(origin['streaming_eval'])/'datasets';comparison={}
 # Only checkpoints from this clean-initialized run participate.
 for label,model in [('talker',talker_hf),('e2e',selected_hf)]:
  comparison[label]={}
  for ds in ['9pair','fleurs']:
   for direction,lang in [('zh2en','en'),('en2zh','zh')]:
    dest=ROOT/'eval/final'/label/ds/direction
    if label=='e2e' and model==talker_hf:
     dest.parent.mkdir(parents=True,exist_ok=True)
     if not dest.exists():dest.symlink_to(ROOT/'eval/final/talker'/ds/direction,target_is_directory=True)
    elif not (dest/'COMPLETE').exists():
     status(f'final: {label} {ds} {direction}')
     run([PY,ROOT/'scripts/evaluate.py','--model',model,'--input',tests/ds/f'test_{ds}_{direction}.jsonl','--output',dest,'--language',lang],ROOT/'audit'/f'final_{label}_{ds}_{direction}.log')
    comparison[label][ds+'/'+direction]=json.loads((dest/'summary.json').read_text())
 (ROOT/'eval/final/comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
 (ROOT/'SELECTED_FINAL').write_text(str(source)+'\n');status('complete')
if __name__=='__main__':
 try:main()
 except BaseException:
  (ROOT/'PIPELINE_FAILED').write_text(traceback.format_exc());raise
