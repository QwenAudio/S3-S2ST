"""Simulate evaluation failure without launching training or GPU tools."""
import tempfile,json
from pathlib import Path
import pipeline as p
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);p.ROOT=root
 for d in ['audit','data/full_v2','runs/talker']:(root/d).mkdir(parents=True)
 (root/'audit/CODE_VALIDATED.json').write_text('{}');(root/'data/full_v2/COMPLETE').touch()
 (root/'audit/origin.json').write_text(json.dumps(dict(reference_checkpoint='/original/iter_0001600',streaming_eval='/unused')))
 (root/'runs/talker/latest_checkpointed_iteration.txt').write_text('500')
 calls=[]
 def fake_run(args,log,env=None):
  args=list(map(str,args))
  if args[1].endswith('/train.sh'):
   stage=args[2];step=int(env['EXIT_INTERVAL']);calls.append((stage,step,env['RESUME_TRAINING']))
   d=root/'runs'/stage;d.mkdir(exist_ok=True);(d/'latest_checkpointed_iteration.txt').write_text(str(step));(d/f'iter_{step:07d}').mkdir()
  elif args[1].endswith('/convert_tp1_s2st_checkpoint_to_hf.sh'):Path(args[3]).mkdir()
  else:
   out=Path(args[args.index('--output')+1]);out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text('{}');(out/'COMPLETE').touch()
 def fake_evaluate(hf,path):
  if path.name=='talker_0005000':raise RuntimeError('simulated evaluator failure')
  return dict(wer=10.,mos=4.,asr_comet=.8,text_comet=.85,failure_rate=0.)
 p.run=fake_run;p.evaluate=fake_evaluate;p.status=lambda x:None
 p.main()
 assert calls[0]==('talker',5000,'1'),calls
 assert ('talker',5500,'1') in calls,calls
 assert (root/'audit/evaluation_failures.jsonl').exists()
 assert (root/'SELECTED_FINAL').exists()
 print('PASS: resume500 directly to5000; evaluation failure continues to5500; E2E/final selection remain reachable')
