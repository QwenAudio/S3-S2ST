#!/usr/bin/env python3
import argparse,os,shutil,subprocess,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description='Local-data training using frozen paper recipe')
p.add_argument('--stage',choices=['streaming-talker','streaming-e2e','offline'],required=True);p.add_argument('--train-list',type=Path,required=True);p.add_argument('--valid-list',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--init',type=Path);p.add_argument('--steps',type=int);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
stage=a.stage.split('-')[-1];out=a.output.resolve();out.relative_to(ROOT);train=a.train_list.resolve();train.relative_to(ROOT)
if not train.is_file():raise SystemExit('Missing local packed data.list; private training corpus is not included')
# Validate list -> shard -> local media dependencies; reject remote/absolute external data.
for line in train.read_text().splitlines():
 if not line.strip():continue
 shard=Path(line.strip());shard=shard if shard.is_absolute() else train.parent/shard;shard=shard.resolve();shard.relative_to(ROOT)
 if not shard.is_file():raise SystemExit('Missing packed shard: '+str(shard))
if out.exists() and any(out.iterdir()) and not (stage=='offline' and int(os.environ.get('NODE_RANK','0'))>0):raise SystemExit('Training output must be new/empty')
out.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env['PYTHON_BIN']=sys.executable
stage=a.stage.split('-')[-1];env.update(EXP_DIR=str(out),S3_ROOT=str(ROOT),TRAIN_DATA_LIST=str(train),VALID_DATA_LIST=str((a.valid_list or train).resolve()),COSYVOICE_FM_INIT_CKPT=str(ROOT/'models/training/talker_native.pth'),RESUME_TRAINING='0',TRAIN_ITERS=str(a.steps or (10000 if stage=='offline' else 5000)))
init=(a.init or ROOT/'checkpoints'/('init_offline' if stage=='offline' else 'init_streaming_'+stage)).resolve();init.relative_to(ROOT)
if not (init/'latest_checkpointed_iteration.txt').is_file():raise SystemExit('Missing local initialization checkpoint tracker: '+str(init))
env['S3_INIT_CKPT']=str(init)
if stage!='offline':
 src=ROOT/'training/streaming'/stage
 shutil.copytree(src/'runtime',out/'runtime');shutil.copytree(src/'backup',out/'backup')
 for name in ['run_train.sh']:shutil.copy2(src/name,out/name)
 # runtime scripts must refer to the run-local template, never the frozen source.
 for f in (out/'runtime').glob('*.sh'):
  s=f.read_text().replace(str(src),str(out));f.write_text(s)
 env.update(EXACT_CODEC_BASE_ENTRY=str(out/'runtime'/('pretrain_joint_layerfusion.py' if stage=='e2e' else 'pretrain_layerfusion_checked.py')))
 # Use the exact codec adapter in both phases, as in the actual continuation.
 launch=out/'runtime'/('run_base.sh' if stage=='e2e' else 'run_codec.sh');s=launch.read_text()
 import re
 s=re.sub(r'(?m)^export TRAIN_SCRIPT=.*$', 'export TRAIN_SCRIPT="'+str(out/'runtime/pretrain_exact_codec.py')+'"',s);launch.write_text(s)
 if stage=='talker':
  (out/'initialization').mkdir(exist_ok=True)
  for name in ['WEIGHT_MANIFEST.json','SOURCE.json','expected_fusion.pt']:
   if (init/name).exists():shutil.copy2(init/name,out/'initialization'/name)
 if stage=='e2e':
  # Verification reads the expected learned fusion vectors from initialization.
  (out/'initialization').mkdir()
  expected=init/'expected_fusion.pt'
  if expected.exists():shutil.copy2(expected,out/'initialization/expected_fusion.pt')
 env['STREAMING_S2ST_PREPARE_ONLY']='1' if a.prepare_only else '0'
 cmd=['bash',str(out/'run_train.sh')]
else:
 import yaml
 cfg=yaml.safe_load((ROOT/'training/offline/config/dataset_v4.yaml').read_text().replace('${S3_ROOT}', str(ROOT)));cfg['train_data_set_list']=str(train);cfg['valid_data_set_list']=str((a.valid_list or train).resolve())
 cfgpath=out/'dataset.yaml';cfgpath.write_text(yaml.safe_dump(cfg,allow_unicode=True));env['DATASET_CONFIG_PATH']=str(cfgpath)
 env['NNODES']=os.environ.get('NNODES','4');env['NODE_RANK']=os.environ.get('NODE_RANK','0');env['NPROC_PER_NODE']=os.environ.get('NPROC_PER_NODE','8')
 env['DRY_RUN']='1' if a.prepare_only else '0';cmd=['bash',str(ROOT/'training/offline/scripts/run.sh')]
if int(os.environ.get('NODE_RANK','0'))==0:
 (out/'LAUNCH.json').write_text(json.dumps(dict(command=cmd,stage=a.stage,train_list=str(train),init=str(init),steps=env['TRAIN_ITERS'],private_training_data_included=False),indent=2))
r=subprocess.run(cmd,env=env,cwd=ROOT/'code/megatron');raise SystemExit(r.returncode)
