import fcntl,hashlib,json,os,time
from pathlib import Path
E=os.environ
out=Path(E['EXP_DIR']);p=out/'run_contract.json'
keys=['NNODES','NPROC_PER_NODE','GLOBAL_BATCH_SIZE','TRAIN_ITERS','LR_DECAY_ITERS','LR_WARMUP_ITERS','LR','MIN_LR','S2ST_TRAIN_STAGE','S2ST_BRIDGE_LR_SCALE','TEXT_LOSS_WEIGHT','DATASET_CONFIG_PATH','TRAIN_SCRIPT']
c={k:E[k] for k in keys}
c['initialization']='exact iter_0005000 model weights; fresh optimizer/scheduler/data traversal; v4-local iteration starts at 0'
for k in ['DATASET_CONFIG_PATH','TRAIN_SCRIPT']:
 c[k+'_sha256']=hashlib.sha256(Path(E[k]).read_bytes()).hexdigest()
c['local_train_list_sha256']=hashlib.sha256(Path(E['TRAIN_DATA_LIST']).read_bytes()).hexdigest()
c['private_training_data_included']=False
with (out/'contract.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if p.exists():
  prior=json.loads(p.read_text())
  if prior!=c:raise RuntimeError('Run configuration changed. Use a new output directory; do not mix topology, batch size, schedule or data.')
 else:
  if E['RESUME_TRAINING']=='1':raise RuntimeError('Resume requires original run contract')
  tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(c,indent=2)+'\n');tmp.replace(p)
print('RUN_CONTRACT_OK')
