#!/usr/bin/env python3
import argparse,os,sys,json,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset-config',type=Path,required=True);p.add_argument('--mode',choices=['streaming','offline'],default='streaming');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
ckpt=a.checkpoint.resolve();out=a.output.resolve();config=a.dataset_config.resolve()
for q in [ckpt,out,config]:q.relative_to(root)
if out.exists():raise SystemExit('Use a new HF output directory')
if not ckpt.is_dir() or not config.is_file():raise SystemExit('Missing local checkpoint/config')
repo=root/'code/megatron';ref=root/'models'/a.mode;env=os.environ.copy()
env.update(CUDA_DEVICE_MAX_CONNECTIONS='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',PYTHONDONTWRITEBYTECODE='1',Channels_Model='True',no_audio_emb_and_head='True',LOADER_RELEASE_LAYER_AFTER_SEND='1',SAVER_STREAMING_MODE='1',COSYVOICE_CODEX_VOCAB_PADDING_MULTIPLE='1',FUN_DATASET_CONFIG_PATH=str(config))
env['PYTHONPATH']=str(repo)+':'+str(root/'vendor')+':'+str(root/'vendor/FunASR')+':'+env.get('PYTHONPATH','')
cmd=[sys.executable,'-m','torch.distributed.run','--nproc-per-node=1','--nnodes=1','--master-port',os.environ.get('CONVERT_MASTER_PORT','29641'),'checkpoint/convert.py','--model-type','GPT','--loader','fun_mcore_qwen3_5_plus','--saver','fun_qwen3_5_hf_plus','--max-queue-size','1','--load-dir',str(ckpt),'--save-dir',str(out),'--shard','5GB','--audio-encoder-ckpt-path',str(ref/'audio_encoder.pt'),'--megatron-path',str(repo),'--fun-dataset-config-path',str(config)]
if a.dry_run:print(json.dumps(cmd));raise SystemExit(0)
subprocess.run(cmd,cwd=repo/'tools',env=env,check=True)
for name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','preprocessor_config.json','processor_config.json']:
 if (ref/name).is_file():shutil.copy2(ref/name,out/name)
if a.mode=='streaming':subprocess.run([sys.executable,str(root/'tools/export_gate.py'),str(ckpt),str(out/'streaming_gate.pt')],cwd=repo,env=env,check=True)
subprocess.run([sys.executable,str(root/'inference/streaming/profiles/layerfusion/export_layer_fusion.py'),str(ckpt),str(out)],cwd=repo,env=env,check=True)
(out/'EXPORT_PROVENANCE.json').write_text(json.dumps(dict(checkpoint=str(ckpt),mode=a.mode,command=cmd),indent=2))
print(out)
