"""Immutable legacy Zarr initialization view and exact fusion sidecar export."""
import argparse,hashlib,json,struct
from pathlib import Path
KEY='thinker_layer_fusion.weights'
ROOT=Path(__file__).resolve().parents[1]
def init(source):
 source=Path(source).resolve()
 if (source/KEY/'.zarray').exists():return source
 raise RuntimeError('Fresh base uses train.sh native initialization; do not create a legacy S2ST fusion view')

def export(source,out):
 import torch
 source=Path(source);out=Path(out);d=source/KEY
 if not d.exists():
  if (source.parent/'layer_fusion_config.json').exists():raise RuntimeError('Fusion declared but missing from checkpoint')
  print('Legacy baseline: no fusion tensor');return
 m=json.loads((d/'.zarray').read_text())
 assert m['shape']==m['chunks']==[8,2560] and m['compressor'] is None and m['order']=='C'
 dtype={'bfloat16':torch.bfloat16,'<f4':torch.float32,'<f2':torch.float16}[m['dtype']]
 w=torch.frombuffer(bytearray((d/'0.0').read_bytes()),dtype=dtype).reshape(8,2560).clone()
 assert torch.isfinite(w).all()
 cfg=json.loads((source.parent/'layer_fusion_config.json').read_text())
 assert cfg['layers']==list(range(4,33,4)) and cfg['hidden_size']==2560 and cfg['talker_profile']=='continuous'
 torch.save({'weights':w},out/'thinker_layer_fusion.pt')
 metadata=dict(version=1,layers=cfg['layers'],hidden_size=2560,hidden_convention='intermediate_post_block_final_post_norm',accumulation_dtype='float32',talker_profile='continuous',parameter_key=KEY,weights_sha256=hashlib.sha256((out/'thinker_layer_fusion.pt').read_bytes()).hexdigest(),source_checkpoint=str(source.resolve()))
 (out/'thinker_layer_fusion.json').write_text(json.dumps(metadata,indent=2))
 hf=json.loads((out/'config.json').read_text());text=hf.get('text_config',hf)
 assert text['hidden_size']==2560 and text['num_hidden_layers']==32
 hf['thinker_layer_fusion']=metadata;(out/'config.json').write_text(json.dumps(hf,indent=2))
 print('FUSION_EXPORTED',metadata['weights_sha256'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['init','export']);p.add_argument('source');p.add_argument('out',nargs='?');a=p.parse_args()
 if a.action=='init':print(init(a.source))
 else:export(a.source,a.out)
