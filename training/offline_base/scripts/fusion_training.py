"""Latest streaming layer-fusion contract, used by both offline training stages."""
import json,os
from pathlib import Path
import torch

def install(model):
 stage=os.environ['S2ST_TRAIN_STAGE']
 if stage=='tts':
  from frozen_layer_fusion import install_thinker_layer_fusion
  fusion=install_thinker_layer_fusion(model,enabled=True,stride=4)
 else:
  from joint_layer_fusion import install_thinker_layer_fusion
  fusion=install_thinker_layer_fusion(model,enabled=True)
 assert fusion.weights.shape==(8,2560) and fusion.weights.requires_grad
 fusion.weights.lr_scale=20.0 if stage=='tts' else 1.0
 calls=[0]
 def audit(grad):
  actual=fusion.weights.main_grad if getattr(fusion.weights,'grad_added_to_main_grad',False) else grad
  if calls[0]<2:
   norms=actual.detach().float().abs().sum(-1)
   if not torch.isfinite(norms).all() or not torch.all(norms>0):raise RuntimeError('Missing/nonfinite layer-fusion gradient')
   if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:print('[OFFLINE FUSION GRAD] row_l1='+str(norms.cpu().tolist()),flush=True)
  calls[0]+=1
  return grad
 fusion.weights.register_hook(audit)
 if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
  metadata=dict(enabled=True,layers=list(fusion.layer_numbers),hidden_size=2560,parameter_key='thinker_layer_fusion.weights',hidden_convention='intermediate_post_block_final_post_norm',talker_profile='continuous',initialization='original base last-only identity at fresh initialization; exact restoration thereafter',formula='sum_i(w_i * h_i), FP32 accumulation, no softmax/extra norm',stage=stage)
  (Path(os.environ['EXP_DIR'])/'layer_fusion_config.json').write_text(json.dumps(metadata,indent=2))
  print('[OFFLINE FUSION] '+json.dumps(metadata),flush=True)
 return fusion
