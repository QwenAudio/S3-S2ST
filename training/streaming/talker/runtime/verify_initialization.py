import os,json,hashlib
from pathlib import Path
import torch

def install_verification(model):
    if os.environ.get('RESUME_TRAINING')=='1' or os.environ['TRAIN_STAGE']!='talker':return
    done=[False]
    def verify(module,inputs):
        if done[0]:return
        manifest=json.loads((Path(os.environ['EXP_DIR'])/'initialization/WEIGHT_MANIFEST.json').read_text())
        by_name=dict(model.named_parameters())
        required=('audio_encoder.','audio_adaptor.','decoder.','embedding.','output_layer.','streaming_gate.','cosyvoice_llm.','cosyvoice_llm_input_proj.','cosyvoice_codec_embed.','cosyvoice_llm_codec_head.','thinker_layer_fusion.')
        checked=[]
        for name,p in by_name.items():
            if not name.startswith(required):continue
            assert name in manifest,('missing initialized parameter',name)
            spec=manifest[name]
            assert list(p.shape)==spec['shape'],(name,p.shape,spec['shape'])
            value=p.detach().to(device='cpu',dtype=torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
            assert hashlib.sha256(value).hexdigest()==spec['sha256_bf16'],('initialization mismatch',name)
            checked.append(name)
        assert len(checked)>500,len(checked)
        done[0]=True
        if torch.distributed.get_rank()==0:
            (Path(os.environ['EXP_DIR'])/'INITIALIZATION_VERIFIED.json').write_text(json.dumps(dict(all_passed=True,checked_parameters=len(checked),parameters=checked),indent=2))
            print('[INIT VERIFIED] all source parameter hashes match: '+str(len(checked)),flush=True)
    model.register_forward_pre_hook(verify)
