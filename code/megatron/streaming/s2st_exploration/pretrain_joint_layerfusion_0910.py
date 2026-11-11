"""Private joint-training entry; parent segmentpad graph plus differentiable fusion."""
import json,os,warnings
from pathlib import Path
import torch
from streaming.s2st_exploration import pretrain_streaming_s2st_e2e_segmentpad_0909 as baseline
from joint_layer_fusion import install_thinker_layer_fusion

def model_provider(pre_process=True,post_process=True):
    model=baseline.model_provider(pre_process,post_process)
    fusion=install_thinker_layer_fusion(model)
    assert fusion.weights.shape==(8,2560) and fusion.weights.requires_grad
    trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
    branches={
        'encoder':'audio_encoder.','audio_adaptor':'audio_adaptor.','thinker':'decoder.',
        'gate':'streaming_gate.','bridge':'cosyvoice_llm_input_proj.',
        'talker':'cosyvoice_llm.','codec_embed':'cosyvoice_codec_embed.',
        'codec_head':'cosyvoice_llm_codec_head.','fusion':'thinker_layer_fusion.'}
    for branch,prefix in branches.items():
        candidates=[(n,p) for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        assert candidates,branch
        n,p=next(((n,p) for n,p in candidates if p.ndim>=2),candidates[0])
        calls=[0]
        def audit(grad,branch=branch,name=n,calls=calls,param=p):
            if calls[0]<2:
                actual_grad = param.main_grad if getattr(param, 'grad_added_to_main_grad', False) else grad
                if not torch.isfinite(actual_grad).all() or not actual_grad.detach().float().abs().sum()>0:
                    raise RuntimeError('Missing/nonfinite joint gradient: '+name)
                if torch.distributed.get_rank()==0:
                    print('[JOINT GRAD] '+branch+' '+name+' abs_sum='+str(actual_grad.detach().float().abs().sum().item()),flush=True)
            calls[0]+=1
            return grad
        p.register_hook(audit)
    checked=[os.environ.get("RESUME_TRAINING","0")=="1"]
    def verify_loaded(module,inputs):
        if checked[0]:return
        expected=torch.load(Path(os.environ['EXP_DIR'])/'initialization/expected_fusion.pt',map_location='cpu',weights_only=True)
        actual=module.weights.detach().cpu()
        if not torch.equal(actual,expected.to(actual.dtype)):raise RuntimeError('Parent fusion weights not restored exactly')
        checked[0]=True
        if torch.distributed.get_rank()==0:print('[INIT VERIFIED] parent fusion vectors exactly restored; all 8 taps differentiable',flush=True)
    fusion.register_forward_pre_hook(verify_loaded)
    if torch.distributed.get_rank()==0:
        root=Path(os.environ['EXP_DIR'])
        meta=dict(enabled=True,layers=list(fusion.layer_numbers),hidden_size=2560,parameter_key='thinker_layer_fusion.weights',
            hidden_convention='intermediate_post_block_final_post_norm',talker_profile='segmentpad',
            initialization='all model weights from talker_layerfusion_0909/iter_0005000',
            full_recompute='uniform_1; tap checkpoint outputs',trainable_parameters=trainable)
        (root/'layer_fusion_config.json').write_text(json.dumps(meta,indent=2)+'\n')
        args=baseline.joint.get_args()
        (root/'actual_arguments.json').write_text(json.dumps({k:str(v) for k,v in sorted(vars(args).items())},indent=2)+'\n')
        print('[JOINT FUSION] trainable='+str(sum(trainable.values())),flush=True)
    return model

if __name__=='__main__':
    joint=baseline.joint
    joint.streaming._patch_torch_dist_writer_compat()
    warnings.filterwarnings('ignore',message='The given NumPy array is not writable.*',category=UserWarning)
    provider=joint.streaming.build_streaming_data_iterators;provider.is_distributed=True
    joint.pretrain(provider,model_provider,joint.ModelType.encoder_or_decoder,joint.streaming.forward_step,
        args_defaults={'tokenizer_type':'QWenVLTokenizer'},use_train_valid_test_dataset_provider_as_data_iterators=True,
        scale_lr_cond=joint.streaming.offline.scale_lr_cond)
