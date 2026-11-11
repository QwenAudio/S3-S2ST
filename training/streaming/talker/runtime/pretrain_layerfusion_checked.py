#!/usr/bin/env python3
"""Talker-first segmentpad experiment with optional frozen-Thinker layer fusion."""
import json
import os
from pathlib import Path
import warnings
import torch
from audio_io_checks import install as install_audio_io_checks
install_audio_io_checks()
import baseline_segmentpad as baseline
from thinker_layer_fusion import install_thinker_layer_fusion

def model_provider(pre_process=True, post_process=True):
    model = baseline.model_provider(pre_process=pre_process, post_process=post_process)
    fusion = install_thinker_layer_fusion(model)
    allowed = ("cosyvoice_llm.", "cosyvoice_llm_input_proj.", "cosyvoice_codec_embed.", "cosyvoice_llm_codec_head.", "thinker_layer_fusion.")
    trainable = {n: p.numel() for n,p in model.named_parameters() if p.requires_grad}
    assert all(n.startswith(allowed) for n in trainable), trainable
    if fusion is not None:
        assert fusion.weights.shape == (8, 2560)
        counts = [0]
        def audit_grad(grad):
            if counts[0] < 3:
                norms = grad.detach().float().abs().sum(dim=-1)
                if not torch.isfinite(norms).all() or not torch.all(norms > 0):
                    raise RuntimeError("Missing/nonfinite gradient in a layer-fusion vector")
                if torch.distributed.get_rank() == 0:
                    print("[LAYER FUSION GRAD] row_l1=" + str(norms.cpu().tolist()), flush=True)
            counts[0] += 1
            return grad
        fusion.weights.register_hook(audit_grad)
        calls=[0]
        def audit_update(module, inputs):
            calls[0]+=1
            if calls[0] in (1,9,17,33) and torch.distributed.get_rank()==0:
                w=module.weights.detach().float()
                print("[LAYER FUSION WEIGHTS] forward=%d earlier_absmax=%.9g last_mean=%.9g" % (calls[0],w[:-1].abs().max().item(),w[-1].mean().item()),flush=True)
        fusion.register_forward_pre_hook(audit_update)
    if torch.distributed.get_rank()==0:
        path=Path(os.environ["EXP_DIR"])
        path.mkdir(parents=True,exist_ok=True)
        metadata={"enabled":fusion is not None,"layers":list(fusion.layer_numbers) if fusion else [],"hidden_size":2560,"parameter_key":"thinker_layer_fusion.weights","initialization":"same mixed initialization as streaming_v6 experiment; see initialization/SOURCE.json","hidden_convention":"intermediate post-block outputs; final layer uses existing post-final-norm output","formula":"sum_i(w_i * h_i) before existing projection; no softmax or added norm","trainable_parameters":trainable}
        (path/"layer_fusion_config.json").write_text(json.dumps(metadata,indent=2))
        print("[LAYER FUSION TRAINABLE] total="+str(sum(trainable.values())),flush=True)
    from verify_initialization import install_verification
    install_verification(model)
    if torch.distributed.get_rank()==0:
        args=baseline.get_args()
        (Path(os.environ["EXP_DIR"])/"actual_arguments.json").write_text(json.dumps({k:str(v) for k,v in sorted(vars(args).items())},indent=2))
    return model

if __name__=="__main__":
    baseline.streaming._patch_torch_dist_writer_compat()
    warnings.filterwarnings("ignore",message="The given NumPy array is not writable.*",category=UserWarning)
    provider=baseline.streaming.build_streaming_data_iterators
    provider.is_distributed=True
    baseline.pretrain(provider,model_provider,baseline.ModelType.encoder_or_decoder,baseline.streaming.forward_step,args_defaults={"tokenizer_type":"QWenVLTokenizer"},use_train_valid_test_dataset_provider_as_data_iterators=True,scale_lr_cond=baseline.streaming.offline.scale_lr_cond)
