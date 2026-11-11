"""Differentiable fusion with taps outside per-layer activation checkpoints.

Keeps the parent checkpoint's parameter key, layer convention and FP32 sum.
The text/gate heads still receive the original final decoder hidden states.
"""
import os
import types
import torch
from torch import nn

class HadamardLayerFusion(nn.Module):
    def __init__(self, layer_numbers, hidden_size):
        super().__init__()
        self.layer_numbers=tuple(layer_numbers)
        self.weights=nn.Parameter(torch.zeros(len(self.layer_numbers),hidden_size))
        with torch.no_grad():self.weights[-1].fill_(1)
    def forward(self, hidden_states):
        if len(hidden_states)!=len(self.layer_numbers):raise RuntimeError('Missing fusion tap')
        shape=hidden_states[-1].shape
        if any(h.shape!=shape for h in hidden_states):raise RuntimeError('Fusion shape mismatch')
        mixed=hidden_states[0].float()*self.weights[0].float()
        for i,h in enumerate(hidden_states[1:],1):mixed=mixed+h.float()*self.weights[i].float()
        return mixed.to(hidden_states[-1].dtype)

def install_thinker_layer_fusion(model, enabled=True, checkpoint_fn=None):
    if not enabled:return None
    if hasattr(model,'thinker_layer_fusion'):raise RuntimeError('Fusion already installed')
    cfg=model.config
    if any(getattr(cfg,n,1)!=1 for n in ['tensor_model_parallel_size','pipeline_model_parallel_size','context_parallel_size']):
        raise ValueError('Fusion requires TP=PP=CP=1')
    if getattr(cfg,'fp8',False):raise ValueError('This experiment uses BF16, not FP8')
    if getattr(cfg,'cosyvoice_codex_use_text_token_input',False):raise ValueError('Hidden conditioning required')
    full=getattr(cfg,'recompute_granularity',None)=='full'
    if full and (cfg.recompute_method!='uniform' or cfg.recompute_num_layers!=1):
        raise ValueError('Joint fusion requires uniform one-layer checkpoints')
    decoder=model.decoder
    selected=tuple(range(4,len(decoder.layers)+1,4))
    if len(decoder.layers)!=32:raise ValueError('Expected the parent 32-layer Thinker')
    for i,layer in enumerate(decoder.layers,1):
        if getattr(layer,'layer_number',i)!=i:raise ValueError('Unexpected layer numbering')
    fusion=HadamardLayerFusion(selected,cfg.hidden_size)
    fusion.to(device=model.cosyvoice_llm_input_proj.weight.device,dtype=model.cosyvoice_llm_input_proj.weight.dtype)
    fusion.weights.lr_scale=1.0
    model.add_module('thinker_layer_fusion',fusion)
    captured={};active=[False];used=[False]
    def remember(index,hidden):
        if index not in selected[:-1]:return
        if not active[0] or index in captured:raise RuntimeError('Invalid fusion capture lifetime')
        # Preserve the graph, protecting the tap from later in-place operations.
        captured[index]=hidden.clone()
    def hook(index):
        def capture(module,inputs,output):
            # Reentrant checkpoint inner forwards run under no_grad and again
            # during backward. Their local outputs must never become taps.
            if active[0] and not (full and decoder.training):
                remember(index,output[0] if isinstance(output,tuple) else output)
        return capture
    for i in selected[:-1]:decoder.layers[i-1].register_forward_hook(hook(i))
    if full:
        if checkpoint_fn is None:
            from megatron.core import tensor_parallel
            checkpoint_fn=tensor_parallel.checkpoint
        def checkpointed_forward(self,hidden_states,attention_mask,context,context_mask,rotary_pos_emb,packed_seq_params):
            # Same uniform=1 execution as TransformerBlock._checkpointed_forward.
            # Capture the checkpoint OUTPUT (autograd-connected), not its inner
            # no_grad layer output, so codec gradients reach every selected layer.
            for index in range(self.num_layers_per_pipeline_rank):
                def custom(h,mask,ctx,ctx_mask,rope,layer_index=index):
                    return self._get_layer(layer_index)(hidden_states=h,attention_mask=mask,context=ctx,
                        context_mask=ctx_mask,rotary_pos_emb=rope,inference_params=None,packed_seq_params=packed_seq_params)
                hidden_states,context=checkpoint_fn(custom,cfg.distribute_saved_activations,
                    hidden_states,attention_mask,context,context_mask,rotary_pos_emb)
                remember(index+1,hidden_states)
            return hidden_states
        decoder._checkpointed_forward=types.MethodType(checkpointed_forward,decoder)
    def bridge_input(module,inputs):
        if not active[0] or used[0] or len(inputs)!=1 or set(captured)!=set(selected[:-1]):
            raise RuntimeError('Incomplete or repeated fusion bridge invocation')
        last=inputs[0]
        hidden=[captured[i] for i in selected[:-1]]+[last]
        if model.training and torch.is_grad_enabled() and not all(h.requires_grad for h in hidden):
            raise RuntimeError('A fusion tap detached the Thinker gradient')
        mixed=fusion(hidden);used[0]=True
        return (mixed,)
    model.cosyvoice_llm_input_proj.register_forward_pre_hook(bridge_input)
    original=model.forward
    def forward(self,*args,**kwargs):
        if active[0]:raise RuntimeError('Reentrant model forward')
        captured.clear();active[0]=True;used[0]=False
        try:
            result=original(*args,**kwargs)
            if not used[0]:raise RuntimeError('Fusion bridge was bypassed')
            return result
        finally:active[0]=False;captured.clear()
    model.forward=types.MethodType(forward,model)
    return fusion
