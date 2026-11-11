# Copyright (c) 2025. Qwen3.5-MoE HF checkpoint loader for Megatron-LM with multimodal support.
"""Load Qwen3.5-MoE HuggingFace checkpoints into Megatron format.

Handles:
- Heterogeneous layers (linear_attention + full_attention)
- Attention output gating (doubled Q projection)
- GatedDeltaNet linear attention weights
- Shared expert with sigmoid gate
- Zero-centered gamma RMSNorm
- Partial rotary embeddings
- Interleaved M-RoPE
- Audio encoder / adaptor multimodal parts
"""

import os
import sys
import torch
import transformers
import gc
from tqdm import tqdm
import types
from contextlib import contextmanager


@contextmanager
def _unpatch_modelopt_load_pretrained():
    """Temporarily restore transformers' original _load_pretrained_model when it
    was broken by NVIDIA modelopt's plugin (TypeError: missing
    pretrained_model_name_or_path).
    """
    from transformers import modeling_utils
    PreTrainedModel = modeling_utils.PreTrainedModel
    current = getattr(PreTrainedModel, '_load_pretrained_model', None)
    original = None
    if current is not None and getattr(current, '__name__', '').startswith('_new'):
        # modelopt wrapper; try to get original from __wrapped__, closure, or cache
        original = getattr(current, '__wrapped__', None)
        if original is None and getattr(current, '__closure__', None):
            for cell in current.__closure__:
                try:
                    v = cell.cell_contents
                    if callable(v) and getattr(v, '__name__', '') == '_load_pretrained_model':
                        original = v
                        break
                except Exception:
                    pass
        if original is None and hasattr(PreTrainedModel, '_modelopt_cache'):
            original = PreTrainedModel._modelopt_cache.get('_load_pretrained_model')
    if original is not None:
        PreTrainedModel._load_pretrained_model = original
    try:
        yield
    finally:
        if original is not None:
            PreTrainedModel._load_pretrained_model = current

random_init_audio_encoder = os.environ.get('Random_Init_Audio_Encoder', 'False')
random_init_audio_encoder = True if random_init_audio_encoder == 'True' else False
random_init_audio_adaptor = os.environ.get('Random_Init_Audio_Adaptor', 'False')
random_init_audio_adaptor = True if random_init_audio_adaptor == 'True' else False
channels_model = os.environ.get('Channels_Model', 'False')
channels_model = True if channels_model == 'True' else False

print(f'random_init_audio_encoder: {random_init_audio_encoder}')
print(f'random_init_audio_adaptor: {random_init_audio_adaptor}')
print(f'channels_model: {channels_model}')


def add_arguments(parser):
    group = parser.add_argument_group(title='Qwen3.5 HF loader.')
    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--audio-adaptor-pt-path', type=str, default=None,
                       help='Path to HF-format audio_adaptor.pt. If omitted, uses '
                            '<load-dir>/audio_adaptor.pt when that file exists.')


def load_args_from_checkpoint_qwen3_5(args, hf_config_path):
    """Read Qwen3.5-MoE config from HF checkpoint and set Megatron args."""
    from transformers import AutoConfig
    full_config = AutoConfig.from_pretrained(hf_config_path)

    if hasattr(full_config, 'text_config'):
        qwen_config = full_config.text_config
    else:
        qwen_config = full_config

    args.seq_length = 4096
    args.global_batch_size = 1024
    args.iteration = 1
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = True
    args.swiglu = True
    args.bf16 = True
    args.add_bias_linear = False
    args.normalization = "RMSNorm"
    args.tokenizer_type = "HuggingFaceTokenizer"
    args.disable_bias_linear = True
    args.add_qkv_bias = getattr(qwen_config, 'qkv_bias', False)
    args.qk_layernorm = getattr(qwen_config, 'use_qk_norm', True)

    tie_word_embeddings = getattr(qwen_config, 'tie_word_embeddings',
                                   getattr(full_config, 'tie_word_embeddings', True))
    args.untie_embeddings_and_output_weights = not tie_word_embeddings
    args.kv_channels = getattr(qwen_config, "head_dim", None)
    args.max_position_embeddings = qwen_config.max_position_embeddings
    args.hidden_size = qwen_config.hidden_size
    args.num_attention_heads = qwen_config.num_attention_heads
    args.num_layers = qwen_config.num_hidden_layers
    args.norm_epsilon = qwen_config.rms_norm_eps

    rope_params = getattr(qwen_config, 'rope_parameters', {}) or {}
    args.rotary_base = getattr(qwen_config, 'rope_theta', None) or rope_params.get('rope_theta', 10000000)
    args.vocab_size = qwen_config.vocab_size
    args.padded_vocab_size = qwen_config.vocab_size
    args.qwen3_5_config = qwen_config
    args.sequence_parallel = True

    if hasattr(qwen_config, 'num_experts'):
        args.num_experts = qwen_config.num_experts
        args.moe_router_topk = qwen_config.num_experts_per_tok
        args.ffn_hidden_size = qwen_config.moe_intermediate_size
    else:
        args.ffn_hidden_size = qwen_config.intermediate_size

    if qwen_config.num_key_value_heads:
        args.group_query_attention = True
        args.num_query_groups = qwen_config.num_key_value_heads

    if hasattr(qwen_config, 'shared_expert_intermediate_size'):
        args.moe_shared_expert_intermediate_size = qwen_config.shared_expert_intermediate_size

    args.apply_layernorm_1p = True
    args.attn_output_gate = getattr(qwen_config, 'attn_output_gate', True)
    args.moe_norm_topk_scores = getattr(qwen_config, 'norm_topk_prob', False)

    if hasattr(qwen_config, 'layer_types'):
        args.layer_types = qwen_config.layer_types
    else:
        args.layer_types = None

    args.rotary_percent = getattr(qwen_config, 'partial_rotary_factor', None) or rope_params.get('partial_rotary_factor', 1.0)
    args.mrope_interleaved = getattr(qwen_config, 'mrope_interleaved', None)
    if args.mrope_interleaved is None:
        args.mrope_interleaved = rope_params.get('mrope_interleaved', False)
    args.mrope_section = getattr(qwen_config, 'mrope_section', None) or rope_params.get('mrope_section', None)
    args.full_attention_interval = getattr(qwen_config, 'full_attention_interval', 0)
    args.linear_attention_conv_kernel_dim = getattr(qwen_config, 'linear_conv_kernel_dim',
                                                     getattr(qwen_config, 'linear_conv_dim', 4))
    args.linear_attention_key_head_dim = getattr(qwen_config, 'linear_key_head_dim', 128)
    args.linear_attention_value_head_dim = getattr(qwen_config, 'linear_value_head_dim', 128)
    args.linear_attention_num_key_heads = getattr(qwen_config, 'linear_num_key_heads', 16)
    args.linear_attention_num_value_heads = getattr(qwen_config, 'linear_num_value_heads', 64)


def load_args_from_checkpoint_qwen3(args, hf_config_path):
    """Read Qwen3 config (for audio encoder) from HF checkpoint."""
    from transformers import AutoConfig
    qwen3_config = AutoConfig.from_pretrained(hf_config_path)

    args.seq_length = 4096
    args.global_batch_size = 1024
    args.iteration = 1
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = True
    args.swiglu = True
    args.bf16 = True
    args.add_bias_linear = False
    args.normalization = "RMSNorm"
    args.tokenizer_type = "HuggingFaceTokenizer"
    args.disable_bias_linear = True
    args.add_qkv_bias = getattr(qwen3_config, 'qkv_bias', False)
    args.qk_layernorm = getattr(qwen3_config, 'use_qk_norm', True)
    args.untie_embeddings_and_output_weights = not qwen3_config.tie_word_embeddings
    args.kv_channels = getattr(qwen3_config, "head_dim", None)
    args.max_position_embeddings = qwen3_config.max_position_embeddings
    args.hidden_size = qwen3_config.hidden_size
    args.num_attention_heads = qwen3_config.num_attention_heads
    args.num_layers = qwen3_config.num_hidden_layers
    args.norm_epsilon = qwen3_config.rms_norm_eps
    args.rotary_base = qwen3_config.rope_theta
    args.vocab_size = qwen3_config.vocab_size
    args.padded_vocab_size = qwen3_config.vocab_size
    args.qwen3 = qwen3_config
    args.sequence_parallel = True
    if "num_experts" in qwen3_config.__dict__.keys():
        args.num_experts = qwen3_config.num_experts
        args.moe_router_topk = qwen3_config.num_experts_per_tok
        args.ffn_hidden_size = qwen3_config.moe_intermediate_size
    else:
        args.ffn_hidden_size = qwen3_config.intermediate_size
    if qwen3_config.num_key_value_heads:
        args.group_query_attention = True
        args.num_query_groups = qwen3_config.num_key_value_heads
    args.moe_norm_topk_scores = getattr(qwen3_config, 'norm_topk_prob', False)


def verify_transformers_version():
    # version_parts = transformers.__version__.split('.')[:2]
    # major, minor = map(int, version_parts)
    # assert major >= 4 and minor >= 36
    return 


def set_preprocess_state(args, model, hf_model):
    """Set embedding params."""
    model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model, model_type, if_decoder=True):
    """Set output layer & norm params."""
    if if_decoder:
        model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    else:
        model.encoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    if args.untie_embeddings_and_output_weights:
        model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def set_full_attn_state(args, layer, hf_layer):
    """Set full attention params (with output gating support)."""
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn
    tp = args.tensor_model_parallel_size
    num_heads = args.num_attention_heads // tp
    num_query_groups = (args.num_query_groups if args.group_query_attention \
                        else args.num_attention_heads) // tp
    num_querys_per_group = num_heads // num_query_groups
    dim = args.kv_channels

    if args.qk_layernorm:
        attn.q_layernorm.weight.data.copy_(hf_attn.q_norm.weight)
        attn.k_layernorm.weight.data.copy_(hf_attn.k_norm.weight)

    if args.attn_output_gate:
        q_weight = hf_attn.q_proj.weight
        num_heads = num_query_groups * num_querys_per_group
        # HF interleaves Q and Gate per head: [Q0(dim), G0(dim), Q1(dim), G1(dim), ...]
        q_per_head = q_weight.reshape(num_heads, 2 * dim, -1)
        q_query = q_per_head[:, :dim, :].reshape(-1, args.hidden_size)
        q_gate = q_per_head[:, dim:, :].reshape(-1, args.hidden_size)
        attn.linear_qkv.weight.data.copy_(torch.cat([
            q_query.reshape((num_query_groups, num_querys_per_group * dim, -1)),
            q_gate.reshape((num_query_groups, num_querys_per_group * dim, -1)),
            hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
            hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
        ], dim=1).reshape((-1, args.hidden_size)))
    else:
        attn.linear_qkv.weight.data.copy_(torch.cat([
            hf_attn.q_proj.weight.reshape((num_query_groups, num_querys_per_group * dim, -1)),
            hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
            hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
        ], dim=1).reshape((-1, args.hidden_size)))

    attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)


def set_linear_attn_state(args, layer, hf_layer):
    """Set GatedDeltaNet linear attention params."""
    attn = layer.self_attention
    hf_attn = hf_layer.linear_attn
    attn.in_proj_qkv.weight.data.copy_(hf_attn.in_proj_qkv.weight)
    attn.in_proj_z.weight.data.copy_(hf_attn.in_proj_z.weight)
    attn.in_proj_b.weight.data.copy_(hf_attn.in_proj_b.weight)
    attn.in_proj_a.weight.data.copy_(hf_attn.in_proj_a.weight)
    attn.conv1d.weight.data.copy_(hf_attn.conv1d.weight)
    if hf_attn.conv1d.bias is not None:
        attn.conv1d.bias.data.copy_(hf_attn.conv1d.bias)
    attn.dt_bias.data.copy_(hf_attn.dt_bias)
    attn.A_log.data.copy_(hf_attn.A_log)
    attn.norm.weight.data.copy_(hf_attn.norm.weight)
    attn.out_proj.weight.data.copy_(hf_attn.out_proj.weight)


def set_attn_state_qwen3(args, layer, hf_layer):
    """Set self-attention params for Qwen3 (audio encoder layers)."""
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn
    tp = args.tensor_model_parallel_size
    num_heads = args.num_attention_heads // tp
    num_query_groups = (args.num_query_groups if args.group_query_attention \
                        else args.num_attention_heads) // tp
    num_querys_per_group = num_heads // num_query_groups
    dim = args.kv_channels
    assert num_heads % num_querys_per_group == 0

    if args.qk_layernorm:
        attn.q_layernorm.weight.data.copy_(hf_attn.q_norm.weight)
        attn.k_layernorm.weight.data.copy_(hf_attn.k_norm.weight)

    attn.linear_qkv.weight.data.copy_(torch.cat([
        hf_attn.q_proj.weight.reshape((num_query_groups, num_querys_per_group * dim, -1)),
        hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
        hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size)))
    attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)

    if args.add_qkv_bias:
        attn.linear_qkv.bias.data.copy_(torch.cat([
            hf_attn.q_proj.bias.reshape((num_query_groups, num_querys_per_group * dim, -1)),
            hf_attn.k_proj.bias.reshape((num_query_groups, dim, -1)),
            hf_attn.v_proj.bias.reshape((num_query_groups, dim, -1)),
        ], dim=1).reshape(num_query_groups * (num_querys_per_group + 2) * dim))


def set_moe_mlp_state(args, layer, hf_layer):
    """Set MOE MLP params."""
    layer.mlp.router.weight.data.copy_(hf_layer.mlp.gate.weight)
    if layer.mlp.shared_experts is not None:
        if hasattr(hf_layer.mlp, 'shared_expert_gate'):
            layer.mlp.shared_experts.gate_weight.data.copy_(
                hf_layer.mlp.shared_expert_gate.weight)

    mcore_experts = layer.mlp.experts.local_experts
    hf_experts = hf_layer.mlp.experts

    if hasattr(hf_experts, 'gate_up_proj'):
        # Qwen3.5 fused expert format: gate_up_proj [num_experts, 2*intermediate, hidden]
        for expert_idx in range(args.num_experts):
            mcore_experts[expert_idx].linear_fc1.weight.data.copy_(
                hf_experts.gate_up_proj[expert_idx])
            mcore_experts[expert_idx].linear_fc2.weight.data.copy_(
                hf_experts.down_proj[expert_idx])
    else:
        # Per-expert format: experts[i].gate_proj / up_proj / down_proj
        for expert_idx in range(args.num_experts):
            mcore_experts[expert_idx].linear_fc1.weight.data.copy_(
                torch.cat([hf_experts[expert_idx].gate_proj.weight,
                           hf_experts[expert_idx].up_proj.weight], dim=0))
            mcore_experts[expert_idx].linear_fc2.weight.data.copy_(
                hf_experts[expert_idx].down_proj.weight)

    if layer.mlp.shared_experts is not None:
        layer.mlp.shared_experts.linear_fc1.weight.data.copy_(
            torch.cat([hf_layer.mlp.shared_expert.gate_proj.weight,
                       hf_layer.mlp.shared_expert.up_proj.weight], dim=0))
        layer.mlp.shared_experts.linear_fc2.weight.data.copy_(
            hf_layer.mlp.shared_expert.down_proj.weight)


def set_mlp_state(args, layer, hf_layer):
    """Set dense MLP params (for audio encoder layers)."""
    layer.mlp.linear_fc1.weight.data.copy_(
        torch.cat([hf_layer.mlp.gate_proj.weight,
                   hf_layer.mlp.up_proj.weight], dim=0))
    layer.mlp.linear_fc2.weight.data.copy_(hf_layer.mlp.down_proj.weight)


def set_layer_state_qwen3_5(args, model, hf_model, layer_idx):
    """Set transformer layer params for Qwen3.5 LLM backbone."""
    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    layer_type = 'full_attention'
    if args.layer_types is not None and layer_idx < len(args.layer_types):
        layer_type = args.layer_types[layer_idx]

    if layer_type == 'linear_attention':
        set_linear_attn_state(args, layer, hf_layer)
        layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)
    else:
        set_full_attn_state(args, layer, hf_layer)
        if hasattr(layer.self_attention.linear_qkv, 'layer_norm_weight'):
            layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(
                hf_layer.input_layernorm.weight)
        else:
            layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)

    if args.num_experts:
        set_moe_mlp_state(args, layer, hf_layer)
        layer.pre_mlp_layernorm.weight.data.copy_(
            hf_layer.post_attention_layernorm.weight)
    else:
        layer.mlp.linear_fc1.weight.data.copy_(
            torch.cat([hf_layer.mlp.gate_proj.weight,
                       hf_layer.mlp.up_proj.weight], dim=0))
        layer.mlp.linear_fc2.weight.data.copy_(hf_layer.mlp.down_proj.weight)
        if hasattr(layer.mlp.linear_fc1, 'layer_norm_weight'):
            layer.mlp.linear_fc1.layer_norm_weight.data.copy_(
                hf_layer.post_attention_layernorm.weight)


def set_layer_state_qwen3(args, model, hf_model, layer_idx, if_decoder=True):
    """Set transformer layer params for Qwen3 audio encoder."""
    if if_decoder:
        layer = model.decoder.layers[layer_idx]
    else:
        layer = model.encoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state_qwen3(args, layer, hf_layer)
    if args.num_experts:
        set_moe_mlp_state(args, layer, hf_layer)
        layer.pre_mlp_layernorm.weight.data.copy_(
            hf_layer.post_attention_layernorm.weight)
    else:
        set_mlp_state(args, layer, hf_layer)
        layer.mlp.linear_fc1.layer_norm_weight.data.copy_(
            hf_layer.post_attention_layernorm.weight)

    layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(
        hf_layer.input_layernorm.weight)


def set_audio_encoder_state(args, model, hf_model):
    """Set audio encoder params (paraformer-style)."""
    model.audio_encoder.conv1.conv.weight.data.copy_(hf_model['audio_encoder.conv1.weight'])
    model.audio_encoder.conv1.conv.bias.data.copy_(hf_model['audio_encoder.conv1.bias'])
    model.audio_encoder.conv2.conv.weight.data.copy_(hf_model['audio_encoder.conv2.weight'])
    model.audio_encoder.conv2.conv.bias.data.copy_(hf_model['audio_encoder.conv2.bias'])
    model.audio_encoder.ln_post.weight.data.copy_(hf_model['audio_encoder.ln_post.weight'])
    model.audio_encoder.ln_post.bias.data.copy_(hf_model['audio_encoder.ln_post.bias'])
    try:
        print(f"num_blocks: {args.audio_num_blocks}")
    except Exception:
        print("num_blocks: 32")
    for block_idx in range(32):
        pfx = f'audio_encoder.blocks.{block_idx}'
        b = model.audio_encoder.blocks[block_idx]
        b.attn.query.weight.data.copy_(hf_model[f'{pfx}.attn.query.weight'])
        b.attn.query.bias.data.copy_(hf_model[f'{pfx}.attn.query.bias'])
        b.attn.key.weight.data.copy_(hf_model[f'{pfx}.attn.key.weight'])
        b.attn.value.weight.data.copy_(hf_model[f'{pfx}.attn.value.weight'])
        b.attn.value.bias.data.copy_(hf_model[f'{pfx}.attn.value.bias'])
        b.attn.out.weight.data.copy_(hf_model[f'{pfx}.attn.out.weight'])
        b.attn.out.bias.data.copy_(hf_model[f'{pfx}.attn.out.bias'])
        b.attn.fsmn_block.weight.data.copy_(hf_model[f'{pfx}.attn.fsmn_block.weight'])
        b.attn_ln.weight.data.copy_(hf_model[f'{pfx}.attn_ln.weight'])
        b.attn_ln.bias.data.copy_(hf_model[f'{pfx}.attn_ln.bias'])
        b.mlp[0].weight.data.copy_(hf_model[f'{pfx}.mlp.0.weight'])
        b.mlp[0].bias.data.copy_(hf_model[f'{pfx}.mlp.0.bias'])
        b.mlp[2].weight.data.copy_(hf_model[f'{pfx}.mlp.2.weight'])
        b.mlp[2].bias.data.copy_(hf_model[f'{pfx}.mlp.2.bias'])
        b.mlp_ln.weight.data.copy_(hf_model[f'{pfx}.mlp_ln.weight'])
        b.mlp_ln.bias.data.copy_(hf_model[f'{pfx}.mlp_ln.bias'])


def set_audio_encoder_conv_state(args, model, hf_model):
    model.audio_encoder.conv1.weight.data.copy_(hf_model['encoder.conv1.weight'])
    model.audio_encoder.conv1.bias.data.copy_(hf_model['encoder.conv1.bias'])
    model.audio_encoder.conv2.weight.data.copy_(hf_model['encoder.conv2.weight'])
    model.audio_encoder.conv2.bias.data.copy_(hf_model['encoder.conv2.bias'])


def set_sensevoicesmall_audio_encoder_state(args, model, hf_model):
    """Set sensevoicesmall audio encoder params."""
    for block_idx in range(1):
        pfx = f'encoder.encoders0.{block_idx}'
        e = model.audio_encoder.encoder.encoders0[block_idx]
        e.self_attn.linear_out.weight.data.copy_(hf_model[f'{pfx}.self_attn.linear_out.weight'])
        e.self_attn.linear_out.bias.data.copy_(hf_model[f'{pfx}.self_attn.linear_out.bias'])
        e.self_attn.linear_q_k_v.weight.data.copy_(hf_model[f'{pfx}.self_attn.linear_q_k_v.weight'])
        e.self_attn.linear_q_k_v.bias.data.copy_(hf_model[f'{pfx}.self_attn.linear_q_k_v.bias'])
        e.self_attn.fsmn_block.weight.data.copy_(hf_model[f'{pfx}.self_attn.fsmn_block.weight'])
        e.feed_forward.w_1.weight.data.copy_(hf_model[f'{pfx}.feed_forward.w_1.weight'])
        e.feed_forward.w_1.bias.data.copy_(hf_model[f'{pfx}.feed_forward.w_1.bias'])
        e.feed_forward.w_2.weight.data.copy_(hf_model[f'{pfx}.feed_forward.w_2.weight'])
        e.feed_forward.w_2.bias.data.copy_(hf_model[f'{pfx}.feed_forward.w_2.bias'])
        e.norm1.weight.data.copy_(hf_model[f'{pfx}.norm1.weight'])
        e.norm1.bias.data.copy_(hf_model[f'{pfx}.norm1.bias'])
        e.norm2.weight.data.copy_(hf_model[f'{pfx}.norm2.weight'])
        e.norm2.bias.data.copy_(hf_model[f'{pfx}.norm2.bias'])

    for block_idx in range(49):
        pfx = f'encoder.encoders.{block_idx}'
        e = model.audio_encoder.encoder.encoders[block_idx]
        e.self_attn.linear_out.weight.data.copy_(hf_model[f'{pfx}.self_attn.linear_out.weight'])
        e.self_attn.linear_out.bias.data.copy_(hf_model[f'{pfx}.self_attn.linear_out.bias'])
        e.self_attn.linear_q_k_v.weight.data.copy_(hf_model[f'{pfx}.self_attn.linear_q_k_v.weight'])
        e.self_attn.linear_q_k_v.bias.data.copy_(hf_model[f'{pfx}.self_attn.linear_q_k_v.bias'])
        e.self_attn.fsmn_block.weight.data.copy_(hf_model[f'{pfx}.self_attn.fsmn_block.weight'])
        e.feed_forward.w_1.weight.data.copy_(hf_model[f'{pfx}.feed_forward.w_1.weight'])
        e.feed_forward.w_1.bias.data.copy_(hf_model[f'{pfx}.feed_forward.w_1.bias'])
        e.feed_forward.w_2.weight.data.copy_(hf_model[f'{pfx}.feed_forward.w_2.weight'])
        e.feed_forward.w_2.bias.data.copy_(hf_model[f'{pfx}.feed_forward.w_2.bias'])
        e.norm1.weight.data.copy_(hf_model[f'{pfx}.norm1.weight'])
        e.norm1.bias.data.copy_(hf_model[f'{pfx}.norm1.bias'])
        e.norm2.weight.data.copy_(hf_model[f'{pfx}.norm2.weight'])
        e.norm2.bias.data.copy_(hf_model[f'{pfx}.norm2.bias'])

    for block_idx in range(20):
        pfx = f'encoder.tp_encoders.{block_idx}'
        e = model.audio_encoder.encoder.tp_encoders[block_idx]
        e.self_attn.linear_out.weight.data.copy_(hf_model[f'{pfx}.self_attn.linear_out.weight'])
        e.self_attn.linear_out.bias.data.copy_(hf_model[f'{pfx}.self_attn.linear_out.bias'])
        e.self_attn.linear_q_k_v.weight.data.copy_(hf_model[f'{pfx}.self_attn.linear_q_k_v.weight'])
        e.self_attn.linear_q_k_v.bias.data.copy_(hf_model[f'{pfx}.self_attn.linear_q_k_v.bias'])
        e.self_attn.fsmn_block.weight.data.copy_(hf_model[f'{pfx}.self_attn.fsmn_block.weight'])
        e.feed_forward.w_1.weight.data.copy_(hf_model[f'{pfx}.feed_forward.w_1.weight'])
        e.feed_forward.w_1.bias.data.copy_(hf_model[f'{pfx}.feed_forward.w_1.bias'])
        e.feed_forward.w_2.weight.data.copy_(hf_model[f'{pfx}.feed_forward.w_2.weight'])
        e.feed_forward.w_2.bias.data.copy_(hf_model[f'{pfx}.feed_forward.w_2.bias'])
        e.norm1.weight.data.copy_(hf_model[f'{pfx}.norm1.weight'])
        e.norm1.bias.data.copy_(hf_model[f'{pfx}.norm1.bias'])
        e.norm2.weight.data.copy_(hf_model[f'{pfx}.norm2.weight'])
        e.norm2.bias.data.copy_(hf_model[f'{pfx}.norm2.bias'])

    model.audio_encoder.encoder.after_norm.weight.data.copy_(hf_model['encoder.after_norm.weight'])
    model.audio_encoder.encoder.after_norm.bias.data.copy_(hf_model['encoder.after_norm.bias'])
    model.audio_encoder.encoder.tp_norm.weight.data.copy_(hf_model['encoder.tp_norm.weight'])
    model.audio_encoder.encoder.tp_norm.bias.data.copy_(hf_model['encoder.tp_norm.bias'])


def load_checkpoint_to_model(args, model_type, args_audio_encoder):
    """Set model params from HF checkpoints."""
    from transformers import AutoModelForCausalLM, AutoConfig

    from pretrain_fun_base import model_provider

    num_layers_override = int(os.environ.get('NUM_LAYERS_OVERRIDE', '0'))
    if num_layers_override > 0:
        hf_config = AutoConfig.from_pretrained(args.load, trust_remote_code=True)
        if hasattr(hf_config, 'text_config'):
            hf_config.text_config.num_hidden_layers = num_layers_override
        else:
            hf_config.num_hidden_layers = num_layers_override
        print(f"[NUM_LAYERS_OVERRIDE] Loading HF model with only {num_layers_override} layers "
              f"(saves memory for {args.qwen3_5_config.num_hidden_layers}-layer model)")
        with torch.device("cpu"), _unpatch_modelopt_load_pretrained():
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.load, config=hf_config, torch_dtype=torch.bfloat16,
                trust_remote_code=True, ignore_mismatched_sizes=True)
    else:
        with torch.device("cpu"), _unpatch_modelopt_load_pretrained():
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.load, torch_dtype=torch.bfloat16, trust_remote_code=True)

    if not args.qwen3_audio_encoder:
        hf_audio_encoder_module = torch.load(args.audio_encoder_ckpt_path)
    else:
        with _unpatch_modelopt_load_pretrained():
            hf_audio_encoder_module = AutoModelForCausalLM.from_pretrained(
                args.audio_encoder_ckpt_path, torch_dtype=torch.bfloat16,
                device_map=None, low_cpu_mem_usage=False)
        from safetensors.torch import load_file
        tensors = load_file(
            args.audio_encoder_ckpt_path + '/model-00001-of-00002.safetensors')
        conv_weights = {
            'encoder.conv1.bias': tensors['encoder.conv1.bias'],
            'encoder.conv1.weight': tensors['encoder.conv1.weight'],
            'encoder.conv2.bias': tensors['encoder.conv2.bias'],
            'encoder.conv2.weight': tensors['encoder.conv2.weight'],
        }

    model = model_provider(True, True).to(args.params_dtype)

    if not random_init_audio_encoder:
        if not args.qwen3_audio_encoder:
            if args.sensevoicesmall_audio_encoder:
                set_sensevoicesmall_audio_encoder_state(args, model,
                                                       hf_audio_encoder_module)
            else:
                set_audio_encoder_state(args, model, hf_audio_encoder_module)

    if args.qwen3_audio_encoder:
        set_audio_encoder_conv_state(args, model, conv_weights)

    set_preprocess_state(args, model, hf_model)
    set_postprocess_state(args, model, hf_model, model_type)
    for layer_idx in tqdm(range(args.num_layers), "set Qwen3.5 layer states"):
        set_layer_state_qwen3_5(args, model, hf_model, layer_idx)

    if args.qwen3_audio_encoder:
        set_postprocess_state(args_audio_encoder, model.audio_encoder,
                              hf_audio_encoder_module, model_type,
                              if_decoder=False)
        for layer_idx in tqdm(range(args_audio_encoder.num_layers),
                              "set audio encoder states"):
            set_layer_state_qwen3(args_audio_encoder, model.audio_encoder,
                                  hf_audio_encoder_module, layer_idx,
                                  if_decoder=False)

    return model


def get_mlp_message_for_dense_model(message, layer, md):
    if md.swiglu:
        chunked = torch.chunk(layer.mlp.linear_fc1.weight.data, 2, dim=0)
        message["mlp l0 weight W"] = chunked[0]
        message["mlp l0 weight V"] = chunked[1]
    else:
        message["mlp l0 weight"] = layer.mlp.linear_fc1.weight.data
    message["mlp l1 weight"] = layer.mlp.linear_fc2.weight.data


def get_mlp_message_for_moe_model(message, layer, md):
    experts = layer.mlp.experts.local_experts
    message["router weight"] = layer.mlp.router.weight.data
    if md.swiglu:
        chunked = [torch.chunk(e.linear_fc1.weight.data, 2, dim=0)
                   for e in experts]
        message["mlp l0 weight W"] = torch.stack(
            [c[0] for c in chunked], dim=0)
        message["mlp l0 weight V"] = torch.stack(
            [c[1] for c in chunked], dim=0)
    else:
        message["mlp l0 weight"] = torch.stack(
            [e.linear_fc1.weight.data for e in experts])
    message["mlp l1 weight"] = torch.stack(
        [e.linear_fc2.weight.data for e in experts], dim=0)

    if layer.mlp.shared_experts is not None:
        if md.swiglu:
            chunked = torch.chunk(
                layer.mlp.shared_experts.linear_fc1.weight.data, 2, dim=0)
            message["shared mlp l0 weight W"] = chunked[0]
            message["shared mlp l0 weight V"] = chunked[1]
        else:
            message["shared mlp l0 weight"] = \
                layer.mlp.shared_experts.linear_fc1.weight.data
        message["shared mlp l1 weight"] = \
            layer.mlp.shared_experts.linear_fc2.weight.data
        message["shared mlp gate weight"] = \
            layer.mlp.shared_experts.gate_weight.data


def put_audio_encoder_message(queue_put, model):
    from types import SimpleNamespace as NS
    ae = model.audio_encoder
    blocks = []
    for i in range(len(ae.blocks)):
        blk = NS(
            attn=NS(
                query=NS(weight=ae.blocks[i].attn.query.weight.data,
                         bias=ae.blocks[i].attn.query.bias.data),
                key=NS(weight=ae.blocks[i].attn.key.weight.data),
                value=NS(weight=ae.blocks[i].attn.value.weight.data,
                         bias=ae.blocks[i].attn.value.bias.data),
                out=NS(weight=ae.blocks[i].attn.out.weight.data,
                       bias=ae.blocks[i].attn.out.bias.data),
                fsmn_block=NS(weight=ae.blocks[i].attn.fsmn_block.weight.data),
            ),
            attn_ln=NS(weight=ae.blocks[i].attn_ln.weight.data,
                       bias=ae.blocks[i].attn_ln.bias.data),
            mlp=[
                NS(weight=ae.blocks[i].mlp[0].weight.data,
                   bias=ae.blocks[i].mlp[0].bias.data),
                None,
                NS(weight=ae.blocks[i].mlp[2].weight.data,
                   bias=ae.blocks[i].mlp[2].bias.data),
            ],
            mlp_ln=NS(weight=ae.blocks[i].mlp_ln.weight.data,
                      bias=ae.blocks[i].mlp_ln.bias.data),
        )
        blocks.append(blk)

    audio_payload = NS(
        conv1=NS(conv=NS(weight=ae.conv1.conv.weight.data,
                         bias=ae.conv1.conv.bias.data)),
        conv2=NS(conv=NS(weight=ae.conv2.conv.weight.data,
                         bias=ae.conv2.conv.bias.data)),
        ln_post=NS(weight=ae.ln_post.weight.data,
                   bias=ae.ln_post.bias.data),
        blocks=blocks,
    )
    queue_put("audio encoder", {"weight": audio_payload})


def put_sensevoicesmall_audio_encoder_message(queue_put, model):
    from types import SimpleNamespace as NS
    ae = model.audio_encoder.encoder

    def _build_encoder_block(src):
        return NS(
            self_attn=NS(
                linear_out=NS(weight=src.self_attn.linear_out.weight.data,
                              bias=src.self_attn.linear_out.bias.data),
                linear_q_k_v=NS(weight=src.self_attn.linear_q_k_v.weight.data,
                                bias=src.self_attn.linear_q_k_v.bias.data),
                fsmn_block=NS(weight=src.self_attn.fsmn_block.weight.data),
            ),
            feed_forward=NS(
                w_1=NS(weight=src.feed_forward.w_1.weight.data,
                       bias=src.feed_forward.w_1.bias.data),
                w_2=NS(weight=src.feed_forward.w_2.weight.data,
                       bias=src.feed_forward.w_2.bias.data),
            ),
            norm1=NS(weight=src.norm1.weight.data, bias=src.norm1.bias.data),
            norm2=NS(weight=src.norm2.weight.data, bias=src.norm2.bias.data),
        )

    encoders0 = [_build_encoder_block(ae.encoders0[i])
                 for i in range(len(ae.encoders0))]
    encoders = [_build_encoder_block(ae.encoders[i])
                for i in range(len(ae.encoders))]
    tp_encoders = [_build_encoder_block(ae.tp_encoders[i])
                   for i in range(len(ae.tp_encoders))]

    audio_payload = NS(
        encoder=NS(
            encoders0=encoders0,
            encoders=encoders,
            tp_encoders=tp_encoders,
            after_norm=NS(weight=ae.after_norm.weight.data,
                          bias=ae.after_norm.bias.data),
            tp_norm=NS(weight=ae.tp_norm.weight.data,
                       bias=ae.tp_norm.bias.data),
        ),
    )
    queue_put("audio encoder", {"weight": audio_payload})


def hf_audio_adaptor_state_dict_to_megatron_namespace(hf_sd):
    """Map HF flat dict (from `audio_adaptor.pt`) to Megatron adaptor payload (SimpleNamespace)."""
    from types import SimpleNamespace as NS

    if not hf_sd:
        return None

    def _resolve_key(name):
        if name in hf_sd:
            return hf_sd[name]
        if not name.startswith("audio_adaptor."):
            alt = f"audio_adaptor.{name}"
            if alt in hf_sd:
                return hf_sd[alt]
        raise KeyError(name)

    block_indices = set()
    for k in hf_sd:
        if k.startswith("audio_adaptor.blocks."):
            parts = k.split(".")
            if len(parts) > 3 and parts[1] == "blocks":
                try:
                    block_indices.add(int(parts[2]))
                except ValueError:
                    pass

    blocks = []
    for block_idx in sorted(block_indices):
        pfx = f"audio_adaptor.blocks.{block_idx}"
        blk = NS(
            self_attn=NS(
                linear_q=NS(
                    weight=_resolve_key(f"{pfx}.self_attn.linear_q.weight"),
                    bias=_resolve_key(f"{pfx}.self_attn.linear_q.bias"),
                ),
                linear_k=NS(
                    weight=_resolve_key(f"{pfx}.self_attn.linear_k.weight"),
                    bias=_resolve_key(f"{pfx}.self_attn.linear_k.bias"),
                ),
                linear_v=NS(
                    weight=_resolve_key(f"{pfx}.self_attn.linear_v.weight"),
                    bias=_resolve_key(f"{pfx}.self_attn.linear_v.bias"),
                ),
                linear_out=NS(
                    weight=_resolve_key(f"{pfx}.self_attn.linear_out.weight"),
                    bias=_resolve_key(f"{pfx}.self_attn.linear_out.bias"),
                ),
            ),
            feed_forward=NS(
                w_1=NS(
                    weight=_resolve_key(f"{pfx}.feed_forward.w_1.weight"),
                    bias=_resolve_key(f"{pfx}.feed_forward.w_1.bias"),
                ),
                w_2=NS(
                    weight=_resolve_key(f"{pfx}.feed_forward.w_2.weight"),
                    bias=_resolve_key(f"{pfx}.feed_forward.w_2.bias"),
                ),
            ),
            norm1=NS(
                weight=_resolve_key(f"{pfx}.norm1.weight"),
                bias=_resolve_key(f"{pfx}.norm1.bias"),
            ),
            norm2=NS(
                weight=_resolve_key(f"{pfx}.norm2.weight"),
                bias=_resolve_key(f"{pfx}.norm2.bias"),
            ),
        )
        blocks.append(blk)

    return NS(
        linear1=NS(
            weight=_resolve_key("audio_adaptor.linear1.weight"),
            bias=_resolve_key("audio_adaptor.linear1.bias"),
        ),
        linear2=NS(
            weight=_resolve_key("audio_adaptor.linear2.weight"),
            bias=_resolve_key("audio_adaptor.linear2.bias"),
        ),
        blocks=blocks,
    )


def put_audio_adaptor_message(queue_put, load_dir, audio_adaptor_pt_path=None):
    """Send audio adaptor in the same shape as `loader_fun_mcore_qwen3_5_plus.get_message_for_audio_adaptor`."""
    from pathlib import Path

    if audio_adaptor_pt_path:
        p = Path(audio_adaptor_pt_path)
    else:
        p = Path(load_dir) / "audio_adaptor.pt"
    if not p.is_file():
        queue_put("audio adaptor", {"weight": None})
        return
    hf_sd = torch.load(str(p), map_location="cpu")
    if not isinstance(hf_sd, dict) or len(hf_sd) == 0:
        queue_put("audio adaptor", {"weight": None})
        return
    try:
        payload = hf_audio_adaptor_state_dict_to_megatron_namespace(hf_sd)
    except Exception as e:
        print(f"Warning: failed to parse audio_adaptor from {p}: {e}. Skipping.")
        queue_put("audio adaptor", {"weight": None})
        return
    queue_put("audio adaptor", {"weight": payload})


def _load_checkpoint(queue, args):
    qwen3_audio_encoder = args.qwen3_audio_encoder

    verify_transformers_version()

    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron "
              "using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    sys.argv = ['script.py',
                '--use-mcore-models',
                '--disable-bias-linear',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--mock-data',
                '--transformer-impl', 'transformer_engine',
                '--load', args.load_dir,
                '--audio-tokenizer-code-book-size', '59049',
                ]
    if channels_model:
        sys.argv.append('--audio-is-dual-channel')
    if args.qwen3_audio_encoder:
        sys.argv.extend(['--audio-encoder-type', 'qwen3-2b',
                         "--audio-encoder-dim", "2048"])
    if args.sensevoicesmall_audio_encoder:
        sys.argv.extend(['--audio-encoder-type', 'sensevoicesmall',
                         "--audio-encoder-dim", "512"])

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint_qwen3_5(margs, margs.load)

    num_layers_override = int(os.environ.get('NUM_LAYERS_OVERRIDE', '0'))
    if num_layers_override > 0:
        print(f"[NUM_LAYERS_OVERRIDE] Overriding num_layers from {margs.num_layers} to {num_layers_override}")
        margs.num_layers = num_layers_override
        if margs.layer_types is not None:
            margs.layer_types = margs.layer_types[:num_layers_override]

    margs.world_size = (margs.tensor_model_parallel_size
                        * margs.pipeline_model_parallel_size)
    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. "
                      "Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('disable_bias_linear')
    check_for_arg('add_qkv_bias')
    check_for_arg('params_dtype')
    check_for_arg('swiglu')

    margs.qwen3_audio_encoder = qwen3_audio_encoder
    margs.audio_encoder_ckpt_path = args.audio_encoder_ckpt_path
    margs.sensevoicesmall_audio_encoder = args.sensevoicesmall_audio_encoder

    assert args.model_type == 'GPT', 'Qwen3.5 is a GPT-MoE model.'
    margs.model_type = ModelType.encoder_or_decoder

    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(
        margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(
        margs.virtual_pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
    fused_kernels.load(margs)

    # Metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = margs.vocab_size
    md.make_vocab_size_divisible_by = None
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.add_qkv_bias = margs.add_qkv_bias
    md.kv_channels = margs.kv_channels
    md.qk_layernorm = margs.qk_layernorm
    md.ffn_hidden_size = margs.ffn_hidden_size

    # Qwen3.5-specific metadata
    md.attn_output_gate = margs.attn_output_gate
    md.layer_types = margs.layer_types
    md.full_attention_interval = margs.full_attention_interval
    md.linear_attention_conv_kernel_dim = margs.linear_attention_conv_kernel_dim
    md.linear_attention_key_head_dim = margs.linear_attention_key_head_dim
    md.linear_attention_value_head_dim = margs.linear_attention_value_head_dim
    md.linear_attention_num_key_heads = margs.linear_attention_num_key_heads
    md.linear_attention_num_value_heads = margs.linear_attention_num_value_heads
    md.mrope_interleaved = margs.mrope_interleaved
    md.mrope_section = margs.mrope_section
    md.apply_layernorm_1p = margs.apply_layernorm_1p

    if margs.num_experts:
        md.num_experts = margs.num_experts
        md.moe_shared_expert_intermediate_size = \
            margs.moe_shared_expert_intermediate_size
        if md.moe_shared_expert_intermediate_size:
            md.moe_shared_experts_gate = True
        else:
            md.moe_shared_experts_gate = False
    else:
        md.num_experts = None

    if args.qwen3_audio_encoder:
        args_audio_encoder = types.SimpleNamespace()
        load_args_from_checkpoint_qwen3(args_audio_encoder,
                                        args.audio_encoder_ckpt_path)
        args_audio_encoder.tensor_model_parallel_size = \
            margs.tensor_model_parallel_size
        args_audio_encoder.pipeline_model_parallel_size = \
            margs.pipeline_model_parallel_size
        args_audio_encoder.virtual_pipeline_model_parallel_size = \
            margs.virtual_pipeline_model_parallel_size
        args_audio_encoder.expert_model_parallel_size = \
            margs.expert_model_parallel_size
        args_audio_encoder.num_experts = 0
        args_audio_encoder.untie_embeddings_and_output_weights = False

        md.audio_encoder_num_layers = args_audio_encoder.num_layers
        print(f'args_audio_encoder: {args_audio_encoder}')
    else:
        args_audio_encoder = types.SimpleNamespace()

    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)

    from transformers import AutoModelForCausalLM

    with _unpatch_modelopt_load_pretrained():
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.load_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    hf_audio_encoder_module = None
    conv_weights = None
    if args.qwen3_audio_encoder:
        with _unpatch_modelopt_load_pretrained():
            hf_audio_encoder_module = AutoModelForCausalLM.from_pretrained(
                args.audio_encoder_ckpt_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        from safetensors.torch import load_file
        tensors = load_file(
            args.audio_encoder_ckpt_path + '/model-00001-of-00002.safetensors')
        conv_weights = {
            'encoder.conv1.bias': tensors['encoder.conv1.bias'],
            'encoder.conv1.weight': tensors['encoder.conv1.weight'],
            'encoder.conv2.bias': tensors['encoder.conv2.bias'],
            'encoder.conv2.weight': tensors['encoder.conv2.weight'],
        }
        del tensors
    elif not random_init_audio_encoder:
        hf_audio_encoder_module = torch.load(
            args.audio_encoder_ckpt_path, map_location="cpu")

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        def _detach_for_queue(obj):
            if torch.is_tensor(obj):
                return obj.detach()
            if isinstance(obj, dict):
                return {k: _detach_for_queue(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_detach_for_queue(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_detach_for_queue(v) for v in obj)
            return obj

        msg = _detach_for_queue(msg)
        msg["name"] = name
        queue.put(msg)

    def build_qkv_weight_from_hf(qargs, hf_attn, attn_output_gate=False):
        tp = qargs.tensor_model_parallel_size
        num_heads = qargs.num_attention_heads // tp
        num_query_groups = (
            qargs.num_query_groups if qargs.group_query_attention
            else qargs.num_attention_heads
        ) // tp
        num_querys_per_group = num_heads // num_query_groups
        dim = qargs.kv_channels

        if attn_output_gate:
            q_weight = hf_attn.q_proj.weight
            num_heads = num_query_groups * num_querys_per_group
            # HF interleaves Q and Gate per head: [Q0(dim), G0(dim), Q1(dim), G1(dim), ...]
            q_per_head = q_weight.reshape(num_heads, 2 * dim, -1)
            q_query = q_per_head[:, :dim, :].reshape(-1, qargs.hidden_size)
            q_gate = q_per_head[:, dim:, :].reshape(-1, qargs.hidden_size)
            return torch.cat([
                q_query.reshape((num_query_groups, num_querys_per_group * dim, -1)),
                q_gate.reshape((num_query_groups, num_querys_per_group * dim, -1)),
                hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
                hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
            ], dim=1).reshape((-1, qargs.hidden_size))

        return torch.cat([
            hf_attn.q_proj.weight.reshape((num_query_groups, num_querys_per_group * dim, -1)),
            hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
            hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
        ], dim=1).reshape((-1, qargs.hidden_size))

    def build_qkv_bias_from_hf(qargs, hf_attn):
        tp = qargs.tensor_model_parallel_size
        num_heads = qargs.num_attention_heads // tp
        num_query_groups = (
            qargs.num_query_groups if qargs.group_query_attention
            else qargs.num_attention_heads
        ) // tp
        num_querys_per_group = num_heads // num_query_groups
        dim = qargs.kv_channels
        return torch.cat([
            hf_attn.q_proj.bias.reshape((num_query_groups, num_querys_per_group * dim, -1)),
            hf_attn.k_proj.bias.reshape((num_query_groups, dim, -1)),
            hf_attn.v_proj.bias.reshape((num_query_groups, dim, -1)),
        ], dim=1).reshape(num_query_groups * (num_querys_per_group + 2) * dim)

    def build_linear_attn_qkv_weight_from_hf(qargs, hf_attn):
        """Split in_proj_qkv into Q/K/V, then interleave by key-head groups like build_qkv_weight_from_hf.

        HF layout: [Q_all | K_all | V_all] with shape [key_dim+key_dim+value_dim, hidden_size].
        Interleaved layout: for each of num_k_heads groups, [Q(128) | K(128) | V0..V3(4*128)].
        """
        num_k_heads = qargs.linear_attention_num_key_heads
        num_v_heads = qargs.linear_attention_num_value_heads
        head_k_dim = qargs.linear_attention_key_head_dim
        head_v_dim = qargs.linear_attention_value_head_dim
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        num_values_per_group = num_v_heads // num_k_heads

        w = hf_attn.in_proj_qkv.weight
        query, key, value = torch.split(w, [key_dim, key_dim, value_dim], dim=0)

        query = query.reshape((num_k_heads, head_k_dim, -1))
        key = key.reshape((num_k_heads, head_k_dim, -1))
        value = value.reshape((num_k_heads, num_values_per_group * head_v_dim, -1))

        return torch.cat([query, key, value], dim=1).reshape((-1, qargs.hidden_size))

    def build_linear_attn_conv1d_from_hf(qargs, hf_attn):
        """Reorder conv1d weight and bias by key-head groups, same pattern as build_linear_attn_qkv_weight_from_hf."""
        num_k_heads = qargs.linear_attention_num_key_heads
        num_v_heads = qargs.linear_attention_num_value_heads
        head_k_dim = qargs.linear_attention_key_head_dim
        head_v_dim = qargs.linear_attention_value_head_dim
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        num_values_per_group = num_v_heads // num_k_heads

        w = hf_attn.conv1d.weight
        q_c, k_c, v_c = torch.split(w, [key_dim, key_dim, value_dim], dim=0)
        q_c = q_c.reshape((num_k_heads, head_k_dim, *q_c.shape[1:]))
        k_c = k_c.reshape((num_k_heads, head_k_dim, *k_c.shape[1:]))
        v_c = v_c.reshape((num_k_heads, num_values_per_group * head_v_dim, *v_c.shape[1:]))
        weight_interleaved = torch.cat([q_c, k_c, v_c], dim=1).reshape(-1, *w.shape[1:])
        # the model use ColumnParallelLinear as weight container so we need to squeeze
        weight_interleaved = weight_interleaved.squeeze(1)

        bias_interleaved = None
        if hf_attn.conv1d.bias is not None:
            b = hf_attn.conv1d.bias
            q_b, k_b, v_b = torch.split(b, [key_dim, key_dim, value_dim], dim=0)
            q_b = q_b.reshape((num_k_heads, head_k_dim))
            k_b = k_b.reshape((num_k_heads, head_k_dim))
            v_b = v_b.reshape((num_k_heads, num_values_per_group * head_v_dim))
            bias_interleaved = torch.cat([q_b, k_b, v_b], dim=1).reshape(-1)

        return weight_interleaved, bias_interleaved

    with torch.no_grad():
        # Send embeddings directly from HF model.
        message = {"word embeddings": hf_model.model.embed_tokens.weight.data}
        if md.position_embedding_type == 'learned_absolute':
            message["position embeddings"] = hf_model.model.embed_positions.weight.data
        queue_put("embeddings", message)
        del message

        # Send Qwen3.5 LLM decoder layers (streaming).
        for layer_idx in range(margs.num_layers):
            hf_layer = hf_model.model.layers[layer_idx]
            message = {}

            layer_type = 'full_attention'
            if md.layer_types is not None and layer_idx < len(md.layer_types):
                layer_type = md.layer_types[layer_idx]
            message["layer_type"] = layer_type
            message["input norm weight"] = hf_layer.input_layernorm.weight.data

            if layer_type == 'linear_attention':
                la = hf_layer.linear_attn
                message["linear_attn in_proj_qkv weight"] = build_linear_attn_qkv_weight_from_hf(
                    margs, la
                )
                message["linear_attn in_proj_z weight"] = la.in_proj_z.weight.data
                message["linear_attn in_proj_b weight"] = la.in_proj_b.weight.data
                message["linear_attn in_proj_a weight"] = la.in_proj_a.weight.data
                conv1d_weight, conv1d_bias = build_linear_attn_conv1d_from_hf(margs, la)
                message["linear_attn conv1d weight"] = conv1d_weight
                if conv1d_bias is not None:
                    message["linear_attn conv1d bias"] = conv1d_bias
                message["linear_attn dt_bias"] = la.dt_bias.data
                message["linear_attn A_log"] = la.A_log.data
                message["linear_attn norm weight"] = la.norm.weight.data
                message["linear_attn out_proj weight"] = la.out_proj.weight.data
            else:
                hf_attn = hf_layer.self_attn
                message["qkv weight"] = build_qkv_weight_from_hf(
                    margs, hf_attn, attn_output_gate=md.attn_output_gate)
                if md.add_qkv_bias and hf_attn.q_proj.bias is not None:
                    message["qkv bias"] = build_qkv_bias_from_hf(margs, hf_attn)
                message["dense weight"] = hf_attn.o_proj.weight.data
                if md.qk_layernorm:
                    message["q layernorm weight"] = hf_attn.q_norm.weight.data
                    message["k layernorm weight"] = hf_attn.k_norm.weight.data

            message["post norm weight"] = hf_layer.post_attention_layernorm.weight.data
            if margs.num_experts:
                message["router weight"] = hf_layer.mlp.gate.weight.data
                hf_experts = hf_layer.mlp.experts

                if hasattr(hf_experts, 'gate_up_proj'):
                    gate_up = hf_experts.gate_up_proj.data
                    message["mlp l0 weight W"] = gate_up[:, :gate_up.shape[1] // 2, :]
                    message["mlp l0 weight V"] = gate_up[:, gate_up.shape[1] // 2:, :]
                    message["mlp l1 weight"] = hf_experts.down_proj.data
                else:
                    message["mlp l0 weight W"] = torch.stack(
                        [hf_experts[i].gate_proj.weight.data for i in range(margs.num_experts)], dim=0)
                    message["mlp l0 weight V"] = torch.stack(
                        [hf_experts[i].up_proj.weight.data for i in range(margs.num_experts)], dim=0)
                    message["mlp l1 weight"] = torch.stack(
                        [hf_experts[i].down_proj.weight.data for i in range(margs.num_experts)], dim=0)

                if md.moe_shared_expert_intermediate_size and hasattr(hf_layer.mlp, 'shared_expert'):
                    message["shared mlp l0 weight W"] = hf_layer.mlp.shared_expert.gate_proj.weight.data
                    message["shared mlp l0 weight V"] = hf_layer.mlp.shared_expert.up_proj.weight.data
                    message["shared mlp l1 weight"] = hf_layer.mlp.shared_expert.down_proj.weight.data
                    if hasattr(hf_layer.mlp, 'shared_expert_gate'):
                        message["shared mlp gate weight"] = hf_layer.mlp.shared_expert_gate.weight.data
            else:
                message["mlp l0 weight W"] = hf_layer.mlp.gate_proj.weight.data
                message["mlp l0 weight V"] = hf_layer.mlp.up_proj.weight.data
                message["mlp l1 weight"] = hf_layer.mlp.down_proj.weight.data

            queue_put(f"transformer layer {layer_idx}", message)
            del message
            if layer_idx % 4 == 0:
                gc.collect()

    # Put audio encoder (non-qwen3-audio) directly from ckpt dict.
    if not args.qwen3_audio_encoder and not random_init_audio_encoder:
        if args.sensevoicesmall_audio_encoder:
            encoders0 = []
            for i in range(1):
                pfx = f'encoder.encoders0.{i}'
                encoders0.append({
                    "self_attn": {
                        "linear_out": {"weight": hf_audio_encoder_module[f'{pfx}.self_attn.linear_out.weight'],
                                       "bias": hf_audio_encoder_module[f'{pfx}.self_attn.linear_out.bias']},
                        "linear_q_k_v": {"weight": hf_audio_encoder_module[f'{pfx}.self_attn.linear_q_k_v.weight'],
                                         "bias": hf_audio_encoder_module[f'{pfx}.self_attn.linear_q_k_v.bias']},
                        "fsmn_block": {"weight": hf_audio_encoder_module[f'{pfx}.self_attn.fsmn_block.weight']},
                    },
                    "feed_forward": {
                        "w_1": {"weight": hf_audio_encoder_module[f'{pfx}.feed_forward.w_1.weight'],
                                "bias": hf_audio_encoder_module[f'{pfx}.feed_forward.w_1.bias']},
                        "w_2": {"weight": hf_audio_encoder_module[f'{pfx}.feed_forward.w_2.weight'],
                                "bias": hf_audio_encoder_module[f'{pfx}.feed_forward.w_2.bias']},
                    },
                    "norm1": {"weight": hf_audio_encoder_module[f'{pfx}.norm1.weight'],
                              "bias": hf_audio_encoder_module[f'{pfx}.norm1.bias']},
                    "norm2": {"weight": hf_audio_encoder_module[f'{pfx}.norm2.weight'],
                              "bias": hf_audio_encoder_module[f'{pfx}.norm2.bias']},
                })

            encoders = []
            for i in range(49):
                pfx = f'encoder.encoders.{i}'
                encoders.append({
                    "self_attn": {
                        "linear_out": {"weight": hf_audio_encoder_module[f'{pfx}.self_attn.linear_out.weight'],
                                       "bias": hf_audio_encoder_module[f'{pfx}.self_attn.linear_out.bias']},
                        "linear_q_k_v": {"weight": hf_audio_encoder_module[f'{pfx}.self_attn.linear_q_k_v.weight'],
                                         "bias": hf_audio_encoder_module[f'{pfx}.self_attn.linear_q_k_v.bias']},
                        "fsmn_block": {"weight": hf_audio_encoder_module[f'{pfx}.self_attn.fsmn_block.weight']},
                    },
                    "feed_forward": {
                        "w_1": {"weight": hf_audio_encoder_module[f'{pfx}.feed_forward.w_1.weight'],
                                "bias": hf_audio_encoder_module[f'{pfx}.feed_forward.w_1.bias']},
                        "w_2": {"weight": hf_audio_encoder_module[f'{pfx}.feed_forward.w_2.weight'],
                                "bias": hf_audio_encoder_module[f'{pfx}.feed_forward.w_2.bias']},
                    },
                    "norm1": {"weight": hf_audio_encoder_module[f'{pfx}.norm1.weight'],
                              "bias": hf_audio_encoder_module[f'{pfx}.norm1.bias']},
                    "norm2": {"weight": hf_audio_encoder_module[f'{pfx}.norm2.weight'],
                              "bias": hf_audio_encoder_module[f'{pfx}.norm2.bias']},
                })

            tp_encoders = []
            for i in range(20):
                pfx = f'encoder.tp_encoders.{i}'
                tp_encoders.append({
                    "self_attn": {
                        "linear_out": {"weight": hf_audio_encoder_module[f'{pfx}.self_attn.linear_out.weight'],
                                       "bias": hf_audio_encoder_module[f'{pfx}.self_attn.linear_out.bias']},
                        "linear_q_k_v": {"weight": hf_audio_encoder_module[f'{pfx}.self_attn.linear_q_k_v.weight'],
                                         "bias": hf_audio_encoder_module[f'{pfx}.self_attn.linear_q_k_v.bias']},
                        "fsmn_block": {"weight": hf_audio_encoder_module[f'{pfx}.self_attn.fsmn_block.weight']},
                    },
                    "feed_forward": {
                        "w_1": {"weight": hf_audio_encoder_module[f'{pfx}.feed_forward.w_1.weight'],
                                "bias": hf_audio_encoder_module[f'{pfx}.feed_forward.w_1.bias']},
                        "w_2": {"weight": hf_audio_encoder_module[f'{pfx}.feed_forward.w_2.weight'],
                                "bias": hf_audio_encoder_module[f'{pfx}.feed_forward.w_2.bias']},
                    },
                    "norm1": {"weight": hf_audio_encoder_module[f'{pfx}.norm1.weight'],
                              "bias": hf_audio_encoder_module[f'{pfx}.norm1.bias']},
                    "norm2": {"weight": hf_audio_encoder_module[f'{pfx}.norm2.weight'],
                              "bias": hf_audio_encoder_module[f'{pfx}.norm2.bias']},
                })

            queue_put("audio encoder", {
                "weight": {
                    "encoder": {
                        "encoders0": encoders0,
                        "encoders": encoders,
                        "tp_encoders": tp_encoders,
                        "after_norm": {"weight": hf_audio_encoder_module['encoder.after_norm.weight'],
                                       "bias": hf_audio_encoder_module['encoder.after_norm.bias']},
                        "tp_norm": {"weight": hf_audio_encoder_module['encoder.tp_norm.weight'],
                                    "bias": hf_audio_encoder_module['encoder.tp_norm.bias']},
                    }
                }
            })
        else:
            blocks = []
            for i in range(32):
                pfx = f'audio_encoder.blocks.{i}'
                blocks.append({
                    "attn": {
                        "query": {"weight": hf_audio_encoder_module[f'{pfx}.attn.query.weight'],
                                  "bias": hf_audio_encoder_module[f'{pfx}.attn.query.bias']},
                        "key": {"weight": hf_audio_encoder_module[f'{pfx}.attn.key.weight']},
                        "value": {"weight": hf_audio_encoder_module[f'{pfx}.attn.value.weight'],
                                  "bias": hf_audio_encoder_module[f'{pfx}.attn.value.bias']},
                        "out": {"weight": hf_audio_encoder_module[f'{pfx}.attn.out.weight'],
                                "bias": hf_audio_encoder_module[f'{pfx}.attn.out.bias']},
                        "fsmn_block": {"weight": hf_audio_encoder_module[f'{pfx}.attn.fsmn_block.weight']},
                    },
                    "attn_ln": {"weight": hf_audio_encoder_module[f'{pfx}.attn_ln.weight'],
                                "bias": hf_audio_encoder_module[f'{pfx}.attn_ln.bias']},
                    "mlp": [
                        {"weight": hf_audio_encoder_module[f'{pfx}.mlp.0.weight'],
                         "bias": hf_audio_encoder_module[f'{pfx}.mlp.0.bias']},
                        None,
                        {"weight": hf_audio_encoder_module[f'{pfx}.mlp.2.weight'],
                         "bias": hf_audio_encoder_module[f'{pfx}.mlp.2.bias']},
                    ],
                    "mlp_ln": {"weight": hf_audio_encoder_module[f'{pfx}.mlp_ln.weight'],
                               "bias": hf_audio_encoder_module[f'{pfx}.mlp_ln.bias']},
                })

            queue_put("audio encoder", {
                "weight": {
                    "conv1": {"conv": {
                        "weight": hf_audio_encoder_module['audio_encoder.conv1.weight'],
                        "bias": hf_audio_encoder_module['audio_encoder.conv1.bias'],
                    }},
                    "conv2": {"conv": {
                        "weight": hf_audio_encoder_module['audio_encoder.conv2.weight'],
                        "bias": hf_audio_encoder_module['audio_encoder.conv2.bias'],
                    }},
                    "ln_post": {
                        "weight": hf_audio_encoder_module['audio_encoder.ln_post.weight'],
                        "bias": hf_audio_encoder_module['audio_encoder.ln_post.bias'],
                    },
                    "blocks": blocks,
                }
            })

    # Audio adaptor (HF `audio_adaptor.pt` -> Megatron namespace); must precede final norm (same as mcore loader).
    if random_init_audio_adaptor:
        queue_put("audio adaptor", {"weight": None})
    else:
        put_audio_adaptor_message(queue_put, args.load_dir, args.audio_adaptor_pt_path)

    queue_put("final norm", {
        "weight": hf_model.model.norm.weight.data,
    })

    if md.output_layer:
        queue_put("output layer", {
            "weight": hf_model.lm_head.weight.data
        })

    if args.qwen3_audio_encoder:
        with torch.no_grad():
            for layer_idx in range(args_audio_encoder.num_layers):
                hf_layer = hf_audio_encoder_module.model.layers[layer_idx]
                message = {
                    "input norm weight": hf_layer.input_layernorm.weight.data,
                    "qkv weight": build_qkv_weight_from_hf(args_audio_encoder, hf_layer.self_attn, attn_output_gate=False),
                    "dense weight": hf_layer.self_attn.o_proj.weight.data,
                    "post norm weight": hf_layer.post_attention_layernorm.weight.data,
                    "mlp l0 weight W": hf_layer.mlp.gate_proj.weight.data,
                    "mlp l0 weight V": hf_layer.mlp.up_proj.weight.data,
                    "mlp l1 weight": hf_layer.mlp.down_proj.weight.data,
                }
                if md.add_qkv_bias and hf_layer.self_attn.q_proj.bias is not None:
                    message["qkv bias"] = build_qkv_bias_from_hf(args_audio_encoder, hf_layer.self_attn)
                if md.qk_layernorm:
                    message["q layernorm weight"] = hf_layer.self_attn.q_norm.weight.data
                    message["k layernorm weight"] = hf_layer.self_attn.k_norm.weight.data
                queue_put(f"audio encoder transformer layer {layer_idx}", message)
                del message
                if layer_idx % 4 == 0:
                    gc.collect()

        queue_put("audio encoder final norm", {
            "weight": hf_audio_encoder_module.model.norm.weight.data,
        })

        queue_put("audio encoder conv1 weights", {
            "weight": conv_weights["encoder.conv1.weight"],
            "bias": conv_weights["encoder.conv1.bias"],
        })

        queue_put("audio encoder conv2 weights", {
            "weight": conv_weights["encoder.conv2.weight"],
            "bias": conv_weights["encoder.conv2.bias"],
        })

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
