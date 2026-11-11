# Copyright (c) 2025. Qwen3.5-MoE HF checkpoint loader for Megatron-LM.
"""Load Qwen3.5-MoE HuggingFace checkpoints into Megatron format.

Handles:
- Heterogeneous layers (linear_attention + full_attention)
- Attention output gating (doubled Q projection)
- GatedDeltaNet linear attention weights
- Shared expert with sigmoid gate
- Zero-centered gamma RMSNorm
- Partial rotary embeddings
- Interleaved M-RoPE
"""

import os
import sys
import torch
import transformers
from tqdm import tqdm
import types


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


def load_args_from_checkpoint(args):
    """Read Qwen3.5-MoE config from HF checkpoint and set Megatron args."""
    from transformers import AutoConfig
    full_config = AutoConfig.from_pretrained(args.load)

    # Qwen3.5-MoE uses nested config: text_config contains the LLM config
    if hasattr(full_config, 'text_config'):
        qwen_config = full_config.text_config
    else:
        qwen_config = full_config

    # Basic args
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

    args.untie_embeddings_and_output_weights = not qwen_config.tie_word_embeddings
    args.kv_channels = getattr(qwen_config, "head_dim", None)
    args.max_position_embeddings = qwen_config.max_position_embeddings
    args.hidden_size = qwen_config.hidden_size
    args.num_attention_heads = qwen_config.num_attention_heads
    args.num_layers = qwen_config.num_hidden_layers
    args.norm_epsilon = qwen_config.rms_norm_eps
    args.rotary_base = qwen_config.rope_theta
    args.vocab_size = qwen_config.vocab_size
    args.padded_vocab_size = qwen_config.vocab_size
    args.qwen3_5_config = qwen_config

    # MoE config
    if hasattr(qwen_config, 'num_experts'):
        args.num_experts = qwen_config.num_experts
        args.moe_router_topk = qwen_config.num_experts_per_tok
        args.ffn_hidden_size = qwen_config.moe_intermediate_size
    else:
        args.ffn_hidden_size = qwen_config.intermediate_size

    # GQA
    if qwen_config.num_key_value_heads:
        args.group_query_attention = True
        args.num_query_groups = qwen_config.num_key_value_heads

    # Shared expert
    if hasattr(qwen_config, 'shared_expert_intermediate_size'):
        args.moe_shared_expert_intermediate_size = qwen_config.shared_expert_intermediate_size

    # Qwen3.5-specific
    args.apply_layernorm_1p = True  # zero-centered gamma
    args.attn_output_gate = getattr(qwen_config, 'attn_output_gate', True)
    args.moe_norm_topk_scores = getattr(qwen_config, 'norm_topk_prob', False)
    args.sequence_parallel = True

    # Layer types
    if hasattr(qwen_config, 'layer_types'):
        args.layer_types = qwen_config.layer_types
    else:
        args.layer_types = None

    # Partial rotary
    args.rotary_percent = getattr(qwen_config, 'partial_rotary_factor', 1.0)

    # M-RoPE
    args.mrope_interleaved = getattr(qwen_config, 'mrope_interleaved', False)
    args.mrope_section = getattr(qwen_config, 'mrope_section', None)

    # Full attention interval
    full_attention_interval = getattr(qwen_config, 'full_attention_interval', 0)
    args.full_attention_interval = full_attention_interval

    # Linear attention config
    args.linear_attention_conv_kernel_dim = getattr(qwen_config, 'linear_conv_dim', 4)
    args.linear_attention_key_head_dim = getattr(qwen_config, 'linear_key_head_dim', 128)
    args.linear_attention_value_head_dim = getattr(qwen_config, 'linear_value_head_dim', 128)
    args.linear_attention_num_key_heads = getattr(qwen_config, 'linear_num_key_heads', 16)
    args.linear_attention_num_value_heads = getattr(qwen_config, 'linear_num_value_heads', 64)


def verify_transformers_version():
    version_parts = transformers.__version__.split('.')[:2]
    major, minor = map(int, version_parts)
    assert major >= 4 and minor >= 36


def set_preprocess_state(args, model, hf_model):
    """Set embedding params."""
    model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model):
    """Set output layer & norm params."""
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    if args.untie_embeddings_and_output_weights:
        model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def set_full_attn_state(args, layer, hf_layer):
    """Set full attention params (with output gating support)."""
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    tp = args.tensor_model_parallel_size
    num_heads = args.num_attention_heads // tp
    num_query_groups = (args.num_query_groups if args.group_query_attention else args.num_attention_heads) // tp
    num_querys_per_group = num_heads // num_query_groups
    dim = args.kv_channels

    # Copy QK norm
    if args.qk_layernorm:
        attn.q_layernorm.weight.data.copy_(hf_attn.q_norm.weight)
        attn.k_layernorm.weight.data.copy_(hf_attn.k_norm.weight)

    # For Qwen3.5 with attn_output_gate, q_proj is doubled (query + gate)
    if args.attn_output_gate:
        # HF q_proj: [num_heads * head_dim * 2, hidden_size]
        q_weight = hf_attn.q_proj.weight
        q_size = num_query_groups * num_querys_per_group * dim
        # Split into query and gate
        q_query = q_weight[:q_size, :]
        q_gate = q_weight[q_size:, :]

        # Reshape and interleave: [ng, (np/ng * hn + np/ng * hn + hn + hn), hidden]
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
    attn = layer.self_attention  # This is a GatedDeltaNet module
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


def set_moe_mlp_state(args, layer, hf_layer):
    """Set MOE MLP params."""
    layer.mlp.router.weight.data.copy_(hf_layer.mlp.gate.weight)

    mcore_experts = layer.mlp.experts.local_experts
    hf_experts = hf_layer.mlp.experts
    for expert_idx in range(args.num_experts):
        mcore_experts[expert_idx].linear_fc1.weight.data.copy_(
            torch.cat([
                hf_experts[expert_idx].gate_proj.weight,
                hf_experts[expert_idx].up_proj.weight
            ], dim=0)
        )
        mcore_experts[expert_idx].linear_fc2.weight.data.copy_(
            hf_experts[expert_idx].down_proj.weight
        )

    # Shared expert with sigmoid gate
    if layer.mlp.shared_experts is not None:
        layer.mlp.shared_experts.linear_fc1.weight.data.copy_(
            torch.cat([
                hf_layer.mlp.shared_expert.gate_proj.weight,
                hf_layer.mlp.shared_expert.up_proj.weight
            ], dim=0)
        )
        layer.mlp.shared_experts.linear_fc2.weight.data.copy_(
            hf_layer.mlp.shared_expert.down_proj.weight
        )
        if hasattr(hf_layer.mlp, 'shared_expert_gate'):
            layer.mlp.shared_experts.gate_weight.data.copy_(
                hf_layer.mlp.shared_expert_gate.weight
            )


def set_layer_state(args, model, hf_model, layer_idx):
    """Set transformer layer params."""
    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    # Determine layer type
    layer_type = 'full_attention'
    if args.layer_types is not None and layer_idx < len(args.layer_types):
        layer_type = args.layer_types[layer_idx]

    # Set attention state based on layer type
    if layer_type == 'linear_attention':
        set_linear_attn_state(args, layer, hf_layer)
        # For linear attention layers, input_layernorm is a separate module
        layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)
    else:
        set_full_attn_state(args, layer, hf_layer)
        # For full attention layers with TE, layernorm is fused into linear_qkv
        if hasattr(layer.self_attention.linear_qkv, 'layer_norm_weight'):
            layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(
                hf_layer.input_layernorm.weight
            )
        else:
            layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)

    # Set MLP state
    if args.num_experts:
        set_moe_mlp_state(args, layer, hf_layer)
        layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)
    else:
        layer.mlp.linear_fc1.weight.data.copy_(
            torch.cat([
                hf_layer.mlp.gate_proj.weight,
                hf_layer.mlp.up_proj.weight
            ], dim=0)
        )
        layer.mlp.linear_fc2.weight.data.copy_(hf_layer.mlp.down_proj.weight)
        if hasattr(layer.mlp.linear_fc1, 'layer_norm_weight'):
            layer.mlp.linear_fc1.layer_norm_weight.data.copy_(
                hf_layer.post_attention_layernorm.weight
            )


def load_checkpoint_to_model(args, model_type):
    """Set model params from HF checkpoint."""
    from transformers import AutoModelForCausalLM

    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.load, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )

    # Init Megatron model
    from pretrain_fun_base_channels import model_provider
    model = model_provider(True, True).to(args.params_dtype)

    # Set model state
    set_preprocess_state(args, model, hf_model)
    set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model, hf_model, layer_idx)

    return model


def get_mlp_message_for_moe_model(message, layer, md):
    """Extract MoE MLP weights into message dict."""
    experts = layer.mlp.experts.local_experts
    message["router weight"] = layer.mlp.router.weight.data
    if md.swiglu:
        chunked = [torch.chunk(e.linear_fc1.weight.data, 2, dim=0) for e in experts]
        message["mlp l0 weight W"] = torch.stack([c[0] for c in chunked], dim=0)
        message["mlp l0 weight V"] = torch.stack([c[1] for c in chunked], dim=0)
    else:
        message["mlp l0 weight"] = torch.stack([e.linear_fc1.weight.data for e in experts])
    message["mlp l1 weight"] = torch.stack([e.linear_fc2.weight.data for e in experts], dim=0)

    if layer.mlp.shared_experts is not None:
        if md.swiglu:
            chunked = torch.chunk(layer.mlp.shared_experts.linear_fc1.weight.data, 2, dim=0)
            message["shared mlp l0 weight W"] = chunked[0]
            message["shared mlp l0 weight V"] = chunked[1]
        else:
            message["shared mlp l0 weight"] = layer.mlp.shared_experts.linear_fc1.weight.data
        message["shared mlp l1 weight"] = layer.mlp.shared_experts.linear_fc2.weight.data
        message["shared mlp gate weight"] = layer.mlp.shared_experts.gate_weight.data


def _load_checkpoint(queue, args):
    verify_transformers_version()

    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
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
        print("Unable to import Megatron. Exiting.")
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
                '--no-initialization',
                '--mock-data',
                '--transformer-impl', 'transformer_engine',
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint(margs)

    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size
    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
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
    check_for_arg('params_dtype')
    check_for_arg('swiglu')

    assert args.model_type == 'GPT', 'Qwen3.5-MoE is a GPT-MoE model.'
    margs.model_type = ModelType.encoder_or_decoder

    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
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
    md.attn_output_gate = margs.attn_output_gate
    md.layer_types = margs.layer_types

    if margs.num_experts:
        md.num_experts = margs.num_experts
        md.moe_shared_expert_intermediate_size = getattr(
            margs, 'moe_shared_expert_intermediate_size', None
        )
        md.moe_shared_experts_gate = md.moe_shared_expert_intermediate_size is not None
    else:
        md.num_experts = None

    # Load model
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs, args.model_type)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings
    message = {
        "word embeddings": model.embedding.word_embeddings.weight.data
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.embedding.position_embeddings.weight.data
    queue_put("embeddings", message)

    # Send layers
    for layer_idx in range(margs.num_layers):
        message = {}
        layer = model.decoder.layers[layer_idx]

        # Determine layer type
        layer_type = 'full_attention'
        if md.layer_types is not None and layer_idx < len(md.layer_types):
            layer_type = md.layer_types[layer_idx]

        message["layer_type"] = layer_type

        if layer_type == 'linear_attention':
            # GatedDeltaNet weights
            message["input norm weight"] = layer.input_layernorm.weight.data
            message["linear_attn in_proj_qkv weight"] = layer.self_attention.in_proj_qkv.weight.data
            message["linear_attn in_proj_z weight"] = layer.self_attention.in_proj_z.weight.data
            message["linear_attn in_proj_b weight"] = layer.self_attention.in_proj_b.weight.data
            message["linear_attn in_proj_a weight"] = layer.self_attention.in_proj_a.weight.data
            message["linear_attn conv1d weight"] = layer.self_attention.conv1d.weight.data
            if layer.self_attention.conv1d.bias is not None:
                message["linear_attn conv1d bias"] = layer.self_attention.conv1d.bias.data
            message["linear_attn dt_bias"] = layer.self_attention.dt_bias.data
            message["linear_attn A_log"] = layer.self_attention.A_log.data
            message["linear_attn norm weight"] = layer.self_attention.norm.weight.data
            message["linear_attn out_proj weight"] = layer.self_attention.out_proj.weight.data
        else:
            # Full attention weights
            if hasattr(layer.self_attention.linear_qkv, 'layer_norm_weight'):
                message["input norm weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data
            else:
                message["input norm weight"] = layer.input_layernorm.weight.data

            message["qkv weight"] = layer.self_attention.linear_qkv.weight.data
            if md.add_qkv_bias:
                message["qkv bias"] = layer.self_attention.linear_qkv.bias.data
            message["dense weight"] = layer.self_attention.linear_proj.weight.data

            if md.qk_layernorm:
                message["q layernorm weight"] = layer.self_attention.q_layernorm.weight.data
                message["k layernorm weight"] = layer.self_attention.k_layernorm.weight.data

        # MLP weights (same for both layer types)
        if margs.num_experts:
            message["post norm weight"] = layer.pre_mlp_layernorm.weight.data
            get_mlp_message_for_moe_model(message, layer, md)
        else:
            if hasattr(layer.mlp.linear_fc1, 'layer_norm_weight'):
                message["post norm weight"] = layer.mlp.linear_fc1.layer_norm_weight.data
            else:
                message["post norm weight"] = layer.pre_mlp_layernorm.weight.data

        queue_put(f"transformer layer {layer_idx}", message)

    queue_put("final norm", {
        "weight": model.decoder.final_layernorm.weight.data,
    })

    if md.output_layer:
        queue_put("output layer", {
            "weight": model.output_layer.weight.data
        })
    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
