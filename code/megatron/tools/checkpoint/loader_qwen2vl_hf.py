import os
import sys
import torch
import transformers
from tqdm import tqdm
import types

from einops import rearrange


def add_arguments(parser):
    group = parser.add_argument_group(title='Qwen2VL HF loader.')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                            'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')


def load_args_from_checkpoint(args):
    # Read Qwen2 args.
    from transformers import AutoConfig
    qwen2vl_config = AutoConfig.from_pretrained(args.load)

    # Update Megatron args.
    args.seq_length = 4096
    args.global_batch_size = 1024
    args.iteration = 1  # '0', 'release' don't work
    args.add_position_embedding = True
    args.use_rotary_position_embeddings = True
    args.swiglu = True
    args.bf16 = True
    args.add_bias_linear = False
    args.normalization = "RMSNorm"
    args.tokenizer_type = "HuggingFaceTokenizer"
    args.disable_bias_linear = True
    args.add_qkv_bias = True

    args.untie_embeddings_and_output_weights = not qwen2vl_config.tie_word_embeddings
    args.kv_channels = getattr(qwen2vl_config, "head_dim", None)
    args.max_position_embeddings = qwen2vl_config.max_position_embeddings
    args.hidden_size = qwen2vl_config.hidden_size
    args.num_attention_heads = qwen2vl_config.num_attention_heads
    args.num_layers = qwen2vl_config.num_hidden_layers
    args.norm_epsilon = qwen2vl_config.rms_norm_eps
    args.rotary_base = qwen2vl_config.rope_theta
    args.vocab_size = qwen2vl_config.vocab_size
    args.padded_vocab_size = qwen2vl_config.vocab_size
    args.qwen2 = qwen2vl_config
    if "num_experts" in qwen2vl_config.__dict__.keys():
        args.num_experts = qwen2vl_config.num_experts
        args.moe_router_topk = qwen2vl_config.num_experts_per_tok
        args.ffn_hidden_size = qwen2vl_config.moe_intermediate_size
        if qwen2vl_config.shared_expert_intermediate_size:
            args.moe_shared_expert_intermediate_size = qwen2vl_config.shared_expert_intermediate_size
    else:
        args.ffn_hidden_size = qwen2vl_config.intermediate_size
    args.sequence_parallel = True

    if qwen2vl_config.num_key_value_heads:
        args.group_query_attention = True
        args.num_query_groups = qwen2vl_config.num_key_value_heads


def verify_transformers_version():
    version_parts = transformers.__version__.split('.')[:2]
    major, minor = map(int, version_parts)
    assert major >= 4 and minor >= 36


def set_vision_model_attn_state(args, layer, hf_blocks, vision_num_attention_heads: int):
    '''Set self-attention params.'''

    # Get attention layer & state.
    layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(hf_blocks.norm1.weight)
    layer.self_attention.linear_qkv.layer_norm_bias.data.copy_(hf_blocks.norm1.bias)
    layer.self_attention.linear_qkv.weight.data.copy_(
        rearrange(hf_blocks.attn.qkv.weight, '(n h d) in -> (h n d) in', n=3, h=vision_num_attention_heads)
    )
    layer.self_attention.linear_qkv.bias.data.copy_(
        rearrange(hf_blocks.attn.qkv.bias, '(n h d) -> (h n d)', n=3, h=vision_num_attention_heads)
    )
    layer.self_attention.linear_proj.weight.data.copy_(hf_blocks.attn.proj.weight)
    layer.self_attention.linear_proj.bias.data.copy_(hf_blocks.attn.proj.bias)


def set_vision_model_mlp_state(args, layer, hf_blocks):
    layer.mlp.linear_fc1.layer_norm_weight.data.copy_(hf_blocks.norm2.weight)
    layer.mlp.linear_fc1.layer_norm_bias.data.copy_(hf_blocks.norm2.bias)

    layer.mlp.linear_fc1.weight.data.copy_(hf_blocks.mlp.fc1.weight)
    layer.mlp.linear_fc1.bias.data.copy_(hf_blocks.mlp.fc1.bias)
    layer.mlp.linear_fc2.weight.data.copy_(hf_blocks.mlp.fc2.weight)
    layer.mlp.linear_fc2.bias.data.copy_(hf_blocks.mlp.fc2.bias)


def set_attn_state(args, layer, hf_layer):
    '''Set self-attention params.'''

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    num_heads = args.num_attention_heads // tp
    num_query_groups = (args.num_query_groups if args.group_query_attention else args.num_attention_heads) // tp
    num_querys_per_group = num_heads // num_query_groups
    dim = args.kv_channels
    assert num_heads % num_querys_per_group == 0

    # Copy weights (re-order dimensions for Megatron).
    attn.linear_qkv.weight.data.copy_(torch.cat([
        hf_attn.q_proj.weight.reshape((num_query_groups, num_querys_per_group * dim, -1)),
        hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
        hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size)))
    attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)

    # Copy bias
    attn.linear_qkv.bias.data.copy_(torch.cat([
        hf_attn.q_proj.bias.reshape((num_query_groups, num_querys_per_group * dim, -1)),
        hf_attn.k_proj.bias.reshape((num_query_groups, dim, -1)),
        hf_attn.v_proj.bias.reshape((num_query_groups, dim, -1)),
    ], dim=1).reshape(num_query_groups * (num_querys_per_group + 2) * dim))

    # Copy input_layernorm
    layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(hf_layer.input_layernorm.weight)


def set_mlp_state(args, layer, hf_layer):
    '''Set MLP params.'''
    layer.mlp.linear_fc1.weight.data.copy_(
        torch.cat([
            hf_layer.mlp.gate_proj.weight,
            hf_layer.mlp.up_proj.weight
        ], dim=0)
    )
    layer.mlp.linear_fc2.weight.data.copy_(
        hf_layer.mlp.down_proj.weight
    )
    layer.mlp.linear_fc1.layer_norm_weight.data.copy_(hf_layer.post_attention_layernorm.weight)


def set_moe_mlp_state(args, layer, hf_layer):
    '''Set MOE MLP params.'''
    layer.mlp.router.weight.data.copy_(hf_layer.mlp.gate.weight)
    layer.mlp.shared_experts.gate_weight.data.copy_(hf_layer.mlp.shared_expert_gate.weight)
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

    # shared exp
    # TODO: lack of shared_exp shared_expert_gate
    layer.mlp.shared_experts.linear_fc1.weight.data.copy_(
        torch.cat([
            hf_layer.mlp.shared_expert.gate_proj.weight,
            hf_layer.mlp.shared_expert.up_proj.weight
        ], dim=0)
    )
    layer.mlp.shared_experts.linear_fc2.weight.data.copy_(
        hf_layer.mlp.shared_expert.down_proj.weight
    )


def set_vision_model_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.vision_model.decoder.layers[layer_idx]
    hf_blocks = hf_model.visual.blocks[layer_idx]

    vision_num_attention_heads = model.vision_model.config.num_attention_heads
    set_vision_model_attn_state(args, layer, hf_blocks, vision_num_attention_heads=vision_num_attention_heads)
    set_vision_model_mlp_state(args, layer, hf_blocks)


def set_lm_model_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.lm_model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)


def load_checkpoint_to_model(args, model_type):
    '''Load model.'''

    # Load Huggingface model.
    from transformers import AutoModelForImageTextToText
    hf_model = AutoModelForImageTextToText.from_pretrained(args.load, torch_dtype=torch.bfloat16, device_map="cpu")

    # Init Megatron model.
    from sft_qwenvl import model_provider
    model = model_provider(True, True).to(args.params_dtype)

    '''Set vision_model patch_embed params.'''
    model.vision_model.patch_embed.proj.weight.data.copy_(
        hf_model.visual.patch_embed.proj.weight)

    '''Set vision_model layers params.'''
    for layer_idx in tqdm(range(len(model.vision_model.decoder.layers)), "set vision_model layer states"):
        set_vision_model_layer_state(args, model, hf_model, layer_idx)

    '''Set vision_model final_layernorm params.'''
    model.vision_model.decoder.final_layernorm.weight.data.copy_(
        hf_model.visual.merger.ln_q.weight)
    model.vision_model.decoder.final_layernorm.bias.data.copy_(
        hf_model.visual.merger.ln_q.bias)

    '''Set vision_model projection params.'''
    model.vision_model.projection.encoder.linear_fc1.weight.data.copy_(
        hf_model.visual.merger.mlp[0].weight)
    model.vision_model.projection.encoder.linear_fc1.bias.data.copy_(
        hf_model.visual.merger.mlp[0].bias)
    model.vision_model.projection.encoder.linear_fc2.weight.data.copy_(
        hf_model.visual.merger.mlp[2].weight)
    model.vision_model.projection.encoder.linear_fc2.bias.data.copy_(
        hf_model.visual.merger.mlp[2].bias)

    # Set lm_model state.
    '''Set lm_model embedding params.'''
    model.lm_model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)
    '''Set lm_model layers params.'''
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_lm_model_layer_state(args, model, hf_model, layer_idx)
    '''Set output layer & norm params.'''
    model.lm_model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    if args.untie_embeddings_and_output_weights:
        model.lm_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)

    return model


def get_mlp_message_for_dense_model(message, layer, md):
    if md.swiglu:
        chunked_mlp_l0_weight = torch.chunk(layer.mlp.linear_fc1.weight.data, 2, dim=0)
        message["mlp l0 weight W"] = chunked_mlp_l0_weight[0]
        message["mlp l0 weight V"] = chunked_mlp_l0_weight[1]
    else:
        message["mlp l0 weight"] = layer.mlp.linear_fc1.weight.data
    message["mlp l1 weight"] = layer.mlp.linear_fc2.weight.data

def get_vision_model_mlp_message_for_dense_model(message, layer, md):
    if md.swiglu:
        chunked_mlp_l0_weight = torch.chunk(layer.mlp.linear_fc1.weight.data, 2, dim=0)
        message["mlp l0 weight W"] = chunked_mlp_l0_weight[0]
        message["mlp l0 weight V"] = chunked_mlp_l0_weight[1]
    else:
        message["mlp l0 weight"] = layer.mlp.linear_fc1.weight.data
    message["mlp l1 weight"] = layer.mlp.linear_fc2.weight.data

def get_mlp_message_for_moe_model(message, layer, md):
    experts = layer.mlp.experts.local_experts
    message["router weight"] = layer.mlp.router.weight.data
    if md.swiglu:
        chunked_mlp_l0_weight = [torch.chunk(local_expert.linear_fc1.weight.data, 2, dim=0) for local_expert in experts]
        message["mlp l0 weight W"] = torch.stack([local_weight[0] for local_weight in chunked_mlp_l0_weight], dim=0)
        message["mlp l0 weight V"] = torch.stack([local_weight[1] for local_weight in chunked_mlp_l0_weight], dim=0)
    else:
        message["mlp l0 weight"] = torch.stack([local_expert.linear_fc1.weight.data for local_expert in experts])
    message["mlp l1 weight"] = torch.stack([local_expert.linear_fc2.weight.data for local_expert in experts], dim=0)

    # shared exp
    if md.swiglu:
        chunked_mlp_l0_weight = torch.chunk(layer.mlp.shared_experts.linear_fc1.weight.data, 2, dim=0)
        message["shared mlp l0 weight W"] = chunked_mlp_l0_weight[0]
        message["shared mlp l0 weight V"] = chunked_mlp_l0_weight[1]
    else:
        message["shared mlp l0 weight"] = layer.mlp.shared_experts.linear_fc1.weight.data
    message["shared mlp l1 weight"] = layer.mlp.shared_experts.linear_fc2.weight.data
    message["shared mlp gate weight"] = layer.mlp.shared_experts.gate_weight.data


def _load_checkpoint(queue, args):
    # Llama-2 requires HF transformers >=4.31.0.
    verify_transformers_version()

    # Search in directory above this.
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
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
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
                '--mock-data',  # To pass the "blend data checks" in arguments.py
                '--transformer-impl', 'transformer_engine',
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
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

    margs.model_type = ModelType.encoder_or_decoder

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
    fused_kernels.load(margs)

    # Metadata.
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
    md.true_vocab_size = margs.vocab_size  # skips padding in saver
    md.make_vocab_size_divisible_by = None
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.qkv_bias = margs.add_qkv_bias
    if margs.num_experts:
        md.num_experts = margs.num_experts
        md.moe_shared_expert_intermediate_size = margs.moe_shared_expert_intermediate_size
        md.moe_shared_experts_gate = True

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs, args.model_type)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {
        "word embeddings": model.lm_model.embedding.word_embeddings.weight.data
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.lm_model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(model.lm_model.embedding, 'position_embeddings')

    queue_put("embeddings", message)

    for layer_idx in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.lm_model.decoder.layers[layer_idx]
        message["input norm weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data

        # Simple concat of the rest.
        message["qkv weight"] = layer.self_attention.linear_qkv.weight.data
        message["qkv bias"] = layer.self_attention.linear_qkv.bias.data
        message["dense weight"] = layer.self_attention.linear_proj.weight.data

        # Grab all parallel tensors for this layer.
        if margs.num_experts:
            message["post norm weight"] = layer.pre_mlp_layernorm.weight.data
            get_mlp_message_for_moe_model(message, layer, md)
        else:
            message["post norm weight"] = layer.mlp.linear_fc1.layer_norm_weight.data
            get_mlp_message_for_dense_model(message, layer, md)

        queue_put(f"transformer layer {layer_idx}", message)

    queue_put("final norm", {
        "weight": model.lm_model.decoder.final_layernorm.weight.data,
    })

    if md.output_layer:
        queue_put("output layer", {
            "weight": model.lm_model.output_layer.weight.data
        })

    queue_put("vision_model patch_embed", {
        "patch_embed": model.vision_model.patch_embed.proj.weight.data,
    })

    queue_put("vision_model final_layernorm", {
        "weight": model.vision_model.decoder.final_layernorm.weight.data,
        "bias": model.vision_model.decoder.final_layernorm.bias.data,
    })

    queue_put("vision_model projection", {
        "l0 weight": model.vision_model.projection.encoder.linear_fc1.weight.data,
        "l0 bias": model.vision_model.projection.encoder.linear_fc1.bias.data,
        "l1 weight": model.vision_model.projection.encoder.linear_fc2.weight.data,
        "l1 bias": model.vision_model.projection.encoder.linear_fc2.bias.data,
    })

    for layer_idx in range(len(model.vision_model.decoder.layers)):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.vision_model.decoder.layers[layer_idx]

        message["attn norm weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data
        message["attn norm bias"] = layer.self_attention.linear_qkv.layer_norm_bias.data
        message["attn qkv weight"] = layer.self_attention.linear_qkv.weight.data
        message["attn qkv bias"] = layer.self_attention.linear_qkv.bias.data
        message["attn proj weight"] = layer.self_attention.linear_proj.weight.data
        message["attn proj bias"] = layer.self_attention.linear_proj.bias.data

        message["mlp l0 norm weight"] = layer.mlp.linear_fc1.layer_norm_weight.data
        message["mlp l0 norm bias"] = layer.mlp.linear_fc1.layer_norm_bias.data
        get_vision_model_mlp_message_for_dense_model(message, layer, md)
        message["mlp l0 bias"] = layer.mlp.linear_fc1.bias.data
        message["mlp l1 bias"] = layer.mlp.linear_fc2.bias.data

        queue_put(f"vision layer {layer_idx}", message)

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
