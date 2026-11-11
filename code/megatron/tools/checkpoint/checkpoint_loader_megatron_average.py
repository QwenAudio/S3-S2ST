# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Support ckpt:torch,torch_dist; model: dense, moe, rm"""

import json
import os
import sys
import types
import torch
import packaging
import io
import re
import pickle
from collections import OrderedDict
from megatron.training import get_args
from utils import get_mcore_transformer_block_key, print_memory_usage


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                            'trim padding from the embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--position-embedding-type',
                       type=str,
                       default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Position embedding type.')
    group.add_argument('--loader-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--average-steps', type=str, default=None,
                       help="specify the step numbers, for example \"1000,2000,3000,4000\" indicates averaging these four models.")


def _copy_bias(_msg, key, _module):
    _msg[key] = (
        _module.bias.data
        if getattr(_module, 'bias', None) is not None
        else None
    )


def _copy_concat_bias(_msg, key, bias_list):
    if any(_bias is None for _bias in bias_list):
        assert all(_bias is None for _bias in bias_list)
        _msg[key] = None
    else:
        _msg[key] = torch.cat(bias_list, dim=0)


def _load_checkpoint(queue, args):
    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    # pylint: disable=import-outside-toplevel
    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_global_variables
        from megatron.training.checkpointing import load_args_from_checkpoint, \
            load_checkpoint  # pylint: disable=redefined-outer-name
        from megatron.core.parallel_state import initialize_model_parallel
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        sys.exit(1)

    def is_picklable(obj):
        """Check if an object is picklable."""
        try:
            pickle.dumps(obj)
            return True
        except (pickle.PicklingError, AttributeError, TypeError, RuntimeError):
            return False

    def _average(arr):
        # print(arr[0])
        if isinstance(arr[0], list):
            return [_average(x) for x in arr]
        elif isinstance(arr[0], dict) or isinstance(arr[0], OrderedDict):
            return {k: _average([item[k] for item in arr]) for k in arr[0].keys()}
        elif hasattr(arr[0], "state_dict"):
            import copy
            res = arr[0]
            # new_state_dict = OrderedDict()
            # for k in arr[0].state_dict():
            #     if k.endswith('_extra_state'):
            #         new_state_dict[k] = arr[0].state_dict()[0]
            #     else:
            #         new_state_dict[k] = _average([x.state_dict()[k] for x in arr])
            res.load_state_dict(_average([x.state_dict() for x in arr]))
            return res
        elif torch.is_tensor(arr[0]):
            return torch.mean(torch.stack(arr, dim=0), dim=0, dtype=torch.float32).to(dtype=arr[0].dtype)
        elif isinstance(arr[0], io.BytesIO):

            return arr[0]
        else:
            return sum(arr) / len(arr)

    def _average_of_all_models(func, input_models=None):
        nonlocal all_models
        if input_models is None:
            arr = [func(models) for models in all_models]
        else:
            arr = [func(models) for models in input_models]
        return _average(arr)

    def get_all_models(*func_args):
        nonlocal consumed_train_samples
        all_models = []
        steps = [int(s.strip()) for s in args.average_steps.split(",")]
        for i in steps:
            get_args().load = re.sub(r"iter_\d{7}", f"iter_{i:07}", args.load_dir)
            consumed_train_samples = None
            all_models.append([get_models(*func_args)])
        return all_models

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--exit-on-missing-checkpoint',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--auto-detect-ckpt-format',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--mock-data',  # To pass the "blend data checks" in arguments.py
                '--load', args.load_dir,
                '--position-embedding-type', args.position_embedding_type,
                '--num-experts', '0',
                ]

    margs = parse_args()

    device_count = torch.cuda.device_count()
    if device_count > 0:
        torch.cuda.set_device(0)
        device_id = torch.device(f'cuda:0')
    else:
        device_id = None

    # Call the init process
    init_process_group_kwargs = {
        'backend': 'nccl',
        'world_size': 1,
        'rank': 0,
    }

    if packaging.version.Version(torch.__version__) >= packaging.version.Version("2.3.0"):
        init_process_group_kwargs['device_id'] = device_id

    margs, checkpoint_args = load_args_from_checkpoint(margs)

    # for now, if load dist ckpt, we load it as tp1pp1ep1vp1 for convenience
    if checkpoint_args.use_dist_ckpt:
        margs.tensor_model_parallel_size = 1
        margs.pipeline_model_parallel_size = 1
        margs.expert_model_parallel_size = 1
        margs.virtual_pipeline_model_parallel_size = 1
        torch.distributed.init_process_group(**init_process_group_kwargs)
        initialize_model_parallel()
        print(f"real initializing distributed")
    else:
        print(f"fake initializing distributed")
        margs.tensor_model_parallel_size = checkpoint_args.tensor_model_parallel_size
        margs.pipeline_model_parallel_size = checkpoint_args.pipeline_model_parallel_size
        margs.expert_model_parallel_size = checkpoint_args.expert_model_parallel_size
        margs.virtual_pipeline_model_parallel_size = checkpoint_args.virtual_pipeline_model_parallel_size
        margs.sequence_parallel = checkpoint_args.sequence_parallel
        margs.ckpt_format = checkpoint_args.ckpt_format

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size * margs.expert_model_parallel_size

    # Explicitly copy data types from checkpoint.
    margs.fp16 = checkpoint_args.fp16
    margs.bf16 = checkpoint_args.bf16

    margs.use_legacy_models = False
    margs.transformer_impl = args.loader_transformer_impl
    margs.norm_epsilon = checkpoint_args.norm_epsilon
    margs.rotary_base = checkpoint_args.rotary_base
    if checkpoint_args.num_experts:
        margs.moe_shared_expert_intermediate_size = checkpoint_args.moe_shared_expert_intermediate_size
        margs.num_experts = checkpoint_args.num_experts
        margs.moe_router_topk = checkpoint_args.moe_router_topk
    if hasattr(checkpoint_args, 'diff_attn'):
        margs.diff_attn = checkpoint_args.diff_attn
        margs.use_vectorized_lambda = checkpoint_args.use_vectorized_lambda
        margs.diff_lambda_init = checkpoint_args.diff_lambda_init

    # Validate margs.
    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                sys.exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('expert_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)
    if checkpoint_args.num_experts:
        check_for_arg('num_experts')
        check_for_arg('moe_shared_expert_intermediate_size')
    print(f"checkpoint_args {checkpoint_args}")
    # Determine how to make our models
    if args.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'Reward':
        from megatron.core.models.gpt.gpt_reward_model import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    # supress warning about torch.distributed not being initialized
    module.MegatronModule.embedding_warning_printed = True

    consumed_train_samples = None
    consumed_valid_samples = None

    def get_models(tp_size, ep_size, dtype):
        nonlocal consumed_train_samples
        nonlocal consumed_valid_samples
        model_array_len = margs.virtual_pipeline_model_parallel_size
        if model_array_len is None:
            model_array_len = 1
        models = [[[] for _ in range(ep_size)] for _ in range(model_array_len)]

        for ep_rank in range(ep_size):
            mpu.set_expert_model_parallel_rank(ep_rank)
            for tp_rank in range(tp_size):
                mpu.set_tensor_model_parallel_rank(tp_rank)
                if margs.virtual_pipeline_model_parallel_size is not None:
                    model_ = []
                    for i in range(margs.virtual_pipeline_model_parallel_size):
                        mpu.set_virtual_pipeline_model_parallel_rank(i)
                        # Set pre_process and post_process only after virtual rank is set.
                        pre_process = mpu.is_pipeline_first_stage()
                        post_process = mpu.is_pipeline_last_stage()
                        this_model = model_provider(
                            pre_process=pre_process,
                            post_process=post_process
                        ).to(dtype)
                        model_.append(this_model)
                else:
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    model_ = [model_provider(pre_process, post_process).to(dtype)]
                margs.consumed_train_samples = 0
                margs.consumed_valid_samples = 0
                margs.exit_on_missing_checkpoint = True
                load_checkpoint(model_, None, None, strict=False)

                if consumed_train_samples is not None:
                    assert (margs.consumed_train_samples == consumed_train_samples)
                else:
                    consumed_train_samples = margs.consumed_train_samples
                if consumed_valid_samples is not None:
                    assert (margs.consumed_valid_samples == consumed_valid_samples)
                else:
                    consumed_valid_samples = margs.consumed_valid_samples
                for vp_rank in range(model_array_len):
                    models[vp_rank][ep_rank].append(model_[vp_rank])

                # Print memory usage.
                print_memory_usage("loader", tp_rank, tp_size)

        return models

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    fused_kernels.load(margs)
    print(f"loader's margs {margs}")
    # Get true (non-padded) vocab size
    if args.true_vocab_size is not None:
        true_vocab_size = args.true_vocab_size
    elif args.vocab_file is not None:
        with open(args.vocab_file) as vocab_file_handler:  # pylint: disable=unspecified-encoding
            vocab = json.load(vocab_file_handler)
        true_vocab_size = len(vocab)
        if args.true_vocab_size is not None and true_vocab_size != args.true_vocab_size:
            print("Both --true-vocab-size and --vocab-file specified and the vocab size does not match, aborting.")
            queue.put("exit")
            sys.exit(1)
    else:
        true_vocab_size = None

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    print(tp_size, pp_size, ep_size, vp_size)
    # Layernorm has bias; RMSNorm does not.
    if hasattr(checkpoint_args, 'normalization'):
        norm_has_bias = checkpoint_args.normalization == "LayerNorm"
    else:
        # older models only supported LayerNorm
        norm_has_bias = True

    # metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.ffn_hidden_size = margs.ffn_hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.untie_embeddings_and_output_weights = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.add_qkv_bias = margs.add_qkv_bias
    md.norm_has_bias = norm_has_bias
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.previous_expert_parallel_size = margs.expert_model_parallel_size
    md.true_vocab_size = true_vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = checkpoint_args
    md.use_legacy_models = margs.use_legacy_models
    md.num_query_groups = margs.num_query_groups
    md.group_query_attention = margs.group_query_attention
    md.layernorm_epsilon = margs.norm_epsilon
    md.rotary_base = margs.rotary_base
    md.padded_vocab_size = margs.padded_vocab_size
    md.num_experts = margs.num_experts
    if md.num_experts is None:
        md.num_experts = 0
    md.moe_router_topk = margs.moe_router_topk
    md.moe_shared_expert_intermediate_size = margs.moe_shared_expert_intermediate_size
    # Get transformer block (named either 'encoder' or 'decoder').
    transformer_block_key = get_mcore_transformer_block_key(md.model_type)

    def get_transformer_block(_model):
        # print(_model)
        return getattr(_model, transformer_block_key)

    # Get first pipe stage
    mpu.set_pipeline_model_parallel_rank(0)
    # all_models: iter, pp_rank, vp_rank, ep_rank, tp_rank
    all_models = get_all_models(tp_size, ep_size, md.params_dtype)
    # if ep_size == 1:
    #     assert len(models) == 1

    md.consumed_train_samples = consumed_train_samples
    md.consumed_valid_samples = consumed_valid_samples
    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings
    message = {
        "word embeddings": _average_of_all_models(lambda models: torch.cat(
            [models[0][0][0][tp_rank].embedding.word_embeddings.weight.data for tp_rank in range(tp_size)],
            dim=0))
    }
    # if md.position_embedding_type == 'learned_absolute':
    #     message["position embeddings"] = _average_of_all_models(
    #         lambda models: models[0].embedding.position_embeddings.weight.data)
    # else:
    #     assert not hasattr(models[0][0].embedding, 'position_embeddings')

    queue_put("embeddings", message)

    def get_message_for_dense_model(message, models):
        # Get non-parallel tensors from tp_rank 0
        layer = _average_of_all_models(lambda m: get_transformer_block(m[0][0]).layers[layer_num], models)

        message["input norm weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data
        if norm_has_bias:
            message["input norm bias"] = layer.self_attention.linear_qkv.layer_norm_bias.data
        message["post norm weight"] = layer.mlp.linear_fc1.layer_norm_weight.data
        if norm_has_bias:
            message["post norm bias"] = layer.mlp.linear_fc1.layer_norm_bias.data
        if md.linear_bias:
            message["dense bias"] = layer.self_attention.linear_proj.bias.data
        if md.diff_attn:
            if md.use_vectorized_lambda:
                message['lambda_k1'] = layer.self_attention.lambda_k1
                message['lambda_q1'] = layer.self_attention.lambda_q1
                message['lambda_k2'] = layer.self_attention.lambda_k2
                message['lambda_q2'] = layer.self_attention.lambda_q2
            else:
                message['lambda_1'] = layer.self_attention.lambda_1
                message['lambda_2'] = layer.self_attention.lambda_2
        # Grab all parallel tensors for this layer
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        mlp_l0_weight = []
        mlp_l0_bias = []
        mlp_l1_weight = []

        for tp_rank, model in enumerate(models[0]):
            # layer = get_transformer_block(model).layers[layer_num]
            qkv_weight.append(layer.self_attention.linear_qkv.weight.data)
            dense_weight.append(layer.self_attention.linear_proj.weight.data)
            mlp_l0_weight.append(layer.mlp.linear_fc1.weight.data)
            mlp_l1_weight.append(layer.mlp.linear_fc2.weight.data)
            if md.linear_bias:
                qkv_bias.append(layer.self_attention.linear_qkv.bias.data)
                mlp_l0_bias.append(layer.mlp.linear_fc1.bias.data)
            elif md.add_qkv_bias:
                qkv_bias.append(layer.self_attention.linear_qkv.bias.data)

        if md.linear_bias:
            # Get non-parallel tensors from tp_rank 0
            # layer = get_transformer_block(models[0][0]).layers[layer_num]
            mlp_l1_bias = layer.mlp.linear_fc2.bias.data

        # Handle gated linear units
        if md.swiglu:
            # concat all the first halves ('W's) and all the second halves ('V's)
            for tp_rank in range(tp_size):
                mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
            message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
            message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

        # simple concat of the rest
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
        if md.linear_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)
            if md.swiglu:
                for tp_rank in range(tp_size):
                    mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
                message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
                message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
            else:
                message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)
            message["mlp l1 bias"] = mlp_l1_bias
        elif md.add_qkv_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)

    def get_message_for_moe_model(message, models):
        # Get non-parallel tensors from tp_rank 0
        layer = _average_of_all_models(lambda models: get_transformer_block(models[0][0]).layers[layer_num], models)
        message["input norm weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data
        if norm_has_bias:
            message["input norm bias"] = layer.self_attention.linear_qkv.layer_norm_bias.data
        message["post norm weight"] = layer.pre_mlp_layernorm.weight.data
        if norm_has_bias:
            message["post norm bias"] = layer.pre_mlp_layernorm.bias.data
        if md.linear_bias:
            message["dense bias"] = layer.self_attention.linear_proj.bias.data

        if md.diff_attn:
            if md.use_vectorized_lambda:
                message['lambda_k1'] = layer.self_attention.lambda_k1
                message['lambda_q1'] = layer.self_attention.lambda_q1
                message['lambda_k2'] = layer.self_attention.lambda_k2
                message['lambda_q2'] = layer.self_attention.lambda_q2
            else:
                message['lambda_1'] = layer.self_attention.lambda_1
                message['lambda_2'] = layer.self_attention.lambda_2
        # Grab all parallel tensors for this layer
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        shared_expert_mlp_l0_weight = []
        shared_expert_mlp_l1_weight = []
        mlp_l0_weight_list = [[] for _ in range(margs.num_experts)]
        mlp_l0_bias_list = [[] for _ in range(margs.num_experts)]
        mlp_l1_weight_list = [[] for _ in range(margs.num_experts)]
        mlp_l1_bias_list = [[] for _ in range(margs.num_experts)]

        # Dense modules
        for tp_rank, model in enumerate(models[0]):
            layer = get_transformer_block(model).layers[layer_num]
            qkv_weight.append(layer.self_attention.linear_qkv.weight.data)
            dense_weight.append(layer.self_attention.linear_proj.weight.data)
            if md.linear_bias:
                qkv_bias.append(layer.self_attention.linear_qkv.bias.data)
            elif md.add_qkv_bias:
                qkv_bias.append(layer.self_attention.linear_qkv.bias.data)
            shared_expert_mlp_l0_weight.append(layer.mlp.shared_experts.linear_fc1.weight.data)
            shared_expert_mlp_l1_weight.append(layer.mlp.shared_experts.linear_fc2.weight.data)

        layer = get_transformer_block(models[0][0]).layers[layer_num]
        router_weight = layer.mlp.router.weight.data
        shared_expert_gate_weight = layer.mlp.shared_experts.gate_weight.data

        # MoE modules
        num_experts_per_rank = margs.num_experts // ep_size
        for ep_rank, tp_models in enumerate(models):
            for tp_rank, model in enumerate(tp_models):
                layer = get_transformer_block(model).layers[layer_num]
                for local_expert_idx in range(num_experts_per_rank):
                    expert_idx = int(ep_rank * num_experts_per_rank + local_expert_idx)
                    mlp_l0_weight_list[expert_idx].append(
                        layer.mlp.experts.local_experts[local_expert_idx].linear_fc1.weight.data)
                    mlp_l1_weight_list[expert_idx].append(
                        layer.mlp.experts.local_experts[local_expert_idx].linear_fc2.weight.data)
                    if md.linear_bias:
                        mlp_l0_bias_list[expert_idx].append(
                            layer.mlp.experts.local_experts[local_expert_idx].linear_fc1.bias.data)

            if md.linear_bias:
                # Get non-parallel tensors from tp_rank 0
                layer = get_transformer_block(tp_models[0])
                for local_expert_idx in range(num_experts_per_rank):
                    expert_idx = int(ep_rank * num_experts_per_rank + local_expert_idx)
                    mlp_l1_bias_list[expert_idx].append(
                        layer.mlp.experts.local_experts[local_expert_idx].linear_fc2.bias.data)

        mlp_l0_weight_w_list = [[] for _ in range(margs.num_experts)]
        mlp_l0_weight_v_list = [[] for _ in range(margs.num_experts)]
        # Concat along the tensor parallel dimension
        for expert_idx in range(margs.num_experts):
            mlp_l0_weight = mlp_l0_weight_list[expert_idx]
            if md.swiglu:
                for tp_rank in range(tp_size):
                    mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
                mlp_l0_weight_w_list[expert_idx] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
                mlp_l0_weight_v_list[expert_idx] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
            else:
                mlp_l0_weight_list[expert_idx] = torch.cat(mlp_l0_weight, dim=0)
            mlp_l1_weight_list[expert_idx] = torch.cat(mlp_l1_weight_list[expert_idx], dim=1)

        # Stack along the expert parallel dimension
        if md.swiglu:
            message["mlp l0 weight W"] = torch.stack(mlp_l0_weight_w_list)
            message["mlp l0 weight V"] = torch.stack(mlp_l0_weight_v_list)
            for tp_rank in range(tp_size):
                shared_expert_mlp_l0_weight[tp_rank] = torch.chunk(shared_expert_mlp_l0_weight[tp_rank], 2, dim=0)
            message["shared mlp l0 weight W"] = torch.cat([w[0] for w in shared_expert_mlp_l0_weight], dim=0)
            message["shared mlp l0 weight V"] = torch.cat([w[1] for w in shared_expert_mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.stack(mlp_l0_weight_list)
            message["shared mlp l0 weight"] = torch.cat(shared_expert_mlp_l0_weight, dim=0)
        message["shared mlp l1 weight"] = torch.cat(shared_expert_mlp_l1_weight, dim=1)
        message["mlp l1 weight"] = torch.stack(mlp_l1_weight_list)

        # Concat along TP and stack along EP to biases
        if md.linear_bias:
            mlp_l0_bias_w_list = [[] for _ in range(margs.num_experts)]
            mlp_l0_bias_v_list = [[] for _ in range(margs.num_experts)]
            # Concat along the tensor parallel dimension
            for expert_idx in range(margs.num_experts):
                mlp_l0_bias = mlp_l0_bias_list[expert_idx]
                if md.swiglu:
                    for tp_rank in range(tp_size):
                        mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
                    mlp_l0_bias_w_list[expert_idx] = torch.cat([w[0] for w in mlp_l0_bias], dim=0)
                    mlp_l0_bias_v_list[expert_idx] = torch.cat([w[1] for w in mlp_l0_bias], dim=0)
                else:
                    mlp_l0_bias_list[expert_idx] = torch.cat(mlp_l0_bias, dim=0)
                assert len(mlp_l1_bias_list[expert_idx]) == 1
                mlp_l1_bias_list[expert_idx] = mlp_l1_bias_list[expert_idx][0]

            # Stack along the expert parallel dimension
            if md.swiglu:
                message["mlp l0 bias W"] = torch.stack(mlp_l0_bias_w_list)
                message["mlp l0 bias V"] = torch.stack(mlp_l0_bias_v_list)
            else:
                message["mlp l0 bias"] = torch.stack(mlp_l0_bias_list)
            message["mlp l1 bias"] = torch.stack(mlp_l1_bias_list)

        # Simple concat of the rest
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        if md.linear_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)
        elif md.add_qkv_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)

        # Do nothing to router
        message["router weight"] = router_weight
        message["shared gate weight"] = shared_expert_gate_weight


    total_layer_num = 0
    for vp_rank in range(vp_size):
        mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
        for pp_rank in range(pp_size):
            if pp_rank > 0:
                mpu.set_pipeline_model_parallel_rank(pp_rank)
                if vp_rank == 0:
                    tmp_all_models = get_all_models(tp_size, ep_size, md.params_dtype)
                    for i in range(len(all_models)):
                        all_models[i].append(tmp_all_models[i])

            models = [all_models[j][pp_rank][vp_rank] for j in range(len(all_models))]
            # print(models) # num_ckpts, ep, tp
            for layer_num in range(len(get_transformer_block(models[0][0][0]).layers)):
                message = {}

                if margs.num_experts:
                    get_message_for_moe_model(message, models)
                else:
                    get_message_for_dense_model(message, models)

                queue_put(f"transformer layer {total_layer_num}", message)
                if layer_num == 0:
                    print(message)
                    aaa
                total_layer_num = total_layer_num + 1

    # Send final norm from tp_rank 0
    message = {
        "weight": _average_of_all_models(
            lambda models: get_transformer_block(models[0][0]).final_layernorm.weight.data, models),
    }
    if norm_has_bias:
        message["bias"] = _average_of_all_models(
            lambda models: get_transformer_block(models[0][0]).final_layernorm.bias.data, models)
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": _average_of_all_models(lambda models: torch.cat(
                [models[0][tp_rank].output_layer.weight.data for tp_rank in range(tp_size)],
                dim=0), models)
        }
        queue_put("output layer", message)

    if md.model_type == 'Reward':
        message = {
            "dense1.weight": _average_of_all_models(lambda models: models[0][0].pooler_head.dense1.weight.data),
            "dense1.bias": _average_of_all_models(lambda models: models[0][0].pooler_head.dense1.bias.data),
            "dense2.weight": _average_of_all_models(lambda models: models[0][0].pooler_head.dense2.weight.data),
            "dense2.bias": _average_of_all_models(lambda models: models[0][0].pooler_head.dense2.bias.data)
        }
        queue_put("pooler head", message)

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
