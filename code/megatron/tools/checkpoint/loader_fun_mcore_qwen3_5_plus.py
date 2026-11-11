# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Support ckpt:torch,torch_dist; model: dense, moe, rm"""

import gc
import json
import os
import sys
import types
import torch
import packaging

from utils import get_mcore_transformer_block_key, print_memory_usage

channels_model = os.environ.get('Channels_Model', 'False')
channels_model = True if channels_model == 'True' else False
if channels_model:
    print ('use channels model')
else:
    print ('use base model')

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
    group.add_argument('--lora-ckpt-dir', type=str, default=None,
                       help='Path to LoRA-only checkpoint directory (trainable params only). '
                       'When set, LoRA deltas are merged into base weights during conversion.')
    group.add_argument('--lora-mode', type=str, default='merge',
                       choices=['merge', 'trainable-only'],
                       help='merge: fold LoRA into base weights; '
                       'trainable-only: save encoder/adaptor/LoRA separately.')
    group.add_argument('--lora-scale', type=float, default=1.0,
                       help='Extra multiplier applied to LoRA deltas during merge. '
                       '1.0 preserves the scale implied by lora_alpha / lora_rank.')
    group.add_argument('--fun-dataset-config-path', type=str, default=None,
                       help='FunASR YAML config used to enable CosyVoiceMixedLMSimpleSysV2 '
                       'codex modules during conversion. If omitted, the value saved in '
                       'checkpoint args is used when available. Can also be set with '
                       'FUN_DATASET_CONFIG_PATH.')


def _load_lora_data(lora_ckpt_dir, num_layers, layer_types):
    """Load LoRA checkpoint and pre-compute full-size weight deltas.

    Supports both torch format (.pt files) and distributed checkpoint format
    (.distcp files).  For dist_ckpt, uses PyTorch DCP ``load_state_dict`` with
    ``no_dist=True`` so that no distributed runtime is required.

    Args:
        lora_ckpt_dir: Path to the LoRA checkpoint directory.
        num_layers: Total number of decoder layers.
        layer_types: Per-layer type list ("full_attention" / "linear_attention").

    Returns:
        Tuple of (lora_deltas, encoder_adaptor_sd, lora_ckpt_args) where:
          - lora_deltas: dict[layer_idx -> {msg_key: delta_tensor}]
          - encoder_adaptor_sd: dict of encoder/adaptor state dict entries
          - lora_ckpt_args: the checkpoint args namespace
    """
    from megatron.training.checkpointing import (
        find_checkpoint_rank_0,
        get_checkpoint_name,
        get_checkpoint_tracker_filename,
        read_metadata,
    )
    from megatron.core.transformer.lora import assemble_lora_deltas

    # ------------------------------------------------------------------ #
    # 1. Read iteration — support both parent dir and direct iter_XXXXXXX
    # ------------------------------------------------------------------ #
    tracker_file = get_checkpoint_tracker_filename(lora_ckpt_dir)
    if os.path.isfile(tracker_file):
        iteration, release = read_metadata(tracker_file)
    else:
        base = os.path.basename(lora_ckpt_dir.rstrip('/'))
        if base.startswith('iter_'):
            iteration = int(base.split('_')[1])
            release = False
            lora_ckpt_dir = os.path.dirname(lora_ckpt_dir.rstrip('/'))
        elif base == 'release':
            iteration = 0
            release = True
            lora_ckpt_dir = os.path.dirname(lora_ckpt_dir.rstrip('/'))
        else:
            raise FileNotFoundError(
                f"Cannot find {tracker_file} and directory name '{base}' "
                f"is not an iter_XXXXXXX format. Please pass the parent "
                f"checkpoint directory or a direct iter_XXXXXXX path.")
    print(f"[LoRA convert] Found LoRA checkpoint at iteration {iteration}, "
          f"dir={lora_ckpt_dir}")

    # ------------------------------------------------------------------ #
    # 2. Locate rank-0 checkpoint and detect format
    # ------------------------------------------------------------------ #
    rank0_path = find_checkpoint_rank_0(lora_ckpt_dir, iteration, release)
    if rank0_path is None:
        raise FileNotFoundError(
            f"Cannot locate checkpoint at {lora_ckpt_dir}, iteration {iteration}")

    is_dist_ckpt = os.path.isdir(rank0_path)

    if is_dist_ckpt:
        # -------------------------------------------------------------- #
        # 2a. Distributed checkpoint format (.distcp)
        # -------------------------------------------------------------- #
        print(f"[LoRA convert] Detected dist_ckpt format at {rank0_path}")

        # Load args from common.pt (plain torch file, no distributed needed)
        common_pt = os.path.join(rank0_path, 'common.pt')
        if not os.path.isfile(common_pt):
            raise FileNotFoundError(
                f"dist_ckpt directory {rank0_path} has no common.pt")
        common_sd = torch.load(common_pt, map_location='cpu', weights_only=False)
        lora_ckpt_args = common_sd.get('args', None)
        if lora_ckpt_args is None:
            raise ValueError(
                f"LoRA dist_ckpt at {rank0_path} has no 'args' in common.pt")

        lora_rank = lora_ckpt_args.lora_rank
        lora_alpha = lora_ckpt_args.lora_alpha
        lora_tp = lora_ckpt_args.tensor_model_parallel_size
        print(f"[LoRA convert] LoRA config: rank={lora_rank}, "
              f"alpha={lora_alpha}, training_tp={lora_tp}")

        # Load all tensor data via PyTorch DCP (no distributed required)
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.metadata import (
            TensorStorageMetadata,
        )
        reader = FileSystemReader(rank0_path)
        metadata = reader.read_metadata()

        # Build only the tensors needed for LoRA merge and encoder/adaptor
        # replacement.  A full training DCP also contains optimizer moments and
        # frozen base-model weights; materializing all of them can consume
        # hundreds of GB and is unnecessary for conversion.
        model_sd = {}
        for key, storage_meta in metadata.state_dict_metadata.items():
            needed = (
                (key.startswith('decoder.') and '.lora_' in key)
                or key.startswith('audio_encoder.')
                or key.startswith('audio_adaptor.')
            )
            if needed and isinstance(storage_meta, TensorStorageMetadata):
                model_sd[key] = torch.empty(
                    storage_meta.size,
                    dtype=storage_meta.properties.dtype,
                )

        dcp.load_state_dict(model_sd, storage_reader=reader, no_dist=True)
        print(f"[LoRA convert] Loaded {len(model_sd)} required LoRA/encoder/"
              f"adaptor tensors from dist_ckpt (full global shapes)")

        # Debug: print first few keys so the user can verify structure
        sample_keys = list(model_sd.keys())[:5]
        print(f"[LoRA convert] Sample keys: {sample_keys}")

        # DCP reconstructs ShardedTensor entries into their global shapes.
        # Correctly-saved LoRA checkpoints shard ColumnParallel lora_B on dim 0
        # and RowParallel lora_A on dim 1, so a single reconstructed state dict
        # already contains the full weight-space delta inputs.  Do not replicate
        # it by TP size; doing so silently duplicates one TP shard into all
        # partitions for legacy checkpoints.
        lora_rank_sds = [model_sd]
        print("[LoRA convert] Using reconstructed global DCP tensors "
              "for delta assembly")

    else:
        # -------------------------------------------------------------- #
        # 2b. Regular torch checkpoint format (.pt per rank)
        # -------------------------------------------------------------- #
        print(f"[LoRA convert] Detected torch format at {rank0_path}")
        rank0_sd = torch.load(rank0_path, map_location='cpu')
        lora_ckpt_args = rank0_sd.get('args', None)
        if lora_ckpt_args is None:
            raise ValueError(
                f"LoRA checkpoint at {rank0_path} has no 'args' entry")

        lora_rank = lora_ckpt_args.lora_rank
        lora_alpha = lora_ckpt_args.lora_alpha
        lora_tp = lora_ckpt_args.tensor_model_parallel_size
        lora_pp = lora_ckpt_args.pipeline_model_parallel_size
        lora_ep = lora_ckpt_args.expert_model_parallel_size
        print(f"[LoRA convert] LoRA config: rank={lora_rank}, "
              f"alpha={lora_alpha}, tp={lora_tp}, pp={lora_pp}, ep={lora_ep}")

        # Load all TP rank state dicts (pp=0, ep=0 — LoRA is not EP-sharded)
        lora_rank_sds = []
        for tp_rank in range(lora_tp):
            ckpt_path = get_checkpoint_name(
                lora_ckpt_dir, iteration, release,
                pipeline_parallel=(lora_pp > 1),
                tensor_rank=tp_rank,
                pipeline_rank=0,
                expert_parallel=(lora_ep > 1),
                expert_rank=0,
            )
            sd = torch.load(ckpt_path, map_location='cpu')
            model_sd = sd.get('model', sd)
            lora_rank_sds.append(model_sd)
            print(f"[LoRA convert] Loaded TP rank {tp_rank}: {ckpt_path} "
                  f"({len(model_sd)} keys)")

    # ------------------------------------------------------------------ #
    # 3. Pre-compute LoRA deltas
    # ------------------------------------------------------------------ #
    lora_deltas = assemble_lora_deltas(
        lora_rank_sds, num_layers, layer_types,
        lora_rank, lora_alpha,
    )
    print(f"[LoRA convert] Assembled deltas for {len(lora_deltas)} layers")

    # ------------------------------------------------------------------ #
    # 4. Extract encoder/adaptor state dict from rank-0 (or single dict)
    # ------------------------------------------------------------------ #
    encoder_adaptor_sd = {}
    for key, val in lora_rank_sds[0].items():
        if key.startswith('audio_encoder.') or key.startswith('audio_adaptor.'):
            encoder_adaptor_sd[key] = val
    print(f"[LoRA convert] Extracted {len(encoder_adaptor_sd)} "
          f"encoder/adaptor keys")

    return lora_deltas, encoder_adaptor_sd, lora_ckpt_args


def _align_lora_delta(delta, target, key, layer_idx):
    """Align a LoRA delta to the extracted base weight shape.

    Some dist-checkpoint LoRA saves expose only the tensor-parallel local
    adapter tensor.  In that case the delta is smaller along exactly one
    dimension.  Pad the missing partition with zeros so conversion can finish
    while keeping the available trained shard instead of duplicating it.
    """
    if delta.shape == target.shape:
        return delta

    if len(delta.shape) != len(target.shape):
        raise RuntimeError(
            f"[LoRA convert] Shape mismatch for layer {layer_idx} {key}: "
            f"delta={tuple(delta.shape)}, target={tuple(target.shape)}")

    mismatch_dims = [
        dim for dim, (delta_size, target_size) in enumerate(zip(delta.shape, target.shape))
        if delta_size != target_size
    ]
    if len(mismatch_dims) != 1:
        raise RuntimeError(
            f"[LoRA convert] Shape mismatch for layer {layer_idx} {key}: "
            f"delta={tuple(delta.shape)}, target={tuple(target.shape)}")

    dim = mismatch_dims[0]
    delta_size = delta.shape[dim]
    target_size = target.shape[dim]
    if delta_size > target_size or target_size % delta_size != 0:
        raise RuntimeError(
            f"[LoRA convert] Cannot align layer {layer_idx} {key}: "
            f"delta={tuple(delta.shape)}, target={tuple(target.shape)}")

    pad_shape = list(delta.shape)
    pad_shape[dim] = target_size - delta_size
    padding = torch.zeros(
        pad_shape,
        dtype=delta.dtype,
        device=delta.device,
    )
    print(
        f"[LoRA convert] WARNING: layer {layer_idx} {key} delta shape "
        f"{tuple(delta.shape)} does not match base shape {tuple(target.shape)}; "
        f"padding missing TP partition with zeros on dim {dim}")
    return torch.cat([delta, padding], dim=dim)


def _load_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    # Dense Qwen3.5 conversion does not use FlashMoE kernels, while Megatron
    # imports the optional package unconditionally during model construction.
    try:
        import flash_moe  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["flash_moe"] = types.ModuleType("flash_moe")

    # pylint: disable=import-outside-toplevel
    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_global_variables
        from megatron.training.checkpointing import load_args_from_checkpoint, load_checkpoint # pylint: disable=redefined-outer-name
        from megatron.core.parallel_state import initialize_model_parallel
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError as e:
        print(f"Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting. Reason: {e}")
        queue.put("exit")
        sys.exit(1)

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--exit-on-missing-checkpoint',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--no-gradient-accumulation-fusion',
                '--use-cpu-initialization',
                '--auto-detect-ckpt-format',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--mock-data', # To pass the "blend data checks" in arguments.py
                '--load', args.load_dir,
                '--position-embedding-type', args.position_embedding_type,
                # audio
                # '--audio-linear-units', '1024',
                # '--audio-attention-heads', '16',
                # '--audio-num-blocks', '12',
                # '--audio-encoder-dim', '1024', # adaptor
                # '--audio-ffn-dim', '2048', # adaptor, 1024 not effect, 2048
                # '--audio-n-layer', '2', # adaptor, 1 not effect, 2
                '--audio-tokenizer-code-book-size', '59049',
                ]

    if channels_model:
        sys.argv.append('--audio-is-dual-channel')
        # 仅当 no_audio_emb_and_head=True 时，允许 checkpoint 中缺少 output_layer_audio 等 channels 组件时部分加载
        no_audio_emb_and_head = os.environ.get('no_audio_emb_and_head', 'False') == 'True'
        if no_audio_emb_and_head:
            sys.argv.extend(['--dist-ckpt-strictness', 'log_unexpected'])
    if args.qwen3_audio_encoder:
        sys.argv.extend(['--audio-encoder-type', 'qwen3-2b', "--audio-encoder-dim", "2048"])
    if args.sensevoicesmall_audio_encoder:
        sys.argv.extend(['--audio-encoder-type', 'sensevoicesmall', "--audio-encoder-dim", "512"])

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

    fun_dataset_config_path = (
        args.fun_dataset_config_path
        or os.environ.get('FUN_DATASET_CONFIG_PATH')
        or getattr(checkpoint_args, 'fun_dataset_config_path', None)
    )
    if fun_dataset_config_path:
        margs.fun_dataset_config_path = fun_dataset_config_path
        print(f"[loader] use fun_dataset_config_path={fun_dataset_config_path}")
    if getattr(checkpoint_args, 'tensor_model_parallel_size', None):
        os.environ["COSYVOICE_CODEX_VOCAB_PADDING_MULTIPLE"] = str(
            checkpoint_args.tensor_model_parallel_size
        )
        print(
            "[loader] use COSYVOICE_CODEX_VOCAB_PADDING_MULTIPLE="
            f"{checkpoint_args.tensor_model_parallel_size}"
        )

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

    margs.moe_norm_topk_scores = checkpoint_args.moe_norm_topk_scores
    # Validate margs.
    margs = validate_args(margs)
    # Weight conversion does not need Apex's fused accumulation extension even
    # when the training checkpoint recorded it as enabled.
    margs.gradient_accumulation_fusion = False

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
        # check_for_arg('moe_shared_expert_intermediate_size')
    print(f"checkpoint_args {checkpoint_args}")
    # Determine how to make our models
    if args.model_type == 'GPT':
        # from pretrain_gpt import model_provider
        from pretrain_fun_base import model_provider
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
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
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
                # 仅当 no_audio_emb_and_head=True 时，允许 checkpoint 缺少 embedding_audio/output_layer_audio 时部分加载
                no_audio_emb_and_head = os.environ.get('no_audio_emb_and_head', 'False') == 'True'
                load_strict = not (channels_model and no_audio_emb_and_head)
                load_checkpoint(model_, None, None, strict=load_strict)

                if consumed_train_samples is not None:
                    assert(margs.consumed_train_samples == consumed_train_samples)
                else:
                    consumed_train_samples = margs.consumed_train_samples
                if consumed_valid_samples is not None:
                    assert(margs.consumed_valid_samples == consumed_valid_samples)
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
        with open(args.vocab_file) as vocab_file_handler: # pylint: disable=unspecified-encoding
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
    md.kv_channels = margs.kv_channels
    md.qk_layernorm = margs.qk_layernorm
    md.moe_norm_topk_scores = margs.moe_norm_topk_scores
    md.layer_types = getattr(margs, 'layer_types', None)
    if md.layer_types is None:
        full_attention_interval = getattr(margs, 'full_attention_interval', None)
        if full_attention_interval is not None and full_attention_interval > 0:
            md.layer_types = [
                "full_attention" if (layer_idx + 1) % full_attention_interval == 0 else "linear_attention"
                for layer_idx in range(margs.num_layers)
            ]
            print(
                "Checkpoint did not provide layer_types; inferred from "
                f"full_attention_interval={full_attention_interval}: {md.layer_types}"
            )
    md.attn_output_gate = getattr(margs, 'attn_output_gate', False)

    # qwen3 audio encoder params
    if args.qwen3_audio_encoder:
        md.audio_encoder_num_layers = 40

    # Get transformer block (named either 'encoder' or 'decoder').
    transformer_block_key = get_mcore_transformer_block_key(md.model_type)
    def get_transformer_block(_model):
        return getattr(_model, transformer_block_key)

    # Get first pipe stage
    mpu.set_pipeline_model_parallel_rank(0)
    # all_models: pp_rank, vp_rank, ep_rank, tp_rank
    all_models = [get_models(tp_size, ep_size, md.params_dtype)]
    models = all_models[0][0]
    if ep_size == 1:
        assert len(models) == 1

    first_model = models[0][0]
    md.has_cosyvoice_codex_head = bool(
        getattr(first_model, "has_cosyvoice_codex_head", False)
        and hasattr(first_model, "cosyvoice_llm")
        and hasattr(first_model, "cosyvoice_codec_embed")
        and hasattr(first_model, "cosyvoice_llm_codec_head")
    )
    md.has_cosyvoice_fm = bool(
        getattr(first_model, "has_cosyvoice_fm", False)
        and hasattr(first_model, "cosyvoice_fm")
    )
    if md.has_cosyvoice_fm:
        # Keep the exact FM architecture beside the exported weights.  This is
        # required now that FM depth/dimensions are no longer globally fixed.
        md.cosyvoice_fm_model_conf = dict(
            getattr(first_model.audio_config, "cosyvoice_fm_model_conf", {})
        )
    if md.has_cosyvoice_codex_head:
        md.cosyvoice_codec_unit = int(getattr(first_model, "cosyvoice_codec_unit"))
        md.cosyvoice_codec_vocab_size = int(getattr(first_model, "cosyvoice_codec_vocab_size"))
        md.cosyvoice_llm_hidden_size = int(getattr(first_model, "cosyvoice_llm_hidden_size"))
        md.cosyvoice_llm_dtype = getattr(first_model, "cosyvoice_llm_dtype", "fp32")
        md.cosyvoice_llm_init_param_path = getattr(
            first_model, "cosyvoice_llm_init_param_path", None
        )
    md.has_embedding_audio = bool(
        channels_model and all(hasattr(model, "embedding_audio") for model in models[0])
    )
    md.has_output_layer_audio = bool(
        channels_model and all(hasattr(model, "output_layer_audio") for model in models[0])
    )

    md.consumed_train_samples = consumed_train_samples
    md.consumed_valid_samples = consumed_valid_samples
    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings
    message = {
        "word embeddings": torch.cat(
            [models[0][tp_rank].embedding.word_embeddings.weight.data for tp_rank in range(tp_size)],
            dim = 0)
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = models[0].embedding.position_embeddings.weight.data
    else:
        assert not hasattr(models[0][0].embedding, 'position_embeddings')

    # sending audio embeddings
    if getattr(md, "has_embedding_audio", False):
        message["embedding audio"] = torch.cat(
            [models[0][tp_rank].embedding_audio.word_embeddings.weight.data for tp_rank in range(tp_size)],
            dim = 0)

    queue_put("embeddings", message)

    def get_message_for_dense_model(message, is_audio_encoder=False):
        # tp is handled in this function
        # Get non-parallel tensors from tp_rank 0
        if not is_audio_encoder:
            layer = get_transformer_block(models[0][0]).layers[layer_num]
        else:
            layer = models[0][0].audio_encoder.encoder.layers[layer_num]
        if hasattr(layer.self_attention.linear_qkv, "layer_norm_weight"):
            message["input norm weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data
            if norm_has_bias:
                message["input norm bias"] = layer.self_attention.linear_qkv.layer_norm_bias.data
        else:
            message["input norm weight"] = layer.input_layernorm.weight.data
            if norm_has_bias:
                message["input norm bias"] = layer.input_layernorm.bias.data
        if hasattr(layer.mlp.linear_fc1, "layer_norm_weight"):
            message["post norm weight"] = layer.mlp.linear_fc1.layer_norm_weight.data
            if norm_has_bias:
                message["post norm bias"] = layer.mlp.linear_fc1.layer_norm_bias.data
        else:
            message["post norm weight"] = layer.pre_mlp_layernorm.weight.data
            if norm_has_bias:
                message["post norm bias"] = layer.pre_mlp_layernorm.bias.data
        if md.linear_bias:
            message["dense bias"] = layer.self_attention.linear_proj.bias.data
        if md.qk_layernorm:
            message["q layernorm weight"] = layer.self_attention.q_layernorm.weight.data
            if norm_has_bias:
                message["q layernorm bias"] = layer.self_attention.q_layernorm.bias.data
            message["k layernorm weight"] = layer.self_attention.k_layernorm.weight.data
            if norm_has_bias:
                message["k layernorm bias"] = layer.self_attention.k_layernorm.bias.data

        # Grab all parallel tensors for this layer
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        mlp_l0_weight = []
        mlp_l0_bias = []
        mlp_l1_weight = []
        for tp_rank, model in enumerate(models[0]):
            if not is_audio_encoder:
                layer = get_transformer_block(model).layers[layer_num]
            else:
                layer = model.audio_encoder.encoder.layers[layer_num]
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
            if not is_audio_encoder:
                layer = get_transformer_block(models[0][0]).layers[layer_num]
            else:
                layer = models[0][0].audio_encoder.encoder.layers[layer_num]
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
                message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias],dim=0)
                message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias],dim=0)
            else:
                message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)
            message["mlp l1 bias"] = mlp_l1_bias
        elif md.add_qkv_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)

    def get_message_for_moe_model(message):
        # Get non-parallel tensors from tp_rank 0
        layer = get_transformer_block(models[0][0]).layers[layer_num]
        if hasattr(layer.self_attention.linear_qkv, 'layer_norm_weight'):
            message["input norm weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data
            if norm_has_bias and hasattr(layer.self_attention.linear_qkv, 'layer_norm_bias'):
                message["input norm bias"] = layer.self_attention.linear_qkv.layer_norm_bias.data
        else:
            message["input norm weight"] = layer.input_layernorm.weight.data
            if norm_has_bias:
                message["input norm bias"] = layer.input_layernorm.bias.data
        message["post norm weight"] = layer.pre_mlp_layernorm.weight.data
        if norm_has_bias:
            message["post norm bias"] = layer.pre_mlp_layernorm.bias.data
        if md.linear_bias:
            message["dense bias"] = layer.self_attention.linear_proj.bias.data
        if md.qk_layernorm:
            message["q layernorm weight"] = layer.self_attention.q_layernorm.weight.data
            if norm_has_bias:
                message["q layernorm bias"] = layer.self_attention.q_layernorm.bias.data
            message["k layernorm weight"] = layer.self_attention.k_layernorm.weight.data
            if norm_has_bias:
                message["k layernorm bias"] = layer.self_attention.k_layernorm.bias.data

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
            if margs.moe_shared_expert_intermediate_size:
                shared_expert_mlp_l0_weight.append(layer.mlp.shared_experts.linear_fc1.weight.data)
                shared_expert_mlp_l1_weight.append(layer.mlp.shared_experts.linear_fc2.weight.data)
                shared_expert_gate_weight = layer.mlp.shared_experts.gate_weight.data

        layer = get_transformer_block(models[0][0]).layers[layer_num]
        router_weight = layer.mlp.router.weight.data

        # MoE modules
        num_experts_per_rank = margs.num_experts // ep_size
        for ep_rank, tp_models in enumerate(models):
            for tp_rank, model in enumerate(tp_models):
                layer = get_transformer_block(model).layers[layer_num]
                for local_expert_idx in range(num_experts_per_rank):
                    expert_idx = int(ep_rank * num_experts_per_rank + local_expert_idx)
                    mlp_l0_weight_list[expert_idx].append(layer.mlp.experts.local_experts[local_expert_idx].linear_fc1.weight.data)
                    mlp_l1_weight_list[expert_idx].append(layer.mlp.experts.local_experts[local_expert_idx].linear_fc2.weight.data)
                    if md.linear_bias:
                        mlp_l0_bias_list[expert_idx].append(layer.mlp.experts.local_experts[local_expert_idx].linear_fc1.bias.data)

            if md.linear_bias:
                # Get non-parallel tensors from tp_rank 0
                layer = get_transformer_block(tp_models[0])
                for local_expert_idx in range(num_experts_per_rank):
                    expert_idx = int(ep_rank * num_experts_per_rank + local_expert_idx)
                    mlp_l1_bias_list[expert_idx].append(layer.mlp.experts.local_experts[local_expert_idx].linear_fc2.bias.data)

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
            if margs.moe_shared_expert_intermediate_size:
                for tp_rank in range(tp_size):
                    shared_expert_mlp_l0_weight[tp_rank] = torch.chunk(shared_expert_mlp_l0_weight[tp_rank], 2, dim=0)
                message["shared mlp l0 weight W"] = torch.cat([w[0] for w in shared_expert_mlp_l0_weight], dim=0)
                message["shared mlp l0 weight V"] = torch.cat([w[1] for w in shared_expert_mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.stack(mlp_l0_weight_list)
            if margs.moe_shared_expert_intermediate_size:
                message["shared mlp l0 weight"] = torch.cat(shared_expert_mlp_l0_weight, dim=0)
        if margs.moe_shared_expert_intermediate_size:
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
        if margs.moe_shared_expert_intermediate_size:
            message["shared gate weight"] = shared_expert_gate_weight

    def get_message_for_linear_attention_dense_model(message):
        """Extract a Qwen3.5 GatedDeltaNet layer with a dense SwiGLU MLP."""
        layer = get_transformer_block(models[0][0]).layers[layer_num]
        attn = layer.self_attention

        message["layer_type"] = "linear_attention"
        message["input norm weight"] = layer.input_layernorm.weight.data
        if hasattr(layer.mlp.linear_fc1, "layer_norm_weight"):
            message["post norm weight"] = layer.mlp.linear_fc1.layer_norm_weight.data
        else:
            message["post norm weight"] = layer.pre_mlp_layernorm.weight.data

        in_proj_qkv_w, in_proj_z_w = [], []
        in_proj_b_w, in_proj_a_w = [], []
        conv1d_w, out_proj_w = [], []
        mlp_l0_weight, mlp_l1_weight = [], []
        for model in models[0]:
            layer_t = get_transformer_block(model).layers[layer_num]
            attn_t = layer_t.self_attention
            in_proj_qkv_w.append(attn_t.in_proj_qkv.weight.data)
            in_proj_z_w.append(attn_t.in_proj_z.weight.data)
            in_proj_b_w.append(attn_t.in_proj_b.weight.data)
            in_proj_a_w.append(attn_t.in_proj_a.weight.data)
            conv1d_w.append(attn_t.conv1d.weight.data)
            out_proj_w.append(attn_t.out_proj.weight.data)
            mlp_l0_weight.append(layer_t.mlp.linear_fc1.weight.data)
            mlp_l1_weight.append(layer_t.mlp.linear_fc2.weight.data)

        message["linear_attn in_proj_qkv weight"] = torch.cat(in_proj_qkv_w, dim=0)
        message["linear_attn in_proj_z weight"] = torch.cat(in_proj_z_w, dim=0)
        message["linear_attn in_proj_b weight"] = torch.cat(in_proj_b_w, dim=0)
        message["linear_attn in_proj_a weight"] = torch.cat(in_proj_a_w, dim=0)
        message["linear_attn conv1d weight"] = torch.cat(conv1d_w, dim=0)
        message["linear_attn out_proj weight"] = torch.cat(out_proj_w, dim=1)
        message["linear_attn dt_bias"] = attn.dt_bias.data
        message["linear_attn A_log"] = attn.A_log.data
        message["linear_attn norm weight"] = attn.norm.weight.data

        if md.swiglu:
            mlp_l0_weight = [torch.chunk(weight, 2, dim=0) for weight in mlp_l0_weight]
            message["mlp l0 weight W"] = torch.cat([weight[0] for weight in mlp_l0_weight], dim=0)
            message["mlp l0 weight V"] = torch.cat([weight[1] for weight in mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)
        message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)

    def get_message_for_linear_attention_moe_model(message):
        """Extract message for linear_attention (GatedDeltaNet) + MoE layers."""
        layer = get_transformer_block(models[0][0]).layers[layer_num]
        attn = layer.self_attention  # GatedDeltaNet

        message["layer_type"] = "linear_attention"
        message["input norm weight"] = layer.input_layernorm.weight.data
        message["post norm weight"] = layer.pre_mlp_layernorm.weight.data

        # GatedDeltaNet: ColumnParallelLinear (concat dim=0), RowParallelLinear (concat dim=1)
        in_proj_qkv_w = []
        in_proj_z_w = []
        in_proj_b_w = []
        in_proj_a_w = []
        conv1d_w = []
        out_proj_w = []
        for tp_rank, model in enumerate(models[0]):
            layer_t = get_transformer_block(model).layers[layer_num]
            attn_t = layer_t.self_attention
            in_proj_qkv_w.append(attn_t.in_proj_qkv.weight.data)
            in_proj_z_w.append(attn_t.in_proj_z.weight.data)
            in_proj_b_w.append(attn_t.in_proj_b.weight.data)
            in_proj_a_w.append(attn_t.in_proj_a.weight.data)
            conv1d_w.append(attn_t.conv1d.weight.data)
            out_proj_w.append(attn_t.out_proj.weight.data)

        message["linear_attn in_proj_qkv weight"] = torch.cat(in_proj_qkv_w, dim=0)
        message["linear_attn in_proj_z weight"] = torch.cat(in_proj_z_w, dim=0)
        message["linear_attn in_proj_b weight"] = torch.cat(in_proj_b_w, dim=0)
        message["linear_attn in_proj_a weight"] = torch.cat(in_proj_a_w, dim=0)
        message["linear_attn conv1d weight"] = torch.cat(conv1d_w, dim=0)
        message["linear_attn out_proj weight"] = torch.cat(out_proj_w, dim=1)
        message["linear_attn dt_bias"] = attn.dt_bias.data
        message["linear_attn A_log"] = attn.A_log.data
        message["linear_attn norm weight"] = attn.norm.weight.data

        # MoE: same as get_message_for_moe_model
        router_weight = layer.mlp.router.weight.data
        shared_expert_mlp_l0_weight = []
        shared_expert_mlp_l1_weight = []
        shared_expert_gate_weight = None
        for tp_rank, model in enumerate(models[0]):
            layer_t = get_transformer_block(model).layers[layer_num]
            if margs.moe_shared_expert_intermediate_size:
                shared_expert_mlp_l0_weight.append(layer_t.mlp.shared_experts.linear_fc1.weight.data)
                shared_expert_mlp_l1_weight.append(layer_t.mlp.shared_experts.linear_fc2.weight.data)
                shared_expert_gate_weight = layer_t.mlp.shared_experts.gate_weight.data

        num_experts_per_rank = margs.num_experts // ep_size
        mlp_l0_weight_list = [[] for _ in range(margs.num_experts)]
        mlp_l1_weight_list = [[] for _ in range(margs.num_experts)]
        for ep_rank, tp_models in enumerate(models):
            for tp_rank, model in enumerate(tp_models):
                layer_t = get_transformer_block(model).layers[layer_num]
                for local_expert_idx in range(num_experts_per_rank):
                    expert_idx = int(ep_rank * num_experts_per_rank + local_expert_idx)
                    mlp_l0_weight_list[expert_idx].append(
                        layer_t.mlp.experts.local_experts[local_expert_idx].linear_fc1.weight.data
                    )
                    mlp_l1_weight_list[expert_idx].append(
                        layer_t.mlp.experts.local_experts[local_expert_idx].linear_fc2.weight.data
                    )

        mlp_l0_weight_w_list = []
        mlp_l0_weight_v_list = []
        for expert_idx in range(margs.num_experts):
            mlp_l0_weight = mlp_l0_weight_list[expert_idx]
            if md.swiglu:
                for tp_rank in range(tp_size):
                    mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
                mlp_l0_weight_w_list.append(torch.cat([w[0] for w in mlp_l0_weight], dim=0))
                mlp_l0_weight_v_list.append(torch.cat([w[1] for w in mlp_l0_weight], dim=0))
            else:
                mlp_l0_weight_list[expert_idx] = torch.cat(mlp_l0_weight, dim=0)
            mlp_l1_weight_list[expert_idx] = torch.cat(mlp_l1_weight_list[expert_idx], dim=1)

        if md.swiglu:
            message["mlp l0 weight W"] = torch.stack(mlp_l0_weight_w_list)
            message["mlp l0 weight V"] = torch.stack(mlp_l0_weight_v_list)
            if margs.moe_shared_expert_intermediate_size:
                for tp_rank in range(tp_size):
                    shared_expert_mlp_l0_weight[tp_rank] = torch.chunk(
                        shared_expert_mlp_l0_weight[tp_rank], 2, dim=0
                    )
                message["shared mlp l0 weight W"] = torch.cat(
                    [w[0] for w in shared_expert_mlp_l0_weight], dim=0
                )
                message["shared mlp l0 weight V"] = torch.cat(
                    [w[1] for w in shared_expert_mlp_l0_weight], dim=0
                )
        else:
            message["mlp l0 weight"] = torch.stack(mlp_l0_weight_list)
            if margs.moe_shared_expert_intermediate_size:
                message["shared mlp l0 weight"] = torch.cat(shared_expert_mlp_l0_weight, dim=0)
        if margs.moe_shared_expert_intermediate_size:
            message["shared mlp l1 weight"] = torch.cat(shared_expert_mlp_l1_weight, dim=1)
        message["mlp l1 weight"] = torch.stack(mlp_l1_weight_list)
        message["router weight"] = router_weight
        if margs.moe_shared_expert_intermediate_size:
            message["shared gate weight"] = shared_expert_gate_weight

    # ---- LoRA: pre-compute deltas and replace encoder/adaptor weights ----
    lora_deltas = None
    lora_scale = 1.0
    if getattr(args, 'lora_ckpt_dir', None) is not None:
        lora_deltas, enc_adp_sd, lora_ckpt_args = _load_lora_data(
            args.lora_ckpt_dir, margs.num_layers, md.layer_types)
        lora_scale = float(getattr(args, 'lora_scale', 1.0))
        print(f"[LoRA convert] Extra LoRA merge scale: {lora_scale}")

        # Replace encoder/adaptor weights in the base model with LoRA-trained versions
        if enc_adp_sd:
            base_model = all_models[0][0][0][0]  # pp=0, vp=0, ep=0, tp=0
            missing, unexpected = base_model.load_state_dict(enc_adp_sd, strict=False)
            print(f"[LoRA convert] Replaced encoder/adaptor: "
                  f"{len(enc_adp_sd)} keys applied, "
                  f"{len(missing)} missing (expected), "
                  f"{len(unexpected)} unexpected")

        md.lora_mode = getattr(args, 'lora_mode', 'merge')
    # ---- end LoRA setup ----

    total_layer_num = 0
    for vp_rank in range(vp_size):
        mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
        for pp_rank in range(pp_size):
            if pp_rank > 0:
                mpu.set_pipeline_model_parallel_rank(pp_rank)
                if vp_rank == 0:
                    all_models.append(get_models(tp_size, ep_size, md.params_dtype))
            models = all_models[pp_rank][vp_rank]
            for layer_num in range(len(get_transformer_block(models[0][0]).layers)):
                message = {}

                # Determine layer type: from md.layer_types or fallback to runtime detection
                if md.layer_types is not None and layer_num < len(md.layer_types):
                    is_linear_attn = md.layer_types[layer_num] == "linear_attention"
                else:
                    layer = get_transformer_block(models[0][0]).layers[layer_num]
                    is_linear_attn = not hasattr(layer.self_attention, "linear_qkv")

                if margs.num_experts:
                    if is_linear_attn:
                        get_message_for_linear_attention_moe_model(message)
                    else:
                        get_message_for_moe_model(message)
                else:
                    if is_linear_attn:
                        get_message_for_linear_attention_dense_model(message)
                    else:
                        get_message_for_dense_model(message)

                # ---- LoRA merge: add pre-computed deltas to extracted weights ----
                if lora_deltas and total_layer_num in lora_deltas:
                    layer_deltas = lora_deltas.pop(total_layer_num)
                    if not is_linear_attn:
                        # Full attention (SelfAttention)
                        qkv_delta = _align_lora_delta(
                            layer_deltas["qkv"], message["qkv weight"],
                            "qkv", total_layer_num)
                        dense_delta = _align_lora_delta(
                            layer_deltas["dense"], message["dense weight"],
                            "dense", total_layer_num)
                        message["qkv weight"] = message["qkv weight"] + lora_scale * qkv_delta
                        message["dense weight"] = message["dense weight"] + lora_scale * dense_delta
                    else:
                        # Linear attention (GatedDeltaNet)
                        in_proj_qkv_delta = _align_lora_delta(
                            layer_deltas["in_proj_qkv"],
                            message["linear_attn in_proj_qkv weight"],
                            "linear_attn in_proj_qkv", total_layer_num)
                        in_proj_z_delta = _align_lora_delta(
                            layer_deltas["in_proj_z"],
                            message["linear_attn in_proj_z weight"],
                            "linear_attn in_proj_z", total_layer_num)
                        out_proj_delta = _align_lora_delta(
                            layer_deltas["out_proj"],
                            message["linear_attn out_proj weight"],
                            "linear_attn out_proj", total_layer_num)
                        message["linear_attn in_proj_qkv weight"] = (
                            message["linear_attn in_proj_qkv weight"] + lora_scale * in_proj_qkv_delta)
                        message["linear_attn in_proj_z weight"] = (
                            message["linear_attn in_proj_z weight"] + lora_scale * in_proj_z_delta)
                        message["linear_attn out_proj weight"] = (
                            message["linear_attn out_proj weight"] + lora_scale * out_proj_delta)
                    print(f"[LoRA convert] Merged LoRA delta into layer {total_layer_num} "
                          f"({'linear_attn' if is_linear_attn else 'full_attn'}, scale={lora_scale})")
                    del layer_deltas
                # ---- end LoRA merge ----

                queue_put(f"transformer layer {total_layer_num}", message)

                # 发送后释放该层内存，避免 loader 持有一份完整模型（大模型 OOM 优化）
                if os.environ.get('LOADER_RELEASE_LAYER_AFTER_SEND', '0') == '1':
                    for ep_idx in range(len(models)):
                        for tp_idx in range(len(models[ep_idx])):
                            layer = get_transformer_block(models[ep_idx][tp_idx]).layers[layer_num]
                            for param in layer.parameters():
                                param.data = torch.empty(0, device=param.device, dtype=param.dtype)
                    del message
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                total_layer_num = total_layer_num + 1

    def get_message_for_audio_encoder(message):
        from types import SimpleNamespace as NS
        
        # Audio encoder only exists in pipeline stage 0
        # Get model from pp_rank=0, vp_rank=0, use first ep_rank and tp_rank
        models_pp0 = all_models[0][0]  # pp_rank=0, vp_rank=0
        
        # Check if audio encoder exists
        if not hasattr(models_pp0[0][0], 'audio_encoder'):
            return
        
        # Use model from ep_rank=0, tp_rank=0 to get audio encoder
        ae = models_pp0[0][0].audio_encoder
        
        # blocks
        blocks = []
        for i in range(len(ae.blocks)):
            blk = NS(
                attn=NS(
                    query=NS(weight=ae.blocks[i].attn.query.weight.data, bias=ae.blocks[i].attn.query.bias.data),
                    key=NS(weight=ae.blocks[i].attn.key.weight.data),
                    value=NS(weight=ae.blocks[i].attn.value.weight.data, bias=ae.blocks[i].attn.value.bias.data),
                    out=NS(weight=ae.blocks[i].attn.out.weight.data, bias=ae.blocks[i].attn.out.bias.data),
                    fsmn_block=NS(weight=ae.blocks[i].attn.fsmn_block.weight.data),
                ),
                attn_ln=NS(weight=ae.blocks[i].attn_ln.weight.data, bias=ae.blocks[i].attn_ln.bias.data),
                mlp=[
                    NS(weight=ae.blocks[i].mlp[0].weight.data, bias=ae.blocks[i].mlp[0].bias.data),
                    None,
                    NS(weight=ae.blocks[i].mlp[2].weight.data, bias=ae.blocks[i].mlp[2].bias.data),
                ],
                mlp_ln=NS(weight=ae.blocks[i].mlp_ln.weight.data, bias=ae.blocks[i].mlp_ln.bias.data),
            )
            blocks.append(blk)
        
        audio_payload = NS(
            conv1=NS(conv=NS(weight=ae.conv1.conv.weight.data, bias=ae.conv1.conv.bias.data)),
            conv2=NS(conv=NS(weight=ae.conv2.conv.weight.data, bias=ae.conv2.conv.bias.data)),
            ln_post=NS(weight=ae.ln_post.weight.data, bias=ae.ln_post.bias.data),
            blocks=blocks,
        )
        
        message["weight"] = audio_payload

    def get_message_for_sensevoicesmall_audio_encoder(message):
        from types import SimpleNamespace as NS
        
        # Audio encoder only exists in pipeline stage 0
        # Get model from pp_rank=0, vp_rank=0, use first ep_rank and tp_rank
        models_pp0 = all_models[0][0]  # pp_rank=0, vp_rank=0
        
        # Check if audio encoder exists
        if not hasattr(models_pp0[0][0], 'audio_encoder'):
            return
        
        # Use model from ep_rank=0, tp_rank=0 to get audio encoder
        ae = models_pp0[0][0].audio_encoder.encoder
        
        # encoders0, 1 block
        encoders0 = []
        for i in range(len(ae.encoders0)):
            blk = NS(
                self_attn=NS(
                    linear_out=NS(weight=ae.encoders0[i].self_attn.linear_out.weight.data, bias=ae.encoders0[i].self_attn.linear_out.bias.data),
                    linear_q_k_v=NS(weight=ae.encoders0[i].self_attn.linear_q_k_v.weight.data, bias=ae.encoders0[i].self_attn.linear_q_k_v.bias.data),
                    fsmn_block=NS(weight=ae.encoders0[i].self_attn.fsmn_block.weight.data),
                ),
                feed_forward=NS(
                    w_1=NS(weight=ae.encoders0[i].feed_forward.w_1.weight.data, bias=ae.encoders0[i].feed_forward.w_1.bias.data),
                    w_2=NS(weight=ae.encoders0[i].feed_forward.w_2.weight.data, bias=ae.encoders0[i].feed_forward.w_2.bias.data),
                ),
                norm1=NS(weight=ae.encoders0[i].norm1.weight.data, bias=ae.encoders0[i].norm1.bias.data),
                norm2=NS(weight=ae.encoders0[i].norm2.weight.data, bias=ae.encoders0[i].norm2.bias.data),
            )
            encoders0.append(blk)
        
        # encoders, 49 blocks
        encoders = []
        for i in range(len(ae.encoders)):
            blk = NS(
                self_attn=NS(
                    linear_out=NS(weight=ae.encoders[i].self_attn.linear_out.weight.data, bias=ae.encoders[i].self_attn.linear_out.bias.data),
                    linear_q_k_v=NS(weight=ae.encoders[i].self_attn.linear_q_k_v.weight.data, bias=ae.encoders[i].self_attn.linear_q_k_v.bias.data),
                    fsmn_block=NS(weight=ae.encoders[i].self_attn.fsmn_block.weight.data),
                ),
                feed_forward=NS(
                    w_1=NS(weight=ae.encoders[i].feed_forward.w_1.weight.data, bias=ae.encoders[i].feed_forward.w_1.bias.data),
                    w_2=NS(weight=ae.encoders[i].feed_forward.w_2.weight.data, bias=ae.encoders[i].feed_forward.w_2.bias.data),
                ),
                norm1=NS(weight=ae.encoders[i].norm1.weight.data, bias=ae.encoders[i].norm1.bias.data),
                norm2=NS(weight=ae.encoders[i].norm2.weight.data, bias=ae.encoders[i].norm2.bias.data),
            )
            encoders.append(blk)
        
        # tp_encoders, 20 blocks
        tp_encoders = []
        for i in range(len(ae.tp_encoders)):
            blk = NS(
                self_attn=NS(
                    linear_out=NS(weight=ae.tp_encoders[i].self_attn.linear_out.weight.data, bias=ae.tp_encoders[i].self_attn.linear_out.bias.data),
                    linear_q_k_v=NS(weight=ae.tp_encoders[i].self_attn.linear_q_k_v.weight.data, bias=ae.tp_encoders[i].self_attn.linear_q_k_v.bias.data),
                    fsmn_block=NS(weight=ae.tp_encoders[i].self_attn.fsmn_block.weight.data),
                ),
                feed_forward=NS(
                    w_1=NS(weight=ae.tp_encoders[i].feed_forward.w_1.weight.data, bias=ae.tp_encoders[i].feed_forward.w_1.bias.data),
                    w_2=NS(weight=ae.tp_encoders[i].feed_forward.w_2.weight.data, bias=ae.tp_encoders[i].feed_forward.w_2.bias.data),
                ),
                norm1=NS(weight=ae.tp_encoders[i].norm1.weight.data, bias=ae.tp_encoders[i].norm1.bias.data),
                norm2=NS(weight=ae.tp_encoders[i].norm2.weight.data, bias=ae.tp_encoders[i].norm2.bias.data),
            )
            tp_encoders.append(blk)
        
        audio_payload = NS(
            encoder=NS(
                encoders0=encoders0,
                encoders=encoders,
                tp_encoders=tp_encoders,
                after_norm=NS(weight=ae.after_norm.weight.data, bias=ae.after_norm.bias.data),
                tp_norm=NS(weight=ae.tp_norm.weight.data, bias=ae.tp_norm.bias.data),
            ),
        )
        
        message["weight"] = audio_payload
        
    
    # Send audio encoder from pipeline stage 0
    message = {}
    if not args.qwen3_audio_encoder:
        if args.sensevoicesmall_audio_encoder:
            get_message_for_sensevoicesmall_audio_encoder(message)
        else:
            get_message_for_audio_encoder(message)
        if "weight" in message:
            queue_put("audio encoder", message)
    else:
        total_layer_num = 0
        # audio encoder is in the first pp layers
        models = all_models[0][0]
        # use model from ep_rank=0, tp_rank=0 to get audio encoder 
        model = models[0][0]
        for layer_num in range(len(model.audio_encoder.encoder.layers)):
            message = {}

            get_message_for_dense_model(message, is_audio_encoder=True)

            queue_put(f"audio encoder transformer layer {total_layer_num}", message)

            total_layer_num = total_layer_num + 1

        # final norm
        message = {
            'weight': model.audio_encoder.encoder.final_layernorm.weight.data
        }
        queue_put("audio encoder final norm", message)

    def get_message_for_audio_adaptor(message):
        from types import SimpleNamespace as NS
        
        # Audio adaptor only exists in pipeline stage 0
        # Get model from pp_rank=0, vp_rank=0, use first ep_rank and tp_rank
        models_pp0 = all_models[0][0]  # pp_rank=0, vp_rank=0
        
        # Check if audio adaptor exists
        if not hasattr(models_pp0[0][0], 'audio_adaptor'):
            return
        
        # Use model from ep_rank=0, tp_rank=0 to get audio adaptor
        aa = models_pp0[0][0].audio_adaptor
        
        # Build blocks
        blocks = []
        if aa.blocks is not None:
            for i in range(len(aa.blocks)):
                blk = NS(
                    self_attn=NS(
                        linear_q=NS(
                            weight=aa.blocks[i].self_attn.w_qs.weight.data,
                            bias=aa.blocks[i].self_attn.w_qs.bias.data
                        ),
                        linear_k=NS(
                            weight=aa.blocks[i].self_attn.w_ks.weight.data,
                            bias=aa.blocks[i].self_attn.w_ks.bias.data
                        ),
                        linear_v=NS(
                            weight=aa.blocks[i].self_attn.w_vs.weight.data,
                            bias=aa.blocks[i].self_attn.w_vs.bias.data
                        ),
                        linear_out=NS(
                            weight=aa.blocks[i].self_attn.linear.weight.data,
                            bias=aa.blocks[i].self_attn.linear.bias.data
                        ),
                    ),
                    feed_forward=NS(
                        w_1=NS(
                            weight=aa.blocks[i].feed_forward.w_1.weight.data,
                            bias=aa.blocks[i].feed_forward.w_1.bias.data
                        ),
                        w_2=NS(
                            weight=aa.blocks[i].feed_forward.w_2.weight.data,
                            bias=aa.blocks[i].feed_forward.w_2.bias.data
                        ),
                    ),
                    norm1=NS(
                        weight=aa.blocks[i].norm1.weight.data,
                        bias=aa.blocks[i].norm1.bias.data
                    ),
                    norm2=NS(
                        weight=aa.blocks[i].norm2.weight.data,
                        bias=aa.blocks[i].norm2.bias.data
                    ),
                )
                blocks.append(blk)
        
        # Build audio adaptor payload
        adaptor_payload = NS(
            linear1=NS(
                weight=aa.linear1.weight.data,
                bias=aa.linear1.bias.data
            ),
            linear2=NS(
                weight=aa.linear2.weight.data,
                bias=aa.linear2.bias.data
            ),
            blocks=blocks,
        )
        
        message["weight"] = adaptor_payload

    # Send audio adaptor from pipeline stage 0
    message = {}
    get_message_for_audio_adaptor(message)
    if "weight" in message:
        queue_put("audio adaptor", message)

    # Send final norm from tp_rank 0
    message = {
        "weight": get_transformer_block(models[0][0]).final_layernorm.weight.data,
    }
    if norm_has_bias:
        message["bias"] = get_transformer_block(models[0][0]).final_layernorm.bias.data
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": torch.cat(
                [models[0][tp_rank].output_layer.weight.data for tp_rank in range(tp_size)],
                dim = 0)
        }
        queue_put("output layer", message)

    if getattr(md, "has_output_layer_audio", False):
        message = {
            "weight": torch.cat(
                [models[0][tp_rank].output_layer_audio.weight.data for tp_rank in range(tp_size)],
                dim = 0)
        }
        queue_put("output layer audio", message)

    if getattr(md, "has_cosyvoice_codex_head", False):
        base_model = models[0][0]

        def _state_dict_cpu(module):
            return {
                key: value.detach().cpu()
                for key, value in module.state_dict().items()
            }

        message = {
            "cosyvoice_llm": _state_dict_cpu(base_model.cosyvoice_llm),
            "cosyvoice_llm_input_proj": (
                _state_dict_cpu(base_model.cosyvoice_llm_input_proj)
                if hasattr(base_model, "cosyvoice_llm_input_proj")
                else {}
            ),
            "cosyvoice_codec_embed": _state_dict_cpu(base_model.cosyvoice_codec_embed),
            "cosyvoice_llm_codec_head": {
                "weight": torch.cat(
                    [
                        models[0][tp_rank].cosyvoice_llm_codec_head.weight.detach().cpu()
                        for tp_rank in range(tp_size)
                    ],
                    dim=0,
                )
            },
            "codec_unit": md.cosyvoice_codec_unit,
            "codec_vocab_size": md.cosyvoice_codec_vocab_size,
            "llm_hidden_size": md.cosyvoice_llm_hidden_size,
            "llm_dtype": md.cosyvoice_llm_dtype,
            "llm_init_param_path": md.cosyvoice_llm_init_param_path,
        }
        queue_put("cosyvoice codex", message)

    if getattr(md, "has_cosyvoice_fm", False):
        # FM is large enough that multiprocessing.Queue shared-storage file
        # descriptors can be exhausted before the saver drains them. Export it
        # directly from the loader process, then send only a completion marker.
        fm_state_dict = _state_dict_cpu(models[0][0].cosyvoice_fm)
        fm_model_conf = getattr(md, "cosyvoice_fm_model_conf", {})
        if not fm_model_conf:
            raise RuntimeError("CosyVoice FM export requires model_conf")
        fm_dir = os.path.join(args.save_dir, "cosyvoice_fm")
        os.makedirs(fm_dir, exist_ok=True)
        torch.save(fm_state_dict, os.path.join(fm_dir, "model.pt"))
        with open(os.path.join(fm_dir, "config.yaml"), "w") as stream:
            json.dump(
                {"model": "CosyVoiceFlowMatching", "model_conf": fm_model_conf},
                stream,
                ensure_ascii=False,
                indent=2,
            )
            stream.write("\n")
        del fm_state_dict
        queue_put("cosyvoice fm", {"exported_by_loader": True})

    if args.qwen3_audio_encoder:
        # audio encoder conv weight and bias
        message = {
            "weight": models[0][0].audio_encoder.conv1.weight.data,
            "bias": models[0][0].audio_encoder.conv1.bias.data
        }
        queue_put("audio encoder conv1 weights", message)
        message = {
            "weight": models[0][0].audio_encoder.conv2.weight.data,
            "bias": models[0][0].audio_encoder.conv2.bias.data
        }
        queue_put("audio encoder conv2 weights", message)

    if md.model_type == 'Reward':
        message = {
            "dense1.weight": models[0][0].pooler_head.dense1.weight.data,
            "dense1.bias": models[0][0].pooler_head.dense1.bias.data,
            "dense2.weight": models[0][0].pooler_head.dense2.weight.data,
            "dense2.bias": models[0][0].pooler_head.dense2.bias.data
        }
        queue_put("pooler head", message)

    # Send BERT lm head and binary head if it exists
    if md.model_type == 'BERT':
        message = {
            "weight": models[0][0].pooler.dense.weight.data,
            "bias": models[0][0].pooler.dense.bias.data
        }
        queue_put("pooler", message)

        message = {
            "dense weight": models[0][0].lm_head.dense.weight.data,
            "dense bias": models[0][0].lm_head.dense.bias.data,
            "norm weight": models[0][0].lm_head.layer_norm.weight.data,
        }
        if norm_has_bias:
            message["norm bias"] = models[0][0].lm_head.layer_norm.bias.data
        queue_put("lm head", message)

        if md.bert_binary_head:
            message = {
                "weight": models[0][0].binary_head.weight.data,
                "bias": models[0][0].binary_head.bias.data
            }
            queue_put("binary head", message)
    queue.put("done")

def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
