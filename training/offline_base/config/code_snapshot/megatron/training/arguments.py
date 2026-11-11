# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron arguments."""

import argparse
import dataclasses
import json
import logging
import os
import types

import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.retro.utils import get_config_path as get_retro_config_path
from megatron.core.models.retro.utils import get_gpt_data_dir as get_retro_data_dir
from megatron.core.transformer import MLATransformerConfig, TransformerConfig, DiffTransformerConfig, VLVisionConfig
from megatron.core import parallel_state
from megatron.training.activations import squared_relu
from megatron.training.utils import update_use_dist_ckpt


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_autoresume_args(parser)
    parser = _add_biencoder_args(parser)
    parser = _add_vision_args(parser)
    parser = _add_moe_args(parser)
    parser = _add_mla_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_straggler_detector_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_transformer_engine_args(parser)
    parser = _add_retro_args(parser)
    parser = _add_experimental_args(parser)
    parser = _add_one_logger_args(parser)
    parser = _add_ft_package_args(parser)
    parser = _add_config_logger_args(parser)
    parser = _add_dpo_args(parser)
    parser = _add_code_args(parser)
    parser = _add_qwenvl_args(parser)
    parser = _add_fun_base_args(parser)
    parser = _add_lora_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Experimental yaml
    if args.yaml_cfg is not None:
        from .yaml_arguments import load_yaml
        assert args.yaml_cfg and not args.use_legacy_models, \
            "Yaml config is not supported with legacy models."
        args = load_yaml(args.yaml_cfg)


    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    return args


def load_retro_config(retro_project_dir):
    '''Load Retro's config.json.'''

    # Retro config path.
    retro_config_path = get_retro_config_path(retro_project_dir)
    assert os.path.exists(retro_config_path), \
        "Retro project dir missing config.json."

    # Load retro config.
    with open(retro_config_path) as f:
        retro_config = types.SimpleNamespace(**json.load(f))

    return retro_config


def load_retro_args(args):
    """Load predefined args from Retro config (if applicable).

    When using Retro (or GPT for comparison purposes), data arguments are
    overridden by the saved config.json within the Retro project directory. This
    is to ensure that the data used for pretraining is consistent with the data
    that was preprocessed using the Retro preprocessing pipeline (see
    `tools/retro/preprocess_data.py`).
    """

    # Return if no project directory is specified.
    if args.retro_project_dir is None:
        return

    # Load retro config.
    retro_config = load_retro_config(args.retro_project_dir)

    # Retro data path is relative to project dir (via hard or soft links).
    data_dir = get_retro_data_dir(args.retro_project_dir)
    data_path = list(retro_config.retro_gpt_data_path)
    if len(data_path) % 2 == 0:
        for i in range(len(data_path) - 1, -1, -2):
            data_path[i] = os.path.join(data_dir, data_path[i])
    else:
        assert len(data_path) == 1
        data_path[0] = os.path.join(data_dir, data_path[0])

    # Update args.
    args.data_cache_path = retro_config.retro_gpt_data_cache_path
    args.data_path = data_path if args.data_path is None else args.data_path
    args.eval_interval = retro_config.retro_gpt_eval_interval
    args.eval_iters = retro_config.retro_gpt_eval_iters
    args.global_batch_size = retro_config.retro_gpt_global_batch_size
    args.max_position_embeddings = retro_config.retro_gpt_seq_length
    args.merge_file = os.path.join(
        args.retro_project_dir,
        retro_config.retro_gpt_merge_file,
    ) if retro_config.retro_gpt_merge_file is not None else None
    args.seed = retro_config.retro_gpt_seed
    args.seq_length = retro_config.retro_gpt_seq_length
    args.tokenizer_model = os.path.join(
        args.retro_project_dir,
        retro_config.retro_gpt_tokenizer_model,
    ) if retro_config.retro_gpt_tokenizer_model is not None else None
    args.tokenizer_type = retro_config.retro_gpt_tokenizer_type
    args.train_samples = retro_config.retro_gpt_train_samples
    args.vocab_file = os.path.join(
        args.retro_project_dir,
        retro_config.retro_gpt_vocab_file,
    ) if retro_config.retro_gpt_vocab_file is not None else None

    # Retro-specific args.
    args.retro_block_size = retro_config.retro_block_size
    args.retro_chunk_length = retro_config.retro_gpt_chunk_length
    args.retro_neighbor_dirs = retro_config.retro_neighbor_dirs
    args.retro_split_preprocessing = retro_config.retro_gpt_split
    args.retro_bert_tokenizer_type = retro_config.retro_bert_tokenizer_type
    args.retro_bert_vocab_file = retro_config.retro_bert_vocab_file


def validate_args(args, defaults={}):

    # Temporary
    assert args.non_persistent_ckpt_type in ['global', None], \
        'Currently only global checkpoints are supported'

    # Load saved args from Retro (if applicable).
    load_retro_args(args)

    # Set args.use_dist_ckpt from args.ckpt_format.
    update_use_dist_ckpt(args)

    if args.encoder_tensor_model_parallel_size > 0:
        assert args.encoder_pipeline_model_parallel_size > 0, "encoder_pipeline_model_parallel_size must be defined."
        assert args.num_attention_heads % args.encoder_tensor_model_parallel_size == 0
        assert args.encoder_tensor_model_parallel_size <= args.tensor_model_parallel_size, "We do not support encoders with more TP than the decoder."

    if args.encoder_pipeline_model_parallel_size > 0 and args.encoder_tensor_model_parallel_size == 0:
        args.encoder_tensor_model_parallel_size = args.tensor_model_parallel_size

    encoder_model_size = args.encoder_tensor_model_parallel_size * args.encoder_pipeline_model_parallel_size * args.context_parallel_size
    decoder_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    total_model_size = encoder_model_size + decoder_model_size

    # Total model size.
    assert args.world_size % total_model_size == 0, (
         f"world size ({args.world_size}) is not divisible by total_model_size ({encoder_model_size} + {decoder_model_size})"
    )

    # Pipeline model parallel size.
    args.transformer_pipeline_model_parallel_size = (
        args.pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.pipeline_model_parallel_size
    )

    args.data_parallel_size = args.world_size // total_model_size

    args.context_parallel_expand_kv_heads = False
    if args.context_parallel_strategy == 'ulysses':
        assert (args.num_attention_heads // args.tensor_model_parallel_size) % \
            args.context_parallel_size == 0, f'Ulysses: num_attention_heads = {args.num_attention_heads} ' \
                f'must be divisible by TPxCP {args.tensor_model_parallel_size * args.context_parallel_size}'
        if (args.group_query_attention and ((args.num_query_groups // \
            args.tensor_model_parallel_size) % args.context_parallel_size != 0)):
            args.context_parallel_expand_kv_heads = True
            if args.rank == 0:
                print(f'Ulysses for GQA: kv heads = {args.num_query_groups} cannot be divided ' \
                    f'by TPxCP = {args.tensor_model_parallel_size * args.context_parallel_size}, ' \
                    f'so set context_parallel_expand_kv_heads to True...')

    # Checks.
    if args.rank == 0:
        print('using world size: {}, data-parallel size: {}, '
              'context-parallel size: {}, '
              'tensor-model-parallel size: {}, '
              'encoder-tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {}, '
              'encoder-pipeline-model-parallel size: {}'.format(
                  args.world_size, args.data_parallel_size,
                  args.context_parallel_size,
                  args.tensor_model_parallel_size,
                  args.encoder_tensor_model_parallel_size,
                  args.pipeline_model_parallel_size,
                  args.encoder_pipeline_model_parallel_size), flush=True)

    # backwards compatibility.
    if args.pipeline_model_parallel_split_rank is not None:
        args.encoder_pipeline_model_parallel_size = args.pipeline_model_parallel_split_rank
        args.pipeline_model_parallel_size -= args.encoder_pipeline_model_parallel_size
        assert args.pipeline_model_parallel_size > 0

    if args.expert_tensor_parallel_size is None:
        args.expert_tensor_parallel_size = args.tensor_model_parallel_size

    if args.tp_comm_overlap:
        assert args.sequence_parallel == True, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'

    # Deprecated arguments
    assert args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead'
    del args.batch_size
    assert args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead'
    del args.warmup
    assert args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead'
    del args.model_parallel_size

    if args.checkpoint_activations:
        if args.rank == 0:
            print('--checkpoint-activations is no longer valid, use --recompute-activations, '
                  'or, for more control, --recompute-granularity and --recompute-method.')
        exit()
    del args.checkpoint_activations

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    if args.data_path is not None and args.split is None:
        legacy_default_split_value = '969, 30, 1'
        if args.rank == 0:
            print('WARNING: Please specify --split when using --data-path. Using legacy default value '
                  f'of "{legacy_default_split_value}"')
        args.split = legacy_default_split_value

    if args.num_layers is not None:
        assert args.encoder_num_layers is None, \
            'cannot have both num-layers and encoder-num-layers specified'
        args.encoder_num_layers = args.num_layers
    else:
        assert args.encoder_num_layers is not None, \
            'either num-layers or encoder-num-layers should be specified'
        args.num_layers = args.encoder_num_layers

    if args.allow_padding_num_layers:
        assert not args.standalone_embedding_stage, "not support standalone embedding stage if allow_padding_num_layers is true"
        assert not args.decoder_first_pipeline_num_layers, "not support decoder first pipeline stage if allow_padding_num_layers is true"
        assert not args.decoder_last_pipeline_num_layers, "not support decoder last pipeline stage if allow_padding_num_layers is true"
        assert not args.pipeline_stages_num_layers_override, "not support pipeline stages num layers override if allow_padding_num_layers is true"
        if args.pipeline_layers is not None:
            args.pipeline_layers = [int(layers)for layers in args.pipeline_layers.split(',')]
            pipeline_layers = args.pipeline_layers
            assert len(pipeline_layers) == args.pipeline_model_parallel_size
            assert sum(pipeline_layers) == args.num_layers, \
                'the sum of pipeline_layers must equal --num-layers'
            num_layers_with_padding = max(pipeline_layers) * args.pipeline_model_parallel_size
        else:
            print('WARNING: expect use --pipeline-layers to specify num layers of each pipeline stage, \
                otherwise will use default num layers settings of uneven (v)pp')
            # pad num_layers to make num_layers % pipeline_model_parallel_size == 0
            remainder = args.num_layers % args.pipeline_model_parallel_size
            num_layers_with_padding = args.num_layers if not remainder > 0 else \
                                     (args.num_layers - remainder + args.pipeline_model_parallel_size)
    else:
        num_layers_with_padding = args.num_layers

    if args.pipeline_model_parallel_size > 1 and args.pipeline_stages_num_layers_override is not None:
        assert len(args.pipeline_stages_num_layers_override) == args.pipeline_model_parallel_size, \
            'the number of pipeline stages num layers override must be equal to the pipeline model parallel size'

    args.num_layers_with_padding = num_layers_with_padding
    args.num_layers_without_padding = args.num_layers

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0
    if args.num_layers_per_virtual_pipeline_stage is not None:
        if args.overlap_p2p_comm:
            assert args.pipeline_model_parallel_size > 1, \
                'when interleaved schedule is used, pipeline-model-parallel size '\
                'should be greater than 1'
        else:
            assert args.pipeline_model_parallel_size > 2, \
                'when interleaved schedule is used and p2p communication overlap is disabled, '\
                'pipeline-model-parallel size should be greater than 2 to avoid having multiple '\
                'p2p sends and recvs between same 2 ranks per communication batch'
        if args.allow_padding_num_layers:
            assert args.num_layers_with_padding % args.transformer_pipeline_model_parallel_size == 0, \
                'number of layers should be divisible by the pipeline parallel size'
            num_layers_per_pipeline_stage = args.num_layers_with_padding // args.transformer_pipeline_model_parallel_size
        else:
            assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                'number of layers should be divisible by the pipeline parallel size'
            num_layers_per_pipeline_stage = args.num_layers // args.transformer_pipeline_model_parallel_size
        assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
            'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
        args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
            args.num_layers_per_virtual_pipeline_stage

        # # if using vpp, buffer mechanism should be carefully implemented
        # assert not args.moe_router_buffer_tokens_per_expert, \
        #     '--moe-router-buffer-tokens-per-expert does not support interleaved schedule'
    else:
        args.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False
        args.align_param_gather = False
        # Only print warning if PP size > 1.
        if args.rank == 0 and args.pipeline_model_parallel_size > 1:
            print('WARNING: Setting args.overlap_p2p_comm and args.align_param_gather to False '
                  'since non-interleaved schedule does not support overlapping p2p communication '
                  'and aligned param AG')

    if args.overlap_param_gather:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather only supported with distributed optimizer'
        assert args.overlap_grad_reduce, \
            'Must use --overlap-param-gather with --overlap-grad-reduce'
        assert not args.use_legacy_models, \
            '--overlap-param-gather only supported with MCore models'

    if args.overlap_param_gather_with_optimizer_step:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather-with-optimizer-step only supported with distributed optimizer'
        assert args.overlap_param_gather, \
            'Must use --overlap-param-gather-with-optimizer-step with --overlap-param-gather'
        assert args.virtual_pipeline_model_parallel_size is not None, \
            '--overlap-param-gather-with-optimizer-step only supported with interleaved pipeline parallelism'
        assert not args.use_dist_ckpt, \
            '--overlap-param-gather-with-optimizer-step not supported with distributed checkpointing yet'

    if args.fp8_param_gather:
        assert args.use_distributed_optimizer, \
            '--fp8-param-gather only supported with distributed optimizer'

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:
            args.check_for_nan_in_loss_and_grad = False
            if args.rank == 0:
                print('WARNING: Setting args.check_for_nan_in_loss_and_grad to False since '
                      'dynamic loss scaling is being used')
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # data
    assert args.num_dataset_builder_threads > 0

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    args.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                     'max_position_embeddings']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.ffn_hidden_size is None:
        if hasattr(args, 'ffn_multi') and args.ffn_multi is not None:
            ffn_multi = args.ffn_multi
        else:
            ffn_multi = 4 * 2 / 3 if args.swiglu else 4

        args.ffn_hidden_size = (
            int(ffn_multi * args.hidden_size) * 2
            if args.swiglu
            else ffn_multi * args.hidden_size
        )
        if args.make_ffn_dim_multiple_of:
            args.ffn_hidden_size = args.make_ffn_dim_multiple_of * \
                                   ((args.ffn_hidden_size + args.make_ffn_dim_multiple_of - 1)
                                    // args.make_ffn_dim_multiple_of)
    if args.kv_channels is None:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None and args.context_parallel_size > 1:
        assert args.seq_length % (args.context_parallel_size * 2) == 0, \
            'seq-length should be a multiple of 2 * context-parallel-size ' \
            'if context-parallel-size > 1.'

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    # if args.seq_length is not None:
    #     assert args.max_position_embeddings >= args.seq_length
    # if args.decoder_seq_length is not None:
    #     assert args.max_position_embeddings >= args.decoder_seq_length
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.moe_grouped_gemm:
        assert args.bf16, 'Currently GroupedGEMM for MoE only supports bf16 dtype.'
        dc = torch.cuda.get_device_capability()
        assert dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels."

    if args.moe_lbl_all_reduce_overlap:
        assert args.moe_router_sync_tokens_per_expert_across_dp, "moe_lbl_all_reduce_overlap is only supported with moe_router_sync_tokens_per_expert_across_dp"
        # assert args.moe_expert_interval == 1, "moe_lbl_all_reduce_overlap might not be correct with moe_expert_interval != 1"
        assert args.reduce_loss_per_iteration, "moe_lbl_all_reduce_overlap is only supported with reduce_loss_per_iteration == True"

    if args.moe_drop_across_ep:
        assert args.use_flashmoe_router, 'Currently --moe-drop-across-ep only supported with flashmoe router.'

    if args.flash_moe_cpu_offload:
        assert args.use_flash_moe, 'Currently, --flash-moe-cpu-offload is only supported with flashmoe layer. '
        assert not args.moe_layer_recompute, 'Currently, --flash-moe-cpu-offload is not compatible with moe_layer_recompute. '

    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    # Persistent fused layer norm.
    if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation recomputing.
    if args.distribute_saved_activations:
        assert args.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups'
        assert args.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity'
        assert args.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method '
        assert (TORCH_MAJOR, TORCH_MINOR) >= (1, 10), \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            'pytorch version is v%s.%s.' % (TORCH_MAJOR, TORCH_MINOR)

    if args.recompute_granularity == 'selective':
        assert args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    if args.recompute_attn_layer:
        assert args.recompute_granularity is None, \
            'recompute granularity is not yet supported for ' \
            'recompute_attn_layer'

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        args.sequence_parallel = False

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.sequence_parallel:
        args.async_tensor_model_parallel_allreduce = False

    if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
        if args.sequence_parallel:
            raise RuntimeError(
                "Using sequence parallelism requires setting the environment variable "
                "CUDA_DEVICE_MAX_CONNECTIONS to 1")
        if args.async_tensor_model_parallel_allreduce:
            raise RuntimeError(
                "Using async gradient all reduce requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Retro checks.
    if args.retro_add_retriever:

        # Train samples should be auto-loaded.
        assert args.train_samples is not None, \
            "args.train_samples should be auto-loaded from the retro config."

        # Sequence parallelism unsupported.
        assert not args.sequence_parallel, \
            "retro currently does not support sequence parallelism."

        # Pipeline parallelism unsupported.
        assert args.pipeline_model_parallel_size == 1, \
            "retro currently does not support pipeline parallelism."

    if args.decoupled_lr is not None or args.decoupled_min_lr is not None:
        assert not args.use_legacy_models, \
            '--decoupled-lr and --decoupled-min-lr is not supported in legacy models.'

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = 'rope'
    if args.rotary_interleaved and args.apply_rope_fusion:
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    if args.rotary_interleaved and args.use_legacy_models:
        raise RuntimeError('--rotary-interleaved is not supported in legacy models.')

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type != 'rope':
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # MoE Spec check
    if args.num_experts == 0:
        args.num_experts = None
    if args.num_experts is not None:
        assert args.spec is None, "Model Spec must be None when using MoEs"

    # Context parallel
    if args.context_parallel_size > 1:
        assert not args.use_legacy_models, "Context parallelism is not supported in legacy models."

    # Expert parallelism check
    if args.expert_model_parallel_size  > 1:
        assert args.num_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."
        assert not args.fp16, \
            "Expert parallelism is not supported with fp16 training."
    if args.use_flash_moe:
        args.use_flashmoe_router = True
    if args.use_flashmoe_hep:
        assert args.use_flash_moe
        assert args.use_flashmoe_router
        args.expert_tensor_parallel_size = 1
        if args.expert_model_parallel_size > 1:
            assert args.expert_model_parallel_size % args.tensor_model_parallel_size == 0, \
                "Number of expert parallel size shall be a multiple of tensor model parallel size in FlashMoE HEP."

    # Distributed checkpointing checks
    if args.use_dist_ckpt and args.use_legacy_models:
        raise RuntimeError('--use-dist-ckpt is not supported in legacy models.')
    if args.dist_ckpt_split_model_and_optim:
        if args.no_save_optim:
            Warning("--dist-ckpt-split-model-and-optim needs save optim, setting to Fasle")
            args.dist_ckpt_split_model_and_optim = False

    # Data blend checks
    assert args.mock_data + \
           bool(args.data_path) + \
           any([args.train_data_path, args.valid_data_path, args.test_data_path]) \
           <= 1, "A single data source must be provided in training mode, else None"

    if args.use_tp_pp_dp_mapping:
        assert args.context_parallel_size * args.expert_model_parallel_size <= 1, \
            "context_parallel and expert_model_parallel can't be used with tp-pp-dp mapping."

    # Deterministic mode
    if args.deterministic_mode:
        assert not args.use_flash_attn, "Flash attention can not be used in deterministic mode."
        assert not args.cross_entropy_loss_fusion, "Cross Entropy Fusion is currently not deterministic."

        all_reduce_choices = ["Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"]
        assert os.getenv("NCCL_ALGO", -1) != -1 and os.getenv("NCCL_ALGO") in all_reduce_choices, \
            f"NCCL_ALGO must be one of {all_reduce_choices}."

        torch.use_deterministic_algorithms(True)

    # Update the printed args to reflect that `apply_query_key_layer_scaling` also controls `attention_softmax_in_fp32`
    if args.apply_query_key_layer_scaling:
        args.attention_softmax_in_fp32 = True

    # Checkpointing
    if args.ckpt_fully_parallel_save_deprecated and args.rank == 0:
        print('--ckpt-fully-parallel-save flag is deprecated and has no effect.'
              ' Use --no-ckpt-fully-parallel-save to disable parallel save.')
    if (
        args.use_dist_ckpt
        and not args.ckpt_fully_parallel_save
        and args.use_distributed_optimizer
        and args.rank == 0
    ):
        print('Warning: With non-parallel ckpt save and DistributedOptimizer,'
              ' it will be impossible to resume training with different parallelism.'
              ' Consider removing flag --no-ckpt-fully-parallel-save.')
    if args.use_dist_ckpt_deprecated and args.rank == 0:
        print('--use-dist-ckpt is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')
    if args.dist_ckpt_format_deprecated and args.rank == 0:
        print('--dist-ckpt-format is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')

    # MoE upcycling check
    if args.moe_use_upcycling:
        assert args.save is not None, "When using upcycling, the --save option must be specified."
        if not args.no_load_optim:
            args.no_load_optim = True
            print('Warning: disabling --no-load-optim for upcycling.')
        if not args.no_load_rng:
            args.no_load_rng = True
            print('Warning: disabling --no-load-rng for upcycling.')

    # ChunkFlow related arguments check
    if args.use_chunkflow_bucket:
        args.use_chunkflow = True
    if args.use_chunkflow:
        # We will remove these constraints later, for now we just add some safe checks
        assert args.reduce_loss_per_iteration, "ChunkFlow requires reduce_loss_per_iteration to be set."
        assert args.chunk_size == -1 or args.chunk_size > 0, "Chunk size must be set greater than 0."
        assert not args.create_attention_mask_in_dataloader, "Attention mask is not supported with chunkflow for now"
        assert not args.dynamic_flash_attn, "Dynamic Flash Attention should be set False for chunkflow"
        assert not args.use_dynamic_collate, "Dynamic collate should be set False for chunkflow"
        assert args.context_parallel_size == 1, "ChunkFlow is not supported with context parallelism for now"
        args.chunk_size = args.seq_length if args.chunk_size == -1 else args.chunk_size
        args.chunks_kept = (args.seq_length + args.chunk_size - 1) // args.chunk_size if args.chunks_kept == -1 else args.chunks_kept

    assert not (args.use_chunkflow and args.use_smart_batching), (
        "Smart batching cannot be used together with Chunkflow. "
        "Please make sure only one is enabled.")

    # Set related arguments for smart batching.
    if args.use_smart_batching_bucket:
        args.use_smart_batching = True
    if args.use_smart_batching:
        args.use_flatten_collate = True
        args.use_dynamic_collate = False
        args.use_sample_average_loss = True
        args.chunk_size = args.seq_length # for smartbatching chunksize is equal to max sql

    if args.use_smart_batching or args.use_chunkflow:
        if args.encoder_pipeline_model_parallel_size > 2:
            assert args.dynamic_padding_full, 'for unknown reasons, variable_sequence_lenths causes hang when pp>2'


    # Print arguments.
    _print_args("arguments", args)

    return args


def _print_args(title, args):
    """Print arguments."""
    if args.rank == 0:
        print(f'------------------------ {title} ------------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------',
              flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


def core_transformer_config_from_args(args, config_class=None):

    # Config class.
    config_class = config_class or TransformerConfig

    if args.multi_latent_attention:
        config_class = MLATransformerConfig
    elif args.diff_attention:
        config_class = DiffTransformerConfig
    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['layernorm_epsilon'] = args.norm_epsilon
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = False
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    kw_args['first_pipeline_num_layers']= args.decoder_first_pipeline_num_layers
    kw_args['last_pipeline_num_layers']= args.decoder_last_pipeline_num_layers
    kw_args['pipeline_stages_num_layers_override'] = args.pipeline_stages_num_layers_override
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion
    if args.squared_relu:
        assert not args.swiglu
        kw_args['activation_func'] = squared_relu
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None
    kw_args['config_logger_dir'] = args.config_logger_dir
    kw_args['num_layers_per_pipeline_stage'],_  = parallel_state.get_pipeline_num_layers_per_stage_and_offset(args)

    # Qwen3.5 hybrid attention config
    kw_args['full_attention_interval'] = args.full_attention_interval
    kw_args['attn_output_gate'] = args.attn_output_gate
    kw_args['linear_attention_conv_kernel_dim'] = args.linear_attention_conv_kernel_dim
    kw_args['linear_attention_key_head_dim'] = args.linear_attention_key_head_dim
    kw_args['linear_attention_value_head_dim'] = args.linear_attention_value_head_dim
    kw_args['linear_attention_num_key_heads'] = args.linear_attention_num_key_heads
    kw_args['linear_attention_num_value_heads'] = args.linear_attention_num_value_heads
    kw_args['mrope_interleaved'] = args.mrope_interleaved
    kw_args['mrope_section'] = args.mrope_section

    # Return config.
    return config_class(**kw_args)


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


def core_vision_model_config_from_args(args):
    config_class = VLVisionConfig
    llm_config = core_transformer_config_from_args(args)

    kw_args = dataclasses.asdict(llm_config)

    # VIT相关参数
    config = config_class(**kw_args)

    config.num_layers = 32 # depth
    config.num_attention_heads = 16 # num_heads
    config.add_bias_linear = True # all nn.Linear has bias (MLP, attn)
    config.add_qkv_bias = True # qkv_proj in attn has bias
    config.hidden_size = 1280 # embed_dim
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.ffn_hidden_size = 1280 * 4 # embed_dim * mlp_ratio
    config.gated_linear_unit = False # no gated
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads # no GQA
    config.layernorm_zero_centered_gamma = False # False
    config.apply_query_key_layer_scaling = False # factor=math.sqrt(head_dim)
    config.activation_func = quick_gelu
    config.bias_activation_fusion = False # no swiglu, set false
    config.bias_dropout_fusion = False # no dropout, set false
    config.attention_softmax_in_fp32 = True # use True
    config.normalization = 'LayerNorm' # use LayerNorm
    config.seq_length = args.seq_length

    config.temporal_patch_size = 2
    config.spatial_patch_size = 14
    config.in_channels = 3
    config.spatial_merge_size = 2
    config.temporal_merge_size = 4

    # no sp and pp for VIT
    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.pipeline_model_parallel_size = 1
    config.first_pipeline_num_layers = None
    config.last_pipeline_num_layers = None
    return config

def core_qwen2_5_vl_vision_model_config_from_args(args):
    config_class = VLVisionConfig
    llm_config = core_transformer_config_from_args(args)

    kw_args = dataclasses.asdict(llm_config)

    # VIT相关参数
    config = config_class(**kw_args)

    config.num_layers = 32 # depth
    config.num_attention_heads = 16 # num_heads
    config.add_bias_linear = True # all nn.Linear has bias (MLP, attn)
    config.add_qkv_bias = True # qkv_proj in attn has bias
    config.hidden_size = 1280 # embed_dim
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.ffn_hidden_size = 3420 # embed_dim * mlp_ratio
    config.gated_linear_unit = True # use gated
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads # no GQA
    config.layernorm_zero_centered_gamma = False # False
    config.apply_query_key_layer_scaling = False # factor=math.sqrt(head_dim)
    config.bias_activation_fusion = True # use silu
    config.bias_dropout_fusion = False # no dropout, set false
    config.attention_softmax_in_fp32 = True # use True
    config.normalization = 'RMSNorm' # use RMSNorm
    config.seq_length = args.seq_length

    config.temporal_patch_size = 2
    config.spatial_patch_size = 14
    config.in_channels = 3
    config.spatial_merge_size = 2
    config.temporal_merge_size = 4
    config.vision_window_size = 112
    config.fullatt_block_indexes = list(range(7, 32, 8))

    # no sp and pp for VIT
    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.pipeline_model_parallel_size = 1
    config.first_pipeline_num_layers = None
    config.last_pipeline_num_layers = None
    return config

def core_mlp_projection_config_from_args(args):

    config_class = TransformerConfig
    config = core_transformer_config_from_args(args, config_class=config_class)


    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = True
    config.ffn_hidden_size = 1280 * (args.spatial_merge_size ** 2)
    config.activation_func = F.gelu

    config.tp_comm_overlap = False
    config.sequence_parallel = False

    return config

def core_audio_model_config_from_args(args):
    """Create audio model configuration from arguments."""
    from megatron.core.transformer.transformer_config import AudioConfig
    
    # Get base transformer config
    base_config = core_transformer_config_from_args(args)
    
    # Create audio config with audio-specific parameters
    audio_config = AudioConfig(
        # Audio processing parameters
        encoder_type=args.audio_encoder_type,
        input_size=args.audio_input_size,
        linear_units=args.audio_linear_units,
        attention_heads=args.audio_attention_heads,
        num_blocks=args.audio_num_blocks,
        downsample_rate=args.audio_downsample_rate,
        use_padmask=args.audio_use_padmask,
        encoder_no_use_fused_attention=args.audio_encoder_no_use_fused_attention,
        encoder_use_packed_attention=args.audio_encoder_use_packed_attention,

        # Tokenizer parameters
        tokenizer_version=args.audio_tokenizer_version,
        tokenizer_frame_rate=args.audio_tokenizer_frame_rate,
        tokenizer_code_book_size=args.audio_tokenizer_code_book_size,
        
        # Audio adaptor parameters
        encoder_dim=args.audio_encoder_dim,
        llm_dim=args.audio_llm_dim if args.audio_llm_dim is not None else base_config.hidden_size,
        adaptor_downsample_rate=args.audio_adaptor_downsample_rate,
        ffn_dim=args.audio_ffn_dim,
        adaptor_num_layers=args.audio_adaptor_num_layers,
        attention_dropout_rate=args.audio_attention_dropout_rate,
        dropout_rate=args.audio_dropout_rate,

        freeze_encoder=not args.audio_no_freeze_encoder,
        encoder_lr_scale=args.audio_encoder_lr_scale,
        freeze_adaptor=args.audio_freeze_adaptor,
        adaptor_lr_scale=args.audio_adaptor_lr_scale,
        freeze_decoder=args.audio_freeze_decoder,
        decoder_lr_scale=args.audio_decoder_lr_scale,
        audio_input_output_lr_scale=args.audio_input_output_lr_scale,
        recompute_encoder=args.audio_recompute_encoder,

        encoder_adaptor_sliding_window_size=args.audio_encoder_adaptor_sliding_window_size,
        encoder_adaptor_sliding_window_left_context=args.audio_encoder_adaptor_sliding_window_left_context,
        encoder_adaptor_bucketing_threshold=args.audio_encoder_adaptor_bucketing_threshold,

        is_dual_channel=args.audio_is_dual_channel,
        audio_output_head=not args.no_audio_output_head,
        
        # Attention configuration
        att_type=args.audio_att_type,
        kernel_size=args.audio_kernel_size,
        sanm_shfit=args.audio_sanm_shift,
        max_position_embeddings=args.audio_max_position_embeddings,
        rope_theta=args.audio_rope_theta,
        
        # Inherit from base config
        **{k: v for k, v in base_config.__dict__.items() 
           if k not in ['input_size', 'linear_units', 'attention_heads', 'num_blocks', 
                       'downsample_rate', 'use_padmask', 'encoder_dim', 'llm_dim', 
                       'ffn_dim', 'n_layer', 'attention_dropout_rate', 'dropout_rate',
                       'att_type', 'kernel_size', 'sanm_shfit', 'max_position_embeddings', 'rope_theta']}
    )
    
    return audio_config

def _add_transformer_engine_args(parser):
    group = parser.add_argument_group(title='Transformer-Engine')

    group.add_argument('--fp8-format', default=None,
                       choices=['e4m3', 'hybrid'],
                       help='Which fp8 format scheme to use for FP8 tensors in the forward and backward pass',
                       dest='fp8')
    group.add_argument('--fp8-margin', type=int, default=0,
                       help='Scaling margin for fp8',
                       dest='fp8_margin')
    group.add_argument('--fp8-interval', type=int, default=1,
                       help='DEPRECATED. This flag is ignored. Scaling update interval for fp8',
                       dest='fp8_interval')
    group.add_argument('--fp8-amax-history-len', type=int, default=1,
                       help='Number of steps for which amax history is recorded per tensor',
                       dest='fp8_amax_history_len')
    group.add_argument('--fp8-amax-compute-algo', default='most_recent',
                       choices=['most_recent', 'max'],
                       help='Algorithm for computing amax from history',
                       dest='fp8_amax_compute_algo')
    group.add_argument('--no-fp8-wgrad', action='store_false',
                       help='Execute wgrad in higher precision even for FP8 runs',
                       dest='fp8_wgrad')
    group.add_argument('--transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--fp8-param-gather', action='store_true',
                       help='Keep the compute param in fp8 (do not use any other intermediate '
                            'dtype) and perform the param all-gather in fp8.')

    return parser

def _add_inference_args(parser):
    group = parser.add_argument_group(title='inference')

    group.add_argument('--inference-batch-times-seqlen-threshold',
                       type=int, default=512,
                       help='During inference, if batch-size times '
                       'sequence-length is smaller than this threshold '
                       'then we will not use pipelining, otherwise we will.')
    group.add_argument('--max-tokens-to-oom',
                       type=int, default=12000,
                       help='Maximum number of tokens during inference'
                       'tokens here is # in prompt + # to generate'
                       'Allows us to throw an error before OOM crashes server')
    group.add_argument('--output-bert-embeddings', action='store_true',
                       help='Output Bert embeddings (via mean pooling) from '
                       'model, rather than its binary head output or entire '
                       'hidden batch.')
    group.add_argument('--bert-embedder-type', default="megatron",
                       choices=["megatron", "huggingface"],
                       help='Select either Megatron or Huggingface as the '
                       'Bert embedder.')
    group.add_argument('--stop-words', type=str, default=None, help='stop words')
    group.add_argument('--stop-words-ids', type=str, default=None, help='stop words ids')
    group.add_argument('--use-dynamic-ntk', action='store_true', help='use dynamic ntk to support long sequence')
    group.add_argument('--use-logn-attn', action='store_true', help='use logn attn')
    # TODO: Merge --use-window-attn-legacy and related arguments into --use-window-attn.
    group.add_argument('--use-window-attn-legacy', action='store_true', help='use window attn (legacy)')
    group.add_argument('--window-interval', type=int, default=4, help='use window attn')
    return parser


def _add_retro_args(parser):
    group = parser.add_argument_group(title='retro')

    group.add_argument('--retro-project-dir', default=None,
                       help='Retro project directory, which contains the '
                       'preprocessed data for pretraining. This directory '
                       'is built during preprocessing (see '
                       'tools/retro/README.md), and contains subdirectories '
                       'for the chunk database and pretraining neighbors.')
    group.add_argument('--retro-add-retriever',
                       action='store_true', default=False,
                       help='Add a retriever to the transformer, for use in '
                       'pretraining a Retro model.')
    group.add_argument('--retro-cyclic-train-iters', type=int, default=None,
                       help='Set number of training iterations for cyclic '
                       'Retro training.')
    group.add_argument('--retro-encoder-layers', type=int, default=2,
                       help='Number of layers to use for the retrieval '
                       'encoder.')
    group.add_argument('--retro-encoder-hidden-dropout',
                       type=float, default=0.1, help='Hidden dropout for '
                       'retrieval encoder.')
    group.add_argument('--retro-encoder-attention-dropout',
                       type=float, default=0.1, help='Attention dropout for '
                       'retrieval encoder.')
    group.add_argument("--retro-num-neighbors", type=int, default=2,
                       help='Number of neighbors to retrieve during '
                       'pretraining.')
    group.add_argument("--retro-num-retrieved-chunks", type=int, default=2,
                       help='Number of chunks to retrieve from the retrieval '
                       'database.')
    group.add_argument("--retro-attention-gate", type=float, default=1,
                       help="Gated cross attention.")
    group.add_argument("--retro-no-verify-neighbor-count", action="store_false",
                       dest="retro_verify_neighbor_count",
                       help="Skip verifying that len(GPT dataset) == len(saved "
                       "neighbors).")

    # Enforce argument naming convention.
    for action in group._group_actions:
        prefix = action.dest.split("_")[0]
        assert prefix == "retro", \
            "Retro args must be prefixed with '--retro-*', for consistent " \
            "styling. Please fix '%s'." % ", ".join(action.option_strings)

    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')

    group.add_argument('--num-layers', type=int, default=None,
                       help='Number of transformer layers.')
    group.add_argument('--encoder-num-layers', type=int, default=None,
                       help='Number of encoder transformer layers.')
    group.add_argument('--decoder-num-layers', type=int, default=None,
                       help='Number of decoder transformer layers.')
    group.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')
    group.add_argument('--ffn-hidden-size', type=int, default=None,
                       help='Transformer Feed-Forward Network hidden size. '
                       'This is set to 4*hidden-size if not provided')
    group.add_argument('--ffn-multi', type=float, default=None,
                       help='modify ffn multi. 4 for normal activations, 8/3 for *glu activations.')
    group.add_argument('--make-ffn-dim-multiple-of', type=int, default=None, help='make ffn_in multiple of ')
    group.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    group.add_argument('--kv-channels', type=int, default=None,
                       help='Projection weights dimension in multi-head '
                       'attention. This is set to '
                       '   args.hidden_size // args.num_attention_heads '
                       'if not provided.')
    group.add_argument('--group-query-attention', action='store_true',
                          help='Use group-query attention.')
    group.add_argument('--num-query-groups', type=int, default=1)

    group.add_argument('--max-position-embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--position-embedding-type', type=str, default='learned_absolute',
                       choices=['learned_absolute', 'rope', 'none'],
                       help='Position embedding type.')
    group.add_argument('--use-rotary-position-embeddings', action='store_true',
                       help='Use rotary positional embeddings or not. '
                       'Deprecated: use --position-embedding-type')
    group.add_argument('--rotary-base', type=int, default=10000,
                       help='Base to use for rotary positional embeddings, default 10000')
    group.add_argument('--rotary-percent', type=float, default=1.0,
                       help='Percent of rotary dimension to use, default 100%%')
    group.add_argument('--rotary-interleaved', action='store_true',
                          help='Use interleaved rotary embedding.')
    group.add_argument('--rotary-seq-len-interpolation-factor', type=int, default=None,
                       help='Sequence length interpolation factor for rotary embeddings.')
    group.add_argument('--use-rope-scaling', action='store_true',
                       help='Apply rope scaling as used in llama3.1')
    group.add_argument('--no-position-embedding',
                       action='store_false',
                       help='Disable position embedding. Deprecated: use --position-embedding-type',
                       dest='add_position_embedding')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--load-padded-vocab-size', action='store_true',
                       help='If set, load the padded vocab size from the checkpoint.')
    group.add_argument('--normalization', default='LayerNorm',
                       choices=['LayerNorm', 'RMSNorm'],
                       help='Which normalization technique to use.')
    group.add_argument('--norm-epsilon', type=float, default=1e-5,
                       help='Epsilon for layer norm and RMS norm.')
    group.add_argument('--apply-layernorm-1p', action='store_true',
                       help='Adjust LayerNorm weights such that they are centered '
                       'around zero. This improves numerical stability.')
    group.add_argument('--apply-residual-connection-post-layernorm',
                       action='store_true',
                       help='If set, use original BERT residula connection '
                       'ordering.')
    group.add_argument('--openai-gelu', action='store_true',
                       help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')
    group.add_argument('--squared-relu', action='store_true',
                       help='Use squared relu activation instead of default gelu')
    group.add_argument('--swiglu', action='store_true',
                       help='Use gated linear units and SiLU activation instead of default gelu')
    group.add_argument('--onnx-safe', type=bool, required=False,
                       help='Use workarounds for known problems with '
                       'Torch ONNX exporter')
    group.add_argument('--bert-no-binary-head', action='store_false',
                       help='Disable BERT binary head.',
                       dest='bert_binary_head')
    group.add_argument('--untie-embeddings-and-output-weights', action='store_true',
                       help='Untie embeddings and output weights.')
    group.add_argument('--multi-latent-attention', action='store_true',
                       help='Use multi-latent attention for model.')
    # regularization arguments
    group.add_argument('--qk-layernorm', action='store_true',
                       help='Whether to layer normalize the q and k attention embeddings.')

    # diff attention
    group.add_argument('--diff-attention', action='store_true')
    group.add_argument('--diff-lambda-init', type=float, default=0.5)
    group.add_argument('--no-use-vectorized-lambda', action='store_false', dest='use_vectorized_lambda',
                       help='Whether to use vectorized lambda for diff attention.')
    group.add_argument('--v-layernorm', action='store_true')

    # Qwen3.5 hybrid attention args
    group.add_argument('--full-attention-interval', type=int, default=0,
                       help='If > 0, every Nth layer uses full attention; rest use linear attention '
                       '(GatedDeltaNet). 0 means all layers use full attention. E.g. 4 for Qwen3.5.')
    group.add_argument('--attn-output-gate', action='store_true',
                       help='Enable attention output gating (Qwen3.5 style). Q projection outputs '
                       '2x size, and attention output is multiplied by sigmoid(gate).')

    # GatedDeltaNet linear attention args
    group.add_argument('--linear-attention-conv-kernel-dim', type=int, default=4,
                       help='Kernel size for causal Conv1d in GatedDeltaNet. Default: 4.')
    group.add_argument('--linear-attention-key-head-dim', type=int, default=128,
                       help='Key head dimension for GatedDeltaNet. Default: 128.')
    group.add_argument('--linear-attention-value-head-dim', type=int, default=128,
                       help='Value head dimension for GatedDeltaNet. Default: 128.')
    group.add_argument('--linear-attention-num-key-heads', type=int, default=16,
                       help='Number of key heads for GatedDeltaNet. Default: 16.')
    group.add_argument('--linear-attention-num-value-heads', type=int, default=64,
                       help='Number of value heads for GatedDeltaNet. Default: 64.')

    # M-RoPE args
    group.add_argument('--mrope-interleaved', action='store_true',
                       help='Use interleaved M-RoPE layout (Qwen3.5) instead of chunked layout.')
    group.add_argument('--mrope-section', type=int, nargs='+', default=None,
                       help='Section sizes for M-RoPE interleaving, e.g. 11 11 10 for Qwen3.5.')

    return parser


def _add_straggler_detector_args(parser):
    group = parser.add_argument_group(title='straggler')
    group.add_argument('--log-straggler', action='store_true',
                       help='If set, tracks and logs straggler per GPU.')
    group.add_argument('--disable-straggler-on-startup', action='store_true',
                       help='If set, StragglerDetector is disabled on startup.')
    group.add_argument('--straggler-ctrlr-port', type=int, default=65535,
                       help='Port number to toggle StragglerDetector on/off at runtime')
    group.add_argument('--straggler-minmax-count', type=int, default=1,
                       help='Number of ranks to report with high/low estimated throughput')
    return parser


def _add_one_logger_args(parser):
    group = parser.add_argument_group(title='one logger')
    group.add_argument('--no-one-logger', action='store_false',
                       help='If set, disable using one_logger to track E2E metrics'
                       'Note that one_logger is an internal tool and not '
                       'available externally. For installation, please go to '
                       'https://confluence.nvidia.com/display/MLWFO/Package+Repositories'
                       'for more details',
                       dest='enable_one_logger')
    group.add_argument('--one-logger-project', type=str, default='megatron-lm',
                       help='The one-logger project name. Will ignore if '
                       '--no-one-logger is set')
    group.add_argument('--one-logger-run-name', type=str, default=None,
                       help='The one-logger run name displayed. Will ignore if '
                       '--no-one-logger is set')
    group.add_argument('--one-logger-async', action='store_true',
                       help='If set, forces one_logger to use async mode.')
    group.add_argument('--app-tag-run-name', type=str, default=None,
                       help='Jobs belonging to same training run, suppose to '
                       'have the same name. It will be used to track progress of '
                       'a training done over multiple different jobs')
    group.add_argument('--app-tag-run-version', type=str, default='0.0.0',
                       help='The version of the training of which current job is '
                       'part of. It will be used to track the changes in the '
                       'application side which might change the performance '
                       'baseline')
    return parser


def _add_ft_package_args(parser):
    group = parser.add_argument_group(title='ft_package')
    group.add_argument('--enable-ft-package', action='store_true',
                       help='If set, Fault Tolerance package is enabled. '
                       'Note: This feature is for Nvidia internal use only.')
    return parser


def _add_config_logger_args(parser):
    group = parser.add_argument_group(title='config logger')
    group.add_argument('--config-logger-dir', type=str, default='',
                       help='If set, will dump all configs to --config-logger-dir',
                       dest='config_logger_dir')
    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')

    group.add_argument('--log-params-norm', action='store_true',
                       help='If set, calculate and log parameters norm.')
    group.add_argument('--log-num-zeros-in-grad', action='store_true',
                       help='If set, calculate and log the number of zeros in gradient.')
    group.add_argument('--log-throughput', action='store_true',
                       help='If set, calculate and log throughput per GPU.')
    group.add_argument('--log-progress', action='store_true',
                       help='If set, log progress (in terms of number of processed tokens and '
                       'number of floating-point operations) to progress.txt file in checkpoint '
                       'directory.')
    group.add_argument('--timing-log-level', type=int,
                       default=0, choices=range(0,3),
                       help='Granularity level to measure and report timing. '
                       '   0: report only iteration time and make sure timing '
                       '      does not introduce extra overhead.'
                       '   1: report timing for operations that are executed '
                       '      very limited times (basically once) during '
                       '      each iteration (such as gradient all-reduce) '
                       '   2: report timing for operations that migh be '
                       '      executed numerous times during each iteration. '
                       'Note that setting the level to 1 or 2 might '
                       'cause increase in iteration time.')
    group.add_argument('--increase-log-level-interval', type=int, default=1000,
                       help='increase log level to 2 interval. ')
    group.add_argument('--increase-log-level-iters', type=int, default=5,
                       help='increase log level to 2 iters. ')

    group.add_argument('--no-barrier-with-level-1-timing', action='store_false',
                       help='If not set, use barrier with level 1 time '
                       'measurements. Note that this is up to the user '
                       'to make sure calling barrier with their timers '
                       'will not result in hangs. This can happen if for '
                       'example the user adds a level 1 timer that is not '
                       'called by all ranks.',
                       dest='barrier_with_L1_time')
    group.add_argument('--timing-log-option', type=str, default='minmax',
                       choices=['max', 'minmax', 'all'],
                       help='Options for logging timing:'
                       '  max: report the max timing across all ranks'
                       '  minmax: report min and max timings across all ranks'
                       '  all: report timings of all ranks.')
    group.add_argument('--tensorboard-log-interval', type=int, default=1,
                       help='Report to tensorboard interval.')
    group.add_argument('--tensorboard-queue-size', type=int, default=1000,
                       help='Size of the tensorboard queue for pending events '
                       'and summaries before one of the ‘add’ calls forces a '
                       'flush to disk.')
    group.add_argument('--log-timers-to-tensorboard', action='store_true',
                       help='If set, write timers to tensorboard.')
    group.add_argument('--no-log-loss-scale-to-tensorboard',
                       action='store_false',
                       help='Disable loss-scale logging to tensorboard.',
                       dest='log_loss_scale_to_tensorboard')
    group.add_argument('--log-validation-ppl-to-tensorboard',
                       action='store_true',
                       help='If set, write validation perplexity to '
                       'tensorboard.')
    group.add_argument('--log-memory-to-tensorboard',
                       action='store_true',
                       help='Enable memory logging to tensorboard.')
    group.add_argument('--log-world-size-to-tensorboard',
                       action='store_true',
                       help='Enable world size logging to tensorboard.')
    group.add_argument('--wandb-project', type=str, default='',
                       help='The wandb project name. Ignore wandb by default.')
    group.add_argument('--wandb-exp-name', type=str, default='',
                       help='The wandb experiment name.')
    group.add_argument('--wandb-save-dir', type=str, default='',
                       help='Path to save the wandb results locally.')
    group.add_argument('--swanlab-project', type=str, default='',
                       help='The swanlab project name. Ignore swanlab by default.')
    group.add_argument('--swanlab-exp-name', type=str, default='',
                       help='The swanlab experiment name.')
    group.add_argument('--swanlab-api-key', type=str, default='',
                       help='The swanlab api key.')
    group.add_argument('--swanlab-host', type=str, default='',
                       help='The swanlab host.')
    group.add_argument('--logging-level', type=int, default=None,
                       help='Set default logging level')
    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Post attention dropout probability.')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--start-weight-decay', type=float,
                       help='Initial weight decay coefficient for L2 regularization.')
    group.add_argument('--end-weight-decay', type=float,
                       help='End of run weight decay coefficient for L2 regularization.')
    group.add_argument('--weight-decay-incr-style', type=str, default='constant',
                       choices=['constant', 'linear', 'cosine'],
                       help='Weight decay increment function.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help='First coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-beta2', type=float, default=0.999,
                       help='Second coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-eps', type=float, default=1e-08,
                       help='Term added to the denominator to improve'
                       'numerical stability')
    group.add_argument('--sgd-momentum', type=float, default=0.9,
                       help='Momentum factor for sgd')
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--rampup-batch-size', nargs='*', default=None,
                       help='Batch size ramp up with the following values:'
                       '  --rampup-batch-size <start batch size> '
                       '                      <batch size incerement> '
                       '                      <ramp-up samples> '
                       'For example:'
                       '   --rampup-batch-size 16 8 300000 \\ '
                       '   --global-batch-size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--decrease-batch-size-if-needed', action='store_true', default=False,
                       help='If set, decrease batch size if microbatch_size * dp_size'
                       'does not divide batch_size. Useful for KSO (Keep Soldiering On)'
                       'to continue making progress if number of healthy GPUs (and'
                       'corresponding dp_size) does not support current batch_size.'
                       'Old batch_size will be restored if training is re-started with'
                       'dp_size that divides batch_size // microbatch_size.')
    group.add_argument('--recompute-activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--recompute-attn-layer', action='store_true',
                       help='recompute the whole attention layer to allow for training '
                            'with larger models, sequences, and batch sizes.')
    group.add_argument('--recompute-mlp-activation-function', action='store_true',
                       help='If set, the activation function in MLP will recompute.')
    group.add_argument('--recompute-layernorm', action='store_true',
                       help='If set, the input_layernorm and pre_mlp_layernorm will recompute.')
    group.add_argument('--recompute-granularity', type=str, default=None,
                       choices=['full', 'selective'],
                       help='Checkpoint activations to allow for training '
                       'with larger models, sequences, and batch sizes. '
                       'It is supported at two granularities 1) full: '
                       'whole transformer layer is recomputed, '
                       '2) selective: core attention part of the transformer '
                       'layer is recomputed.')
    group.add_argument('--no-check-for-nan-in-loss-and-grad', action='store_false',
                       help='Check for NaNs in loss and grad',
                       dest='check_for_nan_in_loss_and_grad')
    group.add_argument('--distribute-saved-activations',
                       action='store_true',
                       help='If set, distribute recomputed activations '
                       'across model parallel group.')
    group.add_argument('--recompute-method', type=str, default=None,
                       choices=['uniform', 'block'],
                       help='1) uniform: uniformly divide the total number of '
                       'Transformer layers and recompute the input activation of '
                       'each divided chunk at specified granularity, '
                       '2) recompute the input activations of only a set number of '
                       'individual Transformer layers per pipeline stage and do the '
                       'rest without any recomputing at specified granularity'
                       'default) do not apply activations recompute to any layers')
    group.add_argument('--recompute-num-layers', type=int, default=None,
                       help='1) uniform: the number of Transformer layers in each '
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.')
    group.add_argument('--no-clone-scatter-output-in-embedding', action='store_false',
                       help='If not set, clone the output of the scatter in embedding layer to GC original tensor.',
                       dest='clone_scatter_output_in_embedding')
    group.add_argument('--profile', action='store_true',
                       help='Enable nsys profiling. When using this option, nsys '
                       'options should be specified in commandline. An example '
                       'nsys commandline is `nsys profile -s none -t nvtx,cuda '
                       '-o <path/to/output_file> --force-overwrite true '
                       '--capture-range=cudaProfilerApi '
                       '--capture-range-end=stop`.')
    group.add_argument('--profile-step-start', type=int, default=10,
                       help='Global step to start profiling.')
    group.add_argument('--profile-step-end', type=int, default=12,
                       help='Global step to stop profiling.')
    group.add_argument('--use-pytorch-profiler', action='store_true',
                       help='Use the built-in pytorch profiler. '
                       'Useful if you wish to view profiles in tensorboard.',
                       dest='use_pytorch_profiler')
    group.add_argument('--profile-ranks', nargs='+', type=int, default=[0],
                       help='Global ranks to profile.')
    group.add_argument('--tp-comm-overlap', action='store_true', help='Enables the '
                       ' overlap of Tensor parallel communication and GEMM kernels.')
    group.add_argument('--tp-comm-overlap-cfg', type=str, default=None,
                       help='Config file when tp_comm_overlap is enabled.')
    group.add_argument('--disable-tp-comm-overlap-ag', action='store_false',
                       help=('Disables the All-Gather overlap with GEMM by '
                             'pipelining the GEMM and All-Gather.'),
                       dest='tp_comm_overlap_ag')
    group.add_argument('--disable-tp-comm-overlap-rs', action='store_false',
                       help=('Disables the Reduce-Scatter overlap with GEMM by '
                             'pipelining the GEMM and Reduce-Scatter.'),
                       dest='tp_comm_overlap_rs')
    group.add_argument('--tp-comm-overlap-rs-dgrad', action='store_true',
                       help = 'Enables the Reduce-Scatter overlap with dgrad GEMM.',
                       dest='tp_comm_overlap_rs_dgrad')
    group.add_argument('--disable-tp-comm-bulk-dgrad', action='store_false',
                       help='Disables the All-Gather overlap with bprop activation gradient GEMM.',
                       dest='tp_comm_bulk_dgrad')
    group.add_argument('--disable-tp-comm-bulk-wgrad', action='store_false',
                       help='Disables the Reduce-Scatter overlap with bprop weight gradient GEMM.',
                       dest='tp_comm_bulk_wgrad')
    group.add_argument('--tp-comm-bootstrap-backend', default='nccl', type=str,
                       choices=['nccl', 'mpi', 'gloo'],
                       help='Set the bootstrapping backend of Tensor parallel communications.')
    group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None,
                       help='If set, initialize weights on the CPU. This eliminates init differences based on tensor parallelism.')
    group.add_argument('--empty-unused-memory-level', default=0, type=int,
                       choices=[0, 1, 2],
                       help='Call torch.cuda.empty_cache() each iteration '
                       '(training and eval), to reduce fragmentation.'
                       '0=off, 1=moderate, 2=aggressive.')
    group.add_argument('--deterministic-mode', action='store_true',
                       help='Choose code that has deterministic execution. This usually '
                       'means slower execution, but is good for debugging and testing.')
    group.add_argument('--check-weight-hash-across-dp-replicas-interval', type=int, default=None,
                       help='Interval to check weight hashes are same across DP replicas. If not specified, weight hashes not checked.')
    group.add_argument('--calculate-per-token-loss', action='store_true',
                       help=('Scale cross entropy loss by the number of non-padded tokens in the '
                             'global batch, versus the default behavior of assuming all tokens are non-padded.'))
    group.add_argument('--train-sync-interval', type=int, default=None,
                       help='Training CPU-GPU synchronization interval, to ensure that CPU is not running too far ahead of GPU.')

    # deprecated
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--advance-lr-on-skipped-iter', action='store_true',
                       default=False,
                       help='Advance the optimizer parameter scheduler even '
                       'when the optimizer step is skipped due to invalid '
                       'gradients. By default, skipped iterations do not '
                       'advance the learning-rate schedule.')
    group.add_argument('--debug-nan-grad', action='store_true',
                       default=False,
                       help='When an optimizer step is skipped due to NaN/Inf '
                       'gradients, scan model gradients and log the first '
                       'non-finite tensors on each rank.')
    group.add_argument('--debug-backward-nan', action='store_true', default=False,
                       help='Register low-memory module input/output-gradient hooks for a selected '
                       'rank and iteration and report where gradients become non-finite.')
    group.add_argument('--debug-backward-nan-rank', type=int, default=0,
                       help='Global rank on which backward NaN hooks are enabled. '
                       'Use -1 to enable hooks on every global rank.')
    group.add_argument('--debug-backward-nan-iteration', type=int, default=0,
                       help='Training iteration on which backward NaN hooks are enabled. '
                       'A fresh run starts at iteration 0.')
    group.add_argument('--debug-backward-nan-module-regex', type=str, default='.*',
                       help='Regular expression selecting named modules to hook.')
    group.add_argument('--debug-backward-nan-max-events', type=int, default=64,
                       help='Maximum number of non-finite backward hook events to print, '
                       'per rank.')
    group.add_argument('--debug-backward-nan-exit-after-iteration', action='store_true',
                       default=False,
                       help='Exit all ranks after the selected diagnostic iteration, before '
                       'evaluation or checkpoint saving.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--exit-duration-in-mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    group.add_argument('--exit-signal-handler', action='store_true',
                       help='Dynamically save the checkpoint and shutdown the '
                       'training if SIGTERM is received')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--no-masked-softmax-fusion',
                       action='store_false',
                       help='Disable fusion of query_key_value scaling, '
                       'masking, and softmax.',
                       dest='masked_softmax_fusion')
    group.add_argument('--no-bias-gelu-fusion', action='store_false',
                       help='Disable bias and gelu fusion.',
                       dest='bias_gelu_fusion')
    group.add_argument('--no-bias-swiglu-fusion', action='store_false',
                       help='Disable bias and swiglu fusion, the fusion is '
                       'available only when using megatron-core.',
                       dest='bias_swiglu_fusion')
    group.add_argument('--no-bias-dropout-fusion', action='store_false',
                       help='Disable bias and dropout fusion.',
                       dest='bias_dropout_fusion')
    group.add_argument('--no-rope-fusion', action='store_false',
                       help='Disable rope fusion, the fusion is available '
                       'only when using megatron-core.',
                       dest='apply_rope_fusion')
    group.add_argument('--cross-entropy-loss-fusion', action='store_true',
                       help='Enabled fusion of cross entropy loss calculation.',
                       dest='cross_entropy_loss_fusion')
    group.add_argument('--use-flash-attn', action='store_true',
                       help='use FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2205.14135')
    group.add_argument('--use-flash-cross-entropy', action='store_true',
                       help='use FlashAttention implementation of cross entropy loss calculation.')
    group.add_argument('--inplace-flash-cross-entropy', action='store_true',
                       help='Use inplace version of FlashAttention implementation of cross entropy loss calculation.')
    group.add_argument('--disable-bias-linear', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_linear')
    group.add_argument('--add-qkv-bias', action='store_true',
                       help='Enable bias only in the QKV linear layers',
                       dest='add_qkv_bias')
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer function')
    group.add_argument('--dataloader-type', type=str, default=None,
                       choices=['single', 'cyclic', 'external'],
                       help='Single pass vs multiple pass data loader')
    group.add_argument('--no-async-tensor-model-parallel-allreduce',
                       action='store_false',
                       help='DEPRECATED. This flag is ignored.',
                       dest='async_tensor_model_parallel_allreduce')
    group.add_argument('--no-persist-layer-norm', action='store_true',
                       help='Disable using persistent fused layer norm kernel. '
                       'This kernel supports only a set of hidden sizes. Please '
                       'check persist_ln_hidden_sizes if your hidden '
                       'size is supported.')
    group.add_argument('--sequence-parallel', action='store_true',
                       help='Enable sequence parallel optimization.')
    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_false',
                       help='Disable fusing gradient accumulation to weight '
                       'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    group.add_argument('--use-mcore-models', action='store_true',
                       dest='deprecated_use_mcore_models',
                       help='DEPRECATED. Use the implementation from megatron core.'
                       'Now ignored and mcore models are the default, use '
                       '--use-legacy-models to not use core models.')
    group.add_argument('--use-legacy-models', action='store_true',
                       help='Use the legacy Megatron models, not Megatron-Core models.')
    group.add_argument('--manual-gc', action='store_true',
                       help='Disable the threshold-based default garbage '
                       'collector and trigger the garbage collection manually. '
                       'Manual garbage collection helps to align the timing of '
                       'the collection across ranks which mitigates the impact '
                       'of CPU-associated jitters. When the manual gc is enabled, '
                       'garbage collection is performed only at the start and the '
                       'end of the validation routine by default.')
    group.add_argument('--manual-gc-interval', type=int, default=0,
                       help='Training step interval to trigger manual garbage '
                       'collection. When the value is set to 0, garbage '
                       'collection is not triggered between training steps.')
    group.add_argument('--no-manual-gc-eval', action='store_false',
                       help='When using manual garbage collection, disable '
                       'garbage collection at the start and the end of each '
                       'evaluation run.', dest='manual_gc_eval')
    group.add_argument('--disable-tp-comm-split-ag', action='store_false',
                       help='Disables the All-Gather overlap with fprop GEMM.',
                       dest='tp_comm_split_ag')
    group.add_argument('--disable-tp-comm-split-rs', action='store_false',
                       help='Disables the Reduce-Scatter overlap with fprop GEMM.',
                       dest='tp_comm_split_rs')
    group.add_argument('--dynamic-flash-attn', action='store_true',
                       help='add packed flash attention')
    group.add_argument('--reduce-loss-per-iteration', action='store_true',
                       help='Whether reduce loss amond data parallel group per iteration.',
                       dest='reduce_loss_per_iteration')

    # ChunkFlow related args
    group.add_argument('--use-chunkflow', action='store_true',
                       help='Enable ChunkFlow')
    group.add_argument('--use-chunkflow-bucket', action='store_true',
                       help='Enable ChunkFlow With Bucket')
    group.add_argument('--chunk-size', type=int, default=-1,
                       help='Chunk size for training')
    group.add_argument('--chunks-kept', type=int, default=-1,
                       help="How many chunks' activation should be preserved in memory")
    group.add_argument('--chunkflow-balance-strategy', type=str, default='global-batch',
                       choices=['micro-batch', 'global-batch', "none"],
                       help='Chunkflow balance strategy')
    # Smart batching args
    group.add_argument('--use-smart-batching', action='store_true',
                       help='Enable Smart Batching')
    group.add_argument('--use-smart-batching-bucket', action='store_true',
                       help='Enable Smart Batching With Bucket Optimization')
    group.add_argument('--smart-batching-balance-strategy', type=str, default='global-batch',
                       choices=['global-batch', 'micro-batch', 'none'],
                       help='Smart Batching Balance Strategy')

    group.add_argument('--dynamic-padding-full', action='store_true',
                       help='padding sequence to max sequence length to avoid some hang cases.')
    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--data-parallel-random-init', action='store_true',
                       help='Enable random initialization of params '
                       'across data parallel ranks')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    group.add_argument('--init-method-xavier-uniform', action='store_true',
                       help='Enable Xavier uniform parameter initialization')
    group.add_argument('--init-clip-std', action='store_true',
                       help='clip to init range to [-3sigma, 3sigma] to improve training stability')

    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learning rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'inverse-square-root', 'WSD'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-wsd-decay-style', type=str, default='exponential',
                       choices=['exponential', 'linear', 'cosine'],
                       help='Decay style for the annealing phase of WSD'),
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--lr-decay-samples', type=int, default=None,
                       help='number of samples to decay learning rate over,'
                       ' If None defaults to `--train-samples`')
    group.add_argument('--lr-wsd-decay-samples', type=int, default=None,
                       help='number of samples for the annealing phase in the wsd schedule')
    group.add_argument('--lr-wsd-decay-iters', type=int, default=None,
                       help='number of iterations for the annealing phase in the wsd schedule')
    group.add_argument('--lr-warmup-fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    group.add_argument('--lr-warmup-iters', type=int, default=0,
                       help='number of iterations to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-samples', type=int, default=0,
                       help='number of samples to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-init', type=float, default=0.0,
                       help='Initial value for learning rate warmup. The '
                       'scheduler starts warmup from this value.')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the'
                       '--lr-warmup-* arguments above')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minimum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--override-opt_param-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-opt_param-scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')
    group.add_argument('--decoupled-lr', type=float, default=None,
                       help='Separate learning rate for the input and output layer')
    group.add_argument('--decoupled-min-lr', type=float, default=None,
                       help='Minimum value for learning rate for the input and output layer. The scheduler'
                       'clip values below this threshold')

    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', '--persistent-save-interval', type=int, default=None,
                       help='Number of iterations between persistent checkpoint saves.')
    group.add_argument('--no-save-optim', action='store_true', default=None,
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true', default=None,
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--allow-partial-load-optim', action='store_true', default=None,
                       help='Allow partial load of optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true', default=None,
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--non-persistent-save-interval', type=int, default=None,
                       help='Number of iterations between non-persistent saves.')
    group.add_argument('--non-persistent-ckpt-type', type=str, default=None,
                       choices=['global', 'local', 'in_memory', None],
                       help='Type of non-persistent model checkpoints. '
                           '"global" - Saved as a standard checkpoint (e.g., on Lustre) with old checkpoints being removed. '
                           '"local" - [TBD] Each rank saves a portion of the checkpoint locally (e.g., on SSD/ramdisk). '
                           '"in_memory" - [TBD] A special kind of local checkpoint that avoids serialization. '
                           'None - No non-persistent checkpointing (default option).')
    group.add_argument('--non-persistent-global-ckpt-dir', type=str, default=None,
                       help='Directory containing global non-persistent model checkpoints.')
    group.add_argument('--non-persistent-local-ckpt-dir', type=str, default=None,
                       help='Directory containing local non-persistent model checkpoints.')
    group.add_argument('--non-persistent-local-ckpt-algo', type=str, default='fully_parallel',
                       choices=['fully_parallel', 'atomic'],
                       help='Algorithm for local non-persistent checkpointing.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Directory containing a pretrained model checkpoint for finetuning.')
    group.add_argument('--ckpt-step', type=int, default=None,
                       help='Checkpoint step to load model from.')
    group.add_argument('--no-initialization', action='store_false',
                       help='Do not perform initialization when building model, '
                       'can reduce startup time when definitely loading from a '
                       'checkpoint',
                       dest='perform_initialization')
    group.add_argument('--use-checkpoint-args', action='store_true',
                       help='Override any command line arguments with arguments '
                       'from the checkpoint')
    group.add_argument('--exit-on-missing-checkpoint', action='store_true',
                       help="If '--load' is set, but checkpoint is not found "
                       "(e.g., path typo), then exit instead of random "
                       "initialization.")
    group.add_argument('--use-dist-ckpt', action='store_true',
                       dest='use_dist_ckpt_deprecated',
                       help='Deprecated: see --ckpt-format.')
    group.add_argument('--auto-detect-ckpt-format', action='store_true',
                       help='Determine if the checkpoint format is in legacy or distributed format.'
                            ' If False, expects distributed checkpoint iff args.ckpt_format != "torch".'
                            ' Might slow down loading a bit (double rank0 ckpt load).')
    group.add_argument('--dist-ckpt-format',
                       dest='dist_ckpt_format_deprecated',
                       help='Deprecated: see --ckpt-format.')
    group.add_argument('--ckpt-format', default='torch_dist',
                       choices=['torch', 'torch_dist', 'zarr'],
                       help='Checkpoint format to use.')
    group.add_argument('--ckpt-convert-format', default=None,
                       choices=['torch', 'torch_dist', 'zarr'],
                       help='Checkpoint format for conversion.')
    group.add_argument('--ckpt-convert-save', default=None,
                       help='Save directory for converted checkpoint.')
    group.add_argument('--ckpt-convert-update-legacy-dist-opt-format', action='store_true',
                       help='When loading a checkpoint, update the legacy format '
                       'for the distributed optimizer, which previously used a '
                       'merged param/grad buffer and a different bucket mapping. '
                       'The legacy format was deprecated on Feb 13, 2024.')
    group.add_argument('--ckpt-fully-parallel-save', action='store_true',
                       dest='ckpt_fully_parallel_save_deprecated',
                       help='Deprecated: see --no-ckpt-fully-parallel-save.')
    group.add_argument('--no-ckpt-fully-parallel-save', action='store_false',
                       dest='ckpt_fully_parallel_save',
                       help='Disable applying full save parallelization across DP for'
                            ' distributed checkpoints. Depending on ckpt format'
                            ' might decrease the number of files in the checkpoint.'
                            ' Makes DistributedOptimizer checkpoint non-reshardable.')
    group.add_argument('--async-save', action='store_true', default=None,
                       help='Apply async checkpointing save. Currently works only with'
                            '`torch_dist` distributed checkpoint format.')
    group.add_argument('--ckpt-fully-parallel-load', action='store_true',
                       help='Apply full load parallelization across DP for'
                            ' distributed checkpoints.')
    group.add_argument('--ckpt-assume-constant-structure', action='store_true',
                       help='If the model and optimizer state dict structure is'
                            'constant throughout a *single training job*, it allows for'
                            'different checkpointing performance optimizations.')
    group.add_argument('--dist-ckpt-strictness', type=str, default='assume_ok_unexpected',
                       choices=[e.value for e in StrictHandling],
                       help='Determine handling of key mismatch during checkpoint load.'
                            ' Check StrictHandling docs for flags meaning.'
                            ' NOTE: This flag controls only distributed checkpoint'
                            ' load from storage, not loading state dict into the model.')
    group.add_argument('--dist-ckpt-split-model-and-optim', action='store_true', help='If set to true, when saving the dist-ckpt, the model state and the optim state will be saved separately, rather than being averaged together. May reduce performace of saving due to unbalance')
    group.add_argument('--allow-overwrite', action='store_true', help='Allow overwrite existing checkpoint.')
    group.add_argument('--dataloader-save', type=str, default=None,
                       help='Path to the dataset state save dir. "$" means relative to the save directory')
    group.add_argument('--dataloader-load', type=str, default=None,
                       help='Path to the dataset state load dir. "$" means relative to the load directory')
    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--initial-loss-scale', type=float, default=2**32,
                       help='Initial loss-scale for dynamic loss scaling.')
    group.add_argument('--min-loss-scale', type=float, default=1.0,
                       help='Minimum loss scale for dynamic loss scaling.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--fp32-residual-connection', action='store_true',
                       help='Move residual connections to fp32.')
    group.add_argument('--apply-query-key-layer-scaling', action='store_true',
                       help='Scale Q * K^T by 1 / layer-number. '
                       'Useful for fp16 training. Also sets `attention_softmax_in_fp32` to True.')
    group.add_argument('--attention-softmax-in-fp32', action='store_true',
                       help='Run attention masking and softmax in fp32.')
    group.add_argument('--accumulate-allreduce-grads-in-fp32',
                       action='store_true',
                       help='Gradient accumulation and all-reduce in fp32.')
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')
    group.add_argument('--bf16-memory-efficient-cross-entropy', action='store_true',
                       help='If set, enable memory efficient bfloat16 cross-entropy for '
                       'lm head.',
                       dest='bf16_memory_efficient_cross_entropy')
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--encoder-tensor-model-parallel-size', type=int, default=0,
                       help='Degree of tensor model parallelism for the encoder.')
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--encoder-pipeline-model-parallel-size', type=int, default=0,
                       help=('Degree of pipeline model parallelism in the encoder. This is '
                             'independent of the amount of pipeline in the decoder.'))
    group.add_argument('--pipeline-model-parallel-split-rank',
                       type=int, default=None,
                       help=('Rank where encoder and decoder should be split. '
                             'Deprecated; use --encoder-pipeline-model-parallel-size instead.'))
    group.add_argument('--decoder-first-pipeline-num-layers',
                       type=int, default=None,
                       help=('The number of transformer layers on the first pipeline stage of the decoder. '
                       'Default None is even split of transformer layers across all pipeline stages'))
    group.add_argument('--decoder-last-pipeline-num-layers',
                       type=int, default=None,
                       help=('The number of transformer layers on the last pipeline stage of the decoder. '
                       'Default None is even split of transformer layers across all pipeline stages'))
    group.add_argument('--pipeline-stages-num-layers-override', type=int, nargs='+', default=None,
                       help=('Override the number of transformer layers for each pipeline stage. '
                       'Default None is not using override. Example: --pipeline-stages-num-layers-override 10 10 10'))
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--pipeline-layers', type=str, default=None,
                       help='Number of layers for different pipeline stages when using unbalanced (v)PP')
    group.add_argument('--allow-padding-num-layers', action='store_true',
                       help='Allow padding num_layers for pipeline parallelism',
                       dest='allow_padding_num_layers')
    group.add_argument('--no-overlap-p2p-communication', action='store_false',
                       help='overlap pipeline parallel communication with forward and backward chunks',
                       dest='overlap_p2p_comm')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Timeout minutes for torch.distributed.')
    group.add_argument('--overlap-grad-reduce', action='store_true',
                       default=False, help='If set, overlap DDP grad reduce.')
    group.add_argument('--defer-embedding-wgrad-compute', action='store_true',
                       default=False, help='If set, defers the vocabulary projection linear layer weight'
                       'gradient compute to pipeline flush.', dest='defer_embedding_wgrad_compute')
    group.add_argument('--wgrad-deferral-limit', type=int, default=0, help='Number of micro-batches for which'
                       'weight gradient computation of vocabulary projection is deferred, defaults to 0 which'
                       'means all the micro-batches are deferred. Invalid if `defer-embedding-wgrad-compute`'
                       'is not set')
    group.add_argument('--no-align-grad-reduce', action='store_false',
                       help='If not set, all PP stages will launch gradient reduces simultaneously. '
                       'Otherwise, each PP stage will independently launch as needed.',
                       dest='align_grad_reduce')
    group.add_argument('--ddp-bucket-size', type=int, default=None,
                       help='Bucket size for data-parallel communication')
    group.add_argument('--ddp-average-in-collective', action='store_true',
                       default=False, help='If set, average directly in data-parallel communication collective.')
    group.add_argument('--overlap-param-gather', action='store_true',
                       default=False, help='If set, overlap param all-gather in distributed optimizer.')
    group.add_argument('--overlap-param-gather-with-optimizer-step', action='store_true',
                       default=False, help='If set, overlap param all-gather of first bucket with optimizer step.')
    group.add_argument('--no-align-param-gather', action='store_false',
                       help='If not set, all PP stages will launch param all-gathers simultaneously. '
                       'Otherwise, each PP stage will independently launch as needed.',
                       dest='align_param_gather')
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='If not set, use scatter/gather to optimize communication of tensors in pipeline.',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--use-ring-exchange-p2p', action='store_true',
                       default=False, help='If set, use custom-built ring exchange '
                       'for p2p communications. Note that this option will require '
                       'a custom built image that support ring-exchange p2p.')
    group.add_argument('--local-rank', type=int, default=int(os.getenv('LOCAL_RANK', '0')),
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead.Also turns on '
                       '--use-cpu-initialization flag. This is for '
                       'external DDP manager.' )
    group.add_argument('--standalone-embedding-stage', action='store_true',
                       default=False, help='If set, *input* embedding layer '
                       'is placed on its own pipeline stage, without any '
                       'transformer layers. (For T5, this flag currently only '
                       'affects the encoder embedding.)')
    group.add_argument('--use-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer.')
    group.add_argument('--context-parallel-size', type=int, default=1,
                       help='Degree of context parallelism.')
    group.add_argument('--context-parallel-strategy', default='ring-attn',
                       choices=['ring-attn', 'ulysses', 'kvgather'],
                       help='Strategy of context parallel.')
    group.add_argument('--nccl-communicator-config-path', type=str, default=None,
                       help='Path to the yaml file with NCCL communicator '
                       'configurations. The number of min/max thread groups and thread '
                       'group cluster size of each communicator can be configured by '
                       'setting `min_ctas`, `max_ctas`, and `cga_cluster_size`.')
    group.add_argument('--use-tp-pp-dp-mapping', action='store_true', default=False,
                        help='If set, distributed ranks initialize order is changed '
                        'from tp-dp-pp to tp-pp-dp. Make sure EP and CP aren\'t used '
                        'with this option enabled')
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')
    group.add_argument("--test-mode", action="store_true", help='Run all real-time test alongside the experiment.')
    group.add_argument('--skip-train', action='store_true',
                       default=False, help='If set, bypass the training loop, '
                       'optionally do evaluation for validation/test, and exit.')

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument('--data-path', nargs='*', default=None,
                       help='The weight and prefix list for a set of train, validation, and test'
                       'datasets which split according to --split. The accepted formats are: '
                       '(1) a single prefix, '
                       '(2) a list of weight prefix pairs e.g. weight1 prefix1 weight2 prefix2, '
                       '(3) a list of prefixes e.g. prefix1 prefix2. '
                       'For (3), weights are inferred from the lengths of the contributing datasets. '
                       'This argument is exclusive to the other independent --*-data-path arguments.')
    group.add_argument('--renormalize-blend-weights', action='store_true',
                       help='Renormalize the blend weights to account for the mid-level dataset '
                       'oversampling done to ensure fulfillment of the requested number of '
                       'samples. Use this option if prompted. Defaults to False for backward '
                       'comparability in the data sample order.')
    group.add_argument('--split', type=str, default=None,
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    group.add_argument('--train-data-path', nargs='*', default=None,
                       help='The weight and prefix list for an independent train dataset. '
                       'Follows the same pattern rules as --data-path.')
    group.add_argument('--valid-data-path', nargs='*', default=None,
                       help='The weight and prefix list for an independent validation dataset. '
                       'Follows the same pattern rules as --data-path.')
    group.add_argument('--test-data-path', nargs='*', default=None,
                       help='The weight and prefix list for an independent test dataset. '
                       'Follows the same pattern rules as --data-path.')
    group.add_argument('--data-cache-path', default=None,
                       help='Path to a directory to hold cached index files.')
    group.add_argument('--no-mmap-bin-files', action='store_false',
                       help='Disable mmap-ing of .bin files.',
                       dest='mmap_bin_files')
    group.add_argument('--mock-data', action='store_true',
                       help='Skip data loading and validation and opt for artificial '
                       'generation of mock data when an implementation is available.')
    group.add_argument('--vocab-size', type=int, default=None,
                       help='Size of vocab before EOD or padding.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file.')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file.')
    group.add_argument('--vocab-extra-ids', type=int, default=0,
                       help='Number of additional vocabulary tokens. '
                            'They are used for span masking in the T5 model')
    group.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--encoder-seq-length', type=int, default=None,
                       help='Maximum encoder sequence length to process.'
                       'This should be exclusive of --seq-length')
    group.add_argument('--decoder-seq-length', type=int, default=None,
                       help="Maximum decoder sequence length to process.")
    group.add_argument('--retriever-seq-length', type=int, default=256,
                       help='Maximum sequence length for the biencoder model '
                       'for retriever')
    group.add_argument('--sample-rate', type=float, default=1.0,
                       help='sample rate for training data. Supposed to be 0 '
                            ' < sample_rate < 1')
    group.add_argument('--mask-prob', type=float, default=0.15,
                       help='Probability of replacing a token with mask.')
    group.add_argument('--short-seq-prob', type=float, default=0.1,
                       help='Probability of producing a short sequence.')
    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")
    group.add_argument('--prefetch-factor', type=int, default=2,
                       help="Number of batches loaded in advance by each worker.")
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'HuggingFaceTokenizer',
                                'Llama2Tokenizer',
                                'TikTokenizer',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--tiktoken-pattern', type=str, default=None,
                       help='Which tiktoken pattern to use. Options: [v1, v2]')
    group.add_argument('--tiktoken-num-special-tokens', type=int, default=1000,
                       help='Number of special tokens in tiktoken tokenizer')
    group.add_argument('--tiktoken-special-tokens', type=str, nargs='+', default=None,
                       help='List of tiktoken special tokens, needs to have ["<unk>", "<s>", "</s>"]')
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token.')
    group.add_argument('--eod-mask-loss', action='store_true',
                       help='Mask loss for the end of document tokens.')
    group.add_argument('--no-create-attention-mask-in-dataloader', action='store_false',
                       help='If set, do not create attention_masks in dataloader.',
                       dest='create_attention_mask_in_dataloader')
    group.add_argument('--num-dataset-builder-threads', type=int, default=1,
                       help='Number of parallel threads per rank for dataset builder')
    group.add_argument('--s3-cache-path', type=str, default=None,
                       help='Path to cache index files when using s3 dataloader')
    group.add_argument('--use-dynamic-collate', action='store_true',
                       help='If set, use dynamic collate.')
    group.add_argument('--mix-pretrain-sft', action='store_true',
                       help='Mix pretrain and SFT data in SFT training')
    group.add_argument('--mask-prompt', action='store_true',
                       help='Mask loss for user prompt.')
    group.add_argument('--mask-chatml-role', action='store_true',
                       help='Mask loss for ChatML role strings')
    group.add_argument('--mask-vision-response', action='store_true',
                       help='Mask loss for `✿VISION✿` string.')
    return parser


def _add_autoresume_args(parser):
    group = parser.add_argument_group(title='autoresume')

    group.add_argument('--adlr-autoresume', action='store_true',
                       help='Enable autoresume on adlr cluster.')
    group.add_argument('--adlr-autoresume-interval', type=int, default=1000,
                       help='Intervals over which check for autoresume'
                       'termination signal')

    return parser


def _add_biencoder_args(parser):
    group = parser.add_argument_group(title='biencoder')

    # network size
    group.add_argument('--ict-head-size', type=int, default=None,
                       help='Size of block embeddings to be used in ICT and '
                        'REALM (paper default: 128)')
    group.add_argument('--biencoder-projection-dim', type=int, default=0,
                       help='Size of projection head used in biencoder (paper'
                        ' default: 128)')
    group.add_argument('--biencoder-shared-query-context-model', action='store_true',
                        help='Whether to share the parameters of the query '
                        'and context models or not')

    # checkpointing
    group.add_argument('--ict-load', type=str, default=None,
                       help='Directory containing an ICTBertModel checkpoint')
    group.add_argument('--bert-load', type=str, default=None,
                       help='Directory containing an BertModel checkpoint '
                       '(needed to start ICT and REALM)')

    # data
    group.add_argument('--titles-data-path', type=str, default=None,
                       help='Path to titles dataset used for ICT')
    group.add_argument('--query-in-block-prob', type=float, default=0.1,
                       help='Probability of keeping query in block for '
                       'ICT dataset')
    group.add_argument('--use-one-sent-docs', action='store_true',
                       help='Whether to use one sentence documents in ICT')
    group.add_argument('--evidence-data-path', type=str, default=None,
                       help='Path to Wikipedia Evidence frm DPR paper')

    # training
    group.add_argument('--retriever-report-topk-accuracies', nargs='+', type=int,
                        default=[], help="Which top-k accuracies to report "
                        "(e.g. '1 5 20')")
    group.add_argument('--retriever-score-scaling', action='store_true',
                       help='Whether to scale retriever scores by inverse '
                        'square root of hidden size')

    # faiss index
    group.add_argument('--block-data-path', type=str, default=None,
                       help='Where to save/load BlockData to/from')
    group.add_argument('--embedding-path', type=str, default=None,
                       help='Where to save/load Open-Retrieval Embedding'
                        ' data to/from')

    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,
                       help='How large of batches to use when doing indexing '
                       'jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,
                       help='After how many batches should the indexer '
                       'report progress')
    return parser

def _add_code_args(parser):
    group = parser.add_argument_group(title="code")
    group.add_argument('--fim-rate', type=float, default=0.,
                       help='Probability to convert a training sample into a "Fill-in-the-Middle" format. Must be between 0 and 1.')
    group.add_argument('--fim-spm-rate', type=float, default=0.5,
                       help='Probability that the a FIM sample uses the SPM format over the PSM format. '
                            'At 1, exclusively train with SPM. At 0, exclusively train with PSM')
    group.add_argument('--fim-data-list-path', type=str, default="",
                       help='the list file of all the data that need to use fim')
    return parser

def _add_vision_args(parser):
    group = parser.add_argument_group(title="vision")

    # general vision arguements
    group.add_argument('--num-classes', type=int, default=1000,
                       help='num of classes in vision classificaiton task')
    group.add_argument('--img-h', type=int, default=224,
                       help='Image height for vision classification task')
    group.add_argument('--img-w', type=int, default=224,
                       help='Image height for vision classification task')
    group.add_argument('--img-dim', type=int, default=224,
                       help='Image dimension for vision task')
    group.add_argument('--img-token-span', type=int, default=32,
                       help='patch dimension used in vit')
    group.add_argument('--vision-model-config', type=str, default=None,
                       help='Vision model config path')
    group.add_argument('--freeze-vision-nlayers', type=int, default=None,
                       help='freeze the first n layers of the model ')
    group.add_argument('--freeze-llm-encoder', action='store_true', default=False,
                       help='freeze LLM encoder of the model ')
    group.add_argument('--freeze-llm-embedding', action='store_true', default=False,
                       help='freeze LLM embedding of the model ')
    group.add_argument('--vision-lr-layer-decay', type=float, default=1.0,
                       help='lr layer decay of vision tower')
    group.add_argument('--add-clip-attention-pool', action='store_true',
                       help='whether add attention pooling for clip')
    group.add_argument('--rm-vit-cls-emb', action='store_true',
                       help='whether rm cls emb')
    group.add_argument('--num-channels', type=int, default=3,
                       help='Number of channels in input image data')
    group.add_argument('--patch-dim', type=int, default=16,
                       help='patch dimension')
    group.add_argument('--classes-fraction', type=float, default=1.0,
                       help='training with fraction of classes.')
    group.add_argument('--data-per-class-fraction', type=float, default=1.0,
                       help='training with fraction of data per class.')
    group.add_argument('--no-data-sharding', action='store_false',
                       help='Disable data sharding.',
                       dest='data_sharding')
    group.add_argument('--head-lr-mult', type=float, default=1.0,
                       help='learning rate multiplier for head during finetuning')

    # pretraining type and backbone selection`
    group.add_argument('--vision-pretraining', action='store_true',
                       help='flag to indicate vision pretraining')
    group.add_argument('--vision-pretraining-type', type=str, default='classify',
                       choices=['classify', 'inpaint', 'dino'],
                       help='pretraining objectives')
    group.add_argument('--vision-backbone-type', type=str, default='vit',
                       choices=['vit', 'mit', 'swin'],
                       help='backbone types types')
    group.add_argument('--swin-backbone-type', type=str, default='tiny',
                       choices=['tiny', 'base', 'h3'],
                       help='pretraining objectives')
    # inpainting arguments
    group.add_argument('--mask-type', type=str, default='random',
                       choices=['random', 'row'],
                       help='mask types')
    group.add_argument('--mask-factor', type=float, default=1.0,
                       help='mask size scaling parameter')

    # dino arguments
    group.add_argument('--iter-per-epoch', type=int, default=1250,
                       help='iterations per epoch')
    group.add_argument('--dino-local-img-size', type=int, default=96,
                       help='Image size for vision classification task')
    group.add_argument('--dino-local-crops-number', type=int, default=10,
                       help='Number of local crops')
    group.add_argument('--dino-head-hidden-size', type=int, default=2048,
                       help='Hidden dimension size in dino head')
    group.add_argument('--dino-bottleneck-size', type=int, default=256,
                       help='Bottle neck dimension in dino head ')
    group.add_argument('--dino-freeze-last-layer', type=float, default=1,
                       help='Freezing last layer weights')
    group.add_argument('--dino-norm-last-layer', action='store_true',
                       help='Disable Norm in last layer.')
    group.add_argument('--dino-warmup-teacher-temp', type=float, default=0.04,
                       help='warump teacher temperature')
    group.add_argument('--dino-teacher-temp', type=float, default=0.07,
                       help='teacher temperature')
    group.add_argument('--dino-warmup-teacher-temp-epochs', type=int, default=30,
                       help='warmup teacher temperaure epochs')

    return parser

def _add_dpo_args(parser):
    group = parser.add_argument_group(title="dpo")
    group.add_argument('--chat_format', type=str, default="chatml",
                       help='chatml/human-assistant')
    group.add_argument('--use_ipo', type=int, default=0,
                       help='use ipo algo.')
    group.add_argument('--use_best_sft', action='store_true')
    group.add_argument('--dpo_loss_weight', type=float, default=0,
                       help='use best sft along with dpo')
    group.add_argument('--dpo_label_smoothing', type=float, default=0,
                       help='use best sft along with dpo')
    group.add_argument('--dpo_weight', default=0.1, type=float)
    group.add_argument('--ref_load', type=str, default=None,
                       help='ref load checkpoint when use dpo')
    group.add_argument('--max_response', default=1, type=int)
    group.add_argument('--batch-response', action='store_true',
                       help='Convert response list to batch of data, make sure tensor shape is right for pp>1 case.' )
    return parser


def _add_moe_args(parser):
    group = parser.add_argument_group(title="moe")
    group.add_argument('--expert-model-parallel-size', type=int, default=1,
                       help='Degree of expert model parallelism.')
    group.add_argument('--expert-tensor-parallel-size', type=int, default=None,
                       help='Degree of expert model parallelism. Default is None, which will be set to the value of --tensor-model-paralle-size.')
    group.add_argument('--num-experts', type=int, default=None,
                       help='Number of Experts in MoE (None means no MoE)')
    group.add_argument('--num-shared-experts', type=int, default=None,
                       help='Number of shared Experts in MoE (None means no shared experts)')
    group.add_argument('--moe-shared-expert-intermediate-size', type=int, default=None,
                       help='Shared expert total ffn hidden size. '
                       'It should be equal to "num_shared_experts * ffn_size_of_each_shared_expert" if there are multiple shared experts. '
                       'None means no shared expert.')
    group.add_argument('--moe-shared-expert-overlap', action='store_true',
                       help='Enable overlapping between shared expert computations and dispatcher communications. '
                       'Without this, the shared epxerts execute after the routed experts. '
                       'Only effective when moe-shared-expert-intermediate-size is set.')
    group.add_argument('--moe-router-load-balancing-type', type=str,
                       choices=['aux_loss', 'sinkhorn', 'none'],
                       default='aux_loss',
                       help='Determines the load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss".')
    group.add_argument('--moe-lbl-all-reduce-overlap', action='store_true',
                       default=False,
                       help='Whether To Overlap the All Reduce in load balancing loss computation.')
    group.add_argument('--moe-router-topk', type=int, default=2,
                       help='Number of experts to route to for each token. The default is 2.')
    group.add_argument('--moe-router-aux-loss-fusion', action='store_true',
                       help='Fuse topk, softmax, bincount operations for router layer with aux loss load balancing and post softmax.')
    group.add_argument('--moe-router-pre-softmax', action='store_true',
                       help='Enable pre-softmax routing for MoE, which means softmax is before the top-k selection. By default, softmax is done after top-k.')
    group.add_argument('--moe-grouped-gemm', action='store_true',
                       help='When there are multiple experts per rank, launch multiple local GEMM kernels in multiple streams to improve the utilization and performance with GroupedLinear in TransformerEngine.')
    group.add_argument('--moe-aux-loss-coeff', type=float, default=0.0,
                       help='Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended.')
    group.add_argument('--moe-z-loss-coeff', type=float, default=None,
                       help='Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended.')
    group.add_argument('--moe-input-jitter-eps', type=float, default=None,
                       help='Add noise to the input tensor by applying jitter with a specified epsilon value.')
    group.add_argument('--moe-token-dispatcher-type', type=str,
                       choices=['allgather', 'alltoall', 'alltoall_seq'],
                       default='alltoall',
                       help="The type of token dispatcher to use. The default is 'allgather'. Options are 'allgather', 'alltoall' and 'alltoall_seq'. We recommend using 'alltoall' when applying expert parallelism. For more information, please refer to the documentation in core/moe/README.")
    group.add_argument('--moe-per-layer-logging', action='store_true',
                       help='Enable per-layer logging for MoE, currently supports auxiliary loss and z loss.')
    # Token dropping arguments
    group.add_argument('--moe-expert-capacity-factor', type=float, default=None,
                       help='The capacity factor for each expert, None means no token will be dropped.')
    group.add_argument('--moe-pad-expert-input-to-capacity', action='store_true',
                       help='Pads the input for each expert to match the expert capacity length, effective only after the --moe-expert-capacity-factor is set.')
    group.add_argument('--moe-token-drop-policy', type=str, default='probs', choices=['probs', 'position'],
                       help='The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.')
    group.add_argument('--moe-layer-recompute', action='store_true',
                       help='Enable checkpointing for moe_layer, should be used when memory is not sufficient.')
    group.add_argument('--moe-use-upcycling', action='store_true',
                       help='Load a checkpoint of a dense model, convert it into an MoE model, and save the converted model to the path specified by --save. '
                       'Upcycling is implemented on the top of distributed checkpointing, so it supports parallel modes different from the dense model.')
    group.add_argument('--moe-permute-fusion', action='store_true',
                       help='Fuse permute/unpermute operations for moe layer.')
    group.add_argument('--use-flashmoe-router', action='store_true',
                       help='Enable FlashMoE router for MoE. This is an experimental feature.')
    group.add_argument('--use-flash-moe', action='store_true',
                       help='Enable FlashMoE for MoE.')
    group.add_argument('--use-flashmoe-hep', action='store_true',
                       help='Enable FlashMoE hyper-expert-parallel for MoE.')
    group.add_argument('--flash-moe-ga-fusion', action='store_true', default=False,
                       help='If True, fuse experts computation and gradient accumulation')
    group.add_argument('--flash-moe-interleaving', action='store_true', default=False,
                       help='If True, interleave experts computation and EP communication')
    group.add_argument('--flash-moe-mem-saving-mlp', action='store_true',
                       default=False, help='Use memory-saving mlp of flashmoe')
    group.add_argument('--flash-moe-cpu-offload', action='store_true',
                       default=False, help='Use cpu offload mlp of flashmoe')
    group.add_argument('--flash-moe-offload-attention-overlap-ratio', type=float,
                       default=0.4, help='The ratio of offloading data to overlap attention.')
    group.add_argument('--moe-drop-across-ep', action='store_true',
                       help='Drop tokens across experts in the same parallel group.')
    group.add_argument('--moe-mock-router-sigma', type=float, default=None,
                       help='The sigma value used in mock router distribution')
    group.add_argument('--moe-expert-dropout', type=float, default=0.0,
                       help='dropout rate for expert during training.')
    group.add_argument('--moe-norm-topk-scores', action='store_true')
    group.add_argument('--moe-router-sync-tokens-per-expert-across-dp', action='store_true')
    group.add_argument('--moe-router-buffer-tokens-per-expert', action='store_true')

    group.add_argument('--moe-sequence-load-balance-weight', type=float, default=0.0)
    group.add_argument('--moe-no-norm-lbl-across-layer', action='store_true')
    return parser

def _add_mla_args(parser):
    group = parser.add_argument_group(title="mla")
    group.add_argument('--q-lora-rank', type=int, default=None,
                       help="Rank of Query tensor's low rank representation.")
    group.add_argument('--kv-lora-rank', type=int, default=32,
                       help="Rank of Key and Value tensors' low rank representation.")
    group.add_argument('--qk-head-dim', type=int, default=128,
                       help="Dimension of the head in the QK projection. q_head_dim = qk_head_dim + qk_pos_emb_head_dim")
    group.add_argument('--qk-pos-emb-head-dim', type=int, default=64,
                       help="Dimension of the position embedding in the QK projection.")
    group.add_argument('--v-head-dim', type=int, default=128,
                       help="Dimension of the head in the V projection.")
    group.add_argument('--rotary-scaling-factor', type=float, default=1.0,
                       help="Rotary scaling factor for the rotary embeddings.")

    return parser

def _add_experimental_args(parser):
    group = parser.add_argument_group(title='experimental')

    group.add_argument('--spec', type=str, default=None, nargs='*',
                       help='Specify the <module_location function_name> pair '
                       'that returns a spec to customize a model, transformer '
                       'block, or transformer layer, depending on the use case.'
                       'To use local spec specify local as the argument.'
                       'For more details, see the model class, '
                       '`transformer_block.py`, or `transformer_layer.py`')
    group.add_argument('--hybrid-attention-ratio', type=float, default=0.0,
                       help='Ratio of attention layers to total layers, in the '
                       'range [0.0, 1.0].')
    group.add_argument('--hybrid-mlp-ratio', type=float, default=0.0,
                       help='Ratio of mlp layers to total layers, in the '
                       'range [0.0, 1.0].')
    group.add_argument('--hybrid-override-pattern', type=str, default=None,
                       help='Force a specific hybrid layer pattern. If a value'
                       'greater than 0.0 is supplied to any of the hybrid ratio'
                       'arguments, then the number of each type of layer in the'
                       'override pattern must match number in the overidden'
                       'pattern')
    group.add_argument('--yaml-cfg', type=str, default=None,
                       help = 'Config file to add additional arguments')
    return parser


def _add_qwenvl_args(parser):
    group = parser.add_argument_group(title="qwenvl")
    group.add_argument('--qwenvl', action='store_true')
    group.add_argument('--qwenclip', action='store_true')
    group.add_argument('--vision-checkpoint-convert-mp', action='store_true',
                       help='convert vision checkpoint from non-mp to mp')
    group.add_argument('--vision-sequence-parallel', action='store_true',
                       help='Enable visual encoder sequence parallel optimization.')
    group.add_argument('--vision-redundant-sync', action='store_true',
                       help='Enable visual encoder do redundant synchronization.')
    group.add_argument('--vision-grad-checkpointing', action='store_true',
                       help='Enable visual encoder grad checkpointing.')
    group.add_argument('--spatial-patch-size', type=int, default=14,
                       help='spatial patch size of vision encoder')
    group.add_argument('--temporal-patch-size', type=int, default=2,
                       help='temporal patch size of vision encoder')
    group.add_argument('--spatial-merge-size', type=int, default=2,
                       help='addtional spatial merge size vision tokens after vision encoder')
    group.add_argument('--temporal-merge-size', type=int, default=4,
                       help='addtional temporal merge size vision tokens after vision encoder')
    group.add_argument('--image-mean', nargs="+", type=float, default=(0.48145466, 0.4578275, 0.40821073))
    group.add_argument('--image-std', nargs="+", type=float, default=(0.26862954, 0.26130258, 0.27577711))
    group.add_argument('--use-mrope', action='store_true',
                       help='whether to use mrope in transformer.')
    group.add_argument('--llm-rotary-tdim', type=int, default=16,
                       help='vision mrope temporal dimension.')
    group.add_argument('--llm-rotary-hdim', type=int, default=24,
                       help='vision mrope height dimension.')
    group.add_argument('--llm-rotary-wdim', type=int, default=24,
                       help='vision mrope width dimension.')
    group.add_argument('--block-causal-attention', action='store_true',
                       help='Enable LLM to use block causal attention.')
    group.add_argument('--clip-loss-type', type=str, default="clip", choices=['clip', 'siglip'],
                       help='Clip-style model loss type.')
    group.add_argument('--clip-output-dim', type=int, default=1024,
                       help='Unified clip output dimension.')
    group.add_argument('--clip-mask-ratio', type=float, default=0.5,
                       help='Vision token mask ratio in CLIP.')
    group.add_argument('--clip-local-loss', action='store_true',
                       help='Enable local loss calculation in CLIP.')
    group.add_argument('--clip-gather-with-grad', action='store_true',
                       help='Enable all_gather with grad in CLIP.')
    group.add_argument('--clip-llm-pooling', action='store_true',
                       help='Enable pooling for LLM representation in CLIP.')

    group.add_argument('--min-pixels', type=float, default=4 * 28 * 28,
                       help='minimum pixels for image')
    group.add_argument('--max-pixels', type=float, default=16384 * 28 * 28,
                       help='maximum pixels for image')
    group.add_argument('--use-shanghai-oss', action='store_true',
                       help='Change data oss path to shanghai oss path')
    return parser


def _add_fun_base_args(parser):
    """Add FunBase specific arguments."""
    group = parser.add_argument_group(title="FunBase")
    
    # Audio processing parameters
    group.add_argument('--audio-encoder-type', type=str, default="sensevoice",
                       choices=["none", "sensevoice", "sensevoicesmall", "qwen3-0.6b", "qwen3-2b"],
                       help='Type of audio encoder to use')
    group.add_argument('--audio-input-size', type=int, default=128,
                       help='Input feature dimension for audio')
    group.add_argument('--audio-linear-units', type=int, default=1280,
                       help='Hidden dimension for audio encoder')
    group.add_argument('--audio-attention-heads', type=int, default=20,
                       help='Number of attention heads for audio encoder')
    group.add_argument('--audio-num-blocks', type=int, default=32,
                       help='Number of encoder blocks for audio')
    group.add_argument('--audio-downsample-rate', type=int, default=4,
                       help='Downsampling rate for audio features')
    group.add_argument('--audio-use-padmask', action='store_true',
                       help='Whether to use padding mask for audio')
    group.add_argument('--audio-encoder-no-use-fused-attention', action='store_true', default=False,
                       help='Whether to use fused attention for audio encoder')
    group.add_argument('--audio-encoder-use-packed-attention', action='store_true', default=False,
                       help='Whether to use packed attention for audio encoder')
    
    # Tokenizer parameters
    group.add_argument('--audio-tokenizer-version', type=str, default="25hz",
                       help='Version of the audio tokenizer')
    group.add_argument('--audio-tokenizer-frame-rate', type=float, default=25,
                       help='Frame rate for audio tokenizer')
    group.add_argument('--audio-tokenizer-code-book-size', type=int, default=6761,
                       help='Code book size for audio tokenizer')
    
    # Audio adaptor parameters
    group.add_argument('--audio-encoder-dim', type=int, default=1280,
                       help='Output dimension of audio encoder')
    group.add_argument('--audio-llm-dim', type=int, default=None,
                       help='Input dimension for language model, default to be the same as llm hidden size if not specified')
    group.add_argument('--audio-adaptor-downsample-rate', type=int, default=2,
                       help='Downsampling rate for audio adaptor')
    group.add_argument('--audio-ffn-dim', type=int, default=2048,
                       help='Hidden dimension for audio adaptor')
    group.add_argument('--audio-adaptor-num-layers', type=int, default=2,
                       help='Number of layers in audio adaptor')
    group.add_argument('--audio-attention-dropout-rate', type=float, default=0.0,
                       help='Dropout rate for attention in audio adaptor')
    group.add_argument('--audio-dropout-rate', type=float, default=0.0,
                       help='General dropout rate for audio adaptor')

    group.add_argument('--audio-no-freeze-encoder', action='store_true', default=False,
                       help='Whether to freeze the audio encoder')
    group.add_argument('--audio-encoder-lr-scale', type=float, default=None,
                       help='LR scaling factor for audio encoder.')
    group.add_argument('--audio-freeze-adaptor', action='store_true', default=False,
                       help='Whether to freeze the audio adaptor')
    group.add_argument('--audio-adaptor-lr-scale', type=float, default=None,
                       help='LR scaling factor for audio adaptor.')
    group.add_argument('--audio-recompute-encoder', action='store_true', default=False,
                       help='Whether to recompute the audio encoder activations')
    group.add_argument('--audio-freeze-decoder', action='store_true', default=False,
                       help='Whether to freeze the decoder')
    group.add_argument('--audio-decoder-lr-scale', type=float, default=None,
                       help='LR scaling factor for decoder.')
    group.add_argument('--audio-input-output-lr-scale', type=float, default=None,
                       help='LR scaling factor for audio input and output.')
    group.add_argument('--audio-encoder-adaptor-sliding-window-size', type=int, default=10000000,
                       help='Sliding window size for audio encoder adaptor')
    group.add_argument('--audio-encoder-adaptor-sliding-window-left-context', type=int, default=0,
                       help='Left context for audio encoder adaptor')
    group.add_argument('--audio-encoder-adaptor-bucketing-threshold', type=int, default=10000000,
                       help='Bucketing threshold for audio encoder adaptor')

    # Attention configuration
    group.add_argument('--audio-att-type', type=str, default="self_att_fsmn_sdpa",
                       help='Attention type to use for audio')
    group.add_argument('--audio-kernel-size', type=int, default=31,
                       help='Kernel size for FSMN')
    group.add_argument('--audio-sanm-shift', type=int, default=0,
                       help='Shift parameter for SANM')
    group.add_argument('--audio-max-position-embeddings', type=int, default=2048,
                       help='Maximum position embeddings for RoPE in audio')
    group.add_argument('--audio-rope-theta', type=float, default=10000,
                       help='Theta parameter for RoPE in audio')
    
    # Dual head parameters
    group.add_argument('--audio-is-dual-channel', action='store_true', default=False,
                       help='Whether to use dual channel for audio')
    group.add_argument('--no-audio-output-head', action='store_true', default=False,
                       help='Disable audio embedding and audio output head '
                       '(useful when only text output is needed).')
    group.add_argument('--cosyvoice-codex-use-text-token-input', action='store_true', default=False,
                       help='Feed teacher-forced main-LLM text label tokens through the codec LLM text '
                       'embedding instead of projecting main-LLM hidden states into the codec LLM. '
                       'The original hidden-state path remains the default.')
    
    # Training parameters
    group.add_argument('--length-normalized-loss', action='store_true',
                       help='Whether to use length normalized loss')
    group.add_argument('--audio-encoder-activation-checkpoint', action='store_true',
                       help='Whether to use activation checkpointing for audio encoder')
    group.add_argument('--feat-permute', action='store_true', default=True,
                       help='Whether to permute features for audio encoder')
    
    # Dataset parameters
    group.add_argument('--fun-audio-dataset', action='store_true',
                       help='Use FunASR audio dataset')
    group.add_argument('--fun-audio-config', type=str, default=None,
                       help='Path to FunASR audio dataset configuration')
    
    group.add_argument('--fun-dataset-config-path', type=str, default=None,
                       help='Path to the fun dataset config file')
    group.add_argument('--fun-no-pad-sample-to-seq-length', action='store_true', default=False,
                       help='Do not pad sample to sequence length. Padding is more efficient for fused attn '
                       'but might cause unbalanced moe experts.')
    group.add_argument('--fun-no-pad-speech-to-bucket', action='store_true', default=False,
                       help='Do not pad speech to bucket.')
    
    return parser


def _add_lora_args(parser):
    """Add LoRA (Low-Rank Adaptation) arguments."""
    group = parser.add_argument_group(title='LoRA')
    group.add_argument('--lora-rank', type=int, default=0,
                       help='LoRA rank. 0 means LoRA is disabled.')
    group.add_argument('--lora-alpha', type=float, default=16.0,
                       help='LoRA scaling factor alpha. Effective scaling = alpha / rank.')
    group.add_argument('--lora-dropout', type=float, default=0.0,
                       help='Dropout rate applied to LoRA input.')
    group.add_argument('--lora-freeze', action='store_true', default=False,
                       help='Inject and load LoRA adapters but keep their parameters frozen. '
                       'Useful when a LoRA-only checkpoint must be reconstructed while only '
                       'training other add-on modules.')
    group.add_argument('--lora-skip-linear-attention-out-proj', action='store_true',
                       default=False,
                       help='Do not inject LoRA into GatedDeltaNet/linear-attention out_proj. '
                       'Full-attention output projections and linear-attention QKV/Z LoRA '
                       'remain enabled. Intended for isolating RowParallel LoRA backward issues.')
    group.add_argument('--lora-save-only-trainable', action='store_true', default=False,
                       help='Save only trainable parameters (LoRA + encoder + adaptor) '
                       'in checkpoint instead of the full model.')
    group.add_argument('--lora-load-path', type=str, default=None,
                       help='Path to load a LoRA-only checkpoint for resuming LoRA training.')
    group.add_argument('--lora-resume-base-load', type=str, default=None,
                       help='Path to the original base model checkpoint for resuming '
                       'from a LoRA-only checkpoint (saved with --lora-save-only-trainable). '
                       'Base model weights will be loaded first, then LoRA weights from --load.')
    group.add_argument('--lora-overlay-load', type=str, default=None,
                       help='Path to a distributed checkpoint whose LoRA tensors are overlaid '
                       'after the main --load checkpoint. Only keys containing .lora_ are '
                       'loaded; encoder/adaptor/base weights in the overlay are ignored.')
    return parser
