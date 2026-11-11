# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

from megatron.core.transformer.moe.flashmoe_router import (
    batched_dropped_token_ratio, clear_num_dropped_tokens,
    get_load_balancing_loss, save_load_balancing_loss, clear_load_balancing_loss, batched_load_balancing_loss,
    clear_router_z_loss, batched_router_z_loss,
)

stimer = StragglerDetector()

def find_last_router(model):
    last_router = None
    for name, module in model.named_children():
        # print(f"search {name}")
        if hasattr(module, "is_last_router"):
            last_router = module
        last_router_in_submodule = find_last_router(module)
        if last_router_in_submodule is not None:
            last_router = last_router_in_submodule
    return last_router

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention, args.diff_attention, args.use_flash_moe, args.fp8)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention, args.diff_attention,args.use_flash_moe)

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling
            )

    if args.moe_lbl_all_reduce_overlap:
        last_router = find_last_router(model)
        if last_router is not None:
            last_router.is_last_router = True

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    # need to get packed_seq_params on each pp rank?
    # if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
    #     return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    if args.reduce_loss_per_iteration:
        reporting_loss = loss.clone().detach()
        loss_dict = {'lm loss': (reporting_loss[0], reporting_loss[1])}
    else:
        reporting_loss = loss.clone().detach()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
        loss_dict = {'lm loss': (reporting_loss[0], reporting_loss[1])}

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        loss_dict
    )


def average_loss(loss, reduce_loss_per_iteration, group):
    if reduce_loss_per_iteration:
        averaged_loss = loss.clone().detach().view(1)
    else:
        reported_loss = loss.clone().detach().view(1)
        torch.distributed.all_reduce(reported_loss, group=group)
        averaged_loss = reported_loss / torch.distributed.get_world_size(group)
    return averaged_loss

def moe_loss_func(loss_mask, output_tensor=None):
    # NOTE: For pipeline parallelism this function will be run on the
    # non-final stages to calculate load balancing loss contribution
    # for the MoE layers within the stage. For these cases, output_tensor
    # will be None.
    loss, local_num_tokens, loss_dict = (None, None,  {})
    if mpu.is_pipeline_last_stage():
        assert output_tensor is not None
        loss, local_num_tokens,  loss_dict = loss_func(loss_mask, output_tensor)

    args = get_args()

    # TODO: support unbalanced pipeline
    # NOTE: For unbalaced pipeline, directly return if there is
    # no transformer layer in this stage.
    num_layers_of_pipeline_stage, _ = \
        mpu.get_pipeline_num_layers_per_stage_and_offset(args)
    if num_layers_of_pipeline_stage == 0:
        return loss, local_num_tokens, loss_dict

    if args.recompute_granularity is not None:
        # Ignore load balancing loss contributions compute during
        # the forward pass if recompute is turned on.
        load_balancing_loss_data = get_load_balancing_loss()
        if args.num_layers * 2 == len(load_balancing_loss_data):
            raise NotImplementedError

    # TODO: add this when moe expert interval is implemented
    #  args.num_layers = args.num_layers // args.moe_expert_interval
    lbl, local_num_tokens_for_lbl = batched_load_balancing_loss()
    if args.moe_router_sync_tokens_per_expert_across_dp:
        lbl, sequence_lbl = lbl
        averaged_sequence_lbl = average_loss(sequence_lbl, args.reduce_loss_per_iteration, mpu.get_data_parallel_group(with_context_parallel=True))
        loss_dict['sequence load balancing loss'] = (averaged_sequence_lbl, local_num_tokens)
        # if local_num_tokens is not None:
        #     if args.use_flashmoe_router:
        #         local_num_tokens_for_lbl //= torch.distributed.get_world_size(
        #                                     group=mpu.get_data_parallel_group(with_context_parallel=True))
        #     assert local_num_tokens_for_lbl == local_num_tokens
        sequence_lbl = sequence_lbl * args.moe_aux_loss_coeff * args.moe_sequence_load_balance_weight
    else:
        if local_num_tokens is not None:
            if args.use_flashmoe_router and args.moe_drop_across_ep:
                local_num_tokens_for_lbl //= (mpu.get_expert_model_flashmoe_hep_parallel_world_size() if args.use_flashmoe_hep else mpu.get_expert_model_parallel_world_size())
            assert local_num_tokens_for_lbl == local_num_tokens

    rzl = batched_router_z_loss()
    clear_load_balancing_loss()
    lbl = lbl * args.moe_aux_loss_coeff # algin displayed lbl
    clear_router_z_loss()

    # TODO: add this when moe expert interval is implemented
    #  args.num_layers = args.num_layers * args.moe_expert_interval

    # Average the load balancing loss across data parallel
    # replicas and save for logging.
    averaged_lbl = average_loss(lbl, args.reduce_loss_per_iteration, mpu.get_data_parallel_group(with_context_parallel=True))
    averaged_rzl = average_loss(rzl, args.reduce_loss_per_iteration, mpu.get_data_parallel_group(with_context_parallel=True))

    loss_dict['load balancing loss'] = (averaged_lbl, local_num_tokens)
    loss_dict['router z loss'] = (averaged_rzl, local_num_tokens)

    if args.moe_expert_capacity_factor > 0:
        dtr = batched_dropped_token_ratio()
        clear_num_dropped_tokens()
        averaged_dtr = average_loss(dtr, args.reduce_loss_per_iteration, mpu.get_data_parallel_group(with_context_parallel=True))
        loss_dict['dropped ratio'] = (averaged_dtr, local_num_tokens)

    if args.moe_router_sync_tokens_per_expert_across_dp and args.moe_sequence_load_balance_weight > 0:
        # 如果使用 global lbl，且 local lbl 的 weight > 0，则将 local lbl 加入 global lbl
        # 限制 local 出现非常不均衡的情况
        lbl = lbl + sequence_lbl

    if args.moe_no_norm_lbl_across_layer:
        # 仍然保留之之前计算过程中的对 layer 进行 averge，使得 plot 出的 lbl 数值范围保持一致
        # 如果不将 lbl 取 layer 的均值，则在此处再乘以 args.num_layers
        lbl = lbl * args.num_layers

    # Compute the total loss, if necessary.
    total_loss = loss + lbl * local_num_tokens + rzl * local_num_tokens if loss is not None else lbl + rzl
    return total_loss, local_num_tokens, loss_dict


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels,packed_seq_params=packed_seq_params)

    return output_tensor, partial(moe_loss_func, loss_mask)


def is_dataset_built_on_rank():
    # dynamic-flash-attn requires build packed_seq_params on each pp rank
    # TODO: find a better way
    return mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path = args.s3_cache_path
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
