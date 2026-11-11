#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DPO"""
import inspect
from contextlib import nullcontext
from typing import Union

import torch
from functools import partial

from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.distributed import finalize_model_grads
from megatron.core.utils import get_model_config, StragglerDetector
from megatron.inference.inference_forward_step import InferenceForwardStep
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import tensor_parallel, mpu
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.distributed import DistributedDataParallel as DDP

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, \
    get_gpt_layer_local_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.training.training import write_args_to_tensorboard, train_step, get_model, setup_model_and_optimizer, \
    build_train_valid_test_data_iterators, save_checkpoint_and_time, training_log, num_floating_point_operations
from megatron.training.utils import get_ltor_masks_and_position_ids, average_losses_across_data_parallel_group, unwrap_model
from megatron.training import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
import time
import megatron.legacy.model
from megatron.core.datasets.preference_dataset import build_train_valid_test_datasets_for_preference, \
    PreferenceDatasetConfig
import torch.nn.functional as F

from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
stimer = StragglerDetector()


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
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention, args.fp8)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention)

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

    return model


def get_ref_batch(data_iterator):
    """Generate a batch"""
    args = get_args()

    # Items and their type.
    keys = ["tokens", "labels", "position_ids", "scores", "num_responses"]

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, torch.int64)
    data_b.update(tensor_parallel.broadcast_data(["loss_mask"], data, torch.float32))
    if args.create_attention_mask_in_dataloader:
        data_b.update(tensor_parallel.broadcast_data(['attention_mask'], data, torch.bool))

    # Unpack.
    seq_len = data_b['tokens'].shape[-1]
    tokens = data_b['tokens'].reshape(-1, seq_len) # (batch*num_response) * sequence_len
    labels = data_b['labels'].reshape(-1, seq_len)
    loss_mask = data_b['loss_mask'].reshape(-1, seq_len)
    position_ids = data_b['position_ids'].reshape(-1, seq_len)
    scores = data_b['scores'].float()
    num_responses = data_b['num_responses'].long()
    if args.create_attention_mask_in_dataloader:
        attention_mask = data_b["attention_mask"].reshape(-1, seq_len)
    else:
        attention_mask = None
    return tokens, labels, scores, num_responses, attention_mask, position_ids, loss_mask


def dpo_loss_fn(loss_mask, rw_scores, num_responses, ref_logits, output_tensor):
    args = get_args()
    logit_label = -output_tensor.float()
    reference_logit_label = -ref_logits.float()

    logit_label = logit_label * loss_mask
    reference_logit_label = reference_logit_label * loss_mask
    length = loss_mask.sum(-1)

    if args.use_ipo:
        scores = logit_label.sum(-1) / length
        reference_scores = reference_logit_label.sum(-1) / length
    else:
        scores = logit_label.sum(-1) #/ length
        reference_scores = reference_logit_label.sum(-1) #/ length

    idx = 0
    loss = 0
    avg_dpo_loss = 0
    avg_sft_loss = 0
    for i, n in enumerate(num_responses):
        # we need to judge which is the higher score relatively
        diff = (scores[idx:idx+n]-reference_scores[idx:idx+n]).unsqueeze(0)-(scores[idx:idx+n]-reference_scores[idx:idx+n]).unsqueeze(-1) # b * b

        rw_score = rw_scores[i].squeeze(0)
        rw_diff = rw_score.unsqueeze(0) - rw_score.unsqueeze(-1) # b * b

        negative_indices = torch.nonzero(torch.lt(rw_diff, 0))
        diff_transformed = diff[negative_indices[:,0], negative_indices[:,1]] * -1
        if diff_transformed.size(0) == 0:
            print(rw_scores)

        if args.use_ipo:
            dpo_loss = ((diff_transformed - 1/(2 * args.dpo_weight)) ** 2).mean()
        else:
            dpo_loss = - F.logsigmoid(args.dpo_weight * diff_transformed) * (1 - args.dpo_label_smoothing) \
                       - F.logsigmoid(-args.dpo_weight * diff_transformed) * args.dpo_label_smoothing
            dpo_loss = dpo_loss.mean()

        avg_dpo_loss += dpo_loss
        if args.use_best_sft:
            max_idx = torch.argmax(rw_score)
            sft_loss = -logit_label[idx+max_idx].mean()
            avg_sft_loss += sft_loss
            loss += args.dpo_loss_weight * dpo_loss + sft_loss
        else:
            loss += dpo_loss
        idx += n
    loss = loss / (len(num_responses) + 1e-5)
    avg_dpo_loss = avg_dpo_loss / (len(num_responses) + 1e-5)
    avg_sft_loss = avg_sft_loss / (len(num_responses) + 1e-5)
    if args.use_best_sft:
        # Reduce loss for logging.
        if args.reduce_loss_per_iteration:
            # Put off loss allreduce for logging.
            averaged_loss = [loss.clone().detach().view(1), avg_dpo_loss.clone().detach().view(1),
                             avg_sft_loss.clone().detach().view(1)]
        else:
            averaged_loss = average_losses_across_data_parallel_group([loss, avg_dpo_loss, avg_sft_loss])
        # print(f'--------------{averaged_loss[0]} {averaged_loss[1]} {averaged_loss[2]}')
        return loss, {'lm loss': averaged_loss[0], 'dpo_loss': averaged_loss[1], 'sft_loss': averaged_loss[2]}
    else:
        # Reduce loss for logging.
        if args.reduce_loss_per_iteration:
            # Put off loss allreduce for logging.
            averaged_loss = [loss.clone().detach().view(1), avg_dpo_loss.clone().detach().view(1)]
        else:
            averaged_loss = average_losses_across_data_parallel_group([loss, avg_dpo_loss])
        # print(f'--------------loss {averaged_loss[0]} {averaged_loss[1]}')
        return loss, {'lm loss': averaged_loss[0], 'dpo_loss': averaged_loss[1]}

def get_batch(data_iterator):
    if data_iterator is not None:
        data = next(data_iterator)
    tokens = data['tokens']
    labels = data['labels']
    loss_mask = data['loss_mask']
    attention_mask = data['attention_mask']
    position_ids = data['position_ids']
    scores = data['scores']
    num_responses = data['num_responses']
    ref_logits = data["ref_logits"]
    return tokens, labels, scores, num_responses, attention_mask, position_ids, ref_logits, loss_mask

def forward_step(data_iterator, model):
    """Forward step."""
    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, scores, num_responses, attention_mask, position_ids, ref_logits, loss_mask = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(
        tokens, position_ids, attention_mask, labels=labels
    )
    return output_tensor, partial(dpo_loss_fn, loss_mask, scores, num_responses, ref_logits)

def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return PreferenceDatasetConfig(
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
        use_dynamic_collate=args.use_dynamic_collate,
        chat_format=args.chat_format,
    )

def train_valid_test_datasets_provider(train_valid_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    print_rank_0("> building train datasets for DPO ...")
    train_ds, _, _ = build_train_valid_test_datasets_for_preference(input_file=args.data_path, config=config)
    print_rank_0("> finished creating DPO datasets ...")

    return train_ds, None, None

def ref_forward(data_iterator, reference_model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator-ref', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, scores, num_responses, attention_mask, position_ids, loss_mask = get_ref_batch(data_iterator)
    timers('batch-generator-ref').stop()

    ref_forward_step = InferenceForwardStep(reference_model)
    ref_logits = ref_forward_step(tokens, position_ids, attention_mask, labels)
    res = []
    data = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "scores": scores,
        "num_responses": num_responses,
    }
    if mpu.is_pipeline_last_stage():
        data["ref_logits"] = ref_logits.detach()
    else:
        data["ref_logits"] = None
    res.append(data)
    return res

def log_mem():
    iter_dict = {}
    mem_stats = torch.cuda.memory_stats()
    iter_dict["mem-reserved-bytes"] = mem_stats["reserved_bytes.all.current"]
    iter_dict["mem-allocated-bytes"] = mem_stats["allocated_bytes.all.current"]
    iter_dict["mem-allocated-count"] = mem_stats["allocation.all.current"]
    return iter_dict


def load_checkpoint_only(model, load_arg):
    # ref无需加载optim
    finetune = args.finetune
    no_load_optim = args.no_load_optim
    args.finetune = True
    args.no_load_optim = True
    load_checkpoint(model, optimizer=None, opt_param_scheduler=None, load_arg=load_arg, strict=True)
    args.finetune = finetune
    args.no_load_optim = no_load_optim


# Setup some training config params
def setup_config(model):
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = get_timers()
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, \
            ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
             'a custom no_sync_func is not supported when overlapping grad-reduce')
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads


if __name__ == "__main__":
    initialize_megatron(None)
    args = get_args()
    assert args.micro_batch_size == 1, "not support micro_batch_size>1 for now"

    # init ref
    reference_model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint_only(reference_model, "ref_load")
    print_rank_0("load ref checkpoint done")
    for model_module in reference_model:
        model_module.eval()
    unwrapped_ref_model = unwrap_model(reference_model[0])

    # init model
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, ModelType.encoder_or_decoder)

    for model_module in model:
        model_module.train()

    setup_config(model)

    if args.optimizer == 'adamw_ondare':
        assert False, "adamw_ondare has not migrated to mcore for now"
        assert args.use_distributed_optimizer, "for now, only support online dare adamw when using distributed optimizer"
        base_model = get_model(model_provider, ModelType.encoder_or_decoder)
        load_checkpoint_only(base_model, "base_load")
        print_rank_0(f"load base checkpoint done")
        for model_module in base_model:
            model_module.eval()
        unwrapped_base_model = unwrap_model(base_model[0])
        #用ref/base model初始化param_diff
        optimizer.optimizer.init_ref_param_diff(unwrapped_base_model, unwrapped_ref_model,
                                                optimizer.opt_group_ranges,
                                                optimizer.model_param_gbuf_map,
                                                optimizer.model_gbuf_ranges)
        del base_model

        # init dataset
    train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_datasets_provider)

    start = time.time()
    # ref model inference also use micro_batch_size
    num_inference_batches_each_episode = args.global_batch_size // (args.micro_batch_size * mpu.get_data_parallel_world_size())
    # Tracking loss.
    total_loss_dict = {}
    iteration = args.iteration
    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far
    report_memory_flag = True
    write_args_to_tensorboard()
    timers = get_timers()
    timers('interval-time', log_level=0).start(barrier=True)
    for i in range(args.train_iters-args.iteration):
        # reference inference
        res = []
        for inference_idx in range(num_inference_batches_each_episode):
            with torch.no_grad():
                res.extend(ref_forward(train_data_iterator, unwrapped_ref_model))
        # TODO support combine res to support micro batch size > 1

        # track timing
        if iteration % args.increase_log_level_interval == 0:
            print_rank_0(f'Iter {iteration} increase timing-log-level to 2.')
            args.init_timing_log_level = args.timing_log_level
            args.timing_log_level = 2
            timers.set_log_level(2)
        if iteration % args.increase_log_level_interval == args.increase_log_level_iters:
            print_rank_0(f'Iter {iteration} reset timing-log-level.')
            args.timing_log_level = args.init_timing_log_level
            timers.set_log_level(args.init_timing_log_level)

        # train
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = (
            train_step(forward_step,
                       iter(res),
                       model,
                       optimizer,
                       opt_param_scheduler,
                       None))
        iteration += 1
        num_floating_point_operations_so_far += num_floating_point_operations(args, args.global_batch_size)
        args.consumed_train_samples += args.global_batch_size
         # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if param_group['is_decoupled_lr']:
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          learning_rate, decoupled_learning_rate,
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, None, num_zeros_in_grad)
        # Checkpointing
        if args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations, {})
    print(f"train takes {time.time()-start} for {args.train_iters} iters")
