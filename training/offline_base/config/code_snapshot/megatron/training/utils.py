# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        multi_tensor_applier = None

    try:
        from amp_C import multi_tensor_l2norm
    except ImportError:
        import warnings
        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_l2norm'
        )

        from megatron.core.utils import (
            local_multi_tensor_l2_norm as multi_tensor_l2norm,
            local_multi_tensor_applier as multi_tensor_applier,
        )

from megatron.training import (
    get_args,
    get_tokenizer,
    get_adlr_autoresume,
)
from megatron.core import DistributedDataParallel as DDP
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate, broadcast_data
from megatron.legacy.model import Float16Module
from megatron.legacy.model.module import param_is_not_shared
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.chunkflow.chunkflow_state_store import state_store_manager

ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if mpu.get_expert_model_parallel_rank() > 0:
                if not getattr(param, 'allreduce', True) and is_not_tp_duplicate:
                    assert param_is_not_shared(param)
                    params_data.append(param.data.float() if args.bf16 else param.data)
            else:
                is_not_shared = param_is_not_shared(param)
                if is_not_shared and is_not_tp_duplicate:
                    params_data.append(param.data.float() if args.bf16 else param.data)

    # Calculate norm
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    norm, _ = multi_tensor_applier(
        multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    if mpu.get_expert_model_parallel_world_size() == 1:
        # Sum across all model-parallel GPUs(tensor + pipeline).
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_model_parallel_group())
    else:
        # Sum across expert tensor, model and pipeline parallel GPUs.
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_expert_tensor_model_pipeline_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.training.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def get_batch_on_this_cp_rank(batch, skip_keys=["packed_seq_params", "kwargs"]):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    args = get_args()
    cp_split_size = 1 if (args.context_parallel_strategy in ['ulysses', 'kvgather']) else 2
    cp_size = args.context_parallel_size
    if cp_size > 1:
        cp_rank = mpu.get_context_parallel_rank()
        for key, val in batch.items():
            if key == 'attention_mask' and \
                (args.context_parallel_strategy in ['ulysses', 'kvgather']):
              continue
            if key in skip_keys:
                continue
            if val is not None:
                seq_dim = 1 if key != 'attention_mask' else 2
                val = val.view(
                    *val.shape[0:seq_dim],
                    cp_split_size * cp_size,
                    val.shape[seq_dim] // (cp_split_size * cp_size),
                    *val.shape[(seq_dim + 1):],
                )
                if cp_split_size == 1:
                    index = torch.tensor([cp_rank], device="cpu", pin_memory=True).cuda(non_blocking=True)
                else:
                    index = torch.tensor([cp_rank, (cp_split_size * cp_size - cp_rank - 1)],
                                          device="cpu", pin_memory=True).cuda(non_blocking=True)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
                batch[key] = val

    return batch


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_last_rank():
    return torch.distributed.get_rank() == (
            torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def append_to_progress_log(string, barrier=True):
    """ Append given string to progress log. """
    args = get_args()
    if args.save is None:
        return
    progress_log_filename = os.path.join(args.save, "progress.txt")
    if barrier:
        torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        with open(progress_log_filename, 'a') as f:
            job_id = os.getenv('SLURM_JOB_ID', '')
            num_gpus = args.world_size
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t"
                    f"# GPUs: {num_gpus}\t{string}\n")


def get_batch_on_this_tp_rank(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        batch = {
            'tokens': data["tokens"].cuda(non_blocking=True),
            'labels': data["labels"].cuda(non_blocking=True),
            'loss_mask': data["loss_mask"].cuda(non_blocking=True),
            'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
            'position_ids': data["position_ids"].cuda(non_blocking=True),
        }

        if args.use_chunkflow or args.use_smart_batching:
            batch['kwargs'] = {}
            broadcast_keys = ['position_ids']
            # Additional data that are constructed by ChunkFlowDataset
            if args.use_chunkflow:
                batch["total_tokens"] = data["total_tokens"].cuda(non_blocking=True)
                batch["seq_length"] = data["seq_length"].cuda(non_blocking=True)
                batch["alloc_state_store"] = data["alloc_state_store"].cuda(non_blocking=True)
                broadcast_keys.extend(['total_tokens', 'seq_length', 'alloc_state_store'])
            # These information are needed for all ranks
            broadcast_data(broadcast_keys, batch, torch.int64)

            if args.pipeline_model_parallel_size == 1:
                broadcast_data(['tokens', 'labels'], batch, torch.int64)
                broadcast_data(['loss_mask'], batch, torch.float32)
            elif mpu.is_pipeline_first_stage():
                broadcast_data(['tokens'], batch, torch.int64)
            elif mpu.is_pipeline_last_stage():
                broadcast_data(['labels'], batch, torch.int64)
                broadcast_data(['loss_mask'], batch, torch.float32)
            else:
                broadcast_data(['tokens'], batch, torch.int64)
        elif args.use_dynamic_collate:
            if args.pipeline_model_parallel_size == 1:
                broadcast_data(['tokens', 'labels', 'position_ids'], batch, torch.int64)
                broadcast_data(['loss_mask'], batch, torch.float32)
                if batch['attention_mask'] is not None:
                    broadcast_data(['attention_mask'], batch, torch.bool)
            elif mpu.is_pipeline_first_stage():
                broadcast_data(['tokens', 'position_ids'], batch, torch.int64)
                if batch['attention_mask'] is not None:
                    broadcast_data(['attention_mask'], batch, torch.bool)
            elif mpu.is_pipeline_last_stage():
                if args.dynamic_flash_attn:
                    broadcast_data(['tokens'], batch, torch.int64)
                broadcast_data(['labels'], batch, torch.int64)
                broadcast_data(['loss_mask'], batch, torch.float32)
                if batch['attention_mask'] is not None:
                    broadcast_data(['attention_mask'], batch, torch.bool)
            else:
                broadcast_data(['tokens'], batch, torch.int64)
        else:
            if args.pipeline_model_parallel_size == 1:
                _broadcast(batch['tokens'])
                _broadcast(batch['labels'])
                _broadcast(batch['loss_mask'])
                _broadcast(batch['attention_mask'])
                _broadcast(batch['position_ids'])
            elif mpu.is_pipeline_first_stage():
                _broadcast(batch['tokens'])
                _broadcast(batch['attention_mask'])
                _broadcast(batch['position_ids'])
            elif mpu.is_pipeline_last_stage():
                if args.dynamic_flash_attn:
                    _broadcast(batch['tokens'])
                _broadcast(batch['labels'])
                _broadcast(batch['loss_mask'])
                _broadcast(batch['attention_mask'])
            else:
                if args.dynamic_flash_attn:
                    _broadcast(batch['tokens'])

    else:
        if args.use_chunkflow or args.use_smart_batching:
            batch = {
                'tokens': None,
                'labels': None,
                'loss_mask': None,
                'attention_mask': None,
                'position_ids': None,
                "kwargs": {}
            }
            broadcast_keys = ['position_ids']
            if args.use_chunkflow:
                batch.update(
                    {
                    'seq_length': None,
                    'total_tokens': None,
                    'alloc_state_store': None,
                    }
                )
                broadcast_keys.extend(['total_tokens', 'seq_length', 'alloc_state_store'])
            batch.update(broadcast_data(broadcast_keys, None, torch.int64))
            if args.pipeline_model_parallel_size == 1:
                batch.update(broadcast_data(['tokens', 'labels'], None, torch.int64))
                batch.update(broadcast_data(['loss_mask'], None, torch.float32))
            elif mpu.is_pipeline_first_stage():
                batch.update(broadcast_data(['tokens'], None, torch.int64))
            elif mpu.is_pipeline_last_stage():
                batch.update(broadcast_data(['labels'], None, torch.int64))
                batch.update(broadcast_data(['loss_mask'], None, torch.float32))
            else:
                batch.update(broadcast_data(['tokens'], None, torch.int64))
        elif args.use_dynamic_collate:
            batch = {
                'tokens': None,
                'labels': None,
                'loss_mask': None,
                'attention_mask': None,
                'position_ids': None
            }
            if args.pipeline_model_parallel_size == 1:
                batch.update(broadcast_data(['tokens', 'labels', 'position_ids'], None, torch.int64))
                batch.update(broadcast_data(['loss_mask'], None, torch.float32))
                if args.create_attention_mask_in_dataloader:
                    batch.update(broadcast_data(['attention_mask'], None, torch.bool))
            elif mpu.is_pipeline_first_stage():
                batch.update(broadcast_data(['tokens', 'position_ids'], None, torch.int64))
                if args.create_attention_mask_in_dataloader:
                    batch.update(broadcast_data(['attention_mask'], None, torch.bool))
            elif mpu.is_pipeline_last_stage():
                if args.dynamic_flash_attn:
                    batch.update(broadcast_data(['tokens'], None, torch.int64))
                batch.update(broadcast_data(['labels'], None, torch.int64))
                batch.update(broadcast_data(['loss_mask'], None, torch.float32))
                if args.create_attention_mask_in_dataloader:
                    batch.update(broadcast_data(['attention_mask'], None, torch.bool))
            else:
                if args.dynamic_flash_attn:
                    batch.update(broadcast_data(['tokens'], None, torch.int64))
        else:
            tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                 device=torch.cuda.current_device())
            labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                 device=torch.cuda.current_device())
            loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32,
                                    device=torch.cuda.current_device())
            if args.create_attention_mask_in_dataloader:
                attention_mask = torch.empty(
                    (args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                    device=torch.cuda.current_device()
                )
            else:
                attention_mask = None
            position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                       device=torch.cuda.current_device())

            if args.pipeline_model_parallel_size == 1:
                _broadcast(tokens)
                _broadcast(labels)
                _broadcast(loss_mask)
                _broadcast(attention_mask)
                _broadcast(position_ids)

            elif mpu.is_pipeline_first_stage():
                labels = None
                loss_mask = None

                _broadcast(tokens)
                _broadcast(attention_mask)
                _broadcast(position_ids)

            elif mpu.is_pipeline_last_stage():
                if args.dynamic_flash_attn:
                    _broadcast(tokens)
                else:
                    tokens = None
                position_ids = None

                _broadcast(labels)
                _broadcast(loss_mask)
                _broadcast(attention_mask)
            else:
                if args.dynamic_flash_attn:
                    _broadcast(tokens)

            batch = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': loss_mask,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }

    # Post process for all ranks
    if args.use_chunkflow or args.use_smart_batching:
        # construct cu_seqlens
        position_ids = batch['position_ids'].contiguous().flatten()
        if position_ids[0] == 0:
            indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)
            cu_seqlens = torch.cat(
                (
                    indices_q[position_ids == 0],
                    torch.tensor([position_ids.size(0)], device=position_ids.device, dtype=torch.int32),
                )
            )
        else:
            cu_seqlens = torch.tensor([0, position_ids.size(0)], device=position_ids.device, dtype=torch.int32)

        batch['kwargs']['cu_seqlens'] = cu_seqlens
        if args.use_chunkflow:
            alloc_state_store = batch.pop("alloc_state_store").long().item()
            seq_length = batch.pop("seq_length").long().item()

            # Set the kwargs for later use
            batch['kwargs']['total_tokens'] = batch.pop('total_tokens').long()
            chunk_idx = position_ids[-1] // args.chunk_size
            chunk_count = (seq_length + args.chunk_size - 1) // args.chunk_size
            chunk_size = args.chunk_size

            # Check if we need to allocate a new state cache
            if alloc_state_store == 1:
                state_store_manager.reset_state_store()

        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen, _ = seqlens.max(dim=0, keepdim=True)
        batch['packed_seq_params'] = {'cu_seqlens_q': cu_seqlens,
                                      'cu_seqlens_kv': cu_seqlens,
                                      'max_seqlen_q': max_seqlen,
                                      'max_seqlen_kv': max_seqlen,
                                      'qkv_format': 'thd',}
        if args.use_chunkflow:
            batch['packed_seq_params'].update(
                                    {"chunk_idx": chunk_idx,
                                      "chunk_count": chunk_count,
                                      "chunk_size":chunk_size})
    elif args.dynamic_flash_attn:
        clone_tokens = batch['tokens'].clone()
        clone_tokens[:, -1] = tokenizer.eod
        cu_seqlens = torch.nonzero(clone_tokens.eq(tokenizer.eod).flatten(), as_tuple=True)[0] + 1
        cu_seqlens = F.pad(cu_seqlens, (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen, _ = seqlens.max(dim=0, keepdim=True)
        batch['packed_seq_params'] = {'cu_seqlens_q': cu_seqlens,
                                      'cu_seqlens_kv': cu_seqlens,
                                      'max_seqlen_q': max_seqlen,
                                      'max_seqlen_kv': max_seqlen,
                                      'qkv_format': 'thd'}
    elif args.context_parallel_size > 1 and args.context_parallel_strategy in ('kvgather', 'ulysses'):
        # pass a full mask
        cu_seqlens = torch.tensor(torch.arange(0, batch["tokens"].shape[0] + 1) * batch["tokens"].shape[1], device='cuda', dtype=torch.int32)
        max_seqlen = torch.tensor([batch["tokens"].shape[1]], device='cuda', dtype=torch.int32)
        batch['packed_seq_params'] = {'cu_seqlens_q': cu_seqlens,
                                      'cu_seqlens_kv': cu_seqlens,
                                      'max_seqlen_q': max_seqlen,
                                      'max_seqlen_kv': max_seqlen,
                                      'qkv_format': 'thd'}

    return batch


def update_use_dist_ckpt(args):
    args.use_dist_ckpt = args.ckpt_format != "torch"
