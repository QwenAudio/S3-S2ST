# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

from .utils import VocabUtility

import triton
import triton.language as tl


@triton.jit
def cross_entropy_fwd_sum_exp_kernel(sum_exp_logits_ptr,
                                     vocab_parallel_logits_ptr,
                                     logits_max_ptr,
                                     n_cols,
                                     BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    vocab_parallel_logits_ptr += row_idx * n_cols

    logits_max = tl.load(logits_max_ptr + row_idx)
    sum_exp_logits = 0.0

    for i in range((n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        vocab_parallel_logits_ptrs = vocab_parallel_logits_ptr + col_offsets
        row = tl.load(
            vocab_parallel_logits_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
        row = row.to(tl.float32)
        row = row - logits_max
        row = tl.exp(row)
        sum_exp_logits += tl.sum(row)

    tl.store(sum_exp_logits_ptr + row_idx, sum_exp_logits)


def cross_entropy_fwd_sum_exp(vocab_parallel_logits, logits_max):
    sum_exp_logits = torch.empty_like(logits_max)

    n_cols = vocab_parallel_logits.size(-1)
    vocab_parallel_logits = vocab_parallel_logits.view(-1, n_cols)
    n_rows = vocab_parallel_logits.size(0)

    logits_max = logits_max.view(n_rows)

    BLOCK_SIZE = 8*1024
    num_warps = 16

    sum_exp_logits_ = sum_exp_logits.view(n_rows)

    cross_entropy_fwd_sum_exp_kernel[(n_rows, )](
        sum_exp_logits_,
        vocab_parallel_logits,
        logits_max,
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return sum_exp_logits


@triton.jit
def cross_entropy_fwd_mean_log_kernel(mean_log_probs_ptr,
                                      vocab_parallel_logits_ptr,
                                      logits_max_ptr,
                                      sum_exp_logits_ptr,
                                      n_cols,
                                      BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    vocab_parallel_logits_ptr += row_idx * n_cols

    logits_max = tl.load(logits_max_ptr + row_idx)
    sum_exp_logits = tl.load(sum_exp_logits_ptr + row_idx)
    mean_log_probs = 0.0

    for i in range((n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        vocab_parallel_logits_ptrs = vocab_parallel_logits_ptr + col_offsets
        row = tl.load(
            vocab_parallel_logits_ptrs, mask=col_offsets < n_cols, other=0)
        row = row.to(tl.float32)
        row = row - logits_max
        row = tl.exp(row)

        sum_exp_row = tl.full((BLOCK_SIZE,), sum_exp_logits, tl.float32)
        row = tl.math.div_rn(row, sum_exp_row)

        row = tl.log(row)
        zeros = tl.full((BLOCK_SIZE,), 0, tl.float32)
        row = tl.where(col_offsets < n_cols, row, zeros)
        mean_log_probs += tl.sum(row)

    mean_log_probs /= n_cols
    tl.store(mean_log_probs_ptr + row_idx, mean_log_probs)


def cross_entropy_fwd_mean_log(vocab_parallel_logits, logits_max, sum_exp_logits):
    mean_log_probs = torch.empty_like(sum_exp_logits)

    n_cols = vocab_parallel_logits.size(-1)
    vocab_parallel_logits = vocab_parallel_logits.view(-1, n_cols)
    n_rows = vocab_parallel_logits.size(0)

    logits_max = logits_max.view(n_rows)
    sum_exp_logits = sum_exp_logits.view(n_rows)

    BLOCK_SIZE = 8*1024
    num_warps = 16

    mean_log_probs_ = mean_log_probs.view(n_rows)

    cross_entropy_fwd_mean_log_kernel[(n_rows, )](
        mean_log_probs_,
        vocab_parallel_logits,
        logits_max,
        sum_exp_logits,
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return mean_log_probs


@triton.jit
def cross_entropy_bwd_kernel(grad_input_ptr,
                             grad_output_ptr,
                             input_ptr,
                             target_mask_ptr,
                             masked_target_1d_ptr,
                             logits_max_ptr,
                             sum_exp_logits_ptr,
                             n_cols,
                             label_smoothing,
                             vocab_size,
                             BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    grad_input_ptr += row_idx * n_cols
    input_ptr += row_idx * n_cols

    grad_output = tl.load(grad_output_ptr + row_idx)
    target_mask = tl.load(target_mask_ptr + row_idx)
    masked_target_1d = tl.load(masked_target_1d_ptr + row_idx)
    logits_max = tl.load(logits_max_ptr + row_idx)
    sum_exp_logits = tl.load(sum_exp_logits_ptr + row_idx)

    for i in range((n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        input_ptrs = input_ptr + col_offsets
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0)
        row = row.to(tl.float32)
        row = row - logits_max
        row = tl.exp(row)
        sum_exp_row = tl.full((BLOCK_SIZE,), sum_exp_logits, tl.float32)
        row = tl.math.div_rn(row, sum_exp_row)

        softmax_update = 1.0 - target_mask.to(tl.float32)
        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            softmax_update *= (1.0 - smoothing)
        row = tl.where(col_offsets == masked_target_1d, row - softmax_update, row)

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            average_grad = 1.0 / vocab_size
            row -= smoothing * average_grad

        row = row * grad_output
        row = row.to(tl.bfloat16)
        grad_input_ptrs = grad_input_ptr + col_offsets
        tl.store(grad_input_ptrs, row, mask=col_offsets < n_cols)


def cross_entropy_bwd(grad_output,
                      inputs,
                      target_mask,
                      masked_target_1d,
                      logits_max,
                      sum_exp_logits,
                      label_smoothing,
                      vocab_size):
    grad_input = torch.empty_like(inputs)

    n_cols = inputs.size(-1)
    inputs = inputs.view(-1, n_cols)
    n_rows = inputs.size(0)

    
    grad_output = grad_output.reshape(n_rows) #TODO do we need to use `view`
    grad_output = grad_output.contiguous()
    target_mask = target_mask.view(n_rows)
    masked_target_1d = masked_target_1d.view(n_rows)
    logits_max = logits_max.view(n_rows)
    sum_exp_logits = sum_exp_logits.view(n_rows)

    BLOCK_SIZE = 8*1024
    num_warps = 16

    grad_input_ = grad_input.view(n_rows, n_cols)

    cross_entropy_bwd_kernel[(n_rows, )](
        grad_input_,
        grad_output,
        inputs,
        target_mask,
        masked_target_1d,
        logits_max,
        sum_exp_logits,
        n_cols,
        label_smoothing,
        vocab_size,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_input


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0):

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )
        logits_max = logits_max.to(torch.float32)

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )
        predicted_logits = predicted_logits.to(torch.float32)

        # Sum of exponential of logits along vocab dimension across all GPUs.
        sum_exp_logits = cross_entropy_fwd_sum_exp(vocab_parallel_logits, logits_max)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        loss = torch.log(sum_exp_logits) - (predicted_logits - logits_max)

        vocab_size = vocab_parallel_logits.size(-1)
        if label_smoothing > 0:
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            mean_log_probs = cross_entropy_fwd_mean_log(vocab_parallel_logits, logits_max, sum_exp_logits)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(vocab_parallel_logits, target_mask, masked_target_1d, logits_max, sum_exp_logits)
        

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        inputs, target_mask, masked_target_1d, logits_max, sum_exp_logits = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        grad_input = cross_entropy_bwd(
            grad_output, inputs, target_mask, masked_target_1d, logits_max, sum_exp_logits, label_smoothing, vocab_size
        )
        
        return grad_input, None, None, None

