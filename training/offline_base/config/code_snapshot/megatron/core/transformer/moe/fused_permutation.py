# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
import triton
import triton.language as tl

HAVE_TE_FLOAT8TENSOR = False
try:
    from transformer_engine.pytorch.float8_tensor import Float8Tensor
    HAVE_TE_FLOAT8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # Float8Tensor not found
    pass

try:
    import transformer_engine_torch as tex
except (ImportError, ModuleNotFoundError):
    try:
        import transformer_engine_extensions as tex
    except (ImportError, ModuleNotFoundError):
        raise ImportError("Cannot import transformer_engine_torch or transformer_engine_extensions")


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor"""
    return HAVE_TE_FLOAT8TENSOR and isinstance(tensor, Float8Tensor)


@triton.jit
def _permute_kernel(
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    routing_map_ptr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    current_start = 0
    while current_start < hidden_size:
        current_offset = current_start + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < hidden_size
        input_offsets = pid * hidden_size + current_offset
        input = tl.load(input_ptr + input_offsets, mask=mask)
        for expert_idx in range(num_experts):
            selected = tl.load(routing_map_ptr + pid * num_experts + expert_idx)
            if selected != 0:
                dst_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid) - 1
                output_offsets = dst_row * hidden_size + current_offset
                tl.store(output_ptr + output_offsets, input, mask=mask)
        current_start += BLOCK_SIZE


@triton.jit
def _unpermute_kernel(
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    routing_map_ptr,
    probs_ptr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    with_probs: tl.constexpr,
    fp8_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if fp8_dtype == tex.DType.kFloat8E5M2:
        compute_type = tl.float16
        data_type = tl.float8e5
        pytorch_tensor_dtype = tl.uint8
    elif fp8_dtype == tex.DType.kFloat8E4M3:
        compute_type = tl.float16
        data_type = tl.float8e4nv
        pytorch_tensor_dtype = tl.uint8
    else:
        compute_type = input_ptr.dtype.element_ty
        assert fp8_dtype is None

    pid = tl.program_id(0)
    current_start = 0
    while current_start < hidden_size:
        current_offset = current_start + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < hidden_size
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
        for expert_idx in range(num_experts):
            selected = tl.load(routing_map_ptr + pid * num_experts + expert_idx)
            if selected != 0:
                src_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid) - 1
                input_offsets = src_row * hidden_size + current_offset
                input = tl.load(input_ptr + input_offsets, mask=mask)
                if fp8_dtype is not None:
                    input = input.to(data_type, bitcast=True).to(compute_type)
                if with_probs:
                    prob = tl.load(probs_ptr + pid * num_experts + expert_idx).to(compute_type)
                    input *= prob
                accumulator += input
        output_offsets = pid * hidden_size + current_offset
        if fp8_dtype is not None:
            if not with_probs:
                # Directly adding these value may cause overflow for fp8, we scale it here.
                # The outside fp8_scale_inv is also scaled in the meantime.
                accumulator /= num_experts
            accumulator = accumulator.to(data_type).to(pytorch_tensor_dtype, bitcast=True)
        tl.store(output_ptr + output_offsets, accumulator, mask=mask)
        current_start += BLOCK_SIZE


@triton.jit
def _unpermute_bwd_with_probs_kernel(
    fwd_output_grad_ptr,
    fwd_input_grad_ptr,
    fwd_input_ptr,
    probs_ptr,
    probs_grad_ptr,
    row_id_map_ptr,
    routing_map_ptr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    fp8_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if fp8_dtype == tex.DType.kFloat8E5M2:
        compute_type = tl.float16
        data_type = tl.float8e5
        pytorch_tensor_dtype = tl.uint8
    elif fp8_dtype == tex.DType.kFloat8E4M3:
        compute_type = tl.float16
        data_type = tl.float8e4nv
        pytorch_tensor_dtype = tl.uint8
    else:
        compute_type = fwd_output_grad_ptr.dtype.element_ty
        assert fp8_dtype is None

    pid = tl.program_id(0)
    for expert_idx in range(num_experts):
        selected = tl.load(routing_map_ptr + pid * num_experts + expert_idx)
        if selected != 0:
            prob_grad_accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            current_start = 0
            while current_start < hidden_size:
                current_offset = current_start + tl.arange(0, BLOCK_SIZE)
                mask = current_offset < hidden_size
                input_offsets = pid * hidden_size + current_offset
                input = tl.load(fwd_output_grad_ptr + input_offsets, mask=mask)
                if fp8_dtype is not None:
                    input = input.to(data_type, bitcast=True).to(compute_type)
                prob = tl.load(probs_ptr + pid *  num_experts + expert_idx).to(compute_type)
                output = input * prob
                if fp8_dtype is not None:
                    output = output.to(data_type).to(pytorch_tensor_dtype, bitcast=True)
                dst_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid) - 1
                output_offsets = dst_row * hidden_size + current_offset
                tl.store(fwd_input_grad_ptr + output_offsets, output, mask=mask)

                fwd_input = tl.load(fwd_input_ptr + output_offsets, mask=mask)
                if fp8_dtype is not None:
                    fwd_input = fwd_input.to(data_type, bitcast=True)
                prob_grad_accum += fwd_input.to(tl.float32) * input.to(tl.float32)
                current_start += BLOCK_SIZE
            probs_grad = tl.sum(prob_grad_accum)
            tl.store(probs_grad_ptr + pid * num_experts + expert_idx, probs_grad)
        else:
            tl.store(probs_grad_ptr + pid * num_experts + expert_idx, 0.0)


class TritonPermuteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, routing_map, num_out_tokens: int):
        if not tokens.numel():
            return tokens, torch.tensor([], device=tokens.device)
        if not tokens.is_contiguous():
            tokens = tokens.contiguous()

        num_tokens = tokens.size(0)
        hidden_size = tokens.size(1)
        num_experts = routing_map.size(1)
        assert num_out_tokens is not None, "num_out_tokens must be provided to the fused permute function."

        # get dest row for each selected token
        row_id_map = torch.cumsum(routing_map.t().reshape(-1), dim=0)

        if is_float8tensor(tokens):
            input_tensor = tokens._data
            output_tensor = torch.empty((num_out_tokens, hidden_size), dtype=input_tensor.dtype, device='cuda')
            output = Float8Tensor(data=output_tensor, fp8_dtype=tokens._fp8_dtype, fp8_scale_inv=tokens._scale_inv)
            fp8 = True
        else:
            input_tensor = tokens
            output_tensor = torch.empty((num_out_tokens, hidden_size), dtype=input_tensor.dtype, device='cuda')
            output = output_tensor
            fp8 = False
        block_size = 512
        grid = (num_tokens,)
        _permute_kernel[grid](input_tensor, output_tensor, row_id_map, routing_map, num_tokens, num_experts, hidden_size, BLOCK_SIZE=block_size)

        ctx.save_for_backward(routing_map, row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        ctx.fp8 = fp8
        return output, row_id_map
    
    @staticmethod
    def backward(ctx, permuted_act_grad, _):
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None
        if not permuted_act_grad.is_contiguous():
            permuted_act_grad = permuted_act_grad.contiguous()

        act_grad = None
        if ctx.needs_input_grad[0]:
            routing_map, row_id_map = ctx.saved_tensors
            if ctx.fp8:
                assert isinstance(
                    permuted_act_grad, Float8Tensor
                ), "Grad of the output must be in Float8Tensor type for FP8 moe_permute."
                input_tensor = permuted_act_grad._data
                output_tensor = torch.empty((ctx.num_tokens, ctx.hidden_size), dtype=input_tensor.dtype, device='cuda')
                act_grad = Float8Tensor(data=output_tensor, fp8_dtype=permuted_act_grad._fp8_dtype, fp8_scale_inv=permuted_act_grad._scale_inv * ctx.num_experts)
                fp8_dtype = permuted_act_grad._fp8_dtype
            else:
                input_tensor = permuted_act_grad
                output_tensor = torch.empty((ctx.num_tokens, ctx.hidden_size), dtype=input_tensor.dtype, device='cuda')
                act_grad = output_tensor
                fp8_dtype = None
            block_size = 512
            grid = (ctx.num_tokens,)
            _unpermute_kernel[grid](input_tensor, output_tensor, row_id_map, routing_map, None, ctx.num_tokens, ctx.num_experts, ctx.hidden_size, with_probs=False, fp8_dtype=fp8_dtype, BLOCK_SIZE=block_size)
        return act_grad, None, None


class TritonUnpermuteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permuted_tokens, probs, routing_map, row_id_map, restore_shape):
        if restore_shape is None:
            restore_shape = permuted_tokens.shape
        num_tokens, hidden_size = restore_shape
        assert routing_map is not None, "routing_map must be provided to the fused unpermute function."
        num_experts = routing_map.size(1)
        with_probs = probs is not None

        if is_float8tensor(permuted_tokens):
            input_tensor = permuted_tokens._data
            output_tensor = torch.empty((num_tokens, hidden_size), dtype=input_tensor.dtype, device='cuda')
            if not with_probs:
                scale_inv = permuted_tokens._scale_inv * num_experts
            else:
                scale_inv = permuted_tokens._scale_inv
            output = Float8Tensor(data=output_tensor, fp8_dtype=permuted_tokens._fp8_dtype, fp8_scale_inv=scale_inv)
            fp8_dtype = permuted_tokens._fp8_dtype
            fp8 = True
        else:
            input_tensor = permuted_tokens
            output_tensor = torch.empty((num_tokens, hidden_size), dtype=input_tensor.dtype, device='cuda')
            output = output_tensor
            fp8_dtype = None
            fp8 = False
        block_size = 512
        grid = (num_tokens,)
        _unpermute_kernel[grid](input_tensor, output_tensor, row_id_map, routing_map, probs, num_tokens, num_experts, hidden_size, with_probs=with_probs, fp8_dtype=fp8_dtype, BLOCK_SIZE=block_size)

        ctx.save_for_backward(permuted_tokens, routing_map, row_id_map, probs)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        ctx.fp8 = fp8
        return output

    @staticmethod
    def backward(ctx, unpermuted_act_grad):
        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, ctx.probs, None, None, None
        if not unpermuted_act_grad.is_contiguous():
            unpermuted_act_grad = unpermuted_act_grad.contiguous()

        act_grad = None
        if ctx.needs_input_grad[0]:
            fwd_input, routing_map, row_id_map, probs = ctx.saved_tensors
            num_permuted_tokens = fwd_input.size(0)
            with_probs = probs is not None

            if with_probs:
                probs_grad = torch.empty((ctx.num_tokens, ctx.num_experts), dtype=probs.dtype, device='cuda')
            else:
                probs_grad = None
            if ctx.fp8:
                assert isinstance(
                    unpermuted_act_grad, Float8Tensor
                ), "Grad of the output must be in Float8Tensor type for FP8 moe_unpermute."
                fwd_input_tensor = fwd_input._data
                input_tensor = unpermuted_act_grad._data
                output_tensor = torch.empty((num_permuted_tokens, ctx.hidden_size), dtype=input_tensor.dtype, device='cuda')
                act_grad = Float8Tensor(data=output_tensor, fp8_dtype=unpermuted_act_grad._fp8_dtype, fp8_scale_inv=unpermuted_act_grad._scale_inv)
                fp8_dtype = unpermuted_act_grad._fp8_dtype
            else:
                fwd_input_tensor = fwd_input
                input_tensor = unpermuted_act_grad
                output_tensor = torch.empty((num_permuted_tokens, ctx.hidden_size), dtype=input_tensor.dtype, device='cuda')
                act_grad = output_tensor
                fp8_dtype = None

            if with_probs:
                block_size = 512
                grid = (ctx.num_tokens,)
                _unpermute_bwd_with_probs_kernel[grid](input_tensor, output_tensor, fwd_input_tensor, probs, probs_grad, row_id_map, routing_map, ctx.num_tokens, ctx.num_experts, ctx.hidden_size, fp8_dtype, BLOCK_SIZE=block_size)
            else:
                block_size = 512
                grid = (ctx.num_tokens,)
                _permute_kernel[grid](input_tensor, output_tensor, row_id_map, routing_map, ctx.num_tokens, ctx.num_experts, ctx.hidden_size, BLOCK_SIZE=block_size)

        if not ctx.needs_input_grad[1]:
            probs_grad = None
        return act_grad, probs_grad, None, None, None


def fused_permute(tokens, routing_map, num_out_tokens):
    return TritonPermuteFunction.apply(tokens, routing_map, num_out_tokens)


def fused_unpermute(permuted_tokens, row_id_map, restore_shape, probs, routing_map):
    return TritonUnpermuteFunction.apply(permuted_tokens, probs, routing_map, row_id_map, restore_shape)
