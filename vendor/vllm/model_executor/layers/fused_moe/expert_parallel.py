"""Expert parallel kernel."""

import contextlib
import functools
import itertools
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    import bladnn
except (ImportError, ModuleNotFoundError):
    bladnn = None

if TYPE_CHECKING:
    from deep_ep import Buffer, EventOverlap

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_data_parallel_world_size,
    get_dp_group,
    get_ep_group,
    get_expert_parallel_rank,
    get_expert_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_drop_ctx, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    persistent_masked_m_silu_mul_quant,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.single_batch_overlap import (
    combine_overlap_consumer,
    combine_overlap_sync,
    get_comb_overlap_args,
    init_and_get_overlap,
    wait_combine_finish,
)
from vllm.model_executor.layers.fused_moe.token_drop import moe_token_drop
from vllm.model_executor.layers.quantization.utils.ep_utils import (
    ep_gather,
    ep_scatter,
    tma_align_input_scale,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    get_nvfp4_global_scale_and_alpha,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    fp8_m_grouped_gemm_nt_masked,
    int8_m_grouped_gemm_nt_masked,
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_int8_gemm_nt_contiguous,
)
from vllm.utils.flashinfer import (
    flashinfer_cutedsl_grouped_gemm_nt_masked,
    flashinfer_cutlass_fused_moe,
    scaled_fp4_grouped_quantize,
    silu_and_mul_scaled_nvfp4_experts_quantize,
)
from vllm.utils.platform_utils import is_ppu
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    is_torch_equal_or_newer,
)
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

dev = torch.device("cuda:0")
sm_major = torch.cuda.get_device_properties(dev).major
sm_minor = torch.cuda.get_device_properties(dev).minor
sm_version = sm_major * 10 + sm_minor

bladnn_dynamic_quantize_fp8_for_deepep = None


def get_device_name(device_id: int = 0) -> str:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(device_id)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_name(device_id)

    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return torch.hpu.get_device_name(device_id)

    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.get_device_name(device_id)


@contextlib.contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None:
        yield
    else:
        import deep_gemm

        original_num_sms = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.set_num_sms(original_num_sms)


@triton.jit
def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):
    expert = tl.program_id(0)
    low = 0
    high = num_toks - 1
    target_location = -1
    while low <= high:
        mid = (low + high) // 2

        if tl.load(reorder_topk_ids + mid) > expert:
            high = mid - 1
        else:
            low = mid + 1
            target_location = mid
    tl.store(seg_indptr + expert + 1, target_location + 1)


@triton.jit
def compute_src2dst_triton_kernel(
    reorder_ids, src2dst, num_toks, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    tl.store(src2dst + src_id, dst_id, mask=mask)


def run_moe_ep_preproess(topk_ids: torch.Tensor, num_experts: int):
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    seg_indptr = torch.zeros(num_experts + 1, device=topk_ids.device, dtype=torch.int64)
    src2dst = torch.zeros(
        topk_ids.numel(), device=topk_ids.device, dtype=topk_ids.dtype
    )

    compute_seg_indptr_triton_kernel[(num_experts,)](
        reorder_topk_ids, seg_indptr, topk_ids.numel()
    )
    seg_sizes = seg_indptr[1:] - seg_indptr[:-1]

    BLOCK_SIZE = 512
    grid = (triton.cdiv(topk_ids.numel(), BLOCK_SIZE),)
    compute_src2dst_triton_kernel[grid](
        reorder_ids, src2dst, topk_ids.numel(), BLOCK_SIZE
    )
    return reorder_topk_ids, src2dst, seg_indptr, seg_sizes


@triton.jit
def compute_src2dst_deepep_triton_kernel(
    reorder_topk_ids,
    reorder_ids,
    src2dst,
    num_toks,
    offsets_per_expert,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    expert_id = tl.load(reorder_topk_ids + dst_id, mask=mask)
    offset = tl.load(offsets_per_expert + expert_id, mask=mask)
    tl.store(src2dst + src_id, dst_id + offset, mask=mask)


@triton.jit
def fused_preprocess_kernel(
    topk_ids_ptr,
    src2dst_ptr,
    seg_sizes_ptr,
    num_toks,
    m_max,
    start_expert_id,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel to compute src2dst and seg_sizes without sorting.
    Uses atomic operations to assign each token a position within its expert group.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_toks

    expert_id = tl.load(topk_ids_ptr + idx, mask=mask, other=-1)
    valid = expert_id >= 0

    # Atomic increment to get local position within expert group
    # This also accumulates seg_sizes
    local_pos = tl.atomic_add(seg_sizes_ptr + expert_id, 1, mask=mask & valid)

    # Compute destination position: (expert_id - start_expert_id) * m_max + local_pos
    target_slot_id = expert_id - start_expert_id
    dst_pos = target_slot_id * m_max + local_pos

    tl.store(src2dst_ptr + idx, dst_pos, mask=mask & valid)


def run_moe_ep_preprocess_deepep(
    topk_ids: torch.Tensor, num_experts: int, num_recv_tokens_per_expert: torch.Tensor
):
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    num_negative_topk_ids = reorder_topk_ids[reorder_topk_ids < 0].size(0)

    seg_indptr = torch.zeros(num_experts + 1, device=topk_ids.device, dtype=torch.int64)
    compute_seg_indptr_triton_kernel[(num_experts,)](
        reorder_topk_ids, seg_indptr, topk_ids.numel()
    )
    seg_indptr.sub_(num_negative_topk_ids)
    num_recv_tokens_per_expert_indptr = torch.zeros(
        num_experts + 1, device=topk_ids.device, dtype=torch.int64
    )
    num_recv_tokens_per_expert_indptr[1:] = num_recv_tokens_per_expert.cumsum(dim=0)
    offsets_per_expert = (num_recv_tokens_per_expert_indptr - seg_indptr)[1:]

    num_tokens = reorder_topk_ids.size(0) - num_negative_topk_ids
    non_negative_reorder_topk_ids = reorder_topk_ids[num_negative_topk_ids:]
    non_negative_reorder_ids = reorder_ids[num_negative_topk_ids:]

    src2dst = torch.full(
        (topk_ids.numel(),), -1, device=topk_ids.device, dtype=topk_ids.dtype
    )
    BLOCK_SIZE = 512
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)
    compute_src2dst_deepep_triton_kernel[grid](
        non_negative_reorder_topk_ids,
        non_negative_reorder_ids,
        src2dst,
        num_tokens,
        offsets_per_expert,
        BLOCK_SIZE,
    )
    return reorder_topk_ids, src2dst
    

def run_moe_ep_preprocess_deepep_like(
    topk_ids: torch.Tensor, num_experts: int, ep_size: int, m_max: int
):
    """
    Fused preprocessing for MoE expert parallel without expensive sorting.
    Uses atomic operations to compute src2dst mapping and seg_sizes in a single kernel.
    """
    num_toks = topk_ids.numel()
    device = topk_ids.device

    num_local_experts = num_experts // ep_size
    rank = get_expert_parallel_rank()
    start_expert_id = rank * num_local_experts

    # Output tensors
    seg_sizes = torch.zeros(num_experts, device=device, dtype=torch.int64)
    src2dst = torch.zeros(num_toks, device=device, dtype=topk_ids.dtype)

    BLOCK_SIZE = 512
    grid = (triton.cdiv(num_toks, BLOCK_SIZE),)

    # Single fused kernel: computes src2dst and seg_sizes using atomic ops
    fused_preprocess_kernel[grid](
        topk_ids.view(-1),
        src2dst,
        seg_sizes,
        num_toks,
        m_max,
        start_expert_id,
        BLOCK_SIZE,
    )

    # Compute seg_indptr from seg_sizes if needed
    seg_indptr = torch.zeros(num_experts + 1, device=device, dtype=torch.int64)
    seg_indptr[1:] = seg_sizes.cumsum(dim=0)

    # reorder_topk_ids is not used downstream, return None
    return None, src2dst, seg_indptr, seg_sizes


@triton.jit
def pre_reorder_triton_kernel(
    input_ptr,
    gateup_input_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    a1_scales_ptr,
    start_expert_id,
    end_expert_id,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty

    src_idx = tl.program_id(0).to(tl.int64)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk

    src_ptr = input_ptr + src_idx * hidden_size
    for idx in range(topk):
        expert_id = tl.load(topk_ids_ptr + idx)
        if expert_id >= start_expert_id and expert_id <= end_expert_id:
            if a1_scales_ptr is not None:
                scale = 1.0 / tl.load(a1_scales_ptr + expert_id - start_expert_id)
            else:
                scale = 1.0

            dst_idx = tl.load(src2dst_ptr + idx)
            dst_ptr = gateup_input_ptr + dst_idx * hidden_size
            for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
                offset = start_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
                mask = offset < hidden_size
                in_data = tl.load(src_ptr + offset, mask=mask).to(tl.float32)
                out_data = (in_data * scale).to(OutDtype)
                tl.store(dst_ptr + offset, out_data, mask=mask)


@triton.jit
def post_reorder_triton_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    start_expert_id,
    end_expert_id,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = down_output_ptr.dtype.element_ty
    AccDtype = tl.float32

    src_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1).to(tl.int64)
    block_offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = block_offset < hidden_size
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk

    store_ptr = output_ptr + src_idx * hidden_size
    sum_vec = tl.zeros([BLOCK_SIZE], dtype=AccDtype)
    for idx in range(topk):
        expert_id = tl.load(topk_ids_ptr + idx)
        if expert_id >= start_expert_id and expert_id <= end_expert_id:
            dst_idx = tl.load(src2dst_ptr + idx)
            weigh_scale = tl.load(topk_weights_ptr + idx).to(AccDtype)
            load_ptr = down_output_ptr + dst_idx * hidden_size
            in_data = tl.load(load_ptr + block_offset, mask=mask).to(AccDtype)
            sum_vec += in_data * weigh_scale
    tl.store(store_ptr + block_offset, sum_vec.to(InDtype), mask=mask)


# Copied from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_silu_and_mul_mix_quant_ep.py
@triton.jit
def fused_silu_and_mul_per_token_group_quant_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    # for blockwise, BLOCK_N=128
    # for perchannal, BLOCK_N=2048(dpsk)/2056->4096(coder),masked when load as 0 (<size_n)  # noqa: E501
    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + hidden_dim_block_index * stride_output_scale_1
    )

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        )
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(
            output_ptr.dtype.element_ty
        )
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_2,
            output_s,
        )


def next_pow2(n: int) -> int:
    """
    >= n 的最小的 2 的幂
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# Copied from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_silu_and_mul_mix_quant_ep.py
def fused_silu_and_mul_per_token_group_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    use_fp4_w4a8=False,
):
    """
    input shape [expert_num, token_num_padded, hidden_dim]
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype fp8
    output_scale [expert_num token_num_paddded, hidden_dim // 2 // 128] dtype float32
                 (use_fp4_w4a8, quant_group_size==size_n)
    quant_group_size  int,
    masked_m shape [expert_num],
    """

    assert input.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0

    expert_num = len(masked_m)

    if expert_num < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    assert BLOCK_N % quant_group_size == 0

    if use_fp4_w4a8:
        BLOCK_N = next_pow2(BLOCK_N)
        hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    fused_silu_and_mul_per_token_group_quant_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        fp8_max,
        fp8_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
    )


@triton.jit
def _silu_and_mul_post_quant_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    quant_max,
    quant_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + hidden_dim_block_index * stride_output_scale_2
    )

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        )
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / quant_max
        if SCALE_UE8M0:
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
        output_q = tl.clamp(gate_up / output_s, quant_min, quant_max).to(
            output_ptr.dtype.element_ty
        )
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_1,
            output_s,
        )


# copy from ppu-sgl-0.4.10
def silu_and_mul_masked_post_quant_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    output_scale: Optional[torch.Tensor],
    scale_ue8m0: bool = False,
    use_int8: bool = False,  # True
    use_fp8: bool = False,
):
    """
    input shape [expert_num, token_num_padded, hidden_dim]
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype fp8
    output_scale [expert_num token_num_paddded, hidden_dim // 2 // 128] dtype float32
    quant_group_size  int,
    masked_m shape [expert_num],
    """

    assert input.is_contiguous()
    # assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    # cueernt we only suport  channel-wise quant
    # so wo do not need to check the align between intermediate_size and quant groupsize
    # assert size_n % quant_group_size == 0

    expert_num = len(masked_m)

    if expert_num < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    quant_group_size = triton.next_power_of_2(quant_group_size)
    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = 1
    assert BLOCK_N % quant_group_size == 0

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    quant_max = 0
    quant_min = 0
    if use_int8:
        quant_min = torch.iinfo(torch.int8).min
        quant_max = torch.iinfo(torch.int8).max
    elif use_fp8:
        finfo = torch.finfo(torch.float8_e4m3fn)
        quant_max = finfo.max
        quant_min = -quant_max

    _silu_and_mul_post_quant_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        quant_max,
        quant_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=scale_ue8m0,
    )
    return


def get_current_alignment_size():
    forward_context: ForwardContext = get_forward_context()
    is_prefill = forward_context.ep_metadata.is_prompt_batch
    if is_prefill:
        return True, 128  # get_m_alignment_for_contiguous_layout()
    else:
        vllm_config = get_current_vllm_config()
        if getattr(vllm_config, "speculative_config", None):
            max_num_seqs = vllm_config.scheduler_config.max_num_seqs * (
                (vllm_config.speculative_config.num_speculative_tokens or 0) + 1
            )
        else:
            max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        return False, (max_num_seqs + 128 - 1) // 128 * 128


def align_local_tokens(
    hidden_states: torch.Tensor,
    batch_sizes: torch.Tensor,
    acc_batch_sizes: torch.Tensor,
    local_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    is_prefill, alignment_size = get_current_alignment_size()
    remaining_sizes = batch_sizes % alignment_size
    if is_prefill:
        remaining_sizes[remaining_sizes == 0] = alignment_size
    alignment_sizes = alignment_size - remaining_sizes

    m_indices, masked_m, expected_m = None, None, None
    if is_prefill:
        m_indices = torch.arange(
            local_num_experts,
            device=hidden_states.device,
            dtype=torch.int,
        ).repeat_interleave(batch_sizes + alignment_sizes)
    else:
        masked_m = batch_sizes.to(torch.int)
        expected_m = max(
            1, (hidden_states.shape[0] + local_num_experts - 1) // local_num_experts
        )

    hidden_states = torch.cat(
        list(
            itertools.chain(
                *(
                    [
                        hidden_states[acc_batch_sizes[i] : acc_batch_sizes[i + 1]],
                        torch.zeros(
                            (alignment_sizes[i], *hidden_states.shape[1:]),
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        ),
                    ]
                    for i in range(len(acc_batch_sizes) - 1)
                )
            )
        ),
        dim=0,
    )
    return hidden_states, alignment_sizes, m_indices, masked_m, expected_m


def unalign_local_tokens(
    hidden_states: torch.Tensor,
    acc_batch_sizes: torch.Tensor,
    batch_sizes: torch.Tensor,
    alignment_sizes: torch.Tensor,
    local_num_experts: int,
) -> torch.Tensor:
    acc_alignment_sizes = torch.zeros(
        alignment_sizes.shape[0] + 1,
        dtype=alignment_sizes.dtype,
        device=alignment_sizes.device,
    )
    acc_alignment_sizes[1:] = torch.cumsum(alignment_sizes, dim=0)
    acc_aligned_batch_sizes = acc_batch_sizes + acc_alignment_sizes
    hidden_states = torch.cat(
        [
            hidden_states[
                acc_aligned_batch_sizes[i] : acc_aligned_batch_sizes[i] + batch_sizes[i]
            ]
            for i in range(local_num_experts)
        ]
    )
    return hidden_states


def remap_tokens_from_ranks(
    received_batch_sizes: torch.Tensor,
    num_local_tokens: int,
    ep_size: int,
    local_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = received_batch_sizes.device
    acc_received_batch_sizes = torch.zeros(
        received_batch_sizes.shape[0] + 1,
        dtype=received_batch_sizes.dtype,
        device=received_batch_sizes.device,
    )
    acc_received_batch_sizes[1:] = torch.cumsum(received_batch_sizes, dim=0)
    received_batch_sizes = received_batch_sizes.view(
        ep_size, local_num_experts
    ).tolist()
    acc_received_batch_sizes = (
        acc_received_batch_sizes[:-1].view(ep_size, local_num_experts).tolist()
    )
    token_mapping, reverse_token_mapping = [], [-1] * num_local_tokens
    for j in range(local_num_experts):
        for i in range(ep_size):
            offset = acc_received_batch_sizes[i][j]
            count = received_batch_sizes[i][j]
            reverse_token_mapping[offset : offset + count] = range(
                len(token_mapping), len(token_mapping) + count
            )
            token_mapping.extend(list(range(offset, offset + count)))
    assert len(token_mapping) == num_local_tokens
    token_mapping = torch.tensor(token_mapping, dtype=torch.int64, device=device)
    reverse_token_mapping = torch.tensor(
        reverse_token_mapping, dtype=torch.int64, device=device
    )
    return token_mapping, reverse_token_mapping


def reorder_tokens_from_ranks(
    recv_hidden_states: torch.Tensor,
    token_mapping: torch.Tensor,
):
    return recv_hidden_states[token_mapping]


@lru_cache(maxsize=1)
def fp8_quant_scale_fmt_from_oracle():
    return DeepGemmQuantScaleFMT.from_oracle()


def _m_grouped_fp8_gemm_nt_contiguous(
    lhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs: torch.Tensor,
    rhs_scales: torch.Tensor,
    out: torch.Tensor,
    m_indices: torch.Tensor,
) -> None:
    m_grouped_fp8_gemm_nt_contiguous(
        (lhs, lhs_scales), (rhs, rhs_scales), out, m_indices
    )


def _m_grouped_fp8_gemm_nt_contiguous_fake(
    lhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs: torch.Tensor,
    rhs_scales: torch.Tensor,
    out: torch.Tensor,
    m_indices: torch.Tensor,
) -> None:
    return


# Wrap the fp8 grouped_gemm_masked as a custom op with specified
# tags as with torch<2.7.0 (including torch 2.7.0a), it requires
# a column-major scales but torch compile will make it contiguous.
direct_register_custom_op(
    op_name="_m_grouped_fp8_gemm_nt_contiguous",
    op_func=_m_grouped_fp8_gemm_nt_contiguous,
    mutates_args=["out"],
    fake_impl=_m_grouped_fp8_gemm_nt_contiguous_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)


def _apply_grouped_gemm_nt_contiguous(
    input: torch.Tensor,
    weight: torch.Tensor,
    m_indices: torch.Tensor,
    block_size: Optional[List[int]] = None,
    weight_scale: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    use_ppu_int8: Optional[bool] = False,
) -> torch.Tensor:
    assert weight_scale is not None, (
        "Weight scale is required for grouped_gemm kernels for MoE."
    )
    assert m_indices is not None, (
        "m_indices is required for grouped_gemm kernels for MoE."
    )

    output_shape = [*input.shape[:-1], weight.shape[1]]
    if output is None:
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # No need to call the kernel if the input is empty
    if input.numel() == 0 or envs.VLLM_TEST_SKIP_MLP_GEMM:
        return output

    assert envs.VLLM_USE_DEEP_GEMM, "group_gemm's contiguous kernel requires deepgemm."
    if input_scale is None:
        if use_ppu_int8:
            q_input, x_scale = per_token_quant_int8(
                input.view(-1, input.shape[-1]),
            )
        else:
            q_input, x_scale = per_token_group_quant_fp8(
                input.view(-1, input.shape[-1]),
                block_size[1],
                column_major_scales=True,
            )
    else:
        q_input, x_scale = input, input_scale
    if use_ppu_int8:
        m_grouped_int8_gemm_nt_contiguous(
            (q_input, x_scale), (weight, weight_scale), output, m_indices=m_indices
        )
    else:
        torch.ops.vllm._m_grouped_fp8_gemm_nt_contiguous(
            q_input, x_scale, weight, weight_scale, output, m_indices=m_indices
        )

    if bias is not None:
        output = output + bias
    return output.view(*output_shape)


def _fp8_m_grouped_gemm_nt_masked(
    lhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs: torch.Tensor,
    rhs_scales: torch.Tensor,
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> None:
    fp8_m_grouped_gemm_nt_masked(
        (lhs, lhs_scales), (rhs, rhs_scales), out, masked_m, expected_m
    )


def _fp8_m_grouped_gemm_nt_masked_fake(
    lhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs: torch.Tensor,
    rhs_scales: torch.Tensor,
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> None:
    return

# Wrap the fp8 grouped_gemm_masked as a custom op with specified
# tags as with torch<2.7.0 (including torch 2.7.0a), it requires
# a column-major scales but torch compile will make it contiguous.
direct_register_custom_op(
    op_name="_fp8_m_grouped_gemm_nt_masked",
    op_func=_fp8_m_grouped_gemm_nt_masked,
    mutates_args=["out"],
    fake_impl=_fp8_m_grouped_gemm_nt_masked_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)

def _apply_grouped_gemm_nt_masked(
    input: torch.Tensor,
    weight: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: Optional[int],
    block_size: Optional[List[int]] = None,
    weight_scale: Optional[torch.Tensor] = None,
    weight_alpha: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    use_fp4_w4a8: bool = False,
    use_ppu_int8: bool = False,
    use_ppu_int4: bool = False,
    use_nvfp4_w4a4: bool = False,
    expert_ids: Optional[torch.Tensor] = None,
    valid_m: Optional[torch.Tensor] = None,
    offset_m: Optional[torch.Tensor] = None,
    num_tokens_post_pad: Optional[torch.Tensor] = None,
    blockm_for_fp4: Optional[int] = -1,
    configs: tuple[int, ...] | None = None,
    overlap: Optional[bool] = False,
) -> torch.Tensor:
    assert weight_scale is not None, (
        "Weight scale is required for grouped_gemm kernels for MoE."
    )
    assert masked_m is not None, (
        "masked_m is required for grouped_gemm kernels for MoE."
    )
    assert use_fp4_w4a8 or not overlap, (
        "Overlap is required for FP4 kernels for MoE."
    )

    E, n, _ = weight.shape
    if use_ppu_int4:
        n = n * 2
    if use_nvfp4_w4a4:
        output_shape = [*input.permute(2, 0, 1).shape[:-1], n]
    else:
        output_shape = [*input.shape[:-1], n]
    if output is None:
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # No need to call the kernel if the input is empty
    if input.numel() == 0 or envs.VLLM_TEST_SKIP_MLP_GEMM:
        return output

    if use_nvfp4_w4a4:
        flashinfer_cutedsl_grouped_gemm_nt_masked(
            (input, input_scale),
            (weight.permute(1, 2, 0), weight_scale),
            output.permute(1, 2, 0),  # the permute is required by the kernel
            masked_m,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            alpha=weight_alpha.view(1, 1, E),
            alpha_dtype=(
                "float32" if weight_alpha.dtype == torch.float32 else "bfloat16"
            ),
        )
    elif use_fp4_w4a8 or use_ppu_int4:
        assert blockm_for_fp4 != -1
        deepgemm_masked_a8w4 = torch.ops.torch_bladnn.deepgemm_masked_a8w4 if use_fp4_w4a8 else torch.ops.torch_bladnn.deepgemm_masked_a8w4_i8 #noqa
        args = []
        if overlap and use_fp4_w4a8:
            comb = get_comb_overlap_args()
            num_sm = comb.device_sms - comb.num_sms
            deepgemm_masked_a8w4 = \
                torch.ops.torch_bladnn.deepgemm_masked_a8w4_with_signal
            args = [comb.signal, masked_m, num_sm, comb.reduce_hidden]
        deepgemm_masked_a8w4(
            input,  #（e*max_m,k）
            weight,  # noqa (local_e=32, n=4096, k//2)  int8,contain 2 fp4; use_ppu_int4:(e,n//2,k) int8,pack in n-dim
            output.view(-1, output.shape[-1]),  # (e*max_m, n)
            input_scale,  # a_scale: per_channel (e*max_m,)
            weight_scale,  # noqa w_scale: (e,k//128=56,n=4096） bf16; use_ppu_int4:(e,n) fp16, perchannel
            valid_m,
            offset_m,
            expert_ids,
            num_tokens_post_pad,
            blockm_for_fp4,
            None,
            *args,
        )
        assert bias is None, "bias is not supported for expert combine overlap"
        return output
    elif use_ppu_int8:
        assert envs.VLLM_USE_DEEP_GEMM, (
            "int8 moe's grouped_gemm decode requires deepgemm."
        )
        assert input_scale is not None, "Inputs must be quantized with use_ppu_int8."
        int8_m_grouped_gemm_nt_masked(
            (input, input_scale), (weight, weight_scale),
            output, masked_m, expected_m, configs=configs,
        )
    else:
        assert envs.VLLM_USE_DEEP_GEMM, (
            "fp8 moe's grouped_gemm decode requires deepgemm."
        )
        if input_scale is None:
            # NOTE(zycao): Quant kernel only support column major scale for 2D
            # tensor, while masked grouped gemm use 3D tensor.
            q_input, x_scale = per_token_group_quant_fp8(
                input.view(-1, input.shape[-1]),
                block_size[1],
                column_major_scales=False,
            )
            q_input = q_input.view(input.shape)
            x_scale = x_scale.view(
                input.shape[:-1] + (input.shape[-1] // block_size[1],)
            )
        else:
            q_input, x_scale = input, input_scale
        if q_input.ndim == 2 and x_scale.ndim == 2:
            q_input = q_input.unflatten(0, (weight.shape[0], -1))
            x_scale = x_scale.unflatten(0, (weight.shape[0], -1))
        if output.ndim == 2:
            output = output.unflatten(0, (weight.shape[0], -1))
        torch.ops.vllm._fp8_m_grouped_gemm_nt_masked(
            q_input, x_scale, weight, weight_scale, output, masked_m, expected_m
        )

    if bias is not None:
        output = output + bias
    return output.view(*output_shape)


def apply_grouped_gemm_moe(
    input: torch.Tensor,
    weight: torch.Tensor,
    m_indices: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: Optional[int] = None,
    block_size: Optional[List[int]] = None,
    weight_scale: Optional[torch.Tensor] = None,
    weight_alpha: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    use_fp4_w4a8: bool = False,
    use_ppu_int8: bool = False,
    use_ppu_int4: bool = False,
    use_nvfp4_w4a4: bool = False,
    expert_ids: Optional[torch.Tensor] = None,
    valid_m: Optional[torch.Tensor] = None,
    offset_m: Optional[torch.Tensor] = None,
    num_tokens_post_pad: Optional[torch.Tensor] = None,
    blockm_for_fp4: Optional[int] = -1,
    configs: tuple[int, ...] | None = None,
    overlap: Optional[bool] = False,
) -> torch.Tensor:
    assert weight_scale is not None, (
        "Weight scale is required for grouped_gemm kernels for MoE."
    )
    assert m_indices is not None or masked_m is not None, (
        "m_indices or masked_m is required for grouped_gemm kernels for MoE."
    )
    
    n = weight.shape[1]
    if use_ppu_int4:
        n = n * 2
    output_shape = [*input.shape[:-1], n]
    if output is None:
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # No need to call the kernel if the input is empty
    if input.numel() == 0 or envs.VLLM_TEST_SKIP_MLP_GEMM:
        return output

    if m_indices is not None:
        return _apply_grouped_gemm_nt_contiguous(
            input,
            weight,
            m_indices,
            block_size,
            weight_scale,
            input_scale,
            bias,
            output,
            use_ppu_int8,
        )
    else:
        return _apply_grouped_gemm_nt_masked(
            input,
            weight,
            masked_m,
            expected_m,
            block_size,
            weight_scale,
            weight_alpha,
            input_scale,
            bias,
            output,
            use_fp4_w4a8,
            use_ppu_int8,
            use_ppu_int4,
            use_nvfp4_w4a4,
            expert_ids,
            valid_m,
            offset_m,
            num_tokens_post_pad,
            blockm_for_fp4,
            configs,
            overlap,
        )


def parallel_experts_without_data_parallel_deepeplike(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    apply_router_weight_on_input: bool = False,
    expert_map: Optional[torch.Tensor] = None,
    quant_config: Optional[FusedMoEQuantConfig] = None,
    allow_deep_gemm: bool = False,
    shared_experts_callback: Optional[Callable] = None,
    allow_cutlass_block_scaled_grouped_gemm: bool = False,
    map_for_skipped: Optional[torch.Tensor] = None,
):
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    ep_size = get_expert_parallel_world_size()
    local_num_experts = global_num_experts // ep_size

    num_tokens, K = hidden_states.shape
    E, N, _ = w1.shape
    topK = topk_ids.shape[1]

    assert quant_config.use_nvfp4_w4a4, ( # noqa
        "NVFP4 is required for expert parallelism"
    )

    vllm_config = get_current_vllm_config()
    m_max = vllm_config.scheduler_config.max_num_batched_tokens
    num_groups = local_num_experts

    (
        reorder_topk_ids,
        row_id_map,
        acc_batch_sizes,
        batch_sizes,
    ) = run_moe_ep_preprocess_deepep_like(topk_ids, global_num_experts, ep_size, m_max)

    masked_m = batch_sizes[
        local_num_experts * get_expert_parallel_rank() : local_num_experts
        * (get_expert_parallel_rank() + 1)
    ].to(torch.int)

    expected_m = max(
        1, (hidden_states.shape[0] + local_num_experts - 1) // local_num_experts
    )

    permuted_align_hidden_states = torch.empty(
        (num_groups, m_max, hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    
    pre_reorder_triton_kernel[hidden_states.shape[0],](
        hidden_states,
        permuted_align_hidden_states,
        row_id_map,
        topk_ids,
        None,
        local_num_experts * get_expert_parallel_rank(),
        local_num_experts * (get_expert_parallel_rank() + 1) - 1,
        topK,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )

    intermediate_cache1 = torch.empty(
        (num_groups, m_max, N), device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache2 = torch.empty(
        (num_groups, m_max, N // 2),
        device=hidden_states.device, 
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = torch.empty(
        (num_groups, m_max, hidden_states.shape[-1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w1_alpha = quant_config.g1_alphas
    w2_alpha = quant_config.g2_alphas

    x_global_scale = quant_config.a1_gscale
    if envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4:
        x_global_scale, w1_alpha = get_nvfp4_global_scale_and_alpha(
            permuted_align_hidden_states, x_global_scale, masked_m, w1_alpha,
        )
    input_quant, input_scale = scaled_fp4_grouped_quantize(
        permuted_align_hidden_states,
        masked_m,
        x_global_scale,
    )

    input_val = input_quant
    input_scale_val = input_scale
    blockm_for_fp4 = None

    apply_grouped_gemm_moe(
        input_val,
        w1,
        m_indices=None,
        masked_m=masked_m,
        expected_m=expected_m,
        block_size=quant_config.block_shape,
        weight_scale=quant_config.w1_scale,
        weight_alpha=w1_alpha,
        input_scale=input_scale_val,
        output=intermediate_cache1,
        use_nvfp4_w4a4=True,
        blockm_for_fp4=blockm_for_fp4,
    )

    if num_groups * m_max > 0:
        x_global_scale = quant_config.a2_gscale
        if envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4:
            x_global_scale, w2_alpha = get_nvfp4_global_scale_and_alpha(
                intermediate_cache1,
                x_global_scale,
                masked_m,
                w2_alpha,
            )
        (intermediate_cache2, intermediate_cache2_scales
            ) = silu_and_mul_scaled_nvfp4_experts_quantize(
            intermediate_cache1,
            masked_m,
            x_global_scale,
        )
    
    apply_grouped_gemm_moe(
        intermediate_cache2,
        w2,
        m_indices=None,
        masked_m=masked_m,
        expected_m=expected_m,
        block_size=quant_config.block_shape,
        weight_scale=quant_config.w2_scale,
        weight_alpha=w2_alpha,
        input_scale=intermediate_cache2_scales,
        output=intermediate_cache3,
        use_nvfp4_w4a4=True,
        blockm_for_fp4=blockm_for_fp4,
    )

    post_reorder_triton_kernel[
        (hidden_states.shape[0], triton.cdiv(hidden_states.shape[1], 512))
    ](
        intermediate_cache3,
        hidden_states,
        row_id_map,
        topk_ids,
        topk_weights,
        local_num_experts * get_expert_parallel_rank(),
        local_num_experts * (get_expert_parallel_rank() + 1) - 1,
        topK,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )
    return hidden_states


def parallel_experts_without_data_parallel(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    apply_router_weight_on_input: bool = False,
    expert_map: Optional[torch.Tensor] = None,
    quant_config: Optional[FusedMoEQuantConfig] = None,
    allow_deep_gemm: bool = False,
    shared_experts_callback: Optional[Callable] = None,
    allow_cutlass_block_scaled_grouped_gemm: bool = False,
    map_for_skipped: Optional[torch.Tensor] = None,
):
    # Check constraints.
    if quant_config.use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[2], (
            f"Hidden size mismatch: {hidden_states.shape} vs {w1.shape}"
        )
    else:
        assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    ep_size = get_expert_parallel_world_size()
    local_num_experts = global_num_experts // ep_size

    num_tokens, K = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens
    topK = topk_ids.shape[1]

    assert quant_config.use_fp8_w8a8, "FP8 w8a8 is required for expert parallelism"

    (
        reorder_topk_ids,
        row_id_map,
        acc_batch_sizes,
        batch_sizes,
    ) = run_moe_ep_preproess(topk_ids, global_num_experts)

    permuted_hidden_states = torch.empty(
        (M * topK, hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    pre_reorder_triton_kernel[(hidden_states.shape[0],)](
        hidden_states,
        permuted_hidden_states,
        row_id_map,
        topk_ids,
        None,
        local_num_experts * get_expert_parallel_rank(),
        local_num_experts * (get_expert_parallel_rank() + 1) - 1,
        topK,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )

    local_batch_sizes = batch_sizes[
        local_num_experts * get_expert_parallel_rank() : local_num_experts
        * (get_expert_parallel_rank() + 1)
    ]
    local_acc_batch_sizes = (
        acc_batch_sizes[
            local_num_experts * get_expert_parallel_rank() : local_num_experts
            * (get_expert_parallel_rank() + 1)
            + 1
        ]
        - acc_batch_sizes[local_num_experts * get_expert_parallel_rank()]
    )

    (
        aligned_local_hidden_states,
        alignment_sizes,
        m_indices,
        masked_m,
        expected_m,
    ) = align_local_tokens(
        permuted_hidden_states[
            acc_batch_sizes[
                local_num_experts * get_expert_parallel_rank()
            ] : acc_batch_sizes[local_num_experts * (get_expert_parallel_rank() + 1)]
        ],
        local_batch_sizes,
        local_acc_batch_sizes,
        local_num_experts,
    )
    aligned_M = aligned_local_hidden_states.shape[0]

    intermediate_cache1 = torch.empty(
        (aligned_M, N), device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache2 = torch.empty(
        (aligned_M, N // 2), device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache3 = torch.empty(
        (aligned_M, hidden_states.shape[-1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    apply_grouped_gemm_moe(
        aligned_local_hidden_states,
        w1,
        m_indices=m_indices,
        masked_m=masked_m,
        expected_m=expected_m,
        block_size=quant_config.block_shape,
        weight_scale=quant_config.w1_scale,
        input_scale=None,
        output=intermediate_cache1,
    )

    if aligned_M > 0:
        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)

    apply_grouped_gemm_moe(
        intermediate_cache2,
        w2,
        m_indices=m_indices,
        masked_m=masked_m,
        expected_m=expected_m,
        block_size=quant_config.block_shape,
        weight_scale=quant_config.w2_scale,
        input_scale=None,
        output=intermediate_cache3,
    )

    permuted_hidden_states[
        acc_batch_sizes[
            local_num_experts * get_expert_parallel_rank()
        ] : acc_batch_sizes[local_num_experts * (get_expert_parallel_rank() + 1)]
    ] = unalign_local_tokens(
        intermediate_cache3,
        local_acc_batch_sizes,
        local_batch_sizes,
        alignment_sizes,
        local_num_experts,
    )

    post_reorder_triton_kernel[(hidden_states.shape[0],)](
        permuted_hidden_states,
        hidden_states,
        row_id_map,
        topk_ids,
        topk_weights,
        local_num_experts * get_expert_parallel_rank(),
        local_num_experts * (get_expert_parallel_rank() + 1) - 1,
        topK,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )
    return hidden_states


@functools.cache
def get_expert_alignment():
    from deep_gemm import get_m_alignment_for_contiguous_layout

    return get_m_alignment_for_contiguous_layout()


@functools.cache
def get_num_max_dispatch_tokens_per_rank(
    num_experts: int,  # num global experts
):
    if envs.VLLM_NUM_MAX_DISPATCHED_TOKENS_PER_RANK:
        return envs.VLLM_NUM_MAX_DISPATCHED_TOKENS_PER_RANK

    from deep_gemm import get_m_alignment_for_contiguous_layout

    dp_size = get_data_parallel_world_size()  # noqa: F841
    ep_size = get_expert_parallel_world_size()
    tp_size = get_tensor_model_parallel_world_size()
    vllm_config = get_current_vllm_config()

    # alignment requirement:
    #
    # num_max_dispatch_tokens_per_rank = max_num_seqs / tp_size
    # max_m = num_max_dispatch_tokens_per_rank * ep_size, aligned to 128/64

    if envs.VLLM_MOE_USE_DEEPEP_NORMAL:
        if getattr(vllm_config, "speculative_config", None):
            max_num_tokens = vllm_config.scheduler_config.max_num_seqs * (
                (vllm_config.speculative_config.num_speculative_tokens or 0) + 1
            )
        else:
            max_num_tokens = vllm_config.scheduler_config.max_num_seqs
    else:
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

    alignment = max(1, get_m_alignment_for_contiguous_layout() // ep_size) * tp_size
    aligned_max_num_tokens = (max_num_tokens + alignment - 1) // alignment * alignment
    num_max_dispatch_tokens_per_rank = aligned_max_num_tokens // tp_size

    # align to 2^k
    num_max_dispatch_tokens_per_rank = (
        1 << (num_max_dispatch_tokens_per_rank - 1).bit_length()
    )

    return num_max_dispatch_tokens_per_rank


DEEP_EP_BUFFER: Optional["Buffer"] = None


def is_deepep_buffer_initialized() -> bool:
    global DEEP_EP_BUFFER
    return DEEP_EP_BUFFER is not None


def get_deepep_rank() -> int:
    assert is_deepep_buffer_initialized(), "Deepep buffer not initialized"
    global DEEP_EP_BUFFER
    return DEEP_EP_BUFFER.rank


def try_deepep_allgather(
    input_tensor: torch.Tensor,
    use_default_stream: bool,
) -> torch.Tensor:
    """
    Perform in-place allgather operation based on DeepEP.
    Each rank should place its input data into
    `input_tensor` at the index corresponding to its `ep_rank`.
    """
    assert is_deepep_buffer_initialized(), "Deepep buffer not initialized"
    global DEEP_EP_BUFFER
    DEEP_EP_BUFFER.low_latency_allgather(input_tensor, use_default_stream)
    return input_tensor


def init_deep_ep_normal_buffer(
    num_experts: int,
    hidden_size: int,
) -> "Buffer":
    from deep_ep import Buffer

    global DEEP_EP_BUFFER

    hidden_bytes = hidden_size * 2
    num_nvl_bytes, num_rdma_bytes = 0, 0
    group = get_ep_group().device_group

    # Prefilling buffer size hint
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Decoding buffer size hint
    num_max_dispatch_tokens_per_rank = get_num_max_dispatch_tokens_per_rank(num_experts)

    num_rdma_bytes = max(
        num_rdma_bytes,
        Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            hidden=hidden_size,
            num_ranks=get_expert_parallel_world_size(),
            num_experts=num_experts,
        ),
    )

    # Allocate a buffer if not existed or not enough buffer size
    if (
        DEEP_EP_BUFFER is None
        or (
            envs.VLLM_MOE_USE_DEEPEP_NORMAL
            and DEEP_EP_BUFFER.num_nvl_bytes < num_nvl_bytes
        )
        or DEEP_EP_BUFFER.num_rdma_bytes < num_rdma_bytes
    ):
        logger.info(
            f"Using DeepEP, allocating normal buffer, "
            f"num max dispatch tokens per rank: {num_max_dispatch_tokens_per_rank}, "
            f"num experts: {num_experts}, "
            f"num local experts: {num_experts // get_expert_parallel_world_size()}, "
            f"expert parallel size: {get_expert_parallel_world_size()}, "
            f"nvl size: {num_nvl_bytes / 1e6} MB, "
            f"rdma size: {num_rdma_bytes / 1e6} MB ..."
        )
        kwargs = dict(
            group=group,
            num_nvl_bytes=num_nvl_bytes if envs.VLLM_MOE_USE_DEEPEP_NORMAL else 0,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            # align with the default value in DeepEP
            num_qps_per_rank=num_experts // get_expert_parallel_world_size(),
        )
        if envs.VLLM_ENABLE_ELASTIC:
            # Currently, DeepEP only supports shrink mode.
            kwargs["enable_shrink"] = True
            kwargs["num_coll_buffer_bytes"] = envs.VLLM_ELASTIC_COLL_BUFFER_BYTES
            logger.info(
                f"Using Elastic mode, coll buffer size: "
                f"{envs.VLLM_ELASTIC_COLL_BUFFER_BYTES / 1e6} MB ..."
            )
        if (is_ppu() and os.environ.get("SAIL_DEEPEP_ICN", "0")
                .strip()
                .lower()
                in ["1", "true", "yes"]
        ):
            kwargs["allow_nvlink_for_low_latency_mode"] = True
            logger.info("Using ICN for low latency kernel on PPU.")
        if envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL:
            kwargs["allow_mnnvl"] = True
            logger.info("Using MNNVL (multi-node NVLink) for low latency kernel.")

        DEEP_EP_BUFFER = Buffer(**kwargs)
    return DEEP_EP_BUFFER


def cleanup_deep_ep_buffer_after_prefill(vllm_config: VllmConfig):
    global DEEP_EP_BUFFER
    if DEEP_EP_BUFFER is None:
        return

    hf_config = vllm_config.model_config.hf_text_config
    hidden_size = hf_config.hidden_size
    if hasattr(hf_config, "num_experts"):
        global_num_experts = hf_config.num_experts
    elif hasattr(hf_config, "n_routed_experts"):
        global_num_experts = hf_config.n_routed_experts
    else:
        raise ValueError("num_experts or n_routed_experts not found in hf_config")

    # Decoding buffer size hint
    num_max_dispatch_tokens_per_rank = get_num_max_dispatch_tokens_per_rank(
        global_num_experts
    )

    # Low latency buffer cleanup
    DEEP_EP_BUFFER.clean_low_latency_buffer(
        num_max_dispatch_tokens_per_rank, hidden_size, global_num_experts
    )


def get_local_ids(global_num_experts, num_tokens):
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    num_tokens_current_tp_rank = num_tokens // tp_size + int(
        tp_rank < num_tokens % tp_size
    )
    start_idx = tp_rank * (num_tokens // tp_size) + min(tp_rank, num_tokens % tp_size)
    end_idx = start_idx + num_tokens_current_tp_rank
    return start_idx, end_idx


@record_function_or_nullcontext("low_latency_dispatch")
def low_latency_dispatch(
    local_hidden_states: torch.Tensor,
    local_topk_ids: torch.Tensor,
    global_num_experts: int,
    return_recv_hook: bool = False,
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    use_ppu_int8: bool = False,
    use_nvfp4_w4a4: bool = False,
    local_topk_weights: torch.Tensor | None = None,
    x_global_scale: torch.Tensor | None = None,  # required by nvfp4
    overlap: bool = False,
):
    _, hidden_size = local_hidden_states.shape

    buffer: Buffer = init_deep_ep_normal_buffer(global_num_experts, hidden_size)

    num_max_dispatch_tokens_per_rank = get_num_max_dispatch_tokens_per_rank(
        global_num_experts
    )

    kwargs = {}
    if use_fp4_w4a8:
        kwargs['pertoken_quant'] = use_fp4_w4a8
    if overlap:
        comb_args = get_comb_overlap_args()
        kwargs['num_next_clean_int'] = comb_args.num_next_clean_int
    if envs.VLLM_MOE_TOKEN_DROP:
        kwargs['topk_weights'] = local_topk_weights
    if use_ppu_int8:
        kwargs['use_int8'] = use_ppu_int8
        kwargs['quant_size'] = local_hidden_states.shape[-1]
    
    # Low latency dispatch
    # support per-channel quant,absorb dynamic_quantize into deepep dispatch kernel
    (
        packed_recv_x,
        packed_recv_count,
        handle,
        event,
        hook,
    ) = buffer.low_latency_dispatch(
        local_hidden_states,
        local_topk_ids,
        num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        num_experts=global_num_experts,
        use_fp8=use_fp8_w8a8,
        **{
            "round_scale": True,
            "use_ue8m0": True,
        } if use_fp8_w8a8 and is_deep_gemm_e8m0_used() else {},
        **{
            "use_nvfp4": True,
        } if use_nvfp4_w4a4 and envs.VLLM_DEEPEPLL_NVFP4_DISPATCH else {},
        **{
            "x_global_scale": x_global_scale,
        } if use_nvfp4_w4a4 and envs.VLLM_DEEPEPLL_NVFP4_DISPATCH else {},
        async_finish=not return_recv_hook,
        return_recv_hook=return_recv_hook,
        **kwargs,
    )

    event.current_stream_wait() if not return_recv_hook else None
    return packed_recv_x, packed_recv_count, handle, event, hook


@record_function_or_nullcontext("low_latency_combine")
def low_latency_combine(
    intermediate_cache3,
    hidden_states,
    local_hidden_states,
    local_topk_ids,
    local_topk_weights,
    handle,
    global_num_experts,
    num_groups,
    drop_map=None,
    return_recv_hook=False,
    masked_m=None,
    overlap=False,
    blockm=64,
):
    num_tokens, hidden_size = hidden_states.shape

    buffer: Buffer = init_deep_ep_normal_buffer(global_num_experts, hidden_size)
    kwargs = {}
    if overlap:
        return_recv_hook = True
        comb_args = get_comb_overlap_args()
        kwargs['num_sms'] = comb_args.num_sms
        kwargs['reduce_hidden'] = comb_args.reduce_hidden
        kwargs['moe_signal_pad'] = comb_args.signal
        kwargs['num_next_clean_int'] = comb_args.num_next_clean_int
        kwargs['recv_stream'] = comb_args.recv_stream
        kwargs['recv_count'] = masked_m if comb_args.num_sms < 6 else None
        kwargs['fc2_blockm'] = blockm
    if envs.VLLM_MOE_TOKEN_DROP:
        kwargs['drop'] = True
        kwargs['drop_map'] = drop_map

    combined_x, event, hook = buffer.low_latency_combine(
        intermediate_cache3.view(num_groups, -1, local_hidden_states.shape[-1]),
        local_topk_ids,
        local_topk_weights,
        handle=handle,
        async_finish=not return_recv_hook,
        return_recv_hook=return_recv_hook,
        **kwargs
    )

    event.current_stream_wait() if not return_recv_hook else None
    return combined_x, event, hook


# copy from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/layer.py
def prefill_forward_deepgemm_contiguous(
    w1: torch.Tensor,
    w2: torch.Tensor,
    hidden_states: tuple[torch.Tensor, torch.Tensor],
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_recv_tokens_per_expert: List[int],
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    use_ppu_int8: bool = False,
) -> torch.Tensor:
    hidden_states, hidden_states_scale = hidden_states
    if num_recv_tokens_per_expert is None:
        return hidden_states.bfloat16()
    all_tokens = sum(num_recv_tokens_per_expert)
    if all_tokens <= 0:
        return hidden_states.bfloat16()

    M, K = hidden_states.size()
    N = w1.size(1)
    scale_block_size = 128

    hidden_states_shape = hidden_states.shape
    hidden_states_device = hidden_states.device
    hidden_states_dtype = hidden_states.dtype

    input_tensor = [
        torch.empty(
            (all_tokens, K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ),
        (
            torch.empty(
                (all_tokens, K // 128 if not use_ppu_int8 else 1),
                device=hidden_states.device,
                dtype=torch.float32,
            )
        ),
    ]
    m_indices = torch.empty(
        all_tokens, device=hidden_states_device, dtype=torch.int32
    )
    output_index = torch.empty_like(topk_idx)

    num_recv_tokens_per_expert_gpu = torch.tensor(
        num_recv_tokens_per_expert,
        dtype=torch.int32,
        pin_memory=True,
        device="cpu",
    ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

    ep_scatter(
        hidden_states,
        hidden_states_scale,
        topk_idx,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor[0],
        input_tensor[1],
        m_indices,
        output_index,
    )
    hidden_states.set_(
        torch.empty(
            (0,), device=hidden_states_device, dtype=hidden_states_dtype
        )
    )

    gateup_output = torch.empty(
        (all_tokens, N),
        device=hidden_states_device,
        dtype=torch.bfloat16,
    )

    # GPU deepgemm: requires tma aligned input scales
    if not use_ppu_int8:
        input_tensor[1] = tma_align_input_scale(input_tensor[1])

    apply_grouped_gemm_moe(
        input_tensor[0],
        w1,
        m_indices=m_indices,
        masked_m=None,
        weight_scale=w1_scale,
        input_scale=input_tensor[1],
        output=gateup_output,
        use_ppu_int8=use_ppu_int8,
    )
    del input_tensor
    down_input = torch.empty(
        (
            all_tokens,
            N // 2,
        ),
        device=gateup_output.device,
        dtype=torch.bfloat16,
    )
    if gateup_output.numel() > 0:
        torch.ops._C.silu_and_mul(down_input, gateup_output.view(-1, N))
    del gateup_output
    down_output = torch.empty(
        (all_tokens, K),
        device=hidden_states_device,
        dtype=torch.bfloat16,
    )
    if use_ppu_int8:
        down_input_weight, down_input_scale = per_token_quant_int8(
            down_input,
        )
    else:
        down_input_weight, down_input_scale = per_token_group_quant_fp8(
            down_input, scale_block_size, column_major_scales=True,
            use_ue8m0=is_deep_gemm_e8m0_used(),
        )
        # GPU deepgemm: requires tma aligned input scales
        down_input_scale = tma_align_input_scale(down_input_scale)
    del down_input

    apply_grouped_gemm_moe(
        down_input_weight,
        w2,
        m_indices=m_indices,
        masked_m=None,
        weight_scale=w2_scale,
        input_scale=down_input_scale,
        output=down_output,
        use_ppu_int8=use_ppu_int8,
    )
    del down_input_weight, down_input_scale

    gather_out = torch.empty(
        hidden_states_shape,
        device=hidden_states_device,
        dtype=torch.bfloat16,
    )
    ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)
    return gather_out


@record_function_or_nullcontext("parallel_experts_with_data_parallel_deepep")
def parallel_experts_with_data_parallel_deepep(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    apply_router_weight_on_input: bool = False,
    expert_map: Optional[torch.Tensor] = None,
    quant_config: Optional[FusedMoEQuantConfig] = None,
    allow_deep_gemm: bool = False,
    previous_event: Optional["EventOverlap"] = None,
    shared_experts_callback: Optional[Callable] = None,
    allow_cutlass_block_scaled_grouped_gemm: bool = False,
    renormalize: bool = False,
    map_for_skipped: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()

    if not forward_context.ep_metadata.is_prompt_batch:
        assert forward_context.ep_metadata.num_fused_moe_chunks == 1, (
            "Fused MoE chunks > 1 is not supported for decode batches."
        )

    if forward_context.ep_metadata.num_fused_moe_chunks > 1:
        required_tokens = get_tensor_model_parallel_world_size()
        output = torch.empty_like(hidden_states)
        for i in range(forward_context.ep_metadata.num_fused_moe_chunks):
            chunk_begin = min(i * envs.VLLM_FUSED_MOE_CHUNK_SIZE, hidden_states.size(0))
            chunk_end = min(
                (i + 1) * envs.VLLM_FUSED_MOE_CHUNK_SIZE, hidden_states.size(0)
            )
            current_hidden_states = hidden_states[chunk_begin:chunk_end]
            current_topk_weights = topk_weights[chunk_begin:chunk_end]
            current_topk_ids = topk_ids[chunk_begin:chunk_end]
            pad_size = max(0, required_tokens - (chunk_end - chunk_begin))
            if pad_size > 0:
                current_hidden_states = torch.cat(
                    [
                        current_hidden_states,
                        torch.empty(
                            (pad_size,) + hidden_states.shape[1:],
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        ),
                    ]
                )
                current_topk_weights = torch.cat(
                    [
                        current_topk_weights,
                        torch.zeros(
                            (pad_size,) + topk_weights.shape[1:],
                            dtype=topk_weights.dtype,
                            device=topk_weights.device,
                        ),
                    ]
                )
                current_topk_ids = torch.cat(
                    [
                        current_topk_ids,
                        torch.full(
                            (pad_size,) + topk_ids.shape[1:],
                            -1,
                            dtype=topk_ids.dtype,
                            device=topk_ids.device,
                        ),
                    ]
                )
            current_output = parallel_experts_with_data_parallel_deepep_chunk(
                hidden_states=current_hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=current_topk_weights,
                topk_ids=current_topk_ids,
                inplace=inplace,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                quant_config=quant_config,
                allow_deep_gemm=allow_deep_gemm,
                previous_event=previous_event,
                shared_experts_callback=shared_experts_callback,
                allow_cutlass_block_scaled_grouped_gemm=allow_cutlass_block_scaled_grouped_gemm,
            )
            if pad_size > 0:
                current_output = current_output[:-pad_size]
            output[chunk_begin:chunk_end] = current_output
        return output
    else:
        return parallel_experts_with_data_parallel_deepep_chunk(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=quant_config,
            allow_deep_gemm=allow_deep_gemm,
            previous_event=previous_event,
            shared_experts_callback=shared_experts_callback,
            allow_cutlass_block_scaled_grouped_gemm=allow_cutlass_block_scaled_grouped_gemm,
        )


def compute_blockm_for_deepgemm_fp4(expectm):
    if is_ppu():
        blockm = 64
        if expectm <= 4 or expectm <= 16:
            blockm = 16
        elif expectm <= 40:
            blockm = 32 
    else: # h20
        if expectm <= 4:
            blockm = 8
        elif expectm <= 16:
            blockm = 16
        elif expectm <= 40:
            blockm = 32
        else:
            blockm = 64
    return blockm

def prepare_for_fp4(global_b, m_max, topK, blockm, local_num_experts, masked_m, device):
    max_tokens = global_b * topK + local_num_experts * (blockm - 1)
    max_mblocks = (max_tokens + blockm - 1) // blockm
    expert_ids = torch.empty(max_mblocks, dtype=torch.int32, device=device)
    valid_m = torch.empty(max_mblocks, dtype=torch.int32, device=device)
    offset_m = torch.empty(max_mblocks, dtype=torch.int32, device=device)

    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    torch.ops.torch_bladnn.deepgemm_preprocess(
        blockm,
        m_max,  # max_shape of A
        local_num_experts,  # local_e
        masked_m,
        expert_ids,
        valid_m,
        offset_m,
        num_tokens_post_pad,
    )
    return expert_ids, valid_m, offset_m, num_tokens_post_pad


@functools.cache
def get_deep_gemm_best_configs():
    import json

    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
    if not os.path.exists(config_dir):
        return None
    configs = os.listdir(config_dir)

    config_dict_in_all = dict()

    device_name = get_device_name().replace(" ", "_")
    device_name_signature = f",device_name={device_name}-"

    logger.info(f"DeepGEMM start loading config: {configs}")

    for config in configs:
        if device_name_signature not in config:
            logger.info(f"skipping device_name_signature {config}")
            continue
        logger.info(f"loading device_name_signature {config}")

        config_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "configs", config
        )
        config_dict = dict()

        try:
            with open(config_file_path) as f:
                config_dict = json.load(f)
            config_dict = {
                (x["M"], x["N"], x["K"], x["num_groups"]): x for x in config_dict
            }
        except Exception:
            logger.warning(f"Empty config found in {config_file_path}.")

        if config_dict:
            config_dict_in_all.update(config_dict)

    return config_dict_in_all


# copy from ppu-sgl
MAX_DECODE_BS = 129
CANDIDATE_Ms = [1, 2, 4, 8] + list(range(16, 129, 16))
non_tuned_mnk_set = set()


def append_non_tuned_case(M, N, K, NUM_GROUP):
    if M < MAX_DECODE_BS:
        # note that we save configs with N before N
        non_tuned_mnk_set.add((M, K, N, NUM_GROUP))


@functools.cache
def get_ppu_deep_gemm_config(M, N, K, num_groups) -> tuple[int, ...] | None:
    best_configs = get_deep_gemm_best_configs()
    if best_configs is not None:
        # find neariest config
        config = None
        if (M, N, K, num_groups) in best_configs:
            config = eval(best_configs[(M, N, K, num_groups)]["config"])
        else:

            def find_closest(lst, x):
                # As a dividing line of P and D
                if M > MAX_DECODE_BS:
                    return -1
                else:
                    return min(lst, key=lambda y: abs(y - x))

            candidate_Ms = CANDIDATE_Ms
            M_candidate = find_closest(candidate_Ms, M)
            if (M_candidate, N, K, num_groups) in best_configs:
                config = eval(best_configs[(M_candidate, N, K, num_groups)]["config"])

        if config is None:
            # Save as a pre-tuning config
            if M < MAX_DECODE_BS:
                append_non_tuned_case(M, N, K, num_groups)
            return None

        num_sms = config["num_min_sms"]
        block_m = config["best_block_m"]
        block_n = config["best_block_n"]
        block_k = config["block_k"]
        warp_m = config["warp_m"]
        warp_n = config["warp_n"]
        num_stages = config["best_num_stages"]
        smem_config = config["best_smem_config"]
        if "extra_info" not in config:
            extra_info = {}
            use_cutlass3 = False
            use_multistage_on_N = False
            if "DG_USE_CUTLASS3" in os.environ:
                use_cutlass3 = int(os.getenv("DG_USE_CUTLASS3"))
            extra_info["use_cutlass3"] = use_cutlass3
            if "DG_USE_MULTISTAGE_ON_N" in os.environ:
                use_multistage_on_N = int(os.getenv("DG_USE_MULTISTAGE_ON_N"))
            extra_info["use_multistage_on_N"] = use_multistage_on_N
        else:
            extra_info = config["extra_info"]
        return (
            num_sms,
            block_m,
            block_n,
            block_k,
            warp_m,
            warp_n,
            num_stages,
            smem_config,
            extra_info,
        )
    else:
        append_non_tuned_case(M, N, K, num_groups)
        return None


@record_function_or_nullcontext("parallel_experts_with_data_parallel_deepep_chunk")
def parallel_experts_with_data_parallel_deepep_chunk(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    quant_config: FusedMoEQuantConfig | None = None,
    allow_deep_gemm: bool = False,
    previous_event: Optional["EventOverlap"] = None,
    shared_experts_callback: Callable | None = None,
    allow_cutlass_block_scaled_grouped_gemm: bool = False,
    map_for_skipped: torch.Tensor | None = None,
) -> torch.Tensor:
    from deep_ep import Buffer, EventOverlap

    dp_size = get_data_parallel_world_size()
    ep_size = get_expert_parallel_world_size()
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    local_num_experts = global_num_experts // ep_size

    num_tokens, hidden_size = hidden_states.shape
    E, N, _ = w1.shape

    fc2_use_int4_w4a8 = False
    if quant_config.use_int4_w4a8:
        N = 2 * N
        if w2.shape[1] != quant_config.w2_scale.shape[-1]:
            fc2_use_int4_w4a8 = True

    M = num_tokens  # noqa: F841
    topK = topk_ids.shape[1]

    assert (
        quant_config.use_fp8_w8a8
        or quant_config.use_int8_w8a8
        or quant_config.use_int4_w4a8
        or quant_config.use_nvfp4_w4a4
    ), ( # noqa
        "FP8/Int8 w8a8/Int4/NVFP4 is required for expert parallelism"
    )
    forward_context: ForwardContext = get_forward_context()
    is_prompt_batch = forward_context.ep_metadata.is_prompt_batch
    overlap = init_and_get_overlap(local_num_experts, global_num_experts,
                                   hidden_size, hidden_states.device)

    num_tokens_current_tp_rank = num_tokens // tp_size + int(
        tp_rank < num_tokens % tp_size
    )
    if (
        tp_size > 1
        and ep_size > 1
        and envs.VLLM_TPEP_REDUCE_SCATTER
    ):
        start_idx = 0
        end_idx = num_tokens
    else:
        start_idx = tp_rank * (num_tokens // tp_size) + min(
            tp_rank, num_tokens % tp_size
        )
        end_idx = start_idx + num_tokens_current_tp_rank

    local_hidden_states = hidden_states[start_idx:end_idx]

    # required by deepep: note that the int64 casting must be done
    # before slicing otherwise deepep may get a strange nullptr
    # for topk_ids in normal dispatch, not know why.
    local_topk_ids = topk_ids.to(torch.int64)[start_idx:end_idx]
    local_topk_weights = topk_weights[start_idx:end_idx]

    # compute config for fp4
    expected_m = max(1, num_tokens * dp_size * topK // global_num_experts)
    # bladnn's fp4 w4a8 store weights in int8.
    use_fp4_w4a8 = False
    use_ppu_int8 = False
    use_ppu_int4 = False
    use_nvfp4_w4a4 = quant_config.use_nvfp4_w4a4
    if w1.dtype == torch.int8 and not quant_config.use_int8_w8a16:
        if current_platform.is_ppu():
            assert allow_deep_gemm
            if quant_config.use_int8_w8a8:
                use_ppu_int8 = True
            elif quant_config.use_int4_w4a8:
                use_ppu_int4 = True
        else:
            use_fp4_w4a8 = True
            assert sm_version >= 90
    blockm_for_fp4 = None
    if use_fp4_w4a8 or use_ppu_int4:
        blockm_for_fp4 = compute_blockm_for_deepgemm_fp4(expected_m) # also compute int4

    # 1. When deepep normal is not enabled, using LL kernels for both prefill
    #    and decode. It will be used when we use piecewise cudagraph with deepep
    #.   where the piecewise compilation will capture with prompt batches.
    #
    # 2. DeepEP normal does not support elastic yet, when elastic is enabled,
    #    use DeepEP LL
    if (is_prompt_batch
        and envs.VLLM_MOE_USE_DEEPEP_NORMAL
        and not envs.VLLM_ENABLE_ELASTIC
    ):
        if envs.VLLM_TEST_SKIP_DEEPEP_PREFILL:
            # Skip deepep prefill for cases where EP is only supported for decode,
            # e.g., EP with fp4 a8w4.
            return hidden_states

        # Some expert parallelism doesn't support prefill forward
        assert not use_fp4_w4a8, (
            "deepep fp4 only supports decode, please run fp8 for prefill, or, "
            "set VLLM_MOE_USE_DEEPEP_NORMAL=0 to use low latency kernels for prefill."
        )

        # PPU normal ep is not well tested.
        assert not use_ppu_int8, (
            "deepep int8 on ppu only supports decode, please run tp for prefill, or, "
            "set VLLM_MOE_USE_DEEPEP_NORMAL=0 to use low latency kernels for prefill."
        )
        # PPU normal ep with int4 doesn't support prefill.
        assert not use_ppu_int4, (
            "deepep int4 only support decode, please run tp for prefill batches"
        )

        # NVFP4: prefill with expert parallel is not supported.
        assert not use_nvfp4_w4a4, (
            "deepep nvfp4 only supports decode, please run tp or fp8 for prefill."
        )

        buffer: Buffer = init_deep_ep_normal_buffer(global_num_experts, hidden_size)

        if quant_config.use_fp8_w8a8 or use_fp4_w4a8:
            h, h_s = per_token_group_quant_fp8(
                local_hidden_states.view(-1, local_hidden_states.shape[-1]),
                quant_config.block_shape[1],
                column_major_scales=False,
                use_ue8m0=is_deep_gemm_e8m0_used(),
            )
            h = h.view(local_hidden_states.shape)
            h_s = h_s.view(
                local_hidden_states.shape[:-1]
                + (local_hidden_states.shape[-1] // quant_config.block_shape[1],)
            )
            local_hidden_states = (h, h_s)
        elif use_ppu_int8:
            h, h_s = per_token_quant_int8(
                local_hidden_states.view(-1, local_hidden_states.shape[-1]),
            )
            h = h.view(local_hidden_states.shape)
            h_s = h_s.view(local_hidden_states.shape[:-1] + (1,))
            local_hidden_states = (h, h_s)
        else:
            raise ValueError("Requires fp8/fp4/int8 w8a8 for deepep prefill")

        # Calculate layout before actual dispatch
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            local_topk_ids,
            global_num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        # Normal dispatch
        (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(
            local_hidden_states,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=local_topk_ids,
            topk_weights=local_topk_weights,
            expert_alignment=get_expert_alignment(),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        recv_x = prefill_forward_deepgemm_contiguous(
            w1=w1,
            w2=w2,
            hidden_states=recv_x,
            topk_idx=recv_topk_ids,
            topk_weights=recv_topk_weights,
            num_recv_tokens_per_expert=num_recv_tokens_per_expert_list,
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            use_ppu_int8=use_ppu_int8,
        )

        # Normal combine
        combined_x, event, hook = buffer.combine(
            recv_x,
            topk_weights=recv_topk_weights,
            handle=handle,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
    else:
        if use_nvfp4_w4a4:
            x_global_scale = quant_config.a1_gscale
            if (envs.VLLM_DEEPEPLL_NVFP4_DISPATCH
                and envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4
            ):
                assert dp_size == 1, (
                    "Dynamic global scale for nvfp4 with EP doesn't support DP."
                )
                x_global_scale, _ = get_nvfp4_global_scale_and_alpha(
                    # N.B., use the global hidden states (rather than local slice)
                    # to compute scales
                    hidden_states,
                    x_global_scale,
                )
        else:
            x_global_scale = None

        w1_alpha = quant_config.g1_alphas
        w2_alpha = quant_config.g2_alphas

        (
            # packed_recv_x[0]:(e,maxm,k),int8 / packed_recv_x[1]:(e,maxm,1),fp32
            packed_recv_x,
            packed_recv_count,
            handle,
            event,
            hook,
        ) = low_latency_dispatch(
            local_hidden_states,
            local_topk_ids,
            global_num_experts,
            return_recv_hook=shared_experts_callback is not None,
            use_fp8_w8a8=quant_config.use_fp8_w8a8 or use_fp4_w4a8,
            use_fp4_w4a8=use_fp4_w4a8,
            use_ppu_int8=use_ppu_int8 or use_ppu_int4,
            use_nvfp4_w4a4=use_nvfp4_w4a4,
            local_topk_weights=local_topk_weights,
            **{
                "x_global_scale": x_global_scale,
            } if envs.VLLM_DEEPEPLL_NVFP4_DISPATCH else {},
            overlap=overlap,
        )

        if not isinstance(packed_recv_x, tuple):
            packed_recv_x = (packed_recv_x, None)

        if quant_config.use_nvfp4_w4a4 and envs.VLLM_DEEPEPLL_NVFP4_DISPATCH:
            num_groups, m_max, K = packed_recv_x[0].permute(2, 0, 1).shape
            K *= 2  # K is in packed form when nvfp4 is on.
        else:
            num_groups, m_max, K = packed_recv_x[0].shape

        # TODO: Use memory pool to avoid repeated allocations
        # fc1
        intermediate_cache1 = torch.empty(
            (num_groups, m_max, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if shared_experts_callback is not None and overlap:
            # When using EPO, shared expert overlap with dispatch,
            # fc2 overlap with combine. Otherwize, shared expert
            # overlap with combine.
            shared_output = shared_experts_callback(hidden_states)

        deep_gemm_configs = None
        expert_ids = valid_m = offset_m = num_tokens_post_pad = None
        if use_nvfp4_w4a4:
            if envs.VLLM_DEEPEPLL_NVFP4_DISPATCH:
                assert not envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4
                input_quant, input_scale = packed_recv_x[0], packed_recv_x[1]
                if envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4:
                    w1_alpha = 1 / x_global_scale * w1_alpha
            else:
                if envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4:
                    x_global_scale, w1_alpha = get_nvfp4_global_scale_and_alpha(
                        packed_recv_x[0], x_global_scale, packed_recv_count, w1_alpha,
                    )
                input_quant, input_scale = scaled_fp4_grouped_quantize(
                    packed_recv_x[0],
                    packed_recv_count,
                    x_global_scale,
                )
        elif use_fp4_w4a8 or use_ppu_int4:
            # only dq last dim, need to merge first 2 dims
            input_quant = packed_recv_x[0].view(  # (e*m,k) int8
                -1, packed_recv_x[0].shape[-1]
            )  # NOTE: bladnn perchanl-fp4-moe use 2-dim tensor
            input_scale_dtype = local_hidden_states.dtype if use_fp4_w4a8 else torch.float16 # noqa
            input_scale = (  # (e*m,) fp16
                packed_recv_x[1]
                .to(input_scale_dtype)
                .view(
                    num_groups * m_max,
                )
                .contiguous()
            )
            assert input_quant.is_contiguous()
            assert input_scale.is_contiguous()
            expert_ids, valid_m, offset_m, num_tokens_post_pad = prepare_for_fp4(
                num_tokens * dp_size,
                m_max,
                topK,
                blockm_for_fp4,
                local_num_experts,
                packed_recv_count,
                packed_recv_x[0].device,
            )
        elif use_ppu_int8:
            n = w1.size(1)
            expected_m = min(expected_m, m_max)
            deep_gemm_configs = get_ppu_deep_gemm_config(
                expected_m, n, K, num_groups=num_groups
            )
            input_quant, input_scale = packed_recv_x[0], packed_recv_x[1]
        else:
            assert packed_recv_x[1] is not None, (
                "BF16 dispatch is not supported for non-nvfp4 quantization "
                "(e.g., expert parallelism with fp8)."
            )
            input_quant, input_scale = packed_recv_x[0], packed_recv_x[1]

        if envs.VLLM_MOE_TOKEN_DROP:
            drop_num, dropped_recv_x, dropped_recv_count, drop_map = moe_token_drop(
                packed_recv_topk_weights=handle[5][0],
                packed_recv_count=packed_recv_count,
                packed_recv_x=(input_quant, input_scale),
                num_avg_token_per_expert=expected_m,
                num_avg_token_per_rank=expected_m*local_num_experts,
            )
            if envs.VLLM_MOE_TOKEN_DROP_STATS > 0:
                # All layers drop ratio
                get_drop_ctx().update(drop_num, packed_recv_count, -1)

        masked_m_val = (dropped_recv_count if envs.VLLM_MOE_TOKEN_DROP
                        else packed_recv_count)
        input_val = (dropped_recv_x[0] if envs.VLLM_MOE_TOKEN_DROP
                     else input_quant)
        input_scale_val = (dropped_recv_x[1] if envs.VLLM_MOE_TOKEN_DROP
                           else input_scale)

        apply_grouped_gemm_moe(
            input_val,
            w1,
            m_indices=None,
            masked_m=masked_m_val,
            expected_m=expected_m,
            block_size=quant_config.block_shape,
            weight_scale=quant_config.w1_scale,
            weight_alpha=w1_alpha,
            input_scale=input_scale_val,
            output=intermediate_cache1,
            use_fp4_w4a8=use_fp4_w4a8,
            use_ppu_int8=use_ppu_int8,
            use_ppu_int4=use_ppu_int4,
            use_nvfp4_w4a4=use_nvfp4_w4a4,
            expert_ids=expert_ids,
            valid_m=valid_m,
            offset_m=offset_m,
            num_tokens_post_pad=num_tokens_post_pad,
            blockm_for_fp4=blockm_for_fp4,
            configs=deep_gemm_configs,
        )
        del packed_recv_x
        if envs.VLLM_MOE_TOKEN_DROP:
            del dropped_recv_x
            
        if use_ppu_int4 and not fc2_use_int4_w4a8:
            use_ppu_int8 = True # fc2 fall back to int8
            use_ppu_int4 = False
        
        # silu and mul
        if num_groups * m_max > 0:
            if use_nvfp4_w4a4:
                x_global_scale = quant_config.a2_gscale
                if envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4:
                    x_global_scale, w2_alpha = get_nvfp4_global_scale_and_alpha(
                        intermediate_cache1,
                        x_global_scale,
                        packed_recv_count,
                        w2_alpha,
                    )
                (intermediate_cache2, intermediate_cache2_scales
                 ) = silu_and_mul_scaled_nvfp4_experts_quantize(
                    intermediate_cache1,
                    packed_recv_count,
                    x_global_scale,
                )
            elif use_fp4_w4a8:
                intermediate_cache2 = torch.empty(
                    (num_groups * m_max, N // 2),
                    device=hidden_states.device,
                    dtype=current_platform.fp8_dtype(),
                )
                intermediate_cache2_scales = torch.empty(
                    (num_groups * m_max, 1),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                fused_silu_and_mul_per_token_group_quant(
                    intermediate_cache1.view(num_groups * m_max, N).unflatten(
                        0, (num_groups, m_max)
                    ),
                    intermediate_cache2.unflatten(0, (num_groups, m_max)),
                    intermediate_cache2_scales.unflatten(0, (num_groups, m_max)),
                    quant_group_size=N // 2,  # per channel
                    masked_m=masked_m_val,
                    use_fp4_w4a8=True,
                )
            elif use_ppu_int8 or use_ppu_int4:
                # perchannel_quant into int8
                intermediate_cache2 = torch.empty(
                    (num_groups, m_max, N // 2),
                    device=hidden_states.device,
                    dtype=torch.int8,
                )
                intermediate_cache2_scales = torch.empty(
                    (num_groups, m_max, 1),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                silu_and_mul_masked_post_quant_fwd(
                    intermediate_cache1,  # in: （e, m, 2*n）  bf16
                    intermediate_cache2,  # out:（e, max_m, n） int8
                    N // 2,  # quant_group_size:perchannel
                    packed_recv_count,
                    intermediate_cache2_scales,  # scale（e, max_m, n/blocksize=1） fp32
                    use_fp8=False,
                    use_int8=True,
                    scale_ue8m0=False,
                )
            elif current_platform.is_device_capability_family(100):
                intermediate_cache2, intermediate_cache2_scales = (
                    persistent_masked_m_silu_mul_quant(
                        intermediate_cache1,
                        packed_recv_count,
                        quant_scale_fmt=fp8_quant_scale_fmt_from_oracle(),
                    )
                )
            else:
                intermediate_cache2 = torch.empty(
                    (num_groups, m_max, N // 2),
                    device=hidden_states.device,
                    dtype=current_platform.fp8_dtype(),
                )
                num_blocks = N // 2 // quant_config.block_shape[-1]
                intermediate_cache2_scales_colmajor = torch.empty(
                    (num_groups, num_blocks, m_max),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                fused_silu_and_mul_per_token_group_quant(
                    intermediate_cache1,
                    intermediate_cache2,
                    intermediate_cache2_scales_colmajor,
                    quant_group_size=quant_config.block_shape[-1],
                    masked_m=masked_m_val,
                )
                intermediate_cache2_scales = (
                    intermediate_cache2_scales_colmajor.permute(0, 2, 1)
                )
        else:
            intermediate_cache2 = torch.empty(
                (num_groups, m_max, N // 2),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            intermediate_cache2_scales = None
        del intermediate_cache1
        combine_overlap_sync(overlap=overlap)

        # fc2
        intermediate_cache3 = torch.empty(
            (num_groups, m_max, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        deep_gemm_configs = None
        if use_fp4_w4a8:
            intermediate_cache2_scales = intermediate_cache2_scales.squeeze(1).to(
                hidden_states.dtype
            )
        elif use_ppu_int4:
            # (e,max, n) -> (e*max,n)
            intermediate_cache2 = intermediate_cache2.view(-1, N // 2)
            # (e, max, 1) -> (e*max,)
            intermediate_cache2_scales = intermediate_cache2_scales.view(-1).to(torch.float16) # noqa
        elif use_ppu_int8:
            n = w2.size(1)
            scale_block_size = N // 2
            expected_m = min(expected_m, m_max)
            deep_gemm_configs = get_ppu_deep_gemm_config(
                expected_m, n, scale_block_size, num_groups=num_groups
            )

        apply_grouped_gemm_moe(
            intermediate_cache2,
            w2,
            m_indices=None,
            masked_m=masked_m_val,
            expected_m=expected_m,
            block_size=quant_config.block_shape,
            weight_scale=quant_config.w2_scale,
            weight_alpha=w2_alpha,
            input_scale=intermediate_cache2_scales,
            output=intermediate_cache3,
            use_fp4_w4a8=use_fp4_w4a8,
            use_ppu_int8=use_ppu_int8,
            use_ppu_int4=use_ppu_int4,
            use_nvfp4_w4a4=use_nvfp4_w4a4,
            expert_ids=expert_ids,
            valid_m=valid_m,
            offset_m=offset_m,
            num_tokens_post_pad=num_tokens_post_pad,
            blockm_for_fp4=blockm_for_fp4,
            configs=deep_gemm_configs,
            overlap=overlap,
        )
        with combine_overlap_consumer(overlap=overlap):
            combined_x, event, hook = low_latency_combine(
                intermediate_cache3,
                hidden_states,
                local_hidden_states,
                local_topk_ids,
                local_topk_weights,
                handle,
                global_num_experts,
                num_groups,
                drop_map=drop_map if envs.VLLM_MOE_TOKEN_DROP else None,
                return_recv_hook=shared_experts_callback is not None,
                masked_m=masked_m_val,
                overlap=overlap,
                blockm=blockm_for_fp4)
        if overlap:
            wait_combine_finish(hook)
        elif shared_experts_callback is not None:
            hook()

        del intermediate_cache2
        del intermediate_cache3

    if tp_size > 1:
        hidden_states.zero_()
        hidden_states[start_idx:end_idx] = combined_x
        combined_x = hidden_states

    if shared_experts_callback is not None and not is_prompt_batch:
        hf_config = get_current_vllm_config().model_config.hf_config
        if hidden_states.dtype != torch.float16:
            combined_x = combined_x * hf_config.routed_scaling_factor + shared_output
        else:
            # Fix FP16 overflow
            # See DeepseekV2DecoderLayer for more details.
            combined_x = combined_x + shared_output * (
                1.0 / hf_config.routed_scaling_factor
            )

    return combined_x


@record_function_or_nullcontext("parallel_experts_with_data_parallel")
def parallel_experts_with_data_parallel(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    apply_router_weight_on_input: bool = False,
    expert_map: Optional[torch.Tensor] = None,
    quant_config: Optional[FusedMoEQuantConfig] = None,
    allow_deep_gemm: bool = False,
    shared_experts_callback: Optional[Callable] = None,
    allow_cutlass_block_scaled_grouped_gemm: bool = False,
    map_for_skipped: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ep_size = get_expert_parallel_world_size()
    local_num_experts = global_num_experts // ep_size

    num_tokens, K = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens
    topK = topk_ids.shape[1]

    assert quant_config.use_fp8_w8a8, "FP8 w8a8 is required for expert parallelism"

    (
        reorder_topk_ids,
        row_id_map,
        acc_batch_sizes,
        batch_sizes,
    ) = run_moe_ep_preproess(topk_ids, global_num_experts)

    permuted_hidden_states = torch.empty(
        (M * topK, hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    pre_reorder_triton_kernel[(hidden_states.shape[0],)](
        hidden_states,
        permuted_hidden_states,
        row_id_map,
        topk_ids,
        None,
        0,
        global_num_experts - 1,
        topK,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )

    # Metadata all-to-all
    recieved_batch_sizes = torch.zeros_like(batch_sizes)
    torch.distributed.all_to_all_single(
        recieved_batch_sizes, batch_sizes, group=get_ep_group().device_group
    )

    num_send_tokens = batch_sizes.view(ep_size, -1).sum(dim=-1)
    num_recv_tokens = recieved_batch_sizes.view(ep_size, -1).sum(dim=1)
    num_local_tokens = num_recv_tokens.sum().item()
    num_recv_tokens_list = num_recv_tokens.tolist()
    num_send_tokens_list = num_send_tokens.tolist()

    recv_hidden_states = torch.empty(
        num_local_tokens,
        hidden_states.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    # Global all-to-all: dispatch
    torch.distributed.all_to_all_single(
        recv_hidden_states,
        permuted_hidden_states,
        num_recv_tokens_list,
        num_send_tokens_list,
        group=get_ep_group().device_group,
    )

    local_batch_sizes = recieved_batch_sizes.view(ep_size, -1).sum(dim=0)
    local_acc_batch_sizes = torch.zeros(
        len(local_batch_sizes) + 1,
        dtype=local_batch_sizes.dtype,
        device=local_batch_sizes.device,
    )
    local_acc_batch_sizes[1:] = torch.cumsum(local_batch_sizes, dim=0)

    # reorder received tokens in the expert order
    remap_mapping, remap_mapping_reversed = remap_tokens_from_ranks(
        recieved_batch_sizes, num_local_tokens, ep_size, local_num_experts
    )
    recv_hidden_states = reorder_tokens_from_ranks(recv_hidden_states, remap_mapping)

    (
        aligned_local_hidden_states,
        alignment_sizes,
        m_indices,
        masked_m,
        expected_m,
    ) = align_local_tokens(
        recv_hidden_states, local_batch_sizes, local_acc_batch_sizes, local_num_experts
    )
    aligned_M = aligned_local_hidden_states.shape[0]

    intermediate_cache1 = torch.empty(
        (aligned_M, N), device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache2 = torch.empty(
        (aligned_M, N // 2), device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache3 = torch.zeros(
        (aligned_M, hidden_states.shape[-1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    apply_grouped_gemm_moe(
        aligned_local_hidden_states,
        w1,
        m_indices=m_indices,
        masked_m=masked_m,
        expected_m=expected_m,
        block_size=quant_config.block_shape,
        weight_scale=quant_config.w1_scale,
        input_scale=None,
        output=intermediate_cache1,
    )

    if aligned_M > 0:
        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)

    apply_grouped_gemm_moe(
        intermediate_cache2,
        w2,
        m_indices=m_indices,
        masked_m=masked_m,
        expected_m=expected_m,
        block_size=quant_config.block_shape,
        weight_scale=quant_config.w2_scale,
        input_scale=None,
        output=intermediate_cache3,
    )

    result_hidden_states = unalign_local_tokens(
        intermediate_cache3,
        local_acc_batch_sizes,
        local_batch_sizes,
        alignment_sizes,
        local_num_experts,
    )
    reorder_tokens_from_ranks(result_hidden_states, remap_mapping_reversed)

    # Reversed global all-to-all: combine
    torch.distributed.all_to_all_single(
        permuted_hidden_states,
        result_hidden_states,
        num_send_tokens_list,
        num_recv_tokens_list,
        group=get_ep_group().device_group,
    )

    post_reorder_triton_kernel[(hidden_states.shape[0],)](
        permuted_hidden_states,
        hidden_states,
        row_id_map,
        topk_ids,
        topk_weights,
        0,
        global_num_experts - 1,
        topK,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )
    return hidden_states


@record_function_or_nullcontext("parallel_experts_with_data_parallel_multi_alltoall")
def parallel_experts_with_data_parallel_multi_alltoall(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    apply_router_weight_on_input: bool = False,
    expert_map: Optional[torch.Tensor] = None,
    quant_config: Optional[FusedMoEQuantConfig] = None,
    allow_deep_gemm: bool = False,
    shared_experts_callback: Optional[Callable] = None,
    allow_cutlass_block_scaled_grouped_gemm: bool = False,
    map_for_skipped: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dp_size = get_data_parallel_world_size()
    ep_size = get_expert_parallel_world_size()
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    local_num_experts = global_num_experts // ep_size

    num_tokens, K = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens
    topK = topk_ids.shape[1]

    assert quant_config.use_fp8_w8a8, "FP8 w8a8 is required for expert parallelism"

    (
        reorder_topk_ids,
        row_id_map,
        acc_batch_sizes,
        batch_sizes,
    ) = run_moe_ep_preproess(topk_ids, global_num_experts)

    permuted_hidden_states = torch.empty(
        (M * topK, hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    pre_reorder_triton_kernel[(hidden_states.shape[0],)](
        hidden_states,
        permuted_hidden_states,
        row_id_map,
        topk_ids,
        None,
        0,
        global_num_experts - 1,
        topK,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )

    # Metadata all-to-all
    send_batch_sizes = (
        batch_sizes.view(dp_size, tp_size, -1)[:, tp_rank].flatten().contiguous()
    )
    recieved_batch_sizes = torch.zeros_like(send_batch_sizes)
    torch.distributed.all_to_all_single(
        recieved_batch_sizes, send_batch_sizes, group=get_dp_group().device_group
    )

    num_recv_tokens = recieved_batch_sizes.view(dp_size, -1).sum(dim=1)
    num_local_tokens = num_recv_tokens.sum().item()
    num_recv_tokens_list = num_recv_tokens.tolist()

    # prepare the hidden states that will be send-to/recv-from other experts
    hidden_states_to_send = [
        permuted_hidden_states[
            acc_batch_sizes[
                (tp_size * i + tp_rank) * local_num_experts
            ] : acc_batch_sizes[(tp_size * i + tp_rank + 1) * local_num_experts]
        ]
        for i in range(dp_size)
    ]
    recv_hidden_states = torch.empty(
        num_local_tokens,
        hidden_states.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    hidden_states_to_recv = list(
        recv_hidden_states.split(tuple(num_recv_tokens_list), dim=0)
    )

    # Multi all-to-all: dispatch
    torch.distributed.all_to_all(
        hidden_states_to_recv, hidden_states_to_send, group=get_dp_group().device_group
    )

    local_batch_sizes = recieved_batch_sizes.view(dp_size, -1).sum(dim=0)
    local_acc_batch_sizes = torch.zeros(
        len(local_batch_sizes) + 1,
        dtype=local_batch_sizes.dtype,
        device=local_batch_sizes.device,
    )
    local_acc_batch_sizes[1:] = torch.cumsum(local_batch_sizes, dim=0)

    # reorder received tokens in the expert order
    remap_mapping, remap_mapping_reversed = remap_tokens_from_ranks(
        recieved_batch_sizes, num_local_tokens, dp_size, local_num_experts
    )
    recv_hidden_states = reorder_tokens_from_ranks(recv_hidden_states, remap_mapping)

    (
        aligned_local_hidden_states,
        alignment_sizes,
        m_indices,
        masked_m,
        expected_m,
    ) = align_local_tokens(
        recv_hidden_states, local_batch_sizes, local_acc_batch_sizes, local_num_experts
    )
    aligned_M = aligned_local_hidden_states.shape[0]

    intermediate_cache1 = torch.empty(
        (aligned_M, N), device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache2 = torch.empty(
        (aligned_M, N // 2), device=hidden_states.device, dtype=hidden_states.dtype
    )
    intermediate_cache3 = torch.zeros(
        (aligned_M, hidden_states.shape[-1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    apply_grouped_gemm_moe(
        aligned_local_hidden_states,
        w1,
        m_indices=m_indices,
        masked_m=masked_m,
        expected_m=expected_m,
        block_size=quant_config.block_shape,
        weight_scale=quant_config.w1_scale,
        input_scale=None,
        output=intermediate_cache1,
    )

    if aligned_M > 0:
        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)

    apply_grouped_gemm_moe(
        intermediate_cache2,
        w2,
        m_indices=m_indices,
        masked_m=masked_m,
        expected_m=expected_m,
        block_size=quant_config.block_shape,
        weight_scale=quant_config.w2_scale,
        input_scale=None,
        output=intermediate_cache3,
    )

    result_hidden_states = unalign_local_tokens(
        intermediate_cache3,
        local_acc_batch_sizes,
        local_batch_sizes,
        alignment_sizes,
        local_num_experts,
    )
    reorder_tokens_from_ranks(result_hidden_states, remap_mapping_reversed)

    splitted_result_hidden_states = list(
        result_hidden_states.split(tuple(num_recv_tokens_list), dim=0)
    )

    # Reversed multi all-to-all: combine, the allreduce after alltoall will be
    # performed along the shared experts in model.py (e.g., qwen2_moe.py).
    torch.distributed.all_to_all(
        hidden_states_to_send,
        splitted_result_hidden_states,
        group=get_dp_group().device_group,
    )

    permuted_hidden_states.zero_()
    for i in range(dp_size):
        permuted_hidden_states[
            acc_batch_sizes[
                (tp_size * i + tp_rank) * local_num_experts
            ] : acc_batch_sizes[(tp_size * i + tp_rank + 1) * local_num_experts]
        ] = hidden_states_to_send[i]
    post_reorder_triton_kernel[(hidden_states.shape[0],)](
        permuted_hidden_states,
        hidden_states,
        row_id_map,
        topk_ids,
        topk_weights,
        0,
        global_num_experts - 1,
        topK,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )
    return hidden_states
