# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUTLASS based Fused MoE kernels."""

from collections.abc import Callable

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm._custom_ops import scaled_fp4_quant
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

try:
    import bladnn
except (ImportError, ModuleNotFoundError):
    bladnn = None
try:
    import flashinfer
except (ImportError, ModuleNotFoundError):
    flashinfer = None

logger = init_logger(__name__)


FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

@torch.compile
def get_nvfp4_global_scale(a: torch.Tensor):
    a = a.nan_to_num(nan=0.0, posinf=FLOAT4_E2M1_MAX, neginf=-FLOAT4_E2M1_MAX)
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.abs(a).max().to(torch.float32)

def bladnn_fused_moe_fp4_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    scales_row: torch.Tensor,
    scales_col: torch.Tensor,
    topk_weights: torch.Tensor,
    global_scale_A: torch.Tensor,
    global_scale_B: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad:  torch.Tensor,
    mul_topk_weights: bool,
    topk: int,
    blockm: int = 32,
    is_per_token_scale: bool = False
) -> torch.Tensor:
    # workspace is necessary for fpaintb check
    workspace = torch.empty((100000,), dtype=torch.int32, device=A.device) 
    bladnn.fused_moe_a4w4(
        A, B, C, 
        scales_row, 
        scales_col, 
        topk_weights, 
        global_scale_A, 
        global_scale_B,
        topk_ids, 
        sorted_token_ids, 
        expert_ids, 
        num_tokens_post_pad, 
        mul_topk_weights, 
        topk, blockm, 
        is_per_token_scale, 
        workspace
    )
    return C

def bladnn_fused_moe_fp4_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    scales_row: torch.Tensor,
    scales_col: torch.Tensor,
    topk_weights: torch.Tensor,
    global_scale_A: torch.Tensor,
    global_scale_B: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad:  torch.Tensor,
    mul_topk_weights: bool,
    topk: int,
    blockm: int = 32,
    is_per_token_scale: bool = False
):
    # Fake implementation for testing
    return torch.empty_like(C)

direct_register_custom_op(
        op_name="bladnn_nvfp4_fused_moe",
        op_func=bladnn_fused_moe_fp4_impl,
        fake_impl=bladnn_fused_moe_fp4_fake,
)

def run_bladnn_fused_moe_fp4(
    output: torch.Tensor,
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    m: int,
    n: int,
    k: int,
    e: int,
    device: torch.device,
    apply_router_weight_on_input: bool = False,
) -> None:
    """
    MoE implementation for FP4 Inputs

    # Gemm 1
    a: Input tensor: [m, k] (half/bfloat16)
    a1_gscale: Activation scale per expert: [e]  (float32)
    w1(gate up) (not an argument to cutlass_moe_fp4): [e, 2 * n, k]
    w1_fp4: [e, 2 * n, k // 2], dtype: torch.uint8 (stacked fp4: E2M1)
    (Note: `n` is the up projection output dim, `k` is the input dim in
     full precision)
    w1_blockscale: [e, 2 * n, k // block_size] (float8_e4m3)
                   (Block size = 16 for NVFP4)

    # Gemm 2
    a2_gscale: Activation scale per expert: [e]
    w2(down projection) (not an argument to cutlass_moe_fp4): [e, k, n]
    w2_fp4: [e, k, n // 2], dtype: torch.uint8 (stacked E2M1)
    w2_blockscale: [e, k, n // block_size], dtype: float8_e4m3

    topk_weights: [m, topk] dtype: float8
    topk_ids: [m, topk] dtype: float8

    m, n, k: Unquantized weight shapes, dtype: int
    e: number of experts, dtype: int

    assumes that topk < k < n to satisfy - up/down projection expectations.
    """
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_fp4.dtype == torch.uint8, "weight 1 must be uint8"
    assert w2_fp4.dtype == torch.uint8, "weight 2 must be uint8"
    assert (
        w1_fp4.ndim == 3
        and w2_fp4.ndim == 3
        and w1_blockscale.ndim == 3
        and w2_blockscale.ndim == 3
    ), "All Weights must be of rank 3 for cutlass_moe_fp4"
    m_a, k_a = a.shape
    e_w1, nx2_w1, half_k_w1 = w1_fp4.shape
    e_w2, k_w2, half_n_w2 = w2_fp4.shape

    assert e_w1 == e_w2 and e_w1 == e, (
        "Number of experts must match",
        f" between weights. {e_w1}, {e_w2}, {e}",
    )
    assert k_a == half_k_w1 * 2 and k == k_w2, (
        "Hidden size mismatch between a, w1 and w2"
    )
    assert nx2_w1 == n * 2 and half_n_w2 * 2 == n, "mismatch in expected `n`"
    assert m == m_a, "input shape mismatch"
    assert 2 * half_k_w1 == k_w2, "Hidden size mismatch w2 and w1"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"
    assert topk_weights.size(0) == m and topk_ids.size(0) == m, (
        "topk must be provided for each row of a"
    )
    topk = topk_ids.size(1)
    out_dtype = a.dtype
    num_topk = topk_ids.size(1)

    if apply_router_weight_on_input:
        # TODO: this only works for topK=1, will need to update for topK>1
        assert num_topk == 1, (
            "apply_router_weight_on_input is only implemented for topk=1"
        )
        a.mul_(topk_weights.to(out_dtype))

    c1 = _resize_cache(workspace13, (m*num_topk, n * 2))
    c2 = _resize_cache(workspace2, (m*num_topk, n))
    c3 = _resize_cache(workspace13, (m*num_topk, k))

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, 32, e
    )
    if envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4:
        # w1_alphas_ is weight global scale using VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4
        global_scale_A = get_nvfp4_global_scale(a)
        a_quant, a_block_scale = flashinfer.fp4_quantize(
            a,
            global_scale_A,
            16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
            is_sf_8x4_layout=False,
            enable_pdl=False
        )
        torch.ops.vllm.bladnn_nvfp4_fused_moe(
            a_quant,
            w1_fp4,
            c1,
            a_block_scale.view(torch.float8_e4m3fn),
            w1_blockscale,
            topk_weights,
            global_scale_A,
            w1_alphas,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk,
            32,
            False
        )
        del a_quant, a_block_scale, global_scale_A
    else:
        a_quant, a_block_scale = flashinfer.fp4_quantize(
            a,
            a1_gscale,
            16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
            is_sf_8x4_layout=False,
            enable_pdl=False
        )
        torch.ops.vllm.bladnn_nvfp4_fused_moe(
            a_quant,
            w1_fp4,
            c1,
            a_block_scale.view(torch.float8_e4m3fn),
            w1_blockscale,
            topk_weights,
            a1_gscale,
            w1_alphas,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk,
            32,
            False
        )
        del a_quant, a_block_scale
    if envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4:
        torch.ops._C.silu_and_mul(c2, c1)
        global_scale_A = get_nvfp4_global_scale(c2)
        a_quant, a_block_scale = flashinfer.fp4_quantize(
            c2,
            global_scale_A,
            16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
            is_sf_8x4_layout=False,
            enable_pdl=False
        )
        torch.ops.vllm.bladnn_nvfp4_fused_moe(
            a_quant,
            w2_fp4,
            c1,
            a_block_scale.view(torch.float8_e4m3fn),
            w2_blockscale,
            topk_weights,
            global_scale_A,
            w2_alphas,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            32,
            False
        )
        del a_quant, a_block_scale, global_scale_A
    else:
        torch.ops._C.silu_and_mul(c2, c1)
        a_quant, a_block_scale = flashinfer.fp4_quantize(
            c2,
            a2_gscale,
            16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
            is_sf_8x4_layout=False,
            enable_pdl=False
        )
        torch.ops.vllm.bladnn_nvfp4_fused_moe(
            a_quant,
            w2_fp4,
            c1,
            a_block_scale.view(torch.float8_e4m3fn),
            w2_blockscale,
            topk_weights,
            a2_gscale,
            w2_alphas,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            32,
            False
        )
        del a_quant, a_block_scale

    assert output.dtype == out_dtype
    output.copy_(c3.view(m, num_topk, k).sum(dim=1), non_blocking=True)
    return


# Split into batched and non-batched
class FusedMoEFp4(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        max_experts_per_worker: int,
        out_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
        use_batched_format: bool = False,
    ):
        super().__init__(quant_config)
        self.max_experts_per_worker = max_experts_per_worker
        self.out_dtype = out_dtype
        self.use_batched_format = use_batched_format

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        if self.use_batched_format:
            return (
                mk.FusedMoEActivationFormat.BatchedExperts,
                mk.FusedMoEActivationFormat.BatchedExperts,
            )
        else:
            return (
                mk.FusedMoEActivationFormat.Standard,
                mk.FusedMoEActivationFormat.Standard,
            )

    def supports_expert_map(self) -> bool:
        return False

    def supports_chunking(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.out_dtype if self.out_dtype is not None else act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1: tuple[int, ...] = ()
        workspace2: tuple[int, ...] = ()
        output: tuple[int, ...] = ()
        if self.use_batched_format:
            workspace1 = (self.max_experts_per_worker, M, max(N, K))
            workspace2 = (self.max_experts_per_worker, M, (N // 2))
            output = (self.max_experts_per_worker, M, K)
        else:
            workspace1 = (M*topk, max(2 * N, K))
            workspace2 = (M*topk, N)
            output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,  # unused
        a2_scale: torch.Tensor | None,  # unused
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        e, m, n, k, _ = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        n = w2.shape[2] * 2

        run_bladnn_fused_moe_fp4(
            output=output,
            a=hidden_states,
            a1_gscale=self.a1_gscale,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_alphas=self.g1_alphas,
            a2_gscale=self.a2_gscale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_alphas=self.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace13=workspace13,
            workspace2=workspace2,
            m=m,
            n=n,
            k=k,
            e=e,
            device=hidden_states.device,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


def bladnn_fused_moe_fp4(
    a: torch.Tensor,
    w1_fp4: torch.Tensor,
    w2_fp4: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    m: int,
    n: int,
    k: int,
    e: int,
    expert_map: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    assert expert_map is None, (
        "Expert Parallelism / expert_map "
        "is currently not supported for "
        "ModelOptNvFp4FusedMoE's cutlass_moe_fp4."
    )

    # TODO(bnell): this feels a bit hacky
    # NVFP4 requires two levels of quantization, which involves
    # computing some scaling factors dynamically. This makes it
    # incompatible with the typical prepare -> MoE -> finalize
    # pipeline. Move the quantization logic into the MoE body.
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=None,  # skip quantization in prepare/finalize
        per_act_token_quant=quant_config.per_act_token_quant,
        per_out_ch_quant=quant_config.per_out_ch_quant,
        block_shape=quant_config.block_shape,
        g1_alphas=quant_config.g1_alphas,
        g2_alphas=quant_config.g2_alphas,
        a1_gscale=quant_config.a1_gscale,
        a2_gscale=quant_config.a2_gscale,
        w1_scale=quant_config.w1_scale,
        w2_scale=quant_config.w2_scale,
    )

    fn = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        FusedMoEFp4(
            max_experts_per_worker=e,
            out_dtype=a.dtype,
            quant_config=quant_config,
            use_batched_format=False,
        ),
    )

    return fn(
        hidden_states=a,
        w1=w1_fp4,
        w2=w2_fp4,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        activation="silu",
        global_num_experts=e,
        expert_map=None,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
