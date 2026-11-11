# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from vllm.model_executor.models.qwen3_next import Indexer

import numpy as np
import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.attention.layer import Attention
from vllm.attention.ops.common import cp_lse_ag_out_rs, cp_sequence_all_to_all
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.attention.utils.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        is_vllm_fa,
        reshape_and_cache_flash,
        scheduler_meta_supported,
    )


from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.platform_utils import is_ppu
from vllm.v1.attention.backends.flash_attn_qsautils import (
    can_use_tl,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
    sparse_gqa_pytorch,
)
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if can_use_tl:
    from vllm.v1.attention.backends.flash_attn_qsautils import (
        sparse_gqa_tl_nopaged_interface,
        sparse_kv_extraction_interface,
    )

import os

from vllm.distributed import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.models.qwen3_next_mtp import with_draft_model_config
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)


if is_flash_attn_varlen_func_available():
    has_seqused_k = "seqused_k" in inspect.signature(flash_attn_varlen_func).parameters
    has_out = "out" in inspect.signature(flash_attn_varlen_func).parameters
    has_num_splits = (
        "num_splits" in inspect.signature(flash_attn_varlen_func).parameters
    )

    if is_ppu() and get_flash_attn_version() == 2:
        has_num_prefill = (
            "num_prefill" in inspect.signature(flash_attn_varlen_func).parameters
        )
        if not has_num_prefill and not envs.VLLM_ATTENTION_DISABLE_PD_SEPARATION:
            logger.info_once(
                "Disabling PD separation in FlashAttention for PPU as not supported, "
                "upgrading ppu's vllm-flash-attn is needed."
            )
            envs.VLLM_ATTENTION_DISABLE_PD_SEPARATION = True
    else:
        has_num_prefill = False
        if not envs.VLLM_ATTENTION_DISABLE_PD_SEPARATION:
            logger.info_once(
                "Disabling PD separation in FlashAttention for PPU as upstream FA is "
                "in use and this feature is not needed."
            )
            envs.VLLM_ATTENTION_DISABLE_PD_SEPARATION = True
else:
    has_seqused_k = False
    has_num_prefill = False
    has_num_splits = False


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return (
                [256]
                if (is_ppu() and not is_vllm_fa and get_flash_attn_version() == 2)
                else [16, 32, 64, 128, 256]
            )
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype.startswith("fp8"):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_reqs: int  # Number of requests in the batch.
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    # For flash-attention backwards compatibility
    cu_seq_lens: torch.Tensor | None
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    req_id_per_token: torch.Tensor
    max_orig_seq_len: int

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    cu_prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None
    cu_suffix_kv_lens: torch.Tensor | None

    # For fp8 attention to store rearrange block table
    rearrange_prefill_block_table: torch.Tensor | None = None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # GPU/PPU FA metadata for setting PD-Seperation mode of varlen_func
    num_prefills: int = -1
    max_seq_len_decode: int = 0
    max_seq_len_prefill: int = 0
    num_decodes: int = -1
    num_decode_tokens: int = -1
    num_prefill_tokens: int = -1
    max_seq_len_q_decode: int = 0

    # Causal attention.
    causal: bool = True
    is_first_chunk: bool = False
    fa_version: int = get_flash_attn_version()

    # Supported for prefill and decode.
    # Backend (FA2 vs FA3 vs Triton) support checked separately.
    cuda_graph_supported: bool = True

    # Whether should reshape-and-write K/V into kv-cache. This is controlled by
    # the model-side logic. If use a fused QK-Norm+RoPE+KVCache-Write kernel,
    # set this to True to avoid duplicate writes.
    skip_write_kvcache: bool = False


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, FlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support = (
        (
            AttentionCGSupport.ALWAYS
            if get_flash_attn_version() == 3
            else AttentionCGSupport.UNIFORM_BATCH
        )
        if is_vllm_fa
        else AttentionCGSupport.NEVER
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        if self.speculative_config:
            self.num_spec = self.speculative_config.num_speculative_tokens  # noqa: E501
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0

        self.enable_chunked_prefill = (
            vllm_config.scheduler_config.enable_chunked_prefill
        )
        self.ck_chunksize = vllm_config.scheduler_config.max_num_batched_tokens

        self.use_qsa = (
            hasattr(vllm_config.model_config.hf_config, "index_topk")
            and vllm_config.model_config.hf_config.index_topk > 0
        )
        if self.use_qsa:
            logger.info_once("Using QSA attention for flash attention backend.")

        # TODO: normal layer and mtp layer share same metadata. only support mtp-qsa now

        if (
            self.use_qsa
            or envs.VLLM_FLASH_ATTN_FP8_ATTENTION
            or (is_ppu() and not envs.VLLM_ATTENTION_DISABLE_PD_SEPARATION)
        ):
            self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = scheduler_meta_supported and get_flash_attn_version() == 3

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size
        self.fa_version: int = get_flash_attn_version()

        if self.use_full_cuda_graph and self.aot_schedule:
            self.scheduler_metadata = torch.zeros(
                round_up(vllm_config.scheduler_config.max_num_seqs, 4) * 4 + 1,
                dtype=torch.int32,
                device=self.device,
            )

            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> FlashAttentionMetadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len

        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens

        cm = common_attn_metadata
        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )
        req_id_per_token = self.req_id_per_token_buffer[:num_actual_tokens]

        if not has_seqused_k:
            cu_seq_lens = torch.zeros(
                seq_lens.shape[0] + 1, dtype=seq_lens.dtype, device=seq_lens.device
            )
            torch.cumsum(seq_lens, dim=0, out=cu_seq_lens[1:])
        else:
            cu_seq_lens = None
        block_table_tensor = common_attn_metadata.block_table_tensor[: num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[: num_actual_tokens]
        causal = common_attn_metadata.causal

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.use_qsa or envs.VLLM_FLASH_ATTN_FP8_ATTENTION:
            query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
            num_decodes = (
                query_lens_cpu[query_lens_cpu <= self.reorder_batch_threshold]
            ).size(0)
            num_decode_tokens = common_attn_metadata.query_start_loc_cpu[
                num_decodes
            ].item()
            num_prefills = query_lens_cpu.size(0) - num_decodes
            num_prefill_tokens = (
                common_attn_metadata.query_start_loc_cpu[-1] - num_decode_tokens
            )
            rearrange_prefill_block_table = None
            if num_prefills > 0:
                prefill_block_table = block_table_tensor[num_decodes:]
                seqs = prefill_block_table.shape[0]
                blocks = prefill_block_table.shape[1]
                rearrange_prefill_block_table = torch.arange(
                    seqs * blocks, device=prefill_block_table.device, dtype=torch.int32
                ).reshape(seqs, blocks)
        else:
            num_prefills = -1
            num_prefill_tokens = -1
            num_decodes = -1
            num_decode_tokens = -1
            prefill_block_table = None
            rearrange_prefill_block_table = None

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if self.use_full_cuda_graph and num_actual_tokens <= self.max_cudagraph_size:
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        if vllm_is_batch_invariant():
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_kv_lens = query_start_loc[1:] - query_start_loc[:-1]
            dcp_context_kv_lens = seq_lens - query_kv_lens

            dcp_context_kv_lens = get_dcp_local_seq_lens(
                dcp_context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            cu_prefix_kv_lens = torch.tensor(
                [0, common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            cu_suffix_kv_lens = torch.zeros(
                suffix_kv_lens.shape[0] + 1,
                dtype=suffix_kv_lens.dtype,
                device=suffix_kv_lens.device,
            )
            torch.cumsum(suffix_kv_lens, dim=0, out=cu_suffix_kv_lens[1:])
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            cu_prefix_kv_lens = None
            suffix_kv_lens = None
            cu_suffix_kv_lens = None
            prefix_scheduler_metadata = None
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )

        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            # Ensure the persistent buffer is large enough
            assert n <= self.scheduler_metadata.shape[0], (
                f"Scheduler metadata size {n} exceeds buffer size "
                + f"{self.scheduler_metadata.shape[0]}"
            )
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        max_seq_len_q_decode = 0
        if num_decodes > 0:
            cu_seqlens_q = query_start_loc
            decode_cu_seqlens_q = cu_seqlens_q[: num_decodes + 1]
            max_seq_len_q_decode = torch.diff(decode_cu_seqlens_q).max().item()

        prefill_seqused_k = seq_lens[num_decodes:]
        query_kv_lens = query_start_loc[1:] - query_start_loc[:-1]
        prefill_query_kv_lens = query_kv_lens[num_decodes:]
        is_first_chunk = torch.all(prefill_seqused_k == prefill_query_kv_lens).item()

        attn_metadata = FlashAttentionMetadata(
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            cu_seq_lens=cu_seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            cu_prefix_kv_lens=cu_prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            cu_suffix_kv_lens=cu_suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            req_id_per_token=req_id_per_token,
            max_seq_len_q_decode=max_seq_len_q_decode,
            is_first_chunk=is_first_chunk,
            max_orig_seq_len=common_attn_metadata.max_orig_seq_len,
            **{
                "num_prefills": num_prefills,
                "num_prefill_tokens": num_prefill_tokens,
                "num_decodes": num_decodes,
                "num_decode_tokens": num_decode_tokens,
                "rearrange_prefill_block_table": rearrange_prefill_block_table,
            }
            if envs.VLLM_FLASH_ATTN_FP8_ATTENTION or self.use_qsa
            else {},  # type: ignore
            causal=causal,
            fa_version=(
                2
                if (
                    envs.VLLM_FLASH_ATTN_PREFILL_VERSION == 2
                    and max_query_len > (self.num_spec + 1)
                )
                else self.fa_version
            ),
        )

        if not envs.VLLM_ATTENTION_DISABLE_PD_SEPARATION:
            # metadata for PD-Separation mode of varlen_func
            # Don't support interleaved DP pattern: DPD, PDP
            num_prefills = 0
            num_decodes = 0
            dp_switch_cnt = 0
            for i in range(1, num_reqs + 1):
                is_p = (
                    common_attn_metadata.query_start_loc_cpu[i]
                    - common_attn_metadata.query_start_loc_cpu[i - 1]
                ).item() > 1
                if i == 1:
                    if is_p:
                        num_prefills += 1
                        prev_pattern = "p"
                    else:
                        num_decodes += 1
                        prev_pattern = "d"
                    continue
                if is_p:
                    num_prefills += 1
                    if prev_pattern == "d":
                        dp_switch_cnt += 1
                        prev_pattern = "p"
                else:
                    num_decodes += 1
                    if prev_pattern == "p":
                        dp_switch_cnt += 1
                        prev_pattern = "d"
            assert dp_switch_cnt < 2, (
                "dp_switch_cnt should < 2, since we already reorder the batch"
            )
            if dp_switch_cnt == 1:
                attn_metadata.num_prefills = num_prefills
                attn_metadata.max_seq_len_decode = (
                    common_attn_metadata.seq_lens_cpu[:num_decodes].max().item()
                )
                attn_metadata.max_seq_len_prefill = (
                    common_attn_metadata.seq_lens_cpu[num_decodes:num_reqs].max().item()
                )

        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        prefix: str = "",
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        target_head_size: int | None = None,
        indexer: Optional["Indexer"] = None,
        total_num_kv_heads: int = -1,
        **extra_impl_args,
    ) -> None:
        if indexer:
            self.topk_indices_buffer = indexer.topk_indices_buffer
        else:
            self.topk_indices_buffer = None
        self.num_heads = num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = vllm_is_batch_invariant()

        if is_quantized_kv_cache(self.kv_cache_dtype) and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )
        self.target_head_size = target_head_size or head_size
        # support fused QK-Norm+RoPE+KVCache-Write kernel in model-side.
        self.support_fuse_kvcache_write = True

        vllm_config = get_current_vllm_config()
        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        prefixes = prefix.split(".")
        self.layer_idx = int(prefixes[prefixes.index("layers") + 1])

        # Check if current layer is eagle layer, note that the layer_idx of draft
        # models are offset by num_hidden_layers of the target model.
        self.is_target_model = self.layer_idx < self.num_layers

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )
        self.prefix = prefix

        self.supports_quant_query_input = True

        self.tp_group = get_tensor_model_parallel_group()
        self.tp_rank: int = get_tensor_model_parallel_rank()
        self.tp_size: int = get_tensor_model_parallel_world_size()

    def prefill_paged_sparse_att_torch_ref(
        self,
        query,
        key_cache,
        value_cache,
        topk_indices,
        block_table,
        prefill_cu_seqlens_q,
        prefill_seqused_k,
        num_decodes,
    ):
        query_lens = (
            (prefill_cu_seqlens_q[1:] - prefill_cu_seqlens_q[:-1]).cpu().tolist()
        )
        kv_lens = prefill_seqused_k.cpu().tolist()

        outputs: list[torch.Tensor] = []
        num_seqs = len(query_lens)
        _, block_size, num_kv_heads, head_size = key_cache.shape
        start_idx = 0
        for i in range(num_seqs):
            query_len = query_lens[i]
            kv_len = kv_lens[i]
            # [num_tokens, num_heads, head_size]
            q = query[start_idx : start_idx + query_len]

            num_kv_blocks = (kv_len + block_size - 1) // block_size
            block_indices = block_table[num_decodes + i, :num_kv_blocks]

            # step2: restore key/value len
            #   k: [num_tokens, num_kv_heads, head_size]
            k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
            k = k[:kv_len]
            v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
            v = v[:kv_len]

            indices = topk_indices[
                start_idx : start_idx + query_len
            ]  # self.topk_indices_buffer[start_idx:start_idx + query_len]

            outputs.append(
                sparse_gqa_pytorch(
                    q, k, v, query_len, kv_len, indices, self.scale, is_causal=True
                )
            )
            start_idx += query_len

        output = torch.cat(outputs, dim=0)
        return output

    def collect_topkcache_torch_ref(
        self, max_seqlen_k, num_decode_tokens, block_table, key_cache, value_cache, topk
    ):
        key_head, val_head, block_size, head_dim = (
            key_cache.shape[-2],
            value_cache.shape[-2],
            key_cache.shape[1],
            key_cache.shape[-1],
        )
        max_blocks = (max_seqlen_k + block_size - 1) // block_size
        raw_topk_indices = self.topk_indices_buffer[:num_decode_tokens]
        valid_mask = (
            (raw_topk_indices >= 0).unsqueeze(-1).unsqueeze(-1)
        )  # [N, topk, 1, 1]. invalid be 0
        decode_topk_ids = raw_topk_indices.clamp(min=0)
        decode_topk_ids = decode_topk_ids + (
            torch.arange(num_decode_tokens, device=key_cache.device, dtype=torch.int32)
            * (max_blocks * block_size)
        ).unsqueeze(-1)
        decode_block_ids = block_table[:num_decode_tokens, :max_blocks].reshape(-1)
        decode_key_cache = (
            key_cache[decode_block_ids]
            .reshape([-1, key_head, head_dim])[decode_topk_ids]
            .reshape([-1, topk, key_head, head_dim])
        )
        decode_val_cache = (
            value_cache[decode_block_ids]
            .reshape([-1, val_head, head_dim])[decode_topk_ids]
            .reshape([-1, topk, val_head, head_dim])
        )
        decode_key_cache = decode_key_cache * valid_mask.to(decode_key_cache.dtype)
        decode_val_cache = decode_val_cache * valid_mask.to(decode_val_cache.dtype)
        return decode_key_cache, decode_val_cache

    def flash_attn_sparse_prefill(
        self,
        attn_metadata,
        query,
        indices,
        key,
        value,
        kv_cache,
        key_cache,
        value_cache,
        block_table,
        num_decodes,
        num_decode_tokens,
        num_actual_tokens,
        prefill_cu_seqlens_q,
        prefill_seqused_k,
        is_first_chunk,
        max_seqlen_k,
        max_seqlen_q,
        use_cp=False,
        query_end=-1,
    ):
        block_size = key_cache.shape[1]
        num_seqs = prefill_seqused_k.shape[0]
        backend = envs.VLLM_QSA_PREFILL_ATTN_BACKEND
        if backend == "pai-fa3":
            assert not use_cp
        if backend == "triton":
            # first chunk goes no_kvcache version for effiency;
            # cp only support kvcache version
            if not is_first_chunk or query.shape[0] != key.shape[0] or use_cp:
                total_kv_tokens = prefill_seqused_k.max().item()
                num_needed_blocks = cdiv(total_kv_tokens, block_size)
                extract_block_table = block_table[
                    num_decodes : num_decodes + num_seqs, :num_needed_blocks
                ]
                # (numblocks,blocksize,heads,h)-> (numneedblocks,blocksize,heads,h)
                #  -> (allneedtoken,kvheads,h)
                k_flat = key_cache[extract_block_table].view(
                    -1, key_cache.shape[2], key_cache.shape[3]
                )
                v_flat = value_cache[extract_block_table].view(
                    -1, key_cache.shape[2], key_cache.shape[3]
                )
                if use_cp:
                    # 1: gather kvheads
                    _, local_kv_heads, _ = k_flat.shape
                    if self.tp_size > self.total_num_kv_heads:
                        #  tp itself may have redudant kvheads in all ranks: 
                        #   total_heads = localheads * num_rank / repeat
                        # each rank only send 1/repeat's this head's data
                        #  after gather,all rank get  no reudant total kvheads:  
                        #   (m * localheads / repeat) * num_rank = m * total_heads
                        repeat = self.tp_size // self.total_num_kv_heads
                        assert k_flat.shape[0] % repeat == 0
                        assert local_kv_heads == 1
                        assert self.total_num_kv_heads != -1
                        chunk_size_k = k_flat.shape[0] // repeat
                        start = (self.tp_rank % repeat) * chunk_size_k
                        end = start + chunk_size_k
                        k_local = k_flat[start:end].view(
                            -1, key_cache.shape[3]
                        )  # （m/repeat * localheads,h）
                        v_local = v_flat[start:end].view(-1, key_cache.shape[3])
                        # gather all kvheads (m * total_heads,d)
                        k_gathered = tensor_model_parallel_all_gather(k_local, dim=0)
                        v_gathered = tensor_model_parallel_all_gather(v_local, dim=0)
                        #  (m * total_heads,d) -> (total_heads,m,d) -> (m,total_heads,d)
                        k_flat = (
                            k_gathered.view(
                                self.total_num_kv_heads, -1, key_cache.shape[3]
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                        v_flat = (
                            v_gathered.view(
                                self.total_num_kv_heads, -1, key_cache.shape[3]
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                    elif self.tp_size > 1:
                        # (M, local_kv_heads, d) -> (M, total_kv_heads, d)
                        k_flat = tensor_model_parallel_all_gather(
                            k_flat, dim=1
                        ).contiguous()
                        v_flat = tensor_model_parallel_all_gather(
                            v_flat, dim=1
                        ).contiguous()

                    #  2: rebuild this chunk's metadata
                    chunk_size = query.shape[0]
                    # qlen:[0-chunk_size]
                    prefill_cu_seqlens_q = torch.tensor(
                        [0, chunk_size], device=query.device, dtype=torch.int32
                    )
                    # consider history ck_chunk, and clamp to current cp_chunk
                    ck_chunk_qlen = attn_metadata.num_prefill_tokens
                    previous_ck_total_klen = prefill_seqused_k - ck_chunk_qlen
                    current_cp_rank_limit = previous_ck_total_klen + query_end
                    prefill_seqused_k = torch.clamp(prefill_seqused_k, 
                                                    max=current_cp_rank_limit)
                    
                if query.shape[0] > 0:
                    # seq token's offset in k_flat
                    stride_k_aligned = num_needed_blocks * block_size
                    cu_seqlens_k_aligned = (
                        torch.arange(
                            num_seqs + 1, device=query.device, dtype=torch.int32
                        )
                        * stride_k_aligned
                    )
                    output = sparse_gqa_fwd_interface_triton_ck(
                        query,
                        k_flat,
                        v_flat,
                        indices,
                        prefill_cu_seqlens_q,
                        cu_seqlens_k_aligned,
                        prefill_seqused_k,
                        self.scale,
                    )
                else:
                    output = query.new_zeros((0, query.shape[1], query.shape[2]))
            else:
                # TODO: when cp, query_end == q.shape[0], can use this。
                assert not use_cp
                assert query.shape[0] == key.shape[0]
                offsets = prefill_cu_seqlens_q
                output = sparse_gqa_fwd_interface_triton(
                    query, key, value, max_seqlen_k, indices, offsets, self.scale
                )
        elif can_use_tl and backend == "tilelang":
            # tl-version
            assert not use_cp
            assert key is not None
            q = query.contiguous()
            k = key.contiguous()
            v = value.contiguous()
            prefill_cu_seqlens_k = prefill_cu_seqlens_q
            output, _ = sparse_gqa_tl_nopaged_interface(
                q,
                k,
                v,
                indices,
                prefill_cu_seqlens_q,
                prefill_cu_seqlens_k,
                self.scale,
                is_causal=True,
            )
        elif backend == "pai-fa3":
            assert not use_cp
            topk_indices = (
                indices  # self.topk_indices_buffer[num_decode_tokens:num_actual_tokens]
            )
            block_size = key_cache.shape[1]
            global_topk_indices = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[num_decode_tokens:],
                block_table, # req_id is global_index in block_table
                topk_indices,
                BLOCK_SIZE=block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            )
            # NOTE: pai-qsa customizid version's fa3, not community version
            from pai_flash_attn_interface import (
                flash_attn_with_kvcache as flash_attn_with_kvcache_qsa,
            )

            # output(m,nqheads,h)
            kv_cache_physical = kv_cache.permute(1, 0, 2, 3, 4)
            assert kv_cache_physical.is_contiguous()
            output = flash_attn_with_kvcache_qsa(
                q=query,  # [num_decode_tokens:num_actual_tokens],# (m,nqheads,h)
                k_cache=kv_cache_physical[
                    :, 0, :, :, :
                ],  # (numblocks,blocksize=544,nkvheads=2,h=256)
                v_cache=kv_cache_physical[
                    :, 1, :, :, :
                ],  # (numblocks,blocksize=544,nkvheads=2,h=256)
                # (bs,) each seq's effetive k
                cache_seqlens=prefill_seqused_k,
                # (m_qtokens,topk)
                page_table=global_topk_indices,
                # (bs+1,)
                cu_seqlens_q=prefill_cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=self.scale,
                causal=True,
                num_splits=1,
                pack_gqa=True,
            )
        else:  # torch ref
            assert not use_cp
            output = self.prefill_paged_sparse_att_torch_ref(
                query,
                key_cache,
                value_cache,
                indices,
                block_table,
                prefill_cu_seqlens_q,
                prefill_seqused_k,
                num_decodes,
            )
        return output

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        if self.target_head_size != self.head_size:
            query = torch.nn.functional.pad(
                query, [0, self.target_head_size - self.head_size]
            )
            key = torch.nn.functional.pad(
                key, [0, self.target_head_size - self.head_size]
            )
            value = torch.nn.functional.pad(
                value, [0, self.target_head_size - self.head_size]
            )
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
            and not attn_metadata.skip_write_kvcache
        ):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            # queries are quantized in the attention layer
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            cu_seq_lens = attn_metadata.cu_seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            num_prefills = attn_metadata.num_prefills
            max_seqlen_k_decode = attn_metadata.max_seq_len_decode
            max_seqlen_k_prefill = attn_metadata.max_seq_len_prefill

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            num_prefills = attn_metadata.num_prefills
            num_decodes = attn_metadata.num_decodes
            num_decode_tokens = attn_metadata.num_decode_tokens
            num_prefill_tokens = attn_metadata.num_prefill_tokens

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                return output

            if self.topk_indices_buffer is not None:  # [token_num, topk] 512
                topk = self.topk_indices_buffer.shape[-1]
                assert num_actual_tokens == num_decode_tokens + num_prefill_tokens
                if num_prefills > 0:
                    prefill_query = query[num_decode_tokens:]
                    prefill_cu_seqlens_q = torch.cat(
                        [
                            torch.tensor(
                                [0], dtype=torch.int32, device=cu_seqlens_q.device
                            ),
                            cu_seqlens_q[num_decodes + 1 :] - cu_seqlens_q[num_decodes],
                        ]
                    )
                    prefill_seqused_k = seqused_k[num_decodes:]

                    qsa_use_cp = envs.VLLM_QSA_PREFILL_USE_CP
                    is_first_chunk = attn_metadata.is_first_chunk
                    use_context_parallel = (
                        num_decode_tokens == 0
                        and (
                            envs.VLLM_DSA_USE_DENSE_PREFILL_THRESHOLD < 0
                            or (
                                attn_metadata.max_orig_seq_len
                                > envs.VLLM_DSA_USE_DENSE_PREFILL_THRESHOLD
                            )
                        )
                        and envs.VLLM_DSA_USE_CONTEXT_PARALLEL_THRESHOLD >= 0
                        and (
                            prefill_query.shape[0]
                            >= envs.VLLM_DSA_USE_CONTEXT_PARALLEL_THRESHOLD
                        )
                        and qsa_use_cp
                    )
                    if (
                        use_context_parallel and num_prefills == 1
                    ):  # only suppoort 1 prefill seq
                        # q use cp, not use tp
                        q = query[num_decode_tokens:num_actual_tokens]
                        # buffer not gather, only contain this's chunk's result
                        indices = self.topk_indices_buffer[
                            num_decode_tokens:num_actual_tokens
                        ]
                        key = key[num_decode_tokens:num_actual_tokens]
                        value = value[num_decode_tokens:num_actual_tokens].contiguous()
                        # 1.cp
                        num_tokens = q.size(0)
                        chunk_size = cdiv(num_tokens, self.tp_size)
                        query_start = self.tp_rank * chunk_size
                        query_end = min(query_start + chunk_size, num_tokens)
                        local_topk_indices = indices[query_start:query_end]

                        # allgather q heads, then slice q sequence,
                        # combine into one all_to_all
                        padded_q = q
                        if num_tokens < self.tp_size * chunk_size:
                            pad = q.new_zeros(
                                self.tp_size * chunk_size - num_tokens, *q.shape[1:]
                            )
                            padded_q = torch.cat([q, pad], dim=0)

                        local_q = cp_sequence_all_to_all(
                            padded_q, 
                            get_tp_group().device_group, 
                            scatter_dim=0, 
                            gather_dim=1,
                        )[:query_end - query_start]

                        # 2: attn
                        # NOTE: current only support 1 prefll seq
                        attn_output = self.flash_attn_sparse_prefill(
                            attn_metadata,
                            local_q,
                            local_topk_indices,
                            key,
                            value,
                            kv_cache,
                            key_cache,
                            value_cache,
                            block_table,
                            num_decodes,
                            num_decode_tokens,
                            num_actual_tokens,
                            prefill_cu_seqlens_q,
                            prefill_seqused_k,
                            is_first_chunk,
                            max_seqlen_k,
                            max_seqlen_q,
                            use_cp=True,
                            query_end=query_end,
                        )
                        # 3: pad output
                        if attn_output.size(0) < chunk_size:
                            # pad to chunk size for all gather
                            padded_output = attn_output.new_zeros(
                                (chunk_size, *attn_output.shape[1:]),
                            )
                            if attn_output.size(0) > 0:
                                padded_output[: attn_output.size(0)].copy_(attn_output)
                            attn_output = padded_output

                        # allgather output sequence, then slice output heads.
                        # combine into one all_to_all
                        local_output = cp_sequence_all_to_all(
                            attn_output, 
                            get_tp_group().device_group, 
                            scatter_dim=1, 
                            gather_dim=0,
                        )
                        output[num_decode_tokens:num_actual_tokens] = local_output[
                            num_decode_tokens:num_actual_tokens
                        ]

                    else:
                        q = query[num_decode_tokens:num_actual_tokens]
                        indices = self.topk_indices_buffer[
                            num_decode_tokens:num_actual_tokens
                        ]
                        key = key[num_decode_tokens:num_actual_tokens]
                        value = value[num_decode_tokens:num_actual_tokens].contiguous()
                        attn_output = self.flash_attn_sparse_prefill(
                            attn_metadata,
                            q,
                            indices,
                            key,
                            value,
                            kv_cache,
                            key_cache,
                            value_cache,
                            block_table,
                            num_decodes,
                            num_decode_tokens,
                            num_actual_tokens,
                            prefill_cu_seqlens_q,
                            prefill_seqused_k,
                            is_first_chunk,
                            max_seqlen_k,
                            max_seqlen_q,
                        )
                        output[num_decode_tokens:num_actual_tokens] = attn_output

                if num_decodes > 0:
                    # assert num_decode_tokens == num_decodes  # num_decodes: seq粒度
                    decode_query = query[:num_decode_tokens]
                    decode_backend = envs.VLLM_QSA_DECODE_ATTN_BACKEND
                    if decode_backend == "pai-fa3":
                        # NOTE: pai-qsa customizid version's fa3, not community version
                        from pai_flash_attn_interface import (
                            flash_attn_with_kvcache as flash_attn_with_kvcache_qsa,
                        )

                        kv_cache_physical = kv_cache.permute(1, 0, 2, 3, 4)
                        assert kv_cache_physical.is_contiguous()
                        block_size = key_cache.shape[1]
                        # decode_seqused_k.fill_(topk)
                        decode_global_topk_indices = (
                            triton_convert_req_index_to_global_index(
                                attn_metadata.req_id_per_token[:num_decode_tokens],
                                block_table, # req_id is global_index in block_table
                                self.topk_indices_buffer[:num_decode_tokens],
                                BLOCK_SIZE=block_size,
                                NUM_TOPK_TOKENS=topk,
                            )
                        )
                        decode_cu_seqlens_q = cu_seqlens_q[: num_decodes + 1]
                        # decode_seqused_k = seqused_k[:num_decodes].clamp(max=topk)
                        decode_seqused_k = seqused_k[:num_decodes]
                        assert attn_metadata.max_seq_len_q_decode > 0
                        output[:num_decode_tokens] = flash_attn_with_kvcache_qsa(
                            q=decode_query,  # (m,nqheads,h)
                            k_cache=kv_cache_physical[
                                :, 0, :, :, :
                            ],  # (numblocks,blocksize=544,nkvheads=2,h=256)
                            v_cache=kv_cache_physical[
                                :, 1, :, :, :
                            ],  # (numblocks,blocksize=544,nkvheads=2,h=256)
                            # (bs,) each seq's effetive k
                            cache_seqlens=decode_seqused_k,
                            # (m_qtokens,topk)
                            page_table=decode_global_topk_indices,
                            # (bs+1,)
                            cu_seqlens_q=decode_cu_seqlens_q,
                            max_seqlen_q=attn_metadata.max_seq_len_q_decode,
                            softmax_scale=self.scale,
                            causal=True,
                            num_splits=1,
                            pack_gqa=True,
                        )
                        return

                    # collect kv
                    if can_use_tl:
                        decode_key_cache, decode_val_cache = (
                            sparse_kv_extraction_interface(
                                kv_cache,
                                block_table[:num_decode_tokens],
                                self.topk_indices_buffer[:num_decode_tokens],
                                seqused_k[:num_decode_tokens],
                            )
                        )
                    else:  # fall back to torch
                        decode_key_cache, decode_val_cache = (
                            self.collect_topkcache_torch_ref(
                                max_seqlen_k,
                                num_decode_tokens,
                                block_table,
                                key_cache,
                                value_cache,
                                topk,
                            )
                        )
                    decode_seqused_k = seqused_k[:num_decode_tokens].clamp(max=topk)
                    descale_shape = (num_decode_tokens, key.shape[1])
                    decode_cu_seqlens_q = torch.arange(
                        num_decode_tokens + 1, device=query.device, dtype=torch.int32
                    )
                    decode_block_table = torch.arange(
                        num_decode_tokens, device=query.device, dtype=torch.int32
                    ).unsqueeze(-1)
                    if decode_backend == "fa2":
                        from flash_attn import (
                            flash_attn_varlen_func as flash_attn_varlen_func_fa2,
                        )

                        decode_cu_seqlens_k = torch.cat(
                            [
                                torch.zeros(1, dtype=torch.int32, device=query.device),
                                decode_seqused_k.cumsum(0).to(torch.int32),
                            ]
                        )
                        output[:num_decode_tokens] = flash_attn_varlen_func_fa2(
                            q=decode_query,
                            k=decode_key_cache,
                            v=decode_val_cache,
                            cu_seqlens_q=decode_cu_seqlens_q,
                            cu_seqlens_k=decode_cu_seqlens_k,
                            max_seqlen_q=1,
                            max_seqlen_k=topk,
                            softmax_scale=self.scale,
                            causal=True,
                            block_table=decode_block_table,
                            window_size=self.sliding_window,
                            **{
                                "alibi_slopes": self.alibi_slopes,
                            }
                            if self.alibi_slopes is not None
                            else {},
                        )
                        return
                    elif decode_backend == "fa3":
                        flash_attn_varlen_func(
                            q=decode_query,
                            k=decode_key_cache,
                            v=decode_val_cache,
                            out=output[:num_decode_tokens],
                            cu_seqlens_q=decode_cu_seqlens_q,
                            max_seqlen_q=1,
                            seqused_k=decode_seqused_k,
                            max_seqlen_k=topk,
                            softmax_scale=self.scale,
                            causal=True,
                            **{
                                "alibi_slopes": self.alibi_slopes,
                            }
                            if self.alibi_slopes is not None
                            else {},
                            window_size=self.sliding_window,
                            block_table=decode_block_table,
                            # softcap=self.logits_soft_cap,
                            **{
                                "scheduler_metadata": None,
                                "num_splits": attn_metadata.max_num_splits,
                            }
                            if scheduler_meta_supported
                            else {},
                            **{
                                "fa_version": attn_metadata.fa_version,
                                "q_descale": layer._q_scale.expand(descale_shape),
                                "k_descale": layer._k_scale.expand(descale_shape),
                                "v_descale": layer._v_scale.expand(descale_shape),
                            }
                            if attn_metadata.fa_version == 3
                            else {},
                        )
                    else:
                        raise NotImplementedError
                return

            if (
                envs.VLLM_FLASH_ATTN_FP8_ATTENTION
                and self.is_target_model
                and num_prefills > 0
            ):
                assert num_actual_tokens == num_decode_tokens + num_prefill_tokens

                prefill_query = query[num_decode_tokens:]
                prefill_cu_seqlens_q = torch.cat(
                    [
                        torch.tensor(
                            [0], dtype=torch.int32, device=cu_seqlens_q.device
                        ),
                        cu_seqlens_q[num_decodes + 1 :] - cu_seqlens_q[num_decodes],
                    ]
                )
                prefill_seqused_k = seqused_k[num_decodes:]

                _sm_major = torch.cuda.get_device_capability(
                    key_cache.device
                )[0]

                if _sm_major >= 10:
                    # Blackwell (SM10.x): use FlashInfer trtllm-gen backend
                    import flashinfer

                    # Clamp and convert query to FP8
                    prefill_query = torch.clamp(
                        prefill_query, max=448.0, min=-448.0
                    ).to(torch.float8_e4m3fn)

                    # Detect KV cache layout and prepare for FlashInfer
                    cache_layout = get_kv_cache_layout()
                    block_size = key_cache.shape[1]
                    head_dim = key_cache.shape[-1]
                    prefill_block_table_2d = block_table[num_decodes:]

                    # Compute paged KV metadata from block table
                    num_pages_per_seq = (
                        (prefill_seqused_k + block_size - 1) // block_size
                    )
                    paged_kv_indptr = torch.zeros(
                        num_prefills + 1,
                        dtype=torch.int32,
                        device=key_cache.device,
                    )
                    torch.cumsum(
                        num_pages_per_seq.int(), dim=0, out=paged_kv_indptr[1:]
                    )

                    max_blocks_per_seq = prefill_block_table_2d.shape[1]
                    block_range = torch.arange(
                        max_blocks_per_seq, device=key_cache.device
                    ).unsqueeze(0)
                    valid_mask = block_range < num_pages_per_seq.unsqueeze(1)
                    paged_kv_indices = prefill_block_table_2d[valid_mask].int()

                    paged_kv_last_page_len = (
                        prefill_seqused_k % block_size
                    ).int()
                    paged_kv_last_page_len[
                        paged_kv_last_page_len == 0
                    ] = block_size

                    # Extract used KV blocks into compact FP8 cache
                    compact_key = torch.clamp(
                        key_cache[paged_kv_indices], max=448.0, min=-448.0
                    ).to(torch.float8_e4m3fn)
                    compact_value = torch.clamp(
                        value_cache[paged_kv_indices], max=448.0, min=-448.0
                    ).to(torch.float8_e4m3fn)

                    if cache_layout == "HND":
                        fi_kv_layout = "HND"
                        fi_key_cache = compact_key.transpose(1, 2)
                        fi_value_cache = compact_value.transpose(1, 2)
                    else:
                        fi_kv_layout = "NHD"
                        fi_key_cache = compact_key
                        fi_value_cache = compact_value

                    # Generate new sequential block table for compact cache
                    total_pages = paged_kv_indices.shape[0]
                    paged_kv_indices = torch.arange(
                        total_pages, dtype=torch.int32,
                        device=key_cache.device,
                    )
                    max_pages_per_seq = num_pages_per_seq.max().item()
                    compact_block_table = (
                        paged_kv_indptr[:-1].unsqueeze(1)
                        + torch.arange(
                            max_pages_per_seq,
                            dtype=torch.int32,
                            device=key_cache.device,
                        ).unsqueeze(0)
                    ).int()

                    # Zero out unused slots in each seq's last block
                    last_block_indices = paged_kv_indptr[1:] - 1
                    for i in range(num_prefills):
                        used = paged_kv_last_page_len[i]
                        lb = last_block_indices[i]
                        if cache_layout == "HND":
                            fi_key_cache[lb, :, used:] = 0
                            fi_value_cache[lb, :, used:] = 0
                        else:
                            fi_key_cache[lb, used:] = 0
                            fi_value_cache[lb, used:] = 0

                    # Lazily initialize FlashInfer workspace
                    if not hasattr(self, '_fi_workspace'):
                        self._fi_workspace = torch.empty(
                            128 * 1024 * 1024,
                            dtype=torch.uint8,
                            device=key_cache.device,
                        )

                    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                        self._fi_workspace,
                        kv_layout=fi_kv_layout,
                        backend="trtllm-gen",
                    )
                    wrapper.plan(
                        prefill_cu_seqlens_q,
                        paged_kv_indptr,
                        paged_kv_indices,
                        paged_kv_last_page_len,
                        self.num_heads,
                        self.num_kv_heads,
                        head_dim,
                        block_size,
                        causal=True,
                        sm_scale=self.scale,
                        q_data_type=torch.float8_e4m3fn,
                        kv_data_type=torch.float8_e4m3fn,
                        o_data_type=output.dtype,
                        seq_lens=prefill_seqused_k.int(),
                        block_tables=compact_block_table,
                    )
                    prefill_attn_output = wrapper.run(
                        prefill_query,
                        (fi_key_cache, fi_value_cache),
                    )

                    if self.target_head_size != self.head_size:
                        output[num_decode_tokens:] = prefill_attn_output[
                            :, :, : self.head_size
                        ]
                    else:
                        output[num_decode_tokens:] = prefill_attn_output

                else:
                    # Pre-Blackwell: use original flash_attn_varlen_func
                    prefill_descale_shape = (num_prefills, key.shape[1])
                    extract_block_table = block_table[num_decodes:].reshape(-1)

                    prefill_key_cache = torch.clamp(
                        key_cache[extract_block_table], max=448.0, min=-448.0
                    ).to(torch.float8_e4m3fn)
                    prefill_value_cache = torch.clamp(
                        value_cache[extract_block_table], max=448.0, min=-448.0
                    ).to(torch.float8_e4m3fn)
                    prefill_query = torch.clamp(
                        prefill_query, max=448.0, min=-448.0
                    ).to(torch.float8_e4m3fn)
                    prefill_block_table = (
                        attn_metadata.rearrange_prefill_block_table
                    )

                    # Avoid the output coping when possible to improve performance
                    direct_output_arg = (
                        self.target_head_size == self.head_size
                        # PPU flash-attn2 doesn't support it.
                        and has_out
                        and (is_vllm_fa or attn_metadata.fa_version == 3)
                    )
                    prefill_attn_output = flash_attn_varlen_func(
                        q=prefill_query,
                        k=prefill_key_cache,
                        v=prefill_value_cache,
                        **{
                            "out": output[num_decode_tokens:],
                        }
                        if direct_output_arg
                        else {},
                        cu_seqlens_q=prefill_cu_seqlens_q,
                        max_seqlen_q=max_seqlen_q,
                        seqused_k=prefill_seqused_k,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=self.scale,
                        causal=True,
                        **{
                            "alibi_slopes": self.alibi_slopes,
                        }
                        if self.alibi_slopes is not None
                        else {},
                        window_size=self.sliding_window,
                        block_table=prefill_block_table,
                        # softcap=self.logits_soft_cap,
                        **{
                            "scheduler_metadata": None,
                            "num_splits": attn_metadata.max_num_splits,
                        }
                        if (scheduler_meta_supported and has_num_splits)
                        else {
                            "scheduler_metadata": None,
                        } if scheduler_meta_supported else {},
                        **{
                            "fa_version": attn_metadata.fa_version,
                            "q_descale": layer._q_scale.expand(
                                prefill_descale_shape
                            ),
                            "k_descale": layer._k_scale.expand(
                                prefill_descale_shape
                            ),
                            "v_descale": layer._v_scale.expand(
                                prefill_descale_shape
                            ),
                        }
                        if attn_metadata.fa_version == 3
                        else {},
                    )
                    if not direct_output_arg:
                        if self.target_head_size != self.head_size:
                            output[num_decode_tokens:] = prefill_attn_output[
                                :, :, : self.head_size
                            ]
                        else:
                            output[num_decode_tokens:] = prefill_attn_output

            if envs.VLLM_FLASH_ATTN_FP8_ATTENTION and self.is_target_model:
                # If enabled fp8 prefill, attn compute below is to do decode compute,
                # otherwise is prefill-decode mixed compute
                num_actual_tokens = num_decode_tokens
                cu_seqlens_q = cu_seqlens_q[: num_decodes + 1]
                seqused_k = seqused_k[:num_decodes]
                block_table = block_table[:num_decodes]
                descale_shape = (num_decodes, key.shape[1])
                scheduler_metadata = (
                    None  # When prefill decode split, not use scheduler_metadata
                )

            direct_output_arg = (
                self.target_head_size == self.head_size
                # PPU flash-attn2 doesn't support it.
                and has_out
                and (is_vllm_fa or attn_metadata.fa_version == 3)
            )
            attn_output = flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                **{
                    "out": output[:num_actual_tokens],
                }
                if direct_output_arg
                else {},
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                **{
                    "seqused_k": seqused_k,
                }
                if has_seqused_k
                else {
                    "cu_seqlens_k": cu_seq_lens,
                },
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                **{
                    "alibi_slopes": self.alibi_slopes,
                }
                if self.alibi_slopes is not None
                else {},
                window_size=self.sliding_window,
                block_table=block_table,
                **{
                    "logits_soft_cap": self.logits_soft_cap,
                }
                if (attn_metadata.fa_version == 3 and self.logits_soft_cap != 0)
                else {},
                **{
                    "scheduler_metadata": scheduler_metadata,
                    "num_splits": attn_metadata.max_num_splits,
                }
                if (scheduler_meta_supported and has_num_splits)
                else {
                    "scheduler_metadata": scheduler_metadata,
                } if scheduler_meta_supported else {},
                **{
                    "fa_version": attn_metadata.fa_version,
                    "q_descale": layer._q_scale.expand(descale_shape),
                    "k_descale": layer._k_scale.expand(descale_shape),
                    "v_descale": layer._v_scale.expand(descale_shape),
                }
                if attn_metadata.fa_version == 3
                else {},
                **{
                    "num_prefill": num_prefills,
                    "max_seqlen_k_decode": max_seqlen_k_decode,
                    "max_seqlen_k_prefill": max_seqlen_k_prefill,
                }
                if not envs.VLLM_ATTENTION_DISABLE_PD_SEPARATION
                else {},
                **{
                    "s_aux": self.sinks,
                }
                if self.sinks is not None
                else {},
            )
            if not direct_output_arg:
                if self.target_head_size != self.head_size:
                    output[:num_actual_tokens] = attn_output[:, :, : self.head_size]
                else:
                    output[:num_actual_tokens] = attn_output
            return output

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            cu_prefix_kv_lens=attn_metadata.cu_prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            cu_suffix_kv_lens=attn_metadata.cu_suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        context_attn_out, context_lse = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.dcp_context_kv_lens,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            **{
                "return_softmax_lse": True,
            }
            if is_vllm_fa
            else {
                "return_attn_probs": True,
            },
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        # FA returns LSE in shape [ H, B ] but cp_lse_ag_out_rs wants [ B, H ]
        context_attn_out_cor, context_lse_cor = cp_lse_ag_out_rs(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            **{
                "return_softmax_lse": True,
            }
            if is_vllm_fa
            else {
                "return_attn_probs": True,
            },
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        # Call flash attention directly on Q, K, V tensors
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            **{
                "alibi_slopes": self.alibi_slopes,
            }
            if self.alibi_slopes is not None
            else {},
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled else 0,
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    cu_prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    cu_suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        **{
            "seqused_k": prefix_kv_lens,
        }
        if has_seqused_k
        else {
            "cu_seqlens_k": cu_prefix_kv_lens,
        },
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=sliding_window,
        block_table=block_table[:1],
        **{
            "softcap": logits_soft_cap,
        }
        if (fa_version == 3 and logits_soft_cap != 0)
        else {},
        **{
            "return_softmax_lse": True,
        }
        if is_vllm_fa
        else {
            "return_attn_probs": True,
        },
        **{
            "num_splits": 1 if vllm_is_batch_invariant() else max_num_splits,
            "scheduler_metadata": prefix_scheduler_metadata,
        }
        if (scheduler_meta_supported and has_num_splits)
        else {
            "scheduler_metadata": prefix_scheduler_metadata,
        } if scheduler_meta_supported else {},
        **{
            "fa_version": fa_version,
            "q_descale": q_descale.expand(descale_shape)
            if q_descale is not None
            else None,
            "k_descale": k_descale.expand(descale_shape)
            if k_descale is not None
            else None,
            "v_descale": v_descale.expand(descale_shape)
            if v_descale is not None
            else None,
        }
        if fa_version == 3
        else {},
        # s_aux is incorporated into prefix_lse inside the GPU kernel,
        # enabling its effect during the final attention merge.
        **{
            "s_aux": s_aux,
        }
        if s_aux is not None
        else {},
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        **{
            "seqused_k": suffix_kv_lens,
        }
        if has_seqused_k
        else {
            "cu_seqlens_k": cu_suffix_kv_lens,
        },
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=sliding_window,
        block_table=block_table[:, num_common_kv_blocks:],
        **{
            "softcap": logits_soft_cap,
        }
        if (fa_version == 3 and logits_soft_cap != 0)
        else {},
        **{
            "return_softmax_lse": True,
        }
        if is_vllm_fa
        else {
            "return_attn_probs": True,
        },
        **{
            "num_splits": 1 if vllm_is_batch_invariant() else max_num_splits,
            "scheduler_metadata": suffix_scheduler_metadata,
        }
        if (scheduler_meta_supported and has_num_splits)
        else {
            "scheduler_metadata": suffix_scheduler_metadata,
        } if scheduler_meta_supported else {},
        **{
            "fa_version": fa_version,
            "q_descale": q_descale.expand(descale_shape)
            if q_descale is not None
            else None,
            "k_descale": k_descale.expand(descale_shape)
            if k_descale is not None
            else None,
            "v_descale": v_descale.expand(descale_shape)
            if v_descale is not None
            else None,
        }
        if fa_version == 3
        else {},
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
