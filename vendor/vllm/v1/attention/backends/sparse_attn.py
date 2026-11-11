# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with SparseFlashAttention."""

import inspect
import math
import os
from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.attention.layer import Attention
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
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.blocksparseutils import block_sparse_attention
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

try:
    from sparse_attn import sparse_attn_func
except (ImportError, ModuleNotFoundError):
    try:
        from vllm.vllm_flash_attn import sparse_attn_func
    except (ImportError, ModuleNotFoundError):
        from vllm_flash_attn import sparse_attn_func

logger = init_logger(__name__)

if is_flash_attn_varlen_func_available():
    has_seqused_k = "seqused_k" in inspect.signature(flash_attn_varlen_func).parameters
    has_num_prefill = False
else:
    has_seqused_k = False
    has_num_prefill = False


class SparseFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # NOTE(tdoublep): while in principle, FA supports
    # MultipleOf(16), these are the block sizes that do not
    # suffer from the NaN propagation problem described here:
    # https://github.com/Dao-AILab/flash-attention/issues/1974
    supported_kernel_block_sizes: ClassVar[list[int | MultipleOf]] = (
        [16, 32, 64, 128, 256]
    )

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """SparseFlashAttention supports all attention types."""

        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["SparseFlashAttentionImpl"]:
        return SparseFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SparseFlashAttentionMetadataBuilder"]:
        return SparseFlashAttentionMetadataBuilder

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
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
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
class SparseFlashAttentionMetadata:
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

    # Causal attention.
    causal: bool = True
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
        assert isinstance(layer.impl, SparseFlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class SparseFlashAttentionMetadataBuilder(AttentionMetadataBuilder
                                          [SparseFlashAttentionMetadata]):
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
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    ) if is_vllm_fa else AttentionCGSupport.NEVER

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

        if self.speculative_config:
            self.num_spec = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0

        # Splits prefill/decode
        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        
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
            self.max_num_splits = envs.VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> SparseFlashAttentionMetadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SparseFlashAttentionMetadata:
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
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
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
                qkv_dtype = SparseFlashAttentionBackend.get_fp8_dtype_for_flashattn(
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
            query_kv_lens_cpu = (
                common_attn_metadata.query_start_loc_cpu[1:]
                - common_attn_metadata.query_start_loc_cpu[:-1]
            )
            dcp_context_kv_lens_cpu = seq_lens_cpu - query_kv_lens_cpu

            dcp_context_kv_lens_cpu = get_dcp_local_seq_lens(
                dcp_context_kv_lens_cpu,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            dcp_context_kv_lens = dcp_context_kv_lens_cpu.to(self.device)
            max_dcp_context_kv_len = dcp_context_kv_lens.max().item()

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
            suffix_kv_lens = (seq_lens_cpu[:num_reqs] - common_prefix_len).to(
                self.device, non_blocking=True
            )
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

        attn_metadata = SparseFlashAttentionMetadata(
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
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            causal=causal,
            fa_version=self.fa_version,
        )

        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False # Sparse attn not support cascade attention yet


class SparseFlashAttentionImpl(AttentionImpl):
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
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
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
                "SparseFlashAttention does not support fp8 kv-cache on this device."
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
        prefixes = prefix.split(".")
        self.layer_idx = int(prefixes[prefixes.index("layers") + 1])
        self.vllm_config = get_current_vllm_config()

        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=self.alibi_slopes is not None)
        self.model_dtype = self.vllm_config.model_config.dtype
        self.vllm_flash_attn_supports_fp8 = flash_attn_supports_fp8()
        self.supports_quant_query_input = True

        self.is_target_model = True
        if speculative_config := self.vllm_config.speculative_config:
            self.is_target_model = (
                    self.layer_idx < self.vllm_config.model_config.get_num_layers(
                        self.vllm_config.parallel_config
                    )
                )
            self.num_speculative_tokens = speculative_config.num_speculative_tokens

        if not self.is_target_model:
            sliding_window = self.vllm_config.speculative_config\
                .draft_model_config.get_sliding_window()
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        if self.is_target_model:
            # load sparse attention config
            from vllm.model_executor.model_loader.weight_utils import (
                get_sparse_attention_config,
            )
            self.sparse_attention_config = get_sparse_attention_config(
                self.vllm_config.model_config, self.vllm_config.load_config)
            
            if not envs.VLLM_VL_BLOCKSPARSE_ATTENTION:
                if not self.sparse_attention_config:
                    logger.warning_once("Sparse attention will not be enabled as "
                                    "sparse attention config is not provided.")
                assert self.sparse_attention_config
                self.sparse_attention_config = {
                    int(i): j
                    for i, j in self.sparse_attention_config[
                        self.layer_idx].items()
                }
                start_head = self.num_heads * get_tensor_model_parallel_rank()
                end_head = start_head + self.num_heads
                self.sparse_attention_config = [
                    self.sparse_attention_config[i]
                    for i in range(start_head, end_head)
                ]
                # temporarily set to 64 like minference
                self.sparse_attention_last_q = 64  
                self.arange = torch.arange(self.sparse_attention_last_q,
                                        device="cuda")
                self.last_q_mask = (self.arange[None, None, :, None]
                                    >= self.arange[None, None, None, :])

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SparseFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with SparseFlashAttention.

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
                "fused output quantization is not yet "
                "supported for SparseFlashAttentionImpl"
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
            dtype = SparseFlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)
            
        fp8_attention = self.kv_cache_dtype.startswith("fp8")

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            cu_seq_lens = attn_metadata.cu_seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            num_prefills = attn_metadata.num_prefills

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            num_prefills = attn_metadata.num_prefills
            
            direct_output_arg = None
            
            # do sparse attn
            if num_prefills > 0 and self.is_target_model:
                assert attn_metadata.query_start_loc.shape[0] == 2
                query = query[:num_actual_tokens]
                kv_length = seqused_k
                k_cache = key_cache
                v_cache = value_cache

                block_tables_kv = attn_metadata.block_table[:, :(
                    kv_length // kv_cache[0].shape[1]) + 1]

                if fp8_attention:
                    query = query.to(torch.bfloat16)
                    k_cache = k_cache.to(torch.bfloat16)
                    v_cache = v_cache.to(torch.bfloat16)

                key = k_cache[block_tables_kv].view(
                    -1, *k_cache.shape[-2:])[:kv_length]
                value = v_cache[block_tables_kv].view(
                    -1, *v_cache.shape[-2:])[:kv_length]

                # calculate vertical and slash
                group_size = int(query.size(-2) / key.size(-2))
                num_device_q_heads = query.size(-2)
                _, num_device_k_heads, head_dim = key.shape
                key_process = (key.unsqueeze(2).repeat(
                    1, 1, group_size,
                    1).reshape(-1, num_device_k_heads * group_size, head_dim))
                value_process = (value.unsqueeze(2).repeat(
                    1, 1, group_size,
                    1).reshape(-1, num_device_k_heads * group_size, head_dim))
                
                if envs.VLLM_VL_BLOCKSPARSE_ATTENTION:
                    query_states = query[:num_actual_tokens]\
                                        .unsqueeze(0).transpose(1, 2)
                    key_states = key_process.unsqueeze(0).transpose(1, 2)
                    value_states = value_process.unsqueeze(0).transpose(1, 2)
                    
                    if fp8_attention:
                        query_states = query_states.to(torch.float8_e4m3fn)
                        key_states = key_states.to(torch.float8_e4m3fn)
                        value_states = value_states.to(torch.float8_e4m3fn)
                        
                    toppvalue = float(os.getenv("SPARSE_TOPP", "0.8"))
                    topkvalue = float(os.getenv("SPARSE_TOPK", "0"))
                        
                    out, sparselse, sparsetime = block_sparse_attention(
                            query_states,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
                            key_states,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
                            value_states,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
                            (toppvalue, topkvalue),
                            64,
                            64,
                            self.scale,
                            fp8_attention=fp8_attention,
                            model_dtype=self.model_dtype
                        ) 
                    attn_output = out.transpose(1,2).squeeze(0).contiguous()
                else:     
                    heads_vertical_size = torch.empty(size=(num_device_q_heads, ),
                                                    dtype=torch.int32)
                    heads_slash_size = torch.empty(size=(num_device_q_heads, ),
                                                dtype=torch.int32)
                    for head_id in range(query.size(-2)):
                        config = self.sparse_attention_config[head_id]
                        ty, vertical_size, slash_size, *_ = \
                            config + [None,] * (4 - len(config))
                        assert ty == "vertical_and_slash"
                        if vertical_size == 30:
                            vertical_size += 100
                        heads_vertical_size[head_id] = vertical_size
                        heads_slash_size[head_id] = slash_size
                    last_q_size = min(query.size(0), self.sparse_attention_last_q)
                    
                    qk = (query.transpose(0, 1)[:, -last_q_size:] *
                        self.scale) @ key_process.permute(1, 2, 0)
                    qk[:, :, -last_q_size:] = torch.where(
                        self.last_q_mask[..., -last_q_size:,
                                        -last_q_size:].to(qk.device),
                        qk[:, :, -last_q_size:], -torch.inf)
                    qk = torch.nn.functional.softmax(qk,
                                                    dim=-1,
                                                    dtype=torch.float32)
                    vertical = qk.sum(-2, keepdim=True)
                    vertical[..., :30] = torch.inf
                    int32_max = torch.iinfo(torch.int32).max
                    int32_min = torch.iinfo(torch.int32).min
                    max_slash_topk = torch.max(heads_slash_size).item()
                    max_vertical_topk = torch.max(heads_vertical_size).item()
                    vertical = vertical.reshape((num_device_q_heads, -1))
                    max_vertical_topk = min(vertical.shape[-1], max_vertical_topk)
                    vertical_topk_buffer = torch.topk(vertical, max_vertical_topk,
                                                    -1).indices
                    slash_topk_buffer = torch.empty(size=(num_device_q_heads,
                                                        max_slash_topk),
                                                    dtype=torch.int64,
                                                    device=qk.device)

                    for head_i in range(num_device_q_heads):
                        #  (nqheads=1, lastq, k_len)
                        head_score = qk[head_i:head_i + 1, :, :]
                        slash_scores = _sum_all_diagonal_matrix(head_score)
                        if head_score.size(1) != 1:
                            # drop right up corner
                            slash_scores = slash_scores[..., :-last_q_size + 1]
                        slash_scores[..., -100:] = torch.inf

                        head_slash_size = heads_slash_size[head_i]
                        head_slash_size = min(head_slash_size, vertical.size(-1))
                        slash_topk = torch.topk(slash_scores, head_slash_size,
                                                -1).indices
                        #（nheads, max_topk）
                        slash_topk_buffer[head_i, :head_slash_size] = slash_topk

                        # reset heads topk
                        heads_slash_size[head_i] = head_slash_size
                        heads_vertical_size[head_i] = min(
                            heads_vertical_size[head_i], max_vertical_topk)

                    # store
                    vertical_buffer = torch.full(
                        (num_device_q_heads, max_vertical_topk),
                        int32_max,
                        dtype=torch.int64,
                        device=query.device)
                    slash_buffer = torch.full((num_device_q_heads, max_slash_topk),
                                            int32_min,
                                            dtype=torch.int64,
                                            device=query.device)
                    vertical_size_buffer = torch.empty(size=(num_device_q_heads, ),
                                                    dtype=torch.int32,
                                                    device=query.device)
                    slash_sizes_buffer = torch.empty(size=(num_device_q_heads, ),
                                                    dtype=torch.int32,
                                                    device=query.device)
                    for head_i in range(num_device_q_heads):
                        vertical_topk = vertical_topk_buffer[
                            head_i, :heads_vertical_size[head_i]]
                        vertical_indices = vertical_topk
                        if vertical_indices.nelement() == 0:
                            vertical_indices = torch.cat([
                                vertical_indices,
                                torch.arange(0,
                                            key_process.size(0),
                                            max(1,
                                                key_process.size(0) / 5),
                                            dtype=torch.int32,
                                            device=vertical_indices.device)
                            ])
                        slash_topk = slash_topk_buffer[
                            head_i, :heads_slash_size[head_i]]
                        slash_indices = ((qk.size(-1) - 1) - slash_topk)
                        # fill buffer
                        v_count = vertical_indices.nelement()
                        s_count = slash_indices.nelement()
                        vertical_size_buffer[head_i] = v_count
                        slash_sizes_buffer[head_i] = s_count
                        vertical_buffer[head_i, :v_count].copy_(vertical_indices)
                        slash_buffer[head_i, :s_count].copy_(slash_indices)

                    q_len = query.shape[0]
                    q_heads = query.shape[1]
                    h_dim = query.shape[-1]

                    query_states = query.unsqueeze(0).transpose(1, 2)
                    key_states = key_process.unsqueeze(0).transpose(1, 2)
                    value_states = value_process.unsqueeze(0).transpose(1, 2)

                    if fp8_attention:
                        query_states = query_states.to(torch.float8_e4m3fn)
                        key_states = key_states.to(torch.float8_e4m3fn)
                        value_states = value_states.to(torch.float8_e4m3fn)

                    out, s_lse = _vertical_slash_sparse_attention(
                        query_states,
                        key_states,
                        value_states,
                        vertical_buffer,
                        slash_buffer,
                        self.scale,
                        causal=True,
                        vertical_indices_count=vertical_size_buffer,
                        slash_indices_count=slash_sizes_buffer,
                        fp8_attention=fp8_attention,
                        model_dtype=self.model_dtype)

                    attn_output = out.view(q_heads, q_len, h_dim).transpose(0, 1)
                    s_lse = s_lse.view(q_heads, q_len,
                                    1).squeeze(-1).unsqueeze(0).float()
            else:
                # Avoid the output coping when possible to improve performance
                direct_output_arg = (
                    self.target_head_size == self.head_size
                    # PPU flash-attn2 doesn't support it.
                    and (is_vllm_fa or attn_metadata.fa_version == 3)
                )
                
                attn_output = flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    **{
                        "out": output[:num_actual_tokens],
                    } if direct_output_arg else {},
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

        raise ValueError("Sparse Flash Attention not support"
                         "cascade attention yet.")

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: SparseFlashAttentionMetadata,
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


def _sum_all_diagonal_matrix(mat: torch.tensor):
    h, n, m = mat.shape
    # Zero matrix used for padding
    zero_mat = torch.zeros((h, n, n), device=mat.device)
    # pads the matrix on left and right
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    # Change the strides
    mat_strided = mat_padded.as_strided((1, n, n + m),
                                        (n * (2 * n + m), 2 * n + m + 1, 1))
    # Sums the resulting matrix's columns
    sum_diags = torch.sum(mat_strided, 1)
    return sum_diags[:, 1:]  # drop left bottom corner


def _vertical_slash_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,  # [BATCH, N_HEADS, N_KV_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_KV_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    softmax_scale: float,
    causal: bool = True,
    stage: str = "intra",
    block_size_M: int = 64,
    block_size_N: int = 64,
    vertical_indices_count: torch.Tensor = None,  # [N_HEADS,]
    slash_indices_count: torch.Tensor = None,
    fp8_attention: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
):
    if stage == "intra":
        assert causal
    else:
        assert not causal

    batch_size, num_heads, context_size, head_dim = query.shape
    _, _, kv_seq_len, _ = key.shape

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2**math.ceil(math.log2(head_dim)) - head_dim
        query = torch.nn.functional.pad(query,
                                        [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value,
                                        [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = v_idx.to(torch.int32).reshape(
        (batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape(
        (batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    q_seqlens = torch.tensor([context_size],
                             dtype=torch.int32,
                             device=query.device)
    kv_seqlens = torch.tensor([kv_seq_len],
                              dtype=torch.int32,
                              device=query.device)
    (
        block_count,
        block_offset,
        column_count,
        column_index,
    ) = ops.convert_vertical_slash_indexes_mergehead(
        q_seqlens, kv_seqlens, v_idx, s_idx, vertical_indices_count,
        slash_indices_count, context_size, block_size_M, block_size_N, causal)
    
    # TODO: not set synchronize will trigger illegal mem, need to fix
    torch.cuda.synchronize() 
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()

    if envs.VLLM_FP8_SPARSE_ATTENTION and not fp8_attention:
        q = torch.clamp(q, max=448.0, min=-448.0).to(torch.float8_e4m3fn)
        k = torch.clamp(k, max=448.0, min=-448.0).to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
    out, lse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        causal=causal,
        softmax_scale=softmax_scale,
        return_softmax_lse=True,
    )
    if envs.VLLM_FP8_SPARSE_ATTENTION or fp8_attention:
        out = out.to(model_dtype)

    out = out.transpose(1, 2).contiguous()
    softmax_lse = lse.reshape(*lse.shape, 1)
    return (out[..., :context_size, :head_dim],
            softmax_lse[..., :context_size, :])
