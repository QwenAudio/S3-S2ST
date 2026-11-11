# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with SparseFlashAttention.
Based on FlashAttention backend implementation."""

import inspect
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm import _custom_ops as ops

# isort: off
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    is_quantized_kv_cache,
)

# isort: on
from vllm.attention.backends.utils import (
    PAD_SLOT_ID,
    CommonAttentionState,
    compute_slot_mapping,
    compute_slot_mapping_start_idx,
    get_num_prefill_decode_query_kv_tokens,
    get_seq_len_block_table_args,
    is_all_cross_attn_metadata_set,
    is_all_encoder_attn_metadata_set,
    is_block_tables_empty,
)
from vllm.attention.utils.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
)
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.multimodal import MultiModalPlaceholderMap
from vllm.utils.torch_utils import async_tensor_h2d, make_tensor_with_pad

if TYPE_CHECKING:
    from vllm.worker.model_runner import (
        ModelInputForGPUBuilder,
        ModelInputForGPUWithSamplingMetadata,
    )

try:
    from sparse_attn import sparse_attn_func
except (ImportError, ModuleNotFoundError):
    try:
        from vllm.vllm_flash_attn import sparse_attn_func
    except (ImportError, ModuleNotFoundError):
        from vllm_flash_attn import sparse_attn_func

if envs.VLLM_USE_BLADNN_FLASH_ATTN:
    from bladnn import flash_attn_varlen_func, flash_attn_with_kvcache
else:
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    except (ImportError, ModuleNotFoundError):
        try:
            from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
        except (ImportError, ModuleNotFoundError):
            from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

logger = init_logger(__name__)


class SparseFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SPARSE_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["SparseFlashAttentionImpl"]:
        return SparseFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return SparseFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["SparseFlashAttentionMetadataBuilder"]:
        return SparseFlashAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]

        ops.copy_blocks(key_caches, value_caches, src_to_dists)


@dataclass
class SparseFlashAttentionMetadata(AttentionMetadata):
    """Metadata for SparseFlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[list[int]] = None
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor] = None

    # Length of the query in the batch.
    query_lens: Optional[list[int]] = None
    # query_lens stored as a tensor.
    query_lens_tensor: Optional[torch.Tensor] = None

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch.
    max_query_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int = None
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.

    max_decode_seq_len: int = None
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor] = None

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool = False

    # Maximum query length in the batch.
    max_query_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    _cached_prefill_metadata: Optional["SparseFlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["SparseFlashAttentionMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[list[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    encoder_seq_start_loc: Optional[torch.Tensor] = None
    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None
    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        """
        All attention metadata required for encoder attention is set.
        """
        return is_all_encoder_attn_metadata_set(self)

    @property
    def is_all_cross_attn_metadata_set(self):
        """
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        """
        return is_all_cross_attn_metadata_set(self)

    @property
    def prefill_metadata(self) -> Optional["SparseFlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None and envs.VLLM_VL_FASTV_K <= 0:
            return self._cached_prefill_metadata

        assert (self.seq_lens is not None) or (self.encoder_seq_lens is not None)
        assert (self.seq_lens_tensor is not None) or (
            self.encoder_seq_lens_tensor is not None
        )

        # Compute some attn_metadata fields which default to None
        query_start_loc = (
            None
            if self.query_start_loc is None
            else self.query_start_loc[: self.num_prefills + 1]
        )
        slot_mapping = (
            None
            if self.slot_mapping is None
            else self.slot_mapping[: self.num_prefill_tokens]
        )
        seq_lens = None if self.seq_lens is None else self.seq_lens[: self.num_prefills]
        seq_lens_tensor = (
            None
            if self.seq_lens_tensor is None
            else self.seq_lens_tensor[: self.num_prefills]
        )
        query_lens = (
            None if self.query_lens is None else self.query_lens[: self.num_prefills]
        )
        query_lens_tensor = (
            None
            if self.query_lens_tensor is None
            else self.query_lens_tensor[: self.num_prefills]
        )
        seq_start_loc = (
            None
            if self.seq_start_loc is None
            else self.seq_start_loc[: self.num_prefills + 1]
        )
        context_lens_tensor = (
            None
            if self.context_lens_tensor is None
            else self.context_lens_tensor[: self.num_prefills]
        )
        block_tables = (
            None
            if self.block_tables is None
            else self.block_tables[: self.num_prefills]
        )

        # Don't use block tables for the attention of the first prefill chunk,
        # as paged kv cache is known to hurt performance.

        self._cached_prefill_metadata = SparseFlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            query_lens=query_lens,
            query_lens_tensor=query_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_query_len=0,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["SparseFlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None and envs.VLLM_VL_FASTV_K <= 0:
            return self._cached_decode_metadata

        assert (self.seq_lens_tensor is not None) or (
            self.encoder_seq_lens_tensor is not None
        )

        # Compute some attn_metadata fields which default to None
        query_start_loc = (
            None
            if self.query_start_loc is None
            else self.query_start_loc[self.num_prefills :]
            - self.query_start_loc[self.num_prefills]
        )
        slot_mapping = (
            None
            if self.slot_mapping is None
            else self.slot_mapping[self.num_prefill_tokens :]
        )
        seq_lens = None if self.seq_lens is None else self.seq_lens[self.num_prefills :]
        seq_lens_tensor = (
            None
            if self.seq_lens_tensor is None
            else self.seq_lens_tensor[self.num_prefills :]
        )
        query_lens = (
            None if self.query_lens is None else self.query_lens[self.num_prefills :]
        )
        query_lens_tensor = (
            None
            if self.query_lens_tensor is None
            else self.query_lens_tensor[self.num_prefills :]
        )
        seq_start_loc = (
            None
            if self.seq_start_loc is None
            else self.seq_start_loc[self.num_prefills :]
            - self.seq_start_loc[self.num_prefills]
        )
        context_lens_tensor = (
            None
            if self.context_lens_tensor is None
            else self.context_lens_tensor[self.num_prefills :]
        )
        block_tables = (
            None
            if self.block_tables is None
            else self.block_tables[self.num_prefills :]
        )

        self._cached_decode_metadata = SparseFlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            query_lens=query_lens,
            query_lens_tensor=query_lens_tensor,
            max_decode_query_len=self.max_decode_query_len,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
        )
        return self._cached_decode_metadata

    def advance_step(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        sampled_token_ids: Optional[torch.Tensor],
        block_size: int,
        num_seqs: int,
        num_queries: int,
        turn_prefills_into_decodes: bool = False,
    ):
        """
        Update metadata in-place to advance one decode step.
        """
        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        if turn_prefills_into_decodes:
            # When Multi-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes. This update reflects that
            # conversion.
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1

            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens is not None
            assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs,), (
            f"{self.slot_mapping.shape} vs. {num_seqs}"
        )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs,)
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1,)
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1,)

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries,)

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        if self._cached_prefill_metadata is not None:
            self._cached_prefill_metadata.max_prefill_seq_len = self.max_prefill_seq_len
        if self._cached_decode_metadata is not None:
            self._cached_decode_metadata.max_decode_seq_len = self.max_decode_seq_len

        ops.advance_step_flashattn(
            num_seqs=num_seqs,
            num_queries=num_queries,
            block_size=block_size,
            input_tokens=model_input.input_tokens,
            sampled_token_ids=sampled_token_ids,
            input_positions=model_input.input_positions,
            seq_lens=self.seq_lens_tensor,
            slot_mapping=self.slot_mapping,
            block_tables=self.block_tables,
        )


class SparseFlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[SparseFlashAttentionMetadata]
):
    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

    def prepare(self):
        self.slot_mapping: list[int] = []
        self.prefill_seq_lens: list[int] = []
        self.context_lens: list[int] = []
        self.block_tables: list[list[int]] = []
        self.curr_seq_lens: list[int] = []
        self.multimodal_placeholder_maps: dict[str, MultiModalPlaceholderMap] = (
            defaultdict(MultiModalPlaceholderMap)
        )
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

    def _add_seq_group(
        self,
        inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
        chunked_prefill_enabled: bool,
        prefix_cache_hit: bool,
    ):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (
            seq_id,
            token_len,
            seq_len,
            curr_seq_len,
            query_len,
            context_len,
            curr_sliding_window_block,
        ) in zip(
            inter_data.seq_ids,
            [len(t) for t in inter_data.input_tokens],
            inter_data.orig_seq_lens,
            inter_data.seq_lens,
            inter_data.query_lens,
            inter_data.context_lens,
            inter_data.curr_sliding_window_blocks,
        ):
            self.context_lens.append(context_len)

            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            #
            # 1. When prefix caching/chunked prefill is enabled, the
            #    block table is requred.
            # 2. For omni's thinker model, the block table is required
            # .   to store the hidden states into cache.
            # 3. When scheduler prefetch is enabled, the block table
            #    will be required when the sequence transit to decoding
            #    phase by the scheduler prefetching (it is treated as
            #    a prompt request, however, it should be in decoding).
            block_table = []
            if prefix_cache_hit or (
                block_tables
                and self.input_builder.runner.model_config.is_omni_thinker_model()
            ):
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif (
                chunked_prefill_enabled or not is_prompt
            ) and block_tables is not None:
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window
            )
            compute_slot_mapping(
                is_profile_run,
                self.slot_mapping,
                seq_id,
                curr_seq_len,
                context_len,
                start_idx,
                self.block_size,
                inter_data.block_tables,
            )

    def _get_graph_runner_block_tables(
        self, num_seqs: int, block_tables: list[list[int]]
    ) -> torch.Tensor:
        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs

        graph_block_tables = self.runner.graph_block_tables[:num_seqs]
        for i, block_table in enumerate(block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    graph_block_tables[i, :num_blocks] = block_table
                else:
                    # It may be possible to have more blocks allocated due
                    # to lookahead slots of multi-step, however, they are
                    # not used anyway, so can be safely ignored.
                    graph_block_tables[i, :max_blocks] = block_table[:max_blocks]

        return torch.from_numpy(graph_block_tables).to(
            device=self.runner.device, non_blocking=True
        )

    def build(
        self,
        seq_lens: list[int],
        query_lens: list[int],
        cuda_graph_pad_size: int,
        batch_size: int,
    ):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any(
            [
                inter_data.prefix_cache_hit
                for inter_data in self.input_builder.inter_data_list
            ]
        )
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(
                inter_data, self.input_builder.chunked_prefill_enabled, prefix_cache_hit
            )

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens) if query_lens else 0
        decode_query_lens = query_lens[self.num_prefills :]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))

        num_seqs = len(seq_lens)
        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens
            block_tables = self._get_graph_runner_block_tables(
                num_seqs, self.block_tables
            )
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )

        assert device is not None
        context_lens_tensor = async_tensor_h2d(
            self.context_lens, torch.int, device, self.runner.pin_memory
        )
        seq_lens_tensor = async_tensor_h2d(
            seq_lens, torch.int32, device, self.runner.pin_memory
        )
        query_lens_tensor = async_tensor_h2d(
            query_lens, torch.int32, device, self.runner.pin_memory
        )
        slot_mapping_tensor = async_tensor_h2d(
            self.slot_mapping, torch.long, device, self.runner.pin_memory
        )
        query_start_loc_tensor = async_tensor_h2d(
            query_start_loc, torch.int32, device, self.runner.pin_memory
        )
        seq_start_loc_tensor = async_tensor_h2d(
            seq_start_loc, torch.int32, device, self.runner.pin_memory
        )
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in self.multimodal_placeholder_maps.items()
        }

        return SparseFlashAttentionMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=True,
            seq_lens_tensor=seq_lens_tensor,
            query_lens=query_lens,
            query_lens_tensor=query_lens_tensor,
            max_query_len=max_query_len,
            max_decode_query_len=max_decode_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )


class SparseFlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        use_irope: bool = False,
        layer_idx: int = -1,
        **extra_impl_args,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in V0.")
        if blocksparse_params is not None:
            raise ValueError("FlashAttention does not support block-sparse attention.")
        if use_irope:
            logger.warning(
                "Using irope in V0 is not supported yet, it will fall back "
                "to global attention for long context."
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = (
            (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=self.alibi_slopes is not None
        )
        if is_quantized_kv_cache(self.kv_cache_dtype) and (
            not self.kv_cache_dtype.startswith("fp8") or not flash_attn_supports_fp8()
        ):
            raise NotImplementedError(
                f"SparseFlashAttention does not support {self.kv_cache_dtype} "
                "kv-cache on this device "
                f"(FA supports fp8 = {flash_attn_supports_fp8()})."
            )
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = SparseFlashAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}."
            )
        self.attn_type = attn_type

        prefixes = prefix.split(".")
        self.layer_idx = int(prefixes[prefixes.index("layers") + 1])

        # load sparse attention config
        from vllm.model_executor.model_loader.weight_utils import (
            get_sparse_attention_config,
        )

        self.sparse_attention_config = get_sparse_attention_config(
            get_current_vllm_config().model_config,
            get_current_vllm_config().load_config,
        )
        if not self.sparse_attention_config:
            logger.warning_once(
                "Sparse attention will not be enabled as "
                "sparse attention config is not provided."
            )

        assert self.sparse_attention_config
        self.sparse_attention_config = {
            int(i): j for i, j in self.sparse_attention_config[self.layer_idx].items()
        }
        start_head = self.num_heads * get_tensor_model_parallel_rank()
        end_head = start_head + self.num_heads
        self.sparse_attention_config = [
            self.sparse_attention_config[i] for i in range(start_head, end_head)
        ]
        self.sparse_attention_last_q = 64  # temporarily set to 64 like minference
        self.arange = torch.arange(self.sparse_attention_last_q, device="cuda")
        self.last_q_mask = (
            self.arange[None, None, :, None] >= self.arange[None, None, None, :]
        )
        if is_quantized_kv_cache(self.kv_cache_dtype) and (
            not self.kv_cache_dtype.startswith("fp8") or not flash_attn_supports_fp8()
        ):
            raise NotImplementedError(
                f"SparseFlashAttention does not support {self.kv_cache_dtype} "
                "kv-cache on this device "
                f"(FA supports fp8 = {flash_attn_supports_fp8()})."
            )
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=self.alibi_slopes is not None
        )
        self.model_dtype = get_current_vllm_config().model_config.dtype
        self.vllm_flash_attn_supports_fp8 = flash_attn_supports_fp8()

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SparseFlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            output: shape = [num_tokens, num_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        NOTE: It in-place updates the output tensor.
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for SparseFlashAttentionImpl"
            )

        # NOTE(woosuk): FlashAttention2 does not support FP8 KV cache.
        if not self.vllm_flash_attn_supports_fp8 or output.dtype != torch.bfloat16:
            assert (
                layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0), (
                    "key/v_scale is only supported in FlashAttention 3 with "
                    "base dtype bfloat16")

        attn_type = self.attn_type
        if attn_type == AttentionType.ENCODER and (
            not attn_metadata.is_all_encoder_attn_metadata_set
        ):
            raise AttributeError(
                "Encoder attention requires setting encoder metadata attributes."
            )
        elif attn_type == AttentionType.ENCODER_DECODER and (
            not attn_metadata.is_all_cross_attn_metadata_set
        ):
            raise AttributeError(
                "Encoder/decoder cross-attention "
                "requires setting cross-attention "
                "metadata attributes."
            )

        kv_cache_dtype: str = self.kv_cache_dtype
        fp8_attention = kv_cache_dtype.startswith("fp8")

        if (
            fp8_attention
            and not flash_attn_supports_fp8()
            and get_flash_attn_version() != 3
        ):
            raise NotImplementedError(
                "SparseFlashAttention does not support FP8 kv-cache on this device."
            )
        softmax_scale: float = self.scale
        window_size = self.sliding_window
        alibi_slopes: Optional[torch.Tensor] = self.alibi_slopes

        if kv_cache.numel() > 0:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
            # We skip updating the KV cache under two conditions:
            #  a. When the Attention Type is ENCODER. In this phase, we compute
            #     only the encoder attention without updating the cache.
            #  b. When both Key and Value are None. This occurs during
            #     cross-attention computation in the decoding phase, where the
            #     KV cache is already populated with the cross-attention
            #     tensor. Thus, we skip cache updates during this time.
            if (
                (attn_type != AttentionType.ENCODER)
                and (key is not None)
                and (value is not None)
            ):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                # profiling run.
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key,
                    value,
                    kv_cache[0],
                    kv_cache[1],
                    updated_slot_mapping.flatten(),  # type: ignore[union-attr]
                    kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

                if fp8_attention:
                    kv_cache = kv_cache.view(torch.float8_e4m3fn)
                    key_cache = key_cache.view(torch.float8_e4m3fn)
                    value_cache = value_cache.view(torch.float8_e4m3fn)

        if fp8_attention:
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape((num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale,
            )
            query = query.reshape((num_tokens, num_heads, head_size))

        (num_prefill_query_tokens, num_prefill_kv_tokens, num_decode_query_tokens) = (
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)
        )
        decode_query = query[num_prefill_query_tokens:]
        decode_output = output[num_prefill_query_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        prefill_output = output[:num_prefill_query_tokens]
        assert query.shape[0] == num_prefill_query_tokens, (
            f"{query.shape[0]=}, {num_prefill_query_tokens=}"
        )
        assert decode_query.shape[0] == num_decode_query_tokens, (
            f"{decode_query.shape[0]=}, {num_decode_query_tokens=}"
        )

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (
                kv_cache.numel() == 0
                or prefill_meta.block_tables is None
                or prefill_meta.block_tables.numel() == 0
                or (
                    envs.VLLM_VL_FASTV_K > 0
                    and prefill_meta.context_lens_tensor[0].item() == 0
                )
            ):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                q_seq_start_loc, q_seq_len, k_seq_start_loc, k_seq_len = (
                    _get_query_key_seq_metadata(prefill_meta, True, attn_type)
                )

                key = key[:num_prefill_kv_tokens]
                value = value[:num_prefill_kv_tokens]

                if fp8_attention:
                    num_kv_tokens, num_kv_heads, head_size = key.shape
                    key, _ = ops.scaled_fp8_quant(
                        key.reshape(
                            (num_kv_tokens, num_kv_heads * head_size)
                        ).contiguous(),
                        layer._k_scale,
                    )
                    key = key.reshape((num_kv_tokens, num_kv_heads, head_size))
                    value, _ = ops.scaled_fp8_quant(
                        value.reshape(
                            (num_kv_tokens, num_kv_heads * head_size)
                        ).contiguous(),
                        layer._v_scale,
                    )
                    value = value.reshape((num_kv_tokens, num_kv_heads, head_size))

                descale_shape = (q_seq_start_loc.shape[0] - 1, key.shape[1])

                out = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=q_seq_start_loc,
                    cu_seqlens_k=k_seq_start_loc,
                    max_seqlen_q=q_seq_len,
                    max_seqlen_k=k_seq_len,
                    softmax_scale=softmax_scale,
                    causal=_get_causal_option(attn_type),
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    **{
                        "fa_version": self.vllm_flash_attn_version,
                        "q_descale": layer._q_scale.expand(descale_shape),
                        "k_descale": layer._k_scale.expand(descale_shape),
                        "v_descale": layer._v_scale.expand(descale_shape),
                    }
                    if fp8_attention
                    else {},
                )

                prefill_output.copy_(out)
            else:
                # prefix-enabled attention
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support prefix caching"
                )
                assert prefill_meta.seq_lens is not None
                assert prefill_meta.query_start_loc is not None

                # assume only process 1 req
                assert prefill_meta.query_start_loc.shape[0] == 2, (
                    f"{prefill_meta.query_start_loc.shape[0]=},{prefill_meta.query_start_loc=}"
                )

                q_seq_start_loc, q_seq_len, k_seq_start_loc, k_seq_len = (
                    _get_query_key_seq_metadata(prefill_meta, True, attn_type)
                )

                kv_length = k_seq_start_loc[1] - k_seq_start_loc[0]
                k_cache = kv_cache[0]
                v_cache = kv_cache[1]
                block_tables_kv = prefill_meta.block_tables[
                    :, : (kv_length // kv_cache[0].shape[1]) + 1
                ]

                # for fp8 kvcache, temporary convert to bf16
                if fp8_attention:
                    query = query.to(torch.bfloat16)
                    k_cache = k_cache.to(torch.bfloat16)
                    v_cache = v_cache.to(torch.bfloat16)

                key = k_cache[block_tables_kv].view(-1, *k_cache.shape[-2:])[:kv_length]
                value = v_cache[block_tables_kv].view(-1, *v_cache.shape[-2:])[
                    :kv_length
                ]

                # calculate vertical and slash
                group_size = int(query.size(-2) / key.size(-2))
                num_device_q_heads = query.size(-2)
                heads_vertical_size = torch.empty(
                    size=(num_device_q_heads,), dtype=torch.int32
                )
                heads_slash_size = torch.empty(
                    size=(num_device_q_heads,), dtype=torch.int32
                )
                for head_id in range(query.size(-2)):
                    (ty, vertical_size, slash_size, _) = self.sparse_attention_config[
                        head_id
                    ]
                    assert ty == "vertical_and_slash"
                    if vertical_size == 30:
                        vertical_size += 100
                    heads_vertical_size[head_id] = vertical_size
                    heads_slash_size[head_id] = slash_size
                last_q_size = min(query.size(0), self.sparse_attention_last_q)
                _, num_device_k_heads, head_dim = key.shape
                key_process = (
                    key.unsqueeze(2)
                    .repeat(1, 1, group_size, 1)
                    .reshape(-1, num_device_k_heads * group_size, head_dim)
                )
                value_process = (
                    value.unsqueeze(2)
                    .repeat(1, 1, group_size, 1)
                    .reshape(-1, num_device_k_heads * group_size, head_dim)
                )
                # qk = self.compiled_compute_qk_softmax
                # (query, key_process, softmax_scale, last_q_size, self.last_q_mask)
                qk = (
                    query.transpose(0, 1)[:, -last_q_size:] * softmax_scale
                ) @ key_process.permute(1, 2, 0)
                qk[:, :, -last_q_size:] = torch.where(
                    self.last_q_mask[..., -last_q_size:, -last_q_size:].to(qk.device),
                    qk[:, :, -last_q_size:],
                    -torch.inf,
                )
                qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
                vertical = qk.sum(-2, keepdim=True)
                vertical[..., :30] = torch.inf
                int32_max = torch.iinfo(torch.int32).max
                int32_min = torch.iinfo(torch.int32).min
                max_slash_topk = torch.max(heads_slash_size).item()
                max_vertical_topk = torch.max(heads_vertical_size).item()
                vertical = vertical.reshape((num_device_q_heads, -1))
                max_vertical_topk = min(vertical.shape[-1], max_vertical_topk)
                vertical_topk_buffer = torch.topk(
                    vertical, max_vertical_topk, -1
                ).indices
                slash_topk_buffer = torch.empty(
                    size=(num_device_q_heads, max_slash_topk),
                    dtype=torch.int64,
                    device=qk.device,
                )

                for head_i in range(num_device_q_heads):
                    #  (nqheads=1, lastq, k_len)
                    head_score = qk[head_i : head_i + 1, :, :]
                    slash_scores = _sum_all_diagonal_matrix(head_score)
                    if head_score.size(1) != 1:
                        # drop right up corner
                        slash_scores = slash_scores[..., : -last_q_size + 1]
                    slash_scores[..., -100:] = torch.inf

                    head_slash_size = heads_slash_size[head_i]
                    head_slash_size = min(head_slash_size, vertical.size(-1))
                    slash_topk = torch.topk(slash_scores, head_slash_size, -1).indices
                    # （nheads, max_topk）
                    slash_topk_buffer[head_i, :head_slash_size] = slash_topk

                    # reset heads topk
                    heads_slash_size[head_i] = head_slash_size
                    heads_vertical_size[head_i] = min(
                        heads_vertical_size[head_i], max_vertical_topk
                    )

                # store
                vertical_buffer = torch.full(
                    (num_device_q_heads, max_vertical_topk),
                    int32_max,
                    dtype=torch.int64,
                    device=query.device,
                )
                slash_buffer = torch.full(
                    (num_device_q_heads, max_slash_topk),
                    int32_min,
                    dtype=torch.int64,
                    device=query.device,
                )
                vertical_size_buffer = torch.empty(
                    size=(num_device_q_heads,), dtype=torch.int32, device=query.device
                )
                slash_sizes_buffer = torch.empty(
                    size=(num_device_q_heads,), dtype=torch.int32, device=query.device
                )
                for head_i in range(num_device_q_heads):
                    vertical_topk = vertical_topk_buffer[
                        head_i, : heads_vertical_size[head_i]
                    ]
                    vertical_indices = vertical_topk
                    if vertical_indices.nelement() == 0:
                        vertical_indices = torch.cat(
                            [
                                vertical_indices,
                                torch.arange(
                                    0,
                                    key_process.size(0),
                                    max(1, key_process.size(0) / 5),
                                    dtype=torch.int32,
                                    device=vertical_indices.device,
                                ),
                            ]
                        )
                    slash_topk = slash_topk_buffer[head_i, : heads_slash_size[head_i]]
                    slash_indices = (qk.size(-1) - 1) - slash_topk
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
                    softmax_scale,
                    causal=True,
                    vertical_indices_count=vertical_size_buffer,
                    slash_indices_count=slash_sizes_buffer,
                    fp8_attention=fp8_attention,
                    model_dtype=self.model_dtype,
                )

                out = out.view(q_heads, q_len, h_dim).transpose(0, 1)
                s_lse = s_lse.view(q_heads, q_len, 1).squeeze(-1).unsqueeze(0).float()

                prefill_output.copy_(out)

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            # Use flash_attn_varlen_func kernel for speculative decoding
            # because different queries might have different lengths.

            assert decode_meta.max_decode_query_len is not None
            # use only for actual varlen decoding
            if decode_meta.max_decode_query_len > 1:
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support max_decode_query_len > 1"
                )
                assert decode_meta.query_start_loc is not None
                descale_shape = (decode_meta.query_start_loc.shape[0] - 1, key.shape[1])
                has_seqused_k = (
                    "seqused_k" in inspect.signature(flash_attn_varlen_func).parameters
                )
                out = flash_attn_varlen_func(
                    q=decode_query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_decode_query_len,
                    **{
                        "seqused_k": prefill_meta.seq_lens_tensor,
                    }
                    if has_seqused_k
                    else {
                        "cu_seqlens_k": prefill_meta.seq_start_loc,
                    },
                    max_seqlen_k=decode_meta.max_decode_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    # softcap=logits_soft_cap,
                    block_table=decode_meta.block_tables,
                    # out=decode_output,
                    **{
                        "fa_version": self.vllm_flash_attn_version,
                        "q_descale": layer._q_scale.expand(descale_shape),
                        "k_descale": layer._k_scale.expand(descale_shape),
                        "v_descale": layer._v_scale.expand(descale_shape),
                    }
                    if fp8_attention
                    else {},
                )
                decode_output.copy_(out)
            else:
                # Use flash_attn_with_kvcache for normal decoding.
                (
                    seq_lens_arg,
                    _,
                    block_tables_arg,
                ) = get_seq_len_block_table_args(decode_meta, False, attn_type)
                descale_shape = (seq_lens_arg.shape[0], key_cache.shape[-2])
                out = flash_attn_with_kvcache(
                    q=decode_query.unsqueeze(1),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    block_table=block_tables_arg,
                    cache_seqlens=seq_lens_arg,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    **{
                        "fa_version": self.vllm_flash_attn_version,
                        "q_descale": layer._q_scale.expand(descale_shape),
                        "k_descale": layer._k_scale.expand(descale_shape),
                        "v_descale": layer._v_scale.expand(descale_shape),
                    }
                    if fp8_attention
                    else {},
                ).squeeze(1)
                decode_output.copy_(out)
        return output


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
    fp8_attention: Optional[bool] = False,
    model_dtype: torch.dtype = torch.bfloat16,
):
    if stage == "intra":
        assert causal
    else:
        assert not causal

    batch_size, num_heads, context_size, head_dim = query.shape
    _, _, kv_seq_len, _ = key.shape

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = F.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = F.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = F.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = (
        v_idx.to(torch.int32)
        .reshape((batch_size, num_heads, -1))
        .sort(dim=-1, descending=False)[0]
    )
    s_idx = (
        s_idx.to(torch.int32)
        .reshape((batch_size, num_heads, -1))
        .sort(dim=-1, descending=True)[0]
    )
    q_seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    kv_seqlens = torch.tensor([kv_seq_len], dtype=torch.int32, device=query.device)

    if vertical_indices_count is not None and slash_indices_count is not None:
        (
            block_count,
            block_offset,
            column_count,
            column_index,
        ) = ops.convert_vertical_slash_indexes_mergehead(
            q_seqlens,
            kv_seqlens,
            v_idx,
            s_idx,
            vertical_indices_count,
            slash_indices_count,
            context_size,
            block_size_M,
            block_size_N,
            causal,
        )
    else:
        (
            block_count,
            block_offset,
            column_count,
            column_index,
        ) = ops.convert_vertical_slash_indexes(
            q_seqlens,
            kv_seqlens,
            v_idx,
            s_idx,
            context_size,
            block_size_M,
            block_size_N,
            causal,
        )

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
    return (out[..., :context_size, :head_dim], softmax_lse[..., :context_size, :])


def _sum_all_diagonal_matrix(mat: torch.tensor):
    h, n, m = mat.shape
    # Zero matrix used for padding
    zero_mat = torch.zeros((h, n, n), device=mat.device)
    # pads the matrix on left and right
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    # Change the strides
    mat_strided = mat_padded.as_strided(
        (1, n, n + m), (n * (2 * n + m), 2 * n + m + 1, 1)
    )
    # Sums the resulting matrix's columns
    sum_diags = torch.sum(mat_strided, 1)
    return sum_diags[:, 1:]  # drop left bottom corner


def _get_query_key_seq_metadata(
    attn_metadata,
    is_prompt: bool,
    attn_type: str,
) -> tuple:
    """
    Returns sequence metadata for key and query based on the specified
    attention type and whether input is a prompt.

    This function computes the starting locations and maximum sequence lengths
    for key and query sequences for different attention types.

    Args:
        attn_metadata: The attention metadata object
        is_prompt (bool): A flag indicating if the input is a prompt
        attn_type (AttentionType): The type of attention being used.

    Returns:
        tuple: A tuple containing four integers:
            - Starting location for the query sequence.
            - Maximum sequence length for the query sequence.
            - Starting location for the key sequence.
            - Maximum sequence length for the key sequence.

    Raises:
        AttributeError: If an invalid attention type is provided.
    """
    if attn_type == AttentionType.DECODER:
        # Decoder self-attention
        # Choose max_seq_len based on whether we are in prompt_run
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (
            attn_metadata.seq_start_loc,
            max_seq_len,
            attn_metadata.seq_start_loc,
            max_seq_len,
        )

    elif attn_type == AttentionType.ENCODER_DECODER:
        # This is cross attention between the where the key
        # is the precomputed encoder attention and query
        # is the input sequence.
        # Choose query max length based on whether it is prompt
        # or not.
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (
            attn_metadata.seq_start_loc,
            max_seq_len,
            attn_metadata.encoder_seq_start_loc,
            attn_metadata.max_encoder_seq_len,
        )
    elif attn_type == AttentionType.ENCODER:
        # For encoder attention both the query and the key are same i.e the
        # encoder sequence.
        return (
            attn_metadata.encoder_seq_start_loc,
            attn_metadata.max_encoder_seq_len,
            attn_metadata.encoder_seq_start_loc,
            attn_metadata.max_encoder_seq_len,
        )
    elif attn_type == AttentionType.ENCODER_ONLY:
        assert is_prompt, "Should not have decode for encoder only model."
        return (
            attn_metadata.seq_start_loc,
            attn_metadata.max_prefill_seq_len,
            attn_metadata.seq_start_loc,
            attn_metadata.max_prefill_seq_len,
        )
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


def _get_causal_option(attn_type: str) -> bool:
    """
    Determine whether the given attention type is suitable for causal
    attention mechanisms.

    Args:
        attn_type (AttentionType): The type of attention being evaluated

    Returns:
        bool: Returns `True` if the attention type is suitable for causal
        attention (i.e., not encoder, encoder-only, or encoder-decoder),
        otherwise returns `False`.
    """
    return not (
        attn_type == AttentionType.ENCODER
        or attn_type == AttentionType.ENCODER_ONLY
        or attn_type == AttentionType.ENCODER_DECODER
    )
