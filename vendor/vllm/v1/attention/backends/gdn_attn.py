# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    num_decode_draft_tokens_cpu: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    # Tree attention
    retrieve_parent_token: torch.Tensor | None = None

class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        if self.speculative_config:
            self.num_spec = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.decode_cudagraph_max_bs = min(
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1),
            self.compilation_config.max_cudagraph_capture_size,
        )

        self.spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.retrieve_parent_token = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )

    def get_retrieve_parent_token(self, num_decodes):
        if self.vllm_config.speculative_config is None:
            return None
        if self.vllm_config.speculative_config.tree_drafter_params is None:
            return None
        if num_decodes == 0:
            return None
        paths = self.vllm_config.speculative_config.tree_drafter_params.draft_nodes
        path_to_index = {(): 0}
        parents = [-1]

        for path in paths:
            parent_path = path[:-1]
            if parent_path not in path_to_index:
                raise ValueError(f"Missing parent path: {parent_path} for {path}")
            path_to_index[path] = len(path_to_index)
            parents.append(path_to_index[parent_path])

        parents = [torch.tensor(parents).cuda() for _ in range(num_decodes)]
        result = torch.stack(parents, dim=0)
        self.retrieve_parent_token[:num_decodes].copy_(
            result, non_blocking=True
        )
        return self.retrieve_parent_token[:num_decodes]

    @record_function_or_nullcontext("GDN.build")
    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        existing_attn_metadata: GDNAttentionMetadata | None = None,
    ) -> GDNAttentionMetadata:

        if existing_attn_metadata is not None:
            return self._build_incremental(
                common_prefix_len,
                common_attn_metadata,
                num_accepted_tokens,
                num_decode_draft_tokens_cpu,
                fast_build,
                existing_attn_metadata,
            )

        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens = m.num_computed_tokens_cpu
        context_lens_tensor = context_lens.to(query_start_loc.device, non_blocking=True)
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            # we use -1 for prefill so 0 is also for decoding.
            or (num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0].numel()
                == 0)
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
            else:
                spec_sequence_masks = spec_sequence_masks.to(
                    query_start_loc.device, non_blocking=True
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            num_accepted_tokens = None
        else:
            query_lens = torch.diff(query_start_loc)
            query_lens_cpu = torch.diff(query_start_loc_cpu)

            non_spec_query_lens_cpu = query_lens_cpu[num_spec_decodes:]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )

                spec_state_indices_tensor = (
                    block_table_tensor[:, : self.num_spec + 1]
                )
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc
                non_spec_query_start_loc = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    : num_spec_decodes, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    num_spec_decodes :, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[: num_spec_decodes], dim=0, out=spec_query_start_loc[1:]
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[num_spec_decodes :],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[: num_spec_decodes]

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks is not None:
                has_initial_state = has_initial_state[num_spec_decodes :]
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(non_spec_query_start_loc)
            )
        else:
            has_initial_state = None
            nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and self.use_spec_decode
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            self.spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]

            self.spec_sequence_masks[num_spec_decodes:].fill_(False)
            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks, non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[non_spec_token_indx.size(0) :].fill_(0)
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[: batch_size]

            self.spec_token_indx[spec_token_indx.size(0) :].fill_(0)
            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: batch_size]

            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            self.spec_query_start_loc[num_spec_decodes + 1 :].fill_(
                spec_num_query_tokens
            )
            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]

            self.num_accepted_tokens[num_spec_decodes:].fill_(1)
            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]

        if (
            self.use_full_cuda_graph
            and not self.use_spec_decode
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]

            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            self.non_spec_query_start_loc[num_decodes + 1 :].fill_(
                non_spec_num_query_tokens
            )
            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(
                non_spec_num_query_tokens
            )

        retrieve_parent_token = self.get_retrieve_parent_token(num_spec_decodes)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            retrieve_parent_token=retrieve_parent_token,
        )
        return attn_metadata

    @record_function_or_nullcontext("GDN._build_incremental")
    def _build_incremental(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None,
        num_decode_draft_tokens_cpu: torch.Tensor,
        fast_build: bool,
        existing_attn_metadata: GDNAttentionMetadata,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata
        em = existing_attn_metadata

        spec_sequence_masks = em.spec_sequence_masks
        num_prefills = em.num_prefills
        num_prefill_tokens = em.num_prefill_tokens
        num_decodes = em.num_decodes
        num_decode_tokens = em.num_decode_tokens
        num_spec_decodes = em.num_spec_decodes
        num_spec_decode_tokens = em.num_spec_decode_tokens

        has_initial_state = em.has_initial_state
        spec_query_start_loc = em.spec_query_start_loc
        non_spec_query_start_loc = em.non_spec_query_start_loc
        spec_token_indx = em.spec_token_indx
        non_spec_token_indx = em.non_spec_token_indx

        num_accepted_tokens = em.num_accepted_tokens
        num_decode_draft_tokens_cpu = em.num_decode_draft_tokens_cpu

        nums_dict = em.nums_dict
        batch_ptr = em.batch_ptr
        token_chunk_offset_ptr = em.token_chunk_offset_ptr

        retrieve_parent_token = em.retrieve_parent_token
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        if spec_sequence_masks is None:
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
        else:
            if num_prefills == 0 and num_decodes == 0:
                spec_state_indices_tensor = (
                    block_table_tensor[:, : self.num_spec + 1]
                )
                non_spec_state_indices_tensor = None
            else:
                spec_state_indices_tensor = block_table_tensor[
                    : num_spec_decodes, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    num_spec_decodes :, 0
                ]

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and self.use_spec_decode
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            self.spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]

        if (
            self.use_full_cuda_graph
            and not self.use_spec_decode
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            retrieve_parent_token=retrieve_parent_token,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        existing_attn_metadata: GDNAttentionMetadata | None = None,
    ) -> GDNAttentionMetadata:
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
        m._num_computed_tokens_cpu = m.seq_lens_cpu - num_accepted_tokens.cpu()

        return self.build(
            0, m, num_accepted_tokens, num_decode_draft_tokens_cpu,
            existing_attn_metadata=existing_attn_metadata
        )
