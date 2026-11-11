# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import contextlib
from dataclasses import replace
from importlib.util import find_spec
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from vllm import envs
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.weight_utils import get_model_loading_tracker
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.qwen3_moe_eagle3 import Eagle3Qwen3MoeForCausalLM
from vllm.model_executor.models.qwen3_next import Qwen3NextIndexerCache
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.tree_attn import (
    TreeAttentionMetadata,
    TreeAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
)
from vllm.v1.tree_spec_decode.utils import (
    apply_accepted_draft_indices,
    copy_kv_cache_slots,
)
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


class EagleProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        self.vllm_config = vllm_config
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.ep_size = (
            vllm_config.parallel_config.expert_parallel_size
            if vllm_config.parallel_config.enable_expert_parallel
            else 1
        )
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.kv_transfer_config = vllm_config.kv_transfer_config

        self.runner = runner
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        self.attn_metadata_builder: AttentionMetadataBuilder | None = None
        self.draft_indexer_metadata_builder: AttentionMetadataBuilder | None = None
        self.attn_layer_names: list[str] = []
        self.indexer_layer_names: list[str] = []
        self.eagle3_use_aux_hidden_state: bool = (
            self._get_eagle3_use_aux_hidden_state_from_config()
        )

        # The Eagle model shares cudagraph_mode with the target model.
        self.compilation_config = self.vllm_config.compilation_config
        self.cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        self.enforce_eager = True
        if self.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            self.enforce_eager = self.speculative_config.enforce_eager
        self.cudagraph_batch_sizes = (
            sorted(self.vllm_config.compilation_config.cudagraph_capture_sizes)
        )

        model_config = self.vllm_config.model_config  
        self.eagle_use_qsa = (
            hasattr(model_config.hf_text_config, "index_topk")
            and model_config.hf_text_config.index_topk > 0
            and "Qwen3Next" in model_config.architecture
        )

        # If the attention backend can only support uniform_batch, e.g., FlashMLA,
        # the first draft step is not FULL cuda graph compatible.
        self.cg_support: AttentionCGSupport | None = None
        self.bs_to_padded_graph_size: list[int] | None = None

        # persistent buffers for cuda graph
        self.use_cuda_graph = bool(self.cudagraph_batch_sizes)
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        self.uses_mrope = self.vllm_config.model_config.uses_mrope
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros(
                (3, self.max_num_tokens + 1), dtype=torch.int64, device=device
            )
        else:
            # RoPE need (max_num_tokens,)
            self.positions = torch.zeros(
                self.max_num_tokens + 1, dtype=torch.int64, device=device
            )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        max_num_slots_for_arange = max(self.max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        # For padding counter used by MoE's EP + DP.
        num_draft_experts = getattr(
            self.draft_model_config, "n_routed_experts",
            getattr(self.draft_model_config, "num_experts", 0)
        )
        if (num_draft_experts > 0
            and envs.VLLM_MOE_USE_DEEPEP
            and vllm_config.parallel_config.enable_expert_parallel
            and vllm_config.parallel_config.expert_parallel_size > 1
        ):
            self.num_pad_tokens = self.runner._make_buffer(1, dtype=torch.int32)
        else:
            self.num_pad_tokens = None

        # Determine allowed attention backends once during initialization.
        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            rocm_types = [TritonAttentionMetadata, FlashAttentionMetadata]
            # ROCM_AITER_FA is an optional backend
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            from vllm.v1.attention.backends.mla.common import MLACommonMetadata

            rocm_types.append(MLACommonMetadata)

            self.allowed_attn_types = tuple(rocm_types)

        tree_drafter_params = self.speculative_config.tree_drafter_params
        self.use_tree_spec_decode = tree_drafter_params is not None
        if self.use_tree_spec_decode:
            self.cu_drafts_per_level = tree_drafter_params.cu_drafts_per_level
            self.child_drafts_per_level = (
                tree_drafter_params.child_drafts_per_level)
            # Precompute draft token positions in flattened tree.
            self.flattened_tree_positions = torch.arange(
                1,
                len(tree_drafter_params.draft_nodes) + 1,
                device=device,
                dtype=torch.int32,
            ).repeat(self.max_batch_size, 1)

        # Make mypy happy.
        self.model: CUDAGraphWrapper | nn.Module

    def _get_positions(self, num_tokens: int):
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        if self.uses_mrope:
            self.mrope_positions[:, :num_tokens] = positions
        else:
            self.positions[:num_tokens] = positions

    def _draft_pad_for_cudagraph(self, batch_size: int) -> int:
        # if batch_size > self.compilation_config.max_cudagraph_capture_size,
        # it should raise an IndexError.
        # the caller should make sure the batch_size is within the range,
        # i.e., batch_size <= self.compilation_config.max_cudagraph_capture_size
        if self.bs_to_padded_graph_size:
            return self.bs_to_padded_graph_size[batch_size]
        return self.vllm_config.compilation_config.bs_to_padded_graph_size[batch_size]

    @record_function_or_nullcontext("propose")
    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        is_prompt_batch: bool = False,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
            assert isinstance(
                getattr(self.model, "unwrap", lambda: self.model)(),
                (Eagle3LlamaForCausalLM, Eagle3Qwen3MoeForCausalLM)
            )
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids
        assert self.runner is not None

        if self.attn_metadata_builder is None:
            self.attn_metadata_builder = self._get_attention_metadata_builder()
        attn_metadata_builder = self.attn_metadata_builder

        # The first step is not uniform decode: can only be dispatched to
        # the decode_mode when decode cuda graph supports ALWAYS.
        uniform_decode = not is_prompt_batch and (
            common_attn_metadata.num_actual_tokens
            == common_attn_metadata.num_reqs * (1 + self.num_speculative_tokens)
        )
        cudagraph_runtime_mode = (
            self.cudagraph_mode.mixed_mode()
            if not uniform_decode
            else self.cudagraph_mode.decode_mode()
        )
        if isinstance(self.attn_metadata_builder, TreeAttentionMetadataBuilder):
            cudagraph_runtime_mode = CUDAGraphMode.NONE
        use_cuda_graph = (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and not self.enforce_eager
        )

        metadata_fast_build = cudagraph_runtime_mode != CUDAGraphMode.FULL

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, 
            draft_index=0, 
            fast_build=metadata_fast_build,
        )
        # FIXME: support hybrid kv for draft model (remove separate indexer)
        if self.draft_indexer_metadata_builder:
            draft_indexer_metadata = (
                self.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=0,
                    fast_build=metadata_fast_build,
                )
            )
        else:
            draft_indexer_metadata = None
        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        for layer_name in self.indexer_layer_names:
            assert draft_indexer_metadata is not None
            per_layer_attn_metadata[layer_name] = draft_indexer_metadata

        if use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            if envs.VLLM_TPEP_REDUCE_SCATTER and self.tp_size > 1 and self.ep_size > 1:
                num_input_tokens = round_up(num_tokens, self.tp_size)
            else:
                num_input_tokens = num_tokens

        # copy inputs to buffer for cudagraph
        self._set_positions(num_tokens, target_positions)
        self.hidden_states[:num_tokens] = target_hidden_states

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        if self.num_pad_tokens is not None:
            # Keep DP padding counter in CpuGpuBuffer and copy to device, for deepep.
            self.num_pad_tokens.cpu[0] = self.runner._get_num_pad_tokens(
                num_input_tokens, num_tokens
            )
            self.num_pad_tokens.copy_to_gpu(1)

        with (
            set_current_vllm_config(self.vllm_config),
            draft_use_dynamic_global_scale_for_fp4(),
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_pad_tokens=self.num_pad_tokens.gpu if self.num_pad_tokens else None,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor = BatchDescriptor(
                    num_tokens=num_input_tokens,
                    num_reqs=num_input_tokens // (1 + self.num_speculative_tokens),
                    uniform=True,
                ) if (
                    uniform_decode
                    and cudagraph_runtime_mode == CUDAGraphMode.FULL
                ) else None,
                is_prompt_batch=is_prompt_batch,
                skip_eplb=True,
                common_attn_metadata=common_attn_metadata,
            ),
            record_function_or_nullcontext(f"eagle-{num_input_tokens}")
        ):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=self._get_positions(num_input_tokens),
                hidden_states=self.hidden_states[:num_input_tokens],
                inputs_embeds=inputs_embeds,
            )
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_token_ids = logits.argmax(dim=-1)
            return draft_token_ids.view(-1, 1)

        if self.uses_mrope:
            positions = target_positions[:, last_token_indices]
        else:
            positions = target_positions[last_token_indices]
        if self.method in (
            "deepseek_mtp",
            "ernie_mtp",
            "longcat_flash_mtp",
            "pangu_ultra_moe_mtp",
        ):
            hidden_states = self.hidden_states[last_token_indices]
        else:
            hidden_states = hidden_states[last_token_indices]

        if self.use_tree_spec_decode:
            # Draft using tree attention.
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
            )
            # [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)

        draft_token_ids = logits.argmax(dim=-1)

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        # The subsequent draft steps are uniform decode.
        if self.eagle_use_qsa:
            cudagraph_runtime_mode = (
                CUDAGraphMode.NONE
                if not uniform_decode
                else self.cudagraph_mode.decode_mode()
            )
        else:
            cudagraph_runtime_mode = self.cudagraph_mode.decode_mode()
        use_cuda_graph = (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and not self.enforce_eager
        )
        metadata_fast_build = cudagraph_runtime_mode != CUDAGraphMode.FULL

        if use_cuda_graph and batch_size <= (
            self.cudagraph_batch_sizes[-1] // (1 + self.num_speculative_tokens)
        ):
            input_batch_size = self._draft_pad_for_cudagraph(batch_size)
        else:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            if envs.VLLM_TPEP_REDUCE_SCATTER and self.tp_size > 1 and self.ep_size > 1:
                input_batch_size = round_up(batch_size, self.tp_size)
            else:
                input_batch_size = batch_size

        if self.num_pad_tokens is not None:
            # Keep DP padding counter in CpuGpuBuffer and copy to device, for deepep.
            self.num_pad_tokens.cpu[0] = self.runner._get_num_pad_tokens(
                input_batch_size, batch_size
            )
            self.num_pad_tokens.copy_to_gpu(1)

        common_attn_metadata.num_reqs = batch_size
        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc[: batch_size + 1].copy_(
            self.arange[: batch_size + 1]
        )
        common_attn_metadata.query_start_loc[batch_size + 1 : ].fill_(batch_size + 1)
        common_attn_metadata.query_start_loc = (
            common_attn_metadata.query_start_loc[: batch_size + 1]
        )
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        )
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            if self.uses_mrope:
                positions += 1
                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length.
                # Since it is complex to remove such requests from the batch,
                # we keep them in the batch but adjust the position ids
                # and slot mappings to avoid the
                # out-of-range access during the model execution.
                # The draft tokens generated with this adjustment
                # should be ignored.
                exceeds_max_model_len = positions[0] >= self.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0),
                    torch.zeros_like(positions),
                    positions,
                )
            else:
                positions += 1
                exceeds_max_model_len = positions >= self.max_model_len
                clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
            # For data integrity when async scheduling, we shouldn't use in place
            # operations in case they are modified in next step's `prepare_input`
            # of main model.
            # Increment the sequence lengths.
            common_attn_metadata.seq_lens += 1
            common_attn_metadata.max_seq_len += 1
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Also update the CPU-side shadow; NOTE: this is hacky and should be
            # removed in when common_attn_metadata.seq_lens_cpu is deprecated.
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu += 1
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu += 1

            # Compute the slot mapping.
            if self.uses_mrope:
                # all dimensions of positions are the same
                block_numbers = clamped_positions[0] // self.block_size
            else:
                block_numbers = clamped_positions // self.block_size
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)
            )
            block_ids = block_ids.view(-1)
            if self.uses_mrope:
                draft_slot_mapping = (
                    block_ids * self.block_size + clamped_positions[0] % self.block_size
                )
            else:
                draft_slot_mapping = (
                    block_ids * self.block_size + clamped_positions % self.block_size
                )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            draft_slot_mapping.masked_fill_(
                exceeds_max_model_len, PADDING_SLOT_ID
            )
            common_attn_metadata.slot_mapping[
                : draft_slot_mapping.shape[0]
            ].copy_(draft_slot_mapping)
            common_attn_metadata.slot_mapping[
                draft_slot_mapping.shape[0] :
            ].fill_(PADDING_SLOT_ID)

            # Rebuild attention metadata
            attn_metadata = attn_metadata_builder.build_for_drafting(  # type: ignore
                common_attn_metadata=common_attn_metadata,
                draft_index=token_index + 1,
                fast_build=metadata_fast_build,
            )
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            if self.draft_indexer_metadata_builder:
                draft_indexer_metadata = (
                    self.draft_indexer_metadata_builder.build_for_drafting(
                        common_attn_metadata=common_attn_metadata,
                        draft_index=token_index + 1,
                        fast_build=metadata_fast_build,
                    )
                )
            else:
                draft_indexer_metadata = None
            for layer_name in self.indexer_layer_names:
                assert draft_indexer_metadata is not None
                per_layer_attn_metadata[layer_name] = draft_indexer_metadata

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None
            
            # Run the model.
            with (
                set_current_vllm_config(self.vllm_config),
                draft_use_dynamic_global_scale_for_fp4(),
                set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=input_batch_size,
                    num_pad_tokens=(
                        self.num_pad_tokens.gpu if self.num_pad_tokens else None
                    ),
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor = BatchDescriptor(
                        num_tokens=input_batch_size,
                        num_reqs=input_batch_size,
                        uniform=True,
                    ) if cudagraph_runtime_mode == CUDAGraphMode.FULL else None,
                    is_prompt_batch=False,  # the subsequent draft steps are decoding
                    skip_eplb=True,
                    common_attn_metadata=common_attn_metadata,
                ),
                record_function_or_nullcontext(f"eagle-{input_batch_size}")
            ):
                ret_hidden_states = self.model(
                    input_ids=input_ids,
                    positions=self._get_positions(input_batch_size),
                    hidden_states=self.hidden_states[:input_batch_size],
                    inputs_embeds=inputs_embeds,
                )
                if self.method == "mtp":
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states
            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size])
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[req_id]
                next_token_id = req_state.get_token_id(
                    seq_len,
                    vocab_size=gpu_input_batch.vocab_size,
                )
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.int32, device=self.input_ids.device
        )
        return next_token_ids

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead. This is denoted
        the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.
        """
        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens_cpu[i].item(),
                    vocab_size=gpu_input_batch.vocab_size,
                )
                for i in range(num_reqs)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        next_token_ids = torch.empty((batch_size,), dtype=torch.int32, device=device)
        valid_sampled_tokens_count = torch.empty(
            (batch_size,), dtype=torch.int32, device=device
        )

        # Kernel grid: one program per request (row)
        grid = (batch_size,)

        # Find the next power of 2 for block sizes
        BLOCK_SIZE_TOKENS = triton.next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        # Kernel grid: one program per request (row)
        grid = (num_reqs,)
        eagle_prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_reqs,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()
        token_indices = self.arange[:total_num_tokens]

        new_slot_mapping = common_attn_metadata.slot_mapping[token_indices]  # copying.
        common_attn_metadata.slot_mapping[:new_slot_mapping.shape[0]].copy_(new_slot_mapping)
        common_attn_metadata.slot_mapping[new_slot_mapping.shape[0]:].fill_(PADDING_SLOT_ID)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            max_orig_seq_len=common_attn_metadata.max_orig_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices_to_sample

    def propose_tree(
        self,
        batch_size: int,
        # [num_tokens, vocab_size]
        logits: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        # [num_tokens, hidden_size]
        hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> list[torch.Tensor]:
        tree_attn_metadata_builder = self.runner.attn_groups[-1][
            0
        ].get_metadata_builder()
        assert isinstance(tree_attn_metadata_builder, TreeAttentionMetadataBuilder)

        total_num_drafts = self.cu_drafts_per_level[0]
        # Sample a draft token for each child at the tree root level.
        num_children = self.child_drafts_per_level[0]
        if num_children == 1:
            draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
        else:
            draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                batch_size, -1
            )
        draft_token_ids_list = [draft_token_ids]
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)

        # Initialize empty tensors for concatenation with the level outputs.
        tree_input_ids = torch.empty(
            0, device=self.input_ids.device, dtype=self.input_ids.dtype
        )
        tree_positions = torch.empty(
            0, device=self.positions.device, dtype=self.positions.dtype
        )
        tree_hidden_states = torch.empty(
            0, device=self.hidden_states.device, dtype=self.hidden_states.dtype
        )
        # Precompute the draft token query positions.
        positions_base = positions.view(batch_size, -1)
        flattened_query_positions = (
            positions_base + self.flattened_tree_positions[:batch_size, :]
        )
        tree_depth = len(self.cu_drafts_per_level)
        for level in range(1, tree_depth - 1):
            # Update the # drafts counters for the current level.
            level_num_drafts = self.cu_drafts_per_level[
                level] - total_num_drafts
            total_num_drafts = self.cu_drafts_per_level[level]

            # Get draft positions for RoPE.
            draft_positions = positions + level
            exceeds_max_model_len = (positions + total_num_drafts) >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            draft_positions = torch.where(
                exceeds_max_model_len,
                0,
                draft_positions,
            ).view(batch_size, -1)

            if level_num_drafts > 1:
                # Repeat the positions for each draft at this level.
                draft_positions = draft_positions.repeat_interleave(
                    level_num_drafts, dim=1
                )

            if num_children > 1:
                # Repeat draft hidden states for each child.
                draft_hidden_states = draft_hidden_states.repeat_interleave(
                    num_children, dim=1
                )

            # Concatenate the draft tokens, positions, and hidden states.
            tree_input_ids = torch.cat([tree_input_ids, draft_token_ids], dim=1)
            tree_positions = torch.cat([tree_positions, draft_positions], dim=1)
            tree_hidden_states = torch.cat(
                [tree_hidden_states, draft_hidden_states], dim=1
            )

            # Build new attention metadata for the next level of drafts.
            # This is necessary to support tree attention.
            query_len = total_num_drafts
            common_attn_metadata = replace(
                common_attn_metadata,
                query_start_loc=query_len * self.arange[: batch_size + 1],
                seq_lens=common_attn_metadata.seq_lens + level_num_drafts,
                num_actual_tokens=batch_size * query_len,
                max_query_len=query_len,
            )
            attn_metadata = tree_attn_metadata_builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata,
                draft_index=level,
            )

            # Apply new attention metadata to all layers.
            per_layer_attn_metadata = {}
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            # Consider max model length.
            attn_metadata.max_seq_len = min(
                attn_metadata.max_seq_len, self.max_model_len
            )
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            query_positions = flattened_query_positions[:, :query_len]
            block_numbers = query_positions // self.block_size
            block_ids = attn_metadata.block_table.gather(dim=1, index=block_numbers)
            slot_mapping = (
                block_ids * self.block_size + query_positions % self.block_size
            )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping[exceeds_max_model_len] = PADDING_SLOT_ID
            attn_metadata.slot_mapping = slot_mapping.view(-1)

            # Copy inputs to buffer for cudagraph.
            num_tokens = attn_metadata.num_actual_tokens
            self.input_ids[:num_tokens] = tree_input_ids.view(-1)
            self.positions[:num_tokens] = tree_positions.view(-1)
            self.hidden_states[:num_tokens] = tree_hidden_states.view(num_tokens, -1)

            if (
                self.use_cuda_graph
                and num_tokens <= self.compilation_config.max_cudagraph_capture_size
            ):
                num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
                cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            else:
                num_input_tokens = num_tokens
                cudagraph_runtime_mode = CUDAGraphMode.NONE
            # Run the model.
            with (
                set_current_vllm_config(self.vllm_config),
                set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    skip_eplb=True,
                ),
            ):
                last_hidden_states_hidden_states = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=None,
                )
                if self.method == "mtp":
                    last_hidden_states = last_hidden_states_hidden_states
                    hidden_states = last_hidden_states_hidden_states
                else:
                    last_hidden_states, hidden_states = last_hidden_states_hidden_states
            # Get the output hidden states for the draft tokens.
            draft_hidden_states = hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]
            draft_last_hidden_states = last_hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]

            # Get the output logits for the draft tokens.
            logits = self.model.compute_logits(
                draft_last_hidden_states.reshape(batch_size * level_num_drafts, -1)
            )

            # Sample a draft token for each child at the next tree level.
            num_children = self.child_drafts_per_level[level]
            if num_children == 1:
                draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
            else:
                draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                    batch_size, -1
                )
            draft_token_ids_list.append(draft_token_ids)

        return draft_token_ids_list

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
        sampled_token_indices: list[list[int]] | None = None,
        kv_caches: list[torch.Tensor] | None = None,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]
        if sampled_token_indices is not None:
            sampled_token_ids_len = sampled_token_indices
        else:
            sampled_token_ids_len = sampled_token_ids
        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids_len[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = common_attn_metadata.seq_lens_cpu - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offests = (
            self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        )

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offests + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=new_seq_lens_cpu.max().item(),
            max_orig_seq_len=common_attn_metadata.max_orig_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )
        if self.use_tree_spec_decode:
            # During tree spec decoding, the accepted draft tokens may not be
            # consecutive. In such cases, we must recompute the token indices,
            # and update the KV cache slots.

            # Compute the accepted token indices.
            apply_accepted_draft_indices(sampled_token_indices,
                                         new_query_start_loc_np,
                                         token_indices_np)
            
            accepted_token_indices = torch.from_numpy(token_indices_np).to(
                device, non_blocking=True)
            if self.vllm_config.model_config.is_hybrid:
                kvs = self.vllm_config.model_config.hf_config.full_attention_interval
            else:
                kvs = 1
            assert kv_caches is not None
            copy_kv_cache_slots(
                (kv_caches[kvs - 1 :: kvs]),
                common_attn_metadata.slot_mapping,
                accepted_token_indices,
                token_indices,
            )
            return spec_common_attn_metadata, accepted_token_indices


        return spec_common_attn_metadata, token_indices

    def get_model_name(self, model: nn.Module) -> str:
        if hasattr(model, "module"):  # multi-GPU
            model = model.module
        return model.__class__.__name__

    def load_model(
        self,
        target_model: nn.Module,
        tensor_queue_union: Union[None, Any, List[Any]] = None,
    ) -> None:
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )
        # FIXME: support hybrid kv for draft model
        target_indexer_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config, (DeepseekV32IndexerCache, Qwen3NextIndexerCache)
            ).keys()
        )

        from vllm.compilation.backends import set_model_tag

        with (
            set_model_tag("eagle_head"),
            draft_use_dynamic_global_scale_for_fp4(),
        ):
            if envs.VLLM_USE_SWAP_ENGINE:
                from vllm.distributed import get_tensor_model_parallel_rank
                from vllm.model_executor.model_loader.quick_model_loader_utils import (
                    get_driver_handle_queue,
                )

                rank = get_tensor_model_parallel_rank()
                driver_tensor_queue_union = get_driver_handle_queue()

                if rank == 0 and isinstance(tensor_queue_union, List):
                    tensor_queue_union[len(tensor_queue_union) - 1] = (
                        driver_tensor_queue_union
                    )

            tracker = get_model_loading_tracker()
            if tracker is not None:
                tracker.set_layer_offset_for_draft_model(
                    self.vllm_config.model_config.get_num_layers(
                        self.vllm_config.parallel_config
                    )
                )

            self.model = get_model(
                vllm_config=self.vllm_config,
                model_config=draft_model_config,
                tensor_queue_union=tensor_queue_union,
                enable_spec_decoding=True,
            )

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
            - target_attn_layer_names
        )
        indexer_layers = get_layers_from_vllm_config(
            self.vllm_config, (DeepseekV32IndexerCache, Qwen3NextIndexerCache)
        )
        draft_indexer_layer_names = indexer_layers.keys() - target_indexer_layer_names
        self.attn_layer_names = list(draft_attn_layer_names - draft_indexer_layer_names)
        self.indexer_layer_names = list(draft_indexer_layer_names)

        if self.indexer_layer_names:
            first_layer = self.indexer_layer_names[0]
            self.draft_indexer_metadata_builder = (
                indexer_layers[first_layer]
                .get_attn_backend()
                .get_builder_cls()(
                    indexer_layers[first_layer].get_kv_cache_spec(self.vllm_config),
                    self.indexer_layer_names,
                    self.vllm_config,
                    self.device,
                )
            )
        else:
            self.draft_indexer_metadata_builder = None

        if self.supports_mm_inputs:
            # Even if the target model is multimodal, we can also use
            # text-only draft models
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Draft model does not support multimodal inputs, "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        if supports_multimodal(target_model):
            # handle multimodality
            if self.get_model_name(target_model) in [
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "Qwen3NextVLForConditionalGeneration",
                "Qwen3NextVLForConditionalGenerationGating",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
            target_language_model = target_model.get_language_model()
        else:
            target_language_model = target_model

        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1:
            if hasattr(target_language_model.model, "embed_tokens"):
                target_embed_tokens = target_language_model.model.embed_tokens
            elif hasattr(target_language_model.model, "embedding"):
                target_embed_tokens = target_language_model.model.embedding
            else:
                raise AttributeError(
                    "Target model does not have 'embed_tokens' or 'embedding' attribute"
                )

            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: Offload to CPU for comparison to avoid extra GPU memory
                    # usage in CI testing environments with limited GPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

        # share lm_head with the target model if needed
        share_lm_head = False
        if hasattr(self.model, "has_own_lm_head"):
            # EAGLE model
            if not self.model.has_own_lm_head:
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model without its own lm_head in the checkpoint. "
                    "Sharing target model lm_head weights with the draft model."
                )
            elif (
                hasattr(target_language_model, "lm_head")
                and isinstance(target_language_model.lm_head.weight, torch.Tensor)
                and isinstance(self.model.lm_head.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_language_model.lm_head.weight.cpu(),
                    self.model.lm_head.weight.cpu(),
                )
            ):
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model with lm_head identical to the target model. "
                    "Sharing target model lm_head weights with the draft model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct lm_head weights. "
                    "Keeping separate lm_head weights from the target model."
                )
        else:
            # MTP/EAGLE model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )

        if share_lm_head and hasattr(target_language_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head

            # MTP models call compute_logits via shared_head.head (a
            # ParallelLMHead inside each MTP layer), not self.model.lm_head.
            # If the checkpoint omits a copy of the lm_head weights at the
            # MTP layer path, shared_head.head stays uninitialised and
            # produces NaN logits. Always share it explicitly.
            inner = getattr(self.model, "model", None)
            layers = getattr(inner, "layers", None) if inner else None
            if layers is not None:
                items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
                for layer in items:
                    sh = getattr(layer, "shared_head", None)
                    if sh is not None and hasattr(sh, "head"):
                        del sh.head
                        sh.head = target_language_model.lm_head
                        logger.info(
                            "Shared target model lm_head with MTP shared_head.head."
                        )

        if not self.enforce_eager and self.cudagraph_mode.has_full_cudagraphs():
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        inputs_embeds: Optional[torch.Tensor] = None,
        common_attn_metadata: Optional[CommonAttentionMetadata] = None,
        attn_metadata: Optional[dict[str, Any]] = None,
        is_profile: bool = False,
        is_prompt_batch: bool = False,
        uniform_decode: bool = False,
        is_graph_capturing: bool = False,
    ) -> None:
        # Determine if CUDA graphs should be used for this run.
        if use_cudagraphs and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)

        # For decode batches, we'll capture two kinds of cuda graphs:
        # 1) multi-query decode: for the first step, align with the behavior when
        # padded disable_padded_drafter_batch is False (by default).
        # 2) single-query decode: for the subsequent steps of draft model forward.
        for fwd_idx in range(
            self.num_speculative_tokens if not is_profile else 1
        ):
            # Note that the logic is not consistent with the "propose()" method:
            #
            # The first step of **dummy run** of decoding batch is also uniform decode:
            # can be dispatched to the decode_mode as well.
            step_cudagraph_runtime_mode = (self.cudagraph_mode.mixed_mode()
                if is_prompt_batch
                else self.cudagraph_mode.decode_mode()
            ) if use_cudagraphs else CUDAGraphMode.NONE

            # The attention metadata can only skip rebuild when piecewise cudagraphs
            # is on, otherwise:
            #
            # 1. it can be FULL when full cuda graph is on;
            # 2. it can be NONE for warmup steps when full cuda graph is on.
            metadata_fast_build = step_cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE

            if not is_prompt_batch and uniform_decode and fwd_idx > 0:
                step_num_tokens = num_tokens // (self.num_speculative_tokens + 1)
            else:
                step_num_tokens = num_tokens
            num_uniform_decode_seqs = num_tokens // (1 + self.num_speculative_tokens)
            batch_descriptor = BatchDescriptor(
                num_tokens=step_num_tokens,
                num_reqs=num_uniform_decode_seqs,
                uniform=True,
            ) if (
                not is_prompt_batch
                and uniform_decode
            ) else None

            # Rebuild attention metadata for full attention of the subsequent steps
            if (not is_prompt_batch
                and uniform_decode
                and attn_metadata is not None
                and common_attn_metadata is not None
                and not metadata_fast_build
            ):
                common_attn_metadata.max_query_len = (
                    (1 + self.num_speculative_tokens) if fwd_idx == 0 else 1
                )
                common_attn_metadata.num_actual_tokens = step_num_tokens
                if fwd_idx > 0:
                    common_attn_metadata.query_start_loc[
                        : num_uniform_decode_seqs + 1
                    ].copy_(self.arange[: num_uniform_decode_seqs + 1])
                common_attn_metadata.query_start_loc[
                    num_uniform_decode_seqs + 1 :
                ].fill_(step_num_tokens + 1)
                common_attn_metadata.query_start_loc = (
                    common_attn_metadata.query_start_loc[: num_uniform_decode_seqs + 1]
                )
                if fwd_idx > 0:
                    common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
                        self.token_arange_np[: num_uniform_decode_seqs + 1]
                    )
                common_attn_metadata.slot_mapping[
                    step_num_tokens :
                ].fill_(PADDING_SLOT_ID)

                if self.attn_metadata_builder is None:
                    self.attn_metadata_builder = self._get_attention_metadata_builder()
                attn_metadata_builder = self.attn_metadata_builder

                # Rebuild attention metadata
                draft_attn_metadata = attn_metadata_builder.build_for_drafting(  # type: ignore
                    common_attn_metadata=common_attn_metadata,
                    draft_index=fwd_idx,
                    fast_build=metadata_fast_build,
                )
                for layer_name in self.attn_layer_names:
                    attn_metadata[layer_name] = draft_attn_metadata

                if self.draft_indexer_metadata_builder:
                    draft_indexer_metadata = (
                        self.draft_indexer_metadata_builder.build_for_drafting(
                            common_attn_metadata=common_attn_metadata,
                            draft_index=fwd_idx,
                            fast_build=metadata_fast_build,
                        )
                    )
                else:
                    draft_indexer_metadata = None
                for layer_name in self.indexer_layer_names:
                    assert draft_indexer_metadata is not None
                    attn_metadata[layer_name] = draft_indexer_metadata

            # TreeAttention: the first draft step is a prefill step and
            # doesn't support cudagraph
            if (fwd_idx == 0
                and isinstance(self.attn_metadata_builder, TreeAttentionMetadataBuilder)
            ):
                step_cudagraph_runtime_mode = CUDAGraphMode.NONE

            with (
                set_current_vllm_config(self.vllm_config),
                draft_use_dynamic_global_scale_for_fp4(),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=step_num_tokens,
                    cudagraph_runtime_mode=step_cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    # Subsequent runs are not prompt batches.
                    is_prompt_batch=is_prompt_batch and fwd_idx == 0,
                    skip_eplb=True,
                ), (
                    # Workaround to avoid IMA during capturing cuda graph
                    # with draft model.
                    get_tp_group().temporarily_disable_custom_all_reduce()
                    if step_cudagraph_runtime_mode != CUDAGraphMode.NONE
                    else contextlib.nullcontext()
                )
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:step_num_tokens]
                else:
                    input_ids = self.input_ids[:step_num_tokens]
                    inputs_embeds = None
                self.model(
                    input_ids=input_ids,
                    positions=self._get_positions(step_num_tokens),
                    hidden_states=self.hidden_states[:step_num_tokens],
                    inputs_embeds=inputs_embeds,
                )

    def _get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        """Find and return the attention metadata builders for EAGLE layers.

        Returns:
            The metadata builders for EAGLE layers.

        Raises:
            AssertionError: If no metadata builders are found for EAGLE layers.
        """
        builder = None
        attn_layers = self.attn_layer_names
        chosen_layer = attn_layers[0]

        for kv_cache_group in self.runner.attn_groups:
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    builder = attn_group.get_metadata_builder()
                    break
            if builder is not None:
                break

        assert builder is not None, (
            "Failed to find attention metadata builder for EAGLE layers."
        )
        # Revise the block_size to the actual kernel block size as it will
        # be used to compute the slot mapping.
        self.block_size = getattr(builder, "block_size", self.block_size)

        chosen_attn_layer: AttentionLayerBase = get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase
        )[chosen_layer]

        self.cg_support = builder.get_cudagraph_support(
            self.vllm_config,
            chosen_attn_layer.get_kv_cache_spec(
                self.vllm_config
            ).copy_with_new_block_size(self.block_size)
        )
        if self.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_batch_sizes = [
                batch_size // (1 + self.num_speculative_tokens)
                for batch_size in self.cudagraph_batch_sizes
            ]
            max_cudagraph_capture_size = max(cudagraph_batch_sizes)
            # pre-compute the mapping from batch size to padded graph size
            self.bs_to_padded_graph_size = [
                0 for i in range(max_cudagraph_capture_size + 1)
            ]
            for end, start in zip(
                cudagraph_batch_sizes + [max_cudagraph_capture_size + 1],
                [0] + cudagraph_batch_sizes,
            ):
                for bs in range(start, end):
                    if bs == start:
                        self.bs_to_padded_graph_size[bs] = start
                    else:
                        self.bs_to_padded_graph_size[bs] = end

        return builder

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """
        Some eagle3 heads (e.g., nvidia/gpt-oss-120b-Eagle3-v2) do not use auxiliary
        hidden states and directly uses the last layer output just like eagle1.
        They might indicate this by setting "use_aux_hidden_state" to False
        inside the "eagle_config" dict of their hf_config.
        """
        if self.method != "eagle3":
            return False
        # Assume that eagle3 heads use aux hidden states by default
        use_aux_hidden_state = True
        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is not None:
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all eagle layers belong to the same KVCacheGroup.
        Need this assumption to ensure all eagle layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self.attn_layer_names
                    ]
                )
            )
            == 1
        ), "All eagle layers should belong to the same kv cache group"

    def _pad_batch_across_dp(
        self,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> tuple[int, torch.Tensor]:
        # TODO(Flechman): support DBO ubatching
        should_ubatch, num_toks_across_dp, _ = coordinate_batch_across_dp(
            num_tokens_unpadded=num_tokens_unpadded,
            parallel_config=self.vllm_config.parallel_config,
            allow_microbatching=False,
            allow_dp_padding=self.use_cuda_graph,
            num_tokens_padded=num_tokens_padded,
            uniform_decode=None,
            num_scheduled_tokens_per_request=None,
        )
        assert not should_ubatch, "DBO ubatching not implemented for EAGLE"

        num_tokens_dp_padded = num_tokens_padded
        if num_toks_across_dp is not None:
            num_tokens_dp_padded = int(num_toks_across_dp[self.dp_rank].item())
        return num_tokens_dp_padded, num_toks_across_dp


# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    assert sampling_metadata.temperature is not None

    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)
    # consistent with sampler.py's _SAMPLING_EPS threshold
    temperature = sampling_metadata.temperature
    # Avoid division by zero if there are greedy requests.
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = torch.empty_like(probs)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    next_token_ids = probs.div(q).argmax(dim=-1).view(-1)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(
            is_greedy,
            greedy_token_ids,
            next_token_ids,
        )
    return next_token_ids, probs


@contextlib.contextmanager
def draft_use_dynamic_global_scale_for_fp4():
    nvfp4_dynamic_global_scale = envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4
    draft_nvfp4_dynamic_global_scale = (
        envs.VLLM_DRAFT_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4
    )
    use_nvfp4_dispatch = envs.VLLM_DEEPEPLL_NVFP4_DISPATCH
    try:
        if not nvfp4_dynamic_global_scale and draft_nvfp4_dynamic_global_scale:
            envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4 = (
                draft_nvfp4_dynamic_global_scale
            )
            logger.info_once(
                "Enabled VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4 for draft model "
                "as VLLM_DRAFT_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4 is set to "
                f"{draft_nvfp4_dynamic_global_scale}."
            )
            if use_nvfp4_dispatch:
                logger.info_once(
                    "Disable VLLM_DEEPEPLL_NVFP4_DISPATCH for draft model "
                    f"as VLLM_DRAFT_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4 is set to "
                    f"{draft_nvfp4_dynamic_global_scale}."
                )
                envs.VLLM_DEEPEPLL_NVFP4_DISPATCH = False
        yield
    finally:
        envs.VLLM_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4 = nvfp4_dynamic_global_scale
        envs.VLLM_DEEPEPLL_NVFP4_DISPATCH = use_nvfp4_dispatch
