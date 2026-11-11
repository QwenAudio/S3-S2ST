# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Optional

import torch
from typing_extensions import deprecated

from vllm._bc_linter import bc_linter_include

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import ExternalData
    from vllm.v1.request import Request
    from vllm.v1.worker.ubatch_utils import UBatchSlices
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object
    UBatchSlices = object
    ExternalData = object


@bc_linter_include
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    output_token_ids: list[int]
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None
    prompt_embeds_shm_key: str | None = None
    chunk_ranks: list[int] | None = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        chunk_ranks: list[int] = []
        if request.chunk_ranks and len(request.chunk_ranks) > 1:
            chunk_ranks = request.chunk_ranks
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=request.output_token_ids[:],
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prompt_embeds_shm_key=request.prompt_embeds_shm_key,
            prefill_token_ids=prefill_token_ids,
            chunk_ranks=chunk_ranks,
        )

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape},"
            f"chunk_ranks={self.chunk_ranks},"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape},"
            f"chunk_ranks={self.chunk_ranks},"
            ")"
        )


@bc_linter_include
@dataclass
class CachedRequestData:
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    # For requests not scheduled in the last step, propagate the token ids to the
    # connector. Won't contain requests that were scheduled in the prior step.
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]
    total_kv_lens: int = 0
    chunk_ranks: list[int] | None = None
    num_prefill_tokens: list[int] | None = None

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    @deprecated("This will be removed in v0.14, use `resumed_req_ids` instead.")
    def resumed_from_preemption(self) -> list[bool]:
        return [req_id in self.resumed_req_ids for req_id in self.req_ids]

    @cached_property
    @deprecated("This will be removed in v0.14, use `all_token_ids` instead.")
    def resumed_req_token_ids(self) -> list[list[int] | None]:
        return [
            self.all_token_ids[req_id] if req_id in self.resumed_req_ids else None
            for req_id in self.req_ids
        ]

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
            total_kv_lens=0,
            chunk_ranks=[],
            num_prefill_tokens=[],
        )


@bc_linter_include
@dataclass
class ScheduledDPMetaData:
    num_pad_tokens: int = 0
    # The number of tokens that will be added to the batch.
    num_tokens_across_dp: Optional[torch.Tensor] = None
    # A tensor containing the total number of
    # tokens per-microbatch for each DP rank including padding. Will be
    # padded up to the max value across all DP ranks when allow_dp_padding
    # is True.
    num_tokens_per_rank: Optional[torch.Tensor] = None
    # Num tokens per rank without padding.
    num_reqs_per_rank: Optional[torch.Tensor] = None
    # Num reqs per rank without padding.
    is_prompt_batch: bool | None = None

    # True if any request in the batch is a prompt batch.
    all_prefill: bool | None = None
    # True if all requests in the batch are prefill-only.
    all_decode: bool | None = None
    # True if all requests in the batch are decode-only.
    should_ubatch: bool = False
    # Microbatch slices if this is set then all DP ranks have agreed to microbatch.
    input_fits_in_drafter: bool = False
    # Whether the input fits in drafter for all DP ranks.
    cudagraph_mode: int = 0  # 0=NONE, 1=PIECEWISE, 2=FULL


@bc_linter_include
@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # req_id -> seq_lens
    # Sequence length of each scheduled request.
    scheduled_seq_lens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are scheduled for the first time.
    scheduled_new_req_ids: set[str]
    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]
    # [num_reqs]
    # The number of tokens generated in the current step.
    kv_lens : list[int]
    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    preempted_req_ids: set[str] | None = None

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # KV Cache Connector metadata.
    kv_connector_metadata: KVConnectorMetadata | None = None

    # EC Cache Connector metadata
    ec_connector_metadata: ECConnectorMetadata | None = None

    # dp related fields
    scheduled_dp_metadata: Optional[ScheduledDPMetaData] = None

    # For Engine to Worker tracing
    trace_carrier: Optional[Mapping[str, str]] = None  # EngineCore to Worker tracing

    # context parallel related fields
    cp_world_size: Optional[int] = 0
    cp_ranks: Optional[list[int]] = None
    cp_rank_idx: Optional[int] = None
    num_prefill_tokens: Optional[list[int]] = None
    # External tensors for decode phase - req_id -> tensor
    # This allows passing external tensors for each request in decode phase
    external_data: dict[str, ExternalData] | None = None

    def __post_init__(self):
        if self.external_data is None:
            self.external_data = {}

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            scheduled_seq_lens={},
            total_num_scheduled_tokens=0,
            kv_lens=[],
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            scheduled_new_req_ids=set(),
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            cp_world_size=1,
            cp_ranks=None,
            cp_rank_idx=None,
            num_prefill_tokens=None,
        )


@dataclass
class GrammarOutput:
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Map of req_id to the offset of the first rejected speculative token.
    # All tokens starting from this offset are invalid (None).
    rejected_structured_output_start_offsets: dict[str, int]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"
