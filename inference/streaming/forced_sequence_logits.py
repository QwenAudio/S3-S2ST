"""Model-level vLLM logits processor for exact per-request token replay."""

from __future__ import annotations

from typing import Any

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


class _ForceSequencePerRequest:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = tuple(int(token_id) for token_id in token_ids)

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        position = len(output_ids)
        if position >= len(self.token_ids):
            raise RuntimeError(
                f"forced sequence exhausted at decode position {position}"
            )
        target = self.token_ids[position]
        keep = logits[target].clone()
        logits.fill_(float("-inf"))
        logits[target] = keep
        return logits


class ForcedSequenceLogitsProcessor(AdapterLogitsProcessor):
    """Force ``SamplingParams.extra_args['forced_token_ids']`` exactly."""

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        value: Any = params.extra_args and params.extra_args.get("forced_token_ids")
        if value is None:
            return
        if not isinstance(value, list) or not value:
            raise ValueError("forced_token_ids must be a non-empty list[int]")
        if any(not isinstance(token_id, int) for token_id in value):
            raise ValueError("forced_token_ids must contain only integers")

    def is_argmax_invariant(self) -> bool:
        return False

    def is_sps_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> RequestLogitsProcessor | None:
        self.validate_params(params)
        value = params.extra_args and params.extra_args.get("forced_token_ids")
        if value is None:
            return None
        return _ForceSequencePerRequest(value)
