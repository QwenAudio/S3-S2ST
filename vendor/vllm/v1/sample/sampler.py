# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that samples the next tokens from the model's outputs."""

import json
import os

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

_SAMPLING_EPS = 1e-5
logger = init_logger(__name__)


class Sampler(nn.Module):
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:

    1. If logprobs are requested:
        a) If `logprobs_mode` is `raw_logprobs`, compute logprobs
           as the final logprobs to return.
        b) If `logprobs_mode` is `raw_logits`, clone the logits
           as the final logprobs to return.
    2. Convert logits to float32.
    3. Apply allowed token ids whitelist.
    4. Apply bad words exclusion.
    5. Apply logit processors which are not argmax-invariant,
       i.e. that can impact greedy sampling.
        a) Min tokens processor
        b) Logit bias processor
    6. Apply penalties
        a) Repetition penalty
        b) Frequency penalty
        c) Presence penalty
    7. Sample the next tokens. `sample` method performs the following steps:
        a) If not `all_random`, perform greedy sampling. If `all_greedy`,
           return the greedily sampled tokens and final logprobs if requested.
        b) Apply temperature.
        c) Apply logit processors which are argmax-invariant, by default
           the min_p processor.
        d) Apply top_k and/or top_p.
        e) Sample the next tokens with the probability distribution.
        f) If `all_random` or temperature >= epsilon (1e-5), return the
           randomly sampled tokens and final logprobs if requested. Else,
           return the greedily sampled tokens and logprobs if requested.
    8. Gather the logprobs of the top `max_num_logprobs` and sampled token
       (if requested). Note that if the sampled token is within the top
       `max_num_logprobs`, the logprob will be eventually merged in
       `LogprobsProcessor` during output processing. Therefore, the
       final output may contain either `max_num_logprobs + 1` or
       `max_num_logprobs` logprobs.
    9. Return the final `SamplerOutput`.
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode)
        self.pin_memory = is_pin_memory_available()
        self.logprobs_mode = logprobs_mode

        self.output_tokens_t = None
        self._codex_trace_counter = 0
        forced_prefix = os.environ.get("CODEX_FORCE_PREFIX_JSON", "").strip()
        self._codex_forced_prefix = (
            [int(token_id) for token_id in json.loads(forced_prefix)]
            if forced_prefix
            else []
        )
        self._codex_force_pos = 0

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
        spec_decode_metadata: SpecDecodeMetadata | None = None,
    ) -> SamplerOutput:
        if envs.VLLM_ENABLE_CODE_SWITCH_GATE:
            [logits, lang_logits] = logits

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        num_logprobs = sampling_metadata.max_num_logprobs

        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )

        # Compute raw logprobs after applying logits processors, like penalties.
        if num_logprobs is not None:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        # Sample the next token.
        if envs.VLLM_ENABLE_CODE_SWITCH_GATE:
            sampled, processed_logprobs = self.sample(
                [logits, lang_logits],
                sampling_metadata,
                spec_decode_metadata=spec_decode_metadata
            )
        else:
            sampled, processed_logprobs = self.sample(
                logits,
                sampling_metadata,
            )
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        if num_logprobs is None:
            logprobs_tensors = None
        elif num_logprobs == -1:
            # Return the full unsorted and unranked logprobs.
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            # Gather the logprobs and ranks of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled,
                top_k=sampling_metadata.top_k
                if envs.VLLM_ENABLE_CODE_SWITCH_GATE else None,
            )

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        # Avoid division by zero if there are greedy requests.
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
        spec_decode_metadata: SpecDecodeMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """
        if envs.VLLM_ENABLE_CODE_SWITCH_GATE:
            [logits, lang_logits] = logits

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        greedy_eos_sampled = None
        codec_trace = None
        if sampling_metadata.greedy_eos_token_ids is not None:
            greedy_eos_sampled = self.greedy_sample(logits)
            if os.environ.get("CODEX_LOGIT_TRACE", "0").lower() in (
                "1", "true", "yes", "on"
            ):
                eos_ids = sampling_metadata.greedy_eos_token_ids
                trace_rows = (eos_ids >= 0).nonzero(as_tuple=False).view(-1)
                trace_limit = int(os.environ.get("CODEX_LOGIT_TRACE_STEPS", "0"))
                codec_trace = []
                for row_tensor in trace_rows:
                    row = int(row_tensor.item())
                    step = self._codex_trace_counter
                    self._codex_trace_counter += 1
                    if trace_limit > 0 and step >= trace_limit:
                        continue
                    row_logits = logits[row]
                    log_norm = torch.logsumexp(row_logits.float(), dim=-1)
                    token_stats = []
                    for token_id in (int(eos_ids[row].item()), 24083, 24074):
                        token_logit = row_logits[token_id].float()
                        token_stats.append((
                            token_id,
                            float(token_logit.item()),
                            float(torch.exp(token_logit - log_norm).item()),
                            int((row_logits > row_logits[token_id]).sum().item()) + 1,
                        ))
                    codec_trace.append((
                        row,
                        step,
                        int(greedy_eos_sampled[row].item()),
                        token_stats,
                    ))
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None

        logits_before_apply_temperature = None
        if (
            envs.VLLM_ENABLE_CODE_SWITCH_GATE 
            and envs.VLLM_CODE_SWITCH_GATE_USE_RAW_LOGITS
        ):
            logits_before_apply_temperature = logits.clone()

        # Apply temperature.
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top_k and/or top_p.
        sampler_input_logits = (
            logits if not envs.VLLM_ENABLE_CODE_SWITCH_GATE
            else [logits, logits_before_apply_temperature, lang_logits]
        )
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            sampler_input_logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
            self.output_tokens_t,
            sampling_metadata,
            spec_decode_metadata,
        )

        if greedy_eos_sampled is not None:
            eos_ids = sampling_metadata.greedy_eos_token_ids
            force_eos = (eos_ids >= 0) & (greedy_eos_sampled == eos_ids)
            random_sampled = torch.where(
                force_eos, greedy_eos_sampled, random_sampled
            )
            if self._codex_force_pos < len(self._codex_forced_prefix):
                codec_rows = (eos_ids >= 0).nonzero(as_tuple=False).view(-1)
                for row_tensor in codec_rows:
                    row = int(row_tensor.item())
                    random_sampled[row] = self._codex_forced_prefix[
                        self._codex_force_pos
                    ]
                    self._codex_force_pos += 1
                    if self._codex_force_pos >= len(self._codex_forced_prefix):
                        break
            if codec_trace:
                for row, step, argmax_id, token_stats in codec_trace:
                    logger.warning(
                        "CODEX_LOGIT_TRACE row=%d step=%d sampled=%d argmax=%d "
                        "eos=%d:logit=%.6f:prob=%.8f:rank=%d "
                        "tok24083=logit=%.6f:prob=%.8f:rank=%d "
                        "tok24074=logit=%.6f:prob=%.8f:rank=%d",
                        row, step, int(random_sampled[row].item()), argmax_id,
                        *token_stats[0],
                        token_stats[1][1], token_stats[1][2], token_stats[1][3],
                        token_stats[2][1], token_stats[2][2], token_stats[2][3],
                    )

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
        top_k: torch.Tensor | None = None,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Code-switch gate breaks the semantic of topk, allows tokens beyond the range
        # of traditional topk, we ensure the maximum rank of sampled token to be the
        # "topk" one. This makes when topk == top_logprobs, the sampled token is always
        # included in the final logprobs output.
        #
        # See also: vllm/v1/sample/sampler.py:gather_logprobs
        if envs.VLLM_ENABLE_CODE_SWITCH_GATE and top_k is not None:
            token_ranks = torch.minimum(token_ranks, top_k)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        return [
            [*out, *spec] if spec else out
            for out, spec in zip(output_token_ids, spec_token_ids)
        ]

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        any_penalties_or_bad_words = (
            bool(bad_words_token_ids) or not sampling_metadata.no_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids
        if predict_bonus_token and any_penalties_or_bad_words:
            # Combine base outputs with spec tokens when speculative decoding
            # is enabled.
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Apply allowed token ids.
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        # Apply logits processors which can impact greedy sampling.
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # Apply penalties (e.g., freq_penalties).
        if envs.VLLM_ENABLE_CODE_SWITCH_GATE:
            if sampling_metadata.no_penalties:
                return logits
            assert sampling_metadata.prompt_token_ids is not None
            logits, self.output_tokens_t = apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                output_token_ids,
            )
        else:
            logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )
