# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

from vllm import envs
from vllm.config import SpeculativeConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.intervention_sampler import InterventionSampler
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
GREEDY_TEMPERATURE: tl.constexpr = 0

# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 128
LOOSE_VERIFICATION_TOP_K: tl.constexpr = 2


class RejectionSampler(nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def __init__(
        self,
        sampler: Sampler,
        speculative_config: SpeculativeConfig,
        device=None,
    ):
        super().__init__()
        self.sampler = sampler
        self.main_sampler = sampler
        self.device = device
        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode.startswith("processed")
        self.is_logits_logprobs_mode = logprobs_mode.endswith("logits")
        if envs.VLLM_ENABLE_CODE_SWITCH_GATE:
            self.intervention_sampler = InterventionSampler()
        self.output_tokens_t = None
        self.speculative_config = speculative_config

    def _get_load_balance_assignment(
        self,
        sizes: list[int],
        num_gpus: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Generate load balancing assignment and metadata
        for distributing data across GPUs.
        The load is determined by the total image sizes,
        not the number of images.

        Args:
            sizes: The size of each image
            num_gpus: Number of GPUs to balance across

        Returns:
            shuffle_indices:
                Indices to reorder data for balanced loading
            gpu_sample_counts:
                Number of samples assigned to each GPU
            grouped_sizes_per_gpu:
                Total size assigned to each GPU

        Example:
            ```
            sizes = [1000, 100, 200, 50]
            num_gpus = 2
            ```

        """

        n_samples = len(sizes)

        # Handle edge cases
        if n_samples == 0:
            return [], [0] * num_gpus, [0] * num_gpus

        # Use greedy algorithm - balance by total size, not sample count
        gpu_assignments = [list[int]() for _ in range(num_gpus)]
        gpu_loads = [0] * num_gpus  # This tracks total SIZE, not sample count

        # Sort indices by size (largest first for better load balancing)
        # sizes = [1000, 100, 200, 50]
        # large_to_small_indices = [0, 2, 1, 3]
        large_to_small_indices = sorted(
            range(n_samples), key=lambda i: sizes[i], reverse=True
        )

        for idx in large_to_small_indices:
            # Find GPU with minimum current load (by total size)
            min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
            gpu_assignments[min_gpu].append(idx)
            gpu_loads[min_gpu] += sizes[idx]

        # Create shuffle indices and counts
        shuffle_indices = list[int]()
        gpu_sample_counts = list[int]()
        for gpu_id in range(num_gpus):
            # GPU_0 = [1000] = [0]
            # GPU_1 = [200, 100, 50] = [2, 1, 3]
            # shuffle_indices = [0, 2, 1, 3]
            shuffle_indices.extend(gpu_assignments[gpu_id])
            # GPU_0 = [1]
            # GPU_1 = [3]
            # gpu_sample_counts = [1, 3]
            gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

        return (shuffle_indices, gpu_sample_counts, gpu_loads)

    def run_sample_dp(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        num_output_tokens: torch.Tensor | None,
    ) -> SamplerOutput:
        """
        run rejection sampling forward in data parallel with tp_group.
        | tp_rank0        | tp_rank1        | tp_rank2        | tp_rank3        |
        | data(0,1,2,3)   | data(0,1,2,3)   | data(0,1,2,3)   | data(0,1,2,3)   |
        | v_dp_rank0      | v_dp_rank1      | v_dp_rank2      | v_dp_rank3      |
        | data(0)         | data(1)         | data(2)         | data(3)         |
        | sampler         | sampler         | sampler         | sampler         |
        | output(0)       | output(1)       | output(2)       | output(3)       |
        | paded_output(0) | paded_output(1) | paded_output(2) | paded_output(3) |
        | ==========================     all_reduce.     ====================== |
        | output(0,1,2,3) | output(0,1,2,3) | output(0,1,2,3) | output(0,1,2,3) |

        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        global_batch_size = len(metadata.num_draft_tokens)
        # shuffle_indices of num_draft_tokens
        # gpu_sample_counts: number of requests per GPU
        # grouped_sizes_per_gpu: total draft tokens assigned to each GPU
        num_sampled_tokens = [num + 1 for num in metadata.num_draft_tokens]
        (
            shuffle_indices,
            gpu_sample_counts,
            grouped_sizes_per_gpu
        ) = self._get_load_balance_assignment(
            num_sampled_tokens, tp_size
        )
        cum_draft_tokens = np.cumsum([0] + metadata.num_draft_tokens).tolist()
        cum_gpu_sample_counts = np.cumsum([0] + gpu_sample_counts).tolist()

        local_bonus_token_indices = shuffle_indices[
            cum_gpu_sample_counts[tp_rank] : cum_gpu_sample_counts[tp_rank + 1]
        ]
        local_draft_token_indices: list[int] = []
        for indices in shuffle_indices[
            cum_gpu_sample_counts[tp_rank]:cum_gpu_sample_counts[tp_rank+1]
        ]:
            local_draft_token_indices = local_draft_token_indices + [
                i for i in range(cum_draft_tokens[indices], cum_draft_tokens[indices+1])
            ]

        torch_to_numpy_dtype_dict = {
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float32: np.float32,
            torch.float64: np.float64,
        }

        local_num_draft_tokens = [
            metadata.num_draft_tokens[i] for i in local_bonus_token_indices
        ]
        num_draft_tokens_tensor = np.array(local_num_draft_tokens,
            dtype=torch_to_numpy_dtype_dict[metadata.cu_num_draft_tokens.dtype])
        num_sampled_tokens_tensor = num_draft_tokens_tensor + 1

        # before used as indices, copy to same device as torch tensor
        local_draft_token_indices_tensor = torch.tensor(
            local_draft_token_indices, dtype=torch.int32).to(
            metadata.draft_token_ids.device,
            non_blocking=True,
        )
        local_bonus_token_indices_tensor = torch.tensor(
            local_bonus_token_indices, dtype=torch.int32).to(
            metadata.bonus_logits_indices.device,
            non_blocking=True,
        )
        local_metadata = replace(
            metadata,
            draft_token_ids=metadata.draft_token_ids[
                local_draft_token_indices_tensor
            ], # [num_tokens]
            num_draft_tokens=local_num_draft_tokens,  # [batch_size]
            cu_num_draft_tokens=torch.from_numpy(
                np.cumsum(num_draft_tokens_tensor,
                    dtype=num_draft_tokens_tensor.dtype)).to(
                metadata.cu_num_draft_tokens.device, non_blocking=True
            ),  # [batch_size]
            cu_num_sampled_tokens=torch.from_numpy(
                np.cumsum(num_sampled_tokens_tensor,
                    dtype=num_sampled_tokens_tensor.dtype)).to(
                metadata.cu_num_sampled_tokens.device, non_blocking=True
            ),  # [batch_size]
            target_logits_indices=metadata.target_logits_indices[
                local_draft_token_indices_tensor
            ],  # [num_tokens]
            bonus_logits_indices=metadata.bonus_logits_indices[
                local_bonus_token_indices_tensor
            ],  # [batch_size]
        )

        new_sampling_metadata = copy.deepcopy(sampling_metadata)
        for key, value in vars(new_sampling_metadata).items():
            if isinstance(value, torch.Tensor) and value.shape[0] > 0:
                assert value.shape[0] >= global_batch_size
                value = value[local_bonus_token_indices_tensor]
            elif isinstance(value, list) and len(value) > 0:
                assert len(value) >= global_batch_size
                value = [value[i] for i in local_bonus_token_indices]
            elif isinstance(value, dict):
                new_value = {}
                for new_idx, old_idx in enumerate(local_bonus_token_indices):
                    if old_idx in value:
                        new_value[new_idx] = value[old_idx]
                value = new_value
            setattr(new_sampling_metadata, key, value)

        sampler_output = self.forward_impl(
            local_metadata,
            draft_probs,
            logits,
            new_sampling_metadata,
            num_output_tokens
        )

        output_token_ids = torch.zeros(
            (global_batch_size, metadata.max_spec_len + 1),
            dtype=sampler_output.sampled_token_ids.dtype,
            device=sampler_output.sampled_token_ids.device
        )
        output_token_ids[local_bonus_token_indices_tensor, :] = torch.tensor(
            PLACEHOLDER_TOKEN_ID, dtype=output_token_ids.dtype).to(
                device=output_token_ids.device, non_blocking=True)
        output_token_ids[
            local_bonus_token_indices_tensor, :sampler_output.sampled_token_ids.shape[1]
        ] = sampler_output.sampled_token_ids

        output_token_ids = tensor_model_parallel_all_reduce(output_token_ids)
        assert sampler_output.logprobs_tensors is None
        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=None,
        )

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        num_output_tokens: torch.Tensor | None,  # [batch_size]
    ) -> SamplerOutput:
        can_run_dp = sampling_metadata.max_num_logprobs is None and \
                    get_tp_group().world_size > 1 and \
                    get_tp_group().world_size <= len(metadata.num_draft_tokens) and \
                    len(sampling_metadata.logitsprocs.argmax_invariant) == 0 and \
                    len(sampling_metadata.logitsprocs.non_argmax_invariant) == 0 and \
                    len(sampling_metadata.logitsprocs.sps_invariant) == 0 and \
                    envs.VLLM_ENABLE_SAMPLER_DP

        if can_run_dp:
            sampler_output = self.run_sample_dp(
                metadata, draft_probs, logits, sampling_metadata, num_output_tokens
            )
        else:
            sampler_output = self.forward_impl(
                metadata, draft_probs, logits, sampling_metadata, num_output_tokens
            )

        return sampler_output

    def forward_impl(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        num_output_tokens: torch.Tensor | None,  # [batch_size]
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        if envs.VLLM_ENABLE_CODE_SWITCH_GATE:
            [logits, lang_logits] = logits
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]

        if envs.VLLM_ENABLE_CODE_SWITCH_GATE:
            bonus_lang_logits = lang_logits[bonus_logits_indices]
            bonus_sampler_output = self.sampler(
                logits=[bonus_logits, bonus_lang_logits],
                sampling_metadata=replace(
                    sampling_metadata,
                    max_num_logprobs=-1,
                ),
                predict_bonus_token=True,
                # Override the logprobs mode to return logits because they are
                # needed later to compute the accepted token logprobs.
                logprobs_mode_override="processed_logits"
                if self.is_processed_logprobs_mode
                else "raw_logits",
            )
        else:
            bonus_sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=replace(
                    sampling_metadata,
                    max_num_logprobs=-1,
                ),
                predict_bonus_token=True,
                # Override the logprobs mode to return logits because they are
                # needed later to compute the accepted token logprobs.
                logprobs_mode_override="processed_logits"
                if self.is_processed_logprobs_mode
                else "raw_logits",
            )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = self.apply_logits_processors(
            raw_target_logits, sampling_metadata, metadata
        )

        if sampling_metadata.max_num_logprobs is not None:
            raw_target_logits = target_logits.clone()

        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
        if envs.VLLM_ENABLE_CODE_SWITCH_GATE:
            target_lang_logits = lang_logits[target_logits_indices]
            target_logits = apply_sampling_constraints(
                [target_logits, target_lang_logits],
                metadata.cu_num_draft_tokens,
                sampling_metadata,
                self.intervention_sampler,
                metadata,
                self.output_tokens_t,
            )
        else:
            target_logits = apply_sampling_constraints(
                target_logits,
                metadata.cu_num_draft_tokens,
                sampling_metadata,
            )
        # Compute probability distribution from target logits.
        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

        if self.speculative_config.use_loose_topk_verify:
            # Compute top-k token ids from the original logits
            # (before temperature scaling)
            topk_token_logits, topk_token_ids = target_logits.topk(
                LOOSE_VERIFICATION_TOP_K, dim=-1
            )  # [num_tokens, TOP_K]
            output_token_ids = rejection_sample_use_loose_top_k(
                metadata.draft_token_ids,
                metadata.num_draft_tokens,
                metadata.max_spec_len,
                metadata.cu_num_draft_tokens,
                draft_probs,
                target_probs,
                bonus_token_ids,
                sampling_metadata,
                topk_token_ids,
                topk_token_logits,
                num_output_tokens,
                self.speculative_config.loose_topk_verify_threshold,
                self.speculative_config.loose_topk_verify_max_tokens,
            )
        else:
            output_token_ids = rejection_sample(
                metadata.draft_token_ids,
                metadata.num_draft_tokens,
                metadata.max_spec_len,
                metadata.cu_num_draft_tokens,
                draft_probs,
                target_probs,
                bonus_token_ids,
                sampling_metadata,
            )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
                sampling_metadata.top_k,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )

    def _get_logprobs_tensors(
        self,
        max_num_logprobs: int,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        top_k: torch.Tensor | None = None,
    ) -> LogprobsTensors:
        _, vocab_size = logits.shape
        cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
        cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

        # Collect target and bonus logits.
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices
        final_logits = torch.zeros_like(logits, dtype=torch.float32)
        final_logits[target_logits_indices] = target_logits.to(torch.float32)
        final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)

        # Compute accepted token indices.
        accepted_mask = sampled_token_ids != PLACEHOLDER_TOKEN_ID
        num_accepted_tokens = accepted_mask.sum(dim=-1)
        accepted_logit_indices = accepted_mask.nonzero(as_tuple=True)[1]
        accepted_logit_indices += cu_num_sampled_tokens.repeat_interleave(
            num_accepted_tokens
        )

        # Compute logprobs for accepted tokens.
        accepted_logits = final_logits[accepted_logit_indices]
        accepted_logprobs = (
            accepted_logits
            if self.is_logits_logprobs_mode
            else self.sampler.compute_logprobs(accepted_logits)
        )

        if envs.VLLM_ENABLE_CODE_SWITCH_GATE and top_k is not None:
            top_k = top_k.view(-1, 1).expand_as(sampled_token_ids)[accepted_mask]
        else:
            top_k = None

        accepted_tokens = sampled_token_ids[accepted_mask]
        return self.sampler.gather_logprobs(
            accepted_logprobs,
            max_num_logprobs,
            accepted_tokens.to(torch.int64).clamp(max=vocab_size - 1),
            top_k=top_k,
        )

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        return_cu_num_tokens: bool = False,
        accept_rate: float | None = None,
    ) -> tuple[list[list[int]], list[list[int]] | None, list[int] | None]:
        """Parse the output of the rejection sampler.
        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID` by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.
            discard_req_indices: Optional row indices to discard tokens in.
            return_cu_num_tokens: Whether to also return cumulative token counts.
            accept_rate: simulate speculative decoding accept rate

        Returns:
            A list of lists of token IDs.
        """
        cu_num_tokens = None
        if accept_rate is None:
            output_token_ids_np = output_token_ids.cpu().numpy()
            # Create mask for valid tokens.
            valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
                output_token_ids_np <= vocab_size
            )
            if return_cu_num_tokens:
                cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            if len(discard_req_indices) > 0:
                valid_mask[discard_req_indices] = False
            outputs = []
            indices = []
            for i, row in enumerate(output_token_ids_np):
                idxs = np.nonzero(valid_mask[i])[0]
                outputs.append(row[idxs].tolist())
                indices.append(idxs.tolist())
            return outputs, indices, cu_num_tokens
        else:
            # IMPORTANT!!!
            # when set a simulate accept rate, output not right
            # 1. fill output_token_ids to number(not PLACEHOLDER_TOKEN_ID)
            # 2. generate a random mat depend on accept_rate (assume i.i.d)
            # 3. select output_token
            output_token_ids[output_token_ids == PLACEHOLDER_TOKEN_ID] = 1
            simulate_accept_rate = torch.bernoulli(
                torch.full(
                    [output_token_ids.shape[0], output_token_ids.shape[1] - 1],
                    accept_rate,
                    device=output_token_ids.device,
                )
            ).bool()
            valid_mask = (simulate_accept_rate == 0).max(1).indices
            valid_mask[~(simulate_accept_rate == 0).any(1)] = (
                output_token_ids.shape[1] - 1
            )
            if return_cu_num_tokens:
                cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            valid_mask = valid_mask.cpu().numpy()
            output_token_ids_np = output_token_ids.cpu().numpy()
            outputs = [
                row[: valid_mask[i] + 1].tolist()
                for i, row in enumerate(output_token_ids_np)
            ]
        return outputs, None, cu_num_tokens

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        if logits.shape[0] == 0:  # num_tokens == 0
            return logits

        for processor in sampling_metadata.logitsprocs.sps_invariant:
            logits = processor.apply(logits, metadata.num_draft_tokens)

        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Calculate indices of target logits.
        if sampling_metadata.allowed_token_ids_mask is not None or has_penalties:
            num_requests = len(sampling_metadata.output_token_ids)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            if envs.VLLM_ENABLE_CODE_SWITCH_GATE:
                logits, self.output_tokens_t = self.apply_penalties(
                    logits,
                    sampling_metadata,
                    metadata,
                    repeat_indices,
                    output_token_ids,
                )
            else:
                logits = self.apply_penalties(
                    logits,
                    sampling_metadata,
                    metadata,
                    repeat_indices,
                    output_token_ids,
                )

            # Apply allowed token ids.
            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        assert repeat_indices.shape[0] == logits.shape[0], (
            "sampling metadata does not expand correctly in rejection sampler"
        )
        assert sampling_metadata.prompt_token_ids is not None

        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]

        # Note that the output_token_ids has already been expanded to align
        # with repeat_indices by self._combine_outputs_with_spec_tokens.
        logits = apply_all_penalties(
            logits,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        return logits

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if not spec_token_ids:
            return output_token_ids

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return output_token_ids


def rejection_sample_use_loose_top_k(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    # [num_tokens, TOP_K]
    topk_token_ids: torch.Tensor | None,
    # [num_tokens, TOP_K]
    topk_token_logits: torch.Tensor | None,
    num_output_tokens: torch.Tensor | None,
    # [batch_size]
    loose_topk_verify_threshold: float,
    loose_topk_verify_max_tokens: int,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)
    assert (
        topk_token_ids is not None
        and topk_token_logits is not None
        and topk_token_ids.ndim == 2
        and topk_token_ids.is_contiguous()
        and topk_token_ids.shape == (num_tokens, LOOSE_VERIFICATION_TOP_K)
    )

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    _2_1_ratio = topk_token_logits[:, 1] / (topk_token_logits[:, 0] + 1e-10)
    # [num_tokens] 50%分位数-0.87, 75%分位数-0.95
    _2_1_threshold = torch.full(
        (num_tokens,),
        loose_topk_verify_threshold,
        dtype=torch.float32,
        device=device,
    )

    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1)
        rejection_greedy_sample_kernel_use_loose_top_k[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            topk_token_ids,
            _2_1_ratio,
            _2_1_threshold,
            num_output_tokens,
            max_spec_len,
            vocab_size,
            loose_topk_verify_max_tokens,
            num_warps=1,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
    rejection_random_sample_kernel_use_loose_top_k[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        topk_token_ids,
        _2_1_ratio,
        _2_1_threshold,
        num_output_tokens,
        max_spec_len,
        vocab_size,
        loose_topk_verify_max_tokens,
        NO_DRAFT_PROBS=draft_probs is None,
        num_warps=1,
    )
    return output_token_ids


def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
    intervention_sampler: InterventionSampler | None = None,
    spec_decode_metadata: SpecDecodeMetadata | None = None,
    output_tokens_t: torch.Tensor = None,
) -> torch.Tensor:
    """Process logits based on sampling metadata.

    This function applies temperature scaling to the logits,
    as well as top-k and top-p. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be processed.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Processed logits if non-greedy sampling is used,
        otherwise returns the original logits.
    """
    if envs.VLLM_ENABLE_CODE_SWITCH_GATE:
        [logits, lang_logits] = logits

        logits_before_apply_temperature = None
        if envs.VLLM_CODE_SWITCH_GATE_USE_RAW_LOGITS:
            logits_before_apply_temperature = logits.clone()

    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    if num_tokens == 0:
        return logits

    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    if envs.VLLM_ENABLE_CODE_SWITCH_GATE:
        if intervention_sampler is not None:
            logits = intervention_sampler(
                logits, logits_before_apply_temperature, lang_logits, 
                None, sampling_metadata, spec_decode_metadata
            )
    else:
        # Get expanded top_k and top_p tensors.
        top_k = None
        if sampling_metadata.top_k is not None:
            top_k = expand_batch_to_tokens(
                sampling_metadata.top_k,
                cu_num_draft_tokens,
                num_tokens,
            )
        top_p = None
        if sampling_metadata.top_p is not None:
            top_p = expand_batch_to_tokens(
                sampling_metadata.top_p,
                cu_num_draft_tokens,
                num_tokens,
            )

        # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
        # which is slow for large vocab sizes. This may cause performance issues.
        logits = apply_top_k_top_p(logits, top_k, top_p)

    return logits


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    )
    return expanded_x


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """
    Generates a batch of uniform random samples, with optional seeding
    if available.

    This method creates a tensor of shape `(num_tokens, )` filled
    with uniform random values in the range [0, 1). If `generators` is provided,
    the requests with their own seeds will use the provided `torch.Generator`
    for reproducibility. The samples for the other requests will be generated
    without a seed.

    Args:
        num_tokens: int
            Total number of tokens.
        num_draft_tokens : list[list[int]]
            Number of draft tokens per request.
        generators : Optional[dict[int, torch.Generator]]
            A dictionary mapping indices in the batch to
            `torch.Generator` objects.
        device: torch.device
            The device on which to allocate the tensor.
    Returns:
        uniform_rand: torch.Tensor
            A tensor of shape `(num_tokens, )` containing uniform
            random values in the range [0, 1).
    """
    # NOTE(woosuk): We deliberately use float64 instead of float32 here
    # because when using float32, there's a non-negligible chance that
    # uniform_prob is sampled to be exact 0.0 as reported in
    # https://github.com/pytorch/pytorch/issues/16706. Using float64
    # mitigates the issue.
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                target_argmax_id,
            )
            if draft_token_id != target_argmax_id:
                # Reject.
                rejected = True

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
            )
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            # NOTE(woosuk): While the draft probability should never be 0,
            # we check it to avoid NaNs. If it happens to be 0, we reject.
            if draft_token_id < vocab_size and (
                draft_prob > 0 and target_prob / draft_prob >= uniform_prob
            ):
                # Accept.
                token_id = draft_token_id
            else:
                # Reject. Use recovered token.
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel_use_loose_top_k(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size]
    topk_token_ids_ptr,  # [num_tokens, TOP_K]
    _2_1_ratio_ptr,  # [num_tokens]
    _2_1_threshold_ptr,  # [num_tokens]
    num_output_tokens_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    loose_topk_verify_max_tokens,
    TOP_K: tl.constexpr = LOOSE_VERIFICATION_TOP_K,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    if is_greedy_ptr is None:
        is_greedy = True
    else:
        is_greedy = tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx
    is_short = tl.load(num_output_tokens_ptr + req_idx) <= loose_topk_verify_max_tokens

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                target_argmax_id,
            )
            # Check if draft token is in top-k
            is_in_topk = False
            is_in_top1 = False
            for i in range(TOP_K):
                topk_id = tl.load(topk_token_ids_ptr + (start_idx + pos) * TOP_K + i)
                if draft_token_id == topk_id:
                    is_in_topk = True
                    if i == 0:
                        is_in_top1 = True

            # Accept logic with relaxation
            if is_in_topk:
                if is_in_top1:
                    # Always accept if in top-1 (same as target_argmax_id match)
                    tl.store(
                        output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                        draft_token_id,
                    )
                else:
                    # For top-k but not top-1, use 2:1 ratio relaxation
                    _2_1_ratio = tl.load(_2_1_ratio_ptr + start_idx + pos)
                    _2_1_threshold = tl.load(_2_1_threshold_ptr + start_idx + pos)
                    if draft_token_id < vocab_size:
                        if is_short and _2_1_ratio > _2_1_threshold:
                            # Accept
                            tl.store(
                                output_token_ids_ptr
                                + req_idx * (max_spec_len + 1)
                                + pos,
                                draft_token_id,
                            )
                        else:
                            # Reject
                            rejected = True
                    else:
                        # Reject
                        rejected = True
            else:
                # Not in top-k, reject
                rejected = True

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel_use_loose_top_k(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    topk_token_ids_ptr,  # [num_tokens, TOP_K]
    _2_1_ratio_ptr,  # [num_tokens]
    _2_1_threshold_ptr,  # [num_tokens]
    num_output_tokens_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    loose_topk_verify_max_tokens,
    NO_DRAFT_PROBS: tl.constexpr,
    TOP_K: tl.constexpr = LOOSE_VERIFICATION_TOP_K,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx
    is_short = tl.load(num_output_tokens_ptr + req_idx) <= loose_topk_verify_max_tokens

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)

            # Check if draft token is in top-k
            is_in_topk = False
            is_in_top1 = False
            for i in range(TOP_K):
                topk_id = tl.load(topk_token_ids_ptr + (start_idx + pos) * TOP_K + i)
                if draft_token_id == topk_id:
                    is_in_topk = True
                    if i == 0:
                        is_in_top1 = True

            # If draft token is in top-k, always accept it
            if is_in_topk:
                if is_in_top1:  # top1直接接收
                    token_id = draft_token_id
                else:
                    _2_1_ratio = tl.load(_2_1_ratio_ptr + start_idx + pos)
                    _2_1_threshold = tl.load(_2_1_threshold_ptr + start_idx + pos)
                    if draft_token_id < vocab_size:
                        if is_short and _2_1_ratio > _2_1_threshold:
                            # accept
                            token_id = draft_token_id
                        else:
                            # reject
                            rejected = True
                            token_id = tl.load(
                                recovered_token_ids_ptr + start_idx + pos
                            )
                    else:
                        # reject
                        rejected = True
                        token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            else:
                # Otherwise, use standard rejection sampling logic
                if NO_DRAFT_PROBS:
                    draft_prob = 1
                else:
                    draft_prob = tl.load(
                        draft_probs_ptr
                        + (start_idx + pos) * vocab_size
                        + draft_token_id
                    )
                target_prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
                uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                # NOTE(woosuk): While the draft probability should never be 0,
                # we check it to avoid NaNs. If it happens to be 0, we reject.
                if draft_token_id < vocab_size and (
                    draft_prob > 0 and target_prob / draft_prob >= uniform_prob
                ):
                    # Accept.
                    token_id = draft_token_id
                else:
                    # Reject. Use recovered token.
                    rejected = True
                    token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)

            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:  # noqa: SIM108
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=((vocab_offset < vocab_size) & (vocab_offset != draft_token_id)),
            other=0,
        )
    else:
        draft_prob = tl.load(
            draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        target_prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        prob = tl.maximum(target_prob - draft_prob, 0)
        # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
        # `tl.argmax` will select the maximum value.

    q = tl.load(
        q_ptr + req_idx * vocab_size + vocab_offset,
        mask=vocab_offset < vocab_size,
        other=float("-inf"),
    )
    recovered_id = tl.argmax(prob / q, axis=-1)
    tl.store(output_token_ids_ptr + start_idx + pos, recovered_id)
