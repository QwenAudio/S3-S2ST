# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from collections import deque

import torch
from .moe_utils import (
    MoEAuxLossAutoScaler,
    sinkhorn,
    switch_load_balancing_loss_func,
    z_loss_func,
    get_capacity,
)
from megatron.core.tensor_parallel.mappings import gather_from_expert_model_parallel_region, \
    scatter_to_expert_model_parallel_region, _gather_along_first_dim_expert_parallel, \
    gather_uneven_from_expert_model_parallel_region, scatter_uneven_to_expert_model_parallel_region
from megatron.core import mpu
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name, \
    gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from .router import Router
from megatron.training import get_args

_LBL_MODEL_CHUNK_ID = 0
def get_lbl_model_chunk_id():
    global _LBL_MODEL_CHUNK_ID
    return _LBL_MODEL_CHUNK_ID
def set_lbl_model_chunk_id(model_chunk_id):
    global _LBL_MODEL_CHUNK_ID
    _LBL_MODEL_CHUNK_ID = model_chunk_id
_NUM_MICROBATCHES = 0
def get_lbl_num_microbatches():
    global _NUM_MICROBATCHES
    return _NUM_MICROBATCHES
def set_lbl_num_microbatches(num_microbatches):
    global _NUM_MICROBATCHES
    _NUM_MICROBATCHES = num_microbatches

_TOKENS_PER_EXPERT = {}

def update_tokens_per_expert(tokens_per_expert):
    global _TOKENS_PER_EXPERT
    model_chunk_id = get_lbl_model_chunk_id()
    if model_chunk_id not in _TOKENS_PER_EXPERT:
        _TOKENS_PER_EXPERT[model_chunk_id] = 0
    _TOKENS_PER_EXPERT[model_chunk_id] = _TOKENS_PER_EXPERT[model_chunk_id] + tokens_per_expert
    return _TOKENS_PER_EXPERT[model_chunk_id]

def reset_tokens_per_expert():
    global _TOKENS_PER_EXPERT
    for _, value in _TOKENS_PER_EXPERT.items():
        value.zero_()


_LOAD_BALANCING_LOSS = []


def save_load_balancing_loss(loss):
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.append(loss)


def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS


def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()

_LBL_WORKLOAD_HANDLE_QUEUE = {}
def pre_launch_lbl_loss_all_reduce():
    args = get_args()
    # tokens_per_expert[i].shape = (num_experts)
    # expert_scores[i].shape = (tokens, num_experts)
    tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())
    # Verify the shape of the tokens_per_expert and expert_scores tensors.
    assert all([
        x.ndim == 1 and x.numel() == args.moe_num_experts
        for x in tokens_per_expert
    ])
    tokens = expert_scores[0].shape[0]
    assert all([
        (x.ndim == 2 and x.shape[1] == args.moe_num_experts and
         x.shape[0] == tokens) for x in expert_scores
    ])
    # Concatenate the contributions of each layer and convert to
    # the correct types and formats for the dot product.
    if args.moe_lbl_in_fp32:
        expert_scores = torch.cat(expert_scores, dim=1).float().mean(dim=0)
    else:
        expert_scores = torch.cat(expert_scores, dim=1).mean(dim=0)
    tokens_per_expert = torch.cat(tokens_per_expert).to(expert_scores.dtype) # (layers, num_experts) -> (layers * num_experts)
    handle = torch.distributed.all_reduce(
        tokens_per_expert,
        group=mpu.get_data_parallel_group(with_context_parallel=True),
        async_op=True
        )
    model_chunk_id = get_lbl_model_chunk_id()
    global _LBL_WORKLOAD_HANDLE_QUEUE
    if model_chunk_id not in _LBL_WORKLOAD_HANDLE_QUEUE:
        _LBL_WORKLOAD_HANDLE_QUEUE[model_chunk_id] = deque()
    _LBL_WORKLOAD_HANDLE_QUEUE[model_chunk_id].append((tokens_per_expert, expert_scores, handle))
_LBL_LOSS_FOR_REDUCE = []
def get_lbl_loss_for_reduce():
    global _LBL_LOSS_FOR_REDUCE
    return _LBL_LOSS_FOR_REDUCE
def clear_lbl_loss_for_reduce():
    global _LBL_LOSS_FOR_REDUCE
    _LBL_LOSS_FOR_REDUCE.clear()
def post_finish_lbl_loss_calculate():
    args = get_args()
    global _LBL_WORKLOAD_HANDLE_QUEUE
    model_chunk_id = get_lbl_model_chunk_id()
    tokens_per_expert, expert_scores, handle = _LBL_WORKLOAD_HANDLE_QUEUE[model_chunk_id].popleft()
    handle.wait()
    tokens_per_expert = tokens_per_expert / torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group(with_context_parallel=True))

    # 由于 69 行 cat 后 tokens_per_expert.shape = (layers * num_experts)，统计 tokens 需要取第 0 层的所有 expert 分配的 tokens
    tokens = tokens_per_expert[:args.moe_num_experts].sum() // args.moe_top_k
    if args.moe_router_buffer_tokens_per_expert:
        tokens_per_expert = update_tokens_per_expert(tokens_per_expert)

    # 由于 69 行 cat 后 tokens_per_expert.shape = (layers * num_experts)，统计 tokens 需要取第 0 层的所有 expert 分配的 tokens
    tokens = tokens_per_expert[:args.moe_num_experts].sum() // args.moe_top_k
    # Calculate the total scale across all factors.
    # loss_weight * num_experts / (num_layers * tokens * top_k)
    scale_numerator = (
            args.moe_num_experts *
            args.moe_loss_weight
    )
    scale_denominator = (
            args.num_layers *
            tokens *
            args.moe_top_k
    )
    scale = scale_numerator / scale_denominator
    lbl_loss = scale * torch.dot(tokens_per_expert, expert_scores)
    global _LBL_LOSS_FOR_REDUCE
    _LBL_LOSS_FOR_REDUCE.append({
        'load balancing loss': lbl_loss.clone().detach().view(1)
    })
    return lbl_loss

def batched_load_balancing_loss():
    args = get_args()
    # tokens_per_expert[i].shape = (num_experts)
    # expert_scores[i].shape = (tokens, num_experts)
    lbl_result = get_load_balancing_loss()
    if len(lbl_result) == 0:
        if args.moe_router_sync_tokens_per_expert_across_dp:
            return (torch.tensor(0.0), torch.tensor(0.0)), torch.tensor(0.0)
        else:
            return torch.tensor(0.0), torch.tensor(0.0)
    tokens_per_expert, expert_scores = zip(*lbl_result)

    # Verify the shape of the tokens_per_expert and expert_scores tensors.
    assert all([
        x.ndim == 1 and x.numel() == args.num_experts
        for x in tokens_per_expert
    ])

    tokens = expert_scores[0].shape[0]
    assert all([
        (x.ndim == 2 and x.shape[1] == args.num_experts and
         x.shape[0] == tokens) for x in expert_scores
    ])

    # Concatenate the contributions of each layer and convert to
    # the correct types and formats for the dot product.
    # if args.moe_lbl_in_fp32:
    expert_scores = torch.cat(expert_scores, dim=1).float().mean(dim=0)
    # else:
    # expert_scores = torch.cat(expert_scores, dim=1).mean(dim=0)
    # tokens = tokens_per_expert[0].sum() // args.moe_router_topk
    tokens_per_expert = torch.cat(tokens_per_expert).to(expert_scores.dtype)

    # reduce tokens_per_expert and expert_scores across context parallel
    if args.context_parallel_size > 1:
        # will work only under evenly split CP
        torch.distributed.all_reduce(expert_scores, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
        torch.distributed.all_reduce(tokens_per_expert, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
        expert_scores /= args.context_parallel_size
        tokens_per_expert /= args.context_parallel_size

    if args.moe_router_sync_tokens_per_expert_across_dp:
        # compute local lbl
        # 将乘 moe_loss_weight 挪到 plot 结束后再进行，方便在 plot 中比较不同 lbl 的影响
        scale_numerator =  args.num_experts # * args.moe_loss_weight
        scale_denominator = args.num_layers * tokens * args.moe_router_topk
        scale = scale_numerator / scale_denominator
        sequence_lbl = scale * torch.dot(tokens_per_expert, expert_scores)

        with torch.no_grad():
            torch.distributed.all_reduce(tokens_per_expert,
                                        group=mpu.get_data_parallel_group(with_context_parallel=True))
            tokens_per_expert = tokens_per_expert / torch.distributed.get_world_size(
                group=mpu.get_data_parallel_group(with_context_parallel=True))

    if args.moe_router_buffer_tokens_per_expert:
        with torch.no_grad():
            tokens_per_expert = update_tokens_per_expert(tokens_per_expert)

    tokens = tokens_per_expert[:args.num_experts].sum() // args.moe_router_topk

    # Calculate the total scale across all factors.
    #
    # loss_weight * num_experts / (num_layers * tokens * top_k)

    scale_numerator = args.num_experts # * args.moe_aux_loss_coeff
    scale_denominator = args.num_layers * tokens * args.moe_router_topk

    # Here we do not use tokens in scale_denominator and return tokens for compatibility with lm_loss.
    scale = scale_numerator / scale_denominator
    lbl = scale * torch.dot(tokens_per_expert, expert_scores)
    if args.moe_router_sync_tokens_per_expert_across_dp:
        return (lbl, sequence_lbl), tokens
    else:
        return lbl, tokens


_ROUTER_Z_LOSS = []


def save_router_z_loss(loss):
    global _ROUTER_Z_LOSS
    _ROUTER_Z_LOSS.append(loss)


def get_router_z_loss():
    global _ROUTER_Z_LOSS
    return _ROUTER_Z_LOSS


def clear_router_z_loss():
    global _ROUTER_Z_LOSS
    _ROUTER_Z_LOSS.clear()


def batched_router_z_loss():
    args = get_args()
    # gate_logits[i].shape = (tokens, num_experts)
    gate_logits = get_router_z_loss()

    if len(gate_logits) == 0:
        return torch.tensor(0.0)

    scale = len(gate_logits) / args.num_layers * args.moe_z_loss_coeff

    gate_logits = torch.cat(gate_logits).float()
    router_z_loss = torch.logsumexp(gate_logits, dim=-1)
    router_z_loss = torch.square(router_z_loss)

    # Here we change mean to sum for compatibility with lm_loss.
    #  router_z_loss = router_z_loss.sum() * scale / args.num_experts
    router_z_loss = router_z_loss.mean() * scale

    return router_z_loss


_MLM_NUM_DROPPED_TOKENS = []


def save_num_dropped_tokens(num_dropped_tokens):
    global _MLM_NUM_DROPPED_TOKENS
    _MLM_NUM_DROPPED_TOKENS.append(num_dropped_tokens)


def get_num_dropped_tokens():
    global _MLM_NUM_DROPPED_TOKENS
    return _MLM_NUM_DROPPED_TOKENS


def clear_num_dropped_tokens():
    global _MLM_NUM_DROPPED_TOKENS
    _MLM_NUM_DROPPED_TOKENS.clear()


def batched_dropped_token_ratio():
    args = get_args()
    # gate_logits[i].shape = (tokens, num_experts)
    num_tokens, num_dropped_tokens = zip(*get_num_dropped_tokens())

    if args.moe_drop_across_ep:
        return (sum(num_dropped_tokens)
                / sum(num_tokens)
                / (mpu.get_expert_model_flashmoe_hep_parallel_world_size()
                   if args.use_flashmoe_hep else mpu.get_expert_model_parallel_world_size()))

    # Here we change mean to sum for compatibility with lm_loss.
    return sum(num_dropped_tokens) / sum(num_tokens)


class FlashMoETopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.is_last_router = False
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.input_jitter = None
        self.moe_aux_loss_func = switch_load_balancing_loss_func

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
            scores = torch.gather(logits, 1, indices)
        else:
            logits = _sinkhorn_activation(logits)
            scores, indices = torch.topk(logits, k=self.topk, dim=1)
        return scores, indices

    def topk_softmax_with_capacity(self, gate_logits):
        num_tokens = gate_logits.shape[0]
        if self.config.moe_mock_router_sigma is not None:
            orig_probs = torch.normal(0, self.config.moe_mock_router_sigma,
                                      size=gate_logits.shape, device=gate_logits.device,
                                      ).softmax(dim=-1, dtype=torch.float32).type_as(gate_logits)
        else:
            orig_probs = gate_logits.softmax(dim=-1, dtype=torch.float32).type_as(gate_logits)

        probs = torch.nn.functional.dropout(orig_probs, p=self.config.moe_expert_dropout, training=self.training)
        if not self.config.moe_router_pre_softmax:
            expert_logits, top_indices = torch.topk(gate_logits, k=self.topk, dim=-1)
            if self.config.moe_router_topk == 1:
                probs = expert_logits.unsqueeze(-1).softmax(dim=-1).squeeze(-1)
            else:
                probs = expert_logits.softmax(dim=-1)
        else:
            probs, top_indices = torch.topk(probs, k=self.topk, dim=-1)

        if self.config.moe_expert_capacity_factor is None or self.config.moe_expert_capacity_factor == 0:
            # TopK without capacity
            tokens_per_expert = torch.histc(top_indices, bins=self.num_experts, min=0, max=self.num_experts)
            if (self.config.sequence_parallel and self.config.use_flashmoe_hep
                    and not self.config.moe_pad_expert_input_to_capacity):
                probs = scatter_to_sequence_parallel_region(probs)
                top_indices = scatter_to_sequence_parallel_region(top_indices)
            return probs, top_indices, orig_probs, gate_logits, tokens_per_expert, None
        else:
            if self.config.moe_drop_across_ep:
                if self.config.use_dynamic_collate:
                    num_global_tokens_per_ep = _gather_along_first_dim_expert_parallel(
                        torch.tensor([num_tokens], device=probs.device),
                        group=mpu.get_expert_model_flashmoe_hep_parallel_group() if self.config.use_flashmoe_hep else None).tolist()
                    probs = gather_uneven_from_expert_model_parallel_region(
                        probs, num_global_tokens_per_ep,
                        group=mpu.get_expert_model_flashmoe_hep_parallel_group() if self.config.use_flashmoe_hep else None)
                    top_indices = gather_uneven_from_expert_model_parallel_region(
                        top_indices, num_global_tokens_per_ep,
                        group=mpu.get_expert_model_flashmoe_hep_parallel_group() if self.config.use_flashmoe_hep else None)
                else:
                    probs = gather_from_expert_model_parallel_region(
                        probs,
                        group=mpu.get_expert_model_flashmoe_hep_parallel_group() if self.config.use_flashmoe_hep else None)
                    top_indices = gather_from_expert_model_parallel_region(
                        top_indices,
                        group=mpu.get_expert_model_flashmoe_hep_parallel_group() if self.config.use_flashmoe_hep else None)

            tokens_per_expert = torch.histc(top_indices, bins=self.num_experts, min=0, max=self.num_experts)
            num_tokens = probs.size(0)

            # TopK with capacity
            expert_capacity = get_capacity(
                num_tokens=num_tokens * self.topk, num_experts=self.num_experts,
                capacity_factor=self.config.moe_expert_capacity_factor,
            )
            # TopK selection, Maskout unused experts
            topk_masked_gates = gate_logits.new_zeros(num_tokens, gate_logits.size(1)).scatter(1, top_indices, probs)
            topk_mask = gate_logits.new_zeros(num_tokens, gate_logits.size(1)).scatter(1, top_indices, 1)

            # Maskout exceeded tokens
            if self.config.moe_token_drop_policy == "probs":
                capacity_probs, capacity_indices = torch.topk(
                    topk_masked_gates, k=expert_capacity, dim=0, sorted=False
                )
                capacity_mask = gate_logits.new_zeros(num_tokens, gate_logits.size(1)).scatter(0, capacity_indices, 1)
            elif self.config.moe_token_drop_policy == "position":
                _, capacity_indices = torch.topk(topk_mask, k=expert_capacity, dim=0, sorted=False)
                capacity_mask = gate_logits.new_zeros(num_tokens, gate_logits.size(1)).scatter(0, capacity_indices, 1)
                capacity_probs = torch.gather(topk_masked_gates, 0, capacity_indices)
            else:
                raise NotImplementedError

            num_dropped_tokens = torch.logical_and(
                topk_mask,
                torch.logical_not(torch.logical_and(topk_mask, capacity_mask))
            ).long().sum()

            if self.config.moe_pad_expert_input_to_capacity:
                final_probs, final_indices = (
                    capacity_probs.T.contiguous(),
                    capacity_indices.T.contiguous(),
                )
            else:
                # Get exceed mask and maskout exceeded probs and indices
                final_mask = torch.logical_and(topk_mask, capacity_mask)
                drop_mask = torch.logical_not(final_mask)
                exceed_mask = torch.gather(drop_mask, 1, top_indices)
                final_probs = probs * torch.logical_not(exceed_mask)
                final_indices = top_indices.clone().masked_fill_(
                    exceed_mask, torch.iinfo(torch.int32).max
                )
            if self.config.moe_drop_across_ep:
                if self.config.use_dynamic_collate:
                    final_probs = scatter_uneven_to_expert_model_parallel_region(
                        final_probs, num_global_tokens_per_ep,
                        group=mpu.get_expert_model_flashmoe_hep_parallel_group() if self.config.use_flashmoe_hep else None)
                    final_indices = scatter_uneven_to_expert_model_parallel_region(
                        final_indices,num_global_tokens_per_ep,
                        group=mpu.get_expert_model_flashmoe_hep_parallel_group() if self.config.use_flashmoe_hep else None)
                else:
                    final_probs = scatter_to_expert_model_parallel_region(
                        final_probs,
                        group=mpu.get_expert_model_flashmoe_hep_parallel_group() if self.config.use_flashmoe_hep else None)
                    final_indices = scatter_to_expert_model_parallel_region(
                        final_indices,
                        group=mpu.get_expert_model_flashmoe_hep_parallel_group() if self.config.use_flashmoe_hep else None)

            if (self.config.use_flashmoe_hep
                    and not self.config.moe_pad_expert_input_to_capacity):
                final_probs = scatter_to_sequence_parallel_region(final_probs)
                final_indices = scatter_to_sequence_parallel_region(final_indices)
            return final_probs, final_indices, orig_probs, gate_logits, tokens_per_expert, \
                   num_dropped_tokens


    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        z_loss = z_loss_func(logits, self.config.moe_router_z_loss_weight)
        save_router_z_loss(z_loss)
        logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def save_aux_loss(self, probs, gate_logits, tokens_per_expert, num_dropped_tokens=None):
        if self.training:
            save_load_balancing_loss((tokens_per_expert, probs))
            save_router_z_loss(gate_logits)
            if num_dropped_tokens is not None:
                save_num_dropped_tokens((probs.size(0) * self.topk, num_dropped_tokens))
            if self.is_last_router:
                pre_launch_lbl_loss_all_reduce()

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply input jitter
        logits = self.apply_input_jitter(logits)

        if self.routing_type == "sinkhorn":
            scores, indices = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            logits_for_aux_loss = None
            probs_for_aux_loss = None
            if self.config.sequence_parallel:
                # Gather the logits from the TP region
                # v3 反向时对grad做了allgather，此处反向则只需split
                if self.config.use_flashmoe_hep and not self.config.moe_pad_expert_input_to_capacity:
                    logits = gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
                else:
                    # NOTE: for lbl and zloss, as its received grad is the same for different tp rank,
                    # backward should use split instead of reduce-scatter
                    logits_for_aux_loss = gather_from_sequence_parallel_region(logits,
                                                                               tensor_parallel_output_grad=False)
                    probs_for_aux_loss = logits_for_aux_loss.softmax(dim=-1, dtype=torch.float32).type_as(
                        logits_for_aux_loss)

                    logits = gather_from_sequence_parallel_region(logits)

            scores, indices, probs, gate_logits, \
                tokens_per_expert, num_dropped_tokens = self.topk_softmax_with_capacity(logits)

            self.save_aux_loss(probs_for_aux_loss if probs_for_aux_loss is not None else probs,
                               logits_for_aux_loss if logits_for_aux_loss is not None else gate_logits,
                               tokens_per_expert, num_dropped_tokens)

        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            top_logits, indices = torch.topk(logits, k=self.topk, dim=1)
            scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        return scores, indices

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, routing_map = self.routing(logits)

        if self.config.moe_norm_topk_scores:
            scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)

        return scores, routing_map
