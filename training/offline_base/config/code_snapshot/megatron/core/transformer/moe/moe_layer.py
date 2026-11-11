# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Union
from typing import Any

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.flashmoe_router import FlashMoETopKRouter
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
try:
    import flash_moe
except ModuleNotFoundError:
    # Checkpoint conversion builds the standard/grouped-GEMM MoE classes and
    # does not require the optional FlashMoE runtime extension.
    flash_moe = None


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        if self.config.use_flashmoe_router:
            self.router = FlashMoETopKRouter(config=self.config)
        else:
            self.router = TopKRouter(config=self.config)

        # Initialize experts
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules.experts, MLPSubmodules):
                self.experts = TEGroupedMLP(
                    self.num_local_experts, self.config, self.submodules.experts
                )
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules.experts, MLPSubmodules)
            self.experts = SequentialMLP(
                self.num_local_experts, self.config, self.submodules.experts
            )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = SharedExpertMLP(self.config, self.submodules.shared_experts)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            probs, routing_map = self.router(hidden_states)
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output += self.shared_experts(hidden_states)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

class FlashMoELayer(BaseMoELayer):
    """Mixture of experts Layer using FlashMoE.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None,
        act_offload_context: Any = nullcontext(),
        act_commit_offload_handle: Any = None,
        act_commit_reload_handle: Any = None,
    ):
        self.submodules = submodules
        super(FlashMoELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.router = FlashMoETopKRouter(config=self.config)

        # Initialize experts
        if isinstance(self.submodules.experts, MLPSubmodules) and config.ckpt_format == "torch_dist":
            self.experts = TEGroupedMLP(
                self.num_local_experts, self.config, self.submodules.experts
            )
        else:
            self.experts = GroupedMLP(self.num_local_experts, self.config)

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = SharedExpertMLP(self.config, self.submodules.shared_experts)
        else:
            self.shared_experts = None

        if self.config.flash_moe_cpu_offload:
          assert act_commit_offload_handle is not None
          assert act_commit_reload_handle is not None

        self.options = flash_moe.nn.MixtureOfExpertsOptions(
            w1_is_transposed=False if isinstance(self.experts, GroupedMLP) else True,
            w2_is_transposed=False if isinstance(self.experts, GroupedMLP) else True,
            fuse_gradient_accumulation=self.config.flash_moe_ga_fusion,
            use_interleaved_expert_parallel=self.config.flash_moe_interleaving,
            recompute_activations=self.config.flash_moe_mem_saving_mlp,
            offload_activations=self.config.flash_moe_cpu_offload,
            offload_context=act_offload_context,
            offload_handle=act_commit_offload_handle,
            reload_handle=act_commit_reload_handle,
        )


    def forward(self, hidden_states: torch.Tensor):
      scores, indices = self.router(hidden_states)
      if isinstance(self.experts, GroupedMLP):
          grouped_weight_1 = self.experts.weight1
          grouped_weight_2 = self.experts.weight2
      else:
          fc1_param_names = sorted(self.experts.linear_fc1._parameters.keys(), key=lambda x: int(x.split('weight')[1]))
          fc2_param_names = sorted(self.experts.linear_fc2._parameters.keys(), key=lambda x: int(x.split('weight')[1]))
          grouped_weight_1 = [self.experts.linear_fc1._parameters[key] for key in fc1_param_names]
          grouped_weight_2 = [self.experts.linear_fc2._parameters[key] for key in fc2_param_names]

      def apply_moe(features, scores, indices):
          if self.config.moe_pad_expert_input_to_capacity:
              return flash_moe.nn.functional.mixture_of_experts_with_padding(
                  features,
                  indices,
                  scores,
                  grouped_weight_1,
                  grouped_weight_2,
                  self.shared_experts.linear_fc1.weight if self.shared_experts is not None else None,
                  self.shared_experts.linear_fc2.weight if self.shared_experts is not None else None,
                  self.shared_experts.gate_weight if self.shared_experts is not None else None,
                  options=self.options,
              )

          return flash_moe.nn.functional.mixture_of_experts(
              features,
              indices,
              scores,
              grouped_weight_1,
              grouped_weight_2,
              self.shared_experts.linear_fc1.weight if self.shared_experts is not None else None,
              self.shared_experts.linear_fc2.weight if self.shared_experts is not None else None,
              self.shared_experts.gate_weight if self.shared_experts is not None else None,
              options=self.options,
          )

      # In current mcore DP allgather overlapping implementation, it assumes
      # all modules should call hooks for local parameters in their own
      # forward function.
      for name, module in self.named_modules():
          if name and 'router' not in name:
              for pre_hook in module._forward_pre_hooks.values():  # noqa: SLF001
                  pre_hook(module)

      if self.moe_layer_recompute:
          output = tensor_parallel.checkpoint(
              apply_moe, False,
              hidden_states, scores, indices)
      else:
          output = apply_moe(hidden_states, scores, indices)

      # In current mcore DP allgather overlapping implementation, it assumes
      # all modules should call hooks for local parameters in their own
      # forward function.
      for name, module in self.named_modules():
          if name and 'router' not in name:
              for hook in module._forward_hooks.values():  # noqa: SLF001
                  hook(module)
      return output, None
