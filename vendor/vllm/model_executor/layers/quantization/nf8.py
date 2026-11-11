# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional

import torch
from torch.nn import Module

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    bladnn_moe_quant_config,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8KVCacheMethod,
    Fp8LinearMethod,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import set_weight_attrs
from vllm.scalar_type import ScalarType, scalar_types

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


class NF8Config(QuantizationConfig):
    """Config class for NF8."""

    def __init__(
        self,
        is_checkpoint_nf8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[list[str]] = None,
        weight_block_size: Optional[list[int]] = None,
        weight_per_tensor: Optional[bool] = False,
        activation_per_tensor: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.is_checkpoint_nf8_serialized = is_checkpoint_nf8_serialized
        self.is_checkpoint_fp8_serialized = is_checkpoint_nf8_serialized  # support nf8
        assert is_checkpoint_nf8_serialized

        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions"
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    "The block-wise quantization only supports "
                    "dynamic activation scheme for now, but got "
                    f"{activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size
        self.weight_per_tensor = weight_per_tensor
        self.activation_per_tensor = activation_per_tensor
        self.attach_model_quantization_config = None

    @classmethod
    def get_name(cls) -> str:
        return "nf8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NF8Config":
        quant_method = cls.get_from_keys_or(config, ["quant_method"], "")  # nf8
        is_checkpoint_nf8_serialized = "nf8" in quant_method
        activation_scheme = cls.get_from_keys_or(
            config, ["activation_scheme"], "dynamic"
        )
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        weight_per_tensor = cls.get_from_keys_or(config, ["weight_per_tensor"], False)
        activation_per_tensor = cls.get_from_keys_or(
            config, ["act_per_tensor", "activation_per_tensor"], False
        )
        return cls(
            is_checkpoint_nf8_serialized=is_checkpoint_nf8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            weight_per_tensor=weight_per_tensor,
            activation_per_tensor=activation_per_tensor,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)  # still FP8
        elif isinstance(layer, FusedMoE):
            return NF8MoEMethod(self)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def __repr__(self):
        return (
            f"NF8Config("
            f"is_checkpoint_nf8_serialized={self.is_checkpoint_nf8_serialized}, "
            f"activation_scheme={self.activation_scheme}, "
            f"ignored_layers={self.ignored_layers}, "
            f"weight_block_size={self.weight_block_size}, "
            f"weight_per_tensor={self.weight_per_tensor}, "
            f"activation_per_tensor={self.activation_per_tensor})"
        )


class NF8MoEMethod(FusedMoEMethodBase):
    """MoE method for NF8.
    Supports loading NF8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: NF8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None
        assert envs.VLLM_MOE_USE_BLADNN
        import bladnn

        self.bladnn_dynamic_quantize_fp8 = (
            bladnn.DynamicQuantizeFP8()
        )  # unchange. still dq for A

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert not self.block_quant
        # WEIGHTS
        assert self.quant_config.is_checkpoint_nf8_serialized

        w13_weight_data = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // 2,  # k//2: pack in k-dim, each element in k contain 2 fp4
            dtype=torch.int8,
        )
        # weight_n = self.out_features  (h)
        # weight_shape = (
        #     self.num_experts, weight_n, self.input_size_per_partition // 2)
        w2_weight_data = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // 2,
            dtype=torch.int8,
        )

        w13_weight = torch.nn.Parameter(w13_weight_data, requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(w2_weight_data, requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.quant_config.activation_scheme == "dynamic"
        # scale_shape = (self.num_experts, self.output_size_per_partition)
        w13_weight_scale_shape = (num_experts, w13_weight.shape[1])  # (e,n) for bladnn
        w2_weight_scale_shape = (num_experts, w2_weight.shape[1])
        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(*w13_weight_scale_shape, dtype=params_dtype), requires_grad=False
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(*w2_weight_scale_shape, dtype=params_dtype), requires_grad=False
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # nf8: use per-channel quant
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        # loading nf8 checkpoint, pass the weight loaders.
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # WEIGHT_SIGN_MAP:
        w13_weight_sign_map_data = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,  # n
            hidden_size // 32,  # k//32
            dtype=torch.int32,
        )
        # weight_n = self.out_features  (h)
        # weight_shape = (self.num_experts, weight_n,
        #                 self.input_size_per_partition // 2)
        # sign_map_shape = (self.num_experts, self.out_features,
        #                   self.input_size_per_partition // 32)
        w2_weight_sign_map_data = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // 32,
            dtype=torch.int32,
        )

        w13_sign_map = torch.nn.Parameter(w13_weight_sign_map_data, requires_grad=False)
        layer.register_parameter("w13_sign_map", w13_sign_map)
        set_weight_attrs(w13_sign_map, extra_weight_attrs)

        w2_sign_map = torch.nn.Parameter(w2_weight_sign_map_data, requires_grad=False)
        layer.register_parameter("w2_sign_map", w2_sign_map)
        set_weight_attrs(w2_sign_map, extra_weight_attrs)

        # Weight not requires after-loading processing
        layer._nf8_quantized = True
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_before_loading(self, layer: torch.nn.Module) -> None:
        if not envs.VLLM_QUANTIZATION_LAYER_WISE:
            return

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_nf8_quantized", False):
            return
        layer._nf8_quantized = True

        assert self.quant_config.is_checkpoint_nf8_serialized

        return

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> Optional[FusedMoEQuantConfig]:
        return bladnn_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_sign_map=layer.w13_sign_map,
            w2_sign_map=layer.w2_sign_map,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )
        if envs.VLLM_MOE_BALANCED_GATING:
            topk_ids = layer.topk_ids[: topk_ids.shape[0]]

        if layer.expert_mapping_func is not None:
            topk_ids = layer.expert_mapping_func(topk_ids)

        # NOTE: not consider ep yet
        fused_experts_forward = fused_experts

        return fused_experts_forward(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=layer.activation,
            use_fp8_w8a8=True,  # same fp8_dp for A
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
            block_shape=self.quant_config.weight_block_size,
        )
