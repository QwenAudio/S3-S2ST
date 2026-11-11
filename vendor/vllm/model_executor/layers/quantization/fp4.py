# SPDX-License-Identifier: Apache-2.0
import copy
import functools
import importlib.util
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from torch.nn import Module

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_data_parallel_world_size,
    get_expert_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
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
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8KVCacheMethod,
    Fp8LinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)
has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None


class FP4Config(QuantizationConfig):
    """Config class for FP4."""

    def __init__(
        self,
        is_checkpoint_fp4_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[list[str]] = None,
        weight_block_size: Optional[list[int]] = None,
        weight_per_tensor: Optional[bool] = False,
        activation_per_tensor: Optional[bool] = False,
        fc1_quant_method: Optional[str] = "fp4_subchannel",
        fc2_quant_method: Optional[str] = "fp8",
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp4_serialized = is_checkpoint_fp4_serialized
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp4_serialized
        assert is_checkpoint_fp4_serialized

        self.activation_scheme = activation_scheme
        # fc1: [fp4_subchannel, fp4_perchannel]
        self.fc1_quant_method = fc1_quant_method
        # fc2: [fp8, fp4_subchannel, fp4_perchannel, fp4_perchannel_group_pack]
        self.fc2_quant_method = fc2_quant_method
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
        return "fp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FP4Config":
        quant_method = cls.get_from_keys_or(config, ["quant_method"], "")
        is_checkpoint_fp4_serialized = "fp4" in quant_method
        activation_scheme = cls.get_from_keys_or(
            config, ["activation_scheme"], "dynamic"
        )
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        weight_per_tensor = cls.get_from_keys_or(config, ["weight_per_tensor"], False)
        activation_per_tensor = cls.get_from_keys_or(
            config, ["act_per_tensor", "activation_per_tensor"], False
        )
        fc1_quant_method = cls.get_from_keys_or(
            config, ["fc1_quant_method"], "fp4_subchannel"
        )
        fc2_quant_method = cls.get_from_keys_or(config, ["fc2_quant_method"], "fp8")
        return cls(
            is_checkpoint_fp4_serialized=is_checkpoint_fp4_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            weight_per_tensor=weight_per_tensor,
            activation_per_tensor=activation_per_tensor,
            fc1_quant_method=fc1_quant_method,
            fc2_quant_method=fc2_quant_method,
        )

    def is_layer_blockwise_quant(self, prefix: str) -> bool:
        if envs.VLLM_FP8_USE_BLADNN and envs.VLLM_MLA_KEEP_BLOCKWISE_QUANT:
            mla_layer_keys = [
                "q_proj",
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj",
                "fused_qkv_a_proj",
                "indexer.wq_b",
                "indexer.wk",
                "indexer.k_norm",
                "indexer.weights_proj",
            ]
            return any(key in prefix for key in mla_layer_keys)
        return False

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        vllm_config = get_current_vllm_config()
        if isinstance(layer, LinearBase):
            if prefix.endswith(".mlp.gate"):  # bladnn fp4 doesn't quantize gate.
                return UnquantizedLinearMethod()
            if (vllm_config.model_config.use_mla
                and self.is_layer_blockwise_quant(prefix)
            ):
                # still use block-wise fp8 for MLA attn.
                quant_config = copy.copy(self)
                quant_config.weight_block_size = [128, 128]
                return Fp8LinearMethod(quant_config)
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if envs.VLLM_QUANTIZE_ROUTED_EXPERTS_ONLY:
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(
                self
            )  # still FP8  shared, use bladnn perchannel fp8, needs VLLM_FP8_USE_BLADNN
        elif isinstance(layer, FusedMoE):
            return FP4MoEMethod(self, layer)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def __repr__(self):
        return (
            f"FP4Config("
            f"is_checkpoint_fp4_serialized={self.is_checkpoint_fp4_serialized}, "
            f"activation_scheme={self.activation_scheme}, "
            f"ignored_layers={self.ignored_layers}, "
            f"weight_block_size={self.weight_block_size}, "
            f"weight_per_tensor={self.weight_per_tensor}, "
            f"activation_per_tensor={self.activation_per_tensor}, "
            f"fc1_quant_method={self.fc1_quant_method}, "
            f"fc2_quant_method={self.fc2_quant_method})"
        )


class FP4MoEMethod(FusedMoEMethodBase):
    """MoE method for FP4.
    Supports loading FP4 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: FP4Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)

        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None
        assert envs.VLLM_MOE_USE_BLADNN, "FP4MoEMethod requires bladnn."

        import bladnn

        self.bladnn_dynamic_quantize_fp8 = (
            bladnn.DynamicQuantizeFP8()
        )  # unchange. still dq for A

        self.allow_deep_gemm = False
        self.fused_experts_func = functools.partial(  # type: ignore
            self.resolve_fused_experts(), allow_deep_gemm=self.allow_deep_gemm
        )

    @functools.lru_cache  # noqa: B019
    def resolve_fused_experts(self):
        vllm_config = get_current_vllm_config()
        use_ep = (
            vllm_config.parallel_config.enable_expert_parallel
            and get_expert_parallel_world_size() > 1
        )
        from vllm.model_executor.layers.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.expert_parallel import (
            parallel_experts_with_data_parallel,
            parallel_experts_with_data_parallel_deepep,
            parallel_experts_with_data_parallel_multi_alltoall,
            parallel_experts_without_data_parallel,
        )

        if not use_ep:
            logger.info_once("Using fused_experts for non-expert parallel.")
            fused_experts_forward = fused_experts
        elif envs.VLLM_MOE_USE_DEEPEP:
            logger.info_once(
                "Using parallel_experts_with_data_parallel_deepep "
                "as VLLM_MOE_USE_DEEPEP is set."
            )
            fused_experts_forward = parallel_experts_with_data_parallel_deepep
        elif get_data_parallel_world_size() <= 1:
            logger.info_once(
                "Using parallel_experts_without_data_parallel for non-data parallel."
            )
            fused_experts_forward = parallel_experts_without_data_parallel
        elif get_tensor_model_parallel_world_size() > 1:
            logger.info_once(
                "Using parallel_experts_with_data_parallel_multi_alltoall "
                "for data parallel + tensor parallel."
            )
            fused_experts_forward = parallel_experts_with_data_parallel_multi_alltoall
        else:
            logger.info_once(
                "Using parallel_experts_with_data_parallel for data parallel."
            )
            fused_experts_forward = parallel_experts_with_data_parallel
        return fused_experts_forward

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
        assert self.quant_config.is_checkpoint_fp4_serialized
        if envs.VLLM_USE_QUICK_LOADER:
            w13_weight_data = torch.empty(num_experts, 1, 1, dtype=torch.int8)
            if self.quant_config.fc2_quant_method.startswith("fp4"):
                w2_weight_data = torch.empty(num_experts, 1, 1, dtype=torch.int8)
            elif self.quant_config.fc2_quant_method == "fp8":
                w2_weight_data = torch.empty(
                    num_experts, 1, 1, dtype=torch.float8_e4m3fn
                )
        else:
            w13_weight_data = torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size
                // 2,  # k//2: pack in k-dim, each element in k contain 2 fp4
                dtype=torch.int8,
            )
            # weight_n = self.out_features  (h)
            # weight_shape = (self.num_experts, weight_n, self.input_size_per_partition)
            if self.quant_config.fc2_quant_method == "fp4_perchannel_group_pack":
                w2_weight_data = torch.empty(
                    num_experts,
                    hidden_size // 2,
                    intermediate_size_per_partition,
                    dtype=torch.int8,
                )
            elif self.quant_config.fc2_quant_method.startswith(
                "fp4"
            ):  # fp4_perchannel or fp4_subchannel
                w2_weight_data = torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // 2,
                    dtype=torch.int8,
                )
            elif self.quant_config.fc2_quant_method == "fp8":
                w2_weight_data = torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    dtype=torch.float8_e4m3fn,
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
        if self.quant_config.fc1_quant_method == "fp4_perchannel":
            w13_weight_scale_shape = (
                num_experts,
                w13_weight.shape[1],
            )  # (e, n) for bladnn
        else:
            w13_weight_scale_shape = (
                num_experts,
                (w13_weight.shape[2] * 2 // 128),
                w13_weight.shape[1],
            )  # (e, k//128 ,n) for bladnn
        if self.quant_config.fc2_quant_method.startswith("fp4"):
            w2_scale_dtype = params_dtype
            if self.quant_config.fc2_quant_method == "fp4_perchannel_group_pack":
                w2_weight_scale_shape = (
                    num_experts,
                    w2_weight.shape[1] * 2,
                )
            elif self.quant_config.fc2_quant_method == "fp4_perchannel":
                w2_weight_scale_shape = (
                    num_experts,
                    w2_weight.shape[1],
                )
            else:  # fp4_subchannel
                w2_weight_scale_shape = (
                    num_experts,
                    (w2_weight.shape[2] * 2 // 128),
                    w2_weight.shape[1],
                )
        elif self.quant_config.fc2_quant_method == "fp8":  # perchannel
            w2_weight_scale_shape = (num_experts, w2_weight.shape[1])
            w2_scale_dtype = torch.float16
        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(*w13_weight_scale_shape, dtype=params_dtype), requires_grad=False
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(*w2_weight_scale_shape, dtype=w2_scale_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # fp4: use per-channel quant
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        # loading fp4 checkpoint, pass the weight loaders.
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight not requires after-loading processing
        layer._fp4_quantized = True
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_before_loading(self, layer: torch.nn.Module) -> None:
        if not envs.VLLM_QUANTIZATION_LAYER_WISE:
            return

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_fp4_quantized", False):
            return
        layer._fp4_quantized = True

        assert self.quant_config.is_checkpoint_fp4_serialized

        return

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> Optional[FusedMoEQuantConfig]:
        return bladnn_moe_quant_config(
            w1_dtype=layer.w13_weight.dtype,
            w2_dtype=layer.w2_weight.dtype,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int4_w4a16=False,
            use_int8_w8a16=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_callback: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        select_result = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        # Note: the order of checks is important since self.fused_experts
        # can override fused_experts or cutlass but not rocm or marlin.
        #
        (
            topk_weights,
            topk_ids,
            zero_expert_result,
            topk_weights_constant,
            topk_ids_constant,
        ) = select_result

        topk_ids = layer.try_fluid_eplb(topk_ids)
        self.topk_weights_constant = topk_weights_constant
        self.topk_ids_constant = topk_ids_constant

        return self.fused_experts_func(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
            **{
                "shared_experts_callback": shared_experts_callback,
            }
            if shared_experts_callback is not None
            else {},
            map_for_skipped=layer.map_for_skipped,
        )
