import functools
import os
from typing import Any, Callable, Optional, Union

import torch
from torch.nn.parameter import Parameter

try:
    import bladnn
except (ImportError, OSError):
    bladnn = None

from typing import TYPE_CHECKING

import triton
import triton.language as tl

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
from vllm.model_executor.layers.fused_moe.expert_parallel import (
    get_ppu_deep_gemm_config,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    set_weight_attrs,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    bladnn_quantization_padding,
    is_layer_skipped,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.parameter import ChannelQuantScaleParameter
from vllm.utils.deep_gemm import (
    int8_gemm_nt,
    is_deep_gemm_supported,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.platform_utils import is_ppu

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper
    
from vllm.model_executor.layers.quantization.gptq_bladnn import (
    BladnnA8W8LinearMethod,
    BladnnA8W8MoEMethod,
)

logger = init_logger(__name__)


class BladnnA8W4Int4Config(QuantizationConfig):
    """Config class for Bladnn A8W4 Int4"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        ignored_layers: list[str] | None = None,
        fallback_layers: list[int] | None = None,
        fc2_quantization_method: Optional[str] = "int8",
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.pack_factor = 1  # TBD
        self.dynamic_quant_input = True
        self.ignored_layers = ignored_layers or []
        self.fallback_layers = fallback_layers or []
        self.fc2_quantization_method = fc2_quantization_method

    def __repr__(self) -> str:
        return (
            f"BladnnA8W4Int4Config(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "bladnn_a8w4_int4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def is_bladnn_a8w4_int4_compatible(cls, quant_config: dict[str, Any]):
        quant_method = quant_config.get("quant_method", "").lower()
        if quant_method and quant_method not in ["bladnn_a8w4_int4"]:
            return False

        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        return not (num_bits != 4 or group_size != -1)


    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        can_convert = cls.is_bladnn_a8w4_int4_compatible(hf_quant_cfg)
        is_valid_user_quant = user_quant is None or user_quant == "bladnn_a8w4_int4"

        if can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to Bladnn a8w4_int4 format. "
                "Using Bladnn A8W4 INT4 kernel."
            )
            logger.info(msg)
            return cls.get_name()

        return None
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BladnnA8W4Int4Config":
        weight_bits = cls.get_from_keys(config, ["bits", "weight_bits"]) # 4
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = config.get("desc_act", False)

        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        fallback_layers = cls.get_from_keys_or(config, ["fallback_layers"], [])
        fc2_quantization_method = cls.get_from_keys_or(
            config, ["fc2_quant_method"], "int8"
        )
        return cls(
            weight_bits=weight_bits,
            group_size=group_size,
            desc_act=desc_act,
            ignored_layers=ignored_layers,
            fallback_layers=fallback_layers,
            fc2_quantization_method=fc2_quantization_method,
        )
    
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str = None
    ) -> Optional[Union["BladnnA8W8LinearMethod", "BladnnA8W4Int4MoEMethod"]]:
        if isinstance(layer, LinearBase):
            if envs.VLLM_QUANTIZE_ROUTED_EXPERTS_ONLY:
                return UnquantizedLinearMethod()
            if is_layer_skipped(prefix, self.ignored_layers):# attn 
                return UnquantizedLinearMethod()
            if prefix.endswith(".mlp.gate"):  # bladnn int8 doesn't quantize gate.
                return UnquantizedLinearMethod()
            return BladnnA8W8LinearMethod(self)  # denselinear still use int8 by default
        elif isinstance(layer, FusedMoE):
            layer_id = extract_layer_index(prefix)
            if layer_id in self.fallback_layers:   # some layers still use int8
                return BladnnA8W8MoEMethod(self, layer)
            else:
                return BladnnA8W4Int4MoEMethod(self, layer)
        return None

class BladnnA8W4Int4MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: BladnnA8W4Int4Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)

        if quant_config.group_size not in [-1]:
            raise ValueError("Bladnn FusedMoE_A8W4_: group_size only support [-1]")
        self.quant_config = quant_config
        assert envs.VLLM_MOE_USE_BLADNN
        self.bladnn_dynamic_quantize = bladnn.DynamicQuantize()
        self.weight_scale_dtype = torch.float16  # fixed now, like int8

        self.allow_deep_gemm = False
        # Check for DeepGemm support.
        if envs.VLLM_USE_DEEP_GEMM:
            if not has_deep_gemm():
                logger.warning_once("Failed to import DeepGemm kernels.")
            elif is_deep_gemm_supported():
                logger.info_once("Using DeepGemm kernels for BladnnA8W8MoEMethod.")
                self.allow_deep_gemm = True
            else:
                logger.warning_once("DeepGemm not supported on the current platform.")

        self.fused_experts_func = functools.partial(  # type: ignore
            self.resolve_fused_experts(),
            allow_deep_gemm=self.allow_deep_gemm,
        )

    @functools.lru_cache
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
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # (e, n//2 ,k) int8, packed in n-dimension
        assert (intermediate_size_per_partition) % 128 == 0  # noqa n-dim may split into different ranks
        w13_weight_data = torch.empty(
            num_experts,
            intermediate_size_per_partition,# noqa origin = 2 * intermediate_size_per_partition
            hidden_size,
            dtype=torch.int8,
        )
        
        # w2 still use int8 a8w8 in fc2 by default
        if self.quant_config.fc2_quantization_method == "int4":
            assert ((hidden_size // 2) * 2) % 128 == 0
            w2_weight_data = torch.empty(
                num_experts, hidden_size // 2, intermediate_size_per_partition, dtype=torch.int8 # noqa
            )
        elif self.quant_config.fc2_quantization_method == "int8":
            w2_weight_data = torch.empty( # each contain one element, no pack in n-dim
                num_experts, hidden_size, intermediate_size_per_partition, dtype=torch.int8 # noqa
            )

        w13_weight = torch.nn.Parameter(w13_weight_data, requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(w2_weight_data, requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # weigth scale (e,n) fp16
        w13_weight_scale_shape = (num_experts, w13_weight.shape[1] * 2) # noqa w13_weight,n-dim is half_n
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(*w13_weight_scale_shape, dtype=torch.float16),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        if self.quant_config.fc2_quantization_method == "int4":
            w2_weight_scale_shape = (num_experts, w2_weight.shape[1] * 2)
        elif self.quant_config.fc2_quantization_method == "int8":  # perchannel
            w2_weight_scale_shape = (num_experts, w2_weight.shape[1])

        w2_weight_scale = torch.nn.Parameter( # (e, n) fp16
            torch.ones(*w2_weight_scale_shape, dtype=torch.float16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)


        layer._int4_quantized = True
        # dynamic scale
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_before_loading(self, layer: torch.nn.Module) -> None:
            return

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert layer._int4_quantized
        assert envs.VLLM_MOE_USE_BLADNN


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
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int4_w4a16=False,
            use_int8_w8a16=False,
            use_int4_w4a8=True,
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
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            renormalize=layer.renormalize,  # add to algn with non-deepep interface
            inplace=True,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
            **{
                "shared_experts_callback": shared_experts_callback,
            } if shared_experts_callback is not None else {},
            map_for_skipped=layer.map_for_skipped,
        )
