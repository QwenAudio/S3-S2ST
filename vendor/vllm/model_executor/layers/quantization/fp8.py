# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import functools
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.attention.layer import Attention
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_data_parallel_world_size,
    get_expert_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEActivationFormat,
    FusedMoEMethodBase,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    apply_flashinfer_per_tensor_scale_fp8,
    build_flashinfer_fp8_cutlass_moe_prepare_finalize,
    flashinfer_cutlass_moe_fp8,
    get_flashinfer_moe_backend,
    register_moe_scaling_factors,
    rotate_flashinfer_fp8_moe_weights,
    select_cutlass_fp8_gemm_impl,
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    deepgemm_post_process_fp8_weight_block,
    maybe_post_process_fp8_weight_block,
    process_fp8_weight_block_strategy,
    process_fp8_weight_tensor_strategy,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
    prepare_moe_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    deepgemm_quantization_padding,
    is_layer_skipped,
    per_block_cast_to_fp8,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp,
    all_close_1d,
    cutlass_block_fp8_supported,
    cutlass_fp8_supported,
    maybe_create_device_identity,
    normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize,
)
from vllm.model_executor.model_loader.weight_utils import get_model_loading_tracker
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.deep_gemm import (
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)
from vllm.utils.flashinfer import has_flashinfer_moe
from vllm.utils.import_utils import has_deep_gemm

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


class Fp8MoeBackend(Enum):
    NONE = 0
    FLASHINFER_TRTLLM = 1
    FLASHINFER_CUTLASS = 2
    DEEPGEMM = 3
    CUTLASS_BLOCK_SCALED_GROUPED_GEMM = 4
    MARLIN = 5
    TRITON = 6


def get_fp8_moe_backend(
    block_quant: bool,
    moe_parallel_config: FusedMoEParallelConfig,
    with_lora_support: bool,
) -> Fp8MoeBackend:
    """
    Select the primary FP8 MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """
    if with_lora_support:
        return Fp8MoeBackend.TRITON
    # Prefer FlashInfer backends on supported GPUs; allow SM90 and SM100.
    if (
        current_platform.is_cuda()
        and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability(90)
        )
        and envs.VLLM_USE_FLASHINFER_MOE_FP8
        and has_flashinfer_moe()
    ):
        backend = get_flashinfer_moe_backend()
        if backend == FlashinferMoeBackend.TENSORRT_LLM:
            logger.info_once("Using FlashInfer FP8 MoE TRTLLM backend for SM100")
            return Fp8MoeBackend.FLASHINFER_TRTLLM
        else:
            if block_quant and current_platform.is_device_capability_family(100):
                raise ValueError(
                    "FlashInfer FP8 MoE throughput backend does not "
                    "support block quantization. Please use "
                    "VLLM_FLASHINFER_MOE_BACKEND=latency "
                    "instead."
                )
            logger.info_once("Using FlashInfer FP8 MoE CUTLASS backend for SM90/SM100")
            return Fp8MoeBackend.FLASHINFER_CUTLASS

    # weight-only path for older GPUs without native FP8
    use_marlin = (
        not current_platform.has_device_capability(89)
        or envs.VLLM_TEST_FORCE_FP8_MARLIN
    )
    if current_platform.is_rocm():
        use_marlin = False
    if use_marlin:
        logger.info_once("Using Marlin backend for FP8 MoE")
        return Fp8MoeBackend.MARLIN

    # Determine if we should use DeepGEMM with block-quantized weights:
    # - If explicitly set by user, respect their choice
    # - If not explicitly set (default), disable when TP size is >= 8
    moe_use_deep_gemm = envs.VLLM_MOE_USE_DEEP_GEMM
    if not envs.is_set("VLLM_MOE_USE_DEEP_GEMM") and moe_parallel_config.tp_size >= 8:
        moe_use_deep_gemm = False
        logger.info_once(
            "DeepGEMM MoE is disabled by default when TP size is >= 8. "
            "Set VLLM_MOE_USE_DEEP_GEMM=1 to enable it.",
            scope="local",
        )

    if envs.VLLM_USE_DEEP_GEMM and moe_use_deep_gemm and block_quant:
        if not has_deep_gemm():
            logger.warning_once(
                "DeepGEMM backend requested but not available.", scope="local"
            )
        elif is_deep_gemm_supported():
            logger.info_once("Using DeepGEMM backend for FP8 MoE", scope="local")
            return Fp8MoeBackend.DEEPGEMM

    # CUTLASS BlockScaled GroupedGemm on SM100 with block-quantized weights
    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and block_quant
    ):
        logger.info_once(
            "Using Cutlass BlockScaled GroupedGemm backend for FP8 MoE", scope="local"
        )
        return Fp8MoeBackend.CUTLASS_BLOCK_SCALED_GROUPED_GEMM

    # default to Triton
    if envs.VLLM_MOE_USE_BLADNN:
        logger.info_once("Using BLADNN backend for FP8 MoE")
    else:
        logger.info_once("Using Triton backend for FP8 MoE")
    return Fp8MoeBackend.TRITON


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = (
            [128, 128] if envs.VLLM_USE_DEEP_GEMM else None
        ),
        weight_per_tensor: bool = True,
        activation_per_tensor: bool = True,
    ) -> None:
        super().__init__()

        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        weight_block_size = weight_block_size or None  # [] -> None
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

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fp8"

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
    def from_config(cls, config: dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys_or(config, ["quant_method"], "")
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys_or(
            config, ["activation_scheme"], "dynamic"
        )
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(
            config,
            ["weight_block_size"],
            [128, 128] if envs.VLLM_USE_DEEP_GEMM else None,
        )
        weight_per_tensor = cls.get_from_keys_or(config, ["weight_per_tensor"], True)
        activation_per_tensor = cls.get_from_keys_or(
            config, ["act_per_tensor", "activation_per_tensor"], True
        )
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            weight_per_tensor=weight_per_tensor,
            activation_per_tensor=activation_per_tensor,
        )

    def get_xpu_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.model_executor.layers.quantization.ipex_quant import (
            XPUFp8LinearMethod,
            XPUFp8MoEMethod,
        )

        fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=self.is_checkpoint_fp8_serialized,
            activation_scheme=self.activation_scheme,
            ignored_layers=self.ignored_layers,
            weight_block_size=self.weight_block_size,
        )

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return XPUFp8LinearMethod(fp8_config)
        elif isinstance(layer, FusedMoE):
            return XPUFp8MoEMethod(fp8_config, layer)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

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
        if current_platform.is_xpu():
            return self.get_xpu_quant_method(layer, prefix)
        vllm_config = get_current_vllm_config()
        if isinstance(layer, LinearBase):
            # bladnn fp8 doesn't quantize gate.
            if prefix.endswith(".mlp.gate") and (
                envs.VLLM_MOE_USE_BLADNN  # skip gate for bladnn fp8.
                or self.weight_block_size  # skip gate for blockwise fp8.
            ):
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
            quant_method = Fp8LinearMethod(self)
            quant_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
            return quant_method
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            moe_quant_method = Fp8MoEMethod(self, layer)
            moe_quant_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
            return moe_quant_method
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_cache_scale(self, name: str) -> str | None:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        if name.endswith(".output_scale") and ".q_proj" in name:
            return name.replace(".q_proj.output_scale", ".attn.q_scale")
        if name.endswith("self_attn.prob_output_scale"):
            return name.replace(".prob_output_scale", ".attn.prob_scale")
        # If no matches, return None
        return None

    def __repr__(self):
        return (
            f"Fp8Config("
            f"is_checkpoint_fp8_serialized={self.is_checkpoint_fp8_serialized}, "
            f"activation_scheme={self.activation_scheme}, "
            f"ignored_layers={self.ignored_layers}, "
            f"weight_block_size={self.weight_block_size}, "
            f"weight_per_tensor={self.weight_per_tensor}, "
            f"activation_per_tensor={self.activation_per_tensor})"
        )


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    loading_layers: dict[str, list[tuple["Fp8LinearMethod", torch.nn.Module]]] = (
        defaultdict(list)
    )

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.out_dtype = torch.get_default_dtype()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.marlin_input_dtype = None
        self.use_marlin = (
            not current_platform.has_device_capability(89)
            or envs.VLLM_TEST_FORCE_FP8_MARLIN
        )
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False
        if vllm_is_batch_invariant():
            self.use_marlin = False

        self.use_aiter_and_is_supported = rocm_aiter_ops.is_linear_fp8_enaled()
        self.use_deep_gemm = is_deep_gemm_supported()

        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None

        if not self.block_quant and envs.VLLM_FP8_USE_BLADNN:
            import bladnn

            self.bladnn_dynamic_quantize_fp8 = bladnn.DynamicQuantizeFP8()
            self.bladnn_gemm_op = bladnn.GemmFP8FusedDq()

            self.weight_scale_dtype = torch.float16
        else:
            self.weight_scale_dtype = torch.float32

        self.act_q_static = self.quant_config.activation_scheme == "static"
        if self.weight_block_size:
            self.act_q_group_shape = GroupShape(1, self.weight_block_size[0])
        else:
            # Use per-token quantization for better perf if dynamic and cutlass
            if not self.act_q_static and cutlass_fp8_supported():
                self.act_q_group_shape = GroupShape.PER_TOKEN
            else:
                self.act_q_group_shape = GroupShape.PER_TENSOR

        if self.block_quant:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            self.w8a8_block_fp8_linear = W8A8BlockFp8LinearOp(
                weight_group_shape=GroupShape(*self.weight_block_size),
                act_quant_group_shape=self.act_q_group_shape,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
                use_aiter_and_is_supported=self.use_aiter_and_is_supported,
            )
        else:
            self.fp8_linear = Fp8LinearOp(
                act_quant_static=self.act_q_static,
                act_quant_group_shape=self.act_q_group_shape,
            )

        if not self.quant_config.is_checkpoint_fp8_serialized:
            # Initialize tracker
            _ = get_model_loading_tracker(do_init=True)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        if (envs.VLLM_USE_QUICK_LOADER) or (
            envs.VLLM_QUANTIZATION_LAYER_WISE
            and not self.quant_config.is_checkpoint_fp8_serialized
        ):
            weight_shape = (1, 1)
        else:
            weight_shape = (output_size_per_partition, input_size_per_partition)

        # WEIGHT
        if self.quant_config.is_checkpoint_fp8_serialized:
            weight = create_fp8_weight_parameter(*weight_shape, weight_loader)
        else:

            def patched_weight_loader(param, loaded_weight, *args, **kwargs):
                # load the current weight chunk
                res = weight_loader(param, loaded_weight, *args, **kwargs)  # type: ignore[misc]

                # track how many elements we have updated
                if not hasattr(layer, "_loaded_numel"):
                    layer._loaded_numel = 0
                layer._loaded_numel += loaded_weight.numel()

                # if we have loaded all of the elements, call
                # process_weights_after_loading
                target_loaded_numel = layer.weight.numel()
                if layer._loaded_numel == target_loaded_numel:
                    self.process_weights_after_loading(layer)

                    # Delete the bookkeeping
                    del layer._loaded_numel
                    # Prevent the usual `process_weights_after_loading` call from doing
                    # anything
                    layer._already_called_process_weights_after_loading = True

                return res

            # For non-serialized checkpoints, use original dtype
            weight = ModelWeightParameter(
                data=torch.empty(
                    *weight_shape,
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                # N.B.: don't use this implementation as it doesn't take TP into
                # account, use VLLM_QUANTIZATION_LAYER_WISE instead.
                #
                # weight_loader=patched_weight_loader,
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight", weight)

        # Weight requires after-loading processing
        layer._fp8_quantized = False

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if not self.block_quant:
                if envs.VLLM_FP8_USE_BLADNN and not self.quant_config.weight_per_tensor:
                    scale = create_fp8_scale_parameter(
                        ChannelQuantScaleParameter,
                        output_partition_sizes,
                        input_size_per_partition,
                        None,
                        weight_loader,
                        dtype=torch.float16,
                    )
                else:
                    scale = create_fp8_scale_parameter(
                        PerTensorScaleParameter,
                        output_partition_sizes,
                        input_size_per_partition,
                        None,
                        weight_loader,
                    )
                set_weight_attrs(scale, {"scale_type": "weight_scale"})
                layer.register_parameter("weight_scale", scale)
            else:
                assert not self.act_q_static
                assert self.weight_block_size is not None
                scale = create_fp8_scale_parameter(
                    BlockQuantScaleParameter,
                    output_partition_sizes,
                    input_size_per_partition,
                    self.weight_block_size,
                    weight_loader,
                )
                set_weight_attrs(scale, {"scale_type": "weight_scale"})
                # The weight_scale_inv name is intentional for deepseekv3
                layer.register_parameter("weight_scale_inv", scale)

            # INPUT ACTIVATION SCALE
            if self.act_q_static:
                scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
                set_weight_attrs(scale, {"scale_type": "input_scale"})
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_before_loading(self, layer: torch.nn.Module) -> None:
        if not envs.VLLM_QUANTIZATION_LAYER_WISE or getattr(
            layer, "_fp8_quantized", False
        ):
            return

        if layer.weight.numel() == 1:
            layer.weight.data = torch.empty(
                layer.output_size_per_partition,
                layer.input_size_per_partition,
                dtype=layer.weight.dtype,
                device=layer.weight.device,
            )

        tracker = get_model_loading_tracker()
        if tracker is None:  # layerwise quant is not supported
            return

        loaded_layers = []
        for layer_name in self.loading_layers:
            if tracker.is_layer_finished(layer_name):
                loaded_layers.append(layer_name)
        for layer_name in loaded_layers:
            for linear_method, ready_layer in self.loading_layers.pop(layer_name):
                linear_method.process_weights_after_loading(ready_layer)

        self.loading_layers[self.prefix].append((self, layer))

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if getattr(layer, "_fp8_quantized", False):
            return
        layer._fp8_quantized = True

        size_k_first = True
        input_scale = None
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            assert not self.act_q_static
            size_k_first = False
            if not self.quant_config.is_checkpoint_fp8_serialized:
                if envs.VLLM_USE_DEEP_GEMM:
                    deepgemm_quantization_padding(self, layer)
                weight, weight_scale_inv = per_block_cast_to_fp8(layer.weight.data)
            else:
                weight = layer.weight.data
                weight_scale_inv = layer.weight_scale_inv.data

            weight, weight_scale_inv = process_fp8_weight_block_strategy(
                weight, weight_scale_inv
            )

            # Update layer with new values
            replace_parameter(layer, "weight", weight)
            replace_parameter(layer, "weight_scale_inv", weight_scale_inv)
            
        # If checkpoint not serialized fp8, quantize the weights.
        else:
            if not self.quant_config.is_checkpoint_fp8_serialized:
                # If rocm, use float8_e4m3fnuz as dtype
                fp8_dtype = (
                    torch.float8_e4m3fnuz
                    if current_platform.is_rocm()
                    else torch.float8_e4m3fn
                )

                if envs.VLLM_FP8_USE_BLADNN:
                    qweight, weight_scale = self.bladnn_dynamic_quantize_fp8(
                        layer.weight.data, fp8_dtype
                    )
                    weight = qweight
                else:
                    qweight, weight_scale = ops.scaled_fp8_quant(
                        layer.weight.data, scale=None
                    )
                    weight = qweight.t()
            else:
                weight = layer.weight.data
                weight_scale = layer.weight_scale.data

                 # If using w8a8, torch._scaled_mm needs per tensor, so
                # requantize the logical shards as a single weight.
                if not envs.VLLM_FP8_USE_BLADNN and not self.use_marlin:
                    weight, weight_scale, input_scale = (
                        process_fp8_weight_tensor_strategy(
                            weight.data,
                            weight_scale,
                            layer.logical_widths,
                            getattr(layer, "input_scale", None),
                        )
                    )
                    if self.act_q_static:
                        assert input_scale is not None
                        input_scale = input_scale.max()
                if not envs.VLLM_FP8_USE_BLADNN:
                    weight = weight.t()

            # Release the memory
            layer.weight.data = torch.empty(
                (0,), dtype=layer.weight.dtype, device=layer.weight.device
            )

            # Update layer with new values.
            replace_parameter(layer, "weight", weight)
            replace_parameter(layer, "weight_scale", weight_scale)

        if input_scale is not None:
            replace_parameter(layer, "input_scale", input_scale)
        else:
            layer.input_scale = None

        if self.use_marlin:
            prepare_fp8_layer_for_marlin(
                layer, size_k_first, input_dtype=self.marlin_input_dtype
            )
            # Activations not quantized for marlin.
            del layer.input_scale
            return

        if self.block_quant:
            maybe_post_process_fp8_weight_block(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.numel() == 0:
            if (
                not self.block_quant and envs.VLLM_FP8_USE_BLADNN
            ) or envs.VLLM_USE_DEEP_GEMM:
                out_shape = x.shape[:-1] + (layer.weight.shape[0],)
            else:
                out_shape = x.shape[:-1] + (layer.weight.shape[1],)
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # if batch invariant mode is enabled, prefer DeepGEMM FP8 path
        # we will use BF16 dequant when DeepGEMM is not supported.
        if vllm_is_batch_invariant():
            if self.block_quant:
                assert self.weight_block_size is not None
                return self.w8a8_block_fp8_linear.apply(
                    input=x,
                    weight=layer.weight,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=layer.input_scale,
                    bias=bias,
                )
            else:
                # per-tensor/channel: dequant to BF16 and run GEMM
                weight_fp8 = layer.weight.to(torch.bfloat16)
                weight_scale = layer.weight_scale.to(torch.bfloat16)
                if weight_scale.numel() == 1:
                    # Per-tensor: simple scalar multiplication
                    weight_bf16 = weight_fp8 * weight_scale
                else:
                    # Multiple scales (fused modules like QKV)
                    # Try to infer correct broadcasting
                    # weight is [K, N], scale could be [num_logical_weights]
                    # Need to figure out how to broadcast - for now just try
                    # direct multiplication
                    if (
                        weight_scale.dim() == 1
                        and weight_scale.shape[0] == weight_fp8.shape[0]
                    ):
                        # Per-row scaling
                        weight_bf16 = weight_fp8 * weight_scale.unsqueeze(1)
                    else:
                        # Fallback
                        weight_bf16 = weight_fp8 * weight_scale
                return torch.nn.functional.linear(x, weight_bf16.t(), bias)

        if self.use_marlin:
            if self.block_quant:
                weight_scale = layer.weight_scale_inv
            else:
                weight_scale = layer.weight_scale

            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                input_dtype=self.marlin_input_dtype,
                bias=bias,
            )

        if not self.block_quant and envs.VLLM_FP8_USE_BLADNN:
            if envs.VLLM_BLADNN_GEMM_CHUNK_SIZE > 0:
                output = torch.empty(
                    [x.shape[0], layer.weight.shape[0]], dtype=x.dtype, device=x.device
                )
                num_chunks = (
                    x.shape[0] + envs.VLLM_BLADNN_GEMM_CHUNK_SIZE - 1
                ) // envs.VLLM_BLADNN_GEMM_CHUNK_SIZE
                for i in range(num_chunks):
                    start = i * envs.VLLM_BLADNN_GEMM_CHUNK_SIZE
                    end = min((i + 1) * envs.VLLM_BLADNN_GEMM_CHUNK_SIZE, x.shape[0])
                    output[start:end] = torch.ops.vllm.apply_bladnn_gemm_fp8_fused_dq(
                        self.bladnn_gemm_op.workspace,
                        x[start:end],
                        layer.weight,
                        layer.weight_scale,
                    )
            else:
                output = torch.ops.vllm.apply_bladnn_gemm_fp8_fused_dq(
                    self.bladnn_gemm_op.workspace, x, layer.weight, layer.weight_scale
                )

            if bias is not None:
                output.add_(bias)
            return output

        if self.block_quant:
            assert self.weight_block_size is not None

            return self.w8a8_block_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
            )

        return self.fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=layer.input_scale,
            bias=bias,
        )


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    loading_layers: dict[str, list[tuple["Fp8MoEMethod", torch.nn.Module]]] = (
        defaultdict(list)
    )

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.layer = layer
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.fp8_backend = get_fp8_moe_backend(
            self.block_quant, layer.moe_parallel_config, self.moe.is_lora_enabled
        )

        self.marlin_input_dtype = None
        self.use_marlin = self.fp8_backend == Fp8MoeBackend.MARLIN
        self.flashinfer_moe_backend: FlashinferMoeBackend | None = None
        if self.fp8_backend == Fp8MoeBackend.FLASHINFER_TRTLLM:
            self.flashinfer_moe_backend = FlashinferMoeBackend.TENSORRT_LLM
        elif self.fp8_backend == Fp8MoeBackend.FLASHINFER_CUTLASS:
            self.flashinfer_moe_backend = FlashinferMoeBackend.CUTLASS
            if self.block_quant:
                assert self.weight_block_size == [128, 128], (
                    f"Only support weight_block_size == [128, 128], "
                    f"got {self.weight_block_size}"
                )
            self.flashinfer_moe_fn = partial(
                flashinfer_cutlass_moe_fp8,
                moe=self.moe,
                use_deepseek_fp8_block_scale=self.block_quant,
            )

        self.allow_deep_gemm = self.fp8_backend == Fp8MoeBackend.DEEPGEMM
        self.allow_cutlass_block_scaled_grouped_gemm = (
            self.fp8_backend == Fp8MoeBackend.CUTLASS_BLOCK_SCALED_GROUPED_GEMM
        )

        if not self.block_quant and envs.VLLM_MOE_USE_BLADNN:
            import bladnn

            self.bladnn_dynamic_quantize_fp8 = bladnn.DynamicQuantizeFP8()

            self.weight_scale_dtype = torch.float16
        else:
            self.weight_scale_dtype = torch.float32

        # Check for CutlassBlockScaledGroupedGemm support.
        self.allow_cutlass_block_scaled_grouped_gemm = False
        if not self.block_quant:
            logger.info_once(
                "Model is not block quantized. Not using "
                "CutlassBlockScaledGroupedGemm kernels"
            )
        elif (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            logger.info_once(
                "Using CutlassBlockScaledGroupedGemm kernels for Fp8MoEMethod."
            )
            self.allow_cutlass_block_scaled_grouped_gemm = True
        else:
            logger.warning_once(
                "CutlassBlockScaledGroupedGemm not supported on the current platform."
            )

        self.fused_experts_func = functools.partial(  # type: ignore
            self.resolve_fused_experts(),
            allow_deep_gemm=self.allow_deep_gemm,
            allow_cutlass_block_scaled_grouped_gemm=(
                self.allow_cutlass_block_scaled_grouped_gemm
            ),
        )

        if not self.quant_config.is_checkpoint_fp8_serialized:
            # Initialize tracker
            _ = get_model_loading_tracker(do_init=True)

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn
        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        layer.num_experts = num_experts
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size

        # if we are doing online quantization, patch the weight
        # loaded to call `process_weights_after_loading` in a streaming fashion
        # as soon as the last weight chunk is loaded
        if not self.quant_config.is_checkpoint_fp8_serialized:
            weight_loader = extra_weight_attrs["weight_loader"]
            # create a new holder to prevent modifying behavior of any other
            # objects which might depend on the old one
            new_extra_weight_attrs = extra_weight_attrs

            def patched_weight_loader(param, loaded_weight, *args, **kwargs):
                # load the current weight chunk
                res = weight_loader(param, loaded_weight, *args, **kwargs)  # type: ignore[misc]

                # add a counter to track how many elements we have updated
                if not hasattr(layer, "_loaded_numel"):
                    layer._loaded_numel = 0
                layer._loaded_numel += loaded_weight.numel()

                # if we have loaded all of the elements, call
                # process_weights_after_loading
                target_loaded_numel = layer.w13_weight.numel() + layer.w2_weight.numel()
                if layer._loaded_numel == target_loaded_numel:
                    self.process_weights_after_loading(layer)

                    # Delete the bookkeeping
                    del layer._loaded_numel
                    # Prevent the usual `process_weights_after_loading` call
                    # from doing anything
                    layer._already_called_process_weights_after_loading = True

                return res

            # N.B.: don't use this implementation as it doesn't take TP into account.
            # use VLLM_QUANTIZATION_LAYER_WISE instead.
            #
            # new_extra_weight_attrs["weight_loader"] = patched_weight_loader
            extra_weight_attrs = new_extra_weight_attrs

        # WEIGHTS
        if (envs.VLLM_USE_QUICK_LOADER) or (
            envs.VLLM_QUANTIZATION_LAYER_WISE
            and not self.quant_config.is_checkpoint_fp8_serialized
        ):
            # Placeholders to make loader recognize this parameter
            # Use 8 to fix the misaligned address issue when using bladnn
            w13_weight_data = torch.empty(
                num_experts,
                8,
                8,
                dtype=params_dtype,
            )
            w2_weight_data = torch.empty(
                num_experts,
                8,
                8,
                dtype=params_dtype,
            )
        else:
            w13_weight_data = torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            )
            w2_weight_data = torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            )

        w13_weight = torch.nn.Parameter(w13_weight_data, requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(w2_weight_data, requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if envs.VLLM_USE_QUICK_LOADER:
            w13_weight_shape = torch.Size([num_experts, 8, 8])
            w2_weight_shape = torch.Size([num_experts, 8, 8])
        else:
            w13_weight_shape = torch.Size(
                [num_experts, 2 * intermediate_size_per_partition, hidden_size]
            )
            w2_weight_shape = torch.Size(
                [num_experts, hidden_size, intermediate_size_per_partition]
            )

        # WEIGHT_SCALES
        if not self.block_quant:
            if envs.VLLM_MOE_USE_BLADNN:
                w13_weight_scale_shape = (num_experts, w13_weight_shape[1])
                w2_weight_scale_shape = (num_experts, w2_weight_shape[1])
            else:
                w13_weight_scale_shape = (num_experts, 2)
                w2_weight_scale_shape = (num_experts,)

            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    *w13_weight_scale_shape,
                    dtype=self.weight_scale_dtype,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    *w2_weight_scale_shape,
                    dtype=self.weight_scale_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        else:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        if self.block_quant:
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            )
        elif envs.VLLM_MOE_USE_BLADNN:
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
            )
        else:
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            )
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight requires after-loading processing
        layer._fp8_quantized = False

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

        self.rocm_aiter_moe_enabled = False

    def process_weights_before_loading(self, layer: torch.nn.Module) -> None:
        if not envs.VLLM_QUANTIZATION_LAYER_WISE or getattr(
            layer, "_fp8_quantized", False
        ):
            return

        if layer.w13_weight.numel() == layer.num_experts * 8 * 8:
            layer.w13_weight.data = torch.empty(
                layer.num_experts,
                2 * layer.intermediate_size_per_partition,
                layer.hidden_size,
                dtype=layer.w13_weight.dtype,
                device=layer.w13_weight.device,
            )
        if layer.w2_weight.numel() == layer.num_experts * 8 * 8:
            layer.w2_weight.data = torch.empty(
                layer.num_experts,
                layer.hidden_size,
                layer.intermediate_size_per_partition,
                dtype=layer.w2_weight.dtype,
                device=layer.w2_weight.device,
            )

        tracker = get_model_loading_tracker()
        if tracker is None:  # layerwise quant is not supported
            return

        loaded_layers = []
        for layer_name in self.loading_layers:
            if tracker.is_layer_finished(layer_name):
                loaded_layers.append(layer_name)
        for layer_name in loaded_layers:
            for linear_method, ready_layer in self.loading_layers.pop(layer_name):
                linear_method.process_weights_after_loading(ready_layer)

        self.loading_layers[self.prefix].append((self, layer))

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if getattr(layer, "_fp8_quantized", False):
            return
        layer._fp8_quantized = True

        # Lazy import to avoid importing triton too early.

        self.rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()

        # TODO (rob): refactor block quant into separate class.
        if self.block_quant:
            assert self.quant_config.activation_scheme == "dynamic"
            if not self.quant_config.is_checkpoint_fp8_serialized:
                fp8_dtype = torch.float8_e4m3fn
                w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
                w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)
                for expert in range(layer.local_num_experts):
                    w13_weight[expert, :, :], layer.w13_weight_scale_inv[expert] = (
                        per_block_cast_to_fp8(layer.w13_weight.data[expert, :, :])
                    )
                    w2_weight[expert, :, :], layer.w2_weight_scale_inv[expert] = (
                        per_block_cast_to_fp8(layer.w2_weight.data[expert, :, :])
                    )
                w13_weight_scale_inv = layer.w13_weight_scale_inv.data
                w2_weight_scale_inv = layer.w2_weight_scale_inv.data
            else:
                w13_weight = layer.w13_weight.data
                w13_weight_scale_inv = layer.w13_weight_scale_inv.data
                w2_weight = layer.w2_weight.data
                w2_weight_scale_inv = layer.w2_weight_scale_inv.data

            if current_platform.is_fp8_fnuz():
                w13_weight, w13_weight_scale_inv, w13_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight,
                        layer.w13_weight_scale_inv,
                        layer.w13_input_scale,
                    )
                )
                w2_weight, w2_weight_scale_inv, w2_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale_inv, layer.w2_input_scale
                    )
                )
            elif self.flashinfer_moe_backend is not None:
                # NOTE: weights have to be swapped since the activation is
                # applied on different half for flashinfer vs vllm
                w13_weight = swap_w13_to_w31(layer.w13_weight.data)
                w13_weight_scale_inv = swap_w13_to_w31(layer.w13_weight_scale_inv.data)
                w2_weight = layer.w2_weight.data
                w2_weight_scale_inv = layer.w2_weight_scale_inv.data

            # renormalize
            if current_platform.is_fp8_fnuz():
                w13_weight, w13_weight_scale_inv, w13_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        w13_weight, w13_weight_scale_inv, layer.w13_input_scale
                    )
                )
                w2_weight, w2_weight_scale_inv, w2_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        w2_weight, w2_weight_scale_inv, layer.w2_input_scale
                    )
                )

            # Reset to release memory
            layer.w13_weight.data = torch.empty(
                (0,), dtype=layer.w13_weight.dtype, device=layer.w13_weight.device
            )
            layer.w2_weight.data = torch.empty(
                (0,), dtype=layer.w2_weight.dtype, device=layer.w2_weight.device
            )

            # torch.compile() cannot use Parameter subclasses.
            replace_parameter(layer, "w13_weight", w13_weight)
            replace_parameter(layer, "w13_weight_scale_inv", w13_weight_scale_inv)
            replace_parameter(layer, "w2_weight", w2_weight)
            replace_parameter(layer, "w2_weight_scale_inv", w2_weight_scale_inv)

            if self.rocm_aiter_moe_enabled:
                # reshaping weights is required for aiter moe kernel.
                shuffled_w13, shuffled_w2 = rocm_aiter_ops.shuffle_weights(
                    layer.w13_weight.data, layer.w2_weight.data
                )

                replace_parameter(layer, "w13_weight", shuffled_w13)
                replace_parameter(layer, "w2_weight", shuffled_w2)

            # DeepGemm scales need to be transposed and aligned. We try to do
            # it ahead of time for performance reasons.
            if self.allow_deep_gemm:
                dg_w13_weight, dg_w13_weight_scale_inv = (
                    deepgemm_post_process_fp8_weight_block(
                        wq=layer.w13_weight.data,
                        ws=layer.w13_weight_scale_inv.data,
                        quant_block_shape=tuple(layer.weight_block_size),
                        use_e8m0=is_deep_gemm_e8m0_used(),
                    )
                )
                dg_w2_weight, dg_w2_weight_scale_inv = (
                    deepgemm_post_process_fp8_weight_block(
                        wq=layer.w2_weight.data,
                        ws=layer.w2_weight_scale_inv.data,
                        quant_block_shape=tuple(layer.weight_block_size),
                        use_e8m0=is_deep_gemm_e8m0_used(),
                    )
                )
                layer.w13_weight = Parameter(dg_w13_weight, requires_grad=False)
                layer.w13_weight_scale_inv = Parameter(
                    dg_w13_weight_scale_inv, requires_grad=False
                )
                layer.w2_weight = Parameter(dg_w2_weight, requires_grad=False)
                layer.w2_weight_scale_inv = Parameter(
                    dg_w2_weight_scale_inv, requires_grad=False
                )

        # If checkpoint is fp16, quantize in place.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = current_platform.fp8_dtype()
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            if envs.VLLM_MOE_USE_BLADNN:
                w13_weight_scale_shape = (layer.local_num_experts, w13_weight.shape[1])
            else:
                w13_weight_scale_shape = (layer.local_num_experts,)
            replace_parameter(
                layer,
                "w13_weight_scale",
                torch.ones(
                    *w13_weight_scale_shape,
                    dtype=self.weight_scale_dtype,
                    device=w13_weight.device,
                ),
            )
            for expert in range(layer.local_num_experts):
                if envs.VLLM_MOE_USE_BLADNN:
                    w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                        self.bladnn_dynamic_quantize_fp8(
                            layer.w13_weight.data[expert, :, :], fp8_dtype
                        )
                    )
                    w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                        self.bladnn_dynamic_quantize_fp8(
                            layer.w2_weight.data[expert, :, :], fp8_dtype
                        )
                    )
                else:
                    w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                        ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                    )
                    w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                        ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                    )

            # Reset to release memory
            layer.w13_weight.data = torch.empty(
                (0,),
                dtype=layer.w13_weight.dtype,
                device=layer.w13_weight.device,
            )
            layer.w2_weight.data = torch.empty(
                (0,),
                dtype=layer.w2_weight.dtype,
                device=layer.w2_weight.device,
            )

            replace_parameter(layer, "w13_weight", w13_weight)
            replace_parameter(layer, "w2_weight", w2_weight)

            if self.rocm_aiter_moe_enabled:
                # reshaping weights is required for aiter moe kernel.
                shuffled_w13, shuffled_w2 = rocm_aiter_ops.shuffle_weights(
                    layer.w13_weight, layer.w2_weight
                )
                replace_parameter(layer, "w13_weight", shuffled_w13)
                replace_parameter(layer, "w2_weight", shuffled_w2)

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                    layer.w2_input_scale
                ):
                    logger.warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer."
                    )
                replace_parameter(layer, "w13_input_scale", layer.w13_input_scale.max())
                replace_parameter(layer, "w2_input_scale", layer.w2_input_scale.max())

            if current_platform.is_fp8_fnuz():
                # Normalize the weights and scales
                w13_weight, w13_weight_scale, w13_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                    )
                )
                w2_weight, w2_weight_scale, w2_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
                    )
                )
                # Reset the parameter
                replace_parameter(layer, "w13_weight", w13_weight)
                replace_parameter(layer, "w13_weight_scale", w13_weight_scale)
                if w13_input_scale is not None:
                    replace_parameter(layer, "w13_input_scale", w13_input_scale)
                replace_parameter(layer, "w2_weight", w2_weight)
                replace_parameter(layer, "w2_weight_scale", w2_weight_scale)
                if w2_input_scale is not None:
                    replace_parameter(layer, "w2_input_scale", w2_input_scale)

            if not envs.VLLM_MOE_USE_BLADNN:
                # Fp8 moe kernel needs single weight scale for w13 per expert.
                # We take the max then dequant and requant each expert.
                assert layer.w13_weight_scale is not None
                shard_size = layer.intermediate_size_per_partition
                max_w13_scales = layer.w13_weight_scale.max(dim=1).values
                for expert_id in range(layer.local_num_experts):
                    start = 0
                    for shard_id in range(2):
                        dq_weight = per_tensor_dequantize(
                            layer.w13_weight[expert_id][start : start + shard_size, :],
                            layer.w13_weight_scale[expert_id][shard_id],
                        )
                        (
                            layer.w13_weight[expert_id][start : start + shard_size, :],
                            _,
                        ) = ops.scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                        start += shard_size
                replace_parameter(layer, "w13_weight_scale", max_w13_scales)

            if self.rocm_aiter_moe_enabled:
                shuffled_w13, shuffled_w2 = rocm_aiter_ops.shuffle_weights(
                    layer.w13_weight, layer.w2_weight
                )

                replace_parameter(layer, "w13_weight", shuffled_w13)
                replace_parameter(layer, "w2_weight", shuffled_w2)

            if self.flashinfer_moe_backend is not None:
                # NOTE: weights have to be swapped since the activation is
                # applied on different half for flashinfer vs vllm
                assert not self.block_quant
                register_moe_scaling_factors(layer)
                w13_weight = swap_w13_to_w31(layer.w13_weight.data)
                if self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
                    rotate_flashinfer_fp8_moe_weights(w13_weight, w2_weight)
                layer.w13_weight.data = w13_weight.data

        if self.use_marlin:
            prepare_moe_fp8_layer_for_marlin(
                layer, False, input_dtype=self.marlin_input_dtype
            )
            # Activations not quantized for marlin.
            del layer.w13_input_scale
            del layer.w2_input_scale

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalize | None:
        if (
            self.rocm_aiter_moe_enabled
            or self.use_marlin
            or self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM
        ):
            return None
        elif self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS:
            if self.block_quant:
                assert self.weight_block_size == [128, 128], (
                    f"Only support weight_block_size == [128, 128], "
                    f"got {self.weight_block_size}"
                )
            # Wire block-scale flag through prepare/finalize when using CUTLASS
            prepare_finalize = build_flashinfer_fp8_cutlass_moe_prepare_finalize(
                self.moe,
                use_deepseek_fp8_block_scale=self.block_quant,
            )
            logger.debug_once("%s", prepare_finalize.__class__.__name__)
            return prepare_finalize
        else:
            return super().maybe_make_prepare_finalize(routing_tables)

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        from vllm.model_executor.layers.fused_moe import (
            BatchedDeepGemmExperts,
            BatchedTritonExperts,
            TritonExperts,
            TritonOrDeepGemmExperts,
        )

        assert not self.use_marlin and not self.rocm_aiter_moe_enabled, (
            "Marlin and ROCm AITER are not supported with all2all yet."
        )

        assert self.moe_quant_config is not None

        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens_per_rank is not None

            experts_impl = (
                BatchedDeepGemmExperts if self.allow_deep_gemm else BatchedTritonExperts
            )
            logger.info_once(
                "%s(%s): max_tokens_per_rank=%s, block_size=%s, per_act_token=%s",
                experts_impl.__name__,
                self.__class__.__name__,
                max_num_tokens_per_rank,
                self.weight_block_size,
                False,
            )
            return experts_impl(
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
            )
        elif self.moe.is_lora_enabled:
            return TritonExperts(quant_config=self.moe_quant_config)
        elif self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS:
            # Select GEMM experts with block-scale when weights are block-quantized
            experts = select_cutlass_fp8_gemm_impl(
                self.moe,
                self.moe_quant_config,
                use_deepseek_fp8_block_scale=self.block_quant,
            )
            logger.info_once("Using %s", experts.__class__.__name__)
            return experts
        else:
            logger.info_once(
                "TritonOrDeepGemmExperts(%s): block_size=%s, per_act_token=%s",
                self.__class__.__name__,
                self.weight_block_size,
                False,
            )
            return TritonOrDeepGemmExperts(
                quant_config=self.moe_quant_config,
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

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.use_marlin:
            return None

        return fp8_w8a8_moe_quant_config(
            w1_scale=(
                layer.w13_weight_scale_inv
                if self.block_quant
                else layer.w13_weight_scale
            ),
            w2_scale=(
                layer.w2_weight_scale_inv if self.block_quant else layer.w2_weight_scale
            ),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int4_w4a16=False,
            use_int8_w8a16=False,
            block_shape=self.weight_block_size,
        )

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def allow_inplace(self) -> bool:
        return True

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_callback: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.flashinfer_moe_backend == FlashinferMoeBackend.TENSORRT_LLM:
            if layer.enable_eplb:
                raise NotImplementedError("EPLB not supported for `Fp8MoEMethod` yet.")
            assert layer.activation == "silu", (
                f"Expected 'silu' activation but got {layer.activation}"
            )

            if self.block_quant:
                import vllm.model_executor.layers.fused_moe.flashinfer_trtllm_moe  # noqa: E501, F401

                e_score_correction_bias = (
                    layer.e_score_correction_bias.to(x.dtype)
                    if layer.e_score_correction_bias is not None
                    else None
                )
                routing_method_type = layer.routing_method_type
                return torch.ops.vllm.flashinfer_fused_moe_blockscale_fp8(
                    routing_logits=router_logits.to(torch.float32)
                    if routing_method_type == RoutingMethodType.DeepSeekV3
                    else router_logits,
                    routing_bias=e_score_correction_bias,
                    x=x,
                    w13_weight=layer.w13_weight,
                    w13_weight_scale_inv=layer.w13_weight_scale_inv,
                    w2_weight=layer.w2_weight,
                    w2_weight_scale_inv=layer.w2_weight_scale_inv,
                    global_num_experts=layer.global_num_experts,
                    top_k=layer.top_k,
                    num_expert_group=layer.num_expert_group,
                    topk_group=layer.topk_group,
                    intermediate_size=layer.intermediate_size_per_partition,
                    expert_offset=layer.ep_rank * layer.local_num_experts,
                    local_num_experts=layer.local_num_experts,
                    block_shape=self.weight_block_size,
                    routing_method_type=routing_method_type,
                    routed_scaling=layer.routed_scaling_factor,
                )
            else:
                assert (
                    not layer.renormalize and layer.custom_routing_function is not None
                )
                result = apply_flashinfer_per_tensor_scale_fp8(
                    layer=layer,
                    hidden_states=x,
                    router_logits=router_logits,
                    routing_bias=layer.e_score_correction_bias,
                    global_num_experts=layer.global_num_experts,
                    top_k=layer.top_k,
                    num_expert_group=layer.num_expert_group,
                    topk_group=layer.topk_group,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                )

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

        if self.rocm_aiter_moe_enabled:
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
                rocm_aiter_fused_experts,  # noqa: E501
            )

            result = rocm_aiter_fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                expert_map=layer.expert_map,
                quant_config=self.moe_quant_config,
            )
        elif self.use_marlin:
            assert layer.activation == "silu", (
                f"{layer.activation} not supported for Marlin MoE."
            )
            result = fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                None,
                None,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                router_logits,
                topk_weights,
                topk_ids,
                quant_type_id=scalar_types.float8_e4m3fn.id,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                input_dtype=self.marlin_input_dtype,
                workspace=layer.workspace,
            )
        elif self.flashinfer_moe_backend == FlashinferMoeBackend.CUTLASS:
            assert layer.activation == "silu", (
                f"Expected 'silu' activation but got {layer.activation}"
            )
            if not self.block_quant:
                assert (
                    not layer.renormalize and layer.custom_routing_function is not None
                )
                assert layer.scoring_func == "sigmoid", (
                    f"Expected 'sigmoid' scoring func but got {layer.scoring_func}"
                )
            # Delegate to CUTLASS FlashInfer path; function already bound with
            # use_deepseek_fp8_block_scale for block-quant when applicable
            result = self.flashinfer_moe_fn(
                x,
                layer,
                topk_weights,
                topk_ids,
                inplace=False,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
            )
        else:
            result = self.fused_experts_func(
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
                allow_deep_gemm=self.allow_deep_gemm,
                allow_cutlass_block_scaled_grouped_gemm=(
                    self.allow_cutlass_block_scaled_grouped_gemm
                ),
                **{
                    "shared_experts_callback": shared_experts_callback,
                } if shared_experts_callback is not None else {},
                map_for_skipped=layer.map_for_skipped,
            )

        if layer.zero_expert_num != 0 and layer.zero_expert_type is not None:
            assert not isinstance(result, tuple), (
                "Shared + zero experts are mutually exclusive not yet supported"
            )
            return result, zero_expert_result
        else:
            return result


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
