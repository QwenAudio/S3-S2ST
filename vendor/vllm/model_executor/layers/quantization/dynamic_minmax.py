import enum
import gc
import os
import queue
from collections import defaultdict
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_expert_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    bladnn_moe_quant_config,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    set_weight_attrs,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq_bladnn import (
    BladnnA8W8LinearMethod,
    BladnnA8W8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_zero_points,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    pack_fp8_to_int32,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    get_weight_perm,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import (
    compress_quantized_24_weight,
    get_weight_perm_24,
    inject_24,
    marlin_permute_scales_24,
    marlin_weights,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    bladnn_quantization_padding,
    minmax_quantize_weights,
    pack_gptq_weight,
    pack_gptq_zeros,
    quantize_weights,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import apply_int8_linear
from vllm.model_executor.model_loader.weight_utils import get_model_loading_tracker
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


class DynamicMinMaxConfig(QuantizationConfig):
    """Config class for dynamic min-max quantization."""

    # For performance reason, maybe a better hyper-parameter should be used instead.
    QGEMM_IS_PROMPT_SIZE = 64

    # Marlin kernels
    GPTQ_MARLIN_TILE = 16
    GPTQ_MARLIN_MIN_THREAD_N = 64
    # MIN_THREAD_K: the original value from vLLM is 128, but it cannot be used to run
    # the qewn2-72b-gqa model with TP=4 (k=7392 and the maximum thread_k would be 32).
    GPTQ_MARLIN_MIN_THREAD_K = 32
    GPTQ_MARLIN_MAX_PARALLEL = 16

    GPTQ_MARLIN_SUPPORTED_NUM_BITS = [4, 8]
    GPTQ_MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

    GPTQ_MARLIN_24_TILE = 16
    GPTQ_MARLIN_24_MIN_THREAD_N = 128
    GPTQ_MARLIN_24_MIN_THREAD_K = 128
    GPTQ_MARLIN_24_MAX_PARALLEL = 64
    GPTQ_MARLIN_24_SUPPORTED_NUM_BITS = [4, 8]
    GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES = [-1, 128]

    def __init__(
        self,
        weight_bits: int = 8,
        group_size: int = -1,
        sym: Optional[bool] = None,
        act_quant: Optional[bool] = False,
        use_marlin: Optional[bool] = None,
        use_cutlass: Optional[bool] = False,
        use_bladnn: Optional[bool] = False,
        use_sparse: Optional[bool] = False,
        use_fp8: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        if use_marlin is None:
            use_marlin = "PPU_HOME" not in os.environ
        self.act_quant = act_quant
        self.use_marlin = use_marlin
        self.use_cutlass = use_cutlass
        self.use_bladnn = use_bladnn
        self.use_sparse = use_sparse
        self.use_fp8 = use_fp8

        if sym is None:
            if use_marlin and (act_quant or use_sparse or use_fp8):
                sym = True
            else:
                sym = False
        self.sym = sym
        self.pack_factor = Fraction(self.weight_bits, self.weight_bits)
        if self.weight_bits not in [4, 8]:
            raise ValueError(
                "Currently, only 4/8-bit weight quantization is supported for "
                f"DynamicMinMax, but got {self.weight_bits} bits."
            )
        elif self.use_marlin and (self.act_quant or self.use_sparse or self.use_fp8):
            assert self.sym, "Marlin-24/FP8-Marlin kernel only works with sym = True"
        elif self.use_cutlass:
            assert self.sym, "Cutlass kernel only works with sym = True"
        elif self.use_bladnn:
            assert self.sym, "Bladnn kernel only works with sym = True"
            assert self.act_quant, "Bladnn kernel only works with act_quant = True"
        if self.use_fp8:
            assert self.weight_bits == 8, (
                "Only 8-bit weight quantization is supported for FP8-Marlin"
            )
        assert group_size == -1, "Only -1 group size is supported"

        # marlin kernels
        self.tile_size = self.GPTQ_MARLIN_TILE
        self.min_thread_n = self.GPTQ_MARLIN_MIN_THREAD_N
        self.min_thread_k = self.GPTQ_MARLIN_MIN_THREAD_K
        self.max_parallel = self.GPTQ_MARLIN_MAX_PARALLEL
        self.tile_size_24 = self.GPTQ_MARLIN_24_TILE
        self.min_thread_n_24 = self.GPTQ_MARLIN_24_MIN_THREAD_N
        self.min_thread_k_24 = self.GPTQ_MARLIN_24_MIN_THREAD_K
        self.max_parallel_24 = self.GPTQ_MARLIN_24_MAX_PARALLEL

        logger.info("DynamicMinMax config: %r", self)

    def __repr__(self) -> str:
        return (
            f"DynamicMinMaxConfig("
            f"weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"sym={self.sym}, "
            f"act_quant={self.act_quant}, "
            f"use_marlin={self.use_marlin}, "
            f"use_cutlass={self.use_cutlass}, "
            f"use_bladnn={self.use_bladnn}, "
            f"use_sparse={self.use_sparse}, "
            f"use_fp8={self.use_fp8})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "dynamic_minmax"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DynamicMinMaxConfig":
        weight_bits = cls.get_from_keys_or(config, ["bits", "wbits", "weight_bits"], 8)
        group_size = cls.get_from_keys_or(config, ["group_size"], -1)
        act_quant = cls.get_from_keys_or(config, ["act_quant"], False)
        use_marlin = cls.get_from_keys_or(
            config, ["use_marlin"], "PPU_HOME" not in os.environ
        )
        use_cutlass = cls.get_from_keys_or(config, ["use_cutlass"], False)
        use_bladnn = cls.get_from_keys_or(config, ["use_bladnn"], False)
        use_sparse = cls.get_from_keys_or(config, ["use_sparse"], False)
        use_fp8 = cls.get_from_keys_or(config, ["use_fp8"], False)
        sym = cls.get_from_keys_or(config, ["sym"], True)

        return cls(
            weight_bits=weight_bits,
            group_size=group_size,
            sym=sym,
            act_quant=act_quant,
            use_marlin=use_marlin,
            use_cutlass=use_cutlass,
            use_bladnn=use_bladnn,
            use_sparse=use_sparse,
            use_fp8=use_fp8,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str = None
    ) -> Optional["DynamicMinMaxLinearMethod"]:
        if isinstance(layer, LinearBase):
            if envs.VLLM_QUANTIZE_ROUTED_EXPERTS_ONLY:
                return UnquantizedLinearMethod(self)
            if prefix.endswith(".mlp.gate"):
                return UnquantizedLinearMethod()
            return DynamicMinMaxLinearMethod(self)
        if isinstance(layer, FusedMoE):
            return DynamicMinMaxMoEMethod(self, layer)
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


class DynamicMinMaxLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    loading_layers: dict[
        str, list[tuple["DynamicMinMaxLinearMethod", torch.nn.Module]]
    ] = defaultdict(list)

    def __init__(self, quant_config: DynamicMinMaxConfig):
        self.quant_config = quant_config

        if self.quant_config.use_bladnn:
            self.bladnn_gemm_op = BladnnA8W8LinearMethod(quant_config)

        self.workspace = None

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
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor.numerator != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        assert self.quant_config.group_size == -1

        weight_attrs = {
            "input_dim": 1,
            "output_dim": 0,
            "weight_bits": self.quant_config.weight_bits,
            "group_size": self.quant_config.group_size,
            "sym": self.quant_config.sym,
            "act_quant": self.quant_config.act_quant,
            "use_marlin": self.quant_config.use_marlin,
            "use_cutlass": self.quant_config.use_cutlass,
            "use_bladnn": self.quant_config.use_bladnn,
            "use_sparse": self.quant_config.use_sparse,
            "use_fp8": self.quant_config.use_fp8,
        }
        weight_attrs.update(extra_weight_attrs)

        layer.output_size_per_partition = output_size_per_partition
        layer.input_size_per_partition = input_size_per_partition

        if envs.VLLM_USE_QUICK_LOADER or envs.VLLM_QUANTIZATION_LAYER_WISE:
            weight_data = torch.empty((1, 1), dtype=params_dtype)
        else:
            weight_data = torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            )
        weight = Parameter(weight_data, requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, weight_attrs)

        # Weight requires after-loading processing
        layer._dynamic_minmax_quantized = False

        # for marlin kernels
        max_workspace_size = (
            output_size_per_partition
            // min(self.quant_config.min_thread_n, self.quant_config.min_thread_n_24)
        ) * max(self.quant_config.max_parallel, self.quant_config.max_parallel_24)
        layer.workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, requires_grad=False
        )
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size
        layer.is_k_full = True
        layer.part_size_k = layer.input_size_per_partition
        layer.part_size_n = layer.output_size_per_partition

    def process_weights_before_loading(self, layer: torch.nn.Module) -> None:
        if not envs.VLLM_QUANTIZATION_LAYER_WISE or getattr(
            layer, "_dynamic_minmax_quantized", False
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_dynamic_minmax_quantized", False):
            return
        layer._dynamic_minmax_quantized = True

        weight_bits = getattr(layer.weight, "weight_bits", 8)
        group_size = getattr(layer.weight, "group_size", -1)
        act_quant = getattr(layer.weight, "act_quant", False)
        use_marlin = getattr(layer.weight, "use_marlin", False)
        use_cutlass = getattr(layer.weight, "use_cutlass", False)
        use_bladnn = getattr(layer.weight, "use_bladnn", False)
        use_sparse = getattr(layer.weight, "use_sparse", False)
        use_fp8 = getattr(layer.weight, "use_fp8", False)
        sym = getattr(layer.weight, "sym", False)

        fp16_weight = layer.weight.data

        # As a special case, for testing only
        if weight_bits == 16:
            return

        # per-channel inside the TP rank
        assert weight_bits in [4, 8], "Only 4/8/16 bits autoquant has been tested"

        ftype = fp16_weight.dtype
        weight, qweight, qzeros, scales = None, None, None, None
        g_idx, g_idx_sort_indices = None, None
        sparse_meta = None

        if use_marlin and use_sparse:
            quant_type = scalar_types.uint4 if weight_bits == 4 else scalar_types.uint8
            size_n = layer.output_size_per_partition
            size_k = layer.input_size_per_partition

            # inject 2:4 sparsity
            fp16_sparse_weight, mask = inject_24(fp16_weight.t(), size_k, size_n)
            weight, scales, zeros = minmax_quantize_weights(
                fp16_sparse_weight, quant_type, group_size=-1, sym=sym, transpose=False
            )
            weight = weight.to(fp16_weight.dtype)

            # compress expects zero-centered weights
            weight.sub_((quant_type.min() + quant_type.max() + 1) // 2)
            # some zeros befores non-zeros after quantize, fixing it
            # weight.mul_(mask)

            # compress sparse quantized weights
            weight_comp, sparse_meta = compress_quantized_24_weight(
                weight, size_k, size_n, quant_type
            )
            size_k_comp = size_k // 2

            # fixes the zeros in the compressed weight
            weight_comp.add_((quant_type.min() + quant_type.max() + 1) // 2)

            # repack to marlin format
            weight_perm = get_weight_perm_24(weight_bits)
            qweight = marlin_weights(
                weight_comp, size_k_comp, size_n, weight_bits, weight_perm
            )
            scales = marlin_permute_scales_24(
                scales, size_k_comp, size_n, group_size=size_k
            )
        elif use_marlin and use_fp8:
            weight, scales = ops.scaled_fp8_quant(fp16_weight, scale=None)
            part_size_n = layer.output_size_per_partition
            part_size_k = layer.input_size_per_partition
            qweight = ops.gptq_marlin_repack(
                pack_fp8_to_int32(weight.t()),
                torch.empty(0, dtype=torch.int, device=weight.device),
                size_k=part_size_k,
                size_n=part_size_n,
                num_bits=weight_bits,
            )
            scales = scales.repeat(1, part_size_n).to(fp16_weight.dtype)
            scales = marlin_permute_scales(
                scales, part_size_k, part_size_n, group_size=-1
            )
        elif use_marlin and (weight_bits == 4 or weight_bits == 8):
            # qtype is uint4
            quant_type = scalar_types.uint4 if weight_bits == 4 else scalar_types.uint8
            weight, scales, zeros = minmax_quantize_weights(
                fp16_weight, quant_type, group_size=-1, sym=sym
            )

            # repack to marlin format
            part_size_n = layer.output_size_per_partition
            part_size_k = layer.input_size_per_partition

            scales_size_k = part_size_k
            scales_size_n = part_size_n

            # reset g_idx and g_idx_sort_indices
            g_idx = torch.empty(0, dtype=torch.int32, device=fp16_weight.device)
            g_idx_sort_indices = torch.empty(
                0, dtype=torch.int32, device=fp16_weight.device
            )

            qweight = pack_gptq_weight(weight, weight_bits)
            qweight = ops.gptq_marlin_repack(
                qweight, g_idx_sort_indices, part_size_k, part_size_n, weight_bits
            )
            scales = marlin_permute_scales(
                scales, scales_size_k, scales_size_n, group_size
            )

            weight_perm = get_weight_perm(weight_bits)
            if sym:  # GPTQ style
                qzeros = pack_gptq_zeros(zeros, weight_bits)
            else:  # AWQ style
                weight_perm = get_weight_perm(quant_type.size_bits)
                qzeros = marlin_zero_points(
                    zeros, size_k=1, size_n=part_size_n, num_bits=weight_bits
                )
        elif use_cutlass and act_quant and weight_bits == 8:
            # cutlass supported int8 format
            quant_type = scalar_types.int8
            weight, scales, zeros = minmax_quantize_weights(
                fp16_weight, quant_type, group_size=-1, sym=sym
            )
            # Cutlass kernels need transposed weight.
            qweight = weight.to(torch.int8)
            scales = scales.to(torch.float).contiguous().view(qweight.size(1), 1)
        elif use_bladnn:
            qtype = torch.int8
            quant_type = scalar_types.int8
            # k needed to be padded to be divided by 64
            n, k = fp16_weight.shape
            if n % 64 != 0:
                padded_n = (n + 63) // 64 * 64
                fp16_weight = F.pad(fp16_weight, (0, 0, 0, padded_n - n), "constant", 0)
                bias = getattr(layer, "bias", None)
                if bias is not None:
                    bias = F.pad(bias, (0, padded_n - n), "constant", 0)
                    layer.bias.data = bias
            if k % 64 != 0:
                padded_k = (k + 63) // 64 * 64
                fp16_weight = F.pad(fp16_weight, (0, padded_k - k, 0, 0), "constant", 0)
            weight, scales, zeros = minmax_quantize_weights(
                fp16_weight, quant_type, group_size=-1, sym=sym
            )
            # The `.contiguous()` is necessary.
            qweight = weight.t().to(qtype).contiguous()
            scales = scales.squeeze_(0).to(torch.float16)  # scale must be float16
            qzeros = zeros.squeeze_(0).to(ftype)
        elif (
            not use_marlin and not act_quant and (weight_bits == 4 or weight_bits == 8)
        ):
            # fallback to vanilla GPTQ
            quant_type = scalar_types.uint4 if weight_bits == 4 else scalar_types.uint8
            weight, scales, zeros = minmax_quantize_weights(
                fp16_weight, quant_type, group_size=-1, sym=sym
            )
            qweight = pack_gptq_weight(weight, weight_bits)
            qzeros = pack_gptq_zeros(zeros, weight_bits)
            g_idx = torch.zeros(
                weight.shape[0], dtype=torch.int32, device=weight.device
            )
        else:
            raise ValueError(
                f"Only weight bits 4/8 are supported, got {self.quant_config=}"
            )

        if qweight is not None:
            if use_cutlass and act_quant and weight_bits == 8:
                qweight = qweight.to(fp16_weight.device).t().contiguous().t()
            else:
                qweight = qweight.to(fp16_weight.device).contiguous()
        if qzeros is not None:
            qzeros = qzeros.to(fp16_weight.device).contiguous()
        if scales is not None:
            scales = scales.to(fp16_weight.device).contiguous()
        if g_idx is not None:
            g_idx = g_idx.to(fp16_weight.device).contiguous()
        if g_idx_sort_indices is not None:
            g_idx_sort_indices = g_idx_sort_indices.to(fp16_weight.device).contiguous()
        if sparse_meta is not None:
            sparse_meta = sparse_meta.to(fp16_weight.device).contiguous()

        # Release memory
        layer.weight.data = torch.empty(
            (0,), dtype=layer.weight.dtype, device=layer.weight.device
        )
        # delete from current layer
        delattr(layer, "weight")

        if qweight is not None:
            param_qweight = Parameter(qweight, requires_grad=False)
            layer.register_parameter("qweight", param_qweight)
        if qzeros is not None:
            param_qzeros = Parameter(qzeros, requires_grad=False)
            layer.register_parameter("qzeros", param_qzeros)
        if scales is not None:
            param_scales = Parameter(scales, requires_grad=False)
            layer.register_parameter("scales", param_scales)
        if g_idx is not None:
            param_g_idx = Parameter(g_idx, requires_grad=False)
            layer.register_parameter("g_idx", param_g_idx)
        if g_idx_sort_indices is not None:
            param_g_idx_sort_indices = Parameter(
                g_idx_sort_indices, requires_grad=False
            )
            layer.register_parameter("g_idx_sort_indices", param_g_idx_sort_indices)
        if sparse_meta is not None:
            param_sparse_meta = Parameter(sparse_meta, requires_grad=False)
            layer.register_parameter("sparse_meta", param_sparse_meta)

        if use_bladnn:
            bladnn_quantization_padding(self, layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(layer, "weight"):
            weight = layer.weight
            output = F.linear(x, weight)
            if bias is not None:
                output.add_(bias)
        elif self.quant_config.use_marlin and self.quant_config.act_quant:
            qweight = layer.qweight
            out_shape = x.shape[:-1] + (layer.part_size_n,)
            reshaped_x = x.reshape(-1, x.shape[-1])
            size_m = reshaped_x.shape[0]
            x_int8, s_tok = ops.scaled_int8_quant(reshaped_x)
            output = ops.marlin_qqq_gemm(
                x_int8,
                qweight,
                s_tok,
                layer.scales,
                torch.empty(0, dtype=torch.float16, device=layer.scales.device),
                layer.workspace,
                size_m,
                layer.part_size_n,
                layer.part_size_k,
            )
            output = output.reshape(out_shape)
            if bias is not None:
                output.add_(bias)
        elif self.quant_config.use_marlin and self.quant_config.use_sparse:
            qweight = layer.qweight
            out_shape = x.shape[:-1] + (layer.part_size_n,)
            reshaped_x = x.reshape(-1, x.shape[-1])
            size_m = reshaped_x.shape[0]
            qtype = (
                scalar_types.uint4b8
                if self.quant_config.weight_bits == 4
                else scalar_types.uint8b128
            )
            output = ops.gptq_marlin_24_gemm(
                reshaped_x,
                qweight,
                layer.sparse_meta,
                layer.scales,
                layer.workspace,
                qtype,
                size_m,
                layer.part_size_n,
                layer.part_size_k,
            )
            output = output.reshape(out_shape)
            if bias is not None:
                output.add_(bias)
        elif self.quant_config.use_marlin and self.quant_config.use_fp8:
            qweight = layer.qweight
            out_shape = x.shape[:-1] + (layer.part_size_n,)
            reshaped_x = x.reshape(-1, x.shape[-1])
            size_m = reshaped_x.shape[0]
            output = ops.fp8_marlin_gemm(
                reshaped_x,
                qweight,
                layer.scales,
                layer.workspace,
                self.quant_config.weight_bits,
                size_m,
                layer.part_size_n,
                layer.part_size_k,
            )
            output = output.reshape(out_shape)
            if bias is not None:
                output.add_(bias)
        elif self.quant_config.use_marlin:
            qweight = layer.qweight
            out_shape = x.shape[:-1] + (layer.part_size_n,)
            reshaped_x = x.reshape(-1, x.shape[-1])
            size_m = reshaped_x.shape[0]
            if self.quant_config.sym:
                qtype = (
                    scalar_types.uint4b8
                    if self.quant_config.weight_bits == 4
                    else scalar_types.uint8b128
                )
            else:
                qtype = (
                    scalar_types.uint4
                    if self.quant_config.weight_bits == 4
                    else scalar_types.uint8
                )
            output = ops.gptq_marlin_gemm(
                reshaped_x,
                qweight,
                layer.scales,
                layer.qzeros,
                layer.g_idx,
                layer.g_idx_sort_indices,
                layer.workspace,
                qtype,
                size_m,
                layer.part_size_n,
                layer.part_size_k,
                is_k_full=layer.is_k_full,
                has_zp=not self.quant_config.sym,
                use_fp32_reduce=True,
            )
            output = output.reshape(out_shape)
            if bias is not None:
                output.add_(bias)
        elif (
            self.quant_config.use_cutlass
            and self.quant_config.act_quant
            and self.quant_config.weight_bits == 8
        ):
            output = apply_int8_linear(x, layer.qweight, layer.scales, bias=bias)
        elif self.quant_config.use_bladnn:
            output = self.bladnn_gemm_op.apply(layer, x, bias)
        elif (
            not self.quant_config.use_marlin
            and not self.quant_config.act_quant
            and (
                self.quant_config.weight_bits == 4 or self.quant_config.weight_bits == 8
            )
        ):
            qweight = layer.qweight
            out_shape = x.shape[:-1] + (qweight.shape[-1],)
            reshaped_x = x.reshape(-1, x.shape[-1])
            # use_exllama yields incorrect result for 8-bit quantization, not sure why.
            output = ops.gptq_gemm(
                reshaped_x,
                layer.qweight,
                layer.qzeros,
                layer.scales,
                layer.g_idx,
                False,
                self.quant_config.weight_bits,
            )
            output = output.reshape(out_shape)
            if bias is not None:
                output.add_(bias)
        return output


class DynamicMinMaxMoEMethod(FusedMoEMethodBase):
    weight_tensors: dict[str, torch.Tensor] = {}
    current_layer_idx: dict[str, int] = defaultdict(lambda: None)
    current_layer: dict[str, tuple["DynamicMinMaxMoEMethod", torch.nn.Module]] = (
        defaultdict(lambda: None)
    )

    def __init__(self, quant_config: DynamicMinMaxConfig, layer: torch.nn.Module):
        super().__init__(layer.moe_config)

        vllm_config = get_current_vllm_config()
        use_ep = (
            vllm_config.parallel_config.enable_expert_parallel
            and get_expert_parallel_world_size() > 1
        )
        assert not use_ep, "DynamicMinMaxMoEMethod does not support expert parallel."

        self.quant_config = quant_config
        if self.quant_config.use_bladnn:
            import bladnn

            self.bladnn_fused_moe_op = BladnnA8W8MoEMethod(quant_config, layer)
            self.bladnn_dynamic_quantize = bladnn.DynamicQuantize()
            self.weight_scale_dtype = torch.float16
        else:
            raise TypeError("DynamicMinMaxMoEMethod only support bladnn now.")
        self.workspace = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight_shape = (2 * intermediate_size_per_partition, hidden_size)
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, *w13_weight_shape, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_shape = (hidden_size, intermediate_size_per_partition)
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, *w2_weight_shape, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        layer._int8_quantized = False
        # dynamic scale
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if getattr(layer, "_int8_quantized", False):
            return
        layer._int8_quantized = True
        assert envs.VLLM_MOE_USE_BLADNN, "DynamicMinMaxMoEMethod requires bladnn."

        w13_weight = torch.empty_like(layer.w13_weight.data, dtype=torch.int8)
        # Re-initialize w13_scale because we directly quantize
        # merged w13 weights and generate a single scaling factor.
        w13_weight_scale_shape = (layer.local_num_experts, w13_weight.shape[1])
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                *w13_weight_scale_shape,
                dtype=self.weight_scale_dtype,
                device=w13_weight.device,
            ),
            requires_grad=False,
        )
        for expert in range(layer.local_num_experts):
            w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                self.bladnn_dynamic_quantize(layer.w13_weight.data[expert, :, :])
            )

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)

        if not envs.VLLM_QUANT_SKIP_FC2:
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=torch.int8)
            w2_weight_scale_shape = (layer.local_num_experts, w2_weight.shape[1])
            layer.w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    *w2_weight_scale_shape,
                    dtype=self.weight_scale_dtype,
                    device=w2_weight.device,
                ),
                requires_grad=False,
            )
            for expert in range(layer.local_num_experts):
                if not envs.VLLM_QUANT_SKIP_FC2:
                    w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                        self.bladnn_dynamic_quantize(layer.w2_weight.data[expert, :, :])
                    )
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        else:
            layer.w2_weight_scale = None

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
            use_int8_w8a8=True,
            use_int4_w4a16=False,
            use_int8_w8a16=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

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

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
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
            } if shared_experts_callback is not None else {},
            map_for_skipped=layer.map_for_skipped,
        )
