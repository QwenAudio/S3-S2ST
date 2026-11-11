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
from vllm.model_executor.parameter import ChannelQuantScaleParameter
from vllm.utils.deep_gemm import (
    int8_gemm_nt,
    is_deep_gemm_supported,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.platform_utils import is_ppu
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


logger = init_logger(__name__)


class BladnnA8W8Config(QuantizationConfig):
    """Config class for Bladnn A8W8"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        ignored_layers: list[str] | None = None,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.pack_factor = 1  # TBD
        self.dynamic_quant_input = True
        self.ignored_layers = ignored_layers or []

    def __repr__(self) -> str:
        return (
            f"BladnnA8W8Config(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "bladnn_a8w8"

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
    def from_config(cls, config: dict[str, Any]) -> "BladnnA8W8Config":
        weight_bits = cls.get_from_keys(config, ["bits", "weight_bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = config.get("desc_act", False)
        return cls(weight_bits, group_size, desc_act)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        can_convert = cls.is_bladnn_a8w8_compatible(hf_quant_cfg)
        is_valid_user_quant = user_quant is None or user_quant == "bladnn_a8w8"

        if can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to Bladnn A8W8 format. "
                "Using Bladnn A8W8 kernel."
            )
            logger.info(msg)
            return cls.get_name()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str = None
    ) -> Optional[Union["BladnnA8W8LinearMethod", "BladnnA8W8MoEMethod"]]:
        if isinstance(layer, LinearBase):
            if envs.VLLM_QUANTIZE_ROUTED_EXPERTS_ONLY:
                return UnquantizedLinearMethod()
            if prefix.endswith(".mlp.gate"):  # bladnn int8 doesn't quantize gate.
                return UnquantizedLinearMethod()
            return BladnnA8W8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return BladnnA8W8MoEMethod(self, layer)
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []

    @classmethod
    def is_bladnn_a8w8_compatible(cls, quant_config: dict[str, Any]):
        quant_method = quant_config.get("quant_method", "").lower()
        if quant_method and quant_method not in ["gptq", "bladnn_a8w8"]:
            return False

        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        desc_act = quant_config.get("desc_act")
        return not (num_bits != 8 or group_size != -1 or desc_act)


class BladnnA8W8GPTQConfig(BladnnA8W8Config):
    """Config class for Bladnn A8W8, from GPTQ weight checkpoint."""

    @classmethod
    def get_name(cls) -> str:
        return "bladnn_a8w8_gptq"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BladnnA8W8GPTQConfig":
        weight_bits = cls.get_from_keys(config, ["bits", "weight_bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = config.get("desc_act", False)

        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            weight_bits=weight_bits,
            group_size=group_size,
            desc_act=desc_act,
            ignored_layers=ignored_layers,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        can_convert = cls.is_bladnn_a8w8_compatible(hf_quant_cfg)
        is_valid_user_quant = user_quant is None or user_quant == "bladnn_a8w8_gptq"

        if can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to Bladnn A8W8 GPTQ format. "
                "Using Bladnn A8W8 GPTQ kernel."
            )
            logger.info(msg)
            return cls.get_name()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str = None
    ) -> Optional[Union["BladnnA8W8LinearMethod", "BladnnA8W8MoEMethod"]]:
        if isinstance(layer, LinearBase):
            if envs.VLLM_QUANTIZE_ROUTED_EXPERTS_ONLY:
                return UnquantizedLinearMethod()
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if prefix.endswith(".mlp.gate"):  # bladnn int8 doesn't quantize gate.
                return UnquantizedLinearMethod()
            return BladnnA8W8LinearMethod(self)  # can change bladnn into deepgemm int8
        elif isinstance(layer, FusedMoE):
            return BladnnA8W8MoEMethod(self, layer)
        return None

    @classmethod
    def is_bladnn_a8w8_compatible(cls, quant_config: dict[str, Any]):
        quant_method = quant_config.get("quant_method", "").lower()
        if quant_method and quant_method not in ["gptq", "bladnn_a8w8_gptq"]:
            return False

        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        desc_act = quant_config.get("desc_act")
        return not (num_bits != 8 or group_size != -1 or desc_act)


def apply_bladnn_gemm_a8w8_fused_dq_fake(
    workspace: torch.Tensor,
    a: torch.Tensor,
    w: torch.Tensor,
    scale_w: torch.Tensor,
    interleaved: Optional[bool] = False,
    user_prov_max_val: Optional[float] = 0.0,
    scale_ub: Optional[float] = 0.0,
) -> torch.Tensor:
    out = torch.empty([a.shape[0], w.shape[0]], dtype=a.dtype, device=a.device)
    return out


def apply_bladnn_gemm_a8w8_fused_dq(
    workspace: torch.Tensor,
    a: torch.Tensor,
    w: torch.Tensor,
    scale_w: torch.Tensor,
    interleaved: Optional[bool] = False,
    user_prov_max_val: Optional[float] = 0.0,
    scale_ub: Optional[float] = 0.0,
) -> torch.Tensor:
    out = torch.empty([a.shape[0], w.shape[0]], dtype=a.dtype, device=a.device)
    torch.ops.torch_bladnn.a8w8_gemm_fused_dq(
        out,
        a,
        w,
        scale_w,
        interleaved,
        user_prov_max_val,
        scale_ub,
        workspace,
    )
    return out


direct_register_custom_op(
    op_name="apply_bladnn_gemm_a8w8_fused_dq",
    op_func=apply_bladnn_gemm_a8w8_fused_dq,
    mutates_args=[],
    fake_impl=apply_bladnn_gemm_a8w8_fused_dq_fake,
)


def apply_deep_gemm_a8w8_fused_dq(
    a: torch.Tensor, w: torch.Tensor, scale_w: torch.Tensor
) -> torch.Tensor:
    x_q, x_scale = per_token_quant_int8(a)  # scale:(m,1) fp32
    out = torch.empty([a.shape[0], w.shape[0]], dtype=torch.bfloat16, device=a.device)
    M = x_q.shape[0]
    N = w.shape[0]
    K = x_q.shape[-1]
    best_config = get_ppu_deep_gemm_config(M, N, K, num_groups=1)
    int8_gemm_nt((x_q, x_scale), (w, scale_w), out, configs=best_config)
    return out


def apply_deep_gemm_a8w8_fused_dq_fake(
    a: torch.Tensor, w: torch.Tensor, scale_w: torch.Tensor
) -> torch.Tensor:
    out = torch.empty([a.shape[0], w.shape[0]], dtype=torch.bfloat16, device=a.device)
    return out


direct_register_custom_op(  # register into torch.ops.vllm.xxx
    op_name="apply_deep_gemm_a8w8_fused_dq",
    op_func=apply_deep_gemm_a8w8_fused_dq,
    mutates_args=[],
    fake_impl=apply_deep_gemm_a8w8_fused_dq_fake,
)


@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    x_sum_ptr,
    stride_x,
    stride_xq,
    N,
    CAL_SUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Adapted from https://github.com/InternLM/lmdeploy/blob/086481ed84b59bee3b8e4274e5fc69620040c048/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py#L282
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)
    if CAL_SUM:
        x_sum = tl.sum(x, axis=0)
        tl.store(x_sum_ptr + row_id, x_sum.to(x_sum_ptr.dtype.element_ty))

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


def per_token_quant_int8(x, scale_dtype=torch.float32, cal_sum=False):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=scale_dtype)
    if cal_sum:
        x_sum = torch.empty(x.shape[:-1], device=x.device, dtype=x.dtype)
    else:
        x_sum = None
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    assert x.is_contiguous()
    _per_token_quant_int8[(M,)](
        x,
        x_q,
        scales,
        x_sum,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        CAL_SUM=cal_sum,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    if cal_sum:
        return x_q, scales, x_sum
    else:
        return x_q, scales


class BladnnA8W8LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: BladnnA8W8Config):
        if quant_config.group_size not in [-1]:
            raise ValueError("Bladnn GEMM_A8W8: group_size only support [-1]")
        self.quant_config = quant_config
        self.bladnn_gemm_op = bladnn.A8W8GemmFusedDq()
        self.is_deep_gemm_supported = is_deep_gemm_supported()

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
        # Warning: sub-channel has NOT fully supported!
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        group_size = self.quant_config.group_size
        assert group_size == -1

        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="cuda",
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        qzeros = Parameter(
            torch.zeros(
                output_size_per_partition,
                device="cuda",
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {
                "input_dim": None,  # currently None for per-channel
                "output_dim": 0,
            },
        )

        scales = ChannelQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                dtype=torch.float16,  # scales must be float16.  ppuint8,must be fp32
            ),
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )

        # load bladnn a8w8 weights in non-GPTQ format, use weight/weight_scale in ckpt
        layer.register_parameter("weight", qweight)
        layer.register_parameter("weight_scale", scales)
        # for other a8w8 weights(use) in GPTQ format, use qweight/scales in ckp
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # use as qweight,scales, to be same as dynamic_minmax
        # (if can load as qweight, skip this name change)
        qweight = getattr(layer, "weight", None)
        if qweight is not None:
            layer.register_parameter("qweight", qweight)
            delattr(layer, "weight")
        scales = getattr(layer, "weight_scale", None)
        if scales is not None:
            scales = Parameter(scales.data, requires_grad=False)
            layer.register_parameter("scales", scales)
            delattr(layer, "weight_scale")

        if envs.VLLM_INT8_DENSE_USE_DEEPGEMM and self.is_deep_gemm_supported:
            layer.scales.data = layer.scales.data.unsqueeze(-1).to(torch.float32)

        # padding the intermediate size to multiple of 64
        bladnn_quantization_padding(self, layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        # we use bladnn's apply_bladnn_gemm_a8w8_fused_dq， for int8 gemm, can run
        # TODO: change into deepgemm int8 gemm
        if envs.VLLM_INT8_DENSE_USE_DEEPGEMM and self.is_deep_gemm_supported:
            output = torch.ops.vllm.apply_deep_gemm_a8w8_fused_dq(
                x, layer.qweight, layer.scales
            )
        elif envs.VLLM_BLADNN_GEMM_CHUNK_SIZE > 0:
            output = torch.empty(
                [x.shape[0], layer.qweight.shape[0]], dtype=x.dtype, device=x.device
            )
            num_chunks = (
                x.shape[0] + envs.VLLM_BLADNN_GEMM_CHUNK_SIZE - 1
            ) // envs.VLLM_BLADNN_GEMM_CHUNK_SIZE
            for i in range(num_chunks):
                start = i * envs.VLLM_BLADNN_GEMM_CHUNK_SIZE
                end = min((i + 1) * envs.VLLM_BLADNN_GEMM_CHUNK_SIZE, x.shape[0])
                output[start:end] = torch.ops.vllm.apply_bladnn_gemm_a8w8_fused_dq(
                    self.bladnn_gemm_op.workspace,
                    x[start:end],
                    layer.qweight,
                    layer.scales,
                )
        else:
            output = torch.ops.vllm.apply_bladnn_gemm_a8w8_fused_dq(
                self.bladnn_gemm_op.workspace, x, layer.qweight, layer.scales
            )

        if bias is not None:
            output.add_(bias)
        return output


class BladnnA8W8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: BladnnA8W8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)

        if quant_config.group_size not in [-1]:
            raise ValueError("Bladnn FusedMoE_A8W8: group_size only support [-1]")
        self.quant_config = quant_config
        assert envs.VLLM_MOE_USE_BLADNN
        self.bladnn_dynamic_quantize = bladnn.DynamicQuantize()
        self.weight_scale_dtype = torch.float16

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
        fc2_dtype = params_dtype if envs.VLLM_QUANT_SKIP_FC2 else torch.int8
        w13_weight_data = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=torch.int8,
        )
        w2_weight_data = torch.empty(
            num_experts, hidden_size, intermediate_size_per_partition, dtype=fc2_dtype
        )

        w13_weight = torch.nn.Parameter(w13_weight_data, requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(w2_weight_data, requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # weigth scale
        w13_weight_scale_shape = (num_experts, w13_weight.shape[1])
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(*w13_weight_scale_shape, dtype=torch.float16),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        if not envs.VLLM_QUANT_SKIP_FC2:
            w2_weight_scale_shape = (num_experts, w2_weight.shape[1])
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(*w2_weight_scale_shape, dtype=torch.float16),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        else:
            layer.w2_weight_scale = None

        layer._int8_quantized = True
        # dynamic scale
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_before_loading(self, layer: torch.nn.Module) -> None:
        if not envs.VLLM_QUANTIZATION_LAYER_WISE:
            return

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if is_ppu() and envs.VLLM_MOE_USE_ACEXT:  # to_fp_32
            layer.w13_weight_scale.data = layer.w13_weight_scale.data.to(torch.float32)
            layer.w2_weight_scale.data = layer.w2_weight_scale.data.to(torch.float32)

        if is_ppu() and envs.VLLM_USE_DEEP_GEMM and self.allow_deep_gemm:
            # ppu deepep use shape [e,n,1]
            layer.w13_weight_scale.data = layer.w13_weight_scale.data.unsqueeze(2).to(
                torch.float32
            )
            layer.w2_weight_scale.data = layer.w2_weight_scale.data.unsqueeze(2).to(
                torch.float32
            )

        if getattr(layer, "_int8_quantized", False):
            return
        if envs.VLLM_QUANT_SKIP_FC2:
            raise NotImplementedError
        assert envs.VLLM_MOE_USE_BLADNN
        layer._int8_quantized = True
        int8_dtype = torch.int8
        fc2_dtype = layer.w2_weight.dtype if envs.VLLM_QUANT_SKIP_FC2 else torch.int8
        w13_weight = torch.empty_like(layer.w13_weight.data, dtype=int8_dtype)
        w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fc2_dtype)

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
            if not envs.VLLM_QUANT_SKIP_FC2:
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    self.bladnn_dynamic_quantize(layer.w2_weight.data[expert, :, :])
                )

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        if not envs.VLLM_QUANT_SKIP_FC2:
            layer.w2_weight = torch.nn.parameter(w2_weight, requires_grad=False)

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
        shared_experts_callback: Callable[[torch.Tensor], torch.Tensor] | None = None,
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
