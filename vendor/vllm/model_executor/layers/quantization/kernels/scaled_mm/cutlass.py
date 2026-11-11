# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .ScaledMMLinearKernel import ScaledMMLinearKernel, ScaledMMLinearLayerConfig


def acext_smoothquant_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.shape[0] == b.shape[1] and bias.dtype == out_dtype
    sqGemm = (
        torch.classes._C.SmoothQuantMatmul_bf16(True, True)
        if out_dtype == torch.bfloat16
        else torch.classes._C.SmoothQuantMatmul(True, True)
    )
    return sqGemm.smootquant_gemm(a, b, scale_a, scale_b, bias)


def acext_smoothquant_gemm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty((m, n), dtype=out_dtype, device=a.device)


direct_register_custom_op(
    op_name="acext_smoothquant_gemm",
    op_func=acext_smoothquant_gemm,
    mutates_args=[],
    fake_impl=acext_smoothquant_gemm_fake,
)


class CutlassScaledMMLinearKernel(ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Requires CUDA."
        if compute_capability is None:
            _cc = current_platform.get_device_capability()
            if _cc is not None:
                compute_capability = _cc.major * 10 + _cc.minor
        if compute_capability is not None and compute_capability < 75:
            return False, f"requires capability 75, got {compute_capability}"
        return True, None

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        try:
            self.sqGemm_fp16 = torch.classes._C.SmoothQuantMatmul(True, True)
            self.sqGemm_bf16 = torch.classes._C.SmoothQuantMatmul_bf16(True, True)
        except Exception:
            self.sqGemm_fp16 = None
            self.sqGemm_bf16 = None

        # WEIGHT
        # Cutlass kernels need transposed weight.
        weight = getattr(layer, self.w_q_name)
        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(weight.t().data, requires_grad=False),
        )

        # WEIGHT SCALE
        # Cutlass kernels support only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, self.w_s_name)
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            self.w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # INPUT SCALE
        if self.config.is_static_input_scheme:
            input_scale = getattr(layer, self.i_s_name)

            if self.config.input_symmetric:
                replace_parameter(
                    layer,
                    self.i_s_name,
                    torch.nn.Parameter(input_scale.max(), requires_grad=False),
                )
                setattr(layer, self.i_zp_name, None)
            else:
                input_zero_point = getattr(layer, self.i_zp_name)

                # reconstruct the ranges
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (input_scale * (int8_traits.max - azps)).max()
                range_min = (input_scale * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
                replace_parameter(
                    layer, self.i_s_name, torch.nn.Parameter(scale, requires_grad=False)
                )

                # AZP loaded as int8 but used as int32
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
                replace_parameter(
                    layer, self.i_zp_name, torch.nn.Parameter(azp, requires_grad=False)
                )

        else:
            setattr(layer, self.i_s_name, None)
            setattr(layer, self.i_zp_name, None)

        # azp_adj is the AZP adjustment term, used to account for weights.
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # For more details, see csrc/quantization/w8a8/cutlass/Epilogues.md
        # https://github.com/vllm-project/vllm/blob/main/csrc/quantization/w8a8/cutlass/Epilogues.md
        if not self.config.input_symmetric:
            weight = getattr(layer, self.w_q_name)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if self.config.is_static_input_scheme:
                # cutlass_w8a8 requires azp to be folded into azp_adj
                # in the per-tensor case
                azp_adj = getattr(layer, self.i_zp_name) * azp_adj
            setattr(
                layer,
                self.azp_adj_name,
                torch.nn.Parameter(azp_adj, requires_grad=False),
            )
        else:
            setattr(layer, self.azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_weight_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=symmetric
        )

        if x_zp is not None:
            # Currently, static is always per-tensor and dynamic is per-token
            static = i_zp is not None
            azp = None if static else x_zp
            return ops.cutlass_scaled_mm_azp(
                x_q,
                w_q,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=x.dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias,
            )
        return ops.cutlass_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )
