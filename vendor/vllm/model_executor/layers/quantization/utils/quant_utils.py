# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This file is used for /tests and /benchmarks"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, NamedTuple

import numpy
import torch
import triton
import triton.language as tl
from torch import fx
from torch.nn import Parameter

from vllm._custom_ops import cutlass_scaled_mm_supports_fp4
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


# Use proxy as NamedTuple direct subclasses cannot have static members
class _GroupShape(NamedTuple):
    row: int
    col: int


class GroupShape(_GroupShape):
    """
    This class describes the quantization group shape.
    It includes static members for common shapes (per-tensor, per-token).
    """

    # Aliases for common quantization group shapes
    PER_TENSOR: ClassVar["GroupShape"]
    PER_TOKEN: ClassVar["GroupShape"]

    def is_per_tensor(self) -> bool:
        return self.row == -1 and self.col == -1

    def is_per_token(self) -> bool:
        return self.row == 1 and self.col == -1

    def is_per_group(self) -> bool:
        return self.row == 1 and self.col >= 1


GroupShape.PER_TENSOR = GroupShape(-1, -1)
GroupShape.PER_TOKEN = GroupShape(1, -1)

BLADNN_DEFAULT_PADDING_FACTOR = 64
DEEPGEMM_DEFAULT_PADDING_FACTOR = 128


@dataclass(frozen=True)
class ScaleDesc:
    """
    Class for describing a single quantization scaling factor.
    dtype: data type of the scale
    static: static scale if True, dynamic if False
    group_shape: group shape of the scale
    """

    dtype: torch.dtype
    static: bool
    group_shape: GroupShape

    def __str__(self):
        group_shape = (
            "per_tensor"
            if self.group_shape == GroupShape.PER_TENSOR
            else (
                "per_token"
                if self.group_shape == GroupShape.PER_TOKEN
                else str(self.group_shape)
            )
        )

        return (
            f"{fx.graph.dtype_abbrs[self.dtype]},"
            f"{'static' if self.static else 'dynamic'},{group_shape}"
        )


@dataclass(frozen=True)
class QuantKey:
    """
    Class for identifying the type of quantization.
    dtype: quantized data type
    scale: scale descriptor
    scale2: second-level scale descriptor
    symmetric: symmetric if True, asymmetric if False
    """

    dtype: torch.dtype
    scale: ScaleDesc
    scale2: ScaleDesc | None = None
    symmetric: bool = True

    def __str__(self):
        scale2_str = f"scale2({self.scale2})," if self.scale2 else ""
        return (
            f"QuantKey({fx.graph.dtype_abbrs[self.dtype]},"
            f"scale({self.scale}),{scale2_str}"
            f"{'a' if not self.symmetric else ''}symmetric)"
        )


kStaticTensorScale = ScaleDesc(torch.float32, True, GroupShape.PER_TENSOR)
kFp8StaticTensorSym = QuantKey(FP8_DTYPE, kStaticTensorScale, symmetric=True)

kDynamicTensorScale = ScaleDesc(torch.float32, False, GroupShape.PER_TENSOR)
kFp8DynamicTensorSym = QuantKey(FP8_DTYPE, kDynamicTensorScale, symmetric=True)

kDynamicTokenScale = ScaleDesc(torch.float32, False, GroupShape.PER_TOKEN)
kFp8DynamicTokenSym = QuantKey(FP8_DTYPE, kDynamicTokenScale, symmetric=True)

kNvfp4GroupScale = ScaleDesc(FP8_DTYPE, False, GroupShape(1, 16))
kNvfp4Quant = QuantKey(FP4_DTYPE, scale=kNvfp4GroupScale, scale2=kStaticTensorScale)

kDynamic128Scale = ScaleDesc(torch.float32, False, GroupShape(1, 128))
kFp8Dynamic128Sym = QuantKey(FP8_DTYPE, kDynamic128Scale, symmetric=True)

kDynamic64Scale = ScaleDesc(torch.float32, False, GroupShape(1, 64))
kFp8Dynamic64Sym = QuantKey(FP8_DTYPE, kDynamic64Scale, symmetric=True)


# Normalize the group_shape to the full extent for any dims that are -1
def _normalize_quant_group_shape(x: torch.Tensor, group_shape: GroupShape):
    # -1 means full extent
    return (
        group_shape[0] if group_shape[0] > 0 else x.shape[-2],
        group_shape[1] if group_shape[1] > 0 else x.shape[-1],
    )


# Useful when treating N-dimensional group scaling as extended numpy-style
# broadcasting in numpy simply stretches dimensions with an extent of 1 to match
# the target shape by repeating the data along that dimension (broadcasting)
# , we extend these semantics to say if the extent of a dimension in the
# source shape is not 1 and does not match the target shape we repeat each
# element along that dimension src_shape[dim] // target_shape[dim] times
# example if we have:
#       a = [[1, 2], and target_shape = (2, 4)
#            [3, 4]]
# then we would expand a to:
#       a = [[1, 1, 2, 2],
#            [3, 3, 4, 4]]
# NOTE this function does not explicitly broadcast dimensions
# with an extent of 1, since this can be done implicitly by pytorch
def group_broadcast(t, shape):
    for i, s in enumerate(shape):
        if t.shape[i] != s and t.shape[i] != 1:
            assert s % t.shape[i] == 0
            t = (
                t.unsqueeze(i + 1)
                .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                .flatten(i, i + 1)
            )
    return t


# Quantize assuming once scale per group of elements with shape group_shape,
# example group shapes:
#  * (-1, -1)   for per-tensor quantization
#  * (1, -1)    for per-row quantization
#  * (-1, 1)    for per-column quantization
#  * (128, 128) for 128x128 deepseek style block quantization
#  * (1, 128)   for deepseek style activation quantization
#               (i.e. per-token-per-group)
def scaled_quantize(
    x: torch.Tensor,
    group_shape: GroupShape,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_shape = _normalize_quant_group_shape(x, group_shape)
    assert quant_dtype.is_floating_point, (
        "currently `scaled_quantize` only supports floating point dtypes "
        "but could be extended to support other dtypes"
    )

    finfo = torch.finfo(quant_dtype)

    # Reshape (M, N) into (BLK_M, BLOCK_SIZE_M, BLK_N, BLOCK_SIZE_N)
    assert x.ndim == 2
    assert x.shape[0] % group_shape[0] == 0 and x.shape[1] % group_shape[1] == 0
    blk_m, blk_n = x.shape[0] // group_shape[0], x.shape[1] // group_shape[1]
    x_blkd = x.reshape(blk_m, group_shape[0], blk_n, group_shape[1])

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    x_blkd_permd = x_blkd.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    x_blkd_permd = x_blkd_permd.flatten(start_dim=2)

    # Compute scales
    min_val, max_val = x_blkd_permd.aminmax(dim=-1)
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax

    # Apply scale and convert form:
    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    x_scl_sat = (
        (x_blkd_permd * scale.unsqueeze(-1))
        .clamp(min=finfo.min, max=finfo.max)
        .reshape(blk_m, blk_n, group_shape[0], group_shape[1])
        .permute(0, 2, 1, 3)
        .reshape(x.shape)
    )

    return x_scl_sat.to(quant_dtype).contiguous(), scale.float().reciprocal()


# inverses `scaled_quantize`
def scaled_dequantize(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    group_shape: GroupShape | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_shape is not None:
        group_shape = _normalize_quant_group_shape(x_q, group_shape)

    if x_s.ndim == 0:  # scalar
        x_s = x_s.unsqueeze(-1).unsqueeze(-1)  # convert to (1, 1) tensor
    if x_s.ndim == 1:
        if group_shape is None:
            raise AssertionError(
                "if x_s is 1D tensor, group_shape must be provided otherwise "
                "its ambiguous which dimension to broadcast x_s to"
            )
        # unsqueeze the scales for the dimension where we want to broadcast
        # across the full extent
        if group_shape[0] == x_q.shape[-2]:
            x_s = x_s.unsqueeze(-2)
        elif group_shape[1] == x_q.shape[-1]:
            x_s = x_s.unsqueeze(-1)
        else:
            raise AssertionError(
                "if x_s is a vector we should be broadcasting it to the full "
                "extent of one of the dimensions"
            )

    if group_shape is not None:
        assert x_s.shape[-1] == x_q.shape[-1] // group_shape[1]
        assert x_s.shape[-2] == x_q.shape[-2] // group_shape[0]
    x_s = group_broadcast(x_s.to(torch.float32), x_q.shape)
    return (x_q.to(torch.float32) * x_s).to(out_dtype)


def pack_quantized_values_into_int32(
    w_q: torch.Tensor, wtype: ScalarType, packed_dim: int = 0
):
    # move dim to pack to the end
    perm = (*[i for i in range(len(w_q.shape)) if i != packed_dim], packed_dim)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    w_q_perm = w_q.permute(perm)

    pack_factor = 32 // wtype.size_bits
    mask = (1 << wtype.size_bits) - 1

    new_shape_perm = list(w_q_perm.shape)
    assert w_q_perm.shape[-1] % pack_factor == 0
    new_shape_perm[-1] //= pack_factor

    res = torch.zeros(new_shape_perm, dtype=torch.int32, device=w_q.device)
    for i in range(pack_factor):
        res |= (w_q_perm[..., i::pack_factor] & mask) << wtype.size_bits * i

    return res.permute(inv_perm)


def unpack_quantized_values_into_int32(
    w_q: torch.Tensor, wtype: ScalarType, packed_dim: int = 0
):
    # move dim to pack to the end
    perm = (*[i for i in range(len(w_q.shape)) if i != packed_dim], packed_dim)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    w_q_perm = w_q.permute(perm)

    pack_factor = 32 // wtype.size_bits
    mask = (1 << wtype.size_bits) - 1

    new_shape_perm = list(w_q_perm.shape)
    new_shape_perm[-1] *= pack_factor

    res = torch.zeros(new_shape_perm, dtype=torch.int32, device=w_q.device)
    for i in range(pack_factor):
        res[..., i::pack_factor] = (w_q_perm >> wtype.size_bits * i) & mask

    return res.permute(inv_perm)


def is_layer_skipped(
    prefix: str,
    ignored_layers: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
    *,
    skip_with_substr: bool = False,
) -> bool:
    def prefix_full_match(prefix: str, ignored_layers: list[str]) -> bool:
        return prefix in ignored_layers

    # For case like: ignored_layers = ["self_attn"]
    def substr_match(prefix: str, ignored_layers: list[str]) -> bool:
        return any(layer in prefix for layer in ignored_layers)

    match_func = substr_match if skip_with_substr else prefix_full_match

    # prefix: model.layers.0.self_attn.q_proj
    # proj_name: q_proj
    proj_name = prefix.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = match_func(shard_prefix, ignored_layers)

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    elif "experts" in prefix and not skip_with_substr:
        expert_ignore_layers = filter(
            lambda layer_name: "experts" in layer_name, ignored_layers
        )
        return any(
            prefix in layer_name if not skip_with_substr else layer_name in prefix
            for layer_name in expert_ignore_layers
        )
    else:
        is_skipped = match_func(prefix, ignored_layers)

    assert is_skipped is not None
    return is_skipped


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def permute_rows(
    q_w: torch.Tensor,
    w_ref: torch.Tensor,
    group_size: int,
    test_perm: torch.Tensor | None = None,
):
    assert q_w.shape == w_ref.shape

    orig_device = q_w.device
    k_size, _ = q_w.shape

    g_idx = torch.zeros((k_size,), dtype=torch.int32)
    for i in range(k_size):
        g_idx[i] = i // group_size

    # Simulate act_order by doing a random permutation on K
    rand_perm = test_perm if test_perm is not None else torch.randperm(k_size)

    g_idx = g_idx[rand_perm].contiguous()
    q_w = q_w[rand_perm, :].contiguous()
    w_ref = w_ref[rand_perm, :].contiguous()

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        rand_perm.to(device=orig_device),
    )


def quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int | None,
    zero_points: bool = False,
    ref_zero_points_after_scales: bool = False,
):
    assert quant_type.is_integer(), (
        "Floating point quantization may work but has not been tested"
    )
    assert not zero_points or group_size is not None, (
        "to have group zero points, group_size must be provided "
        "(-1 group_size is channelwise)"
    )

    orig_device = w.device
    orig_type = w.dtype
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k

    # Reshape to [groupsize, -1]
    if group_size is not None and group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    max_val = torch.max(w, 0, keepdim=True).values
    min_val = torch.min(w, 0, keepdim=True).values

    max_q_val = quant_type.max()
    min_q_val = quant_type.min()

    w_s = torch.Tensor([1.0]).to(w.device)  # unscaled case
    maybe_w_zp = None
    if group_size is not None:
        if zero_points:
            assert not quant_type.is_signed() and quant_type.max() > 0
            w_s = (max_val - min_val).clamp(min=1e-5) / quant_type.max()
            maybe_w_zp = (
                torch.round(torch.abs(min_val / w_s)).clamp(min_q_val, max_q_val).int()
            )
        else:
            # If the bias is such that there are no possible negative/positive
            #  values, set the max value to inf to avoid divide by 0
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)),
            )

    # Quantize
    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)
    w_q = torch.clamp(w_q, min_q_val, max_q_val)

    # Compute ref (dequantized)
    # For some kernels (namely Machete) the zero-points are applied after the
    # scales are applied, for this case computing the reference in similar way
    # allows us to use tighter error tolerances in our unit tests.
    if ref_zero_points_after_scales and maybe_w_zp is not None:
        w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s
    else:
        w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s

    if quant_type.has_bias():
        w_q += quant_type.bias

    # Restore original shapes
    if group_size is not None and group_size < size_k:

        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        w_q = reshape_w(w_q)
        w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()
        maybe_w_zp = maybe_w_zp.to(device=orig_device)

    return (
        w_ref.to(device=orig_device),
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )


SUPPORTED_GPTQ_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


def gptq_quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: torch.Tensor | None = None,
):
    size_k, _ = w.shape

    assert w.is_floating_point(), "w must be float"
    assert quant_type in SUPPORTED_GPTQ_QUANT_TYPES, (
        f"Unsupported gptq type = {quant_type}"
    )
    assert group_size in SUPPORTED_GROUP_SIZES + [size_k], (
        f"Unsupported groupsize = {group_size}"
    )

    w_ref, w_q, w_s, _ = quantize_weights(w, quant_type, group_size)

    # Apply act_order
    g_idx = torch.empty(0, dtype=torch.int, device=w.device)
    rand_perm = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        assert group_size < size_k, (
            "For act_order, groupsize = {} must be less than size_k = {}".format(
                group_size, size_k
            )
        )

        w_ref, w_q, g_idx, rand_perm = permute_rows(w_q, w_ref, group_size, test_perm)

    return w_ref, w_q, w_s, g_idx, rand_perm


def sort_weights(q_w: torch.Tensor, g_idx: torch.Tensor):
    orig_device = q_w.device

    sort_indices = torch.argsort(g_idx).to(dtype=torch.int32)  # Sort based on g_idx

    g_idx = g_idx[sort_indices].contiguous()
    q_w = q_w[sort_indices, :].contiguous()

    return (
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        sort_indices.to(device=orig_device),
    )


def pack_rows(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k // pack_factor, size_n), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[i::pack_factor, :] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    return q_res


def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def unpack_cols(
    packed_q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    assert packed_q_w.shape == (size_k, size_n // pack_factor), (
        "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(
            packed_q_w.shape, size_k, size_n, pack_factor
        )
    )

    orig_device = packed_q_w.device

    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_cpu & mask
        packed_q_w_cpu >>= num_bits
        q_res[:, i::pack_factor] = vals

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def gptq_pack(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    return pack_rows(q_w, num_bits, size_k, size_n)


def awq_pack(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_w = q_w.reshape((-1, len(interleave)))[:, interleave].ravel()
    q_w = q_w.reshape((-1, size_n)).contiguous()

    return pack_cols(q_w, num_bits, size_k, size_n)


def minmax_quantize_weights(
    fp16_weight: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    sym: bool,
    transpose: bool = True,
):
    assert group_size == -1, "Group size is not supported for min/max quantization"
    ftype = fp16_weight.dtype
    if transpose:
        weight = fp16_weight.data.t().to(torch.float32)
    else:
        weight = fp16_weight.data.to(torch.float32)

    t = torch.zeros(weight.shape[1], dtype=weight.dtype, device=weight.device)
    min_val = torch.minimum(weight.amin(0), t)
    max_val = torch.maximum(weight.amax(0), t)
    if sym:
        max_val = torch.maximum(min_val.abs(), max_val)
        t = min_val < 0
        if torch.any(t):
            min_val[t] = -max_val[t]
    zero_indices = (min_val == 0) & (max_val == 0)
    min_val[zero_indices] = -1
    max_val[zero_indices] = +1
    scales = (max_val - min_val).clamp(min=1e-5) / (quant_type.max() - quant_type.min())
    if sym:
        zeros = torch.full_like(scales, (quant_type.min() + quant_type.max() + 1) / 2)
    else:
        zeros = torch.round(quant_type.min() - min_val / scales)
    scale_zeros = zeros * scales

    scales.unsqueeze_(0)
    zeros.unsqueeze_(0)

    # quantize, see also: auto_gptq/quantization/quantizer.py:quantize
    weight.div_(scales).round_().add_(zeros).clamp_(quant_type.min(), quant_type.max())
    weight.sub_(zeros).mul_(scales)

    # pre-pack, see also: auto_gptq/nn_modules/qliner/qlinear_cuda.py:pack
    weight.add_(scale_zeros).div_(scales).round_().clamp_(
        quant_type.min(), quant_type.max()
    )
    return weight, scales.to(ftype), zeros.to(ftype)


def pack_gptq_weight(weight: torch.Tensor, weight_bits: int):
    # pack qweight and qzeros to int32
    if weight_bits == 4:
        wf = torch.tensor(
            list(range(0, 32, weight_bits)),
            dtype=torch.int32,
            device=weight.device,
        ).unsqueeze(0)
        weight = weight.to(torch.int32).view(
            weight.shape[0] // (32 // weight_bits), 32 // weight_bits, -1
        )
        weight.bitwise_left_shift_(wf.unsqueeze(-1))
        qweight = torch.zeros(
            weight.shape[0], weight.shape[-1], dtype=torch.int32, device=weight.device
        )
        _ = [qweight.bitwise_or_(weight[:, i, :]) for i in range(32 // weight_bits)]
        """ Naive implementation from auto-gptq, using numpy, is slow
        intweight = weight.to(torch.int32).numpy(force=True).astype(np.uint32)
        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * weight_bits, intweight.shape[1]),
                           dtype=np.uint32)
        while row < qweight.shape[0]:
            assert weight_bits in [2, 4, 8]
            for j in range(i, i + (32 // weight_bits)):
                qweight[row] |= intweight[j] << (weight_bits * (j - i))
            i += 32 // weight_bits
            row += 1
        qweight = torch.from_numpy(qweight.astype(np.int32)).to(fp16_weight.device)
        """
    elif weight_bits == 8:
        qtype = torch.uint8
        qweight = weight.t().to(qtype).contiguous().view(torch.int32).t().contiguous()
    else:
        raise ValueError(f"Only weight bits 4/8 are supported, got {weight_bits}")
    return qweight


def pack_gptq_zeros(zeros: torch.Tensor, weight_bits: int):
    # pack qweight and qzeros to int32
    if weight_bits == 4:
        wf = torch.tensor(
            list(range(0, 32, weight_bits)),
            dtype=torch.int32,
            device=zeros.device,
        ).unsqueeze(0)
        zeros.sub_(1)
        zeros = zeros.to(torch.int32).view(
            -1, zeros.shape[-1] // (32 // weight_bits), 32 // weight_bits
        )
        zeros.bitwise_left_shift_(wf.unsqueeze(0))
        qzeros = torch.zeros(
            zeros.shape[0], zeros.shape[1], dtype=torch.int32, device=zeros.device
        )
        _ = [qzeros.bitwise_or_(zeros[:, :, i]) for i in range(32 // weight_bits)]
        """
        zeros = zeros.numpy(force=True).astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * weight_bits),
                          dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            assert weight_bits in [2, 4, 8]
            for j in range(i, i + (32 // weight_bits)):
                qzeros[:, col] |= zeros[:, j] << (weight_bits * (j - i))
            i += 32 // weight_bits
            col += 1
        qzeros = torch.from_numpy(qzeros.astype(np.int32)).to(fp16_weight.device)
        """
    elif weight_bits == 8:
        qtype = torch.uint8
        qzeros = zeros.sub_(1).to(qtype).view(torch.int32)
    else:
        raise ValueError(f"Only weight bits 4/8 are supported, got {weight_bits}")
    return qzeros


def pack_gptq_weight_uint4b8(weight: torch.Tensor):
    qweight = weight.t().to(torch.uint8).contiguous()
    qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
    return qweight


def pack_gptq_zeros_uint4b8(zeros: torch.Tensor):
    qzeros = zeros.t().to(torch.uint8).contiguous()
    qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]
    return qzeros


def bladnn_quantization_padding(
    quant_method,
    layer: torch.nn.Module,
    padding_factor: int = BLADNN_DEFAULT_PADDING_FACTOR,
):
    def _pad_gate_up_weight(weight: torch.Tensor, padding_factor: int):
        if (weight.shape[0] // 2) % padding_factor != 0:
            padding_size = padding_factor - (weight.shape[0] // 2) % padding_factor
            new_weight = torch.cat(
                [
                    weight[: weight.shape[0] // 2],
                    torch.zeros(
                        padding_size,
                        *weight.shape[1:],
                        device=weight.device,
                        dtype=weight.dtype,
                    ),
                    weight[weight.shape[0] // 2 :],
                    torch.zeros(
                        padding_size,
                        *weight.shape[1:],
                        device=weight.device,
                        dtype=weight.dtype,
                    ),
                ],
                dim=0,
            )
            return Parameter(new_weight, requires_grad=False)
        return weight

    def _pad_gate_up_scales_and_zeros(scales_or_zeros, padding_factor: int):
        if (scales_or_zeros.shape[0] // 2) % padding_factor != 0:
            padding_size = (
                padding_factor - (scales_or_zeros.shape[0] // 2) % padding_factor
            )
            new_scales_or_zeros = torch.cat(
                [
                    scales_or_zeros[: scales_or_zeros.shape[0] // 2],
                    torch.zeros(
                        padding_size,
                        device=scales_or_zeros.device,
                        dtype=scales_or_zeros.dtype,
                    ),
                    scales_or_zeros[scales_or_zeros.shape[0] // 2 :],
                    torch.zeros(
                        padding_size,
                        device=scales_or_zeros.device,
                        dtype=scales_or_zeros.dtype,
                    ),
                ],
                dim=0,
            )
            return Parameter(new_scales_or_zeros, requires_grad=False)
        return scales_or_zeros

    def _pad_down_weight(weight: torch.Tensor, padding_factor: int):
        if weight.shape[1] % padding_factor != 0:
            padding_size = padding_factor - weight.shape[1] % padding_factor
            new_weight = torch.cat(
                [
                    weight,
                    torch.zeros(
                        weight.shape[0],
                        padding_size,
                        *weight.shape[2:],
                        device=weight.device,
                        dtype=weight.dtype,
                    ),
                ],
                dim=1,
            )
            return Parameter(new_weight, requires_grad=False)
        return weight

    if "gate_up_proj" in quant_method.prefix:
        if hasattr(layer, "weight"):
            layer.weight = _pad_gate_up_weight(layer.weight, padding_factor)
        if hasattr(layer, "qweight"):
            layer.qweight = _pad_gate_up_weight(layer.qweight, padding_factor)
        if hasattr(layer, "scales"):
            layer.scales = _pad_gate_up_scales_and_zeros(layer.scales, padding_factor)
        if hasattr(layer, "weight_scales"):
            layer.weight_scales = _pad_gate_up_scales_and_zeros(
                layer.weight_scales, padding_factor
            )
        if hasattr(layer, "qzeros"):
            layer.qzeros = _pad_gate_up_scales_and_zeros(layer.qzeros, padding_factor)

    if "down_proj" in quant_method.prefix:
        if hasattr(layer, "weight"):
            layer.weight = _pad_down_weight(layer.weight, padding_factor)
        if hasattr(layer, "qweight"):
            layer.qweight = _pad_down_weight(layer.qweight, padding_factor)


def deepgemm_quantization_padding(
    quant_method,
    layer: torch.nn.Module,
    padding_factor: int = DEEPGEMM_DEFAULT_PADDING_FACTOR,
):
    def _pad_gate_up_weight(weight: torch.Tensor, padding_factor: int):
        if (weight.shape[0] // 2) % padding_factor != 0:
            padding_size = padding_factor - (weight.shape[0] // 2) % padding_factor
            new_weight = torch.cat(
                [
                    weight[: weight.shape[0] // 2],
                    torch.zeros(
                        padding_size,
                        *weight.shape[1:],
                        device=weight.device,
                        dtype=weight.dtype,
                    ),
                    weight[weight.shape[0] // 2 :],
                    torch.zeros(
                        padding_size,
                        *weight.shape[1:],
                        device=weight.device,
                        dtype=weight.dtype,
                    ),
                ],
                dim=0,
            )
            return Parameter(new_weight, requires_grad=False)
        return weight

    def _pad_down_weight(weight: torch.Tensor, padding_factor: int):
        if weight.shape[1] % padding_factor != 0:
            padding_size = padding_factor - weight.shape[1] % padding_factor
            new_weight = torch.cat(
                [
                    weight,
                    torch.zeros(
                        weight.shape[0],
                        padding_size,
                        *weight.shape[2:],
                        device=weight.device,
                        dtype=weight.dtype,
                    ),
                ],
                dim=1,
            )
            return Parameter(new_weight, requires_grad=False)
        return weight

    if "gate_up_proj" in quant_method.prefix:
        if hasattr(layer, "weight"):
            layer.weight = _pad_gate_up_weight(layer.weight, padding_factor)

    if "down_proj" in quant_method.prefix:
        if hasattr(layer, "weight"):
            layer.weight = _pad_down_weight(layer.weight, padding_factor)


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization.
            Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError:
            If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def per_token_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    """
    Pad and block-interleave the FP4 block-scales so that they match the data
    layout expected by the CUTLASS / FlashInfer kernels.

    Parameters
    ----------
    scale: torch.Tensor

    Returns
    -------
    torch.Tensor
        The swizzled tensor with the same logical shape as *scale*.
    """
    assert scale.dtype == torch.float8_e4m3fn, (
        "swizzle_blockscale expects the input tensor to be in "
        "torch.float8_e4m3fn format."
    )

    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)  # (1, M, K)
    assert scale.ndim == 3, "Expected a 2-D or 3-D tensor for block scales."

    B, M, K = scale.shape

    def _round_up(x: int, m: int) -> int:
        return (x + m - 1) // m * m

    M_padded = _round_up(M, 128)
    K_padded = _round_up(K, 4)

    padded = torch.zeros(
        (B, M_padded, K_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:B, :M, :K] = scale

    # Reshape / permute to the layout required by the kernel.
    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()

    if scale_ndim == 2:
        return swizzled.reshape(M_padded, K_padded)
    return swizzled.reshape(B, M_padded, K_padded)


def cutlass_fp4_supported() -> bool:
    if not current_platform.is_cuda():
        return False
    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()
    return cutlass_scaled_mm_supports_fp4(capability)


def convert_bf16_scales_to_fp8(
    quant_fp8: Callable, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a BF16 scale tensor into the pair of (fp8_scales, channel_scales)
    expected by W4A8 GEMM kernels.
    """
    assert scales.is_contiguous(), (
        f"scale tensor must be contiguous, got {scales.stride()=}"
    )
    assert scales.is_cuda, "scales must be on gpu"

    orig_shape = scales.shape
    k_groups = orig_shape[-1]
    flat_scales = scales.view(-1, k_groups)

    fp8_scales, chan_scales = quant_fp8(flat_scales)
    fp8_scales = (fp8_scales.float() / 8.0).to(torch.float8_e4m3fn)
    chan_scales *= 8.0

    # restore original shape
    fp8_scales = fp8_scales.view(orig_shape)
    chan_scales = chan_scales.view(orig_shape[:-1], -1)

    return fp8_scales, chan_scales


def convert_packed_uint4b8_to_signed_int4_inplace(t: torch.Tensor) -> torch.Tensor:
    """
    Convert int4b8 (packed to int32) to signed int4
    """
    assert t.is_cuda, "tensor must be on gpu"
    assert t.dtype == torch.int32, f"expected int32 packed weights but got {t.dtype}"

    # loop through the 8 4-bit nibbles in each int32 entry
    for i in range(8):
        shift = 4 * i
        # extract the i-th 4-bit nibble
        nib = (t >> shift) & 0xF
        # clear the original nibble by masking out
        t &= ~(0xF << shift)
        # convert int4b8 [0..15] to signed int4 [-8..7] by subtracting 8
        # and update in-place
        t |= ((nib - 8) & 0xF) << shift

    return t
