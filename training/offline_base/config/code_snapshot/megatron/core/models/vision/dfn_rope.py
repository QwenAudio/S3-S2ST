from functools import partial
from itertools import accumulate
from typing import Union, Optional, Callable
import math
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.layers import to_2tuple, trunc_normal_

from megatron.core.utils import init_method_normal
from megatron.training import print_rank_0
from megatron.training import get_args
from megatron import core
from megatron.core import mpu, tensor_parallel
from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule

from flash_attn.flash_attn_interface import flash_attn_varlen_func
from flash_attn.layers.rotary import apply_rotary_emb
from apex.normalization import FusedLayerNorm
from einops import rearrange


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()
    output = apply_rotary_emb(t_, cos, sin).type_as(t)
    return output


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device)
                    / self.dim
                )
            )
            seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


@jit_fuser
def quick_gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


class QuickGELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)"""

    def __init__(self, inplace: bool = False) -> None:
        super(QuickGELU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quick_gelu(input)


class ApexFusedLayerNorm(FusedLayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        memory_efficient: bool = False,
        sequence_parallel: bool = False,
    ):
        super().__init__(normalized_shape, eps, elementwise_affine, memory_efficient)
        if self.elementwise_affine:
            setattr(self.weight, 'sequence_parallel', sequence_parallel)
            setattr(self.bias, 'sequence_parallel', sequence_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).type_as(x)


class PatchEmbMerger(nn.Module):
    def __init__(self, d_model: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.spatial_merge_unit = spatial_merge_size**2
        self.mlp = nn.Sequential(
            nn.Linear(context_dim * self.spatial_merge_unit, context_dim * self.spatial_merge_unit),
            nn.GELU(),
            nn.Linear(context_dim * self.spatial_merge_unit, d_model),
        )
        self.ln_q = ApexFusedLayerNorm(context_dim, sequence_parallel=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        context_dim = x.shape[-1]
        x = x.reshape(-1, context_dim * self.spatial_merge_unit)
        out = self.mlp(x)
        return out


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_patches = num_patches

        kernel_size = [temporal_patch_size] + list(patch_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.reshape(L, -1, self.temporal_patch_size, self.patch_size[0], self.patch_size[1])
        x = self.proj(x).reshape(L, -1)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = QuickGELU,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=True)
        self.proj = nn.Linear(all_head_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(N, 3, self.num_heads, -1).permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)
        rot_dim = rotary_pos_emb.shape[-1] * 2
        q = torch.cat(
            [
                apply_rotary_pos_emb(q[..., :rot_dim][None], rotary_pos_emb).squeeze(0),
                q[..., rot_dim:],
            ],
            dim=-1,
        )
        k = torch.cat(
            [
                apply_rotary_pos_emb(k[..., :rot_dim][None], rotary_pos_emb).squeeze(0),
                k[..., rot_dim:],
            ],
            dim=-1,
        )

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        x = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
        x = x.reshape(N, -1)
        x = self.proj(x)
        return x


class ParallelMLP(MegatronModule):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = QuickGELU,
    ) -> None:
        super().__init__()
        args = get_args()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = tensor_parallel.ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=True,
            gather_output=False,
            skip_bias_add=True,
            async_tensor_model_parallel_allreduce=args.vision_async_tensor_model_parallel_allreduce,
            params_dtype=args.params_dtype,
            use_cpu_initialization=args.use_cpu_initialization,
            perform_initialization=args.perform_initialization,
            gradient_accumulation_fusion=args.gradient_accumulation_fusion,
            sequence_parallel_enabled=args.vision_sequence_parallel,
        )

        self.act = act_layer()

        self.fc2 = tensor_parallel.RowParallelLinear(
            hidden_features,
            out_features,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=True,
            params_dtype=args.params_dtype,
            use_cpu_initialization=args.use_cpu_initialization,
            perform_initialization=args.perform_initialization,
            gradient_accumulation_fusion=args.gradient_accumulation_fusion,
            sequence_parallel_enabled=args.vision_sequence_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel, bias_parallel = self.fc1(x)
        x_parallel = self.act(x_parallel + bias_parallel)
        x, x_bias = self.fc2(x_parallel)
        return x + x_bias


class ParallelAttention(MegatronModule):
    def __init__(
        self,
        init_method: Callable,
        output_layer_init_method: Callable,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        projection_size: Optional[int] = None,
        query_key_value_name: str = 'qkv',
        dense_name: str = 'proj',
    ) -> None:
        super().__init__()
        args = get_args()
        self.query_key_value_name = query_key_value_name
        self.dense_name = dense_name
        self.projection_size = projection_size

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = core.utils.divide(projection_size, num_heads)
        self.num_attention_heads_per_partition = core.utils.divide(num_heads, world_size)

        self.add_module(
            query_key_value_name,
            tensor_parallel.ColumnParallelLinear(
                embed_dim,
                3 * projection_size,
                bias=True,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.vision_async_tensor_model_parallel_allreduce,
                params_dtype=args.params_dtype,
                use_cpu_initialization=args.use_cpu_initialization,
                perform_initialization=args.perform_initialization,
                gradient_accumulation_fusion=args.gradient_accumulation_fusion,
                sequence_parallel_enabled=args.vision_sequence_parallel,
            ),
        )

        self.add_module(
            self.dense_name,
            tensor_parallel.RowParallelLinear(
                projection_size,
                embed_dim,
                bias=True,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                skip_bias_add=False,
                params_dtype=args.params_dtype,
                use_cpu_initialization=args.use_cpu_initialization,
                perform_initialization=args.perform_initialization,
                gradient_accumulation_fusion=args.gradient_accumulation_fusion,
                sequence_parallel_enabled=args.vision_sequence_parallel,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.get_submodule(self.query_key_value_name)(x)

        # [s, b, head * 3 * head_dim] --> [s, b, head, 3 * head_dim]
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        x = x.view(*new_x_shape)

        # [s, b, head, 3 * head_dim] --> 3 [s, b, head, head_dim]
        q, k, v = tensor_parallel.split_tensor_along_last_dim(x, 3)

        rot_dim = rotary_pos_emb.shape[-1] * 2
        q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous() for x in (q, k, v)]
        with tensor_parallel.get_cuda_rng_tracker().fork():
            q = torch.cat(
                [apply_rotary_pos_emb(q[..., :rot_dim], rotary_pos_emb), q[..., rot_dim:]],
                dim=-1,
            )
            k = torch.cat(
                [apply_rotary_pos_emb(k[..., :rot_dim], rotary_pos_emb), k[..., rot_dim:]],
                dim=-1,
            )
            batch_size, seqlen_q, seqlen_k = q.shape[0], q.shape[1], k.shape[1]
            q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
            if cu_seqlens is not None and self.training:
                cu_seqlens_q = cu_seqlens
                seqlen_q = seqlen_k = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            else:
                cu_seqlens_q = torch.arange(
                    0,
                    (batch_size + 1) * seqlen_q,
                    step=seqlen_q,
                    dtype=torch.int32,
                    device=q.device,
                )
            cu_seqlens_k = cu_seqlens_q
            output = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k, 0, causal=False
            )
            context_layer = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

        output, _ = self.get_submodule(self.dense_name)(context_layer)

        return output


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: nn.Module = QuickGELU,
        enable_mp: bool = False,
        norm_layer: nn.Module = partial(ApexFusedLayerNorm, eps=1e-6),
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim, sequence_parallel=sequence_parallel)
        self.norm2 = norm_layer(dim, sequence_parallel=sequence_parallel)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if enable_mp:
            # from megatron.model.utils import init_method_normal

            self.attn = ParallelAttention(
                init_method_normal(0.02),
                init_method_normal(0.02),
                embed_dim=dim,
                num_heads=num_heads,
                projection_size=dim,
            )
            self.mlp = ParallelMLP(dim, mlp_hidden_dim, act_layer=act_layer)
        else:
            self.attn = Attention(dim, num_heads=num_heads)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rotary_pos_emb=rotary_pos_emb, cu_seqlens=cu_seqlens)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        temporal_merge_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = partial(ApexFusedLayerNorm, eps=1e-6),
        grad_checkpointing: bool = False,
        enable_mp: bool = False,
        vision_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_ratio * embed_dim
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit = spatial_merge_size**2
        self.temporal_merge_size = temporal_merge_size
        self.all_merge_unit = self.spatial_merge_unit * temporal_merge_size
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.head_dim = embed_dim // num_heads
        self.rotary_pos_emb = RotaryEmbedding(self.head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer,
                    enable_mp=enable_mp,
                    sequence_parallel=vision_sequence_parallel,
                )
                for i in range(depth)
            ]
        )

        self.grad_checkpointing = grad_checkpointing

        self.merger = PatchEmbMerger(
            d_model=num_classes, context_dim=embed_dim, spatial_merge_size=spatial_merge_size
        )

        self.vision_bos_eos_token = nn.Embedding(2, num_classes)
        trunc_normal_(self.vision_bos_eos_token.weight, std=0.02)

        self.enable_mp = enable_mp
        self.vision_sequence_parallel = vision_sequence_parallel
        self.tp_world_size = mpu.get_tensor_model_parallel_world_size()

        self.apply(self._init_weights)
        self.set_vit_sync()

    def set_vit_sync(self):
        for name, param in self.named_parameters():
            if not name.startswith("blocks"):
                setattr(param, "vit_sync", True)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = 0.02
        if (
            isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Embedding)
            or isinstance(module, nn.Linear)
        ):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def get_num_layers(self) -> int:
        return len(self.blocks)

    def lock(self, layers: int) -> None:
        self.patch_embed.proj.requires_grad_(False)
        for i in range(layers):
            self.blocks[i].requires_grad_(False)

    def load_state_dict(self, state_dict, strict=False):
        args = get_args()

        def _getattr(obj, key):
            cur = obj
            for k in key.split('.'):
                cur = getattr(cur, k)
            return cur

        def is_match(key):
            return key in state_dict and state_dict[key].shape == _getattr(self, key).shape

        if "norm_pre.weight" in state_dict:
            del state_dict["norm_pre.weight"]
            del state_dict["norm_pre.bias"]

        # inflate 2d conv kernels to 3d
        if "patch_embed.proj.weight" in state_dict and not is_match("patch_embed.proj.weight"):
            v = state_dict["patch_embed.proj.weight"]
            state_dict["patch_embed.proj.weight"] = (
                v.unsqueeze(2).repeat(1, 1, self.temporal_patch_size, 1, 1)
                / self.temporal_patch_size
            )

        if self.merger is not None and not is_match('merger.ln_q.weight'):
            s = self.merger.state_dict(prefix='merger.')
            for key, val in s.items():
                torch.distributed.broadcast(val, 0)
                state_dict[key] = val
            if torch.distributed.get_rank() == 0:
                print(' > visual can not find merger in state_dict')

        if not is_match('vision_bos_eos_token.weight'):
            s = self.vision_bos_eos_token.state_dict()
            torch.distributed.broadcast(s['weight'], 0)
            state_dict['vision_bos_eos_token.weight'] = s['weight']
            if torch.distributed.get_rank() == 0:
                print(' > visual can not vision_bos_eos_token in state_dict')

        if (
            args.vision_checkpoint_convert_mp
            and args.vit_enable_mp
            and mpu.get_tensor_model_parallel_world_size() > 1
        ):
            new_state_dict: dict = {}
            tp_world_size = mpu.get_tensor_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            head_partition = self.num_heads // tp_world_size
            input_size_partition = self.embed_dim // tp_world_size
            mlp_size_partition = self.mlp_hidden_dim // tp_world_size
            for k, v in state_dict.items():
                if k.startswith("blocks.") and "qkv.weight" in k:
                    assert tuple(v.shape) == (3 * self.num_heads * self.head_dim, self.embed_dim)
                    v = v.view(3, self.num_heads, self.head_dim, self.embed_dim)
                    v = v.transpose(0, 1)[
                        tp_rank * head_partition : (tp_rank + 1) * head_partition, ...
                    ]
                    v = v.reshape(-1, self.embed_dim)
                elif k.startswith("blocks.") and "qkv.bias" in k:
                    assert tuple(v.shape) == (3 * self.num_heads * self.head_dim,)
                    v = v.view(3, self.num_heads, self.head_dim)
                    v = v.transpose(0, 1)[
                        tp_rank * head_partition : (tp_rank + 1) * head_partition, ...
                    ]
                    v = v.reshape(-1)
                elif k.startswith("blocks.") and "proj.weight" in k:
                    assert tuple(v.shape) == (self.embed_dim, self.embed_dim)
                    v = v[
                        ..., tp_rank * input_size_partition : (tp_rank + 1) * input_size_partition
                    ]
                elif k.startswith("blocks.") and "fc1.weight" in k:
                    assert tuple(v.shape) == (self.mlp_hidden_dim, self.embed_dim)
                    v = v[tp_rank * mlp_size_partition : (tp_rank + 1) * mlp_size_partition, ...]
                elif k.startswith("blocks.") and "fc1.bias" in k:
                    assert tuple(v.shape) == (self.mlp_hidden_dim,)
                    v = v[tp_rank * mlp_size_partition : (tp_rank + 1) * mlp_size_partition]
                elif k.startswith("blocks.") and "fc2.weight" in k:
                    assert tuple(v.shape) == (self.embed_dim, self.mlp_hidden_dim)
                    v = v[..., tp_rank * mlp_size_partition : (tp_rank + 1) * mlp_size_partition]
                new_state_dict[k] = v
            print_rank_0(super().load_state_dict(new_state_dict, strict=strict))
        else:
            print_rank_0(super().load_state_dict(state_dict, strict=strict))

    def add_special_tokens(self, x: torch.Tensor, cu_seqlens, special_token_weight) -> torch.Tensor:
        output = torch.zeros(
            (x.size(0) + len(cu_seqlens) * 2 - 2, x.size(1)), device=x.device, dtype=x.dtype
        )
        plus_ids = torch.arange(cu_seqlens.size(0) - 1, device=cu_seqlens.device, dtype=torch.int32)
        start_ids = cu_seqlens[:-1] + plus_ids * 2
        end_ids = cu_seqlens[1:] + plus_ids * 2 + 1
        vision_tokens_mask = torch.ones(output.size(0), device=output.device, dtype=torch.bool)
        vision_tokens_mask[start_ids] = False
        vision_tokens_mask[end_ids] = False
        output[start_ids] = special_token_weight[0].to(x.dtype)
        output[end_ids] = special_token_weight[1].to(x.dtype)
        output[vision_tokens_mask] = x
        return output

    def forward(
        self,
        x: torch.Tensor,
        grid_ts: list[int],
        grid_hs: list[int],
        grid_ws: list[int],
        vision_seqlens: list[int],
    ) -> torch.Tensor:

        pos_ids: list = []
        cu_seqlens: list = [0]
        cu_vision_seqlens: list = [0]
        max_grid_size = 0
        for grid_t, grid_h, grid_w in zip(grid_ts, grid_hs, grid_ws):
            max_grid_size = max(max_grid_size, grid_h, grid_w)
            hpos_ids = torch.arange(grid_h).unsqueeze(1).expand(-1, grid_w)
            wpos_ids = torch.arange(grid_w).unsqueeze(0).expand(grid_h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    grid_h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    grid_w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    grid_h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    grid_w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            cu_seqlens.extend(
                [cu_seqlens[-1] + i * grid_h * grid_w for i in range(1, grid_t + 1)]
            )
            cu_vision_seqlens.append(cu_vision_seqlens[-1] + grid_h * grid_w // 4 * grid_t)
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(grid_t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)

        x = self.patch_embed(x)
        seq_len, _ = x.size()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)

        if self.enable_mp:
            x = x.unsqueeze(1)
            x = core.utils.make_viewless_tensor(x, x.requires_grad, keep_graph=True)
            if self.vision_sequence_parallel:
                if seq_len % self.tp_world_size != 0:
                    pad_seqlen = (
                        (seq_len + self.tp_world_size - 1)
                        // self.tp_world_size
                        * self.tp_world_size
                    )
                    x = F.pad(x, (0, 0, 0, 0, 0, pad_seqlen - seq_len), 'constant', 0)
                    rotary_pos_emb = F.pad(
                        rotary_pos_emb, (0, 0, 0, pad_seqlen - seq_len), 'constant', 0
                    )
                    cu_seqlens.append(cu_seqlens[-1] + pad_seqlen - seq_len)
                x = tensor_parallel.scatter_to_sequence_parallel_region(x)
        cu_seqlens = torch.tensor(cu_seqlens, device=x.device, dtype=torch.int32)

        for blk in self.blocks:
            if self.grad_checkpointing:
                # x = checkpoint(blk, x, rotary_pos_emb, cu_seqlens, use_reentrant=False)
                x = tensor_parallel.checkpoint(
                    blk,
                    False,
                    x, rotary_pos_emb, cu_seqlens
                )
            else:
                x = blk(x, rotary_pos_emb=rotary_pos_emb, cu_seqlens=cu_seqlens)

        if self.vision_sequence_parallel:
            x = tensor_parallel.gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False)
            if seq_len % self.tp_world_size > 0:
                x = x[:seq_len]
                cu_seqlens = cu_seqlens[:-1]

        x = self.merger(x)
        cu_vision_seqlens = torch.tensor(cu_vision_seqlens, device=x.device, dtype=torch.int32)
        output = self.add_special_tokens(x, cu_vision_seqlens, self.vision_bos_eos_token.weight)
        return output
