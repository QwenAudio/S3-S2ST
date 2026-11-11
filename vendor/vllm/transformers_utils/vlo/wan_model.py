# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from flash_attn.layers.rotary import apply_rotary_emb_func

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.transformers_utils.vlo.attention import flash_attention
from vllm.utils.deep_gemm import fp8_gemm_nt

__all__ = ["WanModel"]

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def reshape_by_blocks(matrix, a, b=1):
    m, n = matrix.shape
    assert m % a == 0 and n % b == 0, "矩阵维度必须能被块大小整除"

    # 将矩阵重塑为(m//a, a, n//b, b)形状
    reshaped = matrix.reshape(m // a, a, n // b, b)

    # 交换中间两个维度，从(m//a, a, n//b, b)变为(m//a, n//b, a, b)
    permuted = reshaped.permute(0, 2, 1, 3)

    # 重塑为最终形状(-1, a*b)
    final = permuted.reshape(-1, a * b)

    return final


def inverse_reshape_by_blocks(blocked_matrix, m, n, a, b=1):
    assert m % a == 0 and n % b == 0, "矩阵维度必须能被块大小整除"

    # 从(-1, a*b)重塑为(m//a, n//b, a, b)
    reshaped = blocked_matrix.reshape(m // a, n // b, a, b)

    # 交换中间两个维度，从(m//a, n//b, a, b)变为(m//a, a, n//b, b)
    permuted = reshaped.permute(0, 2, 1, 3)

    # 重塑为原始形状(m, n)
    final = permuted.reshape(m, n)

    return final


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply(x, grid_sizes, freqs):
    out_dtype = x.dtype
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(out_dtype)


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.qkv = nn.Linear(dim, dim * 3)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            if self.qkv.weight.dtype == torch.float8_e4m3fn:
                x = x.view(-1, x.shape[-1])
                x_fp8 = per_token_group_quant_fp8(x, 128, column_major_scales=True)
                out_qkv = torch.empty(
                    (x.shape[0], self.qkv.weight.shape[0]),
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                fp8_gemm_nt(
                    x_fp8, (self.qkv.weight, self.qkv.weight_scale_inv), out_qkv
                )
                if self.qkv.bias is not None:
                    out_qkv = out_qkv + self.qkv.bias
                out_q, out_k, out_v = out_qkv.chunk(3, dim=-1)
                q = self.norm_q(out_q).view(b, s, n, d)
                k = self.norm_k(out_k).view(b, s, n, d)
                v = out_v.view(b, s, n, d)

            else:
                q, k, v = self.qkv(x).chunk(3, dim=-1)
                q = self.norm_q(q).view(b, s, n, d)
                k = self.norm_k(k).view(b, s, n, d)
                v = v.view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        # output
        x = x.flatten(2)
        if self.o.weight.dtype == torch.float8_e4m3fn:
            x = x.view(-1, x.shape[-1])
            x_fp8 = per_token_group_quant_fp8(x, 128, column_major_scales=True)
            out_x = torch.empty(
                (x.shape[0], self.o.weight.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
            fp8_gemm_nt(x_fp8, (self.o.weight, self.o.weight_scale_inv), out_x)
            if self.o.bias is not None:
                out_x = out_x + self.o.bias
            x = out_x.view(b, s, -1)
        else:
            x = self.o(x)
        return x


def apply_rotary_pos_emb(t, freqs):
    out_dtype = t.dtype
    if freqs is None:
        return t
    B, L, H, C = t.shape
    t = t.view(B * L, 1, H, C)
    t = t.transpose(0, 1).contiguous()
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()
    output = apply_rotary_emb_func(t_, cos, sin, False, False).type_as(t)
    output = output.transpose(0, 1).contiguous()
    output = output.view(B, L, H, C)
    return output.to(out_dtype)


class WanT2VCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, k_pos_embed, freqs, grid_sizes, k_attn_mask):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            k_pos_embed(Tensor): Shape [B*L2, C / num_heads / 2]
            freqs(Tensor): Shape [1024, C / num_heads / 2]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            k_attn_mask(Tensor): Shape [B, L2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # compute query, key, value
        if self.q.weight.dtype == torch.float8_e4m3fn:
            x = x.view(-1, x.shape[-1])
            context = context.view(-1, context.shape[-1])
            x_fp8 = per_token_group_quant_fp8(x, 128, column_major_scales=True)
            context_fp8 = per_token_group_quant_fp8(
                context, 128, column_major_scales=True
            )
            out_q = torch.empty(
                (x.shape[0], self.q.weight.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
            out_kv = torch.empty(
                (context.shape[0], self.kv.weight.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
            fp8_gemm_nt(x_fp8, (self.q.weight, self.q.weight_scale_inv), out_q)
            fp8_gemm_nt(context_fp8, (self.kv.weight, self.kv.weight_scale_inv), out_kv)
            if self.q.bias is not None:
                out_q = out_q + self.q.bias
            if self.kv.bias is not None:
                out_kv = out_kv + self.kv.bias
            q = self.norm_q(out_q).view(b, -1, n, d)
            out_k, out_v = out_kv.chunk(2, dim=-1)
            k = self.norm_k(out_k).view(b, -1, n, d)
            v = out_v.view(b, -1, n, d)
        else:
            q = self.norm_q(self.q(x)).view(b, -1, n, d)
            k, v = self.kv(context).chunk(2, dim=-1)
            k = self.norm_k(k).view(b, -1, n, d)
            v = v.view(b, -1, n, d)

        # compute attention

        x = flash_attention(
            rope_apply(q, grid_sizes, freqs),
            apply_rotary_pos_emb(k, k_pos_embed),
            v,
            k_lens=k_attn_mask.sum(dim=-1),
            k_attn_mask=k_attn_mask,
        )

        # output
        x = x.flatten(2)
        if self.o.weight.dtype == torch.float8_e4m3fn:
            x = x.view(-1, x.shape[-1])
            x_fp8 = per_token_group_quant_fp8(x, 128, column_major_scales=True)
            out_x = torch.empty(
                (x.shape[0], self.o.weight.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
            fp8_gemm_nt(x_fp8, (self.o.weight, self.o.weight_scale_inv), out_x)
            if self.o.bias is not None:
                out_x = out_x + self.o.bias
            x = out_x.view(b, s, -1)
        else:
            x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, position_ids, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class FFN(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.up_proj = nn.Linear(dim, ffn_dim)
        self.down_proj = nn.Linear(ffn_dim, dim)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        b, s, d = x.shape
        if self.up_proj.weight.dtype == torch.float8_e4m3fn:
            x = x.view(-1, x.shape[-1])
            x_fp8 = per_token_group_quant_fp8(x, 128, column_major_scales=True)
            out_x = torch.empty(
                (x.shape[0], self.up_proj.weight.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
            fp8_gemm_nt(
                x_fp8, (self.up_proj.weight, self.up_proj.weight_scale_inv), out_x
            )
            if self.up_proj.bias is not None:
                out_x = out_x + self.up_proj.bias
            x = out_x
        else:
            x = self.up_proj(x)
        x = self.activation(x)
        if self.down_proj.weight.dtype == torch.float8_e4m3fn:
            x_fp8 = per_token_group_quant_fp8(x, 128, column_major_scales=True)
            out_x = torch.empty(
                (x.shape[0], self.down_proj.weight.shape[0]),
                device=x.device,
                dtype=torch.bfloat16,
            )
            fp8_gemm_nt(
                x_fp8, (self.down_proj.weight, self.down_proj.weight_scale_inv), out_x
            )
            if self.down_proj.bias is not None:
                out_x = out_x + self.down_proj.bias
            x = out_x.view(b, s, -1)
        else:
            x = self.down_proj(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = FFN(dim, ffn_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self, x, e, seq_lens, grid_sizes, freqs, context, k_pos_embed, k_attn_mask
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [L, 6, C] a timestep table for each token (all batch share the same)
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=1)
        e = [u.unsqueeze(0).squeeze(-2) for u in e]

        def modulate(x, scale, shift):
            return x * (1 + scale) + shift

        def gatemodulate(x, gate):
            return x * gate

        # self-attention
        y = self.self_attn(
            modulate(self.norm1(x), scale=e[1], shift=e[0]), seq_lens, grid_sizes, freqs
        )
        x = x + gatemodulate(y, gate=e[2])

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, e):
            x = x + self.cross_attn(
                self.norm3(x),
                context,
                k_pos_embed=k_pos_embed,
                freqs=freqs,
                grid_sizes=grid_sizes,
                k_attn_mask=k_attn_mask,
            )
            y = self.ffn(modulate(self.norm2(x), scale=e[4], shift=e[3]))
            x = x + gatemodulate(y, gate=e[5])
            return x

        x = cross_attn_ffn(x, context, e)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [L, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        e = [u.unsqueeze(0).squeeze(-2) for u in e]
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280)
            )

    def forward(self, image_embeds):
        if hasattr(self, "emb_pos"):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, rotary_base=10000.0):
        super().__init__()
        self.dim = dim
        self.rotary_base = rotary_base
        inv_freq = 1.0 / (rotary_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen):
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (
                self.rotary_base
                ** (
                    torch.arange(0, self.dim, 2, device=self.inv_freq.device).float()
                    / self.dim
                )
            )
            seq = torch.arange(
                seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen):
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        use_mrope=False,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v", "vace"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_mrope = use_mrope

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            WanRMSNorm(text_dim),
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v" or model_type == "flf2v":
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")
        self.rotary_dim = d
        self.rotary_emb = RotaryEmbedding(d, 10000)

        # initialize weights
        self.init_weights()

    def locate_parent(self, root: torch.nn.Module, full_path: str):
        parent = root
        path = full_path.split(".")
        for p in path[:-1]:
            parent = getattr(parent, p)
        return parent

    def fp8_quantize(self, v):
        # pre group
        qmax = torch.finfo(torch.float8_e4m3fn).max
        shape = v.shape

        # per group
        reshaped_v = reshape_by_blocks(v, 128, 128)
        scales = reshaped_v.abs().amax(dim=-1, keepdim=True) / qmax
        reshaped_v = reshaped_v / scales
        scales = inverse_reshape_by_blocks(
            scales, shape[0] // 128, shape[1] // 128, 1, 1
        )
        reshaped_v = inverse_reshape_by_blocks(reshaped_v, shape[0], shape[1], 128, 128)

        reshaped_v = reshaped_v.to(torch.float8_e4m3fn)
        scales = scales.to(torch.float32)

        return reshaped_v, scales

    def quantize(self, quantization_method):
        assert quantization_method in ["fp8"], (
            "currently WanModel only supports fp8 quantization."
        )
        for name, params in self.blocks.named_parameters():
            if "weight" in name and "norm" not in name:
                new_data, new_data_scale = self.fp8_quantize(params.data)
                params.data = new_data
                module = self.locate_parent(self.blocks, name)
                module.weight_scale_inv = new_data_scale

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        position_ids,
        timestep_table=None,
        clip_fea=None,
        y=None,
        k_attn_mask=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v" or self.model_type == "flf2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        # time embeddings
        # we assume all data share the same timestep table
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep_table).bfloat16()
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        # context_lens = k_attn_mask.sum(dim=-1)
        context = self.text_embedding(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        if self.use_mrope:
            position_ids = position_ids.view(3, -1)

            (
                rotary_pos_emb_t,
                rotary_pos_emb_h,
                rotary_pos_emb_w,
            ) = self.rotary_emb(position_ids.size(1))[position_ids].unbind(0)
            rotary_pos_emb = torch.cat(
                (
                    rotary_pos_emb_t[:, 0:16],
                    rotary_pos_emb_h[:, 16:40],
                    rotary_pos_emb_w[:, 40:64],
                ),
                dim=1,
            )
            assert rotary_pos_emb.size(-1) == self.rotary_dim // 2

            k_pos_embed = rotary_pos_emb
        else:
            k_pos_embed = None

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            k_attn_mask=k_attn_mask,
            k_pos_embed=k_pos_embed,
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def get_checkpoint(self, ckpt_path_or_ckpt):
        if not isinstance(ckpt_path_or_ckpt, str):
            checkpoint = ckpt_path_or_ckpt
            ckpt_type = "safetensors"
        else:
            ckpt_type = ckpt_path_or_ckpt.split(".")[-1]
            if ckpt_type == "safetensors":
                from safetensors.torch import load_file

                checkpoint = load_file(ckpt_path_or_ckpt)
            else:
                checkpoint = torch.load(ckpt_path_or_ckpt, weights_only=True)

        # modify key names

        # seperate ffn
        ffn_k = [k for k in checkpoint if "ffn" in k]
        for k in ffn_k:
            if "0.bias" in k:
                checkpoint[k.replace("0.bias", "up_proj.bias")] = checkpoint[k]
            elif "0.weight" in k:
                checkpoint[k.replace("0.weight", "up_proj.weight")] = checkpoint[k]
            elif "2.bias" in k:
                checkpoint[k.replace("2.bias", "down_proj.bias")] = checkpoint[k]
            elif "2.weight" in k:
                checkpoint[k.replace("2.weight", "down_proj.weight")] = checkpoint[k]
            del checkpoint[k]

        # self attn q,k,v --> qkv
        self_ks = [k for k in checkpoint if "self_attn.q.weight" in k]
        for k in self_ks:
            q_weight = checkpoint.pop(k)
            k_weight = checkpoint.pop(k.replace(".q.", ".k."))
            v_weight = checkpoint.pop(k.replace(".q.", ".v."))
            checkpoint[k.replace(".q.", ".qkv.")] = torch.cat(
                [q_weight, k_weight, v_weight], dim=0
            )
        self_ks = [k for k in checkpoint if "self_attn.q.bias" in k]
        for k in self_ks:
            q_bias = checkpoint.pop(k)
            k_bias = checkpoint.pop(k.replace(".q.", ".k."))
            v_bias = checkpoint.pop(k.replace(".q.", ".v."))
            checkpoint[k.replace(".q.", ".qkv.")] = torch.cat(
                [q_bias, k_bias, v_bias], dim=0
            )

        # cross attn k,v --> kv
        cross_ks = [k for k in checkpoint if "cross_attn.k.weight" in k]
        for k in cross_ks:
            k_weight = checkpoint.pop(k)
            v_weight = checkpoint.pop(k.replace(".k.", ".v."))
            checkpoint[k.replace(".k.", ".kv.")] = torch.cat(
                [k_weight, v_weight], dim=0
            )
        cross_ks = [k for k in checkpoint if "cross_attn.k.bias" in k]
        for k in cross_ks:
            k_bias = checkpoint.pop(k)
            v_bias = checkpoint.pop(k.replace(".k.", ".v."))
            checkpoint[k.replace(".k.", ".kv.")] = torch.cat([k_bias, v_bias], dim=0)

        # merge qkv
        return checkpoint
