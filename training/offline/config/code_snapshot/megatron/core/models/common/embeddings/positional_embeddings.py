# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import importlib.util
import torch
from torch import einsum, nn

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
    from einops import rearrange
    use_flash_rotary = True
    print("> use flash_attn rotary")
except ImportError:
    use_flash_rotary = False
    print("> import flash_attn rotary fail")


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.precision = precision

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = einsum("i,j->ij", t, self.inv_freq)
        if self.precision == torch.bfloat16:
            sinusoid_inp = sinusoid_inp.float()
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        if self.precision == torch.bfloat16:
            sin, cos = sin.bfloat16(), cos.bfloat16()
        emb = torch.cat((sin, cos), dim=-1)
        return emb[None, :, :]


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, use_fp32=False, use_outer_in_rope=False):
        super().__init__()
        self.dim = dim
        self.base = base
        self.use_fp32 = use_fp32
        if use_fp32:
            self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self.use_outer_in_rope = use_outer_in_rope
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=self.inv_freq.device).float() / self.dim))
            self._seq_len_cached = seqlen
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(seqlen, device=self.inv_freq.device)
            # Don't do einsum, it converts fp32 to fp16
            if self.use_outer_in_rope:
                freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)
            else:
                freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
            # first part even vector components, second part odd vector components,
            #  2 * dim in dimension size
            emb = torch.cat((freqs, freqs), dim=-1)
            # emb [seq_length, .., dim]
            self._rotary_pos_emb_cache = rearrange(emb, 'n d -> n 1 1 d')

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return self._rotary_pos_emb_cache[offset:offset+max_seq_len]

# rotary pos emb helpers:



def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs, use_fp32=False):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """

    if use_flash_rotary and use_fp32:
        t_ = rearrange(t, 's b ... -> b s ...').contiguous()
        if use_fp32:
            t_ = t_.float()
        freqs = freqs.squeeze((1, 2))
        cos = freqs[:, :freqs.shape[-1] // 2].cos()
        sin = freqs[:, :freqs.shape[-1] // 2].sin()
        output = apply_rotary_emb_func(t_, cos, sin).type_as(t)
        return rearrange(output, 'b s ... -> s b ...')

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
    if use_fp32:
        t_ = t_.float()
        t_pass_ = t_pass_.float()
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t_ = (t_ * freqs.cos()) + (_rotate_half(t_) * freqs.sin())
    return torch.cat((t_, t_pass_), dim=-1).type_as(t)


class AliBi(torch.nn.Module):
    def __init__(self, num_heads, mp_size=1, mp_rank=1):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        assert mp_size <= num_heads and mp_rank <= mp_size
        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(self._get_slopes(num_heads))[
            mp_rank * self.slice_size : (mp_rank + 1) * self.slice_size
        ]
        self.register_buffer("slopes", slopes)

    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(x.device).to(x.dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return x + a