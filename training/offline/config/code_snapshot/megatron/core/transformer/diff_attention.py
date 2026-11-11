# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
import os
from typing import Union

import torch
from einops import rearrange

from megatron.core import parallel_state
from megatron.core.utils import init_method_normal
from megatron.core.models.common.embeddings import (
    apply_rotary_pos_emb,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.shared_layer import SharedParam
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import DiffTransformerConfig

from megatron.core.tensor_parallel import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
_ENABLE_FUSION_FOR_DIFF_ATTN = os.getenv("QWEN_DIFF_ATTN_FUSION", "0").strip() == "1"


@dataclass
class DiffSelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    v_layernorm: Union[ModuleSpec, type] = None
    lambda_module: Union[ModuleSpec, type] = None


class DiffSelfAttention(SelfAttention):
    """Multi-Latent Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
            self,
            config: DiffTransformerConfig,
            submodules: Union[DiffSelfAttentionSubmodules],
            layer_number: int,
            attn_mask_type: AttnMaskType,
            cp_comm_type: str = None,
    ) -> None:
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        assert (
                world_size == 1
        ), "MLA is not supported with Tensor Parallelism yet, \
        use Expert Parallelism and Pipeline Parallelism for better performance."

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            cp_comm_type=cp_comm_type,
            attn_mask_type=attn_mask_type,
        )
        self.lambda_init = config.diff_lambda_init
        self.use_vectorized_lambda = config.use_vectorized_lambda
        if self.config.use_vectorized_lambda:
            # Disable SharedParam as it is not compatible with dp overlap... why..
            self.lambda_q1 = torch.nn.Parameter(torch.zeros(config.kv_channels, dtype=torch.float32))
            self.lambda_k1 = torch.nn.Parameter(torch.zeros(config.kv_channels, dtype=torch.float32))
            self.lambda_q2 = torch.nn.Parameter(torch.zeros(config.kv_channels, dtype=torch.float32))
            self.lambda_k2 = torch.nn.Parameter(torch.zeros(config.kv_channels, dtype=torch.float32))
            if config.perform_initialization:
                with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
                    self.lambda_q1.data.normal_(mean=0, std=config.init_method_std)
                    self.lambda_k1.data.normal_(mean=0, std=config.init_method_std)
                    self.lambda_q2.data.normal_(mean=0, std=config.init_method_std)
                    self.lambda_k2.data.normal_(mean=0, std=config.init_method_std)

            setattr(self.lambda_q1, 'sequence_parallel', config.sequence_parallel)
            setattr(self.lambda_k1, 'sequence_parallel', config.sequence_parallel)
            setattr(self.lambda_q2, 'sequence_parallel', config.sequence_parallel)
            setattr(self.lambda_k2, 'sequence_parallel', config.sequence_parallel)

        else:
            self.lambda_1 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.lambda_2 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            setattr(self.lambda_1, 'sequence_parallel', config.sequence_parallel)
            setattr(self.lambda_2, 'sequence_parallel', config.sequence_parallel)


        assert submodules.v_layernorm is not None
        self.v_layernorm = build_module(
            submodules.v_layernorm,
            hidden_size=self.hidden_size_per_attention_head * 2,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )


    @torch.compile(dynamic=True, disable=not _ENABLE_FUSION_FOR_DIFF_ATTN)  # 用 complie 加速 elementwise 的操作
    def _diff_attention(self, context_layer):
        if self.use_vectorized_lambda:
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        else:
            lambda_1 = torch.exp(self.lambda_1.float())
            lambda_2 = torch.exp(self.lambda_2.float())
        lambda_full = (lambda_1 - lambda_2 + self.lambda_init).view(1, 1, -1, 1).type_as(context_layer)
        context_layer = context_layer[:, :, :context_layer.shape[2] // 2, :] - \
                        lambda_full * context_layer[:, :, context_layer.shape[2] // 2:, :]
        if self.v_layernorm:
            context_layer = self.v_layernorm(context_layer.contiguous())
        if self.lambda_init > 0:
            context_layer = context_layer * (1 - self.lambda_init)
        return context_layer

    def forward(
            self,
            hidden_states,
            attention_mask,
            key_value_states=None,
            inference_params=None,
            rotary_pos_emb=None,
            packed_seq_params=None,
    ):
        """
        Perform a forward pass through the attention module.
        """

        # hidden_states: [sq, b, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        seq_len, bsz = query.size()[0], query.size()[1]

        num_q_heads = query.size()[2]
        value = value.reshape(value.shape[0], value.shape[1], value.shape[2] // 2,
                              -1).repeat(1, 1, 2, 1)
        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb
        )

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            # if packed_seq_params is not None:
            #     cu_seqlens_q = packed_seq_params.cu_seqlens_q
            #     cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            # else:
            # cu_seqlens_q = cu_seqlens_kv = None
            # Follow previous version, reset attention only.
            cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(
                query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
            )
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if packed_seq_params is not None:
            ## convert sbhd to thd
            query, key, value = [rearrange(x, 's b ... -> (b s) ...') for x in (query, key, value)]

        if not self.training and packed_seq_params is None:
            packed_seq_params = {'qkv_format':'sbhd'}
        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        if packed_seq_params is not None and packed_seq_params['qkv_format'] == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case

            core_attn_out = rearrange(core_attn_out, '(b s) (h d) -> s b h d', b=bsz, h=num_q_heads)
        else:
            core_attn_out = rearrange(core_attn_out, 's b (h d) -> s b h d', h=num_q_heads)
        core_attn_out = self._diff_attention(core_attn_out)
        core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)').contiguous()

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)

        return output, bias
