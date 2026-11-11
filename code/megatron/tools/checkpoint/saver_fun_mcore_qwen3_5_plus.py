# Copyright (c) 2025. Megatron M-Core saver for Qwen3.5-MoE multimodal model.
"""Save Megatron checkpoint for Qwen3.5-MoE with multimodal support.

Handles:
- Heterogeneous layers (linear_attention + full_attention)
- Attention output gating (doubled Q projection)
- GatedDeltaNet linear attention weights with TP splitting
- MoE with shared expert + sigmoid gate
- Audio encoder / adaptor multimodal parts
- torch / torch_dist checkpoint formats
"""

import os
import sys
import torch
from importlib.metadata import version
# from pkg_resources import packaging
# from packaging import version
from packaging.version import Version
from importlib.metadata import version  # 获取安装包版本号
import types

from setter import ModelSetter
from utils import get_mcore_transformer_block_key, print_memory_usage

random_init_audio_encoder = os.environ.get('Random_Init_Audio_Encoder', 'False')
random_init_audio_encoder = True if random_init_audio_encoder == 'True' else False
random_init_audio_adaptor = os.environ.get('Random_Init_Audio_Adaptor', 'False')
random_init_audio_adaptor = True if random_init_audio_adaptor == 'True' else False
channels_model = os.environ.get('Channels_Model', 'False')
channels_model = True if channels_model == 'True' else False
print(f'random_init_audio_encoder: {random_init_audio_encoder}')
print(f'random_init_audio_adaptor: {random_init_audio_adaptor}')
print(f'channels_model: {channels_model}')


class MCoreSetter(ModelSetter):

    transformer_block_key = None

    @classmethod
    def get_transformer_block(cls, model):
        return getattr(model, cls.transformer_block_key)

    @classmethod
    def has_position_embeddings(cls, model):
        return hasattr(model.embedding, "position_embeddings")

    @classmethod
    def set_embeddings(cls, model, word=None, pos=None):
        cls.set_tensor(model.embedding.word_embeddings.weight, word)
        if pos is not None:
            cls.set_tensor(model.embedding.position_embeddings.weight, pos)

    @classmethod
    def set_final_norm(cls, model, weight=None, bias=None):
        block = cls.get_transformer_block(model)
        cls.set_tensor(block.final_layernorm.weight, weight)
        if bias is not None:
            cls.set_tensor(block.final_layernorm.bias, bias)

    @classmethod
    def set_output_word_embeddings(cls, model, emb=None):
        cls.set_tensor(model.embedding.word_embeddings.weight, emb)

    @classmethod
    def set_output_layer(cls, model, weight=None):
        cls.set_tensor(model.output_layer.weight, weight)

    @classmethod
    def set_pooler(cls, model, weight=None, bias=None):
        cls.set_tensor(model.pooler.dense.weight, weight)
        if bias is not None:
            cls.set_tensor(model.pooler.dense.bias, bias)

    @classmethod
    def set_audio_encoder(cls, model, audio_encoder=None):
        ae = audio_encoder["weight"]

        def _get(obj, key):
            return getattr(obj, key) if hasattr(obj, key) else obj[key]

        def _try_set(dst_tensor, src_tensor):
            if src_tensor is None:
                return
            try:
                cls.set_tensor(dst_tensor, src_tensor)
            except Exception:
                pass

        conv1 = _get(ae, "conv1")
        conv1_conv = _get(conv1, "conv")
        _try_set(model.audio_encoder.conv1.conv.weight, _get(conv1_conv, "weight"))
        _try_set(model.audio_encoder.conv1.conv.bias, _get(conv1_conv, "bias"))

        conv2 = _get(ae, "conv2")
        conv2_conv = _get(conv2, "conv")
        _try_set(model.audio_encoder.conv2.conv.weight, _get(conv2_conv, "weight"))
        _try_set(model.audio_encoder.conv2.conv.bias, _get(conv2_conv, "bias"))

        ln_post = _get(ae, "ln_post")
        _try_set(model.audio_encoder.ln_post.weight, _get(ln_post, "weight"))
        _try_set(model.audio_encoder.ln_post.bias, _get(ln_post, "bias"))

        src_blocks = _get(ae, "blocks")
        for idx, dst_blk in enumerate(model.audio_encoder.blocks):
            src_blk = src_blocks[idx]
            attn = _get(src_blk, "attn")
            _try_set(dst_blk.attn.query.weight, _get(_get(attn, "query"), "weight"))
            _try_set(dst_blk.attn.query.bias, _get(_get(attn, "query"), "bias"))
            _try_set(dst_blk.attn.key.weight, _get(_get(attn, "key"), "weight"))
            _try_set(dst_blk.attn.value.weight, _get(_get(attn, "value"), "weight"))
            _try_set(dst_blk.attn.value.bias, _get(_get(attn, "value"), "bias"))
            _try_set(dst_blk.attn.out.weight, _get(_get(attn, "out"), "weight"))
            _try_set(dst_blk.attn.out.bias, _get(_get(attn, "out"), "bias"))
            _try_set(dst_blk.attn.fsmn_block.weight, _get(_get(attn, "fsmn_block"), "weight"))

            attn_ln = _get(src_blk, "attn_ln")
            _try_set(dst_blk.attn_ln.weight, _get(attn_ln, "weight"))
            _try_set(dst_blk.attn_ln.bias, _get(attn_ln, "bias"))

            mlp = _get(src_blk, "mlp")
            if mlp[0] is not None:
                _try_set(dst_blk.mlp[0].weight, _get(mlp[0], "weight"))
                _try_set(dst_blk.mlp[0].bias, _get(mlp[0], "bias"))
            if mlp[2] is not None:
                _try_set(dst_blk.mlp[2].weight, _get(mlp[2], "weight"))
                _try_set(dst_blk.mlp[2].bias, _get(mlp[2], "bias"))

            mlp_ln = _get(src_blk, "mlp_ln")
            _try_set(dst_blk.mlp_ln.weight, _get(mlp_ln, "weight"))
            _try_set(dst_blk.mlp_ln.bias, _get(mlp_ln, "bias"))

    @classmethod
    def set_sensevoicesmall_audio_encoder(cls, model, audio_encoder=None):
        ae = audio_encoder["weight"]
        encoder = ae.encoder

        def _get(obj, key):
            return getattr(obj, key) if hasattr(obj, key) else obj[key]

        def _try_set(dst_tensor, src_tensor):
            if src_tensor is None:
                return
            try:
                cls.set_tensor(dst_tensor, src_tensor)
            except Exception:
                pass

        src_encoders0 = _get(encoder, "encoders0")
        for idx, dst_blk in enumerate(model.audio_encoder.encoder.encoders0):
            src_blk = src_encoders0[idx]
            self_attn = _get(src_blk, "self_attn")
            _try_set(dst_blk.self_attn.linear_out.weight, _get(_get(self_attn, "linear_out"), "weight"))
            _try_set(dst_blk.self_attn.linear_out.bias, _get(_get(self_attn, "linear_out"), "bias"))
            _try_set(dst_blk.self_attn.linear_q_k_v.weight, _get(_get(self_attn, "linear_q_k_v"), "weight"))
            _try_set(dst_blk.self_attn.linear_q_k_v.bias, _get(_get(self_attn, "linear_q_k_v"), "bias"))
            _try_set(dst_blk.self_attn.fsmn_block.weight, _get(_get(self_attn, "fsmn_block"), "weight"))
            feed_forward = _get(src_blk, "feed_forward")
            _try_set(dst_blk.feed_forward.w_1.weight, _get(_get(feed_forward, "w_1"), "weight"))
            _try_set(dst_blk.feed_forward.w_1.bias, _get(_get(feed_forward, "w_1"), "bias"))
            _try_set(dst_blk.feed_forward.w_2.weight, _get(_get(feed_forward, "w_2"), "weight"))
            _try_set(dst_blk.feed_forward.w_2.bias, _get(_get(feed_forward, "w_2"), "bias"))
            _try_set(dst_blk.norm1.weight, _get(_get(src_blk, "norm1"), "weight"))
            _try_set(dst_blk.norm1.bias, _get(_get(src_blk, "norm1"), "bias"))
            _try_set(dst_blk.norm2.weight, _get(_get(src_blk, "norm2"), "weight"))
            _try_set(dst_blk.norm2.bias, _get(_get(src_blk, "norm2"), "bias"))

        src_encoders = _get(encoder, "encoders")
        for idx, dst_blk in enumerate(model.audio_encoder.encoder.encoders):
            src_blk = src_encoders[idx]
            self_attn = _get(src_blk, "self_attn")
            _try_set(dst_blk.self_attn.linear_out.weight, _get(_get(self_attn, "linear_out"), "weight"))
            _try_set(dst_blk.self_attn.linear_out.bias, _get(_get(self_attn, "linear_out"), "bias"))
            _try_set(dst_blk.self_attn.linear_q_k_v.weight, _get(_get(self_attn, "linear_q_k_v"), "weight"))
            _try_set(dst_blk.self_attn.linear_q_k_v.bias, _get(_get(self_attn, "linear_q_k_v"), "bias"))
            _try_set(dst_blk.self_attn.fsmn_block.weight, _get(_get(self_attn, "fsmn_block"), "weight"))
            feed_forward = _get(src_blk, "feed_forward")
            _try_set(dst_blk.feed_forward.w_1.weight, _get(_get(feed_forward, "w_1"), "weight"))
            _try_set(dst_blk.feed_forward.w_1.bias, _get(_get(feed_forward, "w_1"), "bias"))
            _try_set(dst_blk.feed_forward.w_2.weight, _get(_get(feed_forward, "w_2"), "weight"))
            _try_set(dst_blk.feed_forward.w_2.bias, _get(_get(feed_forward, "w_2"), "bias"))
            _try_set(dst_blk.norm1.weight, _get(_get(src_blk, "norm1"), "weight"))
            _try_set(dst_blk.norm1.bias, _get(_get(src_blk, "norm1"), "bias"))
            _try_set(dst_blk.norm2.weight, _get(_get(src_blk, "norm2"), "weight"))
            _try_set(dst_blk.norm2.bias, _get(_get(src_blk, "norm2"), "bias"))

        src_tp_encoders = _get(encoder, "tp_encoders")
        for idx, dst_blk in enumerate(model.audio_encoder.encoder.tp_encoders):
            src_blk = src_tp_encoders[idx]
            self_attn = _get(src_blk, "self_attn")
            _try_set(dst_blk.self_attn.linear_out.weight, _get(_get(self_attn, "linear_out"), "weight"))
            _try_set(dst_blk.self_attn.linear_out.bias, _get(_get(self_attn, "linear_out"), "bias"))
            _try_set(dst_blk.self_attn.linear_q_k_v.weight, _get(_get(self_attn, "linear_q_k_v"), "weight"))
            _try_set(dst_blk.self_attn.linear_q_k_v.bias, _get(_get(self_attn, "linear_q_k_v"), "bias"))
            _try_set(dst_blk.self_attn.fsmn_block.weight, _get(_get(self_attn, "fsmn_block"), "weight"))
            feed_forward = _get(src_blk, "feed_forward")
            _try_set(dst_blk.feed_forward.w_1.weight, _get(_get(feed_forward, "w_1"), "weight"))
            _try_set(dst_blk.feed_forward.w_1.bias, _get(_get(feed_forward, "w_1"), "bias"))
            _try_set(dst_blk.feed_forward.w_2.weight, _get(_get(feed_forward, "w_2"), "weight"))
            _try_set(dst_blk.feed_forward.w_2.bias, _get(_get(feed_forward, "w_2"), "bias"))
            _try_set(dst_blk.norm1.weight, _get(_get(src_blk, "norm1"), "weight"))
            _try_set(dst_blk.norm1.bias, _get(_get(src_blk, "norm1"), "bias"))
            _try_set(dst_blk.norm2.weight, _get(_get(src_blk, "norm2"), "weight"))
            _try_set(dst_blk.norm2.bias, _get(_get(src_blk, "norm2"), "bias"))

        after_norm = _get(encoder, "after_norm")
        _try_set(model.audio_encoder.encoder.after_norm.weight, _get(after_norm, "weight"))
        _try_set(model.audio_encoder.encoder.after_norm.bias, _get(after_norm, "bias"))

        tp_norm = _get(encoder, "tp_norm")
        _try_set(model.audio_encoder.encoder.tp_norm.weight, _get(tp_norm, "weight"))
        _try_set(model.audio_encoder.encoder.tp_norm.bias, _get(tp_norm, "bias"))

    @classmethod
    def set_audio_adaptor(cls, model, audio_adaptor=None):
        """Load audio adaptor from HF `audio_adaptor.pt` payload (SimpleNamespace tree).

        Mirrors the structure built in loader_fun_mcore_qwen3_5_plus.get_message_for_audio_adaptor.
        """
        if audio_adaptor is None:
            return
        ae = audio_adaptor["weight"] if isinstance(audio_adaptor, dict) else audio_adaptor
        if ae is None:
            return
        if not hasattr(model, "audio_adaptor") or model.audio_adaptor is None:
            return

        aa = model.audio_adaptor

        def _get(obj, key):
            return getattr(obj, key) if hasattr(obj, key) else obj[key]

        def _try_set(dst_tensor, src_tensor):
            if src_tensor is None:
                return
            try:
                cls.set_tensor(dst_tensor, src_tensor)
            except Exception:
                pass

        linear1 = _get(ae, "linear1")
        _try_set(aa.linear1.weight, _get(linear1, "weight"))
        _try_set(aa.linear1.bias, _get(linear1, "bias"))
        linear2 = _get(ae, "linear2")
        _try_set(aa.linear2.weight, _get(linear2, "weight"))
        _try_set(aa.linear2.bias, _get(linear2, "bias"))

        src_blocks = _get(ae, "blocks")
        for idx, dst_blk in enumerate(aa.blocks):
            if idx >= len(src_blocks):
                break
            src_blk = src_blocks[idx]
            attn = _get(src_blk, "self_attn")
            _try_set(dst_blk.self_attn.w_qs.weight, _get(_get(attn, "linear_q"), "weight"))
            _try_set(dst_blk.self_attn.w_qs.bias, _get(_get(attn, "linear_q"), "bias"))
            _try_set(dst_blk.self_attn.w_ks.weight, _get(_get(attn, "linear_k"), "weight"))
            _try_set(dst_blk.self_attn.w_ks.bias, _get(_get(attn, "linear_k"), "bias"))
            _try_set(dst_blk.self_attn.w_vs.weight, _get(_get(attn, "linear_v"), "weight"))
            _try_set(dst_blk.self_attn.w_vs.bias, _get(_get(attn, "linear_v"), "bias"))
            _try_set(dst_blk.self_attn.linear.weight, _get(_get(attn, "linear_out"), "weight"))
            _try_set(dst_blk.self_attn.linear.bias, _get(_get(attn, "linear_out"), "bias"))

            mlp = _get(src_blk, "feed_forward")
            _try_set(dst_blk.feed_forward.w_1.weight, _get(_get(mlp, "w_1"), "weight"))
            _try_set(dst_blk.feed_forward.w_1.bias, _get(_get(mlp, "w_1"), "bias"))
            _try_set(dst_blk.feed_forward.w_2.weight, _get(_get(mlp, "w_2"), "weight"))
            _try_set(dst_blk.feed_forward.w_2.bias, _get(_get(mlp, "w_2"), "bias"))

            n1 = _get(src_blk, "norm1")
            _try_set(dst_blk.norm1.weight, _get(n1, "weight"))
            _try_set(dst_blk.norm1.bias, _get(n1, "bias"))
            n2 = _get(src_blk, "norm2")
            _try_set(dst_blk.norm2.weight, _get(n2, "weight"))
            _try_set(dst_blk.norm2.bias, _get(n2, "bias"))


class MCoreQwen3_5MoETESetter(MCoreSetter):
    """Setter for Qwen3.5-MoE with heterogeneous layers (full + linear attention)."""

    @classmethod
    def set_full_attn_layer(
        cls,
        model,
        layer_idx,
        router_weight=None,
        self_attn_norm_weight=None,
        self_attn_norm_bias=None,
        self_attn_qkv_weight=None,
        self_attn_qkv_bias=None,
        self_attn_proj_weight=None,
        self_attn_proj_bias=None,
        mlp_norm_weight=None,
        mlp_norm_bias=None,
        mlp_fc1_weight=None,
        mlp_fc1_bias=None,
        mlp_fc2_weight=None,
        mlp_fc2_bias=None,
        mlp_shared_exp_fc1_weight=None,
        mlp_shared_exp_fc2_weight=None,
        shared_mlp_gate_weight=None,
        **kwargs,
    ):
        block = cls.get_transformer_block(model)
        l = block.layers[layer_idx]

        # Self attention layernorm: fused into linear_qkv (TE) or separate input_layernorm
        if hasattr(l.self_attention.linear_qkv, 'layer_norm_weight'):
            cls.set_tensor(l.self_attention.linear_qkv.layer_norm_weight, self_attn_norm_weight)
            if self_attn_norm_bias is not None:
                cls.set_tensor(l.self_attention.linear_qkv.layer_norm_bias, self_attn_norm_bias)
        else:
            cls.set_tensor(l.input_layernorm.weight, self_attn_norm_weight)
            if self_attn_norm_bias is not None:
                cls.set_tensor(l.input_layernorm.bias, self_attn_norm_bias)
        cls.set_tensor(l.self_attention.linear_qkv.weight, self_attn_qkv_weight)
        if self_attn_qkv_bias is not None:
            cls.set_tensor(l.self_attention.linear_qkv.bias, self_attn_qkv_bias)
        cls.set_tensor(l.self_attention.linear_proj.weight, self_attn_proj_weight)
        if self_attn_proj_bias is not None:
            cls.set_tensor(l.self_attention.linear_proj.bias, self_attn_proj_bias)

        if 'q layernorm weight' in kwargs:
            cls.set_tensor(l.self_attention.q_layernorm.weight, kwargs['q layernorm weight'])
        if 'k layernorm weight' in kwargs:
            cls.set_tensor(l.self_attention.k_layernorm.weight, kwargs['k layernorm weight'])

        # Dense and MoE checkpoints use different MLP modules.
        if router_weight is None:
            if hasattr(l.mlp.linear_fc1, 'layer_norm_weight'):
                cls.set_tensor(l.mlp.linear_fc1.layer_norm_weight, mlp_norm_weight)
                if mlp_norm_bias is not None:
                    cls.set_tensor(l.mlp.linear_fc1.layer_norm_bias, mlp_norm_bias)
            else:
                cls.set_tensor(l.pre_mlp_layernorm.weight, mlp_norm_weight)
                if model.config.normalization == "LayerNorm" and mlp_norm_bias is not None:
                    cls.set_tensor(l.pre_mlp_layernorm.bias, mlp_norm_bias)
            cls.set_tensor(l.mlp.linear_fc1.weight, mlp_fc1_weight)
            cls.set_tensor(l.mlp.linear_fc2.weight, mlp_fc2_weight)
        else:
            cls.set_tensor(l.pre_mlp_layernorm.weight, mlp_norm_weight)
            if model.config.normalization == "LayerNorm" and mlp_norm_bias is not None:
                cls.set_tensor(l.pre_mlp_layernorm.bias, mlp_norm_bias)
            cls.set_tensor(l.mlp.router.weight, router_weight)
            for expert_idx in range(mlp_fc1_weight.shape[0]):
                cls.set_tensor(l.mlp.experts.local_experts[expert_idx].linear_fc1.weight, mlp_fc1_weight[expert_idx])
                cls.set_tensor(l.mlp.experts.local_experts[expert_idx].linear_fc2.weight, mlp_fc2_weight[expert_idx])

        if mlp_shared_exp_fc1_weight is not None:
            cls.set_tensor(l.mlp.shared_experts.linear_fc1.weight, mlp_shared_exp_fc1_weight)
        if mlp_shared_exp_fc2_weight is not None:
            cls.set_tensor(l.mlp.shared_experts.linear_fc2.weight, mlp_shared_exp_fc2_weight)
        if shared_mlp_gate_weight is not None:
            cls.set_tensor(l.mlp.shared_experts.gate_weight, shared_mlp_gate_weight)

    @classmethod
    def set_linear_attn_layer(
        cls,
        model,
        layer_idx,
        router_weight=None,
        self_attn_norm_weight=None,
        in_proj_qkv_weight=None,
        in_proj_z_weight=None,
        in_proj_b_weight=None,
        in_proj_a_weight=None,
        conv1d_weight=None,
        conv1d_bias=None,
        dt_bias=None,
        A_log=None,
        norm_weight=None,
        out_proj_weight=None,
        mlp_norm_weight=None,
        mlp_norm_bias=None,
        mlp_fc1_weight=None,
        mlp_fc2_weight=None,
        mlp_shared_exp_fc1_weight=None,
        mlp_shared_exp_fc2_weight=None,
        shared_mlp_gate_weight=None,
        **kwargs,
    ):
        block = cls.get_transformer_block(model)
        l = block.layers[layer_idx]

        # Input layernorm (separate, not fused)
        cls.set_tensor(l.input_layernorm.weight, self_attn_norm_weight)

        # GatedDeltaNet projections
        attn = l.self_attention
        cls.set_tensor(attn.in_proj_qkv.weight, in_proj_qkv_weight)
        cls.set_tensor(attn.in_proj_z.weight, in_proj_z_weight)
        cls.set_tensor(attn.in_proj_b.weight, in_proj_b_weight)
        cls.set_tensor(attn.in_proj_a.weight, in_proj_a_weight)
        cls.set_tensor(attn.conv1d.weight, conv1d_weight)
        if conv1d_bias is not None:
            cls.set_tensor(attn.conv1d.bias, conv1d_bias)
        cls.set_tensor(attn.dt_bias, dt_bias)
        cls.set_tensor(attn.A_log, A_log)
        cls.set_tensor(attn.norm.weight, norm_weight)
        cls.set_tensor(attn.out_proj.weight, out_proj_weight)

        # Dense and MoE checkpoints use different MLP modules.
        if router_weight is None:
            if hasattr(l.mlp.linear_fc1, 'layer_norm_weight'):
                cls.set_tensor(l.mlp.linear_fc1.layer_norm_weight, mlp_norm_weight)
                if mlp_norm_bias is not None:
                    cls.set_tensor(l.mlp.linear_fc1.layer_norm_bias, mlp_norm_bias)
            else:
                cls.set_tensor(l.pre_mlp_layernorm.weight, mlp_norm_weight)
                if model.config.normalization == "LayerNorm" and mlp_norm_bias is not None:
                    cls.set_tensor(l.pre_mlp_layernorm.bias, mlp_norm_bias)
            cls.set_tensor(l.mlp.linear_fc1.weight, mlp_fc1_weight)
            cls.set_tensor(l.mlp.linear_fc2.weight, mlp_fc2_weight)
        else:
            cls.set_tensor(l.pre_mlp_layernorm.weight, mlp_norm_weight)
            if model.config.normalization == "LayerNorm" and mlp_norm_bias is not None:
                cls.set_tensor(l.pre_mlp_layernorm.bias, mlp_norm_bias)
            cls.set_tensor(l.mlp.router.weight, router_weight)
            for expert_idx in range(mlp_fc1_weight.shape[0]):
                cls.set_tensor(l.mlp.experts.local_experts[expert_idx].linear_fc1.weight, mlp_fc1_weight[expert_idx])
                cls.set_tensor(l.mlp.experts.local_experts[expert_idx].linear_fc2.weight, mlp_fc2_weight[expert_idx])

        if mlp_shared_exp_fc1_weight is not None:
            cls.set_tensor(l.mlp.shared_experts.linear_fc1.weight, mlp_shared_exp_fc1_weight)
        if mlp_shared_exp_fc2_weight is not None:
            cls.set_tensor(l.mlp.shared_experts.linear_fc2.weight, mlp_shared_exp_fc2_weight)
        if shared_mlp_gate_weight is not None:
            cls.set_tensor(l.mlp.shared_experts.gate_weight, shared_mlp_gate_weight)


def get_model_setter(model_type, layer_types=None):
    setter = MCoreQwen3_5MoETESetter
    setter.transformer_block_key = get_mcore_transformer_block_key(model_type)
    return setter


def add_arguments(parser):
    group = parser.add_argument_group(title='M-Core Qwen3.5 saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--saver-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--target-expert-parallel-size', type=int, default=1,
                       help='Target expert model parallel size, default to 1')
    group.add_argument('--saver-moe-grouped-gemm', action='store_true',
                       help='Build and save the target MoE model with grouped GEMM.')
    parser.add_argument('--target-ckpt-format', default='torch',
                       choices=['torch', 'torch_dist', 'zarr'],
                       help='Checkpoint format to use.')
    parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.')


def save_checkpoint(queue, args):

    # te_version = packaging.version.Version(version("transformer-engine"))
    # assert te_version >= packaging.version.Version("0.12.0"), \
    #     "transformer engine version: %s (>=0.12.0 required)." % te_version
    te_version = Version(version("transformer-engine"))
    assert te_version >= Version("0.12.0"), \
        f"transformer engine version: {te_version} (>=0.12.0 required)."

    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import (parse_args, validate_args)
        from megatron.training.checkpointing import save_checkpoint
        from megatron.training.global_vars import set_global_variables, get_args
        from megatron.core.parallel_state import initialize_model_parallel
        from megatron.core.enums import ModelType
        from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        from megatron.legacy import fused_kernels
        from megatron.core import mpu
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    md = queue_get()

    if args.target_tensor_parallel_size is None:
        if hasattr(md, 'previous_tensor_parallel_size'):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            print("loader did not provide a tensor parallel size and --target-tensor-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_tensor_parallel_size = 1
    if args.target_pipeline_parallel_size is None:
        if hasattr(md, 'previous_pipeline_parallel_size'):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            print("loader did not provide a pipeline parallel size and --target-pipeline-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_pipeline_parallel_size = 1

    if args.target_ckpt_format == "torch_dist":
        assert args.target_tensor_parallel_size == 1, "Please setting --target-tensor-parallel-size to 1 to use dist ckpt"
        assert args.target_pipeline_parallel_size == 1, "Please setting --target-pipeline-parallel-size to 1 to use dist ckpt"
        assert args.target_expert_parallel_size == 1, "Please setting --target-expert-parallel-size to 1 to use dist ckpt"

    if args.target_tensor_parallel_size is not None and args.target_pipeline_parallel_size is not None:
        if args.target_expert_parallel_size is not None:
            os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size * args.target_expert_parallel_size}'
        else:
            os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}'

    # Extract Qwen3.5-specific metadata from loader (try md first, then md.checkpoint_args)
    def _md_get(attr, default=None):
        val = getattr(md, attr, None)
        if val is not None:
            return val
        if hasattr(md, 'checkpoint_args'):
            val = getattr(md.checkpoint_args, attr, None)
            if val is not None:
                return val
        return default

    layer_types = _md_get('layer_types', None)
    attn_output_gate = _md_get('attn_output_gate', False)
    full_attention_interval = _md_get('full_attention_interval', 0)
    linear_attention_conv_kernel_dim = _md_get('linear_attention_conv_kernel_dim', 4)
    linear_attention_key_head_dim = _md_get('linear_attention_key_head_dim', 128)
    linear_attention_value_head_dim = _md_get('linear_attention_value_head_dim', 128)
    linear_attention_num_key_heads = _md_get('linear_attention_num_key_heads', 16)
    linear_attention_num_value_heads = _md_get('linear_attention_num_value_heads', 64)
    mrope_interleaved = _md_get('mrope_interleaved', False)
    mrope_section = _md_get('mrope_section', None)

    sys.argv = ['script.py',
                '--num-layers', str(md.num_layers),
                '--hidden-size', str(md.hidden_size),
                '--seq-length', str(md.seq_length),
                '--num-attention-heads', str(md.num_attention_heads),
                '--ffn-hidden-size', str(md.ffn_hidden_size),
                '--max-position-embeddings', str(md.max_position_embeddings),
                '--position-embedding-type', str(md.position_embedding_type),
                '--tokenizer-type', str(md.tokenizer_type),
                '--tensor-model-parallel-size', str(args.target_tensor_parallel_size),
                '--pipeline-model-parallel-size', str(args.target_pipeline_parallel_size),
                '--expert-model-parallel-size', str(args.target_expert_parallel_size),
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--save-interval', '1',
                '--save', args.save_dir,
                '--ckpt-format', str(args.target_ckpt_format),
                '--audio-tokenizer-code-book-size', '59049',
                ]
    if getattr(md, "num_experts", None):
        sys.argv.extend(['--num-experts', str(md.num_experts)])
    if channels_model:
        sys.argv.append('--audio-is-dual-channel')
    if args.saver_moe_grouped_gemm:
        sys.argv.append('--moe-grouped-gemm')
    if args.qwen3_audio_encoder:
        if args.sensevoicesmall_audio_encoder:
            sys.argv.extend(['--audio-encoder-type', 'sensevoicesmall', "--audio-encoder-dim", "512"])
        else:
            sys.argv.extend(['--audio-encoder-type', 'qwen3-2b', "--audio-encoder-dim", "2048"])

    # Qwen3.5-specific args
    if full_attention_interval > 0:
        sys.argv.extend(['--full-attention-interval', str(full_attention_interval)])
    if attn_output_gate:
        sys.argv.append('--attn-output-gate')
    sys.argv.extend([
        '--linear-attention-conv-kernel-dim', str(linear_attention_conv_kernel_dim),
        '--linear-attention-key-head-dim', str(linear_attention_key_head_dim),
        '--linear-attention-value-head-dim', str(linear_attention_value_head_dim),
        '--linear-attention-num-key-heads', str(linear_attention_num_key_heads),
        '--linear-attention-num-value-heads', str(linear_attention_num_value_heads),
    ])
    if mrope_interleaved:
        sys.argv.append('--mrope-interleaved')
    if mrope_section is not None:
        sys.argv.extend(['--mrope-section'] + [str(s) for s in mrope_section])

    # Zero-centered gamma RMSNorm
    if _md_get('apply_layernorm_1p', False) or _md_get('layernorm_zero_centered_gamma', False):
        sys.argv.append('--apply-layernorm-1p')

    if md.make_vocab_size_divisible_by is not None:
        sys.argv.extend(['--make-vocab-size-divisible-by', str(md.make_vocab_size_divisible_by)])
    elif args.target_tensor_parallel_size > 1 and args.target_ckpt_format == 'torch':
        sys.argv.extend(['--make-vocab-size-divisible-by', str(args.make_vocab_size_divisible_by)])
    if md.params_dtype == torch.float16:
        sys.argv.append('--fp16')
    elif md.params_dtype == torch.bfloat16:
        sys.argv.append('--bf16')

    if md.output_layer:
        sys.argv.append('--untie-embeddings-and-output-weights')
    if not md.linear_bias:
        sys.argv.append('--disable-bias-linear')

    if md.model_type == 'BERT' and not md.bert_binary_head:
        sys.argv.append('--bert-no-binary-head')

    margs = parse_args()
    if hasattr(md, 'checkpoint_args'):
        args_to_keep = ['tensor_model_parallel_size', 'pipeline_model_parallel_size', 'expert_model_parallel_size',
                        'world_size', 'params_dtype',
                        'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
                        'masked_softmax_fusion', 'bias_gelu_fusion', 'bias_dropout_fusion',
                        'sequence_parallel', 'async_tensor_model_parallel_allreduce',
                        'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng',
                        'vocab_file', 'tokenizer_model',
                        'save_interval', 'save',
                        'perform_initialization', 'use_cpu_initialization',
                        'recompute_granularity', 'recompute_num_layers', 'recompute_method',
                        'encoder_num_layers', 'encoder_seq_length',
                        'distribute_saved_activations',
                        'train_iters', 'lr_decay_iters', 'lr_warmup_iters', 'lr_warmup_fraction',
                        'start_weight_decay', 'end_weight_decay',
                        'ckpt_format', 'ffn_hidden_size', 'num_attention_heads', 'num_layers',
                        # Qwen3.5-specific: keep these from being overwritten
                        'full_attention_interval', 'attn_output_gate',
                        'linear_attention_conv_kernel_dim', 'linear_attention_key_head_dim',
                        'linear_attention_value_head_dim', 'linear_attention_num_key_heads',
                        'linear_attention_num_value_heads',
                        'mrope_interleaved', 'mrope_section',
                        'layer_types', 'layernorm_zero_centered_gamma',
        ]
        if args.target_tensor_parallel_size > 1 and args.target_ckpt_format == 'torch':
            args_to_keep.append('make_vocab_size_divisible_by')
        for arg, value in vars(md.checkpoint_args).items():
            if arg in args_to_keep:
                continue
            if not hasattr(margs, arg):
                print(f"Checkpoint had argument {arg} but new arguments does not have this.")
                continue
            if getattr(margs, arg) != value:
                print(f"Overwriting default {arg} value {getattr(margs, arg)} with value from checkpoint {value}.")
                setattr(margs, arg, value)

    margs.sequence_parallel = md.checkpoint_args.sequence_parallel
    margs.apply_query_key_layer_scaling = md.checkpoint_args.apply_query_key_layer_scaling

    if margs.num_experts is not None and args.target_tensor_parallel_size is not None:
        if margs.num_experts > 1 and args.target_tensor_parallel_size > 1:
            margs.sequence_parallel = True

    validate_args(margs)

    margs.use_legacy_models = False
    margs.blendable_index_path = None
    margs.data_path = []
    margs.load = None
    margs.save = args.save_dir
    margs.tensorboard_dir = None
    margs.tokenizer_model = None
    margs.transformer_impl = args.saver_transformer_impl
    margs.use_flash_cross_entropy = False

    set_global_variables(margs, build_tokenizer=False)
    margs = get_args()

    if hasattr(md, 'consumed_train_samples'):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        print(f"Setting consumed_train_samples to {margs.consumed_train_samples}"
              f" and consumed_valid_samples to {margs.consumed_valid_samples}")
    else:
        print("consumed_train_samples not provided.")

    if args.qwen3_audio_encoder:
        args_audio_encoder = types.SimpleNamespace()
        args_audio_encoder.tensor_model_parallel_size = margs.tensor_model_parallel_size
        args_audio_encoder.pipeline_model_parallel_size = margs.pipeline_model_parallel_size
        args_audio_encoder.virtual_pipeline_model_parallel_size = margs.virtual_pipeline_model_parallel_size
        args_audio_encoder.expert_model_parallel_size = margs.expert_model_parallel_size
        args_audio_encoder.target_pipeline_parallel_size = args.target_pipeline_parallel_size
        args_audio_encoder.target_tensor_parallel_size = args.target_tensor_parallel_size
        args_audio_encoder.target_expert_parallel_size = args.target_expert_parallel_size
        args_audio_encoder.num_experts = 0
        args_audio_encoder.untie_embeddings_and_output_weights = False
        args_audio_encoder.qwen3_audio_encoder = args.qwen3_audio_encoder
    else:
        args_audio_encoder = types.SimpleNamespace()

    # Checkpoint conversion only materializes parameters and never runs a
    # backward pass.  Keeping this enabled unnecessarily requires Apex's
    # fused_weight_gradient_mlp_cuda extension while constructing MCore
    # ColumnParallelLinear modules.
    margs.gradient_accumulation_fusion = False

    if md.model_type == 'GPT':
        from pretrain_fun_base import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif md.model_type == 'Reward':
        from megatron.core.models.gpt.gpt_reward_model import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif md.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')
    print(f"saver's margs {margs}")

    if args.target_ckpt_format == "torch_dist":
        torch.distributed.init_process_group(
            backend=margs.distributed_backend,
            world_size=margs.world_size,
            rank=margs.rank,
        )
        initialize_model_parallel()
        print(f"real initializing distributed")
    else:
        print(f"fake initializing distributed")
        mpu.set_tensor_model_parallel_world_size(args.target_tensor_parallel_size)
        mpu.set_pipeline_model_parallel_world_size(args.target_pipeline_parallel_size)
        mpu.set_expert_model_parallel_world_size(args.target_expert_parallel_size)
        mpu.set_tensor_model_parallel_rank(0)
        mpu.set_pipeline_model_parallel_rank(0)
        mpu.set_expert_model_parallel_rank(0)
    fused_kernels.load(margs)

    # Embeddings
    embeddings_msg = queue_get("embeddings")

    pos_embed = None
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = embeddings_msg.pop("position embeddings")
    orig_word_embed = embeddings_msg.pop("word embeddings")
    check_message(embeddings_msg)

    def pad_weight(orig_word_embed, true_vocab_size):
        if true_vocab_size is not None:
            orig_vocab_size = orig_word_embed.shape[0]
            margs.padded_vocab_size = _vocab_size_with_padding(true_vocab_size, margs)
            if orig_vocab_size > margs.padded_vocab_size:
                full_word_embed = orig_word_embed[0:margs.padded_vocab_size, :]
            elif orig_vocab_size < margs.padded_vocab_size:
                padding_size = margs.padded_vocab_size - orig_vocab_size
                full_word_embed = torch.cat((
                    orig_word_embed,
                    orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))
            else:
                full_word_embed = orig_word_embed
        else:
            print("Original vocab size not specified, leaving embedding table as-is. "
                  "If you've changed the tensor parallel size this could cause problems.")
            margs.padded_vocab_size = orig_word_embed.shape[0]
            full_word_embed = orig_word_embed
        return full_word_embed

    full_word_embed = pad_weight(orig_word_embed, md.true_vocab_size)
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_parallel_size, dim=0)

    setter = get_model_setter(md.model_type, layer_types)

    models = [[[None for _ in range(args.target_tensor_parallel_size)]
               for _ in range(args.target_expert_parallel_size)]
              for _ in range(args.target_pipeline_parallel_size)]

    def get_local_model(pp_rank, ep_rank, tp_rank, is_audio_encoder=False):
        if models[pp_rank][ep_rank][tp_rank] is None:
            pre_process = True if pp_rank == 0 else False
            post_process = True if pp_rank == args.target_pipeline_parallel_size - 1 else False
            models[pp_rank][ep_rank][tp_rank] = model_provider(pre_process, post_process).to(md.params_dtype)
        if is_audio_encoder:
            return models[pp_rank][ep_rank][tp_rank].audio_encoder
        else:
            return models[pp_rank][ep_rank][tp_rank]

    # Set embeddings
    for ep_rank in range(args.target_expert_parallel_size):
        for tp_rank in range(args.target_tensor_parallel_size):
            model = get_local_model(0, ep_rank, tp_rank)
            if pos_embed is None:
                assert not setter.has_position_embeddings(model)
            setter.set_embeddings(
                model,
                word=out_word_embed[tp_rank],
                pos=pos_embed,
            )

    def chunk_weight(weight, parallel_mode, tp_size=1, ep_size=1):
        assert parallel_mode in ["row", "column"]
        if weight.dim() == 3:
            num_experts, out_features, in_features = weight.shape
            if parallel_mode == "column":
                weight = weight.reshape(ep_size, num_experts // ep_size, tp_size, out_features // tp_size, in_features)
                weight = weight.permute(0, 2, 1, 3, 4)
            else:
                weight = weight.reshape(ep_size, num_experts // ep_size, out_features, tp_size, in_features // tp_size)
                weight = weight.permute(0, 3, 1, 2, 4)
            return weight
        else:
            out_features, in_features = weight.shape
            if parallel_mode == "column":
                weight = weight.reshape(tp_size, out_features // tp_size, in_features)
            else:
                weight = weight.reshape(out_features, tp_size, in_features // tp_size).permute(1, 0, 2)
            return weight

    def chunk_bias(bias, parallel_mode, tp_size=1, ep_size=1):
        assert parallel_mode in ["row", "column"]
        if bias.dim() == 2:
            num_experts, hidden_size = bias.shape
            if parallel_mode == 'column':
                bias = bias.reshape(ep_size, num_experts // ep_size, tp_size, hidden_size // tp_size)
                bias = bias.permute(0, 2, 1, 3)
            else:
                bias = bias.reshape(ep_size, num_experts // ep_size, hidden_size)
            return bias
        else:
            hidden_size = bias.shape
            if parallel_mode == "column":
                bias = bias.reshape(tp_size, hidden_size[0] // tp_size)
            return bias

    def chunk_conv1d_weight(weight, tp_size):
        """Chunk depthwise Conv1d weight along channel dim for TP."""
        # weight shape: [channels, 1, kernel_size] for depthwise conv
        channels = weight.shape[0]
        return weight.reshape(tp_size, channels // tp_size, *weight.shape[1:])

    def process_moe_mlp_from_msg(msg, margs, tp_size, ep_size):
        """Extract and chunk dense or MoE MLP weights from a queue message."""
        result = {}

        if not margs.num_experts:
            if md.swiglu:
                mlp_l0_weight_W = chunk_weight(msg.pop("mlp l0 weight W"), "column", tp_size)
                mlp_l0_weight_V = chunk_weight(msg.pop("mlp l0 weight V"), "column", tp_size)
                mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
            else:
                mlp_l0_weight = chunk_weight(msg.pop("mlp l0 weight"), "column", tp_size)
            mlp_l1_weight = chunk_weight(msg.pop("mlp l1 weight"), "row", tp_size)
            result['router'] = None
            # Preserve the [EP, TP, ...] indexing contract used by the caller.
            result['mlp_l0'] = mlp_l0_weight.unsqueeze(0)
            result['mlp_l1'] = mlp_l1_weight.unsqueeze(0)
            return result

        if margs.moe_shared_expert_intermediate_size:
            if md.swiglu:
                shared_mlp_l0_weight_W = chunk_weight(msg.pop("shared mlp l0 weight W"), "column", tp_size)
                shared_mlp_l0_weight_V = chunk_weight(msg.pop("shared mlp l0 weight V"), "column", tp_size)
                shared_mlp_l0_weight = torch.cat((shared_mlp_l0_weight_W, shared_mlp_l0_weight_V), dim=-2)
            else:
                shared_mlp_l0_weight = chunk_weight(msg.pop("shared mlp l0 weight"), "column", tp_size)
            shared_mlp_l1_weight = chunk_weight(msg.pop("shared mlp l1 weight"), "row", tp_size)
            result['shared_mlp_l0'] = shared_mlp_l0_weight
            result['shared_mlp_l1'] = shared_mlp_l1_weight
            if md.moe_shared_experts_gate:
                result['shared_experts_gate'] = msg.pop("shared mlp gate weight")

        result['router'] = msg.pop("router weight")

        if md.swiglu:
            mlp_l0_weight_W = chunk_weight(msg.pop("mlp l0 weight W"), "column", tp_size, ep_size)
            mlp_l0_weight_V = chunk_weight(msg.pop("mlp l0 weight V"), "column", tp_size, ep_size)
            mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
        else:
            mlp_l0_weight = chunk_weight(msg.pop("mlp l0 weight"), "column", tp_size, ep_size)

        mlp_l1_weight = chunk_weight(msg.pop("mlp l1 weight"), "row", tp_size, ep_size)

        result['mlp_l0'] = mlp_l0_weight
        result['mlp_l1'] = mlp_l1_weight
        return result

    def process_transformer_layer(args, margs, is_audio_encoder=False):
        total_layer_num = 0
        for pp_rank in range(args.target_pipeline_parallel_size):
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            get_local_model(pp_rank, 0, 0, is_audio_encoder)

            model = get_local_model(pp_rank, 0, 0, is_audio_encoder)
            if is_audio_encoder:
                setter.transformer_block_key = 'encoder'
            for layer_id in range(len(setter.get_transformer_block(model).layers)):
                if not is_audio_encoder:
                    msg = queue_get(f"transformer layer {total_layer_num}")
                else:
                    msg = queue_get(f"audio encoder transformer layer {total_layer_num}")

                msg_layer_type = msg.pop("layer_type", "full_attention")

                # Determine the actual layer type
                if layer_types is not None and total_layer_num < len(layer_types):
                    actual_layer_type = layer_types[total_layer_num]
                else:
                    actual_layer_type = msg_layer_type

                input_norm_weight = msg.pop("input norm weight")
                post_norm_weight = msg.pop("post norm weight")
                if md.norm_has_bias:
                    input_norm_bias = msg.pop("input norm bias")
                    post_norm_bias = msg.pop("post norm bias")

                tp_size = args.target_tensor_parallel_size
                ep_size = args.target_expert_parallel_size

                if actual_layer_type == 'linear_attention':
                    # GatedDeltaNet linear attention layer
                    # TP-parallel projections
                    in_proj_qkv_weight = chunk_weight(msg.pop("linear_attn in_proj_qkv weight"), "column", tp_size)
                    in_proj_z_weight = chunk_weight(msg.pop("linear_attn in_proj_z weight"), "column", tp_size)
                    out_proj_weight = chunk_weight(msg.pop("linear_attn out_proj weight"), "row", tp_size)

                    # NOT TP-split (replicated across ranks)
                    in_proj_b_weight = msg.pop("linear_attn in_proj_b weight")
                    in_proj_a_weight = msg.pop("linear_attn in_proj_a weight")
                    dt_bias_val = msg.pop("linear_attn dt_bias")
                    A_log_val = msg.pop("linear_attn A_log")
                    norm_weight_val = msg.pop("linear_attn norm weight")

                    # Conv1d: channels are TP-split
                    conv1d_weight_full = msg.pop("linear_attn conv1d weight")
                    conv1d_weight = chunk_conv1d_weight(conv1d_weight_full, tp_size)
                    conv1d_bias_val = msg.pop("linear_attn conv1d bias", None)
                    if conv1d_bias_val is not None:
                        conv1d_bias_chunks = conv1d_bias_val.reshape(tp_size, -1)
                    else:
                        conv1d_bias_chunks = None

                    # MoE MLP
                    mlp_data = process_moe_mlp_from_msg(msg, margs, tp_size, ep_size)

                    for ep_rank in range(ep_size):
                        for tp_rank in range(tp_size):
                            params_dict = {
                                "self_attn_norm_weight": input_norm_weight,
                                "in_proj_qkv_weight": in_proj_qkv_weight[tp_rank],
                                "in_proj_z_weight": in_proj_z_weight[tp_rank],
                                "in_proj_b_weight": in_proj_b_weight,
                                "in_proj_a_weight": in_proj_a_weight,
                                "conv1d_weight": conv1d_weight[tp_rank],
                                "conv1d_bias": conv1d_bias_chunks[tp_rank] if conv1d_bias_chunks is not None else None,
                                "dt_bias": dt_bias_val,
                                "A_log": A_log_val,
                                "norm_weight": norm_weight_val,
                                "out_proj_weight": out_proj_weight[tp_rank],
                                "mlp_norm_weight": post_norm_weight,
                                "mlp_norm_bias": post_norm_bias if md.norm_has_bias else None,
                                "router_weight": mlp_data['router'],
                                "mlp_fc1_weight": mlp_data['mlp_l0'][ep_rank][tp_rank],
                                "mlp_fc2_weight": mlp_data['mlp_l1'][ep_rank][tp_rank],
                            }
                            if 'shared_mlp_l0' in mlp_data:
                                params_dict["mlp_shared_exp_fc1_weight"] = mlp_data['shared_mlp_l0'][tp_rank]
                                params_dict["mlp_shared_exp_fc2_weight"] = mlp_data['shared_mlp_l1'][tp_rank]
                            if 'shared_experts_gate' in mlp_data:
                                params_dict["shared_mlp_gate_weight"] = mlp_data['shared_experts_gate']

                            model = get_local_model(pp_rank, ep_rank, tp_rank, is_audio_encoder)
                            setter.set_linear_attn_layer(model, layer_id, **params_dict)

                else:
                    # Full attention layer (standard or with output gating)
                    qkv_weight = chunk_weight(msg.pop("qkv weight"), "column", tp_size)
                    dense_weight = chunk_weight(msg.pop("dense weight"), "row", tp_size)

                    if md.add_qkv_bias:
                        qkv_bias = chunk_bias(msg.pop("qkv bias"), 'column', tp_size)

                    if md.qk_layernorm:
                        q_layernorm = msg.pop("q layernorm weight")
                        k_layernorm = msg.pop("k layernorm weight")

                    # MoE MLP
                    mlp_data = process_moe_mlp_from_msg(msg, margs, tp_size, ep_size)

                    for ep_rank in range(ep_size):
                        for tp_rank in range(tp_size):
                            params_dict = {
                                "self_attn_norm_weight": input_norm_weight,
                                "self_attn_qkv_weight": qkv_weight[tp_rank],
                                "self_attn_proj_weight": dense_weight[tp_rank],
                                "mlp_norm_weight": post_norm_weight,
                                "self_attn_norm_bias": input_norm_bias if md.norm_has_bias else None,
                                "mlp_norm_bias": post_norm_bias if md.norm_has_bias else None,
                                "router_weight": mlp_data['router'],
                                "mlp_fc1_weight": mlp_data['mlp_l0'][ep_rank][tp_rank],
                                "mlp_fc2_weight": mlp_data['mlp_l1'][ep_rank][tp_rank],
                            }
                            if md.qk_layernorm:
                                params_dict.update({
                                    "q layernorm weight": q_layernorm,
                                    "k layernorm weight": k_layernorm,
                                })
                            if md.add_qkv_bias:
                                params_dict["self_attn_qkv_bias"] = qkv_bias[tp_rank]
                            if 'shared_mlp_l0' in mlp_data:
                                params_dict["mlp_shared_exp_fc1_weight"] = mlp_data['shared_mlp_l0'][tp_rank]
                                params_dict["mlp_shared_exp_fc2_weight"] = mlp_data['shared_mlp_l1'][tp_rank]
                            if 'shared_experts_gate' in mlp_data:
                                params_dict["shared_mlp_gate_weight"] = mlp_data['shared_experts_gate']

                            model = get_local_model(pp_rank, ep_rank, tp_rank, is_audio_encoder)
                            setter.set_full_attn_layer(model, layer_id, **params_dict)

                total_layer_num = total_layer_num + 1
                check_message(msg)

            if pp_rank == 0 and not args.qwen3_audio_encoder:
                pp_local_models = [
                    get_local_model(pp_rank, ep_rank, tp_rank)
                    for ep_rank in range(args.target_expert_parallel_size)
                    for tp_rank in range(args.target_tensor_parallel_size)
                ]
                audio_encoder_msg = queue_get("audio encoder")
                if not random_init_audio_encoder:
                    for model in pp_local_models:
                        if args.sensevoicesmall_audio_encoder:
                            setter.set_sensevoicesmall_audio_encoder(model, audio_encoder_msg)
                        else:
                            setter.set_audio_encoder(model, audio_encoder_msg)

            if pp_rank == 0 and not is_audio_encoder:
                audio_adaptor_msg = queue_get("audio adaptor")
                aa_weight = audio_adaptor_msg.pop("weight", None)
                check_message(audio_adaptor_msg)
                if aa_weight is not None and not random_init_audio_adaptor:
                    pp_local_aa = [
                        get_local_model(0, ep_rank, tp_rank, is_audio_encoder=False)
                        for ep_rank in range(args.target_expert_parallel_size)
                        for tp_rank in range(args.target_tensor_parallel_size)
                    ]
                    for model in pp_local_aa:
                        setter.set_audio_adaptor(model, {"weight": aa_weight})

            if pp_rank == args.target_pipeline_parallel_size - 1:
                msg = queue_get("final norm") if not is_audio_encoder else queue_get("audio encoder final norm")
                final_norm_weight = msg.pop("weight")
                if md.norm_has_bias:
                    final_norm_bias = msg.pop("bias")
                pp_local_models = [get_local_model(pp_rank, ep_rank, tp_rank, is_audio_encoder)
                                   for ep_rank in range(args.target_expert_parallel_size)
                                   for tp_rank in range(args.target_tensor_parallel_size)]
                for eptp_rank, model in enumerate(pp_local_models):
                    tp_rank = eptp_rank % args.target_tensor_parallel_size
                    setter.set_final_norm(
                        model,
                        weight=final_norm_weight,
                        bias=final_norm_bias if md.norm_has_bias else None,
                    )
                    if pp_rank != 0 and not md.output_layer:
                        setter.set_output_word_embeddings(
                            model,
                            emb=out_word_embed[tp_rank],
                        )
                del final_norm_weight
                if md.norm_has_bias:
                    del final_norm_bias
                check_message(msg)

                if md.output_layer:
                    msg = queue_get("output layer")
                    if not hasattr(pp_local_models[0], 'output_layer'):
                        print("ERROR: got an output layer, but model does not have one")
                        exit(1)
                    output_layer_weight = pad_weight(msg.pop("weight"), md.true_vocab_size)
                    output_layer_weight = torch.chunk(output_layer_weight, args.target_tensor_parallel_size, dim=0)
                    for eptp_rank, model in enumerate(pp_local_models):
                        tp_rank = eptp_rank % args.target_tensor_parallel_size
                        setter.set_output_layer(model, output_layer_weight[tp_rank])
                    check_message(msg)

                if is_audio_encoder:
                    msg = queue_get("audio encoder conv1 weights")
                    audio_encoder_conv1_weight = msg.pop("weight")
                    audio_encoder_conv1_bias = msg.pop("bias")
                    msg = queue_get("audio encoder conv2 weights")
                    audio_encoder_conv2_weight = msg.pop("weight")
                    audio_encoder_conv2_bias = msg.pop("bias")
                    for model in pp_local_models:
                        setter.set_tensor(model.conv1.weight, audio_encoder_conv1_weight)
                        setter.set_tensor(model.conv1.bias, audio_encoder_conv1_bias)
                        setter.set_tensor(model.conv2.weight, audio_encoder_conv2_weight)
                        setter.set_tensor(model.conv2.bias, audio_encoder_conv2_bias)

                if (args.qwen3_audio_encoder and is_audio_encoder) or (not args.qwen3_audio_encoder and not is_audio_encoder):
                    msg = queue_get()
                else:
                    msg = {"name": "convert audio encoder next step"}
                if msg != "done" and msg["name"] == "pooler":
                    if not hasattr(models[pp_rank][0][0], 'pooler'):
                        print("ERROR: got a pooler, but model does not have one")
                        exit(1)
                    print("received pooler")
                    pooler_weight = msg.pop("weight")
                    pooler_bias = msg.pop("bias")
                    for model in pp_local_models:
                        setter.set_pooler(
                            model=model,
                            weight=pooler_weight,
                            bias=pooler_bias,
                        )
                    del pooler_weight
                    del pooler_bias
                    check_message(msg)
                    msg = queue_get()

                if msg != "done" and msg["name"] == "binary head":
                    if not hasattr(models[pp_rank][0][0], 'binary_head'):
                        print("ERROR: got a binary head, but model does not have one")
                        exit(1)
                    print("received binary head")
                    binary_head_weight = msg.pop("weight")
                    binary_head_bias = msg.pop("bias")
                    for model in pp_local_models:
                        setter.set_tensor(model.binary_head.weight, binary_head_weight)
                        if binary_head_bias is not None:
                            setter.set_tensor(model.binary_head.bias, binary_head_bias)
                    check_message(msg)
                    msg = queue_get()

                if args.qwen3_audio_encoder and is_audio_encoder:
                    if msg != "done":
                        print("ERROR: got some more data but was expecting to be done")
                elif not args.qwen3_audio_encoder and not is_audio_encoder:
                    if msg != "done":
                        print("ERROR: got some more data but was expecting to be done")

            if (args.qwen3_audio_encoder and is_audio_encoder) or (not args.qwen3_audio_encoder and not is_audio_encoder):
                for ep_rank in range(args.target_expert_parallel_size):
                    for tp_rank in range(args.target_tensor_parallel_size):
                        save_checkpoint(md.iteration, [get_local_model(pp_rank, ep_rank, tp_rank, is_audio_encoder=False)], None, None,
                                        num_floating_point_operations_so_far=0,
                                        pipeline_rank=pp_rank, pipeline_parallel=args.target_pipeline_parallel_size > 1,
                                        expert_rank=ep_rank, expert_parallel=args.target_expert_parallel_size > 1,
                                        tensor_rank=tp_rank)
                        models[pp_rank][ep_rank][tp_rank] = None

    process_transformer_layer(args, margs, is_audio_encoder=False)
    if args.qwen3_audio_encoder:
        # Audio encoder uses dense attention, reset setter
        from utils import get_mcore_transformer_block_key
        setter.transformer_block_key = get_mcore_transformer_block_key(md.model_type)
        margs.moe_shared_expert_intermediate_size = None
        margs.num_experts = 0
        md.output_layer = False
        process_transformer_layer(args_audio_encoder, margs, is_audio_encoder=True)

    print("Done!")
