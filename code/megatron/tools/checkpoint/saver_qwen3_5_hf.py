# Copyright (c) 2025. Qwen3.5-MoE HF checkpoint saver for Megatron-LM.
"""Save Megatron Qwen3.5-MoE checkpoints to HuggingFace format.

Handles:
- Heterogeneous layers (linear_attention + full_attention)
- Attention output gating (doubled Q projection -> split back)
- GatedDeltaNet linear attention weights
- Shared expert with sigmoid gate
- Zero-centered gamma RMSNorm
"""

import argparse
import collections
import json
import pprint
import re
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file as safe_save_file

try:
    from transformers.pytorch_utils import id_tensor_storage
except ImportError:
    id_tensor_storage = id

LAYER_RE = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z_]+)")
# For params without .weight/.bias suffix: layers.0.self_attention.dt_bias
LAYER_BARE_PARAM_RE = re.compile(r"layers\.(\d+)\.self_attention\.(dt_bias|A_log)$")

try:
    from transformers.modeling_utils import shard_checkpoint

    def save_state_dict(path, state_dict, max_shard_size):
        shards_dict, shards_index = shard_checkpoint(
            state_dict, max_shard_size, weights_name='model.safetensors')
        if shards_index:
            safe_index_filename = path / "model.safetensors.index.json"
            with open(safe_index_filename, 'w', encoding='utf-8') as f:
                f.write(json.dumps(shards_index, indent=2, sort_keys=True) + "\n")
        for shard_file, shard in shards_dict.items():
            shard_filename = path / shard_file
            print(f'Saving to shard checkpoint {shard_filename} ...')
            safe_save_file(shard, shard_filename, metadata={"format": "pt"})

except ImportError:
    from huggingface_hub import split_torch_state_dict_into_shards

    def save_state_dict(path, state_dict, max_shard_size):
        # 标准格式: model.safetensors-00001-of-00014.safetensors
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern="model.safetensors{suffix}.safetensors", max_shard_size=max_shard_size)
        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            safe_save_file(shard, path / filename, metadata={"format": "pt"})
        if state_dict_split.is_sharded:
            shards_index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            safe_index_filename = path / "model.safetensors.index.json"
            with open(safe_index_filename, 'w', encoding='utf-8') as f:
                f.write(json.dumps(shards_index, indent=2, sort_keys=True) + "\n")


KNOWN_TIED_WEIGHT_KEYS = ["lm_head.weight"]


def _parse_shard_size(shard_str):
    """Parse shard size string like '5GB', '2GB' to bytes.
    必须按单位长度从长到短匹配，否则 '10GB' 会错误匹配 'B' 导致解析失败。
    """
    if shard_str is None:
        return None
    shard_str = str(shard_str).strip().upper()
    if not shard_str:
        return None
    # 按单位长度降序，确保 "GB" 在 "B" 之前匹配
    units = [("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)]
    for unit, mult in units:
        if shard_str.endswith(unit):
            try:
                return int(float(shard_str[:-len(unit)].strip()) * mult)
            except ValueError:
                return None
    return None


def save_state_dict_streaming(path, tensor_iterator, max_shard_size):
    """Stream tensors (name, tensor) into shards, write when current shard exceeds max_shard_size.
    tensor_iterator: yields (name, tensor) tuples.
    始终输出 model.safetensors-00001-of-XXXXX.safetensors 格式，与 HF 标准一致。
    """
    path.mkdir(exist_ok=True, parents=True)
    max_bytes = _parse_shard_size(max_shard_size) if isinstance(max_shard_size, str) else max_shard_size
    if max_bytes is None:
        # 默认 10GB 分片，确保输出 safetensors 格式而非 pytorch_model.bin
        max_bytes = _parse_shard_size("10GB")
        print(f'Shard size not specified, using default 10GB for safetensors format')
    if max_bytes is None:
        state_dict = dict(tensor_iterator)
        ckpt_filename = path / 'pytorch_model.bin'
        print(f'Saving to no-shard checkpoint ...')
        torch.save(state_dict, ckpt_filename)
        return

    current_shard = {}
    current_size = 0
    shard_idx = 0
    weight_map = {}
    total_size = 0

    def _tensor_numel(t):
        d = t.dtype
        if d in (torch.float16, torch.bfloat16):
            return t.numel() * 2
        if d == torch.float32:
            return t.numel() * 4
        return t.numel() * 4  # default

    for name, tensor in tensor_iterator:
        if not isinstance(tensor, torch.Tensor):
            continue
        nbytes = _tensor_numel(tensor)
        total_size += nbytes
        if current_size + nbytes > max_bytes and current_shard:
            shard_idx += 1
            shard_name = f"model.safetensors-{shard_idx:05d}.safetensors"
            print(f'Saving to shard checkpoint {path / shard_name} ...')
            safe_save_file(current_shard, path / shard_name, metadata={"format": "pt"})
            for k in current_shard:
                weight_map[k] = shard_name
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += nbytes

    if current_shard:
        shard_idx += 1
        shard_name = f"model.safetensors-{shard_idx:05d}.safetensors"
        print(f'Saving to shard checkpoint {path / shard_name} ...')
        safe_save_file(current_shard, path / shard_name, metadata={"format": "pt"})
        for k in current_shard:
            weight_map[k] = shard_name

    # Rename to standard format model.safetensors-00001-of-00014.safetensors
    for i in range(1, shard_idx + 1):
        old_name = f"model.safetensors-{i:05d}.safetensors"
        new_name = f"model.safetensors-{i:05d}-of-{shard_idx:05d}.safetensors"
        if (path / old_name).exists():
            (path / old_name).rename(path / new_name)
            for k in list(weight_map.keys()):
                if weight_map[k] == old_name:
                    weight_map[k] = new_name

    shards_index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    safe_index_filename = path / "model.safetensors.index.json"
    with open(safe_index_filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(shards_index, indent=2, sort_keys=True) + "\n")
    print(f'Successfully saved checkpoint to {path}')


def save_hf_checkpoint(path, state_dict, max_shard_size):
    path.mkdir(exist_ok=True, parents=True)
    if max_shard_size is None:
        ckpt_filename = path / 'pytorch_model.bin'
        print(f'Saving to no-shard checkpoint ...')
        torch.save(state_dict, ckpt_filename)
    else:
        ptrs = collections.defaultdict(list)
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                ptrs[id_tensor_storage(tensor)].append(name)
            else:
                ptrs[id(tensor)].append(name)

        shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
        for names in shared_ptrs.values():
            found = 0
            for name in sorted(names):
                matches_pattern = any(re.search(pat, name) for pat in KNOWN_TIED_WEIGHT_KEYS)
                if matches_pattern and name in state_dict:
                    found += 1
                    if found < len(names):
                        del state_dict[name]

        save_state_dict(path, state_dict, max_shard_size)
    print(f'Successfully saved checkpoint to {path}')


def add_arguments(parser):
    group = parser.add_argument_group(title='HF Qwen3.5 saver')
    parser.add_argument(
        "--shard", type=str, default=None,
        help='Sharded size, e.g. "2GB", "8GB", default None (no shards)',
    )
    return group


def build_linear_attn_qkv_weight_to_hf(megatron_cfg, val):
    """Convert Megatron interleaved layout to HF [Q_all|K_all|V_all] layout.
    Inverse of build_linear_attn_qkv_weight_from_hf in loader_fun_qwen3_5_hf_plus.
    """
    num_k_heads = getattr(megatron_cfg, 'linear_attention_num_key_heads', 16)
    num_v_heads = getattr(megatron_cfg, 'linear_attention_num_value_heads', 64)
    head_k_dim = getattr(megatron_cfg, 'linear_attention_key_head_dim', 128)
    head_v_dim = getattr(megatron_cfg, 'linear_attention_value_head_dim', 128)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    num_values_per_group = num_v_heads // num_k_heads
    group_size = head_k_dim + head_k_dim + num_values_per_group * head_v_dim

    # Megatron: (conv_dim, hidden_size), per group [Q|K|V]
    val = val.reshape((num_k_heads, group_size, -1))
    query = val[:, :head_k_dim, :].reshape(key_dim, -1)
    key = val[:, head_k_dim:head_k_dim * 2, :].reshape(key_dim, -1)
    value = val[:, head_k_dim * 2:, :].reshape(value_dim, -1)
    return torch.cat([query, key, value], dim=0)


def build_linear_attn_conv1d_to_hf(megatron_cfg, weight=None, bias=None):
    """Convert Megatron interleaved conv1d to HF [Q_all|K_all|V_all] layout.
    Inverse of build_linear_attn_conv1d_from_hf in loader_fun_qwen3_5_hf_plus.
    """
    num_k_heads = getattr(megatron_cfg, 'linear_attention_num_key_heads', 16)
    num_v_heads = getattr(megatron_cfg, 'linear_attention_num_value_heads', 64)
    head_k_dim = getattr(megatron_cfg, 'linear_attention_key_head_dim', 128)
    head_v_dim = getattr(megatron_cfg, 'linear_attention_value_head_dim', 128)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    num_values_per_group = num_v_heads // num_k_heads
    group_size = head_k_dim + head_k_dim + num_values_per_group * head_v_dim

    weight_hf = None
    if weight is not None:
        extra_dims = weight.shape[1:]
        val = weight.reshape((num_k_heads, group_size, *extra_dims))
        q_c = val[:, :head_k_dim].reshape(key_dim, *extra_dims)
        k_c = val[:, head_k_dim:head_k_dim * 2].reshape(key_dim, *extra_dims)
        v_c = val[:, head_k_dim * 2:].reshape(value_dim, *extra_dims)
        weight_hf = torch.cat([q_c, k_c, v_c], dim=0)

    bias_hf = None
    if bias is not None:
        val_b = bias.reshape((num_k_heads, group_size))
        q_b = val_b[:, :head_k_dim].reshape(key_dim)
        k_b = val_b[:, head_k_dim:head_k_dim * 2].reshape(key_dim)
        v_b = val_b[:, head_k_dim * 2:].reshape(value_dim)
        bias_hf = torch.cat([q_b, k_b, v_b], dim=0)
    return weight_hf, bias_hf


def fix_qkv_ordering(param, num_heads, hidden_size, num_kv_heads, head_dim, has_gate=False):
    """Split fused QKV weight back to separate Q, K, V (and optionally Gate)."""
    if has_gate:
        heads_per_group = num_heads // num_kv_heads
        per_group_dim = (heads_per_group * 2 + 2) * head_dim  # Q + Gate + K + V
        saved_shape = [num_kv_heads, per_group_dim] + list(param.shape[1:])
        qkv = param.view(*saved_shape)
        q_dim = heads_per_group * head_dim
        query, gate, key, value = qkv.split([q_dim, q_dim, head_dim, head_dim], dim=1)
        # Megatron: [Q_block|Gate_block|K|V] per group. HF expects q_proj: [Q0,G0,Q1,G1,...] interleaved per head.
        query = query.reshape(num_kv_heads, heads_per_group, head_dim, *param.shape[1:])
        gate = gate.reshape(num_kv_heads, heads_per_group, head_dim, *param.shape[1:])
        stacked = torch.stack([query, gate], dim=2)  # (ng, np/ng, 2, hn, ...)
        q_proj = stacked.reshape(num_heads, 2 * head_dim, *param.shape[1:]).reshape(-1, *param.shape[1:])
        key = key.contiguous().view(-1, *param.shape[1:])
        value = value.contiguous().view(-1, *param.shape[1:])
        return q_proj, key, value
    else:
        heads_per_group = num_heads // num_kv_heads
        per_group_dim = (heads_per_group + 2) * head_dim
        saved_shape = [num_kv_heads, per_group_dim] + list(param.shape[1:])
        qkv = param.view(*saved_shape)
        q_dim = heads_per_group * head_dim
        query, key, value = qkv.split([q_dim, head_dim, head_dim], dim=1)
        query = query.contiguous().view(-1, *param.shape[1:])
        key = key.contiguous().view(-1, *param.shape[1:])
        value = value.contiguous().view(-1, *param.shape[1:])
        return query, key, value


def convert_megatron_to_qwen3_5_moe(megatron_cfg, megatron_sd, num_query_groups=None):
    """Convert Megatron state dict to Qwen3.5-MoE HF format."""

    print("Converting from Megatron to Qwen3.5-MoE HF format ...")
    print('Megatron config:', pprint.pformat(megatron_cfg.__dict__))

    hidden_size = megatron_cfg.hidden_size
    num_heads = megatron_cfg.num_attention_heads
    num_kv_heads = getattr(megatron_cfg, 'num_query_groups', num_heads)
    if num_query_groups is not None:
        num_kv_heads = num_query_groups
    head_dim = megatron_cfg.kv_channels
    num_experts = getattr(megatron_cfg, 'num_experts', None)
    num_layers = getattr(megatron_cfg, 'num_layers_without_padding', megatron_cfg.num_layers)
    has_gate = getattr(megatron_cfg, 'attn_output_gate', False)
    layer_types = getattr(megatron_cfg, 'layer_types', None)
    if layer_types is None:
        # Infer from megatron_sd: in_proj_qkv -> linear_attention, linear_qkv -> full_attention
        layer_types = []
        for layer_idx in range(num_layers):
            if f"layers.{layer_idx}.self_attention.in_proj_qkv.weight" in megatron_sd:
                layer_types.append("linear_attention")
            else:
                layer_types.append("full_attention")
        print(f"Inferred layer_types from checkpoint: {layer_types[:5]}... ({len(layer_types)} layers)")

    # 标准格式使用 model.language_model. 前缀 (Qwen3_5MoeForConditionalGeneration)
    prefix = "model.language_model."

    # Output state dict
    hf_sd = {}

    # Embeddings（仅当 chunk 包含时处理，支持流式 partial chunk）
    word_embeddings = megatron_sd.get("embedding.word_embeddings.weight")
    if word_embeddings is not None:
        vocab_size = megatron_cfg.padded_vocab_size
        word_embeddings = word_embeddings[:vocab_size, :]
        hf_sd[f"{prefix}embed_tokens.weight"] = word_embeddings

    # Simple name mapping
    megatron_to_hf = {
        "self_attention.linear_proj": "self_attn.o_proj",
        "mlp.shared_experts.gate_weight": "mlp.shared_expert_gate",
        "mlp.router": "mlp.gate",
        "mlp.shared_experts.linear_fc2": "mlp.shared_expert.down_proj",
    }

    # Handle bare params (dt_bias, A_log) that don't match LAYER_RE
    for key, val in megatron_sd.items():
        m_bare = LAYER_BARE_PARAM_RE.match(key)
        if m_bare is not None:
            layer_idx = int(m_bare.group(1))
            param_name = m_bare.group(2)
            if layer_types is not None and layer_idx < len(layer_types) and layer_types[layer_idx] == 'linear_attention':
                hf_sd[f"{prefix}layers.{layer_idx}.linear_attn.{param_name}"] = val.clone()
            continue

    for key, val in megatron_sd.items():
        m = LAYER_RE.match(key)
        if m is None:
            continue

        layer_idx = int(m.group(1))
        op_name = m.group(2)
        weight_or_bias = m.group(3)
        layer_name = f"{prefix}layers.{layer_idx}"

        # Determine layer type
        is_linear_attn = (
            layer_types is not None
            and layer_idx < len(layer_types)
            and layer_types[layer_idx] == 'linear_attention'
        )

        # Shared expert FC1 (gate_proj + up_proj)
        if op_name == 'mlp.shared_experts.linear_fc1':
            w1, w2 = val.chunk(2, dim=0)
            hf_sd[f"{layer_name}.mlp.shared_expert.gate_proj.weight"] = w1.clone()
            hf_sd[f"{layer_name}.mlp.shared_expert.up_proj.weight"] = w2.clone()

        # Expert FC1: 标准格式使用 gate_up_proj (合并 gate+up)，形状 (num_experts, 2*intermediate_dim, hidden_dim)
        elif op_name == 'mlp.experts.local_experts.linear_fc1':
            # val 已是 (num_experts, 2*intermediate_dim, hidden_dim)，直接作为 gate_up_proj
            hf_sd[f"{layer_name}.mlp.experts.gate_up_proj"] = val.clone()

        # Expert FC2: 标准格式使用 down_proj，形状 (num_experts, hidden_dim, intermediate_dim)
        elif op_name == 'mlp.experts.local_experts.linear_fc2':
            # val 已是 (num_experts, hidden_dim, intermediate_dim)
            hf_sd[f"{layer_name}.mlp.experts.down_proj"] = val.clone()

        # Dense SwiGLU MLP.
        elif op_name == 'mlp.linear_fc1' and weight_or_bias == 'weight':
            gate_proj, up_proj = val.chunk(2, dim=0)
            hf_sd[f"{layer_name}.mlp.gate_proj.weight"] = gate_proj.clone()
            hf_sd[f"{layer_name}.mlp.up_proj.weight"] = up_proj.clone()

        elif op_name == 'mlp.linear_fc2' and weight_or_bias == 'weight':
            hf_sd[f"{layer_name}.mlp.down_proj.weight"] = val.clone()

        # LayerNorm weights
        elif op_name.endswith("layernorm"):
            if op_name.endswith("q_layernorm"):
                hf_sd[f"{layer_name}.self_attn.q_norm.{weight_or_bias}"] = val
            elif op_name.endswith("k_layernorm"):
                hf_sd[f"{layer_name}.self_attn.k_norm.{weight_or_bias}"] = val
            elif op_name.startswith("self_attention.") or op_name == "input_layernorm":
                hf_sd[f"{layer_name}.input_layernorm.{weight_or_bias}"] = val.clone()
            elif op_name.startswith("pre_mlp_layernorm"):
                hf_sd[f"{layer_name}.post_attention_layernorm.{weight_or_bias}"] = val.clone()

        # GatedDeltaNet linear attention weights (layout conversion for in_proj_qkv, conv1d)
        elif is_linear_attn and op_name == "self_attention.in_proj_qkv" and weight_or_bias == "weight":
            hf_val = build_linear_attn_qkv_weight_to_hf(megatron_cfg, val)
            hf_sd[f"{layer_name}.linear_attn.in_proj_qkv.weight"] = hf_val
        elif is_linear_attn and op_name == "self_attention.conv1d":
            conv_weight = val if weight_or_bias == "weight" else megatron_sd.get(f"layers.{layer_idx}.self_attention.conv1d.weight")
            conv_bias = val if weight_or_bias == "bias" else megatron_sd.get(f"layers.{layer_idx}.self_attention.conv1d.bias")
            weight_hf, bias_hf = build_linear_attn_conv1d_to_hf(megatron_cfg, weight=conv_weight, bias=conv_bias)
            if weight_hf is not None:
                # HF Conv1d expects (out_channels, 1, kernel_size) for depthwise
                if weight_hf.dim() == 2:
                    weight_hf = weight_hf.unsqueeze(1)
                hf_sd[f"{layer_name}.linear_attn.conv1d.weight"] = weight_hf
            if bias_hf is not None:
                hf_sd[f"{layer_name}.linear_attn.conv1d.bias"] = bias_hf
        elif is_linear_attn and op_name.startswith("self_attention."):
            # Map Megatron GatedDeltaNet params to HF naming (no layout change)
            param_name = op_name.replace("self_attention.", "")
            hf_key = f"{layer_name}.linear_attn.{param_name}.{weight_or_bias}"
            hf_sd[hf_key] = val.clone()

        # Full attention QKV
        elif op_name == "self_attention.linear_qkv" and weight_or_bias == "weight":
            q, k, v = fix_qkv_ordering(
                val, num_heads, hidden_size, num_kv_heads, head_dim, has_gate=has_gate
            )
            hf_sd[f"{layer_name}.self_attn.q_proj.weight"] = q
            hf_sd[f"{layer_name}.self_attn.k_proj.weight"] = k
            hf_sd[f"{layer_name}.self_attn.v_proj.weight"] = v

        elif op_name == "self_attention.linear_qkv" and weight_or_bias == "bias":
            q, k, v = fix_qkv_ordering(
                val, num_heads, hidden_size, num_kv_heads, head_dim, has_gate=has_gate
            )
            hf_sd[f"{layer_name}.self_attn.q_proj.bias"] = q
            hf_sd[f"{layer_name}.self_attn.k_proj.bias"] = k
            hf_sd[f"{layer_name}.self_attn.v_proj.bias"] = v

        # Generic weight mapping
        elif weight_or_bias == "weight" and op_name in megatron_to_hf:
            out_name = megatron_to_hf[op_name]
            hf_sd[f"{layer_name}.{out_name}.weight"] = val.clone()

        elif weight_or_bias == "bias" and op_name in megatron_to_hf:
            out_name = megatron_to_hf[op_name]
            hf_sd[f"{layer_name}.{out_name}.bias"] = val.clone()

    # Final layernorm (仅当 chunk 包含时处理，支持流式 partial chunk)
    if "decoder.final_layernorm.weight" in megatron_sd:
        hf_sd[f"{prefix}norm.weight"] = megatron_sd["decoder.final_layernorm.weight"]

    # LM head (仅当 chunk 包含时处理)
    tie_word_embeddings = not megatron_cfg.untie_embeddings_and_output_weights
    # Tied HF checkpoints keep only embed_tokens.weight; safetensors rejects
    # writing both aliases for the same storage.
    if not tie_word_embeddings and "output_layer.weight" in megatron_sd:
        hf_sd["lm_head.weight"] = megatron_sd["output_layer.weight"]

    return hf_sd


def save_checkpoint(queue, args):
    """Save checkpoint from queue to HF format."""

    # Receive metadata
    md = queue.get()
    if md == "exit":
        return

    output_dir = Path(args.save_dir)

    # Collect full state dict from queue
    megatron_sd = {}

    def queue_get():
        msg = queue.get()
        if isinstance(msg, str) and msg == "exit":
            raise RuntimeError("Got 'exit' from loader")
        return msg

    # Get embeddings
    msg = queue_get()
    assert msg["name"] == "embeddings"
    megatron_sd["embedding.word_embeddings.weight"] = msg["word embeddings"]

    # Get layers
    for layer_idx in range(md.num_layers):
        msg = queue_get()
        assert msg["name"] == f"transformer layer {layer_idx}"

        layer_type = msg.get("layer_type", "full_attention")
        prefix = f"layers.{layer_idx}."

        if layer_type == "linear_attention":
            # GatedDeltaNet weights
            megatron_sd[prefix + "input_layernorm.weight"] = msg["input norm weight"]
            for key_name in [
                "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                "conv1d", "dt_bias", "A_log", "norm", "out_proj"
            ]:
                weight_key = f"linear_attn {key_name} weight"
                if weight_key in msg:
                    megatron_sd[prefix + f"self_attention.{key_name}.weight"] = msg[weight_key]
                bias_key = f"linear_attn {key_name} bias"
                if bias_key in msg:
                    megatron_sd[prefix + f"self_attention.{key_name}.bias"] = msg[bias_key]
                # Handle non-weight/bias params
                bare_key = f"linear_attn {key_name}"
                if bare_key in msg and weight_key not in msg and bias_key not in msg:
                    megatron_sd[prefix + f"self_attention.{key_name}"] = msg[bare_key]
        else:
            # Full attention weights
            attn_gate = getattr(md, 'attn_output_gate', False)
            if attn_gate:
                megatron_sd[prefix + "input_layernorm.weight"] = msg["input norm weight"]
            else:
                megatron_sd[prefix + "self_attention.linear_qkv.layer_norm_weight"] = msg["input norm weight"]
            megatron_sd[prefix + "self_attention.linear_qkv.weight"] = msg["qkv weight"]
            if "qkv bias" in msg:
                megatron_sd[prefix + "self_attention.linear_qkv.bias"] = msg["qkv bias"]
            megatron_sd[prefix + "self_attention.linear_proj.weight"] = msg["dense weight"]
            if "q layernorm weight" in msg:
                megatron_sd[prefix + "self_attention.q_layernorm.weight"] = msg["q layernorm weight"]
                megatron_sd[prefix + "self_attention.k_layernorm.weight"] = msg["k layernorm weight"]

        # MLP weights
        megatron_sd[prefix + "pre_mlp_layernorm.weight"] = msg["post norm weight"]

        if hasattr(md, 'num_experts') and md.num_experts:
            megatron_sd[prefix + "mlp.router.weight"] = msg["router weight"]
            if md.swiglu:
                w = torch.cat([msg["mlp l0 weight W"], msg["mlp l0 weight V"]], dim=-2)
                megatron_sd[prefix + "mlp.experts.local_experts.linear_fc1.weight"] = w
            megatron_sd[prefix + "mlp.experts.local_experts.linear_fc2.weight"] = msg["mlp l1 weight"]

            if "shared mlp l0 weight W" in msg:
                sw = torch.cat([msg["shared mlp l0 weight W"], msg["shared mlp l0 weight V"]], dim=0)
                megatron_sd[prefix + "mlp.shared_experts.linear_fc1.weight"] = sw
                megatron_sd[prefix + "mlp.shared_experts.linear_fc2.weight"] = msg["shared mlp l1 weight"]
                megatron_sd[prefix + "mlp.shared_experts.gate_weight.weight"] = msg["shared mlp gate weight"]

    # Final norm
    msg = queue_get()
    assert msg["name"] == "final norm"
    megatron_sd["decoder.final_layernorm.weight"] = msg["weight"]

    # Output layer
    if md.output_layer:
        msg = queue_get()
        assert msg["name"] == "output layer"
        megatron_sd["output_layer.weight"] = msg["weight"]

    # Done signal
    msg = queue.get()
    assert msg == "done"

    # Build megatron config namespace for conversion
    megatron_cfg = md.checkpoint_args
    megatron_cfg.attn_output_gate = getattr(md, 'attn_output_gate', False)
    megatron_cfg.layer_types = getattr(md, 'layer_types', None)
    megatron_cfg.num_experts = getattr(md, 'num_experts', None)

    # Convert to HF format
    hf_sd = convert_megatron_to_qwen3_5_moe(megatron_cfg, megatron_sd)

    # Save
    save_hf_checkpoint(output_dir, hf_sd, args.shard)

    # Save config (minimal, user should provide full config separately)
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "hidden_size": megatron_cfg.hidden_size,
        "num_attention_heads": megatron_cfg.num_attention_heads,
        "num_hidden_layers": megatron_cfg.num_layers,
        "vocab_size": megatron_cfg.padded_vocab_size,
        "model_type": "qwen3_5_moe",
    }
    config_path = output_dir / "config.json"
    if not config_path.exists():
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved minimal config to {config_path}")
