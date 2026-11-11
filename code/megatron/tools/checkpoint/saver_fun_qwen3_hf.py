# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
import argparse
import collections
import json
import pprint
import re
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file as safe_save_file
from transformers.pytorch_utils import id_tensor_storage

channels_model = os.environ.get('Channels_Model', 'False')
channels_model = True if channels_model == 'True' else False

# The regex to extract layer names.
LAYER_RE = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

LOCAL_HF_FILES_PATH = Path(__file__).absolute().parent.parent / "local-hf-files"

try:
    from transformers.modeling_utils import shard_checkpoint


    def save_state_dict(path, state_dict, max_shard_size):
        shards_dict, shards_index = shard_checkpoint(
            state_dict, max_shard_size, weights_name='model.safetensors')

        # Save index.
        if shards_index:
            # Only save non-empty shards index.
            safe_index_filename = path / "model.safetensors.index.json"
            with open(safe_index_filename, 'w', encoding='utf-8') as f_safe_index:
                content = json.dumps(shards_index, indent=2, sort_keys=True) + "\n"
                f_safe_index.write(content)

        # Save shards.
        for shard_file, shard in shards_dict.items():
            shard_filename = path / shard_file
            print(f'Saving to shard checkpoint {shard_filename} ...')
            safe_save_file(shard, shard_filename, metadata={"format": "pt"})

except ImportError:
    print('WARNING: Cannot import `transformers.modeling_utils.shard_checkpoint`, '
          'use `huggingface_hub.split_torch_state_dict_into_shards` instead.')

    from huggingface_hub import split_torch_state_dict_into_shards


    def save_state_dict(path, state_dict, max_shard_size):
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern="model{suffix}.safetensors", max_shard_size=max_shard_size)

        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            safe_save_file(
                shard,
                path / filename,
                metadata={"format": "pt"},
            )
        if state_dict_split.is_sharded:
            shards_index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            # Only save non-empty shards index.
            safe_index_filename = path / "model.safetensors.index.json"
            with open(safe_index_filename, 'w', encoding='utf-8') as f_safe_index:
                content = json.dumps(shards_index, indent=2, sort_keys=True) + "\n"
                f_safe_index.write(content)

KNOWN_TIED_WEIGHT_KEYS = ["lm_head.weight"]


def save_hf_checkpoint(
        path: Path,
        state_dict: dict,
        max_shard_size: Optional[str],
):
    path.mkdir(exist_ok=True, parents=True)

    if max_shard_size is None:
        ckpt_filename = path / 'pytorch_model.bin'
        print(f'Saving to no-shard checkpoint ...')
        torch.save(state_dict, ckpt_filename)
    else:
        # Remove tensor aliases before sharding.
        # See `transformers.modeling_utils.PreTrainedModel.save_pretrained` for more details.

        # Safetensors does not allow tensor aliasing.
        # We're going to remove aliases before saving
        ptrs = collections.defaultdict(list)
        for name, tensor in state_dict.items():
            # Sometimes in the state_dict we have non-tensor objects.
            # e.g. in bitsandbytes we have some `str` objects in the state_dict
            if isinstance(tensor, torch.Tensor):
                ptrs[id_tensor_storage(tensor)].append(name)
            else:
                # In the non-tensor case, fall back to the pointer of the object itself
                ptrs[id(tensor)].append(name)

        # These are all the pointers of shared tensors.
        shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

        # [NOTE] (suyang.fy):
        #  To simplify the implementation, this function only allowed tensor aliases in `KNOWN_TIED_WEIGHT_KEYS`.
        warn_names = set()
        for names in shared_ptrs.values():
            found = 0
            for name in sorted(names):
                matches_pattern = any(re.search(pat, name) for pat in KNOWN_TIED_WEIGHT_KEYS)
                if matches_pattern and name in state_dict:
                    found += 1
                    if found < len(names):
                        print(f'Remove known tensor alias {name} from the state dict.')
                        del state_dict[name]

            found = 0
            for name in names:
                if name in state_dict:
                    found += 1
                    if found > 1:
                        del state_dict[name]
                        warn_names.add(name)

            if len(warn_names) > 0:
                raise RuntimeError(
                    f"Unexpected tensor aliases {warn_names} detected while saving. Please check the checkpoint again.",
                )

        save_state_dict(path, state_dict, max_shard_size)
    print(f'Successful saved checkpoint to {path}')


def add_arguments(parser):
    group = parser.add_argument_group(title='Hf qwen saver')
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help='Sharded size of converted HF checkpoint, e.g. "2GB", "8GB", default to None (no shards)',
    )
    return group


def fix_query_key_value_ordering_qwen3(
        param, num_heads, hidden_size, num_key_value_heads, channels=None,
):
    input_shape = param.size()
    if channels is None:
        channels = hidden_size // num_heads
    saved_shape = [num_key_value_heads, (num_heads // num_key_value_heads + 2) * channels] + list(input_shape[1:])
    qkv_weight = param.view(*saved_shape)  # 8+1+1, 4, 128, hid
    query, key, value = qkv_weight.split(
        [num_heads // num_key_value_heads * channels, channels, channels], dim=1)

    query, key, value = query.contiguous().view([-1] + list(input_shape[1:])), \
        key.contiguous().view([-1] + list(input_shape[1:])), \
        value.contiguous().view([-1] + list(input_shape[1:]))

    return query, key, value

def convert_megatron_to_qwen3_moe(
        megatron_cfg: argparse.Namespace,
        megatron_sd,
        num_query_groups: int = None,
):
    assert getattr(megatron_cfg, 'num_experts', 1) > 1, 'Not a MoE model'

    try:
        from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
    except ImportError:
        print('Cannot import Qwen3MoeModel from transformers, please check your transformers install, make sure >=4.51.0')

    print("Converting from megatron to qwen3-moe ...")
    print('Megatron config:', pprint.pformat(megatron_cfg.__dict__))

    if megatron_cfg.moe_shared_expert_intermediate_size is not None:
        moe_shared_expert_intermediate_size = megatron_cfg.moe_shared_expert_intermediate_size
    else:
        moe_shared_expert_intermediate_size = 0

    # Spell out all parameters in case the defaults change.
    qwen3_moe_cfg = Qwen3MoeConfig(
        bos_token_id=151643,
        eos_token_id=151645,
        hidden_size=megatron_cfg.hidden_size,
        intermediate_size=megatron_cfg.ffn_hidden_size * megatron_cfg.moe_router_topk,
        max_position_embeddings=megatron_cfg.max_position_embeddings,
        num_attention_heads=megatron_cfg.num_attention_heads,
        num_key_value_heads=megatron_cfg.num_attention_heads,
        num_hidden_layers=getattr(megatron_cfg, "num_layers_without_padding", megatron_cfg.num_layers),
        max_window_layers=getattr(megatron_cfg, "num_layers_without_padding", megatron_cfg.num_layers),
        rms_norm_eps=megatron_cfg.layernorm_epsilon,
        rope_theta=megatron_cfg.rotary_base,
        torch_dtype="bfloat16",
        vocab_size=megatron_cfg.padded_vocab_size,
        tie_word_embeddings=not megatron_cfg.untie_embeddings_and_output_weights,
        decoder_sparse_step=1,   #megatron_cfg.moe_expert_interval,
        moe_intermediate_size=megatron_cfg.ffn_hidden_size,
        num_experts_per_tok=megatron_cfg.moe_router_topk,
        num_experts=megatron_cfg.num_experts,
        output_router_logits=True,
        router_aux_loss_coef=0.001, #megatron_cfg.moe_loss_weight,
        norm_topk_prob=megatron_cfg.moe_norm_topk_scores
    )
    if num_query_groups is not None:
        qwen3_moe_cfg.num_key_value_heads = num_query_groups

    if getattr(megatron_cfg, 'group_query_attention', False):
        # Set from megatron config.
        qwen3_moe_cfg.num_key_value_heads = megatron_cfg.num_query_groups

    head_dim = megatron_cfg.kv_channels
    if head_dim * megatron_cfg.num_attention_heads != megatron_cfg.hidden_size:
        qwen3_moe_cfg.head_dim = head_dim

    qwen3_moe_cfg.architectures = ["Qwen3MoeForCausalLM"]

    print('Qwen3-MoE config:', qwen3_moe_cfg)

    def _name(*parts) -> str:
        """Make parameter name from parts."""
        name = '.'.join(parts)
        return name

    # The converted output model.
    qwen3_moe_sd = {}
    # The hidden_size
    hidden_size = qwen3_moe_cfg.hidden_size
    # The number of heads.
    heads = qwen3_moe_cfg.num_attention_heads
    num_key_value_heads = qwen3_moe_cfg.num_key_value_heads
    # Number of routed experts
    num_experts = qwen3_moe_cfg.num_experts

    print(f"megatron_sd: {megatron_sd.keys()}")
    prefix_name = "model."

    # The word embeddings.
    word_embeddings = megatron_sd["embedding.word_embeddings.weight"]
    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: qwen3_moe_cfg.vocab_size, :]
    qwen3_moe_sd[f"{prefix_name}embed_tokens.weight"] = word_embeddings
    print(f"word_embeddings: {word_embeddings.size()}. vocab_size: {qwen3_moe_cfg.vocab_size}.")

    # The transformer. now encoder
    transformer = megatron_sd

    # The simple map of names for "automated" rules.
    megatron_to_transformers = {
        "self_attention.linear_proj": "self_attn.o_proj",
        "mlp.shared_experts.gate_weight": "mlp.shared_expert_gate",
        "mlp.router": "mlp.gate",
        "mlp.shared_experts.linear_fc2": "mlp.shared_expert.down_proj",
    }

    # Extract the layers.
    for key, val in transformer.items():
        m = LAYER_RE.match(key)

        # Stop if that's not a layer
        if m is None:
            continue

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"{prefix_name}layers.{layer_idx}"
        # For layernorm(s), simply store the layer norm.
        # print(op_name)
        if op_name == 'mlp.shared_experts.linear_fc1':
            w1, w2 = val.chunk(2, dim=0)
            w1 = w1.clone().detach()
            w2 = w2.clone().detach()
            w1_name = '.'.join([layer_name + f".mlp.shared_expert.gate_proj.weight"])
            w2_name = '.'.join([layer_name + f".mlp.shared_expert.up_proj.weight"])
            qwen3_moe_sd[w1_name] = w1
            qwen3_moe_sd[w2_name] = w2

        # For layernorm(s), simply store the layer norm.
        elif op_name.endswith("layernorm"):
            if op_name.endswith("q_layernorm"):
                qwen3_moe_sd[
                    _name('model', 'layers', str(layer_idx), 'self_attn.q_norm', weight_or_bias)
                ] = val
            elif op_name.endswith("k_layernorm"):
                qwen3_moe_sd[
                    _name('model', 'layers', str(layer_idx), 'self_attn.k_norm', weight_or_bias)
                ] = val
            elif op_name.startswith("self_attention."):
                qwen3_moe_sd[
                    _name(layer_name, 'input_layernorm', weight_or_bias)
                ] = val.clone()
            elif op_name.startswith("pre_mlp_layernorm"):
                qwen3_moe_sd[
                    _name(layer_name, 'post_attention_layernorm', weight_or_bias)
                ] = val.clone()

        elif op_name == "mlp.experts.local_experts.linear_fc1":
            experts = val
            assert len(experts) == num_experts
            for expert_idx in range(num_experts):
                w1, w2 = experts[expert_idx].chunk(2, dim=0)
                w1 = w1.clone().detach()
                w2 = w2.clone().detach()
                qwen3_moe_sd[layer_name + f".mlp.experts.{expert_idx}.gate_proj.weight"] = w1
                qwen3_moe_sd[layer_name + f".mlp.experts.{expert_idx}.up_proj.weight"] = w2
        elif op_name == "mlp.experts.local_experts.linear_fc2":
            experts = val
            assert len(experts) == num_experts
            for expert_idx in range(num_experts):
                qwen3_moe_sd[layer_name + f".mlp.experts.{expert_idx}.down_proj.weight"] = experts[expert_idx].clone()
        # Transpose the QKV matrix.
        elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.linear_qkv"
        ) and weight_or_bias == "weight":
            q, k, v = fix_query_key_value_ordering_qwen3(
                val, heads, hidden_size, num_key_value_heads, channels=head_dim,
            )

            qwen3_moe_sd[layer_name + ".self_attn.q_proj.weight"] = q
            qwen3_moe_sd[layer_name + ".self_attn.k_proj.weight"] = k
            qwen3_moe_sd[layer_name + ".self_attn.v_proj.weight"] = v
        # Transpose the bias.
        elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.linear_qkv"
        ) and weight_or_bias == "bias":
            q, k, v = fix_query_key_value_ordering_qwen3(
                val, heads, hidden_size, num_key_value_heads, channels=head_dim,
            )
            qwen3_moe_sd[layer_name + ".self_attn.q_proj.bias"] = q
            qwen3_moe_sd[layer_name + ".self_attn.k_proj.bias"] = k
            qwen3_moe_sd[layer_name + ".self_attn.v_proj.bias"] = v
        # Transpose the weights. Add clone() to avoid shared data.
        elif weight_or_bias == "weight":
            out_name = megatron_to_transformers[op_name]
            qwen3_moe_sd[_name(layer_name, out_name, "weight")] = val.clone()
        # Copy the bias. Add clone() to avoid shared data.
        elif weight_or_bias == "bias":
            assert False
            out_name = megatron_to_transformers[op_name]
            qwen3_moe_sd[_name(layer_name, out_name, "bias")] = val.clone()
        else:
            assert False

    # DEBUG.
    assert qwen3_moe_cfg.num_hidden_layers == layer_idx + 1

    # The final layernorm.
    qwen3_moe_sd[f"{prefix_name}norm.weight"] = transformer[
        "decoder.final_layernorm.weight"
    ]

    # LM head
    if not qwen3_moe_cfg.tie_word_embeddings:
        output_layer = transformer["output_layer.weight"]
        qwen3_moe_sd["lm_head.weight"] = output_layer
    else:
        qwen3_moe_sd["lm_head.weight"] = word_embeddings
    # It should be done!
    return qwen3_moe_cfg, qwen3_moe_sd

def convert_megatron_to_qwen3(
        megatron_cfg: argparse.Namespace,
        megatron_sd,
        num_query_groups: int = None,
):
    try:
        from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
    except ImportError as e:
        print('Cannot import Qwen3Model, please check your transformers install, make sure >=4.51.0')
        exit(1)

    print("Converting from megatron to qwen3 ...")
    print('Megatron config:', pprint.pformat(megatron_cfg.__dict__))

    config_dict = dict(
        bos_token_id=151643,
        eos_token_id=151643,
        hidden_size=megatron_cfg.hidden_size,
        intermediate_size=megatron_cfg.ffn_hidden_size,
        max_position_embeddings=megatron_cfg.max_position_embeddings,
        num_attention_heads=megatron_cfg.num_attention_heads,
        num_key_value_heads=megatron_cfg.num_attention_heads,
        num_hidden_layers=megatron_cfg.num_layers,
        rms_norm_eps=megatron_cfg.layernorm_epsilon,
        rope_theta=megatron_cfg.rotary_base,
        torch_dtype='bfloat16',
        vocab_size=megatron_cfg.padded_vocab_size,
        tie_word_embeddings=not megatron_cfg.untie_embeddings_and_output_weights,
        qkv_bias=megatron_cfg.add_qkv_bias,
        # for hf sft triaing
        output_router_logits=True,
    )

    qwen3_cfg = Qwen3Config(**config_dict)

    if num_query_groups is not None:
        qwen3_cfg.num_key_value_heads = num_query_groups
    if getattr(megatron_cfg, 'group_query_attention', False):
        # Set from megatron config.
        qwen3_cfg.num_key_value_heads = megatron_cfg.num_query_groups

    head_dim = megatron_cfg.kv_channels
    if head_dim * megatron_cfg.num_attention_heads != megatron_cfg.hidden_size:
        qwen3_cfg.head_dim = head_dim

    if megatron_cfg.qk_layernorm:
        qwen3_cfg.use_qk_norm = True

    qwen3_cfg.architectures = ["Qwen3ForCausalLM"]
    print('Qwen3 config:', qwen3_cfg)

    def _name(*parts) -> str:
        """Make parameter name from parts."""
        name = '.'.join(parts)
        return name

    # The converted output model.
    qwen3_sd = {}

    # The number of heads.
    heads = qwen3_cfg.num_attention_heads
    num_key_value_heads = qwen3_cfg.num_key_value_heads
    # The hidden_size .
    hidden_size = qwen3_cfg.hidden_size
    print(megatron_sd.keys())
    # The word embeddings.
    word_embeddings = megatron_sd["embedding.word_embeddings.weight"]

    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: qwen3_cfg.vocab_size, :]
    qwen3_sd[_name("model", "embed_tokens", "weight")] = word_embeddings
    print(f"word_embeddings: {word_embeddings.size()}. vocab_size: {qwen3_cfg.vocab_size}.")

    # The transformer. now encoder
    transformer = megatron_sd

    # The simple map of names for "automated" rules.

    megatron_to_transformers = {
        "mlp.linear_fc2": "mlp.down_proj",
        "self_attention.linear_proj": "self_attn.o_proj"
    }

    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = LAYER_RE.match(key)

        # Stop if that's not a layer
        if m is None:
            continue

        # The index of the layer.
        layer_idx = m.group(1)
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)
        if op_name == 'mlp.linear_fc1':
            w1, w2 = val.chunk(2, dim=0)
            w1 = w1.clone().detach()
            w2 = w2.clone().detach()
            w1_name = _name('model', 'layers', layer_idx, 'mlp', 'gate_proj', 'weight')
            w2_name = _name('model', 'layers', layer_idx, 'mlp', 'up_proj', 'weight')
            qwen3_sd[w1_name] = w1
            qwen3_sd[w2_name] = w2

        # For layernorm(s), simply store the layer norm.
        elif op_name.endswith("layernorm"):
            if op_name.endswith("q_layernorm"):
                qwen3_sd[
                    _name('model', 'layers', layer_idx, 'self_attn.q_norm', weight_or_bias)
                ] = val
            elif op_name.endswith("k_layernorm"):
                qwen3_sd[
                    _name('model', 'layers', layer_idx, 'self_attn.k_norm', weight_or_bias)
                ] = val
            elif op_name.startswith("self_attention."):
                qwen3_sd[
                    _name('model', 'layers', layer_idx, 'input_layernorm', weight_or_bias)
                ] = val
            elif op_name.startswith("mlp"):
                qwen3_sd[
                    _name('model', 'layers', layer_idx, 'post_attention_layernorm', weight_or_bias)
                ] = val

        # Transpose the QKV matrix.
        elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.linear_qkv"
        ) and weight_or_bias == "weight":
            q, k, v = fix_query_key_value_ordering_qwen3(
                val, heads, hidden_size, num_key_value_heads, channels=head_dim,
            )
            q_name = _name('model', 'layers', layer_idx, 'self_attn', 'q_proj', 'weight')
            k_name = _name('model', 'layers', layer_idx, 'self_attn', 'k_proj', 'weight')
            v_name = _name('model', 'layers', layer_idx, 'self_attn', 'v_proj', 'weight')
            qwen3_sd[q_name] = q
            qwen3_sd[k_name] = k
            qwen3_sd[v_name] = v

        # Transpose the bias.
        elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.linear_qkv"
        ) and weight_or_bias == "bias":
            q, k, v = fix_query_key_value_ordering_qwen3(
                val, heads, hidden_size, num_key_value_heads, channels=head_dim,
            )
            q_name = _name('model', 'layers', layer_idx, 'self_attn', 'q_proj', 'bias')
            k_name = _name('model', 'layers', layer_idx, 'self_attn', 'k_proj', 'bias')
            v_name = _name('model', 'layers', layer_idx, 'self_attn', 'v_proj', 'bias')
            qwen3_sd[q_name] = q
            qwen3_sd[k_name] = k
            qwen3_sd[v_name] = v

        # Transpose the weights.
        elif weight_or_bias == "weight":
            out_name = megatron_to_transformers[op_name]
            qwen3_key = _name('model', 'layers', layer_idx, out_name, 'weight')
            qwen3_sd[qwen3_key] = val
        else:
            raise ValueError(f'param {key} failed')

    # The final layernorm.
    qwen3_sd[_name("model", "norm", "weight")] = transformer["decoder.final_layernorm.weight"]

    # LM head
    if not qwen3_cfg.tie_word_embeddings:
        output_layer = transformer["output_layer.weight"]
        qwen3_sd[_name("lm_head", "weight")] = output_layer
    else:
        qwen3_sd["lm_head.weight"] = word_embeddings

    return qwen3_cfg, qwen3_sd

def convert_megatron_to_hf_audio_encoder(md, mcore_state):
    """Convert megatron audio encoder format to HF audio encoder format and save."""
    if 'audio_encoder.weight' not in mcore_state:
        print("Warning: audio_encoder.weight not found in mcore_state, skipping audio encoder conversion.")
        return
    
    from types import SimpleNamespace as NS
    ae_weight = mcore_state['audio_encoder.weight']
    
    # Convert SimpleNamespace to state dict format
    hf_audio_encoder_state = {}
    
    # conv1
    def _get(obj, key):
        return getattr(obj, key) if hasattr(obj, key) else obj[key]
    
    conv1 = _get(ae_weight, "conv1")
    conv1_conv = _get(conv1, "conv")
    hf_audio_encoder_state['audio_encoder.conv1.weight'] = _get(conv1_conv, "weight")
    hf_audio_encoder_state['audio_encoder.conv1.bias'] = _get(conv1_conv, "bias")
    
    # conv2
    conv2 = _get(ae_weight, "conv2")
    conv2_conv = _get(conv2, "conv")
    hf_audio_encoder_state['audio_encoder.conv2.weight'] = _get(conv2_conv, "weight")
    hf_audio_encoder_state['audio_encoder.conv2.bias'] = _get(conv2_conv, "bias")
    
    # ln_post
    ln_post = _get(ae_weight, "ln_post")
    hf_audio_encoder_state['audio_encoder.ln_post.weight'] = _get(ln_post, "weight")
    hf_audio_encoder_state['audio_encoder.ln_post.bias'] = _get(ln_post, "bias")
    
    # blocks
    blocks = _get(ae_weight, "blocks")
    num_blocks = len(blocks)
    print(f"Saving audio encoder with {num_blocks} blocks")
    
    for block_idx in range(num_blocks):
        block = blocks[block_idx]
        
        # attn
        attn = _get(block, "attn")
        # query
        query = _get(attn, "query")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.attn.query.weight'] = _get(query, "weight")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.attn.query.bias'] = _get(query, "bias")
        # key
        key = _get(attn, "key")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.attn.key.weight'] = _get(key, "weight")
        # value
        value = _get(attn, "value")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.attn.value.weight'] = _get(value, "weight")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.attn.value.bias'] = _get(value, "bias")
        # out
        out = _get(attn, "out")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.attn.out.weight'] = _get(out, "weight")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.attn.out.bias'] = _get(out, "bias")
        # fsmn_block
        fsmn_block = _get(attn, "fsmn_block")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.attn.fsmn_block.weight'] = _get(fsmn_block, "weight")
        
        # attn_ln
        attn_ln = _get(block, "attn_ln")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.attn_ln.weight'] = _get(attn_ln, "weight")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.attn_ln.bias'] = _get(attn_ln, "bias")
        
        # mlp (Linear-ReLU-Linear in list indices 0 and 2)
        mlp = _get(block, "mlp")
        if mlp[0] is not None:
            hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.mlp.0.weight'] = _get(mlp[0], "weight")
            hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.mlp.0.bias'] = _get(mlp[0], "bias")
        if mlp[2] is not None:
            hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.mlp.2.weight'] = _get(mlp[2], "weight")
            hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.mlp.2.bias'] = _get(mlp[2], "bias")
        
        # mlp_ln
        mlp_ln = _get(block, "mlp_ln")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.mlp_ln.weight'] = _get(mlp_ln, "weight")
        hf_audio_encoder_state[f'audio_encoder.blocks.{block_idx}.mlp_ln.bias'] = _get(mlp_ln, "bias")
    
    return hf_audio_encoder_state

def convert_megatron_to_hf_audio_adaptor(md, mcore_state):
    """Convert megatron audio adaptor format to HF audio adaptor format and save."""
    if 'audio_adaptor.weight' not in mcore_state:
        print("Warning: audio_adaptor.weight not found in mcore_state, skipping audio adaptor conversion.")
        return
    
    from types import SimpleNamespace as NS
    aa_weight = mcore_state['audio_adaptor.weight']
    
    # Convert SimpleNamespace to state dict format
    hf_audio_adaptor_state = {}
    
    # Helper function to get attribute or item
    def _get(obj, key):
        return getattr(obj, key) if hasattr(obj, key) else obj[key]
    
    # linear1
    linear1 = _get(aa_weight, "linear1")
    hf_audio_adaptor_state['audio_adaptor.linear1.weight'] = _get(linear1, "weight")
    hf_audio_adaptor_state['audio_adaptor.linear1.bias'] = _get(linear1, "bias")
    
    # linear2
    linear2 = _get(aa_weight, "linear2")
    hf_audio_adaptor_state['audio_adaptor.linear2.weight'] = _get(linear2, "weight")
    hf_audio_adaptor_state['audio_adaptor.linear2.bias'] = _get(linear2, "bias")
    
    # blocks
    blocks = _get(aa_weight, "blocks")
    num_blocks = len(blocks)
    print(f"Saving audio adaptor with {num_blocks} blocks")
    
    for block_idx in range(num_blocks):
        block = blocks[block_idx]
        
        # self_attn
        self_attn = _get(block, "self_attn")
        # linear_q
        linear_q = _get(self_attn, "linear_q")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.self_attn.linear_q.weight'] = _get(linear_q, "weight")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.self_attn.linear_q.bias'] = _get(linear_q, "bias")
        # linear_k
        linear_k = _get(self_attn, "linear_k")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.self_attn.linear_k.weight'] = _get(linear_k, "weight")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.self_attn.linear_k.bias'] = _get(linear_k, "bias")
        # linear_v
        linear_v = _get(self_attn, "linear_v")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.self_attn.linear_v.weight'] = _get(linear_v, "weight")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.self_attn.linear_v.bias'] = _get(linear_v, "bias")
        # linear_out
        linear_out = _get(self_attn, "linear_out")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.self_attn.linear_out.weight'] = _get(linear_out, "weight")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.self_attn.linear_out.bias'] = _get(linear_out, "bias")
        
        # feed_forward
        feed_forward = _get(block, "feed_forward")
        # w_1
        w_1 = _get(feed_forward, "w_1")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.feed_forward.w_1.weight'] = _get(w_1, "weight")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.feed_forward.w_1.bias'] = _get(w_1, "bias")
        # w_2
        w_2 = _get(feed_forward, "w_2")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.feed_forward.w_2.weight'] = _get(w_2, "weight")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.feed_forward.w_2.bias'] = _get(w_2, "bias")
        
        # norm1
        norm1 = _get(block, "norm1")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.norm1.weight'] = _get(norm1, "weight")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.norm1.bias'] = _get(norm1, "bias")
        
        # norm2
        norm2 = _get(block, "norm2")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.norm2.weight'] = _get(norm2, "weight")
        hf_audio_adaptor_state[f'audio_adaptor.blocks.{block_idx}.norm2.bias'] = _get(norm2, "bias")
    
    return hf_audio_adaptor_state

def set_dense_mlp(mcore_state, prefix, msg):
    post_norm_weight = msg.pop("post norm weight")

    mlp_l1_weight = msg.pop("mlp l1 weight")

    mlp_l0_weight_W = msg.pop("mlp l0 weight W")
    mlp_l0_weight_V = msg.pop("mlp l0 weight V")
    mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
    mcore_state[f'{prefix}.mlp.linear_fc1.layernorm.weight'] = post_norm_weight
    mcore_state[f'{prefix}.mlp.linear_fc1.weight'] = mlp_l0_weight
    mcore_state[f'{prefix}.mlp.linear_fc2.weight'] = mlp_l1_weight

def set_moe_mlp(mcore_state, prefix, msg):
    post_norm_weight = msg.pop("post norm weight")
    mcore_state[f'{prefix}.pre_mlp_layernorm.weight'] = post_norm_weight
    if "shared mlp l0 weight W" in msg.keys(): # hacky may improve
        shared_expert_mlp_l0_weight_W = msg.pop("shared mlp l0 weight W")
        shared_expert_mlp_l0_weight_V = msg.pop("shared mlp l0 weight V")
        shared_expert_mlp_l0_weight = torch.cat((shared_expert_mlp_l0_weight_W, shared_expert_mlp_l0_weight_V), dim=-2)
        shared_expert_mlp_l1_weight = msg.pop("shared mlp l1 weight")
        shared_expert_gate_weight = msg.pop("shared gate weight")
        mcore_state[f'{prefix}.mlp.shared_experts.gate_weight.weight'] = shared_expert_gate_weight
        mcore_state[f'{prefix}.mlp.shared_experts.linear_fc1.weight'] = shared_expert_mlp_l0_weight
        mcore_state[f'{prefix}.mlp.shared_experts.linear_fc2.weight'] = shared_expert_mlp_l1_weight

    router_weight = msg.pop("router weight")
    mcore_state[f'{prefix}.mlp.router.weight'] = router_weight

    mlp_l0_weight_W = msg.pop("mlp l0 weight W")
    mlp_l0_weight_V = msg.pop("mlp l0 weight V")
    mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
    mcore_state[f'{prefix}.mlp.experts.local_experts.linear_fc1.weight'] = mlp_l0_weight
    mlp_l1_weight = msg.pop("mlp l1 weight")
    mcore_state[f'{prefix}.mlp.experts.local_experts.linear_fc2.weight'] = mlp_l1_weight

def save_checkpoint(queue, args):
    import os
    import sys
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

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

    md = queue_get()
    mcore_state = {}
    # Embeddings
    # -----------
    embeddings_msg = queue_get("embeddings")
    mcore_state['embedding.word_embeddings.weight'] = embeddings_msg['word embeddings']
    if channels_model:
        mcore_state['embedding_audio.word_embeddings.weight'] = embeddings_msg['embedding audio']

    # Transformer layers.
    # ------------------
    total_layer_num = 0
    for layer_idx in range(md.num_layers):
        prefix = f'layers.{layer_idx}'

        msg = queue_get(f"transformer layer {total_layer_num}")

        input_norm_weight = msg.pop("input norm weight")

        qkv_weight = msg.pop("qkv weight")
        dense_weight = msg.pop("dense weight")

        # Save them to the model state
        # attention
        mcore_state[f'{prefix}.self_attention.linear_proj.weight'] = dense_weight
        mcore_state[f'{prefix}.self_attention.linear_qkv.weight'] = qkv_weight
        mcore_state[f'{prefix}.self_attention.linear_qkv.layernorm.weight'] = input_norm_weight

        if md.qk_layernorm:
            mcore_state[f'{prefix}.self_attention.q_layernorm.weight'] = msg.pop("q layernorm weight")
            mcore_state[f'{prefix}.self_attention.k_layernorm.weight'] = msg.pop("k layernorm weight")

        if md.add_qkv_bias:
            qkv_bias = msg.pop("qkv bias")
            mcore_state[f'{prefix}.self_attention.linear_qkv.bias'] = qkv_bias

        # mlp
        if md.num_experts:
            set_moe_mlp(mcore_state, prefix, msg)
        else:
            set_dense_mlp(mcore_state, prefix, msg)

        total_layer_num = total_layer_num + 1

    # get audio encoder weight
    msg = queue_get("audio encoder")
    audio_encoder_weight = msg.pop("weight")
    mcore_state['audio_encoder.weight'] = audio_encoder_weight

    # get audio adaptor weight
    msg = queue_get("audio adaptor")
    audio_adaptor_weight = msg.pop("weight")
    mcore_state['audio_adaptor.weight'] = audio_adaptor_weight

    msg = queue_get("final norm")
    final_norm_weight = msg.pop("weight")
    mcore_state[f'decoder.final_layernorm.weight'] = final_norm_weight

    if md.output_layer:
        msg = queue_get("output layer")
        output_layer_weight = msg.pop("weight")
        mcore_state['output_layer.weight'] = output_layer_weight
    if channels_model:
        msg = queue_get("output layer audio")
        output_layer_audio_weight = msg.pop("weight")
        mcore_state['output_layer_audio.weight'] = output_layer_audio_weight

    if md.num_experts:
        qwen3_cfg, output_state_dict = convert_megatron_to_qwen3_moe(md, mcore_state, num_query_groups=md.num_query_groups)
    else:
        qwen3_cfg, output_state_dict = convert_megatron_to_qwen3(md, mcore_state, num_query_groups=md.num_query_groups)
    save_hf_checkpoint(Path(args.save_dir), output_state_dict, args.shard)
    qwen3_cfg.save_pretrained(Path(args.save_dir))

    # save audio encoder
    audio_encoder_weight = convert_megatron_to_hf_audio_encoder(md, mcore_state)
    audio_encoder_path = Path(args.save_dir) / "audio_encoder.pt"
    torch.save(audio_encoder_weight, audio_encoder_path)

    # save audio adaptor
    audio_adaptor_weight = convert_megatron_to_hf_audio_adaptor(md, mcore_state)
    audio_adaptor_path = Path(args.save_dir) / "audio_adaptor.pt"
    torch.save(audio_adaptor_weight, audio_adaptor_path)

    if channels_model:
        # save embedding audio and output layer audio
        embedding_audio_weight = mcore_state['embedding_audio.word_embeddings.weight']
        embedding_audio_path = Path(args.save_dir) / "embedding_audio.pt"
        torch.save(embedding_audio_weight, embedding_audio_path)

        output_layer_audio_weight = mcore_state['output_layer_audio.weight']
        output_layer_audio_path = Path(args.save_dir) / "output_layer_audio.pt"
        torch.save(output_layer_audio_weight, output_layer_audio_path)

    # Save generation config to avoid RL error (eos_token_id).
    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        bos_token_id=151643,
        do_sample=True,
        eos_token_id=[
            151645,
            151643
        ],
        pad_token_id=151643,
        repetition_penalty=1.0,
        temperature=0.6,
        top_k=40,
        top_p=0.95,
        transformers_version="4.51.0",
    )
    gen_config.save_pretrained(Path(args.save_dir))

    print("Done!")
