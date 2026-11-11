import argparse
import collections
import json
import pprint
import re
from pathlib import Path
from typing import Optional
from einops import rearrange

import torch
from safetensors.torch import save_file as safe_save_file
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.pytorch_utils import id_tensor_storage

# The regex to extract layer names.
LAYER_RE = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
VISION_LAYER_RE = re.compile(r"vision_model.decoder.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z_]+)")

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


def fix_query_key_value_ordering_qwen2(
        param, num_heads, hidden_size, num_key_value_heads
):
    input_shape = param.size()
    channels = hidden_size // num_heads
    saved_shape = [num_key_value_heads, (num_heads // num_key_value_heads + 2) * channels] + list(input_shape[1:])
    qkv_weight = param.view(*saved_shape)  # 8+1+1, 4, 128, hid
    query, key, value = qkv_weight.split(
        [num_heads // num_key_value_heads * channels, channels, channels], dim=1)

    query, key, value = query.contiguous().view([-1] + list(input_shape[1:])), \
        key.contiguous().view([-1] + list(input_shape[1:])), \
        value.contiguous().view([-1] + list(input_shape[1:]))

    return query, key, value


def convert_megatron_to_qwen2vl(
        megatron_cfg: argparse.Namespace,
        megatron_sd,
        num_query_groups: int = None,
):
    try:
        from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
    except ImportError as e:
        print('Cannot import Qwen2VLModel, please check your transformers install.')
        exit(1)

    print("Converting from megatron to qwen2vl ...")
    print('Megatron config:', pprint.pformat(megatron_cfg.__dict__))

    config_dict = dict(
        bos_token_id=151643,
        eos_token_id=151643,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        vision_token_id=151654,
        image_token_id=151655,
        video_token_id=151656,
        hidden_size=megatron_cfg.hidden_size,
        intermediate_size=megatron_cfg.ffn_hidden_size,
        max_position_embeddings=megatron_cfg.max_position_embeddings,
        num_attention_heads=megatron_cfg.num_attention_heads,
        num_key_value_heads=megatron_cfg.num_query_groups,
        max_window_layers=megatron_cfg.num_layers,
        num_hidden_layers=megatron_cfg.num_layers,
        rms_norm_eps=megatron_cfg.layernorm_epsilon,
        rope_theta=megatron_cfg.rotary_base,
        torch_dtype='bfloat16',
        vocab_size=megatron_cfg.padded_vocab_size,
        tie_word_embeddings=not megatron_cfg.untie_embeddings_and_output_weights,
        rope_scaling={
            "type": "mrope",
            "mrope_section": [
                megatron_cfg.checkpoint_args.llm_rotary_tdim,
                megatron_cfg.checkpoint_args.llm_rotary_hdim,
                megatron_cfg.checkpoint_args.llm_rotary_wdim
            ]
        }
    )
    qwen2vl_cfg = Qwen2VLConfig(**config_dict)

    if num_query_groups is not None:
        qwen2vl_cfg.num_key_value_heads = num_query_groups
    if getattr(megatron_cfg, 'group_query_attention', False):
        # Set from megatron config.
        qwen2vl_cfg.num_key_value_heads = megatron_cfg.num_query_groups


    qwen2vl_cfg.architectures = ["Qwen2VLForConditionalGeneration"]
    print('Qwen2VL config:', qwen2vl_cfg)

    def _name(*parts) -> str:
        """Make parameter name from parts."""
        name = '.'.join(parts)
        return name

    # The converted output model.
    qwen2vl_sd = {}

    # The number of heads.
    heads = qwen2vl_cfg.num_attention_heads
    num_key_value_heads = qwen2vl_cfg.num_key_value_heads
    # The hidden_size .
    hidden_size = qwen2vl_cfg.hidden_size
    print(megatron_sd.keys())
    # The word embeddings.
    word_embeddings = megatron_sd["embedding.word_embeddings.weight"]

    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: qwen2vl_cfg.vocab_size, :]
    qwen2vl_sd[_name("model", "embed_tokens", "weight")] = word_embeddings
    print(f"word_embeddings: {word_embeddings.size()}. vocab_size: {qwen2vl_cfg.vocab_size}.")

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
            qwen2vl_sd[w1_name] = w1
            qwen2vl_sd[w2_name] = w2

        # For layernorm(s), simply store the layer norm.
        elif op_name.endswith("layernorm"):

            if op_name.startswith("self_attention."):
                qwen2vl_sd[
                    _name('model', 'layers', layer_idx, 'input_layernorm', weight_or_bias)
                ] = val
            elif op_name.startswith("mlp"):
                qwen2vl_sd[
                    _name('model', 'layers', layer_idx, 'post_attention_layernorm', weight_or_bias)
                ] = val

        # Transpose the QKV matrix.
        elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.linear_qkv"
        ) and weight_or_bias == "weight":
            q, k, v = fix_query_key_value_ordering_qwen2(
                val, heads, hidden_size, num_key_value_heads
            )
            q_name = _name('model', 'layers', layer_idx, 'self_attn', 'q_proj', 'weight')
            k_name = _name('model', 'layers', layer_idx, 'self_attn', 'k_proj', 'weight')
            v_name = _name('model', 'layers', layer_idx, 'self_attn', 'v_proj', 'weight')
            qwen2vl_sd[q_name] = q
            qwen2vl_sd[k_name] = k
            qwen2vl_sd[v_name] = v

        # Transpose the bias.
        elif (
                op_name == "attention.query_key_value"
                or op_name == "self_attention.linear_qkv"
        ) and weight_or_bias == "bias":
            q, k, v = fix_query_key_value_ordering_qwen2(
                val, heads, hidden_size, num_key_value_heads
            )
            q_name = _name('model', 'layers', layer_idx, 'self_attn', 'q_proj', 'bias')
            k_name = _name('model', 'layers', layer_idx, 'self_attn', 'k_proj', 'bias')
            v_name = _name('model', 'layers', layer_idx, 'self_attn', 'v_proj', 'bias')
            qwen2vl_sd[q_name] = q
            qwen2vl_sd[k_name] = k
            qwen2vl_sd[v_name] = v

        # Transpose the weights.
        elif weight_or_bias == "weight":
            out_name = megatron_to_transformers[op_name]
            qwen2_key = _name('model', 'layers', layer_idx, out_name, 'weight')
            qwen2vl_sd[qwen2_key] = val
        else:
            raise ValueError(f'param {key} failed')

    # The final layernorm.
    qwen2vl_sd[_name("model", "norm", "weight")] = transformer["decoder.final_layernorm.weight"]

    # LM head
    if not qwen2vl_cfg.tie_word_embeddings:
        output_layer = transformer["output_layer.weight"]
        qwen2vl_sd[_name("lm_head", "weight")] = output_layer
    else:
        qwen2vl_sd["lm_head.weight"] = word_embeddings

    # save vision_model patch_embed
    # -----------
    qwen2vl_sd[_name("visual", "patch_embed", "proj", "weight")] = transformer["vision_model.patch_embed.proj.weight"]

    # save vision_model final_layernorm
    # -----------
    qwen2vl_sd[_name("visual", "merger", "ln_q", "weight")] = transformer["vision_model.decoder.final_layernorm.weight"]
    qwen2vl_sd[_name("visual", "merger", "ln_q", "bias")] = transformer["vision_model.decoder.final_layernorm.bias"]

    # save vision_model projection
    # -----------
    qwen2vl_sd[_name("visual", "merger", "mlp.0", "weight")] = transformer["vision_model.projection.encoder.linear_fc1.weight"]
    qwen2vl_sd[_name("visual", "merger", "mlp.0","bias")] = transformer["vision_model.projection.encoder.linear_fc1.bias"]
    qwen2vl_sd[_name("visual", "merger", "mlp.2","weight")] = transformer["vision_model.projection.encoder.linear_fc2.weight"]
    qwen2vl_sd[_name("visual", "merger", "mlp.2","bias")] = transformer["vision_model.projection.encoder.linear_fc2.bias"]

    # Extract the vision layers.
    for key, val in transformer.items():
        # Match the name.
        m = VISION_LAYER_RE.match(key)

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
            if weight_or_bias == 'layer_norm_weight':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'norm2', 'weight')] = val
            elif weight_or_bias == 'layer_norm_bias':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'norm2', 'bias')] = val
            elif weight_or_bias == 'weight':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'mlp', 'fc1', 'weight')] = val
            elif weight_or_bias == 'bias':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'mlp', 'fc1', 'bias')] = val
            else:
                print(f'Unexpected key: "{key}"')
                exit(1)
        if op_name == 'mlp.linear_fc2':
            if weight_or_bias == 'weight':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'mlp', 'fc2', 'weight')] = val
            elif weight_or_bias == 'bias':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'mlp', 'fc2', 'bias')] = val
            else:
                print(f'Unexpected key: "{key}"')
                exit(1)
        if op_name == 'self_attention.linear_qkv':
            if weight_or_bias == 'layer_norm_weight':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'norm1', 'weight')] = val
            elif weight_or_bias == 'layer_norm_bias':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'norm1', 'bias')] = val
            elif weight_or_bias == 'weight':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'attn', 'qkv', 'weight')] = rearrange(
                    val, '(h n d) in -> (n h d) in', n=3, h=megatron_cfg.vision_num_attention_heads
                )
            elif weight_or_bias == 'bias':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'attn', 'qkv', 'bias')] = rearrange(
                    val, '(h n d) -> (n h d)', n=3, h=megatron_cfg.vision_num_attention_heads
                )
            else:
                print(f'Unexpected key: "{key}"')
                exit(1)
        if op_name == 'self_attention.linear_proj':
            if weight_or_bias == 'weight':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'attn', 'proj', 'weight')] = val
            elif weight_or_bias == 'bias':
                qwen2vl_sd[_name('visual', 'blocks', layer_idx, 'attn', 'proj', 'bias')] = val
            else:
                print(f'Unexpected key: "{key}"')
                exit(1)
    for key, value in qwen2vl_sd.items():
        print(f"{key} {str(value.size())}")
    return qwen2vl_cfg, qwen2vl_sd

def set_dense_mlp(mcore_state, prefix, msg):
    post_norm_weight = msg.pop("post norm weight")

    mlp_l1_weight = msg.pop("mlp l1 weight")

    mlp_l0_weight_W = msg.pop("mlp l0 weight W")
    mlp_l0_weight_V = msg.pop("mlp l0 weight V")
    mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
    mcore_state[f'{prefix}.mlp.linear_fc1.layernorm.weight'] = post_norm_weight
    mcore_state[f'{prefix}.mlp.linear_fc1.weight'] = mlp_l0_weight
    mcore_state[f'{prefix}.mlp.linear_fc2.weight'] = mlp_l1_weight


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

        if md.add_qkv_bias:
            qkv_bias = msg.pop("qkv bias")
            mcore_state[f'{prefix}.self_attention.linear_qkv.bias'] = qkv_bias

        # mlp
        if md.num_experts:
            print("expert is not supported yet.")
            exit(0)
        else:
            set_dense_mlp(mcore_state, prefix, msg)

        total_layer_num = total_layer_num + 1

    msg = queue_get("final norm")
    final_norm_weight = msg.pop("weight")
    mcore_state[f'decoder.final_layernorm.weight'] = final_norm_weight

    if md.output_layer:
        msg = queue_get("output layer")
        output_layer_weight = msg.pop("weight")
        mcore_state['output_layer.weight'] = output_layer_weight
    if args.model_type == 'Reward':
        pooler_head = queue_get("pooler head")
        mcore_state['pooler_head'] = pooler_head

    # save vision_model patch_embed
    # -----------
    patch_embed_msg = queue_get("vision_model patch_embed")
    mcore_state['vision_model.patch_embed.proj.weight'] = patch_embed_msg['patch_embed']

    # save vision_model final_layernorm
    # -----------
    final_layernorm_msg = queue_get("vision_model final_layernorm")
    mcore_state['vision_model.decoder.final_layernorm.weight'] = final_layernorm_msg['weight']
    mcore_state['vision_model.decoder.final_layernorm.bias'] = final_layernorm_msg['bias']

    # save vision_model projection
    # -----------
    projection_msg = queue_get("vision_model projection")
    mcore_state['vision_model.projection.encoder.linear_fc1.weight'] = projection_msg['l0 weight']
    mcore_state['vision_model.projection.encoder.linear_fc1.bias'] = projection_msg['l0 bias']
    mcore_state['vision_model.projection.encoder.linear_fc2.weight'] = projection_msg['l1 weight']
    mcore_state['vision_model.projection.encoder.linear_fc2.bias'] = projection_msg['l1 bias']

    # save vision_model Transformer layers.
    # ------------------
    vision_model_total_layer_num = 0
    for layer_idx in range(md.vision_num_layers):
        prefix = f'vision_model.decoder.layers.{layer_idx}'

        msg = queue_get(f"vision layer {vision_model_total_layer_num}")

        input_norm_weight = msg.pop("attn norm weight")
        mcore_state[f'{prefix}.self_attention.linear_qkv.layer_norm_weight'] = input_norm_weight
        input_norm_bias = msg.pop("attn norm bias")
        mcore_state[f'{prefix}.self_attention.linear_qkv.layer_norm_bias'] = input_norm_bias
        post_norm_weight = msg.pop("mlp l0 norm weight")
        mcore_state[f'{prefix}.mlp.linear_fc1.layer_norm_weight'] = post_norm_weight
        post_norm_bias = msg.pop("mlp l0 norm bias")
        mcore_state[f'{prefix}.mlp.linear_fc1.layer_norm_bias'] = post_norm_bias
        dense_bias = msg.pop("attn proj bias")
        mcore_state[f'{prefix}.self_attention.linear_proj.bias'] = dense_bias
        mlp_l1_bias = msg.pop("mlp l1 bias")
        mcore_state[f'{prefix}.mlp.linear_fc2.bias'] = mlp_l1_bias

        qkv_weight = msg.pop("attn qkv weight")
        mcore_state[f'{prefix}.self_attention.linear_qkv.weight'] = qkv_weight
        dense_weight = msg.pop("attn proj weight")
        mcore_state[f'{prefix}.self_attention.linear_proj.weight'] = dense_weight

        mlp_l1_weight = msg.pop("mlp l1 norm weight")
        mcore_state[f'{prefix}.mlp.linear_fc2.weight'] = mlp_l1_weight

        qkv_bias = msg.pop("attn qkv bias")
        mcore_state[f'{prefix}.self_attention.linear_qkv.bias'] = qkv_bias

        if md.swiglu:
            mlp_l0_weight_W = msg.pop("mlp l0 weight W")
            mlp_l0_weight_V = msg.pop("mlp l0 weight V")
            mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
            mcore_state[f'{prefix}.mlp.linear_fc1.weight'] = mlp_l0_weight
            mlp_l0_bias_W = msg.pop("mlp l0 bias W")
            mlp_l0_bias_V = msg.pop("mlp l0 bias V")
            mlp_l0_bias = torch.cat((mlp_l0_bias_W, mlp_l0_bias_V), dim=-1)
            mcore_state[f'{prefix}.mlp.linear_fc1.bias'] = mlp_l0_bias
        else:
            print("not swiglu, please check")
            exit(0)
        vision_model_total_layer_num = vision_model_total_layer_num + 1

    if md.num_experts:
        print("expert is not supported yet.")
        exit(0)
        # qwen2_cfg, output_state_dict = convert_megatron_to_qwen2_moe(md, mcore_state, num_query_groups=md.num_query_groups)
    else:
        qwen2_cfg, output_state_dict = convert_megatron_to_qwen2vl(md, mcore_state, num_query_groups=md.num_query_groups)
    save_hf_checkpoint(Path(args.save_dir), output_state_dict, args.shard)
    qwen2_cfg.save_pretrained(Path(args.save_dir))
    print("Done!")
