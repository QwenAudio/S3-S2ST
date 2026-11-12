# Copyright (c) 2025. Qwen3.5-MoE Fun HF checkpoint saver for Megatron-LM.
"""Save Megatron Qwen3.5-MoE Fun checkpoints (with audio encoder/adaptor) to HuggingFace format.

Extends saver_qwen3_5_hf with support for:
- Audio encoder (blocks-based, SenseVoiceSmall, or Qwen3-2B transformer)
- Audio adaptor
- Channels model (embedding_audio, output_layer_audio)
"""

import gc
import os
import argparse
import json
import tempfile
from pathlib import Path

import torch

channels_model = os.environ.get('Channels_Model', 'False')
channels_model = True if channels_model == 'True' else False

# 复用 saver_qwen3_5_hf 的核心转换逻辑
from saver_qwen3_5_hf import (
    save_hf_checkpoint,
    save_state_dict_streaming,
    convert_megatron_to_qwen3_5_moe,
    LAYER_RE,
)

# 复用 saver_fun_qwen3_hf_plus 的 audio 转换逻辑
from saver_fun_qwen3_hf_plus import (
    convert_megatron_to_hf_audio_encoder,
    convert_megatron_to_hf_audio_adaptor,
    convert_megatron_to_hf_sensevoicesmall_audio_encoder,
    convert_megatron_to_qwen3,
    set_dense_mlp,
    set_moe_mlp,
)


def add_arguments(parser):
    group = parser.add_argument_group(title='HF Qwen3.5 Fun saver')
    parser.add_argument(
        "--shard", type=str, default="10GB",
        help='Sharded size for model.safetensors-00001-of-XXXXX.safetensors, e.g. "2GB", "5GB", "10GB", default 10GB',
    )
    return group


def save_checkpoint(queue, args):
    """Save checkpoint from queue to HF format (Qwen3.5-MoE + audio)."""
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
        if name is not None and args.checking and val.get("name") != name:
            val_name = val.get("name", "?")
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    md = queue_get()
    streaming_mode = os.environ.get('SAVER_STREAMING_MODE', '0') == '1'
    output_dir = Path(args.save_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    temp_dir = None
    if streaming_mode:
        temp_dir = Path(tempfile.mkdtemp(prefix="hf_ckpt_streaming_", dir=str(output_dir)))
        print(f"SAVER_STREAMING_MODE=1: Using temp dir {temp_dir} for low-memory streaming")

    megatron_sd = {}
    mcore_state = {}  # 用于 audio 组件的转换（audio 组件较小，可保留在内存）

    def _write_layer_to_temp(chunk_dict, chunk_name):
        if streaming_mode and temp_dir and chunk_dict:
            torch.save(chunk_dict, temp_dir / f"{chunk_name}.pt")
            chunk_dict.clear()
            gc.collect()

    # Embeddings
    embeddings_msg = queue_get("embeddings")
    megatron_sd["embedding.word_embeddings.weight"] = embeddings_msg["word embeddings"]
    if getattr(md, "has_embedding_audio", False):
        mcore_state["embedding_audio.word_embeddings.weight"] = embeddings_msg["embedding audio"]
    if streaming_mode:
        _write_layer_to_temp({"embedding.word_embeddings.weight": megatron_sd["embedding.word_embeddings.weight"]}, "embeddings")
        del megatron_sd["embedding.word_embeddings.weight"]
        gc.collect()

    # Transformer layers (LLM backbone)
    for layer_idx in range(md.num_layers):
        msg = queue_get()
        assert msg["name"] == f"transformer layer {layer_idx}"

        layer_type = msg.get("layer_type", "full_attention")
        prefix = f"layers.{layer_idx}."

        layer_chunk = {}
        if layer_type == "linear_attention":
            layer_chunk[prefix + "input_layernorm.weight"] = msg["input norm weight"]
            for key_name in [
                "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                "conv1d", "dt_bias", "A_log", "norm", "out_proj"
            ]:
                weight_key = f"linear_attn {key_name} weight"
                if weight_key in msg:
                    layer_chunk[prefix + f"self_attention.{key_name}.weight"] = msg[weight_key]
                bias_key = f"linear_attn {key_name} bias"
                if bias_key in msg:
                    layer_chunk[prefix + f"self_attention.{key_name}.bias"] = msg[bias_key]
                bare_key = f"linear_attn {key_name}"
                if bare_key in msg and weight_key not in msg and bias_key not in msg:
                    layer_chunk[prefix + f"self_attention.{key_name}"] = msg[bare_key]
        else:
            attn_gate = getattr(md, 'attn_output_gate', False)
            if attn_gate:
                layer_chunk[prefix + "input_layernorm.weight"] = msg["input norm weight"]
            else:
                layer_chunk[prefix + "self_attention.linear_qkv.layer_norm_weight"] = msg["input norm weight"]
            layer_chunk[prefix + "self_attention.linear_qkv.weight"] = msg["qkv weight"]
            if "qkv bias" in msg:
                layer_chunk[prefix + "self_attention.linear_qkv.bias"] = msg["qkv bias"]
            layer_chunk[prefix + "self_attention.linear_proj.weight"] = msg["dense weight"]
            if "q layernorm weight" in msg:
                layer_chunk[prefix + "self_attention.q_layernorm.weight"] = msg["q layernorm weight"]
                layer_chunk[prefix + "self_attention.k_layernorm.weight"] = msg["k layernorm weight"]

        layer_chunk[prefix + "pre_mlp_layernorm.weight"] = msg["post norm weight"]

        if hasattr(md, 'num_experts') and md.num_experts:
            layer_chunk[prefix + "mlp.router.weight"] = msg["router weight"]
            if md.swiglu:
                w = torch.cat([msg["mlp l0 weight W"], msg["mlp l0 weight V"]], dim=-2)
                layer_chunk[prefix + "mlp.experts.local_experts.linear_fc1.weight"] = w
            layer_chunk[prefix + "mlp.experts.local_experts.linear_fc2.weight"] = msg["mlp l1 weight"]

            shared_gate_key = "shared mlp gate weight" if "shared mlp gate weight" in msg else "shared gate weight"
            if shared_gate_key in msg or "shared mlp l0 weight W" in msg:
                sw = torch.cat([msg["shared mlp l0 weight W"], msg["shared mlp l0 weight V"]], dim=0)
                layer_chunk[prefix + "mlp.shared_experts.linear_fc1.weight"] = sw
                layer_chunk[prefix + "mlp.shared_experts.linear_fc2.weight"] = msg["shared mlp l1 weight"]
                layer_chunk[prefix + "mlp.shared_experts.gate_weight.weight"] = msg.get(
                    "shared mlp gate weight", msg.get("shared gate weight")
                )
        else:
            if md.swiglu:
                layer_chunk[prefix + "mlp.linear_fc1.weight"] = torch.cat(
                    [msg["mlp l0 weight W"], msg["mlp l0 weight V"]], dim=0
                )
            else:
                layer_chunk[prefix + "mlp.linear_fc1.weight"] = msg["mlp l0 weight"]
            layer_chunk[prefix + "mlp.linear_fc2.weight"] = msg["mlp l1 weight"]

        if streaming_mode:
            _write_layer_to_temp(layer_chunk, f"layer_{layer_idx}")
        else:
            megatron_sd.update(layer_chunk)
        del msg
        if streaming_mode and layer_idx % 4 == 0:
            gc.collect()

    # Audio encoder
    if args.qwen3_audio_encoder:
        for layer_idx in range(md.audio_encoder_num_layers):
            msg = queue_get(f"audio encoder transformer layer {layer_idx}")
            prefix = f'audio_encoder.layers.{layer_idx}'
            mcore_state[f'{prefix}.self_attention.linear_proj.weight'] = msg["dense weight"]
            mcore_state[f'{prefix}.self_attention.linear_qkv.weight'] = msg["qkv weight"]
            mcore_state[f'{prefix}.self_attention.linear_qkv.layernorm.weight'] = msg["input norm weight"]
            set_dense_mlp(mcore_state, prefix, msg)
        msg = queue_get("audio encoder final norm")
        mcore_state['audio_encoder.final_layernorm.weight'] = msg["weight"]
    else:
        msg = queue_get("audio encoder")
        mcore_state['audio_encoder.weight'] = msg["weight"]

    # Audio adaptor
    msg = queue_get("audio adaptor")
    mcore_state['audio_adaptor.weight'] = msg["weight"]

    # Final norm
    msg = queue_get("final norm")
    final_norm_chunk = {"decoder.final_layernorm.weight": msg["weight"]}
    if streaming_mode:
        _write_layer_to_temp(final_norm_chunk, "final_norm")
    else:
        megatron_sd["decoder.final_layernorm.weight"] = msg["weight"]
    del msg

    # Output layer
    if md.output_layer:
        msg = queue_get("output layer")
        output_chunk = {"output_layer.weight": msg["weight"]}
        if streaming_mode:
            _write_layer_to_temp(output_chunk, "output_layer")
        else:
            megatron_sd["output_layer.weight"] = msg["weight"]
        del msg
    if getattr(md, "has_output_layer_audio", False):
        msg = queue_get("output layer audio")
        mcore_state['output_layer_audio.weight'] = msg["weight"]

    cosyvoice_codex_state = None
    if getattr(md, "has_cosyvoice_codex_head", False):
        msg = queue_get("cosyvoice codex")
        cosyvoice_codex_state = {
            key: value
            for key, value in msg.items()
            if key != "name"
        }
    cosyvoice_fm_state = None
    cosyvoice_fm_model_conf = None
    if getattr(md, "has_cosyvoice_fm", False):
        msg = queue_get("cosyvoice fm")
        if not msg.get("exported_by_loader", False):
            cosyvoice_fm_state = msg["state_dict"]
            cosyvoice_fm_model_conf = msg.get("model_conf")

    # Qwen3 audio encoder conv weights (在 output layer 之后发送)
    if args.qwen3_audio_encoder:
        msg = queue_get("audio encoder conv1 weights")
        mcore_state['audio_encoder.conv1.weight'] = msg["weight"]
        mcore_state['audio_encoder.conv1.bias'] = msg["bias"]
        msg = queue_get("audio encoder conv2 weights")
        mcore_state['audio_encoder.conv2.weight'] = msg["weight"]
        mcore_state['audio_encoder.conv2.bias'] = msg["bias"]

    # Done signal
    msg = queue.get()
    assert msg == "done"

    # 转换 LLM 部分为 Qwen3.5-MoE HF 格式
    megatron_cfg = md.checkpoint_args
    megatron_cfg.attn_output_gate = getattr(md, 'attn_output_gate', False)
    megatron_cfg.layer_types = getattr(md, 'layer_types', None)
    megatron_cfg.num_experts = getattr(md, 'num_experts', None)

    if streaming_mode and temp_dir:
        # Phase 2: 从 temp 文件流式加载、转换、写入 shard
        def _stream_hf_tensors():
            layer_types = megatron_cfg.layer_types
            if layer_types is None:
                layer_types = []
                for i in range(md.num_layers):
                    layer_pt = temp_dir / f"layer_{i}.pt"
                    if layer_pt.exists():
                        chunk = torch.load(layer_pt, map_location='cpu', weights_only=True)
                        if any("in_proj_qkv" in k for k in chunk):
                            layer_types.append("linear_attention")
                        else:
                            layer_types.append("full_attention")
                    else:
                        layer_types.append("full_attention")
                megatron_cfg.layer_types = layer_types

            word_embeddings = None
            emb_pt = temp_dir / "embeddings.pt"
            if emb_pt.exists():
                chunk = torch.load(emb_pt, map_location='cpu', weights_only=True)
                hf_chunk = convert_megatron_to_qwen3_5_moe(megatron_cfg, chunk)
                word_embeddings = chunk.get("embedding.word_embeddings.weight")
                for k, v in hf_chunk.items():
                    yield k, v
                del chunk
                gc.collect()

            for i in range(md.num_layers):
                layer_pt = temp_dir / f"layer_{i}.pt"
                chunk = torch.load(layer_pt, map_location='cpu', weights_only=True)
                hf_chunk = convert_megatron_to_qwen3_5_moe(megatron_cfg, chunk)
                for k, v in hf_chunk.items():
                    yield k, v
                del chunk
                if i % 8 == 0:
                    gc.collect()

            final_chunk = {}
            fn_pt = temp_dir / "final_norm.pt"
            if fn_pt.exists():
                final_chunk.update(torch.load(fn_pt, map_location='cpu', weights_only=True))
            out_pt = temp_dir / "output_layer.pt"
            if out_pt.exists():
                final_chunk.update(torch.load(out_pt, map_location='cpu', weights_only=True))
            # tied 时 convert 需要 embedding.word_embeddings.weight 来设置 lm_head
            if not megatron_cfg.untie_embeddings_and_output_weights and word_embeddings is not None:
                final_chunk["embedding.word_embeddings.weight"] = word_embeddings
            if final_chunk:
                hf_chunk = convert_megatron_to_qwen3_5_moe(megatron_cfg, final_chunk)
                prefix = "model.language_model."
                for k, v in hf_chunk.items():
                    # 跳过 embed_tokens（已在前面 yield），只 yield norm 和 lm_head
                    if k == prefix + "embed_tokens.weight":
                        continue
                    yield k, v

        save_state_dict_streaming(output_dir, _stream_hf_tensors(), args.shard)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        hf_sd = convert_megatron_to_qwen3_5_moe(megatron_cfg, megatron_sd)
        save_hf_checkpoint(output_dir, hf_sd, args.shard)

    # 保存 config，按 Qwen3.5-35B-A3B 标准格式：LLM 配置放在 text_config 下
    layer_types = getattr(megatron_cfg, 'layer_types', None)
    if layer_types is None:
        num_layers = megatron_cfg.num_layers
        layer_types = []
        for i in range(num_layers):
            if f"layers.{i}.self_attention.in_proj_qkv.weight" in megatron_sd:
                layer_types.append("linear_attention")
            else:
                layer_types.append("full_attention")

    vocab_size = getattr(md, 'true_vocab_size', None) or megatron_cfg.padded_vocab_size
    # rope_parameters 与标准 Qwen3.5-35B-A3B 一致
    mrope_section = getattr(megatron_cfg, 'mrope_section', None)
    if mrope_section is None:
        mrope_section = [11, 11, 10]
    rope_params = {
        "mrope_interleaved": getattr(megatron_cfg, 'mrope_interleaved', True) or True,
        "mrope_section": mrope_section,
        "rope_type": "default",
        "rope_theta": getattr(megatron_cfg, 'rotary_base', 10000000),
        "partial_rotary_factor": getattr(megatron_cfg, 'rotary_percent', 0.25),
    }

    is_moe = bool(getattr(megatron_cfg, 'num_experts', None))
    text_config = {
        # OmniDRCA2's runtime uses the nested architecture to construct the
        # audio adaptor output projection.  Without it transformers falls back
        # to the old 2048-wide text model although this checkpoint is 2560.
        "architectures": [
            "Qwen3_5MoeForCausalLM" if is_moe else "Qwen3_5ForCausalLM"
        ],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": getattr(megatron_cfg, 'attn_output_gate', True),
        "dtype": (
            "bfloat16" if getattr(megatron_cfg, "bf16", False)
            else "float16" if getattr(megatron_cfg, "fp16", False)
            else "float32"
        ),
        "eos_token_id": 248044,
        "full_attention_interval": getattr(megatron_cfg, 'full_attention_interval', 4),
        "head_dim": megatron_cfg.kv_channels,
        "hidden_act": "silu",
        "hidden_size": megatron_cfg.hidden_size,
        "initializer_range": 0.02,
        "layer_types": layer_types,
        "linear_conv_kernel_dim": getattr(megatron_cfg, 'linear_attention_conv_kernel_dim', 4),
        "linear_key_head_dim": getattr(megatron_cfg, 'linear_attention_key_head_dim', 128),
        "linear_num_key_heads": getattr(megatron_cfg, 'linear_attention_num_key_heads', 16),
        "linear_num_value_heads": getattr(megatron_cfg, 'linear_attention_num_value_heads', 32),
        "linear_value_head_dim": getattr(megatron_cfg, 'linear_attention_value_head_dim', 128),
        "max_position_embeddings": getattr(megatron_cfg, 'max_position_embeddings', 262144),
        "mlp_only_layers": [],
        "model_type": "qwen3_5_moe_text" if is_moe else "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": megatron_cfg.num_attention_heads,
        "num_hidden_layers": megatron_cfg.num_layers,
        "num_key_value_heads": getattr(megatron_cfg, 'num_query_groups', 2),
        "rms_norm_eps": getattr(megatron_cfg, 'norm_epsilon', 1e-6),
        "use_cache": True,
        "vocab_size": vocab_size,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": rope_params,
        "tie_word_embeddings": not megatron_cfg.untie_embeddings_and_output_weights,
    }
    if is_moe:
        text_config.update({
            "moe_intermediate_size": megatron_cfg.ffn_hidden_size,
            "num_experts": megatron_cfg.num_experts,
            "num_experts_per_tok": megatron_cfg.moe_router_topk,
            "router_aux_loss_coef": 0.001,
            "shared_expert_intermediate_size": megatron_cfg.moe_shared_expert_intermediate_size,
        })
    else:
        text_config["intermediate_size"] = megatron_cfg.ffn_hidden_size

    config = {
        "architectures": ["OmniDRCA2NextForConditionalGeneration"],
        "audio_token_index": 248076,
        "model_type": "omnidrca2",
        "text_config": text_config,
        "tie_word_embeddings": not megatron_cfg.untie_embeddings_and_output_weights,
        "transformers_version": "5.2.0",
    }
    if cosyvoice_codex_state is not None:
        config["cosyvoice_codex"] = {
            "enabled": True,
            "llm_file": "talker.pt",
            "input_proj_file": "talker_input_proj.pt",
            "codec_embed_file": "talker_codec_embed.pt",
            "codec_head_file": "talker_codec_head.pt",
            "pre_matching_file": "pre_matching.pt",
            "codec_unit": cosyvoice_codex_state.get("codec_unit"),
            "codec_vocab_size": cosyvoice_codex_state.get("codec_vocab_size"),
            "llm_hidden_size": cosyvoice_codex_state.get("llm_hidden_size"),
            "llm_dtype": cosyvoice_codex_state.get("llm_dtype"),
            "llm_init_param_path": cosyvoice_codex_state.get("llm_init_param_path"),
        }
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # 保存 audio encoder
    if args.qwen3_audio_encoder:
        audio_encoder_dir = output_dir / "audio_encoder"
        audio_encoder_dir.mkdir(exist_ok=True, parents=True)
        # convert_megatron_to_qwen3 需要 decoder.final_layernorm.weight
        audio_transformer = {k: v for k, v in mcore_state.items()
                            if k.startswith('audio_encoder.layers.')}
        audio_transformer["decoder.final_layernorm.weight"] = mcore_state["audio_encoder.final_layernorm.weight"]
        qwen3_cfg, audio_encoder_sd = convert_megatron_to_qwen3(
            md.checkpoint_args, audio_transformer,
            is_audio_encoder=True, args=args
        )
        if audio_encoder_sd:
            save_hf_checkpoint(audio_encoder_dir, audio_encoder_sd, args.shard)
        if qwen3_cfg:
            qwen3_cfg.save_pretrained(audio_encoder_dir)
        conv1_path = audio_encoder_dir / "conv1.pt"
        conv2_path = audio_encoder_dir / "conv2.pt"
        conv1_sd = {'audio_encoder.conv1.weight': mcore_state['audio_encoder.conv1.weight'],
                    'audio_encoder.conv1.bias': mcore_state['audio_encoder.conv1.bias']}
        conv2_sd = {'audio_encoder.conv2.weight': mcore_state['audio_encoder.conv2.weight'],
                    'audio_encoder.conv2.bias': mcore_state['audio_encoder.conv2.bias']}
        torch.save(conv1_sd, conv1_path)
        torch.save(conv2_sd, conv2_path)
    else:
        if args.sensevoicesmall_audio_encoder:
            audio_encoder_sd = convert_megatron_to_hf_sensevoicesmall_audio_encoder(md, mcore_state)
        else:
            audio_encoder_sd = convert_megatron_to_hf_audio_encoder(md, mcore_state)
        if audio_encoder_sd:
            audio_encoder_path = output_dir / "audio_encoder.pt"
            torch.save(audio_encoder_sd, audio_encoder_path)

    # 保存 audio adaptor
    audio_adaptor_sd = convert_megatron_to_hf_audio_adaptor(md, mcore_state)
    if audio_adaptor_sd:
        audio_adaptor_path = output_dir / "audio_adaptor.pt"
        torch.save(audio_adaptor_sd, audio_adaptor_path)

    # Channels model: embedding_audio, output_layer_audio
    if getattr(md, "has_embedding_audio", False) or getattr(md, "has_output_layer_audio", False):
        if 'embedding_audio.word_embeddings.weight' in mcore_state:
            torch.save(mcore_state['embedding_audio.word_embeddings.weight'],
                      output_dir / "embedding_audio.pt")
        if 'output_layer_audio.weight' in mcore_state:
            torch.save(mcore_state['output_layer_audio.weight'],
                      output_dir / "output_layer_audio.pt")

    if cosyvoice_codex_state is not None:
        torch.save(cosyvoice_codex_state["cosyvoice_llm"], output_dir / "talker.pt")
        torch.save(
            cosyvoice_codex_state.get("cosyvoice_llm_input_proj", {}),
            output_dir / "talker_input_proj.pt",
        )
        torch.save(
            cosyvoice_codex_state["cosyvoice_codec_embed"],
            output_dir / "talker_codec_embed.pt",
        )
        torch.save(
            cosyvoice_codex_state["cosyvoice_llm_codec_head"],
            output_dir / "talker_codec_head.pt",
        )
        torch.save(
            cosyvoice_codex_state.get("cosyvoice_llm_input_proj", {}),
            output_dir / "pre_matching.pt",
        )
        torch.save(
            cosyvoice_codex_state["cosyvoice_codec_embed"],
            output_dir / "codec_embedding.pt",
        )
        torch.save(
            cosyvoice_codex_state["cosyvoice_llm_codec_head"],
            output_dir / "codec_head.pt",
        )
    if cosyvoice_fm_state is not None:
        fm_dir = output_dir / "cosyvoice_fm"
        fm_dir.mkdir(parents=True, exist_ok=True)
        torch.save(cosyvoice_fm_state, fm_dir / "model.pt")
        if not cosyvoice_fm_model_conf:
            raise RuntimeError(
                "CosyVoice FM weights were exported without model_conf; "
                "refusing to create an architecture-ambiguous checkpoint"
            )
        # JSON is valid YAML, so codex_to_wav.py and legacy yaml.safe_load
        # callers can both consume this file without another dependency.
        (fm_dir / "config.yaml").write_text(
            json.dumps(
                {
                    "model": "CosyVoiceFlowMatching",
                    "model_conf": cosyvoice_fm_model_conf,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

    # Generation config
    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        bos_token_id=151643,
        do_sample=True,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
        repetition_penalty=1.0,
        temperature=0.6,
        top_k=40,
        top_p=0.95,
    )
    gen_config.save_pretrained(output_dir)

    print("Done!")
