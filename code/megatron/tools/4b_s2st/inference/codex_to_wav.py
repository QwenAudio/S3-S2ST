#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

from s2s_key_utils import make_output_key


WAV_PATTERN = re.compile(r"<\|startofspeech\|>!(.*)<\|endofspeech\|>")


def _add_funasr_to_path(funasr_root):
    if funasr_root:
        sys.path.insert(0, funasr_root)


def _load_codec(path, codec_unit, codec_sos_token, codec_eos_token):
    if path.suffix == ".npy":
        codec = np.load(path)
    else:
        text = path.read_text().strip()
        codec = np.asarray([int(x) for x in re.split(r"[\s,]+", text) if x], dtype=np.int64)
    codec = np.asarray(codec, dtype=np.int64).reshape(-1)
    if codec_eos_token >= 0:
        eos_pos = np.flatnonzero(codec == codec_eos_token)
        if eos_pos.size > 0:
            codec = codec[: int(eos_pos[0])]
    if codec_sos_token >= 0:
        codec = codec[codec != codec_sos_token]
    if codec_eos_token >= 0:
        codec = codec[codec != codec_eos_token]
    if codec_unit > 0:
        codec = codec[(codec >= 0) & (codec < codec_unit)]
    return codec


def _load_optional_hidden(path):
    if path is None or not path.exists():
        return None
    hidden = np.load(path)
    hidden = torch.from_numpy(hidden)
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0)
    return hidden


def _save_silence_wav(wav_path, sample_rate):
    silence = torch.zeros(1, int(sample_rate), dtype=torch.float32)
    torchaudio.save(
        str(wav_path),
        silence,
        sample_rate=int(sample_rate),
        encoding="PCM_S",
        bits_per_sample=16,
    )


def _iter_codec_files(codec_dir):
    for suffix in ("*.npy", "*.txt", "*.codec"):
        yield from sorted(codec_dir.rglob(suffix))


def _resolve_checkpoint_path(path):
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No .pt checkpoint found in directory: {path}")
        return candidates[0]
    return path


def _load_model_from_config(config_path, ckpt_path, device):
    from funasr.register import tables

    import funasr.models.cosyvoice.flow_matching_model  # noqa: F401
    import funasr.models.cosyvoice.causal_bigvgan  # noqa: F401

    config = yaml.safe_load(Path(config_path).read_text())
    # Training configs keep the standalone CosyVoice FM definition under
    # model_conf.fm_conf. Accept both that form and the legacy standalone
    # {model, model_conf} config.yaml used by converted checkpoints.
    if "model" not in config:
        nested_fm = (config.get("model_conf") or {}).get("fm_conf")
        if not isinstance(nested_fm, dict) or "model" not in nested_fm:
            raise RuntimeError(
                f"No standalone model or model_conf.fm_conf in {config_path}"
            )
        config = nested_fm
    model_class = tables.model_classes.get(config["model"])
    if model_class is None:
        raise RuntimeError(f"Model class is not registered: {config['model']}")
    model_conf = config.get("model_conf", {}) or {}
    mel_feat_conf = model_conf.get("mel_feat_conf")
    if isinstance(mel_feat_conf, dict):
        mel_feat_conf.setdefault("device", str(device))
    model = model_class(**model_conf)
    state_dict = torch.load(str(ckpt_path), map_location=device)
    # Accept both the standalone state dict emitted by our converter and the
    # native CosyVoice training checkpoint container.
    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and isinstance(state_dict.get("model"), dict):
        state_dict = state_dict["model"]
    if isinstance(state_dict, dict) and any(k.startswith("fm_model.") for k in state_dict):
        state_dict = {
            k[len("fm_model."):]: v
            for k, v in state_dict.items()
            if k.startswith("fm_model.")
        }
    # x-transformers renamed this rotary buffer; use the same compatibility
    # mapping as FM initialization in pretrain_fun_base.py.
    rotary_old = "dit_model.rotary_embed.inv_freq"
    rotary_new = "dit_model.rotary_embed.freqs"
    target_keys = set(model.state_dict())
    if rotary_old in state_dict and rotary_new in target_keys and rotary_old not in target_keys:
        state_dict[rotary_new] = state_dict.pop(rotary_old)
    checkpoint_entries = len(state_dict)
    state_dict = {key: value for key, value in state_dict.items() if key in target_keys}
    ignored_checkpoint_entries = checkpoint_entries - len(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        json.dumps(
            {
                "model": config["model"],
                "checkpoint": str(ckpt_path),
                "missing": len(missing),
                "unexpected": len(unexpected),
                "ignored_checkpoint_entries": ignored_checkpoint_entries,
                "missing_preview": missing[:10],
                "unexpected_preview": unexpected[:10],
            },
            ensure_ascii=False,
        )
    )
    # Parameter mismatches indicate that weights and architecture came from
    # different FM variants.  Non-persistent/cache buffers may legitimately be
    # absent in older native checkpoints, but trainable weights may not.
    parameter_names = set(dict(model.named_parameters()))
    missing_parameters = [name for name in missing if name in parameter_names]
    if config["model"] == "CosyVoiceFlowMatching" and (missing_parameters or unexpected):
        raise RuntimeError(
            "FM checkpoint/config mismatch: "
            f"missing_parameters={missing_parameters[:20]}, "
            f"unexpected={unexpected[:20]}"
        )
    return model


def _sanitize_key(key):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def _load_jsonl_spk_sources(jsonl_path, use_oss_cache):
    if not jsonl_path:
        return {}
    spk_sources = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            messages = data["messages"]
            user_content = messages[1].get("content", messages[1].get("text"))
            if user_content is None:
                raise ValueError("user message contains neither 'content' nor 'text'")
            wav_match = WAV_PATTERN.search(user_content)
            if wav_match is None:
                raise ValueError("user message does not contain a !<wav-path> speech span")
            wav_path = wav_match.group(1)
            try:
                duration = (float(messages[1]["start_time"]), float(messages[1]["end_time"]))
            except Exception:
                duration = (0, None)

            key = wav_path.replace(" ", "")
            if duration[1] is not None:
                key = f"{wav_path}_{duration[0]}_{duration[1]}"

            if use_oss_cache:
                wav_path = wav_path.replace("oss://example-storage-4", "/nls_spider_sh_cache", 1)
                wav_path = wav_path.replace("oss://example-storage-1", "/fun_resource_sh_cache", 1)
            spk_sources[make_output_key(key)] = (wav_path, duration)
    return spk_sources


def _load_spk_embedding(key, spk_sources, embedding_extractor, load_audio_fn, device):
    if embedding_extractor is None or key not in spk_sources:
        return None
    wav_path, duration = spk_sources[key]
    speech = load_audio_fn(wav_path, fs=16000, duration=duration, use_ffmpeg=False)
    if isinstance(speech, np.ndarray):
        speech = torch.from_numpy(speech)
    if speech.ndim == 1:
        speech = speech.unsqueeze(0)
    elif speech.ndim > 2:
        speech = speech.reshape(1, -1)
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    spk_emb = embedding_extractor.extract_embedding(speech.cpu()).view(1, -1)
    return spk_emb


def main():
    parser = argparse.ArgumentParser(description="Convert inferred codex/codec tokens to wav with CosyVoice FM+Vocoder.")
    parser.add_argument("--codec-dir", required=True, help="Directory containing inferred codec .npy/.txt files.")
    parser.add_argument("--output-dir", required=True, help="Output directory. Wavs are written to <output-dir>/wav.")
    parser.add_argument("--fm-ckpt-path", required=True, help="CosyVoice FM checkpoint path, e.g. .../model.pt")
    parser.add_argument("--voc-ckpt-path", required=True, help="CosyVoice vocoder checkpoint path, e.g. .../model.pt")
    parser.add_argument("--jsonl", default="", help="Input jsonl used by inference. Used to load wav and extract speaker embedding.")
    parser.add_argument("--use-oss-cache", action="store_true")
    parser.add_argument("--disable-spk-embedding", action="store_true")
    parser.add_argument("--funasr-root", default=os.environ.get("FUNASR_ROOT", ""))
    parser.add_argument("--device", default=os.environ.get("CODEX_TO_WAV_DEVICE", "cuda:0"))
    parser.add_argument("--codec-unit", type=int, default=int(os.environ.get("CODEX_CODEC_UNIT", "59049")))
    parser.add_argument("--codec-sos-token", type=int, default=int(os.environ.get("CODEX_CODEC_SOS_TOKEN_ID", "59049")))
    parser.add_argument("--codec-eos-token", type=int, default=int(os.environ.get("CODEX_CODEC_EOS_TOKEN_ID", "59050")))
    parser.add_argument("--fm-dtype", default=os.environ.get("CODEX_FM_DTYPE", "fp16"), choices=["fp32", "bf16", "fp16"])
    parser.add_argument(
        "--fm-config-path",
        default=os.environ.get(
            "CODEX_FM_CONFIG_PATH",
            os.environ.get("FUN_DATASET_CONFIG_PATH", ""),
        ),
    )
    parser.add_argument("--voc-dtype", default=os.environ.get("CODEX_VOC_DTYPE", "fp32"), choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--llm-hidden-dir", default=os.environ.get("CODEX_LLM_HIDDEN_DIR", ""))
    parser.add_argument("--prompt-llm-hidden-dir", default=os.environ.get("CODEX_PROMPT_LLM_HIDDEN_DIR", ""))
    parser.add_argument("--chunk-size", type=int, default=int(os.environ.get("CODEX_FM_CHUNK_SIZE", "-1")))
    parser.add_argument("--infer-causal-mask-type", type=int, default=int(os.environ.get("CODEX_INFER_CAUSAL_MASK_TYPE", "0")))
    parser.add_argument("--max-codec-len", type=int, default=int(os.environ.get("CODEX_MAX_CODEC_LEN", "3750")))
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split codec files deterministically across decoder processes.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based decoder shard index.")
    parser.add_argument("--summary-path", default="",
                        help="Optional per-worker summary path; avoids parallel writers clobbering one file.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num-shards >= 1 and 0 <= shard-index < num-shards")

    codec_dir = Path(args.codec_dir)
    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wav"
    feat_dir = output_dir / "feat"
    wav_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    if not codec_dir.is_dir():
        raise FileNotFoundError(f"codec-dir not found: {codec_dir}")

    _add_funasr_to_path(args.funasr_root)
    from funasr.datasets.cosyvoice_dataset.datasets import EmbeddingExtraction
    from funasr.utils.load_utils import load_audio_text_image_video
    from funasr.models.cosyvoice.utils import dtype_map

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spk_sources = _load_jsonl_spk_sources(args.jsonl, args.use_oss_cache)
    embedding_extractor = None if args.disable_spk_embedding else EmbeddingExtraction()

    fm_ckpt_path = _resolve_checkpoint_path(args.fm_ckpt_path)
    fm_config_path = (
        Path(args.fm_config_path)
        if args.fm_config_path
        else fm_ckpt_path.parent / "config.yaml"
    )
    if not fm_config_path.exists():
        raise FileNotFoundError(f"FM config.yaml not found next to checkpoint: {fm_config_path}")
    fm_model = _load_model_from_config(fm_config_path, fm_ckpt_path, device)
    fm_model.to(device=device, dtype=dtype_map[args.fm_dtype])
    fm_model.eval()
    fm_param = next(fm_model.parameters())
    fm_device = fm_param.device
    fm_dtype = fm_param.dtype

    voc_ckpt_path = _resolve_checkpoint_path(args.voc_ckpt_path)
    voc_config_path = voc_ckpt_path.parent / "config.yaml"
    if not voc_config_path.exists():
        raise FileNotFoundError(f"Vocoder config.yaml not found next to checkpoint: {voc_config_path}")
    voc_model = _load_model_from_config(voc_config_path, voc_ckpt_path, device)
    voc_model.to(device=device, dtype=dtype_map[args.voc_dtype])
    voc_model.eval()
    sample_rate = int(getattr(voc_model, "sample_rate", getattr(fm_model, "sample_rate", 24000)))

    llm_hidden_dir = Path(args.llm_hidden_dir) if args.llm_hidden_dir else None
    prompt_llm_hidden_dir = Path(args.prompt_llm_hidden_dir) if args.prompt_llm_hidden_dir else None

    all_codec_files = list(_iter_codec_files(codec_dir))
    codec_files = all_codec_files[args.shard_index::args.num_shards]
    summary = {
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "codec_candidates_total": len(all_codec_files),
        "codec_candidates_assigned": len(codec_files),
        "converted": 0,
        "silence_fallback": 0,
        "skipped": 0,
        "skipped_invalid": [],
        "failed": [],
    }
    with torch.no_grad():
        for codec_path in codec_files:
            key = codec_path.stem
            wav_path = wav_dir / f"{key}.wav"
            feat_path = feat_dir / f"{key}.npy"
            if wav_path.exists() and not args.overwrite:
                summary["skipped"] += 1
                continue

            try:
                codec_np = _load_codec(
                    codec_path,
                    args.codec_unit,
                    args.codec_sos_token,
                    args.codec_eos_token,
                )
                if codec_np.size == 0:
                    _save_silence_wav(wav_path, sample_rate)
                    summary["converted"] += 1
                    summary["silence_fallback"] += 1
                    summary.setdefault("silence_fallback_files", []).append(
                        {
                            "codec": str(codec_path),
                            "wav": str(wav_path),
                            "reason": "empty_codec_after_filtering",
                            "duration_sec": 1.0,
                            "sample_rate": sample_rate,
                        }
                    )
                    continue
                if args.max_codec_len > 0 and codec_np.size > args.max_codec_len:
                    summary["skipped_invalid"].append(
                        {
                            "codec": str(codec_path),
                            "reason": "codec_too_long",
                            "codec_len": int(codec_np.size),
                            "max_codec_len": args.max_codec_len,
                        }
                    )
                    continue
                codec = torch.from_numpy(codec_np).long().unsqueeze(0).to(device)
                codec_lengths = torch.tensor([codec.shape[1]], dtype=torch.int64, device=device)
                spk_emb = _load_spk_embedding(
                    key,
                    spk_sources,
                    embedding_extractor,
                    load_audio_text_image_video,
                    device,
                )

                infer_kwargs = {
                    "fm_dtype": args.fm_dtype,
                    "infer_causal_mask_type": args.infer_causal_mask_type,
                }
                if spk_emb is not None:
                    infer_kwargs["xvec"] = spk_emb.to(
                        device=fm_device, dtype=fm_dtype
                    )
                    infer_kwargs["xvec_lengths"] = torch.tensor([1], dtype=torch.int64, device=device)
                if bool(getattr(fm_model, "unified", False)):
                    llm_hidden = _load_optional_hidden(llm_hidden_dir / f"{key}.npy" if llm_hidden_dir else None)
                    if llm_hidden is None:
                        raise RuntimeError(
                            "FM model is unified=True, but no llm_hidden was provided. "
                            "Set CODEX_LLM_HIDDEN_DIR or use a non-unified token2wav FM checkpoint."
                        )
                    prompt_llm_hidden = _load_optional_hidden(
                        prompt_llm_hidden_dir / f"{key}.npy" if prompt_llm_hidden_dir else None
                    )
                    if prompt_llm_hidden is None:
                        prompt_llm_hidden = llm_hidden[:, :0, :]
                    infer_kwargs["llm_hidden"] = llm_hidden.to(
                        device=fm_device, dtype=fm_dtype
                    )
                    infer_kwargs["prompt_llm_hidden"] = prompt_llm_hidden.to(
                        device=fm_device, dtype=fm_dtype
                    )

                feat = fm_model._inference(
                    codec=codec,
                    codec_lengths=codec_lengths,
                    chunk_size=args.chunk_size,
                    finalize=True,
                    **infer_kwargs,
                ).float()
                wav = voc_model.inference(
                    [feat[0]],
                    None,
                    [key],
                    device=str(device),
                    voc_dtype=args.voc_dtype,
                    finalize=True,
                )
                np.save(feat_path, feat[0].detach().cpu().numpy())
                torchaudio.save(
                    str(wav_path),
                    wav.detach().cpu(),
                    sample_rate=sample_rate,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
                summary["converted"] += 1
            except Exception as exc:
                summary["failed"].append(
                    {
                        "codec": str(codec_path),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    summary_path = Path(args.summary_path) if args.summary_path else output_dir / "codex_to_wav.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    if summary["failed"]:
        raise SystemExit(f"codex_to_wav failed for {len(summary['failed'])} file(s); see {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
