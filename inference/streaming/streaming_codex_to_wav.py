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
        for path in sorted(codec_dir.rglob(suffix)):
            # Inference stores a parallel ``*.control.npy`` audit stream with
            # injected BOS/EOS tokens.  It is not an acoustic codec sequence
            # and must never be sent to FM/vocoder or counted as a decode
            # failure.
            if path.name.endswith(".control.npy"):
                continue
            yield path


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

    config = yaml.safe_load(Path(config_path).read_text().replace('${S3_ROOT}', str(next(p for p in Path(__file__).resolve().parents if (p / 'env.sh').is_file() and (p / 'tools').is_dir()))))
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


def _load_streaming_segments(metadata_jsonl):
    """Return codec WRITE lengths keyed by the emitted codec filename stem."""
    if not metadata_jsonl:
        return {}
    result = {}
    with open(metadata_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            codec_path = row.get("codec_path")
            segments = row.get("codec_segments") or []
            if not codec_path:
                continue
            raw_lengths = [int(segment["codec_len"]) for segment in segments]
            if any(length < 0 for length in raw_lengths):
                raise ValueError(
                    f"invalid codec_segments for {codec_path}: {segments!r}"
                )
            # A WRITE may immediately emit codec EOS and therefore contribute
            # zero acoustic tokens.  It is a valid no-op for FM/Vocoder state.
            lengths = [length for length in raw_lengths if length > 0]
            result[Path(codec_path).stem] = lengths
    return result


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


def _decode_one_envelope(codec, segment_lengths, fm_model, voc_model, args, device, infer_kwargs, summary):
    # Replay Talker WRITE commits incrementally.  FM deliberately
    # recomputes the committed prefix with a causal attention mask;
    # only stable, previously un-emitted mel frames advance the
    # cursor.  BigVGAN carries its convolution cache across calls.
    feed_boundaries = []
    committed = 0
    for segment_length in segment_lengths:
        remaining = segment_length
        while remaining > 0:
            take = min(remaining, args.streaming_feed_codec_tokens)
            committed += take
            remaining -= take
            feed_boundaries.append(committed)
    if not feed_boundaries or feed_boundaries[-1] != codec.shape[1]:
        raise RuntimeError(f"invalid streaming feed boundaries for {key}")

    # A partial attention block is not stable: its earlier mel
    # frames attend to later codec in that same block when it arrives.
    # Schedule complete blocks as soon as FM lookahead is available.
    fm_ratio = int(fm_model.feat_token_ratio)
    fm_chunk_frames = int(args.chunk_size) * fm_ratio
    assert fm_ratio > 0 and fm_chunk_frames > 0
    feed_boundaries = sorted(set(feed_boundaries) | set(range(
        int(fm_model.context_size) + int(args.chunk_size),
        int(codec.shape[1]), int(args.chunk_size))))
    mel_cursor = 0
    vocoder_cache = {}
    wav_chunks = []
    final_feat = None
    vocoder_lookahead = int(
        getattr(voc_model.generator.h, "conv_pre_look_right", 0)
    )
    for call_index, boundary in enumerate(feed_boundaries):
        finalize = call_index == len(feed_boundaries) - 1
        prefix = codec[:, :boundary]
        # Non-final FM calls reserve lookahead_length codec tokens;
        # skip prefixes that cannot yet produce a stable frame.
        if not finalize and boundary <= int(fm_model.context_size):
            continue
        if not finalize:
            complete_frames = ((boundary - int(fm_model.context_size))
                               // int(args.chunk_size)) * fm_chunk_frames
            if complete_frames - vocoder_lookahead <= mel_cursor:
                continue
        prefix_lengths = torch.tensor(
            [boundary], dtype=torch.int64, device=device
        )
        feat = fm_model._inference(
            codec=prefix,
            codec_lengths=prefix_lengths,
            chunk_size=args.chunk_size,
            finalize=finalize,
            **infer_kwargs,
        ).float()
        # Keep incomplete rightmost blocks away from the vocoder,
        # including its lookahead/cache. Final source drain makes
        # the utterance's last partial block definitive.
        fm_stable_end = (int(feat.shape[1]) if finalize else
                         int(feat.shape[1]) // fm_chunk_frames * fm_chunk_frames)
        feat = feat[:, :fm_stable_end, :]
        final_feat = feat
        summary["streaming_fm_calls"] += 1
        stable_end = (
            int(feat.shape[1])
            if finalize
            else max(mel_cursor, int(feat.shape[1]) - vocoder_lookahead)
        )
        if stable_end <= mel_cursor:
            continue
        # Include the right-lookahead mel frames in the input while
        # advancing the committed cursor only through stable_end.
        mel_chunk = feat[:, mel_cursor:, :]
        wav_chunk, vocoder_cache = voc_model.generator.inference_cache(
            mel_chunk.transpose(1, 2),
            finalize=finalize,
            cache=vocoder_cache,
        )
        wav_chunks.append(wav_chunk.float())
        mel_cursor = stable_end
        summary["streaming_vocoder_calls"] += 1

    if final_feat is None or not wav_chunks:
        raise RuntimeError(f"streaming FM/Vocoder emitted no audio for {key}")
    wav = torch.cat(wav_chunks, dim=-1)
    return wav, final_feat


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
    parser.add_argument(
        "--streaming-metadata-jsonl",
        default=os.environ.get("CODEX_STREAMING_METADATA_JSONL", ""),
        help="Streaming inference JSONL containing codec_segments for each utterance.",
    )
    parser.add_argument(
        "--streaming-feed-codec-tokens",
        type=int,
        default=int(os.environ.get("CODEX_STREAMING_FEED_CODEC_TOKENS", "15")),
        help="Maximum newly committed codec tokens supplied per incremental FM call.",
    )
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
    streaming_segments = _load_streaming_segments(args.streaming_metadata_jsonl)
    if not streaming_segments:
        raise ValueError(
            "true-streaming codec decode requires --streaming-metadata-jsonl "
            "with non-empty codec_segments"
        )
    if args.chunk_size <= 0:
        raise ValueError("true-streaming codec decode requires --chunk-size > 0")
    if args.streaming_feed_codec_tokens <= 0:
        raise ValueError("--streaming-feed-codec-tokens must be > 0")
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
        "streaming": True,
        "fm_stability_contract": "independent_exact_codec_envelopes_stable_fm_v1",
        "fm_chunk_size": args.chunk_size,
        "streaming_feed_codec_tokens": args.streaming_feed_codec_tokens,
        "streaming_fm_calls": 0,
        "streaming_vocoder_calls": 0,
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
                raw_codec_np = (
                    np.asarray(np.load(codec_path), dtype=np.int64).reshape(-1)
                    if codec_path.suffix == ".npy"
                    else None
                )
                if codec_np.size == 0:
                    raise RuntimeError("empty_audio: no acoustic codec; no synthetic silence is permitted")
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
                segment_lengths = streaming_segments.get(key)
                if segment_lengths is None:
                    raise RuntimeError(f"missing codec_segments metadata for {key}")
                if sum(segment_lengths) != int(codec.shape[1]):
                    # The forced final WRITE path can preserve one returned BOS
                    # in the saved envelope and count it in the last segment's
                    # codec_len.  _load_codec correctly removes BOS before FM;
                    # reconcile only an exactly matching number of observed BOS
                    # tokens, never an unexplained length discrepancy.
                    excess = sum(segment_lengths) - int(codec.shape[1])
                    observed_bos = (
                        int(np.sum(raw_codec_np == args.codec_sos_token))
                        if raw_codec_np is not None and args.codec_sos_token >= 0
                        else 0
                    )
                    if excess > 0 and excess == observed_bos:
                        segment_lengths = list(segment_lengths)
                        for index in range(len(segment_lengths) - 1, -1, -1):
                            take = min(excess, segment_lengths[index])
                            segment_lengths[index] -= take
                            excess -= take
                            if excess == 0:
                                break
                        segment_lengths = [x for x in segment_lengths if x > 0]
                    if sum(segment_lengths) != int(codec.shape[1]):
                        raise RuntimeError(
                            f"codec segment length mismatch for {key}: "
                            f"segments={segment_lengths} sum={sum(segment_lengths)} "
                            f"codec={codec.shape[1]} observed_bos={observed_bos}"
                        )
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

                assert not bool(getattr(fm_model, "unified", False)), "Envelope probe supports token-only FM"
                envelope_waves, envelope_feats = [], []
                codec_offset = 0
                for segment_length in segment_lengths:
                    envelope = codec[:, codec_offset:codec_offset + segment_length]
                    chunk_wav, chunk_feat = _decode_one_envelope(
                        envelope, [segment_length], fm_model, voc_model, args,
                        device, infer_kwargs, summary)
                    envelope_waves.append(chunk_wav)
                    envelope_feats.append(chunk_feat)
                    codec_offset += segment_length
                assert codec_offset == int(codec.shape[1])
                wav = torch.cat(envelope_waves, dim=-1)
                final_feat = torch.cat(envelope_feats, dim=1)
                np.save(feat_path, final_feat[0].detach().cpu().numpy())
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
