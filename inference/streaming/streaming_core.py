#!/usr/bin/env python3
"""Closed-loop S2TT evaluation for the 640-ms Thinker READ/WRITE gate.

In true-streaming mode the waveform is read once but exposed to the speech
frontend as growing 640-ms prefixes.  Each frontend call uses reference
block-causal encoder and only newly available adaptor embeddings (normally eight
80-ms tokens) are appended to the Thinker.  This recomputes historical encoder
states because the current encoder has no inference cache, but never exposes
future audio.  WRITE opens a normal assistant turn and normal ``<|im_end|>``
returns control to audio reading.  End of audio forces the final WRITE.
"""

from __future__ import annotations
_S3_ROOT = next(p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'env.sh').is_file() and (p / 'tools').is_dir())

import argparse
import asyncio
import hashlib
import json
import os
import queue
import re
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, WhisperFeatureExtractor


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
OVERRIDES = SCRIPT_DIR / "vllm_overrides"
sys.path.insert(0, str(REPO))

import vllm.model_executor.models as _vllm_models

if str(OVERRIDES) not in _vllm_models.__path__:
    _vllm_models.__path__.insert(0, str(OVERRIDES))
_vllm_models.ModelRegistry.register_model(
    "Qwen3_5ForCausalLM",
    "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM",
)

from vllm.config import CompilationConfig
from vllm.engine.omnidrca2_engine import OmniDRCA2Engine
from vllm.engine.shared_hidden_buffer import SharedHiddenBuffer
from vllm.inputs import TextPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils.load_audio_utils import load_audio_text_image_video
from vllm.v1.engine.async_llm import AsyncEngineArgs
from vllm.transformers_utils.config import get_config
from vllm.model_executor.models.omnidrca2_next_processor import OmniDRCA2NextProcessor
from vllm.model_executor.models.whisper import WhisperFrontend as VllmWhisperFrontend
from vllm.model_executor.models.omnidrca2_next_thinker import (
    SenseVoiceEncoder,
    Transformer,
    audio_adaptor_conf,
    encoder_conf,
    load_pretrained_module,
)
from true_streaming_encoder import (
    enable_true_streaming_adaptor,
    enable_true_streaming_encoder,
)


SPEECH_RE = re.compile(r"<\|startofspeech\|>!?(.*?)<\|endofspeech\|>")
SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
# Mean of 7,756 decodable segments sampled deterministically across all 512
# shuffled training shards.  This defines one absolute noise floor at
# FIXED_LOG_MEL_MAX - 8 instead of using future utterance audio.
FIXED_LOG_MEL_MAX = 1.3904691073
STREAMING_LOOKAHEAD_MS = 2.5
MEL_FILTERS_PATH = ("__S3_ROOT__/models/frontend/mel_filters.npz".replace('__S3_ROOT__', str(_S3_ROOT)))


class FixedMaxWhisperFrontend(VllmWhisperFrontend):
    """Whisper frontend with the same causal-safe fixed clipping floor as training."""

    def log_mel_spectrogram(
        self, audio: torch.Tensor, ilens: torch.Tensor | None = None
    ):
        window = torch.hann_window(self.win_length, device=audio.device)
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, window=window, return_complex=True
        )

        # Keep Whisper's final-frame deletion.  A 2.5-ms waveform lookahead at
        # each non-final chunk boundary makes the last retained centered frame
        # identical to full-record extraction.
        magnitudes = stft[..., :-1].abs().square()
        if self.filters_path is not None:
            filters = self.mel_filters(audio.device, self.n_mels, self.filters_path)
        else:
            filters = self.mel_filters(audio.device, self.n_mels)
        log_spec = torch.clamp(filters @ magnitudes, min=1e-10).log10()
        olens = ilens // self.hop_length if ilens is not None else None
        max_value = log_spec.new_full((audio.size(0), 1, 1), FIXED_LOG_MEL_MAX)
        log_spec = torch.maximum(log_spec, max_value - 8.0)
        return (log_spec + 4.0) / 4.0, olens


class ReadWriteGate(nn.Module):
    def __init__(self, hidden_size: int = 2560, bottleneck: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, 2),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class FullRecordEncoder:
    def __init__(self, model_dir: Path, device: str, true_streaming: bool = False,
                 chunk_ms: float = 640.0):
        self.device = torch.device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        extractor_path = Path(os.environ.get(
            "STREAMING_FEATURE_EXTRACTOR",
            ('__S3_ROOT__/models/offline'.replace('__S3_ROOT__', str(_S3_ROOT))),
        ))
        extractor = WhisperFeatureExtractor.from_pretrained(extractor_path)
        self.processor = OmniDRCA2NextProcessor(
            feature_extractor=extractor, tokenizer=tokenizer
        )
        # OmniDRCA2NextProcessor otherwise lazily constructs vLLM's original
        # utterance-max frontend.  Install the calibrated implementation before
        # the first processor call.
        self.processor._frontend = FixedMaxWhisperFrontend(
            fs=16000,
            n_mels=128,
            do_pad_trim=False,
            filters_path=MEL_FILTERS_PATH,
        )
        self.encoder = SenseVoiceEncoder(**encoder_conf).float().eval().to(self.device)
        cfg = get_config(str(model_dir), trust_remote_code=True)
        dim = int(cfg.text_config.hidden_size)
        self.adaptor = Transformer(
            **{**audio_adaptor_conf, "encoder_dim": encoder_conf["linear_units"], "llm_dim": dim}
        ).float().eval().to(self.device)
        load_pretrained_module(
            str(model_dir / "audio_encoder.pt"), self.encoder,
            remove_prefix="audio_encoder."
        )
        load_pretrained_module(
            str(model_dir / "audio_adaptor.pt"), self.adaptor,
            remove_prefix="audio_adaptor."
        )
        if true_streaming:
            enable_true_streaming_encoder(self.encoder, chunk_ms=chunk_ms)
            enable_true_streaming_adaptor(
                self.adaptor, self.encoder.streaming_chunk_frames
            )
            print(
                f"[TRUE STREAMING FRONTEND] block-causal chunk={chunk_ms:g} ms ",
                f"({self.encoder.streaming_chunk_frames} encoder frames, ",
                f"{self.adaptor.streaming_chunk_size} adaptor tokens)",
                flush=True,
            )
        self.true_streaming = true_streaming

    @torch.inference_mode()
    def __call__(self, wav: np.ndarray) -> torch.Tensor:
        item = self.processor(text="", audio=wav)
        feats = item["input_features"].to(self.device, dtype=torch.float32)
        lengths = item["audio_feature_lengths"].to(self.device)
        encoded, encoded_lengths = self.encoder(feats.permute(0, 2, 1), lengths)
        embeds, output_lengths = self.adaptor(encoded, encoded_lengths)
        length = int(output_lengths[0].item())
        return embeds[0, :length].to(dtype=torch.bfloat16, device="cpu").contiguous()

    @torch.inference_mode()
    def stream_prefixes(
        self, wav: np.ndarray, chunk_samples: int, lookahead_samples: int
    ):
        """Yield new embeddings after each closed chunk plus STFT lookahead."""
        if not self.true_streaming:
            raise RuntimeError("stream_prefixes requires --true-streaming-encoder")
        if chunk_samples <= 0:
            raise ValueError(f"chunk_samples must be positive: {chunk_samples}")
        if lookahead_samples < 0:
            raise ValueError(f"lookahead_samples must be non-negative: {lookahead_samples}")
        emitted = 0
        feature_prefix_samples = chunk_samples
        while True:
            if feature_prefix_samples >= len(wav):
                feature_prefix_samples = len(wav)
                available_prefix_samples = len(wav)
                end_of_audio = True
            else:
                available_prefix_samples = min(
                    feature_prefix_samples + lookahead_samples, len(wav)
                )
                end_of_audio = available_prefix_samples == len(wav)

            prefix = np.ascontiguousarray(wav[:available_prefix_samples])
            prefix_embeds = self(prefix)
            available = len(prefix_embeds)
            if available < emitted:
                raise RuntimeError(
                    f"streaming frontend length regressed: {available} < {emitted}"
                )
            new_embeds = prefix_embeds[emitted:available].contiguous()
            yield (
                feature_prefix_samples,
                available_prefix_samples,
                new_embeds,
                available,
                end_of_audio,
            )
            emitted = available
            if end_of_audio:
                break
            feature_prefix_samples += chunk_samples


class ThinkerBackend:
    def __init__(self, args):
        self.args = args
        self.model_dir = Path(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        self.config = get_config(str(self.model_dir), trust_remote_code=True)
        self.audio_token_id = int(self.config.audio_token_index)
        self.im_end_id = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        per_engine_util = min(args.gpu_memory_utilization, 0.44)
        common_engine_args = dict(
            model=str(self.model_dir), trust_remote_code=True,
            dtype="bfloat16", tensor_parallel_size=len(args.devices),
            gpu_memory_utilization=per_engine_util,
            distributed_executor_backend="mp", enforce_eager=True,
            enable_prefix_caching=False, enable_chunked_prefill=False,
            max_model_len=args.max_model_len, max_num_seqs=1, block_size=16,
            limit_mm_per_prompt={"audio": 64},
            enable_mm_embeds=True,
            skip_mm_profiling=True,
            enable_hidden_state_caching=False,
        )
        self.gate_engine = None
        if not args.offline_thinker:
            os.environ["VLLM_STREAM_GATE_PATH"] = str(Path(args.gate).resolve())
            os.environ["VLLM_STREAM_GATE_THRESHOLD"] = str(args.threshold)
            self.gate_engine = OmniDRCA2Engine(
                AsyncEngineArgs(**common_engine_args),
                thinker_visible_devices=args.devices,
            )
            os.environ.pop("VLLM_STREAM_GATE_PATH", None)
            os.environ.pop("VLLM_STREAM_GATE_THRESHOLD", None)
        self.text_engine = OmniDRCA2Engine(
            AsyncEngineArgs(**common_engine_args),
            thinker_visible_devices=args.devices,
        )

    def _request(self, engine, prompt: TextPrompt, params: SamplingParams):
        request_id = f"stream_{uuid.uuid4().hex}"
        output_queue = engine.add_request(request_id, prompt, params)
        last = None
        while True:
            output = output_queue.get(timeout=600)
            if output is None:
                break
            if isinstance(output, RequestOutput):
                last = output
        if last is None:
            raise RuntimeError(f"no RequestOutput for {request_id}")
        return last

    @staticmethod
    def _mm(segments: list[torch.Tensor]):
        # DictEmbeddingItems supports a list of variable-length [T,H] tensors;
        # avoiding padded tensors also avoids treating a scalar length field as
        # a separate embedding item in vLLM's multimodal hasher.
        return {"audio_embeds": [x.contiguous() for x in segments]}

    def gate_decision(self, prompt_text: str, segments: list[torch.Tensor]):
        prompt = TextPrompt(
            prompt=prompt_text,
            multi_modal_data={"audio": self._mm(segments)},
            mm_processor_kwargs={},
        )
        params = SamplingParams(
            temperature=0.0, top_k=1, max_tokens=1,
            stop_token_ids=[self.im_end_id], detokenize=False,
        )
        output = self._request(self.gate_engine, prompt, params)
        token_ids = list(output.outputs[0].token_ids or [])
        if not token_ids or token_ids[0] not in (0, 1):
            raise RuntimeError(f"gate returned invalid class token(s): {token_ids}")
        return token_ids[0] == 1

    def generate(self, prompt_text: str, segments: list[torch.Tensor]):
        prompt = TextPrompt(
            prompt=prompt_text,
            multi_modal_data={"audio": self._mm(segments)},
            mm_processor_kwargs={},
        )
        params = SamplingParams(
            temperature=0.0, top_k=1, max_tokens=self.args.max_new_tokens,
            stop_token_ids=[self.im_end_id], detokenize=True,
        )
        output = self._request(self.text_engine, prompt, params)
        return str(output.outputs[0].text or "").replace("<|im_end|>", "").strip()

    def close(self):
        if self.gate_engine is not None:
            self.gate_engine.shutdown()
        self.text_engine.shutdown()


def make_key(wav_path: str, index: int) -> str:
    base = Path(wav_path).name
    digest = hashlib.sha1(f"{index}:{wav_path}".encode()).hexdigest()[:16]
    return f"{base}_{digest}"


def load_row(obj: dict):
    messages = obj["messages"]
    user = messages[1]
    content = user.get("content", user.get("text", ""))
    match = SPEECH_RE.search(content)
    if not match:
        raise ValueError("missing speech path")
    path = match.group(1).replace(" ", "")
    path = path.replace(
        "oss://example-storage-1/", "/fun_resource_sh_cache/", 1
    ).replace(
        "oss://example-storage-4/", "/nls_spider_sh_cache/", 1
    )
    task_prompt = content.split("<|startofspeech|>", 1)[0]
    reference = messages[2].get("content", messages[2].get("text", ""))
    source = obj.get("transcript", "")
    return path, task_prompt, source, reference.replace("\n", " ")


def build_prompt(committed: str, task_prompt: str, pending: torch.Tensor,
                 committed_segments: list[torch.Tensor], open_for_write: bool = True):
    first = not committed_segments
    user_prefix = task_prompt if first else ""
    audio_tail = "<|audio_end|><|im_end|>\n" + ASSISTANT if open_for_write else ""
    text = committed + f"<|im_start|>user\n{user_prefix}<|audio_start|><|audio_pad|>" + audio_tail
    return text, committed_segments + [pending]


def _finish_result(index, wav_path, source, reference, pieces, first_write,
                   total_tokens, decisions):
    prediction = "".join(pieces) if reference and any("\u4e00" <= c <= "\u9fff" for c in reference) else " ".join(pieces)
    return {
        "index": index, "key": make_key(wav_path, index),
        "wav": wav_path, "source": source, "reference": reference,
        "prediction": prediction.strip(), "pieces": pieces,
        "first_write_audio_tokens": first_write,
        "total_audio_tokens": total_tokens,
        "first_write_ratio": (first_write / total_tokens) if first_write and total_tokens else None,
        "decisions": decisions,
    }


def evaluate_one_true_streaming(index, obj, encoder, backend, args):
    """Run one sample without ever giving the frontend future waveform samples."""
    wav_path, task_prompt, source, reference = load_row(obj)
    wav = np.asarray(load_audio_text_image_video(wav_path, use_ffmpeg=False))
    chunk_samples = round(args.encoder_chunk_ms * args.audio_sample_rate / 1000.0)
    lookahead_samples = round(STREAMING_LOOKAHEAD_MS * args.audio_sample_rate / 1000.0)
    committed = SYSTEM
    committed_segments = []
    pending_parts = []
    pieces = []
    decisions = []
    first_write = None
    total_tokens = 0

    for feature_prefix_samples, available_prefix_samples, new_embeds, consumed, end_of_audio in encoder.stream_prefixes(
        wav, chunk_samples, lookahead_samples
    ):
        if new_embeds.numel():
            pending_parts.append(new_embeds)

        # The final waveform prefix can be shorter than the frontend's output
        # stride.  In that case the encoder/adaptor legitimately produces no
        # new Thinker token.  Do not create an empty <audio_pad> segment: vLLM
        # cannot parse a zero-length DictEmbeddingItems entry, and no new audio
        # information needs another gate/text invocation anyway.
        if not pending_parts:
            decisions.append({
                "audio_prefix_samples": available_prefix_samples,
                "audio_prefix_ms": 1000.0 * available_prefix_samples / args.audio_sample_rate,
                "feature_prefix_samples": feature_prefix_samples,
                "feature_prefix_ms": 1000.0 * feature_prefix_samples / args.audio_sample_rate,
                "new_audio_tokens": 0,
                "consumed_audio_tokens": consumed,
                "p_write": 0.0,
                "logits": [],
                "decision": "END_NO_NEW_AUDIO" if end_of_audio else "READ_NO_NEW_AUDIO",
                "forced_eos": end_of_audio,
                "skipped_empty_audio_segment": True,
            })
            total_tokens = consumed
            if end_of_audio:
                break
            continue

        pending = torch.cat(pending_parts, dim=0) if pending_parts else new_embeds
        total_tokens = consumed
        # A 640-ms block normally adds eight 80-ms Thinker tokens.  Keep the
        # measured count in the trace so boundary/padding behavior is auditable.
        if end_of_audio:
            write, forced = True, True
            p_write, logits = 1.0, [float("nan"), float("nan")]
        else:
            gate_prompt, segments = build_prompt(
                committed, task_prompt, pending, committed_segments,
                open_for_write=False,
            )
            write, forced = backend.gate_decision(gate_prompt, segments), False
            p_write, logits = (1.0 if write else 0.0), []
        decisions.append({
            "audio_prefix_samples": available_prefix_samples,
            "audio_prefix_ms": 1000.0 * available_prefix_samples / args.audio_sample_rate,
            "feature_prefix_samples": feature_prefix_samples,
            "feature_prefix_ms": 1000.0 * feature_prefix_samples / args.audio_sample_rate,
            "new_audio_tokens": len(new_embeds),
            "consumed_audio_tokens": consumed,
            "p_write": p_write, "logits": logits,
            "decision": "WRITE" if write else "READ",
            "forced_eos": forced,
            "end_of_audio": end_of_audio,
        })
        if not write:
            continue
        if first_write is None:
            first_write = consumed
        prompt_text, segments = build_prompt(
            committed, task_prompt, pending, committed_segments,
            open_for_write=True,
        )
        if hasattr(backend, "set_write_context"):
            backend.set_write_context(final_write=end_of_audio)
        text = backend.generate(prompt_text, segments)
        pieces.append(text)
        committed = prompt_text + text + "<|im_end|>\n"
        committed_segments = segments
        pending_parts = []

    return _finish_result(
        index, wav_path, source, reference, pieces, first_write,
        total_tokens, decisions,
    )


def evaluate_one_offline_thinker(index, obj, encoder, backend, args):
    """Stream the frontend, then make one final offline Thinker call."""
    wav_path, task_prompt, source, reference = load_row(obj)
    wav = np.asarray(load_audio_text_image_video(wav_path, use_ffmpeg=False))
    decisions = []
    if encoder.true_streaming:
        chunk_samples = round(args.encoder_chunk_ms * args.audio_sample_rate / 1000.0)
        lookahead_samples = round(STREAMING_LOOKAHEAD_MS * args.audio_sample_rate / 1000.0)
        parts = []
        for feature_prefix, available_prefix, new_embeds, consumed, end_of_audio in encoder.stream_prefixes(
            wav, chunk_samples, lookahead_samples
        ):
            if new_embeds.numel():
                parts.append(new_embeds)
            decisions.append({
                "feature_prefix_samples": feature_prefix,
                "available_prefix_samples": available_prefix,
                "lookahead_samples": available_prefix - feature_prefix,
                "new_audio_tokens": len(new_embeds),
                "consumed_audio_tokens": consumed,
                "decision": "END_FRONTEND" if end_of_audio else "READ_FRONTEND",
                "end_of_audio": end_of_audio,
                "offline_thinker": True,
            })
        full = torch.cat(parts, dim=0)
    else:
        full = encoder(wav)
    total_tokens = len(full)
    prompt_text, segments = build_prompt(
        SYSTEM, task_prompt, full, [], open_for_write=True,
    )
    if hasattr(backend, "set_write_context"):
        backend.set_write_context(final_write=True)
    text = backend.generate(prompt_text, segments)
    decisions.append({
        "consumed_audio_tokens": total_tokens,
        "p_write": 1.0,
        "logits": [float("nan"), float("nan")],
        "decision": "WRITE",
        "forced_eos": True,
        "offline_thinker": True,
    })
    return _finish_result(
        index, wav_path, source, reference, [text], total_tokens,
        total_tokens, decisions,
    )


def evaluate_one_deferred_talker(index, obj, encoder, backend, args):
    """Estimate READ/WRITE online, but synthesize once after all audio is read.

    Gate decisions are advisory only in this quality-first evaluator.  The
    frontend still receives closed 640-ms blocks with the configured lookahead,
    and the first predicted WRITE is retained as the latency estimate.  No
    assistant text is committed at an intermediate boundary.  At end of audio
    one synchronized request lets Thinker finish the whole translation and then
    gives its continuous hidden-state span to Talker.  Consequently there is no
    prior-WRITE request reconstruction, local padding, or codec-history replay.
    """
    wav_path, task_prompt, source, reference = load_row(obj)
    wav = np.asarray(load_audio_text_image_video(wav_path, use_ffmpeg=False))
    chunk_samples = round(args.encoder_chunk_ms * args.audio_sample_rate / 1000.0)
    lookahead_samples = round(STREAMING_LOOKAHEAD_MS * args.audio_sample_rate / 1000.0)
    parts = []
    decisions = []
    first_write = None
    total_tokens = 0

    for feature_prefix_samples, available_prefix_samples, new_embeds, consumed, end_of_audio in encoder.stream_prefixes(
        wav, chunk_samples, lookahead_samples
    ):
        if new_embeds.numel():
            parts.append(new_embeds)
        total_tokens = consumed
        if not parts:
            decisions.append({
                "audio_prefix_samples": available_prefix_samples,
                "feature_prefix_samples": feature_prefix_samples,
                "new_audio_tokens": 0,
                "consumed_audio_tokens": consumed,
                "decision": "END_NO_NEW_AUDIO" if end_of_audio else "READ_NO_NEW_AUDIO",
                "advisory_gate": True,
                "end_of_audio": end_of_audio,
            })
            continue

        full_prefix = torch.cat(parts, dim=0)
        if end_of_audio:
            write, forced = True, True
        else:
            gate_prompt, gate_segments = build_prompt(
                SYSTEM, task_prompt, full_prefix, [], open_for_write=False
            )
            write = backend.gate_decision(gate_prompt, gate_segments)
            forced = False
        if write and first_write is None:
            first_write = consumed
        decisions.append({
            "audio_prefix_samples": available_prefix_samples,
            "audio_prefix_ms": 1000.0 * available_prefix_samples / args.audio_sample_rate,
            "feature_prefix_samples": feature_prefix_samples,
            "feature_prefix_ms": 1000.0 * feature_prefix_samples / args.audio_sample_rate,
            "new_audio_tokens": len(new_embeds),
            "consumed_audio_tokens": consumed,
            "decision": "WRITE_DEFERRED" if write else "READ",
            "forced_eos": forced,
            "advisory_gate": True,
            "end_of_audio": end_of_audio,
        })

    if not parts:
        raise RuntimeError("streaming frontend produced no Thinker audio embeddings")
    full = torch.cat(parts, dim=0)
    prompt_text, segments = build_prompt(
        SYSTEM, task_prompt, full, [], open_for_write=True
    )
    if hasattr(backend, "set_write_context"):
        backend.set_write_context(final_write=True)
    text = backend.generate(prompt_text, segments)
    result = _finish_result(
        index, wav_path, source, reference, [text], first_write,
        total_tokens, decisions,
    )
    result.update({
        "inference_contract": "deferred_talker_final_only_v1",
        "gate_is_advisory": True,
        "talker_write_count": 1,
    })
    return result


def evaluate_one_streaming_thinker_deferred_talker(
    index, obj, encoder, backend, args
):
    """Run source-causal Thinker WRITEs and lockstep Talker codec generation.

    A WRITE is turned into codec before the frontend reads the next block.  A
    no-KV evaluator rebuilds prior Talker positions from the same causal
    Thinker prefix and sampled codec embeddings, then feeds current hidden
    states one at a time.  Thus rebuilding is compute-only; it cannot expose
    later audio or turn the current text span into a Talker prefill.
    """
    wav_path, task_prompt, source, reference = load_row(obj)
    wav = np.asarray(load_audio_text_image_video(wav_path, use_ffmpeg=False))
    chunk_samples = round(args.encoder_chunk_ms * args.audio_sample_rate / 1000.0)
    lookahead_samples = round(STREAMING_LOOKAHEAD_MS * args.audio_sample_rate / 1000.0)
    committed = SYSTEM
    committed_segments = []
    pending_parts = []
    pieces = []
    decisions = []
    first_write = None
    total_tokens = 0

    for feature_prefix_samples, available_prefix_samples, new_embeds, consumed, end_of_audio in encoder.stream_prefixes(
        wav, chunk_samples, lookahead_samples
    ):
        if new_embeds.numel():
            pending_parts.append(new_embeds)
        total_tokens = consumed
        if not pending_parts:
            decisions.append({
                "audio_prefix_samples": available_prefix_samples,
                "feature_prefix_samples": feature_prefix_samples,
                "new_audio_tokens": 0,
                "consumed_audio_tokens": consumed,
                "decision": "END_NO_NEW_AUDIO" if end_of_audio else "READ_NO_NEW_AUDIO",
                "forced_eos": end_of_audio,
                "end_of_audio": end_of_audio,
            })
            if end_of_audio:
                break
            continue

        pending = torch.cat(pending_parts, dim=0)
        if end_of_audio:
            write, forced = True, True
        else:
            gate_prompt, segments = build_prompt(
                committed, task_prompt, pending, committed_segments,
                open_for_write=False,
            )
            write = backend.gate_decision(gate_prompt, segments)
            forced = False
        decisions.append({
            "audio_prefix_samples": available_prefix_samples,
            "audio_prefix_ms": 1000.0 * available_prefix_samples / args.audio_sample_rate,
            "feature_prefix_samples": feature_prefix_samples,
            "feature_prefix_ms": 1000.0 * feature_prefix_samples / args.audio_sample_rate,
            "new_audio_tokens": len(new_embeds),
            "consumed_audio_tokens": consumed,
            "decision": "WRITE" if write else "READ",
            "forced_eos": forced,
            "end_of_audio": end_of_audio,
        })
        if not write:
            continue
        prompt_text, segments = build_prompt(
            committed, task_prompt, pending, committed_segments,
            open_for_write=True,
        )
        text = backend.generate_text(prompt_text, segments)
        if not text:
            if end_of_audio and pieces and hasattr(
                backend, "finalize_without_new_write"
            ):
                # A unified-EOS Talker already closed the preceding envelope.
                # With no new final text, source-end control simply promotes
                # that boundary to utterance-final; it must not invent a blank
                # WRITE or add synthetic Thinker hidden states.
                backend.finalize_without_new_write(prompt_text, segments)
                decisions[-1]["decision"] = "FINALIZE_LAST_WRITE"
                pending_parts = []
                break
            if not end_of_audio:
                # An advisory WRITE with no emitted token commits nothing.
                # Keep all pending audio so the next causal request sees the
                # same uncommitted prefix plus the following source block.
                decisions[-1]["decision"] = "READ_EMPTY_WRITE"
                continue
            raise RuntimeError("streaming Thinker WRITE produced empty text")
        if first_write is None:
            first_write = consumed
        # Do this before reading the next encoder block.  ``segments`` has
        # exactly the committed audio plus this WRITE's pending audio, never a
        # later source block.  The backend forces the already generated text
        # only to recover its causal hidden states; the Talker itself consumes
        # those states in hidden/codec lockstep rather than bulk-prefilling.
        if hasattr(backend, "set_write_context"):
            backend.set_write_context(final_write=end_of_audio)
        replayed_text = backend.generate_forced_speech(prompt_text, segments, text)
        if replayed_text != text:
            raise RuntimeError(
                "forced streaming Talker conditioning text mismatch: "
                f"expected={text!r} actual={replayed_text!r}"
            )
        pieces.append(text)
        committed = prompt_text + text + "<|im_end|>\n"
        committed_segments = segments
        pending_parts = []

    result = _finish_result(
        index, wav_path, source, reference, pieces, first_write,
        total_tokens, decisions,
    )
    if not result["prediction"]:
        raise RuntimeError("streaming Thinker produced empty final translation")

    result.update({
        "inference_contract": "streaming_thinker_lockstep_codec_v3",
        "gate_is_advisory": False,
        "talker_write_count": len(pieces),
        "talker_conditioning_text": pieces,
        "thinker_write_count": len(pieces),
    })
    return result


def evaluate_one(index, obj, encoder, backend, args):
    if args.offline_thinker:
        return evaluate_one_offline_thinker(index, obj, encoder, backend, args)
    if args.true_streaming_encoder:
        return evaluate_one_true_streaming(index, obj, encoder, backend, args)
    wav_path, task_prompt, source, reference = load_row(obj)
    wav = load_audio_text_image_video(wav_path, use_ffmpeg=False)
    full = encoder(np.asarray(wav))
    total_tokens = len(full)
    consumed = 0
    pending_start = 0
    committed = SYSTEM
    committed_segments = []
    pieces = []
    decisions = []
    first_write = None

    while consumed < total_tokens:
        consumed = min(total_tokens, consumed + args.interval_tokens)
        pending = full[pending_start:consumed]
        end_of_audio = consumed == total_tokens
        if end_of_audio:
            p_write, logits, write, forced = 1.0, [float("nan"), float("nan")], True, True
        else:
            gate_prompt, segments = build_prompt(
                committed, task_prompt, pending, committed_segments,
                open_for_write=False,
            )
            write, forced = backend.gate_decision(gate_prompt, segments), False
            p_write, logits = (1.0 if write else 0.0), []
        decisions.append({
            "consumed_audio_tokens": consumed, "p_write": p_write,
            "logits": logits, "decision": "WRITE" if write else "READ",
            "forced_eos": forced,
        })
        if not write:
            continue
        if first_write is None:
            first_write = consumed
        prompt_text, segments = build_prompt(
            committed, task_prompt, pending, committed_segments,
            open_for_write=True,
        )
        if hasattr(backend, "set_write_context"):
            backend.set_write_context(final_write=end_of_audio)
        text = backend.generate(prompt_text, segments)
        pieces.append(text)
        committed = prompt_text + text + "<|im_end|>\n"
        committed_segments = segments
        pending_start = consumed

    return _finish_result(
        index, wav_path, source, reference, pieces, first_write,
        total_tokens, decisions,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--devices", type=json.loads, default=[0, 1])
    ap.add_argument("--encoder-device", default="cuda:0")
    ap.add_argument("--true-streaming-encoder", action="store_true")
    ap.add_argument(
        "--offline-thinker", action="store_true",
        help="Stream the frontend, then run one final offline Thinker generation.",
    )
    ap.add_argument("--encoder-chunk-ms", type=float, default=640.0)
    ap.add_argument("--audio-sample-rate", type=int, default=16000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--interval-tokens", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.input).read_text().splitlines() if x.strip()]
    end = len(rows) if args.end is None else min(args.end, len(rows))
    selected = list(enumerate(rows))[args.start:end]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        completed = {json.loads(x)["index"] for x in out.read_text().splitlines() if x.strip()}

    encoder = FullRecordEncoder(
        Path(args.model), args.encoder_device,
        true_streaming=args.true_streaming_encoder,
        chunk_ms=args.encoder_chunk_ms,
    )
    backend = ThinkerBackend(args)
    try:
        with out.open("a", encoding="utf-8") as fout:
            for pos, (index, obj) in enumerate(selected, 1):
                if index in completed:
                    continue
                started = time.time()
                try:
                    result = evaluate_one(index, obj, encoder, backend, args)
                    result["elapsed_s"] = time.time() - started
                except Exception as exc:
                    result = {"index": index, "error": repr(exc)}
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                print(json.dumps({
                    "progress": f"{pos}/{len(selected)}", "index": index,
                    "first_write": result.get("first_write_audio_tokens"),
                    "total": result.get("total_audio_tokens"),
                    "error": result.get("error"),
                }, ensure_ascii=False), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
