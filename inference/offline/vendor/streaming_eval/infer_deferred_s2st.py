#!/usr/bin/env python3
"""Causal streaming S2ST inference with one utterance-global codec Talker.

Each Thinker WRITE is evaluated as soon as its source-audio block closes: no
Thinker request ever contains a later source-audio block.  Its causal hidden
timeline is reconstructed after each WRITE, while one persistent Talker request
keeps its KV cache for the complete utterance.  At every Talker position the
controller supplies exactly one Thinker hidden plus either a real codec
embedding, an injected BOS embedding, or a zero codec branch (training ``-60``).
Outputs sampled at zero-codec positions are discarded and never fed back.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from threading import RLock, Thread

import numpy as np
import torch
from transformers import AutoTokenizer

from forced_sequence_logits import ForcedSequenceLogitsProcessor


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
sys.path.insert(0, str(REPO))

# Importing this module installs the streaming vLLM model override and provides
# the audited encoder and READ/WRITE state machine.  Do not replace it with the
# offline S2ST inference module.
import streaming_core as s2tt  # noqa: E402
from vllm.engine.omnidrca2_engine import (  # noqa: E402
    OmniDRCA2Engine,
    SynchronizedGenerator,
)
from vllm.engine.shared_hidden_buffer import SharedHiddenBuffer  # noqa: E402
from vllm.inputs import TextPrompt  # noqa: E402
from vllm.outputs import RequestOutput  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402


# Codec-channel controls shared by the historical unified-EOS checkpoint and
# the later 0903 injected-dual-BOS checkpoint.  They are not acoustic tokens.
TRAININFER_CONTINUE_BOS_ID = 59049
TRAININFER_EOS_ID = 59050
TRAININFER_PAUSE_ID = 59051
TRAININFER_FINAL_BOS_ID = 59052
TRAININFER_BLANK_COUNT = 5
TRAININFER_ALIGNMENT_MASK_ID = -60
TRAININFER_ENCODER_CHUNK_MS = 640.0
TRAININFER_WAVEFORM_LOOKAHEAD_MS = 162.5
BOUNDARY_POLICY_UNIFIED_EOS = "unified_eos"
BOUNDARY_POLICY_DUAL_BOS = "dual_bos_pause_eos"
from vllm.v1.engine.async_llm import AsyncEngineArgs  # noqa: E402
from vllm.v1.engine import ExternalData  # noqa: E402


class CodecHistoryReplayEngine(OmniDRCA2Engine):
    """Keep Talker inputs aligned to the causal Thinker/codec training timeline.

    Training runs one causal Talker forward over every READ/WRITE turn.  At a
    later WRITE, the Talker therefore sees projected Thinker hidden states plus
    teacher-forced codec embeddings from earlier WRITE spans.  The stock
    synchronized inference engine rebuilds the Thinker prefix for each WRITE,
    but drops both the earlier codec-length padding and codec embeddings.

    Legacy mode rebuilds prior WRITE inputs.  Global mode instead keeps one
    Talker request alive and explicitly supplies each new dual-stream position,
    without changing installed vLLM.
    """

    def __init__(self, *args, codec_embed_path: Path, **kwargs):
        super().__init__(*args, **kwargs)
        # The installed aligned-Talker engine consumes this request-scoped
        # prefix map, but its constructor does not create it.  Legacy paths
        # never exercised the lookup because they did not use the aligned
        # CosyVoice branch.  Keep the compatibility state local to streaming
        # inference instead of modifying the shared vLLM installation.
        if not hasattr(self, "cosyvoice_codec_prefix_ids"):
            self.cosyvoice_codec_prefix_ids: dict[str, list[int]] = {}
        state = torch.load(codec_embed_path, map_location="cpu", weights_only=True)
        weight = state["weight"] if isinstance(state, dict) else state
        if weight.ndim != 2:
            raise ValueError(f"invalid codec embedding shape: {tuple(weight.shape)}")
        self._replay_codec_embedding = weight.detach().cpu().contiguous()
        self._codec_replay_context: dict[str, dict] = {}
        self._active_codec_replay_request: str | None = None
        # The legacy evaluator starts a separate Talker request for every
        # Thinker WRITE.  That makes the Talker drain local text spans with
        # synthetic padding even though future real Thinker positions will
        # arrive after the next READ/WRITE cycle.  The global mode below keeps
        # one Talker request alive for the complete source utterance.
        self._global_timeline_enabled = False
        self._global_talker_request_id: str | None = None
        self._global_real_hidden_len = 0
        self._global_available_end = 0
        self._global_available_hidden: torch.Tensor | None = None
        self._global_write_specs: list[dict] = []
        self._global_next_bos_index = 0
        self._global_previous_boundary = -1
        self._global_processed_position = -1
        self._global_expect_codec_output = False
        self._global_waiting_after_output = False
        self._global_pending_feedback_token: int | None = None
        self._global_active_bos_id: int | None = None
        self._global_active_write_is_final = False
        self._global_codec_tokens: list[int] = []
        self._global_audio_by_position: dict[int, int] = {}
        self._global_failure: str | None = None
        self._global_lock = RLock()
        self._global_final_requested = False
        self._global_final_eos_observed = False
        self._global_final_eos_token_count = 0
        self._global_next_write_is_final = False
        self._global_next_write_finalize_only = False
        self._global_alignment_epoch = 0
        self.global_codec_bos_id = 59049
        self.global_codec_final_bos_id = 59052
        self.global_codec_pause_id = 59051
        self.global_codec_eos_id = 59050
        self.global_boundary_policy = BOUNDARY_POLICY_UNIFIED_EOS

    def enable_global_timeline(self) -> None:
        self._global_timeline_enabled = True

    def reset_global_timeline(self) -> None:
        request_id = self._global_talker_request_id
        if request_id is not None:
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(
                    self.talker_engine.abort(request_id), self._loop
                ).result()
            self.output_queue.pop(request_id, None)
            self.talker_listener.pop(request_id, None)
            self.talker_speech_ids.pop(request_id, None)
            self.thinker_output_len.pop(request_id, None)
            self.thinker_generate_speech.pop(request_id, None)
            self.request_prompts.pop(request_id, None)
            self.cosyvoice_talker_hidden.pop(request_id, None)
            self.cosyvoice_talker_hidden_cursor.pop(request_id, None)
            self.cosyvoice_talker_debug_step.pop(request_id, None)
        self._global_talker_request_id = None
        self._global_real_hidden_len = 0
        self._global_available_end = 0
        self._global_available_hidden = None
        self._global_write_specs = []
        self._global_next_bos_index = 0
        self._global_previous_boundary = -1
        self._global_processed_position = -1
        self._global_expect_codec_output = False
        self._global_waiting_after_output = False
        self._global_pending_feedback_token = None
        self._global_active_bos_id = None
        self._global_active_write_is_final = False
        self._global_codec_tokens = []
        self._global_audio_by_position = {}
        self._global_failure = None
        self._global_final_requested = False
        self._global_final_eos_observed = False
        self._global_final_eos_token_count = 0
        self._global_next_write_is_final = False
        self._global_next_write_finalize_only = False
        self._global_alignment_epoch = 0

    def set_global_write_context(
        self, *, final_write: bool, finalize_only: bool = False
    ) -> None:
        self._global_next_write_is_final = bool(final_write)
        self._global_next_write_finalize_only = bool(finalize_only)

    def finalize_global_at_last_boundary(self) -> None:
        """Declare the most recent unified EOS to be utterance-final.

        The historical checkpoint emits the same EOS after every WRITE.  If
        the source-complete Thinker turn produces no additional text, there is
        no new envelope to run: the controller must reinterpret the already
        completed last envelope as final and stop the persistent Talker there.
        """
        if self.global_boundary_policy != BOUNDARY_POLICY_UNIFIED_EOS:
            raise RuntimeError(
                "finalizing without a new WRITE is only valid for unified_eos"
            )
        request_id = self._global_talker_request_id
        with self._global_lock:
            if request_id is None:
                raise RuntimeError("cannot finalize before global Talker starts")
            if self._global_final_eos_observed:
                return
            if self._global_active_bos_id is not None:
                raise RuntimeError("cannot finalize with an open codec envelope")
            if self._global_next_bos_index != len(self._global_write_specs):
                raise RuntimeError("cannot finalize before all WRITE envelopes start")
            if (
                not self._global_codec_tokens
                or int(self._global_codec_tokens[-1]) != self.global_codec_eos_id
            ):
                raise RuntimeError("last completed codec envelope has no EOS")
            self._global_final_requested = True
            self._global_final_eos_observed = True
            self._global_final_eos_token_count = len(self._global_codec_tokens)
            self.thinker_generate_speech[request_id] = False
        print(
            "[INJECTED BOS TIMELINE] source ended with empty Thinker WRITE; "
            f"finalize prior unified EOS request={request_id} "
            f"tokens={self._global_final_eos_token_count}",
            flush=True,
        )
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(
                self.talker_engine.abort(request_id), self._loop
            ).result()

    def _next_injected_bos(self) -> tuple[int, int, bool] | None:
        if self._global_next_bos_index >= len(self._global_write_specs):
            return None
        spec = self._global_write_specs[self._global_next_bos_index]
        position = max(
            int(spec["text_start"]) + int(spec["blank_count"]),
            self._global_previous_boundary + 1,
        )
        return position, int(spec["bos_id"]), bool(spec["final_write"])

    def _send_global_position(
        self,
        position: int,
        *,
        codec_input_id: int | None,
        expect_codec_output: bool,
    ) -> bool:
        """Advance one Talker position with explicit dual-stream input.

        ``codec_input_id=None`` reproduces the training -60 sentinel: only the
        real Thinker hidden is sent and the sampled output for this position is
        ignored.  It is a zero codec branch, not codec id 0.  Non-negative IDs
        are added from the exported codec embedding table, so no incidental
        vLLM output token can leak into the next input.
        """
        if (
            self._global_available_hidden is None
            or position >= self._global_available_end
        ):
            return False
        prompt_embeds = (
            self._global_available_hidden[position : position + 1]
            .detach()
            .cpu()
            .clone()
            .contiguous()
        )
        if codec_input_id is not None:
            if not 0 <= codec_input_id < self._replay_codec_embedding.size(0):
                raise RuntimeError(
                    f"codec input id {codec_input_id} outside embedding table"
                )
            flat = prompt_embeds.reshape(1, -1)
            codec_embed = self._replay_codec_embedding[codec_input_id : codec_input_id + 1]
            if flat.size(1) != codec_embed.size(1):
                raise RuntimeError(
                    "Talker hidden/codec embedding width mismatch: "
                    f"hidden={flat.size(1)} codec={codec_embed.size(1)}"
                )
            flat += codec_embed.to(dtype=flat.dtype)
            self._global_audio_by_position[position] = int(codec_input_id)
        self._global_processed_position = position
        self._global_expect_codec_output = bool(expect_codec_output)
        self._global_waiting_after_output = False
        self._global_pending_feedback_token = None
        self._add_talker_external_data(
            request_id=self._global_talker_request_id,
            prompt_embeds=prompt_embeds,
            # vLLM interprets is_padding=True as replay this ExternalData
            # automatically forever, not as a text-alignment padding marker.
            # Every global position has fresh hidden AND codec feedback, even
            # on final drain; auto-replay would feed stale codec embeddings.
            is_padding=False,
            # Codec input is already added explicitly above.  Never add the
            # scheduler's previous sampled token implicitly.
            add_token_embedding=False,
        )
        return True

    def _advance_global_after_ignored_output(self) -> bool:
        """Feed a gap position or inject the next queued BOS."""
        next_position = self._global_processed_position + 1
        next_bos = self._next_injected_bos()
        if next_bos is not None:
            bos_position, bos_id, final_write = next_bos
            if next_position > bos_position:
                raise RuntimeError(
                    "missed injected codec BOS position: "
                    f"next={next_position} bos={bos_position}"
                )
            if next_position == bos_position:
                if not self._send_global_position(
                    next_position,
                    codec_input_id=bos_id,
                    expect_codec_output=True,
                ):
                    return False
                self._global_active_bos_id = bos_id
                self._global_active_write_is_final = final_write
                self._global_next_bos_index += 1
                self._global_codec_tokens.append(bos_id)
                print(
                    "[INJECTED BOS TIMELINE] inject BOS "
                    f"index={self._global_next_bos_index - 1} "
                    f"position={next_position} token={bos_id}",
                    flush=True,
                )
                return True
        return self._send_global_position(
            next_position,
            codec_input_id=None,
            expect_codec_output=False,
        )

    def _resume_global_talker_if_waiting(self) -> None:
        if (
            self._global_talker_request_id is None
            or not self._global_waiting_after_output
        ):
            return
        if self._global_pending_feedback_token is not None:
            token = int(self._global_pending_feedback_token)
            self._send_global_position(
                self._global_processed_position + 1,
                codec_input_id=token,
                expect_codec_output=token
                not in (self.global_codec_pause_id, self.global_codec_eos_id),
            )
        else:
            self._advance_global_after_ignored_output()

    def _add_talker_external_data(
        self,
        request_id: str,
        prompt_embeds: torch.Tensor,
        is_padding: bool,
        add_token_embedding: bool,
    ) -> None:
        """Send one-step Talker conditioning directly, avoiding SHM expiry.

        The initial Talker prompt remains on the established SHM path.  Only
        the tiny per-token continuation vector uses vLLM's native direct tensor
        field, because queued temporary SHM names can be unlinked before a
        long/coalesced Talker step reaches them.
        """
        external_data = ExternalData(
            request_id=request_id,
            external_embeds=self._compact_cpu_tensor(prompt_embeds),
            external_embeds_shm_key=None,
            is_padding=is_padding,
            add_token_embedding=add_token_embedding,
        )
        asyncio.run_coroutine_threadsafe(
            self.talker_engine.update_request_with_data(external_data),
            self._loop,
        ).result()

    def _listen_talker(self, request_id, generator, output_queue) -> None:
        """Run one Talker request over codec, blank, and injected-BOS states."""
        if request_id != self._global_talker_request_id:
            return super()._listen_talker(request_id, generator, output_queue)

        talker_hf_config = self._get_talker_hf_config()
        group_size = int(talker_hf_config.group_size)
        if group_size != 1:
            raise RuntimeError(
                "global streaming EOS listener requires group_size=1, "
                f"got {group_size}"
            )
        codec_unit = int(getattr(talker_hf_config, "codec_unit", -1))
        pause_id = int(self.global_codec_pause_id)
        eos_id = int(self.global_codec_eos_id)

        def _listen() -> None:
            total_speech_token_num = 0
            self.talker_speech_ids[request_id] = []
            stopped_on_final_eos = False
            try:
                for output in SynchronizedGenerator(generator, self._loop):
                    if output.outputs is None:
                        break
                    token_ids = list(output.outputs[0].token_ids or [])
                    current_speech_token_num = len(token_ids)
                    if current_speech_token_num == total_speech_token_num:
                        continue
                    if current_speech_token_num < total_speech_token_num:
                        raise RuntimeError(
                            "global Talker token stream regressed: "
                            f"old={total_speech_token_num} new={current_speech_token_num}"
                        )
                    if current_speech_token_num - total_speech_token_num != group_size:
                        raise RuntimeError(
                            "Talker advanced without one fresh input per codec step: "
                            f"previous={total_speech_token_num} "
                            f"current={current_speech_token_num} group_size={group_size}. "
                            "Check ExternalData auto-replay before consuming outputs."
                        )
                    previous_speech_token_num = total_speech_token_num
                    total_speech_token_num = current_speech_token_num
                    if current_speech_token_num % group_size:
                        continue
                    for offset in range(previous_speech_token_num, current_speech_token_num, group_size):
                        group = token_ids[offset:offset + group_size]
                        self.talker_speech_ids[request_id].extend(group)
                        with contextlib.suppress(Exception):
                            # Copy because vLLM reuses the backing list in later
                            # RequestOutput updates.
                            output_queue.put(list(self.talker_speech_ids[request_id]))
                        token = int(group[0])
                        with self._global_lock:
                            if self._global_expect_codec_output:
                                if token in (
                                    self.global_codec_bos_id,
                                    self.global_codec_final_bos_id,
                                ):
                                    raise RuntimeError(
                                        f"Talker sampled controller-only BOS {token}"
                                    )
                                self._global_codec_tokens.append(token)
                                token_position = self._global_processed_position + 1
                                if token in (pause_id, eos_id):
                                    envelope_is_final = bool(
                                        self._global_active_write_is_final
                                    )
                                    if (
                                        self.global_boundary_policy
                                        == BOUNDARY_POLICY_UNIFIED_EOS
                                    ):
                                        expected = eos_id
                                    else:
                                        expected = (
                                            eos_id if envelope_is_final else pause_id
                                        )
                                    if token != expected:
                                        raise RuntimeError(
                                            "Talker boundary contradicts training policy: "
                                            f"policy={self.global_boundary_policy} "
                                            f"bos={self._global_active_bos_id} "
                                            f"final={envelope_is_final} "
                                            f"boundary={token} expected={expected}"
                                        )
                                    self._global_previous_boundary = token_position
                                    self._global_active_bos_id = None
                                    self._global_active_write_is_final = False
                                    if envelope_is_final:
                                        if token != eos_id:
                                            raise RuntimeError(
                                                "final codec envelope did not emit EOS"
                                            )
                                        self._global_audio_by_position[token_position] = token
                                        self._global_final_eos_observed = True
                                        self._global_final_eos_token_count = len(
                                            self._global_codec_tokens
                                        )
                                        self.thinker_generate_speech[request_id] = False
                                        stopped_on_final_eos = True
                                    else:
                                        # PAUSE (dual-BOS) or EOS (historical
                                        # unified-EOS) is a real codec-channel
                                        # input.  Its own logit has no loss and
                                        # is discarded before entering gap mode.
                                        self._global_pending_feedback_token = token
                                        self._global_waiting_after_output = True
                                        self._send_global_position(
                                            token_position,
                                            codec_input_id=token,
                                            expect_codec_output=False,
                                        )
                                else:
                                    if not 0 <= token < codec_unit:
                                        raise RuntimeError(
                                            f"invalid acoustic codec token {token}"
                                        )
                                    self._global_pending_feedback_token = token
                                    self._global_waiting_after_output = True
                                    self._send_global_position(
                                        token_position,
                                        codec_input_id=token,
                                        expect_codec_output=True,
                                    )
                            else:
                                # This logit came from a codec-empty or PAUSE
                                # input position.  It is intentionally ignored
                                # and can never be fed back into Talker.
                                self._global_waiting_after_output = True
                                self._global_pending_feedback_token = None
                                self._advance_global_after_ignored_output()
                        if stopped_on_final_eos:
                            print(
                                "[INJECTED BOS TIMELINE] final codec EOS "
                                f"request={request_id} "
                                f"tokens={self._global_final_eos_token_count}",
                                flush=True,
                            )
                            asyncio.run_coroutine_threadsafe(
                                self.talker_engine.abort(request_id), self._loop
                            ).result()
                            break
                    if stopped_on_final_eos:
                        break
                if not self._global_final_eos_observed:
                    raise RuntimeError(
                        "Talker request ended before the final WRITE produced codec EOS"
                    )
            except Exception as exc:
                self._global_failure = repr(exc)
                print(
                    f"[INJECTED BOS TIMELINE] Talker listener failure "
                    f"request={request_id}: {exc!r}",
                    flush=True,
                )
                with contextlib.suppress(Exception):
                    asyncio.run_coroutine_threadsafe(
                        self.talker_engine.abort(request_id), self._loop
                    ).result()
            finally:
                # Keep output_queue and token history until finish_sample()
                # consumes the final EOS.  reset_global_timeline() owns their
                # cleanup; the base listener would remove them too early.
                self.talker_listener.pop(request_id, None)

        self.talker_listener[request_id] = Thread(target=_listen, daemon=True)
        self.talker_listener[request_id].start()

    def _run_global_alignment(self, request_id, thinker_output):
        """Recover real causal Thinker states, adding pads only on final drain."""
        generated_token_ids = self._strip_thinker_stop_tokens(
            list(thinker_output.outputs[0].token_ids or [])
        )
        # im_end is a request stop token, but this checkpoint's text_config
        # eos_token_id is endoftext. Strip the actual chat stop explicitly
        # before materializing exactly one training turn suffix below.
        im_end_id = int(self.thinker_tokenizer.convert_tokens_to_ids("<|im_end|>"))
        while generated_token_ids and generated_token_ids[-1] == im_end_id:
            generated_token_ids.pop()
        finalize_only = bool(self._global_next_write_finalize_only)
        if finalize_only:
            # The forced stop token closes the Thinker request but is not a
            # new translation token or a Talker-alignment position.
            generated_token_ids = []
        blank_count = self._get_cosyvoice_blank_count()
        original_prompt = self.request_prompts.get(request_id)
        base_prompt_token_ids = self._get_alignment_base_prompt_token_ids(
            original_prompt, list(thinker_output.prompt_token_ids or [])
        )
        final_write = bool(self._global_next_write_is_final)
        # Training appends tokenizer pad positions only after the *complete*
        # utterance has no more real Thinker positions.  Keep enough of those
        # final positions to permit the codec request to reach EOS.
        pad_count = 0
        if final_write:
            sampling = self.talker_requests.get(request_id)
            pad_count = int(getattr(sampling, "max_tokens", 0) or 0) + blank_count + 2
        pad_token_id = int(getattr(self.thinker_tokenizer, "pad_token_id", 0) or 0)
        # The training collator retains the assistant's <|im_end|>\\n
        # before utterance-tail padding. These are real Thinker positions,
        # including at the last WRITE; skipping them changes every drain hidden.
        suffix_ids = [] if finalize_only else list(self.thinker_tokenizer.encode(
            "<|im_end|>\n", add_special_tokens=False
        ))
        real_segment = generated_token_ids + suffix_ids
        aligned_segment = real_segment + [pad_token_id] * pad_count
        align_prompt = {"prompt_token_ids": base_prompt_token_ids + aligned_segment}
        if isinstance(original_prompt, Mapping):
            for key in ("multi_modal_data", "mm_processor_kwargs"):
                if original_prompt.get(key) is not None:
                    align_prompt[key] = original_prompt.get(key)
        align_request_id = f"{request_id}_global_codec_align"
        align_params = SamplingParams(temperature=0.0, top_k=1, max_tokens=1, detokenize=False)
        generator = self.thinker_engine.generate(
            request_id=align_request_id,
            prompt=align_prompt,
            sampling_params=align_params,
            using_external_data=False,
        )
        hidden = None
        for output in SynchronizedGenerator(generator, self._loop):
            if hidden is None:
                hidden = SharedHiddenBuffer.read_global_by_request_id(
                    request_id=f"{align_request_id}_prefill", device="cpu", wait=True, timeout=30.0
                )
            if output.finished:
                break
        if hidden is None:
            raise RuntimeError(f"global codec alignment hidden missing for {request_id}")
        target_len = len(aligned_segment)
        prefix_len = int(hidden.size(0)) - target_len
        if prefix_len < 0:
            raise RuntimeError(
                f"global codec alignment length underflow: hidden={hidden.size(0)} target={target_len}"
            )
        real_end = prefix_len + len(real_segment)
        return (
            hidden,
            prefix_len,
            real_end,
            int(hidden.size(0)),
            blank_count,
            final_write,
            len(generated_token_ids),
            finalize_only,
        )

    def _start_global_talker_from_alignment(self, request_id, thinker_output) -> None:
        (
            hidden,
            prefix_len,
            real_end,
            all_end,
            blank_count,
            final_write,
            generated_token_count,
            finalize_only,
        ) = self._run_global_alignment(request_id, thinker_output)
        sampling = self.talker_requests.pop(request_id)
        start_payload = None
        abort_at_existing_boundary = False
        with self._global_lock:
            if self._global_final_eos_observed:
                raise RuntimeError("received a WRITE after final codec EOS")
            if real_end < self._global_real_hidden_len:
                raise RuntimeError(
                    "global Thinker timeline regressed: "
                    f"old={self._global_real_hidden_len} new={real_end}"
                )
            self._global_real_hidden_len = real_end
            self._global_available_end = all_end if final_write else real_end
            self._global_available_hidden = hidden[: self._global_available_end]
            empty_finalization = bool(
                finalize_only and final_write and self._global_write_specs
            )
            if empty_finalization:
                # No new Thinker token means no new Talker envelope/BOS.  The
                # final alignment request contributes only source-complete pad
                # hidden states, allowing the preceding envelope to drain.
                if self._global_final_requested:
                    raise RuntimeError("more than one final WRITE")
                self._global_final_requested = True
                self._global_write_specs[-1]["final_write"] = True
                if self._global_talker_request_id is None:
                    # A short first WRITE can end exactly at its delayed BOS
                    # position, so no Talker request exists yet.  Source-end
                    # pad hidden states now make that queued BOS available;
                    # start the prior (now final) envelope without inventing a
                    # new WRITE.
                    first_bos = self._next_injected_bos()
                    if first_bos is None or first_bos[0] >= self._global_available_end:
                        raise RuntimeError(
                            "final drain still cannot reach the queued codec BOS"
                        )
                    bos_position, first_bos_id, first_write_is_final = first_bos
                    prompt = hidden[: bos_position + 1].detach().cpu().clone().contiguous()
                    flat = prompt.reshape(prompt.size(0), -1)
                    bos_embed = self._replay_codec_embedding[
                        first_bos_id : first_bos_id + 1
                    ]
                    if flat.size(1) != bos_embed.size(1):
                        raise RuntimeError(
                            "Talker hidden/codec embedding width mismatch: "
                            f"hidden={flat.size(1)} codec={bos_embed.size(1)}"
                        )
                    flat[bos_position : bos_position + 1] += bos_embed.to(
                        dtype=flat.dtype
                    )
                    global_request_id = f"global_codec_{uuid.uuid4().hex}"
                    self._global_talker_request_id = global_request_id
                    self._global_processed_position = bos_position
                    self._global_expect_codec_output = True
                    self._global_active_bos_id = first_bos_id
                    self._global_active_write_is_final = first_write_is_final
                    self._global_next_bos_index += 1
                    self._global_codec_tokens.append(first_bos_id)
                    self._global_audio_by_position[bos_position] = first_bos_id
                    queued_sampling = self._global_write_specs[0]["sampling"]
                    queued_sampling.stop_token_ids = []
                    queued_sampling.ignore_eos = True
                    start_payload = (
                        global_request_id,
                        flat,
                        queued_sampling,
                        blank_count,
                        bos_position + 1,
                    )
                elif self._global_next_bos_index == len(self._global_write_specs):
                    if self._global_active_bos_id is not None:
                        self._global_active_write_is_final = True
                    elif (
                        self._global_codec_tokens
                        and int(self._global_codec_tokens[-1])
                        == self.global_codec_eos_id
                    ):
                        # The shared EOS may have arrived just before source
                        # completion.  Controller state now declares it final.
                        self._global_final_eos_observed = True
                        self._global_final_eos_token_count = len(
                            self._global_codec_tokens
                        )
                        self.thinker_generate_speech[
                            self._global_talker_request_id
                        ] = False
                        abort_at_existing_boundary = True
                    else:
                        raise RuntimeError(
                            "empty final WRITE found no open or completed envelope"
                        )
                if not self._global_final_eos_observed:
                    self._resume_global_talker_if_waiting()
                print(
                    "[INJECTED BOS TIMELINE] append final drain without "
                    f"new WRITE real_end={real_end} available_end={all_end}",
                    flush=True,
                )
            else:
                bos_id = int(self.global_codec_bos_id)
                if (
                    self.global_boundary_policy == BOUNDARY_POLICY_DUAL_BOS
                    and final_write
                ):
                    bos_id = int(self.global_codec_final_bos_id)
                self._global_write_specs.append(
                    {
                        "text_start": prefix_len,
                        "blank_count": blank_count,
                        "bos_id": bos_id,
                        "final_write": final_write,
                        "sampling": sampling,
                    }
                )
                if final_write:
                    if self._global_final_requested:
                        raise RuntimeError("more than one final WRITE")
                    self._global_final_requested = True

                if self._global_talker_request_id is None:
                    first_bos = self._next_injected_bos()
                    if first_bos is not None and first_bos[0] < self._global_available_end:
                        bos_position, first_bos_id, first_write_is_final = first_bos
                        prompt = hidden[: bos_position + 1].detach().cpu().clone().contiguous()
                        flat = prompt.reshape(prompt.size(0), -1)
                        bos_embed = self._replay_codec_embedding[
                            first_bos_id : first_bos_id + 1
                        ]
                        if flat.size(1) != bos_embed.size(1):
                            raise RuntimeError(
                                "Talker hidden/codec embedding width mismatch: "
                                f"hidden={flat.size(1)} codec={bos_embed.size(1)}"
                            )
                        flat[bos_position : bos_position + 1] += bos_embed.to(
                            dtype=flat.dtype
                        )
                        global_request_id = f"global_codec_{uuid.uuid4().hex}"
                        self._global_talker_request_id = global_request_id
                        self._global_processed_position = bos_position
                        self._global_expect_codec_output = True
                        self._global_active_bos_id = first_bos_id
                        self._global_active_write_is_final = first_write_is_final
                        self._global_next_bos_index += 1
                        self._global_codec_tokens.append(first_bos_id)
                        self._global_audio_by_position[bos_position] = first_bos_id
                        sampling.stop_token_ids = []
                        sampling.ignore_eos = True
                        start_payload = (
                            global_request_id,
                            flat,
                            sampling,
                            blank_count,
                            bos_position + 1,
                        )
                else:
                    self._resume_global_talker_if_waiting()

                print(
                    "[INJECTED BOS TIMELINE] append WRITE "
                    f"index={len(self._global_write_specs) - 1} final={final_write} "
                    f"text_start={prefix_len} real_end={real_end} "
                    f"available_end={self._global_available_end}",
                    flush=True,
                )

        if start_payload is not None:
            (
                global_request_id,
                prompt,
                sampling,
                start_blank_count,
                prompt_frame_count,
            ) = start_payload
            self.output_queue[global_request_id]
            self.thinker_output_len[global_request_id] = 0
            self.thinker_generate_speech[global_request_id] = True
            print(
                "[INJECTED BOS TIMELINE] start persistent Talker "
                f"request={global_request_id} bos_position={prompt_frame_count - 1} "
                f"bos={self._global_active_bos_id} prefill={prompt_frame_count}",
                flush=True,
            )
            self._start_cosyvoice_talker_from_hidden(
                request_id=global_request_id,
                hidden_states=prompt,
                sampling_params=sampling,
                blank_count=start_blank_count,
                prompt_frame_count=prompt_frame_count,
            )
        if abort_at_existing_boundary and self._global_talker_request_id is not None:
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(
                    self.talker_engine.abort(self._global_talker_request_id),
                    self._loop,
                ).result()
        self._global_alignment_epoch += 1

    def register_codec_history(self, request_id: str, context: dict) -> None:
        self._codec_replay_context[request_id] = context

    def _start_cosyvoice_aligned_talker(self, request_id, thinker_output):
        # This must precede the legacy replay path below: a global session is
        # intentionally one Talker request across all WRITE turns, whereas
        # replay rebuilds a new local Talker prefix for every turn.
        if self._global_timeline_enabled:
            return self._start_global_talker_from_alignment(request_id, thinker_output)
        context = self._codec_replay_context.get(request_id)
        if not context or not context["codec_by_expanded_position"]:
            return super()._start_cosyvoice_aligned_talker(request_id, thinker_output)

        original_prompt = self.request_prompts.get(request_id)
        replay_prompt = {"prompt_token_ids": context["prompt_token_ids"]}
        if isinstance(original_prompt, Mapping):
            for key in ("multi_modal_data", "mm_processor_kwargs"):
                if original_prompt.get(key) is not None:
                    replay_prompt[key] = original_prompt.get(key)
        self.request_prompts[request_id] = replay_prompt
        self._active_codec_replay_request = request_id
        try:
            return super()._start_cosyvoice_aligned_talker(request_id, thinker_output)
        finally:
            self._active_codec_replay_request = None
            if original_prompt is not None:
                self.request_prompts[request_id] = original_prompt

    def _start_cosyvoice_talker_from_hidden(
        self,
        request_id,
        hidden_states,
        sampling_params,
        blank_count,
        prompt_frame_count=None,
        codec_prefix_ids=None,
    ):
        context = self._codec_replay_context.pop(request_id, None)
        if context and context["codec_by_expanded_position"]:
            talker_config = self._get_talker_hf_config()
            group_size = int(talker_config.group_size)
            if group_size != 1:
                raise RuntimeError(
                    "codec history replay currently requires Talker group_size=1, "
                    f"got {group_size}"
                )
            expected_prefix = int(context["expanded_prefix_len"])
            if int(hidden_states.size(0)) < expected_prefix:
                raise RuntimeError(
                    "aligned hidden prefix shorter than replay plan: "
                    f"hidden={int(hidden_states.size(0))} expected={expected_prefix}"
                )
            positions = sorted(context["codec_by_expanded_position"])
            if positions[-1] >= expected_prefix:
                raise RuntimeError(
                    f"codec replay position {positions[-1]} outside prefix {expected_prefix}"
                )
            token_ids = [
                context["codec_by_expanded_position"][position]
                for position in positions
            ]
            if min(token_ids) < 0 or max(token_ids) >= self._replay_codec_embedding.size(0):
                raise RuntimeError(
                    f"codec replay id range [{min(token_ids)}, {max(token_ids)}] "
                    f"outside embedding size {self._replay_codec_embedding.size(0)}"
                )
            hidden_states = hidden_states.clone()
            flat = hidden_states.reshape(hidden_states.size(0), -1)
            position_tensor = torch.tensor(positions, dtype=torch.long)
            token_tensor = torch.tensor(token_ids, dtype=torch.long)
            codec_embeds = self._replay_codec_embedding.index_select(0, token_tensor)
            flat[position_tensor] += codec_embeds.to(dtype=flat.dtype)
            context["applied_codec_tokens"] = len(token_ids)
            print(
                "[CODEC HISTORY REPLAY] "
                f"request={request_id} prior_writes={context['prior_writes']} "
                f"prefix={expected_prefix} codec_tokens={len(token_ids)} "
                f"inserted_mask_tokens={context['inserted_mask_tokens']}",
                flush=True,
            )
        # The H100/H100-2 vLLM build predates the base-engine
        # ``codec_prefix_ids`` keyword.  This compatibility branch belongs to
        # legacy per-WRITE replay; global mode supplies its codec embeddings
        # explicitly and therefore leaves ``codec_prefix_ids`` empty.
        if codec_prefix_ids is None:
            codec_prefix_ids = self.cosyvoice_codec_prefix_ids.pop(request_id, [])
        prefix_ids = list(codec_prefix_ids or [])
        if prefix_ids:
            group_size = int(self._get_talker_hf_config().group_size)
            if group_size != 1:
                raise RuntimeError(
                    "codec prefix replay currently requires Talker group_size=1, "
                    f"got {group_size}"
                )
            cursor = int(prompt_frame_count or blank_count)
            if cursor + len(prefix_ids) > int(hidden_states.size(0)):
                raise RuntimeError(
                    "codec prefix exceeds aligned Talker hidden states: "
                    f"cursor={cursor} prefix={len(prefix_ids)} "
                    f"hidden={int(hidden_states.size(0))}"
                )
            if min(prefix_ids) < 0 or max(prefix_ids) >= self._replay_codec_embedding.size(0):
                raise RuntimeError(
                    f"codec prefix id range [{min(prefix_ids)}, {max(prefix_ids)}] "
                    f"outside embedding size {self._replay_codec_embedding.size(0)}"
                )
            hidden_states = hidden_states.clone()
            flat = hidden_states.reshape(hidden_states.size(0), -1)
            positions = torch.arange(cursor, cursor + len(prefix_ids), dtype=torch.long)
            token_tensor = torch.tensor(prefix_ids, dtype=torch.long)
            prefix_embeds = self._replay_codec_embedding.index_select(0, token_tensor)
            flat[positions] += prefix_embeds.to(dtype=flat.dtype)
            prompt_frame_count = cursor + len(prefix_ids)
        return super()._start_cosyvoice_talker_from_hidden(
            request_id=request_id,
            hidden_states=hidden_states,
            sampling_params=sampling_params,
            blank_count=blank_count,
            prompt_frame_count=prompt_frame_count,
        )


def _is_int_sequence(value):
    """True only for objects that can genuinely be Talker codec tokens."""
    try:
        import torch
        if torch.is_tensor(value):
            return value.numel() > 0 and not value.dtype.is_floating_point
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return value.size > 0 and np.issubdtype(value.dtype, np.integer)
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(x, (int, np.integer)) for x in value)
    )


def _to_codec_array(value):
    if not _is_int_sequence(value):
        return None
    try:
        import torch
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _extract_codec(obj, depth=0, max_depth=4):
    """Extract only explicitly named Talker outputs; never use text token_ids."""
    if obj is None or depth > max_depth:
        return None
    direct = _to_codec_array(obj)
    if direct is not None:
        return direct
    preferred = (
        "codec", "codex", "codec_ids", "codex_ids", "audio_tokens",
        "audio_token_ids", "talker_token_ids", "talker_tokens",
        "speech_tokens", "speech_token_ids",
    )
    if isinstance(obj, dict):
        attrs = obj
    else:
        try:
            attrs = vars(obj)
        except TypeError:
            attrs = {}
    for name in preferred:
        if name in attrs:
            value = _extract_codec(attrs[name], depth + 1, max_depth)
            if value is not None:
                return value
    values = obj if isinstance(obj, (list, tuple)) else attrs.values()
    for value in values:
        candidate = _extract_codec(value, depth + 1, max_depth)
        if candidate is not None:
            return candidate
    return None


class StreamingS2STBackend:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_dir = Path(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        self.im_end_id = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        common = dict(
            model=str(self.model_dir),
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=len(args.gate_devices),
            gpu_memory_utilization=args.gate_gpu_memory_utilization,
            distributed_executor_backend="mp",
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            block_size=16,
            limit_mm_per_prompt={"audio": 64},
            enable_mm_embeds=True,
            skip_mm_profiling=True,
            enable_hidden_state_caching=False,
        )

        # Gate classification and synchronized S2ST decoding are deliberately
        # separate engines.  The gate engine has the two-class output override;
        # the S2ST engine keeps normal text logits and forwards hidden states to
        # the Talker.
        os.environ["VLLM_STREAM_GATE_PATH"] = str(Path(args.gate).resolve())
        os.environ["VLLM_STREAM_GATE_THRESHOLD"] = str(args.threshold)
        self.gate_engine = OmniDRCA2Engine(
            AsyncEngineArgs(**common),
            thinker_visible_devices=args.gate_devices,
        )
        os.environ.pop("VLLM_STREAM_GATE_PATH", None)
        os.environ.pop("VLLM_STREAM_GATE_THRESHOLD", None)

        text_args = AsyncEngineArgs(
            **{
                **common,
                "tensor_parallel_size": len(args.text_devices),
                "gpu_memory_utilization": args.text_gpu_memory_utilization,
            }
        )
        self.text_engine = OmniDRCA2Engine(
            text_args,
            thinker_visible_devices=args.text_devices,
        )

        talker_args = AsyncEngineArgs(
            model=str(self.model_dir),
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.talker_gpu_memory_utilization,
            distributed_executor_backend="mp",
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            block_size=16,
            enable_prompt_embeds=True,
        )
        s2st_thinker_args = AsyncEngineArgs(
            **{
                **common,
                "tensor_parallel_size": len(args.thinker_devices),
                "gpu_memory_utilization": args.thinker_gpu_memory_utilization,
                "logits_processors": [ForcedSequenceLogitsProcessor],
            }
        )
        # A later streaming WRITE has to recreate the Talker causal prefix
        # (previous Thinker hiddens + the sampled codec embedding at each
        # position).  Keep that replay local to this evaluator rather than
        # retaining an opaque cross-request KV cache in shared vLLM.
        self.s2st_engine = CodecHistoryReplayEngine(
            s2st_thinker_args,
            talker_args,
            thinker_visible_devices=args.thinker_devices,
            talker_visible_devices=args.talker_devices,
            codec_embed_path=self.model_dir / "cosyvoice_codec_embed.pt",
        )
        # The installed aligned-Talker engine consumes this request-scoped BOS
        # prefix map but its base constructor does not create it.  The legacy
        # evaluator accidentally obtained the attribute only from its replay
        # subclass.  Deferred mode intentionally disables replay, so install
        # the compatibility container directly on the base engine.
        if not hasattr(self.s2st_engine, "cosyvoice_codec_prefix_ids"):
            self.s2st_engine.cosyvoice_codec_prefix_ids = {}
        self.codec_eos_id = args.codec_eos_id
        self.codec_bos_id = args.codec_bos_id
        self.codec_final_bos_id = args.codec_final_bos_id
        self.codec_pause_id = args.codec_pause_id
        self.codec_boundary_policy = str(args.codec_boundary_policy)
        if (
            self.codec_boundary_policy == BOUNDARY_POLICY_DUAL_BOS
            and self.codec_final_bos_id == self.codec_bos_id
        ):
            raise ValueError("continue and final codec BOS ids must differ")
        self.codec_envelope = bool(args.codec_envelope)
        self.final_eos_only = bool(args.final_eos_only)
        self.codec_history_replay = bool(args.codec_history_replay)
        self.codec_blank_count = int(args.codec_blank_count)
        self.codec_alignment_mask_id = int(args.codec_alignment_mask_id)
        self._next_write_is_final = False
        self.audio_pad_id = int(
            self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        )
        self._segments: list[dict] = []
        self._global_codec_full: np.ndarray | None = None
        self.s2st_engine.enable_global_timeline()
        self.s2st_engine.global_codec_bos_id = int(self.codec_bos_id)
        self.s2st_engine.global_codec_final_bos_id = int(self.codec_final_bos_id)
        self.s2st_engine.global_codec_pause_id = int(self.codec_pause_id)
        self.s2st_engine.global_codec_eos_id = int(self.codec_eos_id)
        self.s2st_engine.global_boundary_policy = self.codec_boundary_policy

    @staticmethod
    def _mm(segments):
        return {"audio_embeds": [segment.contiguous() for segment in segments]}

    @staticmethod
    def _request(engine, prompt, params, request_id=None, **kwargs):
        request_id = request_id or f"stream_s2st_{uuid.uuid4().hex}"
        queue = engine.add_request(request_id, prompt, params, **kwargs)
        outputs = []
        while True:
            output = queue.get(timeout=600)
            if output is None:
                break
            outputs.append(output)
        return outputs

    def _close_consumed_writer_shm_handles(self) -> None:
        """Close writer FDs for SHM segments already consumed by Talker.

        The installed vLLM unlinks each segment in the reader process but keeps
        every writer-side ``SharedMemory`` object in ``_created_shm`` until
        engine shutdown. Streaming creates many requests per utterance, so the
        default 1024 FD limit is exhausted after only a few samples. At the end
        of a synchronized request the Talker has consumed all external data;
        closing the writer handles here is safe and does not unlink live data.
        """
        shared = SharedHiddenBuffer.get_global()
        backend = getattr(shared, "backend", None)
        created = getattr(backend, "_created_shm", None)
        lock = getattr(backend, "_lock", None)
        if not isinstance(created, dict):
            return
        if lock is None:
            items = list(created.items())
            created.clear()
        else:
            with lock:
                items = list(created.items())
                created.clear()
        for _, shm in items:
            try:
                shm.close()
            except Exception:
                pass

    def start_sample(self) -> None:
        self._segments = []
        self._next_write_is_final = False
        self._global_codec_full = None
        self.s2st_engine.reset_global_timeline()

    def set_write_context(self, *, final_write: bool) -> None:
        """Tell Talker whether this WRITE returns to READ or ends the utterance."""
        self._next_write_is_final = bool(final_write)
        self.s2st_engine.set_global_write_context(final_write=final_write)

    @staticmethod
    def _find_subsequence(haystack, needle, start):
        if not needle:
            return start
        stop = len(haystack) - len(needle) + 1
        for index in range(start, max(start, stop)):
            if haystack[index:index + len(needle)] == needle:
                return index
        return -1

    def _build_codec_history_replay(self, prompt_text, audio_segments):
        """Build the same dual-channel prefix layout used by S2ST training."""
        prompt_ids = list(self.tokenizer.encode(
            prompt_text, add_special_tokens=False
        ))
        augmented_ids = list(prompt_ids)
        codec_by_raw_position: dict[int, int] = {}
        cursor = 0
        inserted = 0
        assistant_ids = list(self.tokenizer.encode(
            s2tt.ASSISTANT, add_special_tokens=False
        ))
        turn_end = "<|im_end|>\n"
        turn_end_ids = list(self.tokenizer.encode(
            turn_end, add_special_tokens=False
        ))

        for write_index, history in enumerate(self._segments):
            pattern_ids = list(self.tokenizer.encode(
                s2tt.ASSISTANT + history["text"] + turn_end,
                add_special_tokens=False,
            ))
            pattern_start = self._find_subsequence(
                augmented_ids, pattern_ids, cursor
            )
            if pattern_start < 0:
                raise RuntimeError(
                    "cannot locate prior assistant span while building codec replay: "
                    f"write={write_index} text={history['text']!r}"
                )
            text_start = pattern_start + len(assistant_ids)
            end_start = pattern_start + len(pattern_ids) - len(turn_end_ids)
            text_span = end_start - text_start
            replay_codec = list(np.asarray(
                history["codec"], dtype=np.int64
            ).reshape(-1))
            if self.codec_envelope:
                replay_codec.insert(0, int(self.codec_bos_id))
            # Training uses PAUSE between WRITE spans and EOS only at utterance end.
            replay_codec.append(int(history["boundary_token_id"]))
            audio_span = self.codec_blank_count + len(replay_codec)
            padding = max(0, audio_span - text_span)
            if padding:
                augmented_ids[end_start:end_start] = [
                    self.codec_alignment_mask_id
                ] * padding
                inserted += padding
            for offset, token_id in enumerate(replay_codec):
                raw_position = text_start + self.codec_blank_count + offset
                codec_by_raw_position[raw_position] = int(token_id)
            cursor = pattern_start + len(pattern_ids) + padding

        codec_by_expanded_position: dict[int, int] = {}
        expanded_position = 0
        audio_index = 0
        for raw_position, token_id in enumerate(augmented_ids):
            if raw_position in codec_by_raw_position:
                codec_by_expanded_position[expanded_position] = (
                    codec_by_raw_position[raw_position]
                )
            if token_id == self.audio_pad_id:
                if audio_index >= len(audio_segments):
                    raise RuntimeError(
                        "more <|audio_pad|> tokens than audio embedding segments"
                    )
                expanded_position += int(audio_segments[audio_index].shape[0])
                audio_index += 1
            else:
                expanded_position += 1
        if audio_index != len(audio_segments):
            raise RuntimeError(
                "audio embedding/placeholder mismatch in codec replay: "
                f"placeholders={audio_index} segments={len(audio_segments)}"
            )
        return {
            "prompt_token_ids": augmented_ids,
            "codec_by_expanded_position": codec_by_expanded_position,
            "expanded_prefix_len": expanded_position,
            "prior_writes": len(self._segments),
            "inserted_mask_tokens": inserted,
        }

    def gate_decision(self, prompt_text, segments):
        prompt = TextPrompt(
            prompt=prompt_text,
            multi_modal_data={"audio": self._mm(segments)},
            mm_processor_kwargs={},
        )
        params = SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=1,
            stop_token_ids=[self.im_end_id],
            detokenize=False,
        )
        outputs = self._request(self.gate_engine, prompt, params)
        last = next((x for x in reversed(outputs) if isinstance(x, RequestOutput)), None)
        if last is None:
            raise RuntimeError("gate request returned no RequestOutput")
        token_ids = list(last.outputs[0].token_ids or [])
        if not token_ids or token_ids[0] not in (0, 1):
            raise RuntimeError(f"gate returned invalid class token(s): {token_ids}")
        return token_ids[0] == 1

    def generate_text(self, prompt_text, segments):
        prompt = TextPrompt(
            prompt=prompt_text,
            multi_modal_data={"audio": self._mm(segments)},
            mm_processor_kwargs={},
        )
        params = SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=self.args.max_new_tokens,
            stop_token_ids=[self.im_end_id],
            detokenize=True,
            seed=self.args.seed,
        )
        outputs = self._request(self.text_engine, prompt, params)
        last = next((x for x in reversed(outputs) if isinstance(x, RequestOutput)), None)
        if last is None:
            raise RuntimeError("streaming Thinker WRITE returned no output")
        if last.outputs[0].finish_reason == "length":
            raise RuntimeError("Thinker reached token cap: refusing truncated evaluation")
        return str(last.outputs[0].text or "").replace("<|im_end|>", "").strip()

    def generate_forced_speech(
        self, prompt_text, segments, forced_text, *, record_segment=True
    ):
        # Also cleans handles from a preceding request if its normal return was
        # interrupted after Talker consumption.
        self._close_consumed_writer_shm_handles()
        prompt = TextPrompt(
            prompt=prompt_text,
            multi_modal_data={"audio": self._mm(segments)},
            mm_processor_kwargs={},
        )
        forced_text_ids = list(self.tokenizer.encode(
            forced_text, add_special_tokens=False
        ))
        forced_token_ids = forced_text_ids + [self.im_end_id]
        thinker_params = SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=len(forced_token_ids),
            stop_token_ids=[self.im_end_id],
            detokenize=True,
            seed=self.args.seed,
            extra_args={"forced_token_ids": forced_token_ids},
        )
        talker_params = SamplingParams(
            temperature=self.args.codec_temperature,
            top_p=self.args.codec_top_p,
            top_k=self.args.codec_top_k,
            max_tokens=self.args.max_codec_tokens,
            # The request spans codec-active and codec-empty positions.  EOS
            # sampled from an ignored gap logit must not auto-stop vLLM; the
            # listener terminates only a meaningful EOS under FINAL_BOS.
            stop_token_ids=[],
            ignore_eos=True,
            detokenize=False,
            seed=self.args.seed,
            # BOS is controller-injected and is never a valid prediction.
            # PAUSE is also invalid for historical unified-EOS checkpoints.
            # Masking these logits is the inference counterpart of excluding
            # controller-only tokens from supervised acoustic targets.
            logit_bias={
                int(token_id): -10000.0
                for token_id in (
                    [self.codec_bos_id, self.codec_final_bos_id]
                    + (
                        [self.codec_pause_id]
                        if self.codec_boundary_policy
                        == BOUNDARY_POLICY_UNIFIED_EOS
                        else []
                    )
                )
            },
        )
        request_id = f"stream_s2st_{uuid.uuid4().hex}"
        # The global session must receive this flag before the Thinker listener
        # reaches its aligned-hidden callback.
        self.s2st_engine.set_global_write_context(
            final_write=self._next_write_is_final,
            finalize_only=bool(not record_segment and not forced_text),
        )
        expected_alignment_epoch = self.s2st_engine._global_alignment_epoch + 1
        output_queue = self.s2st_engine.add_request(
            request_id=request_id,
            prompt=prompt,
            params=thinker_params,
            talker_params=talker_params,
        )
        last = None
        deadline = time.monotonic() + 600.0
        while time.monotonic() < deadline:
            # A forced Thinker WRITE can take far longer than 5 seconds on a
            # cold model.  This is a response wait, not a polling cadence.
            item = output_queue.get(timeout=max(0.1, deadline - time.monotonic()))
            if isinstance(item, RequestOutput):
                last = item
                if item.finished:
                    break
        if last is None:
            raise RuntimeError("S2ST write request returned no Thinker output")
        while (
            self.s2st_engine._global_alignment_epoch < expected_alignment_epoch
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        if self.s2st_engine._global_alignment_epoch < expected_alignment_epoch:
            raise RuntimeError("global Talker alignment did not complete after Thinker WRITE")
        decoded_text = str(last.outputs[0].text or "").replace("<|im_end|>", "").strip()
        actual_ids = list(last.outputs[0].token_ids or [])
        if actual_ids and actual_ids[-1] == self.im_end_id:
            actual_ids = actual_ids[:-1]
        if actual_ids != forced_text_ids:
            raise RuntimeError(
                "forced Thinker replay token mismatch: "
                f"expected={forced_text_ids} actual={actual_ids}"
            )
        if self._next_write_is_final:
            global_request_id = self.s2st_engine._global_talker_request_id
            if global_request_id is None:
                raise RuntimeError("final WRITE completed without a global Talker")
            deadline = time.monotonic() + 600.0
            while (
                not self.s2st_engine._global_final_eos_observed
                and self.s2st_engine._global_failure is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            if self.s2st_engine._global_failure is not None:
                raise RuntimeError(
                    "global Talker failed: " + self.s2st_engine._global_failure
                )
            codec = np.asarray(
                self.s2st_engine._global_codec_tokens, dtype=np.int64
            ).reshape(-1).copy()
            if (
                not self.s2st_engine._global_final_eos_observed
                or len(codec) != self.s2st_engine._global_final_eos_token_count
                or not len(codec)
                or int(codec[-1]) != self.codec_eos_id
            ):
                raise RuntimeError("global Talker did not reach final codec EOS")
            self._global_codec_full = codec
        if record_segment:
            self._segments.append({
                "text": forced_text,
                "decoded_text": decoded_text,
                "global_final_write": bool(self._next_write_is_final),
            })
        self._close_consumed_writer_shm_handles()
        return forced_text

    def finalize_without_new_write(self, prompt_text, segments) -> None:
        """Drain the prior envelope when source-end emits no new text."""
        if not self._segments:
            raise RuntimeError("cannot finalize an utterance with no prior WRITE")
        if self.codec_boundary_policy != BOUNDARY_POLICY_UNIFIED_EOS:
            raise RuntimeError(
                "empty final WRITE is only supported by unified_eos checkpoints"
            )
        self._next_write_is_final = True
        self.generate_forced_speech(
            prompt_text,
            segments,
            "",
            record_segment=False,
        )
        self._segments[-1]["global_final_write"] = True

    def finish_sample(self):
        if self._global_codec_full is None:
            raise RuntimeError("sample ended before global Talker emitted codec EOS")
        if not self._segments:
            raise RuntimeError("sample completed without a WRITE/Talker segment")
        codec_unit = int(getattr(
            self.s2st_engine._get_talker_hf_config(), "codec_unit", -1
        ))
        if codec_unit <= 0:
            raise RuntimeError(f"invalid Talker codec_unit={codec_unit}")
        acoustic_segments: list[list[int]] = []
        current: list[int] | None = None
        current_bos: int | None = None
        for token in np.asarray(self._global_codec_full, dtype=np.int64).reshape(-1):
            token = int(token)
            valid_bos_ids = {self.codec_bos_id}
            if self.codec_boundary_policy == BOUNDARY_POLICY_DUAL_BOS:
                valid_bos_ids.add(self.codec_final_bos_id)
            if token in valid_bos_ids:
                if current is not None:
                    raise RuntimeError("codec BOS appeared before the prior boundary")
                current = []
                current_bos = token
            elif token in (self.codec_pause_id, self.codec_eos_id):
                if current is None:
                    raise RuntimeError("codec boundary appeared without an open envelope")
                if self.codec_boundary_policy == BOUNDARY_POLICY_UNIFIED_EOS:
                    expected = self.codec_eos_id
                else:
                    expected = (
                        self.codec_eos_id
                        if current_bos == self.codec_final_bos_id
                        else self.codec_pause_id
                    )
                if token != expected:
                    raise RuntimeError(
                        f"codec BOS/boundary mismatch: bos={current_bos} "
                        f"boundary={token} expected={expected}"
                    )
                acoustic_segments.append(current)
                current = None
                current_bos = None
            elif 0 <= token < codec_unit:
                if current is None:
                    raise RuntimeError(
                        f"acoustic codec token {token} appeared before envelope BOS"
                    )
                current.append(token)
            else:
                raise RuntimeError(f"unexpected non-codec Talker token {token}")
        if current is not None:
            raise RuntimeError("Talker stream ended with an unterminated codec envelope")
        if len(acoustic_segments) != len(self._segments):
            raise RuntimeError(
                "training-envelope boundary mismatch: "
                f"codec_boundary_count={len(acoustic_segments)} "
                f"thinker_writes={len(self._segments)}"
            )
        final_write_flags = [
            bool(segment.get("global_final_write")) for segment in self._segments
        ]
        expected_final_write_flags = [False] * (len(self._segments) - 1) + [True]
        if final_write_flags != expected_final_write_flags:
            raise RuntimeError(
                "invalid source-end WRITE control sequence: "
                f"actual={final_write_flags} expected={expected_final_write_flags}"
            )
        annotated_segments = []
        for segment, acoustic in zip(self._segments, acoustic_segments):
            annotated = dict(segment)
            annotated["codec_len"] = len(acoustic)
            annotated_segments.append(annotated)
        acoustic = np.asarray(
            [token for segment in acoustic_segments for token in segment],
            dtype=np.int64,
        )
        return acoustic, annotated_segments

    def close(self):
        self._close_consumed_writer_shm_handles()
        self.gate_engine.shutdown()
        self.text_engine.shutdown()
        self.s2st_engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codec-dir", required=True)
    parser.add_argument("--gate-devices", type=json.loads, default=[0, 1])
    parser.add_argument("--thinker-devices", type=json.loads, default=[2, 3])
    parser.add_argument("--talker-devices", type=json.loads, default=[4])
    parser.add_argument("--text-devices", type=json.loads, default=[5, 6])
    parser.add_argument("--encoder-device", default="cuda:0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--encoder-chunk-ms", type=float, default=640.0)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-codec-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gate-gpu-memory-utilization", type=float, default=0.42)
    parser.add_argument("--thinker-gpu-memory-utilization", type=float, default=0.42)
    parser.add_argument("--talker-gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--text-gpu-memory-utilization", type=float, default=0.42)
    parser.add_argument("--codec-temperature", type=float, default=0.8)
    parser.add_argument("--codec-top-p", type=float, default=0.8)
    parser.add_argument("--codec-top-k", type=int, default=25)
    parser.add_argument(
        "--codec-eos-id", type=int, default=TRAININFER_EOS_ID
    )
    parser.add_argument(
        "--codec-bos-id", type=int, default=TRAININFER_CONTINUE_BOS_ID
    )
    parser.add_argument(
        "--codec-final-bos-id",
        type=int,
        default=TRAININFER_FINAL_BOS_ID,
        help="Controller-injected BOS for the utterance-final codec envelope.",
    )
    parser.add_argument(
        "--codec-boundary-policy",
        choices=(BOUNDARY_POLICY_UNIFIED_EOS, BOUNDARY_POLICY_DUAL_BOS),
        default=BOUNDARY_POLICY_UNIFIED_EOS,
        help=(
            "unified_eos matches historical checkpoints trained with EOS at "
            "every WRITE; dual_bos_pause_eos matches the 0903 dual-BOS model."
        ),
    )
    parser.add_argument(
        "--codec-envelope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expect BOS before acoustic codec tokens and exclude it from FM input.",
    )
    parser.add_argument(
        "--codec-pause-id",
        type=int,
        default=TRAININFER_PAUSE_ID,
        help="Non-acoustic boundary token used between WRITE spans.",
    )
    parser.add_argument(
        "--final-eos-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use PAUSE at intermediate WRITEs and EOS only at utterance end. "
            "Required by this train/infer-consistent runner."
        ),
    )
    parser.add_argument(
        "--codec-history-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Legacy per-WRITE replay mode. It must remain disabled because this "
            "runner keeps one utterance-global Talker request and KV cache."
        ),
    )
    parser.add_argument(
        "--codec-blank-count", type=int, default=TRAININFER_BLANK_COUNT
    )
    parser.add_argument(
        "--codec-alignment-mask-id",
        type=int,
        default=TRAININFER_ALIGNMENT_MASK_ID,
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    required_contract = {
        "--codec-bos-id": (args.codec_bos_id, TRAININFER_CONTINUE_BOS_ID),
        "--codec-eos-id": (args.codec_eos_id, TRAININFER_EOS_ID),
        "--codec-blank-count": (
            args.codec_blank_count,
            TRAININFER_BLANK_COUNT,
        ),
        "--codec-alignment-mask-id": (
            args.codec_alignment_mask_id,
            TRAININFER_ALIGNMENT_MASK_ID,
        ),
    }
    for option, (actual, expected) in required_contract.items():
        if actual != expected:
            parser.error(
                f"{option}={actual} violates the streaming training contract; "
                f"expected {expected}"
            )
    if args.codec_boundary_policy == BOUNDARY_POLICY_UNIFIED_EOS:
        if args.final_eos_only:
            parser.error(
                "unified_eos requires --no-final-eos-only because EOS is the "
                "trained boundary at every WRITE"
            )
    else:
        if args.codec_final_bos_id != TRAININFER_FINAL_BOS_ID:
            parser.error(
                f"dual_bos_pause_eos requires --codec-final-bos-id "
                f"{TRAININFER_FINAL_BOS_ID}"
            )
        if args.codec_pause_id != TRAININFER_PAUSE_ID:
            parser.error(
                f"dual_bos_pause_eos requires --codec-pause-id "
                f"{TRAININFER_PAUSE_ID}"
            )
        if not args.final_eos_only:
            parser.error("dual_bos_pause_eos requires --final-eos-only")
    if not args.codec_envelope:
        parser.error("--codec-envelope is required by the 0903 training contract")
    if args.codec_history_replay:
        parser.error(
            "--codec-history-replay is incompatible with the utterance-global "
            "Talker; use --no-codec-history-replay"
        )
    if args.encoder_chunk_ms != TRAININFER_ENCODER_CHUNK_MS:
        parser.error(
            f"--encoder-chunk-ms must be {TRAININFER_ENCODER_CHUNK_MS:g}"
        )
    actual_lookahead_ms = float(getattr(s2tt, "STREAMING_LOOKAHEAD_MS", -1.0))
    if abs(actual_lookahead_ms - TRAININFER_WAVEFORM_LOOKAHEAD_MS) > 1e-6:
        parser.error(
            "the train/infer-consistent entry is run_block640_lookahead160.py; "
            f"expected waveform lookahead {TRAININFER_WAVEFORM_LOOKAHEAD_MS} ms, "
            f"got {actual_lookahead_ms} ms"
        )
    # Global-timeline mode preserves the active Talker state across WRITE
    # turns, so the legacy per-WRITE codec-history reconstruction is neither
    # needed nor allowed to create local drain padding.

    # OmniDRCA2Engine reads these at Talker startup rather than from this
    # evaluator's argparse namespace. Keeping the two paths in sync is
    # essential: a CLI value of five must result in exactly five delayed
    # hidden frames before the first codec step, not the engine default (six).
    os.environ["CODEX_AUDIO_BLANK_TOKEN_COUNT"] = str(args.codec_blank_count)
    os.environ["CODEX_ALIGNMENT_MASK_TOKEN_ID"] = str(args.codec_alignment_mask_id)
    # The synchronized Thinker lazily loads pre_matching.pt from the active
    # HF checkpoint during its profiling pass.  The suite entrypoints set this
    # already, but make direct invocation of this reusable runner equivalent.
    os.environ.setdefault("OMNIDRCA2_MODEL_DIR", str(args.model))
    print(
        "[TRUE STREAMING GLOBAL-TIMELINE CONTRACT] each WRITE sees only "
        "available source audio; one Talker waits for real Thinker hidden "
        f"frames after its {args.codec_blank_count}-frame prefill, and only "
        "the final source-complete WRITE may append pad-token drain frames; "
        f"boundary_policy={args.codec_boundary_policy}; "
        f"BOS={args.codec_bos_id}, final_BOS={args.codec_final_bos_id}, "
        f"PAUSE={args.codec_pause_id}, EOS={args.codec_eos_id}",
        flush=True,
    )

    rows = [json.loads(x) for x in Path(args.input).read_text().splitlines() if x.strip()]
    end = len(rows) if args.end is None else min(args.end, len(rows))
    selected = list(enumerate(rows))[args.start:end]
    output = Path(args.output)
    codec_dir = Path(args.codec_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    codec_dir.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size:
        raise FileExistsError(f"refusing append/duplicate-prone output: {output}")

    encoder = s2tt.FullRecordEncoder(
        Path(args.model),
        args.encoder_device,
        true_streaming=True,
        chunk_ms=args.encoder_chunk_ms,
    )
    backend = StreamingS2STBackend(args)
    try:
        with output.open("w", encoding="utf-8") as writer:
            for position, (index, row) in enumerate(selected, 1):
                started = time.time()
                try:
                    backend.start_sample()
                    result = s2tt.evaluate_one_streaming_thinker_deferred_talker(
                        index, row, encoder, backend, args
                    )
                    codec, codec_segments = backend.finish_sample()
                    codec_path = codec_dir / f"{result['key']}.npy"
                    np.save(codec_path, codec)
                    codec_control_path = codec_dir / f"{result['key']}.control.npy"
                    np.save(codec_control_path, backend._global_codec_full)
                    result.update({
                        "codec_path": str(codec_path),
                        "codec_control_path": str(codec_control_path),
                        "codec_len": int(len(codec)),
                        "codec_control_ends_with_eos": True,
                        "codec_envelope_count": len(codec_segments),
                        "codec_segments": codec_segments,
                        "codec_boundary_policy": str(args.codec_boundary_policy),
                        "codec_continue_bos_id": int(args.codec_bos_id),
                        "codec_final_bos_id": int(args.codec_final_bos_id),
                        "codec_pause_id": int(args.codec_pause_id),
                        "codec_eos_id": int(args.codec_eos_id),
                        "codec_blank_count": int(args.codec_blank_count),
                        "codec_alignment_mask_id": int(args.codec_alignment_mask_id),
                        "elapsed_s": time.time() - started,
                        "inference_contract": (
                            "streaming_global_talker_unified_eos_traininfer_v1"
                            if args.codec_boundary_policy
                            == BOUNDARY_POLICY_UNIFIED_EOS
                            else "streaming_global_talker_dualbos_traininfer_v2"
                        ),
                    })
                except Exception as exc:
                    result = {"index": index, "error": repr(exc)}
                writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                writer.flush()
                print(json.dumps({
                    "progress": f"{position}/{len(selected)}",
                    "index": index,
                    "writes": len(result.get("pieces", [])),
                    "codec_len": result.get("codec_len"),
                    "error": result.get("error"),
                }, ensure_ascii=False), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
