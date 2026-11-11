#!/usr/bin/env python3
"""Talker-first true-streaming S2ST training entrypoint."""

from __future__ import annotations

import os
import types
import warnings

import torch

os.environ.setdefault("STREAMING_DEBUG_DISABLE_GATE", "1")
# The inherited offline collator truncates token tensors to SEQ_LENGTH but
# leaves audio feature metadata untouched.  Reject such S2ST packs in the
# streaming loader so audio embeddings can never be paired with truncated
# placeholder spans; the dataset's existing retry path supplies another bin.
os.environ.setdefault("STREAMING_FULL_RECORD_REJECT_TRUNCATED_AUDIO", "1")

if os.environ.get("STREAMING_S2ST_BLOCK640_LOOKAHEAD160", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    # Importing this entrypoint installs the isolated 640-ms encoder with
    # layer-0-only 160-ms lookahead and the block-causal adaptor before the
    # offline FunBaseModel class is imported.  Loading a checkpoint trained
    # with that frontend through the generic streaming entrypoint silently
    # changes its forward contract.
    from streaming import pretrain_block640_lookahead160 as streaming
else:
    from streaming import pretrain_streaming as streaming
from streaming.data import full_record_audio as full_record_audio
from streaming.model.read_write_gate import attach_streaming_gate
from streaming.s2st_exploration.cosyvoice_zarr_overlay import (
    register_cosyvoice_zarr_overlay,
)
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain, print_rank_0


def _install_s2st_source_only_full_record_grouping() -> None:
    """Keep target TTS audio out of the source encoder's full-record grouping."""
    original_group_specs = full_record_audio._group_specs
    if getattr(original_group_specs, "_streaming_s2st_source_only", False):
        return

    def source_only_group_specs(packed_items, dataset_class, pattern, use_oss_cache):
        source_only_items = []
        for item in packed_items:
            source_item = dict(item)
            # The shared S2TT full-record adapter must see only source speech.
            # The private S2ST loader still receives the original item later,
            # so assistant audio remains available as codec supervision.
            source_item["assistant"] = [None] * len(item["assistant"])
            source_only_items.append(source_item)
        return original_group_specs(
            source_only_items, dataset_class, pattern, use_oss_cache
        )

    source_only_group_specs._streaming_s2st_source_only = True
    full_record_audio._group_specs = source_only_group_specs


_install_s2st_source_only_full_record_grouping()


def _install_codec_input_token_dropout(model) -> None:
    """Regularize teacher-forced codec history without changing inference."""
    probability = float(
        os.environ.get("STREAMING_S2ST_CODEC_INPUT_TOKEN_DROPOUT", "0")
    )
    if not 0.0 <= probability < 1.0:
        raise ValueError(
            "STREAMING_S2ST_CODEC_INPUT_TOKEN_DROPOUT must be in [0, 1)"
        )
    if probability == 0.0:
        return

    threshold = int(probability * 1_000_000)

    def drop_codec_tokens(module, inputs, output):
        del module
        if not model.training:
            return output
        token_ids = inputs[0].to(dtype=torch.int64)
        sequence = torch.arange(
            token_ids.shape[-1], device=token_ids.device, dtype=torch.int64
        )
        view_shape = [1] * token_ids.ndim
        view_shape[-1] = token_ids.shape[-1]
        sequence = sequence.view(view_shape)
        batch = torch.arange(
            token_ids.shape[0], device=token_ids.device, dtype=torch.int64
        )
        batch_shape = [1] * token_ids.ndim
        batch_shape[0] = token_ids.shape[0]
        batch = batch.view(batch_shape)
        args = get_args()
        iteration = int(
            getattr(args, "iteration", getattr(args, "curr_iteration", 0))
        )
        # A stateless mask keeps tensor-parallel ranks identical even when their
        # process-local RNG streams differ. Drop complete codec-token vectors.
        hashed = (
            token_ids * 1_103_515_245
            + sequence * 12_345
            + batch * 2_654_435_761
            + iteration * 1_013_904_223
        ) & 0x7FFFFFFF
        drop = (hashed % 1_000_000) < threshold
        return output.masked_fill(drop.unsqueeze(-1), 0.0) / (1.0 - probability)

    model.cosyvoice_codec_embed.register_forward_hook(drop_codec_tokens)
    print_rank_0(
        "[STREAMING S2ST] enabled deterministic whole-token codec-input "
        f"dropout p={probability:.3f} (training only)"
    )


def _install_talker_packed_attention_isolation(model) -> None:
    """Give the replicated HF Talker the same packed-sample boundaries.

    Megatron's Thinker consumes ``packed_seq_params`` directly.  The embedded
    HuggingFace Qwen2 Talker does not: without this adapter it uses one causal
    sequence across every independently packed utterance.  At TP=1 we can pass
    a block-diagonal additive mask and reset Talker RoPE positions per sample.
    """
    enabled = os.environ.get(
        "STREAMING_S2ST_TALKER_PACK_ISOLATION", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return
    args = get_args()
    if int(args.tensor_model_parallel_size) != 1:
        raise ValueError(
            "Talker packed-attention isolation requires TP=1; TP>1 also "
            "sequence-shards the replicated HF Talker and is unsupported"
        )

    original_model_forward = model.forward
    original_talker_forward = model.cosyvoice_llm.forward

    def model_forward_with_packed_context(self, *forward_args, **forward_kwargs):
        packed = forward_kwargs.get("packed_seq_params")
        self.cosyvoice_llm._streaming_packed_seq_params = packed
        return original_model_forward(*forward_args, **forward_kwargs)

    def talker_forward_with_isolation(self, *forward_args, **forward_kwargs):
        inputs_embeds = forward_kwargs.get("inputs_embeds")
        packed = getattr(self, "_streaming_packed_seq_params", None)
        if inputs_embeds is None or packed is None:
            raise RuntimeError(
                "Talker pack isolation requires inputs_embeds and packed_seq_params"
            )
        cu_seqlens = packed.get("cu_seqlens_q")
        if cu_seqlens is None:
            raise RuntimeError("packed_seq_params has no cu_seqlens_q")
        sequence_length = int(inputs_embeds.shape[1])
        boundaries = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
        boundaries = [value for value in boundaries if 0 <= value <= sequence_length]
        boundaries = sorted(set(boundaries))
        if not boundaries or boundaries[0] != 0:
            boundaries.insert(0, 0)
        if boundaries[-1] != sequence_length:
            boundaries.append(sequence_length)
        if len(boundaries) < 2:
            raise RuntimeError(f"invalid packed boundaries: {boundaries}")

        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        minimum = torch.finfo(dtype).min
        additive_mask = torch.full(
            (1, 1, sequence_length, sequence_length),
            minimum,
            dtype=dtype,
            device=device,
        )
        position_ids = torch.empty(
            (1, sequence_length), dtype=torch.long, device=device
        )
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= begin:
                continue
            length = end - begin
            block = torch.triu(
                torch.full((length, length), minimum, dtype=dtype, device=device),
                diagonal=1,
            )
            additive_mask[0, 0, begin:end, begin:end] = block
            position_ids[0, begin:end] = torch.arange(length, device=device)

        attention_types = set(getattr(self.config, "layer_types", ["full_attention"]))
        forward_kwargs["attention_mask"] = {
            attention_type: additive_mask for attention_type in attention_types
        }
        forward_kwargs["position_ids"] = position_ids
        if not getattr(self, "_streaming_pack_isolation_logged", False):
            self._streaming_pack_isolation_logged = True
            print_rank_0(
                "[STREAMING S2ST TALKER PACK ISOLATION] "
                f"sequence={sequence_length} samples={len(boundaries) - 1} "
                f"boundaries={boundaries[:24]}"
            )
        return original_talker_forward(*forward_args, **forward_kwargs)

    model.forward = types.MethodType(model_forward_with_packed_context, model)
    model.cosyvoice_llm.forward = types.MethodType(
        talker_forward_with_isolation, model.cosyvoice_llm
    )
    print_rank_0("[STREAMING S2ST] enabled Talker packed-attention isolation")


def _install_codec_envelope_label_shift(model) -> None:
    """Install streaming-only BOS/codec/PAUSE-or-EOS supervision."""
    enabled = os.environ.get("STREAMING_S2ST_CODEC_ENVELOPE", "0").lower() in (
        "1", "true", "yes", "on"
    )
    if not enabled:
        return
    codec_bos = int(os.environ.get("STREAMING_S2ST_CODEC_BOS_TOKEN_ID", "59049"))
    codec_final_bos = int(
        os.environ.get("STREAMING_S2ST_CODEC_FINAL_BOS_TOKEN_ID", "59052")
    )
    if codec_final_bos == codec_bos:
        raise ValueError("continue and final codec BOS token ids must differ")
    codec_eos = int(os.environ.get("STREAMING_S2ST_CODEC_EOS_TOKEN_ID", "59050"))
    codec_pause = int(os.environ.get("STREAMING_S2ST_CODEC_PAUSE_TOKEN_ID", "59051"))
    codec_mask = int(
        os.environ.get("STREAMING_S2ST_CODEC_ALIGNMENT_MASK_ID", "-60")
    )

    def shift_envelope_labels(self, input_ids_audio, labels_ids_audio):
        # convert_batch_to_megatron has already shifted labels one position to
        # the left.  Both BOS variants are controller-injected inputs and must
        # never be next-token targets.  Only -50 codec placeholders need to be
        # replaced by the corresponding token from input_ids_audio[:, 1:].
        shifted_labels = labels_ids_audio.clone()
        raw_bos_label_count = int(
            (
                (shifted_labels == codec_bos)
                | (shifted_labels == codec_final_bos)
            ).sum().detach().cpu()
        )
        if raw_bos_label_count:
            raise RuntimeError(
                "controller-injected codec BOS unexpectedly has prediction loss: "
                f"count={raw_bos_label_count}"
            )
        raw_eos_count = int((shifted_labels == codec_eos).sum().detach().cpu())
        raw_pause_count = int((shifted_labels == codec_pause).sum().detach().cpu())
        placeholder_count = int((labels_ids_audio == -50).sum().detach().cpu())

        # Dynamic collation can truncate the final sample in a packed batch.
        # Drop only an incomplete trailing envelope; all preceding complete
        # BOS...EOS segments remain supervised.  Any non-tail structural error
        # is still fatal.
        dropped_tail_envelopes = 0
        complete_continue_bos = 0
        complete_final_bos = 0
        complete_pause = 0
        complete_eos = 0
        cleared_stale_boundaries = 0
        for row_index in range(shifted_labels.shape[0]):
            row = input_ids_audio[row_index]
            boundary_positions = torch.nonzero(
                (row == codec_bos)
                | (row == codec_final_bos)
                | (row == codec_pause)
                | (row == codec_eos),
                as_tuple=False,
            ).flatten().detach().cpu().tolist()
            open_bos = None
            open_bos_token = None
            last_closed_position = None
            last_closed_token = None
            for position in boundary_positions:
                token = int(row[position].detach().cpu())
                if token in (codec_bos, codec_final_bos):
                    if open_bos is not None:
                        # A packed row can contain a locally truncated codec
                        # envelope followed by the next intact sample.  Mask
                        # only that incomplete span and restart at this BOS.
                        shifted_labels[row_index, open_bos:position] = -100
                        dropped_tail_envelopes += 1
                    open_bos = position
                    open_bos_token = token
                else:
                    if open_bos is None:
                        # Dynamic speech-token replacement can shorten the
                        # final placeholder run.  It moves the supervised
                        # boundary forward, but the legacy replacement code
                        # leaves the old, now-unsupervised boundary input at
                        # the original tail position.  Clear only this exact,
                        # provable duplicate: same just-closed boundary, zero
                        # or more mask positions between them, and no label at
                        # the stale position.  Any other orphan remains fatal.
                        stale_duplicate = (
                            last_closed_position is not None
                            and token == last_closed_token
                            and position > last_closed_position
                            and int(
                                shifted_labels[row_index, position]
                                .detach()
                                .cpu()
                            ) < 0
                            and bool(
                                torch.all(
                                    row[last_closed_position + 1 : position]
                                    == codec_mask
                                ).detach().cpu()
                            )
                        )
                        if stale_duplicate:
                            row[position] = codec_mask
                            shifted_labels[row_index, position] = -100
                            cleared_stale_boundaries += 1
                            continue
                        window_start = max(0, position - 12)
                        window_end = min(int(row.numel()), position + 13)
                        input_window = row[window_start:window_end].detach().cpu().tolist()
                        label_window = shifted_labels[
                            row_index, window_start:window_end
                        ].detach().cpu().tolist()
                        raise RuntimeError(
                            "codec PAUSE/EOS without BOS: "
                            f"row={row_index} boundary={position} token={token} "
                            f"window=[{window_start}:{window_end}] "
                            f"input={input_window} labels={label_window}"
                        )
                    expected_boundary = (
                        codec_eos if open_bos_token == codec_final_bos else codec_pause
                    )
                    if token != expected_boundary:
                        raise RuntimeError(
                            "injected codec BOS/boundary mismatch: "
                            f"row={row_index} bos={open_bos_token} "
                            f"boundary={token} expected={expected_boundary}"
                        )
                    if open_bos_token == codec_final_bos:
                        complete_final_bos += 1
                    else:
                        complete_continue_bos += 1
                    if token == codec_eos:
                        complete_eos += 1
                    else:
                        complete_pause += 1
                    last_closed_position = position
                    last_closed_token = token
                    open_bos = None
                    open_bos_token = None
            if open_bos is not None:
                shifted_labels[row_index, open_bos:] = -100
                dropped_tail_envelopes += 1

        bos_count = complete_continue_bos + complete_final_bos
        eos_count = complete_eos
        pause_count = complete_pause
        boundary_count = eos_count + pause_count
        if bos_count != boundary_count or bos_count == 0:
            raise RuntimeError(
                "invalid codec envelope batch: "
                f"raw_bos_labels={raw_bos_label_count} raw_pause={raw_pause_count} "
                f"raw_eos={raw_eos_count} complete_bos={bos_count} "
                f"complete_pause={pause_count} complete_eos={eos_count} "
                f"placeholders={placeholder_count}"
            )
        if not getattr(self, "_streaming_codec_envelope_debug_logged", False):
            self._streaming_codec_envelope_debug_logged = True
            print_rank_0(
                "[STREAMING S2ST ENVELOPE BATCH] "
                f"continue_bos={complete_continue_bos} "
                f"final_bos={complete_final_bos} pause={pause_count} eos={eos_count} "
                f"placeholders={placeholder_count} "
                f"cleared_stale_boundaries={cleared_stale_boundaries} "
                f"dropped_incomplete_tail={dropped_tail_envelopes} "
                f"sequence_shape={tuple(labels_ids_audio.shape)}"
            )

        placeholder_mask = shifted_labels == -50
        input_id_shift = torch.zeros_like(input_ids_audio)
        input_id_shift[:, :-1] = input_ids_audio[:, 1:]
        # When the extracted codec sequence is shorter than its reserved
        # placeholder run, _replace_audio_placeholders moves its PAUSE/EOS
        # boundary forward.  The final placeholder then predicts that boundary;
        # Either injected BOS and unknown control values remain invalid.
        valid_boundary_target = (
            (input_id_shift == codec_pause) | (input_id_shift == codec_eos)
        )
        invalid_codec_target = placeholder_mask & (
            (input_id_shift < 0)
            | ((input_id_shift >= codec_bos) & (~valid_boundary_target))
        )
        if torch.any(invalid_codec_target):
            bad_count = int(invalid_codec_target.sum().detach().cpu())
            raise RuntimeError(
                f"invalid shifted codec targets inside complete envelopes: {bad_count}"
            )
        result = torch.where(placeholder_mask, input_id_shift, shifted_labels)

        # If PAUSE/EOS was moved onto an input placeholder position, the
        # following label is a duplicate self-label. Keep the preceding
        # placeholder -> boundary target and mask the duplicate.
        previous_placeholder = torch.zeros_like(placeholder_mask)
        previous_placeholder[:, 1:] = placeholder_mask[:, :-1]
        duplicate_boundary_self_label = (
            previous_placeholder
            & (~placeholder_mask)
            & (
                ((shifted_labels == codec_pause) & (input_ids_audio == codec_pause))
                | ((shifted_labels == codec_eos) & (input_ids_audio == codec_eos))
            )
        )
        return torch.where(
            duplicate_boundary_self_label,
            torch.tensor(
                -100,
                device=result.device,
                dtype=result.dtype,
            ),
            result,
        )

    model._shift_cosyvoice_audio_labels = types.MethodType(
        shift_envelope_labels, model
    )
    print_rank_0(
        "[STREAMING S2ST] codec envelope labels enabled: "
        f"CONTINUE_BOS={codec_bos}, FINAL_BOS={codec_final_bos}, "
        f"PAUSE={codec_pause}, EOS={codec_eos}"
    )


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    if args.pipeline_model_parallel_size != 1 or args.context_parallel_size != 1:
        raise ValueError("streaming S2ST v1 requires PP=1 and CP=1")
    model = streaming.offline.model_provider(
        pre_process=pre_process, post_process=post_process
    )
    streaming.register_encoder_weight_overlay(
        model, streaming._streaming_repo_root / "streaming"
    )
    streaming.install_full_record_encoder_split(model)

    # Keep the trained READ/WRITE policy in the checkpoint for later joint
    # training/inference, but do not optimize it during Talker-first training.
    attach_streaming_gate(
        model,
        bottleneck=int(os.environ.get("STREAM_GATE_BOTTLENECK", "256")),
        dropout=float(os.environ.get("STREAM_GATE_DROPOUT", "0.1")),
    )
    for parameter in model.streaming_gate.parameters():
        parameter.requires_grad = False

    register_cosyvoice_zarr_overlay(model)
    _install_codec_envelope_label_shift(model)
    _install_codec_input_token_dropout(model)
    _install_talker_packed_attention_isolation(model)
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith(
            (
                "cosyvoice_llm.",
                "cosyvoice_llm_input_proj.",
                "cosyvoice_codec_embed.",
                "cosyvoice_llm_codec_head.",
            )
        )
    ]
    if unexpected:
        raise RuntimeError(
            "Talker-first stage left non-Talker parameters trainable: "
            + repr(unexpected[:20])
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank_0(
        f"[STREAMING S2ST] Talker-first trainable parameters={trainable:,}; "
        "encoder/Thinker/gate/FM frozen"
    )
    return model


if __name__ == "__main__":
    streaming._patch_torch_dist_writer_compat()
    warnings.filterwarnings(
        "ignore",
        message="The given NumPy array is not writable.*",
        category=UserWarning,
    )
    provider = streaming.build_streaming_data_iterators
    provider.is_distributed = True
    pretrain(
        provider,
        model_provider,
        ModelType.encoder_or_decoder,
        streaming.forward_step,
        args_defaults={"tokenizer_type": "QWenVLTokenizer"},
        use_train_valid_test_dataset_provider_as_data_iterators=True,
        scale_lr_cond=streaming.offline.scale_lr_cond,
    )
