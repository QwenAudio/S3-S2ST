import codecs
import logging
import os
import random
import tempfile
import time
from contextlib import contextmanager
from distutils.version import LooseVersion
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
from torch.nn.utils.rnn import pad_sequence

from funasr.losses.label_smoothing_loss import LabelSmoothingLoss  # noqa: H301
from funasr.metrics.common import ErrorCalculator
from funasr.models.ctc.ctc import CTC
from funasr.models.llm_asr.tts_models.ctc_alignment import ctc_forced_align
from funasr.models.paraformer2.nets_utils import get_minimal_ctc_task_io, th_accuracy
from funasr.models.transducer.beam_search_transducer import Hypothesis as HypothesisTransducer
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.register import tables
from funasr.tokenizer.token_id_converter import TokenIDConverter
from funasr.train_utils.device_funcs import force_gatherable, to_device
from funasr.utils import postprocess_utils
from funasr.utils.load_utils import load_audio_text_image_video

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class AlignformerUnified(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[torch.nn.Module],
        specaug: Optional[torch.nn.Module],
        normalize: Optional[torch.nn.Module],
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        decoder_model2: torch.nn.Module,
        ctc: CTC,
        ctc_weight: float = 1.0,
        att_weight: float = 1.0,
        pre_loss_att_weight: float = 0.0,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        predictor_bias: int = 0,
        sampling_ratio: float = 0.2,
        share_embedding: bool = False,
        preencoder: Optional[torch.nn.Module] = None,
        postencoder: Optional[torch.nn.Module] = None,
        use_1st_decoder_loss: bool = False,
        decoder_chunk_size: int = 1,
        num_left_chunks: int = -1,
        num_right_chunks: int = 0,
        decoder_chunk_size2: int = 1,
        num_left_chunks2: int = -1,
        num_right_chunks2: int = 0,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = blank_id
        self.sos = vocab_size - 1 if sos is None else sos
        self.eos = vocab_size - 1 if eos is None else eos
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.att_weight = att_weight
        self.pre_loss_att_weight = pre_loss_att_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(vocab_size, self.encoder.output_size())

        self.error_calculator = None

        self.decoder = decoder
        self.decoder_model2 = decoder_model2

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(token_list, sym_space, sym_blank, report_cer, report_wer)

        if ctc_weight == 0.0:
            self.ctc = None
            self.ctc_model2 = None
        else:
            self.ctc = ctc
            self.ctc_model2 = CTC(odim=vocab_size, encoder_output_size=self.encoder.output_size())

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.predictor_bias = predictor_bias
        self.sampling_ratio = sampling_ratio
        self.step_cur = 0

        self.share_embedding = share_embedding
        if self.share_embedding:
            self.decoder.embed = None
            self.decoder_model2.embed = None

        self.use_1st_decoder_loss = use_1st_decoder_loss

        self.decoder_chunk_size = decoder_chunk_size
        self.num_left_chunks = num_left_chunks
        self.num_right_chunks = num_right_chunks

        self.decoder_chunk_size2 = decoder_chunk_size2
        self.num_left_chunks2 = num_left_chunks2
        self.num_right_chunks2 = num_right_chunks2

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert speech.shape[0] == speech_lengths.shape[0] == text.shape[0] == text_lengths.shape[0], (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
        )
        batch_size = speech.shape[0]
        self.step_cur += 1
        # for data-parallel
        text = text[:, : text_lengths.max()]
        speech = speech[:, : speech_lengths.max()]

        # 1. Encoder
        if hasattr(self.encoder, "overlap_chunk_cls") and hasattr(self.encoder, "overlap_chunk_cls2"):
            ind_1st = self.encoder.overlap_chunk_cls.random_choice(self.training, decoding_ind=None)
            encoder_out, encoder_out_lens, encoder_out_tp = self.encode(
                speech, speech_lengths, ind_1st=ind_1st, ind_2st=ind_1st
            )
            self.decoder_chunk_size = self.encoder.overlap_chunk_cls.get_chunk_size(ind=ind_1st)[1]
            self.decoder_chunk_size2 = self.encoder.overlap_chunk_cls2.get_chunk_size(ind=ind_1st)[1]
        else:
            encoder_out, encoder_out_lens, encoder_out_tp = self.encode(speech, speech_lengths)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, pre_loss_att, acc_att, cer_att, wer_att, acc_ctc = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(encoder_out, encoder_out_lens, text, text_lengths)

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

            loss_ctc_2, cer_ctc_2 = self._calc_ctc_loss2(encoder_out_tp, encoder_out_lens, text, text_lengths)

            # Collect CTC branch stats
            stats["loss_ctc_2"] = loss_ctc_2.detach() if loss_ctc_2 is not None else None
            stats["cer_ctc_2"] = cer_ctc_2

        # 2. Transducer-related I/O preparation
        target, t_len, u_len = get_minimal_ctc_task_io(
            text,
            encoder_out_lens,
            ignore_id=self.ignore_id,
        )
        log_probs = self.ctc.log_softmax(encoder_out)
        with torch.no_grad():
            alignment = ctc_forced_align(
                log_probs.float(),
                target.long(),
                t_len.long(),
                u_len.long(),
                ignore_id=self.ignore_id,
            )

        log_probs_2 = self.ctc_model2.log_softmax(encoder_out_tp)
        with torch.no_grad():
            alignment_2 = ctc_forced_align(
                log_probs_2.float(),
                target.long(),
                t_len.long(),
                u_len.long(),
                ignore_id=self.ignore_id,
            )

        # 2b. Attention decoder branch
        if self.att_weight != 0.0:
            loss_att, acc_att, pre_loss_att, acc_ctc, loss_att_2, acc_att_2 = self._calc_att_loss(
                encoder_out,
                encoder_out_tp,
                encoder_out_lens,
                text,
                text_lengths,
                alignment,
                alignment_2,
            )

        # 3. CTC-Att loss definition
        loss = self.ctc_weight * (loss_ctc + loss_ctc_2) + self.att_weight * (loss_att + loss_att_2)

        if self.use_1st_decoder_loss and pre_loss_att is not None:
            loss = loss + self.pre_loss_att_weight * pre_loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["loss_att_2"] = loss_att_2.detach() if loss_att_2 is not None else None
        stats["pre_loss_att"] = pre_loss_att.detach() if pre_loss_att is not None else None
        stats["acc"] = acc_att
        stats["acc_2"] = acc_att_2
        stats["acc_ctc"] = acc_ctc
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        ind_1st: int = 0,
        ind_2st: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, encoder_out_tp = self.encoder(
                feats, feats_lengths, ctc=self.ctc, ind_1st=ind_1st, ind_2st=ind_2st
            )
        else:
            encoder_out, encoder_out_lens, encoder_out_tp = self.encoder(
                feats, feats_lengths, ind_1st=ind_1st, ind_2st=ind_2st
            )
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(encoder_out, encoder_out_lens)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens, encoder_out_tp

    def _extract_feats(self, speech: torch.Tensor, speech_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]
        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def predictor(
        self,
        am: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        alignment,
        decoder_chunk_size,
        num_left_chunks,
        num_right_chunks,
    ):
        B, T, C = am.shape
        enc_probs = self.ctc.softmax(am).data
        acoustic_embeds = []
        mask = torch.zeros(B, ys_pad_lens.max(), encoder_out_lens.max())
        # for enc_prob, ali, y, y_lens in zip(enc_probs, alignment, ys_pad, ys_pad_lens):
        for i, (enc_prob, ali, y, y_lens) in enumerate(zip(enc_probs, alignment, ys_pad, ys_pad_lens)):
            pred = groupby(ali[: encoder_out_lens[i]])

            acoustic_embed = []
            _start = 0
            token_id = 0

            for pred_token, pred_frame in pred:
                _end = _start + len(list(pred_frame))
                if pred_token != 0:
                    acoustic_embed.append(torch.mean(enc_prob[_start:_end, :], 0, keepdim=True))

                    chunk_id = (_end - 1) // decoder_chunk_size
                    if num_left_chunks < 0:
                        chunk_start = 0
                    else:
                        chunk_start = max((chunk_id - num_left_chunks) * decoder_chunk_size, 0)
                    chunk_end = min(
                        (chunk_id + 1 + num_right_chunks) * decoder_chunk_size,
                        encoder_out_lens[i],
                    )
                    mask[i, token_id, chunk_start:chunk_end] = 1
                    token_id += 1
                _start = _end

            if len(acoustic_embed) != y_lens:
                acoustic_embeds.append(torch.nn.functional.one_hot(y[:y_lens], num_classes=self.vocab_size).type_as(am))
                mask[i, :, : encoder_out_lens[i]] = 1
            else:
                acoustic_embeds.append(torch.cat(acoustic_embed, dim=0))
        acoustic_embeds = pad_sequence(acoustic_embeds, batch_first=True, padding_value=0)
        return acoustic_embeds, mask.to(am.device)

    def predictor2(
        self,
        am: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        alignment,
        decoder_chunk_size,
        num_left_chunks,
        num_right_chunks,
    ):
        B, T, C = am.shape
        enc_probs = self.ctc_model2.softmax(am).data
        acoustic_embeds = []
        mask = torch.zeros(B, ys_pad_lens.max(), encoder_out_lens.max())
        # for enc_prob, ali, y, y_lens in zip(enc_probs, alignment, ys_pad, ys_pad_lens):
        for i, (enc_prob, ali, y, y_lens) in enumerate(zip(enc_probs, alignment, ys_pad, ys_pad_lens)):
            pred = groupby(ali[: encoder_out_lens[i]])

            acoustic_embed = []
            _start = 0
            token_id = 0

            for pred_token, pred_frame in pred:
                _end = _start + len(list(pred_frame))
                if pred_token != 0:
                    acoustic_embed.append(torch.mean(enc_prob[_start:_end, :], 0, keepdim=True))

                    chunk_id = (_end - 1) // decoder_chunk_size
                    if num_left_chunks < 0:
                        chunk_start = 0
                    else:
                        chunk_start = max((chunk_id - num_left_chunks) * decoder_chunk_size, 0)
                    chunk_end = min(
                        (chunk_id + 1 + num_right_chunks) * decoder_chunk_size,
                        encoder_out_lens[i],
                    )
                    mask[i, token_id, chunk_start:chunk_end] = 1
                    token_id += 1
                _start = _end

            if len(acoustic_embed) != y_lens:
                acoustic_embeds.append(torch.nn.functional.one_hot(y[:y_lens], num_classes=self.vocab_size).type_as(am))
                mask[i, :, : encoder_out_lens[i]] = 1
            else:
                acoustic_embeds.append(torch.cat(acoustic_embed, dim=0))
        acoustic_embeds = pad_sequence(acoustic_embeds, batch_first=True, padding_value=0)
        return acoustic_embeds, mask.to(am.device)

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_2: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        alignment,
        alignment_2,
    ):
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device
        )

        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias

        pre_acoustic_embeds, decoder_mask = self.predictor(
            encoder_out,
            encoder_out_lens,
            ys_pad,
            ys_pad_lens.long(),
            alignment,
            self.decoder_chunk_size,
            self.num_left_chunks,
            self.num_right_chunks,
        )

        pre_acoustic_embeds_2, decoder_mask_2 = self.predictor2(
            encoder_out_2,
            encoder_out_lens,
            ys_pad,
            ys_pad_lens.long(),
            alignment_2,
            self.decoder_chunk_size2,
            self.num_left_chunks2,
            self.num_right_chunks2,
        )
        # 0. sampler
        decoder_out_1st = None
        pre_loss_att = None

        if self.sampling_ratio > 0.0:
            sematic_embeds, decoder_out_1st = self.sampler(
                encoder_out,
                encoder_out_lens,
                ys_pad,
                ys_pad_lens,
                pre_acoustic_embeds,
                decoder_mask,
            )
        else:
            sematic_embeds = self.decoder.embed(pre_acoustic_embeds)

        # 1. Forward decoder
        decoder_outs = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, decoder_mask)
        decoder_out, _ = decoder_outs[0], decoder_outs[1]

        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out_1st.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        acc_ctc = th_accuracy(
            pre_acoustic_embeds.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )

        if self.sampling_ratio > 0.0:
            sematic_embeds_2, _ = self.sampler2(
                encoder_out_2,
                encoder_out_lens,
                ys_pad,
                ys_pad_lens,
                pre_acoustic_embeds_2,
                decoder_mask_2,
            )
        else:
            sematic_embeds_2 = self.decoder_model2.embed(pre_acoustic_embeds_2)

        decoder_outs_2 = self.decoder_model2(
            encoder_out_2,
            encoder_out_lens,
            sematic_embeds_2,
            ys_pad_lens,
            decoder_mask_2,
        )
        decoder_out_2, _ = decoder_outs_2[0], decoder_outs_2[1]

        loss_att_2 = self.criterion_att(decoder_out_2, ys_pad)
        acc_att_2 = th_accuracy(
            decoder_out_2.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att, pre_loss_att, acc_ctc, loss_att_2, acc_att_2

    def sampler(
        self,
        encoder_out,
        encoder_out_lens,
        ys_pad,
        ys_pad_lens,
        pre_acoustic_embeds,
        decoder_mask,
    ):

        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        ys_pad_one_hot = torch.nn.functional.one_hot(ys_pad_masked, num_classes=self.vocab_size).type_as(
            pre_acoustic_embeds
        )

        ys_pad_embed = self.decoder.embed(ys_pad_one_hot)

        pre_acoustic_embeds = self.decoder.embed(pre_acoustic_embeds)
        with torch.no_grad():
            decoder_outs = self.decoder(
                encoder_out,
                encoder_out_lens,
                pre_acoustic_embeds,
                ys_pad_lens,
                decoder_mask,
            )
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
                if target_num > 0:
                    input_mask[li].scatter_(
                        dim=0,
                        index=torch.randperm(seq_lens[li])[:target_num].cuda(),
                        value=0,
                    )
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0
        )
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def sampler2(
        self,
        encoder_out,
        encoder_out_lens,
        ys_pad,
        ys_pad_lens,
        pre_acoustic_embeds,
        decoder_mask,
    ):

        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        ys_pad_one_hot = torch.nn.functional.one_hot(ys_pad_masked, num_classes=self.vocab_size).type_as(
            pre_acoustic_embeds
        )

        ys_pad_embed = self.decoder_model2.embed(ys_pad_one_hot)

        pre_acoustic_embeds = self.decoder_model2.embed(pre_acoustic_embeds)
        with torch.no_grad():
            decoder_outs = self.decoder_model2(
                encoder_out,
                encoder_out_lens,
                pre_acoustic_embeds,
                ys_pad_lens,
                decoder_mask,
            )
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
                if target_num > 0:
                    input_mask[li].scatter_(
                        dim=0,
                        index=torch.randperm(seq_lens[li])[:target_num].cuda(),
                        value=0,
                    )
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0
        )
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def sampler_with_grad(
        self,
        encoder_out,
        encoder_out_lens,
        ys_pad,
        ys_pad_lens,
        pre_acoustic_embeds,
        decoder_mask,
    ):
        # print("target", ys_pad)
        # print("pred", torch.argmax(pre_acoustic_embeds, dim=-1))
        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        ys_pad_one_hot = torch.nn.functional.one_hot(ys_pad_masked, num_classes=self.vocab_size).type_as(
            pre_acoustic_embeds
        )

        ys_pad_embed = self.decoder.embed(ys_pad_one_hot)

        pre_acoustic_embeds = self.decoder.embed(pre_acoustic_embeds)
        decoder_outs = self.decoder(
            encoder_out,
            encoder_out_lens,
            pre_acoustic_embeds,
            ys_pad_lens,
            decoder_mask,
        )
        pre_loss_att = self.criterion_att(decoder_outs[0], ys_pad)
        decoder_out, _ = decoder_outs[0], decoder_outs[1]
        pred_tokens = decoder_out.argmax(-1)
        nonpad_positions = ys_pad.ne(self.ignore_id)
        seq_lens = (nonpad_positions).sum(1)
        same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
        input_mask = torch.ones_like(nonpad_positions)
        bsz, seq_len = ys_pad.size()
        for li in range(bsz):
            target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
            # target_ratio = (((seq_lens[li] * self.sampling_ratio - same_num[li].sum()).float()) / (seq_lens[li]- same_num[li].sum()).float())
            # target_ratio = min(target_ratio, 1)
            # target_num = (target_ratio * seq_lens[li]).long()
            if target_num > 0:
                input_mask[li].scatter_(
                    dim=0,
                    index=torch.randperm(seq_lens[li])[:target_num].cuda(),
                    value=0,
                )
        input_mask = input_mask.eq(1)
        input_mask = input_mask.masked_fill(~nonpad_positions, False)
        input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0
        )

        return sematic_embeds * tgt_mask, decoder_out * tgt_mask, pre_loss_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_ctc_loss2(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc_model2(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc_model2.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc


@tables.register("model_classes", "EncBiasAlignformerUnified")
class EncBiasAlignformerUnified(AlignformerUnified):
    # alignformer model with hotword customization ability
    # use encoder bias only (without seaco bias)
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[torch.nn.Module],
        specaug: Optional[str],
        specaug_conf: Optional[Dict],
        normalize: Optional[str],
        normalize_conf: Optional[Dict],
        encoder: str,
        encoder_conf: Optional[Dict],
        decoder: str,
        decoder_conf: Optional[Dict],
        decoder_model2: str,
        decoder_model2_conf: Optional[Dict],
        decoder2: str,
        decoder2_conf: Optional[Dict],
        preencoder: Optional[str],
        preencoder_conf: Optional[Dict],
        postencoder: Optional[str],
        postencoder_conf: Optional[Dict],
        ctc_conf: Optional[Dict],
        ctc_weight: float = 1.0,
        att_weight: float = 1.0,
        pre_loss_att_weight: float = 0.0,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        predictor_bias: int = 0,
        sampling_ratio: float = 0.2,
        share_embedding: bool = False,
        use_1st_decoder_loss: bool = False,
        decoder_chunk_size: int = 1,
        num_left_chunks: int = -1,
        num_right_chunks: int = 0,
        decoder_chunk_size2: int = 1,
        num_left_chunks2: int = -1,
        num_right_chunks2: int = 0,
        inner_dim: int = 256,  # seaco related
        bias_decoder_dropout: float = 0.1,
        bias_bd_lstm: bool = False,
        bias_encoder_layer: int = 2,
        eahc_weight: float = 1.0,
        eahc_bias_rate: float = 0.8,
        **kwargs,
    ):
        # normalize_class = tables.normalize_class.get(normalize)
        # preencoder_class = tables.preencoder_classes.get(preencoder)
        # postencoder_class = tables.postencoder_classes.get(postencoder)
        specaug_class = tables.specaug_classes.get(specaug)
        encoder_class = tables.encoder_classes.get(encoder)
        decoder_class = tables.decoder_classes.get(decoder)

        # normalize = normalize_class(**kwargs.get("normalize_conf", {}))
        # preencoder = preencoder_class(**kwargs.get("preencoder_conf", {}))
        # postencoder = postencoder_class(**kwargs.get("postencoder_conf", {}))
        normalize, preencoder, postencoder = None, None, None
        specaug = specaug_class(**specaug_conf)
        encoder = encoder_class(frontend.output_size(), **encoder_conf)
        ctc = CTC(odim=vocab_size, encoder_output_size=inner_dim, **ctc_conf)
        decoder = decoder_class(vocab_size, encoder.output_size(), **decoder_conf)
        decoder_model2 = decoder_class(vocab_size, encoder.output_size(), **decoder_model2_conf)
        decoder2 = decoder_class(vocab_size, encoder.output_size(), **decoder2_conf)
        self.converter = TokenIDConverter(token_list=token_list)
        self.tokenizer = kwargs.get("tokenizer", "CharTokenizer")

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            decoder_model2=decoder_model2,
            ctc=ctc,
            ctc_weight=ctc_weight,
            att_weight=att_weight,
            pre_loss_att_weight=pre_loss_att_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            blank_id=blank_id,
            sos=sos,
            eos=eos,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            predictor_bias=predictor_bias,
            sampling_ratio=sampling_ratio,
            share_embedding=share_embedding,
            preencoder=preencoder,
            postencoder=postencoder,
            use_1st_decoder_loss=use_1st_decoder_loss,
            decoder_chunk_size=decoder_chunk_size,
            num_left_chunks=num_left_chunks,
            num_right_chunks=num_right_chunks,
            decoder_chunk_size2=decoder_chunk_size2,
            num_left_chunks2=num_left_chunks2,
            num_right_chunks2=num_right_chunks2,
        )

        # seaco related init
        self.inner_dim = inner_dim
        self.bias_encoder = torch.nn.LSTM(
            self.inner_dim,
            self.inner_dim,
            bias_encoder_layer,
            batch_first=True,
            dropout=bias_decoder_dropout,
            bidirectional=bias_bd_lstm,
        )
        if bias_bd_lstm:
            self.lstm_proj = torch.nn.Linear(self.inner_dim * 2, self.inner_dim)
        else:
            self.lstm_proj = None

        # encoder bias parameters and modules
        self.sac_embedding = torch.nn.Embedding(self.vocab_size, self.inner_dim)
        self.decoder2 = decoder2  # use for bias encoder output
        self.ctc2 = CTC(odim=vocab_size, encoder_output_size=self.inner_dim)
        self.eahc_weight = eahc_weight
        self.eahc_bias_rate = eahc_bias_rate

        self.length_normalized_loss = length_normalized_loss

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        hotword_pad: torch.Tensor,
        hotword_lengths: torch.Tensor,
        cpp_label_pad: torch.Tensor,
        cpp_label_lengths: torch.Tensor,
        dha_pad: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check if batch_size is unified
        assert speech.shape[0] == speech_lengths.shape[0] == text.shape[0] == text_lengths.shape[0], (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
        )
        # for data-parallel
        text = text[:, : text_lengths.max()]
        speech = speech[:, : speech_lengths.max()]

        # step init
        batch_size = speech.shape[0]
        self.step_cur += 1
        stats = dict()

        # 1. Encoder
        # encoder_out for pass-1 (small chunk) encoder output
        # encoder_out_tp for pass-2 (big chunk) encoder output
        if hasattr(self.encoder, "overlap_chunk_cls") and hasattr(self.encoder, "overlap_chunk_cls2"):
            ind_1st = self.encoder.overlap_chunk_cls.random_choice(self.training, decoding_ind=None)
            encoder_out, encoder_out_lens, encoder_out_tp = self.encode(
                speech, speech_lengths, ind_1st=ind_1st, ind_2st=ind_1st
            )
            self.decoder_chunk_size = self.encoder.overlap_chunk_cls.get_chunk_size(ind=ind_1st)[1]
            self.decoder_chunk_size2 = self.encoder.overlap_chunk_cls2.get_chunk_size(ind=ind_1st)[1]
        else:
            encoder_out, encoder_out_lens, encoder_out_tp = self.encode(speech, speech_lengths)

        # hotword representation
        # use bias encoder to represent hotword and repeat for attention calculation
        hotword_vec = self._hotword_representation(hotword_pad, hotword_lengths)
        # repeat B times in 0th dimension
        hotword_vec = hotword_vec.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
        num_hot_word = hotword_vec.shape[1]
        hotword_vec_length = torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)

        if self.eahc_weight > 0 and random.random() < self.eahc_bias_rate:
            # use decoder2 to bias encoder_out_tp with hotwords
            # and add encoder_out_tp_bias to encoder_out_tp
            # bias is conducted under probability of self.eahc_bias_rate
            # else encoder_out_tp is remained unbiased
            encoder_out_tp_bias, _ = self.decoder2(
                hotword_vec, hotword_vec_length, encoder_out_tp, encoder_out_lens
            )  # B*T*D
            encoder_out_tp = self._eahc_combiner(encoder_out_tp, encoder_out_tp_bias)
            loss_eahc = self.ctc2(encoder_out_tp_bias, encoder_out_lens, cpp_label_pad, cpp_label_lengths)
            stats["loss_eahc"] = loss_eahc.detach()

            encoder_out_bias, _ = self.decoder2(hotword_vec, hotword_vec_length, encoder_out, encoder_out_lens)  # B*T*D
            encoder_out = self._eahc_combiner(encoder_out, encoder_out_bias)
            loss_eahc_model1 = self.ctc2(encoder_out_bias, encoder_out_lens, cpp_label_pad, cpp_label_lengths)
            stats["loss_eahc_model1"] = loss_eahc_model1.detach()
        else:
            loss_eahc = 0.0
            stats["loss_eahc"] = loss_eahc
            loss_eahc_model1 = 0.0
            stats["loss_eahc_model1"] = loss_eahc_model1

        loss_att, pre_loss_att, acc_att, _, _, acc_ctc = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        loss_ctc, cer_ctc = None, None

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(encoder_out, encoder_out_lens, text, text_lengths)

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

            loss_ctc_2, cer_ctc_2 = self._calc_ctc_loss2(encoder_out_tp, encoder_out_lens, text, text_lengths)

            # Collect CTC branch stats
            stats["loss_ctc_2"] = loss_ctc_2.detach() if loss_ctc_2 is not None else None
            stats["cer_ctc_2"] = cer_ctc_2

        # 2. Transducer-related I/O preparation
        target, t_len, u_len = get_minimal_ctc_task_io(
            text,
            encoder_out_lens,
            ignore_id=self.ignore_id,
        )
        log_probs = self.ctc.log_softmax(encoder_out)
        log_probs_2 = self.ctc_model2.log_softmax(encoder_out_tp)

        with torch.no_grad():
            alignment = ctc_forced_align(
                log_probs.float(),
                target.long(),
                t_len.long(),
                u_len.long(),
                ignore_id=self.ignore_id,
            )
            alignment_2 = ctc_forced_align(
                log_probs_2.float(),
                target.long(),
                t_len.long(),
                u_len.long(),
                ignore_id=self.ignore_id,
            )

        # 2b. Attention decoder branch
        if self.att_weight != 0.0:
            loss_att, acc_att, pre_loss_att, acc_ctc, loss_att_2, acc_att_2 = self._calc_att_loss(
                encoder_out,
                encoder_out_tp,
                encoder_out_lens,
                text,
                text_lengths,
                alignment,
                alignment_2,
            )

        # 3. CTC-Att loss definition
        loss = (
            self.ctc_weight * (loss_ctc + loss_ctc_2)
            + self.att_weight * (loss_att + loss_att_2)
            + self.eahc_weight * loss_eahc
            + self.eahc_weight * loss_eahc_model1
        )

        if self.use_1st_decoder_loss and pre_loss_att is not None:
            loss = loss + self.pre_loss_att_weight * pre_loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["loss_att_2"] = loss_att_2.detach() if loss_att_2 is not None else None
        stats["pre_loss_att"] = pre_loss_att.detach() if pre_loss_att is not None else None
        stats["acc"] = acc_att
        stats["acc_2"] = acc_att_2
        stats["acc_ctc"] = acc_ctc
        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = (text_lengths + self.predictor_bias).sum().type_as(batch_size)
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return loss, stats, weight

    def _eahc_combiner(self, encoder_output, encoder_output_bias):
        return encoder_output + encoder_output_bias * 0.6

    def _hotword_representation(
        self,
        hotword_pad,
        hotword_lengths,
    ):
        hw_embed = self.sac_embedding(hotword_pad)
        hw_embed, (_, _) = self.bias_encoder(hw_embed)
        if self.lstm_proj is not None:
            hw_embed = self.lstm_proj(hw_embed)
        _ind = np.arange(0, hw_embed.shape[0]).tolist()
        selected = hw_embed[_ind, [i - 1 for i in hotword_lengths.detach().cpu().tolist()]]
        # TODO: bid rnnt support
        return selected

    def bias_encode_ASF(self, encoder_out, encoder_out_lens, hotword_pad, hotword_lengths, nfilter=50):
        logging.warning("nfilter: {}".format(nfilter))

        selected = self._hotword_representation(
            hotword_pad.to(encoder_out.device),
            torch.Tensor(hotword_lengths).int().to(encoder_out.device),
        )
        contextual_info = selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
        num_hot_word = contextual_info.shape[1]
        _contextual_length = torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)

        if num_hot_word > nfilter > 0:
            # filter the incoming hotword list with decoder2
            hotword_scores = self.decoder2.forward_asf(
                3, contextual_info, _contextual_length, encoder_out, encoder_out_lens
            )
            hotword_scores = hotword_scores[0].sum(0).sum(0)[:-1]

            dec_filter = torch.topk(hotword_scores, min(nfilter, num_hot_word - 1))[1].tolist()
            dec_filter.append(len(hotword_pad) - 1)
            selected = selected[dec_filter]
            contextual_info = selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
            num_hot_word = contextual_info.shape[1]
            _contextual_length = torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)

        encoder_out_b, _ = self.decoder2(contextual_info, _contextual_length, encoder_out, encoder_out_lens)  # B*T*D
        encoder_out = self._eahc_combiner(encoder_out, encoder_out_b)

        return encoder_out, contextual_info, _contextual_length, encoder_out_b

    def dualbias_decode(
        self,
        enc_out,
        enc_out_lens,
        pre_acoustic_embeds,
        contextual_info=None,
        _contextual_length=None,
        mode="decoder",
    ):
        # eahc_out = self.ctc2.softmax(enc_out_b)
        if mode == "ctc":
            ctc_out = self.ctc.softmax(enc_out)
            ctc_pred = ctc_out.argmax(dim=-1)[0].cpu()
            _ctc_pred = []
            if ctc_pred[0] != 0:
                _ctc_pred.append(ctc_pred[0])
            for i in range(1, len(ctc_pred)):
                if ctc_pred[i] != ctc_pred[i - 1] and ctc_pred[i] != 0:
                    _ctc_pred.append(ctc_pred[i])
            _ctc_pred = torch.Tensor(_ctc_pred).long()
            return _ctc_pred
        else:
            decoder_out, decoder_hidden, _ = self.decoder(
                enc_out,
                enc_out_lens,
                pre_acoustic_embeds,
                torch.tensor(pre_acoustic_embeds.size(1)).unsqueeze(0),
                return_hidden=True,
                return_both=True,
            )
            # ys_pad_lens = torch.tensor(pre_acoustic_embeds.size(1)).unsqueeze(0)
            decoder_pred = torch.log_softmax(decoder_out, dim=-1)
            decoder_pred = decoder_pred.argmax(dim=-1)[0].cpu()
            return decoder_pred

    def generate_hotwords_list(self, hotword_list_or_file):
        # for None
        if hotword_list_or_file is None or not len(hotword_list_or_file):
            hotword_list = [[1]]
        # for local txt inputs
        elif os.path.exists(hotword_list_or_file) and (
            hotword_list_or_file.endswith("bpe.txt") or hotword_list_or_file.endswith("bpe.hwlist")
        ):
            logging.info("Attempting to parse hotwords from local txt...")
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, "r") as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hotword_str_list.append(hw)
                    hotword_list.append(self.converter.tokens2ids([i for i in hw.split()]))
                hotword_list.append([1])
                hotword_str_list.append("<s>")
            logging.info(
                "Initialized hotword list from file: {}, hotword list: {}.".format(
                    hotword_list_or_file, hotword_str_list
                )
            )
            logging.info("hotword list int: {}".format(hotword_list))
        elif os.path.exists(hotword_list_or_file) and (
            hotword_list_or_file.endswith(".txt") or hotword_list_or_file.endswith(".hwlist")
        ):
            logging.info("Attempting to parse hotwords from local txt...")
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, "r") as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hotword_str_list.append(hw)
                    hotword_list.append(self.converter.tokens2ids([i for i in hw]))
                hotword_list.append([1])
                hotword_str_list.append("<s>")
            logging.info(
                "Initialized hotword list from file: {}, hotword list: {}.".format(
                    hotword_list_or_file, hotword_str_list
                )
            )
        # for url, download and generate txt
        elif hotword_list_or_file.startswith("http"):
            logging.info("Attempting to parse hotwords from url...")
            work_dir = tempfile.TemporaryDirectory().name
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            text_file_path = os.path.join(work_dir, os.path.basename(hotword_list_or_file))
            local_file = requests.get(hotword_list_or_file)
            open(text_file_path, "wb").write(local_file.content)
            hotword_list_or_file = text_file_path
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, "r") as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hotword_str_list.append(hw)
                    hotword_list.append(self.converter.tokens2ids([i for i in hw]))
                hotword_list.append([self.sos])
                hotword_str_list.append("<s>")
            logging.info(
                "Initialized hotword list from file: {}, hotword list: {}.".format(
                    hotword_list_or_file, hotword_str_list
                )
            )
        # for text str input
        elif not hotword_list_or_file.endswith(".txt"):
            logging.info("Attempting to parse hotwords as str...")
            hotword_list = []
            hotword_str_list = []
            for hw in hotword_list_or_file.strip().split():
                hotword_str_list.append(hw)
                hotword_list.append(self.converter.tokens2ids([i for i in hw]))
            hotword_list.append([self.sos])
            hotword_str_list.append("<s>")
            logging.info("Hotword list: {}.".format(hotword_str_list))
        else:
            hotword_list = None
        return hotword_list

    def hypotheses_to_results(self, nbest_hyps: List[HypothesisTransducer]) -> List[Any]:
        """Build partial or final results from the hypotheses.
        Args:
            nbest_hyps: N-best hypothesis.
        Returns:
            results: Results containing different representation for the hypothesis.
        """
        results = []

        for hyp in nbest_hyps:
            token_int = list(filter(lambda x: x != 0, hyp))
            token = self.converter.ids2tokens(token_int)
            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
                if not hasattr(self.tokenizer, "bpemodel"):
                    text, _ = postprocess_utils.sentence_postprocess(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        return results

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        hotword_list_file = ""
        decoding_mode = kwargs.get("decoding_mode", "model2")
        decmode_eahc = "decoder"

        meta_data = {}
        time1 = time.perf_counter()
        speech = load_audio_text_image_video(data_in)
        speech_lengths = torch.tensor([s.size(0) for s in speech])
        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        meta_data["batch_data_time"] = speech_lengths.sum().item() / frontend.fs

        speech = torch.stack(speech)
        speech = to_device(speech, device=kwargs["device"])
        speech_lengths = to_device(speech_lengths, device=kwargs["device"])
        enc_out_1st, enc_out_lens, enc_out_2st = self.encode(speech, speech_lengths)
        contextual_info, _contextual_length = None, None

        # if not hotword_list_file.startswith('no'):
        #     if hotword_list_file.startswith('default'):
        #         hotword_list_file = ''
        #     self.hotword_list = self.generate_hotwords_list(hotword_list_file)
        #     self.hotword_lengths = [len(i) for i in self.hotword_list]
        #     hotword_list_ = [torch.Tensor(i).long() for i in self.hotword_list]
        #     self.hotword_pad = pad_list(hotword_list_, 0)
        # if self.hotword_pad is not None:
        #     enc_out_2st_bias, contextual_info, _contextual_length, encoder_out_b = (
        #         self.bias_encode_ASF(
        #             enc_out_2st,
        #             enc_out_lens,
        #             self.hotword_pad,
        #             self.hotword_lengths,
        #             nfilter=self.nfilter,
        #         )
        #     )
        #     enc_out_2st = enc_out_2st_bias

        if decoding_mode == "model1":
            enc_out = enc_out_1st
            decoder_chunk_size = self.decoder_chunk_size
            num_left_chunks = self.num_left_chunks
            num_right_chunks = self.num_right_chunks
            ctc = self.ctc
            decoder = self.decoder
        else:
            enc_out = enc_out_2st
            decoder_chunk_size = self.decoder_chunk_size2
            num_left_chunks = self.num_left_chunks2
            num_right_chunks = self.num_right_chunks2
            ctc = self.ctc_model2
            decoder = self.decoder_model2

        enc_probs = ctc.softmax(enc_out).data
        T_enc = enc_out.size(1)
        ys_hat = []
        for idx, enc_prob in enumerate(enc_probs):
            pred = groupby(enc_prob.argmax(-1).cpu())
            ctc_pred = [tmp[0] for tmp in groupby(enc_prob.argmax(-1).cpu()) if tmp[0] != 0 and tmp[0] != -1]
            if len(ctc_pred) == 0:
                break
            acoustic_embeds = []
            mask = torch.zeros(1, len(ctc_pred), enc_out.size(1))
            _start = 0
            j = 0
            for pred_token, pred_frame in pred:
                _end = _start + len(list(pred_frame))
                current_t = _end - 1
                if pred_token != 0 and pred_token != -1:
                    chunk_id = current_t // decoder_chunk_size
                    if num_left_chunks < 0:
                        chunk_start = 0
                    else:
                        chunk_start = max((chunk_id - num_left_chunks) * decoder_chunk_size, 0)
                    chunk_end = min((chunk_id + 1 + num_right_chunks) * decoder_chunk_size, T_enc)
                    acoustic_embeds.append(torch.mean(enc_prob[_start:_end, :], 0, keepdim=True))
                    mask[idx, j, chunk_start:chunk_end] = 1
                    j += 1
                _start = _end
            acoustic_embeds = torch.cat(acoustic_embeds, dim=0)

            pre_acoustic_embeds = decoder.embed(acoustic_embeds.unsqueeze(0))
            decoder_out, _ = decoder(
                enc_out,
                enc_out_lens,
                pre_acoustic_embeds,
                torch.tensor(pre_acoustic_embeds.size(1)).unsqueeze(0),
                mask.to(enc_out.device)
            )
            pred = decoder_out.argmax(dim=-1)[0].cpu()
            ys_hat.append(torch.LongTensor(pred))
        return self.hypotheses_to_results(ys_hat), meta_data
