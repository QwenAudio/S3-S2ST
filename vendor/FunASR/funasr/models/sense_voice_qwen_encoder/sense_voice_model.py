import logging
from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional
import types
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.cuda.amp import autocast
from funasr.metrics.compute_acc import compute_accuracy, th_accuracy
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils.datadir_writer import DatadirWriter
#from funasr.models.ctc.ctc import CTC
from funasr.models.ctc.ctc_zk import CTC

from funasr.models.sense_voice import whisper_lib as whisper
from funasr.frontends.whisper_frontend import WhisperFrontend
from funasr.register import tables


def calculate_cer_ctc(xs_label, ys_pad, idx_blank=0, idx_space=-1, idx_append=-1, xs_len=None):
    """Calculate sentence-level CER score for CTC.

    :param torch.Tensor ys_hat: prediction (batch, seqlen)
    :param torch.Tensor ys_pad: reference (batch, seqlen)
    :return: average sentence-level CER score
    :rtype float
    """
    from itertools import groupby
    import editdistance
    import numpy as np
    
    if xs_len is None:
        xs_len = [xs_label.shape[1]] * xs_label.shape[0]
    # xs_label = np.argmax(xs_pre,axis=-1)
    cers, char_ref_lens = [], []
    for i, y in enumerate(xs_label):
        y_hat = [x[0] for x in groupby(y)]
        y_true = ys_pad[i]
        seq_hat, seq_true = [], []
        for idx in y_hat[:xs_len[i]]:
            idx = int(idx)
            if idx != idx_append and idx != idx_blank and idx != idx_space:
                seq_hat.append(int(idx))

        for idx in y_true:
            idx = int(idx)
            if idx != idx_append and idx != idx_blank and idx != idx_space:
                seq_true.append(int(idx))

        if len(seq_true) > 0:
            cers.append(editdistance.eval(seq_hat, seq_true))
            char_ref_lens.append(len(seq_true))
    cer_ctc = float(sum(cers)) / sum(char_ref_lens) if cers else 0.0
    return cer_ctc

@tables.register("model_classes", "SenseVoiceQwenEncoder")
class SenseVoiceQwenEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        audio_encoder = kwargs.get("audio_encoder", None)
        encoder_conf = kwargs.get("audio_encoder_conf", {})
        from funasr.models.minmo.qwen_audio_encoder import QwenAudioEncoder
        encoder = QwenAudioEncoder(
            audio_encoder, 
            llm_dtype = encoder_conf.get("llm_dtype", "bf16"), 
            output_size=encoder_conf.get("output_size", 2048),
            fix_layer = -1
        )


        import re
        freeze = encoder_conf.get("freeze", True)
        freeze_layer_num = int(encoder_conf.get("freeze_layer_num", -1))
        freeze_exclude_key = encoder_conf.get("freeze_exclude_key", None)
        # import pdb; pdb.set_trace()
        if freeze:
            skip_flag = False
            for name, param in encoder.named_parameters():
                if freeze_exclude_key is not None:
                    for key in freeze_exclude_key:
                        # print(f"name: {name}")
                        if key in name:
                            skip_flag = True
                    if skip_flag:
                        skip_flag = False
                        continue
                if freeze_layer_num > 0:
                    idx = re.search(r"\.\d+\.", name)
                    if idx is not None:
                        beg, end = idx.regs[0]
                        layer_id = int(name[beg + 1: end - 1])
                        if layer_id < freeze_layer_num:
                            param.requires_grad = False
                    elif "ln_post." not in name and 'norm.' not in name:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

            encoder.eval()


        from funasr import AutoModel
        model = AutoModel(
            model=kwargs.get("sensevoice_init_param", "/path/to/SenseVoiceSANM"),
        ).model.model

        print("model", model)
        # encoder
        del model.encoder
        model.encoder = encoder

        decoder_conf = kwargs.get("decoder_conf", {})
        if decoder_conf.get("freeze", False):
            freeze_exclude_key = decoder_conf.get("freeze_exclude_key", None)
            for name, param in model.decoder.named_parameters():
                if not freeze_exclude_key in name:
                    logging.info(f"name: {name} is freeze")
                    param.requires_grad = False

        # decoder
        model.decoder.use_padmask = kwargs.get("use_padmask", True)
        from funasr.models.sense_voice.decoder import sense_voice_decode_forward

        model.decoder.forward = types.MethodType(sense_voice_decode_forward, model.decoder)

        self.model = model

        self.encoder_output_size = self.model.dims.n_audio_state 
        self.projector = torch.nn.Linear(kwargs.get("encoder_output_size", 2048), self.encoder_output_size)



        self.activation_checkpoint = kwargs.get("activation_checkpoint", False)
        self.ignore_id = kwargs.get("ignore_id", -1)
        self.vocab_size = kwargs.get("vocab_size", -1)
        self.length_normalized_loss = kwargs.get("length_normalized_loss", True)
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )

        specaug = kwargs.get("specaug", None)
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**kwargs.get("specaug_conf", {}))
        self.specaug = specaug

        ctc_conf = kwargs.get("ctc_conf", None)
        if ctc_conf is None:
            ctc_conf = {}
        blank_id = kwargs.get("blank_id", -1) if kwargs.get("blank_id", -1) > -1 else self.vocab_size - 1
        self.blank_id = blank_id
        ctc = CTC(odim=self.vocab_size, encoder_output_size=self.encoder_output_size, blank_id=blank_id, **ctc_conf)
        self.ctc = ctc
        ctc_weight = kwargs.get("ctc_weight", 0.3)
        self.ctc_weight = ctc_weight
        self.error_calculator = None
        self.ctc_conf = ctc_conf
        
        if ctc_conf.get("ctc_intermiddle_layer", -1) > -1:
            ctc_intermiddle_layer = CTC(odim=self.vocab_size, encoder_output_size=self.encoder_output_size, blank_id=blank_id, **ctc_conf)
            self.ctc_intermiddle_layer = ctc_intermiddle_layer
            ctc_weight_intermiddle_layer = kwargs.get("ctc_weight_intermiddle_layer", 0.3)
            self.ctc_weight_intermiddle_layer = ctc_weight_intermiddle_layer
            self.projector_inner = torch.nn.Linear(kwargs.get("encoder_output_size", 2048), self.encoder_output_size)

        else:
            self.ctc_intermiddle_layer = -1
            self.ctc_weight_intermiddle_layer = -1
            self.projector_inner = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        text = kwargs.get("ctc_labels", None)
        text_lengths = kwargs.get("ctc_labels_len", None)

        text_att = kwargs.get("att_labels", None)
        text_lengths_att = kwargs.get("att_labels_len", None)

        # text_lengths[text_lengths>text.size(1)] = text.size(1)
        # text_lengths_att[text_lengths_att>text_att.size(1)] = text_att.size(1)

        target_mask = kwargs.get("target_mask", None)

        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        if self.activation_checkpoint:
            from torch.utils.checkpoint import checkpoint

            encoder_out, encoder_out_lens = checkpoint(
                self.encode, speech, speech_lengths, ctc_intermiddle_layer=self.ctc_conf["ctc_intermiddle_layer"], use_reentrant=False
            )
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ctc_intermiddle_layer=self.ctc_conf["ctc_intermiddle_layer"])

        ctc_intermediate_layer = None
        if isinstance(encoder_out, tuple):
            encoder_out, ctc_intermediate_layer = encoder_out[0], encoder_out[1]
            # encoder_out = encoder_out[0]
            # print(ctc_intermediate_layer.size())
            ctc_intermediate_layer = self.projector_inner(ctc_intermediate_layer)

        encoder_out = self.projector(encoder_out)
        try:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text_att, text_lengths_att, target_mask=target_mask
            )
        except Exception as e:
            print(
                f"Train Failed as {str(e)}, {traceback.format_exc()}"
            )
            print(text_att)
            print(text_lengths_att)


        stats = {}
        if self.ctc_weight > -1:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            if self.ctc_intermiddle_layer is not None:
                # print(ctc_intermediate_layer.size())
                # print(encoder_out.size())
                # exit()
                loss_ctc_intermiddle_layer, cer_ctc_intermiddle_layer = self._calc_ctc_loss2(
                    ctc_intermediate_layer, encoder_out_lens, text, text_lengths
                )
                loss = (1 - self.ctc_weight - self.ctc_weight_intermiddle_layer) * loss_att + self.ctc_weight * loss_ctc + self.ctc_weight_intermiddle_layer * loss_ctc_intermiddle_layer
            else:
                loss = (1 - self.ctc_weight) * loss_att + self.ctc_weight * loss_ctc
        else:
            loss_ctc, cer_ctc = None, None
            loss = loss_att
        

        stats["cer"] = cer_att
        stats["acc"] = acc_att
        stats["cer_ctc"] = cer_ctc
        stats["loss_att"] = torch.clone(loss_att.detach())
        stats["loss_ctc"] = torch.clone(loss_ctc.detach()) if loss_ctc is not None else None
        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        batch_size_speech, frames, _ = speech.shape
        stats["batch_size_speech"] = batch_size_speech
        stats["batch_size_x_frames"] = frames * batch_size_speech
        stats["batch_size_real_frames"] = speech_lengths.sum().item()
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]
        
        batch_size, token_num = text_att.shape
        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = text_lengths_att.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        """Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        # with autocast(False):
        #
        #     # Data augmentation
        #     if self.specaug is not None and self.training:
        #         speech, speech_lengths = self.specaug(speech, speech_lengths)
        # Forward encoder
        encoder_out, encoder_out_lens = self.model.encoder(speech.permute(0, 2, 1), speech_lengths, ctc_intermiddle_layer=self.ctc_conf["ctc_intermiddle_layer"])
        # print("encode", encoder_out[0].size(), encoder_out[1].size())
        return encoder_out, encoder_out_lens

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
        # if not self.training and self.error_calculator is not None:
        if True:
            ys_hat = self.ctc.argmax(encoder_out).data
            # cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)

            cer_ctc = calculate_cer_ctc(
                ys_hat.float().detach().cpu().numpy(), 
                ys_pad.detach().cpu().numpy(), 
                idx_blank = self.blank_id
            )


        return loss_ctc, cer_ctc

    def _calc_ctc_loss2(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc_intermiddle_layer(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        # if not self.training and self.error_calculator is not None:
        if True:
            ys_hat = self.ctc_intermiddle_layer.argmax(encoder_out).data
            # cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)

            cer_ctc = calculate_cer_ctc(
                ys_hat.float().detach().cpu().numpy(),
                ys_pad.detach().cpu().numpy(),
                idx_blank = self.blank_id
            )


        return loss_ctc, cer_ctc

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        **kwargs,
    ):
        target_mask = kwargs.get("target_mask", None)
        stats = {}

        # 1. Forward decoder
        decoder_out = self.model.decoder(
            x=ys_pad, xa=encoder_out, hlens=encoder_out_lens, ys_in_lens=ys_pad_lens
        )

        # 2. Compute attention loss
        mask = torch.ones_like(ys_pad) * (-1)
        ys_pad_mask = (ys_pad * target_mask + mask * (1 - target_mask)).to(torch.int64)
        ys_pad_mask[ys_pad_mask == 0] = -1. # why need?

        loss_att = self.criterion_att(decoder_out[:, :-1, :], ys_pad_mask[:, 1:])

        with torch.no_grad():
            preds = torch.argmax(decoder_out, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], ys_pad_mask[:, 1:], ignore_label=self.ignore_id
            )

        return loss_att, acc_att, None, None

    def inference_decoder(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        if frontend is None and not hasattr(self, "frontend"):
            frontend_class = tables.frontend_classes.get("WhisperFrontend")
            frontend = frontend_class(
                n_mels=self.model.dims.n_mels, do_pad_trim=kwargs.get("do_pad_trim", True)
            )
            self.frontend = frontend
        else:
            frontend = frontend if frontend is not None else self.frontend

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs if hasattr(frontend, "fs") else 16000,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
            lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

        speech = speech.to(device=kwargs["device"])[0, :, :]
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        # if isinstance(frontend, WhisperFrontend):
        #     speech = speech.permute(1, 0)
        DecodingOptions = kwargs.get("DecodingOptions", {"fp16": kwargs.get("fp16", True)})
        task = DecodingOptions.get("task", "ASR")
        if isinstance(task, str):
            task = [task]
        task = "".join([f"<|{x}|>" for x in task])
        initial_prompt = kwargs.get("initial_prompt", f"<|startoftranscript|>{task}")
        DecodingOptions["initial_prompt"] = initial_prompt

        language = DecodingOptions.get("language", None)
        language = None if language == "auto" else language
        DecodingOptions["language"] = language

        DecodingOptions["vocab_path"] = kwargs["tokenizer_conf"].get("vocab_path", None)

        if "without_timestamps" not in DecodingOptions:
            DecodingOptions["without_timestamps"] = True

        options = whisper.DecodingOptions(**DecodingOptions)

        result = whisper.decode(self.model, speech, options)
        text = f"{result.text}"
        results = []
        result_i = {"key": key[0], "text": text}

        results.append(result_i)

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"1best_recog"]
        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = text

        return results, meta_data

    # def inference(
    #     self,
    #     data_in,
    #     data_lengths=None,
    #     key: list = None,
    #     tokenizer=None,
    #     frontend=None,
    #     **kwargs,
    # ):


    #     if frontend is None and not hasattr(self, "frontend"):
    #         frontend_class = tables.frontend_classes.get("WhisperFrontend")
    #         frontend = frontend_class(
    #             n_mels=self.encoder.input_size, do_pad_trim=kwargs.get("do_pad_trim", True)
    #         )
    #         self.frontend = frontend
    #     else:
    #         frontend = frontend if frontend is not None else self.frontend

    #     meta_data = {}
    #     if (
    #         isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
    #     ):  # fbank
    #         speech, speech_lengths = data_in, data_lengths
    #         if len(speech.shape) < 3:
    #             speech = speech[None, :, :]
    #         if speech_lengths is None:
    #             speech_lengths = speech.shape[1]
    #     else:
    #         # extract fbank feats
    #         time1 = time.perf_counter()
    #         audio_sample_list = load_audio_text_image_video(
    #             data_in,
    #             fs=frontend.fs if hasattr(frontend, "fs") else 16000,
    #             audio_fs=kwargs.get("fs", 16000),
    #             data_type=kwargs.get("data_type", "sound"),
    #             tokenizer=tokenizer,
    #         )

    #         if (
    #             isinstance(kwargs.get("data_type", None), (list, tuple))
    #             and len(kwargs.get("data_type", [])) > 1
    #         ):
    #             audio_sample_list, text_token_int_list = audio_sample_list
    #             text_token_int = text_token_int_list[0]
    #         else:
    #             text_token_int = None

    #         time2 = time.perf_counter()
    #         meta_data["load_data"] = f"{time2 - time1:0.3f}"
    #         speech, speech_lengths = extract_fbank(
    #             audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
    #         )
    #         time3 = time.perf_counter()
    #         meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
    #         frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
    #         lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
    #         meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

    #     speech = speech.to(device=kwargs["device"])
    #     speech_lengths = speech_lengths.to(device=kwargs["device"])
    #     if isinstance(frontend, WhisperFrontend):
    #         speech = speech.permute(0, 2, 1)

    #     encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
    #     if isinstance(encoder_out, tuple):
    #         encoder_out = encoder_out[0]

    #     encoder_out = self.projector(encoder_out)
    #     # c. Passed the encoder result and the beam search
    #     ctc_logits = self.ctc.log_softmax(encoder_out)

    #     results = []
    #     b, n, d = encoder_out.size()
    #     if isinstance(key[0], (list, tuple)):
    #         key = key[0]
    #     if len(key) < b:
    #         key = key * b
    #     for i in range(b):
    #         x = ctc_logits[i, : encoder_out_lens[i].item(), :]
    #         yseq = x.argmax(dim=-1)
    #         yseq = torch.unique_consecutive(yseq, dim=-1)

    #         ibest_writer = None
    #         if kwargs.get("output_dir") is not None:
    #             if not hasattr(self, "writer"):
    #                 self.writer = DatadirWriter(kwargs.get("output_dir"))
    #             ibest_writer = self.writer[f"1best_recog"]

    #         mask = yseq != self.blank_id
    #         token_int = yseq[mask].tolist()

    #         # Change integer-ids to tokens
    #         text = tokenizer.decode(token_int)

    #         result_i = {"key": key[i], "text": text}
    #         results.append(result_i)

    #         if ibest_writer is not None:
    #             ibest_writer["text"][key[i]] = text

    #     return results, meta_data


    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        eos = kwargs.get("model_conf").get("eos")
        if isinstance(eos, str):
            eos_int = tokenizer.encode(eos, allowed_special="all")
        else:
            eos_int = [eos]

        # init beamsearch
        if not hasattr(self, "beam_search") or self.beam_search is None:
            logging.info("enable beam_search")
            self.init_beam_search(eos = eos_int[0], **kwargs)
            self.nbest = kwargs.get("nbest", 1)

        if frontend is None and not hasattr(self, "frontend"):
            frontend_class = tables.frontend_classes.get("WhisperFrontend")
            frontend = frontend_class(
                n_mels=self.model.dims.n_mels, do_pad_trim=kwargs.get("do_pad_trim", True)
            )
            self.frontend = frontend
        else:
            frontend = frontend if frontend is not None else self.frontend

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs if hasattr(frontend, "fs") else 16000,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )

            if (
                isinstance(kwargs.get("data_type", None), (list, tuple))
                and len(kwargs.get("data_type", [])) > 1
            ):
                audio_sample_list, text_token_int_list = audio_sample_list
                text_token_int = text_token_int_list[0]
            else:
                text_token_int = None

            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
            lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

        speech = speech.to(device=kwargs["device"])[0, :, :]
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        # print(speech.size())
        speech = speech.transpose(0,1)
        
        DecodingOptions = kwargs.get("DecodingOptions", {})
        task = DecodingOptions.get("task", "ASR")
        if isinstance(task, str):
            task = [task]
        task = "".join([f"<|{x}|>" for x in task])

        sos = kwargs.get("model_conf").get("sos")
        sos = tokenizer.encode(sos, allowed_special="all")
        # if isinstance(sos, str):
        #     initial_prompt = kwargs.get("initial_prompt", f"<|startoftranscript|>{task}")

        #     language = DecodingOptions.get("language", None)
        #     language = None if language == "auto" else language

        #     sos = f"{initial_prompt}<|{language}|>" if language is not None else initial_prompt
        #     sos_int = tokenizer.encode(sos, allowed_special="all")
        # else:
        #     language = DecodingOptions.get("language", None)
        #     language = None if language == "auto" else language
        #     initial_prompt = kwargs.get("initial_prompt", f"{task}")
        #     initial_prompt_lid = (
        #         f"{initial_prompt}<|{language}|>" if language is not None else initial_prompt
        #     )
        #     initial_prompt_lid_int = tokenizer.encode(initial_prompt_lid, allowed_special="all")
        #     sos_int = [sos] + initial_prompt_lid_int

        sos_int = [sos]


        self.beam_search.sos = sos_int
        self.beam_search.eos = eos_int[0]

        # Paramterts for rich decoding
        self.beam_search.emo_unk = tokenizer.encode(
            DecodingOptions.get("emo_unk_token", "<|SPECIAL_TOKEN_1|>"), allowed_special="all"
        )[0]
        self.beam_search.emo_unk_score = 1
        self.beam_search.emo_tokens = tokenizer.encode(
            DecodingOptions.get("emo_target_tokens", "<|HAPPY|><|SAD|><|ANGRY|>"),
            allowed_special="all",
        )
        self.beam_search.emo_scores = DecodingOptions.get("emo_target_threshold", [0.1, 0.1, 0.1])

        self.beam_search.event_bg_token = tokenizer.encode(
            DecodingOptions.get("gain_tokens_bg", "<|Speech|><|BGM|><|Applause|><|Laughter|>"),
            allowed_special="all",
        )
        self.beam_search.event_ed_token = tokenizer.encode(
            DecodingOptions.get("gain_tokens_ed", "<|/Speech|><|/BGM|><|/Applause|><|/Laughter|>"),
            allowed_special="all",
        )
        # self.beam_search.event_score_ga = DecodingOptions.get("gain_tokens_score", [1, 1, 1, 1])

        encoder_out, encoder_out_lens = self.encode(speech[None, :, :], speech_lengths)
        encoder_out = self.projector(encoder_out)

        if text_token_int is not None:
            i = 0
            results = []
            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer[f"1best_recog"]

            # 1. Forward decoder
            ys_pad = torch.tensor(sos_int + text_token_int, dtype=torch.int64).to(kwargs["device"])[
                None, :
            ]
            ys_pad_lens = torch.tensor([len(sos_int + text_token_int)], dtype=torch.int64).to(
                kwargs["device"]
            )[None, :]
            decoder_out = self.model.decoder(
                x=ys_pad, xa=encoder_out, hlens=encoder_out_lens, ys_in_lens=ys_pad_lens
            )

            token_int = decoder_out.argmax(-1)[0, :].tolist()
            text = tokenizer.decode(token_int)

            result_i = {"key": key[i], "text": text}
            results.append(result_i)

            if ibest_writer is not None:
                # ibest_writer["token"][key[i]] = " ".join(token)
                ibest_writer["text"][key[i]] = text
            return results, meta_data

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=encoder_out[0],
            maxlenratio=kwargs.get("maxlenratio", 0.0),
            minlenratio=kwargs.get("minlenratio", 0.0),
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        b, n, d = encoder_out.size()
        for i in range(b):

            for nbest_idx, hyp in enumerate(nbest_hyps):
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx + 1}best_recog"]

                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # # remove blank symbol id, which is assumed to be 0
                # token_int = list(
                #     filter(
                #         lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                #     )
                # )

                # Change integer-ids to tokens
                # token = tokenizer.ids2tokens(token_int)
                text = tokenizer.decode(token_int)

                result_i = {"key": key[i], "text": text}
                results.append(result_i)

                if ibest_writer is not None:
                    # ibest_writer["token"][key[i]] = " ".join(token)
                    ibest_writer["text"][key[i]] = text

        return results, meta_data

    def init_beam_search(
        self,
        **kwargs,
    ):
        from funasr.models.sense_voice.search import BeamSearch
        from funasr.models.transformer.scorers.ctc import CTCPrefixScorer
        from funasr.models.transformer.scorers.length_bonus import LengthBonus
        # 1. Build ASR model
        scorers = {}

        eos = kwargs.get("eos")
        # print(eos)
        # exit()
        scorers.update(
            decoder=self.model.decoder,
            # ctc = CTCPrefixScorer(ctc=self.ctc, eos=eos),
            length_bonus=LengthBonus(self.vocab_size),
        )

        weights = dict(
            decoder=1.0,
            ctc=0.0,
            lm=0.0,
            ngram=0.0,
            length_bonus=kwargs.get("penalty", 0.0),
        )
        beam_search = BeamSearch(
            beam_size=kwargs.get("beam_size", 1),
            weights=weights,
            scorers=scorers,
            sos=None,
            eos=None,
            vocab_size=self.vocab_size,
            token_list=None,
            pre_beam_score_key="full",
        )

        self.beam_search = beam_search
