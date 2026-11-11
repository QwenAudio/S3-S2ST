import logging
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.metrics.compute_acc import th_accuracy, compute_accuracy
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
import numpy as np
from funasr.models.cosyvoice.utils import dtype_map
from funasr.utils.hinter import hint_once
import math
from typing import Dict, Optional
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from .utils.mask import make_pad_mask
from .language_diffusion_model import print_memory_usage, Audio2Mel, LayerNorm, LLMDiffusion
import torchaudio

class CausalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True  # 输入形状为 (batch_size, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)
        # 生成下三角掩码 (seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        # 应用掩码
        output = self.encoder(x, mask=~causal_mask)  # 注意: mask=True 表示被屏蔽
        return output

@tables.register("model_classes", "LLMDiffusion_ASR_Reg_SS")
class LLMDiffusion_ASR_Reg_SS(LLMDiffusion):
    """ """

    def __init__(
        self,
        llm: str = None,
        llm_conf: dict = None,
        input_size: int = 80,
        length_normalized_loss: bool = False,
        del_lm_head: bool = True,
        dit_conf: Dict = {},
        mel_feat_conf: Dict = {},
        downsample_encoder_conf: Dict = {},
        upsample_encoder_conf: Dict = {},
        asr_upsample_encoder_conf: Dict = {},
        lm_use_spk_emb: bool = False,
        pad_left: bool = True,
        init_cv2: bool = True,
        bce_weight: list = [1, 1],
        asr_model_path: str = None,
        asr_frontend: str = None,
        fs: int = 24000,
        use_binary_encoder: bool = False,
        asr_reg_weight: float = 1,
        use_l1_loss: bool = False,
        **kwargs,
    ):
        super().__init__(
            llm,
            llm_conf,
            input_size,
            length_normalized_loss,
            del_lm_head,
            dit_conf,
            mel_feat_conf,
            downsample_encoder_conf,
            upsample_encoder_conf,
            lm_use_spk_emb,
            pad_left,
            init_cv2,
            bce_weight,
            use_l1_loss,
            **kwargs,            
        )
        from funasr import AutoModel
        model = AutoModel(model=asr_model_path, model_revision="master")

        self.audio_encoder = (
            model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
        )

        for name, param in self.audio_encoder.named_parameters():
            param.requires_grad = False

        self.audio_encoder.eval()

        frontend_class = tables.frontend_classes.get(asr_frontend)
        self.frontend = frontend_class(**kwargs.get("asr_frontend_conf", {}))

        self.asr_frontend_fs = 16000 if self.frontend is None else self.frontend.fs

        self.fs = fs
        self.resampler = torchaudio.transforms.Resample(fs, self.asr_frontend_fs)

        llm_dim = self.llm.get_input_embeddings().weight.shape[-1]
        from .upsample_encoder import LinearUpsample1D_V2
        self.asr_upsample_encoder = LinearUpsample1D_V2(input_size=llm_dim, **asr_upsample_encoder_conf)

        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        self.use_binary_encoder = use_binary_encoder
        if self.use_binary_encoder:
            self.binary_encoder = CausalTransformerEncoder(input_dim=llm_dim * 2, d_model=llm_dim, nhead=8, num_layers=2)

        self.asr_reg_weight = asr_reg_weight
        self.use_l1_loss = use_l1_loss
        print("v_prediction", self.v_prediction)
        print("asr_reg_weight", asr_reg_weight)

    def forward(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        audio: torch.Tensor,
        audio_len: torch.Tensor,
        spk_emb: Optional[torch.Tensor] = None,
        spk_emb_len: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        if self.pad_left:
            stride_lens = self.subsampling_factor * self.mel_extractor.hop_length
            audio_len = (audio_len + stride_lens - 1) // stride_lens * stride_lens
            pad_lens = audio_len.max()-audio.size(1)

            if pad_lens > 0:
                left_pad = torch.zeros(audio.size(0), pad_lens).to(audio.device)
                audio = torch.cat([left_pad, audio], dim=1)

        asr_audio = []
        asr_audio_len = []
        for i in range(len(audio_len)):
            asr_audio.append(self.resampler(audio[i]))
            asr_audio_len.append(audio_len[i] * (self.asr_frontend_fs / self.fs))

        asr_audio_len = torch.cat(asr_audio_len, dim=0)
        asr_audio_len = torch.tensor(asr_audio_len, dtype=torch.int32)
        asr_audio = pad_sequence(asr_audio, batch_first=True, padding_value=0.0)
        asr_speech, asr_speech_lengths = self.frontend(asr_audio, asr_audio_len)

        with torch.no_grad():
            asr_encoder_out, asr_encoder_out_lens = self.audio_encoder(asr_speech, asr_speech_lengths)

        with torch.cuda.amp.autocast(enabled=False):
            feat, feat_lengths = self.extract_feat(audio.float(), audio_len)
            if not self.pad_left:
                if feat_lengths.max() < self.subsampling_factor:
                    feat_pad = torch.zeros(feat.size(0), self.subsampling_factor-feat_lengths.max(), feat.size(2)).to(audio.device)
                    feat = torch.cat([feat_pad, feat], dim=1)
                    feat_lengths = (feat_lengths + self.subsampling_factor) // self.subsampling_factor * self.subsampling_factor
                else:
                    feat_lengths = feat_lengths // self.subsampling_factor * self.subsampling_factor
                    feat = feat[:,:feat_lengths.max(),:]     
 
        feat_lengths = feat_lengths.squeeze(-1)
        text_len = text_len.squeeze(-1)

        if self.diff_use_global_emb or self.llm_use_global_emb:
            segments, seg_mask = self.extract_segment_features(feat, feat_lengths)

        if self.diff_use_global_emb:
            global_emb = self.seg_transformer(segments, seg_mask)
        else:
            global_emb = None

        if self.llm_use_global_emb:
            global_emb_llm = self.seg_transformer_llm(segments, seg_mask)
        else:
            global_emb_llm = None

        encoder_in = feat.clone()
        if self.specaug is not None:
            encoder_in, _ = self.specaug(encoder_in)

        encoder_out, encoder_mask = self.downsample_encoder(encoder_in, feat_lengths)
        encoder_out_lengths = encoder_mask.sum(dim=2).squeeze(dim=1) 
        
        text_mask = text > 0
        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            text_embeds = self.llm.model.get_input_embeddings()(text * text_mask) * text_mask.unsqueeze(-1)

        if self.lm_use_spk_emb and spk_emb is not None:
            spk_emb_llm = spk_emb[:, :spk_emb_len.max()]
            spk_emb_llm = self.norm_spk_emb(spk_emb_llm)
            spk_emb_llm = self.xvec_proj_llm(spk_emb_llm).unsqueeze(dim=1)
        else:
            spk_emb_llm = None

        inputs_embeds, inputs_embeds_len, attention_mask = self.pad_unpad_sequence(text_embeds, text_len, encoder_out, encoder_out_lengths, spk_emb_llm, global_emb_llm)

        stats = {}
        batch_size, feat_num, dims = inputs_embeds.shape
        #print_memory_usage("Before llm:")
        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            attention_mask[attention_mask < 0] = 0
            outputs = self.llm.model(
                # inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            lm_outputs = outputs[0]
        #print_memory_usage("After llm:")
        if self.lm_use_spk_emb:
            if self.llm_use_global_emb:
                lm_binary_target = [torch.tensor([self.ignore_id] * (3 + text_len[i]) + encoder_out_lengths[i] * [0] + [1]) for i in range(batch_size)]
            else:
                lm_binary_target = [torch.tensor([self.ignore_id] * (2 + text_len[i]) + encoder_out_lengths[i] * [0] + [1]) for i in range(batch_size)]
        else:
            if self.llm_use_global_emb:
                lm_binary_target = [torch.tensor([self.ignore_id] * (2 + text_len[i]) + encoder_out_lengths[i] * [0] + [1]) for i in range(batch_size)]
            else:
                lm_binary_target = [torch.tensor([self.ignore_id] * (1 + text_len[i]) + encoder_out_lengths[i] * [0] + [1]) for i in range(batch_size)]
        lm_binary_target = pad_sequence(lm_binary_target, batch_first=True, padding_value=self.ignore_id).to(lm_outputs.device)

        if self.use_binary_encoder:
            binary_logits = self.binary_decoder(self.binary_encoder(torch.cat((lm_outputs, inputs_embeds), dim=2).detach()))
            #binary_logits = self.binary_decoder(self.binary_encoder(torch.cat((lm_outputs, inputs_embeds), dim=2))) 
        else:
            binary_logits = self.binary_decoder(lm_outputs)

        # binary loss
        loss_binary = self.criterion_ce(binary_logits, lm_binary_target)
        acc_binary = th_accuracy(binary_logits.view(-1, 2), lm_binary_target, ignore_label=self.ignore_id)

        hidden = self.get_llm_output(lm_outputs, text_len, encoder_out_lengths, encoder_out)

        asr_hidden, asr_lengths = self.asr_upsample_encoder(hidden, encoder_out_lengths)

        asr_reg_loss = self.cos(asr_hidden, asr_encoder_out)
        asr_mask = (~make_pad_mask(asr_lengths.sum(dim=-1).squeeze(dim=1))).to(asr_hidden)

        asr_reg_loss = -(asr_reg_loss * asr_mask).sum() / torch.sum(asr_mask)

        hidden, h_lengths = self.upsample_encoder(hidden, encoder_out_lengths)
        if spk_emb is not None:
            spk_emb_diff = spk_emb[:, :spk_emb_len.max()]
            spk_emb_diff = self.norm_spk_emb(spk_emb_diff)
            spk_emb_diff = self.xvec_proj(spk_emb_diff).unsqueeze(dim=1)

        #print_memory_usage("Before diffusion:")
        mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(hidden)

        b, _, _ = hidden.shape
        t = torch.rand([b], device=hidden.device, dtype=hidden.dtype)
        z = torch.randn_like(feat)
        cfg_mask = torch.rand([b, 1, 1], device=hidden.device, dtype=hidden.dtype) > self.training_cfg_rate

        if self.use_l1_loss:
            diff_loss, feat_pred, diff_loss_l1 = self.diffusion_forward(feat, hidden, spk_emb_diff, feat_lengths, mask, global_emb, z, t, None, cfg_mask)
            loss = diff_loss + diff_loss_l1 + loss_binary + asr_reg_loss * self.asr_reg_weight
        else:
            diff_loss, feat_pred = self.diffusion_forward(feat, hidden, spk_emb_diff, feat_lengths, mask, global_emb, z, t, None, cfg_mask)
            loss = diff_loss + loss_binary + asr_reg_loss * self.asr_reg_weight

        # schedualed sampling
        pred_encoder_in = feat_pred.detach()
        if self.specaug is not None:
            pred_encoder_in, _ = self.specaug(pred_encoder_in)

        pred_encoder_out, _ = self.downsample_encoder(pred_encoder_in, feat_lengths)

        pred_inputs_embeds, _, _ = self.pad_unpad_sequence(text_embeds, text_len, pred_encoder_out, encoder_out_lengths, spk_emb_llm, global_emb_llm)

        #print_memory_usage("Before llm:")
        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            attention_mask[attention_mask < 0] = 0
            pred_outputs = self.llm.model(
                # inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                inputs_embeds=pred_inputs_embeds,
                attention_mask=attention_mask,
            )

            pred_lm_outputs = pred_outputs[0]
        #print_memory_usage("After llm:")
        if self.use_binary_encoder:
            pred_binary_logits = self.binary_decoder(self.binary_encoder(torch.cat((pred_lm_outputs, pred_inputs_embeds), dim=2).detach()))
            #binary_logits = self.binary_decoder(self.binary_encoder(torch.cat((lm_outputs, inputs_embeds), dim=2))) 
        else:
            pred_binary_logits = self.binary_decoder(pred_lm_outputs)

        # binary loss
        pred_loss_binary = self.criterion_ce(pred_binary_logits, lm_binary_target)
        pred_acc_binary = th_accuracy(pred_binary_logits.view(-1, 2), lm_binary_target, ignore_label=self.ignore_id)

        pred_hidden = self.get_llm_output(pred_lm_outputs, text_len, encoder_out_lengths, pred_encoder_out)

        pred_asr_hidden, _ = self.asr_upsample_encoder(pred_hidden, encoder_out_lengths)

        pred_asr_reg_loss = self.cos(pred_asr_hidden, asr_encoder_out)
        pred_asr_reg_loss = -(pred_asr_reg_loss * asr_mask).sum() / torch.sum(asr_mask)

        pred_hidden, h_lengths = self.upsample_encoder(pred_hidden, encoder_out_lengths)
        #print_memory_usage("Before diffusion:")
        if self.use_l1_loss:
            diff_loss_ss, feat_pred_ss, diff_loss_l1_ss = self.diffusion_forward(feat, pred_hidden, spk_emb_diff, feat_lengths, mask, global_emb, z, t, feat_pred.detach(), cfg_mask)
            loss = loss + diff_loss_ss + diff_loss_l1_ss + pred_loss_binary + pred_asr_reg_loss * self.asr_reg_weight
        else:
            diff_loss_ss, feat_pred = self.diffusion_forward(feat, pred_hidden, spk_emb_diff, feat_lengths, mask, global_emb, z, t, feat_pred.detach(), cfg_mask)
            loss = loss + diff_loss_ss + pred_loss_binary + pred_asr_reg_loss * self.asr_reg_weight


        stats["diff_loss"] = torch.clone(diff_loss.detach())
        stats["diff_loss_ss"] = torch.clone(diff_loss_ss.detach())
        if self.use_l1_loss:
            stats["diff_loss_l1"] = torch.clone(diff_loss_l1.detach())
            stats["diff_loss_l1_ss"] = torch.clone(diff_loss_l1_ss.detach())

        stats["loss"] = torch.clone(loss.detach())
 
        stats["asr_reg_loss"] = torch.clone(asr_reg_loss.detach())
        stats["asr_reg_loss_ss"] = torch.clone(pred_asr_reg_loss.detach())
        stats["binary_loss"] = torch.clone(loss_binary.detach())
        stats["binary_loss_ss"] = torch.clone(pred_loss_binary.detach())
        stats["acc_binary"] = acc_binary
        stats["acc_binary_ss"] = pred_acc_binary

        stats["batch_size"] = batch_size
        stats["batch_size_x_tokens"] = feat_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def build_masker(self):
        masker_type = self.masker_conf.get("prompt_type", "free")
        print("masker_type", masker_type)
        if masker_type == "prefix":
            from funasr.models.cosyvoice.utils.mask_along_axis import MaskTailVariableMaxWidth
            masker = MaskTailVariableMaxWidth(
                mask_width_ratio_range=self.masker_conf["prompt_width_ratio_range"],
            )
        else:
            raise NotImplementedError

        return masker

    @torch.no_grad()
    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        **kwargs,
    ):
        uttid = key[0]
        streaming_mode = kwargs.get("lm_streaming", False)
        if isinstance(streaming_mode, bool):
            streaming_mode = str(streaming_mode).lower()
        inputs_emb, batch = self.inference_prepare(
            data_in, data_lengths, key, **kwargs
        )
        text, text_lengths = batch["text"], batch["text_lengths"]
        prompt_text, prompt_text_lens = batch["prompt_text"], batch["prompt_text_lengths"]
        prompt_mel, prompt_mel_lens = batch["prompt_mel"], batch["prompt_mel_lengths"]

        if self.diff_use_global_emb or self.diff_use_global_emb:
            #segments, seg_mask = self.extract_segment_features(prompt_mel, prompt_mel_lens)
            segments = prompt_mel
            max_seg_len = prompt_mel_lens.max().item()
            seg_mask = torch.ones(segments.shape[0], max_seg_len, dtype=torch.bool, device=prompt_mel.device)
            for i in range(segments.shape[0]):
                seg_len = prompt_mel_lens[i].item()
                seg_mask[i, :seg_len] = False

        if self.diff_use_global_emb:
            global_emb = self.seg_transformer(segments, seg_mask)
        else:
            global_emb = None

        sys_instruct, instruct_lengths = batch["sys_instruct"], batch["sys_instruct_lengths"]
        
        max_length = kwargs.get("max_length", 60 * 25)
        if "max_length" in kwargs:
            kwargs.pop("max_length")
        min_length = kwargs.get("min_length", None)
        if "min_length" in kwargs:
            kwargs.pop("min_length")
        min_length_ratio = kwargs.get("min_length_ratio", None)
        max_length_ratio = kwargs.get("max_length_ratio", None)
        if min_length_ratio is not None and prompt_text_lens is not None and prompt_mel_lens is not None:
            min_length = int(prompt_mel_lens / prompt_text_lens * text_lengths * min_length_ratio)
        if max_length_ratio is not None and prompt_text_lens is not None and prompt_mel_lens is not None:
            max_length = int(prompt_mel_lens / prompt_text_lens * text_lengths * max_length_ratio)
        logging.info(f"{uttid}: min length: {min_length}, max length: {max_length}")

        spk_emb = batch["xvec"]
        spk_emb_len = batch["xvec_lengths"]
        if spk_emb is not None:
            spk_emb = spk_emb[:, :spk_emb_len.max()]
            spk_emb = self.norm_spk_emb(spk_emb.reshape(1, 1, -1)).float()
            spk_emb = self.xvec_proj(spk_emb).unsqueeze(dim=1)

        use_llm_cache = kwargs.get("use_llm_cache", True)
        states = kwargs.get("states", {})
        llm_cache = states.get("llm_cache", None)

        diff_input_h = None

        diff_input_mel = torch.zeros(1, self.left_size, 80).to(inputs_emb.device)

        if prompt_mel is not None:
            diff_input_mel = torch.cat((diff_input_mel, prompt_mel), dim=1)[:,-self.left_size:, :]

        output = None

        hidden_cache = None
        inputs_emb_cache = None

        for t in range(max_length):
        #for t in range(50):
            with torch.cuda.amp.autocast(
                enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype] 
            ):
                outputs = self.llm.model(
                    input_ids=None,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=llm_cache,
                    inputs_embeds=inputs_emb,
                    use_cache=use_llm_cache,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                    cache_position=None,
                )

                if hidden_cache is None:
                    hidden_cache = outputs.hidden_states[-1]
                else:
                    hidden_cache = torch.concat([hidden_cache, outputs.hidden_states[-1]], dim=1)

                if inputs_emb_cache is None:
                    inputs_emb_cache = inputs_emb
                else:
                    inputs_emb_cache = torch.concat([inputs_emb_cache, inputs_emb[:,-1:,:]], dim=1)

                lm_hidden_states = outputs.hidden_states[-1]
                if self.use_binary_encoder:
                    logp = self.binary_decoder(self.binary_encoder(torch.cat((hidden_cache, inputs_emb_cache), dim=2)))[:, -1].log_softmax(dim=-1)
                else:
                    logp = self.binary_decoder(lm_hidden_states[:, -1]).log_softmax(dim=-1)
                # logp = h.log_softmax(dim=-1).squeeze(0)
                eos = (logp.argmax().item() == 1)

                if t>0 and eos is True:
                    break

                if diff_input_h is None:
                    diff_input_h = torch.zeros(1, self.left_size, self.upsample_encoder.out_channels).to(lm_hidden_states.device)
                    hidden, _ = self.upsample_encoder(lm_hidden_states[:, -(1+prompt_mel_lens//self.subsampling_factor):], None)
                else:
                    hidden, _ = self.upsample_encoder(lm_hidden_states[:, -1:], None)
          
                diff_input_h = torch.cat((diff_input_h, hidden), dim=1)[:,-(self.left_size+self.subsampling_factor):, :]
                pred_mel = self.diffusion_infer(
                    mu=diff_input_h.contiguous(),
                    spks=spk_emb,
                    n_timesteps=10,
                    mel_history=diff_input_mel.contiguous() if self.left_size>0 else None,
                    global_emb=global_emb,
                )

                pred_mel = pred_mel[:,-self.subsampling_factor:,:]

                if output is None:
                    output = pred_mel
                else:
                    output = torch.concat([output, pred_mel], dim=1)
                
                diff_input_mel = torch.cat((diff_input_mel, pred_mel), dim=1)[:,-self.left_size:, :]
                this_input_embed, _ = self.downsample_encoder(pred_mel, torch.IntTensor([pred_mel.shape[1]]).to(lm_hidden_states.device))

                if use_llm_cache:
                    llm_cache = outputs.past_key_values

                if use_llm_cache:
                    inputs_emb = this_input_embed
                else:
                    inputs_emb = torch.cat([
                        inputs_emb,
                        this_input_embed,
                    ], dim=1)


        kwargs["max_length"] = max_length
        kwargs["min_length"] = min_length

        output_dir = kwargs.get("output_dir", None)
        if output_dir is not None:
            output_dir = os.path.join(output_dir, "mel")
            os.makedirs(output_dir, exist_ok=True)
            np.save(
                os.path.join(output_dir, f"{key[0]}.npy"),
                output[0].cpu().numpy()
            )
        return output
