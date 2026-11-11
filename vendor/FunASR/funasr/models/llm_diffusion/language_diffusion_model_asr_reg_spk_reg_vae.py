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
from speakerlab.models.eres2net.ERes2Net import ERes2Net as NET
from .spk_utils import FBank, supports
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

@tables.register("model_classes", "LLMDiffusion_ASR_Reg_Spk_Reg_VAE")
class LLMDiffusion_ASR_Reg_Spk_Reg_VAE(LLMDiffusion):
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
        spk_upsample_encoder_conf: Dict = {},
        lm_use_spk_emb: bool = False,
        pad_left: bool = True,
        init_cv2: bool = True,
        bce_weight: list = [1, 1],
        asr_model_path: str = None,
        asr_frontend: str = None,
        fs: int = 24000,
        use_binary_encoder: bool = False,
        asr_reg_weight: float = 1,
        spk_reg_weight: float = 1,
        use_l1_loss: bool = False,
        spk_model_path: str = None,
        spk_model_id: str = None,
        spk_fs: int=16000,
        spk_lm_use_spk_emb: bool = False,
        pretrain_vae: bool = False,
        kld_weight: float = 0.00001,
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

        spk_conf = supports[spk_model_id]

        spk_pretrained_model = os.path.join(spk_model_path, spk_model_id.split('/')[1], spk_conf['model_pt'])

        print(spk_pretrained_model)
        rank = int(os.environ.get("LOCAL_RANK", 0))
        spk_pretrained_state = torch.load(spk_pretrained_model, map_location=f'cuda:{rank}')

        spk_model = spk_conf['model']
        self.spk_embed_model = NET(**spk_model['args'])
        self.spk_embed_model.load_state_dict(spk_pretrained_state)
        for name, param in self.spk_embed_model.named_parameters():
            param.requires_grad = False
        self.spk_embed_model.eval()
        self.spk_fs = spk_fs
        self.spk_resampler = torchaudio.transforms.Resample(fs, self.spk_fs)
        self.spk_feature_extractor = FBank(80, sample_rate=self.spk_fs, mean_nor=True)
        self.spk_upsample_encoder = LinearUpsample1D_V2(input_size=llm_dim, **spk_upsample_encoder_conf)
        self.spk_cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        self.use_binary_encoder = use_binary_encoder
        if self.use_binary_encoder:
            self.binary_encoder = CausalTransformerEncoder(input_dim=llm_dim * 2, d_model=llm_dim, nhead=8, num_layers=2)

        self.asr_reg_weight = asr_reg_weight
        self.spk_reg_weight = spk_reg_weight
        self.use_l1_loss = use_l1_loss
        print("v_prediction", self.v_prediction)
        print("asr_reg_weight", asr_reg_weight)

        self.pretrain_vae = pretrain_vae
        from .vae import TransformerVAE, vae_loss_function
        self.vae = TransformerVAE(
            input_dim=80, 
            latent_dim=16, 
            d_model_encoder=1024, 
            d_model_decoder=1024,
            encoder_layers=12, 
            decoder_layers=12, 
            nhead=64, 
            dim_feedforward=4096, 
            dropout=0.0
        )
        self.vae_loss = vae_loss_function
        self.kld_weight = kld_weight
  
    @torch.no_grad()
    def spk_extract_feat(self, y: torch.Tensor, y_lengths: torch.Tensor):
        feats=[]
        feat_lengths=[]
        for i in range(y.shape[0]):
            audio = y[i]
            feat = self.spk_feature_extractor(audio)
            feat_length = y_lengths[i] / 160
            feats.append(feat.unsqueeze(0))
            feat_lengths.append(feat_length.unsqueeze(0))
        feats = torch.cat(feats, dim=0).to(y)
        # print(feat_lengths)
        feat_lengths = torch.cat(feat_lengths, dim=0).to(y)
        with torch.no_grad():
            outputs = self.spk_embed_model.custom_forward(feats)
        B, T = outputs.shape[0], outputs.shape[3]
        # print(outputs.shape)
        outputs = outputs.permute(0, 3, 1, 2).contiguous().view(B, T, -1)
        return outputs, None

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

        if self.pretrain_vae:
            recon_feat, mu, sigma = self.vae(feat)
            recon_loss, kld_loss = self.vae_loss(recon_feat, feat, mu, sigma, feat_lengths)   
            loss = recon_loss + self.kld_weight *  kld_loss
            stats = {}
            stats["loss"] = torch.clone(loss.detach())

            stats["recon_loss"] = torch.clone(recon_loss.detach())
            stats["kld_loss"] = torch.clone(kld_loss.detach())
            batch_size = feat.shape[0]
            # force_gatherable: to-device and to-tensor if scalar for DataParallel
            loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
            return loss, stats, weight
        else:
            with torch.no_grad():
                feat_enc, _ = self.vae.encode(feat)

        asr_audio = []
        asr_audio_len = []
        for i in range(len(audio_len)):
            asr_audio.append(self.resampler(audio[i]))
            asr_audio_len.append(audio_len[i] * (self.asr_frontend_fs / self.fs))

        asr_audio_len = torch.cat(asr_audio_len, dim=0)
        asr_audio_len = torch.tensor(asr_audio_len, dtype=torch.int32)
        asr_audio = pad_sequence(asr_audio, batch_first=True, padding_value=0.0)
        asr_speech, asr_speech_lengths = self.frontend(asr_audio, asr_audio_len)

        spk_audio = []
        spk_audio_len = []
        for i in range(len(audio_len)):
            spk_audio.append(self.spk_resampler(audio[i]))
            spk_audio_len.append(audio_len[i] * (self.spk_fs / self.fs))

        spk_audio_len = torch.cat(spk_audio_len, dim=0)
        spk_audio_len = torch.tensor(spk_audio_len, dtype=torch.int32)
        spk_audio = pad_sequence(spk_audio, batch_first=True, padding_value=0.0)
        spk_feats, _ = self.spk_extract_feat(spk_audio.float(), spk_audio_len)
        with torch.no_grad():
            asr_encoder_out, asr_encoder_out_lens = self.audio_encoder(asr_speech, asr_speech_lengths)

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

        encoder_in = feat_enc.clone()
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

        spk_hidden, spk_lengths = self.spk_upsample_encoder(hidden, encoder_out_lengths)

        asr_reg_loss = self.cos(asr_hidden, asr_encoder_out)
        asr_mask = (~make_pad_mask(asr_lengths.sum(dim=-1).squeeze(dim=1))).to(asr_hidden)
        asr_reg_loss = -(asr_reg_loss * asr_mask).sum() / torch.sum(asr_mask)

        spk_reg_loss = self.spk_cos(spk_hidden, spk_feats)
        spk_mask = (~make_pad_mask(spk_lengths.sum(dim=-1).squeeze(dim=1))).to(spk_hidden)
        spk_reg_loss = -(spk_reg_loss * spk_mask).sum() / torch.sum(spk_mask)

        hidden, h_lengths = self.upsample_encoder(hidden, encoder_out_lengths)
        if spk_emb is not None:
            spk_emb_diff = spk_emb[:, :spk_emb_len.max()]
            spk_emb_diff = self.norm_spk_emb(spk_emb_diff)
            spk_emb_diff = self.xvec_proj(spk_emb_diff).unsqueeze(dim=1)

        mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(hidden)

        if self.use_l1_loss:
            diff_loss, feat_pred, diff_loss_l1 = self.diffusion_forward(feat_enc, hidden, spk_emb_diff, feat_lengths, mask, global_emb)
            loss = diff_loss + diff_loss_l1 + loss_binary + asr_reg_loss * self.asr_reg_weight +  spk_reg_loss * self.spk_reg_weight
        else:
            diff_loss, feat_pred = self.diffusion_forward(feat_enc, hidden, spk_emb_diff, feat_lengths, mask, global_emb)
            loss = diff_loss + loss_binary + asr_reg_loss * self.asr_reg_weight + spk_reg_loss * self.spk_reg_weigh

        stats["diff_loss"] = torch.clone(diff_loss.detach())
        if self.use_l1_loss:
            stats["diff_loss_l1"] = torch.clone(diff_loss_l1.detach())

        stats["loss"] = torch.clone(loss.detach())
 
        stats["asr_reg_loss"] = torch.clone(asr_reg_loss.detach())
        stats["spk_reg_loss"] = torch.clone(spk_reg_loss.detach())
        stats["binary_loss"] = torch.clone(loss_binary.detach())
        stats["acc_binary"] = acc_binary

        stats["batch_size"] = batch_size
        stats["batch_size_x_tokens"] = feat_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def inference_prepare(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            **kwargs,
    ):
        uttid = key[0]
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        kwargs.update(**contents["meta_dict"])
        kwargs["uttid"] = uttid
        output = self.load_data(
            contents, **kwargs
        )
        batch = to_device(output, kwargs["device"])
        source_ids = batch["source_ids"]
        prompt_mel = batch["prompt_mel"]
        prompt_mel_lengths = batch["prompt_mel_lengths"]
        text_embeder = self.llm.model.get_input_embeddings()
        inputs_text_embeds = text_embeder(source_ids)

        if prompt_mel is not None:
            encoder_in, _ = self.vae.encode(prompt_mel)
            encoder_out, encoder_mask = self.downsample_encoder(encoder_in, prompt_mel_lengths)
            encoder_out_lengths = encoder_mask.sum(dim=2).squeeze(dim=1)

        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        turn_of_speech_emb = self.llm_embedding.weight[self.turn_of_speech].reshape(1, 1, -1)

        if self.llm_use_global_emb:
            #segments, seg_mask = self.extract_segment_features(prompt_mel, prompt_mel_lens)
            segments = prompt_mel
            max_seg_len = prompt_mel_lengths.max().item()
            seg_mask = torch.ones(segments.shape[0], max_seg_len, dtype=torch.bool, device=prompt_mel.device)
            for i in range(segments.shape[0]):
                seg_len = prompt_mel_lengths[i].item()
                seg_mask[i, :seg_len] = False

        if self.llm_use_global_emb:
            global_emb_llm = self.seg_transformer_llm(segments, seg_mask).unsqueeze(dim=1)
        else:
            global_emb_llm = None

        if self.lm_use_spk_emb:
            spk_emb = batch["xvec"]
            spk_emb_len = batch["xvec_lengths"]

            spk_emb_llm = spk_emb[:, :spk_emb_len.max()]
            spk_emb_llm = self.norm_spk_emb(spk_emb_llm.reshape(1, 1, -1)).float()
            spk_emb_llm = self.xvec_proj_llm(spk_emb_llm).unsqueeze(dim=1)

            if prompt_mel is not None:
                if self.llm_use_global_emb:
                    inputs_embeds = torch.concat([sos_emb, spk_emb_llm, global_emb_llm, inputs_text_embeds, turn_of_speech_emb, encoder_out], dim=1)
                else:
                    inputs_embeds = torch.concat([sos_emb, spk_emb_llm, inputs_text_embeds, turn_of_speech_emb, encoder_out], dim=1)
            else:
                if self.llm_use_global_emb:
                    inputs_embeds = torch.concat([sos_emb, spk_emb_llm, global_emb_llm, inputs_text_embeds, turn_of_speech_emb], dim=1)
                else:
                    inputs_embeds = torch.concat([sos_emb, spk_emb_llm, inputs_text_embeds, turn_of_speech_emb], dim=1)
        else:
            if prompt_mel is not None:
                if self.llm_use_global_emb:
                    inputs_embeds = torch.concat([sos_emb, global_emb_llm, inputs_text_embeds, turn_of_speech_emb, encoder_out], dim=1)
                else:
                    inputs_embeds = torch.concat([sos_emb, inputs_text_embeds, turn_of_speech_emb, encoder_out], dim=1)
            else:
                if self.llm_use_global_emb:
                    inputs_embeds = torch.concat([sos_emb, global_emb_llm, inputs_text_embeds, turn_of_speech_emb], dim=1)
                else:
                    inputs_embeds = torch.concat([sos_emb, inputs_text_embeds, turn_of_speech_emb], dim=1)
        return inputs_embeds, batch

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
            diff_input_mel_enc, _ = self.vae.encode(diff_input_mel)

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
                    mel_history=diff_input_mel_enc.contiguous() if self.left_size>0 else None,
                    global_emb=global_emb,
                )

                pred_mel = pred_mel[:,-self.subsampling_factor:,:]

                if output is None:
                    output = pred_mel
                else:
                    output = torch.concat([output, pred_mel], dim=1)
                
                diff_input_mel_enc = torch.cat((diff_input_mel_enc, pred_mel), dim=1)[:,-self.left_size:, :]
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

        prompt_in, _ = self.vae.encode(prompt_mel)
        output = torch.cat((prompt_in, output), dim=1)
        output = self.vae.decode(output)[:,prompt_in.shape[1]:,:]

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
