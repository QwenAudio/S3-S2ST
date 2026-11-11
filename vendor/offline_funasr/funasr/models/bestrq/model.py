#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Dict, List, Tuple, Optional
from .vector_quantize_pytorch.random_projection_quantizer import RandomProjectionQuantizer
from funasr.train_utils.device_funcs import to_device
from funasr.register import tables
from funasr.models.transformer.utils.repeat import repeat
from funasr.train_utils.device_funcs import force_gatherable
from funasr.metrics.compute_acc import th_accuracy
from funasr.models.transformer.layer_norm import LayerNorm
from transformers import AutoConfig, AutoModelForCausalLM

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

def calculate_cer_ctc(xs_pre, ys_pad, idx_blank=0, idx_space=-1, idx_append=-1, xs_len=None):
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
        xs_len = [xs_pre.shape[1]] * xs_pre.shape[0]
    xs_label = np.argmax(xs_pre,axis=-1)
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

class Conv1d(nn.Conv1d):
    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

class QwenAudioEncoder(nn.Module):
    def __init__(
        self,
        audio_encoder,
        llm_dtype,
        input_size: int = 128,
        linear_units: int = 1280,
        output_size: int = 1536,
        fix_layer: int = -1,
        **kwargs,
    ):
        super().__init__()

        self.time_first = True
        self.fix_layer = fix_layer
        self.conv1 = Conv1d(input_size, linear_units, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv1d(linear_units, output_size, kernel_size=3, stride=2, padding=1)

        if audio_encoder.endswith("json"):
            config = AutoConfig.from_pretrained(audio_encoder)
            self.encoder = AutoModelForCausalLM.from_config(
                config,
                attn_implementation="flash_attention_2",
                # load_in_8bit=None,
                # device_map=None,
                # use_cache=None,
            ).model
        else:
            self.encoder = AutoModelForCausalLM.from_pretrained(
                audio_encoder,
                attn_implementation="flash_attention_2",
                load_in_8bit=None,
                device_map=None,
                use_cache=None,
            ).model
            # set conv1, and conv2 weight
            import os
            conv1_weight_path = os.path.join(audio_encoder, "conv1.pt")
            conv2_weight_path = os.path.join(audio_encoder, "conv2.pt")
            conv1_weight = torch.load(conv1_weight_path)
            conv2_weight = torch.load(conv2_weight_path)
            self.conv1.weight.data = conv1_weight['audio_encoder.conv1.weight']
            self.conv1.bias.data = conv1_weight['audio_encoder.conv1.bias']
            self.conv2.weight.data = conv2_weight['audio_encoder.conv2.weight']
            self.conv2.bias.data = conv2_weight['audio_encoder.conv2.bias']

        del self.encoder.embed_tokens

        for block in self.encoder.layers:
            block.self_attn.is_causal = False

        self.downsample_rate = kwargs.get("downsample_rate", 4)
        self.llm_dtype = llm_dtype
        self.output_size = output_size

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor = None,
        time_first: bool = False,
        **kwargs,
    ):
        if time_first:
            x = x.transpose(1,2)

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        inputs_embeds = x.permute(0, 2, 1)

        n_frames = inputs_embeds.size(1)
        max_pos = n_frames

        if ilens is not None:
            if self.downsample_rate == 4:
                olens = (
                    1
                    + (ilens - self.conv1.kernel_size[0] + 2 * self.conv1.padding[0])
                    // self.conv1.stride[0]
                )
            else:
                olens = ilens
            olens = (
                1
                + (olens - self.conv2.kernel_size[0] + 2 * self.conv2.padding[0])
                // self.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        position_ids = cache_position.unsqueeze(0)

        # 计算 attention_mask
        if olens is not None:
            # 创建 attention_mask: 形状为 (batch_size, sequence_length)
            # 1 表示有效位置，0 表示 padding 位置
            attention_mask = torch.arange(n_frames, device=inputs_embeds.device).unsqueeze(0) < olens.unsqueeze(1)
            attention_mask = attention_mask.to(dtype=torch.long)  # 转换为与 hidden_states 相同的 dtype
        else:
            # 如果没有长度信息，创建全1的mask（所有位置都有效）
            attention_mask = torch.ones(
                (inputs_embeds.size(0), n_frames),
                device=inputs_embeds.device,
                dtype=torch.long
            )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.encoder.rotary_emb(hidden_states, position_ids)

        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            for i, decoder_layer in enumerate(self.encoder.layers):
                
                layer_outputs = decoder_layer(
                    hidden_states,
                     attention_mask=attention_mask,
                     position_ids=position_ids,
                     past_key_value=None,
                     output_attentions=False,
                     use_cache=False,
                     cache_position=cache_position,
                     position_embeddings=position_embeddings,
                )

                hidden_states = layer_outputs#[0 ]

                if i == self.fix_layer:
                    hidden_states = hidden_states.detach()

        hidden_states = self.encoder.norm(hidden_states)
        return hidden_states, olens




@tables.register("model_classes", "BestRQ_Qwen2")
class BestRQ_Qwen(torch.nn.Module):
    def __init__(
          self,
          encoder: str = None,
          encoder_conf: Optional[Dict] = None,
          codebook_size: int = 1024,
          codebook_dim: int = 256,
          mask_prob: float = 0.1,
          mask_length: int = 6,
          input_size: int = 80,
          num_softmax_layer: int = 1,
          causal: bool = False,
          **kwargs,
    ):
        super().__init__()

        encoder = QwenAudioEncoder(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size


        import re
        freeze = encoder_conf.get("freeze", True)
        freeze_layer_num = int(encoder_conf.get("freeze_layer_num", -1))
        freeze_exclude_key = encoder_conf.get("freeze_exclude_key", None)

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


        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.num_softmax_layer = num_softmax_layer

        self.rpq = RandomProjectionQuantizer(
            dim = input_size * encoder.downsample_rate,
            num_codebooks = num_softmax_layer, # in USM, they used up to 16 for 5% gain
            codebook_size = codebook_size,
            codebook_dim = codebook_dim,
            norm = True,
        )

        self.out_linear = nn.Linear(encoder_output_size, (codebook_size + 1) * num_softmax_layer)

        self.rpq.requires_grad = False
        self.pad_id = 0
        self.encoder = encoder
        self.causal = causal
        self.encoder_output_size = encoder_output_size
    
    def rq_mask(self,
        x,
        lengths,
        mask_indices=None,
    ):
        batch_size, sequence_length, feature_dim = x.shape

        if mask_indices is None:
            mask_starts = torch.bernoulli(
                torch.ones(batch_size, sequence_length) * self.mask_prob
            ).bool().to(x.device)

        for i in range(batch_size):
            padding_length = sequence_length - lengths[i]
            mask_starts[i, -(self.mask_length + padding_length) + 1:] = False
            if torch.all(mask_starts[i] == False).item():
                mask_starts[i, 0] = True

        mask_indices = torch.zeros(batch_size, sequence_length,
                                   dtype=torch.bool, device=x.device)

        for i in range(self.mask_length):
            mask_indices = mask_indices | torch.roll(mask_starts, shifts=i, dims=1)

        masked_x = x.clone()
        #masked_x[mask_indices] = 0.0
        noise = 0.1 * torch.randn_like(x)
        masked_x[mask_indices.unsqueeze(-1).expand_as(x)] = noise[mask_indices.unsqueeze(-1).expand_as(x)]

        return masked_x, mask_indices

    def forward(self, 
        speech: torch.Tensor, 
        speech_lengths: torch.Tensor,
        # text: torch.Tensor,
        # text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        # print(speech.size())
        # print(max(speech_lengths))
        speech = speech[:, :3000]
        speech_lengths[speech_lengths>3000] = 3000

        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size, time, dims = speech.shape

        target_size = speech.size(1) // 4 * 4
        ds_speech = speech[:,:target_size].contiguous().view(speech.size(0), target_size // 4, -1)
        labels = self.rpq(ds_speech) # bsz, time, num_softmax_layer


        if self.num_softmax_layer == 1:
            labels = torch.unsqueeze(labels, dim=-1)

        for i in range(batch_size):
            padding_length = target_size // 4 - speech_lengths[i] // 4
            if padding_length > 0:
                labels[i, -padding_length:, :] = self.pad_id

        mask_feats, mask_indices = self.rq_mask(ds_speech, speech_lengths // 4)
        mask_feats = mask_feats.contiguous().view(speech.size(0), -1, speech.size(-1))

        enc_out, _ = self.encoder(mask_feats, speech_lengths // 4 * 4, time_first = True)

        target_size = min(enc_out.size(1), labels.size(1))

        enc_out = enc_out[:,:target_size]
        labels = labels[:,:target_size]
        mask_indices = mask_indices[:,:target_size]

        loss, acc = 0.0, 0.0
        logits = self.out_linear(enc_out)
        logits = logits.contiguous().view(batch_size, target_size, self.num_softmax_layer, self.codebook_size + 1)
        logits = logits.permute(0, 2, 1, 3).contiguous().view(-1, self.codebook_size + 1)
        
        masked_labels = labels.masked_fill(~mask_indices.unsqueeze(-1), self.pad_id)
        masked_labels = masked_labels.permute(0, 2, 1).contiguous().view(-1, target_size)

        try:
            acc = th_accuracy(logits, masked_labels, ignore_label=self.pad_id)
        except:
            acc = 0

        masked_labels = masked_labels.contiguous().view(-1)
        loss = F.cross_entropy(logits, masked_labels, ignore_index = self.pad_id)


        pos = masked_labels != self.pad_id

        # print(mask_feats.size())
        # print(mask_indices.size())
        # print(ds_speech.size())
        # print(labels.size())
        # print(mask_indices[0, pos])

        # print(masked_labels[pos].squeeze(-1))
        # print(logits.argmax(-1)[pos])

        stats = dict()
        stats["loss"] = loss.detach()
        stats["acc"] = acc
        stats["batch_size"] = speech.size(0)
        stats["batch_token"] = speech.size(0) * speech.size(1)
        

        batch_size_speech, frames, _ = speech.shape
        stats["batch_size_speech"] = batch_size_speech
        stats["batch_size_x_frames"] = frames * batch_size_speech
        stats["batch_size_real_frames"] = speech_lengths.sum().item()
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]
        
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
      

@tables.register("model_classes", "BestRQ_QwenCTC2")
class BestRQ_QwenCTC(torch.nn.Module):
    def __init__(
        self,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        mask_prob: float = 0.1,
        mask_length: int = 6,
        input_size: int = 80,
        num_softmax_layer: int = 1,
        causal: bool = False,
        audio_adaptor_conf: Optional[Dict] = None,
        ctc_size: int = 58000,
        **kwargs,
    ):
        super().__init__()

        encoder = QwenAudioEncoder(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size


        import re
        freeze = encoder_conf.get("freeze", True)
        freeze_layer_num = int(encoder_conf.get("freeze_layer_num", -1))
        freeze_exclude_key = encoder_conf.get("freeze_exclude_key", None)

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

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.num_softmax_layer = num_softmax_layer

        self.rpq = RandomProjectionQuantizer(
            dim = input_size,
            num_codebooks = num_softmax_layer, # in USM, they used up to 16 for 5% gain
            codebook_size = codebook_size,
            codebook_dim = codebook_dim,
            norm = True,
        )

        self.out_linear = nn.Linear(encoder_output_size, (codebook_size + 1) * num_softmax_layer)

        self.rpq.requires_grad = False
        self.pad_id = 0
        self.encoder = encoder
        self.causal = causal  

        from whisper.tokenizer import get_tokenizer
        self.tokenizer = get_tokenizer(multilingual=True, num_languages=99, language=None, task="transcribe")

        adaptor_class = tables.adaptor_classes.get("Transformer")
        self.audio_adaptor = adaptor_class(**audio_adaptor_conf)
        self.ctc_linear = torch.nn.Linear(audio_adaptor_conf["llm_dim"], ctc_size)
        self.ctc_loss = torch.nn.CTCLoss(reduction="none")


    def rq_mask(self,
        x,
        lengths,
        mask_indices=None,
    ):
        batch_size, sequence_length, feature_dim = x.shape

        if mask_indices is None:
            mask_starts = torch.bernoulli(
                torch.ones(batch_size, sequence_length) * self.mask_prob
            ).bool().to(x.device)

        for i in range(batch_size):
            padding_length = sequence_length - lengths[i]
            mask_starts[i, -(self.mask_length + padding_length) + 1:] = False
            if torch.all(mask_starts[i] == False).item():
                mask_starts[i, 0] = True

        mask_indices = torch.zeros(batch_size, sequence_length,
                                   dtype=torch.bool, device=x.device)

        for i in range(self.mask_length):
            mask_indices = mask_indices | torch.roll(mask_starts, shifts=i, dims=1)

        masked_x = x.clone()
        #masked_x[mask_indices] = 0.0
        noise = 0.1 * torch.randn_like(x)
        masked_x[mask_indices.unsqueeze(-1).expand_as(x)] = noise[mask_indices.unsqueeze(-1).expand_as(x)]

        return masked_x, mask_indices

    def forward(self, 
        speech: torch.Tensor, 
        speech_lengths: torch.Tensor,
        ctc_labels: torch.Tensor = None,
        ctc_labels_len: torch.Tensor = None,
        # text: torch.Tensor,
        # text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # print(speech.size())

        speech = speech[:, :3000]
        speech_lengths[speech_lengths>3000] = 3000

        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]
        target_ids = ctc_labels
        if len(target_ids.shape) == 3:
            target_ids = 1 + target_ids.squeeze(-1)
        target_lengths = ctc_labels_len
        if len(target_lengths.size()) > 1:
            target_lengths = target_lengths.squeeze(-1)

        batch_size, time, dims = speech.shape

        enc_out, enc_length = self.encoder(speech, speech_lengths, time_first = True)

        ctc_logits, th_ilen = self.audio_adaptor(enc_out, enc_length)
        ctc_logits = self.ctc_linear(ctc_logits)
        ctc_log_probs = ctc_logits.log_softmax(2).transpose(0,1)

        mask = th_ilen > 2 * target_lengths

        if torch.any(mask):
            good_ctc_log_probs = ctc_log_probs[:, mask]
            good_target_ids = target_ids[mask]
            good_th_ilen = th_ilen[mask]
            good_target_lengths = target_lengths[mask]
        else:
            good_ctc_log_probs = ctc_log_probs
            good_target_ids = target_ids
            good_th_ilen = th_ilen
            good_target_lengths = target_lengths
            good_target_lengths[:] = 1

        loss = self.ctc_loss(good_ctc_log_probs, good_target_ids, good_th_ilen, good_target_lengths)
        loss = loss.mean()

        asr_cer = calculate_cer_ctc(ctc_logits.float().detach().cpu().numpy(), target_ids.detach().cpu().numpy())


        if torch.isinf(loss) or torch.isnan(loss):
            print(th_ilen[~mask])
            print(target_lengths[~mask])
            print(self.tokenizer.decode(target_ids[~mask][0,:target_lengths[0]]))
            print(loss)

        # exit()
        stats = dict()
        stats["loss"] = loss.detach()
        stats["asr_cer"] = asr_cer
        stats["batch_size"] = len(speech)
        stats["batch_token"] = len(speech) * speech.size(1)


        batch_size_speech, frames, _ = speech.shape
        stats["batch_size_speech"] = batch_size_speech
        stats["batch_size_x_frames"] = frames * batch_size_speech
        stats["batch_size_real_frames"] = speech_lengths.sum().item()
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]
        
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight