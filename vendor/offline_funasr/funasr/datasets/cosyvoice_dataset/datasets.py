_S3_ROOT = next((p for p in __import__('pathlib').Path(__file__).resolve().parents if (p / 'env.sh').is_file() and (p / 'tools').is_dir()))
import logging
import re
import torch
import random
import traceback
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
import math
import numpy as np
import onnxruntime
import torchaudio.compliance.kaldi as kaldi

@tables.register('dataset_classes', 'CosyVoice')
class CosyVoice(torch.utils.data.Dataset):
    """
    Dataset for CosyVoice
    """

    def __init__(self, path, index_ds: str=None, frontend=None, tokenizer=None, int_pad_value: int=-1, float_pad_value: float=0.0, **kwargs):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get('preprocessor_speech', None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(**kwargs.get('preprocessor_speech_conf'))
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get('preprocessor_text', None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get('preprocessor_text_conf'))
        self.preprocessor_text = preprocessor_text
        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = 'sound'
        self.tokenizer = tokenizer
        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get('sos', '<|startoftranscript|>')
        self.eos = kwargs.get('eos', '<|endoftext|>')
        self.batch_size = kwargs.get('batch_size')
        self.batch_type = kwargs.get('batch_type')
        self.prompt_ids_len = 0
        self.retry = kwargs.get('retry', 100)
        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend
        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True
        self.pattern = re.compile('(<\\|startofspeech\\|>.*?<\\|endofspeech\\|>)')
        self.max_token_length = kwargs.get('max_token_length', 1500)
        self.batch_size_scale_ratio_max = kwargs.get('batch_size_scale_ratio_max', 1.5)
        self.batch_size_token_max = kwargs.get('batch_size_token_max', 2500)
        self.multiturn_num_max = kwargs.get('multiturn_num_max', 5)
        self.max_source_length = kwargs.get('max_source_length', 3000)
        self.num_codec = kwargs.get('num_codec', 4096)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        output = None
        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()
            item = self.index_ds[index_cur]
            key = item.get('key', None)
            system = item['system']
            user = item['user']
            assistant = item['assistant']
            input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len, codec, codec_len, audio, audio_len, input_mask, input_mask_beg, spk_emb, spk_emb_len, codec_beg, fake_codec_len = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
            multiturn_num = len(system)
            for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(f'input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}')
                    break
                if user_prompt.startswith('语音合成：'):
                    user_prompt = user_prompt.replace('语音合成：', '', 1)
                elif user_prompt.startswith('Speech Synthesis:'):
                    user_prompt = user_prompt.replace('Speech Synthesis:', '', 1)
                source_input = f'<|im_start|>{user_prompt}<|im_end|>'
                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                input_mask_i = []
                target_ids = []
                for k, sub_str in enumerate(splits):
                    if len(sub_str) < 1:
                        continue
                    if not sub_str.startswith('<|startofspeech|>'):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace('<|startofspeech|>', '').replace('<|endofspeech|>', '')
                        if sub_str.startswith('!'):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                            except Exception as e:
                                logging.error(f'Loading wav failed! {str(e)}, {traceback.format_exc()}')
                                badcase_flag = True
                                continue
                            speech, speech_lengths = extract_fbank(data_src, data_type=self.data_type, frontend=self.frontend, is_final=True)
                            if speech_lengths > self.max_source_length:
                                logging.info(f'speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}')
                                badcase_flag = True
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            fbank_mask_i += [1] * len(fake_token)
                if badcase_flag:
                    continue
                meta = None
                if isinstance(target_out, (list, tuple)):
                    target_out, meta = target_out
                    if isinstance(meta, dict):
                        if 'wav_path' in meta:
                            wav_path = meta['wav_path']
                            wav_samples = load_audio_text_image_video(wav_path, fs=self.fs)
                            audio.append(wav_samples)
                            audio_len.append(len(wav_samples))
                        if 'spk_emb':
                            spk_emb_path = meta['spk_emb']
                            sub_spk_emb = np.load(spk_emb_path)
                            sub_spk_emb = torch.from_numpy(sub_spk_emb)
                            spk_emb.append(sub_spk_emb)
                            spk_emb_len.append(len(sub_spk_emb))
                target_out = f'{target_out}<|im_end|>'
                splits = self.pattern.split(target_out)
                codec_i = []
                sub_token = []
                codec_beg_i = ([],)
                fake_codec_len_i = ([],)
                for k, sub_str in enumerate(splits):
                    if len(sub_str) < 1:
                        continue
                    if not sub_str.startswith('<|startofspeech|>'):
                        sub_token = self.tokenizer.encode(sub_str)
                    else:
                        sub_str = sub_str.replace('<|startofspeech|>', '').replace('<|endofspeech|>', '')
                        if sub_str.startswith('!'):
                            sub_token_codec = np.load(sub_str[1:])
                            fake_token = sub_token_codec.tolist() + [self.num_codec]
                            codec_i = torch.tensor(sub_token_codec, dtype=torch.int64)
                            codec_i_len = len(sub_token_codec)
                            target_ids += fake_token
                if len(codec_i) > 0:
                    codec.append(codec_i)
                    codec_len.append(codec_i_len)
                fbank_beg += [fbank_beg_i + len(input_ids)]
                fake_token_len += [fake_token_len_i]
                source_mask = [-100] * len(source_ids)
                codec_beg_i = len(source_ids)
                fake_codec_len_i = len(codec_i)
                codec_beg += [codec_beg_i + len(input_ids)]
                fake_codec_len += [fake_codec_len_i]
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                fbank_mask += fbank_mask_i
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)
            if badcase_flag:
                continue
            input_ids = torch.tensor(input_ids, dtype=torch.int64)
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)
            fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
            output = {'fbank_beg': fbank_beg, 'fake_token_len': fake_token_len, 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels_ids': labels}
            if len(codec) > 0:
                codec_len = torch.tensor(codec_len, dtype=torch.int32)
                codec_beg = torch.tensor(codec_beg, dtype=torch.int32)
                fake_codec_len = torch.tensor(fake_codec_len, dtype=torch.int32)
                output['codec'] = codec
                output['codec_len'] = codec_len
                output['codec_beg'] = codec_beg
                output['fake_codec_len'] = fake_codec_len
            if len(fbank) > 0:
                output['speech'] = fbank
                output['speech_lengths'] = fbank_lens
            if len(audio) > 0:
                output['audio'] = audio
                output['audio_len'] = audio_len
            if len(spk_emb) > 0:
                output['spk_emb'] = spk_emb
                output['spk_emb_len'] = torch.tensor(spk_emb_len, dtype=torch.int32)
            if key is not None:
                output['key'] = key
            break
        return output

    def collator(self, samples: list=None):
        for idx in range(self.retry):
            badcase_flag = False
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])
            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value
                    outputs[key] = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=pad_value)
            if self.batch_type != 'example':
                b, t = outputs['input_ids'].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(f'Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data')
                    samples = samples[:-1]
                    continue
            break
        return outputs

@tables.register('dataset_classes', 'CosyVoiceMixedLM')
class CosyVoiceMixedLM(torch.utils.data.Dataset):
    """
    Dataset for Mixed LM of CosyVoice
    """

    def __init__(self, path, index_ds: str=None, frontend=None, tokenizer=None, int_pad_value: int=-1, float_pad_value: float=0.0, **kwargs):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = 'sound'
        self.tokenizer = tokenizer
        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = tokenizer.encode(kwargs.get('sos', '<|im_start|>'))[0]
        self.eos = tokenizer.encode(kwargs.get('eos', '<|im_end|>'))[0]
        self.batch_size = kwargs.get('batch_size')
        self.batch_type = kwargs.get('batch_type')
        self.retry = kwargs.get('retry', 100)
        self.max_token_length = kwargs.get('max_token_length', 1500)
        self.batch_size_scale_ratio_max = kwargs.get('batch_size_scale_ratio_max', 1.5)
        self.batch_size_token_max = kwargs.get('batch_size_token_max', 2500)
        self.multiturn_num_max = kwargs.get('multiturn_num_max', 1)
        self.max_source_length = kwargs.get('max_source_length', 3000)
        self.codebook_size = kwargs.get('codebook_size', 4096)
        self.codec_eos_token = kwargs.get('codec_eos_token', self.codebook_size)
        self.fill_token = kwargs.get('fill_token', self.codebook_size + 1)
        self.token_text_ratio = kwargs.get('token_text_ratio', 2)
        self.mixup_prob = kwargs.get('mixup_prob', 0.5)
        self.ignore_id = kwargs.get('ignore_id', -100)
        specaug = kwargs.get('specaug', None)
        specaug_conf = kwargs.get('specaug_conf', {})
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        self.specaug = specaug

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def mixup_text_codec(self, text: torch.Tensor, aug_codec: torch.Tensor, codec: torch.Tensor):
        text_len = text.shape[0]
        codec_len = codec.shape[0]
        can_mixup = text_len * self.token_text_ratio < codec_len
        if random.random() <= self.mixup_prob and can_mixup:
            mixed_seq = torch.zeros(text_len * (self.token_text_ratio + 1), dtype=torch.int64)
            mixed_seq[0::self.token_text_ratio + 1] = text
            for i in range(self.token_text_ratio):
                mixed_seq[i + 1::self.token_text_ratio + 1] = aug_codec[:text_len * self.token_text_ratio][i::self.token_text_ratio]
            sequence = [self.sos, *mixed_seq.tolist(), self.eos, *aug_codec[text_len * self.token_text_ratio:].tolist(), self.codec_eos_token]
            input_ids = torch.tensor(sequence, dtype=torch.int64)
            mixed_seq = torch.zeros(text_len * (self.token_text_ratio + 1), dtype=torch.int64)
            mixed_seq[0::self.token_text_ratio + 1] = self.fill_token
            for i in range(self.token_text_ratio):
                mixed_seq[i + 1::self.token_text_ratio + 1] = codec[:text_len * self.token_text_ratio][i::self.token_text_ratio]
            label_sequence = [self.ignore_id, *mixed_seq.tolist(), self.ignore_id, *codec[text_len * self.token_text_ratio:].tolist(), self.codec_eos_token]
            labels = torch.tensor(label_sequence, dtype=torch.int64)
            flag_seq = torch.zeros(text_len * (self.token_text_ratio + 1), dtype=torch.int64)
            flag_seq[0::self.token_text_ratio + 1] = 1
            text_flag = [1, *flag_seq.tolist(), 1, *torch.zeros_like(codec[text_len * self.token_text_ratio:]).tolist(), 0]
            text_flag = torch.tensor(text_flag, dtype=torch.int64)
            codec_flag = 1 - text_flag
            return (input_ids, labels, text_flag, codec_flag)
        else:
            sequence = [self.sos, *text.tolist(), self.eos, *aug_codec.tolist(), self.codec_eos_token]
            input_ids = torch.tensor(sequence, dtype=torch.int64)
            text_flag = torch.zeros(len(sequence), dtype=torch.float32)
            text_flag[:text_len + 2] = 1
            codec_flag = 1 - text_flag
            sequence = [self.sos, *text.tolist(), self.eos, *codec.tolist(), self.codec_eos_token]
            labels = torch.tensor(sequence, dtype=torch.int64)
            labels[:text_len + 2] = self.ignore_id
            return (input_ids, labels, text_flag, codec_flag)

    def __getitem__(self, index):
        output = None
        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()
            item = self.index_ds[index_cur]
            key = item.get('key', None)
            user_prompt = item['user'][0]
            target_out = item['assistant'][0]
            if user_prompt.startswith('语音合成：'):
                user_prompt = user_prompt.replace('语音合成：', '', 1)
            elif user_prompt.startswith('Speech Synthesis:'):
                user_prompt = user_prompt.replace('Speech Synthesis:', '', 1)
            text_ids = torch.tensor(self.tokenizer.encode(user_prompt), dtype=torch.int32)
            meta = None
            if isinstance(target_out, (list, tuple)):
                target_out, meta = target_out
            target_out = target_out.replace('<|startofspeech|>!', '').replace('<|endofspeech|>', '')
            codec = torch.from_numpy(np.load(target_out))
            aug_codec = codec.clone()
            if self.specaug is not None:
                aug_codec, _ = self.specaug(aug_codec.float().unsqueeze(0).unsqueeze(-1))
                aug_codec = aug_codec.squeeze(0).squeeze(-1).long()
            codec_len = codec.shape[0]
            input_ids, labels, text_flag, codec_flag = self.mixup_text_codec(text_ids, aug_codec, codec)
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            audio, spk_emb = (None, None)
            if meta is not None and isinstance(meta, dict):
                if 'wav_path' in meta:
                    wav_path = meta['wav_path']
                    audio = load_audio_text_image_video(wav_path, fs=self.fs)
                    audio_len = audio.shape[0]
                if 'spk_emb':
                    spk_emb_path = meta['spk_emb']
                    spk_emb = np.load(spk_emb_path)
                    spk_emb = torch.from_numpy(spk_emb)
                    spk_emb_len = spk_emb.shape[0]
            output = {'input_ids': input_ids, 'labels_ids': labels, 'attention_mask': attention_mask, 'text_flag': text_flag, 'codec_flag': codec_flag}
            if len(codec) > 0:
                output['codec'] = codec
                output['codec_len'] = torch.tensor([codec_len], dtype=torch.int32)
            if audio is not None:
                output['audio'] = audio
                output['audio_len'] = torch.tensor([audio_len], dtype=torch.int32)
            if spk_emb is not None:
                output['spk_emb'] = spk_emb
                output['spk_emb_len'] = torch.tensor([spk_emb_len], dtype=torch.int32)
            if key is not None:
                output['key'] = key
            break
        return output

    def collator(self, samples: list=None):
        for idx in range(self.retry):
            badcase_flag = False
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])
            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value
                    outputs[key] = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=pad_value)
            if self.batch_type != 'example':
                b, t = outputs['input_ids'].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(f'Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data')
                    samples = samples[:-1]
                    continue
            break
        return outputs

@tables.register('dataset_classes', 'CosyVoiceFlow')
class CosyVoiceFlow(torch.utils.data.Dataset):
    """
    Dataset for CosyVoice
    """

    def __init__(self, path, index_ds: str=None, frontend=None, tokenizer=None, int_pad_value: int=-1, float_pad_value: float=0.0, **kwargs):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        self.frontend = frontend
        self.fs = 24000 if frontend is None else frontend.fs
        self.data_type = 'sound'
        self.tokenizer = tokenizer
        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.batch_size = kwargs.get('batch_size')
        self.batch_type = kwargs.get('batch_type')
        self.retry = kwargs.get('retry', 100)
        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend
        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True
        self.batch_size_token_max = kwargs.get('batch_size_token_max', 2500)
        self.wav_token_ratio = kwargs.get('wav_token_ratio', 960)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def _align_wav_token_length(self, token, audio, uid: str):
        if self.wav_token_ratio is None:
            return (token, audio)
        exp_audio_len = token.shape[0] * self.wav_token_ratio
        if audio.shape[0] > exp_audio_len + 5 * self.wav_token_ratio or audio.shape[0] < exp_audio_len - 5 * self.wav_token_ratio:
            logging.warning(f'{uid}: may have wrong shape: {audio.shape}, {token.shape}.')
        if audio.shape[0] > exp_audio_len:
            audio = audio[:exp_audio_len]
        elif audio.shape[0] < exp_audio_len:
            exp_token_len = audio.shape[0] // self.wav_token_ratio
            token = token[:exp_token_len]
            audio = audio[:exp_token_len * self.wav_token_ratio]
        return (token, audio)

    def __getitem__(self, index):
        output = None
        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()
            item = self.index_ds[index_cur]
            key = item.get('key', None)
            system = item['system']
            user = item['user']
            assistant = item['assistant']
            codec, codec_len, audio, audio_len, spk_emb, spk_emb_len = ([], [], [], [], [], [])
            for i, (system_prompt, user_prompt, assistant_content) in enumerate(zip(system, user, assistant)):
                assert isinstance(assistant_content, (list, tuple))
                codec_path, meta_dict = assistant_content
                assert codec_path.startswith('<|startofspeech|>!')
                codec_path = codec_path.replace('<|startofspeech|>!', '').replace('<|endofspeech|>', '')
                codec_i = np.load(codec_path)
                codec_i = torch.tensor(codec_i, dtype=torch.int64)
                assert isinstance(meta_dict, dict)
                assert 'wav_path' in meta_dict
                wav_path = meta_dict['wav_path']
                wav_samples = load_audio_text_image_video(wav_path, fs=self.fs)
                codec_i, wav_samples = self._align_wav_token_length(codec_i, wav_samples, key)
                codec_i_len = len(codec_i)
                codec.append(codec_i)
                codec_len.append(codec_i_len)
                audio.append(wav_samples)
                audio_len.append(len(wav_samples))
                if 'spk_emb':
                    spk_emb_path = meta_dict['spk_emb']
                    sub_spk_emb = np.load(spk_emb_path)
                    sub_spk_emb = torch.from_numpy(sub_spk_emb)
                    spk_emb.append(sub_spk_emb)
                    spk_emb_len.append(len(sub_spk_emb))
                break
            output = {}
            if len(codec) > 0:
                output['codec'] = codec
                output['codec_len'] = torch.tensor(codec_len, dtype=torch.int32)
            if len(audio) > 0:
                output['audio'] = audio
                output['audio_len'] = torch.tensor(audio_len, dtype=torch.int32)
            if len(spk_emb) > 0:
                output['spk_emb'] = spk_emb
                output['spk_emb_len'] = torch.tensor(spk_emb_len, dtype=torch.int32)
            if key is not None:
                output['key'] = key
            break
        return output

    def collator(self, samples: list=None):
        for idx in range(self.retry):
            badcase_flag = False
            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])
            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:
                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value
                    outputs[key] = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=pad_value)
            if self.batch_type != 'example':
                b, t = outputs['codec'].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(f'Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data')
                    samples = samples[:-1]
                    continue
            break
        return outputs
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
    if kwargs.get('debug', False):
        import pdb
        pdb.set_trace()
    from index_ds import OpenAIIndexDSJsonlForFullDuplexVAD
    index_ds = OpenAIIndexDSJsonlForFullDuplexVAD(path='/path/to/vad_test_samples.jsonl')
    print(index_ds.contents)
    kwargs = OmegaConf.to_container(kwargs, resolve=True)
    frontend = kwargs.get('frontend', None)
    if frontend is not None:
        frontend_class = tables.frontend_classes.get(frontend)
        frontend = frontend_class(**kwargs['frontend_conf'])
        kwargs['frontend'] = frontend
        kwargs['input_size'] = frontend.output_size()
        tokenizer = kwargs.get('tokenizer', None)
        if tokenizer is not None:
            tokenizer_class = tables.tokenizer_classes.get(tokenizer)
            tokenizer = tokenizer_class(**kwargs.get('tokenizer_conf', {}))
            kwargs['token_list'] = tokenizer.token_list if hasattr(tokenizer, 'token_list') else None
            kwargs['token_list'] = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else kwargs['token_list']
            vocab_size = len(kwargs['token_list']) if kwargs['token_list'] is not None else -1
            if vocab_size == -1 and hasattr(tokenizer, 'get_vocab_size'):
                vocab_size = tokenizer.get_vocab_size()
        else:
            vocab_size = -1
        kwargs['tokenizer'] = tokenizer
    dataset_class = tables.dataset_classes.get(kwargs.get('dataset', 'AudioDataset'))
    dataset_train = dataset_class(kwargs.get('train_data_set_list'), frontend=frontend, tokenizer=tokenizer, is_training=False, **kwargs.get('dataset_conf'))
    batch_sampler = kwargs['dataset_conf'].get('batch_sampler', 'BatchSampler')
    batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
    dataset_conf = kwargs.get('dataset_conf')
    print(f'dataset_conf:\n{dataset_conf}')
    batch_sampler_train = batch_sampler_class(dataset_train, is_training=False, **dataset_conf)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, collate_fn=dataset_train.collator, **batch_sampler_train)
    for batch_idx, batch in enumerate(dataloader_train):
        print(batch_idx)
        for key in batch.keys():
            print(key)
            print(batch[key].size())
            if key != 'speech':
                print(batch[key].tolist())
        break

class EmbeddingExtraction:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.max_len = 10 * 16000
        self.campplus_session = onnxruntime.InferenceSession('__S3_ROOT__/models/speaker/campplus.onnx'.replace('__S3_ROOT__', str(_S3_ROOT)), sess_options=option, providers=['CPUExecutionProvider'])

    def extract_embedding(self, speech):
        speech = load_audio_text_image_video('/path/to/longanhuan.wav', fs=16000)
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        if speech.ndim == 1:
            speech = speech.unsqueeze(0)
        if speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)
        if speech.shape[1] > self.max_len:
            start_index = random.randint(0, speech.shape[1] - self.max_len)
            speech = speech[:, start_index:start_index + self.max_len]
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None, {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        return torch.tensor(embedding).to(speech.device)
if __name__ == '__main__':
    main_hydra()
    '\n    local_path="/path/to/vad"\n    config="Speech2Text_Align_val_exp3_last8.yaml"\n\n    python funasr/datasets/openai_datasets/datasets.py     --config-path "${local_path}"     --config-name "${config}"     ++train_data_set_list="/path/to/vad_test_samples.jsonl"\n    '
