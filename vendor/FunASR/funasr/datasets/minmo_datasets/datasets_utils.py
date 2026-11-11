import logging
import os.path
import re
import torch
import random
import traceback
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
import math
import numpy as np
import random

from funasr.utils import load_utils

@tables.register("dataset_classes", "ClearWavData")
class ClearWavData(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.do_think = kwargs.get("do_think", True)
        
        self.kwargs = kwargs

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = {}

        for idx in range(self.retry):
            badcase_flag = False
            if idx <= 20:
                index_cur = index
            else:
                output['item'] = item
                output['item']['badcase'] = True
                logging.warning(f"badcase, failed: {idx}: {item}")
                return output

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break

                splits = self.pattern.split(user_prompt)
                audio_path_new_exist = False
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        if len(sub_str) > 0:
                            usr_prompt_prefix = sub_str
                        # sub_token = self.tokenizer.encode(sub_str)
                        # source_ids += sub_token
                        # fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            audio_path_new_exist, audio_path_new = load_utils.exist_audio_byteio_in_oss(sub_str[1:], data=None,
                                                                                  key_words=self.kwargs.get("key_words",
                                                                                                            "gzf"),
                                                                                  convert_flac=self.kwargs.get(
                                                                                      'convert_flac', True),
                                                                                  overwrite_outfile=self.kwargs.get(
                                                                                      "overwrite_outfile", False))
                            if audio_path_new_exist:
                                speech_length = -1
                                continue
                                
    
                            # if load_utils.valid_file_exist_oss(sub_str[1:], key_words=self.kwargs.get("key_words", "gzf")) and not self.kwargs.get("overwrite_outfile", False):
                            #     badcase_flag = True
                            #     logging.warning(f"file_exist_oss!")
                            #     continue
                            try:
                                if not os.path.exists(sub_str[1:]) and (sub_str[1:].startswith("/mnt/") or sub_str[1:].startswith("oss") or sub_str[1:].startswith("https://oss")):
                                    audio_byteio = load_utils.get_audio_byteio_from_oss(sub_str[1:])
                                    is_path = False
                                else:
                                    audio_byteio = sub_str[1:]
                                    is_path = True
                                if self.kwargs.get('convert_flac', True):
                                    audio_flac, duration = load_utils.convert_audio_to_flac(audio_byteio, sr=self.fs, is_path=is_path, raw_path=sub_str[1:])
                                else:
                                    audio_flac = audio_byteio
                                    data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                                    duration = int(data_src.shape[0]/ self.fs)
                                speech_length = int(duration*100)
                                audio_path_new = load_utils.write_audio_byteio_to_oss(sub_str[1:], audio_flac, key_words=self.kwargs.get("key_words", "gzf"), convert_flac=self.kwargs.get('convert_flac', True), overwrite_outfile=self.kwargs.get("overwrite_outfile", False))
                            except Exception as e:
                                logging.warning(
                                    f"convert_audio_to_flac failed! {str(e)}, {traceback.format_exc()}\nitem{item}"
                                )
                                badcase_flag = True
                                continue
                                
                text_length = len(self.tokenizer.encode(target_out))
                has_repeat_tokens = self.preprocessor_text(target_out)
                # if has_repeat_tokens:
                #     # badcase_flag = True
                #     logging.info(f"item has_repeat_tokens: {item}")
                if badcase_flag or audio_path_new_exist:
                    continue


            if badcase_flag:
                logging.warning(f"badcase, retry: {idx}: {item}")
                continue
            # if audio_path_new_exist:
            #     break

            output = {}
            output["item"] = item
            user_prompt_new = usr_prompt_prefix + f'<|startofspeech|>!/mnt/{audio_path_new}<|endofspeech|>'
            output['item']['meta']['messages'][1]['content'] = user_prompt_new
            if speech_length < 0:
                speech_length = output['item']['meta'].get('speech_length', 0)
                if isinstance(speech_length, str):
                    speech_length = int(speech_length)
                elif isinstance(speech_length, (list, tuple)):
                    speech_length = int(speech_length[0])
                    
            output['item']['meta']['speech_length'] = speech_length
            output['item']['meta']['text_length'] = text_length
            output['item']['has_repeat_tokens'] = has_repeat_tokens
            break


        return output

    def collator(self, samples: list = None):

        return samples

@tables.register("dataset_classes", "GenLength")
class GenLength(torch.utils.data.Dataset):
    """
    GenLength
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.do_think = kwargs.get("do_think", True)
        from funasr.utils.load_utils import get_audio_byteio_from_oss
        from funasr.utils.load_utils import write_audio_byteio_to_oss
        from funasr.utils.load_utils import convert_audio_to_flac
        self.get_audio_byteio_from_oss = get_audio_byteio_from_oss
        self.write_audio_byteio_to_oss = write_audio_byteio_to_oss
        self.convert_audio_to_flac = convert_audio_to_flac

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break

                splits = self.pattern.split(user_prompt)
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        if len(sub_str) > 0:
                            usr_prompt_prefix = sub_str
                        # sub_token = self.tokenizer.encode(sub_str)
                        # source_ids += sub_token
                        # fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                                speech_length = int(data_src.shape[0]/self.fs)
                            except Exception as e:
                                logging.warning(
                                    f"convert_audio_to_flac failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                text_length = len(self.tokenizer.encode(target_out))
                has_repeat_tokens = self.preprocessor_text(target_out)
                if has_repeat_tokens:
                    badcase_flag = True
                    logging.info(f"item has_repeat_tokens: {item}")
                if badcase_flag:
                    continue

            if badcase_flag:
                continue

            output = {}
            output["item"] = item
            # output["audio_path_new"] = audio_path_new
            output['item']['meta']['speech_length'] = speech_length
            output['item']['meta']['text_length'] = text_length
            break

        return output

    def collator(self, samples: list = None):

        return samples


import hydra
import logging
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
    if kwargs.get("debug", False):
        import pdb

        pdb.set_trace()

    from index_ds import OpenAIIndexDSJsonlForFullDuplexVAD

    index_ds = OpenAIIndexDSJsonlForFullDuplexVAD(
        path="/path/to/vad_test_samples.jsonl"
    )
    print(index_ds.contents)

    kwargs = OmegaConf.to_container(kwargs, resolve=True)
    # build frontend if frontend is none None
    frontend = kwargs.get("frontend", None)
    if frontend is not None:
        frontend_class = tables.frontend_classes.get(frontend)
        frontend = frontend_class(**kwargs["frontend_conf"])
        kwargs["frontend"] = frontend
        kwargs["input_size"] = frontend.output_size()

        # build tokenizer
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            tokenizer_class = tables.tokenizer_classes.get(tokenizer)
            tokenizer = tokenizer_class(**kwargs.get("tokenizer_conf", {}))
            kwargs["token_list"] = (
                tokenizer.token_list if hasattr(tokenizer, "token_list") else None
            )
            kwargs["token_list"] = (
                tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else kwargs["token_list"]
            )
            vocab_size = len(kwargs["token_list"]) if kwargs["token_list"] is not None else -1
            if vocab_size == -1 and hasattr(tokenizer, "get_vocab_size"):
                vocab_size = tokenizer.get_vocab_size()
        else:
            vocab_size = -1
        kwargs["tokenizer"] = tokenizer

    # dataset
    dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "AudioDataset"))
    dataset_train = dataset_class(
        kwargs.get("train_data_set_list"),
        frontend=frontend,
        tokenizer=tokenizer,
        is_training=False,
        **kwargs.get("dataset_conf"),
    )

    # dataloader
    batch_sampler = kwargs["dataset_conf"].get("batch_sampler", "BatchSampler")
    batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
    dataset_conf = kwargs.get("dataset_conf")
    print(f"dataset_conf:\n{dataset_conf}")
    batch_sampler_train = batch_sampler_class(dataset_train, is_training=False, **dataset_conf)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, collate_fn=dataset_train.collator, **batch_sampler_train
    )

    for batch_idx, batch in enumerate(dataloader_train):
        print(batch_idx)
        for key in batch.keys():
            print(key)
            print(batch[key].size())
            if key != "speech":
                print(batch[key].tolist())

        break


if __name__ == "__main__":
    main_hydra()
    """
    local_path="/path/to/vad"
    config="Speech2Text_Align_val_exp3_last8.yaml"

    python funasr/datasets/openai_datasets/datasets.py \
    --config-path "${local_path}" \
    --config-name "${config}" \
    ++train_data_set_list="/path/to/vad_test_samples.jsonl"
    """
