import os
import json
import torch
import logging

import librosa
import random
import torch.distributed as dist

from funasr.register import tables


@tables.register("index_ds_classes", "CosyVoice")
class CosyVoice(torch.utils.data.Dataset):  # torch.utils.data.Dataset

    def __init__(self, path: str, **kwargs):
        super().__init__()

        self.max_source_length = kwargs.get("max_source_length", 6000)
        # self.min_source_length = kwargs.get("min_source_length", 0)
        self.max_target_length = kwargs.get("max_target_length", 2048)
        # self.min_target_length = kwargs.get("min_target_length", 0)
        # self.max_token_length = kwargs.get("max_token_length", 2200)

        load_meta_data_key = kwargs.get("load_meta_data_key", None)

        is_training = kwargs.get("is_training", True)
        if not (path.endswith(".jsonl") or path.endswith(".json")):
            # jsonl list file
            data_split_num = kwargs.get("data_split_num", 1)
            data_split_i = kwargs.get("data_split_i", 0)

            if not is_training:
                data_split_num = 1
                data_split_i = 0
            with open(path, encoding="utf-8") as fin:
                file_list_all = fin.readlines()

                num_per_slice = (len(file_list_all) - 1) // data_split_num + 1  # 16
                file_list = file_list_all[
                    data_split_i * num_per_slice : (data_split_i + 1) * num_per_slice
                ]
                logging.info(
                    f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}, \nfile_list_all: {file_list_all}"
                )

        else:
            file_list = [path]

        contents = []
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    data = data_dict["messages"]
                    speech_length = data_dict.get("speech_length", -1) // 8
                    text_length = data_dict.get("text_length", 0)
                    if speech_length * 1.3 > self.max_source_length:
                        logging.info(
                            f"speech_length: {speech_length*8} > {self.max_source_length}, drop it: {data_dict}"
                        )
                        continue
                    if text_length > self.max_target_length:
                        logging.info(
                            f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
                        )
                        continue

                    system, user, assistant = [], [], []
                    for i, item in enumerate(data):
                        role = item["role"]
                        content = item["content"]
                        if role == "system":
                            system.append(content)
                        elif role == "user":
                            user.append(content)
                        elif role == "assistant":
                            if load_meta_data_key is not None:
                                meta_dict = {}
                                if isinstance(load_meta_data_key, str):
                                    load_meta_data_key = load_meta_data_key.split(",")
                                for key_tmp in load_meta_data_key:
                                    if key_tmp in item:
                                        wav_path = item[key_tmp]
                                        meta_dict[key_tmp] = wav_path
                                if not isinstance(content, (list, tuple)):
                                    content = [content]
                                content.append(meta_dict)

                            assistant.append(content)
                    if len(system) == 0:
                        system = ["You are a helpful assistant."]
                    system = system * len(user)

                    contents_i = {
                        "system": system,
                        "user": user,
                        "assistant": assistant,
                        "source_len": speech_length + text_length,
                    }
                    if "key" in data_dict:
                        contents_i["key"] = data_dict["key"]
                    contents.append(contents_i)

        self.contents = contents

        logging.info("total_num of samplers: {}, {}".format(len(self.contents), path))

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):

        data = self.contents[index]

        return data

    def get_source_len(self, data_dict):
        source_len = data_dict.get("source_len", -1)
        if source_len < 0:
            source_len = len(data_dict["system"]) + len(data_dict["user"])
        return source_len

    def get_target_len(self, data_dict):

        return 0


if __name__ == "__main__":
    index_ds = CosyVoice(path="/path/to/data.1.jsonl")
    print(index_ds.contents)
    pass
