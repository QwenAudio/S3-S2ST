import os
import json
import torch
import logging

import librosa
import random
import torch.distributed as dist

from funasr.register import tables


@tables.register("index_ds_classes", "IndexDSJsonlScpList")
class IndexDSJsonlScpList(torch.utils.data.Dataset):

    def __init__(self, path: str, **kwargs):
        super().__init__()
        self.max_source_length = kwargs.get("max_source_length", 2048)
        self.min_source_length = kwargs.get("min_source_length", 0)
        self.max_target_length = kwargs.get("max_target_length", 2048)
        self.min_target_length = kwargs.get("min_target_length", 0)
        self.max_token_length = kwargs.get("max_token_length", 2200)

        is_training = kwargs.get("is_training", True)
        if not (path.endswith(".jsonl") or path.endswith(".json")):
            # jsonl list file
            data_split_num = kwargs.get("data_split_num", 1)
            data_split_i = kwargs.get("data_split_i", 0)

            # if not is_training:
            #     data_split_num = 1
            #     data_split_i = 0
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



        # contents = []
        # for file_json in file_list_rank:
        contents = []
        for file_json in file_list:
            wav_scp = file_json.split()[0].strip()
            text_scp = file_json.split()[1].strip()
            with open(wav_scp, encoding="utf-8") as fin, open(text_scp, encoding="utf-8") as fin_text:
                while True:        # 读取wav_scp和text_scp
                    wav_line = fin.readline().strip()
                    text_line = fin_text.readline().strip()
                    if not wav_line or not text_line:
                        break
                        
                    key, wav_path = wav_line.split()
                    key2, text = text_line.split(maxsplit=1)
                    assert key == key2
                    contents_i = {
                        "wav_path": wav_path,
                        "text": text,
                        "key": key,
                    }

                    contents.append(contents_i)

        self.contents = contents

        logging.info("total_num of samplers: {}, {}".format(len(self.contents), path))

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):

        data = self.contents[index]

        return data

    def get_source_len(self, data_dict):
        return data_dict.get("source_len", 1)

    def get_target_len(self, data_dict):

        return data_dict.get("target_len", 0)
