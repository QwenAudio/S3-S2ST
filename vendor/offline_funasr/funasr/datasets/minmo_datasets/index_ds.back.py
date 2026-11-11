import os
try:
	#import orjson as json
	import ujson as json
except:
	import json
import torch
import logging

import librosa
import random
import torch.distributed as dist

from funasr.register import tables

import base64
import hashlib
import time
import numpy as np
import uuid
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, Manager
# @tables.register("index_ds_classes", "OpenAIIndexDSJsonl")
# @tables.register("index_ds_classes", "MinMo_S2T")
# class MinMo_S2T(torch.utils.data.Dataset):  # torch.utils.data.Dataset
#
# 	def __init__(self, path: str, **kwargs):
# 		super().__init__()
#
# 		self.max_source_length = kwargs.get("max_source_length", 6000)
# 		self.min_source_length = kwargs.get("min_source_length", 0)
# 		self.max_target_length = kwargs.get("max_target_length", 2048)
# 		# self.min_target_length = kwargs.get("min_target_length", 0)
# 		# self.max_token_length = kwargs.get("max_token_length", 2200)
# 		audio_downsample_rate = int(kwargs.get("audio_downsample_rate", 8))
#
# 		is_training = kwargs.get("is_training", True)
# 		if not (path.endswith(".jsonl") or path.endswith(".json")):
# 			# jsonl list file
# 			data_split_num = kwargs.get("data_split_num", 1)
# 			data_split_i = kwargs.get("data_split_i", 0)
#
# 			if not is_training:
# 				data_split_num = 1
# 				data_split_i = 0
# 			with open(path, encoding="utf-8") as fin:
# 				file_list_all = fin.readlines()
#
# 				num_per_slice = (len(file_list_all) - 1) // data_split_num + 1  # 16
# 				file_list = file_list_all[
# 							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
# 							]
# 				logging.warning(
# 					f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}, \nfile_list_all: {file_list_all}"
# 				)
#
# 		else:
# 			file_list = [path]
#
# 		contents = []
# 		total_whrs = 0.0
# 		total_token_for_llm_B = 0.0
# 		for file_json in file_list:
# 			with open(file_json.strip(), encoding="utf-8") as fin:
# 				for line in fin:
# 					data_dict = json.loads(line.strip())
# 					data = data_dict["messages"]
# 					if isinstance(data_dict.get("speech_length", 0), (list, tuple)):
# 						speech_length = int(data_dict.get("speech_length", [0])[0]) / audio_downsample_rate
# 						text_length = int(data_dict.get("text_length", 0)[0])
# 					else:
# 						speech_length = int(data_dict.get("speech_length", 0)) / audio_downsample_rate
# 						text_length = int(data_dict.get("text_length", 0))
# 					# if "<|startofspeech|>" not in line or "<|endofspeech|>" not in line:
# 					#     logging.error(
# 					#         f"drop it, Speech not in S2T data : {data}, file_json: {file_json}"
# 					#     )
# 					#     continue
# 					if speech_length > 0 and speech_length < 1:
# 						logging.warning(
# 							f"drop it, speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
# 						)
# 						continue
# 					# speech_length = int(speech_length) if speech_length < 1 or text_length < 1:
# 					if text_length < 1:
# 						logging.warning(
# 							f"speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
# 						)
# 						if len(data) > 2:
# 							text_length = len(data[2]['content'])
# 						continue
# 					if speech_length > self.max_source_length:
# 						# logging.warning(
# 						#     f"speech_length: {speech_length} > {self.max_source_length}, drop it: {data_dict}"
# 						# )
# 						continue
# 					if speech_length < self.min_source_length:
# 						# logging.warning(
# 						#     f"speech_length: {speech_length} < {self.min_source_length}, drop it: {data_dict}"
# 						# )
# 						continue
# 					if text_length > self.max_target_length:
# 						# logging.warning(
# 						#     f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
# 						# )
# 						continue
#
# 					system, user, assistant = [], [], []
# 					for i, item in enumerate(data):
# 						role = item["role"]
# 						content = item["content"]
# 						if role == "system":
# 							system.append(content)
# 						elif role == "user":
# 							user.append(content)
# 						elif role == "assistant":
# 							# if "wav_path" in item:
# 							#     wav_path = item["wav_path"]
# 							#     assistant.append([content, {"wav_path": wav_path}])
# 							# else:
# 							#     assistant.append(content)
# 							# assistant.append(content)
# 							if "prev_content" in item:
# 								prev_content = item["prev_content"]
# 								assistant.append([content, {"prev_content": prev_content}])
# 							else:
# 								assistant.append(content)
# 					text_length = len("".join(assistant))
# 					# if text_length > self.max_target_length:
# 					# 	# logging.warning(
# 					# 	#     f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
# 					# 	# )
# 					# 	continue
# 					if len(system) == 0:
# 						system = ["You are a helpful assistant."]
# 					system = system * len(user)
#
# 					contents_i = {
# 						"system": system,
# 						"user": user,
# 						"assistant": assistant,
# 						"source_len": speech_length + text_length,
# 						"speech_length": speech_length,
# 						"text_length": text_length
# 					}
# 					if "key" in data_dict:
# 						contents_i["key"] = data_dict["key"] if not isinstance(data_dict.get("speech_length", 0),
# 																			   (list, tuple)) else data_dict["key"][0]
#
# 					if kwargs.get("save_meta", False):
# 						contents_i["meta"] = data_dict
# 					total_whrs += speech_length / 100.0 / 3600 / 10000 * audio_downsample_rate
# 					total_token_for_llm_B += (text_length + speech_length / 8) / 1000 / 1000 / 1000
# 					contents.append(contents_i)
#
# 		self.contents = contents
#
# 		logging.info(
# 			f"\n\ntotal_num of samplers: {len(self.contents)}, total_whrs: {total_whrs:.5f}, total_token_for_llm_B: {total_token_for_llm_B:.5g}, {path}, {file_list}\n\n")
#
# 	def __len__(self):
# 		return len(self.contents)
#
# 	def __getitem__(self, index):
#
# 		data = self.contents[index]
#
# 		return data
#
# 	def get_source_len(self, data_dict):
# 		source_len = data_dict.get("source_len", -1)
# 		if source_len < 0:
# 			source_len = len(data_dict["system"]) + len(data_dict["user"])
# 		return source_len
#
# 	def get_target_len(self, data_dict):
#
# 		return 0

@tables.register("index_ds_classes", "OpenAIIndexDSJsonl")
@tables.register("index_ds_classes", "MinMo_S2T")
@tables.register("index_ds_classes", "MinMo_S2T_plus")
class MinMo_S2T_plus(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()

		self.max_source_length = kwargs.get("max_source_length", 6000)
		self.min_source_length = kwargs.get("min_source_length", 0)
		self.max_target_length = kwargs.get("max_target_length", 2048)
		# self.min_target_length = kwargs.get("min_target_length", 0)
		# self.max_token_length = kwargs.get("max_token_length", 2200)
		audio_downsample_rate = int(kwargs.get("audio_downsample_rate", 8))

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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
							]
				logging.warning(
					f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}, \nfile_list_all: {file_list_all}"
				)

		else:
			file_list = [path]

		contents = []
		total_whrs = 0.0
		total_token_for_llm_B = 0.0
		for file_json in file_list:
			with open(file_json.strip(), encoding="utf-8") as fin:
				for line in fin:
					data_dict = json.loads(line.strip())
					data = data_dict["messages"]
					if isinstance(data_dict.get("speech_length", 0), (list, tuple)):
						speech_length = int(data_dict.get("speech_length", [0])[0]) / audio_downsample_rate
						text_length = int(data_dict.get("text_length", 0)[0])
					else:
						speech_length = int(data_dict.get("speech_length", 0)) / audio_downsample_rate
						text_length = int(data_dict.get("text_length", 0))
					# if "<|startofspeech|>" not in line or "<|endofspeech|>" not in line:
					#     logging.error(
					#         f"drop it, Speech not in S2T data : {data}, file_json: {file_json}"
					#     )
					#     continue
					if speech_length > 0 and speech_length < 1:
						logging.warning(
							f"drop it, speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
						)
						continue
					# speech_length = int(speech_length) if speech_length < 1 or text_length < 1:
					if text_length < 1:
						logging.warning(
							f"speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
						)
						if len(data) > 2:
							text_length = len(data[2]['content'])
						continue
					if speech_length > self.max_source_length:
						# logging.warning(
						#     f"speech_length: {speech_length} > {self.max_source_length}, drop it: {data_dict}"
						# )
						continue
					if speech_length < self.min_source_length:
						# logging.warning(
						#     f"speech_length: {speech_length} < {self.min_source_length}, drop it: {data_dict}"
						# )
						continue
					if text_length > self.max_target_length:
						# logging.warning(
						#     f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
						# )
						continue

					system, user, assistant = [], [], []
					duration = []
					for i, item in enumerate(data):
						role = item["role"]
						content = item["content"]
						if role == "system":
							system.append(content)
						elif role == "user":
							user.append(content)
							duration.append([item.get("start_time", 0), item.get("end_time", None)])
						elif role == "assistant":
							# if "wav_path" in item:
							#     wav_path = item["wav_path"]
							#     assistant.append([content, {"wav_path": wav_path}])
							# else:
							#     assistant.append(content)
							# assistant.append(content)
							if "prev_content" in item:
								prev_content = item["prev_content"]
								assistant.append([content, {"prev_content": prev_content}])
							else:
								assistant.append(content)
					text_length = len("".join(assistant))
					# if text_length > self.max_target_length:
					# 	# logging.warning(
					# 	#     f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
					# 	# )
					# 	continue
					if len(system) == 0:
						system = ["You are a helpful assistant."]
					system = system * len(user)

					contents_i = {
						"system": system,
						"user": user,
						"assistant": assistant,
						"source_len": speech_length + text_length,
						"speech_length": speech_length,
						"text_length": text_length,
						"duration": duration,
					}
					if "key" in data_dict:
						contents_i["key"] = data_dict["key"] if not isinstance(data_dict.get("speech_length", 0),
																			   (list, tuple)) else data_dict["key"][0]

					if kwargs.get("save_meta", False):
						contents_i["meta"] = data_dict
					total_whrs += speech_length / 100.0 / 3600 / 10000 * audio_downsample_rate
					total_token_for_llm_B += (text_length + speech_length / 8) / 1000 / 1000 / 1000
					contents.append(contents_i)

		self.contents = contents

		logging.info(
			f"\n\ntotal_num of samplers: {len(self.contents)}, total_whrs: {total_whrs:.5f}, total_token_for_llm_B: {total_token_for_llm_B:.5g}, {path}, {file_list}\n\n")

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


@tables.register("index_ds_classes", "MinMo_S2T_PathFix")
class MinMo_S2T_PathFix(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()

		self.max_source_length = kwargs.get("max_source_length", 6000)
		self.min_source_length = kwargs.get("min_source_length", 0)
		self.max_target_length = kwargs.get("max_target_length", 2048)
		# self.min_target_length = kwargs.get("min_target_length", 0)
		# self.max_token_length = kwargs.get("max_token_length", 2200)
		audio_downsample_rate = int(kwargs.get("audio_downsample_rate", 8))

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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
							]
				logging.warning(
					f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}, \nfile_list_all: {file_list_all}"
				)

		else:
			file_list = [path]

		contents = []
		total_whrs = 0.0
		total_token_for_llm_B = 0.0
		for file_json in file_list:
			with open(file_json.strip(), encoding="utf-8") as fin:
				for line in fin:
					data_dict = json.loads(line.strip())
					data = data_dict["messages"]
					if isinstance(data_dict.get("speech_length", 0), (list, tuple)):
						speech_length = int(data_dict.get("speech_length", [0])[0]) / audio_downsample_rate
						text_length = int(data_dict.get("text_length", 0)[0])
					else:
						speech_length = int(data_dict.get("speech_length", 0)) / audio_downsample_rate
						text_length = int(data_dict.get("text_length", 0))
					# if "<|startofspeech|>" not in line or "<|endofspeech|>" not in line:
					#     logging.error(
					#         f"drop it, Speech not in S2T data : {data}, file_json: {file_json}"
					#     )
					#     continue
					if speech_length > 0 and speech_length < 1:
						logging.warning(
							f"drop it, speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
						)
						continue
					# speech_length = int(speech_length) if speech_length < 1 or text_length < 1:
					if text_length < 1:
						logging.warning(
							f"speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
						)
						if len(data) > 2:
							text_length = len(data[2]['content'])
						continue
					if speech_length > self.max_source_length:
						# logging.warning(
						#     f"speech_length: {speech_length} > {self.max_source_length}, drop it: {data_dict}"
						# )
						continue
					if speech_length < self.min_source_length:
						# logging.warning(
						#     f"speech_length: {speech_length} < {self.min_source_length}, drop it: {data_dict}"
						# )
						continue
					if text_length > self.max_target_length:
						# logging.warning(
						#     f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
						# )
						continue

					system, user, assistant = [], [], []
					for i, item in enumerate(data):
						role = item["role"]
						content = item["content"]
						if role == "system":
							system.append(content)
						elif role == "user":
							content = content.replace('.wav', '.flac')
							content = content.replace('.mp3', '.flac')
							content = '/nfs/processed-gzf-train-flac' + content
							user.append(content)
						elif role == "assistant":
							# if "wav_path" in item:
							#     wav_path = item["wav_path"]
							#     assistant.append([content, {"wav_path": wav_path}])
							# else:
							#     assistant.append(content)
							# assistant.append(content)
							if "prev_content" in item:
								prev_content = item["prev_content"]
								assistant.append([content, {"prev_content": prev_content}])
							else:
								assistant.append(content)
					text_length = len("".join(assistant))
					if text_length > self.max_target_length:
						# logging.warning(
						#     f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
						# )
						continue
					if len(system) == 0:
						system = ["You are a helpful assistant."]
					system = system * len(user)

					contents_i = {
						"system": system,
						"user": user,
						"assistant": assistant,
						"source_len": speech_length + text_length,
						"speech_length": speech_length,
						"text_length": text_length
					}
					if "key" in data_dict:
						contents_i["key"] = data_dict["key"] if not isinstance(data_dict.get("speech_length", 0),
																			   (list, tuple)) else data_dict["key"][0]

					if kwargs.get("save_meta", False):
						contents_i["meta"] = data_dict
					total_whrs += speech_length / 100.0 / 3600 / 10000 * audio_downsample_rate
					total_token_for_llm_B += (text_length + speech_length / 8) / 1000 / 1000 / 1000
					contents.append(contents_i)

		self.contents = contents

		logging.info(
			f"\n\ntotal_num of samplers: {len(self.contents)}, total_whrs: {total_whrs:.5f}, total_token_for_llm_B: {total_token_for_llm_B:.5g}, {path}, {file_list}\n\n")

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


@tables.register("index_ds_classes", "MinMoSense")
class MinMoSense(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()

		self.max_source_length = kwargs.get("max_source_length", 8000)
		self.min_source_length = kwargs.get("min_source_length", 10)
		self.max_target_length = kwargs.get("max_target_length", 2048)
		self.min_target_length = kwargs.get("min_target_length", 0)
		# self.max_token_length = kwargs.get("max_token_length", 2200)+
		audio_downsample_rate = int(kwargs.get("audio_downsample_rate", 8))

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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
							]
				logging.info(
					f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}, \nfile_list_all: {file_list_all}"
				)

		else:
			file_list = [path]

		contents = []
		total_whrs = 0.0
		total_token_for_llm_B = 0.0
		for file_json in file_list:
			with open(file_json.strip(), encoding="utf-8") as fin:
				for line in fin:
					try:
						data_dict = json.loads(line.strip())
					except Exception as e:
						logging.error(
							f"drop it, json error: {e}, line: {line}, file_json: {file_json}"
						)
						continue

					data = data_dict["messages"]
					if isinstance(data_dict.get("speech_length", 0), (list, tuple)):
						speech_length = int(data_dict.get("speech_length", [0])[0])
						text_length = int(data_dict.get("text_length", 0)[0])
					else:
						speech_length = int(data_dict.get("speech_length", 0))
						text_length = int(data_dict.get("text_length", 0))
					speech_length = int(speech_length)
					text_length = int(text_length)
					# if "<|startofspeech|>" not in line or "<|endofspeech|>" not in line:
					#     logging.error(
					#         f"drop it, Speech not in S2T data : {data}, file_json: {file_json}"
					#     )
					#     continue
					if speech_length > 0 and speech_length < 1:
						# logging.warning(
						# 	f"drop it, speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
						# )
						continue
					# speech_length = int(speech_length) if speech_length < 1 or text_length < 1:
					if text_length < 1:
						logging.warning(
							f"speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
						)
						if len(data) > 2:
							text_length = len(data[2]['content'])
						continue
					if speech_length > self.max_source_length:
						# logging.warning(
						#     f"speech_length: {speech_length} > {self.max_source_length}, drop it: {data_dict}"
						# )
						continue
					if speech_length < self.min_source_length:
						# logging.warning(
						#     f"speech_length: {speech_length} < {self.min_source_length}, drop it: {data_dict}"
						# )
						continue
					if text_length > self.max_target_length:
						# logging.warning(
						#     f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
						# )
						continue

					system, user, assistant = [], [], []
					duration = []
					for i, item in enumerate(data):
						try:
							role = item["role"]
							content = item["content"]
						except KeyError:
							logging.error(
								f"drop it, KeyError: {item}, file_json: {file_json}"
							)
							continue

						if role == "system":
							system.append(content)
						elif role == "user":
							user.append(content)
							duration.append([item.get("start_time", 0), item.get("end_time", None)])
						elif role == "assistant":
							# if "wav_path" in item:
							#     wav_path = item["wav_path"]
							#     assistant.append([content, {"wav_path": wav_path}])
							# else:
							#     assistant.append(content)
							# assistant.append(content)
							if "prev_content" in item:
								prev_content = item["prev_content"]
								assistant.append([content, {"prev_content": prev_content}])
							else:
								assistant.append(content)
					# text_length = len("".join(assistant))
					# if text_length > self.max_target_length*1.2:
					# 	# logging.warning(
					# 	# 	f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
					# 	# )
					# 	continue
					if len(system) == 0:
						system = ["You are a helpful assistant."]
					system = system * len(user)

					# speech_length = speech_length/audio_downsample_rate
					contents_i = {
						"system": system,
						"user": user,
						"assistant": assistant,
						"source_len": speech_length + text_length,
						"duration": duration,
					}
					# if "key" in data_dict:
					#     contents_i["key"] = data_dict["key"] if not isinstance(data_dict.get("speech_length", 0),
					#                                                            (list, tuple)) else data_dict["key"][0]
					if "key" in data_dict:
						contents_i["key"] = data_dict["key"] if not isinstance(data_dict.get("key", "key_01234"),
																			   (list, tuple)) else data_dict["key"][0]

					if "hist_context" in data_dict:
						contents_i["hist_context"] = data_dict["hist_context"]
					if "hotwords" in data_dict:
						contents_i["hotwords"] = data_dict["hotwords"]
					if "asr_hotwords" in data_dict:
						contents_i["asr_hotwords"] = data_dict["asr_hotwords"]
					if "vad_segs" in data_dict:
						contents_i["vad_segs"] = data_dict["vad_segs"]
					if "word_list" in data_dict:
						contents_i["word_list"] = data_dict["word_list"]
					if "one_pass_result" in data_dict:
						contents_i["one_pass_result"] = data_dict["one_pass_result"]
					if "one_pass_wer" in data_dict:
						contents_i["one_pass_wer"] = data_dict["one_pass_wer"]
					if "noised" in data_dict:
						contents_i["noised"] = data_dict["noised"]

					if kwargs.get("save_meta", False):
						contents_i["meta"] = data_dict

					total_whrs += speech_length / 100.0 / 3600 / 10000 * audio_downsample_rate
					total_token_for_llm_B += (text_length + speech_length / 8) / 1000 / 1000 / 1000
					contents.append(contents_i)

		self.contents = contents

		logging.info(
			f"\n\ntotal_num of samplers: {len(self.contents)}, total_whrs: {total_whrs:.5f}, total_token_for_llm_B: {total_token_for_llm_B:.5g}, {path}, {file_list}\n\n")

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


@tables.register("index_ds_classes", "MinMoSense_PathFix")
class MinMoSense_PathFix(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()

		self.max_source_length = kwargs.get("max_source_length", 8000)
		self.min_source_length = kwargs.get("min_source_length", 10)
		self.max_target_length = kwargs.get("max_target_length", 2048)
		self.min_target_length = kwargs.get("min_target_length", 0)
		# self.max_token_length = kwargs.get("max_token_length", 2200)+
		audio_downsample_rate = int(kwargs.get("audio_downsample_rate", 8))

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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
							]
				logging.info(
					f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}, \nfile_list_all: {file_list_all}"
				)

		else:
			file_list = [path]

		contents = []
		total_whrs = 0.0
		total_token_for_llm_B = 0.0
		for file_json in file_list:
			with open(file_json.strip(), encoding="utf-8") as fin:
				for line in fin:
					try:
						data_dict = json.loads(line.strip())
					except Exception as e:
						logging.error(
							f"drop it, json error: {e}, line: {line}, file_json: {file_json}"
						)
						continue

					data = data_dict["messages"]
					if isinstance(data_dict.get("speech_length", 0), (list, tuple)):
						speech_length = data_dict.get("speech_length", [0])[0]
						text_length = data_dict.get("text_length", 0)[0]
					else:
						speech_length = data_dict.get("speech_length", 0)
						text_length = data_dict.get("text_length", 0)
					speech_length = int(speech_length)
					text_length = int(text_length)
					# if "<|startofspeech|>" not in line or "<|endofspeech|>" not in line:
					#     logging.error(
					#         f"drop it, Speech not in S2T data : {data}, file_json: {file_json}"
					#     )
					#     continue
					if speech_length > 0 and speech_length < 1:
						logging.warning(
							f"drop it, speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
						)
						continue
					# speech_length = int(speech_length) if speech_length < 1 or text_length < 1:
					if text_length < 1:
						logging.warning(
							f"speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
						)
						if len(data) > 2:
							text_length = len(data[2]['content'])
						continue
					if speech_length > self.max_source_length:
						# logging.warning(
						#     f"speech_length: {speech_length} > {self.max_source_length}, drop it: {data_dict}"
						# )
						continue
					if speech_length < self.min_source_length:
						# logging.warning(
						#     f"speech_length: {speech_length} < {self.min_source_length}, drop it: {data_dict}"
						# )
						continue
					if text_length > self.max_target_length:
						# logging.warning(
						#     f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
						# )
						continue

					system, user, assistant = [], [], []
					for i, item in enumerate(data):
						try:
							role = item["role"]
							content = item["content"]
						except KeyError:
							logging.error(
								f"drop it, KeyError: {item}, file_json: {file_json}"
							)
							continue

						if role == "system":
							system.append(content)
						elif role == "user":
							if not "/mnt" in content:
								content = content.replace('.wav', '.flac')
								content = content.replace('.mp3', '.flac')
								content = '/nfs/processed-gzf-train-flac' + content
							user.append(content)
						elif role == "assistant":
							# if "wav_path" in item:
							#     wav_path = item["wav_path"]
							#     assistant.append([content, {"wav_path": wav_path}])
							# else:
							#     assistant.append(content)
							# assistant.append(content)
							if "prev_content" in item:
								prev_content = item["prev_content"]
								assistant.append([content, {"prev_content": prev_content}])
							else:
								assistant.append(content)
					# text_length = len("".join(assistant))
					# if text_length > self.max_target_length:
					# 	logging.info(
					# 		f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
					# 	)
					# 	continue
					if len(system) == 0:
						system = ["You are a helpful assistant."]
					system = system * len(user)

					# speech_length = speech_length/audio_downsample_rate
					contents_i = {
						"system": system,
						"user": user,
						"assistant": assistant,
						"source_len": speech_length + text_length,
					}
					# if "key" in data_dict:
					#     contents_i["key"] = data_dict["key"] if not isinstance(data_dict.get("speech_length", 0),
					#                                                            (list, tuple)) else data_dict["key"][0]
					if "key" in data_dict:
						contents_i["key"] = data_dict["key"] if not isinstance(data_dict.get("key", "key_01234"),
																			   (list, tuple)) else data_dict["key"][0]

					if "hist_context" in data_dict:
						contents_i["hist_context"] = data_dict["hist_context"]
					if "hotwords" in data_dict:
						contents_i["hotwords"] = data_dict["hotwords"]
					if "asr_hotwords" in data_dict:
						contents_i["asr_hotwords"] = data_dict["asr_hotwords"]
					if "vad_segs" in data_dict:
						contents_i["vad_segs"] = data_dict["vad_segs"]
					if "word_list" in data_dict:
						contents_i["word_list"] = data_dict["word_list"]
					if "one_pass_result" in data_dict:
						contents_i["one_pass_result"] = data_dict["one_pass_result"]
					if "one_pass_wer" in data_dict:
						contents_i["one_pass_wer"] = data_dict["one_pass_wer"]

					if kwargs.get("save_meta", False):
						contents_i["meta"] = data_dict

					total_whrs += speech_length / 100.0 / 3600 / 10000 * audio_downsample_rate
					total_token_for_llm_B += (text_length + speech_length / 8) / 1000 / 1000 / 1000
					contents.append(contents_i)

		self.contents = contents

		logging.info(
			f"\n\ntotal_num of samplers: {len(self.contents)}, total_whrs: {total_whrs:.5f}, total_token_for_llm_B: {total_token_for_llm_B:.5g}, {path}, {file_list}\n\n")

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


@tables.register("index_ds_classes", "MinMo_S2S_Interleave")
class MinMo_S2S_Interleave(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()

		self.max_source_length = kwargs.get("max_source_length", 6000)
		self.min_source_length = kwargs.get("min_source_length", 0)
		self.max_target_length = kwargs.get("max_target_length", 2048)

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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
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
					speech_length = data_dict.get("speech_length", 0)
					text_length = data_dict.get("text_length", 0)
					if speech_length > 0 and speech_length < 1:
						logging.error(
							f"drop it, speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
						)
						continue
					if speech_length * 1.3 > self.max_source_length:
						logging.info(
							f"speech_length: {speech_length} > {self.max_source_length}, drop it: {data_dict}"
						)
						continue
					if speech_length < self.min_source_length:
						logging.info(
							f"speech_length: {speech_length} < {self.min_source_length}, drop it: {data_dict}"
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
							if "prev_content" in item:
								prev_content = item["prev_content"]
								assistant.append([content, {"prev_content": prev_content}])
							else:
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
		logging.info(f"total_num of samplers: {len(self.contents)}, {path}, {file_list}")

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

@tables.register("index_ds_classes", "MinMo_S2S_Interleave_Pretrain")
class MinMo_S2S_Interleave_Pretrain(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	AUDIO_PT_MODES_CONFIG = {
		"dual_channel_v2": {
			# "unimodal-text": (7, ['t']), # unimodal-text
			"unimodal-audio": (1, [['td']]), # unimodal-audio
			"mapping-audio-to-text": (1, [['c', 'td']]), # mapping-audio-to-text
			"mapping-text-to-audio": (1, [['t', 'td']]), # mapping-text-to-audio 
			# "interleave-audio-to-semantic": (1, [['c'], ['d']]), # interleave-audio-to-semantic 
			# "interleave-audio-to-text": (1, [['c'], ['t']]), # interleave-audio-to-text
			"interleave-audio-to-semantic-and-text": (4, [['c'], ['td']]), # interleave-audio-to-semantic-and-text
		},
		"dual_channel": {
			# "unimodal-text": (7, ['t']), # unimodal-text
			"unimodal-audio": (1, [['d']]), # unimodal-audio
			"mapping-audio-to-text": (1, [['c', 't']]), # mapping-audio-to-text
			"mapping-text-to-audio": (1, [['t', 'd']]), # mapping-text-to-audio 
			"interleave-audio-to-semantic": (1, [['c'], ['d']]), # interleave-audio-to-semantic 
			"interleave-audio-to-text": (1, [['c'], ['t']]), # interleave-audio-to-text
			"interleave-audio-to-semantic-and-text": (2, [['c'], ['d'], ['c'], ['t']]), # interleave-audio-to-semantic-and-text
		},
		"default": {
			# "unimodal-text": (7, ['t']), # unimodal-text
			"unimodal-audio": (1, [['d']]), # unimodal-audio
			"mapping-audio-to-text": (1, [['c', 't']]), # mapping-audio-to-text
			"mapping-text-to-audio": (1, [['t', 'd']]), # mapping-text-to-audio 
			"interleave-audio-to-semantic": (1, [['c'], ['d']]), # interleave-audio-to-semantic 
			"interleave-audio-to-text": (1, [['c'], ['t']]), # interleave-audio-to-text
			"interleave-audio-to-semantic-and-text": (2, [['c'], ['d']]), # interleave-audio-to-semantic-and-text
		}
	}

	NEW_SLICE = {
		"messages": [
			{
				"role": "system",
				"content": "You are a helpful assistant."
			},
			{
				"role": "user",
				"content": "",
				"start_time": None,
				"end_time": None,
				"speech_tokens": []
			},
			{
				"role": "assistant",
				"content": "",
				"text_tokens": None
			}
		],
	}

	@staticmethod
	def get_audio_slice_mode(config_name, pt_mode, current_slice_mode_idx):
		# print(f'pt_mode: {pt_mode}, current_slice_mode_idx: {current_slice_mode_idx}')
		slice_modes = MinMo_S2S_Interleave_Pretrain.AUDIO_PT_MODES_CONFIG[config_name][pt_mode][1]
		# print(f'slice_modes: {slice_modes}')
		return slice_modes[current_slice_mode_idx % len(slice_modes)]

	@staticmethod
	def merge_slices(audio_pt_modes_config_name, slices, slice_modes, pt_mode, current_slice_mode_idx, max_duration_sec=30):
		'''
		max_duration_sec: float for a fixed value, or tuple of float meaning min and max values for random sampling
		'''
		if not isinstance(max_duration_sec, float):
			max_duration_sec = random.uniform(max_duration_sec[0], max_duration_sec[1])
		merged_slices = []
		merged_slice_modes = []

		current_slice = None

		for slice in slices:
			# print(f'slice: {slice}')
			messages = slice['messages']
			tokens = messages[1]['speech_tokens']
			text_tokens = messages[2]['text_tokens']
			content = messages[2]['content']
			start_time = messages[1]['start_time'] # + float(segment['start_time'])
			end_time = messages[1]['end_time'] # + float(segment['start_time'])

			if current_slice is not None:
				current_slice_duration = current_slice['messages'][1]['end_time'] - current_slice['messages'][1]['start_time']
				if start_time - current_slice['messages'][1]['end_time'] < 0.1 and current_slice_duration + (end_time - start_time) < max_duration_sec:
					# merge with current slice
					current_slice['messages'][1]['end_time'] = end_time
					current_slice['messages'][2]['content'] += content
					# current_slice['messages'][2]['text_tokens'] += text_tokens
				else:
					# new slice
					merged_slices.append(current_slice)
					merged_slice_modes.append(MinMo_S2S_Interleave_Pretrain.get_audio_slice_mode(audio_pt_modes_config_name, pt_mode, current_slice_mode_idx))
					current_slice_mode_idx += 1
					current_slice = None
			if current_slice is None:
				current_slice = deepcopy(MinMo_S2S_Interleave_Pretrain.NEW_SLICE)
				current_slice['messages'][1]['start_time'] = start_time
				current_slice['messages'][1]['end_time'] = end_time
				current_slice['messages'][2]['content'] += content
				# current_slice['messages'][2]['text_tokens'] += text_tokens
			# print('speech_tokens: ', tokens, 'text_tokens: ', text_tokens, 'content: ', content)
		if current_slice is not None:
			merged_slices.append(current_slice)
			merged_slice_modes.append(MinMo_S2S_Interleave_Pretrain.get_audio_slice_mode(audio_pt_modes_config_name, pt_mode, current_slice_mode_idx))
			current_slice_mode_idx += 1
		return merged_slices, merged_slice_modes, current_slice_mode_idx

	@staticmethod
	def load_line(line: str,
				  min_slice_duration_sec,
				  min_slice_duration_drop_audio,
				  text_data_random_drop_ratio):
		try:
			data_dict_ = json.loads(line.strip())
		except:
			return None
		# transform Matterhorn format to chatml format
		if 'raw_audio_path' in data_dict_ and 'messages' not in data_dict_:
			try:
				data_dict = {}
				segments_chatml = []
				messages_chatml = []
				for segment in data_dict_["segments"]:
					if len(segment['slices']) == 0:
						return None
					if 'wav_path' in segment:
						wav_path = segment['wav_path']
					else:
						wav_path = segment['slices'][0]['messages'][1]['content'].replace('<|startofspeech|>!', '').replace('<|endofspeech|>', '')
					segment_chatml = {
						# 'lid': segment['postprocess']["lid"],
						# 'text': segment['postprocess']["text"],
						'start_time': float(segment['start_time']),
						'end_time': float(segment['end_time']),
						'segment_id': len(segments_chatml),
						'wav_path': wav_path
					}
					for slice in segment['slices']:
						messages = slice['messages']
						start_time = messages[1]['start_time'] + float(segment['start_time'])
						end_time = messages[1]['end_time'] + float(segment['start_time'])
						content = messages[2]['content']
						tokens = messages[2]['slice_tokens']
						segment_id = len(segments_chatml)

						role = 'user' if len(messages_chatml) % 2 == 0 else 'assistant'
						messages_chatml.append({
							'role': role,
							'content': content,
							'start_time': start_time,
							'end_time': end_time,
							'segment_id': segment_id,
							'tokens': tokens
						})
					segments_chatml.append(segment_chatml)
				data_dict = {
					'messages': messages_chatml,
					'source': data_dict_['raw_audio_path'],
					'segments': segments_chatml,
				}
			except Exception as e:
				logging.error(f"Error loading jsonl file: {e}; data_dict_: {line}")
				return None
		else:
			data_dict = data_dict_

		data = data_dict["messages"]
		wav_path = data_dict.get("source", None)
		segments = data_dict.get("segments", [])
		segments_dict = {}
		for segment in segments:
			segments_dict[segment["segment_id"]] = segment

		system, user, assistant = [], [], []
		referenced_segment_ids = set()
		should_drop = False
		is_audio = False
		source_len = 0
		for i, item in enumerate(data):
			role = item["role"]
			content = item["content"]
			if role != "system":
				start_time = item.get("start_time", None)
				end_time = item.get("end_time", None)
				segment_id = item.get("segment_id", None)
				if segment_id is not None:
					referenced_segment_ids.add(segment_id)
				# check for too short slice
				if start_time is not None and end_time is not None:
					is_audio = True

					segment = segments_dict[segment_id]
					segment_start_time = segment["start_time"]
					segment_end_time = segment["end_time"]
					if segment_start_time is None or segment_end_time is None:
						continue
					if start_time < segment_start_time or end_time > segment_end_time:
						# logging.info(f"! slice out of segment, slice: {item}, segment: {segment}")
						should_drop = True

					duration = end_time - start_time
					if duration < min_slice_duration_sec:
						# logging.info(f"! too short slice, slice duration: {duration} < {min_slice_duration_sec}, item: {item}")
						if min_slice_duration_drop_audio:
							should_drop = True
							break
						else:
							continue

			if role == "system":
				system.append({"content": content, "start_time": None, "end_time": None})
			elif role == "user":
				user.append({"content": content, "start_time": start_time, "end_time": end_time, "segment_id": segment_id, "tokens": item['tokens'] if 'tokens' in item else None})
				if is_audio:
					source_len += int((end_time - start_time) * 25)
				else:
					source_len += int(len(content) / 2)
			elif role == "assistant":
				if "prev_content" in item:
					prev_content = item["prev_content"]
					assistant.append([content, {"prev_content": prev_content}])
					raise ValueError("prev_content is not supported")
				else:
					assistant.append({"content": content, "start_time": start_time, "end_time": end_time, "segment_id": segment_id, "tokens": item['tokens'] if 'tokens' in item else None})
				# assume assistant is always text
				source_len += int(len(content) / 2)

		if is_audio:
			# num_audio += 1
			# if should_drop_audio:
			# 	# logging.info(f"!! drop audio, should_drop_audio: {should_drop_audio}")
			# 	num_audio_drop += 1
				
			# else:
			# 	num_audio_keep += 1
			pass
		else:
			# num_text += 1
			if text_data_random_drop_ratio and random.random() < text_data_random_drop_ratio:
				# num_text_drop += 1
				should_drop = True
			else:
				# num_text_keep += 1
				should_drop = False

		# user， assistant并没有区分含义，
		assert len(user) - len(assistant) <= 1, f"user turn does not match assistant turn"
		if len(user) < len(assistant):
			user.append(None)
		elif len(user) > len(assistant):
			assistant.append(None)

		if len(system) == 0:
			system = [{"content": "You are a helpful assistant.", "start_time": None, "end_time": None}]
		system = system * len(user)

		referenced_segments = []
		for segment_id in sorted(referenced_segment_ids):
			referenced_segments.append(segments_dict[segment_id])

		contents_i = {
			"system": system,
			"user": user,
			"assistant": assistant,
			"source_wav": wav_path,
			"segments": referenced_segments,
			"segments_dict": segments_dict,
			"source_len": source_len,
		}
		if "key" in data_dict:
			contents_i["key"] = data_dict["key"]
		return is_audio, should_drop, contents_i

	@staticmethod
	def load_lines(lines: list,
					index_entry: dict=None,
					tokenizer=None,
					if_merge_slices=False,
					max_merged_slice_duration_sec=30.0,
					audio_pt_modes_config_name="default"):
		try:
			segments = []
			slice_modes_list = []
			pt_modes = set()
			for line, segment_index, pt_mode in lines:
				slice_modes = segment_index['extra'].get('slice_modes', None)
				pt_modes.add(pt_mode)
				segment = json.loads(line.strip())

				# print(f'index_entry: {segment_index}')
				if 'merged_slices' in segment_index['extra']:
					# print('loading merged slices from index')
					merged_slice_infos = segment_index['extra']['merged_slices']
					merged_slices = []
					merged_slice_modes = []
					for slice_info in merged_slice_infos:
						slice = deepcopy(MinMo_S2S_Interleave_Pretrain.NEW_SLICE)
						slice['messages'][1]['start_time'] = slice_info['start_time']
						slice['messages'][1]['end_time'] = slice_info['end_time']
						slice['messages'][2]['content'] = slice_info['text']
						merged_slices.append(slice)
						merged_slice_modes.append(slice_info['pt_slice_mode'])
					slice_modes = merged_slice_modes
					segment['slices'] = merged_slices
				slice_modes_list.append(slice_modes)
				segments.append(segment)
		except Exception as e:
			import traceback
			# logging.error(f"Error load_lines: {e}")
			# logging.error(f"lines: {lines}")
			# logging.error(f"Traceback: {traceback.format_exc()}")
			return None

		if len(pt_modes) > 1:
			raise ValueError(f"Multiple pt_modes are not supported: {pt_modes}")
		pt_mode = list(pt_modes)[0]

		index_segments = None
		if index_entry is not None and 'items' in index_entry:
			index_segments = index_entry['items']
			assert len(index_segments) == len(segments), f"index_segments and segments length mismatch"

		data_dict = {}
		segments_chatml = []
		messages_chatml = []
		# print(f'pt_mode: {pt_mode}, num_segments: {len(segments)}')
		current_raw_audio_path = None
		current_slice_mode_idx = 0
		for segment_idx, (segment, slice_modes) in enumerate(zip(segments, slice_modes_list)):

			raw_audio_path = segment.get('raw_audio_path', 'empty')
			# print(f'segment_idx: {segment_idx}, raw_audio_path: {raw_audio_path}')
			if raw_audio_path != current_raw_audio_path and raw_audio_path != 'empty':
				# print(f'new raw_audio_path, setting current_slice_mode_idx: {current_slice_mode_idx}')
				current_raw_audio_path = raw_audio_path
				current_slice_mode_idx = 0

			# audio data 
			if pt_mode != 'unimodal-text':
				slices = segment['slices']
				assert len(slices) == len(slice_modes), f"slices and slice_modes length mismatch"

				if if_merge_slices:
					# num_slices_before_merge = len(slices)
					# print(f'num_slices_before_merge: {num_slices_before_merge}')
					slices, slice_modes, current_slice_mode_idx = MinMo_S2S_Interleave_Pretrain.merge_slices(audio_pt_modes_config_name, slices, slice_modes, pt_mode, current_slice_mode_idx, max_duration_sec=max_merged_slice_duration_sec)
					# print(f'num_slices before and after merge: {num_slices_before_merge} => {len(slices)}')

				for slice, slice_mode in zip(slices, slice_modes):
					messages = slice['messages']
					tokens = messages[1]['speech_tokens']
					text_tokens = messages[2]['text_tokens']
					content = messages[2]['content']

					start_time = messages[1]['start_time'] + float(segment['start_time'])
					end_time = messages[1]['end_time'] + float(segment['start_time'])

					# print(f'  slice_mode: {slice_mode}, start_time: {start_time}, end_time: {end_time}, text_tokens: {text_tokens}, speech_tokens: {tokens}, content: {content}')

					segment_id = len(segments_chatml)
					role = 'user' if len(messages_chatml) % 2 == 0 else 'assistant'

					messages_chatml.append({
						'role': role,
						'content': content,
						'start_time': start_time,
						'end_time': end_time,
						'segment_id': segment_id,
						'tokens': tokens,
						'text_tokens': text_tokens,
						'slice_mode': ''.join(slice_mode)
					})
			else:
				# text data
				messages_chatml.append({
					'role': 'user',
					'content': segment['messages'][2]['content'],
					'start_time': None,
					'end_time': None,
					'segment_id': len(segments_chatml),
					'tokens': None,
					'text_tokens': None,
					'slice_mode': None
				})
			
			# segments
			if pt_mode != 'unimodal-text':
				if 'audio_data'in segment:
					# base64 encoded audio data, save to tmp file
					audio_base64 = segment['audio_data']
					audio_bytes = base64.b64decode(audio_base64)
					
					segment_id = len(segments_chatml)
					# raw_audio_path = segment.get('raw_audio_path', 'empty')
					start_time = float(segment['start_time'])
					end_time = float(segment['end_time'])
					current_time = int(time.time() * 1000000)
					unique_id_with_time = hashlib.sha256(f"{raw_audio_path}_{start_time}_{end_time}_{current_time}_{uuid.uuid4()}".encode('utf-8')).hexdigest()
					unique_id = hashlib.sha256(f"{raw_audio_path}_{start_time}_{end_time}".encode('utf-8')).hexdigest()

					wav_path = os.path.join('/tmp/tmp_flacs', f"{unique_id_with_time}.flac")
					os.makedirs(os.path.dirname(wav_path), exist_ok=True)
					with open(wav_path, 'wb') as f:
						f.write(audio_bytes)
				elif 'segment_path' in segment:
					# on local or oss, pass through
					wav_path = segment['segment_path']
					unique_id = None

				# override postprocess fields with index_segment['extra']
				post_keys = ('speechscore', 'avg_wer')
				index_segment = None
				if index_entry is not None:
					index_segment = index_segments[segment_idx]
				if 'postprocess' not in segment:
					segment['postprocess'] = {}
				for key in post_keys:
					value = None
					if key in segment['postprocess']:
						value = segment['postprocess'][key]
						if isinstance(value, list):
							try:
								value = np.mean(value)
								value = round(value, 4)
							except Exception as e:
								value = None
						# print(f"segment['postprocess'][{key}]: {segment['postprocess'][key]}")
					if index_entry is not None and key in index_segment['extra']:
						value = index_segment['extra'][key]
						# print(f"index_segment['extra'][{key}]: {index_segment['extra'][key]}")
					if value is not None:
						# print(f"key: {key}, value: {value}")
						segment['postprocess'][key] = value

				segment_chatml = {
					'start_time': float(segment['start_time']),
					'end_time': float(segment['end_time']),
					'segment_id': len(segments_chatml),
					'wav_path': wav_path,
					'postprocess': segment['postprocess'],
					'raw_audio_path': raw_audio_path,
					'unique_id': unique_id
				}
			else:
				segment_chatml = {
					'start_time': None,
					'end_time': None,
					'segment_id': len(segments_chatml),
					'wav_path': None,
					'postprocess': {},
					'raw_audio_path': None,
					'unique_id': None
				}

			segments_chatml.append(segment_chatml)

		data_dict = {
			'messages': messages_chatml,
			'segments': segments_chatml,
		}

		messages = data_dict["messages"]
		segments = data_dict.get("segments", [])

		user, assistant = [], []
		for i, item in enumerate(messages):
			role = item["role"]
			if role == "user":
				user.append({"content": item['content'], "start_time": item['start_time'], "end_time": item['end_time'], 
							"segment_id": item['segment_id'], "tokens": item['tokens'], "text_tokens": item['text_tokens'], 
							"slice_mode": item['slice_mode']})
			elif role == "assistant":
				assistant.append({"content": item['content'], "start_time": item['start_time'], "end_time": item['end_time'], 
							"segment_id": item['segment_id'], "tokens": item['tokens'], "text_tokens": item['text_tokens'], 
							"slice_mode": item['slice_mode']})

		assert len(user) - len(assistant) <= 1, f"user turn does not match assistant turn"
		if len(user) < len(assistant):
			user.append(None)
		elif len(user) > len(assistant):
			assistant.append(None)

		contents_i = {
			"user": user,
			"assistant": assistant,
			"segments_dict": {segment["segment_id"]: segment for segment in segments},
			"pt_mode": pt_mode,
		}

		return contents_i

	@staticmethod
	def load_jsonl_file(file_json,
						min_slice_duration_sec,
						min_slice_duration_drop_audio,
						text_data_random_drop_ratio,
						seed=2333):
		random.seed(seed)
		import tqdm
		contents = []
		num_total = 0
		num_audio = 0
		num_audio_drop = 0
		num_audio_keep = 0
		num_text = 0
		num_text_drop = 0
		num_text_keep = 0
		num_ok = 0
		with open(file_json.strip(), encoding="utf-8") as fin:
			for line in fin:
			# for line in tqdm.tqdm(fin, position=1, leave=False):
				num_total += 1
				line_result = MinMo_S2S_Interleave_Pretrain.load_line(line, min_slice_duration_sec, min_slice_duration_drop_audio, text_data_random_drop_ratio)
				if line_result is None:
					continue
				is_audio, should_drop, contents_i = line_result
				if is_audio:
					num_audio += 1
					if should_drop:
						num_audio_drop += 1
					else:
						num_audio_keep += 1
				else:
					num_text += 1
					if should_drop:
						num_text_drop += 1
					else:
						num_text_keep += 1
				contents.append(contents_i)
				num_ok += 1
		return contents, num_total, num_audio, num_audio_drop, num_audio_keep, num_text, num_text_drop, num_text_keep, num_ok


	def __init__(self,
				 path: str,
				 tokenizer=None,
				 **kwargs):
		super().__init__()

		self.max_source_length = kwargs.get("max_source_length", 6000)
		self.min_source_length = kwargs.get("min_source_length", 0)
		self.max_target_length = kwargs.get("max_target_length", 2048)

		self.tokenizer = tokenizer
		self.if_merge_slices = kwargs.get("merge_slices", False)
		self.max_merged_slice_duration_sec = kwargs.get("max_merged_slice_duration_sec", 30.0)
		self.audio_pt_modes_config_name = kwargs.get("audio_pt_modes_config_name", "dual_channel")
		logging.info(f"MinMo_S2S_Interleave_Pretrain index_ds audio_pt_modes_config_name: {self.audio_pt_modes_config_name}")
		# self.min_segment_duration_sec = kwargs.get("min_segment_duration", 0.1)
		# self.min_segment_duration_drop_audio = kwargs.get("min_segment_duration_drop_audio", True)
		self.min_slice_duration_sec = kwargs.get("min_slice_duration", 0.1)
		logging.info(f"MinMo_S2S_Interleave_Pretrain index_ds min_slice_duration_sec: {self.min_slice_duration_sec}")
		self.min_slice_duration_drop_audio = kwargs.get("min_slice_duration_drop_audio", True)
		logging.info(f"MinMo_S2S_Interleave_Pretrain index_ds min_slice_duration_drop_audio: {self.min_slice_duration_drop_audio}")
		self.text_data_random_drop_ratio = kwargs.get("text_data_random_drop_ratio", 0.0)
		logging.info(f"MinMo_S2S_Interleave_Pretrain index_ds text_data_random_drop_ratio: {self.text_data_random_drop_ratio}")

		is_training = kwargs.get("is_training", True)
		self.single_process_read = kwargs.get("single_process_read", True)
		if not (path.endswith(".jsonl") or path.endswith(".json")):
			# jsonl list file
			data_split_num = kwargs.get("data_split_num", 1)
			# TODO: data_split_i怎么传入参数设置的？
			data_split_i = kwargs.get("data_split_i", 0)
			# when data_split_size, which determins the size of each data split, is not None, data_split_num will be ignored
			data_split_size = kwargs.get("data_split_size", None)

			if not is_training:
				data_split_num = 1
				data_split_i = 0
			with open(path, encoding="utf-8") as fin:
				file_list_all = fin.readlines()

				if data_split_size is not None:
					num_per_slice = data_split_size
				else:
					num_per_slice = (len(file_list_all) - 1) // data_split_num + 1  # 16
				file_list = file_list_all[
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
							]
				if data_split_size is not None:
					logging.info(f"is_training: {is_training}, data_split_size: {data_split_size}, data_split_i: {data_split_i}, \nfile_list[:10]: {file_list[:10]}, \nfile_list_all[:10]: {file_list_all[:10]}")
				else:
					logging.info(f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list[:10]: {file_list[:10]}, \nfile_list_all[:10]: {file_list_all[:10]}")

		else:
			file_list = [path]

		self.use_file_index = kwargs.get("use_file_index", False)
		logging.info(f"MinMo_S2S_Interleave_Pretrain index_ds use_file_index: {self.use_file_index}")
		if self.use_file_index:
			from funasr.datasets.minmo_datasets.cache_utils.file_index import FileIndex
			# max_file_handles_to_keep=4, max_file_handle_idle_time=600
			self.file_index_max_file_handles_to_keep = kwargs.get("file_index_max_file_handles_to_keep", 4)
			logging.info(f"MinMo_S2S_Interleave_Pretrain index_ds file_index_max_file_handles_to_keep: {self.file_index_max_file_handles_to_keep}")
			self.file_index_max_file_handle_idle_time = kwargs.get("file_index_max_file_handle_idle_time", 600)
			logging.info(f"MinMo_S2S_Interleave_Pretrain index_ds file_index_max_file_handle_idle_time: {self.file_index_max_file_handle_idle_time}")
			self.file_index = FileIndex(max_file_handles_to_keep=self.file_index_max_file_handles_to_keep, max_file_handle_idle_time=self.file_index_max_file_handle_idle_time)
			logging.info(f"MinMo_S2S_Interleave_Pretrain index_ds adding file_list len: {len(file_list)}, file_list[:10]: {file_list[:10]}")
			file_list = [file.strip() for file in file_list]
			self.file_index.add_index_files(file_list)
			logging.info(f"MinMo_S2S_Interleave_Pretrain index_ds file_index len: {len(self.file_index)}")
		else:
			contents = []
			num_total = 0
			num_audio = 0
			num_audio_drop = 0
			num_audio_keep = 0
			num_text = 0
			num_text_drop = 0
			num_text_keep = 0
			num_ok = 0
			import tqdm
			if not self.single_process_read:
				from multiprocessing.pool import Pool
				from functools import partial
				load_jsonl_file = partial(MinMo_S2S_Interleave_Pretrain.load_jsonl_file,
											min_slice_duration_sec=self.min_slice_duration_sec,
											min_slice_duration_drop_audio=self.min_slice_duration_drop_audio,
											text_data_random_drop_ratio=self.text_data_random_drop_ratio)
				with Pool(processes=20) as pool:
					for file_contents, file_num_total, file_num_audio, file_num_audio_drop, \
						file_num_audio_keep, file_num_text, file_num_text_drop, file_num_text_keep, file_num_ok in tqdm.tqdm(pool.imap(load_jsonl_file, file_list), total=len(file_list), position=0, leave=True):
							contents.extend(file_contents)
							num_total += file_num_total
							num_audio += file_num_audio
							num_audio_drop += file_num_audio_drop
							num_audio_keep += file_num_audio_keep
							num_text += file_num_text
							num_text_drop += file_num_text_drop
							num_text_keep += file_num_text_keep
							num_ok += file_num_ok
			else:
				for file_json in tqdm.tqdm(file_list, position=0, leave=True):
					file_contents, file_num_total, file_num_audio, file_num_audio_drop, \
						file_num_audio_keep, file_num_text, file_num_text_drop, file_num_text_keep, file_num_ok = MinMo_S2S_Interleave_Pretrain.load_jsonl_file(
							file_json,
							min_slice_duration_sec=self.min_slice_duration_sec,
							min_slice_duration_drop_audio=self.min_slice_duration_drop_audio,
							text_data_random_drop_ratio=self.text_data_random_drop_ratio
						)
					contents.extend(file_contents)
					num_total += file_num_total
					num_audio += file_num_audio
					num_audio_drop += file_num_audio_drop
					num_audio_keep += file_num_audio_keep
					num_text += file_num_text
					num_text_drop += file_num_text_drop
					num_text_keep += file_num_text_keep
					num_ok += file_num_ok

			self.contents = contents
			logging.info(f"total_num of samplers: {len(self.contents)}, {path}, {file_list}, num_total: {num_total}, num_audio: {num_audio}, num_audio_drop: {num_audio_drop}, num_audio_keep: {num_audio_keep}, num_text: {num_text}, num_text_drop: {num_text_drop}, num_text_keep: {num_text_keep}, num_ok: {num_ok}")

	def __len__(self):
		if self.use_file_index:
			return len(self.file_index)
		else:
			return len(self.contents)

	def __getitem__(self, index):

		if self.use_file_index:
			lines = self.file_index[index]
			index_entry = self.file_index.get_index_entry(index)
			contents_i = self.load_lines(lines,
											index_entry=index_entry,
											tokenizer=self.tokenizer,
											if_merge_slices=self.if_merge_slices,
											max_merged_slice_duration_sec=self.max_merged_slice_duration_sec,
											audio_pt_modes_config_name=self.audio_pt_modes_config_name)
			data = contents_i
		else:
			data = self.contents[index]

		return data

	def get_source_len(self, data_dict, index=None):
		if self.use_file_index:
			index_entry = self.file_index.get_index_entry(index)
			source_len = index_entry.get("extra", {}).get("source_len", -1)
		else:
			data_dict = self[index]
			source_len = data_dict.get("source_len", -1)
			if source_len < 0:
				source_len = len(data_dict["system"]) + len(data_dict["user"])
		return source_len

	def get_target_len(self, index):

		return 0

@tables.register("index_ds_classes", "MinMo_S2S_Channels_SFT")
class MinMo_S2S_Channels_SFT(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()

		self.max_source_length = kwargs.get("max_source_length", 8000)
		self.min_source_length = kwargs.get("min_source_length", 0)
		self.max_target_length = kwargs.get("max_target_length", 8000)
		self.min_target_length = kwargs.get("min_target_length", 0)
		# self.max_token_length = kwargs.get("max_token_length", 2200)+
		# audio_downsample_rate = int(kwargs.get("audio_downsample_rate", 8))

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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
							]
				logging.info(
					f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}, \nfile_list_all: {file_list_all}"
				)

		else:
			file_list = [path]

		contents = []
		total_whrs = 0.0
		total_token_for_llm_B = 0.0
		for file_json in file_list:
			with open(file_json.strip(), encoding="utf-8") as fin:
				for line in fin:
					try:
						data_dict = json.loads(line.strip())
					except Exception as e:
						logging.error(
							f"drop it, json error: {e}, line: {line}, file_json: {file_json}"
						)
						continue

					data = data_dict["messages"]
					if isinstance(data_dict.get("speech_length", 0), (list, tuple)):
						speech_length = int(data_dict.get("speech_length", [0])[0])
						text_length = int(data_dict.get("text_length", 0)[0])
					else:
						speech_length = int(data_dict.get("speech_length", 0))
						text_length = int(data_dict.get("text_length", 0))
					speech_length = int(speech_length)
					text_length = int(text_length)

					if speech_length > self.max_source_length:
						# logging.warning(
						#     f"speech_length: {speech_length} > {self.max_source_length}, drop it"
						# )
						continue
					if speech_length < self.min_source_length:
						# logging.warning(
						#     f"speech_length: {speech_length} < {self.min_source_length}, drop it"
						# )
						continue
					if text_length > self.max_target_length:
						# logging.warning(
						#     f"text_length: {text_length} > {self.max_target_length}, drop it"
						# )
						continue

					system, user, assistant = [], [], []
					duration = []
					for i, item in enumerate(data):
						try:
							role = item["role"]
							text = item["text"]
							audio = item["audio"]
						except KeyError:
							logging.error(
								f"drop it, KeyError: {item}, file_json: {file_json}"
							)
							continue

						if role == "system":
							system.append(item)
						elif role == "user":
							user.append(item)
							duration.append([item.get("start_time", 0), item.get("end_time", None)])
						elif role == "assistant":
							assistant.append(item)
					system = system * len(user)

					contents_i = {
						"system": system,
						"user": user,
						"assistant": assistant,
						"source_len": speech_length + text_length,
						"duration": duration,
					}
					
					contents.append(contents_i)

		self.contents = contents

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

@tables.register("index_ds_classes", "MinMo_S2S_Channels_SFT_Pack")
class MinMo_S2S_Channels_SFT_Pack(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()

		self.max_source_length = kwargs.get("max_source_length", 8000)
		self.min_source_length = kwargs.get("min_source_length", 0)
		self.max_target_length = kwargs.get("max_target_length", 8000)
		self.min_target_length = kwargs.get("min_target_length", 0)
		# self.max_token_length = kwargs.get("max_token_length", 2200)+
		# audio_downsample_rate = int(kwargs.get("audio_downsample_rate", 8))

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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
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
					try:
						data_dict_list = json.loads(line.strip())
					except Exception as e:
						logging.error(
							f"drop it, json error: {e}, line: {line}, file_json: {file_json}"
						)
						continue
					contents_pack = []
					for data_dict in data_dict_list:
						data = data_dict["messages"]
						if isinstance(data_dict.get("speech_length", 0), (list, tuple)):
							speech_length = int(data_dict.get("speech_length", [0])[0])
							text_length = int(data_dict.get("text_length", 0)[0])
						else:
							speech_length = int(data_dict.get("speech_length", 0))
							text_length = int(data_dict.get("text_length", 0))
						speech_length = int(speech_length)
						text_length = int(text_length)


						system, user, assistant = [], [], []
						duration = []
						for i, item in enumerate(data):
							try:
								role = item["role"]
								text = item["text"]
								audio = item["audio"]
							except KeyError:
								logging.error(
									f"drop it, KeyError: {item}, file_json: {file_json}"
								)
								continue

							if role == "system":
								system.append(item)
							elif role == "user":
								user.append(item)
								duration.append([item.get("start_time", 0), item.get("end_time", None)])
							elif role == "assistant":
								assistant.append(item)
						system = system * len(user)

						contents_i = {
							"system": system,
							"user": user,
							"assistant": assistant,
							"source_len": speech_length + text_length,
							"duration": duration,
						}
						contents_pack.append(contents_i)
					
					contents.append(contents_pack)

		self.contents = contents

	def __len__(self):
		return len(self.contents)

	def __getitem__(self, index):

		data = self.contents[index]

		return data

	def get_source_len(self, data_dict):
		# source_len = data_dict.get("source_len", -1)
		# if source_len < 0:
		# 	source_len = len(data_dict["system"]) + len(data_dict["user"])
		# return source_len

		# for packed data, set source_len to 1 to ignore the check
		return 1

	def get_target_len(self, data_dict):

		return 0


@tables.register("index_ds_classes", "MinMo_S2S_Channels_SFT_Pack_MultiProcess")
class MinMo_S2S_Channels_SFT_Pack_MultiProcess(torch.utils.data.Dataset):

	def __init__(self, path: str, **kwargs):
		super().__init__()
		self.num_loading_process = kwargs.get("num_loading_process", 8)
		self.max_source_length = kwargs.get("max_source_length", 8000)
		self.min_source_length = kwargs.get("min_source_length", 0)
		self.max_target_length = kwargs.get("max_target_length", 8000)
		self.min_target_length = kwargs.get("min_target_length", 0)
		# self.max_token_length = kwargs.get("max_token_length", 2200)+
		# audio_downsample_rate = int(kwargs.get("audio_downsample_rate", 8))

		is_training = kwargs.get("is_training", True)
		if not (path.endswith(".jsonl") or path.endswith(".json")):
			# jsonl list file
			data_split_num = kwargs.get("data_split_num", 1)
			data_split_num = 50
			data_split_i = kwargs.get("data_split_i", 0)

			if not is_training:
				data_split_num = 1
				data_split_i = 0
			with open(path, encoding="utf-8") as fin:
				file_list_all = fin.readlines()

				num_per_slice = (len(file_list_all) - 1) // data_split_num + 1  # 16
				file_list = file_list_all[
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
							]
				logging.info(
					f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}"
				)

		else:
			file_list = [path]

		st = time.time()
		contents_all = []
		with ProcessPoolExecutor(max_workers=self.num_loading_process) as executor:
			for contents in executor.map(self.load_jsonl_file, file_list):
				contents_all.extend(contents) # receive dumped json in case pickle time spend (load in worker)
		print(f"load {len(file_list)} data split, len contents {len(contents_all)}, timepass {time.time()-st:.2f}s")

		self.contents = contents_all

	def load_jsonl_file(self, path):
		contents = []
		with open(path.strip(), encoding="utf-8") as fin:
			for line in fin:
				contents_pack = self.load_single_line(line)
				contents.append(contents_pack)
		return contents

	def load_single_line(self, line):
		try:
			data_dict_list = json.loads(line.strip())
			contents_pack = []
			for data_dict in data_dict_list:
				data = data_dict["messages"]
				if isinstance(data_dict.get("speech_length", 0), (list, tuple)):
					speech_length = int(data_dict.get("speech_length", [0])[0])
					text_length = int(data_dict.get("text_length", 0)[0])
				else:
					speech_length = int(data_dict.get("speech_length", 0))
					text_length = int(data_dict.get("text_length", 0))
				speech_length = int(speech_length)
				text_length = int(text_length)


				system, user, assistant = [], [], []
				duration = []
				for i, item in enumerate(data):
					try:
						role = item["role"]
						text = item["text"]
						audio = item["audio"]
					except KeyError:
						logging.error(
							f"drop it, KeyError: {item}, file_json: {file_json}"
						)
						return []

					if role == "system":
						system.append(item)
					elif role == "user":
						user.append(item)
						duration.append([item.get("start_time", 0), item.get("end_time", None)])
					elif role == "assistant":
						assistant.append(item)
				system = system * len(user)

				contents_i = {
					"system": system,
					"user": user,
					"assistant": assistant,
					"source_len": speech_length + text_length,
					"duration": duration,
				}
				contents_pack.append(contents_i)
			return json.dumps(contents_pack, ensure_ascii=False)

		except Exception as e:
			logging.error(
				f"drop it, json error: {e}, line: {line}, file_json: {file_json}"
			)
			return []

	def __len__(self):
		return len(self.contents)

	def __getitem__(self, index):
		data = json.loads(self.contents[index])
		return data

	def get_source_len(self, data_dict):
		# for packed data, set source_len to 1 to ignore the check
		return 1

	def get_target_len(self, data_dict):
		return 0


@tables.register("index_ds_classes", "MinMo_AV2T")
class MinMo_AV2T(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()

		self.max_source_length = kwargs.get("max_source_length", 6000)
		self.min_source_length = kwargs.get("min_source_length", 0)
		self.max_target_length = kwargs.get("max_target_length", 2048)
		# self.min_target_length = kwargs.get("min_target_length", 0)
		# self.max_token_length = kwargs.get("max_token_length", 2200)

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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
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
					speech_length = data_dict.get("speech_length", 0)
					text_length = data_dict.get("text_length", 0)
					vision_length = data_dict.get("vision_length", 0)
					if speech_length > 0 and speech_length < 1:
						logging.error(
							f"drop it, speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
						)
						continue
					# speech_length = int(speech_length)
					# if speech_length < 1 or text_length < 1:
					#     logging.error(
					#         f"speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
					#     )
					if speech_length > self.max_source_length:
						logging.info(
							f"speech_length: {speech_length} > {self.max_source_length}, drop it: {data_dict}"
						)
						continue
					if speech_length < self.min_source_length:
						logging.info(
							f"speech_length: {speech_length} < {self.min_source_length}, drop it: {data_dict}"
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
							if "prev_content" in item:
								prev_content = item["prev_content"]
								assistant.append([content, {"prev_content": prev_content}])
							else:
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

		logging.info(f"total_num of samplers: {len(self.contents)}, {path}, {file_list}")

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


@tables.register("index_ds_classes", "OpenAIIndexDSJsonlForFullDuplexVAD")
class OpenAIIndexDSJsonlForFullDuplexVAD(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()

		self.max_source_length = kwargs.get("max_source_length", 3000)
		self.min_source_length = kwargs.get("min_source_length", 0)
		self.max_target_length = kwargs.get("max_target_length", 2048)
		self.min_target_length = kwargs.get("min_target_length", 0)
		self.max_token_length = kwargs.get("max_token_length", 2200)

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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
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
					for message in data:
						if message["role"] == "user":
							message["content"] = message["content"].replace(
								"/path/to/vad",
								"/path/to/vad",
							)
							message["content"] = message["content"].replace(
								"/path/to/wav",
								"/path/to/wav",
							)

					speech_length = data_dict.get("speech_length", -1)
					text_length = data_dict.get("text_length", 0)
					task = data_dict["task"]
					last_total_time = data[-1]["end_time"] - data[-1]["start_time"]
					if task == "turn-taking":
						true_time_span = data[-1]["turn-taking-gap_time-added"]
					elif task == "barge-in":
						true_time_span = last_total_time - data[-1]["barge-in-0"]
					if speech_length > self.max_source_length:
						logging.info(
							f"speech_length: {speech_length} > {self.max_source_length}, drop it: {data_dict}"
						)
						continue
					if speech_length < self.min_source_length:
						logging.info(
							f"speech_length: {speech_length} < {self.min_source_length}, drop it: {data_dict}"
						)
						continue
					if text_length > self.max_target_length:
						continue

					self.max_target_length = kwargs.get("max_target_length", 2048)

					system, user, assistant = [], [], []
					for i, item in enumerate(data):
						role = item["role"]
						content = item["content"]
						if role == "system":
							system.append(content)
						elif role == "user":
							user.append(content)
						elif role == "assistant":
							assistant.append(content)

					system = system * len(user)
					assert len(user) - 1 == len(assistant)
					assistant.append("")

					contents_i = {
						"system": system,
						"user": user,
						"assistant": assistant,
						"source_len": speech_length + text_length,
						"task": task,
						"true_time_span": true_time_span,
						"last_total_time": last_total_time,
					}

					contents.append(contents_i)

		self.contents = contents

		logging.info(f"total_num of samplers: {len(self.contents)}, {path}, {file_list}")

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


@tables.register("index_ds_classes", "OpenAIIndexDSJsonlMel")
class OpenAIIndexDSJsonlMel(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()

		# tts text tokenizer related
		tts_token_type = "whisper_rich_ttsfrd"
		ttsfrd_res_dir = "/nfs/neo.dzh/pip_wheels/ttsfrd/9.5.5"
		from funasr.models.llm_asr.tts_text_tokenizer.build_tokenizer import build_tokenizer

		self.tts_text_tokenizer = build_tokenizer(
			tts_token_type,
			bpemodel=ttsfrd_res_dir,
			p_word2phn=1.0,
		)

		self.max_source_length = kwargs.get("max_source_length", 3000)
		self.min_source_length = kwargs.get("min_source_length", 0)
		self.max_target_length = kwargs.get("max_target_length", 2048)
		self.min_target_length = kwargs.get("min_target_length", 0)
		self.max_token_length = kwargs.get("max_token_length", 2200)

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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
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
					speech_length = data_dict.get("speech_length", -1)
					text_length = data_dict.get("text_length", 0)
					if speech_length > self.max_source_length:
						logging.info(
							f"speech_length: {speech_length} > {self.max_source_length}, drop it: {data_dict}"
						)
						continue
					if speech_length < self.min_source_length:
						logging.info(
							f"speech_length: {speech_length} < {self.min_source_length}, drop it: {data_dict}"
						)
						continue
					if text_length > self.max_target_length:
						continue

					self.max_target_length = kwargs.get("max_target_length", 2048)

					system, user, assistant = [], [], []
					for i, item in enumerate(data):
						role = item["role"]
						content = item["content"]
						if role == "system":
							system.append(content)
						elif role == "user":
							user.append(content)
						elif role == "assistant":
							if "wav_path" in item:
								wav_path = item["wav_path"]
								assistant.append([content, {"wav_path": wav_path}])
							else:
								assistant.append(content)

					system = system * len(user)

					for i, (system_i, user_i, assistant_i) in enumerate(
						zip(system, user, assistant)
					):
						contents_i = {
							"system": system[: i + 1],
							"user": user[: i + 1],
							"assistant": assistant[: i + 1],
							"source_len": speech_length + text_length,
						}
						contents.append(contents_i)

		self.contents = contents

		logging.info(f"total_num of samplers: {len(self.contents)}, {path}, {file_list}")

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


@tables.register("index_ds_classes", "MinMo_T2S")
@tables.register("index_ds_classes", "MinMo_S2S")
class MinMo_T2S(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()

		self.max_source_length = kwargs.get("max_source_length", 6000)
		self.min_source_length = kwargs.get("min_source_length", 0)
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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
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
					speech_length = data_dict.get("speech_length", -1)
					text_length = data_dict.get("text_length", 0)
					if speech_length > self.max_source_length:
						logging.info(
							f"speech_length: {speech_length} > {self.max_source_length}, drop it: {data_dict}"
						)
						continue
					if speech_length < self.min_source_length:
						logging.info(
							f"speech_length: {speech_length} < {self.min_source_length}, drop it: {data_dict}"
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
										var_tmp = item[key_tmp]

										meta_dict[key_tmp] = var_tmp
								if len(meta_dict) > 0:
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

		logging.info(f"total_num of samplers: {len(self.contents)}, {path}, {file_list}")

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


@tables.register("index_ds_classes", "MinMo_S2T_data")
class MinMo_S2T_data(torch.utils.data.Dataset):  # torch.utils.data.Dataset

	def __init__(self, path: str, **kwargs):
		super().__init__()
		# import pdb; pdb.set_trace()
		self.max_source_length = kwargs.get("max_source_length", 6000)
		self.min_source_length = float(kwargs.get("min_source_length", -10))
		self.max_target_length = kwargs.get("max_target_length", 2048)
		self.min_target_length = float(kwargs.get("min_target_length", -10))
		# self.max_token_length = kwargs.get("max_token_length", 2200)

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
							data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
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
					try:
						data_dict = json.loads(line.strip())
					except:
						logging.warning(f"bad jsonl line: {file_json}, {line}")
					data = data_dict["messages"]
					if isinstance(data_dict.get("speech_length", 0), (list, tuple)):
						speech_length = int(data_dict.get("speech_length", [0])[0])
						text_length = int(data_dict.get("text_length", 0)[0])
					else:
						speech_length = int(data_dict.get("speech_length", 0))
						text_length = int(data_dict.get("text_length", 0))
					# if "<|startofspeech|>" not in line or "<|endofspeech|>" not in line:
					#     logging.error(
					#         f"drop it, Speech not in S2T data : {data}, file_json: {file_json}"
					#     )
					#     continue
					# if speech_length > 0 and speech_length < 1:
					# 	logging.error(
					# 		f"drop it, speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
					# 	)
					# 	continue
					# speech_length = int(speech_length) if speech_length < 1 or text_length < 1:
					# if text_length < 1:
					# 	logging.error(
					# 		f"speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
					# 	)
					# if speech_length * 1.3 > self.max_source_length:
					# 	logging.info(
					# 		f"speech_length: {speech_length * 8} > {self.max_source_length}, drop it: {data_dict}"
					# 	)
					# 	continue
					# if speech_length < self.min_source_length:
					# 	logging.info(
					# 		f"speech_length: {speech_length * 8} < {self.max_source_length}, drop it: {data_dict}"
					# 	)
					# 	continue
					# if text_length > self.max_target_length:
					# 	logging.info(
					# 		f"text_length: {text_length} > {self.max_target_length}, drop it: {data_dict}"
					# 	)
					# 	continue

					system, user, assistant = [], [], []
					for i, item in enumerate(data):
						role = item["role"]
						content = item["content"]
						if role == "system":
							system.append(content)
						elif role == "user":
							user.append(content)
						elif role == "assistant":
							# if "wav_path" in item:
							#     wav_path = item["wav_path"]
							#     assistant.append([content, {"wav_path": wav_path}])
							# else:
							#     assistant.append(content)
							# assistant.append(content)
							if "prev_content" in item:
								prev_content = item["prev_content"]
								assistant.append([content, {"prev_content": prev_content}])
							else:
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
						contents_i["key"] = data_dict["key"] if not isinstance(data_dict.get("speech_length", 0),
																			   (list, tuple)) else data_dict["key"][0]

					if kwargs.get("save_meta", False):
						contents_i["meta"] = data_dict
					contents.append(contents_i)

		self.contents = contents

		logging.info(f"total_num of samplers: {len(self.contents)}, {path}, {file_list}")

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
	index_ds = MinMo_S2T(path="/path/to/tmp_wav_10.jsonl")
	print(index_ds.contents)
	pass
