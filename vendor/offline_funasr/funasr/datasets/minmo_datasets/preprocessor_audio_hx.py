import os
import json
from io import BytesIO

import torch
import logging
import concurrent.futures
import librosa
import torch.distributed as dist
from typing import Collection
import torch
import torchaudio
from torch import nn
import random
import regex as re
import string
from funasr.tokenizer.cleaner import TextCleaner
from funasr.register import tables
import numpy as np
from funasr.utils.load_utils import load_audio_text_image_video
import soundfile as sf
from scipy import signal
import ctypes

try:
    from funasr.datasets.minmo_datasets.wham_room import WhamRoom
    from numba import njit
except ImportError as e:
    logging.error(f"Failed to import required libraries: {e}")

"""
需要支持音频的处理

1. 单麦 加噪 加混响
2. 多麦 加噪 加混响 + 抽取第二通道/钉钉前端

"""

# oss init
oss_connect_timeout = 1  # second

try:
    import oss2
    from oss2.credentials import EnvironmentVariableCredentialsProvider

    # 设置Endpoint和Region
    OSS_ACCESS_KEY_ID ="" # supply through the caller environment
    OSS_ACCESS_KEY_SECRET ="" # supply through the caller environment

    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    bucket_sh_spider = oss2.Bucket(auth, os.environ.get("OSS_ENDPOINT", "https://oss-cn-shanghai.aliyuncs.com"), "example-storage-4",
                                   region="cn-shanghai")
    bucket_sh_fun = oss2.Bucket(auth, os.environ.get("OSS_ENDPOINT", "https://oss-cn-shanghai.aliyuncs.com"), "example-storage-1",
                                region="cn-shanghai")
    bucket_wlcb_spider = oss2.Bucket(auth, os.environ.get("OSS_ENDPOINT_WLCB", "https://oss-cn-wulanchabu.aliyuncs.com"), "example-storage-5",
                                     region="cn-wulanchabu")
    bucket_wlcb = oss2.Bucket(auth, os.environ.get("OSS_ENDPOINT_WLCB", "https://oss-cn-wulanchabu.aliyuncs.com"), os.environ.get("OSS_BUCKET_WLCB", "example-storage-5"),
                              region="cn-wulanchabu")



except ImportError:
    logging.warning("OSS not installed, please intall oss2.")
    pass
except AttributeError:
    logging.warning("Environment variable of OSS not set.")
    pass


def read_audiobuffer_from_oss_support_multimic(wav_oss_path):
    if 'oss://' in wav_oss_path:
        wav_oss_path = wav_oss_path.replace('oss://', '')
    wav_oss_path_s = wav_oss_path.split('/')
    bucket_name = wav_oss_path_s[0]

    object_name = '/'.join(wav_oss_path_s[1:])  # "audio_processed/shengkui.zhao/test/wav2.wav"
    if not object_name.endswith(".wav"):
        logging.error(f"Object name {object_name} does not end with .wav")
        return None

    # print(f"bucket_name: {bucket_name}, object_name: {object_name}")
    bucket_now = None

    if bucket_name == 'example-storage-1':
        bucket_now = bucket_sh_fun
    elif bucket_name == 'example-storage-4':
        bucket_now = bucket_sh_spider
    elif bucket_name == 'example-storage-5':
        bucket_now = bucket_wlcb_spider
    elif bucket_name == 'nls-resource-wulanchabu':
        bucket_now = bucket_wlcb
    else:
        logging.warning(f'{bucket_name} is not supported')
        return None

    try:
        buffer = BytesIO(bucket_now.get_object(object_name).read())
        return buffer
    except Exception as e:
        logging.error(f'Error loading from {wav_oss_path}: {e}')
        return None


def apply_rir_multi_channel(audio_data, rir):
    # audio_data = audio_data[:,None].repeat(rir.shape[1], axis=1) #T mic
    # convolved_results = signal.fftconvolve(audio_data, rir, mode='same', axes=0)
    # return convolved_results #T*C

    # def add_pyreverb(clean_speech, rir):
    orginal_length = audio_data.shape[0]
    audio_data = audio_data[:, None].repeat(rir.shape[1], axis=1)  # T mic
    # print(f"audio_data.shape: {audio_data.shape}, {rir.shape}")
    # reverb_speech = signal.fftconvolve(audio_data, rir, mode="full")
    # print(f"reverb_speech.shape: {reverb_speech.shape}")
    reverb_speech = np.stack([
        signal.fftconvolve(audio_data[:, c], rir[:, c], mode='full')
        for c in range(audio_data.shape[1])
    ], axis=1)

    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0: audio_data.shape[0]]

    return reverb_speech


def read_multimic_wav_file(filename, src=None, mic=None, force_mono=True, target_sr=16000, random_source=None):
    if filename.startswith("oss://"):
        # try:
        buffer = read_audiobuffer_from_oss_support_multimic(filename)
        if buffer is None:
            logging.error(f'Failed to read buffer from {filename}')
            return None
        audio_data, mic_sr = sf.read(buffer, dtype='float32')
        # except Exception as e:
        #     logging.error(f'oss read error: {filename}, error: {e}')
        #     return None, None
    else:
        try:
            audio_data, mic_sr = sf.read(filename, dtype='float32')
        except Exception as e:
            logging.error(f'Local file read error: {filename}, error: {e}')
            return None
        # (frames, channel) / (frames, )

    # audio_data = read_wav_file(filename)
    if mic_sr != target_sr:
        try:
            audio_data = librosa.resample(audio_data, orig_sr=mic_sr, target_sr=target_sr)
        except Exception as e:
            logging.error(f'Resample error: {filename}, error: {e}')
            return None

    if force_mono and src is None and mic is None:
        if audio_data.ndim > 1:
            if random_source is None:
                random_source = random
            chn = random_source.randint(0, audio_data.shape[1] - 1)
            audio_data = audio_data[:, chn]

    # 如果提供了src和mic参数，且音频是2D的，则尝试reshape
    if src is not None and mic is not None and audio_data.ndim == 2 and src * mic == audio_data.shape[1]:
        # Reshape into (T, src, mic)
        audio_data = audio_data.reshape((audio_data.shape[0], src, mic))

    if audio_data.ndim == 1:
        return audio_data[:, None]
    else:
        return audio_data


def adjust_noise_for_snr(speech, noise, snr_db):
    speech_power = np.mean(speech ** 2)
    # 避免除零错误
    if speech_power == 0:
        return noise
    snr_linear = 10 ** (snr_db / 10)
    noise_power = speech_power / snr_linear
    noise_mean_square = np.mean(noise ** 2)
    # 避免除零错误
    if noise_mean_square == 0:
        return noise
    noise = noise * np.sqrt(noise_power / noise_mean_square)
    return noise


def extend_or_trim_noise(noise, target_length, random_source=None):
    if noise is None or noise.shape[0] == 0:
        logging.warning("Noise is empty or None")
        return noise

    if random_source is None:
        random_source = random

    noise_len = noise.shape[0]
    if noise.ndim == 1:  # Single channel case
        if noise_len < target_length:
            if noise_len < target_length // 2 or random_source.random() < 0.9:
                repeat_count = (target_length + noise_len - 1) // noise_len
                noise = np.tile(noise, repeat_count)[:target_length]
            else:
                pad_length = target_length - noise_len
                pad = (pad_length, 0) if random_source.random() < 0.5 else (0, pad_length)
                noise = np.pad(noise, pad, mode='constant', constant_values=0)
        elif noise_len > target_length:
            start = random_source.randint(0, noise_len - target_length)
            noise = noise[start:start + target_length]
    else:  # Multi-channel case
        if noise_len < target_length:
            if noise_len < target_length // 2 or random_source.random() < 0.9:
                repeat_count = (target_length + noise_len - 1) // noise_len
                noise = np.tile(noise, (repeat_count, 1))[:target_length, :]
            else:
                pad_length = target_length - noise_len
                pad = (pad_length, 0) if random_source.random() < 0.5 else (0, pad_length)
                noise = np.pad(noise, (pad, (0, 0)), mode='constant', constant_values=0)
        elif noise_len > target_length:
            start = random_source.randint(0, noise_len - target_length)
            noise = noise[start:start + target_length, :]
    return noise


def rescale(data, random_source=None):
    if data is None or data.size == 0:
        return data

    if random_source is None:
        random_source = random

    RANGE = [0.8, 0.99]
    min_norm, max_norm = RANGE
    sample_norm = random_source.uniform(min_norm, max_norm - 1e-6) # substract eps to exclude max_norm
    # print(sample_norm)
    data_norm = max(1, np.linalg.norm(data, np.inf))  # 避免除零错误
    scale = sample_norm / data_norm  # (data_norm + MIN_AMPLITUDE)
    return data * scale


def fe(mic, dingding_fe_path=None):
    assert dingding_fe_path is not None
    LIB_PATH = dingding_fe_path
    lib_a1_process = ctypes.cdll.LoadLibrary(LIB_PATH)
    mic_data_1 = mic[:, 0].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    mic_data_2 = mic[:, 1].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    mic_data_3 = mic[:, 2].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    mic_data_4 = mic[:, 3].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    mic_data_5 = mic[:, 4].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    mic_data_6 = mic[:, 5].flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_data_1 = np.zeros_like(mic[:, 0].flatten(), dtype=np.float32)
    out_data_c = out_data_1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib_a1_process.process_6ch(
        16000,
        mic.shape[0],
        mic_data_1,
        mic_data_2,
        mic_data_3,
        mic_data_4,
        mic_data_5,
        mic_data_6,
        out_data_c
    )
    return out_data_1


def process_single_audio_simu_6mic(
        input_audio,
        noise_file,
        rir_6mic_file,
        isotropic_noise_file=None,
        snr_range=(5, 20),
        isotropic_snr_range=(5, 20),
        noise_probability=0.9,
        isotropic_probability=0.9,
        random_source=None
):
    """
    Process single audio with 6mic noise and reverberation simulation.

    Args:
        input_audio: Input audio data (numpy array)
        noise_file: Path to noise file
        rir_6mic_file: Path to 6mic RIR file
        isotropic_noise_file: Path to isotropic noise file
        snr_range: SNR range for noise
        isotropic_snr_range: SNR range for isotropic noise
        noise_probability: Probability of adding noise
        isotropic_probability: Probability of adding isotropic noise
        random_source: Random number generator to use for reproducible randomness

    Returns:
        Processed 6mic audio with noise and reverberation
    """

    if random_source is None:
        random_source = random

    # 处理speech pad

    processed_speech = input_audio

    # 读取6mic RIR文件
    rir_data = read_multimic_wav_file(rir_6mic_file, src=2, mic=6, force_mono=False, random_source=random_source)
    if rir_data is None:
        logging.warning(f"Failed to read RIR file: {rir_6mic_file}")
        return processed_speech

    # 分离speech和noise的RIR
    if rir_data.ndim == 3 and rir_data.shape[1] == 2 and rir_data.shape[2] == 6:
        # RIR数据已经是(T, src, mic)格式
        rir_speech = rir_data[:, 0, :]  # First source for speech (all 6 mics)
        rir_noise = rir_data[:, 1, :]  # Second source for noise (all 6 mics)
    elif rir_data.ndim == 2 and rir_data.shape[1] == 12:
        # RIR数据是(T, src*mic)格式，需要重塑
        rir_data = rir_data.reshape((rir_data.shape[0], 2, 6))
        rir_speech = rir_data[:, 0, :]  # First source for speech (all 6 mics)
        rir_noise = rir_data[:, 1, :]  # Second source for noise (all 6 mics)
    else:
        # 如果RIR格式不符合预期，直接使用整个RIR
        logging.warning(f"Unexpected RIR format in file: {rir_6mic_file}, shape: {rir_data.shape}")
        rir_speech = rir_data
        rir_noise = None

    # 应用RIR到语音
    # print(f"processed_speech: {processed_speech.shape}, rir_speech: {rir_speech.shape}")
    processed_speech = apply_rir_multi_channel(processed_speech, rir_speech)
    # print("processed_speech:", processed_speech.shape)

    # 添加点噪声
    if noise_file is not None and random_source.random() < noise_probability:
        
        noise_data = read_multimic_wav_file(noise_file, force_mono=True, random_source=random_source)
        if noise_data is not None and noise_data.shape[0] >= 1600:
            # 如果noise_data是多维的，随机选择一个通道
            if noise_data.ndim > 1:
                n_channels = noise_data.shape[1]
                noise_data = noise_data[:, random_source.randint(0, n_channels - 1)]

            # 应用RIR到噪声
            if rir_noise is not None:
                processed_noise = apply_rir_multi_channel(noise_data, rir_noise)
            else:
                processed_noise = noise_data[:, None]

            # 调整噪声的SNR
            selected_snr = random_source.uniform(snr_range[0], snr_range[1])
            processed_noise = adjust_noise_for_snr(processed_speech, processed_noise, selected_snr)

            # 扩展或裁剪噪声以匹配语音长度
            processed_noise = extend_or_trim_noise(processed_noise, processed_speech.shape[0], random_source)

            # print("add noise:", noise_file)

            # 添加噪声到语音
            processed_speech += processed_noise

    # 添加各向同性噪声
    if isotropic_noise_file is not None and random_source.random() < isotropic_probability:
        
        isotropic_noise_data = read_multimic_wav_file(isotropic_noise_file, force_mono=True, random_source=random_source)
        if isotropic_noise_data is not None and isotropic_noise_data.shape[0] >= 1600:
            # 如果isotropic_noise_data是多维的，随机选择一个通道
            if isotropic_noise_data.ndim > 1:
                n_channels = isotropic_noise_data.shape[1]
                isotropic_noise_data = isotropic_noise_data[:, random_source.randint(0, n_channels - 1)]

            # 调整各向同性噪声的SNR
            isotropic_snr = random_source.uniform(isotropic_snr_range[0], isotropic_snr_range[1])
            isotropic_noise_scaled = adjust_noise_for_snr(
                processed_speech.mean(axis=1),
                isotropic_noise_data,
                isotropic_snr
            )

            # 扩展或裁剪各向同性噪声以匹配语音长度
            isotropic_noise_scaled = extend_or_trim_noise(isotropic_noise_scaled, processed_speech.shape[0], random_source)
            
            # print(f"Adding isotropic noise from: {isotropic_noise_file}")

            # 添加各向同性噪声到语音
            processed_speech += isotropic_noise_scaled[:, np.newaxis]

    # 重新缩放
    processed_speech = rescale(processed_speech, random_source)

    # 根据距离调整第一个麦克风的音量
    try:
        # 从文件名中提取距离信息，例如: 1000_3.4972_1.1272_3.7255_0.68_rir.wav
        spk_dis = float(os.path.basename(rir_6mic_file).split("_")[1])
        if spk_dis > 0.3:  # >0.3m 时抑制chn-0 24db
            processed_speech[:, 0] = processed_speech[:, 0] * np.sqrt(10 ** (-24 / 10))
            # print(f"Adjusting volume of first channel by {spk_dis}m")
    except (ValueError, IndexError):
        # 如果无法解析距离信息，则不进行调整
        pass

    return processed_speech


def process_single_audio_simu2(
        input_audio,
        noise_file,
        rir_file,
        isotropic_noise_file=None,
        snr_range=None,
        isotropic_snr_range=None,
        noise_probability=None,
        isotropic_probability=None,
        random_source=None
):
    """
    Process single audio with noise and reverberation simulation.

    Args:
        input_audio: Input audio data
        noise_file: Path to noise file
        rir_file: Path to RIR file
        isotropic_noise_file: Path to isotropic noise file
        snr_range: SNR range for noise
        isotropic_snr_range: SNR range for isotropic noise
        noise_probability: Probability of adding noise
        isotropic_probability: Probability of adding isotropic noise
        random_source: Random number generator to use for reproducible randomness

    Returns:
        Processed audio with noise and reverberation
    """

    if random_source is None:
        random_source = random
        
    src = None
    mic = None

    if isotropic_noise_file is None:
        if random_source.random() <= 0.3:  # 30% probability to regard pnt as iso noise
            isotropic_noise_file = noise_file
            noise_file = None
        else:
            isotropic_noise_file = None

    # speech_data = read_wav_file(speech_file, st, et)
    # 这里speech_data是int16的，我应该以float的形式处理
    speech_data = input_audio  # (T, )

    processed_speech = speech_data

    rir_data = read_multimic_wav_file(rir_file, src, mic, random_source=random_source)
    # print("rir_data.shape:", rir_data.shape)
    if rir_data.ndim == 3:
        rir_speech = rir_data[:, 0, :]  # All mics for first source (speech)
        rir_noise = rir_data[:, 1, :]  # All mics for second source (noise)
    else:
        rir_speech = rir_data
        rir_noise = None

    # processed_speech = apply_rir_multi_channel(speech_data, rir_speech) #T*C
    processed_speech = apply_rir_multi_channel(processed_speech, rir_speech)  # T*C
    # print("processed_speech.shape:", processed_speech.shape)

    if noise_file is not None and random_source.random() < noise_probability:
        noise_data = read_multimic_wav_file(noise_file, random_source=random_source)
        if noise_data is None or noise_data.shape[0] < 1600:  # too short
            logging.warning("too short noise_file:", noise_file)
        else:
            if noise_data.ndim > 1:
                n_channels = noise_data.shape[1]
                noise_data = noise_data[:, random_source.randint(0, n_channels - 1)]
            if rir_noise is not None:
                processed_noise = apply_rir_multi_channel(noise_data, rir_noise)
            else:
                processed_noise = noise_data[:, None]
            selected_snr = random_source.uniform(snr_range[0], snr_range[1])
            processed_noise = adjust_noise_for_snr(processed_speech, processed_noise, selected_snr)
            processed_noise = extend_or_trim_noise(processed_noise, processed_speech.shape[0], random_source)
            # print(f"processed_noise.shape: {processed_noise.shape}")
            processed_speech += processed_noise

    if isotropic_noise_file is not None and random_source.random() < isotropic_probability:
        # isotropic_noise_data = read_wav_file(isotropic_noise_file)
        isotropic_noise_data = read_multimic_wav_file(isotropic_noise_file, random_source=random_source)
        if isotropic_noise_data is None or isotropic_noise_data.shape[0] < 1600:  # too short
            logging.warning("too short isotropic_noise_file:", isotropic_noise_file)
        else:
            if isotropic_noise_data.ndim > 1:  # random select one channel
                n_channels = isotropic_noise_data.shape[1]
                isotropic_noise_data = isotropic_noise_data[:, random_source.randint(0, n_channels - 1)]
            isotropic_snr = random_source.uniform(isotropic_snr_range[0], isotropic_snr_range[1])
            isotropic_noise_scaled = adjust_noise_for_snr(processed_speech.mean(axis=1), isotropic_noise_data,
                                                          isotropic_snr)
            isotropic_noise_scaled = extend_or_trim_noise(isotropic_noise_scaled, processed_speech.shape[0], random_source)
            # print(f"isotropic_noise_scaled.shape: {isotropic_noise_scaled.shape}")
            processed_speech += isotropic_noise_scaled[:, np.newaxis]

    processed_speech = rescale(processed_speech, random_source)
    # if src == 1 and mic == 1:  # for far-field 1-chn simulation
    #     processed_speech = processed_speech.flatten()
    #     out_path = os.path.join(output_dir, name) + "_" + gen_with_choice() + ".wav"
    # else:
    #     spk_dis = float(os.path.basename(rir_file).split("_")[1])  # 1000_3.4972_1.1272_3.7255_0.68_rir.wav
    #     rt = float(os.path.basename(rir_file).split("_")[4])
    #     out_path = os.path.join(output_dir, name) + "_" + gen_with_choice() + "_" + str(spk_dis) + "_" + str(
    #         rt) + ".wav"
    #     if spk_dis > 0.3:  # >0.3m suppress chn-0 24db
    #         processed_speech[:, 0] = processed_speech[:, 0] * np.sqrt(10 ** (-24 / 10))

    return processed_speech


@tables.register("preprocessor_classes", "AudioPreprocessAddNoiseReverbMultiMic")
class AudioPreprocessAddNoiseReverbMultiMic(nn.Module):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

        self.kwargs = kwargs

        self.simu_1mic_conf = self.kwargs.get("simu_1mic_conf", dict(
            snr_range=(5, 20),
            isotropic_snr_range=(5, 20),
            noise_probability=0.0,
            isotropic_probability=0.9
        ))
        self.simu_6mic_conf = self.kwargs.get("simu_6mic_conf", dict(
            snr_range=(5, 20),
            isotropic_snr_range=(5, 20),
            noise_probability=0.9,
            isotropic_probability=0.9
        ))

        self.use_6mic_probability = self.kwargs.get("use_6mic_probability", 0.5)

        self.fs = self.kwargs.get("fs", 16000)
        self.dingding_fe_path = self.kwargs.get("dingding_fe_path", None)
        self.dingding_fe_probability = self.kwargs.get("dingding_fe_probability", 0.5)

        # Load noise and RIR resources
        self.load_noise_rir_resource()

    def load_noise_rir_resource(self):
        # Initialize lists even if files are not provided or fail to load
        self.noise_list = []
        self.isotropic_noise_list = []
        self.rir_list = []
        self.rir_6mic_list = []
        self.noise_list_4_6mic = []
        self.isotropic_noise_list_4_6mic = []

        self.noise_list_file = self.kwargs.get("noise_list", None)
        if self.noise_list_file is not None:
            try:
                with open(self.noise_list_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.noise_list.append(line)
            except Exception as e:
                logging.error(f"Load noise list failed: {e}")
        self.noise_list_length = len(self.noise_list)

        self.isotropic_noise_list_file = self.kwargs.get("isotropic_noise_list", None)
        if self.isotropic_noise_list_file is not None:
            try:
                with open(self.isotropic_noise_list_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.isotropic_noise_list.append(line)
            except Exception as e:
                logging.error(f"Load isotropic noise list failed: {e}")
        self.isotropic_noise_list_length = len(self.isotropic_noise_list)

        self.rir_list_file = self.kwargs.get("rir_list", None)
        if self.rir_list_file is not None:
            try:
                with open(self.rir_list_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.rir_list.append(line)
            except Exception as e:
                logging.error(f"Load rir list failed: {e}")
        self.rir_list_length = len(self.rir_list)

        self.rir_6mic_list_file = self.kwargs.get("rir_6mic_list", None)
        if self.rir_6mic_list_file is not None:
            try:
                with open(self.rir_6mic_list_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.rir_6mic_list.append(line)
            except Exception as e:
                logging.error(f"Load rir 6mic list failed: {e}")
        self.rir_6mic_list_length = len(self.rir_6mic_list)

        self.noise_list_4_6mic_file = self.kwargs.get("noise_list_4_6mic", None)
        if self.noise_list_4_6mic_file is not None:
            try:
                with open(self.noise_list_4_6mic_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.noise_list_4_6mic.append(line)
            except Exception as e:
                logging.error(f"Load noise list 4_6mic failed: {e}")
        self.noise_list_4_6mic_length = len(self.noise_list_4_6mic)

        self.isotropic_noise_list_4_6mic_file = self.kwargs.get("isotropic_noise_list_4_6mic", None)
        if self.isotropic_noise_list_4_6mic_file is not None:
            try:
                with open(self.isotropic_noise_list_4_6mic_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.isotropic_noise_list_4_6mic.append(line)
            except Exception as e:
                logging.error(f"Load isotropic noise list 4_6mic failed: {e}")
        self.isotropic_noise_list_4_6mic_length = len(self.isotropic_noise_list_4_6mic)

    def forward(self, input_audio: np.ndarray, use_6mic=True, random_source=None):
        """
        Apply noise and reverberation augmentation to input audio.

        Args:
            input_audio: Input audio data as numpy array
            use_6mic: Whether to use 6mic processing, True of False or 'random'
            random_source: Random number generator to use for reproducible randomness

        Returns:
            Augmented audio data
        """
        if random_source is None:
            random_source = random

        # 根据可用资源选择增强类型
        # use_6mic = (self.rir_6mic_list_length > 0 and
        #            (len(self.rir_list) == 0 or random.random() < 0.5))
        # use_6mic = random.random() < 0.5
        # use_6mic = True
        if use_6mic == 'random':
            use_6mic = random_source.random() < self.use_6mic_probability

        if not use_6mic:

            # 使用标准simu2增强
            noise_file = random_source.choice(self.noise_list)
            rir_file = random_source.choice(self.rir_list)
            isotropic_noise_file = random_source.choice(self.isotropic_noise_list)

            # snr_range = (5, 20)
            # isotropic_snr_range = (5, 20)
            # noise_probability = 0.  # 修复：使用配置的噪声概率而不是硬编码的0.0
            # isotropic_probability = 1.0
            snr_range = self.simu_1mic_conf["snr_range"]
            isotropic_snr_range = self.simu_1mic_conf["isotropic_snr_range"]
            noise_probability = self.simu_1mic_conf["noise_probability"]
            isotropic_probability = self.simu_1mic_conf["isotropic_probability"]

            input_audio = process_single_audio_simu2(
                input_audio,
                noise_file=noise_file,
                rir_file=rir_file,
                isotropic_noise_file=isotropic_noise_file,
                snr_range=snr_range,
                isotropic_snr_range=isotropic_snr_range,
                noise_probability=noise_probability,
                isotropic_probability=isotropic_probability,
                random_source=random_source
            )
        else:
            assert self.rir_6mic_list_length > 0
            # 使用6mic增强
            noise_file = random_source.choice(self.noise_list_4_6mic)
            rir_6mic_file = random_source.choice(self.rir_6mic_list)
            isotropic_noise_file = random_source.choice(self.isotropic_noise_list_4_6mic)

            # snr_range = (5, 20)
            # isotropic_snr_range = (5, 20)
            # noise_probability = 1.0  # 修复：使用配置的噪声概率而不是硬编码的0.0
            # isotropic_probability = 1.0
            snr_range = self.simu_6mic_conf["snr_range"]
            isotropic_snr_range = self.simu_6mic_conf["isotropic_snr_range"]
            noise_probability = self.simu_6mic_conf["noise_probability"]
            isotropic_probability = self.simu_6mic_conf["isotropic_probability"]

            input_audio = process_single_audio_simu_6mic(
                input_audio,
                noise_file=noise_file,
                rir_6mic_file=rir_6mic_file,
                isotropic_noise_file=isotropic_noise_file,
                snr_range=snr_range,
                isotropic_snr_range=isotropic_snr_range,
                noise_probability=noise_probability,
                isotropic_probability=isotropic_probability,
                random_source=random_source
            )
            if self.dingding_fe_path is not None and os.path.exists(self.dingding_fe_path):
                if random_source.random() < self.dingding_fe_probability:
                    input_audio = fe(input_audio, self.dingding_fe_path)
                else:
                    input_audio = input_audio[:, 1]
            else:
                input_audio = input_audio[:, 1]

        input_audio = self.validate_audio(input_audio)
        return input_audio

    def validate_audio(self, data):
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("Audio contains NaN/inf values")
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.flatten()
        return np.clip(data, -1.0, 1.0)


def noise_test():

    # 下面这个noise_list用不上

    noise_list = "/path/to/noise_simu2.lst_cpfs_exist"
    isotropic_noise_list = "/path/to/iso_noise_simu2.lst_cpfs_exist"
    rir_1mic_list = "/path/to/rir_all2_oss.lst_cpfs_exist"

    rir_6mic_list = "/path/to/rir_all1_oss.lst_cpfs_exist"
    noise_list_4_6mic = "/path/to/noise_list_4_6mic.lst_cpfs_exist"
    isotropic_noise_list_4_6mic = "/path/to/isotropic_noise_list_4_6mic.lst_cpfs_exist"

    dingding_fe_path = "/path/to/lib_fe.so"

    noise_preprocess = AudioPreprocessAddNoiseReverbMultiMic(
        noise_list=noise_list,
        isotropic_noise_list=isotropic_noise_list,
        rir_list=rir_1mic_list,
        rir_6mic_list=rir_6mic_list,
        noise_list_4_6mic=noise_list_4_6mic,
        isotropic_noise_list_4_6mic=isotropic_noise_list_4_6mic,
        simu_1mic_conf=dict(
            snr_range=(5, 20),
            isotropic_snr_range=(5, 20),
            noise_probability=0.0,
            isotropic_probability=0.9
        ),
        simu_6mic_conf=dict(
            snr_range=(5, 20),
            noise_probability=0.9,
            isotropic_snr_range=(5, 20),
            isotropic_probability=0.9,
        ),
        fs=16000,
        dingding_fe_path=dingding_fe_path
    )

    input_audio_path = "tmp/concat_mono16k.wav"

    # Example input audio
    input_audio = load_audio_text_image_video(input_audio_path, 16000)
    print("Input audio shape:", input_audio)
    sf.write("tmp/asr_example.wav", input_audio, 16000)

    output_audio = noise_preprocess(input_audio.numpy(), use_6mic=False)
    print("Processed audio shape:", output_audio.shape)

    # sf.write("/mnt/nas/wucong.lyb/add_noise/asr_example_noise.wav", output_audio, 16000)
    sf.write("tmp/asr_example_noise.wav", output_audio, 16000)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # rep_test()
    noise_test()