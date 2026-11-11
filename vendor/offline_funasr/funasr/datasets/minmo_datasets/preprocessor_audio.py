import os
import json
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

try:
    from funasr.datasets.minmo_datasets.wham_room import WhamRoom
    from numba import njit
except ImportError as e:
    logging.error(f"Failed to import required libraries: {e}")

@njit(fastmath=True)
def fast_mix(signal, noise, target_snr_db):
    signal_sqr = signal**2
    noise_sqr = noise**2
    avg_signal_power = np.mean(signal_sqr)
    avg_noise_power = max(np.mean(noise_sqr), 1e-12)
    if len(signal_sqr[signal_sqr>avg_signal_power]) > 0:
        signal_power = np.mean(signal_sqr[signal_sqr>avg_signal_power])
    else:
        signal_power = avg_signal_power + 1e-12
    if len(noise_sqr[noise_sqr>avg_noise_power]) > 0:
        noise_power = np.mean(noise_sqr[noise_sqr>avg_noise_power])
    else:
        noise_power = avg_noise_power + 1e-12

    if noise_power > 0:
        scale = np.sqrt(signal_power / (noise_power * 10**(target_snr_db/10) + 1e-8))
    else:
        scale = 1.0
    mixed = signal + noise * scale
    return mixed.clip(-1.0, 1.0)

@njit(fastmath=True)
def fast_mix_v0(signal, noise, target_snr_db):
    signal_sqr = signal**2
    noise_sqr = noise**2
    avg_signal_power = np.mean(signal_sqr)
    avg_noise_power = max(np.mean(noise_sqr), 1e-12)
    signal_power = np.mean(signal_sqr[signal_sqr>avg_signal_power])
    noise_power = np.mean(noise_sqr[noise_sqr>avg_noise_power])
    scale = np.sqrt(signal_power / (noise_power * 10**(target_snr_db/10)))
    mixed = signal + noise * scale
    #peak = max(np.max(np.abs(mixed)), 1e-8)
    return mixed.clip(-1.0, 1.0)

@tables.register("preprocessor_classes", "AudioPreprocessAddNoiseReverb")
class AudioPreprocessAddNoiseReverb(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        self.kwargs = kwargs
        self.noise_list = []
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
        self.noise_length = len(self.noise_list)
        self.noise_prob = self.kwargs.get("noise_prob", 0.2)
        self.reverb_prob = self.kwargs.get("reverb_prob", 0)
        self.noise_mean_prob = self.kwargs.get("noise_mean_prob", 0.8)
        self.noise_mean = self.kwargs.get("noise_mean", 10)
        self.noise_std = self.kwargs.get("noise_std", 5)
        self.noise_mean2 = self.kwargs.get("noise_mean2", 5)
        self.noise_std2 = self.kwargs.get("noise_std2", 2)
        self.noise_pad = self.kwargs.get("noise_pad", False)
        self.fs = self.kwargs.get("fs", 16000)
    
    def forward(self, input_audio:np.ndarray):
        """
        """
        noise_snr = -1
        random_noise_prob = random.random()
        if random_noise_prob < self.noise_prob:
            ## noise pad
            if self.noise_pad:
                num_zeros_to_pad_left  = random.randint(0, 16000)
                num_zeros_to_pad_right = random.randint(0, 16000)
                input_audio = np.pad(input_audio, (num_zeros_to_pad_left, num_zeros_to_pad_right), mode='constant', constant_values=0)                
            random_noise_mean_prob = random.random()
            while noise_snr < 0:
                if random_noise_mean_prob < self.noise_mean_prob:
                    noise_snr = int(round(np.random.normal(self.noise_mean, self.noise_std)))
                else:
                    noise_snr = int(round(np.random.normal(self.noise_mean2, self.noise_std2)))
        
        random_reverb_prob = random.random()
        if random_reverb_prob < self.reverb_prob:
            try:
                input_audio = self.gen_reverb_wav(input_audio, fs=self.fs)
                input_audio = input_audio[:, 0]
            except Exception as e:
                logging.error(f"Generate reverb audio failed: {e}")

        if random_noise_prob < self.noise_prob:
            input_audio = np.clip(input_audio, -1.0, 1.0)
            data_len  = input_audio.shape[0]

            maxlen_noise = -1
            index = random.randint(0, self.noise_length-1)
            noise_data = load_audio_text_image_video(self.noise_list[index], fs=self.fs)
            noise_data = noise_data.numpy()
            maxlen_noise = noise_data.shape[0]
            while maxlen_noise < data_len:
                index = random.randint(0, self.noise_length-1)
                noise_data_tmp = load_audio_text_image_video(self.noise_list[index], fs=self.fs)
                noise_data_tmp = noise_data_tmp.numpy()
                noise_data = np.concatenate([noise_data, noise_data_tmp], axis=0)
                maxlen_noise = noise_data.shape[0]
            
            input_audio = self.mix_noise_at_snr(input_audio, noise_data, noise_snr)
            input_audio = self.validate_audio(input_audio)
        return input_audio

    def adjust_noise_length(self, signal, noise):
        signal_len = len(signal)
        noise_len = len(noise)
        if noise_len == 0:
            return None
        if noise_len < signal_len:
            repeat_times = int(np.ceil(signal_len / noise_len))
            extended_noise = np.tile(noise, repeat_times)
            return extended_noise[:signal_len]
        elif noise_len > signal_len:
            start_idx = np.random.randint(0, noise_len - signal_len)
            return noise[start_idx:start_idx+signal_len]
        return noise

    def adjust_noise_length_silence(self, signal, noise):
        signal_len = len(signal)
        noise_len = len(noise)

        if noise_len < signal_len:
            repeat_times = int(np.ceil(signal_len / noise_len))
            extended_noise = np.tile(noise, repeat_times)
            return extended_noise[:signal_len]
        elif noise_len >= signal_len:
            retry_count = 0
            while True:
                start_idx = np.random.randint(0, noise_len - signal_len)
                if sum(noise[start_idx:start_idx+signal_len]) != 0 or retry_count > 2:
                    break
                logging.warning(f"Retrying to find a valid noise segment, attempt {retry_count}")
                retry_count += 1
            return noise[start_idx:start_idx+signal_len]
        return noise

    def mix_noise_at_snr(self, signal, noise, target_snr_db):
        noise = self.adjust_noise_length(signal, noise)
        if noise is None:
            return signal
        merged = fast_mix(signal, noise, target_snr_db)
        return merged

    def gen_reverb_wav(self, input_audio, fs):
        orig_len = len(input_audio)
        room_set, mic_set, speaker_set, RT60 = self.get_rir(fs)
        #print(f'RT60: {RT60} s')
        try:
            room = self.gen_room(room_set, mic_set, speaker_set, RT60)
        except Exception as e:
            print(f'generating room Error: {e}')
            return None
            
        room.generate_rirs()
        #input_audio = quantize(input_audio)
        room.add_audio([input_audio])
        reverberant = room.generate_audio(fs=fs)
        reverberant_audio = reverberant[0].T
        
        return reverberant_audio[:orig_len,:]

    def get_rir(self, fs=16000):
        
        #reverberation time
        RT60 = round(np.clip(random.gauss(0.4, 0.3), 0.2, 0.8),2)
        
        #room dimensions
        room_set = {}
        room_set['room_x'] = round(random.uniform(3,11),2)
        room_set['room_y'] = round(random.uniform(3,7),2)
        room_set['room_z'] = round(random.uniform(3.0,4.0),2)
        
        #microphone position
        mic_set = {}
        mic_set['mic_x'] = round(random.uniform(0.1,room_set['room_x']-0.1),2)
        mic_set['mic_y'] = round(random.uniform(0.1,room_set['room_y']-0.1),2)
        mic_set['mic_z'] = round(random.uniform(0.5,2.5),2)
        
        #source/speaker position
        speaker_set = {}
        speaker_set['s_x'] = round(random.uniform(0.3,room_set['room_x']-0.3),2)
        speaker_set['s_y'] = round(random.uniform(0.3,room_set['room_y']-0.3),2)
        speaker_set['s_z'] = round(random.uniform(0.8,1.8),2)

        return room_set, mic_set, speaker_set, RT60
            
    def gen_room(self, room_set, mic_set, speaker_set, RT60):
        room = WhamRoom([room_set['room_x'], room_set['room_y'], room_set['room_z']],
                        [[mic_set['mic_x'], mic_set['mic_y'], mic_set['mic_z']]],
                        [speaker_set['s_x'], speaker_set['s_y'], speaker_set['s_z']],
                        RT60)
        return room

    def validate_audio(self, data):
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("Audio contains NaN/inf values")
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.flatten()
        return np.clip(data, -1.0, 1.0)






def noise_test():
    noise_preprocess = AudioPreprocessAddNoiseReverb(
        noise_list="/mnt/nas/wucong.lyb/add_noise/tmp_1000.lst",
        noise_prob=1,
        reverb_prob=1,
        noise_mean=10,
        noise_std=5,
        fs=16000
    )

    # Example input audio
    input_audio = load_audio_text_image_video("/mnt/nas/wucong.lyb/add_noise/asr_example.wav", 16000)

    output_audio = noise_preprocess(input_audio.numpy())
    print("Processed audio shape:", output_audio.shape)
    import soundfile as sf
    sf.write("/mnt/nas/wucong.lyb/add_noise/asr_example_noise.wav", output_audio, 16000)
    
if __name__ == "__main__":
    # rep_test()
    noise_test()