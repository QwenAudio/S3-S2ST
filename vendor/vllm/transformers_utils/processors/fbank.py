from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)


def hertz_to_mel(
    freq: Union[float, np.ndarray], mel_scale: str = "htk"
) -> Union[float, np.ndarray]:
    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    elif mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0

    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = (
            min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
        )
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep

    return mels


def mel_to_hertz(
    mels: Union[float, np.ndarray], mel_scale: str = "htk"
) -> Union[float, np.ndarray]:
    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    elif mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0

    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(
            logstep * (mels[log_region] - min_log_mel)
        )
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))

    return freq


def _create_triangular_filter_bank(
    fft_freqs: np.ndarray, filter_freqs: np.ndarray
) -> np.ndarray:
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
    triangularize_in_mel_space: bool = False,
) -> np.ndarray:
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # center points of the triangular mel filters
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

    if triangularize_in_mel_space:
        # frequencies of FFT bins in Hz, but filters triangularized in mel space
        fft_bin_width = sampling_rate / (num_frequency_bins * 2)
        fft_freqs = hertz_to_mel(
            fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale
        )
        filter_freqs = mel_freqs
    else:
        # frequencies of FFT bins in Hz
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (
            filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters]
        )
        mel_filters *= np.expand_dims(enorm, 0)

    if (mel_filters.max(axis=0) == 0.0).any():
        logger.warning(
            "At least one mel filter has all zero values. "
            "The value for `num_mel_filters` (%d) may be set too high. "
            "Or, the value for `num_frequency_bins` () may be set too low.",
            num_mel_filters,
            num_frequency_bins,
        )

    return mel_filters


class FastAudioFeature:

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
        device: str = "cpu",
    ):
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window = torch.hann_window(n_fft, device=device)
        mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.mel_filters = torch.from_numpy(mel_filters).to(device, torch.float32).T
        self.device = device

    def get_fbanks(
        self, raw_speech: Union[np.ndarray, List[np.ndarray]], padding_length: int = 0
    ):
        is_numpy_array = isinstance(raw_speech, np.ndarray)
        # padding
        if is_numpy_array:
            audio_ins = torch.from_numpy(raw_speech).to(self.device, torch.float32)
            audio_len = audio_ins.size(0)
            mask_ins = torch.ones(
                audio_len // self.hop_length, dtype=torch.int32, device="cpu"
            )
            if padding_length > 0:
                if audio_len > padding_length:
                    audio_ins = audio_ins[:padding_length]
                    mask_ins = mask_ins[: padding_length // self.hop_length]
                elif audio_len < padding_length:
                    audio_padding_size = padding_length - audio_len
                    mask_padding_size = (
                        padding_length // self.hop_length - audio_len // self.hop_length
                    )
                    audio_ins = F.pad(audio_ins, (0, audio_padding_size))
                    mask_ins = F.pad(mask_ins, (0, mask_padding_size))
        else:
            audio_lst = []
            mask_lst = []
            for speech in raw_speech:
                audio_cur = torch.from_numpy(speech).to(self.device, torch.float32)
                audio_len = audio_cur.size(0)
                mask_cur = torch.ones(
                    audio_len // self.hop_length, dtype=torch.int32, device="cpu"
                )
                if padding_length > 0:
                    if audio_len > padding_length:
                        audio_cur = audio_cur[:padding_length]
                        mask_cur = mask_cur[: padding_length // self.hop_length]
                    elif audio_len < padding_length:
                        audio_padding_size = padding_length - audio_len
                        mask_padding_size = (
                            padding_length // self.hop_length
                            - audio_len // self.hop_length
                        )
                        audio_cur = F.pad(audio_cur, (0, audio_padding_size))
                        mask_cur = F.pad(mask_cur, (0, mask_padding_size))
                audio_lst.append(audio_cur)
                mask_lst.append(mask_cur)
            audio_ins = torch.stack(audio_lst, dim=0)
            mask_ins = torch.stack(mask_lst, dim=0)
        # get fbank
        stft = torch.stft(
            audio_ins,
            self.n_fft,
            self.hop_length,
            window=self.window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if audio_ins.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.cpu(), mask_ins
