import numpy as np


def _hz_to_mel(frequencies):
    scalar = np.isscalar(frequencies)
    frequencies = np.atleast_1d(frequencies).astype(float)
    f_sp = 200.0 / 3
    mels = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = frequencies >= min_log_hz
    mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    return mels[0] if scalar else mels


def _mel_to_hz(mels):
    scalar = np.isscalar(mels)
    mels = np.atleast_1d(mels).astype(float)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs[0] if scalar else freqs


def mel_filter(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
    """librosa-compatible Slaney mel filterbank without importing librosa/numba."""
    if fmax is None:
        fmax = float(sr) / 2
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    mel_f = _mel_to_hz(mel_points)

    fdiff = np.diff(mel_f)
    ramps = mel_f[:, np.newaxis] - fftfreqs[np.newaxis, :]
    lower = -ramps[:-2] / fdiff[:-1, np.newaxis]
    upper = ramps[2:] / fdiff[1:, np.newaxis]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    return weights.astype(np.float32)
