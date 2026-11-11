import io
import numpy as np
import soundfile as sf


def load_audio(source, filters=None, offset=0, duration=None, cache_url=False, **kwargs):
    """Small audiolab compatibility shim for FunASR load_utils."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    elif hasattr(source, "read"):
        pass

    start = int(offset or 0) if isinstance(offset, int) else 0
    frames = -1
    data, sample_rate = sf.read(source, start=start, frames=frames, dtype="float32", always_2d=False)
    if duration is not None:
        n = int(float(duration) * sample_rate)
        data = data[:n]
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), sample_rate
