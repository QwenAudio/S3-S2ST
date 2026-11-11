import os
import torch
import json
import torch.distributed as dist
import numpy as np
import kaldiio
import librosa
try:
    import torchaudio
except:
    print("torchaudio is not installed, if you use torchaudio to load audio, please install it first.")
import time
import logging
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime
from audiolab import load_audio
from audiolab.av import aformat
import re
import io
import itertools
import resource

try:
    from funasr.download.file import download_from_url
except:
    print("urllib is not installed, if you infer from url, please install it first.")
import pdb
import subprocess
from subprocess import CalledProcessError, run

try:
    from .oss_torchaudio import oss_remote_torchaudio_load
except:
    pass


# From oss 团队：https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYZnRbNYCo6m7ry98akx1Z5N
def lexical_normalize_abspath(p):
    assert p.startswith('/')
    stack = []
    for part in p.split('/'):
        if part == '' or part == '.':
            continue
        elif part == '..':
            if stack:
                stack.pop()
        else:
            stack.append(part)
    res = '/'.join(stack)
    return f'/{res}'


def lexical_join_path(base, p):
    assert base.startswith('/')
    if p.startswith('/'):
        return p
    return f'{base}/{p}'


class PathTransformer:
    def __init__(self, roots):
        self._roots = [lexical_normalize_abspath(root) for root in roots]

    def _transform_with_root(self, root, path, cwd=None):
        if cwd is None:
            cwd = os.getcwd()
        abspath = lexical_join_path(cwd, path)
        normpath = lexical_normalize_abspath(abspath)
        if normpath == root or not normpath.startswith(root + '/'):
            return False, path
        if '.efc' in normpath.split('/'):
            return False, path
        relpath = normpath[len(root) + 1:]
        for sep in ('#', '@', ':'):
            if sep not in relpath:
                encoded = relpath.replace('/', sep)
                return True, f'{root}/.efc/flat/{sep}/{encoded}'
        return False, path

    def transform(self, path, cwd=None):
        for root in self._roots:
            encoded, transformed = self._transform_with_root(root, path, cwd)
            if encoded:
                return transformed
        return path


oss_roots = ["/fun_resource_sh_cache", "/nls_spider_sh_cache", "/oss-fun-asr-train-data-shanghai"]
PATH_TRANSFORMER = PathTransformer(oss_roots)


def is_ffmpeg_installed():
    try:
        output = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        return "ffmpeg version" in output.decode("utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


use_ffmpeg = False
if is_ffmpeg_installed():
    use_ffmpeg = True
else:
    print(
        "Notice: ffmpeg is not installed. torchaudio is used to load audio\n"
        "If you want to use ffmpeg backend to load audio, please install it by:"
        "\n\tsudo apt install ffmpeg # ubuntu"
        "\n\t# brew install ffmpeg # mac"
    )

def load_audio_text_image_video(
    data_or_path_or_list,
    fs: int = 16000,
    audio_fs: int = 16000,
    data_type="sound",
    tokenizer=None,
    duration=(0, None),
    **kwargs,
):
    duration = (duration[0], float(duration[1])) if duration[1] is not None else duration
    if isinstance(data_or_path_or_list, (list, tuple)):
        if data_type is not None and isinstance(data_type, (list, tuple)):
            data_types = [data_type] * len(data_or_path_or_list)
            data_or_path_or_list_ret = [[] for d in data_type]
            for i, (data_type_i, data_or_path_or_list_i) in enumerate(zip(data_types, data_or_path_or_list)):
                for j, (data_type_j, data_or_path_or_list_j) in enumerate(zip(data_type_i, data_or_path_or_list_i)):
                    data_or_path_or_list_j = load_audio_text_image_video(
                        data_or_path_or_list_j,
                        fs=fs,
                        audio_fs=audio_fs,
                        data_type=data_type_j,
                        tokenizer=tokenizer,
                        **kwargs,
                    )
                    data_or_path_or_list_ret[j].append(data_or_path_or_list_j)

            return data_or_path_or_list_ret
        else:
            return [
                load_audio_text_image_video(audio, fs=fs, audio_fs=audio_fs, data_type=data_type, **kwargs)
                for audio in data_or_path_or_list
            ]
    if isinstance(data_or_path_or_list, str) and data_or_path_or_list.startswith("http"):  # download url to local file
        data_or_path_or_list = download_from_url(data_or_path_or_list)
    # import pdb; pdb.set_trace()
    if isinstance(data_or_path_or_list, str) and os.path.exists(data_or_path_or_list):  # local file
        if data_type is None or data_type in ["sound", "kaldi_ark_or_sound"]:
            interval = duration
            offset = interval[0]
            duration = None if interval[1] is None else interval[1] - interval[0]
            if kwargs.get("use_ffmpeg", False):
                data_or_path_or_list = _load_audio_ffmpeg(data_or_path_or_list, sr=fs, offset=offset, duration=duration)
                data_or_path_or_list = torch.from_numpy(data_or_path_or_list.copy()).squeeze()  # [n_samples,]
            else:
                try:
                    if data_or_path_or_list.endswith(".pcm"):  # pcm
                        data_or_path_or_list = load_pcm(data_or_path_or_list)
                        audio_fs = fs
                    else:  # wav file
                        frame_offset = offset * fs
                        num_frames = duration * fs if duration is not None else -1
                        data_or_path_or_list, audio_fs = torchaudio.load(data_or_path_or_list, frame_offset=frame_offset, num_frames=num_frames)
                    if kwargs.get("reduce_channels", True):
                        data_or_path_or_list = data_or_path_or_list.mean(0)
                    if audio_fs != fs and data_type != "text":
                        resampler = torchaudio.transforms.Resample(audio_fs, fs)
                        data_or_path_or_list = resampler(data_or_path_or_list[None, :])[0, :]
                except:
                    data_or_path_or_list = _load_audio_ffmpeg(data_or_path_or_list, sr=fs, offset=offset, duration=duration)
                    data_or_path_or_list = torch.from_numpy(data_or_path_or_list.copy()).squeeze()  # [n_samples,]
        elif data_type == "text" and tokenizer is not None:
            data_or_path_or_list = tokenizer.encode(data_or_path_or_list)
        elif data_type == "image":  # undo
            pass
        elif data_type == "video":  # undo
            pass

        # if data_in is a file or url, set is_final=True
        if "cache" in kwargs:
            kwargs["cache"]["is_final"] = True
            kwargs["cache"]["is_streaming_input"] = False
    elif isinstance(data_or_path_or_list, str) and data_type == "text" and tokenizer is not None:
        data_or_path_or_list = tokenizer.encode(data_or_path_or_list)
    elif isinstance(data_or_path_or_list, np.ndarray):  # audio sample point
        data_or_path_or_list = torch.from_numpy(data_or_path_or_list.copy()).squeeze()  # [n_samples,]
    elif isinstance(data_or_path_or_list, str) and data_type in ["kaldi_ark", "kaldi_ark_or_sound"]:
        if len(data_or_path_or_list.split()) == 2:
            data_or_path_or_list, audio_fs = data_or_path_or_list.split()
            audio_fs = int(audio_fs)
        data_mat = kaldiio.load_mat(data_or_path_or_list)
        if isinstance(data_mat, tuple):
            audio_fs, mat = data_mat
        else:
            mat = data_mat
        if mat.dtype == "int16":
            mat = mat.astype(np.float64)
            mat = mat / (2**16)
        elif mat.dtype == "int32":
            mat = mat.astype(np.float64)
            mat = mat / (2**32)
        if mat.ndim == 2:
            mat = mat[:, 0]
        data_or_path_or_list = torch.from_numpy(mat)
    elif isinstance(data_or_path_or_list, bytes):  # audio bytes
        data_or_path_or_list = load_bytes(data_or_path_or_list)
    elif isinstance(data_or_path_or_list, str) and (
        data_or_path_or_list.startswith("oss") or data_or_path_or_list.startswith("/mnt")
    ):
        interval = duration # kwargs.get("duration", [0, None])
        offset = interval[0]
        duration = None if (interval[1] is None) else float(interval[1]) - interval[0]
        # import pdb; pdb.set_trace()
        if kwargs.get("oss_remote_torchaudio_load", False):
            data_or_path_or_list, audio_fs = oss_remote_torchaudio_load(data_or_path_or_list)
        else:
            if data_or_path_or_list.startswith("oss://") or offset > 0 or duration is not None:
                _bucket_name, object_name = data_or_path_or_list[len("oss://"):].split("/", maxsplit=1)
                if _bucket_name not in buckets:
                    buckets[_bucket_name] = oss2.Bucket(auth, endpoint, _bucket_name, region=region, connect_timeout=oss_connect_timeout)
                if not buckets[_bucket_name].object_exists(object_name):
                    raise RuntimeError(f"{object_name} not found")
                url = buckets[_bucket_name].sign_url("GET", object_name, 3600)
                data_or_path_or_list = _load_audio_ffmpeg(url, sr=fs, offset=offset, duration=duration)
                data_or_path_or_list = torch.from_numpy(data_or_path_or_list).squeeze()
            else:
                _bucket_name, object_name = resolve_oss_audio_object_name(data_or_path_or_list)
                if _bucket_name is not None:
                    if _bucket_name not in buckets:
                        buckets[_bucket_name] = oss2.Bucket(auth, endpoint, _bucket_name, region=region, connect_timeout=oss_connect_timeout)
                    url = buckets[_bucket_name].sign_url("GET", object_name, 3600)
                    data_or_path_or_list = _load_audio_ffmpeg(url, sr=fs, offset=offset, duration=duration)
                    data_or_path_or_list = torch.from_numpy(data_or_path_or_list).squeeze()
                else:
                    raise RuntimeError(f"{data_or_path_or_list} not found")
    else:
        pass
        print(f"unsupport data type: {data_or_path_or_list}, return raw data")

    if audio_fs != fs and data_type != "text":
        resampler = torchaudio.transforms.Resample(audio_fs, fs)
        data_or_path_or_list = resampler(data_or_path_or_list[None, :])[0, :]
    return data_or_path_or_list


def _load_audio_oss_bytes(bucket, object_name, sr=16000, offset=0, duration=None):
    # 从OSS直接取字节流
    result = bucket.get_object(object_name)
    audio_bytes = result.read()
    result.close()

    # 用BytesIO包装，audiolab可以当作文件句柄读取
    audio_buf = io.BytesIO(audio_bytes)

    filters = [aformat(rate=sr, dtype=float, to_mono=True)]
    samples, _ = load_audio(audio_buf, filters=filters, offset=offset, duration=duration, cache_url=False)
    return samples.flatten()
def get_oss_object_size(oss_path: str):
    """获取 OSS 上文件的 size（字节）。oss_path 格式为 oss://bucket_name/object_key。失败返回 None。"""
    if not oss_path or not oss_path.startswith("oss://"):
        return None
    try:
        _bucket_name, object_name = oss_path[len("oss://"):].split("/", maxsplit=1)
        if _bucket_name not in buckets:
            return None
        result = buckets[_bucket_name].head_object(object_name)
        return getattr(result, "content_length", None)
    except Exception as e:
        print(f"Error getting OSS object size for {oss_path}: {e}")
        return None


def load_pcm(input):
    with open(input, "rb") as f:
        bytes = f.read()
    return load_bytes(bytes)


def load_bytes(input):
    middle_data = np.frombuffer(input, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    if middle_data.dtype.kind not in "iu":
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype("float32")
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(middle_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
    return array


def extract_fbank(data, data_len=None, data_type: str = "sound", frontend=None, **kwargs):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
        if len(data.shape) < 2:
            data = data[None, :]  # data: [batch, N]
        data_len = [data.shape[1]] if data_len is None else data_len
    elif isinstance(data, torch.Tensor):
        if len(data.shape) < 2:
            data = data[None, :]  # data: [batch, N]
        data_len = [data.shape[1]] if data_len is None else data_len
    elif isinstance(data, (list, tuple)):
        data_list, data_len = [], []
        for data_i in data:
            if isinstance(data_i, np.ndarray):
                data_i = torch.from_numpy(data_i)
            data_list.append(data_i)
            data_len.append(data_i.shape[0])
        data = pad_sequence(data_list, batch_first=True)  # data: [batch, N]

    data, data_len = frontend(data, data_len, **kwargs)

    if isinstance(data_len, (list, tuple)):
        data_len = torch.tensor([data_len])
    return data.to(torch.float32), data_len.to(torch.int32)


_AUDIO_LOAD_COUNTER = itertools.count(1)


def _slow_audio_threshold_s():
    try:
        return float(os.environ.get("FUNASR_SLOW_AUDIO_LOG_THRESHOLD_S", "1.0"))
    except ValueError:
        return 1.0


def _log_slow_audio(decoder, original_path, resolved_path, load_seq, elapsed_s, trace):
    if elapsed_s < _slow_audio_threshold_s():
        return
    try:
        open_fds = len(os.listdir("/proc/self/fd"))
    except OSError:
        open_fds = -1
    logging.warning(
        "[SLOW_AUDIO_IO] decoder=%s elapsed_s=%.3f transform_ms=%.1f "
        "open_ms=%.1f read_ms=%.1f mono_ms=%.1f resample_ms=%.1f "
        "rank=%s host=%s pid=%d load_seq=%d open_fds=%d maxrss_kb=%d "
        "source_sr=%s channels=%s frames=%s offset=%s duration=%s "
        "original_path=%s resolved_path=%s",
        decoder,
        elapsed_s,
        trace.get("transform_ms", 0.0),
        trace.get("open_ms", 0.0),
        trace.get("read_ms", 0.0),
        trace.get("mono_ms", 0.0),
        trace.get("resample_ms", 0.0),
        os.environ.get("RANK", "unknown"),
        os.uname().nodename,
        os.getpid(),
        load_seq,
        open_fds,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        trace.get("source_sr", "unknown"),
        trace.get("channels", "unknown"),
        trace.get("frames", "unknown"),
        trace.get("offset", "unknown"),
        trace.get("duration", "unknown"),
        original_path,
        resolved_path,
    )


def load_audio_ffmpeg(wav_path, sr: int = 16000, offset=0, duration=None, speed=1):
    load_start = time.perf_counter()
    load_seq = next(_AUDIO_LOAD_COUNTER)
    original_wav_path = wav_path
    transform_start = time.perf_counter()
    wav_path = PATH_TRANSFORMER.transform(wav_path)
    trace = {"transform_ms": (time.perf_counter() - transform_start) * 1000}

    # Use FFmpeg for every audio format. Keeping one decoder path makes the
    # training behavior independent of SoundFile/EFC compatibility.
    cmd = ["ffmpeg", "-nostdin", "-threads", "1"]
    if offset > 0:
        cmd += ["-ss", str(offset)]
    cmd += ["-i", wav_path, "-map", "0:a:0"]

    # 修复：duration 和 speed 不应互斥！
    if duration is not None:
        cmd += ["-t", str(duration + 1)]  # 多读 1s，读出来采样点后再裁掉
    if speed != 1:
        if not (0.5 <= speed <= 2.0):
            raise ValueError(f"atempo only supports speed in [0.5, 2.0], got {speed}")
        cmd += ["-af", f"atempo={speed}"]

    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", f"{sr}", "-"]

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            audio_bytes, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise TimeoutError("FFmpeg timeout")

    stderr_str = stderr.decode('utf-8', errors='replace')

    # 关键：即使 returncode == 0，也要检查是否真的有输出！
    if len(audio_bytes) == 0:
        raise RuntimeError(
            f"FFmpeg produced empty audio. Possible causes: "
            f"offset ({offset}) exceeds audio duration, or invalid parameters. "
            f"FFmpeg stderr:\n{stderr_str}\nCommand: {' '.join(cmd)}"
        )

    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
    if duration is not None:
        audio_array = audio_array[:int(duration * sr / speed)]
    trace.update({
        "read_ms": (time.perf_counter() - load_start) * 1000,
        "source_sr": sr,
        "channels": 1,
        "frames": audio_array.shape[0],
        "offset": offset,
        "duration": duration,
    })
    # Disable per-file SLOW_AUDIO_IO warnings during training. Keep the
    # instrumentation above so this call can be restored for diagnosis.
    # _log_slow_audio(
    #     "ffmpeg",
    #     original_wav_path,
    #     wav_path,
    #     load_seq,
    #     time.perf_counter() - load_start,
    #     trace,
    # )
    return audio_array, sr

def _load_audio_ffmpeg(file: str, sr: int = 16000, offset: float = 0, duration: float = None):
    audio_array, _ = load_audio_ffmpeg(file, sr=sr, offset=offset, duration=duration)
    return audio_array


# oss init
oss_connect_timeout = 1  # second

try:
    import oss2
    from oss2.credentials import EnvironmentVariableCredentialsProvider
    from subprocess import CalledProcessError, run
    from subprocess import Popen, PIPE
except ImportError as e:
    logging.warning(f"OSS not installed, please intall oss2. error: {e}")
    pass

try:
    # 设置Endpoint和Region
    OSS_ACCESS_KEY_ID ="" # supply through the caller environment
    OSS_ACCESS_KEY_SECRET ="" # supply through the caller environment
    endpoint = os.environ.get("OSS_ENDPOINT", None)
    region = os.environ.get("OSS_REGION", None)
    bucket_name = os.environ.get("OSS_BUCKET_NAME", None)
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region, connect_timeout=oss_connect_timeout)
    buckets = {bucket_name: bucket}

    # shanghai bucket
    endpoint_sh = os.environ.get("OSS_ENDPOINT_SH", None)
    region_sh = os.environ.get("OSS_REGION_SH", None)
    bucket_name_sh = os.environ.get("OSS_BUCKET_NAME_SH", None)
    if endpoint_sh is not None and region_sh is not None and bucket_name_sh is not None:
        bucket_sh = oss2.Bucket(auth, endpoint_sh, bucket_name_sh, region=region_sh, connect_timeout=oss_connect_timeout)
        buckets[bucket_name_sh] = bucket_sh

    # wulanchabu bucket
    endpoint_wlcb = os.environ.get("OSS_ENDPOINT_WLCB", None)
    region_wlcb = os.environ.get("OSS_REGION_WLCB", None)
    bucket_name_wlcb = os.environ.get("OSS_BUCKET_NAME_WLCB", None)
    if endpoint_wlcb is not None and region_wlcb is not None and bucket_name_wlcb is not None:
        bucket_wlcb = oss2.Bucket(auth, endpoint_wlcb, bucket_name_wlcb, region=region_wlcb, connect_timeout=oss_connect_timeout)
        buckets[bucket_name_wlcb] = bucket_wlcb
except AttributeError as e:
    logging.warning(f"Environment variable of OSS not set. error: {e}")
    pass

def object_exists(object_name):
    try:
         return bucket.object_exists(object_name)
    except oss2.exceptions.NoSuchBucket:
        print(f"Bucket {bucket_name} 不存在")
        return False
    except oss2.exceptions.OssError as e:
        # 其它异常，例如权限不足、网络问题
        print("OSS Error:", e)
        return False

def resolve_oss_audio_object_name(file_path: str):
    # replace pattern by lyb
    if file_path.startswith("/mnt/processed-gzf-train-flac/mnt/wlcb-resource/pengzhendong.pzd"):
        file_path = file_path.replace("/mnt/processed-gzf-train-flac/mnt/wlcb-resource/pengzhendong.pzd", "/mnt/processed-gzf-train-flac/pengzhendong.pzd")

    if file_path.startswith("/mnt/spider-data"):
        file_path = file_path.replace("/mnt/spider-data", "spider-data/aicheng")
        object_name = file_path
    elif file_path.startswith("/mnt/spider-resource/"):
        file_path = file_path[len("/mnt/spider-resource/"):]
        object_name = file_path
    elif file_path.startswith("/mnt/spider-resource-nichongjia/"):
        file_path = file_path[len("/mnt/spider-resource-nichongjia/"):]
        object_name = file_path
    elif file_path.startswith("/mnt/wlcb-resource/"):
        file_path = file_path[len("/mnt/wlcb-resource/"):]
        object_name = file_path
        input_data = bucket_wlcb_res.get_object(object_name).read()
        return input_data
    elif file_path.startswith("/mnt/"):
        file_path = file_path[len("/mnt/"):]
        object_name = file_path
    else:
        object_name = file_path[len("oss://" + bucket_name):]

    if object_exists(object_name):
        return bucket_name, object_name
    else:
        # replace pattern by lyb
        if object_name.startswith("processed-gzf-train-flac/mnt/spider-resource-nichongjia/"):
            object_name = object_name[len("processed-gzf-train-flac/mnt/spider-resource-nichongjia/"):]
            if object_exists(object_name):
                return bucket_name, object_name
        elif object_name.startswith("processed-gzf-train-flac/mnt/spider-resource/"):
            object_name = object_name[len("processed-gzf-train-flac/mnt/spider-resource/"):]
            if object_exists(object_name):
                return bucket_name, object_name
        if "//" in object_name:
            object_name = object_name.replace("//", "/")
            if object_exists(object_name):
                return bucket_name, object_name
        if object_name.endswith(".wav"):
            new_object_name = re.sub(r"\.wav$", ".flac", object_name, flags=re.IGNORECASE)
            if object_exists(new_object_name):
                return bucket_name, new_object_name
        elif object_name.endswith(".flac"):
            new_object_name = re.sub(r"\.flac$", ".wav", object_name, flags=re.IGNORECASE)
            if object_exists(new_object_name):
                return bucket_name, new_object_name

    return None, None

def write_audio_byteio_to_oss(file_path: str, data, key_words="gzf", convert_flac=True, overwrite_outfile=True):
    current_date = datetime.now()
    date = current_date.date()
    if file_path.startswith("/mnt/spider-data"):
        file_path = file_path.replace("/mnt/spider-data", f"processed-{key_words}/")
        object_name = file_path
    elif file_path.startswith("/mnt/processed-gzf-train-flac/"):
        file_path = file_path.replace("/mnt/processed-gzf-train-flac/", f"processed-{key_words}/")
        object_name = file_path
    else:
        object_name = f"processed-{key_words}{file_path}"

    if convert_flac:
        object_name = object_name.replace(".wav", ".flac")
        object_name = object_name.replace(".mp3", ".flac")
        if not overwrite_outfile and bucket.object_exists(object_name):
            logging.warning(f"{object_name} already exists, skip")
            return object_name
        if data is not None:
            result = bucket.put_object(object_name, data)
    else:
        if not overwrite_outfile and bucket.object_exists(object_name):
            logging.warning(f"{object_name} already exists, skip")
            return object_name
        if data is not None:
            result = bucket.put_object(object_name, data)
    return object_name


def exist_audio_byteio_in_oss(file_path: str, data=None, key_words="gzf", convert_flac=True, overwrite_outfile=True):
    current_date = datetime.now()
    date = current_date.date()
    if file_path.startswith("/mnt/spider-data"):
        file_path = file_path.replace("/mnt/spider-data", f"processed-{key_words}/")
        object_name = file_path
    elif file_path.startswith("/mnt/processed-gzf-train-flac/"):
        file_path = file_path.replace("/mnt/processed-gzf-train-flac/", f"processed-{key_words}/")
        object_name = file_path
    else:
        object_name = f"processed-{key_words}{file_path}"

    if convert_flac:
        object_name = object_name.replace(".wav", ".flac")
        object_name = object_name.replace(".mp3", ".flac")
        if not overwrite_outfile and bucket.object_exists(object_name):
            # logging.warning(f"{object_name} already exists, skip")
            return True, object_name
        # if data is not None:
        #     result = bucket.put_object(object_name, data)
    else:
        if not overwrite_outfile and bucket.object_exists(object_name):
            # logging.warning(f"{object_name} already exists, skip")
            return True, object_name
        # if data is not None:
        #     result = bucket.put_object(object_name, data)
    logging.warning(f"{object_name} already not exist")
    return False, object_name

def load_bytesio_audio(content, sr: int = 16000):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        "pipe:",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "pipe:",
    ]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=-1)
    out, _ = p.communicate(input=content)
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def convert_audio_to_flac(content, sr: int = 16000, is_path=False, raw_path=None):
    if is_path:
        input_path = content
    else:
        input_path = "pipe:"
    # 构建 FFmpeg 命令
    cmd = [
        "ffmpeg",
        "-nostdin",  # 禁用标准输入，避免与管道输入冲突
        "-threads",
        "1",  # 设置线程数为 1（关闭多线程）
        "-i",
        input_path,  # 从标准输入读取数据
        "-f",
        "flac",  # 输出格式为 FLAC
        "-ac",
        "1",  # 单声道输出
        "-ar",
        str(sr),  # 设置目标采样率
        "pipe:",  # 将处理后的数据输出到标准输出
    ]

    # 执行 FFmpeg 命令
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=-1)
    out, err = p.communicate(input=content) if not is_path else p.communicate()

    # 检查 FFmpeg 是否成功执行
    if p.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {err.decode('utf-8')}")

    # 解析 ffmpeg 日志中的音频时长
    duration = parse_ffmpeg_duration(err, raw_path=raw_path)
    if duration < 0:
        wav_samples = load_bytesio_audio(out, sr=sr)
        duration = len(wav_samples) / sr

    return out, duration


def valid_file_exist_oss(file_path: str, key_words: str):

    if file_path.startswith("/mnt/spider-data"):
        file_path = file_path.replace("/mnt/spider-data", f"processed-{key_words}/")
        object_name = file_path
    else:
        object_name = f"processed-{key_words}{file_path}"

    return bucket.object_exists(object_name)


def delete_file_oss(file_path: str):
    if file_path.startswith("/mnt/spider-data"):
        file_path = file_path.replace("/mnt/spider-data", "spider-data/aicheng")
        object_name = file_path
    elif file_path.startswith("/mnt/spider-resource/"):
        file_path = file_path.replace("/mnt/spider-resource/", "")
        object_name = file_path
    elif file_path.startswith("/mnt/spider-resource-nichongjia/"):
        file_path = file_path.replace("/mnt/spider-resource-nichongjia/", "")
        object_name = file_path
    elif file_path.startswith("/mnt/"):
        file_path = file_path.replace("/mnt/", "")
        object_name = file_path
    else:
        object_name = file_path[len("oss://" + bucket_name) :]
    bucket.delete_object(object_name)
    return


import re


def parse_ffmpeg_duration(ffmpeg_log, raw_path):
    """
    从 ffmpeg 的日志中解析音频时长。
    :param ffmpeg_log: ffmpeg 的标准错误输出日志（字节流）
    :return: 音频时长（秒）
    """
    try:
        # 匹配 Duration 字段（直接匹配字节模式）
        duration_match = re.search(
            rb"Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})", ffmpeg_log
        )
        if duration_match:
            hours = int(duration_match.group(1))
            minutes = int(duration_match.group(2))
            seconds = int(duration_match.group(3))
            milliseconds = int(duration_match.group(4)) * 10  # 转换为毫秒
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
            return total_seconds
        else:
            # print(f"{raw_path}: 无法从 ffmpeg 日志中解析音频时长")
            return -1
    except Exception as e:
        # print(f"{raw_path}, Error parsing audio duration: {e}")
        return -1

if __name__ == "__main__":
    import os
    from funasr.frontends.whisper_frontend import WhisperFrontend

    frontend = WhisperFrontend(
        fs=16000,
        n_mels=128,
        do_pad_trim=False,
        filters_path="/mnt/data/yunjie.jyj.cpfs.speech/FunASR/external_reliance/SenseVoiceSANM/assets/mel_filters.npz"
    )

    data_or_path_or_list = "oss://example-storage-4/spider-data/aicheng/source/20250715/youtube/audio/ptLauJajknI.webm"
    duration = [45.06, 46.12]
    buckets = {}
    endpoint = 'oss-cn-shanghai-internal.aliyuncs.com'
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    region = 'cn-shanghai'

    os.environ['OSS_ACCESS_KEY_ID'] = ''
    os.environ['OSS_ACCESS_KEY_SECRET'] = 'aDmOlzUt5Umq9K5fYLiiDfJFgsZilR'
    os.environ['OSS_SESSION_TOKEN'] = ''
    data_src = load_audio_text_image_video(data_or_path_or_list, duration=duration, buckets=buckets, endpoint=endpoint, auth=auth, region=region)
    print (f'data_src shape: {data_src.shape}')
    print (f'data_src: {data_src}')

    speech, speech_lengths = extract_fbank(
                            data_src,
                            data_type='sound',
                            frontend=frontend,
                            is_final=True,
                        )  # speech: [b, T, d]

    print (f'speech shape: {speech.shape}')
    print (f'speech: {speech}')
    print (f'speech_lengths: {speech_lengths}')
