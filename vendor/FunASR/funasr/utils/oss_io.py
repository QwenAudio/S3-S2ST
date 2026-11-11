import logging
import time

import torchaudio
import numpy as np
import os
import io

try:
    import oss2
    from oss2.credentials import EnvironmentVariableCredentialsProvider

    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    region = os.environ.get("OSS_REGION", None)
    endpoint = os.environ.get("OSS_ENDPOINT", None)

    session = oss2.http.Session(pool_size=500)

except ImportError:
    logging.warning("OSS not installed, please intall oss2.")

except AttributeError:
    logging.warning("Environment variable of OSS not set.")


# Per-call timeouts. requests accepts a (connect, read) tuple; with stream=True
# the read timeout becomes the per-chunk socket timeout for iter_content().
# Both can be overridden via env vars so users can tune without code changes.
OSS_CONNECT_TIMEOUT = float(os.environ.get("FUNASR_OSS_CONNECT_TIMEOUT", 2))
OSS_READ_TIMEOUT = float(os.environ.get("FUNASR_OSS_READ_TIMEOUT", 10))
OSS_MAX_RETRIES = int(os.environ.get("FUNASR_OSS_MAX_RETRIES", 3))
OSS_BASE_DELAY = float(os.environ.get("FUNASR_OSS_BASE_DELAY", 0.5))


def _oss_get_object_with_retry(
    bucket_name: str,
    object_name: str,
    max_retries: int = None,
    base_delay: float = None,
):
    """从 OSS 获取对象字节，带超时和指数退避重试。

    超时通过 ``(connect, read)`` 元组下传给 ``requests``，因此连接握手和分块读取
    都会在 socket 层硬截断，避免单次 GET 在数据加载中无限等待。
    """
    if max_retries is None:
        max_retries = OSS_MAX_RETRIES
    if base_delay is None:
        base_delay = OSS_BASE_DELAY

    timeout = (OSS_CONNECT_TIMEOUT, OSS_READ_TIMEOUT)
    bucket = oss2.Bucket(
        auth, endpoint, bucket_name,
        region=region, session=session,
        connect_timeout=timeout,
    )
    last_exc = None
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            object_stream = bucket.get_object(object_name)
            return object_stream.read()
        except Exception as e:
            last_exc = e
            elapsed = time.perf_counter() - t0
            delay = base_delay * (2 ** attempt)
            logging.warning(
                f"OSS get_object failed (attempt {attempt + 1}/{max_retries}, "
                f"elapsed {elapsed:.1f}s, timeout=({OSS_CONNECT_TIMEOUT},{OSS_READ_TIMEOUT})), "
                f"retrying in {delay:.1f}s. obj=oss://{bucket_name}/{object_name} err={e}"
            )
            time.sleep(delay)
    raise last_exc


def oss_remote_torchaudio_load(oss_path: str):
    bucket_name, object_name = oss_path[len("oss://"):].split("/", maxsplit=1)

    object_bytes = _oss_get_object_with_retry(bucket_name, object_name)
    object_io = io.BytesIO(object_bytes)

    return torchaudio.load(object_io)

def oss_remote_numpy_load(oss_path: str):
    bucket_name, object_name = oss_path[len("oss://"):].split("/", maxsplit=1)

    object_bytes = _oss_get_object_with_retry(bucket_name, object_name)
    object_io = io.BytesIO(object_bytes)

    return np.load(object_io)