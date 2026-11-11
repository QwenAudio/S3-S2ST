#!/usr/bin/env python3


import concurrent.futures
import functools
import hashlib
import json
import logging
import os
import shutil
import subprocess
from typing import Iterable, Tuple, Dict, Any, List
import time

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import oss2.http
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_bucket(bucket_name, endpoint="https://oss-cn-wulanchabu-internal.aliyuncs.com", region="cn-wulanchabu"):
    # 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())

    s = oss2.http.Session(pool_size=200)

    # yourBucketName填写存储空间名称。
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region, session=s)

    return bucket


def download_file(bucket: oss2.Bucket, object_name: str, file_path: str):
    """
    下载文件到本地
    :param bucket: oss2.Bucket 实例
    :param object_name: OSS上的对象名称
    :param file_path: 本地保存路径
    :return: 下载是否成功
    """
    try:
        bucket.get_object_to_file(object_name, file_path)
        return True
    except oss2.exceptions.NoSuchKey:
        print(f"文件不存在: {object_name}")
        return None # so hacky
    except Exception as e:
        print(f"下载失败: {e}")
        return False


auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
endpoint = os.environ.get("OSS_ENDPOINT", None)
region = os.environ.get("OSS_REGION", None)
bucket_name = os.environ.get("OSS_BUCKET", None)
buckets = {}


def download_audio_to_local(source_wav: str, file_path: str):
    global buckets, auth, endpoint, region, bucket_name

    _bucket_name, object_name = source_wav[len("oss://"):].split("/", maxsplit=1)
    if bucket_name is not None:
        # override bucket name
        _bucket_name = bucket_name
    if _bucket_name not in buckets:
        buckets[_bucket_name] = oss2.Bucket(auth, endpoint, _bucket_name, region=region)
    if not buckets[_bucket_name].object_exists(object_name):
        raise RuntimeError(f"{object_name} not found")

    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Retry download up to 5 times with incremental backoff 1s,2s,3s,4s,5s
    for attempt in range(1, 6):
        ok = download_file(buckets[_bucket_name], object_name, file_path)
        if ok and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return True
        if attempt < 5:
            sleep_s = attempt
            logger.warning(f"download failed (attempt {attempt}/5), retry in {sleep_s}s: {object_name}")
            time.sleep(sleep_s)
    return False


def run_shell(command: str, cwd: str = None, env: dict = None) -> tuple[int, str, str]:
    """
    Run a shell command and return the output.

    :param command: The shell command to run.
    :param cwd: The working directory to run the command in.
    :param env: Environment variables to set for the command.
    :return: (returncode, stdout, stderr)
    """
    try:
        if isinstance(command, list):
            command = ' '.join(command)
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        stdout = e.stdout.strip() if e.stdout else ""
        stderr = e.stderr.strip() if e.stderr else ""
        return e.returncode, stdout, stderr


def ffmpeg_segment_audio(input_file: str, output_file: str,
                         start_time: float, end_time: float) -> bool:
    """
    使用ffmpeg切分音频

    Args:
        input_file: 输入音频文件路径
        output_file: 输出音频文件路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）

    Returns:
        切分是否成功
    """
    if not os.path.exists(input_file):
        return False

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", f"{input_file}",
        "-ss", f"{start_time}",
        "-to", f"{end_time}",
        f"{output_file}",
        "-y"
    ]

    # print(f"ffmpeg_segment_audio: {cmd}")

    retcode, _stdout, stderr = run_shell(cmd)
    if retcode != 0:
        logger.error(f"音频切分失败: {stderr}")
        return False

    return os.path.exists(output_file) and os.path.getsize(output_file) > 0


def ffmpeg_convert_to_wav(input_file: str, wav_file: str, sample_rate: int = 16000, channels: int = 1) -> bool:
    """Convert arbitrary audio to mono PCM WAV at target sample rate."""
    if not os.path.exists(input_file):
        return False

    os.makedirs(os.path.dirname(wav_file), exist_ok=True)

    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", f"{input_file}",
        "-ac", f"{channels}",
        "-ar", f"{sample_rate}",
        "-f", "wav",
        "-acodec", "pcm_s16le",
        f"{wav_file}",
        "-y"
    ]

    # print(f"ffmpeg_convert_to_wav: {cmd}")

    retcode, _stdout, stderr = run_shell(cmd)
    if retcode != 0:
        logger.error(f"音频转WAV失败: {stderr}")
        return False

    return os.path.exists(wav_file) and os.path.getsize(wav_file) > 0


def _sha256_hex(text: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()


def _sharded_dirs(base: str, digest: str) -> str:
    return os.path.join(base, digest[0:2], digest[2:4], digest[4:6], digest)


def _format_time_component(value: float) -> str:
    return f"{value:.4f}"


def _is_file_intact(path: str) -> bool:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        # check permission
        if os.access(path, os.R_OK):
            return True
        else:
            logger.warning(f"file {path} is not readable")
            return False
    return False


def _expected_output_paths(items: List[Dict[str, Any]], output_dir: str, slice_suffix: str, prefix: str) -> List[str]:
    expected = []
    for msg in items:
        try:
            start_time = float(msg.get("start_time", None))
            end_time = float(msg.get("end_time", None))
        except Exception:
            start_time = None
            end_time = None
        if start_time is None or end_time is None:
            continue
        ofn = f"{prefix}-{_format_time_component(start_time)}-{_format_time_component(end_time)}.{slice_suffix}"
        expected.append(os.path.join(output_dir, ofn))
    return expected


def _line_processor(line_json: Dict[str, Any], output_base: str, temp_base: str, slice_suffix: str, slice_workers: int, mode: str) -> Tuple[bool, str, str]:
    """
    Process a single JSONL line. Returns (ok, key, message)
    key is the sha256 digest key for the line's source object.
    """
    source: str = line_json.get("source")
    messages: List[Dict[str, Any]] = line_json.get("messages", [])
    if not source or not messages:
        return False, "", "missing source or messages"

    if not source.startswith("oss://"):
        return False, "", "source is not oss://"

    # object name part after oss://bucket_name/
    try:
        bucket_and_rest = source[len("oss://"):]
        bucket_name, object_name = bucket_and_rest.split("/", maxsplit=1)
    except Exception:
        return False, "", "malformed oss url"

    digest = _sha256_hex(object_name)
    temp_dir = _sharded_dirs(temp_base, digest)
    out_dir = _sharded_dirs(output_base, digest)

    os.makedirs(temp_dir, exist_ok=True)

    try:
        items = line_json.get("messages", []) if mode == "slices" else line_json.get("segments", [])
        prefix = "audio_slice" if mode == "slices" else "audio_segment"
        if not items:
            return False, "", f"no items for mode={mode}"
        # Idempotency check: all expected outputs exist and intact
        expected_outputs = _expected_output_paths(items, out_dir, slice_suffix, prefix)
        if expected_outputs and all(_is_file_intact(p) for p in expected_outputs):
            return True, digest, "skipped (already done)"

        # Download original audio into temp_dir with original suffix
        src_suffix = os.path.splitext(object_name)[1].lstrip('.') or 'webm'
        local_audio_path = os.path.join(temp_dir, f"audio.{src_suffix}")

        if not _is_file_intact(local_audio_path):
            try:
                download_audio_to_local(source, local_audio_path)
            except Exception as e:
                return False, digest, f"download failed: {e}"

        if not _is_file_intact(local_audio_path):
            return False, digest, "download produced empty file"

        # Convert to wav once per source and reuse for slicing
        local_wav_path = os.path.join(temp_dir, "audio_converted.wav")
        if not _is_file_intact(local_wav_path):
            ok = ffmpeg_convert_to_wav(local_audio_path, local_wav_path, sample_rate=16000, channels=1)
            if not ok:
                return False, digest, "wav conversion failed"
        
        os.makedirs(out_dir, exist_ok=True)

        # Slice with ffmpeg into temp, then move to output using thread pool
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _process_one(start_time: float, end_time: float) -> Tuple[bool, str]:
            out_name = f"{prefix}-{_format_time_component(start_time)}-{_format_time_component(end_time)}.{slice_suffix}"
            final_out_path = os.path.join(out_dir, out_name)
            if _is_file_intact(final_out_path):
                return True, out_name
            temp_out_path = os.path.join(temp_dir, out_name)
            if not _is_file_intact(temp_out_path):
                ok_local = ffmpeg_segment_audio(local_wav_path, temp_out_path, start_time, end_time)
                if not ok_local:
                    return False, out_name
            os.makedirs(os.path.dirname(final_out_path), exist_ok=True)
            try:
                if os.path.exists(final_out_path):
                    os.remove(final_out_path)
                shutil.move(temp_out_path, final_out_path)
            except Exception:
                return False, out_name
            return True, out_name

        futures = []
        with ThreadPoolExecutor(max_workers=max(1, slice_workers)) as pool:
            for msg in items:
                try:
                    start_time = float(msg.get("start_time", None))
                    end_time = float(msg.get("end_time", None))
                except Exception:
                    start_time = None
                    end_time = None
                if start_time is None or end_time is None:
                    continue
                futures.append(pool.submit(_process_one, start_time, end_time))

            for fut in as_completed(futures):
                ok_one, name_one = fut.result()
                if not ok_one:
                    return False, digest, f"ffmpeg slice failed for {name_one}"

        return True, digest, "ok"
    finally:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        pass


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skip invalid json line")


def process_jsonl(jsonl_path: str, output_base: str, temp_base: str,
                  num_workers: int = 8, slice_suffix: str = 'wav', slice_workers: int = 10, mode: str = 'slices') -> Dict[str, int]:
    """
    Process a JSONL file: download OSS audio and extract message slices.

    - Idempotent: skips lines if all slices already exist in output.
    - Uses a process pool with num_workers.

    Returns counters dict.
    """
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(temp_base, exist_ok=True)

    lines = list(_read_jsonl(jsonl_path))
    total = len(lines)
    logger.info(f"Loaded {total} lines from {jsonl_path}")

    counters = {"total": total, "ok": 0, "skipped": 0, "failed": 0}

    worker = functools.partial(_line_processor, output_base=output_base, temp_base=temp_base, slice_suffix=slice_suffix, slice_workers=slice_workers, mode=mode)

    # Use ProcessPoolExecutor with imap-like behavior via map
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, num_workers)) as executor:
        with tqdm(total=total, desc="Processing", dynamic_ncols=True) as pbar:
            for ok, key, msg in executor.map(worker, lines, chunksize=16 if total > 256 else 1):
                if ok and msg.startswith("skipped"):
                    counters["skipped"] += 1
                elif ok:
                    counters["ok"] += 1
                else:
                    counters["failed"] += 1
                    if key:
                        logger.error(f"{key}: {msg}")
                    else:
                        logger.error(f"{msg}")
                pbar.update(1)
                pbar.set_postfix(ok=counters["ok"], skipped=counters["skipped"], failed=counters["failed"], refresh=False)

    logger.info(f"Done. ok={counters['ok']} skipped={counters['skipped']} failed={counters['failed']}")
    return counters


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process JSONL to slice audios from OSS")
    parser.add_argument("jsonl", type=str, help="Path to input JSONL file")
    parser.add_argument("output_base", type=str, help="Output base directory for final slices")
    parser.add_argument("temp_base", type=str, help="Temporary base directory for downloads and intermediate slices")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of process workers")
    parser.add_argument("--slice-suffix", type=str, default="wav", choices=["wav", "flac"], help="Slice audio format")
    parser.add_argument("--slice-workers", type=int, default=10, help="Number of threads for per-line slicing")
    parser.add_argument("--mode", type=str, default="segments", choices=["slices", "segments"], help="Whether to cut from messages or segments")

    args = parser.parse_args()

    process_jsonl(args.jsonl, args.output_base, args.temp_base, num_workers=args.num_workers, slice_suffix=args.slice_suffix, slice_workers=args.slice_workers, mode=args.mode)
