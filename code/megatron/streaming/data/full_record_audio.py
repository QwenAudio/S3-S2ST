"""Streaming-only full-record audio loading for packed S2TT.

Each interleaved source record references one contiguous interval of one audio
object.  The offline Mega loader normally seeks every turn independently.  We
keep its token/label construction unchanged, but serve those turn reads from a
single full-record waveform and replace the per-turn fbanks with one full
fbank per source record.  The model-side adapter splits encoder output later.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from multiprocessing.pool import ThreadPool
from typing import Iterable

import numpy as np


# Keep the runtime slice validation aligned with the packed-data audio audit.
# That audit accepts a trailing shortfall of at most 80 ms.  FFmpeg/codec
# duration rounding can make the decoded waveform slightly shorter than the
# metadata interval, so rejecting at the previous fixed 32-sample threshold
# silently replaced otherwise-audited 9pair bins during dataloader retries.
MAX_TAIL_SHORTFALL_SECONDS = 0.08


def _install_streaming_io_timeouts() -> None:
    """Relax the offline short-segment I/O limits for full-record reads.

    The shared offline loader gives FFmpeg 10 seconds and OSS connections one
    second. That is normally enough for a short turn, but not for a 10--30s
    contiguous record when OSS is congested. Keep this process-local so the
    shared FunASR source and offline jobs remain unchanged.
    """
    import funasr.utils.load_utils as load_utils

    ffmpeg_timeout = float(os.environ.get("STREAMING_FFMPEG_TIMEOUT_SEC", "60"))
    oss_timeout = float(os.environ.get("STREAMING_OSS_CONNECT_TIMEOUT_SEC", "10"))
    if getattr(load_utils.load_audio_ffmpeg, "_streaming_full_record_timeout", False):
        return

    def load_audio_ffmpeg_streaming(
        wav_path, sr: int = 16000, offset=0, duration=None, speed=1
    ):
        wav_path = load_utils.PATH_TRANSFORMER.transform(wav_path)
        cmd = ["ffmpeg", "-nostdin", "-threads", "1"]
        if offset > 0:
            cmd += ["-ss", str(offset)]
        cmd += ["-i", wav_path, "-map", "0:a:0"]
        if duration is not None:
            cmd += ["-t", str(duration + 1)]
        if speed != 1:
            if not 0.5 <= speed <= 2.0:
                raise ValueError(
                    f"atempo only supports speed in [0.5, 2.0], got {speed}"
                )
            cmd += ["-af", f"atempo={speed}"]
        cmd += [
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"
        ]
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            try:
                audio_bytes, stderr = process.communicate(timeout=ffmpeg_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise TimeoutError(
                    f"FFmpeg timeout after {ffmpeg_timeout:g}s"
                ) from None
        stderr_text = stderr.decode("utf-8", errors="replace")
        if not audio_bytes:
            raise RuntimeError(
                "FFmpeg produced empty audio. "
                f"offset={offset}, duration={duration}. stderr:\n{stderr_text}"
            )
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        if duration is not None:
            audio_array = audio_array[: int(duration * sr / speed)]
        return audio_array, sr

    load_audio_ffmpeg_streaming._streaming_full_record_timeout = True
    load_utils.load_audio_ffmpeg = load_audio_ffmpeg_streaming
    load_utils.oss_connect_timeout = oss_timeout
    for bucket in getattr(load_utils, "buckets", {}).values():
        if bucket is not None and hasattr(bucket, "connect_timeout"):
            bucket.connect_timeout = oss_timeout
    print(
        "[STREAMING DATA] I/O limits: "
        f"ffmpeg_timeout={ffmpeg_timeout:g}s oss_connect_timeout={oss_timeout:g}s",
        flush=True,
    )


def _replace_audio_path(path: str, use_oss_cache: bool) -> str:
    path = path.replace(
        "oss://example-storage-2/",
        "oss://example-storage-4/",
    )
    path = path.replace(
        "oss://example-storage-6/",
        "oss://example-storage-4/",
    )
    path = path.replace(
        "oss://example-storage-5/",
        "oss://example-storage-1/",
    )
    path = path.replace(
        "oss://example-storage-3/",
        "oss://example-storage-1/",
    )
    if use_oss_cache:
        path = path.replace(
            "oss://example-storage-4", "/nls_spider_sh_cache", 1
        )
        path = path.replace(
            "oss://example-storage-1", "/fun_resource_sh_cache", 1
        )
    return path


def _fake_token_length(fbank_frames: int) -> int:
    length = 1 + (int(fbank_frames) - 3 + 2) // 2
    length = 1 + (length - 3 + 2) // 2
    return (length - 1) // 2 + 1


def _group_specs(packed_items, dataset_class, pattern, use_oss_cache):
    groups = []
    for item_index, item in enumerate(packed_items):
        specs = dataset_class._collect_audio_specs_from_packed_items([item], pattern)
        if not specs:
            raise ValueError(f"streaming source item {item_index} has no audio")
        paths = {_replace_audio_path(path, use_oss_cache) for path, _ in specs}
        if len(paths) != 1:
            raise ValueError(
                f"streaming source item {item_index} references {len(paths)} audio objects"
            )
        intervals = [(float(duration[0]), float(duration[1])) for _, duration in specs]
        for previous, current in zip(intervals, intervals[1:]):
            if abs(previous[1] - current[0]) > 2e-4:
                raise ValueError(
                    f"streaming source item {item_index} has noncontiguous audio: "
                    f"{previous} -> {current}"
                )
        groups.append(
            {
                "item_index": item_index,
                "path": next(iter(paths)),
                "intervals": intervals,
                "full_interval": [intervals[0][0], intervals[-1][1]],
                "expected_tokens": sum(
                    int(duration_item["speech_length"])
                    for duration_item in item["user"]
                    if duration_item is not None
                    and "<|startofspeech|>" in str(duration_item.get("text", ""))
                ),
            }
        )
    return groups


def _duration_key(path: str, duration: Iterable[float]):
    start, end = duration
    return path, round(float(start), 6), round(float(end), 6)


def _truncated_audio_feature(
    feature_beg, feature_type, fake_token_len, max_token_length: int
):
    """Return the first source-audio feature cut by fixed-length collation."""
    if max_token_length <= 0:
        return None
    for begin_value, type_value, length_value in zip(
        feature_beg, feature_type, fake_token_len
    ):
        begin = int(begin_value)
        feature_kind = int(type_value)
        length = int(length_value)
        if feature_kind != 1 or begin <= 0 or length <= 0:
            continue
        if begin + length > max_token_length:
            return begin, length
    return None


def install_full_record_audio_loader() -> None:
    """Install a process-local adapter; the external FunASR tree stays read-only."""
    import funasr.datasets.minmo_datasets.datasets as dataset_module

    _install_streaming_io_timeouts()

    dataset_class = dataset_module.MinMo_S2S_Channels_SFT_Pack_Mega
    original_prepare = dataset_class._prepare_single_data
    if getattr(original_prepare, "_streaming_full_record_audio", False):
        return
    original_load = dataset_module.load_audio_text_image_video

    def prepare_full_record(packed_items, *args, **kwargs):
        # These are positional in the offline static method, but production
        # calls use keywords.  Fail closed if a future caller changes that.
        pattern = kwargs.get("pattern")
        frontend = kwargs.get("frontend")
        fs = int(kwargs.get("fs", 16000))
        data_type = kwargs.get("data_type", "sound")
        permute = bool(kwargs.get("permute", False))
        use_oss_cache = bool(kwargs.get("use_oss_cache", False))
        load_threads = int(kwargs.get("load_audio_num_threads", 4))
        if pattern is None or frontend is None:
            raise RuntimeError("full-record streaming loader requires pattern and frontend")
        try:
            groups = _group_specs(
                packed_items, dataset_class, pattern, use_oss_cache
            )
        except Exception as exc:
            logging.warning("full-record audio layout rejected: %s", exc)
            return None

        def load_group(group):
            attempts = max(1, int(os.environ.get("STREAMING_AUDIO_READ_ATTEMPTS", "2")))
            error = None
            for attempt in range(attempts):
                try:
                    waveform = original_load(
                        group["path"],
                        fs=fs,
                        duration=group["full_interval"],
                        use_ffmpeg=True,
                    )
                    return waveform, None
                except Exception as exc:
                    error = exc
                    if attempt + 1 < attempts:
                        time.sleep(0.25 * (attempt + 1))
            return None, error

        with ThreadPool(max(1, min(load_threads, len(groups)))) as pool:
            loaded = list(pool.map(load_group, groups))
        if any(waveform is None for waveform, _ in loaded):
            for group, (waveform, error) in zip(groups, loaded):
                if waveform is None:
                    logging.warning(
                        "full-record audio read failed; drop packed bin: path=%s "
                        "duration=%s error=%s",
                        group["path"],
                        group["full_interval"],
                        error,
                    )
            return None

        full_fbanks = []
        full_fbank_lengths = []
        slice_cache = {}
        for group, (waveform, _error) in zip(groups, loaded):
            full_start, _full_end = group["full_interval"]
            for start, end in group["intervals"]:
                begin_sample = round((start - full_start) * fs)
                sample_count = int((end - start) * fs)
                slice_end = begin_sample + sample_count
                available_samples = int(waveform.shape[-1])
                # Match the packed-data audit's trailing-shortfall tolerance.
                # The offline loader accepts the available waveform as-is, so
                # clamp the slice end after validating against the same 80-ms
                # standard used when the data was produced.
                tail_shortfall = max(0, slice_end - available_samples)
                max_tail_shortfall = round(MAX_TAIL_SHORTFALL_SECONDS * fs)
                if (
                    begin_sample < 0
                    or begin_sample >= available_samples
                    or tail_shortfall > max_tail_shortfall
                ):
                    logging.warning(
                        "full-record slice out of bounds; drop packed bin: "
                        "path=%s interval=%s full=%s samples=%s shortfall=%d",
                        group["path"],
                        [start, end],
                        group["full_interval"],
                        tuple(waveform.shape),
                        tail_shortfall,
                    )
                    return None
                slice_end = min(slice_end, available_samples)
                slice_cache[_duration_key(group["path"], [start, end])] = waveform[
                    begin_sample:slice_end
                ]

            speech, speech_lengths = dataset_module.extract_fbank(
                waveform,
                data_type=data_type,
                frontend=frontend,
                is_final=True,
            )
            if permute:
                speech = speech.permute(0, 2, 1)
            actual_tokens = _fake_token_length(int(speech_lengths[0]))
            if abs(actual_tokens - group["expected_tokens"]) > 2:
                logging.warning(
                    "full-record token length mismatch; drop packed bin: path=%s "
                    "expected=%d actual=%d interval=%s",
                    group["path"],
                    group["expected_tokens"],
                    actual_tokens,
                    group["full_interval"],
                )
                return None
            full_fbanks.append(speech[0])
            full_fbank_lengths.append(speech_lengths)

        def cached_slice_loader(path, *load_args, **load_kwargs):
            duration = load_kwargs.get("duration", (0, None))
            if duration[1] is not None:
                cached = slice_cache.get(_duration_key(str(path), duration))
                if cached is not None:
                    return cached
            return original_load(path, *load_args, **load_kwargs)

        # The original method still builds exactly the proven token/label and
        # feature layout.  Its per-turn audio requests are now cheap views of
        # the one full-record waveform loaded above.
        previous_load = dataset_module.load_audio_text_image_video
        dataset_module.load_audio_text_image_video = cached_slice_loader
        try:
            output = original_prepare(packed_items, *args, **kwargs)
        finally:
            dataset_module.load_audio_text_image_video = previous_load
        if output is None:
            return None
        if os.environ.get("STREAMING_FULL_RECORD_REJECT_TRUNCATED_AUDIO", "0") == "1":
            truncated = _truncated_audio_feature(
                output["feature_beg"],
                output["feature_type"],
                output["fake_token_len"],
                int(kwargs.get("max_token_length", -1)),
            )
            if truncated is not None:
                begin, length = truncated
                logging.warning(
                    "full-record audio feature crosses fixed sequence boundary; "
                    "drop packed bin: begin=%d length=%d limit=%d",
                    begin,
                    length,
                    int(kwargs.get("max_token_length", -1)),
                )
                return None
        source_count = int(output["cu_seqlens"].numel()) - 1
        feature_count = int((output["feature_type"] == 1).sum())
        allow_trailing_pack_drop = os.environ.get(
            "STREAMING_S2ST_TALKER_ONLY_SEGMENT_PADDING", "0"
        ).lower() in ("1", "true", "yes", "on")
        retained_indices = output.get("retained_item_indices")
        if retained_indices is not None:
            if (len(retained_indices) != source_count
                or retained_indices != sorted(set(retained_indices))
                or any(index < 0 or index >= len(groups) for index in retained_indices)):
                logging.warning("invalid retained item mapping; drop packed bin")
                return None
            retained_groups = [groups[index] for index in retained_indices]
        else:
            retained_indices = list(range(source_count))
            retained_groups = (
                groups[:source_count] if allow_trailing_pack_drop else groups
            )
        expected_feature_count = sum(
            len(group["intervals"]) for group in retained_groups
        )
        if (
            source_count != len(retained_groups)
            or feature_count != expected_feature_count
        ):
            logging.warning(
                "full-record packing conservation failed; drop packed bin: "
                "sources=%d/%d features=%d/%d",
                source_count,
                len(retained_groups),
                feature_count,
                expected_feature_count,
            )
            return None
        if source_count < len(groups):
            logging.warning(
                "full-record mode retained complete source items: sources=%d/%d",
                source_count,
                len(groups),
            )
        output["speech"] = [full_fbanks[index] for index in retained_indices]
        output["speech_lengths"] = [full_fbank_lengths[index] for index in retained_indices]
        output["speech_info"] = [
            {
                "path": group["path"],
                "full_interval": group["full_interval"],
                "turns": len(group["intervals"]),
            }
            for group in retained_groups
        ]
        return output

    prepare_full_record._streaming_full_record_audio = True
    dataset_class._prepare_single_data = staticmethod(prepare_full_record)
    print(
        "[STREAMING DATA] full-record audio mode: one OSS read and one fbank "
        "per interleaved source record",
        flush=True,
    )
