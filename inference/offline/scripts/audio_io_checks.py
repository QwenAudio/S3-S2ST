"""Audio decoding checks and bounded recovery for training data reads.
"""
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from urllib.parse import urlsplit, unquote

import numpy as np

_lock = threading.Lock()
_local = threading.local()
_counts = {}
_strict_sources = set()


def source_key(path):
    if path.startswith(('http://', 'https://')):
        url = urlsplit(path)
        return 'oss://' + url.hostname.split('.')[0] + '/' + unquote(url.path.lstrip('/'))
    return path


def redact(value):
    if isinstance(value, str):
        return re.sub(r"(https?://[^\s?'\"<>]+)\?[^\s'\"<>]*", r"\1?[REDACTED]", value)
    if isinstance(value, dict):
        return {str(k): redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def diag_root():
    root = Path(os.environ['EXP_DIR']).resolve() / 'diagnostics' / 'audio_reads'
    root.mkdir(parents=True, exist_ok=True)
    return root


def event(kind, **fields):
    # Per-process files avoid interleaved writes from ranks/workers; bounded size.
    row = redact(dict(time=time.time(), host=socket.gethostname(),
                      pid=os.getpid(), rank=os.environ.get('RANK'), kind=kind, **fields))
    with _lock:
        p = diag_root() / f'events.rank{os.environ.get("RANK", "na")}.pid{os.getpid()}.jsonl'
        if not p.exists() or p.stat().st_size < 64 * 1024 * 1024:
            with p.open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
    return row


def retain_pcm(data):
    # At most 8 x 2 MiB per process, enough to inspect truncation without filling CPFS.
    if not data:
        return None
    with _lock:
        n = _counts.get(('pcm', os.getpid()), 0)
        if n >= 8:
            return None
        _counts['pcm', os.getpid()] = n + 1
        path = diag_root() / f'failed.rank{os.environ.get("RANK", "na")}.pid{os.getpid()}.{n}.f32le'
        path.write_bytes(data[:2 * 1024 * 1024])
        return str(path)


def decode_checked(wav_path, sr=16000, offset=0, duration=None, speed=1):
    import funasr.utils.load_utils as lu
    source = lu.PATH_TRANSFORMER.transform(wav_path)
    # Keep the proven offline decoding command, including +1s then PCM trim.
    cmd = ['ffmpeg', '-nostdin', '-threads', '1']
    if offset > 0:
        cmd += ['-ss', str(offset)]
    cmd += ['-i', source, '-map', '0:a:0']
    if duration is not None:
        cmd += ['-t', str(duration + 1)]
    if speed != 1:
        if not .5 <= speed <= 2:
            raise ValueError('atempo speed outside [0.5, 2]')
        cmd += ['-af', f'atempo={speed}']
    cmd += ['-f', 'f32le', '-acodec', 'pcm_f32le', '-ac', '1', '-ar', str(sr), '-']
    timeout = float(os.environ.get('STREAMING_FFMPEG_TIMEOUT_SEC', '60'))
    t = time.monotonic()
    timed_out = False
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        try:
            raw, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            raw, err = p.communicate()
            timed_out = True
    expected = int(duration * sr / speed) if duration is not None else None
    count = len(raw) // 4
    reasons = []
    if timed_out:
        reasons.append('timeout')
    if p.returncode != 0:
        reasons.append('nonzero_exit')
    if not raw:
        reasons.append('empty_pcm')
    if len(raw) % 4:
        reasons.append('incomplete_float')
    stderr = err.decode('utf-8', errors='replace')
    if expected is not None and count < expected - 32:
        # Source waveforms still use the original 32-sample strict limit.
        # Target codec supervision rounds its final interval to an 80ms grid.
        # Accept <=80ms only at the declared physical EOF, never in mid-file,
        # never for a source, and never with a transport/nonzero-exit error.
        match = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', stderr)
        eof = (int(match[1])*3600 + int(match[2])*60 + float(match[3])) if match else None
        eof_tail = (source_key(wav_path) not in _strict_sources and eof is not None
                    and offset + duration >= eof
                    and expected - count <= int(.08 * sr / speed) + 1
                    and abs(count - max(0, eof-offset)*sr/speed) <= int(.011 * sr / speed) + 1)
        if not eof_tail:
            reasons.append('short_pcm')
    # Some demuxers exit 0 even after an HTTP/partial-file error.
    if re.search(r'HTTP error|Server returned [45]\d\d|partial file|Input/output error|Error during demuxing', stderr, re.I):
        reasons.append('transport_error')
    if reasons:
        artifact = retain_pcm(raw)
        event('audio_read_rejected', source=source, offset=offset, duration=duration,
              sr=sr, speed=speed, expected_samples=expected, actual_samples=count,
              returncode=p.returncode, elapsed=time.monotonic()-t, reasons=reasons,
              stderr=stderr[-16000:], command=cmd, pcm_prefix=artifact)
        raise RuntimeError(f'AUDIO_IO {",".join(reasons)} samples={count}/{expected}; '
                           'see experiment diagnostics/audio_reads')
    audio = np.frombuffer(raw, dtype=np.float32)
    if expected is not None:
        audio = audio[:expected]
    return audio, sr


def install_io():
    import funasr.utils.load_utils as lu
    lu.load_audio_ffmpeg = decode_checked
    lu.oss_connect_timeout = float(os.environ.get('STREAMING_OSS_CONNECT_TIMEOUT_SEC', '10'))
    for bucket in getattr(lu, 'buckets', {}).values():
        if bucket is not None and hasattr(bucket, 'connect_timeout'):
            bucket.connect_timeout = lu.oss_connect_timeout


class CaptureWarnings(logging.Handler):
    def emit(self, record):
        context = getattr(_local, 'pack', None)
        if context is not None and len(context['warnings']) < 24:
            message = record.getMessage()
            if any(x in message for x in ('full-record', 'len mismatch', 'skip codec', 'invalid retained')):
                context['warnings'].append(redact(message.split('item:')[0][:3000]))


def save_pack(items, context, reason):
    payload = json.dumps(redact(items), ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    artifact = None
    with _lock:
        n = _counts.get(('pack', os.getpid()), 0)
        if n < 128:
            _counts['pack', os.getpid()] = n + 1
            path = diag_root() / f'pack.pid{os.getpid()}.{digest}.json'
            if not path.exists():
                path.write_text(payload)
            artifact = str(path)
    event('pack_rejected', sha256=digest, artifact=artifact,
          dataset_index=getattr(_local, 'dataset_index', None),
          reason=reason, warnings=context['warnings'])


def valid_sample(sample):
    return isinstance(sample, dict) and 'input_ids' in sample and sample['input_ids'].numel() > 0


def install_dataset_guards(cls):
    if getattr(cls, '_audio_reads', False):
        return
    prepare = cls._prepare_single_data
    getitem = cls.__getitem__
    collate = cls.collator

    @functools.wraps(prepare)
    def guarded_prepare(packed_items, *args, **kwargs):
        previous = getattr(_local, 'pack', None)
        context = {'warnings': []}
        _local.pack = context
        _strict_sources.clear()
        try:
            result = prepare(packed_items, *args, **kwargs)
            if not valid_sample(result):
                save_pack(packed_items, context, 'prepare_returned_empty')
            return result
        except Exception as exc:
            save_pack(packed_items, context, f'{type(exc).__name__}: {exc}')
            raise
        finally:
            _local.pack = previous

    @functools.wraps(getitem)
    def guarded_getitem(self, index):
        _local.dataset_index = int(index)
        return getitem(self, index)

    @functools.wraps(collate)
    def guarded_collator(self, samples=None):
        import torch
        samples = list(samples or [])
        if not samples:
            event('empty_collator_input')
            raise RuntimeError('AUDIO_IO empty collator input')
        for slot, sample in enumerate(samples):
            if valid_sample(sample):
                continue
            event('batch_replacement_begin', slot=slot)
            recovered = False
            t = time.monotonic()
            for attempt in range(4):
                if time.monotonic() - t > 300:
                    break
                index = torch.randint(0, len(self), ()).item()
                replacement = self[index]  # Existing dataset retry and validation.
                if valid_sample(replacement):
                    samples[slot] = replacement
                    event('batch_replacement_success', slot=slot, index=index, attempt=attempt+1)
                    recovered = True
                    break
            if not recovered:
                event('batch_replacement_exhausted', slot=slot)
                raise RuntimeError('AUDIO_IO exhausted valid-pack recovery; see diagnostics')
        result = collate(self, samples)
        if not valid_sample(result):
            event('invalid_collator_output', keys=list(result or {}))
            raise RuntimeError('AUDIO_IO collator returned invalid batch')
        return result

    guarded_prepare._streaming_full_record_audio = True
    cls._prepare_single_data = staticmethod(guarded_prepare)
    cls.__getitem__ = guarded_getitem
    cls.collator = guarded_collator
    cls._audio_reads = True
    handler = CaptureWarnings(level=logging.WARNING)
    logging.getLogger().addHandler(handler)
    event('dataset_guard_installed', dataset=cls.__name__)


def install():
    from streaming.data import full_record_audio as full
    if getattr(full, '_audio_reads', False):
        return
    # Hook before baseline imports streaming, which installs this loader.
    original = full.install_full_record_audio_loader
    original_groups = full._group_specs
    def capture_source_groups(*args, **kwargs):
        groups = original_groups(*args, **kwargs)
        _strict_sources.update(source_key(g['path']) for g in groups)
        return groups
    full._group_specs = capture_source_groups
    full._install_streaming_io_timeouts = install_io

    def guarded_install():
        original()
        import funasr.datasets.minmo_datasets.datasets as dm
        install_dataset_guards(dm.MinMo_S2S_Channels_SFT_Pack_Mega)

    full.install_full_record_audio_loader = guarded_install
    full._audio_reads = True
    event('defense_entry_installed')
    print('[AUDIO_IO] checked audio decoding and batch recovery', flush=True)
