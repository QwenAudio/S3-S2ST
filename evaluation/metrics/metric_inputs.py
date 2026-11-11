"""Strict keyed metric inputs: empty predictions are data, missing IDs are errors."""
from pathlib import Path

def read_keyed(path):
    rows = {}
    for lineno, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split('\t', 1) if '\t' in line else line.split(maxsplit=1)
        key = parts[0]
        if not key or key in rows:
            raise ValueError(f'{path}:{lineno}: empty or duplicate key {key!r}')
        rows[key] = parts[1] if len(parts) == 2 else ''
    return rows

def read_asr_tsv(path):
    refs, hyps = {}, {}
    for lineno, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 4:
            raise ValueError(f'{path}:{lineno}: malformed ASR TSV')
        key = Path(parts[0]).stem
        if not key or key in hyps:
            raise ValueError(f'{path}:{lineno}: empty or duplicate ASR key {key!r}')
        refs[key], hyps[key] = parts[2], parts[3]
    return refs, hyps

def full_keys(reference, hypothesis):
    if not reference:
        raise ValueError('No reference samples')
    if set(reference) != set(hypothesis):
        missing = sorted(set(reference)-set(hypothesis))
        extra = sorted(set(hypothesis)-set(reference))
        raise ValueError(f'Metric coverage mismatch: missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}')
    return list(reference)
