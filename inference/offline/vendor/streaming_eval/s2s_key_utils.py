"""Stable, filesystem-safe utterance keys for S2S inference outputs."""

import hashlib
import re


def make_output_key(original_key: str, basename_limit: int = 180) -> str:
    """Return a readable key whose use as a codec/wav basename is always safe."""
    original_key = original_key.replace(" ", "")
    basename = original_key.rsplit("/", 1)[-1]
    basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename).strip("._")
    if not basename:
        basename = "audio"
    digest = hashlib.sha1(original_key.encode("utf-8")).hexdigest()[:16]
    return f"{basename[:basename_limit]}_{digest}"
