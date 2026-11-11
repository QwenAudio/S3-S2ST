#!/usr/bin/env python3
"""Execute the proven offline entrypoint without modifying its source tree."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


offline_root = Path(
    os.environ.get(
        "STREAMING_OFFLINE_CODE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
).resolve()
entrypoint = offline_root / "pretrain_fun_base.py"
if not entrypoint.is_file():
    raise FileNotFoundError(entrypoint)

compat_root = Path(__file__).resolve().parent
sys.path.insert(0, str(compat_root))
sys.path.insert(0, str(offline_root))
runpy.run_path(str(entrypoint), run_name="__main__")
