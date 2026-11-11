#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export S3_ROOT="$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/vendor:$ROOT/code/megatron:$ROOT/vendor/FunASR:$ROOT/vendor/offline_funasr:${PYTHONPATH:-}"
exec "${PYTHON_BIN:-python}" "$ROOT/tools/train.py" "$@"
