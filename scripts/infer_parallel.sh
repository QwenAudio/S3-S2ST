#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/env.sh"
if [[ -z "${PYTHON_BIN:-}" && -x "$ROOT/envs/inference312/bin/python" ]]; then
  PYTHON_BIN="$ROOT/envs/inference312/bin/python"
fi
exec "${PYTHON_BIN:-python}" "$ROOT/tools/infer_parallel.py" "$@"
