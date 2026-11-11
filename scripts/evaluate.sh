#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/env.sh"
if [[ -z "${PYTHON_BIN:-}" && -x "$ROOT/envs/metrics312/bin/python" ]]; then
  PYTHON_BIN="$ROOT/envs/metrics312/bin/python"
fi
export LOCAL_PYTHON_BIN=${PYTHON_BIN:-python}
exec "${PYTHON_BIN:-python}" "$ROOT/tools/score.py" "$@"
