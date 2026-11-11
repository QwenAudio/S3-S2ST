#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$root/scripts/oss_env.sh"
base=/path/to/Qwen-public-0803-hejia
export PYTHONPATH="${EXP_DIR}/runtime:${S3_ROOT}/training/offline/config/code_snapshot:${S3_ROOT}/training/offline_base/config/code_snapshot:${S3_ROOT}/vendor/offline_funasr:${S3_ROOT}/vendor/FunASR:${S3_ROOT}/code/megatron:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 NVTE_FLASH_ATTN=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMNIDRCA2_TALKER_ADD_INPUT_EMBEDDING=0
export TMPDIR="$EXP_DIR/tmp"
export TRITON_CACHE_DIR="$root/cache/triton" VLLM_CACHE_ROOT="$root/cache/vllm"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
ulimit -Sn 65536
exec /usr/local/bin/python "$root/scripts/infer_offline.py" "$@"
