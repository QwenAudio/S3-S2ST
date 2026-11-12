#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
stage=${1:?Usage: train.sh talker|e2e [exact iter directory]}
source_iter=${2:-$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["reference_checkpoint"])' "$root/audit/origin.json")}
[[ "$stage" == talker || "$stage" == e2e ]] || exit 2
[[ "$(basename "$source_iter")" =~ ^iter_[0-9]{7}$ ]] || { echo 'Exact iteration required'; exit 2; }
[[ -f "$source_iter/.metadata" || -f "$source_iter/metadata.json" ]] || exit 2
base_init=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["reference_checkpoint"])' "$root/audit/origin.json")
case "$source_iter" in
 "$base_init"|"$root"/runs/*/iter_*) ;;
 *) echo 'Refusing checkpoint outside original base or this fresh experiment' >&2; exit 2 ;;
esac
export EXP_DIR=${EXP_DIR:-$root/runs/$stage}
export RESUME_TRAINING=${RESUME_TRAINING:-0}
mkdir -p "$EXP_DIR"
exec 9>"$EXP_DIR/launch.lock"
flock -n 9 || { echo 'Experiment is already running'; exit 2; }
if [[ "$RESUME_TRAINING" == 1 ]]; then
 export INIT_CKPT_DIR="$EXP_DIR"
 [[ -f "$EXP_DIR/latest_checkpointed_iteration.txt" ]] || exit 2
else
 [[ ! -f "$EXP_DIR/latest_checkpointed_iteration.txt" ]] || { echo 'Use RESUME_TRAINING=1 for an existing checkpoint'; exit 2; }
 base_init=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["reference_checkpoint"])' "$root/audit/origin.json")
 if [[ "$source_iter" == "$base_init" ]]; then
  export OFFLINE_FRESH_BASE_INIT=1
 else
  export OFFLINE_FRESH_BASE_INIT=0
 fi
 export INIT_CKPT_DIR="$EXP_DIR/init_view"
 mkdir -p "$INIT_CKPT_DIR"
 ln -sfn "$source_iter" "$INIT_CKPT_DIR/$(basename "$source_iter")"
 iter=$(basename "$source_iter"); printf '%d\n' "$((10#${iter#iter_}))" > "$INIT_CKPT_DIR/latest_checkpointed_iteration.txt"
fi
export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH="${EXP_DIR}/runtime:${S3_ROOT}/training/offline/config/code_snapshot:${S3_ROOT}/training/offline_base/config/code_snapshot:${S3_ROOT}/vendor/offline_funasr:${S3_ROOT}/vendor/FunASR:${S3_ROOT}/code/megatron:${PYTHONPATH:-}"
export TRAIN_SCRIPT="$root/config/pretrain_fun_base_talkerpackiso.py"
export DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-$root/config/dataset.yaml}
export S2ST_TRAIN_STAGE=tts TEXT_LOSS_WEIGHT=0.0
export LR=${LR:-5e-6} S2ST_BRIDGE_LR_SCALE=20.0
if [[ "$stage" == e2e ]]; then
 export S2ST_TRAIN_STAGE=s2tt TEXT_LOSS_WEIGHT=1.0
 export LR=${E2E_LR:-1e-6} S2ST_BRIDGE_LR_SCALE=1.0
 export RECOMPUTE_ARGS='--recompute-granularity full --recompute-method uniform --recompute-num-layers 1'
fi
export TRAIN_ITERS=${TRAIN_ITERS:-10000} SAVE_INTERVAL=${SAVE_INTERVAL:-500}
export LR_DECAY_ITERS=${LR_DECAY_ITERS:-10000} LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-100}
export MIN_LR=${MIN_LR:-1e-6}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8} TP_SIZE=1 GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64} MICRO_BATCH_SIZE=1
export MASTER_PORT=${MASTER_PORT:-12971}
export S2ST_PRESERVE_COSYVOICE_CHECKPOINT=$((1-${OFFLINE_FRESH_BASE_INIT:-0})) S2ST_TALKER_PACK_ISOLATION=1 S2ST_SEGMENT_ALIGN_ISOLATED=1
export ENABLE_COSYVOICE_FM=0 COSYVOICE_CODEX_LOSS_WEIGHT=1.0 COSYVOICE_EOS_TOKEN_LOSS_WEIGHT=1.0 COSYVOICE_FM_LOSS_WEIGHT=0.0 NO_SAVE_OPTIM=0
export COSYVOICE_TTS_INIT_CKPT=${S3_ROOT}/models/training/talker_native.pth
export COSYVOICE_FM_INIT_CKPT=$COSYVOICE_TTS_INIT_CKPT
export DATALOADER_STATE_DIR="$EXP_DIR/dataloader_states" TENSORBOARD_DIR="$EXP_DIR/tensorboard"
export NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EXP_NAME="qwen3_5_4b_s2st_offline_fresh_layerfusion8_${stage}_20260918"
printf 'stage=%s\nsource=%s\nbase_lr=%s\nbridge_scale=%s\n' "$stage" "$source_iter" "$LR" "$S2ST_BRIDGE_LR_SCALE" > "$EXP_DIR/launch_contract.txt"
exec bash "$root/config/run_4b_s2st_e2e_talkerpackiso.sh"
