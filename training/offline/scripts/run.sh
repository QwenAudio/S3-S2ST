#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
base=${S3_ROOT}/training/offline_base
v4=${S3_ROOT}/data/train
export PYTHON_BIN=${PYTHON_BIN:-python}
export NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export GLOBAL_BATCH_SIZE=64 MICRO_BATCH_SIZE=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-12981}
export RESUME_TRAINING=${RESUME_TRAINING:-0} DRY_RUN=${DRY_RUN:-0}
export FINETUNE_ARGS=--finetune
for name in NNODES NODE_RANK NPROC_PER_NODE GLOBAL_BATCH_SIZE; do
 [[ ${!name} =~ ^[0-9]+$ ]] || { echo "Invalid $name" >&2; exit 2; }
done
((NNODES>0 && NPROC_PER_NODE>0 && NODE_RANK<NNODES)) || exit 2
((GLOBAL_BATCH_SIZE>0 && GLOBAL_BATCH_SIZE%(NNODES*NPROC_PER_NODE)==0)) || { echo 'GBS must be divisible by total GPUs (TP=PP=CP=1, micro batch=1).' >&2; exit 2; }
if ((NNODES>1)) && [[ "$MASTER_ADDR" == 127.0.0.1 || "$MASTER_ADDR" == localhost ]]; then
 echo 'Multi-node run needs MASTER_ADDR=node0 reachable IP/hostname.' >&2; exit 2
fi
[[ $RESUME_TRAINING == 0 || $RESUME_TRAINING == 1 ]] || exit 2
export WORLD_SIZE=$NNODES RANK=$NODE_RANK TP_SIZE=1
export EXP_DIR=${EXP_DIR:?}

[[ -f "${TRAIN_DATA_LIST:?}" ]] || exit 2
[[ -f "${S3_INIT_CKPT}/iter_0005000/.metadata" || -f "${S3_INIT_CKPT}/iter_0005000/metadata.json" ]] || exit 2
[[ -f "${S3_INIT_CKPT}/latest_checkpointed_iteration.txt" ]] || exit 2
# local copied initialization, verified by asset manifest
export TRAIN_ITERS=${TRAIN_ITERS:-10000} # reported continuation checkpoint; LR decay schedule stays 68000
export LR_DECAY_ITERS=68000 EXIT_INTERVAL=$TRAIN_ITERS
export LR=${LR:-5e-6} MIN_LR=${MIN_LR:-1e-6} LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-100}
export SAVE_INTERVAL=${SAVE_INTERVAL:-500}
export DATASET_CONFIG_PATH="${DATASET_CONFIG_PATH:?}"
export TRAIN_SCRIPT="$base/config/pretrain_fun_base_talkerpackiso.py"
export PYTHONPATH="${EXP_DIR}/runtime:${S3_ROOT}/training/offline/config/code_snapshot:${S3_ROOT}/training/offline_base/config/code_snapshot:${S3_ROOT}/vendor/offline_funasr:${S3_ROOT}/vendor/FunASR:${S3_ROOT}/code/megatron:${PYTHONPATH:-}"
export S2ST_TRAIN_STAGE=tts TEXT_LOSS_WEIGHT=0.0 S2ST_BRIDGE_LR_SCALE=20.0
export OFFLINE_FRESH_BASE_INIT=0 S2ST_PRESERVE_COSYVOICE_CHECKPOINT=1
export S2ST_TALKER_PACK_ISOLATION=1 S2ST_SEGMENT_ALIGN_ISOLATED=1
export ENABLE_COSYVOICE_FM=0 COSYVOICE_CODEX_LOSS_WEIGHT=1.0 COSYVOICE_EOS_TOKEN_LOSS_WEIGHT=1.0 COSYVOICE_FM_LOSS_WEIGHT=0.0 NO_SAVE_OPTIM=0
export COSYVOICE_TTS_INIT_CKPT=${S3_ROOT}/models/training/talker_native.pth
export COSYVOICE_FM_INIT_CKPT=$COSYVOICE_TTS_INIT_CKPT
export DATALOADER_STATE_DIR="$EXP_DIR/dataloader_states" TENSORBOARD_DIR="$EXP_DIR/tensorboard"
export NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EXP_NAME=qwen3_5_4b_s2st_offline_v4_no9pair_scale_from5k_1epoch_20260923
if [[ $RESUME_TRAINING == 1 ]]; then
 export INIT_CKPT_DIR="$EXP_DIR"
 [[ -f "$EXP_DIR/latest_checkpointed_iteration.txt" && -d "$DATALOADER_STATE_DIR" && -f "$EXP_DIR/run_contract.json" ]] || { echo 'No complete v4 run to resume.' >&2; exit 2; }
else
 export INIT_CKPT_DIR="${S3_INIT_CKPT:?}"
 [[ ! -f "$EXP_DIR/latest_checkpointed_iteration.txt" ]] || { echo 'Existing v4 checkpoint: use RESUME_TRAINING=1.' >&2; exit 2; }
 if [[ -d "$DATALOADER_STATE_DIR" ]] && [[ -n $(find "$DATALOADER_STATE_DIR" -type f -print -quit) ]]; then
  echo 'Refusing fresh data traversal with existing loader states; use a fresh EXP_DIR.' >&2; exit 2
 fi
fi
mkdir -p "$EXP_DIR"
if [[ $DRY_RUN != 1 ]]; then
 exec 9>"$EXP_DIR/node_${NODE_RANK}.lock"
 flock -n 9 || { echo 'This node rank is already running.' >&2; exit 2; }
 "$PYTHON_BIN" "$root/scripts/contract.py"
fi
printf 'v4 new-stage steps=%s, GBS=%s, nodes=%s, node_rank=%s, source=iter_0005000, resume_v4=%s\n' "$TRAIN_ITERS" "$GLOBAL_BATCH_SIZE" "$NNODES" "$NODE_RANK" "$RESUME_TRAINING"
exec bash "$root/config/run_backend.sh"
