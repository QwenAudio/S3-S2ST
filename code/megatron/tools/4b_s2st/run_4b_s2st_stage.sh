#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
stage=${1:-}
case "${stage}" in
    bridge) stage=adapter ;;
    joint) stage=tts ;;
    full) stage=s2tt ;;
esac
if [[ "${stage}" != adapter && "${stage}" != tts && "${stage}" != s2tt ]]; then
    echo "Usage: $0 {adapter|tts|s2tt} [training arguments ...]" >&2
    exit 2
fi
shift

export S2ST_TRAIN_STAGE=${stage}
# Optimizer state is required for a numerically continuous same-stage resume.
# Individual debug launches may still opt out with NO_SAVE_OPTIM=1.
export NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-0}
case "${stage}" in
    adapter)
        # Stage 1: train only the random 2560 -> 896 hidden-state adapter.
        export S2ST_PRESERVE_COSYVOICE_CHECKPOINT=0
        export ENABLE_COSYVOICE_FM=0
        export COSYVOICE_CODEX_LOSS_WEIGHT=${COSYVOICE_CODEX_LOSS_WEIGHT:-1.0}
        export TEXT_LOSS_WEIGHT=${TEXT_LOSS_WEIGHT:-0.0}
        export COSYVOICE_FM_LOSS_WEIGHT=0.0
        export LR=${LR:-1e-4}
        export TRAIN_ITERS=${TRAIN_ITERS:-5000}
        export EXP_NAME=${EXP_NAME:-qwen3_5_4b_s2st_stage1_adapter}
        ;;
    tts)
        if [[ -z "${INIT_CKPT_DIR:-}" && "${TTS_INIT_FROM_BASE:-0}" != 1 ]]; then
            echo "ERROR: tts stage requires INIT_CKPT_DIR=<stage1 checkpoint parent>" >&2
            exit 2
        fi
        # Stage 2: adapt the bridge, TTS Transformer, and its codec-token I/O
        # embedding/head, matching reference Talker finetune.  The neural speech
        # codec itself, the original 4B audio head, FM, and vocoder stay fixed.
        # A normal stage-to-stage load retains the learned bridge.  A direct
        # reference-Talker initialization starts from the 4B base and intentionally
        # creates a fresh 2560 -> 896 bridge, because reference bridge is 2048 -> 896.
        if [[ "${TTS_INIT_FROM_BASE:-0}" == 1 ]]; then
            export S2ST_PRESERVE_COSYVOICE_CHECKPOINT=0
        else
            export S2ST_PRESERVE_COSYVOICE_CHECKPOINT=1
        fi
        export ENABLE_COSYVOICE_FM=0
        export COSYVOICE_CODEX_LOSS_WEIGHT=${COSYVOICE_CODEX_LOSS_WEIGHT:-1.0}
        export TEXT_LOSS_WEIGHT=${TEXT_LOSS_WEIGHT:-0.0}
        export COSYVOICE_FM_LOSS_WEIGHT=0.0
        export LR=${LR:-2e-5}
        # One 8192-token packed-data epoch is about 280k steps at global batch 8.
        export TRAIN_ITERS=${TRAIN_ITERS:-280000}
        export EXP_NAME=${EXP_NAME:-qwen3_5_4b_s2st_stage2_tts_llm}
        ;;
    s2tt)
        if [[ -z "${INIT_CKPT_DIR:-}" ]]; then
            echo "ERROR: s2tt stage requires INIT_CKPT_DIR=<stage2 checkpoint parent>" >&2
            exit 2
        fi
        # Stage 3: add the complete pretrained S2TT path.  Continue adapting the
        # bridge/TTS LLM/codec-token I/O, while the neural codec, original 4B
        # audio head, FM and vocoder stay fixed.
        # Retain both the bridge and TTS LLM learned in stage 2.
        export S2ST_PRESERVE_COSYVOICE_CHECKPOINT=1
        export ENABLE_COSYVOICE_FM=0
        export COSYVOICE_CODEX_LOSS_WEIGHT=${COSYVOICE_CODEX_LOSS_WEIGHT:-1.0}
        export TEXT_LOSS_WEIGHT=${TEXT_LOSS_WEIGHT:-1.0}
        export COSYVOICE_FM_LOSS_WEIGHT=0.0
        export LR=${LR:-5e-6}
        export TRAIN_ITERS=${TRAIN_ITERS:-20000}
        export EXP_NAME=${EXP_NAME:-qwen3_5_4b_s2st_stage3_s2tt}
        ;;
esac

exec bash "${repo_root}/tools/4b_s2st/run_4b_s2st_e2e.sh" "$@"
