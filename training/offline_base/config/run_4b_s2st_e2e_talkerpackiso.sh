#!/usr/bin/env bash
set -euo pipefail

repo_root=${S3_ROOT}/code/megatron
workspace_root=/path/to/Qwen-public-0803-hejia
isolated_config_dir=${S3_ROOT}/training/offline_base/config
source_checkpoint=${SOURCE_CHECKPOINT:-/path/to/iter_0001600}
checkpoint_view=${workspace_root}/artifacts/4b_s2st/checkpoint_view
fm_init=${COSYVOICE_FM_INIT_CKPT:-/path/to/epoch_276_step_2209.pth}
dataset_config=${DATASET_CONFIG_PATH:-${repo_root}/scripts/sft/fun/conf/megatron_fun_sft_4b_s2st_0803_e2e_pretrained.8192.yaml}
packed_list=$("${PYTHON_BIN:-python3}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["train_data_set_list"])' "$dataset_config")
oss_credential_source=${OSS_CREDENTIAL_SOURCE:-${repo_root}/test/run_debug_local_dense_dc_online_sft.sh}

for required in "${fm_init}" "${dataset_config}" "${packed_list}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: missing required path: ${required}" >&2
        exit 1
    fi
done

# Match the proven 8B S2TT launch environment. Non-interactive SSH jobs do not
# inherit the caller's OSS credentials, so, only when they are absent, import
# the three credential exports from an existing target-local training script.
# The values remain process-local: they are never printed or copied to a new
# file. Callers can provide their own environment or override the read-only
# source with OSS_CREDENTIAL_SOURCE.
# Local project data needs no object-store credentials.
export OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET
[[ -z "${OSS_SESSION_TOKEN:-}" ]] || export OSS_SESSION_TOKEN
export OSS_ENDPOINT=${OSS_ENDPOINT:-oss-cn-shanghai-internal.aliyuncs.com}
export OSS_REGION=${OSS_REGION:-cn-shanghai}
export OSS_BUCKET_NAME=${OSS_BUCKET_NAME:-example-storage-1}
export OSS_ENDPOINT_SH=${OSS_ENDPOINT_SH:-oss-cn-shanghai-internal.aliyuncs.com}
export OSS_REGION_SH=${OSS_REGION_SH:-cn-shanghai}
export OSS_BUCKET_NAME_SH=${OSS_BUCKET_NAME_SH:-example-storage-4}

# Megatron expects a parent checkpoint directory with a tracker.  This view is
# target-local and points read-only at the supplied 4B iteration.
if [[ -n "${INIT_CKPT_DIR:-}" ]]; then
    if [[ ! -f "${INIT_CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
        echo "ERROR: INIT_CKPT_DIR is not a checkpoint parent: ${INIT_CKPT_DIR}" >&2
        exit 1
    fi
    load_ckpt_dir=${INIT_CKPT_DIR}
else
    if [[ ! -e "${source_checkpoint}/.metadata" ]]; then
        echo "ERROR: missing supplied 4B checkpoint: ${source_checkpoint}/.metadata" >&2
        exit 1
    fi
    mkdir -p "${checkpoint_view}"
    if [[ -L "${checkpoint_view}/iter_0001600" ]]; then
        [[ "$(readlink "${checkpoint_view}/iter_0001600")" = "${source_checkpoint}" ]] || {
            echo "ERROR: checkpoint view points to an unexpected source" >&2
            exit 1
        }
    elif [[ -e "${checkpoint_view}/iter_0001600" ]]; then
        echo "ERROR: refusing to replace ${checkpoint_view}/iter_0001600" >&2
        exit 1
    else
        ln -s "${source_checkpoint}" "${checkpoint_view}/iter_0001600"
    fi
    printf '1600\n' > "${checkpoint_view}/latest_checkpointed_iteration.txt"
    load_ckpt_dir=${checkpoint_view}
fi

export NUM_LAYERS=${NUM_LAYERS:-32}
export HIDDEN_SIZE=2560
export FFN_HIDDEN_SIZE=9216
export KV_CHANNELS=256
export NUM_ATTENTION_HEADS=16
export NUM_QUERY_GROUPS=4
export FULL_ATTENTION_INTERVAL=4
export MROPE_ARGS="--mrope-interleaved --mrope-section 11 11 10"
export UNTIE_EMBEDDINGS_ARGS=" "
export USE_MOE=false
export USE_FLASHMOE=false

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-12901}
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export TP_SIZE=${TP_SIZE:-2}
export PP_SIZE=1
export CP_SIZE=1
export EP_SIZE=1

export SEQ_LENGTH=${SEQ_LENGTH:-8192}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export GAS_STEPS=${GAS_STEPS:-1}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export LR=${LR:-1e-5}
export TRAIN_ITERS=${TRAIN_ITERS:-20000}

export LOAD_CKPT_DIR="${load_ckpt_dir}"
export TOKENIZER_MODEL=${S3_ROOT}/models/offline
export DATASET_CONFIG_PATH="${dataset_config}"
export PYTHONPATH="${EXP_DIR}/runtime:${S3_ROOT}/training/offline/config/code_snapshot:${S3_ROOT}/training/offline_base/config/code_snapshot:${S3_ROOT}/vendor/offline_funasr:${S3_ROOT}/vendor/FunASR:${S3_ROOT}/code/megatron:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export AUDIO_TOKENIZER_FRAME_RATE=12.5
export DUAL_CHANNEL=true
export AUDIO_ENCODER_TYPE=0.7B

export ENABLE_COSYVOICE_FM=${ENABLE_COSYVOICE_FM:-1}
export COSYVOICE_FM_ONLY=0
export COSYVOICE_TTS_INIT_FULL=1
export COSYVOICE_TTS_INIT_CKPT=${COSYVOICE_TTS_INIT_CKPT:-${fm_init}}
export COSYVOICE_FM_INIT_CKPT="${fm_init}"
export COSYVOICE_FM_LOSS_WEIGHT=${COSYVOICE_FM_LOSS_WEIGHT:-1.0}
export TEXT_LOSS_WEIGHT=${TEXT_LOSS_WEIGHT:-1.0}

export EXP_NAME=${EXP_NAME:-qwen3_5_4b_s2st_0803_e2e}
size_str="gpu${WORLD_SIZE}x${NPROC_PER_NODE}_tp${TP_SIZE}_pp1_cp1_ep1"
export EXP_DIR=${EXP_DIR:-"${workspace_root}/artifacts/4b_s2st/experiments/${EXP_NAME}_${size_str}"}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"${workspace_root}/artifacts/4b_s2st/cache/torchinductor"}
export SWANLAB_PROJECT=${SWANLAB_PROJECT:-${EXP_NAME}}
export SWANLAB_EXP_NAME=${SWANLAB_EXP_NAME:-"${EXP_NAME}_${size_str}"}
export ENABLE_SWANLAB=${ENABLE_SWANLAB:-false}

export LOAD_PADDED_VOCAB_SIZE="--load-padded-vocab-size"
# Stage transitions intentionally reset the iteration counter.  A same-stage
# resume must not pass --finetune, otherwise Megatron loads the weights but
# silently restarts at iteration 0.  The launcher below treats an empty value
# as "use --finetune", so use the already-valid no-load-rng flag as a benign
# non-empty replacement in resume mode.
if [[ "${RESUME_TRAINING:-0}" == 1 ]]; then
    export FINETUNE_ARGS="--use-checkpoint-opt_param-scheduler"
    # A whitespace value suppresses the downstream launcher's default
    # --no-load-optim while expanding to no command-line argument.
    export NO_LOAD_OPTIM_ARGS=" "
else
    export FINETUNE_ARGS=${FINETUNE_ARGS:---finetune}
    # A new branch loads model weights from its parent but must not import the
    # parent's optimizer/scheduler state.
    export NO_LOAD_OPTIM_ARGS="--no-load-optim"
fi
export DATALOADER_STATE_DIR=${DATALOADER_STATE_DIR:-"${EXP_DIR}/dataloader_states"}
mkdir -p "${DATALOADER_STATE_DIR}"
if [[ "${RESUME_TRAINING:-0}" == 1 ]]; then
    export DATALOADER_LOAD_ARGS="--dataloader-load ${DATALOADER_STATE_DIR}"
else
    export DATALOADER_LOAD_ARGS=" "
fi
export DIST_CKPT_STRICTNESS=ignore_all
export CKPT_FORMAT_ARGS=${CKPT_FORMAT_ARGS:-"--ckpt-format zarr"}
export NO_INIT_LOAD=0
export DYNAMIC_FLASH_ATTN=${DYNAMIC_FLASH_ATTN-" "}
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export DISABLE_OVERLAP_PARAM_GATHER=${DISABLE_OVERLAP_PARAM_GATHER:-1}
export DISABLE_OVERLAP_ALL=${DISABLE_OVERLAP_ALL:-1}
export EXTRA_NAME=${EXTRA_NAME:-""}
export EXTRA_ARGS="
--audio-no-freeze-encoder
--audio-recompute-encoder
--audio-encoder-no-use-fused-attention
--audio-encoder-use-packed-attention
--audio-adaptor-lr-scale 1.0
--audio-decoder-lr-scale 1.0
--audio-input-output-lr-scale 1.0
--no-audio-output-head
--fun-no-pad-speech-to-bucket
--audio-encoder-adaptor-sliding-window-size 10000000
--audio-encoder-adaptor-sliding-window-left-context 0
--audio-encoder-adaptor-bucketing-threshold 200
--increase-log-level-iters 1000

"
if [[ "${NO_SAVE_OPTIM:-0}" == 1 ]]; then
    export EXTRA_ARGS="${EXTRA_ARGS}
--no-save-optim
"
fi

echo "4B S2ST initialization: checkpoint_parent=${load_ckpt_dir}, TTS=${fm_init}"
echo "S2ST train stage: ${S2ST_TRAIN_STAGE:-legacy}; FM=${ENABLE_COSYVOICE_FM}"
echo "Preserve previous-stage CosyVoice checkpoint: ${S2ST_PRESERVE_COSYVOICE_CHECKPOINT:-0}"
echo "Offline packed data: ${packed_list}"
echo "Experiment output: ${EXP_DIR}"
cd "${repo_root}/scripts/sft"
exec bash "${isolated_config_dir}/fun_qwen3_5_moe_35BA3B_sft_talkerpackiso.sh" "$@"
