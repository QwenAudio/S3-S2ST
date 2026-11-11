#!/usr/bin/env bash
set -euo pipefail

repo_root=${S3_ROOT}/code/megatron
workspace_root=$(cd -- "${repo_root}/.." && pwd)
s2st_root=${repo_root}/streaming/s2st_exploration
runtime_dir=${EXP_DIR:?}/runtime
mkdir -p "${runtime_dir}"
export PYTHON_BIN=${PYTHON_BIN:-python}
export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"
streaming_site_packages=$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')
source_checkpoint=${SOURCE_CHECKPOINT:-${S3_ROOT}/checkpoints/offline/iter_0010000}
checkpoint_view=${runtime_dir}/checkpoint_view
fm_init=${COSYVOICE_FM_INIT_CKPT:?}
config_template=${STREAMING_S2ST_CONFIG_TEMPLATE:-${s2st_root}/config/streaming_s2st_talker.yaml}
dataset_config=${DATASET_CONFIG_PATH:-${runtime_dir}/streaming_s2st_talker.yaml}
packed_list=${TRAIN_DATA_LIST:?}
valid_list=${VALID_DATA_LIST:-$TRAIN_DATA_LIST}
INIT_CKPT_DIR=${INIT_CKPT_DIR:-${repo_root}/streaming/s2st_exploration/checkpoint_views/talker_codec_envelope_iter2000}
talker_checkpoint=${S3_ROOT}/checkpoints/offline/iter_0010000


for required in "${fm_init}" "${config_template}" "${packed_list}" "${valid_list}" "${talker_checkpoint}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: missing required path: ${required}" >&2
        exit 1
    fi
done

cp "${config_template}" "${dataset_config}"
sed -i "s#\${S3_ROOT}#${S3_ROOT}#g" "${dataset_config}"
sed -i -e "s#STREAMING_S2ST_TRAIN_LIST_PLACEHOLDER#${packed_list}#" -e "s#STREAMING_S2ST_VALID_LIST_PLACEHOLDER#${valid_list}#" "${dataset_config}"

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
export GAS_STEPS=${GAS_STEPS:-8}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export LR=${LR:-1e-6}
export TRAIN_ITERS=${TRAIN_ITERS:-20000}

export LOAD_CKPT_DIR="${load_ckpt_dir}"
export TOKENIZER_MODEL=${S3_ROOT}/models/offline
export DATASET_CONFIG_PATH="${dataset_config}"
export PYTHONPATH="${EXP_DIR}/runtime:${S3_ROOT}/training/offline/config/code_snapshot:${S3_ROOT}/training/offline_base/config/code_snapshot:${S3_ROOT}/vendor/offline_funasr:${S3_ROOT}/vendor/FunASR:${S3_ROOT}/code/megatron:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export AUDIO_TOKENIZER_FRAME_RATE=12.5
export DUAL_CHANNEL=true
export AUDIO_ENCODER_TYPE=0.7B

export ENABLE_COSYVOICE_FM=0
export COSYVOICE_FM_ONLY=0
export COSYVOICE_TTS_INIT_FULL=1
export COSYVOICE_TTS_INIT_CKPT=${COSYVOICE_TTS_INIT_CKPT:-${fm_init}}
export COSYVOICE_FM_INIT_CKPT="${fm_init}"
export COSYVOICE_FM_LOSS_WEIGHT=0
export COSYVOICE_CODEX_LOSS_WEIGHT=1.0
export TEXT_LOSS_WEIGHT=${TEXT_LOSS_WEIGHT:-1.0}
export AUDIO_LOSS_WEIGHT=0.0
export S2ST_TRAIN_STAGE=tts
export S2ST_PRESERVE_COSYVOICE_CHECKPOINT=1
export STREAMING_DEBUG_DISABLE_GATE=0
export STREAM_GATE_LOSS_WEIGHT=${STREAM_GATE_LOSS_WEIGHT:-0.1}
export STREAM_GATE_WRITE_WEIGHT=${STREAM_GATE_WRITE_WEIGHT:-1.0}
export STREAM_GATE_BACKPROP_TO_THINKER=1
export STREAM_GATE_THINKER_GRAD_SCALE=${STREAM_GATE_THINKER_GRAD_SCALE:-1.0}
export STREAM_GATE_LR_SCALE=${STREAM_GATE_LR_SCALE:-10.0}
export STREAMING_E2E_ENCODER_LR_SCALE=${STREAMING_E2E_ENCODER_LR_SCALE:-1.0}
export STREAMING_E2E_AUDIO_ADAPTOR_LR_SCALE=${STREAMING_E2E_AUDIO_ADAPTOR_LR_SCALE:-1.0}
export STREAMING_E2E_THINKER_LR_SCALE=${STREAMING_E2E_THINKER_LR_SCALE:-1.0}
export STREAMING_E2E_BRIDGE_LR_SCALE=${STREAMING_E2E_BRIDGE_LR_SCALE:-1.0}
export STREAMING_E2E_TALKER_LR_SCALE=${STREAMING_E2E_TALKER_LR_SCALE:-1.0}
export STREAMING_E2E_CODEC_IO_LR_SCALE=${STREAMING_E2E_CODEC_IO_LR_SCALE:-1.0}
export STREAMING_TRUE_ENCODER=1
export STREAMING_TRUE_ENCODER_CHUNK_MS=640
export STREAMING_TRUE_ENCODER_OVERLAY=0
export STREAMING_OFFLINE_CODE_ROOT="${repo_root}"
export STREAMING_S2ST_TALKER_CKPT="${talker_checkpoint}"
export STREAMING_S2ST_CODEC_ENVELOPE=1
export STREAMING_S2ST_CODEC_BOS_TOKEN_ID=59049
export STREAMING_S2ST_CODEC_EOS_TOKEN_ID=59050
export STREAMING_S2ST_CODEC_ENVELOPE_BLANK_COUNT=5
export STREAMING_S2ST_CODEC_INPUT_TOKEN_DROPOUT=0
export TRAIN_SCRIPT="$EXP_DIR/runtime/pretrain_exact_codec.py"

export EXP_NAME=${EXP_NAME:-qwen3_5_4b_true_streaming_s2st_e2e_allmodules_from_talker2k_lr1e6_gate1e5_nowarmup}
size_str="gpu${WORLD_SIZE}x${NPROC_PER_NODE}_tp${TP_SIZE}_pp1_cp1_ep1"
export EXP_DIR=${EXP_DIR:-"${repo_root}/streaming/artifacts/experiments/${EXP_NAME}_${size_str}"}
if [[ "${RESUME_TRAINING:-0}" != 1 && -f "${EXP_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "ERROR: fresh E2E stage requires a new EXP_NAME/EXP_DIR; checkpoint output already exists: ${EXP_DIR}" >&2
    exit 1
fi
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${EXP_DIR}/tensorboard}
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
    export FINETUNE_ARGS="--no-load-rng"
    # A whitespace value suppresses the downstream launcher's default
    # --no-load-optim while expanding to no command-line argument.
    export NO_LOAD_OPTIM_ARGS=" "
    # A same-stage resume must preserve the Talker/bridge/codec-I/O tensors
    # from the streaming checkpoint.  Re-applying the offline Talker overlay
    # here silently resets those tensors while retaining the resumed optimizer
    # state, producing an invalid continuation.
    export STREAMING_S2ST_SKIP_TALKER_OVERLAY=1
else
    export FINETUNE_ARGS=${FINETUNE_ARGS:---finetune}
    # A new branch loads model weights from its parent but must not import the
    # parent's optimizer/scheduler state.
    export NO_LOAD_OPTIM_ARGS="--no-load-optim"
    # The parent checkpoint already contains the trained streaming Talker.
    # Re-applying the offline Talker overlay would silently destroy the 2k init.
    export STREAMING_S2ST_SKIP_TALKER_OVERLAY=${STREAMING_S2ST_SKIP_TALKER_OVERLAY:-1}
fi
export DATALOADER_STATE_DIR=${DATALOADER_STATE_DIR:-"${EXP_DIR}/dataloader_states"}
mkdir -p "${DATALOADER_STATE_DIR}"
if [[ "${RESUME_TRAINING:-0}" == 1 ]]; then
    export DATALOADER_LOAD_ARGS="--dataloader-load ${DATALOADER_STATE_DIR}"
else
    export DATALOADER_LOAD_ARGS=" "
fi
export DIST_CKPT_STRICTNESS=ignore_all
# Python 3.10 can read the Zarr parent through auto-detection, but the bundled
# tensorstore binary is built for Python 3.12.  Save the new branch with the
# repository's proven torch_dist backend so checkpoints remain resumable.
export CKPT_FORMAT_ARGS=${CKPT_FORMAT_ARGS:-"--ckpt-format torch_dist"}
export NO_INIT_LOAD=0
export DYNAMIC_FLASH_ATTN=${DYNAMIC_FLASH_ATTN-" "}
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export DISABLE_OVERLAP_PARAM_GATHER=${DISABLE_OVERLAP_PARAM_GATHER:-1}
export DISABLE_OVERLAP_ALL=${DISABLE_OVERLAP_ALL:-1}
export EXTRA_NAME=${EXTRA_NAME:-""}
export EXTRA_ARGS="
--no-load-rng
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
--allow-overwrite
"
if [[ "${NO_SAVE_OPTIM:-0}" == 1 ]]; then
    export EXTRA_ARGS="${EXTRA_ARGS}
--no-save-optim
"
fi

echo "4B S2ST initialization: checkpoint_parent=${load_ckpt_dir}, TTS=${fm_init}"
echo "S2ST train stage: end-to-end codec supervision; FM=${ENABLE_COSYVOICE_FM}"
echo "Effective LRs: encoder/Thinker/adaptor/Talker/bridge/codec_io=${LR}; gate=${LR} x ${STREAM_GATE_LR_SCALE}"
echo "Offline Talker overlay: ${STREAMING_S2ST_TALKER_CKPT}"
echo "Skip offline Talker overlay: ${STREAMING_S2ST_SKIP_TALKER_OVERLAY}"
echo "Preserve previous-stage CosyVoice checkpoint: ${S2ST_PRESERVE_COSYVOICE_CHECKPOINT:-0}"
if [[ "${STREAMING_S2ST_FINAL_CODEC_EOS_ONLY:-0}" == 1 ]]; then
    echo "Codec envelope: 5xMASK -> CONTINUE_BOS(59049) -> codec -> PAUSE(59051); final uses FINAL_BOS(59052) -> codec -> EOS(59050)"
else
    echo "Codec envelope: 5xMASK -> BOS(59049) -> codec -> EOS(59050)"
fi
echo "Balanced packed data: ${packed_list}"
echo "Experiment output: ${EXP_DIR}"
runtime_launcher="${runtime_dir}/fun_streaming_s2st.sh"
cp "${EXP_DIR}/backup/launcher_source.sh" "${runtime_launcher}"
sed -i \
    -e 's#^GLOBAL_BATCH_SIZE=.*#GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}#' \
    -e 's#--lr-warmup-iters 1000#--lr-warmup-iters ${LR_WARMUP_ITERS:-0}#' \
    -e 's#--tensorboard-dir tensorboard/35B-A3B-fun-test-run1#--tensorboard-dir ${TENSORBOARD_DIR}#' \
    -e "s#^FIXED_TOPO_FILE=.*#FIXED_TOPO_FILE=${EXP_DIR}/runtime/fixed_topo.xml#" \
    "${runtime_launcher}"
sed -i '/^[[:space:]]*--no-check-for-nan-in-loss-and-grad$/d' "${runtime_launcher}"
if [[ "${TP_SIZE}" == 1 ]]; then
    # The embedded HF Talker is replicated. At TP1, sequence parallelism is
    # both meaningless and incompatible with packed-attention isolation.
    sed -i '/^[[:space:]]*--sequence-parallel$/d' "${runtime_launcher}"
fi
if [[ "${STREAMING_S2ST_PREPARE_ONLY:-0}" == 1 ]]; then
    echo "Prepared streaming S2ST Talker runtime: ${runtime_dir}"
    echo "train_data_set_list=${packed_list}"
    echo "valid_data_set_list=${valid_list}"
    echo "global_batch_size=${GLOBAL_BATCH_SIZE:-64}"
    exit 0
fi
cd "${repo_root}/scripts/sft"
export EXACT_CODEC_BASE_ENTRY="$EXP_DIR/runtime/pretrain_joint_layerfusion.py"
export TRAIN_SCRIPT="$EXP_DIR/runtime/pretrain_exact_codec.py"
export STREAMING_S2ST_CODEC_RESERVATION_MARGIN=0
exec bash "${runtime_launcher}" "$@"
