#!/usr/bin/env bash
set -o pipefail
# Qwen3.5-MoE 35B A3B pretrain script (config from Qwen3.5-35B-A3B/config.json)
# Based on fun_qwen3_5_moe_397BA17B_pretrain.sh with 35B-A3B model structure

export CUDA_DEVICE_MAX_CONNECTIONS=1

# fix the f*cking topo
echo "Fixing NCCL topo ..."
FIXED_TOPO_FILE=${S3_ROOT}/training/streaming/talker/runtime/fixed_topo.xml
python ../../tools/fix_nccl_topo/fix_topo.py ${FIXED_TOPO_FILE}
export NCCL_TOPO_FILE=${FIXED_TOPO_FILE}

# for flashmoe
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-0}
export FLASH_MOE_SWIGLU_USE_SILU_ON_A=1
export FLASH_MOE_PERMUTATION_KERNEL_FUSION=1

# for FunASR
export PYTHONPATH="${EXP_DIR}/runtime:${S3_ROOT}/training/offline/config/code_snapshot:${S3_ROOT}/training/offline_base/config/code_snapshot:${S3_ROOT}/vendor/offline_funasr:${S3_ROOT}/vendor/FunASR:${S3_ROOT}/code/megatron:${PYTHONPATH:-}"
echo "PYTHONPATH: ${PYTHONPATH}"

export CUDA_PROFILER_OUTPUT=${CUDA_PROFILER_OUTPUT:-${S3_ROOT}/}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}

SEQ_LENGTH=${SEQ_LENGTH:-4096}

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GAS_STEPS=${GAS_STEPS:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}

DECODER_FIRST_PIPE_LAYER_NUM=${DECODER_FIRST_PIPE_LAYER_NUM:-}
DECODER_LAST_PIPE_LAYER_NUM=${DECODER_LAST_PIPE_LAYER_NUM:-}
DECODER_PIPELINE_STAGES_NUM_LAYERS_OVERRIDE=${DECODER_PIPELINE_STAGES_NUM_LAYERS_OVERRIDE:-}
# only allow DECODER_FIRST_PIPE_LAYER_NUM and DECODER_LAST_PIPE_LAYER_NUM if PP_SIZE > 1
if [ ${PP_SIZE} -eq 1 ] && ( [ ! -z "${DECODER_FIRST_PIPE_LAYER_NUM}" ] || [ ! -z "${DECODER_LAST_PIPE_LAYER_NUM}" ] || [ ! -z "${DECODER_PIPELINE_STAGES_NUM_LAYERS_OVERRIDE}" ] )
then
    echo "DECODER_FIRST_PIPE_LAYER_NUM and DECODER_LAST_PIPE_LAYER_NUM are not allowed when PP_SIZE is 1"
    exit 1
fi
echo "DECODER_FIRST_PIPE_LAYER_NUM: ${DECODER_FIRST_PIPE_LAYER_NUM}"
echo "DECODER_LAST_PIPE_LAYER_NUM: ${DECODER_LAST_PIPE_LAYER_NUM}"
if [ ! -z "${DECODER_FIRST_PIPE_LAYER_NUM}" ]
then
    DECODER_FIRST_PIPE_LAYER_NUM_ARGS="--decoder-first-pipeline-num-layers ${DECODER_FIRST_PIPE_LAYER_NUM}"
fi
if [ ! -z "${DECODER_LAST_PIPE_LAYER_NUM}" ]
then
    DECODER_LAST_PIPE_LAYER_NUM_ARGS="--decoder-last-pipeline-num-layers ${DECODER_LAST_PIPE_LAYER_NUM}"
fi
if [ ! -z "${DECODER_PIPELINE_STAGES_NUM_LAYERS_OVERRIDE}" ]
then
    DECODER_PIPELINE_STAGES_NUM_LAYERS_OVERRIDE_ARGS="--pipeline-stages-num-layers-override ${DECODER_PIPELINE_STAGES_NUM_LAYERS_OVERRIDE}"
fi
DYNAMIC_FLASH_ATTN=${DYNAMIC_FLASH_ATTN:---dynamic-flash-attn}
echo "DYNAMIC_FLASH_ATTN: ${DYNAMIC_FLASH_ATTN}"

USE_FLASHMOE=${USE_FLASHMOE:-true}
echo "USE_FLASHMOE: ${USE_FLASHMOE}"

MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.005}
echo "MOE_AUX_LOSS_COEFF: ${MOE_AUX_LOSS_COEFF}"

MOE_Z_LOSS_COEFF=${MOE_Z_LOSS_COEFF:-0.02}
echo "MOE_Z_LOSS_COEFF: ${MOE_Z_LOSS_COEFF}"

# 35B-A3B: num_hidden_layers=40
NUM_LAYERS=${NUM_LAYERS:-40}

LOAD_CKPT_DIR=${LOAD_CKPT_DIR:-""}
echo "LOAD_CKPT_DIR: ${LOAD_CKPT_DIR}"

TOKENIZER_MODEL=${TOKENIZER_MODEL}
echo "TOKENIZER_MODEL: ${TOKENIZER_MODEL}"

DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"${S3_ROOT}/data/train/dataset.yaml"}
echo "DATASET_CONFIG_PATH: ${DATASET_CONFIG_PATH}"

AUDIO_TOKENIZER_FRAME_RATE=${AUDIO_TOKENIZER_FRAME_RATE:-"25"}
echo "AUDIO_TOKENIZER_FRAME_RATE: ${AUDIO_TOKENIZER_FRAME_RATE}"

AUDIO_ENCODER_TYPE=${AUDIO_ENCODER_TYPE:-"0.7B"}
echo "AUDIO_ENCODER_TYPE: ${AUDIO_ENCODER_TYPE}"

DUAL_CHANNEL=${DUAL_CHANNEL:-false}
echo "DUAL_CHANNEL: ${DUAL_CHANNEL}"

MROPE_ARGS=${MROPE_ARGS:-"--mrope-interleaved --mrope-section 11 11 10"}
echo "MROPE_ARGS: ${MROPE_ARGS}"

UNTIE_EMBEDDINGS_ARGS=${UNTIE_EMBEDDINGS_ARGS:---untie-embeddings-and-output-weights}
echo "UNTIE_EMBEDDINGS_ARGS: ${UNTIE_EMBEDDINGS_ARGS}"

LR=${LR:-"5e-5"}
echo "LR: ${LR}"

SAVE_INTERVAL=${SAVE_INTERVAL:-"1000"}
echo "SAVE_INTERVAL: ${SAVE_INTERVAL}"

EXTRA_ARGS=${EXTRA_ARGS:-""}
echo "EXTRA_ARGS: ${EXTRA_ARGS}"

MOE_EXPERT_CAPACITY_FACTOR=${MOE_EXPERT_CAPACITY_FACTOR:-0}
echo "MOE_EXPERT_CAPACITY_FACTOR: ${MOE_EXPERT_CAPACITY_FACTOR}"

FULL_ATTENTION_INTERVAL=${FULL_ATTENTION_INTERVAL:-4}
echo "FULL_ATTENTION_INTERVAL: ${FULL_ATTENTION_INTERVAL}"

if [ -z "${EXP_DIR}" ]
then
    echo "EXP_DIR is not set"
    exit 1
fi
echo "EXP_DIR: ${EXP_DIR}"

FINETUNE_ARGS=${FINETUNE_ARGS:-"--finetune"}
echo "FINETUNE_ARGS: ${FINETUNE_ARGS}"

NO_LOAD_OPTIM_ARGS=${NO_LOAD_OPTIM_ARGS:-"--no-load-optim"}
echo "NO_LOAD_OPTIM_ARGS: ${NO_LOAD_OPTIM_ARGS}"

LOAD_PADDED_VOCAB_SIZE=${LOAD_PADDED_VOCAB_SIZE:-""}
echo "LOAD_PADDED_VOCAB_SIZE: ${LOAD_PADDED_VOCAB_SIZE}"

TRAIN_ITERS=${TRAIN_ITERS:-10000000}
echo "TRAIN_ITERS: ${TRAIN_ITERS}"

SWANLAB_PROJECT=${SWANLAB_PROJECT:-"${EXP_NAME}"}
echo "SWANLAB_PROJECT: ${SWANLAB_PROJECT}"

SWANLAB_EXP_NAME=${SWANLAB_EXP_NAME:-"35BA3B-fun"}
echo "SWANLAB_EXP_NAME: ${SWANLAB_EXP_NAME}"


TRAIN_SCRIPT=${TRAIN_SCRIPT:-pretrain_fun_base.py}
echo "TRAIN_SCRIPT: ${TRAIN_SCRIPT}"

PYTHON_BIN=${PYTHON_BIN:-python}
echo "PYTHON_BIN: ${PYTHON_BIN}"

TORCHRUN_ARGS="
    --nproc_per_node ${NPROC_PER_NODE:-1}
    --nnodes ${WORLD_SIZE:-1}
    --node_rank ${RANK:-0}
    --master_addr ${MASTER_ADDR:-127.0.0.1}
    --master_port ${MASTER_PORT:-12700}
"
if command -v numarun >/dev/null 2>&1; then
    TORCHRUN_ARGS="${TORCHRUN_ARGS}
--no-python numarun ${PYTHON_BIN}
"
else
    echo "numarun not found; fallback to ${PYTHON_BIN} without NUMA binding."
    TORCHRUN_ARGS="${TORCHRUN_ARGS}
--no-python ${PYTHON_BIN}
"
fi


# ===== Qwen3.5 MoE 35B-A3B LLM args (from Qwen3.5-35B-A3B/config.json) =====
# hidden_size=2048, num_hidden_layers=40, num_attention_heads=16, num_key_value_heads=2
# head_dim=256, moe_intermediate_size=512, num_experts=256, num_experts_per_tok=8
# linear_num_key_heads=16, linear_num_value_heads=32, linear_key/value_head_dim=128
HIDDEN_SIZE=${HIDDEN_SIZE:-2048}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-512}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-16}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-2}
KV_CHANNELS=${KV_CHANNELS:-256}

LLM_MODEL_ARGS="
    --qk-layernorm
    --rotary-base 10000000
    --rotary-percent 0.25
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --group-query-attention
    --kv-channels ${KV_CHANNELS}
    --num-query-groups ${NUM_QUERY_GROUPS}
    --disable-bias-linear
    --use-rotary-position-embeddings
    --no-position-embedding
    --swiglu
    --normalization RMSNorm
    --apply-layernorm-1p
    --num-layers ${NUM_LAYERS:-40}
    --hidden-size ${HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTENTION_HEADS}
    ${UNTIE_EMBEDDINGS_ARGS}
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --vocab-size 248320
    ${LOAD_PADDED_VOCAB_SIZE}
    --full-attention-interval ${FULL_ATTENTION_INTERVAL}
    --attn-output-gate
    ${MROPE_ARGS}
    --linear-attention-conv-kernel-dim 4
    --linear-attention-key-head-dim 128
    --linear-attention-value-head-dim 128
    --linear-attention-num-key-heads 16
    --linear-attention-num-value-heads 32
"


if [ "${AUDIO_TOKENIZER_FRAME_RATE}" == "none" ]; then
    AUDIO_TOKENIZER_ARGS="
        --audio-tokenizer-version none
        --audio-tokenizer-frame-rate 12.5
        --audio-tokenizer-code-book-size 59049
    "
elif [ "${AUDIO_TOKENIZER_FRAME_RATE}" == "12.5" ]; then
    AUDIO_TOKENIZER_ARGS="
        --audio-tokenizer-version 12.5hz
        --audio-tokenizer-frame-rate 12.5
        --audio-tokenizer-code-book-size 59049
    "
else
    AUDIO_TOKENIZER_ARGS="
        --audio-tokenizer-version 25hz
        --audio-tokenizer-frame-rate 25
        --audio-tokenizer-code-book-size 6761
    "
fi


if [ "${AUDIO_ENCODER_TYPE}" == "none" ]
then
    AUDIO_ENCODER_ARGS="--audio-encoder-type none"
elif [[ "${AUDIO_ENCODER_TYPE}" == "0.7B" ]]
then
    AUDIO_ENCODER_ARGS=""
elif [[ "${AUDIO_ENCODER_TYPE}" == "0.2B" ]]
then
    AUDIO_ENCODER_ARGS="
        --audio-encoder-type sensevoicesmall
        --audio-encoder-dim 512
    "
elif [[ "${AUDIO_ENCODER_TYPE}" == "Qwen3-0.6B" ]]
then
    AUDIO_ENCODER_ARGS="
        --audio-encoder-type qwen3-0.6b
        --audio-encoder-dim 1024
    "
    echo "Using Qwen Encoder, setting NVTE_FUSED_ATTN=0 to speed up training"
    export NVTE_FUSED_ATTN=0
elif [[ "${AUDIO_ENCODER_TYPE}" == "Qwen3-2B" ]]
then
    AUDIO_ENCODER_ARGS="
        --audio-encoder-type qwen3-2b
        --audio-encoder-dim 2048
    "
    echo "Using Qwen Encoder, setting NVTE_FUSED_ATTN=0 to speed up training"
else
    echo "Error: unknown AUDIO_ENCODER_TYPE: ${AUDIO_ENCODER_TYPE}"
    exit 1
fi
echo "AUDIO_ENCODER_ARGS: ${AUDIO_ENCODER_ARGS}"

if [ "${DUAL_CHANNEL}" == "true" ]; then
    AUDIO_DUAL_CHANNEL_ARGS="--audio-is-dual-channel"
fi

AUDIO_MODEL_ARGS="
    ${AUDIO_DUAL_CHANNEL_ARGS}
    ${AUDIO_TOKENIZER_ARGS}
    ${AUDIO_ENCODER_ARGS}
"
echo "AUDIO_MODEL_ARGS: ${AUDIO_MODEL_ARGS}"

if [ "${USE_FLASHMOE}" == "true" ]; then
    FLASHMOE_ARGS="
        --moe-grouped-gemm
        --use-flash-moe
        --use-flashmoe-router
        --use-flashmoe-hep
    "
else
    FLASHMOE_ARGS=""
fi
echo "FLASHMOE_ARGS: ${FLASHMOE_ARGS}"


# ===== Qwen3.5 MoE args (disabled for dense variants) =====
# Key changes from Qwen3:
#   - num-experts: 128 -> 512
#   - moe-router-topk: 8 -> 10
#   - moe-shared-expert-intermediate-size: 1024 (new, with sigmoid gate)
NUM_EXPERTS=${NUM_EXPERTS:-512}
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-10}
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-${FFN_HIDDEN_SIZE}}

USE_MOE=${USE_MOE:-true}
if [ "${USE_MOE}" = "true" ]; then
    MOE_ARGS="
        --moe-shared-expert-overlap
        --num-experts ${NUM_EXPERTS}
        --moe-router-load-balancing-type aux_loss
        --num-shared-experts 1
        --moe-shared-expert-intermediate-size ${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE}
        --moe-router-topk ${MOE_ROUTER_TOPK}
        --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF}
        --moe-z-loss-coeff ${MOE_Z_LOSS_COEFF}
        --moe-token-dispatcher-type alltoall
        --moe-expert-capacity-factor ${MOE_EXPERT_CAPACITY_FACTOR}
        --moe-sequence-load-balance-weight 0.0
        --moe-norm-topk-scores
        --expert-model-parallel-size ${EP_SIZE}
        ${FLASHMOE_ARGS}
    "
else
    if [ "${EP_SIZE}" -ne 1 ]; then
        echo "Dense Qwen3.5 requires EP_SIZE=1, got ${EP_SIZE}" >&2
        exit 1
    fi
    MOE_ARGS=""
fi
echo "USE_MOE: ${USE_MOE}"


DATALOADER_LOAD_ARGS=${DATALOADER_LOAD_ARGS:-"--dataloader-load $/dataloader_states"}
DATALOADER_STATE_DIR=${DATALOADER_STATE_DIR:-"$/dataloader_states"}
echo "DATALOADER_LOAD_ARGS: ${DATALOADER_LOAD_ARGS}"
echo "DATALOADER_STATE_DIR: ${DATALOADER_STATE_DIR}"

DATASET_ARGS="
    --fun-dataset-config-path ${DATASET_CONFIG_PATH}
    --renormalize-blend-weights
    --no-create-attention-mask-in-dataloader
    --data-cache-path ${EXP_DIR}/data_cache
    --num-workers 2
    --dataloader-type cyclic
    --dataloader-save
    ${DATALOADER_STATE_DIR}
    ${DATALOADER_LOAD_ARGS}
"

if [ ${CP_SIZE} -gt 1 ]
then
    echo "kvgather not support recompute now, set RECOMPUTE_ARGS to empty"
    RECOMPUTE_ARGS=""
else
    RECOMPUTE_ARGS=${RECOMPUTE_ARGS:-"--recompute-granularity selective"}
fi
echo "RECOMPUTE_ARGS: ${RECOMPUTE_ARGS}"


TRAINING_ARGS="
    --weight-decay 0.01
    --clip-grad 0.25
    --hidden-dropout 0
    --attention-dropout 0
    --seed 1024
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.009
    --lr-decay-iters 200000
    --lr-warmup-iters 1000
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings 32768
    --train-iters ${TRAIN_ITERS}
    --lr ${LR}
    --min-lr 1e-6
    --norm-epsilon 1e-06
    --lr-decay-style cosine
    --split 10000
    --hysteresis 2
    ${FINETUNE_ARGS}
    ${RECOMPUTE_ARGS}
"


if [ "${NO_INIT_LOAD}" = "1" ]; then
    LOAD=""
elif [ ! -z "${LOAD_CKPT_DIR}" ]; then
    LOAD="--load ${LOAD_CKPT_DIR}"
else
    LOAD="--load ${EXP_DIR}"
fi
echo "LOAD: ${LOAD}"


CHECKPOINT_ARGS="
    --save-interval ${SAVE_INTERVAL}
    ${NO_LOAD_OPTIM_ARGS}
    --use-dist-ckpt
    ${CKPT_FORMAT_ARGS:-}
    --dist-ckpt-strictness ${DIST_CKPT_STRICTNESS:-log_all}
    --auto-detect-ckpt-format
    --save ${EXP_DIR}
    ${LOAD}
"

if [ "${DISABLE_OVERLAP_ALL:-0}" = "1" ]; then
    OVERLAP_ARGS=" "
elif [ "${DISABLE_OVERLAP_PARAM_GATHER:-0}" = "1" ]; then
    OVERLAP_ARGS="--overlap-grad-reduce"
else
    OVERLAP_ARGS="--overlap-grad-reduce --overlap-param-gather"
fi
echo "OVERLAP_ARGS: ${OVERLAP_ARGS}"

ACCEL_ARGS="
    --use-flash-cross-entropy
    --inplace-flash-cross-entropy
    ${DYNAMIC_FLASH_ATTN}
    ${OVERLAP_ARGS}
    --reduce-loss-per-iteration
    --manual-gc
    --manual-gc-interval
    100
    --cross-entropy-loss-fusion
    --distributed-backend nccl
    --bf16
    --loss-scale 1.0
    --use-flash-attn
    --use-distributed-optimizer
    --tensor-model-parallel-size
    ${TP_SIZE}
    --pipeline-model-parallel-size
    ${PP_SIZE}
    ${DECODER_FIRST_PIPE_LAYER_NUM_ARGS}
    ${DECODER_LAST_PIPE_LAYER_NUM_ARGS}
${DECODER_PIPELINE_STAGES_NUM_LAYERS_OVERRIDE_ARGS}
    "--context-parallel-size"
    "${CP_SIZE}"
    "--context-parallel-strategy"
    "kvgather"
    --use-mcore-models
    --attention-softmax-in-fp32
"


SWANLAB_LOGGING_ARGS=""
if [[ "${ENABLE_SWANLAB:-false}" == "true" ]]; then
    SWANLAB_AUTH_ARGS=""
    if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
        SWANLAB_AUTH_ARGS="--swanlab-api-key ${SWANLAB_API_KEY}"
    fi
    SWANLAB_HOST=${SWANLAB_HOST:-http://47.100.3.166:10087}
    SWANLAB_LOGGING_ARGS="
        --swanlab-project ${SWANLAB_PROJECT}
        --swanlab-exp-name ${SWANLAB_EXP_NAME}
        ${SWANLAB_AUTH_ARGS}
        --swanlab-host ${SWANLAB_HOST}
    "
fi

LOGGING_ARGS="
    --logging-level 20
    --increase-log-level-interval 1000
    --log-interval 1
    --tensorboard-queue-size 500
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --tensorboard-dir ${TENSORBOARD_DIR}
    ${SWANLAB_LOGGING_ARGS}
"


current_time=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir="${EXP_DIR}/logs/${current_time}"
mkdir -p ${log_dir} || true
log_file="${log_dir}/log_${RANK:-0}.txt"

torchrun \
   ${TORCHRUN_ARGS} \
   ../../${TRAIN_SCRIPT} \
   ${DATASET_ARGS} \
   ${LLM_MODEL_ARGS} \
   ${AUDIO_MODEL_ARGS} \
   ${TRAINING_ARGS} \
   ${ACCEL_ARGS} \
   ${CHECKPOINT_ARGS} \
   ${LOGGING_ARGS} \
   ${MOE_ARGS} \
   ${EXTRA_ARGS} 2>&1 | tee ${log_file}
