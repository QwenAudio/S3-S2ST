#!/bin/bash
# Qwen3.5 Fun 模型: MCore checkpoint -> HF checkpoint 转换脚本
# 输入模型支持 torch、torch_dist 格式
# 参考: convert_mcore_to_hf_fun_encoder2B.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

WORK_DIR=/path/to/Qwen-public-0611
FUNASR_ROOT=${FUNASR_ROOT:-/path/to/FunASR_0611}
export PYTHONPATH=${WORK_DIR}:${FUNASR_ROOT}:${PYTHONPATH:-}

# MCore 格式 checkpoint 目录 (请根据实际情况修改)
MCORE_FORMAT_DIR=${WORK_DIR}/MCore-Qwen3.5-35B-A3B.sep.2026011501
# HF 格式输出目录
HF_FORMAT_DIR=${WORK_DIR}/HF-Qwen3.5-35B-A3B-from-mcore-sep-2026011501-debug

# Tokenizer 文件路径 (用于复制 tokenizer 到 HF 输出目录)
TOKENIZER_FILE_PATH=${WORK_DIR}/Qwen3.5-35B-A3B.sep

MEGATRON_PATH=${WORK_DIR}

SHARD=${SHARD:-5GB}

# ckpt supplement (如果存在则复制到 mcore 目录)
SUPPLEMENT_DIR="${WORK_DIR}/tools/supplement_ckpt"
if [[ -d "${SUPPLEMENT_DIR}" ]]; then
    cp ${SUPPLEMENT_DIR}/.metadata ${MCORE_FORMAT_DIR}
    cp ${SUPPLEMENT_DIR}/metadata.json ${MCORE_FORMAT_DIR}
fi

if [[ "x${MEGATRON_PATH}" != "x" ]]; then
    MEGATRON_PATH_ARGS="--megatron-path ${MEGATRON_PATH}"
fi
FUN_DATASET_CONFIG_ARGS=()
if [[ -n "${FUN_DATASET_CONFIG_PATH:-}" ]]; then
    FUN_DATASET_CONFIG_ARGS+=(--fun-dataset-config-path "${FUN_DATASET_CONFIG_PATH}")
fi

export Channels_Model=True

# 当 checkpoint 缺少 embedding_audio/output_layer_audio 时，设置此变量以允许部分加载
export no_audio_emb_and_head=True

# 大模型转换：loader 每发送一层后释放该层内存，避免持有一份完整模型导致 OOM
export LOADER_RELEASE_LAYER_AFTER_SEND=1

# 大模型转换：saver 收到一层写磁盘并释放，流式构建 HF shard，降低峰值内存
export SAVER_STREAMING_MODE=1

# 若使用 Qwen3-2B 结构的 audio encoder，添加: --qwen3-audio-encoder
# 若使用 SenseVoiceSmall audio encoder，添加: --sensevoicesmall-audio-encoder
cd ${WORK_DIR}/tools || exit 1

torchrun --nproc-per-node=1 --nnodes=1 checkpoint/convert.py \
    --model-type GPT \
    --loader fun_mcore_qwen3_5_plus \
    --saver fun_qwen3_5_hf_plus \
    --max-queue-size 1 \
    --load-dir ${MCORE_FORMAT_DIR} \
    --save-dir ${HF_FORMAT_DIR} \
    --shard ${SHARD} \
    --audio-encoder-ckpt-path /path/to/audio_encoder_only.pt \
    "${FUN_DATASET_CONFIG_ARGS[@]}" \
    ${MEGATRON_PATH_ARGS}

# Unified FM is saved by the checkpoint saver.  Materialize the matching
# standalone FunASR config next to it for codex_to_wav.py.
if [[ -f "${HF_FORMAT_DIR}/cosyvoice_fm/model.pt" ]]; then
    if [[ -z "${FUN_DATASET_CONFIG_PATH:-}" || ! -f "${FUN_DATASET_CONFIG_PATH}" ]]; then
        echo "FM was exported but FUN_DATASET_CONFIG_PATH is unavailable" >&2
        exit 1
    fi
    python - "${FUN_DATASET_CONFIG_PATH}" "${HF_FORMAT_DIR}/cosyvoice_fm/config.yaml" <<'PY'
import sys
import yaml

source, target = sys.argv[1:]
with open(source) as f:
    cfg = yaml.safe_load(f)
fm_conf = cfg.get("model_conf", {}).get("fm_conf", {})
if fm_conf.get("model") != "CosyVoiceFlowMatching":
    raise SystemExit("Dataset config has no CosyVoiceFlowMatching fm_conf")
with open(target, "w") as f:
    yaml.safe_dump(fm_conf, f, sort_keys=False)
PY
fi

if [ ! -e ${TOKENIZER_FILE_PATH} ]; then
    echo "Cannot find tokenizer file path ${TOKENIZER_FILE_PATH}"
    echo "Please copy tokenizer files (tokenizer.json, tokenizer_config.json, vocab.json, merges.txt) manually to ${HF_FORMAT_DIR}"
    exit 1
else
    cp -vu ${TOKENIZER_FILE_PATH}/tokenizer.json ${HF_FORMAT_DIR}
    cp -vu ${TOKENIZER_FILE_PATH}/tokenizer_config.json ${HF_FORMAT_DIR}
    cp -vu ${TOKENIZER_FILE_PATH}/vocab.json ${HF_FORMAT_DIR}
    cp -vu ${TOKENIZER_FILE_PATH}/merges.txt ${HF_FORMAT_DIR}
fi

echo "Conversion completed. HF checkpoint saved to ${HF_FORMAT_DIR}"
