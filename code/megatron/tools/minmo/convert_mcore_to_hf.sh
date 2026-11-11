#!/usr/bin/env bash


if [ $# -ne 3 ]
then
    echo "Usage: $0 <base_model_hf_dir> <input_dir> <output_dir>"
    exit 1
fi


BASE_MODEL_HF_DIR=$1
echo "BASE_MODEL_HF_DIR: $BASE_MODEL_HF_DIR"

INPUT_DIR=$2
echo "INPUT_DIR: $INPUT_DIR"

OUTPUT_DIR=$3
echo "OUTPUT_DIR: $OUTPUT_DIR"


TOKENIZER_FILE_PATH=$BASE_MODEL_HF_DIR
MEGATRON_PATH=$(dirname $0)/..


# below is convert_mcore_to_hf.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

MCORE_FORMAT_DIR=${INPUT_DIR:-"NOT_EXIST"}
HF_FORMAT_DIR=${OUTPUT_DIR:-"NOT_EXIST"}
SHARD=${SHARD:-5GB}

if [[ "x${MEGATRON_PATH}" != "x" ]]; then
    MEGATRON_PATH_ARGS="--megatron-path ${MEGATRON_PATH}"
fi

torchrun --nproc-per-node=1 --nnodes=1 checkpoint/convert.py \
--model-type GPT \
--loader mcore_all \
--saver qwen3_hf \
--load-dir ${MCORE_FORMAT_DIR} \
--save-dir ${HF_FORMAT_DIR} \
--shard $SHARD \
${MEGATRON_PATH_ARGS}

TOKENIZER_FILE_PATH=${TOKENIZER_FILE_PATH:-"../qwen_tokenizer"}
if [ ! -e ${TOKENIZER_FILE_PATH} ]; then
    echo "Cannot find tokenizer file path ${TOKENIZER_FILE_PATH}"
    exit 1
else
    cp -vu ${TOKENIZER_FILE_PATH}/tokenizer.json ${HF_FORMAT_DIR}
    cp -vu ${TOKENIZER_FILE_PATH}/tokenizer_config.json ${HF_FORMAT_DIR}
    cp -vu ${TOKENIZER_FILE_PATH}/vocab.json ${HF_FORMAT_DIR}
    cp -vu ${TOKENIZER_FILE_PATH}/merges.txt ${HF_FORMAT_DIR}
fi


if [ $? -ne 0 ]; then
    echo "Failed to convert MCore to HF"
    exit 1
fi

echo "Successfully converted MCore to HF"
echo "Saved HF model to ${HF_FORMAT_DIR}"