#!/usr/bin/env bash


if [ $# -ne 5 ]
then
    echo "Usage: $0 <input_dir> <target_tp_size> <target_ep_size> <target_pp_size> <output_dir>"
    exit 1
fi


INPUT_DIR=$1
echo "INPUT_DIR: $INPUT_DIR"

TARGET_TP_SIZE=$2
echo "TARGET_TP_SIZE: $TARGET_TP_SIZE"

TARGET_PP_SIZE=$3
echo "TARGET_PP_SIZE: $TARGET_PP_SIZE"

TARGET_EP_SIZE=$4
echo "TARGET_EP_SIZE: $TARGET_EP_SIZE"

OUTPUT_DIR=$5
echo "OUTPUT_DIR: $OUTPUT_DIR"


TOKENIZER_FILE_PATH=$INPUT_DIR
MEGATRON_PATH=$(dirname $0)/../..


# below is convert_hf_to_mcore.sh
TOKENIZER_FILE_PATH=${TOKENIZER_FILE_PATH:-"../qwen_tokenizer"}
export CUDA_DEVICE_MAX_CONNECTIONS=1

TARGET_TP_SIZE=${TARGET_TP_SIZE:-1}
TARGET_EP_SIZE=${TARGET_EP_SIZE:-1}
TARGET_PP_SIZE=${TARGET_PP_SIZE:-1}

TARGET_CKPT_FORMAT=${TARGET_CKPT_FORMAT:-"torch_dist"}

HF_FORMAT_DIR=${INPUT_DIR:-"NOT_EXIST"}
MCORE_FORMAT_DIR=${OUTPUT_DIR:-"NOT_EXIST"}

torchrun --nproc-per-node=1 --nnodes=1 checkpoint/convert.py \
--model-type GPT \
--loader qwen3_hf \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MCORE_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_FILE_PATH} \
--target-ckpt-format ${TARGET_CKPT_FORMAT} \
--megatron-path ${MEGATRON_PATH}

if [ $? -ne 0 ]; then
    echo "Failed to convert HF to MCore"
    exit 1
fi

echo "Successfully converted HF to MCore"
echo "Saved MCore model to ${MCORE_FORMAT_DIR}"