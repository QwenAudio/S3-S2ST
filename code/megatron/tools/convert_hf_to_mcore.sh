# 使用dist_ckpt时，目前要求tp，ep，pp均设为1
# 不使用dist_ckpt时，TARGET_CKPT_FORMAT 设为 torch

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
--megatron-path '..'