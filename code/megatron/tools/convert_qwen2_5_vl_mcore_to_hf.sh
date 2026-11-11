# 输入模型torch,torch_dist格式均支持
export CUDA_DEVICE_MAX_CONNECTIONS=1

MCORE_FORMAT_DIR=${INPUT_DIR:-"Qwen2.5-VL-7B-Instruct-mcore"}
HF_FORMAT_DIR=${OUTPUT_DIR:-"Qwen2.5-VL-7B-Instruct-HF2"}
SHARD=${SHARD:-5GB}

if [[ "x${MEGATRON_PATH}" != "x" ]]; then
    MEGATRON_PATH_ARGS="--megatron-path ${MEGATRON_PATH}"
fi

torchrun --nproc-per-node=1 --nnodes=1 checkpoint/convert.py \
--model-type GPT \
--loader qwen2_5_vl_mcore \
--saver qwen2_5_vl_hf \
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

QWEN2_5_VL_CONFIG_FILE_PATH=${QWEN2_5_VL_CONFIG_FILE_PATH:-"../qwen2_5_vl_config"}
if [ ! -e ${QWEN2_5_VL_CONFIG_FILE_PATH} ]; then
    echo "Cannot find qwen2_5_vl_config file path ${QWEN2_5_VL_CONFIG_FILE_PATH}"
    exit 1
else
    cp -vu ${QWEN2_5_VL_CONFIG_FILE_PATH}/chat_template.json ${HF_FORMAT_DIR}
    cp -vu ${QWEN2_5_VL_CONFIG_FILE_PATH}/configuration.json ${HF_FORMAT_DIR}
    cp -vu ${QWEN2_5_VL_CONFIG_FILE_PATH}/generation_config.json ${HF_FORMAT_DIR}
    cp -vu ${QWEN2_5_VL_CONFIG_FILE_PATH}/preprocessor_config.json ${HF_FORMAT_DIR}
fi
