#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/path/to/FunASR

WORK_DIR=/Path/to/your/Qwen-public
hf_path=/Path/to/your/HF-ckpt

cd ${WORK_DIR}/tools || exit 1

export Random_Init_Audio_Encoder=False
export Random_Init_Audio_Adaptor=False
export Channels_Model=True
# export NUM_LAYERS_OVERRIDE=4

/usr/local/bin/torchrun --nproc-per-node=1 --nnodes=1 --master-port=29508 \
    checkpoint/convert.py \
    --model-type GPT \
    --loader fun_qwen3_5_hf_plus \
    --saver fun_mcore_qwen3_5_plus \
    --max-queue-size 1 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --load-dir ${hf_path} \
    --save-dir ${WORK_DIR}/MCore-Qwen3.5-397B-A17B.sep.2026011501 \
    --tokenizer-model ${hf_path} \
    --target-ckpt-format torch_dist \
    --megatron-path ${WORK_DIR} \
    --audio-encoder-ckpt-path /path/to/your/audio_encoder.pt \
    --audio-adaptor-pt-path /path/to/your/audio_adaptor.pt

