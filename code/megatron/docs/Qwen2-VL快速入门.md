# Qwen2-VL快速入门

## 准备环境
请参考[readme](../README.md#环境准备)使用合适的镜像，并安装以下依赖库：

```shell
pip install transformers qwen_vl_utils -U -i https://mirrors.aliyun.com/pypi/simple/
```

## SFT

### 下载权重

从modelscope下载Qwen2-VL-7B-Instruct权重：
```shell
# 模型存储目录: './Qwen2-VL-7B-Instruct'
# 安装git git-lfs
sudo apt install git git-lfs
git lfs install

# 下载Qwen2-VL-7B-Instruct
GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/Qwen/Qwen2-VL-7B-Instruct.git
cd Qwen2-VL-7B-Instruct
git lfs pull

# 回到工作目录
cd ..
```

### 数据集准备与预处理

下载数据集，并转换数据集格式：

（为了简化整体链路，我们使用少量开源的Latex-OCR数据集跑通流程）

Qwen2-VL的微调所需**数据集格式**如下（jsonl）：
```jsonl
{"type": "chatml", "messages": [{"role": "system", "content": [{"text": "You are a helpful assistant."}]}, {"role": "user", "content": [{"image": "<image_path/image_url>"}, {"text": "<text>"}]}, {"role": "assistant", "content": [{"text": "<text>"}]}]}
{"type": "chatml", "messages": [{"role": "system", "content": [{"text": "You are a helpful assistant."}]}, {"role": "user", "content": [{"image": "<image_path/image_url>"}, {"image": "<image_path/image_url>"}, {"text": "<text>"}]}, {"role": "assistant", "content": [{"text": "<text>"}]}]}
{"type": "chatml", "messages": [{"role": "system", "content": [{"text": "You are a helpful assistant."}]}, {"role": "user", "content": [{"image": "<image_path/image_url>"}, {"text": "<text>"}]}, {"role": "assistant", "content": [{"text": "<text>"}]}, {"role": "user", "content": [{"image": "<image_path/image_url>"}, {"text": "<text>"}]}, {"role": "assistant", "content": [{"text": "<text>"}]}]}
```

我们已经对Latex-OCR数据集的格式进行了转换，我们直接下载转换后的数据集。（原始数据集repo查看[这里](https://modelscope.cn/datasets/AI-ModelScope/LaTeX_OCR)）

```shell
# jsonl存储位置: ./preprocess/qwen2_vl_sft_data/latex_ocr.jsonl
cd preprocess/qwen2_vl_sft_data

wget "https://modelscope.cn/datasets/huangjintao/latex_ocr_test/resolve/master/latex_ocr.jsonl"
 
# 回到工作目录
cd ../..
```

### HF to MCore

```shell
cd tools

export TARGET_CKPT_FORMAT=torch_dist
export INPUT_DIR='../Qwen2-VL-7B-Instruct'
export TOKENIZER_FILE_PATH="../qwen_tokenizer"
export OUTPUT_DIR='../Qwen2-VL-7B-Instruct-mcore'

bash convert_qwen2vl_hf_to_mcore.sh

# 回到工作目录
cd ..
```

### 微调

```shell
cd scripts/sft

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4
export MP_SIZE=4
export PP_SIZE=1
export SEQ_LEN=8192

# 设置ckpt的载入目录
export PRETRAINED_CKPT_ROOT_PATH=../..
export PRETRAINED_CKPT_NAME=Qwen2-VL-7B-Instruct-mcore
# 设置ckpt的输出目录
export OUTPUT_CHECKPOINT_PATH=../../output/qwen2_vl_sft
export PRETRAINED_CKPT_ID=qwen2_vl_test
# tokenizer
export VOCAB_PATH=../../qwen_15w.json
export MERGE_PATH=../../qwen_15w.bpe

export DATA_PATH=../../preprocess/qwen2_vl_sft_data/latex_ocr.jsonl

# 训练参数（更多训练参数请直接修改shell脚本）
export GLOBAL_BATCH_SIZE=16
export LR=1e-5 MIN_LR=1e-6
export TRAIN_ITERS=100 LR_WARMUP_ITERS=40 LR_DECAY_ITERS=100
export SAVE_INTERVAL=100

bash qwen2_vl_sft_template.sh

# 回到工作目录
cd ../..
```

显存占用：4卡*45GiB

### MCore to HF

```shell
cd tools

# 存储的训练权重路径
export INPUT_DIR='../output/qwen2_vl_sft/checkpoint/qwen2_vl_test-7B-bf16-mp4-pp1-lr-1e-5-minlr-1e-6-bs-16-gpus--seqlen-8192'

export TOKENIZER_FILE_PATH="../qwen_tokenizer"

# 导出的权重/tokenizer路径
export OUTPUT_DIR='../output/qwen2_vl_sft/hf'
export hf_model_path=${OUTPUT_DIR}

bash convert_qwen2vl_mcore_to_hf.sh

# 回到工作目录
cd ..
```

最终我们将训练好的权重保存在'output/qwen2_vl_sft/hf'目录下。


### HF推理

使用transformers进行推理：

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
model_dir = 'output/qwen2_vl_sft/hf'

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_dir)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://modelscope.cn/datasets/huangjintao/latex_ocr_test/resolve/master/images/151e0ddeec07100559f17fa706e958a9483f3f9ea919a01f641d7ede7faccd8d.png",
            },
            {"type": "text", "text": "Using LaTeX to perform OCR on the image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

"""输出：
['e ^ { A } = e ^ { A _ { 0 } \\left( t _ { 0 } - \\mathrm { s i g n } ( m ) t \\right) ^ { - \\frac { m } { 2 } , \\chi = \\chi _ { 0 } \\left( t _ { 0 } - \\mathrm { s i g n } ( m ) t \\right) ^ { m } ,']
"""
```
