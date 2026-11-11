

# Qwen-Public
## 框架介绍
Qwen-Public项目基于NVIDIA开源的Megatron-LM进行深度定制优化，提供Dense模型及MoE两种结构模型的预训练及SFT微调能力，旨在为Qwen系列模型提供高效、灵活的分布式训练支持。核心功能包括：

1. 分布式训练优化

Qwen-Public采用Mcore的GPU优化技术，支持张量并行（Tensor Parallelism, TP）、流水线并行（Pipeline Parallelism, PP）及数据并行（Data Parallelism, DP）策略，此外针对MoE模型，我们使用PAI团队开发的flash_moe实现了高效的专家并行（Expert Parallelism, EP）。

2. 双模型架构支持
    - Dense模型：基于经典的Transformer架构，提供高效的预训练及微调能力
    - MoE模型：集成稀疏门控机制与专家网络动态路由，并提供多种expert负载均衡策略，实验结果显示MoE模型可以达到激活量5-7倍的Dense模型能力
3. 专项模型支持

在纯文本通用模型以外，Qwen系列同样提供了各种专项模型。在Qwen-Public中，我们目前提供了代码专项Qwen-coder和视觉专项模型Qwen-VL的继续训练能力，供开发者进行二次开发。

4. 上下游工具链

除了高效的分布式训练能力外，Qwen-Public框架也提供了适配LLM上下游的各种工具，包括：

    - 预训练数据格式转换
    - 微调数据格式转换
    - Mcore ckpt to HF ckpt转换
    - HF ckpt to Mcore ckpt 转换
    - Qwen3 HF模型实现



答疑交流群请搜索 Qwen内部开发者交流群 或 钉钉群号： 43370002280，欢迎各团队同学贡献与二次开发！

# Release Note
## 2025.07.30
+ fix KNOWN ISSUES
    + torch 在某个版本之后添加了对载入 tensor 时的严格检查，使得原本的 padding vocab 的逻辑可能失效，添加 --load-padded-vocab-size，直接从 ckpt 中载入 vocab size 从而避免这个报错，注意此时 vocab size 需要能被 tp_size 整除
    + 添加 dispatcher 中的模型切分 size 和 rank 的rank 值，避免在模型转换时报错
## 2025.05.15
+ 添加了 context parallel 的支持，以支持长序列的训练，添加 ulysses 和 kvgather 的 cp 策略；
    + ulysses 策略下，需要保证 cp size 可以被 kv head 整除
    + kv gather 策略，则没有这个限制
+ 更新了 flash-moe 的 whl 包，以支持 cp
## 2025.04.28
+ 添加了 pretrain 对 inbalance vpp 的支持
    + 在使用 vpp 的时候，用 **--allow-padding-num-layers** **--pipeline-layers INBALANCED_LAYERS** 组合来设置 uneven 的流水并行
    + 在不使用 vpp 的时候，用 **--decoder-first-pipeline-num-layers FIRST_PIPE_LAYER_NUM** **--decoder-last-pipeline-num-layers LAST_PIPE_LAYER_NUM**来设置 uneven 的流水并行
+ 添加 **--dist-ckpt-split-model-and-optim** 接口
    + 可以在 torch-dist 格式下提供将 model state 和 optim state 分开保存的功能，在训练大尺寸模型时，可以尽量减少硬盘的使用
+ 修复了一些 typo 以及代码造成的可能的循环引用的报错

## 2025.04.24
+ 添加了 --recompute-attn-layer 接口，支持大尺寸的长序列训练
+ 完善 chunkflow/smart_batching 功能，优化 sft 训练效率
    + chunkflow/smart_batching 都可以做到智能的拼接 batch 以最大化训练效率
    + 在训练 sft 时,推荐开启，strategy 选择 global-batch，在 moe 训练时不建议开启 bucket 功能
    + chunkflow/ smart_batching 开启后在 pp>2 时会出现已知的 hang 问题，需要添加 --dynamic-padding-full，来进行规避，此时会稍微浪费一些 token 用于padding
+ pretrain 的 moe 训练添加 vpp 支持

## 2025.04.23
 + 删除了 qwen3 系列转换的临时方案，现在 qwen3 系列模型的转换直接使用 hf transformers 包里的 qwen3 modeling 文件和 configuration 文件，以适配 model scope 中的文件。
    +  **注意**: transformers>=4.51.0
    + 仍保留了 qwen2 模型的转换，将 convert_xx_to_xx.sh 中的 loader/saver 改为 qwen2_hf 即可
 + 添加 --allow-overwrite 开关，允许在保存 ckpt 的时候覆盖

# Usage
![画板](https://intranetproxy.alipay.com/skylark/lark/0/2025/jpeg/224925/1744951093435-d99b67af-29d9-4013-8e95-865ab7710e92.jpeg)

## 使用Qwen-Public进行SFT微调
这里介绍**30分钟**在八卡80GiB A100/H100上对Qwen3进行微调。

### 准备工作
#### 镜像环境
**单租镜像：**

+ 平湖/上海集群：m6-docker-registry-vpc.cn-shanghai.cr.aliyuncs.com/eflops/lingjun-pytorch-training:2.4-24.07-gu8-gpu-py310-cu125-ubuntu22.04
+ 乌兰C集群：dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/lingjun-pytorch-training:2.4-24.07-gu8-gpu-py310-cu125-ubuntu22.04_accelerated
+ 新加坡集群：qwen-sg-docker-registry-vpc.ap-southeast-1.cr.aliyuncs.com/eflops/lingjun-pytorch-training:2.4-24.07-gu8-gpu-py310-cu125-ubuntu22.04

**多租镜像：**

+ dsw-registry.[region].cr.aliyuncs.com/pai/lingjun-pytorch-training:2.4-24.07-gu8-gpu-py310-cu125-ubuntu22.04

其中 [region] 换成实际的区域，比如 上海就是 cn-shanghai

推荐灵骏集群使用环境变量：

```shell
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_bond
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_MIN_NCHANNELS=4
export NCCL_NET_PLUGIN=none
export ACCL_C4_STATS_MODE=CONN
export ACCL_IB_SPLIT_DATA_NUM=4
export ACCL_IB_QPS_LOAD_BALANCE=1
export ACCL_IB_GID_INDEX_FIX=1
export ACCL_LOG_TIME=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USER=whoami

# 训练 moe 需要额外安装 flash-moe
pip uninstall -y flash-moe
pip install ./flash_moe-0.2.0-cp310-cp310-linux_x86_64.whl
```

+ 训练moe需要额外安装flashmoe：[本地连接](./flash_moe-0.2.0-cp310-cp310-linux_x86_64.whl)

#### 下载权重
从modelscope下载Qwen3权重：

```shell
# 模型存储目录: './Qwen3-4B'
# 安装git git-lfs
sudo apt install git git-lfs
git lfs install

# 下载Qwen3-4B
GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/models/Qwen3-4B/Qwen3-4B-Base.git
cd Qwen3-4B
git lfs pull

# 回到工作目录
cd ..
```

+ 可用模型及模型链接见[Qwen3模型说明]

#### 数据集准备与预处理
下载数据集，并转换数据集格式：

（为了简化整体链路，我们只使用自我认知数据集跑通流程）

Qwen-Public的文本模型微调所需**数据集格式**如下（jsonl）：

```plain
{"type": "chatml", "messages": [{"role": "system", "content": "<system>"}, {"role": "user", "content": "<query>"}, {"role": "assistant", "content": "<response>"}]}
{"type": "chatml", "messages": [{"role": "system", "content": "<system>"}, {"role": "user", "content": "<query>"}, {"role": "assistant", "content": "<response>"}, {"role": "user", "content": "<query2>"}, {"role": "assistant", "content": "<response2>"}]}
```

我们已经对自我认知数据集的格式进行了转换，我们直接下载转换后的数据集。（原始数据集repo查看[这里](https://modelscope.cn/datasets/swift/self-cognition)）

```shell
# jsonl存储位置: ./preprocess/sft_data/self_cognition/self_cognition.jsonl
cd preprocess/sft_data

mkdir self_cognition
cd self_cognition
wget "https://modelscope.cn/datasets/huangjintao/self_cognition_test/resolve/master/self_cognition.jsonl"

# 回到工作目录
cd ../../..
```

接下来，需要将jsonl格式数据集进行预处理，转成**.bin/.idx格式**：

```shell
cd preprocess/sft_data
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# ROOT_DIR文件夹下有self-cognition文件夹
export ROOT_DIR='./'
python binarize.py self_cognition --root-dir $ROOT_DIR --mask-prompt --mask-chatml-role --mask-vision-response --seq-len 8192

# 回到工作目录
cd ../..
```

关于数据**随机**：

1. Pretrain将在packing的时候进行随机，并在之后cache下来。
2. SFT使用的cyclic dataloader会在训练开始时进行随机（具体搜索`MegatronPretrainingRandomSampler`）。数据预处理`binarize`中，不存在随机的逻辑。



### CKPT转换：HF to Mcore
```bash
cd tools

export MODEL_NAME=${MODEL_NAME:-'qwen3-15.6BA2B'} # 指定所需的模型

export INPUT_DIR="../${MODEL_NAME}-hf" # 需要提前下载相关的 hf 版本模型
export TOKENIZER_FILE_PATH="../qwen_tokenizer"
export OUTPUT_DIR="../${MODEL_NAME}"

export TARGET_TP_SIZE=1
export TARGET_EP_SIZE=1
export TARGET_PP_SIZE=1

bash convert_hf_to_mcore.sh

# 回到工作目录
cd ..
```
版本需求：qwen3 模型需要保证 transformers >=4.51.0


### 微调脚本示例
#### Qwen3 4B DENSE 微调
```bash
cd scripts/sft

# 推荐 TP1
export MP_SIZE=1

# 设置ckpt的载入目录
export PRETRAINED_CKPT_ROOT_PATH=../..
export PRETRAINED_CKPT_NAME=qwen3-4B
# 设置ckpt的输出目录
export OUTPUT_CHECKPOINT_PATH=../../output/sft
export PRETRAINED_CKPT_ID=self-cognition

export DATA_PATH=../../preprocess/sft_data/self_cognition
export DATA_CACHE_PATH=../../output/sft/data_cache

# 训练参数（更多训练参数请直接修改shell脚本）
export GLOBAL_BATCH_SIZE=16
export LR=1e-5 MIN_LR=1e-6
export TRAIN_ITERS=100 LR_DECAY_ITERS=60 LR_WARMUP_ITERS=40
export SAVE_INTERVAL=100

bash qwen3_4B_sft_template.sh

# 回到工作目录
cd ../..
```

#### Qwen3 15.6BA2B MOE 微调
```shell
cd scripts/sft

# 推荐 TP2 EP8
export MP_SIZE=2
export EP_SIZE=8

# 设置ckpt的载入目录
export PRETRAINED_CKPT_ROOT_PATH=../..
export PRETRAINED_CKPT_NAME=qwen3-15.6BA2B
# 设置ckpt的输出目录
export OUTPUT_CHECKPOINT_PATH=../../output/sft
export PRETRAINED_CKPT_ID=self-cognition

export DATA_PATH=../../preprocess/sft_data/self_cognition
export DATA_CACHE_PATH=../../output/sft/data_cache

export CHUNKFLOW=true
export CHUNKFLOW_STRATEGY='global-batch'

# 训练参数（更多训练参数请直接修改shell脚本）
export GLOBAL_BATCH_SIZE=16
export LR=1e-5 MIN_LR=1e-6
export TRAIN_ITERS=100 LR_DECAY_ITERS=60 LR_WARMUP_ITERS=40
export SAVE_INTERVAL=100

bash qwen3_moe_15.6BA2B_sft_template.sh

# 回到工作目录
cd ../..
```



Tips：

1. 默认将数据样本**静态padding**到seq-len，超出的样本进行**右裁剪**。
2. 如果需要进行**动态padding**，则需在`qwen3_4B_sft_template`脚本中额外设置`--use-dynamic-collate`，这通常可以加快训练速度。
3. 也可以使用 **Chunkflow** 来进行智能数据拼接，以最大化 sft 的训练效率。
4. 此处超参数仅为跑通脚本的示例，请根据实际数据情况进行调整。



### 推理/评测
#### CKPT转换：MCore to HF
```shell
cd tools

# 存储的训练权重路径
export INPUT_DIR='../output/sft/checkpoint/xxx-4B-bf16-mp1-pp1-lr-1e-5-minlr-1e-6-bs-16-gpus-1-seqlen-32768'

export TOKENIZER_FILE_PATH="../qwen_tokenizer"
# or
# export TOKENIZER_FILE_PATH="../qwen3_chat_tokenizer"

# 导出的权重/tokenizer路径
export OUTPUT_DIR='../output/sft/self-cognition-hf'

bash convert_mcore_to_hf.sh

# 回到工作目录
cd ..
```

最终我们将训练好的权重保存在'output/sft/self-cognition-hf'目录下。

#### 使用HF格式模型CKPT推理
使用HF 的 transformers进行推理：

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = 'output/sft/self-cognition-hf'  # hf格式的checkpoint
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype='auto', device_map='auto')

tokenizer = AutoTokenizer.from_pretrained(model_dir)

query = '你是谁？'
messages = [{
    'role': 'system',
    'content': 'You are a helpful assistant.'
}, {
    'role': 'user',
    'content': query
}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors='pt').to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f'query: {query}')
print(f'response: {response}')

"""输出：
query: 你是谁？
response: 我是一个由云云开发的人工智能助手，被称为坤坤。我主要的目的是通过文本交流为用户提供帮助、信息和娱乐。如果您有任何疑问或者需要帮助，请随时提出，我们会尽力协助您。
"""
```



+ 更多预处理、训练的参数文档查看[这里](./docs/参数含义介绍.md)。



## 使用Qwen-Public做文本模型-预训练
这里将忽略模型下载和格式转换，该部分同SFT的部分。

### 数据准备与预处理
预训练/继续预训练**数据集格式**如下（jsonl）：

```plain
{"text": "<text>"}
{"text": "<text>"}
```

首先我们需要准备若干jsonl文件。这里，我们直接使用repo中预先准备的jsonl文件，存储在`./preprocess/pretrain_data/raw_data_example`文件夹下。

接下来，需要将jsonl格式数据集进行预处理，转成**.bin/.idx格式**：

```shell
pip install python-rapidjson zstandard -i https://mirrors.aliyun.com/pypi/simple/
pip install tiktoken_qwen-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

cd preprocess/pretrain_data
# 可选，将已经预处理好的文件进行删除
# rm -rf processed_data_example

# 处理目录内所有数据并输出为一组数据
python encoder_qwen_15w.py raw_data_example/ --output-prefix processed_data_example/data_source_12 --workers 8 --file-match-str jsonl

# 或处理单个数据
python encoder_qwen_15w.py raw_data_example/data_source_1.jsonl --output-prefix processed_data_example/data_source_1


# 回到工作目录
cd ../..
```



### 预训练-继续预训练示例
这里将忽略模型下载和格式转换，该部分同SFT的部分。

**从头开始预训练：**

```shell
cd scripts/pretrain

# 设置ckpt的载入目录
export PRETRAINED_CKPT_ROOT_PATH=../..
export PRETRAINED_CKPT_NAME=Qwen3-4B

# 设置ckpt的输出目录
export OUTPUT_CHECKPOINT_PATH=../../output/pretrain
export OUTPUT_TENSORBOARD_PATH=$OUTPUT_CHECKPOINT_PATH

export DATA_BLEND_PATH_SHELL=data_blend_example.sh

# 具体训练参数请直接修改shell脚本
bash qwen3_4B_pretrain_template.sh

# 回到工作目录
cd ../..
```

+ 注释1：`data_blend_example.sh`的作用是设置变量`DATA_BLEND`传入`--data-path`，对数据进行混合。
+ 注释2：`DATA_BLEND`变量的格式例如：`1 /path/to/dataset_A 1.5 /path/to/dataset_B`，数字与数据源成对出现，数字表示在实际训练中对应数据源的采样权重（权重会进行归一化处理）。训练过程中框架会构建BlendableDataset，根据权重从不同数据源采样数据构建batch。
+ 注释3：若是断点续训，需要确保`OUTPUT_CHECKPOINT_PATH`指向断点对应的ckpt目录

## 更多示例
### Qwen-Coder 预训练-继续预训练
这里将忽略模型下载和格式转换以及训练部分，该部分同qwen3预训练的部分。

**以继续预训练举例：**

```shell
cd scripts/pretrain

# 设置ckpt的载入目录
export PRETRAINED_CKPT_ROOT_PATH=../..
export PRETRAINED_CKPT_NAME=Qwen3-4B-mcore

# 设置ckpt的输出目录
export OUTPUT_CHECKPOINT_PATH=../../output/pretrain
export OUTPUT_TENSORBOARD_PATH=$OUTPUT_CHECKPOINT_PATH

export DATA_BLEND_PATH_SHELL=data_blend_example.sh

# 具体训练参数请直接修改shell脚本
bash qwen3_coder_4B_cpt_template.sh

# 回到工作目录
cd ../..
```

+ 注释1：动态 FIM 数据比例参考 `--fim-rate` 和 `--fim-spm-rate` 两个参数，0 表示不开 ，1表示 100%。
+ 注释2：`--fim-data-list-path` 指向的文件中列举所有需要进行 FIM 的数据源，以绝对路径一行一列进行列举。

###
### Qwen2-VL SFT训练
#### 下载权重
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

#### 数据集准备与预处理
下载数据集，并转换数据集格式：

（为了简化整体链路，我们使用少量开源的Latex-OCR数据集跑通流程）

Qwen2-VL的微调所需**数据集格式**如下（jsonl）：

```plain
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

#### HF to MCore
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

#### 微调
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

#### MCore to HF
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



#### HF推理
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

###
# FAQ
## Qwen3
### **<font style="color:rgb(64, 64, 64);">模型说明</font>**
<font style="color:rgb(64, 64, 64);">Qwen3 （语言模型，非多模态）系列包含多个 size 的 dense 和 MoE 模型，其中 Dense 模型包括：0.6B、1.7B、4B、8B、14B 和 32B，MoE 模型包括 15B-A2B（内部使用不开源）、30B-A3B 和 235B-A22B（总参数量及激活参数量），后续还会有更大的模型提供。</font>

<font style="color:rgb(64, 64, 64);">内部提供的 ckpt 比外部开源更多，其中包括多个阶段的预训练模型，如 S1 （不开源）、S2（不开源）、Base（Dense 模型开源，MoE 模型不开源）。它们分别为不同的预训练阶段，其中 S1 为第一阶段（约 30 T tokens），S2 为 第二阶段（STEM 和 Coding 占大部分，约 5 T）、Base（退火阶段，打分器筛选的高质量数据以及长序列数据为主，约 200B）。如无 S1，即该模型不存在 S1 阶段，这是因为部分模型是使用蒸馏以及 upscaling 得到的原因。</font>

<font style="color:rgb(64, 64, 64);">Posttrain 模型均对外开源，但内部版本会在开源后持续更新。Qwen3 的 Posttrain 模型较为特殊，实现了 thinking 和 non-thinking 两种模式的混合（即 Qwen 和 QwQ 合二为一），用户可以强制要求模型思考或不思考，也可以在多轮场景中通过输入指令控制，详情请查看 modelcard （</font>[<font style="color:rgb(24, 144, 255);">https://modelscope.cn/models/Qwen3-32B/Qwen3-32B-250415</font>](https://modelscope.cn/models/Qwen3-32B/Qwen3-32B-250415)<font style="color:rgb(64, 64, 64);">）。</font>

<font style="color:rgb(64, 64, 64);">序列长度预训练模型和 Posttrain 模型不一样。其中，S1 和 S2 为 4096，Base 为 32768，Posttrain 模型默认为 40960，使用 YARN 外推可达到 131072。</font>

<font style="color:rgb(64, 64, 64);">不同 size 的模型存在不同的 org 下，具体如下：</font>

<font style="color:rgb(64, 64, 64);">Qwen3-235B-A22B：</font>[<font style="color:rgb(24, 144, 255);">https://modelscope.cn/organization/Qwen3-235B-A22B</font>](https://modelscope.cn/organization/Qwen3-235B-A22B)

<font style="color:rgb(64, 64, 64);">Qwen3-30B-A3B：</font>[<font style="color:rgb(24, 144, 255);">https://modelscope.cn/organization/Qwen3-30B-A3B</font>](https://modelscope.cn/organization/Qwen3-30B-A3B)

<font style="color:rgb(64, 64, 64);">Qwen3-32B：</font>[<font style="color:rgb(24, 144, 255);">https://modelscope.cn/organization/Qwen3-32B</font>](https://modelscope.cn/organization/Qwen3-32B)

<font style="color:rgb(64, 64, 64);">Qwen3-14B：</font>[<font style="color:rgb(24, 144, 255);">https://modelscope.cn/organization/Qwen3-14B</font>](https://modelscope.cn/organization/Qwen3-14B)

<font style="color:rgb(64, 64, 64);">Qwen3-8B：</font>[<font style="color:rgb(24, 144, 255);">https://modelscope.cn/organization/Qwen3-8B</font>](https://modelscope.cn/organization/Qwen3-8B)

<font style="color:rgb(64, 64, 64);">Qwen3-4B：</font>[<font style="color:rgb(24, 144, 255);">https://modelscope.cn/organization/Qwen3-4B</font>](https://modelscope.cn/organization/Qwen3-4B)

<font style="color:rgb(64, 64, 64);">Qwen3-1.7B：</font>[<font style="color:rgb(24, 144, 255);">https://modelscope.cn/organization/Qwen3-1B</font>](https://modelscope.cn/organization/Qwen3-1B)<font style="color:rgba(0, 0, 0, 0.65);"> </font>

<font style="color:rgb(64, 64, 64);">Qwen3-0.6B：</font>[<font style="color:rgb(24, 144, 255);">https://modelscope.cn/organization/Qwen3-0.6B</font>](https://modelscope.cn/organization/Qwen3-0.6B)



### 对比Qwen2系列模型结构改动
+ Dense模型
    - 为了增加训练稳定性引入QK_layernorm
    - 去除了QKV bias

+ MoE模型
    - 为了增加训练稳定性引入QK_layernorm
    - 去除了QKV bias
    - 去除了shared experts
    - 对selected expert的输出分数进行归一化（norm_topk_prob)

### huggingface 格式说明
+ 由于Qwen3模型在外部还未正式发布，目前如果基于hf格式存取有以下两种方式
    - 直接用tools/qwen3_hf(dense)和tools/qwen3_moe_hf内的文件
        * 这一方法需要在hf config中设置model为Qwen2ForCausalLM或Qwen2MoeForCausalLM
    - 直接使用huggingface最新main分支代码（2025.04.02后） 或 [https://github.com/bozheng-hit/transformers/tree/qwen3](https://github.com/bozheng-hit/transformers/tree/qwen3)
        * 这一方法需要在hf config中设置model为Qwen3ForCausalLM或Qwen3MoeForCausalLM
    - 如果基于Qwen-Public进行微调并转换回hf格式，请检查生成config.json是否符合预期或手动进行修改


## 继续预训练/微调 注意事项
+ 如上文所述，Qwen系列预训练包含多个阶段
    - 其中cpt阶段使用rotray_emb_base为**100w**,序列长度**32k**训练；
    - 其他阶段均为rotray_emb_base为**1w**，序列长度**4k**训练；
    - 训练时请注意使用的ckpt与训练脚本中的rotary_emb_base/序列长度是否保持一致


## HEP/EP

EP 和 TP 同时作用时，可能会引起一些歧义这里做一些解释
+ TP 指模型中的 linear tensor 进行切分的维度
+ EP 指将 moe 的按 expert 进行切分的维度

由于 attention 部分的 tp 和 expert 部分的 tp 是可以分别考虑的，所以存在 EP_TP 即，单个 expert 中 linear 被分为多少份，此时又存在两种情况

- 对于 mcore 来说，如果不直接设置 expert_tensor_parallel_size 则使 ep_tp 等于 tp，即共享这个参数
- 在开启 flashmoe hep 后，认为expert_tensor_parallel_size恒等于1

下面用一个最简单的例子进行说明，下图中模型由2个 expert 组成。


可以看到同样是将模型的参数分到2张卡上，不同的 ep/ep/ep_tp 的配置参数分布是不一样的。在 flashmoe 中，对 hep 的情况下对 router 做了特殊优化， hep 的可以等价为 tp_ep 为 1 的特殊情况。注意在这种情况下，同时需要保证 ep 可以被 tp 整除。