## VL评测

### VLMEvalKit
请参考：https://github.com/open-compass/VLMEvalKit 以下只划重点
#### 第1步 准备环境
```Shell
git clone https://github.com/open-compass/VLMEvalKit
cd VLMEvalKit
pip install -e .
```
#### 第2步 配置
**VLM 配置**：所有 VLMs 都在 `vlmeval/config.py` 中配置。在评估时，你应该使用 `vlmeval/config.py` 中 `supported_VLM` 指定的模型名称来选择 VLM。确保在开始评估之前，你可以成功使用 VLM 进行推理，使用以下命令 `vlmutil check {MODEL_NAME}`。
```Shell
#在vlmeval/config.py中的qwen2vl_series添加一条指向本地finetune模型路径
vi vlmeval/config.py

qwen2vl_series = {
    'qwen2vl-finetune': partial(Qwen2VLChat, model_path='{YOUR_MODEL_PATH}', min_pixels=1280*28*28, max_pixels=16384*28*28),
}
vlmutil check qwen2vl-finetune
```
#### 第3步 评测
（可选）部署本地语言模型作为评判 / 选择提取器，参考官方文档：https://github.com/open-compass/VLMEvalKit
##### 图像评测

```Shell
# 在 MMBench_DEV_EN、MME 和 SEEDBench_IMG 上使用 IDEFICS-80B-Instruct 进行推理和评估
python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model qwen2vl-finetune --verbose
# 在 MMBench_DEV_EN、MME 和 SEEDBench_IMG 上使用 IDEFICS-80B-Instruct 仅进行推理
python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model qwen2vl-finetune --verbose --mode infer
#（如果缺少libGLX.so.0等组件，尝试安装OpenCL解决）
```

##### 视频评测

```Shell
# 在 MMBench-Video 上评测 IDEFCIS2-8B, 视频采样 8 帧作为输入，不采用 pack 模式评测. MMBench_Video_8frame_nopack 是一个定义在 `vlmeval/dataset/video_dataset_config.py` 的数据集设定.
torchrun --nproc-per-node=8 run.py --data MMBench_Video_8frame_nopack --model qwen2vl_finetune
```
评估结果将作为日志打印出来。此外，**结果文件**也会在目录 `$YOUR_WORKING_DIRECTORY/{model_name}` 中生成。以 `.csv` 结尾的文件包含评估的指标。

