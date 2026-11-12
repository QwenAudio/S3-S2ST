# 评测与并行推理

[返回 README](../README.zh-CN.md) · [English](EVALUATION.md)

先按照主 README 安装环境并下载模型。以下命令均在代码仓库根目录运行。

## FLEURS 数据准备与评测

```bash
envs/inference/bin/python tools/prepare_fleurs.py
bash scripts/infer.sh --mode streaming --preset adaptive --input data/fleurs/test_all.jsonl --output runs/fleurs_stream --gpu 0
bash scripts/evaluate.sh --run fleurs_stream --gpu 0 --full --skip-comet
envs/metrics/bin/python tools/report.py --run runs/fleurs_stream
```

准备脚本下载官方元数据与音频，校验 SHA256，构建论文子集：945 条中译英和 646 条英译中。离线服务器先将下载文件放入 `cache/downloads/fleurs_hf/`；文件名和哈希见 `configs/fleurs_subset.json` 与 `configs/fleurs_archives.json`。

评测模型分别放入 `evaluation/third_party/cer_tool_parallel/models/`、`evaluation/third_party/torch_cache/` 和 `evaluation/voice_eval/thirdparty/3D-Speaker/pretrained/`。`--skip-comet` 跳过 COMET 和 ASR-COMET；需要这两个指标时，获取官方 XCOMET-XL 权重，运行 `tools/prepare_assets.py` 并去掉该参数。得分写入 `quality/summary.tsv`，按两个方向等权平均。

可选的整条音频增益预处理与并行流式推理：

```bash
envs/inference/bin/python tools/boost_fleurs.py "<path_to_fleurs_audio>" --output-dir "<path_to_new_audio_directory>"
bash scripts/infer_parallel.sh --output runs/fleurs_parallel --gpus 0,1 --workers-per-gpu 2 --skip-metrics
```

并行入口自动应用增益：RMS 目标 0.0743324953、峰值上限 0.95，仅放大。使用 `fixed-0.5`、FM8，每个引擎分配 1 GiB KV 缓存。单 worker 入口使用原音频。增益预处理是独立于论文结果的可选流程；并发时延反映吞吐场景，单请求时延应在无竞争 worker 时测量。单独预处理音频后，需要更新输入清单中的音频路径。
