# S3-S2ST: Semantically Supervised Streaming Speech-to-Speech Translation

[![arXiv](https://img.shields.io/badge/arXiv-2610.36777-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2610.36777)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/zhaoyx39/S3-S2ST)
[![ModelScope](https://img.shields.io/badge/ModelScope-Models-624AFF?style=flat-square)](https://modelscope.cn/models/zhaoyx239/S3-S2ST)

[中文](README.zh-CN.md)

## Abstract

Streaming speech-to-speech translation requires timely decisions about when partial input is sufficient for accurate translation. We present S3-S2ST, a model that learns to translate and speak from semantically aligned streaming supervision. By converting utterance-level speech pairs into incremental training examples, we teach a learned policy when to commit translations. Its adjustable threshold controls the quality–latency tradeoff on a fixed 640 ms input grid. S3-S2ST delivers competitive English–Chinese translation on FLEURS and leads the evaluated streaming systems in translation quality on the internal 9pair test set. On RealSI, adapting the threshold to buffered speech further improves quality over a low fixed threshold with little change in first-audio latency. We plan to release the model and the training, inference, and data-construction code.

![Figure 2: S3-S2ST architecture](docs/assets/fig2.png)

A chunkwise encoder and Thinker support translation and READ/WRITE decisions. Eight-layer fusion conditions the Talker; flow matching and a vocoder generate speech.

## Installation

### Clone and system dependencies

Use Linux x86_64 with an NVIDIA GPU and a CUDA-compatible driver.

```bash
git clone https://github.com/QwenAudio/S3-S2ST.git
cd S3-S2ST
sudo apt-get update
sudo apt-get install -y git ffmpeg libsndfile1 cmake ninja-build build-essential python3-venv
```

### Inference environment

Dependencies are defined in `pyproject.toml`, with `inference`, `train`, `eval` and `data` extras. For Conda setup, see the [environment guide](docs/ENVIRONMENT.md).

The reference environment uses Python 3.12, NVIDIA PyTorch 2.7.0a0 (25.03), CUDA 12.8, torchaudio 2.6.0a0 and Triton 3.2.0. Use the NVIDIA PyTorch 25.03 container or a compatible CUDA/PyTorch base. The bundled vLLM Python code requires compiled vLLM kernels compatible with commit `6ac0cd370316e7c940d237e803b0eb8eda282f01`.

```bash
python3.12 -m venv --system-site-packages envs/inference
envs/inference/bin/python -m pip install --upgrade pip
# Install your compatible compiled vLLM wheel before the Python dependencies.
envs/inference/bin/python -m pip install "<path_to_compatible_vllm.whl>"
PIP_CACHE_DIR="$PWD/cache/pip" envs/inference/bin/python -m pip install -e ".[inference]"
source env.sh
envs/inference/bin/python -c 'from vllm.engine.omnidrca2_engine import OmniDRCA2Engine; import onnxruntime'
```

### Training and evaluation environments

Install CUDA-matched PyTorch/torchaudio, NVIDIA Apex, Transformer Engine and flash-attn in the training environment. Compile the bundled Megatron fused kernels against that environment.

```bash
python3.10 -m venv envs/train
envs/train/bin/python -m pip install -e ".[train]"

python3.12 -m venv --system-site-packages envs/metrics
envs/metrics/bin/python -m pip install -e ".[eval]"
```

The launchers select their corresponding inference or metrics environment automatically. Activate `envs/train` for training. Set `PYTHON_BIN="<path_to_python>"` to use a different interpreter.

## Inference

### Model setup

Download and install either model from Hugging Face:

```bash
envs/inference/bin/python tools/download_model.py --mode streaming --repo-id zhaoyx39/S3-S2ST --subfolder streaming
envs/inference/bin/python tools/download_model.py --mode offline --repo-id zhaoyx39/S3-S2ST --subfolder offline
# For files downloaded manually, replace --repo-id with --local-dir "<path_to_model_download>".
```

Alternatively, download from ModelScope and use the same installer:

```bash
envs/inference/bin/python -m pip install modelscope
envs/inference/bin/modelscope download --model zhaoyx239/S3-S2ST --include "streaming/**" --local_dir cache/model_downloads/modelscope
envs/inference/bin/python tools/download_model.py --mode streaming --local-dir cache/model_downloads/modelscope --subfolder streaming
envs/inference/bin/modelscope download --model zhaoyx239/S3-S2ST --include "offline/**" --local_dir cache/model_downloads/modelscope
envs/inference/bin/python tools/download_model.py --mode offline --local-dir cache/model_downloads/modelscope --subfolder offline
```

One model repository contains `streaming/` and `offline/` variants. Each subdirectory includes its matching 4B weights, Talker configuration and rendering assets. The installer verifies checksums and places files under `models/`; no external model paths are needed. Pin `--revision "<hf_commit>"` to select an exact release. For verification without installation, add `--verify-only`.

The two releases are named `streaming` and `offline`, without step or iteration suffixes. Place the bundles and shared assets inside the repository:

```text
models/
  streaming/              # HF export, audio_encoder.pt, streaming_gate.pt, fusion weights
  offline/                # HF export and audio_encoder.pt
  speaker/campplus.onnx
  flow_matching/          # shared flow-matching weights
  vocoder/                # shared vocoder weights and configuration
  frontend/               # mel_filters.npz and fm_mel_filters.npz
  training/talker_native.pth
checkpoints/
  init_streaming_talker/   # Megatron checkpoint parent and iteration tracker
  init_streaming_e2e/
  init_offline/
```

### Translate an audio file

```bash
envs/inference/bin/python tools/make_input.py --audio "<path_to_input.wav>" --direction zh2en --output data/input.jsonl
bash scripts/infer.sh --mode streaming --preset adaptive --model "<path_to_streaming_model>" --input data/input.jsonl --no-warmup --output runs/example --gpu 0
```

Use `--direction en2zh` for English-to-Chinese translation. For offline inference, use `--mode offline --model "<path_to_offline_model>"` and omit `--preset`. Streaming supports `fixed-0.5`, `fixed-0.1` and `adaptive`. Audio, model and output paths resolve inside the repository; use a new output directory for each run. Outputs include `raw.jsonl`, generated audio and codec tokens.

### Evaluation

See the [evaluation guide](docs/EVALUATION.md) for FLEURS preparation, scoring, optional audio gain preprocessing and parallel inference.

## Training

### Prepare data

Supply licensed bilingual audio, semantic chunks and word alignments. The original private training corpus is not distributed.

```bash
source envs/train/bin/activate
python training/data_tools/build_interleaved_data.py --direction zh2en --mapping "<path_to_mapping.jsonl>" --text-labels "<path_to_semantic_chunks.jsonl>" --word-alignments "<path_to_word_alignments.jsonl>" --output-dir data/train/zh2en --sqlite cache/zh2en.sqlite
python training/data_tools/pack_streaming_s2tt.py --input-list "<path_to_generated_data.list>" --output-dir data/train/packed --funasr-root vendor/FunASR --tokenizer-model models/offline --cutoff-len 8192
```

Repeat for `en2zh`. The packer builds streaming S2TT examples; S2ST additionally needs aligned target speech and precomputed FSQ codec slices. See the schema in `training/streaming/talker/runtime/exact_codec_loader.py`: 12.5 Hz codec frames, blank count 5, BOS 59049 and EOS 59050. Each `data.list` contains one local packed JSONL shard path per line. Keep data and media dependencies inside the repository.

### Run training

```bash
# Streaming: 8 GPUs, Talker adaptation followed by end-to-end training.
bash scripts/train.sh --stage streaming-talker --train-list "<path_to_train_data.list>" --valid-list "<path_to_valid_data.list>" --output runs/train_talker --steps 5000
bash scripts/train.sh --stage streaming-e2e --train-list "<path_to_train_data.list>" --valid-list "<path_to_valid_data.list>" --output runs/train_e2e --steps 4000

# Offline: 4 nodes × 8 GPUs; set NODE_RANK=0..3 on the corresponding nodes.
NNODES=4 NODE_RANK=0 MASTER_ADDR="<node0_address>" MASTER_PORT=12981 bash scripts/train.sh --stage offline --train-list "<path_to_offline_data.list>" --output runs/train_offline --steps 10000
```

Each stage defaults to its initializer under `checkpoints/`. Pass `--init "<path_to_checkpoint_parent>"` to select another initialization, including the preceding stage's output. Use `--prepare-only` to generate stage configuration before training. Offline nodes share the same project, data and output paths.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zhao2026s3s2st,
  title={{S3-S2ST}: Semantically Supervised Streaming Speech-to-Speech Translation},
  author={Yuxiang Zhao and Zhiyun Fan and Yu Pu and Yanjie An and Yushen Chen and
          Wenrui Liang and Baiji Liu and Yunchong Xiao and Zhaoci Liu and Keqi Deng and
          Kai Yu and Zhifu Gao and Xiangang Li and Xie Chen},
  year={2026},
  eprint={2610.36777},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2610.36777}
}
```

## License

Original project code and documentation use MIT. Third-party code, models and datasets retain their own licenses; see `LICENSE` and `THIRD_PARTY_NOTICES.md`.
