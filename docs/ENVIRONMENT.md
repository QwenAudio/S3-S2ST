# Installation

Use Linux x86_64/CUDA, ffmpeg, libsndfile, cmake, ninja and a C++ compiler. Keep inference, training and metrics in independent environments.

## Inference

Validated base: Python 3.12, NVIDIA PyTorch 2.7.0a0 (25.03), CUDA 12.8, torchaudio 2.6.0a0, Triton 3.2.0 and custom vLLM commit `6ac0cd370316e7c940d237e803b0eb8eda282f01`. This repository supplies custom vLLM Python modules; a compatible installed build supplies compiled kernels. An arbitrary stock wheel is not a validated replacement.

The complete framework code and kernel configurations are retained. The released models use dense BF16 inference.

With a compatible base environment:

```bash
cd "<path_to_S3-S2ST>"
python3.12 -m venv --system-site-packages envs/inference312
PIP_CACHE_DIR="$PWD/cache/pip" envs/inference312/bin/python -m pip install -r requirements-inference.txt
source env.sh
envs/inference312/bin/python -c 'from vllm.engine.omnidrca2_engine import OmniDRCA2Engine; import onnxruntime'
```

`scripts/infer.sh` chooses `envs/inference312` when present. Otherwise activate a compatible environment or set `PYTHON_BIN="<path_to_python>"`. Each run uses its own compiler caches; abstract Unix sockets support shared filesystems.

## Training

```bash
python3.10 -m venv envs/train
source envs/train/bin/activate
# Install CUDA-matched PyTorch/torchaudio first.
pip install -r requirements-training.txt
# Install matching NVIDIA Apex, Transformer Engine and flash-attn.
```

Use this Megatron fork and compile its fused kernels against the same PyTorch/CUDA. Checkpoint loading needs Zarr 2, numcodecs and tensorstore. Training uses BF16 and sequence length 8192; see [training](TRAINING.md).

## Metrics and validation

```bash
python3.12 -m venv --system-site-packages envs/metrics312
envs/metrics312/bin/python -m pip install -r requirements-metrics.txt
python tools/check_release.py
```

Static release checks cover syntax, local documentation links, credentials and excluded assets. Model assets and training data are configured separately; see [models](MODELS.md) and [training](TRAINING.md).
