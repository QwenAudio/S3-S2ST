# Installation

Use Linux x86_64/CUDA, ffmpeg, libsndfile, cmake, ninja and a C++ compiler. Keep inference, training and metrics in independent environments.

## Inference

Validated base: Python 3.12, NVIDIA PyTorch 2.7.0a0 (25.03), CUDA 12.8, torchaudio 2.6.0a0, Triton 3.2.0 and custom vLLM commit `6ac0cd370316e7c940d237e803b0eb8eda282f01`. This repository supplies custom vLLM Python modules; a compatible installed build supplies compiled kernels. An arbitrary stock wheel is not a validated replacement.

The complete framework code and kernel configurations are retained. The released models use dense BF16 inference.

With a compatible base environment:

```bash
cd "<path_to_S3-S2ST>"
python3.12 -m venv --system-site-packages envs/inference
PIP_CACHE_DIR="$PWD/cache/pip" envs/inference/bin/python -m pip install -e ".[inference]"
source env.sh
envs/inference/bin/python -c 'from vllm.engine.omnidrca2_engine import OmniDRCA2Engine; import onnxruntime'
```

`scripts/infer.sh` chooses `envs/inference` when present. Otherwise activate a compatible environment or set `PYTHON_BIN="<path_to_python>"`. Each run uses its own compiler caches; abstract Unix sockets support shared filesystems.

## Training

```bash
python3.10 -m venv envs/train
source envs/train/bin/activate
# Install CUDA-matched PyTorch/torchaudio first.
pip install -e ".[train]"
# Install matching NVIDIA Apex, Transformer Engine and flash-attn.
```

Use this Megatron fork and compile its fused kernels against the same PyTorch/CUDA. Checkpoint loading needs Zarr 2, numcodecs and tensorstore. Training uses BF16 and sequence length 8192; see [training](TRAINING.md).

## Metrics and validation

```bash
python3.12 -m venv --system-site-packages envs/metrics
envs/metrics/bin/python -m pip install -e ".[eval]"
python tools/check_release.py
```

Static release checks cover syntax, local documentation links, credentials and excluded assets. Model assets and training data are configured separately; see [models](MODELS.md) and [training](TRAINING.md).

## Conda

From the repository root, create one environment for each role. Install the matching CUDA/PyTorch and compiled kernels described above in the environment before installing its dependencies.

```bash
# Inference (Python 3.12)
conda create --prefix ./envs/inference python=3.12 -y
conda activate ./envs/inference
python -m pip install --upgrade pip
# Install matching PyTorch/torchaudio and the compatible compiled vLLM wheel.
python -m pip install -e ".[inference]"
source env.sh

# Training (Python 3.10)
conda create --prefix ./envs/train python=3.10 -y
conda activate ./envs/train
# Install matching PyTorch/torchaudio, Apex, Transformer Engine and flash-attn.
python -m pip install -e ".[train]"

# Evaluation (Python 3.12)
conda create --prefix ./envs/metrics python=3.12 -y
conda activate ./envs/metrics
# Install matching PyTorch/torchaudio.
python -m pip install -e ".[eval]"
```

For the optional dataset tooling, install `python -m pip install -e ".[data]"` in the environment where it is needed. The inference extra uses Transformers 5, while training and evaluation use Transformers 4; install these roles in separate environments.

The editable installation manages dependencies; run the project scripts from this checkout with `env.sh` as described in the README.
