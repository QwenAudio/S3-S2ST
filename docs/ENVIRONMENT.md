# Installation

Use Linux x86_64/CUDA, ffmpeg, libsndfile, cmake, ninja and a C++ compiler. Keep inference, training and metrics in independent environments.

## Inference

Use the NVIDIA base image `nvcr.io/nvidia/pytorch:25.03-py3` on Linux x86_64 with an NVIDIA GPU and NVIDIA Container Toolkit:

```bash
cd "<path_to_S3-S2ST>"
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/workspace/S3-S2ST -w /workspace/S3-S2ST \
  nvcr.io/nvidia/pytorch:25.03-py3 bash
```

The reference runtime uses:

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | `2.7.0a0+7c8ec84dab.nv25.03` |
| CUDA toolkit | 12.8 |
| torchaudio | `2.6.0a0` |
| Triton | 3.2.0 |
| Transformers | 5.3.0 |
| FlashInfer | 0.4.0 |
| S3-S2ST vLLM wheel | `0.11.1+s3.1.cu128` |

The wheel packages this repository's vLLM Python snapshot and the three matching native extensions from the reference runtime. `0.11.1+s3.1.cu128` identifies this S3-S2ST runtime release. The companion FlashInfer and TVM FFI wheels preserve the reference builds. Third-party licenses are included. Models use dense BF16 inference.

Download the following assets from [Runtime v1](https://github.com/QwenAudio/S3-S2ST/releases/tag/runtime-v1) into `cache/vllm_release/`:

- `vllm-0.11.1+s3.1.cu128-cp312-cp312-linux_x86_64.whl`
- `flashinfer_python-0.4.0-py3-none-any.whl`
- `apache_tvm_ffi-0.1.0b15-cp312-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`
- `inference-runtime.txt`
- `SHA256SUMS`

Download and install the runtime from the repository root inside the container:

```bash
RUNTIME="$PWD/cache/vllm_release"
mkdir -p "$RUNTIME"
RELEASE="https://github.com/QwenAudio/S3-S2ST/releases/download/runtime-v1"
for file in \
  "vllm-0.11.1+s3.1.cu128-cp312-cp312-linux_x86_64.whl" \
  "flashinfer_python-0.4.0-py3-none-any.whl" \
  "apache_tvm_ffi-0.1.0b15-cp312-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl" \
  "inference-runtime.txt" "SHA256SUMS"; do
  curl -fL --retry 3 "$RELEASE/$file" -o "$RUNTIME/$file"
done
(cd "$RUNTIME" && sha256sum -c SHA256SUMS)
python3.12 -m venv --system-site-packages envs/inference
envs/inference/bin/python -m pip install --no-deps -r "$RUNTIME/inference-runtime.txt"
envs/inference/bin/python -m pip install --no-deps "$RUNTIME"/*.whl
envs/inference/bin/python -m pip install --no-deps -e .
source env.sh
envs/inference/bin/python -c 'from vllm.engine.omnidrca2_engine import OmniDRCA2Engine; import onnxruntime'
```

`inference-runtime.txt` pins the inference dependencies separately from the container's PyTorch/CUDA stack. `--no-deps` preserves that stack and the recorded versions. FunASR comes from `vendor/FunASR`; structured-generation-only XGrammar is not needed for the speech inference commands.

`scripts/infer.sh` chooses `envs/inference` when present. Otherwise set `PYTHON_BIN="<path_to_python>"`.

To package the same runtime from matching installed native extensions:

```bash
python tools/build_vllm_wheel.py \
  --native-dir "<path_to_installed_vllm>" \
  --output-dir "<path_to_release_directory>"
```

The packager verifies the native extension SHA256 values and copies the Python snapshot without changing its code. Rebuilding the native extensions themselves requires their corresponding C++/CUDA source.

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

For inference, use the container and venv installation above to retain the reference PyTorch build. For training and evaluation, Conda can manage separate environments. Install the matching CUDA/PyTorch and compiled kernels before installing each role’s dependencies.

```bash
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
