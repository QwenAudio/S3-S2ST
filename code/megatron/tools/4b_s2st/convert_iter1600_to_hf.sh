#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
workspace_root=$(cd -- "${repo_root}/.." && pwd)
mcore_dir=${1:-/path/to/iter_0001600}
hf_dir=${2:-${workspace_root}/artifacts/4b_s2st/validation/iter_0001600_HF}
tokenizer=${S3_ROOT}/models/offline
audio_encoder=/path/to/audio_encoder_only.pt
python_bin=${PYTHON_BIN:-/path/to/python}

if [[ ! -e "${mcore_dir}/.metadata" && ! -e "${mcore_dir}/metadata.json" ]]; then
    echo "ERROR: missing checkpoint metadata in ${mcore_dir}" >&2
    exit 1
fi
for required in "${tokenizer}/tokenizer.json" "${audio_encoder}"; do
    [[ -e "${required}" ]] || { echo "ERROR: missing ${required}" >&2; exit 1; }
done
if [[ -e "${hf_dir}" ]]; then
    echo "ERROR: refusing to overwrite ${hf_dir}" >&2
    exit 1
fi
mkdir -p "$(dirname "${hf_dir}")"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="${workspace_root}/deps/python:${workspace_root}/deps/FunASR_0611:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export Channels_Model=True
export no_audio_emb_and_head=True
export LOADER_RELEASE_LAYER_AFTER_SEND=1
export SAVER_STREAMING_MODE=1

cd "${repo_root}/tools"
"${python_bin}" -m torch.distributed.run --nproc-per-node=1 --nnodes=1 \
    --master-port "${CONVERT_MASTER_PORT:-29581}" \
    checkpoint/convert.py \
    --model-type GPT \
    --loader fun_mcore_qwen3_5_plus \
    --saver fun_qwen3_5_hf_plus \
    --max-queue-size 1 \
    --load-dir "${mcore_dir}" \
    --save-dir "${hf_dir}" \
    --shard "${SHARD:-5GB}" \
    --audio-encoder-ckpt-path "${audio_encoder}" \
    --megatron-path "${repo_root}"

for name in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    cp "${tokenizer}/${name}" "${hf_dir}/"
done
[[ -s "${hf_dir}/model.safetensors.index.json" ]] || { echo "ERROR: missing HF index" >&2; exit 1; }
[[ -s "${hf_dir}/config.json" ]] || { echo "ERROR: missing HF config" >&2; exit 1; }
echo "4B conversion passed: ${hf_dir}"
