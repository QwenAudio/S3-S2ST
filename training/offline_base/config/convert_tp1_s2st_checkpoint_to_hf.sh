#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
experiment_dir=$(cd -- "${script_dir}/.." && pwd)
repo_root=${S3_ROOT}/code/megatron
workspace_root=$(cd -- "${repo_root}/.." && pwd)

mcore_iter_dir=${1:-}
hf_dir=${2:-}
if [[ -z "${mcore_iter_dir}" ]]; then
    echo "Usage: $0 <iter_NNNNNNN> [output_HF_dir]" >&2
    exit 2
fi
mcore_iter_dir=$(cd -- "${mcore_iter_dir}" && pwd)
if [[ -z "${hf_dir}" ]]; then
    hf_dir="${mcore_iter_dir}_HF"
fi

[[ -s "${mcore_iter_dir}/metadata.json" || -s "${mcore_iter_dir}/.metadata" ]] || {
    echo "ERROR: incomplete checkpoint metadata: ${mcore_iter_dir}" >&2
    exit 1
}
[[ ! -e "${hf_dir}" ]] || {
    echo "ERROR: refusing to overwrite existing output: ${hf_dir}" >&2
    exit 1
}

tokenizer=${S3_ROOT}/models/offline
audio_encoder=/path/to/audio_encoder_only.pt
python_bin=${PYTHON_BIN:-/path/to/python}
native_tts_ckpt=${COSYVOICE_TTS_INIT_CKPT:-/path/to/epoch_276_step_2209.pth}
snapshot_fun_model="${script_dir}/code_snapshot/megatron/core/models/multimodal/fun_model_channels.py"
for required in "${snapshot_fun_model}" "${audio_encoder}" "${native_tts_ckpt}" "${tokenizer}/tokenizer.json"; do
    [[ -s "${required}" ]] || { echo "ERROR: missing ${required}" >&2; exit 1; }
done

overlay_dir=$(mktemp -d /tmp/s2st_tp1_convert.XXXXXX)
tmp_hf="${hf_dir}.tmp.$$"
cleanup() {
    if [[ "${overlay_dir}" == /tmp/s2st_tp1_convert.* && -d "${overlay_dir}" ]]; then
        rm -rf -- "${overlay_dir}"
    fi
    if [[ -e "${tmp_hf}" ]]; then
        mv -- "${tmp_hf}" "${tmp_hf}.failed"
    fi
}
trap cleanup EXIT

# The shared converter reconstructs codec_unit rows, while this TP=1 experiment
# was trained with native CosyVoice/vLLM codec_vocab_size=codec_unit+1=59250.
# Load the experiment snapshot in a local overlay; shared code is never edited.
cp -a "${repo_root}/megatron" "${overlay_dir}/"
install -m 0644 "${snapshot_fun_model}"     "${overlay_dir}/megatron/core/models/multimodal/fun_model_channels.py"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="${EXP_DIR}/runtime:${S3_ROOT}/training/offline/config/code_snapshot:${S3_ROOT}/training/offline_base/config/code_snapshot:${S3_ROOT}/vendor/offline_funasr:${S3_ROOT}/vendor/FunASR:${S3_ROOT}/code/megatron:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export Channels_Model=True
export no_audio_emb_and_head=True
export LOADER_RELEASE_LAYER_AFTER_SEND=1
export SAVER_STREAMING_MODE=1
export ENABLE_COSYVOICE_FM=1
export COSYVOICE_FM_ONLY=0
export COSYVOICE_TTS_INIT_FULL=1
export COSYVOICE_TTS_INIT_CKPT="${native_tts_ckpt}"
export COSYVOICE_FM_INIT_CKPT="${native_tts_ckpt}"

cd "${repo_root}/tools"
"${python_bin}" -m torch.distributed.run --nproc-per-node=1 --nnodes=1     --master-port "${CONVERT_MASTER_PORT:-29681}"     checkpoint/convert.py     --model-type GPT     --loader fun_mcore_qwen3_5_plus     --saver fun_qwen3_5_hf_plus     --max-queue-size 1     --load-dir "${mcore_iter_dir}"     --save-dir "${tmp_hf}"     --shard "${SHARD:-5GB}"     --audio-encoder-ckpt-path "${audio_encoder}"     --megatron-path "${overlay_dir}"

for name in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    cp "${tokenizer}/${name}" "${tmp_hf}/"
done
install -m 0644 "${repo_root}/tools/4b_s2st/inference/config/preprocessor_config.json"     "${tmp_hf}/preprocessor_config.json"
install -m 0644 "${repo_root}/tools/4b_s2st/inference/config/processor_config.json"     "${tmp_hf}/processor_config.json"

for required in config.json model.safetensors.index.json preprocessor_config.json     processor_config.json cosyvoice_fm/model.pt cosyvoice_fm/config.yaml; do
    [[ -s "${tmp_hf}/${required}" ]] || {
        echo "ERROR: converted checkpoint is missing ${required}" >&2
        exit 1
    }
done
grep -q '"codec_unit": 59249' "${tmp_hf}/config.json" || {
    echo "ERROR: unexpected codec_unit in converted config" >&2
    exit 1
}
grep -q '"codec_vocab_size": 59250' "${tmp_hf}/config.json" || {
    echo "ERROR: expected TP=1 codec_vocab_size=59250" >&2
    exit 1
}

"${python_bin}" "${experiment_dir}/scripts/fusion_checkpoint.py" export "${mcore_iter_dir}" "${tmp_hf}"

mv -- "${tmp_hf}" "${hf_dir}"
trap - EXIT
if [[ "${overlay_dir}" == /tmp/s2st_tp1_convert.* && -d "${overlay_dir}" ]]; then
    rm -rf -- "${overlay_dir}"
fi
echo "TP=1 S2ST conversion passed: ${hf_dir}"
