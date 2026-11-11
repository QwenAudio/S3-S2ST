#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mcore_iter_dir=${1:-}
hf_dir=${2:-}
native_tts_ckpt=${COSYVOICE_TTS_INIT_CKPT:-/path/to/epoch_276_step_2209.pth}
if [[ -z "${mcore_iter_dir}" || -z "${hf_dir}" ]]; then
    echo "Usage: $0 <experiment/iter_NNNNNNN> <output_hf_dir>" >&2
    exit 2
fi

# The three-stage schedule deliberately omits FM from its trainable checkpoint.
# Initialize the fixed native TTS/FM state before the MCore checkpoint is loaded;
# trained adapter/TTS-LLM/S2TT tensors then overwrite their initialized values.
# This prevents the converter from exporting a freshly randomized FM module.
[[ -s "${native_tts_ckpt}" ]] || {
    echo "ERROR: missing native TTS/FM initialization: ${native_tts_ckpt}" >&2
    exit 1
}
export ENABLE_COSYVOICE_FM=1
export COSYVOICE_FM_ONLY=0
export COSYVOICE_TTS_INIT_FULL=1
export COSYVOICE_TTS_INIT_CKPT="${native_tts_ckpt}"
export COSYVOICE_FM_INIT_CKPT="${native_tts_ckpt}"

bash "${repo_root}/tools/4b_s2st/convert_iter1600_to_hf.sh" "${mcore_iter_dir}" "${hf_dir}"

# The checkpoint converter saves model/tokenizer state, while the custom S2S
# runtime also resolves its audio frontend through AutoProcessor.  Keep these
# small configs in this repository so a converted checkpoint is self-contained
# and does not depend on the old qwen3.5-4B reproduction directory.
processor_config_dir="${repo_root}/tools/4b_s2st/inference/config"
install -m 0644 "${processor_config_dir}/preprocessor_config.json" "${hf_dir}/preprocessor_config.json"
install -m 0644 "${processor_config_dir}/processor_config.json" "${hf_dir}/processor_config.json"

for required in "${hf_dir}/config.json" "${hf_dir}/model.safetensors.index.json" "${hf_dir}/preprocessor_config.json" "${hf_dir}/processor_config.json" "${hf_dir}/cosyvoice_fm/model.pt" "${hf_dir}/cosyvoice_fm/config.yaml"; do
    [[ -s "${required}" ]] || {
        echo "ERROR: converted S2ST checkpoint is missing ${required}" >&2
        exit 1
    }
done
echo "S2ST conversion validation passed: ${hf_dir}"
