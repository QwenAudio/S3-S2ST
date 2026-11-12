#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
workspace_root=$(cd -- "${repo_root}/.." && pwd)
hf_dir=${1:-${workspace_root}/artifacts/4b_s2st/validation/model}
output_dir=${2:-${workspace_root}/artifacts/4b_s2st/validation/inference}
source_reproduction=/path/to/qwen3.5-4B

if [[ ! -s "${hf_dir}/model.safetensors.index.json" ]]; then
    bash "${repo_root}/tools/4b_s2st/export_checkpoint.sh" \
        /path/to/checkpoint \
        "${hf_dir}"
fi
if [[ -e "${output_dir}" ]]; then
    echo "ERROR: refusing to overwrite ${output_dir}" >&2
    exit 1
fi

# Reuse the already-vendored evaluator and immutable six-direction FLEURS list;
# all predictions and logs are written under the project workspace.
RUN_METRIC=${RUN_METRIC:-false} \
    bash "${source_reproduction}/reproduce/run_inference_and_eval.sh" \
    "${hf_dir}" \
    "${source_reproduction}/data/test_s2tt.fleurs.monitor.list.6" \
    "${output_dir}"

expected=0
actual=0
while IFS= read -r jsonl; do
    [[ -n "${jsonl}" && "${jsonl}" != \#* ]] || continue
    expected=$((expected + $(wc -l < "${jsonl}")))
done < "${source_reproduction}/data/test_s2tt.fleurs.monitor.list.6"
while IFS= read -r text_file; do
    actual=$((actual + $(wc -l < "${text_file}")))
done < <(find "${output_dir}" -path '*/1best_recog/text' -type f | sort)
[[ "${actual}" -eq "${expected}" ]] || {
    echo "ERROR: inference rows ${actual}, expected ${expected}" >&2
    exit 1
}
echo "4B inference validation passed: ${actual}/${expected} predictions"
