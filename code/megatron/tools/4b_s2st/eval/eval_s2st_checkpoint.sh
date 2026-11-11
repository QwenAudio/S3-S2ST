#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
checkpoint=${1:-}
output_root=${2:-}
if [[ -z "${checkpoint}" ]]; then
  echo "Usage: $0 <experiment-dir|iter_NNNNNNN|converted-HF-dir> [output-root]" >&2
  exit 2
fi
cd "${repo_root}"

if [[ -f "${checkpoint}/latest_checkpointed_iteration.txt" ]]; then
  iteration=$(<"${checkpoint}/latest_checkpointed_iteration.txt")
  iter_name=$(printf 'iter_%07d' "${iteration}")
  mcore_dir="${checkpoint}/${iter_name}"
elif [[ -f "${checkpoint}/metadata.json" && $(basename "${checkpoint}") == iter_* ]]; then
  mcore_dir="${checkpoint}"
  iter_name=$(basename "${checkpoint}")
else
  mcore_dir=
  iter_name=$(basename "${checkpoint}")
fi

if [[ -f "${checkpoint}/config.json" ]]; then
  hf_dir="${checkpoint}"
elif [[ -n "${mcore_dir}" ]]; then
  hf_dir="${mcore_dir}_HF"
  if [[ ! -s "${hf_dir}/config.json" || ! -s "${hf_dir}/model.safetensors.index.json" ]]; then
    [[ ! -e "${hf_dir}" ]] || {
      echo "ERROR: incomplete HF directory already exists: ${hf_dir}" >&2
      exit 1
    }
    tmp_hf="${hf_dir}.tmp.$$"
    trap '[[ -n "${tmp_hf:-}" && -e "${tmp_hf}" ]] && mv "${tmp_hf}" "${tmp_hf}.failed"' EXIT
    bash tools/4b_s2st/convert_s2st_checkpoint_to_hf.sh "${mcore_dir}" "${tmp_hf}"
    mv "${tmp_hf}" "${hf_dir}"
    tmp_hf=
    trap - EXIT
  fi
else
  echo "ERROR: cannot recognize checkpoint type: ${checkpoint}" >&2
  exit 2
fi

if [[ -z "${output_root}" ]]; then
  if [[ -n "${mcore_dir}" ]]; then
    experiment_name=$(basename "$(dirname "${mcore_dir}")")
    eval_name="${experiment_name}_${iter_name}_fast"
  else
    eval_name="${iter_name}_fast"
  fi
  output_root="/path/to/_{eval_name}"
fi

echo "checkpoint=${checkpoint}"
echo "hf_dir=${hf_dir}"
echo "output_root=${output_root}"
echo "profile=${EVAL_PROFILE:-full}"
exec bash tools/4b_s2st/eval/run_9pair_s2st_eval_fast.sh "${hf_dir}" "${output_root}"

