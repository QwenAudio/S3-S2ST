#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
workspace_root=$(cd -- "${repo_root}/.." && pwd)

suite_name=${SUITE_NAME:-qwen3_5_4b_s2st_adapter_tts_s2tt_v1}
size_str="gpu${WORLD_SIZE:-1}x${NPROC_PER_NODE:-8}_tp${TP_SIZE:-2}_pp1_cp1_ep1"
stage1_dir="${workspace_root}/artifacts/4b_s2st/experiments/${suite_name}_stage1_adapter_${size_str}"
stage2_dir="${workspace_root}/artifacts/4b_s2st/experiments/${suite_name}_stage2_tts_llm_${size_str}"
stage3_dir="${workspace_root}/artifacts/4b_s2st/experiments/${suite_name}_stage3_s2tt_${size_str}"

run_stage() {
    local stage=$1
    local exp_name=$2
    local exp_dir=$3
    local init_dir=${4:-}
    echo "===== starting ${stage}: ${exp_dir} ====="
    if [[ -n "${init_dir}" ]]; then
        INIT_CKPT_DIR="${init_dir}" EXP_NAME="${exp_name}" EXP_DIR="${exp_dir}" \
            bash "${repo_root}/tools/4b_s2st/run_4b_s2st_stage.sh" "${stage}"
    else
        EXP_NAME="${exp_name}" EXP_DIR="${exp_dir}" \
            bash "${repo_root}/tools/4b_s2st/run_4b_s2st_stage.sh" "${stage}"
    fi
    [[ -s "${exp_dir}/latest_checkpointed_iteration.txt" ]] || {
        echo "ERROR: ${stage} finished without a checkpoint tracker in ${exp_dir}" >&2
        exit 1
    }
}

run_stage adapter "${suite_name}_stage1_adapter" "${stage1_dir}"
run_stage tts "${suite_name}_stage2_tts_llm" "${stage2_dir}" "${stage1_dir}"
run_stage s2tt "${suite_name}_stage3_s2tt" "${stage3_dir}" "${stage2_dir}"

echo "Three-stage S2ST training completed:"
printf '  %s\n' "${stage1_dir}" "${stage2_dir}" "${stage3_dir}"
