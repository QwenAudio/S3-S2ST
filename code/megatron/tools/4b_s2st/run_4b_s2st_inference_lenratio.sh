#!/usr/bin/env bash
set -euo pipefail

nofile_limit=${INFER_NOFILE_LIMIT:-65536}
current_nofile_limit=$(ulimit -Sn)
hard_nofile_limit=$(ulimit -Hn)
if [[ "${nofile_limit}" =~ ^[0-9]+$ ]] && (( nofile_limit > current_nofile_limit )); then
    if [[ "${hard_nofile_limit}" != unlimited ]] && (( nofile_limit > hard_nofile_limit )); then
        nofile_limit=${hard_nofile_limit}
    fi
    ulimit -Sn "${nofile_limit}"
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
workspace_root=$(cd -- "${repo_root}/.." && pwd)
infer_dir=${repo_root}/tools/4b_s2st/inference

hf_dir=${1:-}
input_list=${2:-}
output_list=${3:-}
if [[ -z "${hf_dir}" || -z "${input_list}" || -z "${output_list}" ]]; then
    echo "Usage: $0 <converted_hf_dir> <input_jsonl.list> <output_dir.list>" >&2
    exit 2
fi

vocoder_ckpt=${COSYVOICE_VOCODER_CKPT:-${S3_ROOT}/models/vocoder}
infer_python=${INFER_PYTHON_BIN:-/usr/local/bin/python}
wav_python=${CODEX_TO_WAV_PYTHON_BIN:-/path/to/python}
wav_pythonpath=${CODEX_TO_WAV_PYTHONPATH:-/path/to/rotary_only_site}
funasr_root=${FUNASR_ROOT:-${workspace_root}/deps/FunASR_0611}

use_codec_llm_hidden_for_fm=${USE_CODEC_LLM_HIDDEN_FOR_FM:-0}
case "${use_codec_llm_hidden_for_fm,,}" in
    1|true|yes|on)
        use_codec_llm_hidden_for_fm=1
        fm_ckpt=${COSYVOICE_FM_CKPT:-${hf_dir}/cosyvoice_fm/model.pt}
        ;;
    0|false|no|off)
        use_codec_llm_hidden_for_fm=0
        fm_ckpt=${COSYVOICE_FM_CKPT:-${S3_ROOT}/models/flow_matching/epoch_2_step_780000.pt}
        ;;
    *)
        echo "ERROR: invalid USE_CODEC_LLM_HIDDEN_FOR_FM=${use_codec_llm_hidden_for_fm}" >&2
        exit 2
        ;;
esac

for required in "${hf_dir}/config.json" "${input_list}" "${output_list}" "${fm_ckpt}" "${vocoder_ckpt}"; do
    [[ -e "${required}" ]] || { echo "ERROR: missing ${required}" >&2; exit 1; }
done

export PYTHONPATH="${infer_dir}:${workspace_root}/deps/python:${funasr_root}:${PYTHONPATH:-}"
export VLLM_COSYVOICE_EXPORT_LLM_HIDDEN_FOR_FM=${use_codec_llm_hidden_for_fm}
export VLLM_EXPORT_CODEX_LLM_HIDDEN=${use_codec_llm_hidden_for_fm}
export USE_CODEC_LLM_HIDDEN_FOR_FM=${use_codec_llm_hidden_for_fm}
export OMNIDRCA2_MODEL_DIR="${hf_dir}"

talker_input_mode=${TALKER_INPUT_MODE:-hidden}
case "${talker_input_mode,,}" in
    hidden)
        unset VLLM_CODEX_USE_TEXT_TOKEN_INPUT
        ;;
    thinker_text)
        export VLLM_CODEX_USE_TEXT_TOKEN_INPUT=1
        ;;
    *)
        echo "ERROR: TALKER_INPUT_MODE must be hidden or thinker_text, got ${talker_input_mode}" >&2
        exit 2
        ;;
esac
echo "Talker input mode: ${talker_input_mode}"

extra_args=()
if [[ "${USE_OSS_CACHE:-0}" == 1 ]]; then
    extra_args+=(--use-oss-cache)
fi
# The current 0803 vLLM image's piecewise CUDA-graph profiler assumes the MoE
# call signature and crashes on the dense 4B GatedDeltaNet graph.  Eager mode
# is the reliable default for this adapted model; opt out only after the image
# gains a dense-compatible compiler path.
if [[ "${INFER_ENFORCE_EAGER:-1}" == 1 ]]; then
    extra_args+=(--enforce-eager)
fi

cd "${infer_dir}"
infer_cmd=("${infer_python}" infer_jsonllist_new_s2s_lenratio.py "${hf_dir}" "${input_list}" "${output_list}")
infer_cmd+=(--njobs "${INFER_NJOBS:-2}")
infer_cmd+=(--thinker-devices "${THINKER_DEVICES:-[0,1,2,3,4,5,6,7]}")
infer_cmd+=(--talker-devices "${TALKER_DEVICES:-[0]}")
infer_cmd+=(--thinker-gpu-memory-utilization "${THINKER_GPU_MEMORY_UTILIZATION:-0.45}")
infer_cmd+=(--talker-gpu-memory-utilization "${TALKER_GPU_MEMORY_UTILIZATION:-0.15}")
infer_cmd+=(--max-tokens "${MAX_TOKENS:-2048}")
if [[ "${talker_input_mode,,}" == thinker_text ]]; then
    infer_cmd+=(--use-thinker-text-for-talker)
fi
if [[ "${ALLOW_TOKEN_IDS_CODEC_FALLBACK:-0}" == 1 ]]; then
    infer_cmd+=(--allow-token-ids-codec-fallback)
fi
if [[ "${use_codec_llm_hidden_for_fm}" == 1 ]]; then
    infer_cmd+=(--save-codec-llm-hidden)
fi
infer_cmd+=("${extra_args[@]}")
if [[ "${RUN_MODEL_INFERENCE:-1}" == 1 ]]; then
    "${infer_cmd[@]}"
fi

decode_one_output() {
    local jsonl=$1 out_dir=$2
    [[ -n "${jsonl}" && -n "${out_dir}" ]] || return 0
    local -a wav_args=()
    if [[ "${USE_OSS_CACHE:-0}" == 1 ]]; then
        wav_args+=(--use-oss-cache)
    fi
    local -a wav_cmd_base=("${wav_python}" "${infer_dir}/codex_to_wav.py")
    wav_cmd_base+=(--codec-dir "${out_dir}/1best_recog/codec")
    if [[ "${use_codec_llm_hidden_for_fm}" == 1 ]]; then
        wav_cmd_base+=(--llm-hidden-dir "${out_dir}/1best_recog/codec_llm_hidden")
    fi
    wav_cmd_base+=(--output-dir "${out_dir}/1best_recog/s2st_wav")
    wav_cmd_base+=(--fm-ckpt-path "${fm_ckpt}" --voc-ckpt-path "${vocoder_ckpt}")
    wav_cmd_base+=(--jsonl "${jsonl}" --funasr-root "${funasr_root}")
    if [[ "${DISABLE_SPK_EMBEDDING:-0}" == 1 ]]; then
        wav_cmd_base+=(--disable-spk-embedding)
    fi
    wav_cmd_base+=("${wav_args[@]}")

    local workers=${CODEX_TO_WAV_WORKERS:-1}
    local devices_csv=${CODEX_TO_WAV_DEVICES:-${CODEX_TO_WAV_DEVICE#cuda:}}
    local -a decode_devices=()
    IFS=',' read -r -a decode_devices <<< "${devices_csv}"
    if (( workers > ${#decode_devices[@]} )); then
        echo "ERROR: CODEX_TO_WAV_WORKERS=${workers} but only ${#decode_devices[@]} devices in CODEX_TO_WAV_DEVICES=${devices_csv}" >&2
        return 2
    fi
    local decode_root="${out_dir}/1best_recog/s2st_wav/decode_workers"
    mkdir -p "${decode_root}"

    if (( workers > 1 )) && [[ "${CODEX_PREWARM_CUDA_EXTENSION:-1}" == 1 ]]; then
        local prewarm_lock="${funasr_root}/funasr/models/cosyvoice/alias_free_activation/cuda/build/codex_prewarm.lock"
        (
            flock -x 8
            timeout "${CODEX_PREWARM_TIMEOUT:-600}" env \
                PYTHONPATH="${wav_pythonpath}:${workspace_root}/deps/python:${funasr_root}:${PYTHONPATH:-}" \
                "${wav_python}" -c \
                'from funasr.models.cosyvoice.alias_free_activation.cuda.activation1d import FusedAntiAliasActivation'
        ) 8>"${prewarm_lock}" >"${decode_root}/prewarm.out" 2>"${decode_root}/prewarm.err"
    fi

    local -a pids=()
    local idx device status=0
    for ((idx=0; idx<workers; idx++)); do
        device=${decode_devices[$idx]}
        device=${device#cuda:}
        PYTHONPATH="${wav_pythonpath}:${workspace_root}/deps/python:${funasr_root}:${PYTHONPATH:-}" \
        "${wav_cmd_base[@]}" --device "cuda:${device}" \
            --num-shards "${workers}" --shard-index "${idx}" \
            --summary-path "${decode_root}/part${idx}.summary.json" \
            >"${decode_root}/part${idx}.out" 2>"${decode_root}/part${idx}.err" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" || status=$?
    done
    (( status == 0 )) || return "${status}"
}

if [[ "${RUN_CODEX_TO_WAV:-1}" == 1 ]]; then
    while IFS=$'\t' read -r jsonl out_dir; do
        [[ -n "${jsonl}" && -n "${out_dir}" ]] || continue
        decode_one_output "${jsonl}" "${out_dir}"
    done < <(paste "${input_list}" "${output_list}")
fi

echo "S2ST inference finished. Outputs are listed in ${output_list}"
