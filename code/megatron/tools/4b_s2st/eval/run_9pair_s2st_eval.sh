#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
eval_dir=${repo_root}/tools/4b_s2st/eval

hf_dir=${1:-}
output_root=${2:-}
if [[ -z "${hf_dir}" || -z "${output_root}" ]]; then
  echo "Usage: $0 <converted_hf_dir> <output_root>" >&2
  exit 2
fi

data_list=${DATA_LIST:-/path/to/zhen.list}
local_python=${LOCAL_PYTHON_BIN:-/usr/local/bin/python}
metric_python=${METRIC_PYTHON_BIN:-/path/to/python}
comet_python=${COMET_PYTHON_BIN:-${metric_python}}
wer_tool=${WER_TOOL_DIR:-/path/to/cer_tool_from_lyx2}
voice_eval=${VOICE_EVAL_SCRIPT:-/path/to/run_voice_eval.sh}
wer_jobs=${WER_NUM_JOBS:-8}
run_infer=${RUN_INFERENCE:-1}
run_text=${RUN_TEXT_METRICS:-1}
run_asr=${RUN_ASR_METRICS:-1}
run_comet=${RUN_COMET:-1}
run_voice=${RUN_VOICE_METRICS:-1}
dry_run=${DRY_RUN:-0}

run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  [[ "${dry_run}" == 1 ]] || "$@"
}

mkdir -p "${output_root}/manifests"
cp "${data_list}" "${output_root}/manifests/input_data.list"
: > "${output_root}/manifests/output_dir.list"
while IFS= read -r jsonl; do
  [[ -n "${jsonl}" ]] || continue
  name=$(basename "${jsonl}" .jsonl)
  printf '%s\n' "${output_root}/${name}.out" >> "${output_root}/manifests/output_dir.list"
done < "${output_root}/manifests/input_data.list"

if [[ "${run_infer}" == 1 ]]; then
  run bash "${repo_root}/tools/4b_s2st/run_4b_s2st_inference.sh" \
    "${hf_dir}" "${output_root}/manifests/input_data.list" "${output_root}/manifests/output_dir.list"
fi

paste "${output_root}/manifests/input_data.list" "${output_root}/manifests/output_dir.list" | \
while IFS=$'\t' read -r jsonl result_dir; do
  [[ -n "${jsonl}" && -n "${result_dir}" ]] || continue
  name=$(basename "${jsonl}" .jsonl)
  case "${name}" in
    *zh2en*) target_lang=en ;;
    *en2zh*) target_lang=zh ;;
    *) echo "Cannot infer target language from ${name}" >&2; exit 1 ;;
  esac
  recog=${result_dir}/1best_recog
  s2st_wav=${recog}/s2st_wav
  metrics=${result_dir}/metrics
  run mkdir -p "${metrics}"
  run "${local_python}" "${eval_dir}/prepare_eval.py" --jsonl "${jsonl}" \
    --recog-dir "${recog}" --wav-dir "${s2st_wav}/wav" --output-dir "${metrics}"

  if [[ "${run_text}" == 1 ]]; then
    run "${metric_python}" "${eval_dir}/score_bleu.py" --ref "${recog}/label" \
      --hyp "${recog}/text" --language "${target_lang}" --output "${metrics}/text_bleu.json"
  fi
  if [[ "${run_asr}" == 1 ]]; then
    run env WER_PYTHON="${metric_python}" bash "${wer_tool}/cal_wer.sh" \
      "${metrics}/wer_ref.txt" "${s2st_wav}" "${target_lang}" "${wer_jobs}"
    if [[ "${dry_run}" != 1 ]]; then
      asr_tsv=$(find "${s2st_wav}/tmp" -type f -path '*/results/merge.out' -printf '%T@ %p\n' \
        | sort -nr | head -1 | cut -d' ' -f2-)
      [[ -n "${asr_tsv}" ]] || { echo "ASR detail TSV not found under ${s2st_wav}/tmp" >&2; exit 1; }
      cp "${asr_tsv}" "${metrics}/asr_details.tsv"
    fi
    run "${metric_python}" "${eval_dir}/score_bleu.py" \
      --ref "${recog}/label" --asr-tsv "${metrics}/asr_details.tsv" \
      --language "${target_lang}" \
      --output "${metrics}/asr_bleu.json"
  fi
  if [[ "${run_comet}" == 1 ]]; then
    run "${comet_python}" "${eval_dir}/eval_comet.py" \
      "${metrics}/comet_input.jsonl" "${metrics}/comet.json"
  fi
  if [[ "${run_voice}" == 1 ]]; then
    run env NUMBA_CACHE_DIR="${output_root}/.numba_cache" bash "${voice_eval}" \
      --prompt_wavs "${metrics}/prompt_wav.scp" --hyp_wavs "${metrics}/hyp_wav.scp" \
      --output_dir "${metrics}/voice" --devices "${VOICE_DEVICE:-0}" --python "${metric_python}"
  fi
done

run "${local_python}" "${eval_dir}/summarize_eval.py" "${output_root}"

echo "9pair S2ST evaluation finished: ${output_root}"
