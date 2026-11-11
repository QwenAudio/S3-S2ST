#!/usr/bin/env bash
set -euo pipefail

eval_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

hf_dir=${1:-}
output_root=${2:-}
if [[ -z "${hf_dir}" || -z "${output_root}" ]]; then
  echo "Usage: $0 <converted_hf_dir> <output_root>" >&2
  exit 2
fi
# Output containment is checked by tools/score.py.


data_list=${DATA_LIST:?Set DATA_LIST to an evaluation manifest list}
local_python=${LOCAL_PYTHON_BIN:-python}
metric_python=${METRIC_PYTHON_BIN:-python}
utmos_python=${UTMOS_PYTHON_BIN:-${eval_dir}/../third_party/utmos_env/bin/python}
comet_python=${COMET_PYTHON_BIN:-${metric_python}}
wer_tool=${WER_TOOL_DIR:-${eval_dir}/../third_party/cer_tool_parallel}
voice_eval=${VOICE_EVAL_SCRIPT:-${eval_dir}/../voice_eval/scripts/run_voice_eval.sh}
wer_jobs=${WER_NUM_JOBS:-8}
run_text=${RUN_TEXT_METRICS:-1}
run_asr=${RUN_ASR_METRICS:-1}
run_comet=${RUN_COMET:-1}
run_asr_comet=${RUN_ASR_COMET:-1}
run_utmos=${RUN_UTMOS:-1}
run_spk_sim=${RUN_SPK_SIM:-1}
run_wer_tn=${RUN_WER_TN:-1}
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

# Inference is intentionally not part of this helper.  The deferred-Talker
# launcher has already produced and validated the raw text, codec and WAV files
# before metrics start.

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
      cp "${asr_tsv}" "${metrics}/asr_details_backend.tsv"
      "${local_python}" "${eval_dir}/complete_empty_asr.py" "${metrics}/wer_ref.txt" "${asr_tsv}" "${recog}" "${metrics}/asr_details_raw.tsv"
      cp "${metrics}/asr_details_raw.tsv" "${metrics}/asr_details.tsv"
    fi
    run "${metric_python}" "${eval_dir}/score_bleu.py" \
      --ref "${recog}/label" --asr-tsv "${metrics}/asr_details_raw.tsv" \
      --language "${target_lang}" \
      --output "${metrics}/asr_bleu.json"
    if [[ "${run_asr_comet}" == 1 ]]; then
      run "${local_python}" "${eval_dir}/prepare_asr_comet.py" \
        --recog-dir "${recog}" --asr-tsv "${metrics}/asr_details_raw.tsv" \
        --target-language "${target_lang}" \
        --output "${metrics}/asr_comet_input.jsonl"
      run "${comet_python}" "${eval_dir}/eval_comet.py" \
        "${metrics}/asr_comet_input.jsonl" "${metrics}/asr_comet.json"
    fi
    if [[ "${run_wer_tn}" == 1 ]]; then
      # Preserve raw scoring for audit, then make symmetric-TN WER/CER the
      # official summary value.  Both reference and ASR hypothesis are TN'd.
      run cp "${s2st_wav}/wav_res_ref_text.wer" "${s2st_wav}/wav_res_ref_text.raw.wer"
      run "${metric_python}" "${eval_dir}/tn_wer.py" \
        --input "${metrics}/asr_details_raw.tsv" --output "${metrics}/asr_details.tsv" \
        --report "${s2st_wav}/wav_res_ref_text.wer" --audit "${metrics}/wer_tn_audit.jsonl" \
        --lang "${target_lang}"
    fi
  fi
  if [[ "${run_comet}" == 1 ]]; then
    run "${comet_python}" "${eval_dir}/eval_comet.py" \
      "${metrics}/comet_input.jsonl" "${metrics}/comet.json"
  fi
  if [[ "${run_utmos}" == 1 ]]; then
    [[ -x "${utmos_python}" ]] || { echo "missing fixed UTMOS env: ${utmos_python}; run setup_utmos.sh" >&2; exit 1; }
    run "${utmos_python}" "${eval_dir}/eval_utmos.py" \
      --wav-scp "${metrics}/hyp_wav.scp" --output "${metrics}/utmos.json" \
      --csv "${metrics}/utmos.csv" --device "${UTMOS_DEVICE:-cuda:0}"
  fi
  # Keep the established speaker-similarity metric.  The legacy helper may
  # also create DNSMOS side files; they are intentionally ignored by summary.
  if [[ "${run_spk_sim}" == 1 ]]; then
    run env NUMBA_CACHE_DIR="${output_root}/.numba_cache" bash "${voice_eval}" \
      --prompt_wavs "${metrics}/prompt_wav.scp" --hyp_wavs "${metrics}/hyp_wav.scp" \
      --skip_dnsmos --output_dir "${metrics}/voice" --devices "${VOICE_DEVICE:-0}" --python "${metric_python}"
  fi
done

run "${local_python}" "${eval_dir}/summarize_eval.py" "${output_root}"

echo "S2ST evaluation finished: ${output_root}"
