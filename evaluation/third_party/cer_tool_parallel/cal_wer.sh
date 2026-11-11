#!/usr/bin/env bash
set -euo pipefail
set -x

if [ "$#" -lt 4 ]; then
  echo "Usage: bash cal_wer.sh <ref_text> <output_dir> <lang> <num_job> [zh_model] [whisper_model]" >&2
  echo "  ref_text: each row is '<utt> <reference text>'" >&2
  echo "  output_dir: contains wavs/<utt>.wav, or wav/<utt>.wav" >&2
  echo "  lang: zh, hard_zh, en, ..." >&2
  exit 2
fi

ref_text=$(readlink -f "$1")
output_dir=$(readlink -f "$2")
lang=$3
num_job=$4
zh_model=${5:-}
whisper_model=${6:-}
wer_python=${WER_PYTHON:-python3}
gpu_offset=${WER_GPU_OFFSET:-0}

tool_dir=$(cd "$(dirname "$0")" && pwd)
wav_dir=${output_dir}/wavs
if [ ! -d "${wav_dir}" ] && [ -d "${output_dir}/wav" ]; then
  wav_dir=${output_dir}/wav
fi
if [ ! -d "${wav_dir}" ]; then
  echo "Cannot find wav directory: ${output_dir}/wavs or ${output_dir}/wav" >&2
  exit 1
fi

wav_ref=${output_dir}/wav_res_ref_text
score_file=${output_dir}/wav_res_ref_text.wer
thread_dir=${output_dir}/tmp/wer_thread_metas_$(date +%s)
out_dir=${thread_dir}/results
mkdir -p "${out_dir}"

"${wer_python}" "${tool_dir}/prepare_wav_ref.py" "${ref_text}" "${wav_dir}" "${wav_ref}"
num=$(wc -l "${wav_ref}" | awk '{print $1}')
if [ "${num}" -eq 0 ]; then
  echo "No wav/ref rows to score." >&2
  exit 1
fi
num_per_thread=$((num / num_job + 1))
split -l "${num_per_thread}" --additional-suffix=.lst -d "${wav_ref}" "${thread_dir}/thread-"

pids=()
for rank in $(seq 0 $((num_job - 1))); do
  input_part=$(printf "%s/thread-%02d.lst" "${thread_dir}" "${rank}")
  [ -s "${input_part}" ] || continue
  sub_score_file=$(printf "%s/thread-%02d.wer.out" "${out_dir}" "${rank}")
  cmd=("${wer_python}" "${tool_dir}/run_wer.py" "${input_part}" "${sub_score_file}" "${lang}")
  if [ -n "${zh_model}" ]; then
    cmd+=(--zh-model "${zh_model}")
  fi
  if [ -n "${whisper_model}" ]; then
    cmd+=(--whisper-model "${whisper_model}")
  fi
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$((rank + gpu_offset))} "${cmd[@]}" &
  pids+=("$!")
done
worker_status=0
for pid in "${pids[@]}"; do
  wait "$pid" || worker_status=$?
done
(( worker_status == 0 )) || exit "$worker_status"

cat "${out_dir}"/thread-*.wer.out > "${out_dir}/merge.out"
metric_name=WER
case "${lang}" in
  *zh|hard_zh|ja|ko|th) metric_name=CER ;;
  *) metric_name=WER ;;
esac
"${wer_python}" "${tool_dir}/average_wer.py" "${out_dir}/merge.out" "${score_file}" --metric-name "${metric_name}"
