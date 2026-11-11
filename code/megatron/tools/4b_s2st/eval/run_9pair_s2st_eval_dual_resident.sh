#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
hf_dir=${1:-}
output_root=${2:-}
if [[ -z "$hf_dir" || -z "$output_root" ]]; then
  echo "Usage: $0 <converted_hf_dir> <output_root>" >&2
  exit 2
fi
case "$output_root" in
  /path/to/_) ;;
  *) echo "Refusing output outside workspace: $output_root" >&2; exit 2 ;;
esac

zh_jsonl=${ZH2EN_JSONL:-/path/to/test_9pair_zh2en.jsonl}
en_jsonl=${EN2ZH_JSONL:-/path/to/test_9pair_en2zh.jsonl}
nofile_limit=${INFER_NOFILE_LIMIT:-65536}
infer_njobs=${INFER_NJOBS:-8}

cd "$repo_root"
for required in "$hf_dir/config.json" "$zh_jsonl" "$en_jsonl"; do
  [[ -e "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
mkdir -p "$output_root"/{manifests,lists,logs}
exec 9> "$output_root/.eval.lock"
flock -n 9 || { echo "ERROR: another evaluation is using $output_root" >&2; exit 1; }

printf '%s\n%s\n' "$zh_jsonl" "$en_jsonl" > "$output_root/manifests/input_data.list"
printf '%s\n%s\n' "$output_root/test_9pair_zh2en.out" "$output_root/test_9pair_en2zh.out" > "$output_root/manifests/output_dir.list"

run_direction() {
  local direction=$1 jsonl=$2 outdir=$3 thinker=$4 talker=$5 wavdev=$6
  local input_list="$output_root/lists/$direction.input.list"
  local output_list="$output_root/lists/$direction.output.list"
  printf '%s\n' "$jsonl" > "$input_list"
  printf '%s\n' "$outdir" > "$output_list"
  env USE_OSS_CACHE=1 USE_CODEC_LLM_HIDDEN_FOR_FM=0 INFER_ENFORCE_EAGER=1 MAX_TOKENS=2048 \
    INFER_NOFILE_LIMIT="$nofile_limit" INFER_NJOBS="$infer_njobs" \
    THINKER_DEVICES="$thinker" TALKER_DEVICES="$talker" CODEX_TO_WAV_DEVICE="$wavdev" \
    bash tools/4b_s2st/run_4b_s2st_inference.sh "$hf_dir" "$input_list" "$output_list" \
    > "$output_root/logs/$direction.out" 2> "$output_root/logs/$direction.err"
}

run_direction zh2en "$zh_jsonl" "$output_root/test_9pair_zh2en.out" \
  "${WORKER0_THINKER_DEVICES:-[0,1]}" "${WORKER0_TALKER_DEVICES:-[2]}" "${WORKER0_WAV_DEVICE:-cuda:3}" &
p0=$!
run_direction en2zh "$en_jsonl" "$output_root/test_9pair_en2zh.out" \
  "${WORKER1_THINKER_DEVICES:-[4,5]}" "${WORKER1_TALKER_DEVICES:-[6]}" "${WORKER1_WAV_DEVICE:-cuda:7}" &
p1=$!

s0=0
s1=0
wait "$p0" || s0=$?
wait "$p1" || s1=$?
[[ "$s0" -eq 0 && "$s1" -eq 0 ]] || {
  echo "ERROR: inference workers failed: zh2en=$s0 en2zh=$s1" >&2
  exit 1
}

for direction in zh2en en2zh; do
  if [[ "$direction" == zh2en ]]; then jsonl=$zh_jsonl; else jsonl=$en_jsonl; fi
  recog="$output_root/test_9pair_$direction.out/1best_recog"
  expected=$(wc -l < "$jsonl")
  text_rows=$(wc -l < "$recog/text")
  quality_rows=$(wc -l < "$recog/codec_quality.jsonl")
  ok_rows=$(grep -c '"ok": true' "$recog/codec_quality.jsonl" || true)
  wav_rows=$(find "$recog/s2st_wav/wav" -maxdepth 1 -type f -name '*.wav' | wc -l)
  [[ "$text_rows" -eq "$expected" && "$quality_rows" -eq "$expected" && "$wav_rows" -eq "$ok_rows" ]] || {
    echo "ERROR: incomplete $direction expected=$expected text=$text_rows quality=$quality_rows codec_ok=$ok_rows wav=$wav_rows" >&2
    exit 1
  }
  printf '%s\texpected=%s\ttext=%s\tcodec_ok=%s\twav=%s\n' "$direction" "$expected" "$text_rows" "$ok_rows" "$wav_rows"
done

env RUN_INFERENCE=0 RUN_TEXT_METRICS=1 RUN_ASR_METRICS=1 RUN_COMET=1 \
  RUN_VOICE_METRICS=1 WER_NUM_JOBS="${WER_NUM_JOBS:-8}" VOICE_DEVICE="${VOICE_DEVICE:-0}" \
  bash tools/4b_s2st/eval/run_9pair_s2st_eval.sh "$hf_dir" "$output_root" \
  > "$output_root/logs/metrics.out" 2> "$output_root/logs/metrics.err"

touch "$output_root/COMPLETE"
echo "complete=$output_root"
