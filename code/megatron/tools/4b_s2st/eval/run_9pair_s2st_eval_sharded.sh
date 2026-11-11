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
  *) echo "Refusing output outside /path/to/workspace $output_root" >&2; exit 2 ;;
esac

zh_jsonl=${ZH2EN_JSONL:-/path/to/test_9pair_zh2en.jsonl}
en_jsonl=${EN2ZH_JSONL:-/path/to/test_9pair_en2zh.jsonl}
shard_lines=${EVAL_SHARD_LINES:-100}
infer_njobs=${INFER_NJOBS:-8}
nofile_limit=${INFER_NOFILE_LIMIT:-65536}

cd "$repo_root"
for required in "$hf_dir/config.json" "$zh_jsonl" "$en_jsonl"; do
  [[ -e "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done
mkdir -p "$output_root"/{shards,shard_outputs,lists,logs,manifests,attempts,done}
exec 9> "$output_root/.eval.lock"
flock -n 9 || { echo "ERROR: another evaluation is already using $output_root" >&2; exit 1; }
if [[ -f "$output_root/manifests/shard_lines.txt" ]]; then
  saved_shard_lines=$(<"$output_root/manifests/shard_lines.txt")
  [[ "$saved_shard_lines" == "$shard_lines" ]] || {
    echo "ERROR: existing output uses EVAL_SHARD_LINES=$saved_shard_lines, requested $shard_lines" >&2
    exit 1
  }
else
  printf '%s\n' "$shard_lines" > "$output_root/manifests/shard_lines.txt"
fi
printf '%s\n%s\n' "$zh_jsonl" "$en_jsonl" > "$output_root/manifests/input_data.list"
printf '%s\n%s\n' "$output_root/test_9pair_zh2en.out" "$output_root/test_9pair_en2zh.out" > "$output_root/manifests/output_dir.list"

prepare_shards() {
  local direction=$1 jsonl=$2 prefix="$output_root/shards/test_9pair_$1.part"
  if ! compgen -G "$prefix*.jsonl" >/dev/null; then
    split -d -a 3 -l "$shard_lines" --additional-suffix=.jsonl "$jsonl" "$prefix"
  fi
  local shard part name
  for shard in "$prefix"*.jsonl; do
    part=${shard##*.part}
    part=${part%.jsonl}
    name="test_9pair_$direction.part$part"
    printf '%s\n' "$shard" > "$output_root/lists/$name.input.list"
    printf '%s\n' "$output_root/shard_outputs/$name.out" > "$output_root/lists/$name.output.list"
  done
}
prepare_shards zh2en "$zh_jsonl"
prepare_shards en2zh "$en_jsonl"

run_shard() {
  local name=$1 thinker=$2 talker=$3 wavdev=$4
  local input="$output_root/lists/$name.input.list"
  local output="$output_root/lists/$name.output.list"
  local jsonl outdir expected text_rows quality_rows ok_rows wav_rows stamp
  jsonl=$(<"$input")
  outdir=$(<"$output")
  expected=$(wc -l < "$jsonl")
  if [[ -f "$output_root/done/$name" ]]; then
    echo "skip complete shard: $name"
    return 0
  fi
  if [[ -e "$outdir/1best_recog/text" && -e "$outdir/1best_recog/codec_quality.jsonl" ]]; then
    text_rows=$(wc -l < "$outdir/1best_recog/text")
    quality_rows=$(wc -l < "$outdir/1best_recog/codec_quality.jsonl")
    ok_rows=$(grep -c '"ok": true' "$outdir/1best_recog/codec_quality.jsonl" || true)
    wav_rows=$(find "$outdir/1best_recog/s2st_wav/wav" -maxdepth 1 -type f -name '*.wav' | wc -l)
    if [[ "$text_rows" -eq "$expected" && "$quality_rows" -eq "$expected" && "$wav_rows" -eq "$ok_rows" ]]; then
      printf '%s\ttext=%s\tcodec_ok=%s\twav=%s\n' "$name" "$text_rows" "$ok_rows" "$wav_rows" > "$output_root/done/$name"
      echo "adopt complete shard: $name"
      return 0
    fi
  fi
  if [[ -e "$outdir" ]]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mv "$outdir" "$output_root/attempts/$name.$stamp.out"
  fi
  env USE_OSS_CACHE=1 USE_CODEC_LLM_HIDDEN_FOR_FM=0 INFER_ENFORCE_EAGER=1 MAX_TOKENS=2048 \
    INFER_NOFILE_LIMIT="$nofile_limit" INFER_NJOBS="$infer_njobs" \
    THINKER_DEVICES="$thinker" TALKER_DEVICES="$talker" CODEX_TO_WAV_DEVICE="$wavdev" \
    bash tools/4b_s2st/run_4b_s2st_inference.sh "$hf_dir" "$input" "$output" \
    > "$output_root/logs/$name.out" 2> "$output_root/logs/$name.err"
  text_rows=$(wc -l < "$outdir/1best_recog/text")
  quality_rows=$(wc -l < "$outdir/1best_recog/codec_quality.jsonl")
  ok_rows=$(grep -c '"ok": true' "$outdir/1best_recog/codec_quality.jsonl" || true)
  wav_rows=$(find "$outdir/1best_recog/s2st_wav/wav" -maxdepth 1 -type f -name '*.wav' | wc -l)
  [[ "$text_rows" -eq "$expected" && "$quality_rows" -eq "$expected" && "$wav_rows" -eq "$ok_rows" ]] || {
    echo "ERROR: incomplete $name expected=$expected text=$text_rows quality=$quality_rows codec_ok=$ok_rows wav=$wav_rows" >&2
    return 1
  }
  printf '%s\ttext=%s\tcodec_ok=%s\twav=%s\n' "$name" "$text_rows" "$ok_rows" "$wav_rows" > "$output_root/done/$name"
}

mapfile -t shards < <(find "$output_root/shards" -maxdepth 1 -type f -name 'test_9pair_*.part*.jsonl' -printf '%f\n' | sed 's/\.jsonl$//' | sort)
worker() {
  local wid=$1 thinker=$2 talker=$3 wavdev=$4 i name
  for ((i=wid; i<${#shards[@]}; i+=2)); do
    name=${shards[$i]}
    run_shard "$name" "$thinker" "$talker" "$wavdev"
  done
}
worker 0 "${WORKER0_THINKER_DEVICES:-[0,1]}" "${WORKER0_TALKER_DEVICES:-[2]}" "${WORKER0_WAV_DEVICE:-cuda:3}" > "$output_root/logs/worker_0.out" 2> "$output_root/logs/worker_0.err" &
p0=$!
worker 1 "${WORKER1_THINKER_DEVICES:-[4,5]}" "${WORKER1_TALKER_DEVICES:-[6]}" "${WORKER1_WAV_DEVICE:-cuda:7}" > "$output_root/logs/worker_1.out" 2> "$output_root/logs/worker_1.err" &
p1=$!
s0=0
s1=0
wait "$p0" || s0=$?
wait "$p1" || s1=$?
[[ "$s0" -eq 0 && "$s1" -eq 0 ]] || {
  echo "ERROR: inference workers failed: $s0,$s1" >&2
  exit 1
}

for direction in zh2en en2zh; do
  final="$output_root/test_9pair_$direction.out/1best_recog"
  if [[ -e "$final" ]]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mv "$final" "$output_root/attempts/test_9pair_$direction.merged.$stamp"
  fi
  mkdir -p "$final/codec" "$final/s2st_wav/feat" "$final/s2st_wav/wav"
  : > "$final/source"
  : > "$final/text"
  : > "$final/label"
  : > "$final/key_map.scp"
  : > "$final/codec_quality.jsonl"
  while IFS= read -r rec; do
    for f in source text label key_map.scp codec_quality.jsonl; do
      [[ -f "$rec/$f" ]] && cat "$rec/$f" >> "$final/$f"
    done
    cp -a --update=none "$rec/codec/." "$final/codec/"
    cp -a --update=none "$rec/s2st_wav/feat/." "$final/s2st_wav/feat/"
    cp -a --update=none "$rec/s2st_wav/wav/." "$final/s2st_wav/wav/"
  done < <(find "$output_root/shard_outputs" -type d -path "*/test_9pair_$direction.part*.out/1best_recog" | sort)
  expected=$(wc -l < "$output_root/manifests/input_data.list")
  if [[ "$direction" == zh2en ]]; then expected=$(wc -l < "$zh_jsonl"); else expected=$(wc -l < "$en_jsonl"); fi
  text_rows=$(wc -l < "$final/text")
  quality_rows=$(wc -l < "$final/codec_quality.jsonl")
  ok_rows=$(grep -c '"ok": true' "$final/codec_quality.jsonl" || true)
  wav_rows=$(find "$final/s2st_wav/wav" -maxdepth 1 -type f -name '*.wav' | wc -l)
  [[ "$text_rows" -eq "$expected" && "$quality_rows" -eq "$expected" && "$wav_rows" -eq "$ok_rows" ]] || {
    echo "ERROR: merged $direction expected=$expected text=$text_rows quality=$quality_rows codec_ok=$ok_rows wav=$wav_rows" >&2
    exit 1
  }
done

env RUN_INFERENCE=0 RUN_TEXT_METRICS=1 RUN_ASR_METRICS=1 RUN_COMET=1 \
  RUN_VOICE_METRICS=1 WER_NUM_JOBS="${WER_NUM_JOBS:-8}" VOICE_DEVICE="${VOICE_DEVICE:-0}" \
  bash tools/4b_s2st/eval/run_9pair_s2st_eval.sh "$hf_dir" "$output_root" \
  > "$output_root/logs/metrics.out" 2> "$output_root/logs/metrics.err"

touch "$output_root/COMPLETE"
echo "complete=$output_root"
