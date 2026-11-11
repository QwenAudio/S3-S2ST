#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
hf_dir=${1:-}
output_root=${2:-}
if [[ -z "${hf_dir}" || -z "${output_root}" ]]; then
  echo "Usage: $0 <converted_hf_dir> <output_root>" >&2
  exit 2
fi

profile=${EVAL_PROFILE:-full}
case "${profile}" in
  eos|core|full) ;;
  *) echo "ERROR: EVAL_PROFILE must be eos, core, or full" >&2; exit 2 ;;
esac
max_samples=${EVAL_MAX_SAMPLES_PER_DIRECTION:-0}
seed=${EVAL_SUBSET_SEED:-20260812}
infer_njobs=${INFER_NJOBS:-8}
decode_workers_per_direction=${DECODE_WORKERS_PER_DIRECTION:-4}
zh_source=${ZH2EN_JSONL:-/path/to/test_9pair_zh2en.jsonl}
en_source=${EN2ZH_JSONL:-/path/to/test_9pair_en2zh.jsonl}

cd "${repo_root}"
for required in "${hf_dir}/config.json" "${zh_source}" "${en_source}"; do
  [[ -e "${required}" ]] || { echo "ERROR: missing ${required}" >&2; exit 1; }
done
mkdir -p "${output_root}"/{eval_data,lists,logs,manifests,attempts,done}
exec 9> "${output_root}/.eval.lock"
flock -n 9 || { echo "ERROR: another evaluation is using ${output_root}" >&2; exit 1; }

completion_marker="${output_root}/${profile^^}_COMPLETE"
if [[ -f "${completion_marker}" ]]; then
  recorded_hf=$(<"${output_root}/manifests/hf_checkpoint.txt")
  recorded_profile=$(<"${output_root}/manifests/profile.txt")
  recorded_max_samples=$(<"${output_root}/manifests/max_samples_per_direction.txt")
  [[ "${recorded_hf}" == "${hf_dir}" ]] || {
    echo "ERROR: completed output belongs to a different checkpoint: ${recorded_hf}" >&2
    exit 1
  }
  [[ "${recorded_profile}" == "${profile}" ]] || {
    echo "ERROR: completed output uses profile=${recorded_profile}, requested=${profile}" >&2
    exit 1
  }
  [[ "${recorded_max_samples}" == "${max_samples}" ]] || {
    echo "ERROR: completed output uses max_samples=${recorded_max_samples}, requested=${max_samples}" >&2
    exit 1
  }
  [[ -s "${output_root}/eos_summary.tsv" && -s "${output_root}/eos_summary.json" ]] || {
    echo "ERROR: completion marker exists but EOS summary is incomplete" >&2
    exit 1
  }
  if [[ "${profile}" != eos ]]; then
    [[ -s "${output_root}/summary.tsv" && -s "${output_root}/summary.json" ]] || {
      echo "ERROR: completion marker exists but metric summary is incomplete" >&2
      exit 1
    }
  fi
  echo "already_complete=${output_root}"
  exit 0
fi

printf '%s\n' "${hf_dir}" > "${output_root}/manifests/hf_checkpoint.txt"
printf '%s\n' "${profile}" > "${output_root}/manifests/profile.txt"
printf '%s\n' "${max_samples}" > "${output_root}/manifests/max_samples_per_direction.txt"
printf '%s\n' "${CODEX_MIN_CODEC_TEXT_RATIO:-1.0}" > "${output_root}/manifests/min_codec_text_ratio.txt"
printf '%s\n' "${CODEX_MAX_CODEC_TEXT_RATIO:-8.0}" > "${output_root}/manifests/max_codec_text_ratio.txt"

prepare_data() {
  local source=$1 destination=$2
  if [[ "${max_samples}" =~ ^[0-9]+$ ]] && (( max_samples > 0 )); then
    python3 - "${source}" "${destination}" "${max_samples}" "${seed}" <<'PY'
import random
import sys
from pathlib import Path
source, destination, count, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lines = Path(source).read_text(encoding="utf-8", errors="replace").splitlines(True)
if count >= len(lines):
    selected = lines
else:
    rng = random.Random(seed)
    selected = [lines[i] for i in sorted(rng.sample(range(len(lines)), count))]
Path(destination).write_text("".join(selected), encoding="utf-8")
PY
  else
    ln -sfn "${source}" "${destination}"
  fi
}
zh_jsonl="${output_root}/eval_data/test_9pair_zh2en.jsonl"
en_jsonl="${output_root}/eval_data/test_9pair_en2zh.jsonl"
prepare_data "${zh_source}" "${zh_jsonl}"
prepare_data "${en_source}" "${en_jsonl}"
printf '%s\n%s\n' "${zh_jsonl}" "${en_jsonl}" > "${output_root}/manifests/metric_input_data.list"

run_inference_direction() {
  local direction=$1 jsonl=$2 outdir=$3 thinker=$4 talker=$5
  local expected text_rows quality_rows codec_rows ok_rows stamp
  expected=$(wc -l < "${jsonl}")
  if [[ -f "${output_root}/done/${direction}.inference" ]]; then
    text_rows=$(wc -l < "${outdir}/1best_recog/text" 2>/dev/null || echo 0)
    quality_rows=$(wc -l < "${outdir}/1best_recog/codec_quality.jsonl" 2>/dev/null || echo 0)
    if [[ "${text_rows}" -eq "${expected}" && "${quality_rows}" -eq "${expected}" ]]; then
      echo "skip inference: ${direction}"
      return 0
    fi
  fi
  if [[ -e "${outdir}" ]]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mv "${outdir}" "${output_root}/attempts/${direction}.${stamp}.out"
  fi
  local input_list="${output_root}/lists/${direction}.input.list"
  local output_list="${output_root}/lists/${direction}.output.list"
  printf '%s\n' "${jsonl}" > "${input_list}"
  printf '%s\n' "${outdir}" > "${output_list}"
  env RUN_CODEX_TO_WAV=0 USE_OSS_CACHE=1 USE_CODEC_LLM_HIDDEN_FOR_FM=0 \
    TALKER_INPUT_MODE=hidden \
    INFER_ENFORCE_EAGER=1 MAX_TOKENS=2048 INFER_NJOBS="${infer_njobs}" \
    THINKER_DEVICES="${thinker}" TALKER_DEVICES="${talker}" \
    bash tools/4b_s2st/run_4b_s2st_inference_lenratio.sh "${hf_dir}" "${input_list}" "${output_list}" \
    > "${output_root}/logs/${direction}.infer.out" \
    2> "${output_root}/logs/${direction}.infer.err"
  text_rows=$(wc -l < "${outdir}/1best_recog/text")
  quality_rows=$(wc -l < "${outdir}/1best_recog/codec_quality.jsonl")
  codec_rows=$(find "${outdir}/1best_recog/codec" -maxdepth 1 -type f -name '*.npy' | wc -l)
  ok_rows=$(grep -c '"ok": true' "${outdir}/1best_recog/codec_quality.jsonl" || true)
  [[ "${text_rows}" -eq "${expected}" && "${quality_rows}" -eq "${expected}" && "${codec_rows}" -eq "${ok_rows}" ]] || {
    echo "ERROR: incomplete inference ${direction}: expected=${expected} text=${text_rows} quality=${quality_rows} codec=${codec_rows} ok=${ok_rows}" >&2
    return 1
  }
  printf 'expected=%s\ttext=%s\tcodec=%s\n' "${expected}" "${text_rows}" "${codec_rows}" > "${output_root}/done/${direction}.inference"
}

zh_out="${output_root}/test_9pair_zh2en.out"
en_out="${output_root}/test_9pair_en2zh.out"
run_inference_direction zh2en "${zh_jsonl}" "${zh_out}" \
  "${WORKER0_THINKER_DEVICES:-[0,1]}" "${WORKER0_TALKER_DEVICES:-[2]}" &
p0=$!
run_inference_direction en2zh "${en_jsonl}" "${en_out}" \
  "${WORKER1_THINKER_DEVICES:-[4,5]}" "${WORKER1_TALKER_DEVICES:-[6]}" &
p1=$!
s0=0; s1=0
wait "${p0}" || s0=$?
wait "${p1}" || s1=$?
[[ "${s0}" -eq 0 && "${s1}" -eq 0 ]] || {
  echo "ERROR: resident inference failed: zh2en=${s0} en2zh=${s1}" >&2
  exit 1
}

python3 - "${output_root}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for direction in ("zh2en", "en2zh"):
    path = root / f"test_9pair_{direction}.out/1best_recog/codec_quality.jsonl"
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    eos = sum(x.get("termination") == "codec_eos" for x in items)
    no_eos = sum(x.get("termination") == "no_codec_eos" for x in items)
    missing = sum(x.get("termination") == "missing_codec" for x in items)
    lengths = sorted(int(x.get("codec_len", 0)) for x in items)
    rows.append({
        "direction": direction,
        "total": len(items),
        "codec_eos": eos,
        "no_codec_eos": no_eos,
        "missing_codec": missing,
        "eos_rate": eos / len(items) if items else None,
        "codec_len_p50": lengths[len(lengths)//2] if lengths else None,
        "codec_len_p95": lengths[int((len(lengths)-1)*0.95)] if lengths else None,
    })
(root / "eos_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
with (root / "eos_summary.tsv").open("w") as f:
    cols = list(rows[0])
    f.write("\t".join(cols) + "\n")
    for row in rows:
        f.write("\t".join(str(row[c]) for c in cols) + "\n")
PY

if [[ "${profile}" != eos ]]; then
  decode_direction() {
    local direction=$1 jsonl=$2 outdir=$3 devices=$4
    local input_list="${output_root}/lists/${direction}.input.list"
    local output_list="${output_root}/lists/${direction}.output.list"
    local ok_rows wav_rows
    ok_rows=$(grep -c '"ok": true' "${outdir}/1best_recog/codec_quality.jsonl" || true)
    mkdir -p "${outdir}/1best_recog/s2st_wav/wav"
    wav_rows=$(find "${outdir}/1best_recog/s2st_wav/wav" -maxdepth 1 -type f -name '*.wav' | wc -l)
    if [[ -f "${output_root}/done/${direction}.decode" && "${wav_rows}" -eq "${ok_rows}" ]]; then
      echo "skip decode: ${direction}"
      return 0
    fi
    env RUN_MODEL_INFERENCE=0 RUN_CODEX_TO_WAV=1 USE_OSS_CACHE=1 \
      USE_CODEC_LLM_HIDDEN_FOR_FM=0 CODEX_TO_WAV_WORKERS="${decode_workers_per_direction}" \
      CODEX_TO_WAV_DEVICES="${devices}" \
      bash tools/4b_s2st/run_4b_s2st_inference_lenratio.sh "${hf_dir}" "${input_list}" "${output_list}" \
      > "${output_root}/logs/${direction}.decode.out" \
      2> "${output_root}/logs/${direction}.decode.err"
    wav_rows=$(find "${outdir}/1best_recog/s2st_wav/wav" -maxdepth 1 -type f -name '*.wav' | wc -l)
    [[ "${wav_rows}" -eq "${ok_rows}" ]] || {
      echo "ERROR: incomplete decode ${direction}: codec_ok=${ok_rows} wav=${wav_rows}" >&2
      return 1
    }
    printf 'codec_ok=%s\twav=%s\n' "${ok_rows}" "${wav_rows}" > "${output_root}/done/${direction}.decode"
  }

  decode_direction zh2en "${zh_jsonl}" "${zh_out}" "${WORKER0_DECODE_DEVICES:-0,1,2,3}" &
  p0=$!
  decode_direction en2zh "${en_jsonl}" "${en_out}" "${WORKER1_DECODE_DEVICES:-4,5,6,7}" &
  p1=$!
  s0=0; s1=0
  wait "${p0}" || s0=$?
  wait "${p1}" || s1=$?
  [[ "${s0}" -eq 0 && "${s1}" -eq 0 ]] || {
    echo "ERROR: parallel decode failed: zh2en=${s0} en2zh=${s1}" >&2
    exit 1
  }

  run_comet=0
  run_voice=0
  if [[ "${profile}" == full ]]; then
    run_comet=1
    run_voice=1
  fi
  env DATA_LIST="${output_root}/manifests/metric_input_data.list" \
    RUN_INFERENCE=0 RUN_TEXT_METRICS=1 RUN_ASR_METRICS=1 \
    RUN_COMET="${run_comet}" RUN_VOICE_METRICS="${run_voice}" \
    WER_NUM_JOBS="${WER_NUM_JOBS:-8}" VOICE_DEVICE="${VOICE_DEVICE:-0}" \
    bash tools/4b_s2st/eval/run_9pair_s2st_eval.sh "${hf_dir}" "${output_root}" \
    > "${output_root}/logs/metrics.out" 2> "${output_root}/logs/metrics.err"
fi

touch "${output_root}/${profile^^}_COMPLETE"
[[ "${profile}" == full ]] && touch "${output_root}/COMPLETE"
echo "complete=${output_root}"
