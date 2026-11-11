#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_voice_eval.sh --prompt_wavs PROMPT_WAVS --hyp_wavs HYP_WAVS --output_dir OUT_DIR [options]

Runs speaker similarity first, then DNSMOS.

Required:
  --prompt_wavs PATH     SCP file: utt_id prompt_wav_path
  --hyp_wavs PATH        SCP file: utt_id generated_wav_path
  --output_dir PATH      Directory for metric outputs

Options:
  --ref_wavs PATH        Optional reference wav SCP for ref similarity column
  --devices IDS          CUDA device list for speaker similarity, default: 0
  --model_name NAME      Speaker model branch, default: eres2net
  --model_id ID          Speaker model id, default: damo/speech_eres2net_sv_en_voxceleb_16k
  --python PATH          Python executable, default: cosyvoice_eval env python
  --dnsmos_personalized  Use personalized DNSMOS model
  --skip_spk             Skip speaker similarity
  --skip_dnsmos          Skip DNSMOS
  -h, --help             Show this help

Outputs:
  OUT_DIR/spk_simi_scores.<model_name>.txt
  OUT_DIR/mos.csv
  OUT_DIR/dnsmos_mean.txt
EOF
}

VOICE_EVAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_NAME=eres2net
MODEL_ID=damo/speech_eres2net_sv_en_voxceleb_16k
DEVICES=0
PROMPT_WAVS=
HYP_WAVS=
REF_WAVS=
OUTPUT_DIR=
DNSMOS_PERSONALIZED=0
SKIP_SPK=0
SKIP_DNSMOS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt_wavs) PROMPT_WAVS=${2:?}; shift 2 ;;
    --hyp_wavs) HYP_WAVS=${2:?}; shift 2 ;;
    --ref_wavs) REF_WAVS=${2:?}; shift 2 ;;
    --output_dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --devices) DEVICES=${2:?}; shift 2 ;;
    --model_name) MODEL_NAME=${2:?}; shift 2 ;;
    --model_id) MODEL_ID=${2:?}; shift 2 ;;
    --python) PYTHON_BIN=${2:?}; shift 2 ;;
    --dnsmos_personalized) DNSMOS_PERSONALIZED=1; shift ;;
    --skip_spk) SKIP_SPK=1; shift ;;
    --skip_dnsmos) SKIP_DNSMOS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$PROMPT_WAVS" || -z "$HYP_WAVS" || -z "$OUTPUT_DIR" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 2
fi

for f in "$PROMPT_WAVS" "$HYP_WAVS"; do
  if [[ ! -f "$f" ]]; then
    echo "Input file not found: $f" >&2
    exit 1
  fi
done
if [[ -n "$REF_WAVS" && ! -f "$REF_WAVS" ]]; then
  echo "Reference SCP not found: $REF_WAVS" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$VOICE_EVAL_ROOT/thirdparty/3D-Speaker:${PYTHONPATH:-}"

SPK_SCRIPT="$VOICE_EVAL_ROOT/scripts/eval_speaker_similarity_lyuxiang.py"
DNSMOS_SCRIPT="$VOICE_EVAL_ROOT/scripts/dnsmos_local_wavscp.py"
SPK_MODEL_DIR="$VOICE_EVAL_ROOT/thirdparty/3D-Speaker/pretrained"
DNSMOS_ROOT="$VOICE_EVAL_ROOT/thirdparty/DNSMOS"
SPK_LOG="$OUTPUT_DIR/spk_simi_scores.${MODEL_NAME}.txt"
MOS_CSV="$OUTPUT_DIR/mos.csv"
DNSMOS_MEAN="$OUTPUT_DIR/dnsmos_mean.txt"

if [[ "$SKIP_SPK" -eq 0 ]]; then
  echo "[voice_eval] Running speaker similarity..."
  spk_cmd=("$PYTHON_BIN" "$SPK_SCRIPT"
    --model_name "$MODEL_NAME"
    --model_id "$MODEL_ID"
    --local_model_dir "$SPK_MODEL_DIR"
    --prompt_wavs "$PROMPT_WAVS"
    --hyp_wavs "$HYP_WAVS"
    --log_file "$SPK_LOG"
    --devices "$DEVICES")
  if [[ -n "$REF_WAVS" ]]; then
    spk_cmd+=(--ref_wavs "$REF_WAVS")
  fi
  "${spk_cmd[@]}"
  echo "[voice_eval] Speaker similarity written to $SPK_LOG"
fi

if [[ "$SKIP_DNSMOS" -eq 0 ]]; then
  echo "[voice_eval] Running DNSMOS..."
  dnsmos_cmd=("$PYTHON_BIN" "$DNSMOS_SCRIPT" -t "$HYP_WAVS" -e "$DNSMOS_ROOT" -o "$MOS_CSV")
  if [[ "$DNSMOS_PERSONALIZED" -eq 1 ]]; then
    dnsmos_cmd+=(-p)
  fi
  "${dnsmos_cmd[@]}"
  awk -F ',' 'NR > 1 { sum += $NF; count++ } END { if (count > 0) print sum / count; else print "nan" }' "$MOS_CSV" > "$DNSMOS_MEAN"
  echo "[voice_eval] DNSMOS csv written to $MOS_CSV"
  echo "[voice_eval] DNSMOS mean written to $DNSMOS_MEAN"
fi

echo "[voice_eval] Done. Output dir: $OUTPUT_DIR"
