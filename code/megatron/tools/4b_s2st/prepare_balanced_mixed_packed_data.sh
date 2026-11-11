#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
workspace_root=$(cd -- "${repo_root}/.." && pwd)
artifact_root=${workspace_root}/artifacts/4b_s2st/data
balanced_list=${artifact_root}/balanced_raw/data.list
mixed_dir=${artifact_root}/balanced_mixed_raw_256
packed_dir=${artifact_root}/balanced_mixed_packed_8192
funasr_root=${FUNASR_ROOT:-${workspace_root}/deps/FunASR_0611}
tokenizer_model=${TOKENIZER_MODEL:-${S3_ROOT}/models/offline}
workers=${DATA_WORKERS:-32}

if [[ ! -s "${balanced_list}" ]]; then
    echo "ERROR: missing balanced input list: ${balanced_list}" >&2
    exit 1
fi

if [[ ! -s "${mixed_dir}/data.list" ]]; then
    python "${repo_root}/tools/4b_s2st/mix_balanced_directions.py" \
        --data-list "${balanced_list}" \
        --output "${mixed_dir}" \
        --num-shards 256
else
    echo "Reuse mixed balanced data: ${mixed_dir}/data.list"
fi

if [[ ! -s "${packed_dir}/data.list" ]]; then
    python "${repo_root}/tools/4b_s2st/pack_balanced_s2st.py" \
        --data-list "${mixed_dir}/data.list" \
        --output "${packed_dir}" \
        --funasr-root "${funasr_root}" \
        --tokenizer-model "${tokenizer_model}" \
        --cutoff-len 8192 \
        --num-workers "${workers}" \
        --seed 20260804
else
    echo "Reuse mixed packed data: ${packed_dir}/data.list"
fi

python - "${mixed_dir}/mix_stats.json" "${packed_dir}/data.list" <<'PY'
import json
import pathlib
import sys

stats = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(set(stats["output_counts"].values())) == 1, stats
paths = [line.strip() for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
assert len(paths) == 256, len(paths)
assert all(pathlib.Path(path).is_file() for path in paths)
print("mixed balanced packed validation passed:", stats["output_counts"], "shards=", len(paths))
PY
