#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
artifact_root=$(cd -- "${repo_root}/.." && pwd)/artifacts/4b_s2st/data
source_list=/path/to/data.list
balanced_dir=${artifact_root}/balanced_raw
packed_dir=${artifact_root}/balanced_packed_8192
funasr_root=${FUNASR_ROOT:-${repo_root}/../deps/FunASR_0611}
tokenizer_model=${TOKENIZER_MODEL:-${S3_ROOT}/models/offline}
workers=${DATA_WORKERS:-32}

mkdir -p "${artifact_root}"
if [[ ! -s "${balanced_dir}/data.list" ]]; then
    python "${repo_root}/tools/4b_s2st/balance_two_directions.py" \
        --data-list "${source_list}" \
        --output "${balanced_dir}" \
        --workers "${workers}"
else
    echo "Reuse balanced data: ${balanced_dir}/data.list"
fi

if [[ ! -s "${packed_dir}/data.list" ]]; then
    python "${repo_root}/tools/4b_s2st/pack_balanced_s2st.py" \
        --data-list "${balanced_dir}/data.list" \
        --output "${packed_dir}" \
        --funasr-root "${funasr_root}" \
        --tokenizer-model "${tokenizer_model}" \
        --cutoff-len 8192 \
        --num-workers "${workers}"
else
    echo "Reuse packed data: ${packed_dir}/data.list"
fi

python - "${balanced_dir}/balance_stats.json" "${packed_dir}/data.list" <<'PY'
import json, pathlib, sys
stats = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(set(stats["output_counts"].values())) == 1, stats
paths = [line.strip() for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
assert paths and all(pathlib.Path(path).is_file() for path in paths)
print("balanced/packed validation passed:", stats["output_counts"], "packed_shards=", len(paths))
PY
