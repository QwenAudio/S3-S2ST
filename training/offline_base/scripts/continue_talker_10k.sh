#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec 8>"$root/pipeline.lock"
flock -n 8 || exit 2
export RESUME_TRAINING=1 TRAIN_ITERS=10000 EXIT_INTERVAL=10000 LR_DECAY_ITERS=10000 SAVE_INTERVAL=500
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
printf 'talker: resuming 5000 -> 10000; full evaluation on H100\n' > "$root/PIPELINE_STATUS"
if bash "$root/scripts/train.sh" talker "$root/runs/talker/iter_0005000"; then
 printf 'talker: completed 10000; awaiting next stage decision\n' > "$root/PIPELINE_STATUS"
else
 printf 'talker continuation failed; inspect audit/continue_talker_10k.log\n' > "$root/PIPELINE_FAILED"
 exit 1
fi
