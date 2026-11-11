#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Same scheduler convention as the previous multinode launcher:
# WORLD_SIZE = node count, RANK = node rank (not total GPU count).
export NNODES=${NNODES:-${WORLD_SIZE:?Set NNODES or scheduler WORLD_SIZE to node count}}
export NODE_RANK=${NODE_RANK:-${RANK:?Set NODE_RANK or scheduler RANK}}
export MASTER_ADDR=${MASTER_ADDR:?Set MASTER_ADDR to node 0 reachable address}
export MASTER_PORT=${MASTER_PORT:-12981}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
[[ $NNODES =~ ^[0-9]+$ ]] && ((NNODES>1)) || { echo 'Multinode launcher requires NNODES > 1.' >&2; exit 2; }
if [[ ${DRY_RUN:-0} != 1 ]]; then
 echo "Checking ffmpeg on $(hostname)"
 if ! ffmpeg -version >/dev/null 2>&1; then
  echo "Repairing libGL on $(hostname)"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install --reinstall -y libgl1
  ldconfig
 fi
 if ! ffmpeg -version; then
  echo "ERROR: ffmpeg is still broken on $(hostname)" >&2
  ls -lhL /usr/lib/x86_64-linux-gnu/libGL.so.1 || true
  exit 1
 fi
fi
export GLOBAL_BATCH_SIZE=64
export EXP_DIR=${EXP_DIR:-$root/runs/gpu${NNODES}x${NPROC_PER_NODE}_gbs64}
export RESUME_TRAINING=${RESUME_TRAINING:-0}
cd ${S3_ROOT}/code/megatron
bash "$root/scripts/run.sh"
echo 'now exit'
