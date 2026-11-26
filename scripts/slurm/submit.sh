#!/bin/bash
# The only way to submit project jobs. It caps the cluster footprint at MAX_RUNNING GPUs:
#   - "%MAX_RUNNING" throttles each array, and
#   - "--dependency=singleton" (same job name "sss") lets only one sss array run at a time.
# Usage: bash scripts/slurm/submit.sh <jobs.txt> [PACK=3] [extra sbatch args...]
set -euo pipefail
MAX_RUNNING=4
jobs=$1; pack=${2:-3}; shift $(( $# >= 2 ? 2 : 1 ))
n=$(grep -c . "$jobs")
tasks=$(( (n + pack - 1) / pack ))
cd "$(dirname "$0")/../.."
sbatch --array=1-${tasks}%${MAX_RUNNING} --dependency=singleton --export=ALL,PACK=$pack "$@" \
    scripts/slurm/array.sh "$jobs"
