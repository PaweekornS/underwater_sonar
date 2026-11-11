#!/bin/bash
# Run one line of a job list per array task, 1 GPU each.
# Usage: sbatch --array=1-$(wc -l < jobs.txt) [-p gpu-devel -t 2:00:00] scripts/slurm/array.sh jobs.txt
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 6:00:00
#SBATCH -A lt200304
#SBATCH -J sss
#SBATCH -o /scratch/lt200304-dipmt/paweekorn/logs/%x_%A_%a.out

set -euo pipefail
ml Mamba/23.11.0-0
conda activate unsloth_env
export YOLO_OFFLINE=1
cd /project/lt200304-dipmt/paweekorn/sss-project
cmd=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "$cmd"
eval "$cmd"
