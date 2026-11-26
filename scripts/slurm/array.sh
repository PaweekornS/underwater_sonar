#!/bin/bash
# Run PACK lines of a job list concurrently on one GPU per array task.
# Do not sbatch this directly: submit through scripts/slurm/submit.sh (caps running GPUs at 4).
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 6:00:00
#SBATCH -A lt200304
#SBATCH -J sss
#SBATCH -o /scratch/lt200304-dipmt/paweekorn/logs/%x_%A_%a.out

set -uo pipefail
ml Mamba/23.11.0-0
conda activate unsloth_env
export YOLO_OFFLINE=1
cd /project/lt200304-dipmt/paweekorn/sss-project
PACK=${PACK:-1}
first=$(( (SLURM_ARRAY_TASK_ID - 1) * PACK + 1 ))
for n in $(seq "$first" $(( first + PACK - 1 ))); do
    cmd=$(sed -n "${n}p" "$1")
    [ -z "$cmd" ] && continue
    echo "[$n] $cmd"
    eval "$cmd --workers $(( 16 / PACK ))" > "/scratch/lt200304-dipmt/paweekorn/logs/line_${SLURM_ARRAY_JOB_ID}_${n}.log" 2>&1 &
done
wait
