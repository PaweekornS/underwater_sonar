#!/bin/bash
# Run every line of a command list in sequence on one CPU node (synthetic data generation, data prep).
# Usage: sbatch --dependency=singleton scripts/slurm/cpu.sh <commands.txt>
#SBATCH -p compute
#SBATCH -N 1 -c 32
#SBATCH -t 4:00:00
#SBATCH -A lt200304
#SBATCH -J sss
#SBATCH -o /scratch/lt200304-dipmt/paweekorn/logs/cpu_%j.out

set -uo pipefail
ml Mamba/23.11.0-0
conda activate unsloth_env
cd /project/lt200304-dipmt/paweekorn/sss-project
while read -r cmd; do
    [ -z "$cmd" ] && continue
    echo "+ $cmd"
    eval "$cmd" || echo "FAILED: $cmd"
done < "$1"
