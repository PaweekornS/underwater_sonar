#!/bin/bash
# Total GPU-hours used by this project's Slurm jobs (job name "sss") since the remake started.
# Budget: 100 GPU-hours. Usage: bash scripts/slurm/gpu_hours.sh
sacct -X -n -P -S 2026-11-12 --name=sss -o JobID,ElapsedRaw,AllocTRES |
  awk -F'|' '{ g = 0; if (match($3, /gres\/gpu=[0-9]+/)) g = substr($3, RSTART + 9, RLENGTH - 9);
               s += $2 * g } END { printf "GPU-hours used: %.1f / 100\n", s / 3600 }'
