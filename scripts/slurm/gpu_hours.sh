#!/bin/bash
# Budget check for the remake: 1000 LANTA service hours (SHr) from 2026-11-12 (project usage then: 3655.04 SHr).
# SHr is project-wide (other members count too); GPU-hours are this project's own "sss" jobs.
# Usage: bash scripts/slurm/gpu_hours.sh
START_SHR=3655.04
BUDGET_SHR=1000
sacct -X -n -P -S 2026-11-12 --name=sss -o JobID,ElapsedRaw,AllocTRES |
  awk -F'|' '{ g = 0; if (match($3, /gres\/gpu=[0-9]+/)) g = substr($3, RSTART + 9, RLENGTH - 9);
               s += $2 * g } END { printf "GPU-hours (sss jobs): %.1f\n", s / 3600 }'
used=$(sbalance 2>/dev/null | awk '$1 == "lt200304" { print $5 }')
awk -v u="$used" -v s="$START_SHR" -v b="$BUDGET_SHR" \
  'BEGIN { printf "SHr used since start: %.0f / %d (project-wide)\n", u - s, b }'
