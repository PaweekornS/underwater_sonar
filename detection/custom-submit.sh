#!/bin/bash
#SBATCH -p gpu                          # Partition name (standard GPU on LANTA)
#SBATCH -N 1                            # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=16              # CPU cores (matches our workers logic)
#SBATCH --gres=gpu:1                    # Request 1 GPU (change to 2 or 4 for DDP)
#SBATCH --mem=64G                       # Memory request
#SBATCH -t 24:00:00                     # Walltime (HH:MM:SS)
#SBATCH -J custom                       # Job name
#SBATCH -o logs/job_%j.out              # Standard output log

# --- Run the Automation Pipeline ---
echo "🤖 Starting YOLO Automation Pipeline..."

cd /project/lt200304-dipmt/paweekorn/sss_project
python train_custom.py --config 11m-sonar.yaml --epochs 200 --patience 10

echo "✅ Job completed at $(date)"