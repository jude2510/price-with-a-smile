#!/bin/bash
#SBATCH -J dpairs
#SBATCH -o logs/dp_%A_%a.out
#SBATCH -e logs/dp_%A_%a.err
#SBATCH -p htc                 # short jobs; great throughput for ~12 min tasks
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-21            # 22 pairs => indices 0..21
# #SBATCH --array=0-6%4        # (optional) throttle to 4 concurrent tasks

set -euo pipefail
mkdir -p logs

# Ensure user-installed packages (e.g., pyarrow) are visible
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"
export MPLBACKEND=Agg

echo "JOB=${SLURM_JOB_ID} TASK=${SLURM_ARRAY_TASK_ID}"

python run_pair.py \
  --pairs pairs.txt \
  --idx ${SLURM_ARRAY_TASK_ID} \
  --outdir artifacts/pairs
