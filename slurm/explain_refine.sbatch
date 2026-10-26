#!/bin/bash
# 512-step rerun of the people whose 64-step completeness error exceeds 0.01.
# Submit from the repository root with the Python environment active:
#   mkdir -p logs && sbatch --array=0-99 slurm/explain_refine.sbatch
#SBATCH --job-name=explain_refine
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/explain_refine_%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u src/explain_refine.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
