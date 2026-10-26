#!/bin/bash
# Leave-one-site-out, 20 folds. From the repository root, with the Python environment active:
#   mkdir -p logs && sbatch --array=0-19 slurm/loso.sbatch
#SBATCH --job-name=loso
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/loso_%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u src/loso.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
