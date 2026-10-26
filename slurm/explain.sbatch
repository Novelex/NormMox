#!/bin/bash
# Explanation maps (integrated gradients) and deletion faithfulness for every test fold.
# Submit from the repository root with the Python environment active:
#   mkdir -p logs && sbatch --array=0-99 slurm/explain.sbatch
#   shuffled-label folds (after normmox.py perm): sbatch --array=0-1 slurm/explain.sbatch perm
#SBATCH --job-name=explain
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/explain_%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u src/explain.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT ${1:-main}
