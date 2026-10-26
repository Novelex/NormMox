#!/bin/bash
# Group reset: test AUROC with all fMRI, all sMRI or all inputs reset to typical.
# Submit from the repository root with the Python environment active:
#   mkdir -p logs && sbatch --array=0-99 slurm/explain_reset_all.sbatch
#SBATCH --job-name=explain_reset_all
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/explain_reset_all_%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u src/explain_reset_all.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
