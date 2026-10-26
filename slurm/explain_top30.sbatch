#!/bin/bash
# Per-modality deletion: top-k vs random-k inputs reset within fMRI or within sMRI.
# Submit from the repository root with the Python environment active:
#   mkdir -p logs && sbatch --array=0-99 slurm/explain_top30.sbatch
#SBATCH --job-name=explain_top30
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/explain_top30_%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u src/explain_top30.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
