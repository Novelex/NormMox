#!/bin/bash
# fMRI expert of the deep fusion baselines (src/fmri_branch.py), the 100 folds spread over the array tasks.
# Submit from the repository root with the Python environment active: sbatch slurm/fmri_branch.sbatch
#SBATCH --job-name=fmri_branch
#SBATCH --array=0-99
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u src/fmri_branch.py "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT"
