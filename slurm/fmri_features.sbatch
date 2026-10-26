#!/bin/bash
# fMRI features for all 1,035 people as a job array: task t takes every 8th person (t, t+8, ...), 8 people at once per task.
# Restartable: a person whose per-person file exists is skipped.
# From the repository root, after `python src/fmri_features.py prepare`:
#   mkdir -p logs && sbatch slurm/fmri_features.sbatch
#SBATCH --job-name=fmri_features
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fmri_features_%A_%a.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1          # one thread per worker process
export SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))}
SECONDS=0
python src/fmri_features.py
echo "task ${SLURM_ARRAY_TASK_ID} finished in $SECONDS s"
