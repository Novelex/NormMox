#!/bin/bash
# Deep fusion baseline fine-tuned jointly (src/fusion_finetuned.py); needs the outputs of slurm/fusion_frozen.sbatch.
# Submit from the repository root with the Python environment active: sbatch slurm/fusion_finetuned.sbatch
#SBATCH --job-name=fusion_finetuned
#SBATCH --array=0-99
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -u src/fusion_finetuned.py "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT"
