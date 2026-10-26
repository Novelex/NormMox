#!/bin/bash
# NormMoX, one array task per outer fold. Activate the Python environment (requirements.txt), then from the repository root:
#   mkdir -p logs
#   sbatch --array=0-99 slurm/normmox.sbatch main    # the 100 folds of the 10 x 10 cross-validation (outputs/main)
#   sbatch --array=0-1 slurm/normmox.sbatch perm     # shuffled-label control, repeat 0 folds 0-1 (outputs/perm)
#   sbatch --array=0 slurm/normmox.sbatch smoke      # quick check: K = 3, 2 seeds, 3 epochs (outputs/smoke)
# Leave-one-site-out runs the same code through slurm/loso.sbatch.
#SBATCH --job-name=normmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/normmox_%A_%a.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
python -u src/normmox.py "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT" "${1:-main}"
