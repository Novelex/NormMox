#!/bin/bash
# Download the preprocessed functional scan and ABIDE's brain mask of the 1,035 people (16 downloads at a time, about 110 GB).
# ABIDE Preprocessed, pipeline C-PAC, strategy nofilt_noglobal (no band-pass filter, no global signal regression).
# Run from the repository root after src/download_phenotypic_and_freesurfer.sh (it writes data/raw/file_ids.txt):
#   bash src/download_func.sh     (or: sbatch src/download_func.sh)
# Writes data/raw/func_preproc/<FILE_ID>_func_preproc.nii.gz and data/raw/func_mask/<FILE_ID>_func_mask.nii.gz.
# Files already present are skipped; a failed file is removed and listed in data/raw/func_missing.txt (run again to retry).
#SBATCH --job-name=download_func
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p data/raw/func_preproc data/raw/func_mask
export BASE=${ABIDE_BASE:-https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative}   # ABIDE Preprocessed
[ -s data/raw/file_ids.txt ] || { echo "data/raw/file_ids.txt missing: run src/download_phenotypic_and_freesurfer.sh first"; exit 1; }

fetch() {
  out="data/raw/$2/$1_$2.nii.gz"
  [ -s "$out" ] && return 0
  curl -sS -f -L --retry 3 --retry-delay 5 -o "$out" "$BASE/Outputs/cpac/nofilt_noglobal/$2/$1_$2.nii.gz" 2>/dev/null || { rm -f "$out"; echo "$1 $2"; }
}
export -f fetch
SECONDS=0
for id in $(cat data/raw/file_ids.txt); do for d in func_preproc func_mask; do echo "$id $d"; done; done \
  | xargs -P 16 -n 2 bash -c 'fetch "$0" "$1"' > data/raw/func_missing.txt
echo "download finished in $SECONDS s"
for d in func_preproc func_mask; do echo "$d: $(ls data/raw/$d/*_$d.nii.gz 2>/dev/null | wc -l) of 1035"; done
echo "failed downloads: $(wc -l < data/raw/func_missing.txt)"
