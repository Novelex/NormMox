#!/bin/bash
# Download the ABIDE Preprocessed phenotypic file and the FreeSurfer 5.1 stats files of the 1,035 people (24 downloads at a time).
# Run from the repository root:  bash src/download_phenotypic_and_freesurfer.sh     (or: sbatch src/download_phenotypic_and_freesurfer.sh)
# Files per person: stats/lh.aparc.a2009s.stats, stats/rh.aparc.a2009s.stats, stats/aseg.stats -> data/raw/freesurfer/<FILE_ID>/stats/
# Files already present are skipped; a failed file is removed and listed in data/raw/freesurfer_missing.txt (run again to retry).
#SBATCH --job-name=download_freesurfer
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p data/raw/freesurfer
export BASE=${ABIDE_BASE:-https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative}   # ABIDE Preprocessed
export FILES="stats/lh.aparc.a2009s.stats stats/rh.aparc.a2009s.stats stats/aseg.stats"

echo "== 1. the phenotypic file and the list of FILE_IDs"
[ -s data/raw/Phenotypic_V1_0b_preprocessed1.csv ] || curl -sS -f -L --retry 3 -o data/raw/Phenotypic_V1_0b_preprocessed1.csv "$BASE/Phenotypic_V1_0b_preprocessed1.csv"
python src/ids.py

echo "== 2. download"
fetch() {
  out="data/raw/freesurfer/$1/$2"
  [ -s "$out" ] && return 0
  mkdir -p "$(dirname "$out")"
  curl -sS -f -L --retry 3 --retry-delay 5 -o "$out" "$BASE/Outputs/freesurfer/5.1/$1/$2" 2>/dev/null || { rm -f "$out"; echo "$1 $2"; }
}
export -f fetch
SECONDS=0
for id in $(cat data/raw/file_ids.txt); do for f in $FILES; do echo "$id $f"; done; done \
  | xargs -P 24 -n 2 bash -c 'fetch "$0" "$1"' > data/raw/freesurfer_missing.txt
echo "download finished in $SECONDS s"

echo "== 3. report"
for f in $FILES; do echo "$f: $(ls data/raw/freesurfer/*/$f 2>/dev/null | wc -l) of 1035"; done
echo "failed downloads: $(wc -l < data/raw/freesurfer_missing.txt)"
