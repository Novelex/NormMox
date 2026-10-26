# Data

This repository contains no imaging or phenotypic data. The only data files it ships are:

- `folds.csv.gz`: the frozen classification folds (scheme, repeat, fold, participant ID, role). Two schemes: `cv10_strat_dx_site`
  (10 repeats of 10-fold cross-validation, stratified by diagnosis and site) and `loso` (leave-one-site-out).
- `thickness_region_to_yeo17.csv`: an atlas-level table assigning each Destrieux thickness region to a Yeo-17 network (used by the
  transformer baseline).

Everything else is downloaded into `data/raw/` and derived into `data/derived/` by the scripts below. Both folders are ignored by git.

## 1. Downloads

**ABIDE I, ABIDE Preprocessed** (public bucket `https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative`):

| What | Path in the bucket | Saved as |
|---|---|---|
| Phenotypic file | `Phenotypic_V1_0b_preprocessed1.csv` | `data/raw/Phenotypic_V1_0b_preprocessed1.csv` |
| FreeSurfer 5.1 stats | `Outputs/freesurfer/5.1/<FILE_ID>/stats/{lh.aparc.a2009s,rh.aparc.a2009s,aseg}.stats` | `data/raw/freesurfer/<FILE_ID>/stats/` |
| C-PAC functional scans (no band-pass filter, no global signal regression) | `Outputs/cpac/nofilt_noglobal/func_preproc/<FILE_ID>_func_preproc.nii.gz` | `data/raw/func_preproc/` |
| C-PAC brain masks | `Outputs/cpac/nofilt_noglobal/func_mask/<FILE_ID>_func_mask.nii.gz` | `data/raw/func_mask/` |

```
bash src/download_phenotypic_and_freesurfer.sh   # also writes data/raw/file_ids.txt (1,035 people)
bash src/download_func.sh                        # about 110 GB
```
Both scripts can also be submitted with `sbatch`; files already present are skipped, so a script can be re-run to retry failures.

**Normative charts** (PCNtoolkit braincharts; Rutherford et al., 2022), at the commit used here:
```
git clone https://github.com/predictive-clinical-neuroscience/braincharts data/raw/braincharts
git -C data/raw/braincharts checkout 5a1a7822e6f54405b0a826cf137d584240af307c
cd data/raw/braincharts/models
unzip -q lifespan_23K_57sites_mqc2.zip                # sMRI chart (covariates: age, sex)
tar xzf lifespan_yeo17_22K_40sites.tar.gz             # fMRI chart, Yeo-17 network pairs (covariates: age, sex, mean FD)
cd -
```

**Yeo et al. (2011) 17-network atlas**, FreeSurfer distribution: `Yeo_JNeurophysiol11_MNI152.zip` (2,654,759 bytes,
md5 `c64533b6396723d473424f96270bfe42`), the file that nilearn's `fetch_atlas_yeo_2011` downloads from
`ftp://surfer.nmr.mgh.harvard.edu/pub/data/Yeo_JNeurophysiol11_MNI152.zip`. Save it as `data/raw/yeo/Yeo_JNeurophysiol11_MNI152.zip`.
Only the thin-mask map `Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz` is used.

## 2. Derived inputs (run from the repository root, in this order)

| Step | Command | Writes (in `data/derived/`) |
|---|---|---|
| 1 | `python src/covariates.py` | `covariates_base.csv`: age, sex, site, mean FD, diagnosis |
| 2 | `python src/smri_features.py` | `structure_features.csv`: 185 sMRI measures (148 Destrieux thickness, 2 mean thickness, 35 volumes) |
| 3 | `python src/fmri_features.py prepare` | `data/raw/yeo/yeo17_thin_3mm.nii.gz`: the atlas on the 3 mm scan grid |
| 4 | `mkdir -p logs && sbatch slurm/fmri_features.sbatch` (or `python src/fmri_features.py`) | `fmri_per_person/<FILE_ID>.npz`: 17 x 17 Fisher-z matrix per scan |
| 5 | `python src/fmri_merge.py` | `yeo17_s0_thin_hp128.csv` (136 network pairs), `coverage.csv` |
| 6 | `python src/chart_inputs.py` | `struct_185.csv`, `func_s0_thin_hp128.csv`, `covariates.csv`, `blank_rows.csv` |
| 7 | `python src/assignments.py` | `assignments.csv`: within-site folds for cross-fitting the site calibration |
| 8 | `python src/chart_residuals.py` | `resid_struct.csv`, `resid_func.csv`: chart residuals before site calibration |
| 9 | `python src/folds.py` (optional) | `folds.csv.gz`, checked to be identical to the shipped `data/folds.csv.gz` |

fMRI connectivity: C-PAC `nofilt_noglobal` scans, ABIDE brain mask, no smoothing, Yeo-17 thin mask, 128-s cosine high-pass filter,
Pearson correlation of the 17 network mean signals, Fisher z.

Nobody is excluded. One person has no mean FD in the phenotypic file (Yale_0050610); it is filled with the site median. Scans with
more than half of the 17 networks outside the brain mask get a blank fMRI row (three Caltech scans); these people have no fMRI evidence
in the models. The fMRI chart uses fMRIPrep (Power) framewise displacement, so mean FD is multiplied by 1.76 before scoring.
