# NormMoX

Code for **NormMoX: Explainable multimodal autism classification using deviations from normative brain charts**
(Muhammad Hasan Masrur and Fani Deligianni, University of Glasgow).

NormMoX scores 185 structural MRI measures and 136 resting-state network connectivities of ABIDE I (1,035 participants, 20 site
samples) against pretrained lifespan normative charts, calibrates the deviations to each site with training-fold controls only
(Eq. 1), trains one TabM-mini expert per modality, and fuses the two with a 7-parameter mixture-of-experts head (Eqs. 2-4).
Typical-brain integrated gradients explain each prediction (Eq. 5).

## Setup

Python 3.11, CPU only.

```
pip install -r requirements.txt
```

Run every command from the repository root. Paths, seeds and hyperparameters are in `settings.json`.

## Quick check: Table 1 from the reference results

```
python src/table1.py
```

`results/per_fold_metrics.csv` holds the per-fold metrics of every Table 1 row (8 models x 100 test folds of the 10 x 10
cross-validation). The script prints Table 1 (mean +- SD over repeats) and the comparison of NormMoX with every other row (corrected
resampled t-test, Holm correction).

## Full pipeline

**1. Data.** Download ABIDE I (ABIDE Preprocessed), the two normative charts and the Yeo-17 atlas, then build the derived inputs:
see [data/README.md](data/README.md). No participant data are included in this repository.

**2. Table 1.** The array jobs run one outer fold per task (2 CPUs each); wait for each step to finish before the next.

```
mkdir -p logs
sbatch --array=0-99 slurm/normmox.sbatch main     # NormMoX, sMRI expert, fMRI expert
python src/logistic.py                             # logistic-regression rows
sbatch slurm/fmri_branch.sbatch                    # deep fusion baselines: fMRI expert on a 70/15/15 split,
sbatch slurm/fusion_frozen.sbatch                  #   then experts frozen + MoX head,
sbatch slurm/fusion_finetuned.sbatch               #   then jointly fine-tuned
python src/per_fold.py                             # -> outputs/per_fold_metrics.csv
python src/table1.py --per-fold outputs/per_fold_metrics.csv
```

**3. Leave-one-site-out.**

```
sbatch --array=0-19 slurm/loso.sbatch
python src/loso_summary.py
```

**4. Explanations and Table 2.**

```
sbatch --array=0-1 slurm/normmox.sbatch perm      # shuffled-label control (repeat 0, folds 0-1)
sbatch --array=0-99 slurm/explain.sbatch           # integrated gradients from the typical brain (64 steps)
sbatch --array=0-1 slurm/explain.sbatch perm
sbatch --array=0-99 slurm/explain_refine.sbatch    # 512 steps where the completeness error exceeds 0.01
sbatch --array=0-99 slurm/explain_reset_all.sbatch # AUROC with fMRI, sMRI or all inputs reset to typical
sbatch --array=0-99 slurm/explain_top30.sbatch     # top-30 vs random-30 inputs reset within each modality
python src/explain_summary.py
python src/table2.py
```

**5. Figures 2 and 3.**

```
python src/fig_maps.py
python src/fig_values.py
python src/fig_stats.py
python src/figures.py                              # -> outputs/figures
```

Without Slurm, run an array script directly for every task, e.g. `python src/normmox.py TASK 100 main` for TASK = 0, ..., 99.
The main 10 x 10 run used about 24 CPU-hours.

## Code

| File | Contents |
|---|---|
| `src/charts.py`, `src/chart_residuals.py` | pretrained charts applied to ABIDE; residuals before site calibration |
| `src/rulers.py`, `src/data.py` | site calibration inside each fold (Eq. 1); fold inputs |
| `src/tabm.py`, `src/fusion.py`, `src/normmox.py` | TabM-mini experts, MoX head, cross-fitting (5 seeds x 10 parts) and test-time averaging (Eqs. 2-4) |
| `src/logistic.py` | logistic-regression baselines (raw, ABIDE-normed, chart deviations) |
| `src/transformer.py`, `src/fmri_branch.py`, `src/fusion_frozen.py`, `src/fusion_finetuned.py` | deep fusion baselines |
| `src/explain*.py`, `src/table2.py` | typical-brain integrated gradients (Eq. 5) and Table 2 |
| `src/loso.py`, `src/loso_summary.py` | leave-one-site-out |
| `src/stats.py`, `src/per_fold.py`, `src/table1.py` | metrics, corrected resampled t-test, Holm correction, Table 1 |
| `src/fig_*.py`, `src/figures.py` | Figures 2 and 3 |

## License

MIT (see [LICENSE](LICENSE)).
