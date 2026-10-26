"""Pre-trained normative charts (PCNtoolkit braincharts), applied as in the authors' transfer notebooks
(braincharts apply_normative_models_*.ipynb):
  - design matrix: create_design_matrix(cov, site_ids, all_sites = the chart's training sites, basis 'bspline', xmin -5, xmax 110);
    the site dummies are all 0 for a site the chart was not trained on;
  - prediction = X @ blr.m in the warped space; the measurement is warped with the model's own warp (blr.warp.f, parameters
    blr.hyp[1:1 + n]).
Also the within-site fold assignment (fixed seed) used to cross-fit the site calibration among a site's own controls.
"""
import pickle
import numpy as np
import pandas as pd

XMIN, XMAX = -5, 110
N_FOLDS, SEED = 10, 20261014


def design_matrix(df, cov_cols, sites_tr):
    """The notebooks' design matrix: intercept, covariates, dummy columns for the TRAINING sites (all 0 for a new site), B-spline of age."""
    from pcntoolkit.util.utils import create_design_matrix
    return create_design_matrix(df[cov_cols], site_ids=df["site"], all_sites=sites_tr, basis="bspline", xmin=XMIN, xmax=XMAX)


def load_chart(mdir, feature):
    with open(f"{mdir}/{feature}/Models/NM_0_0_estimate.pkl", "rb") as h:
        nm = pickle.load(h)
    blr = nm.blr
    wp = blr.hyp[1:1 + blr.warp.get_n_params()]
    return blr, wp


def warp_values(blr, wp, y):
    """warped measurement, blank-safe"""
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan)
    m = np.isfinite(y)
    if m.any():
        out[m] = blr.warp.f(y[m], wp)
    return out


def make_folds(df, n_folds=N_FOLDS, seed=SEED):
    """fold 0..n_folds-1 per person: controls shuffled within site, autistic people shuffled within site (fixed seed)"""
    rng = np.random.default_rng(seed)
    fold = pd.Series(-1, index=df.index, dtype=int)
    for site in sorted(df["site"].unique()):
        for lab in (0, 1):
            idx = df.index[(df["site"] == site) & (df["autism"] == lab)].to_numpy()
            idx = idx[rng.permutation(len(idx))]
            fold.loc[idx] = np.arange(len(idx)) % n_folds
    return fold
