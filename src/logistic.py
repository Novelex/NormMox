"""Logistic-regression baselines of Table 1 (L2, C = 1) on 150 cortical thickness + 136 fMRI measures, built inside each training fold:
  lr_raw            raw values with age, age^2, sex and site (+ mean FD for fMRI) regressed out (least squares on the training people)
  lr_abide_normed   raw values standardised per site with the training controls' mean and SD
  lr_deviations     chart deviations (Eq. 1)
Blanks get the training median; inputs are standardised on the training people.

usage: python src/logistic.py
writes outputs/logistic/predictions.csv.gz (test scores) and outputs/per_fold_logistic.csv (per-fold metrics, read by per_fold.py)
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import ROOT, S, Data
from stats import METRICS, metrics

P = S["paths"]
D = Data()
cv = pd.read_csv(os.path.join(ROOT, P["covariates"])).set_index("FILE_ID").loc[D.ids]
n = D.n
age, sex, fd = cv["age"].to_numpy(float), cv["sex"].to_numpy(float), cv["mean_FD_power"].to_numpy(float)
site_onehot = pd.get_dummies(cv["site"]).astype(float).to_numpy()
struct_raw = pd.read_csv(os.path.join(ROOT, P["struct_inputs"])).set_index("FILE_ID").loc[D.ids]
func_raw = pd.read_csv(os.path.join(ROOT, P["func_inputs"])).set_index("FILE_ID").loc[D.ids]
thick = [c for c in D.struct_cols if c.endswith("_thickness")]
assert len(thick) == 150 and list(D.struct_cols) == list(struct_raw.columns), "structure columns"
assert list(D.func_cols) == list(func_raw.columns), "fMRI columns"


def residualise(X, train, with_fd):
    """remove age, age^2, sex, site (+ mean FD) with a least-squares fit on the training people (per feature, blanks skipped)"""
    C = np.column_stack([np.ones(n), age, age ** 2, sex, site_onehot] + ([fd] if with_fd else []))
    C = C[:, [0, 1, 2, 3] + [4 + k for k in range(site_onehot.shape[1]) if site_onehot[train, k].sum() > 0] + ([C.shape[1] - 1] if with_fd else [])]
    Xr = np.full(X.shape, np.nan)
    for j in range(X.shape[1]):
        m = train & np.isfinite(X[:, j])
        if m.sum() < C.shape[1] + 2:
            continue
        beta = np.linalg.lstsq(C[m], X[m, j], rcond=None)[0]
        Xr[:, j] = X[:, j] - C @ beta
    return Xr


def impute_median(X, train):
    med = np.nanmedian(X[train], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med)
    return Xi


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def arms_for_fold(rep, fold):
    train, test = D.masks(rep, fold)
    D.eng["scheme"], D.eng["rep"] = S["scheme"], int(rep)
    is_adapter, adapt_fold = D.eng["ruler_setup"](train, test, int(fold))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zs = pd.DataFrame(D.eng["site_score"](D.Rs.to_numpy(float), is_adapter, adapt_fold)[0], columns=D.struct_cols)
        zf = D.eng["site_score"](D.Rf.to_numpy(float), is_adapter, adapt_fold)[0]
        rz = pd.DataFrame(D.eng["site_score"](struct_raw.to_numpy(float), is_adapter, adapt_fold)[0], columns=struct_raw.columns)
        rzf = D.eng["site_score"](func_raw.to_numpy(float), is_adapter, adapt_fold)[0]
    rs = pd.DataFrame(residualise(struct_raw.to_numpy(float), train, False), columns=struct_raw.columns)
    rf = residualise(func_raw.to_numpy(float), train, True)
    arms = dict(lr_raw=np.hstack([rs[thick].to_numpy(), rf]),
                lr_abide_normed=np.hstack([rz[thick].to_numpy(), rzf]),
                lr_deviations=np.hstack([zs[thick].to_numpy(), zf]))
    return train, test, arms


def main():
    out = os.path.join(ROOT, P["outputs"])
    os.makedirs(os.path.join(out, "logistic"), exist_ok=True)
    rows, per_fold = [], []
    for rep, fold in D.fold_list():
        train, test, arms = arms_for_fold(rep, fold)
        for name, X in arms.items():
            Xi = impute_median(X, train)
            s = np.round(fit_predict(Xi[train], D.y[train], Xi[test]), 6)
            rows.append(pd.DataFrame(dict(model=name, repeat=rep, fold=fold, FILE_ID=D.ids[test], y=D.y[test], score=s)))
            per_fold.append(dict(model=name, repeat=rep, fold=fold, **metrics(D.y[test], s, 0.5)))
        print(f"repeat {rep} fold {fold}: " + ", ".join(f"{r['model']} {r['auroc']:.3f}" for r in per_fold[-3:]), flush=True)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(out, "logistic", "predictions.csv.gz"), index=False)
    pd.DataFrame(per_fold)[["model", "repeat", "fold", *METRICS]].to_csv(os.path.join(out, "per_fold_logistic.csv"), index=False)


if __name__ == "__main__":
    main()
