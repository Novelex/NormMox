"""Model inputs, built inside each fold.
- z: chart deviations with site rulers from the training-fold controls only (rulers.py, Eq. 1).
- 3-way split of the training fold per seed, stratified by diagnosis x site: 70 % training / 15 % early stopping / 15 % head fitting
  (deep fusion baselines).
- tokens: a blank cell gets the training-fold median and a missing flag; then every feature is standardised with the training-fold
  mean and SD.
- people whose fMRI residual row is fully blank (no usable fMRI) are flagged."""
import json
import os
import warnings

import numpy as np
import pandas as pd

import rulers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(ROOT, "settings.json")))
P = S["paths"]
SPLIT_SEED = 20261019                          # base seed of the 3-way split


class Data:
    def __init__(self):
        cov = os.path.join(ROOT, P["covariates"])
        cv = pd.read_csv(cov).set_index("FILE_ID")
        self.ids = np.array(cv.index)
        self.n = len(self.ids)
        self.sitenum = cv["sitenum"].to_numpy(int)
        self.y = cv["autism"].to_numpy(int)
        site_fold = pd.read_csv(os.path.join(ROOT, P["assignments"])).set_index("FILE_ID").loc[self.ids, "fold10"].to_numpy(int)
        self.Rs = pd.read_csv(os.path.join(ROOT, P["resid_struct"])).set_index("FILE_ID").loc[self.ids]
        self.Rf = pd.read_csv(os.path.join(ROOT, P["resid_func"])).set_index("FILE_ID").loc[self.ids]
        self.struct_cols, self.func_cols = list(self.Rs.columns), list(self.Rf.columns)
        self.blank_fmri = self.Rf.isna().all(axis=1).to_numpy()
        self.eng = vars(rulers)                    # namespace of the ruler functions; scheme and rep are set per fold
        self.eng.update(n=self.n, sitenum=self.sitenum, y_all=self.y, site_fold=site_fold)
        self.scheme = S["scheme"]
        fo = pd.read_csv(os.path.join(ROOT, P["folds"]))
        self.folds = fo[fo["scheme"] == self.scheme]
        self.pos = {fid: i for i, fid in enumerate(self.ids)}

    def fold_list(self):
        return sorted({(int(r), int(f)) for r, f in zip(self.folds["repeat"], self.folds["fold"])})

    def masks(self, rep, fold):
        g = self.folds[(self.folds["repeat"] == rep) & (self.folds["fold"] == fold)]
        idx, role = g["FILE_ID"].map(self.pos).to_numpy(), g["role"].to_numpy()
        train, test = np.zeros(self.n, bool), np.zeros(self.n, bool)
        train[idx[role == "train"]] = True
        test[idx[role == "test"]] = True
        assert not (train & test).any(), "a person is both train and test"
        return train, test

    def fold_z(self, rep, fold):
        """z for everyone, rulers from this fold's training controls"""
        train, test = self.masks(rep, fold)
        self.eng["scheme"], self.eng["rep"] = self.scheme, int(rep)
        is_adapter, adapt_fold = self.eng["ruler_setup"](train, test, int(fold))
        with warnings.catch_warnings():                # silences numpy's "empty slice" notes for rulers with < 2 values (z stays blank)
            warnings.simplefilter("ignore", RuntimeWarning)
            zs = self.eng["site_score"](self.Rs.to_numpy(float), is_adapter, adapt_fold)[0]
            zf = self.eng["site_score"](self.Rf.to_numpy(float), is_adapter, adapt_fold)[0]
        return dict(train=train, test=test, is_adapter=is_adapter, zs=zs, zf=zf)

    def split3(self, train, rep, fold, seed):
        """training / early-stopping / head-fitting rows inside the training fold: stratified by diagnosis x site, different for
        every seed"""
        rng = np.random.default_rng([SPLIT_SEED, int(rep), int(fold), int(seed)])
        cuts = np.array([S["split"]["train"], S["split"]["train"] + S["split"]["val_es"]])
        part = np.full(self.n, -1)
        for dx, st in sorted(set(zip(self.y[train].tolist(), self.sitenum[train].tolist()))):
            members = np.where(train & (self.y == dx) & (self.sitenum == st))[0]
            members = members[rng.permutation(len(members))]
            frac = (np.arange(len(members)) + rng.random()) / len(members)
            part[members] = np.searchsorted(cuts, frac, side="right")          # 0 training, 1 early stopping, 2 head fitting
        return part == 0, part == 1, part == 2

    @staticmethod
    def tokens(Z, fit):
        """blank -> training-fold median (0 if the whole column is blank) + miss flag 1; then standardise with training-fold mean / SD"""
        miss = ~np.isfinite(Z)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(Z[fit], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = np.where(miss, med, Z)
        mu, sd = X[fit].mean(axis=0), X[fit].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        return ((X - mu) / sd).astype(np.float32), miss.astype(np.float32)
