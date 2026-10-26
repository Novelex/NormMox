"""Classification folds, one long table: scheme, repeat, fold, FILE_ID, role (train / test). Fixed seeds, so the table is
reproducible to the byte.
  cv10_strat_dx_site  10 repeats x 10-fold, folds balanced for diagnosis and site (sklearn StratifiedKFold on the 40 diagnosis x site
                      strata), seed 20261015 + repeat
  loso                leave-one-site-out: 20 folds, one site sample = test, the other 19 = train
Site calibration uses the fold's training controls; in loso the test site has no training controls, so its people are calibrated by
cross-fitting among that site's own controls.
Reads data/derived/covariates.csv; writes data/derived/folds.csv.gz and checks it against the shipped data/folds.csv.gz.
Checks: everyone is test exactly once per repeat; every site with test people keeps >= 2 training controls in every cv10 fold.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = OUT = f"{ROOT}/data/derived"
SHIPPED = f"{ROOT}/data/folds.csv.gz"
BASE_SEED, K, REP_CV = 20261015, 10, 10
ok = True


def check(cond, msg):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {msg}")
    ok = ok and bool(cond)


def build(cv):
    ids = np.array(cv.index)
    dx = cv["autism"].to_numpy(int)
    site = cv["site"].to_numpy()
    dx_site = np.array([f"{s}_{d}" for s, d in zip(site, dx)])
    idx = np.arange(len(ids))
    rows = []

    def add(scheme, rep, fold, role, members):
        rows.append(pd.DataFrame({"scheme": scheme, "repeat": rep, "fold": fold, "FILE_ID": ids[members], "role": role}))

    scheme, n = "cv10_strat_dx_site", 0
    for rep in range(REP_CV):
        seed = BASE_SEED + 100 * n + rep
        splitter = StratifiedKFold(K, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(splitter.split(idx, dx_site)):
            add(scheme, rep, fold, "train", tr); add(scheme, rep, fold, "test", te)
    for fold, s in enumerate(sorted(set(site))):
        te = idx[site == s]; tr = idx[site != s]
        add("loso", 0, fold, "train", tr); add("loso", 0, fold, "test", te)
    return pd.concat(rows, ignore_index=True)


cv = pd.read_csv(f"{INP}/covariates.csv").set_index("FILE_ID")
check(len(cv) == 1035, f"covariates: {len(cv)} people")
folds = build(cv)
again = build(cv)
check(folds.equals(again), "reproducible: building the folds twice gives identical tables (fixed seeds)")
os.makedirs(OUT, exist_ok=True)
folds.to_csv(f"{OUT}/folds.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

site_of = cv["site"]; dx_of = cv["autism"]
min_adapt_overall = 10 ** 9
for (scheme, rep), g in folds.groupby(["scheme", "repeat"]):
    test_counts = g[g["role"] == "test"].groupby("FILE_ID").size()
    check(len(test_counts) == 1035 and (test_counts == 1).all(), f"{scheme} repeat {rep}: everyone is test exactly once")
    if scheme == "loso":
        continue
    for fold, gf in g.groupby("fold"):
        tr = gf.loc[gf["role"] == "train", "FILE_ID"]; te = gf.loc[gf["role"] == "test", "FILE_ID"]
        tr_ctrl_per_site = site_of.loc[tr][dx_of.loc[tr] == 0].value_counts()
        min_adapt_overall = min(min_adapt_overall, int(min(tr_ctrl_per_site.get(s, 0) for s in set(site_of.loc[te]))))
check(min_adapt_overall >= 2, f"every cv10 fold: each site with test people keeps >= 2 training controls (minimum found {min_adapt_overall})")
if os.path.exists(SHIPPED):
    check(pd.read_csv(SHIPPED).equals(pd.read_csv(f"{OUT}/folds.csv.gz")), f"identical to the shipped {os.path.relpath(SHIPPED, ROOT)}")
print(f"written: {OUT}/folds.csv.gz ({len(folds)} rows)")
print("CHECK: PASS" if ok else "CHECK: FAIL")
sys.exit(0 if ok else 1)
