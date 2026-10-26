"""Within-site fold assignment used to cross-fit the site calibration among a site's own controls, fixed before any scoring.
Reads data/derived/covariates.csv. Writes data/derived/assignments.csv, one row per person: site, sitenum, autism, fold10 (fold 0-9;
controls shuffled within site, autistic people shuffled within site; fixed seed 20261014, charts.make_folds). A person in fold j is
calibrated with the site's controls outside fold j (10-fold, as in Wolfers et al. 2021); autistic people get a fold too, so both groups
are calibrated with the same number of controls. Nobody is removed.
"""
import os, sys
import numpy as np
import pandas as pd

from charts import make_folds, N_FOLDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = f"{ROOT}/data/derived"
ok = True


def check(cond, msg):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {msg}")
    ok = ok and bool(cond)


cv = pd.read_csv(f"{INP}/covariates.csv").set_index("FILE_ID")
check(len(cv) == 1035 and {"site", "sitenum", "autism"} <= set(cv.columns), f"covariates.csv: {len(cv)} people with site and autism")
out = cv[["site", "sitenum", "autism"]].copy()
out["fold10"] = make_folds(cv).values
rows = []
for site in sorted(cv["site"].unique()):
    ctrl = out.index[(out["site"] == site) & (out["autism"] == 0)]
    rows.append(dict(site=site, controls=len(ctrl), autistic=int(((out["site"] == site) & (out["autism"] == 1)).sum()),
                     fold_sizes_controls=np.bincount(out.loc[ctrl, "fold10"], minlength=N_FOLDS).tolist()))
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))

check((out["fold10"] >= 0).all() and out["fold10"].nunique() == N_FOLDS, f"every person has a fold 0-{N_FOLDS - 1}; fold sizes overall {np.bincount(out['fold10']).tolist()}")
min_adapt = min(len(c) - max(np.bincount(out.loc[c, 'fold10'], minlength=N_FOLDS)) for c in [out.index[(out['site'] == s) & (out['autism'] == 0)] for s in tab['site']])
check(min_adapt >= 2, f"smallest number of calibrating controls in any site x fold = {min_adapt} (the toolkit needs >= 2)")

out.to_csv(f"{INP}/assignments.csv")
again = make_folds(cv).values
check((again == out["fold10"].values).all(), "reproducible: rebuilding the folds with the fixed seed gives the same file")
print(f"written: {INP}/assignments.csv ({out.shape[0]} x {out.shape[1]}: site, sitenum, autism, fold10)")
print("CHECK: PASS" if ok else "CHECK: FAIL")
sys.exit(0 if ok else 1)
