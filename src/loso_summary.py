"""Leave-one-site-out summary: AUROC within each held-out site that has >= 5 autistic and >= 5 control people, averaged over those
sites (unweighted; the n-weighted mean and the pooled AUROC over all test people are given as well), for NormMoX and the two experts.
usage: python src/loso_summary.py -> outputs/{loso_summary.txt, loso_summary.json, loso_per_site.csv}"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
ARMS = {"MOX": "NormMoX", "F_TABM": "fMRI expert", "S_TABM": "sMRI expert"}
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def check(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def main():
    cov = pd.read_csv(os.path.join(ROOT, S["paths"]["covariates"])).set_index("FILE_ID")
    site_name = cov.groupby("sitenum")["site"].first().to_dict()
    preds = sorted(glob.glob(f"{OUT}/loso/pred/r0_f*.csv.gz"))
    check(len(preds) == 20, f"{len(preds)} LOSO prediction files (need 20)")
    P = {int(os.path.basename(p).split("_f")[1].split(".")[0]): json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/loso/params/r0_f*.json"))}
    check(len(P) == 20 and all(v["complete"] for v in P.values()), "20 complete LOSO params files")
    rows = []
    for p in preds:
        fold = int(os.path.basename(p).split("_f")[1].split(".")[0])
        d = pd.read_csv(p)
        w = d.pivot_table(index=["FILE_ID", "y"], columns="arm", values="score").reset_index()
        w["fold"] = fold
        w["sitenum"] = w["FILE_ID"].map(cov["sitenum"])
        check(w["sitenum"].nunique() == 1, f"fold {fold}: more than one site in the test fold")
        rows.append(w)
    W = pd.concat(rows, ignore_index=True)
    per = []
    for fold, g in W.groupby("fold"):
        n1, n0 = int((g["y"] == 1).sum()), int((g["y"] == 0).sum())
        r = dict(fold=fold, sitenum=int(g["sitenum"].iloc[0]), site=site_name[int(g["sitenum"].iloc[0])], n_aut=n1, n_ctrl=n0, qualifies=n1 >= 5 and n0 >= 5)
        for a in ARMS:
            r[a] = float(roc_auc_score(g["y"], g[a])) if r["qualifies"] else np.nan
        per.append(r)
    T = pd.DataFrame(per)
    T.to_csv(f"{OUT}/loso_per_site.csv", index=False)
    Q = T[T["qualifies"]]
    say(f"CHECKS: 20 LOSO folds complete; {len(Q)} sites with >= 5 autistic and >= 5 controls (excluded: "
        f"{', '.join(T[~T['qualifies']]['site'].tolist()) or 'none'})")
    say("\nLEAVE-ONE-SITE-OUT (train on 19 sites, test on the held-out site; the held-out site's own controls calibrate its site offsets)")
    say(f"  {'model':22s} {'mean AUROC over sites':>22s} {'SD':>7s} {'n-weighted':>11s} {'pooled':>8s}")
    report = dict(arms={}, sites=T.to_dict("records"))
    for a, name in ARMS.items():
        wmean = float(np.average(Q[a], weights=Q["n_aut"] + Q["n_ctrl"]))
        pooled = float(roc_auc_score(W["y"], W[a]))
        report["arms"][name] = dict(mean_over_sites=float(Q[a].mean()), sd_over_sites=float(Q[a].std(ddof=1)), n_weighted=wmean, pooled=pooled)
        say(f"  {name:22s} {Q[a].mean():22.4f} {Q[a].std(ddof=1):7.4f} {wmean:11.4f} {pooled:8.4f}")
    say("\nPer site (AUROC; '-' = fewer than 5 autistic or 5 controls)")
    say(f"  {'site':16s} {'n_aut':>5s} {'n_ctrl':>6s} " + " ".join(f"{n:>12s}" for n in ARMS.values()))
    for _, r in T.sort_values("sitenum").iterrows():
        vals = " ".join(f"{r[a]:12.3f}" if r["qualifies"] else f"{'-':>12s}" for a in ARMS)
        say(f"  {r['site'][6:]:16s} {r['n_aut']:5d} {r['n_ctrl']:6d} {vals}")
    json.dump(report, open(f"{OUT}/loso_summary.json", "w"), indent=1, default=float)
    open(f"{OUT}/loso_summary.txt", "w").write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
