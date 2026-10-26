"""Figure 3b numbers: per-person atypical-evidence patterns in all 10 repeats (each person once per repeat). A person's fMRI or sMRI
evidence is flagged when it exceeds the 95th percentile of the training controls' out-of-fold evidence of that fold; people without an
fMRI scan are never fMRI-flagged. Shares per group and pattern are averaged over the repeats and checked against
outputs/explain_summary.json; one-sided Fisher exact test (autistic > controls) per pattern and repeat, the largest p over the 10
repeats is reported.
Writes outputs/isbi/fig2_patterns.json.
usage: python src/fig_stats.py"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import fusion as fu

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/outputs"


def nan0(a):
    return np.where(np.isfinite(a), a, 0.0)


def evidence_per_fold(oo, pp):
    """test people's e_F, e_S (mean over the inner models) and the training controls' OOF e_F, e_S (mean over seeds)"""
    m = pp["heads"]["MOX"]
    p, cF, cS = np.array(m["p"]), float(m["cF"]), float(m["cS"])
    sF_, sS_ = fu.softplus(p[4]), fu.softplus(p[5])
    train, test, has = oo["train"], oo["test"], oo["has_fmri"]
    eF_tr, eS_tr, eF_te, eS_te = [], [], [], []
    for s in range(pp["n_seeds"]):
        c = pp["seed_constants"][s]
        zF = np.where(has, (nan0(oo["lF_oof"][:, s]) - c["muF"]) / c["sdF"], 0.0)
        zS = (nan0(oo["lS_oof"][:, s]) - c["muS"]) / c["sdS"]
        eF_tr.append(np.where(has, sF_ * zF + cF, np.nan)[train])
        eS_tr.append((sS_ * zS + cS)[train])
        for j in range(pp["k"]):
            zFt = np.where(has, (nan0(oo["lF_test"][:, s, j]) - c["muF"]) / c["sdF"], 0.0)[test]
            zSt = ((oo["lS_test"][:, s, j] - c["muS"]) / c["sdS"])[test]
            eF_te.append(np.where(has[test], sF_ * zFt + cF, np.nan))
            eS_te.append(sS_ * zSt + cS)
    with np.errstate(all="ignore"):
        return (np.nanmean(eF_tr, axis=0), np.mean(eS_tr, axis=0), np.nanmean(eF_te, axis=0), np.mean(eS_te, axis=0),
                oo["y_fit"][train] == 0, oo["ids"][test])


E = json.load(open(f"{RES}/explain_summary.json"))["E"]
pat_names = ("F_only", "S_only", "both", "any", "none")
rows = []
for rep in range(10):
    yy, F, S = [], [], []
    for fold in range(10):
        tag = f"r{rep}_f{fold}"
        pp = json.load(open(f"{RES}/main/params/{tag}.json"))
        oo = np.load(f"{RES}/main/oof/{tag}.npz", allow_pickle=True)
        eF_tr, eS_tr, eF_te, eS_te, ctrl_tr, _ = evidence_per_fold(oo, pp)
        tF, tS = np.nanpercentile(eF_tr[ctrl_tr], 95), np.percentile(eS_tr[ctrl_tr], 95)
        yy.append(oo["y"][oo["test"]])                       # everyone; no fMRI scan -> not fMRI-atypical
        F.append(np.where(np.isfinite(eF_te), eF_te, -np.inf) > tF)
        S.append(eS_te > tS)
    y, F, S = np.concatenate(yy), np.concatenate(F), np.concatenate(S)
    pats = dict(F_only=F & ~S, S_only=S & ~F, both=F & S, any=F | S, none=~F & ~S)
    for g, gname in ((1, "aut"), (0, "ctrl")):
        for p in pat_names:
            rows.append(dict(rep=rep, group=gname, pattern=p, n=int((y == g).sum()), k=int(pats[p][y == g].sum())))
    for p in pat_names[:4]:
        a, c = pats[p][y == 1], pats[p][y == 0]
        odds, pv = fisher_exact([[a.sum(), (~a).sum()], [c.sum(), (~c).sum()]], alternative="greater")
        rows.append(dict(rep=rep, group="test", pattern=p, n=0, k=0, odds=float(odds), p=float(pv)))
T = pd.DataFrame(rows)
out = {}
for p in pat_names:
    for gname in ("aut", "ctrl"):
        s = T[(T.group == gname) & (T.pattern == p)]
        out[f"{gname}_{p}"] = float((s.k / s.n).mean())
for p in pat_names[:4]:
    s = T[(T.group == "test") & (T.pattern == p)]
    out[f"p_{p}_median"], out[f"p_{p}_max"] = float(s.p.median()), float(s.p.max())
    out[f"odds_{p}_median"] = float(s.odds.median())
n_aut = int(T[(T.group == "aut") & (T.pattern == "any")].n.iloc[0])
n_ctl = int(T[(T.group == "ctrl") & (T.pattern == "any")].n.iloc[0])
out["n_aut"], out["n_ctrl"] = n_aut, n_ctl
print("reproduced vs explain_summary E (mean over repeats):")
for p in ("F_only", "S_only", "both", "none"):
    print(f"  {p:7s} aut {out['aut_' + p]:.4f} (E {E['aut_' + p]:.4f})  ctrl {out['ctrl_' + p]:.4f} (E {E['ctrl_' + p]:.4f})")
print(f"  any     aut {out['aut_any']:.4f}  ctrl {out['ctrl_any']:.4f}   | n per repeat: autistic {n_aut}, controls {n_ctl}")
for p in pat_names[:4]:
    print(f"  Fisher (autistic > controls) {p:7s}: odds ratio median {out['odds_' + p + '_median']:.2f}; p median {out['p_' + p + '_median']:.2e}, "
          f"max over repeats {out['p_' + p + '_max']:.2e}")
os.makedirs(f"{RES}/isbi", exist_ok=True)
json.dump(out, open(f"{RES}/isbi/fig2_patterns.json", "w"), indent=1)
print("written", f"{RES}/isbi/fig2_patterns.json")
