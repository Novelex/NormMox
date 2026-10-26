"""Explanation summary over the 100 test folds (reads outputs/main/{explain, explain_512, oof, params, pred}).
A  group maps: mean contribution per input (IG from the typical person, log-odds units) over the autistic test people and over the
   control test people of each fold; mean +- SD over folds; top inputs of the fMRI networks and of the structure regions (z = direction
   of the deviation, |z| = its size); modality shares
E  evidence profile: per modality the share of test people whose control-calibrated evidence e_m exceeds the 95th percentile of the
   training-fold controls (out-of-fold; never test people): fMRI only / sMRI only / both / none, autistic vs control test people
Checks: 100 folds; completeness error of the 64-step maps and after the 512-step refinement; the evidence recomputed from the saved
out-of-fold logits reproduces the saved MOX test scores.
usage: python src/explain_summary.py -> outputs/{explain_summary.txt, explain_summary.json, maps_features.csv}"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

import fusion as fu
import normmox as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
LINES = []
TOP = 10


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def check(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def fold_id(path):
    b = os.path.basename(path)
    return int(b[1:b.index("_")]), int(b.split("_f")[1].split(".")[0])


def load_explain(mode):
    files = sorted(glob.glob(f"{OUT}/{mode}/explain/r*_f*.npz"))
    out = []
    for f in files:
        e = np.load(f, allow_pickle=True)
        r, k = fold_id(f)
        out.append(dict(rep=r, fold=k, y=e["y"], has=e["has_fmri"], A=np.concatenate([e["attr_F"], e["attr_S_z"], e["attr_S_abs"]], axis=1),
                        f_x=e["f_x"], f0=float(e["f_typical"]), comp=e["completeness_err"], dele=json.loads(str(e["deletion_auroc"])),
                        fcols=list(e["fmri_cols"]), scols=list(e["struct_cols"])))
    return out


def main():
    E = load_explain("main")
    check(len(E) == 100, f"{len(E)} explanation files (need 100)")
    fcols, scols = E[0]["fcols"], E[0]["scols"]
    nF, nS = len(fcols), len(scols)
    names = [f"fMRI {c}" for c in fcols] + [f"z {c}" for c in scols] + [f"|z| {c}" for c in scols]
    report = {}
    comp = np.concatenate([e["comp"] for e in E])
    say(f"CHECKS: 100 folds; completeness error at 64 steps max {comp.max():.4f}, mean {comp.mean():.5f} (log-odds); f(typical person) mean "
        f"{np.mean([e['f0'] for e in E]):+.3f} (SD over folds {np.std([e['f0'] for e in E], ddof=1):.3f})")
    ref = sorted(glob.glob(f"{OUT}/main/explain_512/r*_f*.npz"))
    if ref:
        R = [np.load(f, allow_pickle=True) for f in ref]
        c512 = np.concatenate([r["completeness_err"] for r in R])
        h512 = np.concatenate([r["has_fmri"] for r in R]).astype(bool)
        no_f = float(c512[~h512].max()) if (~h512).any() else 0.0
        say(f"  after the 512-step refinement ({len(ref)} folds): completeness error max {c512[h512].max():.4f} (people with fMRI), "
            f"{no_f:.4f} (people without fMRI: f(typical) includes fMRI evidence, which they lack); mean {c512.mean():.5f}")
        report["completeness_512"] = dict(n_folds=len(ref), max=float(c512.max()), max_with_fmri=float(c512[h512].max()),
                                          max_without_fmri=no_f, mean=float(c512.mean()))
    # ---- A group maps
    GA = np.array([e["A"][e["y"] == 1].mean(0) for e in E])        # folds x inputs, autistic test people
    GC = np.array([e["A"][e["y"] == 0].mean(0) for e in E])
    mA, sA, mC = GA.mean(0), GA.std(0, ddof=1), GC.mean(0)
    sign_cons = (np.sign(GA) == np.sign(mA)).mean(0)
    maps = pd.DataFrame(dict(feature=names, mean_autistic=mA, sd_autistic=sA, mean_control=mC, sign_consistency=sign_cons,
                             mean_abs_autistic=np.array([np.abs(e["A"][e["y"] == 1]).mean(0) for e in E]).mean(0)))
    maps.to_csv(f"{OUT}/maps_features.csv", index=False)
    shares = []
    for e in E:
        a = np.abs(e["A"])
        tot = a.sum(1)
        shares.append([(a[:, :nF].sum(1) / tot).mean(), (a[:, nF:nF + nS].sum(1) / tot).mean(), (a[:, nF + nS:].sum(1) / tot).mean()])
    shares = np.array(shares).mean(0)
    report["A"] = dict(share_fmri=float(shares[0]), share_struct_z=float(shares[1]), share_struct_abs=float(shares[2]))
    say(f"\nA. GROUP MAPS (contribution to the fused log-odds vs the typical person; + = towards autism). Share of |contribution|: fMRI "
        f"{shares[0]:.2f}, structure z {shares[1]:.2f}, structure |z| {shares[2]:.2f}")
    for label, idx in (("fMRI network pairs", slice(0, nF)), ("structure z (direction)", slice(nF, nF + nS)), ("structure |z| (size)", slice(nF + nS, None))):
        sub = maps.iloc[idx].copy()
        sub = sub.reindex(sub["mean_autistic"].abs().sort_values(ascending=False).index)
        say(f"  top {TOP} {label} among autistic people (mean over folds, SD, sign consistency over folds):")
        for _, r in sub.head(TOP).iterrows():
            say(f"    {r['feature']:48s} {r['mean_autistic']:+.4f} +- {r['sd_autistic']:.4f}  sign {r['sign_consistency']:.2f}  controls {r['mean_control']:+.4f}")
        report[f"top_{label}"] = sub.head(TOP)[["feature", "mean_autistic", "sd_autistic", "sign_consistency"]].to_dict("records")
    # ---- E evidence profile (from oof + params)
    prof = []
    for e in E:
        tag = f"r{e['rep']}_f{e['fold']}"
        oo = np.load(f"{OUT}/main/oof/{tag}.npz", allow_pickle=True)
        pp = json.load(open(f"{OUT}/main/params/{tag}.json"))
        m = pp["heads"]["MOX"]
        p, cF, cS = np.array(m["p"]), float(m["cF"]), float(m["cS"])
        sF_, sS_ = fu.softplus(p[4]), fu.softplus(p[5])
        train, test, has, y = oo["train"], oo["test"], oo["has_fmri"], oo["y"]
        k, n_seeds = pp["k"], pp["n_seeds"]
        eF_tr, eS_tr, eF_te, eS_te = [], [], [], []
        mox_full = []
        for s in range(n_seeds):
            c = pp["seed_constants"][s]
            zF = np.where(has, (sf.nan0(oo["lF_oof"][:, s]) - c["muF"]) / c["sdF"], 0.0)
            zS = (sf.nan0(oo["lS_oof"][:, s]) - c["muS"]) / c["sdS"]
            eF_tr.append(np.where(has, sF_ * zF + cF, np.nan)[train])
            eS_tr.append((sS_ * zS + cS)[train])
            for j in range(k):
                LF, LS = oo["lF_test"][:, s, j], oo["lS_test"][:, s, j]
                zFt = np.where(has, (sf.nan0(LF) - c["muF"]) / c["sdF"], 0.0)[test]
                zSt = ((LS - c["muS"]) / c["sdS"])[test]
                ht = has[test]
                eF_te.append(np.where(ht, sF_ * zFt + cF, np.nan))
                eS_te.append(sS_ * zSt + cS)
                mox_full.append(fu.mox_f(p, cF, cS, zFt, zSt, ht))
        with warnings.catch_warnings():                      # people without fMRI have no fMRI evidence (all NaN)
            warnings.simplefilter("ignore", RuntimeWarning)
            eF_tr, eS_tr = np.nanmean(eF_tr, axis=0), np.mean(eS_tr, axis=0)
            eF_te, eS_te = np.nanmean(eF_te, axis=0), np.mean(eS_te, axis=0)
        ctrl = oo["y_fit"][train] == 0
        tF, tS = np.nanpercentile(eF_tr[ctrl], 95), np.percentile(eS_tr[ctrl], 95)
        yt = y[test]
        xF, xS = np.where(np.isfinite(eF_te), eF_te > tF, False), eS_te > tS
        row = dict(rep=e["rep"], fold=e["fold"])
        for lab, msk in (("aut", yt == 1), ("ctrl", yt == 0)):
            row[f"{lab}_F_only"], row[f"{lab}_S_only"] = float((xF & ~xS)[msk].mean()), float((~xF & xS)[msk].mean())
            row[f"{lab}_both"], row[f"{lab}_none"] = float((xF & xS)[msk].mean()), float((~xF & ~xS)[msk].mean())
            row[f"{lab}_F_any"], row[f"{lab}_S_any"] = float(xF[msk].mean()), float(xS[msk].mean())
        prof.append(row)
        pr = pd.read_csv(f"{OUT}/main/pred/{tag}.csv.gz")
        saved = pr[pr["arm"] == "MOX"].set_index("FILE_ID").loc[oo["ids"][test], "score"].to_numpy()
        full = np.mean(mox_full, axis=0)
        check(np.max(np.abs(full - saved)) <= 1e-9, f"{tag}: recomputed MOX differs from the saved scores")
    Pf = pd.DataFrame(prof)
    rp = Pf.groupby("rep").mean()
    say("\nE. EVIDENCE PROFILE: share of test people whose control-calibrated evidence exceeds the 95th percentile of the training-fold "
        "controls (mean +- SD over repeats)")
    say(f"  {'':10s} {'fMRI only':>10s} {'struct only':>12s} {'both':>8s} {'none':>8s} | {'fMRI any':>9s} {'struct any':>11s} {'either':>7s}")
    for lab, nm in (("aut", "autistic"), ("ctrl", "controls")):
        say(f"  {nm:10s} " + " ".join(f"{rp[f'{lab}_{q}'].mean():.3f}+-{rp[f'{lab}_{q}'].std(ddof=1):.3f}".rjust(w_) for q, w_ in
                                      (("F_only", 10), ("S_only", 12), ("both", 8), ("none", 8))) +
            f" | {rp[f'{lab}_F_any'].mean():.3f}".rjust(10) + f" {rp[f'{lab}_S_any'].mean():.3f}".rjust(12)
            + f" {1 - rp[f'{lab}_none'].mean():.3f}".rjust(8))
    say("  (controls: 0.05 expected per modality by construction; 'both' expected 0.0025 if independent)")
    report["E"] = {c: float(rp[c].mean()) for c in rp.columns}
    json.dump(report, open(f"{OUT}/explain_summary.json", "w"), indent=1, default=float)
    open(f"{OUT}/explain_summary.txt", "w").write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
