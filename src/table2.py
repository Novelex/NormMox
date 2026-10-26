"""Table 2 (validation of the explanations) by input group: fMRI (136 network pairs), sMRI (370 structure tokens: 185 z + 185 |z|)
and all 506 inputs, with the definitions of explain_summary.py:
  share       mean over folds of each person's share of |contribution| in the group
  reset       test AUROC when every input of the group is reset to typical (explain_reset_all.py); mean of the repeat means; reset - full
              over the 100 folds with the corrected resampled t (ratio 1/9)
  deletion    test AUROC after resetting each person's top-k inputs of the group vs k random inputs of the group (k = 10, 30); mean of the
              repeat means; top - random over the 100 folds with the corrected resampled t; all-input values from explain.py
  stability   Spearman between the autistic group maps of every pair of folds (group inputs only); overlap of each fold's top-10 with the
              global top-10; sign consistency over folds of the group's global top-10
  shuffled    mean |contribution| of the real folds / of each of the 2 shuffled-label folds (group inputs only)
  Holm        over the 6 tests: reset of fMRI / sMRI / all, top-30 vs random-30 of fMRI / sMRI / all
Also printed from outputs/explain_summary.json: test controls flagged (calibration) and the largest completeness error.
Reads outputs/main/{explain, explain_top30, explain_reset_all}, outputs/perm/explain. Writes outputs/isbi/table2.json and prints the table.
usage: python src/table2.py"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import explain_summary as es
from stats import corrected_t, holm

OUT = es.OUT
TOP = 10


def paired(diff):
    """corrected resampled t-test on per-fold differences, with the number of folds where the difference is positive"""
    diff = np.asarray(diff, float)
    return dict(corrected_t(diff), wins=int((diff > 0).sum()), J=int(len(diff)))


def main():
    E = es.load_explain("main")
    P = es.load_explain("perm")
    assert len(E) == 100 and len(P) == 2, (len(E), len(P))
    nF, nS = len(E[0]["fcols"]), len(E[0]["scols"])
    groups = {"fMRI": np.arange(0, nF), "sMRI": np.arange(nF, nF + 2 * nS), "all": np.arange(0, nF + 2 * nS)}
    GA = np.array([e["A"][e["y"] == 1].mean(0) for e in E])
    tags = [f"r{e['rep']}_f{e['fold']}" for e in E]
    mods = pd.DataFrame([dict(rep=int(t.split("_")[0][1:]), fold=int(t.split("_")[1][1:]),
                              **json.load(open(f"{OUT}/main/explain_top30/{t}.json"))) for t in tags])
    resets = pd.DataFrame([dict(rep=int(t.split("_")[0][1:]), fold=int(t.split("_")[1][1:]),
                                **json.load(open(f"{OUT}/main/explain_reset_all/{t}.json"))) for t in tags])
    dele_all = pd.DataFrame([dict(rep=e["rep"], fold=e["fold"], **e["dele"]) for e in E])
    out = {}
    for g, cols in groups.items():
        r = {}
        r["n_inputs"] = int(len(cols))
        r["share"] = float(np.mean([(np.abs(e["A"][:, cols]).sum(1) / np.abs(e["A"]).sum(1)).mean() for e in E]))
        ga = GA[:, cols]
        rho_all, rho_within = [], []
        for i in range(len(E)):
            for j in range(i + 1, len(E)):
                rr = spearmanr(ga[i], ga[j]).statistic
                rho_all.append(rr)
                if E[i]["rep"] == E[j]["rep"]:
                    rho_within.append(rr)
        m = ga.mean(0)
        top = np.argsort(-np.abs(m))[:TOP]
        sign_cons = (np.sign(ga) == np.sign(m)).mean(0)
        r["rho_folds"], r["rho_within_repeat"] = float(np.mean(rho_all)), float(np.mean(rho_within))
        r["top10_overlap"] = float(np.mean([len(set(top) & set(np.argsort(-np.abs(x))[:TOP])) / TOP for x in ga]))
        r["top10_same_sign"] = float(sign_cons[top].mean())
        real = float(np.mean([np.abs(e["A"][:, cols]).mean() for e in E]))
        perm = [float(np.abs(p["A"][:, cols]).mean()) for p in P]
        r["shuffled_times_smaller"] = [real / x for x in perm]
        r["shuffled_rho_with_real"] = [float(spearmanr(p["A"][p["y"] == 1].mean(0)[cols], m).statistic) for p in P]
        for k in (10, 30):
            if g == "all":
                top_k, rnd_k = dele_all[f"top{k}"], dele_all[f"rand{k}_auroc"]
                reps = dele_all.groupby("rep").mean()
                r[f"top{k}"], r[f"rand{k}"] = float(reps[f"top{k}"].mean()), float(reps[f"rand{k}_auroc"].mean())
            else:
                key = "F" if g == "fMRI" else "S"
                top_k, rnd_k = mods[f"{key}_top{k}"], mods[f"{key}_rand{k}"]
                reps = mods.groupby("rep").mean()
                r[f"top{k}"], r[f"rand{k}"] = float(reps[f"{key}_top{k}"].mean()), float(reps[f"{key}_rand{k}"].mean())
            c = paired((top_k - rnd_k).to_numpy())
            r[f"delta{k}"] = {kk: (float(v) if not isinstance(v, (int, np.integer)) else int(v)) for kk, v in c.items()}
        off = {"fMRI": "fmri_off", "sMRI": "smri_off", "all": "all_off"}[g]
        rr_ = resets.groupby("rep").mean()
        r["reset"] = dict(auroc=float(rr_[off].mean()), sd=float(rr_[off].std(ddof=1)), **paired((resets[off] - resets["none"]).to_numpy()))
        out[g] = r
    full = float(dele_all.groupby("rep").mean()["none"].mean())
    out["full_auroc"] = full
    out["check_all_reproduced"] = bool(np.allclose(mods["all_top10"], dele_all["top10"], atol=1e-6)
                                       and np.allclose(mods["all_top30"], dele_all["top30"], atol=1e-6))
    out["check_reset_full_reproduced"] = bool(np.allclose(resets["none"], dele_all["none"], atol=1e-6))
    tests = [(f"reset {g}", out[g]["reset"]["p"]) for g in groups] + [(f"top-30 vs random {g}", out[g]["delta30"]["p"]) for g in groups]
    adj = holm([p for _, p in tests])
    out["holm"] = {name: dict(p=float(p), p_holm=float(a)) for (name, p), a in zip(tests, adj)}
    os.makedirs(f"{OUT}/isbi", exist_ok=True)
    json.dump(out, open(f"{OUT}/isbi/table2.json", "w"), indent=1)
    print(f"full model AUROC {full:.4f}; all-input deletion reproduced in every fold: {out['check_all_reproduced']}; "
          f"reset files reproduce the full AUROC: {out['check_reset_full_reproduced']}")
    print(f"{'inputs':6s} {'n':>4s} {'share':>6s} | {'reset':>6s} {'p':>8s} | {'top10':>6s} {'rand10':>6s} {'delta10 [95% CI]':>28s} {'p':>8s} | "
          f"{'top30':>6s} {'rand30':>6s} {'delta30 [95% CI]':>28s} {'p':>8s} | {'rho':>5s} {'rhoW':>5s} {'ovl':>5s} {'sign':>5s} | "
          f"shuffled x smaller; rho")
    for g in ("fMRI", "sMRI", "all"):
        r = out[g]
        d10, d30 = r["delta10"], r["delta30"]
        print(f"{g:6s} {r['n_inputs']:4d} {r['share']:6.3f} | {r['reset']['auroc']:6.3f} {r['reset']['p']:8.2g} | {r['top10']:6.3f} {r['rand10']:6.3f} "
              f"{d10['diff']:+.4f} [{d10['ci_low']:+.4f}, {d10['ci_high']:+.4f}] {d10['p']:8.2g} | {r['top30']:6.3f} {r['rand30']:6.3f} "
              f"{d30['diff']:+.4f} [{d30['ci_low']:+.4f}, {d30['ci_high']:+.4f}] {d30['p']:8.2g} | {r['rho_folds']:5.2f} "
              f"{r['rho_within_repeat']:5.2f} {r['top10_overlap']:5.2f} {r['top10_same_sign']:5.2f} | "
              f"{min(r['shuffled_times_smaller']):.1f}-{max(r['shuffled_times_smaller']):.1f}; "
              f"{', '.join(f'{x:+.3f}' for x in r['shuffled_rho_with_real'])}")
    print("Holm over the 6 tests:")
    for name, v in out["holm"].items():
        print(f"  {name:24s} p {v['p']:.3g} -> Holm {v['p_holm']:.3g}")
    summ = f"{OUT}/explain_summary.json"
    if os.path.exists(summ):
        S = json.load(open(summ))
        ev = S["E"]
        print(f"test controls flagged: fMRI {ev['ctrl_F_any']:.3f}, sMRI {ev['ctrl_S_any']:.3f}, either {1 - ev['ctrl_none']:.3f}")
        if "completeness_512" in S:
            c = S["completeness_512"]
            print(f"largest completeness error after the 512-step refinement: {c['max_with_fmri']:.4f} log-odds (people with fMRI), "
                  f"{c['max_without_fmri']:.4f} (people without fMRI)")


if __name__ == "__main__":
    main()
