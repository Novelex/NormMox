"""Table 1: mean +- SD over the 10 repeats of 10-fold cross-validation, and the AUROC of NormMoX against every other
row (corrected resampled t-test on the 100 per-fold differences, Holm correction over the comparisons).

usage: python src/13_tables.py [--per-fold results/per_fold_metrics.csv] [--out outputs]
"""
import argparse
import os

import pandas as pd

from stats import METRICS, corrected_t, holm, summarise

ROWS = [("lr_raw", "Logistic regression, raw"),
        ("lr_abide_normed", "Logistic regression, ABIDE-normed"),
        ("lr_deviations", "Logistic regression, chart deviations"),
        ("smri_expert", "sMRI expert (z, |z|)"),
        ("fmri_expert", "fMRI expert (z)"),
        ("fusion_frozen", "Deep fusion, experts frozen"),
        ("fusion_finetuned", "Deep fusion, jointly fine-tuned"),
        ("normmox", "NormMoX")]
REF = "normmox"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-fold", default="results/per_fold_metrics.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    df = pd.read_csv(args.per_fold)
    ref = df[df["model"] == REF].set_index(["repeat", "fold"]).sort_index()
    rows = []
    for key, name in ROWS:
        d = df[df["model"] == key].set_index(["repeat", "fold"]).sort_index()
        if len(d) != 100 or not d.index.equals(ref.index):
            raise ValueError(f"{key}: expected the same 100 test folds as NormMoX")
        mean, sd = summarise(d.reset_index())
        row = dict(model=key, name=name, **{f"{m}_mean": mean[m] for m in METRICS}, **{f"{m}_sd": sd[m] for m in METRICS})
        if key != REF:
            t = corrected_t(ref["auroc"].to_numpy() - d["auroc"].to_numpy())
            row.update(auroc_diff=t["diff"], ci_low=t["ci_low"], ci_high=t["ci_high"], p=t["p"])
        rows.append(row)
    T = pd.DataFrame(rows)
    cmp = T["p"].notna()
    T.loc[cmp, "p_holm"] = holm(T.loc[cmp, "p"].to_numpy())

    w = max(len(n) for _, n in ROWS)
    print("Table 1 (10 x 10 cross-validation, mean +- SD over repeats, threshold 0.5)")
    print(f"{'':{w}s}  {'AUROC':>14s}  " + "  ".join(f"{h:>6s}" for h in ("AUPRC", "BAcc", "Sens", "Spec", "F1"))
          + "   NormMoX - row [95% CI]          p   p (Holm)")
    for _, r in T.iterrows():
        line = f"{r['name']:{w}s}  {r['auroc_mean']:.3f} +- {r['auroc_sd']:.3f}  " + "  ".join(f"{r[m + '_mean']:6.3f}" for m in METRICS[1:])
        if r["model"] != REF:
            line += f"   {r['auroc_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  {r['p']:9.3g}  {r['p_holm']:9.3g}"
        print(line)
    print(f"Largest SD of the other metrics: {T[[m + '_sd' for m in METRICS[1:]]].to_numpy().max():.3f}")

    os.makedirs(args.out, exist_ok=True)
    T.to_csv(os.path.join(args.out, "table1.csv"), index=False)


if __name__ == "__main__":
    main()
