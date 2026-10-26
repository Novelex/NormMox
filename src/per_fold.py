"""Per-fold test metrics of NormMoX, the two experts and the two deep fusion baselines (and the logistic-regression rows when
outputs/per_fold_logistic.csv exists), in the format read by table1.py.
Inputs: outputs/main/pred (normmox.py: MOX, F_TABM, S_TABM), outputs/fusion/main/pred (fusion_frozen.py: MOX_FROZEN) and
outputs/fusion/full/pred (fusion_finetuned.py: FULL); every score is a log-odds, threshold 0.
usage: python src/per_fold.py [--outputs outputs] [--out outputs/per_fold_metrics.csv]"""
import argparse
import glob
import os

import pandas as pd

from stats import METRICS, metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = (("main/pred", {"S_TABM": "smri_expert", "F_TABM": "fmri_expert", "MOX": "normmox"}),
           ("fusion/main/pred", {"MOX_FROZEN": "fusion_frozen"}),
           ("fusion/full/pred", {"FULL": "fusion_finetuned"}))
ORDER = ["lr_raw", "lr_abide_normed", "lr_deviations", "smri_expert", "fmri_expert", "fusion_frozen", "fusion_finetuned", "normmox"]


def fold_id(path):
    b = os.path.basename(path)
    return int(b[1:b.index("_")]), int(b.split("_f")[1].split(".")[0])


def read_preds(folder, arms):
    files = sorted(glob.glob(f"{folder}/r*_f*.csv.gz"))
    if len(files) != 100:
        raise RuntimeError(f"{folder}: {len(files)} fold files (need 100)")
    d = pd.concat([pd.read_csv(f).assign(repeat=fold_id(f)[0], fold=fold_id(f)[1]) for f in files], ignore_index=True)
    return d[d["arm"].isin(arms)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.outputs, "per_fold_metrics.csv")

    rows, people = [], []
    for sub, names in SOURCES:
        d = read_preds(os.path.join(args.outputs, sub), names)
        for (r, k, arm), g in d.groupby(["repeat", "fold", "arm"]):
            rows.append(dict(model=names[arm], repeat=r, fold=k, **metrics(g["y"], g["score"], 0.0)))
            people.append(g[["repeat", "fold", "FILE_ID", "y"]].sort_values("FILE_ID").assign(model=names[arm]))
    M = pd.DataFrame(rows)
    W = pd.concat(people, ignore_index=True)
    base = W[W["model"] == "normmox"].drop(columns="model").sort_values(["repeat", "fold", "FILE_ID"]).reset_index(drop=True)
    for m, g in W.groupby("model"):
        if not g.drop(columns="model").sort_values(["repeat", "fold", "FILE_ID"]).reset_index(drop=True).equals(base):
            raise RuntimeError(f"{m}: test people or labels differ from NormMoX's")
    lg = os.path.join(args.outputs, "per_fold_logistic.csv")
    if os.path.exists(lg):
        M = pd.concat([pd.read_csv(lg), M], ignore_index=True)
    for m, g in M.groupby("model"):
        if len(g) != 100:
            raise RuntimeError(f"{m}: {len(g)} folds (need 100)")
    M["model"] = pd.Categorical(M["model"], [m for m in ORDER if m in set(M["model"])], ordered=True)
    M = M.sort_values(["model", "repeat", "fold"])[["model", "repeat", "fold", *METRICS]]
    M.to_csv(out, index=False)
    print(f"wrote {out}: {len(M)} rows, models {list(M['model'].cat.categories)}")


if __name__ == "__main__":
    main()
