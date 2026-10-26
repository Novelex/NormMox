"""Leave-one-site-out: the NormMoX fold code (normmox.run_fold, unchanged) on the 20 'loso' folds (train on 19 site samples, test on
the 20th). Site calibration follows the leave-one-site-out rule: the held-out site's own controls calibrate its site offsets and SDs
(cross-fitted within the site). Everything else (inner parts, experts, heads, thresholds) is the main-scheme code.
Outputs: outputs/loso/{pred, oof, models, params}.
usage: python src/loso.py TASK NTASKS"""
import os
import sys

import pandas as pd

import normmox as sf
from data import S, Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEME = "loso"


class LosoData(Data):
    """Data with the leave-one-site-out folds and site-calibration rule"""
    def __init__(self):
        super().__init__()
        fo = pd.read_csv(os.path.join(ROOT, S["paths"]["folds"]))
        self.folds = fo[fo["scheme"] == SCHEME]
        self.scheme = SCHEME
        if len(self.fold_list()) != 20:
            raise RuntimeError(f"{len(self.fold_list())} LOSO folds, expected 20")


def main():
    task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    D = LosoData()
    ctx = sf.context("main")
    ctx["out"] = f"{sf.OUT_BASE}/loso"
    for d in ("pred", "oof", "models", "params"):
        os.makedirs(f"{ctx['out']}/{d}", exist_ok=True)
    mine = D.fold_list()[task::ntasks]
    print(f"task {task}/{ntasks} LOSO: {len(mine)} folds {mine}; K {ctx['k']}, seeds {ctx['n_seeds']}", flush=True)
    first = True
    for rep, fold in mine:
        if sf.run_fold(D, ctx, rep, fold, "main", first):
            first = False
    print("LOSO TASK DONE", flush=True)


if __name__ == "__main__":
    main()
