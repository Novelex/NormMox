"""Per-fold maps behind Figures 2 and 3, from the explanation files of the 100 test folds (outputs/main/explain).
GA = mean Integrated-Gradients contribution over the autistic test people, GC = over the control test people, DIFF = GA - GC
(506 inputs: 136 fMRI pairs, 185 structural z, 185 structural |z|). ZA / ZC = mean in-fold chart z of the autistic / control test
people, AA / AC = their mean structural |z|.
Writes outputs/isbi/per_fold.npz.
usage: python src/fig_maps.py"""
import glob
import os

import numpy as np

from data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/outputs/isbi"
EXPL = f"{ROOT}/outputs/main/explain"


def main():
    os.makedirs(OUT, exist_ok=True)
    D = Data()
    files = sorted(glob.glob(f"{EXPL}/r*_f*.npz"))
    if len(files) != 100:
        raise RuntimeError(f"{len(files)} explanation files (need 100)")
    GA, GC, ZA, ZC, AA, AC, tags = [], [], [], [], [], [], []
    for f in files:
        e = np.load(f, allow_pickle=True)
        A = np.concatenate([e["attr_F"], e["attr_S_z"], e["attr_S_abs"]], axis=1)
        y = e["y"].astype(int)
        GA.append(A[y == 1].mean(0)); GC.append(A[y == 0].mean(0))
        b = os.path.basename(f)
        r, k = int(b[1:b.index("_")]), int(b.split("_f")[1].split(".")[0])
        zi = D.fold_z(r, k)
        t = zi["test"]
        ids_ok = np.array_equal(np.asarray(D.ids[t], dtype=str), e["ids"].astype(str))
        if not ids_ok:
            raise RuntimeError(f"{b}: test people differ between the explanation file and the fold")
        Zf, Zs = zi["zf"][t], zi["zs"][t]
        Zall = np.concatenate([Zf, Zs], axis=1)
        ZA.append(np.nanmean(Zall[y == 1], axis=0)); ZC.append(np.nanmean(Zall[y == 0], axis=0))
        AA.append(np.nanmean(np.abs(Zs[y == 1]), axis=0)); AC.append(np.nanmean(np.abs(Zs[y == 0]), axis=0))
        tags.append(b[:-4])
    GA, GC, ZA, ZC, AA, AC = map(np.array, (GA, GC, ZA, ZC, AA, AC))
    DIFF = GA - GC
    np.savez(f"{OUT}/per_fold.npz", GA=GA, GC=GC, DIFF=DIFF, ZA=ZA, ZC=ZC, AA=AA, AC=AC, tags=np.array(tags),
             func_cols=np.array(D.func_cols), struct_cols=np.array(D.struct_cols))
    print(f"written {OUT}/per_fold.npz ({len(tags)} folds)")


if __name__ == "__main__":
    main()
