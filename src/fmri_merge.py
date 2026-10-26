"""Merge the per-person fMRI files into one table (1,035 x 136) and a coverage table.
Reads data/derived/fmri_per_person/<FILE_ID>.npz (from src/fmri_features.py), data/raw/file_ids.txt and the chart's pair list
data/raw/braincharts/docs/phenotypes_yeo17.txt.
Writes data/derived/yeo17_s0_thin_hp128.csv (FILE_ID + 136 columns in the chart's order) and data/derived/coverage.csv.
Naming: label k of the atlas is the k-th network name in the chart's pair list (the 17 names in order of first appearance).
Nobody is excluded; a network outside a person's mask gives NaN cells.
"""
import os, sys, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/data/derived"
PP = f"{OUT}/fmri_per_person"
DOCS = f"{ROOT}/data/raw/braincharts/docs"
VARIANT = "s0_thin_hp128"
EXPECT_N = 1035
NET = 17
ok = True


def check(cond, msg):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {msg}")
    ok = ok and bool(cond)


pairs = open(f"{DOCS}/phenotypes_yeo17.txt").read().split()
H1 = []
for n in pairs:
    for part in n.split("_"):
        if part not in H1:
            H1.append(part)
pos = {n: i for i, n in enumerate(H1)}
expected_pairs = [f"{H1[i]}_{H1[j]}" for i in range(NET) for j in range(i + 1, NET)]
check(len(H1) == NET and len(pairs) == 136 and pairs == expected_pairs, f"chart pair list = upper triangle, row by row, of the 17 names {H1[:3]} ... {H1[-1]} (136 pairs)")


def col_of(name_a, name_b):
    a, b = sorted([name_a, name_b], key=lambda n: pos[n])
    return f"{a}_{b}"


def table(zmat, naming):
    """17x17 label-indexed matrices (n x 17 x 17) -> 136-column table under a naming (list: label k -> name)."""
    cols, vals = [], []
    for i in range(NET):
        for j in range(i + 1, NET):
            cols.append(col_of(naming[i], naming[j])); vals.append(zmat[:, i, j])
    df = pd.DataFrame(np.stack(vals, axis=1), columns=cols)
    return df[pairs]                                                   # chart column order


ids = open(f"{ROOT}/data/raw/file_ids.txt").read().split()
files = {os.path.basename(f)[:-4]: f for f in glob.glob(f"{PP}/*.npz") if not f.endswith(".tmp.npz")}
missing = [i for i in ids if i not in files]
check(len(ids) == EXPECT_N and not missing, f"per-person files: {len(ids) - len(missing)} of {len(ids)} people in file_ids.txt (expected {EXPECT_N}; missing e.g. {missing[:5]})")
if missing:
    print("CHECK: FAIL"); sys.exit(1)

Z = np.full((len(ids), NET, NET), np.nan)
cov = np.zeros((len(ids), NET))
bad = []
for n, i in enumerate(ids):
    try:
        d = np.load(files[i], allow_pickle=True)
        Z[n], cov[n] = d["z"], d["coverage"]
    except Exception as e:
        bad.append(f"{i} ({type(e).__name__})")
check(not bad, f"every per-person file can be read ({len(bad)} unreadable: {bad[:5]}; delete those files and run src/fmri_features.py again, finished people are skipped)")
if bad:
    print("CHECK: FAIL"); sys.exit(1)

covdf = pd.DataFrame({f"thin_{H1[k]}": cov[:, k] for k in range(NET)}, index=pd.Index(ids, name="FILE_ID"))
covdf.to_csv(f"{OUT}/coverage.csv")
print(f"networks with < 70 % of their voxels inside the mask: {int((cov < 0.70).sum())} network-cells in {int((cov < 0.70).any(axis=1).sum())} people; "
      f"empty networks (NaN cells): {int((cov == 0).sum())}")

t = table(Z, H1)
t.index = pd.Index(ids, name="FILE_ID")
fin = t.values[np.isfinite(t.values)]
check(t.shape == (len(ids), 136) and list(t.columns) == pairs and fin.size > 0 and np.abs(fin).max() < 5,
      f"{VARIANT}: {t.shape[0]} x {t.shape[1]}, chart order, NaN {int(t.isna().sum().sum())}, min {fin.min():.3f}, median {np.median(fin):.3f}, max {fin.max():.3f}")
t.to_csv(f"{OUT}/yeo17_{VARIANT}.csv")
print(f"written: {OUT}/yeo17_{VARIANT}.csv ({len(ids)} x 136), {OUT}/coverage.csv")
print("CHECK: PASS" if ok else "CHECK: FAIL")
sys.exit(0 if ok else 1)
