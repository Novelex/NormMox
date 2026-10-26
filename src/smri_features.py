"""Structural features: 185 of the sMRI chart's 187 measures for all 1,035 people, from ABIDE's FreeSurfer 5.1 files
stats/lh.aparc.a2009s.stats, stats/rh.aparc.a2009s.stats and stats/aseg.stats (many people at once; workers = SLURM_CPUS_PER_TASK, default 4).
Writes data/derived/structure_features.csv (FILE_ID + 185 columns in the chart's order).
  - region names: FreeSurfer writes 'G_and_S_frontomargin', the chart uses 'G&S_frontomargin' -> '_and_' replaced by '&'
  - 148 thickness (ThickAvg) + 2 MeanThickness + 32 aseg volumes (Volume_mm3) + 3 global volumes = 185
  - SubCortGrayVol rebuilt with the FreeSurfer >= 5.2 definition: the 16 deep structures only (5.1 also added cerebellum cortex + brain stem)
  - TotalGrayVol rebuilt as lhCortexVol + rhCortexVol + SubCortGrayVol(16) + Left/Right-Cerebellum-Cortex
  - EstimatedTotalIntraCranialVol = the 5.1 line 'IntraCranialVol, ICV' (same number, renamed in 5.2)
  - SupraTentorialVol and SupraTentorialVolNotVent are dropped (no 5.1 equivalent)
  - a region FreeSurfer could not find has no row in the .stats file; it is written as 0 (as FreeSurfer's own table tool does) and
    then set to missing (NaN); volumes are kept as they are
Checks: every row name is a chart region (<= 74 rows, no duplicates); the 5.1 identities SubCortGray(5.1) = 16 structures + cerebellum
cortex + brain stem and TotalGray(5.1) = Cortex + SubCortGray(5.1) hold for everyone; 185 columns; missing only where a region is
absent; no constant column; thickness 0.5-6 mm; volumes >= 0.
"""
import os, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = f"{ROOT}/data/raw/freesurfer"
DOCS = f"{ROOT}/data/raw/braincharts/docs"
OUT = f"{ROOT}/data/derived"
DROP = ["SupraTentorialVol", "SupraTentorialVolNotVent"]
GLOBALS5 = ["SubCortGrayVol", "TotalGrayVol", "SupraTentorialVol", "SupraTentorialVolNotVent", "EstimatedTotalIntraCranialVol"]
DEEP16 = [f"{s}-{r}" for s in ("Left", "Right") for r in ("Thalamus-Proper", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "Accumbens-area", "VentralDC")]
FILES = ["stats/lh.aparc.a2009s.stats", "stats/rh.aparc.a2009s.stats", "stats/aseg.stats"]

LH = open(f"{DOCS}/phenotypes_ct_lh.txt").read().split()      # 74 regions + lh_MeanThickness_thickness
RH = open(f"{DOCS}/phenotypes_ct_rh.txt").read().split()
SC = open(f"{DOCS}/phenotypes_sc.txt").read().split()          # 32 structures + 5 global volumes
SC32 = [s for s in SC if s not in GLOBALS5]
FEATS = LH + RH + [s for s in SC if s not in DROP]               # 185, chart order


def parse_stats(path):
    """A FreeSurfer .stats file -> (measures {short name: value}, table of the rows)."""
    measures, cols, rows = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith("# Measure"):
                parts = [p.strip() for p in line[len("# Measure"):].split(",")]
                measures[parts[1]] = float(parts[-2])             # e.g. '# Measure Cortex, MeanThickness, Mean Thickness, 3.11944, mm'
            elif line.startswith("# ColHeaders"):
                cols = line.split()[2:]
            elif line.startswith("#") or not line.strip():
                continue
            else:
                rows.append(line.split())
    if cols is None or not rows:
        raise ValueError(f"no table in {path}")
    return measures, pd.DataFrame(rows, columns=cols)


def person(fid):
    try:
        base = f"{FS}/{fid}"
        out = {"FILE_ID": fid}
        for hemi, expected in (("lh", LH), ("rh", RH)):
            meas, df = parse_stats(f"{base}/stats/{hemi}.aparc.a2009s.stats")
            cols = [f"{hemi}_{n.replace('_and_', '&')}_thickness" for n in df["StructName"]]
            unknown = [c for c in cols if c not in expected]
            if unknown or len(set(cols)) != len(cols) or len(cols) > 74:
                raise ValueError(f"{hemi}.aparc.a2009s.stats: {len(cols)} rows, unknown names {unknown[:3]}, duplicates {len(cols) - len(set(cols))}")
            vals = dict(zip(cols, df["ThickAvg"].astype(float)))
            for c in expected:
                if c != f"{hemi}_MeanThickness_thickness":
                    out[c] = vals.get(c, 0.0)      # a region FreeSurfer could not find has NO row (FreeSurfer's own table tool writes 0 there) -> becomes missing below
            out[f"raw_{hemi}_regions_absent"] = 74 - len(cols)
            out[f"{hemi}_MeanThickness_thickness"] = meas["MeanThickness"]
        meas, df = parse_stats(f"{base}/stats/aseg.stats")
        vol = dict(zip(df["StructName"], df["Volume_mm3"].astype(float)))
        for name in SC32:
            out[name] = vol[name]                                  # KeyError if a structure is missing
        sub16 = sum(vol[n] for n in DEEP16)
        cereb = vol["Left-Cerebellum-Cortex"] + vol["Right-Cerebellum-Cortex"]
        out["SubCortGrayVol"] = sub16
        out["TotalGrayVol"] = meas["lhCortexVol"] + meas["rhCortexVol"] + sub16 + cereb
        out["EstimatedTotalIntraCranialVol"] = meas["ICV"]
        out["chk_subcort51"] = meas["SubCortGrayVol"] - (sub16 + cereb + vol["Brain-Stem"])      # must be 0 for everyone
        out["chk_totalgray51"] = meas["TotalGrayVol"] - (meas["lhCortexVol"] + meas["rhCortexVol"] + meas["SubCortGrayVol"])
        return out
    except Exception as e:
        return {"FILE_ID": fid, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    ok = True

    def check(cond, msg):
        global ok
        print(f"{'PASS' if cond else 'FAIL'}  {msg}")
        ok = ok and bool(cond)

    print(f"chart lists: {len(LH)} + {len(RH)} + {len(SC)} = {len(LH) + len(RH) + len(SC)} | kept: {len(FEATS)} (dropped {DROP})")
    ids = open(f"{ROOT}/data/raw/file_ids.txt").read().split()
    missing = [f"{i}/{f}" for i in ids for f in FILES if not (os.path.exists(f"{FS}/{i}/{f}") and os.path.getsize(f"{FS}/{i}/{f}") > 0)]
    if missing:
        print(f"FAIL  {len(missing)} input files missing or empty, e.g. {missing[:5]} -> stop"); sys.exit(1)
    check(len(ids) == 1035, f"people: {len(ids)} (expect 1035); all 3 input files present for everyone")

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    with Pool(workers) as pool:
        res = pool.map(person, ids, chunksize=8)
    errors = [r for r in res if "error" in r]
    if errors:
        for r in errors[:20]:
            print("ERROR", r["FILE_ID"], r["error"])
        print(f"FAIL  {len(errors)} people could not be read -> stop"); sys.exit(1)
    df = pd.DataFrame(res).set_index("FILE_ID").loc[ids]
    print(f"workers: {workers} | people read: {len(df)}")

    feats = df[FEATS].copy()
    th_cols = [c for c in FEATS if c.endswith("_thickness")]
    vo_cols = [c for c in FEATS if not c.endswith("_thickness")]
    check(list(feats.columns) == FEATS and feats.shape[1] == 185, f"185 columns named and ordered as the chart lists (got {feats.shape[1]})")
    check(abs(df["chk_subcort51"]).max() < 1e-3, f"5.1 identity SubCortGray = 16 structures + cerebellum cortex + brain stem, max |diff| = {abs(df['chk_subcort51']).max():.2e}")
    check(abs(df["chk_totalgray51"]).max() < 1e-3, f"5.1 identity TotalGray = Cortex + SubCortGray(5.1), max |diff| = {abs(df['chk_totalgray51']).max():.2e}")
    th = feats[th_cols]
    zero = th == 0
    n_zero, n_zero_people = int(zero.sum().sum()), int(zero.any(axis=1).sum())
    print(f"thickness regions absent from the .stats file (FreeSurfer found no vertices): {n_zero} cells in {n_zero_people} people -> set to missing")
    feats[th_cols] = th.mask(zero)
    th = feats[th_cols]
    check(feats[vo_cols].isna().sum().sum() == 0 and th.isna().sum().sum() == n_zero and int(df[["raw_lh_regions_absent", "raw_rh_regions_absent"]].sum().sum()) == n_zero, f"missing values only for absent regions ({int(feats.isna().sum().sum())} cells = {int(df[['raw_lh_regions_absent', 'raw_rh_regions_absent']].sum().sum())} absent rows)")
    const = [c for c in FEATS if feats[c].nunique() <= 1]
    check(not const, f"no constant column ({len(const)}: {const[:5]})")
    check(len(th_cols) == 150 and th.min().min() >= 0.5 and th.max().max() <= 6, f"150 thickness columns, all non-missing values between 0.5 and 6 mm (min {th.min().min():.3f}, max {th.max().max():.3f})")
    vo = feats[vo_cols]
    check(len(vo_cols) == 35 and (vo >= 0).all().all(), f"35 volume columns, all >= 0; {int((vo == 0).sum().sum())} zero cells kept (tiny labels can be absent)")

    os.makedirs(OUT, exist_ok=True)
    feats.to_csv(f"{OUT}/structure_features.csv")
    print(f"written: {OUT}/structure_features.csv ({feats.shape[0]} x {feats.shape[1]})")
    print("CHECK: PASS" if ok else "CHECK: FAIL")
    sys.exit(0 if ok else 1)
