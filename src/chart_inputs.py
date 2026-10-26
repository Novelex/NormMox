"""Inputs of the charts: join, check, apply the blank-row rule and add the fMRI chart's head-motion column. Nothing is scored here;
nobody is removed.
Reads data/derived/{structure_features.csv, yeo17_s0_thin_hp128.csv, coverage.csv, covariates_base.csv}, data/raw/file_ids.txt and
data/raw/braincharts (feature lists, site lists, unpacked models).
Writes data/derived/{struct_185.csv, func_s0_thin_hp128.csv, covariates.csv, blank_rows.csv}.
Checks: 1,035 rows in the same order everywhere; the charts' feature names and order; a model folder for every feature in the sMRI chart
(lifespan_23K_57sites_mqc2) and the fMRI chart (lifespan_yeo17_22K_40sites); no ABIDE site among the charts' training sites (every site
is new to the charts and is calibrated with its own controls); sex 0/1; age inside the charts' B-spline range (-5..110); mean FD > 0;
every site has >= 2 controls.
Blank-row rule: a scan with more than half of its 17 networks empty (coverage 0) gets a fully blank fMRI row and func_empty_thin = 1.
Blank cells are never filled.
Head motion: the fMRI chart uses fMRIPrep (Power) FD; mean_FD_power = 1.76 x ABIDE (Jenkinson) mean FD.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BC = f"{ROOT}/data/raw/braincharts"
INP = OUT = f"{ROOT}/data/derived"
STRUCT_MAIN, FUNC_CHART = "lifespan_23K_57sites_mqc2", "lifespan_yeo17_22K_40sites"
VARIANT = "s0_thin_hp128"
FD_SCALE = 1.76                                               # fMRIPrep (Power) FD = 1.76 x ABIDE (Jenkinson) FD
ok = True


def check(cond, msg):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {msg}")
    ok = ok and bool(cond)


def read_list(p):
    return open(p).read().split()


ids = read_list(f"{ROOT}/data/raw/file_ids.txt")
cv = pd.read_csv(f"{INP}/covariates_base.csv").set_index("FILE_ID")
st = pd.read_csv(f"{INP}/structure_features.csv").set_index("FILE_ID")
cov = pd.read_csv(f"{INP}/coverage.csv").set_index("FILE_ID")
fn = pd.read_csv(f"{INP}/yeo17_{VARIANT}.csv").set_index("FILE_ID")
same_order = all(list(d.index) == ids for d in [cv, st, cov, fn])
check(len(ids) == 1035 and same_order, f"1,035 people, same order in covariates, structure, coverage and the fMRI table ({len(ids)} ids)")

# --- feature names vs the charts
docs = f"{BC}/docs"
ct_list = read_list(f"{docs}/phenotypes_ct_lh.txt") + read_list(f"{docs}/phenotypes_ct_rh.txt") + read_list(f"{docs}/phenotypes_sc.txt")
yeo_list = read_list(f"{docs}/phenotypes_yeo17.txt")
check(list(st.columns) == [f for f in ct_list if f not in ("SupraTentorialVol", "SupraTentorialVolNotVent")],
      f"structure columns = the chart's 187 names minus the 2 dropped: {st.shape[1]} columns")
check(list(fn.columns) == yeo_list, "the fMRI table has the chart's 136 columns in the chart's order")


def folders_present(chart, feats):
    mdir = f"{BC}/models/{chart}"
    if not os.path.isdir(mdir):
        return None
    return [f for f in feats if not os.path.exists(f"{mdir}/{f}/Models/NM_0_0_estimate.pkl")]


miss = folders_present(STRUCT_MAIN, list(st.columns))
check(miss == [], f"structure chart {STRUCT_MAIN}: a model folder exists for every structure column (missing {None if miss is None else len(miss)}: {(miss or [])[:3]})")
missf = folders_present(FUNC_CHART, yeo_list)
check(missf == [], f"function chart {FUNC_CHART}: a model folder exists for every fMRI column (missing {None if missf is None else len(missf)}: {(missf or [])[:3]})")

# --- sites vs the charts' training sites
our_sites = sorted(cv["site"].unique())
for name, lst in (("structure chart", "site_ids_ct_57sites.txt"), ("function chart", "site_ids_yeo17_40sites.txt")):
    inter = sorted(set(our_sites) & set(read_list(f"{docs}/{lst}")))
    check(not inter, f"no ABIDE site inside the {name}'s training list (all 20 sites are new): overlap {inter}")

# --- covariates for the charts
check(set(cv["sex"].unique()) <= {0, 1}, f"sex coded 0 = female, 1 = male (chart coding): counts {cv['sex'].value_counts().to_dict()}")
check(cv["age"].between(-5, 110).all(), f"age inside the charts' B-spline range -5..110: {cv['age'].min():.1f}-{cv['age'].max():.1f} years")
check((cv["mean_FD"] > 0).all() and cv["mean_FD"].notna().all(), f"mean_FD present and > 0 for everyone (imputed: {int(cv['mean_FD_imputed'].sum())})")
ctrl = cv["autism"] == 0
per_site = cv[ctrl].groupby("site").size().reindex(our_sites).fillna(0).astype(int)
check((per_site >= 2).all(), f"controls per site: min {per_site.min()} (the toolkit needs >= 2), max {per_site.max()}, total {int(ctrl.sum())}")
print(f"autistic per site: {cv[~ctrl].groupby('site').size().reindex(our_sites).fillna(0).astype(int).to_dict()}")

# --- blank-row rule, from the coverage table
cols = [c for c in cov.columns if c.startswith("thin_")]
empty = (cov[cols] == 0).sum(axis=1)                            # empty networks per person
flag = empty > 17 / 2                                            # more than half of the 17 networks
cv["func_empty_thin"] = flag.astype(int).values
blank_rows = list(cov.index[flag])
print(f"people with >= 1 empty network: {int((empty > 0).sum())}; with more than half empty -> whole fMRI row blank: {int(flag.sum())} {blank_rows}")
check(len(cols) == 17 and flag.sum() <= 5, "17 networks in coverage.csv and at most a handful of blank rows")
before = int(fn.isna().sum().sum())
fn.loc[blank_rows, :] = np.nan
after = int(fn.isna().sum().sum())
print(f"  {VARIANT}: blank cells before {before}, after the rule {after}")
check(after >= before and fn.shape == (1035, 136), f"  {VARIANT}: table 1,035 x 136, blanks only added, never filled")

# --- head motion column for the function chart
cv["mean_FD_power"] = FD_SCALE * cv["mean_FD"]
print(f"mean_FD_power = {FD_SCALE} x mean_FD: median {cv['mean_FD_power'].median():.3f} mm (ABIDE median {cv['mean_FD'].median():.3f} mm)")

# --- write
os.makedirs(OUT, exist_ok=True)
st.to_csv(f"{OUT}/struct_185.csv")
fn.to_csv(f"{OUT}/func_{VARIANT}.csv")
cv.to_csv(f"{OUT}/covariates.csv")
pd.DataFrame({"FILE_ID": blank_rows, "thin": 1}).to_csv(f"{OUT}/blank_rows.csv", index=False)
print(f"written: {OUT}/struct_185.csv (1035 x 185), func_{VARIANT}.csv (1035 x 136), covariates.csv ({cv.shape[1]} columns), blank_rows.csv")
print("CHECK: PASS" if ok else "CHECK: FAIL")
sys.exit(0 if ok else 1)
