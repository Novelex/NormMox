"""Chart residuals: for every person and measure, r = warped measured value - the chart's prediction, before any site calibration.
The site calibration (Eq. 1: mean and SD of r over a site's training controls) is redone inside every classification fold
(src/rulers.py), so no test person ever contributes to it. Nobody is removed; blank cells stay blank.
Reads data/derived/{struct_185.csv, func_s0_thin_hp128.csv, covariates.csv} and data/raw/braincharts/{docs, models}.
Writes data/derived/resid_struct.csv (185 measures; chart lifespan_23K_57sites_mqc2; covariates age, sex) and
data/derived/resid_func.csv (136 pairs; chart lifespan_yeo17_22K_40sites; covariates age, sex, mean_FD_power).
"""
import os, sys, time
import numpy as np
import pandas as pd

from charts import design_matrix, load_chart, warp_values

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BC = f"{ROOT}/data/raw/braincharts"
INP = OUT = f"{ROOT}/data/derived"
STRUCT = ("resid_struct", "lifespan_23K_57sites_mqc2", "site_ids_ct_57sites.txt")
FUNC = ("resid_func", "lifespan_yeo17_22K_40sites", "site_ids_yeo17_40sites.txt")
VARIANT = "s0_thin_hp128"
ok = True


def check(cond, msg):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {msg}")
    ok = ok and bool(cond)


def read_list(p):
    return open(p).read().split()


cv = pd.read_csv(f"{INP}/covariates.csv").set_index("FILE_ID")
ids = list(cv.index)
check(len(ids) == 1035, f"covariates: {len(ids)} people")
is_ctrl = cv["autism"].to_numpy() == 0
docs = f"{BC}/docs"


def run_table(name, mdir, sites_tr, feats, values, cov_cols, frame):
    t0 = time.time()
    X = design_matrix(frame, cov_cols, sites_tr)
    r = np.full((len(ids), len(feats)), np.nan)
    for j, f in enumerate(feats):
        blr, wp = load_chart(mdir, f)
        pred = X @ blr.m
        yw = warp_values(blr, wp, values[f].to_numpy(float))
        r[:, j] = yw - pred                                      # blank stays blank
    rt = pd.DataFrame(r, index=pd.Index(ids, name="FILE_ID"), columns=feats)
    rt.to_csv(f"{OUT}/{name}.csv")
    n_blank = int(np.isnan(r).sum())
    check(n_blank == int(values[feats].isna().sum().sum()), f"{name}: {len(feats)} features, {n_blank} blanks = the input blanks")
    rc = r[is_ctrl]
    print(f"   {name}: controls' r mean {np.nanmean(rc):+.3f}, SD {np.nanstd(rc):.3f} (warped units); {time.time() - t0:.0f} s")


# structure: 185 measures (150 thickness + 35 volumes)
st = pd.read_csv(f"{INP}/struct_185.csv").set_index("FILE_ID").loc[ids]
name, chart, sites_file = STRUCT
frame = cv[["age", "sex", "site"]].copy()
run_table(name, f"{BC}/models/{chart}", read_list(f"{docs}/{sites_file}"), list(st.columns), st, ["age", "sex"], frame)
# function: 136 network pairs, the chart's FD scale
name, chart, sites_file = FUNC
frame = cv[["age", "sex", "site"]].copy()
frame["mean_FD"] = cv["mean_FD_power"]
fn = pd.read_csv(f"{INP}/func_{VARIANT}.csv").set_index("FILE_ID").loc[ids]
run_table(name, f"{BC}/models/{chart}", read_list(f"{docs}/{sites_file}"), list(fn.columns), fn, ["age", "sex", "mean_FD"], frame)

print("CHECK: PASS" if ok else "CHECK: FAIL")
sys.exit(0 if ok else 1)
