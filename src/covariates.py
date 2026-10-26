"""One covariate table for all 1,035 people (nobody is excluded).
Reads data/raw/Phenotypic_V1_0b_preprocessed1.csv and data/raw/file_ids.txt.
Writes data/derived/covariates_base.csv with the charts' column names:
  sub_id, age (AGE_AT_SCAN), sex (ABIDE SEX 1 = male -> 1, 2 = female -> 0; chart coding 0 = female, 1 = male), site ('ABIDE_' + SITE_ID),
  sitenum, mean_FD (ABIDE func_mean_fd, Jenkinson), mean_FD_imputed (1 if the site median was filled in), perc_FD, num_FD,
  group (DX_GROUP: 1 autism, 2 control), autism (1/0).
One person has no mean FD in the phenotypic file (Yale_0050610); it is filled with the site median.
Checks: 1,035 rows; no missing values after the one fill; sex and group coding; age and mean FD ranges.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/data/derived"
ok = True


def check(cond, msg):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {msg}")
    ok = ok and bool(cond)


ids = open(f"{ROOT}/data/raw/file_ids.txt").read().split()
ph = pd.read_csv(f"{ROOT}/data/raw/Phenotypic_V1_0b_preprocessed1.csv")
ph = ph[ph["FILE_ID"] != "no_filename"].set_index("FILE_ID")
check(len(ids) == 1035 and all(i in ph.index for i in ids), f"{len(ids)} people, all in the phenotype file")
ph = ph.loc[ids]

cov = pd.DataFrame(index=pd.Index(ids, name="FILE_ID"))
cov["sub_id"] = ids
cov["age"] = ph["AGE_AT_SCAN"].astype(float)
cov["sex"] = ph["SEX"].map({1: 1, 2: 0})
cov["site"] = "ABIDE_" + ph["SITE_ID"].astype(str)
sites = sorted(cov["site"].unique())
cov["sitenum"] = cov["site"].map({s: i for i, s in enumerate(sites)})
cov["mean_FD"] = ph["func_mean_fd"].astype(float)
cov["mean_FD_imputed"] = cov["mean_FD"].isna().astype(int)
site_median = cov.groupby("site")["mean_FD"].transform("median")
cov["mean_FD"] = cov["mean_FD"].fillna(site_median)
cov["perc_FD"] = ph["func_perc_fd"].astype(float)
cov["num_FD"] = ph["func_num_fd"].astype(float)
cov["group"] = ph["DX_GROUP"].astype(int)
cov["autism"] = (cov["group"] == 1).astype(int)

# --- checks
imputed = cov.index[cov["mean_FD_imputed"] == 1].tolist()
print(f"mean_FD filled with the site median for: {imputed} (expected: ['Yale_0050610'])")
nan_cols = cov.columns[cov.isna().any()].tolist()
other_nan = cov.isna()
check(other_nan.drop(index=imputed).sum().sum() == 0 and set(other_nan.columns[other_nan.any()]) <= {"perc_FD", "num_FD"},
      f"no missing values, except the motion fields of the imputed person: columns with NaN {nan_cols}")
check(set(cov["sex"].unique()) <= {0, 1} and set(cov["group"].unique()) == {1, 2}, f"sex coded 0/1, group 1/2: {cov['sex'].value_counts().to_dict()} | {cov['group'].value_counts().to_dict()}")
print(f"sex (1 = male) x group (1 = autism):\n{pd.crosstab(cov['sex'], cov['group']).to_string()}")
check(cov["age"].between(5, 65).all(), f"age between 5 and 65: min {cov['age'].min():.1f}, max {cov['age'].max():.1f}")
check(cov["mean_FD"].between(0, 5).all(), f"mean_FD between 0 and 5 mm: min {cov['mean_FD'].min():.3f}, median {cov['mean_FD'].median():.3f}, max {cov['mean_FD'].max():.3f}")
print(f"people per site x group (1 = autism, 2 = control):\n{pd.crosstab(cov['site'], cov['group']).to_string()}")

os.makedirs(OUT, exist_ok=True)
cov.to_csv(f"{OUT}/covariates_base.csv")
print(f"written: {OUT}/covariates_base.csv ({cov.shape[0]} x {cov.shape[1]})")
print("CHECK: PASS" if ok else "CHECK: FAIL")
sys.exit(0 if ok else 1)
