"""Completeness refinement: in every fold, the test people whose Integrated-Gradients completeness error (|sum of contributions -
(f(person) - f(typical))|) exceeds TOL at 64 steps are recomputed with STEPS = 512 midpoint steps (same models, baseline, path).
The 64-step files stay untouched; refined copies go to outputs/main/explain_512/ (all arrays copied, refined rows replaced,
refined_rows and steps_refined added). usage: python src/explain_refine.py TASK NTASKS"""
import os
import sys
import time

import numpy as np
import torch

import explain as se
import normmox as sf
from data import Data

STEPS, TOL = 512, 0.01
SRC, DST = f"{sf.OUT_BASE}/main/explain", f"{sf.OUT_BASE}/main/explain_512"


def refine(D, rep, fold, cfg):
    tag = f"r{rep}_f{fold}"
    dst = f"{DST}/{tag}.npz"
    if os.path.exists(dst):
        return None
    t0 = time.time()
    e = np.load(f"{SRC}/{tag}.npz", allow_pickle=True)
    data = {k: np.array(e[k], copy=True) for k in e.files}
    err = data["completeness_err"]
    idx = np.where(err > TOL)[0]
    new = err.copy()
    if len(idx):
        zi = D.fold_z(rep, fold)
        train, test = zi["train"], zi["test"]
        XF, x0F, XS, x0S = se.baseline_tokens(zi, train)
        models, _ = se.load_models(tag, f"{sf.OUT_BASE}/main", cfg, XF.shape[1], XS.shape[1])
        f32 = torch.float32
        has = (~D.blank_fmri)[test][idx]
        aF, aS = se.integrated_gradients(models, torch.as_tensor(XF[test][idx], dtype=f32), torch.as_tensor(XS[test][idx], dtype=f32),
                                         torch.as_tensor(has), torch.as_tensor(x0F, dtype=f32), torch.as_tensor(x0S, dtype=f32), steps=STEPS)
        nz = data["attr_S_z"].shape[1]
        new[idx] = np.abs(aF.sum(1) + aS.sum(1) - (data["f_x"][idx] - float(data["f_typical"])))
        data["attr_F"][idx], data["attr_S_z"][idx], data["attr_S_abs"][idx] = aF, aS[:, :nz], aS[:, nz:]
    data["completeness_err"] = new
    data["refined_rows"] = idx
    data["steps_refined"] = np.array(STEPS)
    tmp = f"{dst}.tmp{os.getpid()}.npz"
    np.savez(tmp, **data)
    os.replace(tmp, dst)
    print(f"{tag}: {len(idx)} people refined; max error {err.max():.4f} -> {new.max():.4f}; {time.time() - t0:.0f} s", flush=True)
    return len(idx), float(err.max()), float(new.max())


def main():
    task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    D = Data()
    cfg = sf.context("main")["cfg"]
    os.makedirs(DST, exist_ok=True)
    torch.set_num_threads(int(cfg["threads"]))
    for rep, fold in D.fold_list()[task::ntasks]:
        refine(D, rep, fold, cfg)
    print("explain_refine: task done", flush=True)


if __name__ == "__main__":
    main()
