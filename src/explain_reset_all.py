"""Group reset for Table 2, including all inputs reset: test AUROC of the fused model when ALL inputs of one modality (fMRI: the 136
network-pair tokens; sMRI: the 370 structure tokens), or all 506 inputs, are reset to the typical value, every other input unchanged.
Same models, tokens and baseline as explain.py; check in every fold: no reset reproduces the saved full-model AUROC.
Output: outputs/main/explain_reset_all/r{r}_f{f}.json. usage: python src/explain_reset_all.py TASK NTASKS"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import explain as se
import normmox as sf
from data import Data


def run_fold(D, out, rep, fold, cfg):
    tag = f"r{rep}_f{fold}"
    path = f"{out}/explain_reset_all/{tag}.json"
    if os.path.exists(path):
        return
    t0 = time.time()
    zi = D.fold_z(rep, fold)
    train, test = zi["train"], zi["test"]
    XF, x0F, XS, x0S = se.baseline_tokens(zi, train)
    has = ~D.blank_fmri
    models, _ = se.load_models(tag, out, cfg, XF.shape[1], XS.shape[1])
    f32 = torch.float32
    xF, xS = torch.as_tensor(XF[test], dtype=f32), torch.as_tensor(XS[test], dtype=f32)
    hf = torch.as_tensor(has[test])
    x0F_t, x0S_t = torch.as_tensor(x0F, dtype=f32), torch.as_tensor(x0S, dtype=f32)
    y = D.y[test].astype(int)
    e = np.load(f"{out}/explain/{tag}.npz", allow_pickle=True)
    saved = json.loads(str(e["deletion_auroc"]))
    res = {"none": float(roc_auc_score(y, se.fused_scores(models, xF, xS, hf)))}
    if abs(res["none"] - saved["none"]) > 1e-6:
        raise RuntimeError(f"{tag}: full AUROC {res['none']:.6f} does not reproduce the saved {saved['none']:.6f}")
    res["fmri_off"] = float(roc_auc_score(y, se.fused_scores(models, x0F_t.expand_as(xF).clone(), xS, hf)))
    res["smri_off"] = float(roc_auc_score(y, se.fused_scores(models, xF, x0S_t.expand_as(xS).clone(), hf)))
    res["all_off"] = float(roc_auc_score(y, se.fused_scores(models, x0F_t.expand_as(xF).clone(), x0S_t.expand_as(xS).clone(), hf)))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, path)
    print(f"{tag}: check ok; full {res['none']:.3f} fMRI off {res['fmri_off']:.3f} sMRI off {res['smri_off']:.3f}; {time.time() - t0:.0f} s",
          flush=True)


def main():
    task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    D = Data()
    ctx = sf.context("main")
    os.makedirs(f"{ctx['out']}/explain_reset_all", exist_ok=True)
    torch.set_num_threads(int(ctx["cfg"]["threads"]))
    for rep, fold in D.fold_list()[task::ntasks]:
        run_fold(D, ctx["out"], rep, fold, ctx["cfg"])
    print("explain_reset_all: task done", flush=True)


if __name__ == "__main__":
    main()
