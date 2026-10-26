"""Per-modality faithfulness (deletion) for Table 2. As explain.deletion, but each test person's top-k inputs are ranked and reset to
typical WITHIN one modality (fMRI: the 136 network pairs; sMRI: the 370 structure tokens = 185 z + 185 |z|), against k random inputs
of the same modality (RANDOM_DRAWS draws, own random stream); k = 10, 30. Uses the saved 64-step attributions (outputs/main/explain/)
and the saved models. Check in every fold: the saved all-input AUROCs (no reset, top-10, top-30) are reproduced before any new number
is computed. Output: outputs/main/explain_top30/r{r}_f{f}.json. usage: python src/explain_top30.py TASK NTASKS"""
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

STREAM_MOD = 11                                              # own random stream (explain.py uses 5)
TOL = 1e-6                                                   # AUROC reproduction tolerance


def run_fold(D, out, rep, fold, cfg):
    tag = f"r{rep}_f{fold}"
    path = f"{out}/explain_top30/{tag}.json"
    if os.path.exists(path):
        print(f"{tag}: done, skipped", flush=True)
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
    e = np.load(f"{out}/explain/{tag}.npz", allow_pickle=True)
    if list(e["ids"]) != [str(i) for i in D.ids[test]]:
        raise RuntimeError(f"{tag}: test people differ from the saved explanation file")
    attr = np.concatenate([e["attr_F"], e["attr_S_z"], e["attr_S_abs"]], axis=1)
    y = D.y[test].astype(int)
    saved = json.loads(str(e["deletion_auroc"]))
    nF, n_all = xF.shape[1], attr.shape[1]

    def auroc_reset(reset):
        xf, xs = xF.clone(), xS.clone()
        rF, rS = torch.as_tensor(reset[:, :nF]), torch.as_tensor(reset[:, nF:])
        xf[rF] = x0F_t.expand_as(xf)[rF]
        xs[rS] = x0S_t.expand_as(xs)[rS]
        return float(roc_auc_score(y, se.fused_scores(models, xf, xs, hf)))

    res = {"none": auroc_reset(np.zeros(attr.shape, bool))}
    order = np.argsort(-np.abs(attr), axis=1)
    for k in (10, 30):                                        # check: the saved all-input deletion is reproduced
        reset = np.zeros(attr.shape, bool)
        np.put_along_axis(reset, order[:, :k], True, axis=1)
        res[f"all_top{k}"] = auroc_reset(reset)
    for key, sk in (("none", "none"), ("all_top10", "top10"), ("all_top30", "top30")):
        if abs(res[key] - saved[sk]) > TOL:
            raise RuntimeError(f"{tag}: {key} {res[key]:.6f} does not reproduce the saved {saved[sk]:.6f}")
    rng = np.random.default_rng([sf.SEED_ROOT, int(rep), int(fold), STREAM_MOD])
    for mod, cols in (("F", np.arange(0, nF)), ("S", np.arange(nF, n_all))):
        o = np.argsort(-np.abs(attr[:, cols]), axis=1)
        for k in (10, 30):
            reset = np.zeros(attr.shape, bool)
            for i in range(attr.shape[0]):
                reset[i, cols[o[i, :k]]] = True
            res[f"{mod}_top{k}"] = auroc_reset(reset)
            rs = []
            for _ in range(se.RANDOM_DRAWS):
                reset = np.zeros(attr.shape, bool)
                for i in range(attr.shape[0]):
                    reset[i, rng.choice(cols, k, replace=False)] = True
                rs.append(auroc_reset(reset))
            res[f"{mod}_rand{k}"] = float(np.mean(rs))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, path)
    print(f"{tag}: check ok; " + " ".join(f"{k} {v:.3f}" for k, v in res.items()) + f"; {time.time() - t0:.0f} s", flush=True)


def main():
    task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    D = Data()
    ctx = sf.context("main")
    os.makedirs(f"{ctx['out']}/explain_top30", exist_ok=True)
    mine = D.fold_list()[task::ntasks]
    torch.set_num_threads(int(ctx["cfg"]["threads"]))
    print(f"task {task}/{ntasks}: {len(mine)} folds", flush=True)
    for rep, fold in mine:
        run_fold(D, ctx["out"], rep, fold, ctx["cfg"])
    print("explain_top30: task done", flush=True)


if __name__ == "__main__":
    main()
