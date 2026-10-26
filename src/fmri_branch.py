"""fMRI expert of the deep fusion baselines: TabM-mini on every fold of the 10 x 10 scheme.
usage: python src/fmri_branch.py TASK NTASKS
This task runs the folds fold_list()[TASK::NTASKS]. Per fold and seed the expert is trained on the TRAIN people with fMRI (70 % of the
training fold) and stops early on the VAL-ES people with fMRI (15 %). Saved per fold: raw logits of everyone, MC-dropout mean and SD
(20 passes) and the eval-mode SD over the 32 members for the VAL-HEAD and test people with fMRI, split codes, epochs, VAL-ES BCE and
AUROC, and the 5 state dicts (used by fusion_frozen.py and fusion_finetuned.py). No test AUROC is computed here.
Writes outputs/fbranch/main/."""
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data import S, Data
from tabm import mc_dropout, n_params, train_tabm
from transformer import predict, torch_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = S["tabm"]


def run_fold(D, rep, fold, out_dir):
    t0 = time.time()
    zi = D.fold_z(rep, fold)
    train, test = zi["train"], zi["test"]
    assert not (train & test).any() and train.sum() + test.sum() == D.n
    X, M = D.tokens(zi["zf"], train)
    X = X.astype(np.float64)
    has = ~D.blank_fmri
    y = D.y.astype(int)
    n_seeds = int(CFG["seeds"])
    logits = np.full((D.n, n_seeds), np.nan)
    mc_mean, mc_sd = np.full((D.n, n_seeds), np.nan), np.full((D.n, n_seeds), np.nan)
    member_sd = np.full((D.n, n_seeds), np.nan)
    split = np.full((D.n, n_seeds), -1, dtype=np.int8)
    epochs, best, bce, auc_es, states = [], [], [], [], []
    for s in range(n_seeds):
        tr, es, hd = D.split3(train, rep, fold, s)
        assert not (tr & es).any() and not (tr & hd).any() and not (es & hd).any() and ((tr | es | hd) == train).all()
        tr_f, es_f = tr & has, es & has
        model, info = train_tabm(X, y, tr_f, es_f, CFG, torch_seed(rep, fold, s, int(CFG["seed_stream"])))
        logits[:, s] = predict(model, X, M)
        rows = (hd | test) & has
        mm, sd = mc_dropout(model, X[rows], M[rows], int(CFG["mc_passes"]), torch_seed(rep, fold, s, int(CFG["mc_dropout_stream"])))
        mc_mean[rows, s], mc_sd[rows, s] = mm, sd
        member_sd[rows, s] = model.member_sd(torch.as_tensor(X[rows], dtype=torch.float32)).numpy().astype(np.float64)
        split[tr, s], split[es, s], split[hd, s], split[test, s] = 0, 1, 2, 3
        epochs.append(info["epochs"]); best.append(info["best_epoch"]); bce.append(info["val_es_bce"])
        auc_es.append(roc_auc_score(y[es_f], logits[es_f, s]))
        states.append({k: v.detach().clone() for k, v in model.net.state_dict().items()})
    assert np.isfinite(logits).all(), "non-finite logits"
    sec = round(time.time() - t0, 1)
    tag = f"r{rep}_f{fold}"
    os.makedirs(f"{out_dir}/models", exist_ok=True)
    torch.save(states, f"{out_dir}/models/{tag}.pt")
    np.savez(f"{out_dir}/{tag}.npz", logits=logits, mc_mean=mc_mean, mc_sd=mc_sd, member_sd=member_sd, split=split, y=y, has_fmri=has,
             ids=D.ids, epochs=np.array(epochs), best_epoch=np.array(best), val_es_bce=np.array(bce), val_es_auroc=np.array(auc_es),
             n_params=n_params(model), seconds=sec, scheme=S["scheme"])
    print(f"{tag}: train {train.sum()} test {test.sum()} | epochs {epochs} best {best} | VAL-ES AUROC {np.round(auc_es, 3).tolist()} "
          f"(mean {np.mean(auc_es):.3f}) | {sec} s", flush=True)


def main():
    task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    D = Data()
    folds = D.fold_list()
    assert len(folds) == 100
    mine = folds[task::ntasks]
    out_dir = os.path.join(ROOT, "outputs/fbranch/main")
    os.makedirs(out_dir, exist_ok=True)
    print(f"task {task}/{ntasks}: {len(mine)} folds {mine}", flush=True)
    for rep, fold in mine:
        run_fold(D, rep, fold, out_dir)
    print("done", flush=True)


if __name__ == "__main__":
    main()
