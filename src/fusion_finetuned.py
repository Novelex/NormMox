"""Deep fusion baseline, experts and head fine-tuned jointly; one fold of the 10 x 10 scheme at a time, per seed:
  start   the stored TabM weights (fmri_branch.py), the sMRI transformer weights and the fold's MOX_FROZEN head (theta, a_F, a_S, b) and
          offsets c_F, c_S (fusion_frozen.py); the seed's standardisation constants (muF, sdF, muS, sdS) and the offsets stay fixed
  model   f = b + log(pi0 + piF e^eF + piS e^eS + piB e^(eF + eS)), e_m = softplus(a_m) (l_m - mu_m) / sd_m + c_m, pi = softmax(theta);
          l_F = TabM's ensemble logit (logit of the mean member probability), l_S = the transformer's logit; e_F = 0 for people
          without fMRI during fine-tuning (training batches and VAL-ES)
  loss    mean BCE(f) over the batch + 0.3 [mean BCE(l_F) over the batch's people with fMRI + mean BCE(l_S) over the batch]
  train   AdamW lr 1e-4, two weight-decay groups (TabM 1e-4; transformer + head 1e-2), batch 64, gradient clip 1.0 over all parameters;
          early stopping on the VAL-ES BCE of f (patience 5, at most 30 epochs); epoch 0 (the MOX_FROZEN start) takes part and is kept
          if no epoch improves on it; VAL-HEAD is not used
  after   people without fMRI: exp(e_F) = its mean over the TRAIN controls with fMRI under the fine-tuned model
  checks  epoch 0 reproduces the seed's MOX_FROZEN test log-odds of the test people with fMRI (<= 1e-4); every loss finite
Outputs: outputs/fusion/full/pred/r{r}_f{f}.csv.gz (arm FULL + MoX posteriors) and params/r{r}_f{f}.json (per seed: kept epoch, VAL-ES
history, epoch-0 check value; complete: true last).
usage: python src/fusion_finetuned.py TASK NTASKS"""
import copy
import gzip
import io
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import fusion as fu
import fusion_frozen as sf
from data import S, Data
from tabm import FMRIExpert
from transformer import STRUCT_TYPES, Branch, torch_seed

FT = S["finetune"]


class FullModel(nn.Module):
    def __init__(self, tabm, struct, p, cF, cS, consts):
        super().__init__()
        self.tabm, self.struct = tabm, struct
        self.theta = nn.Parameter(torch.tensor(p[:4], dtype=torch.float32))
        self.a = nn.Parameter(torch.tensor(p[4:6], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(float(p[6]), dtype=torch.float32))
        self.register_buffer("c", torch.tensor([cF, cS], dtype=torch.float32))
        self.register_buffer("mu", torch.tensor([consts["muF"], consts["muS"]], dtype=torch.float32))
        self.register_buffer("sd", torch.tensor([consts["sdF"], consts["sdS"]], dtype=torch.float32))

    def parts(self, xf, mf, xs, ms):
        lF = self.tabm(xf, mf)
        lS = self.struct(xs, ms)
        sp = F.softplus(self.a)
        eF = sp[0] * (lF - self.mu[0]) / self.sd[0] + self.c[0]
        eS = sp[1] * (lS - self.mu[1]) / self.sd[1] + self.c[1]
        return lF, lS, eF, eS

    def fused(self, eF, eS, has, eF_missing=None):
        """eF of people without fMRI: 0 (fine-tuning) or log of a given mean of exp(eF) (after fine-tuning)"""
        fill = torch.zeros_like(eF) if eF_missing is None else torch.full_like(eF, float(eF_missing))
        eF = torch.where(has, eF, fill)
        logpi = self.theta - torch.logsumexp(self.theta, 0)
        t = torch.stack([logpi[0].expand_as(eS), logpi[1] + eF, logpi[2] + eS, logpi[3] + eF + eS])
        return self.b + torch.logsumexp(t, 0), t

    def forward(self, xf, mf, xs, ms, has, eF_missing=None):
        lF, lS, eF, eS = self.parts(xf, mf, xs, ms)
        f, t = self.fused(eF, eS, has, eF_missing)
        return f, lF, lS, t


def param_groups(model):
    tab = list(model.tabm.parameters())
    ids = {id(p) for p in tab}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return [dict(params=tab, weight_decay=1e-4), dict(params=rest, weight_decay=float(S["train"]["weight_decay"]))]


def tens(a, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype)


@torch.no_grad()
def evaluate(model, T, rows, eF_missing=None):
    model.eval()
    f, lF, lS, t = model(T["xf"][rows], T["mf"][rows], T["xs"][rows], T["ms"][rows], T["has"][rows], eF_missing)
    return f, t


def fine_tune(model, T, tr, es, y, seed, max_epochs, patience):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(param_groups(model), lr=float(FT["lr"]))
    g = torch.Generator().manual_seed(seed)
    yt = tens(y)
    tr_idx, es_idx = torch.as_tensor(np.where(tr)[0]), torch.as_tensor(np.where(es)[0])

    def val_loss():
        f, _ = evaluate(model, T, es_idx)
        v = float(F.binary_cross_entropy_with_logits(f, yt[es_idx]))
        if not math.isfinite(v):
            raise RuntimeError(f"non-finite VAL-ES loss in FULL (seed {seed})")
        return v

    best, state, best_ep, bad, hist = val_loss(), copy.deepcopy(model.state_dict()), 0, 0, []
    hist.append(best)
    ep = 0
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        for s0 in range(0, len(perm), 64):
            idx = perm[s0:s0 + 64]
            f, lF, lS, _ = model(T["xf"][idx], T["mf"][idx], T["xs"][idx], T["ms"][idx], T["has"][idx])
            h = T["has"][idx]
            loss = F.binary_cross_entropy_with_logits(f, yt[idx]) + float(FT["aux_weight"]) * (
                (F.binary_cross_entropy_with_logits(lF[h], yt[idx][h]) if bool(h.any()) else torch.zeros(()))
                + F.binary_cross_entropy_with_logits(lS, yt[idx]))
            if not math.isfinite(loss.detach().item()):
                raise RuntimeError(f"non-finite FULL training loss (seed {seed}, epoch {ep})")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = val_loss()
        hist.append(v)
        if v < best - 1e-6:
            best, state, best_ep, bad = v, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    model.zero_grad(set_to_none=True)
    return dict(kept_epoch=best_ep, epochs=ep, val_es_bce=best, history=hist)


def run_fold(D, ctx, rep, fold):
    t0 = time.time()
    tag, out = f"r{rep}_f{fold}", ctx["out"]
    if os.path.exists(f"{out}/params/{tag}.json") and json.load(open(f"{out}/params/{tag}.json")).get("complete"):
        print(f"{tag}: complete, skipped", flush=True)
        return False
    A = ctx["stage_a"]
    pa = json.load(open(f"{A}/params/{tag}.json"))
    if not pa.get("complete"):
        raise RuntimeError(f"{tag}: fusion_frozen.py output not complete")
    npz, pt = f"{sf.FB}/main/{tag}.npz", f"{sf.FB}/main/models/{tag}.pt"
    z = np.load(npz, allow_pickle=True)
    tab_states = torch.load(pt)
    st_states = torch.load(f"{A}/models/{tag}.pt")
    sa = np.load(f"{A}/struct/{tag}.npz")
    zi = D.fold_z(rep, fold)
    train, test = zi["train"], zi["test"]
    XF, MF = D.tokens(zi["zf"], train)
    XS, MS = D.tokens(zi["zs"], train)
    has, y = ~D.blank_fmri, D.y.astype(int)
    y_fit = sa["y_fit"].astype(int)
    if not (y_fit == z["y"]).all() or not (y_fit == y).all():
        raise RuntimeError(f"{tag}: the labels of the fusion_frozen and fmri_branch outputs differ from the data")
    T = dict(xf=tens(XF), mf=tens(MF), xs=tens(XS), ms=tens(MS), has=torch.as_tensor(has))
    hm = pa["heads"]["MOX_FROZEN"]
    p, cF, cS = np.array(hm["p"], float), float(hm["cF"]), float(hm["cS"])
    n_seeds = int(pa["n_seeds"])
    test_idx = torch.as_tensor(np.where(test)[0])
    fused, posts, seeds = [], [], []
    for s in range(n_seeds):
        sp = sa["split"][:, s]
        tr, es = sp == 0, sp == 1
        if not ((sp == 3) == test).all() or not (sp == z["split"][:, s]).all():
            raise RuntimeError(f"{tag} seed {s}: fusion_frozen / fmri_branch splits differ")
        consts = pa["seed_constants"][s]
        tabm = FMRIExpert(XF.shape[1], sf.CFG_T)
        tabm.net.load_state_dict(tab_states[s])
        struct = Branch(ctx["RS"], len(STRUCT_TYPES), S["model"])
        struct.load_state_dict(st_states[s])
        model = FullModel(tabm, struct, p, cF, cS, consts)
        # epoch-0 check against the seed's MOX_FROZEN scores (people with fMRI)
        f0, _ = evaluate(model, T, test_idx)
        zF = np.where(has, (z["logits"][:, s] - consts["muF"]) / consts["sdF"], 0.0)[test]
        zS = ((sa["lS"][:, s] - consts["muS"]) / consts["sdS"])[test]
        mox = fu.mox_f(p, cF, cS, zF, zS, has[test])
        d0 = float(np.max(np.abs(f0.numpy()[has[test]] - mox[has[test]])))
        if d0 > 1e-4:
            raise RuntimeError(f"{tag} seed {s}: FULL at epoch 0 differs from MOX_FROZEN by {d0:.2e}")
        info = fine_tune(model, T, tr, es, y_fit, torch_seed(rep, fold, s, int(FT["seed_stream"])), ctx["max_epochs"], ctx["patience"])
        # people without fMRI: exp(e_F) = mean over the TRAIN controls with fMRI under the fine-tuned model
        with torch.no_grad():
            model.eval()
            ctrl = torch.as_tensor(np.where(tr & (y_fit == 0) & has)[0])
            _, _, eF_c, _ = model.parts(T["xf"][ctrl], T["mf"][ctrl], T["xs"][ctrl], T["ms"][ctrl])
            eF_missing = float(torch.logsumexp(eF_c, 0) - math.log(len(ctrl)))
        f, t = evaluate(model, T, test_idx, eF_missing)
        fused.append(f.numpy().astype(np.float64))
        posts.append(torch.softmax(t, 0).T.numpy().astype(np.float64))
        seeds.append(dict(seed=s, epoch0_max_diff=d0, eF_missing=eF_missing, **info))
    score, post = np.mean(fused, axis=0), np.mean(posts, axis=0)
    if not np.isfinite(score).all():
        raise RuntimeError(f"{tag}: non-finite FULL scores")
    ids = D.ids
    rows = [dict(FILE_ID=ids[i], y=int(y[i]), arm="FULL", score=float(score[k]), has_fmri=bool(has[i]), post_none=post[k, 0],
                 post_F=post[k, 1], post_S=post[k, 2], post_both=post[k, 3]) for k, i in enumerate(np.where(test)[0])]
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(pd.DataFrame(rows).to_csv(index=False).encode())
    sf.atomic_write(f"{out}/pred/{tag}.csv.gz", buf.getvalue())
    params = dict(fold=tag, n_seeds=n_seeds, seeds=seeds, kept_epoch0=sum(s_["kept_epoch"] == 0 for s_ in seeds),
                  seconds=round(time.time() - t0, 1), complete=True)
    sf.atomic_write(f"{out}/params/{tag}.json", json.dumps(params, indent=1, allow_nan=False).encode())
    print(f"{tag}: kept epochs {[s_['kept_epoch'] for s_ in seeds]}, epoch-0 check max {max(s_['epoch0_max_diff'] for s_ in seeds):.1e}, "
          f"{params['seconds']} s", flush=True)
    return True


def main():
    task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    torch.set_num_threads(int(S["train"]["threads"]))
    D = Data()
    net, _, n_match = sf.network_map(os.path.join(sf.ROOT, S["paths"]["yeo_map"]), D.struct_cols)
    if n_match != 148:
        raise RuntimeError("Yeo map does not match 148 structure columns")
    ctx = dict(RS=sf.struct_relations(D.struct_cols, net), stage_a=f"{sf.OUT_BASE}/main", out=f"{sf.OUT_BASE}/full",
               max_epochs=int(FT["max_epochs"]), patience=int(FT["patience"]))
    for d in ("pred", "params"):
        os.makedirs(f"{ctx['out']}/{d}", exist_ok=True)
    folds = D.fold_list()
    mine = folds[task::ntasks]
    print(f"task {task}/{ntasks}: {len(mine)} folds", flush=True)
    for rep, fold in mine:
        run_fold(D, ctx, rep, fold)
    print("done", flush=True)


if __name__ == "__main__":
    main()
