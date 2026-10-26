"""Explanation maps (Eq. 5): what moves each test person's fused MoX log-odds away from the typical person (z = 0 in every chart
input, i.e. a person exactly at their normative expectation for age, sex and site), and how faithful the maps are.
Per test person: Integrated Gradients (Sundararajan et al., ICML 2017) of the fused log-odds f through every inner TabM expert and
the fold's MoX head, along the straight path from the typical person's inputs (z = 0 and |z| = 0, tokenised with the fold's training
constants) to the person's inputs, STEPS midpoint steps. The contributions of the 136 fMRI network pairs, the 185 structure z and the
185 structure |z| sum to f(person) - f(typical) (completeness; the error is saved). IG is linear in f, so the fused map is the mean
over the 5 x 10 inner models of each model's map. People without fMRI get zero fMRI contributions (e_F = 0 by construction).
Faithfulness (deletion): the person's top-k inputs by |contribution| (k = 10, 30 of the 506) are set back to typical and the fused
score recomputed with every model; the same with k random inputs (RANDOM_DRAWS draws).
Checks in every fold: the torch fused model reproduces the saved MOX test scores (<= 1e-5); resetting no input reproduces them.
Outputs: outputs/<mode>/explain/r{r}_f{f}.npz (ids, y, has_fmri, attr_F, attr_S_z, attr_S_abs, f_x, f_typical, completeness
error, deletion AUROCs). usage: python src/explain.py TASK NTASKS [main|perm]"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import normmox as sf
from data import Data
from tabm import FMRIExpert

STEPS, CHUNK, TOP_K, RANDOM_DRAWS, STREAM_DEL = 64, 26, (10, 30), 5, 5


def baseline_tokens(zi, train):
    """the people's tokens and the typical person's tokens (z = 0, |z| = 0) under the fold's training statistics: the zero row is
    appended outside the fitting rows, so the people's tokens are exactly those of normmox.py"""
    fit = np.append(train, False)
    XF, MF = Data.tokens(np.vstack([zi["zf"], np.zeros((1, zi["zf"].shape[1]))]), fit)
    XS, MS = sf.struct_inputs(np.vstack([zi["zs"], np.zeros((1, zi["zs"].shape[1]))]), fit)
    return XF[:-1].astype(np.float64), XF[-1].astype(np.float64), XS[:-1], XS[-1]


class Fused(torch.nn.Module):
    """one inner model's fused MoX log-odds: the two TabM experts, the seed's standardisation and the fold's MoX head"""
    def __init__(self, exF, exS, c, m):
        super().__init__()
        self.exF, self.exS = exF, exS
        for k in ("muF", "sdF", "muS", "sdS"):
            self.register_buffer(k, torch.tensor(float(c[k])))
        self.register_buffer("p", torch.tensor(np.asarray(m["p"], float), dtype=torch.float32))
        self.register_buffer("cF", torch.tensor(float(m["cF"])))
        self.register_buffer("cS", torch.tensor(float(m["cS"])))

    def forward(self, xF, xS, has_f):
        zF = (self.exF(xF) - self.muF) / self.sdF
        zS = (self.exS(xS) - self.muS) / self.sdS
        eF = torch.where(has_f, F.softplus(self.p[4]) * zF + self.cF, torch.zeros_like(zF))
        eS = F.softplus(self.p[5]) * zS + self.cS
        logpi = self.p[:4] - torch.logsumexp(self.p[:4], 0)
        terms = torch.stack([logpi[0].expand_as(zS), logpi[1] + eF, logpi[2] + eS, logpi[3] + eF + eS])
        return self.p[6] + torch.logsumexp(terms, 0)


def load_models(tag, out, cfg, dF, dS):
    pp = json.load(open(f"{out}/params/{tag}.json"))
    states = torch.load(f"{out}/models/{tag}.pt")
    models = []
    for st in states:
        exF, exS = FMRIExpert(dF, cfg), FMRIExpert(dS, cfg)
        exF.net.load_state_dict(st["F"])
        exS.net.load_state_dict(st["S"])
        fm = Fused(exF, exS, pp["seed_constants"][st["seed"]], pp["heads"]["MOX"]).eval()
        for q in fm.parameters():
            q.requires_grad_(False)
        models.append(fm)
    return models, pp


@torch.no_grad()
def fused_scores(models, xF, xS, has):
    return np.mean([m(xF, xS, has).numpy().astype(np.float64) for m in models], axis=0)


def integrated_gradients(models, xF, xS, has, x0F, x0S, steps=STEPS, chunk=CHUNK):
    """mean over the models of each model's IG along the straight path from the typical person; returns attributions in token space
    (people x inputs) for the fMRI and structure inputs"""
    n = xF.shape[0]
    attrF, attrS = np.zeros((n, xF.shape[1])), np.zeros((n, xS.shape[1]))
    alphas = ((torch.arange(steps, dtype=torch.float32) + 0.5) / steps)[:, None, None]
    for a in range(0, n, chunk):
        b = slice(a, min(a + chunk, n))
        xf, xs, hf = xF[b], xS[b], has[b]
        B = xf.shape[0]
        pf = (x0F + alphas * (xf - x0F)).reshape(steps * B, -1).clone().requires_grad_(True)
        ps = (x0S + alphas * (xs - x0S)).reshape(steps * B, -1).clone().requires_grad_(True)
        hh = hf.repeat(steps)
        gF, gS = torch.zeros_like(pf), torch.zeros_like(ps)
        for m in models:
            f = m(pf, ps, hh).sum()
            g = torch.autograd.grad(f, (pf, ps))
            gF += g[0]
            gS += g[1]
        gF, gS = gF / len(models), gS / len(models)
        attrF[b] = ((xf - x0F) * gF.view(steps, B, -1).mean(0)).numpy()
        attrS[b] = ((xs - x0S) * gS.view(steps, B, -1).mean(0)).numpy()
    return attrF, attrS


def deletion(models, xF, xS, has, x0F, x0S, attr, y, rep, fold):
    """AUROC after resetting each person's top-k inputs (by |attribution| over all 506) to typical, and k random inputs"""
    nF = xF.shape[1]
    order = np.argsort(-np.abs(attr), axis=1)
    rng = np.random.default_rng([sf.SEED_ROOT, int(rep), int(fold), STREAM_DEL])
    out = {}

    def score_with(reset):                   # reset: people x 506 boolean
        xf, xs = xF.clone(), xS.clone()
        rF, rS = torch.as_tensor(reset[:, :nF]), torch.as_tensor(reset[:, nF:])
        xf[rF] = x0F.expand_as(xf)[rF]
        xs[rS] = x0S.expand_as(xs)[rS]
        return fused_scores(models, xf, xs, has)

    out["none"] = score_with(np.zeros(attr.shape, bool))
    for k in TOP_K:
        reset = np.zeros(attr.shape, bool)
        np.put_along_axis(reset, order[:, :k], True, axis=1)
        out[f"top{k}"] = score_with(reset)
        rs = []
        for _ in range(RANDOM_DRAWS):
            reset = np.zeros(attr.shape, bool)
            for i in range(attr.shape[0]):
                reset[i, rng.choice(attr.shape[1], k, replace=False)] = True
            rs.append(float(roc_auc_score(y, score_with(reset))))
        out[f"rand{k}_auroc"] = float(np.mean(rs))
    return out


def run_fold(D, out, rep, fold, cfg):
    t0 = time.time()
    tag = f"r{rep}_f{fold}"
    path = f"{out}/explain/{tag}.npz"
    if os.path.exists(path):
        print(f"{tag}: done, skipped", flush=True)
        return False
    zi = D.fold_z(rep, fold)
    train, test = zi["train"], zi["test"]
    XF, x0F, XS, x0S = baseline_tokens(zi, train)
    has = ~D.blank_fmri
    models, pp = load_models(tag, out, cfg, XF.shape[1], XS.shape[1])
    pr = pd.read_csv(f"{out}/pred/{tag}.csv.gz")
    saved = pr[pr["arm"] == "MOX"].set_index("FILE_ID").loc[D.ids[test], "score"].to_numpy()
    f32 = torch.float32
    xF, xS = torch.as_tensor(XF[test], dtype=f32), torch.as_tensor(XS[test], dtype=f32)
    hf = torch.as_tensor(has[test])
    x0F_t, x0S_t = torch.as_tensor(x0F, dtype=f32), torch.as_tensor(x0S, dtype=f32)
    f_x = fused_scores(models, xF, xS, hf)
    d = float(np.max(np.abs(f_x - saved)))
    if d > 1e-5:
        raise RuntimeError(f"{tag}: the torch fused model differs from the saved MOX scores by {d:.2e}")
    f0 = fused_scores(models, x0F_t[None], x0S_t[None], torch.tensor([True]))[0]
    attrF, attrS = integrated_gradients(models, xF, xS, hf, x0F_t, x0S_t)
    nz = len(D.struct_cols)
    total = attrF.sum(1) + attrS.sum(1)
    comp = np.abs(total - (f_x - f0))
    y = D.y[test].astype(int)
    dele = deletion(models, xF, xS, hf, x0F_t, x0S_t, np.concatenate([attrF, attrS], axis=1), y, rep, fold)
    if float(np.max(np.abs(dele["none"] - f_x))) > 1e-6:
        raise RuntimeError(f"{tag}: resetting no input changed the scores")
    aur = {k: float(roc_auc_score(y, v)) for k, v in dele.items() if not k.endswith("_auroc")}
    aur.update({k: v for k, v in dele.items() if k.endswith("_auroc")})
    buf = io.BytesIO()
    np.savez(buf, ids=np.asarray(D.ids[test], dtype=str), y=y, has_fmri=has[test], attr_F=attrF, attr_S_z=attrS[:, :nz], attr_S_abs=attrS[:, nz:],
             f_x=f_x, f_typical=f0, completeness_err=comp, saved_mox_max_diff=d, steps=STEPS, n_models=len(models),
             deletion_auroc=json.dumps(aur), fmri_cols=np.asarray(D.func_cols, dtype=str), struct_cols=np.asarray(D.struct_cols, dtype=str))
    sf.atomic_write(path, buf.getvalue())
    print(f"{tag}: f(typical) {f0:+.3f}; completeness error max {comp.max():.4f} mean {comp.mean():.4f}; AUROC full {aur['none']:.3f} "
          f"top10 {aur['top10']:.3f} rand10 {aur['rand10_auroc']:.3f} top30 {aur['top30']:.3f} rand30 {aur['rand30_auroc']:.3f}; "
          f"{time.time() - t0:.0f} s", flush=True)
    return True


def main():
    task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "main"
    assert mode in ("main", "perm")
    D = Data()
    ctx = sf.context(mode)
    os.makedirs(f"{ctx['out']}/explain", exist_ok=True)
    folds = [(0, 0), (0, 1)] if mode == "perm" else D.fold_list()
    mine = folds[task::ntasks]
    torch.set_num_threads(int(ctx["cfg"]["threads"]))
    print(f"task {task}/{ntasks} mode {mode}: {len(mine)} folds; steps {STEPS}", flush=True)
    for rep, fold in mine:
        run_fold(D, ctx["out"], rep, fold, ctx["cfg"])
    print("explain: task done", flush=True)


if __name__ == "__main__":
    main()
