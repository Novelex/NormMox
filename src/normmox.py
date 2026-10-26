"""NormMoX on one outer fold: cross-fitted TabM-mini experts on the chart deviations and the MoX head.
One outer fold of the frozen folds, with in-fold z (site rulers from the training-fold controls, data.Data.fold_z):
  experts   fMRI: TabM-mini (tabm.train_tabm) on the 136 fMRI z; sMRI: the same TabM-mini on the 185 sMRI z and their absolute values
            |z| (370 inputs)
  cross-fit per seed s (5 seeds) the training fold is dealt into K = 10 inner parts (stratified by diagnosis x site); inner model j
            trains on 8 parts, stops early on part j + 1 and scores part j out of fold; so every training person has one out-of-fold
            (OOF) logit per seed from a model that never saw them, and every expert learned from 80 % of the training fold
  head      per seed the OOF logits are standardised with their own mean / SD over the training people (fMRI over those with fMRI);
            MoX (fusion.fit_mox) is fitted on the OOF rows pooled over the seeds (~4,655 rows) and applied to every inner model's test
            logits (same standardisation); the fused log-odds are averaged over the 5 x 10 inner models; F_TABM / S_TABM = the experts
            alone (mean raw logit)
  threshold of every arm = Youden's J maximum on the OOF scores of the training fold (stored); the test fold is scored once, at the end
  outputs   pred/r{r}_f{f}.csv.gz (FILE_ID, y, arm, score, has_fmri; MoX pattern shares), oof/r{r}_f{f}.npz (inner parts, OOF logits,
            every inner model's test logits, OOF scores), models/r{r}_f{f}.pt (all 2 x 5 x 10 TabM state dicts, for the attributions),
            params/r{r}_f{f}.json (seed constants, head, self-test, OOF AUROCs, thresholds; complete last)
Modes: main = the 100 folds (outputs/main); perm = repeat 0 folds 0-1 with the training-fold labels permuted within site (test labels
real; the shuffled-label control, outputs/perm); smoke = K = 3, 2 seeds, 3 epochs, repeat 0 fold 0 (outputs/smoke).
No test AUROC is computed here.
usage: python src/normmox.py TASK NTASKS [main|perm|smoke]"""
import gzip
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

import fusion as fu
from data import S, Data
from tabm import train_tabm
from transformer import predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_T = S["tabm"]
OUT_BASE = os.path.join(ROOT, "outputs")
SEED_ROOT = 20261021                                   # base of every random stream (inner parts, experts, permutations)
K, N_SEEDS = 10, 5
STREAM_F, STREAM_S, STREAM_PART, STREAM_PERM = 1, 2, 7, 9
ARMS = ("F_TABM", "S_TABM", "MOX")


def stream_seed(rep, fold, s, j, stream):
    """one fixed 32-bit seed per (repeat, fold, seed, inner model, stream)"""
    return int(np.random.SeedSequence([SEED_ROOT, int(rep), int(fold), int(s), int(j), int(stream)]).generate_state(1)[0])


def inner_parts(D, train, rep, fold, s, k):
    """the training fold dealt into k parts, stratified by diagnosis x site: within each stratum the people are shuffled and dealt in
    turn from a random starting part; -1 outside the training fold"""
    rng = np.random.default_rng([SEED_ROOT, int(rep), int(fold), int(s), STREAM_PART])
    part = np.full(D.n, -1)
    for dx, st in sorted(set(zip(D.y[train].tolist(), D.sitenum[train].tolist()))):
        members = np.where(train & (D.y == dx) & (D.sitenum == st))[0]
        members = members[rng.permutation(len(members))]
        part[members] = (np.arange(len(members)) + int(rng.integers(k))) % k
    return part


def inner_masks(part, j, k):
    """inner model j: held out (OOF) = part j, early stopping = part j + 1, training = the other k - 2 parts"""
    oof, es = part == j, part == (j + 1) % k
    tr = (part >= 0) & ~oof & ~es
    return tr, es, oof


def struct_inputs(zs, train):
    """the sMRI expert's inputs: the 185 z and their |z| (a blank stays blank), median-filled and standardised on the training fold"""
    X, M = Data.tokens(np.concatenate([zs, np.abs(zs)], axis=1), train)
    return X.astype(np.float64), M


def permuted_labels(D, train, rep, fold):
    """shuffled-label control: the training people's labels permuted within site; test labels untouched"""
    rng = np.random.default_rng([SEED_ROOT, int(rep), int(fold), STREAM_PERM])
    yp = D.y.astype(int).copy()
    for st in np.unique(D.sitenum[train]):
        idx = np.where(train & (D.sitenum == st))[0]
        yp[idx] = yp[idx][rng.permutation(len(idx))]
    return yp


def youden(y, s):
    """the decision threshold t (people with score > t are called autistic) that maximises sensitivity + specificity - 1 on the fitting
    rows: the midpoint between the chosen ROC threshold and the largest score below it"""
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmax(tpr - fpr))
    ss = np.unique(s)
    if not np.isfinite(thr[i]):
        return float(ss[-1])
    kk = int(np.searchsorted(ss, thr[i]))
    return float((ss[kk - 1] + ss[kk]) / 2.0) if kk > 0 else float(ss[0] - 1.0)


def threshold_metrics(y, s, t):
    pred = s > t
    tp, tn = int((pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    fn, fp = int((~pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    return dict(sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1))


def nan0(a):
    return np.where(np.isfinite(a), a, 0.0)


def atomic_write(path, data_bytes):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data_bytes)
    os.replace(tmp, path)


def train_experts(D, XF, MF, XS, MS, y_fit, train, test, has, rep, fold, k, n_seeds, cfg):
    """every inner model of every seed. Returns the OOF logits (people x seeds; NaN outside the training fold), every inner model's test
    logits (people x seeds x k; NaN outside the test fold), the parts, the state dicts and one summary per model"""
    n = D.n
    lF_oof, lS_oof = np.full((n, n_seeds), np.nan), np.full((n, n_seeds), np.nan)
    lF_test, lS_test = np.full((n, n_seeds, k), np.nan), np.full((n, n_seeds, k), np.nan)
    parts = np.full((n, n_seeds), -1)
    states, info = [], []
    for s in range(n_seeds):
        part = inner_parts(D, train, rep, fold, s, k)
        parts[:, s] = part
        for j in range(k):
            tr, es, oof = inner_masks(part, j, k)
            if ((tr | es | oof) & test).any() or not ((tr | es | oof) == train).all() or (tr & es).any() or (tr & oof).any() or (es & oof).any():
                raise RuntimeError(f"seed {s} inner {j}: training / early-stopping / OOF parts are not disjoint, or touch the test fold")
            exF, iF = train_tabm(XF, y_fit, tr & has, es & has, cfg, stream_seed(rep, fold, s, j, STREAM_F))
            exS, iS = train_tabm(XS, y_fit, tr, es, cfg, stream_seed(rep, fold, s, j, STREAM_S))
            lf, ls = predict(exF, XF, MF), predict(exS, XS, MS)
            if not (np.isfinite(lf).all() and np.isfinite(ls).all()):
                raise RuntimeError(f"seed {s} inner {j}: non-finite expert logits")
            lF_oof[oof, s], lS_oof[oof, s] = lf[oof], ls[oof]
            lF_test[test, s, j], lS_test[test, s, j] = lf[test], ls[test]
            states.append(dict(seed=s, inner=j, F={a: v.detach().clone() for a, v in exF.net.state_dict().items()},
                               S={a: v.detach().clone() for a, v in exS.net.state_dict().items()}))
            info.append(dict(seed=s, inner=j, n_train=int(tr.sum()), n_es=int(es.sum()), n_oof=int(oof.sum()),
                             F_epochs=iF["epochs"], F_best_epoch=iF["best_epoch"], F_val_es_bce=iF["val_es_bce"],
                             F_val_es_auroc=float(roc_auc_score(y_fit[es & has], lf[es & has])),
                             F_oof_auroc=float(roc_auc_score(y_fit[oof & has], lf[oof & has])),
                             S_epochs=iS["epochs"], S_best_epoch=iS["best_epoch"], S_val_es_bce=iS["val_es_bce"],
                             S_val_es_auroc=float(roc_auc_score(y_fit[es], ls[es])), S_oof_auroc=float(roc_auc_score(y_fit[oof], ls[oof]))))
    if not (np.isfinite(lF_oof[train]).all() and np.isfinite(lS_oof[train]).all() and np.isnan(lF_oof[test]).all()
            and np.isfinite(lF_test[test]).all() and np.isfinite(lS_test[test]).all() and np.isnan(lS_test[train]).all()):
        raise RuntimeError("OOF / test logits incomplete: every training person needs one OOF logit per seed, every test person one per model")
    return lF_oof, lS_oof, lF_test, lS_test, parts, states, info


def arm_scores(heads, r):
    """log-odds of every arm for one set of rows built by fusion.seed_rows"""
    m = heads["MOX"]
    return dict(F_TABM=r["lF"], S_TABM=r["lS"], MOX=fu.mox_f(m["p"], m["cF"], m["cS"], r["zF"], r["zS"], r["has_f"]))


def fit_and_apply(D, lF_oof, lS_oof, lF_test, lS_test, y_fit, y_true, train, test, has, k, n_seeds, do_self_test):
    """MoX on the pooled OOF rows; OOF scores, AUROCs and thresholds (training fold only); then every inner model's test rows"""
    ids = D.ids
    consts, oof_rows = [], []
    for s in range(n_seeds):
        c = fu.seed_scaler(nan0(lF_oof[:, s]), nan0(lS_oof[:, s]), has, train, train)
        consts.append(c)
        oof_rows.append(fu.seed_rows(c, nan0(lF_oof[:, s]), nan0(lS_oof[:, s]), has, y_fit, train, ids))
    P = fu.pool(oof_rows)
    heads = dict(MOX=fu.fit_mox(P["zF"], P["zS"], P["y"], P["has_f"]),
                 n_rows=int(len(P["y"])), n_autistic=int(P["y"].sum()), n_without_fmri=int((~P["has_f"]).sum()))
    if not heads["MOX"]["ok"]:
        raise RuntimeError(f"MoX fit check failed: dev {heads['MOX']['dev_F']:.1e} / {heads['MOX']['dev_S']:.1e}, gradient per row "
                           f"{heads['MOX']['grad_per_row']:.1e}, optimiser: {heads['MOX']['message']}")
    st = fu.self_test(P["zF"], P["zS"], P["y"], P["has_f"], heads["MOX"]) if do_self_test else None
    if st is not None and not st["ok"]:
        raise RuntimeError(f"self-test failed: {fu.summary(st)}")
    per = [arm_scores(heads, r) for r in oof_rows]
    oof_scores = {a: np.full(D.n, np.nan) for a in ARMS}
    for a in ARMS:
        oof_scores[a][train] = np.mean([p[a] for p in per], axis=0)
    oof_auroc = {a: float(roc_auc_score(y_fit[train], oof_scores[a][train])) for a in ARMS}
    thresholds = {a: youden(y_fit[train], oof_scores[a][train]) for a in ARMS}
    oof_thr = {a: threshold_metrics(y_fit[train], oof_scores[a][train], thresholds[a]) for a in ARMS}
    test_scores, posts = {a: [] for a in ARMS}, []
    m = heads["MOX"]
    for s in range(n_seeds):
        for j in range(k):
            LF, LS = nan0(lF_oof[:, s]), nan0(lS_oof[:, s])
            LF[test], LS[test] = lF_test[test, s, j], lS_test[test, s, j]
            r = fu.seed_rows(consts[s], LF, LS, has, y_true, test, ids)
            sc = arm_scores(heads, r)
            for a in ARMS:
                test_scores[a].append(sc[a])
            posts.append(fu.mox_posterior(m["p"], m["cF"], m["cS"], r["zF"], r["zS"], r["has_f"]))
    scores = {a: np.mean(test_scores[a], axis=0) for a in ARMS}
    for a in ARMS:
        if not np.isfinite(scores[a]).all():
            raise RuntimeError(f"non-finite test scores of {a}")
    return dict(heads=heads, consts=consts, self_test=st, oof_scores=oof_scores, oof_auroc=oof_auroc, thresholds=thresholds,
                oof_threshold_metrics=oof_thr, scores=scores, post=np.mean(posts, axis=0))


def run_fold(D, ctx, rep, fold, mode, first_in_task):
    t0 = time.time()
    out, k, n_seeds, cfg = ctx["out"], ctx["k"], ctx["n_seeds"], ctx["cfg"]
    tag = f"r{rep}_f{fold}"
    if os.path.exists(f"{out}/params/{tag}.json") and json.load(open(f"{out}/params/{tag}.json")).get("complete"):
        print(f"{tag}: complete, skipped", flush=True)
        return False
    zi = D.fold_z(rep, fold)
    train, test = zi["train"], zi["test"]
    if (train & test).any() or not (train | test).any():
        raise RuntimeError(f"{tag}: bad fold masks")
    y_true, has, ids = D.y.astype(int), ~D.blank_fmri, D.ids
    y_fit = permuted_labels(D, train, rep, fold) if mode == "perm" else y_true.copy()
    if mode == "perm" and not ((y_fit[test] == y_true[test]).all() and (y_fit[train] != y_true[train]).any()):
        raise RuntimeError(f"{tag}: the permutation did not change the training labels or touched the test labels")
    XF, MF = D.tokens(zi["zf"], train)
    XF = XF.astype(np.float64)
    XS, MS = struct_inputs(zi["zs"], train)
    lF_oof, lS_oof, lF_test, lS_test, parts, states, info = train_experts(D, XF, MF, XS, MS, y_fit, train, test, has, rep, fold, k,
                                                                           n_seeds, cfg)
    R = fit_and_apply(D, lF_oof, lS_oof, lF_test, lS_test, y_fit, y_true, train, test, has, k, n_seeds, first_in_task)
    scores, post = R["scores"], R["post"]
    rows = []
    for a in ARMS:
        for i, idx in enumerate(np.where(test)[0]):
            r = dict(FILE_ID=ids[idx], y=int(y_true[idx]), arm=a, score=float(scores[a][i]), has_fmri=bool(has[idx]))
            if a == "MOX":
                r.update(post_none=post[i, 0], post_F=post[i, 1], post_S=post[i, 2], post_both=post[i, 3])
            rows.append(r)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(pd.DataFrame(rows).to_csv(index=False).encode())
    atomic_write(f"{out}/pred/{tag}.csv.gz", buf.getvalue())
    obuf = io.BytesIO()
    np.savez(obuf, ids=np.asarray(ids, dtype=str), y=y_true, y_fit=y_fit, has_fmri=has, train=train, test=test, parts=parts,
             lF_oof=lF_oof, lS_oof=lS_oof, lF_test=lF_test, lS_test=lS_test, **{f"oof_{a}": R["oof_scores"][a] for a in ARMS})
    atomic_write(f"{out}/oof/{tag}.npz", obuf.getvalue())
    mbuf = io.BytesIO()
    torch.save(states, mbuf)
    atomic_write(f"{out}/models/{tag}.pt", mbuf.getvalue())
    params = dict(fold=tag, mode=mode, k=k, n_seeds=n_seeds, n_models=len(states), tabm=dict(cfg), models=info, seed_constants=R["consts"],
                  heads=fu.summary(R["heads"]), self_test=fu.summary(R["self_test"]) if R["self_test"] is not None else None,
                  oof_auroc=R["oof_auroc"], thresholds=R["thresholds"], oof_threshold_metrics=R["oof_threshold_metrics"],
                  largest_abs_test_log_odds={a: float(np.max(np.abs(scores[a]))) for a in ARMS}, n_train=int(train.sum()),
                  n_test=int(test.sum()), n_test_without_fmri=int((test & ~has).sum()), torch=torch.__version__,
                  seconds=round(time.time() - t0, 1), complete=True)
    atomic_write(f"{out}/params/{tag}.json", json.dumps(params, indent=1, allow_nan=False).encode())
    oa = R["oof_auroc"]
    print(f"{tag}: OOF AUROC F {oa['F_TABM']:.3f} S {oa['S_TABM']:.3f} MOX {oa['MOX']:.3f} | "
          f"epochs F {np.median([m_['F_best_epoch'] for m_ in info]):.0f} S {np.median([m_['S_best_epoch'] for m_ in info]):.0f} (median best) | "
          f"MoX scales {R['heads']['MOX']['scale_F']:.2f} / {R['heads']['MOX']['scale_S']:.2f} | {params['seconds']} s", flush=True)
    return True


def context(mode):
    cfg, k, n_seeds = dict(CFG_T), K, N_SEEDS
    if mode == "smoke":
        cfg.update(max_epochs=3, patience=2)
        k, n_seeds = 3, 2
    ctx = dict(out=f"{OUT_BASE}/{mode}", k=k, n_seeds=n_seeds, cfg=cfg)
    for d in ("pred", "oof", "models", "params"):
        os.makedirs(f"{ctx['out']}/{d}", exist_ok=True)
    return ctx


def main():
    task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "main"
    assert mode in ("main", "perm", "smoke")
    D = Data()
    ctx = context(mode)
    folds = [(0, 0), (0, 1)] if mode == "perm" else ([(0, 0)] if mode == "smoke" else D.fold_list())
    mine = folds[task::ntasks]
    print(f"task {task}/{ntasks} mode {mode}: {len(mine)} folds {mine}; K {ctx['k']}, seeds {ctx['n_seeds']}", flush=True)
    first = True
    for rep, fold in mine:
        if run_fold(D, ctx, rep, fold, mode, first):
            first = False
    print("FOLD TASK DONE", flush=True)


if __name__ == "__main__":
    main()
