"""Deep fusion baseline with frozen experts, one fold of the 10 x 10 scheme at a time:
  1 checks: the stored per-seed splits of fmri_branch.py equal D.split3 and the test fold; the stored TabM logits equal a re-prediction
    from the stored weights (<= 1e-5)
  2 sMRI transformer, 5 seeds: transformer.train_branch on the stored TRAIN / VAL-ES rows; logits for everyone; MC-dropout 20 passes on
    VAL-HEAD + test (random state restored); VAL-ES AUROC; layer-1 relation-bias table
  3 heads (fusion.py) on the pooled VAL-HEAD rows; the invariance self-test in the first fold of every task; the F_BRANCH test scores must
    equal fbranch_logits (<= 1e-9). The paper's "experts frozen" row is the MOX_FROZEN arm.
  4 outputs (a fold whose params file says complete is skipped): pred/r{r}_f{f}.csv.gz (FILE_ID, y, arm, score, MoX posteriors),
    struct/r{r}_f{f}.npz (sMRI logits, MC mean / SD, splits), models/r{r}_f{f}.pt (5 sMRI state dicts, for fusion_finetuned.py),
    params/r{r}_f{f}.json (seed constants, heads, self-test, diagnostics; complete: true last)
Reads outputs/fbranch/main/, writes outputs/fusion/main/. No test AUROC is computed here.
usage: python src/fusion_frozen.py TASK NTASKS"""
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
from sklearn.metrics import roc_auc_score

import fusion as fu
from data import S, Data
from tabm import FMRIExpert, fbranch_logits, mc_dropout
from transformer import STRUCT_TYPES, network_map, predict, struct_relations, torch_seed, train_branch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_T = S["tabm"]
FB = os.path.join(ROOT, "outputs/fbranch")
OUT_BASE = os.path.join(ROOT, "outputs/fusion")


def train_structure(XS, MS, y, tr, es, RS, seed, tcfg):
    """train_branch, then: every VAL-ES loss finite, eval mode, gradients cleared"""
    model, info = train_branch(XS, MS, y.astype(np.float64), tr, es, RS, len(STRUCT_TYPES), S["model"], tcfg, seed)
    if not all(math.isfinite(v) for v in info["history"]):
        raise RuntimeError(f"non-finite VAL-ES loss in the structure branch (seed {seed}): {info['history'][-3:]}")
    model.eval()
    model.zero_grad(set_to_none=True)
    return model, info


def atomic_write(path, data_bytes):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data_bytes)
    os.replace(tmp, path)


def tabm_reprediction(states, XF, MF):
    out = []
    for st in states:
        ex = FMRIExpert(XF.shape[1], CFG_T)
        ex.net.load_state_dict(st)
        ex.eval()
        out.append(ex)
    return out


def run_fold(D, ctx, rep, fold, first_in_task):
    t0 = time.time()
    out = ctx["out"]
    tag = f"r{rep}_f{fold}"
    if os.path.exists(f"{out}/params/{tag}.json") and json.load(open(f"{out}/params/{tag}.json")).get("complete"):
        print(f"{tag}: complete, skipped", flush=True)
        return False
    npz, pt = f"{FB}/main/{tag}.npz", f"{FB}/main/models/{tag}.pt"
    z = np.load(npz, allow_pickle=True)
    states = torch.load(pt)
    zi = D.fold_z(rep, fold)
    train, test = zi["train"], zi["test"]
    y_true = D.y.astype(int)
    has = ~D.blank_fmri
    ids = D.ids
    if not ((z["ids"] == ids).all() and (z["y"] == y_true).all() and (z["has_fmri"] == has).all()):
        raise RuntimeError(f"{tag}: the fMRI-branch file's people, labels or fMRI flags differ")
    y_fit = y_true
    XF, MF = D.tokens(zi["zf"], train)
    XF = XF.astype(np.float64)
    XS, MS = D.tokens(zi["zs"], train)
    n_seeds = ctx["n_seeds"]
    split = z["split"].astype(int)
    experts = tabm_reprediction(states, XF, MF)
    seeds, lS_all, mS_all, sS_all, s_states = [], [], [], [], []
    for s in range(n_seeds):
        tr, es, hd = D.split3(train, rep, fold, s)
        sp = split[:, s]
        if not ((sp == 0) == tr).all() or not ((sp == 1) == es).all() or not ((sp == 2) == hd).all() or not ((sp == 3) == test).all():
            raise RuntimeError(f"{tag} seed {s}: the stored split differs from D.split3 / the test fold")
        if (tr & es).any() or (tr & hd).any() or (es & hd).any() or ((tr | es | hd) & test).any() or not ((tr | es | hd) == train).all():
            raise RuntimeError(f"{tag} seed {s}: TRAIN / VAL-ES / VAL-HEAD / test are not disjoint or do not cover the fold")
        lF_re = predict(experts[s], XF, MF)
        dmax = float(np.max(np.abs(lF_re - z["logits"][:, s])))
        if dmax > 1e-5:
            raise RuntimeError(f"{tag} seed {s}: stored TabM logits differ from the re-prediction by {dmax:.2e}")
        model, info = train_structure(XS, MS, y_fit, tr, es, ctx["RS"], torch_seed(rep, fold, s, 1), ctx["tcfg"])
        lS = predict(model, XS, MS)
        rows = hd | test
        mS, sS = np.full(D.n, np.nan), np.full(D.n, np.nan)
        mS[rows], sS[rows] = mc_dropout(model, XS[rows], MS[rows], 20, torch_seed(rep, fold, s, 11))
        if not (np.isfinite(lS).all() and (sS[rows] > 0).all()):
            raise RuntimeError(f"{tag} seed {s}: non-finite structure logits or an MC-dropout SD of 0")
        mcF_ok = np.isfinite(z["mc_mean"][rows & has, s]).all() and (z["mc_sd"][rows & has, s] > 0).all()
        if not mcF_ok:
            raise RuntimeError(f"{tag} seed {s}: TabM MC-dropout missing or SD 0 on VAL-HEAD / test people with fMRI")
        seeds.append(dict(seed=s, structure_epochs=info["epochs"], structure_best_epoch=info["best_epoch"],
                          structure_val_es_bce=info["val_es_bce"], structure_val_es_auroc=float(roc_auc_score(y_fit[es], lS[es])),
                          tabm_val_es_auroc=float(z["val_es_auroc"][s]), tabm_best_epoch=int(z["best_epoch"][s]),
                          bias_layer1=model.rel_bias.detach().numpy().tolist(), tabm_reprediction_max_diff=dmax))
        lS_all.append(lS); mS_all.append(mS); sS_all.append(sS)
        s_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    # heads
    val_list, test_list, consts = [], [], []
    for s in range(n_seeds):
        sp = split[:, s]
        tr, hd = sp == 0, sp == 2
        lF = z["logits"][:, s]
        c = fu.seed_scaler(lF, lS_all[s], has, hd, tr)
        consts.append(c)
        mcF, mcS = (z["mc_mean"][:, s], z["mc_sd"][:, s]), (mS_all[s], sS_all[s])
        val_list.append(fu.seed_rows(c, lF, lS_all[s], has, y_fit, hd, ids, mcF=mcF, mcS=mcS))
        test_list.append(fu.seed_rows(c, lF, lS_all[s], has, y_true, test, ids, mcF=mcF, mcS=mcS))
    heads = fu.fit_heads(val_list)
    fu.assert_fit(heads)
    st = None
    if first_in_task:
        P = fu.pool(val_list)
        st = fu.self_test(P["zF"], P["zS"], P["y"], P["has_f"], heads["MOX_FROZEN"], heads=heads, test_rows=test_list)
        if not st["ok"]:
            raise RuntimeError(f"{tag}: invariance self-test failed: {fu.summary(st)}")
    scores, post = fu.combine_seeds(heads, test_list)
    fb = np.mean([fbranch_logits(z["logits"][:, s], has, split[:, s] == 0)[test] for s in range(n_seeds)], axis=0)
    dfb = float(np.max(np.abs(scores["F_BRANCH"] - fb)))
    if dfb > 1e-9:
        raise RuntimeError(f"{tag}: F_BRANCH scores differ from fbranch_logits by {dfb:.2e}")
    # outputs
    rows = []
    for a in fu.ARMS:
        for i, idx in enumerate(np.where(test)[0]):
            r = dict(FILE_ID=ids[idx], y=int(y_true[idx]), arm=a, score=float(scores[a][i]), has_fmri=bool(has[idx]))
            if a == "MOX_FROZEN":
                r.update(post_none=post[i, 0], post_F=post[i, 1], post_S=post[i, 2], post_both=post[i, 3])
            rows.append(r)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(pd.DataFrame(rows).to_csv(index=False).encode())
    atomic_write(f"{out}/pred/{tag}.csv.gz", buf.getvalue())
    sbuf = io.BytesIO()
    np.savez(sbuf, ids=np.asarray(ids, dtype=str), split=split[:, :n_seeds], lS=np.column_stack(lS_all), mS=np.column_stack(mS_all),
             sS=np.column_stack(sS_all), y_fit=y_fit, y=y_true, has_fmri=has)
    atomic_write(f"{out}/struct/{tag}.npz", sbuf.getvalue())
    mbuf = io.BytesIO()
    torch.save(s_states, mbuf)
    atomic_write(f"{out}/models/{tag}.pt", mbuf.getvalue())
    params = dict(fold=tag, n_seeds=n_seeds, seeds=seeds, seed_constants=consts, heads=fu.summary(heads),
                  self_test=fu.summary(st) if st is not None else None, f_branch_rule_max_diff=dfb,
                  largest_abs_test_log_odds={a: float(np.max(np.abs(scores[a]))) for a in fu.ARMS},
                  n_test=int(test.sum()), n_test_without_fmri=int((test & ~has).sum()),
                  torch=torch.__version__, seconds=round(time.time() - t0, 1), complete=True)
    atomic_write(f"{out}/params/{tag}.json", json.dumps(params, indent=1, allow_nan=False).encode())
    va = np.mean([s_["structure_val_es_auroc"] for s_ in seeds])
    print(f"{tag}: structure epochs {[s_['structure_epochs'] for s_ in seeds]} best {[s_['structure_best_epoch'] for s_ in seeds]} | "
          f"VAL-ES AUROC structure {va:.3f}, TabM {np.mean([s_['tabm_val_es_auroc'] for s_ in seeds]):.3f} | MoX scales "
          f"{heads['MOX_FROZEN']['scale_F']:.2f} / {heads['MOX_FROZEN']['scale_S']:.2f} | {params['seconds']} s", flush=True)
    return True


def main():
    task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    D = Data()
    net, _, n_match = network_map(os.path.join(ROOT, S["paths"]["yeo_map"]), D.struct_cols)
    if n_match != 148:
        raise RuntimeError(f"Yeo map matches {n_match} structure columns, expected 148")
    tcfg = dict(S["train"])
    n_seeds = len(S["seeds"])
    ctx = dict(RS=struct_relations(D.struct_cols, net), tcfg=tcfg, n_seeds=n_seeds, out=f"{OUT_BASE}/main")
    for d in ("pred", "struct", "models", "params"):
        os.makedirs(f"{ctx['out']}/{d}", exist_ok=True)
    folds = D.fold_list()
    mine = folds[task::ntasks]
    print(f"task {task}/{ntasks}: {len(mine)} folds {mine}", flush=True)
    first = True
    for rep, fold in mine:
        if run_fold(D, ctx, rep, fold, first):
            first = False
    print("done", flush=True)


if __name__ == "__main__":
    main()
