"""Fusion heads on the experts' out-of-fold logits (numpy / scipy only).
Each expert's logits are standardised per seed; a head is fitted on the rows pooled over the seeds, applied to each seed's test rows,
and the fused log-odds are averaged over the seeds. Binary cross-entropy summed over people; L-BFGS-B.
  MOX    b + log(pi0 + piF e^eF + piS e^eS + piB e^(eF + eS)), e_m = softplus(a_m) z_m + c_m, where c_m = -log mean over the fitting
         controls of exp(softplus(a_m) z_m); 7 parameters; penalty mox_theta_ridge ||theta||^2 + mox_scale_ridge sum_m (softplus(a_m) - 1)^2;
         several starting points, the best fit is kept. People without fMRI get eF = 0.
  ADD    w0 + wF zF + wS zS, ridge on wF, wS.   EQUAL (zF + zS) / 2.
  BARY   lambda_S mS + (1 - lambda_S) mF, lambda_S = sigmoid(alpha + beta (log sF - log sS)) on the MC-dropout means and SDs.
Missing fMRI: BARY lambda_S = 1; ADD and EQUAL use the seed's mean training fMRI logit."""
import json
import math
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

S = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "settings.json")))
H = S["heads"]
OPTIONS = dict(H["optimizer_options"])
ONE = float(np.log(np.e - 1.0))                                   # softplus(ONE) = 1
STARTS = tuple(np.array(s, float) for s in H["mox_starts"])       # starting points of the MoX fit
ARMS = ("F_BRANCH", "S_BRANCH", "EQUAL", "ADD", "MOX_FROZEN", "BARY")


def softplus(x):
    return np.logaddexp(0.0, x)


def bce(f, y):
    """binary cross-entropy summed over people; f = log-odds, y = 0 / 1"""
    return float(np.sum(np.logaddexp(0.0, f) - y * f))


def _message(res):
    return res.message if isinstance(res.message, str) else res.message.decode()


def _finite(name, *arrays):
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError(f"{name}: non-finite values")


def _mask(m, n, name):
    m = np.asarray(m)
    if m.dtype != bool or m.shape != (n,):
        raise ValueError(f"{name} must be a boolean mask of length {n} (got dtype {m.dtype}, shape {m.shape})")
    return m


def _labels(y):
    y = np.asarray(y, float)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 (control) or 1 (autism)")
    return y


# ---------- per-seed standardisation ----------
def seed_scaler(lF, lS, has_f, val_head, train):
    """this seed's constants: mean and SD (ddof 1) of each expert's logits on the head-fitting rows (fMRI over people with fMRI), and
    the fill value for missing fMRI (the mean fMRI logit over this seed's training people with fMRI; raw and standardised)"""
    n = len(lF)
    has_f, val_head, train = _mask(has_f, n, "has_f"), _mask(val_head, n, "val_head"), _mask(train, n, "train")
    vf = val_head & has_f
    c = dict(muF=float(lF[vf].mean()), sdF=float(lF[vf].std(ddof=1)), muS=float(lS[val_head].mean()), sdS=float(lS[val_head].std(ddof=1)))
    if not all(np.isfinite(v) for v in c.values()) or c["sdF"] <= 0 or c["sdS"] <= 0:
        raise ValueError(f"degenerate VAL-HEAD logits, cannot standardise: {c}")
    c["fillF_raw"] = float(lF[train & has_f].mean())
    c["fillF"] = (c["fillF_raw"] - c["muF"]) / c["sdF"]
    _finite("fill value", c["fillF"])
    return c


def seed_rows(c, lF, lS, has_f, y, rows, ids, mcF=None, mcS=None):
    """the heads' inputs for the people in `rows` (boolean mask) of one seed; ids let combine_seeds check that every seed has the same
    people in the same order; mcF / mcS = (mean, SD) of the MC-dropout logits for everyone (NaN where not computed)"""
    n = len(lF)
    rows, has_f = _mask(rows, n, "rows"), _mask(has_f, n, "has_f")
    if len(lS) != n or len(y) != n or len(ids) != n:
        raise ValueError("lF, lS, y and ids must have one entry per person")
    hf = has_f[rows]
    zF = np.where(hf, (lF[rows] - c["muF"]) / c["sdF"], 0.0)
    r = dict(y=_labels(np.asarray(y)[rows]), has_f=hf, lF=np.where(hf, np.asarray(lF[rows], float), c["fillF_raw"]),
             lS=np.asarray(lS[rows], float), zF=zF, zF_fill=np.where(hf, zF, c["fillF"]), zS=(np.asarray(lS[rows], float) - c["muS"]) / c["sdS"])
    if mcF is not None:
        r["mF"] = np.where(hf, (mcF[0][rows] - c["muF"]) / c["sdF"], 0.0)
        r["sF"] = np.where(hf, mcF[1][rows] / c["sdF"], 1.0)
        r["mS"] = (mcS[0][rows] - c["muS"]) / c["sdS"]
        r["sS"] = mcS[1][rows] / c["sdS"]
        if np.any(r["sF"] <= 0) or np.any(r["sS"] <= 0):
            raise ValueError("an MC-dropout SD is 0: BARY needs SD > 0")
    _finite("head inputs", *r.values())
    r["ids"] = np.asarray(ids)[rows]
    return r


def pool(rows_list):
    """concatenate the per-seed row dicts into one fitting set"""
    keys = rows_list[0].keys()
    return {k: np.concatenate([r[k] for r in rows_list]) for k in keys}


# ---------- MoX ----------
def mox_offset(a, z, ctrl):
    """c = -log mean over the controls `ctrl` of exp(softplus(a) z); then exp(softplus(a) z + c) averages exactly 1 over them"""
    v = softplus(a) * z[ctrl]
    return float(np.log(v.size) - logsumexp(v))


def mox_terms(p, cF, cS, zF, zS, has_f):
    """log of the 4 pattern terms: log pi_k + evidence of the modalities in pattern k (eF = 0 for people without fMRI)"""
    logpi = p[:4] - logsumexp(p[:4])
    eF = np.where(has_f, softplus(p[4]) * zF + cF, 0.0)
    eS = softplus(p[5]) * zS + cS
    return np.stack([np.full(np.shape(zS), logpi[0]), logpi[1] + eF, logpi[2] + eS, logpi[3] + eF + eS])


def mox_f(p, cF, cS, zF, zS, has_f):
    return p[6] + logsumexp(mox_terms(p, cF, cS, zF, zS, has_f), axis=0)


def mox_posterior(p, cF, cS, zF, zS, has_f):
    """per person: share of each pattern in the fused evidence (rows sum to 1)"""
    t = mox_terms(p, cF, cS, zF, zS, has_f)
    return np.exp(t - logsumexp(t, axis=0)).T


def mox_penalty(p):
    """ridge on theta plus the scale penalty, pulling each scale softplus(a_m) towards 1"""
    return H["mox_theta_ridge"] * float(np.sum(p[:4] ** 2)) + H["mox_scale_ridge"] * float(np.sum((softplus(p[4:6]) - 1.0) ** 2))


def mox_objective(p, zF, zS, y, has_f):
    """summed BCE + penalty, with the offsets computed from the controls (those with fMRI for c_F)"""
    ctrl = y == 0
    cF, cS = mox_offset(p[4], zF, ctrl & has_f), mox_offset(p[5], zS, ctrl)
    return bce(mox_f(p, cF, cS, zF, zS, has_f), y) + mox_penalty(p)


def mox_ok(p, dev_F, dev_S, grad_per_row):
    lim = H["fit_checks"]
    return bool(np.all(np.isfinite(p)) and dev_F < lim["control_mean_exp_e_minus_1"] and dev_S < lim["control_mean_exp_e_minus_1"]
                and grad_per_row < lim["max_abs_gradient_per_row"])


def fit_mox(zF, zS, y, has_f, options=None):
    y, has_f = _labels(y), _mask(has_f, len(zF), "has_f")
    ctrl, ctrl_f = y == 0, (y == 0) & has_f
    fits = [minimize(mox_objective, x0, args=(zF, zS, y, has_f), method="L-BFGS-B", options=options or OPTIONS) for x0 in STARTS]
    k = int(np.argmin([res.fun for res in fits]))
    best = fits[k]
    p = np.array(best.x, float)
    cF, cS = mox_offset(p[4], zF, ctrl_f), mox_offset(p[5], zS, ctrl)
    applied = bce(mox_f(p, cF, cS, zF, zS, has_f), y) + mox_penalty(p)
    if np.isfinite(best.fun) and not abs(applied - best.fun) <= H["self_test"]["rel_tol"] * max(1.0, abs(best.fun)):
        raise RuntimeError(f"code error: MoX objective with the stored offsets {applied:.12g} differs from the optimiser's {best.fun:.12g}")
    dev_F = abs(float(np.mean(np.exp(softplus(p[4]) * zF[ctrl_f] + cF))) - 1.0)
    dev_S = abs(float(np.mean(np.exp(softplus(p[5]) * zS[ctrl] + cS))) - 1.0)
    grad = float(np.max(np.abs(best.jac))) / len(y)
    return dict(p=p, cF=cF, cS=cS, pi=np.exp(p[:4] - logsumexp(p[:4])), scale_F=float(softplus(p[4])), scale_S=float(softplus(p[5])),
                dev_F=dev_F, dev_S=dev_S, grad_per_row=grad, message=_message(best), objective=float(best.fun), best_start=k,
                start_objectives=[float(res.fun) for res in fits], n=int(len(y)), n_controls=int(ctrl.sum()), n_controls_fmri=int(ctrl_f.sum()),
                ok=mox_ok(p, dev_F, dev_S, grad))


# ---------- ADD and BARY ----------
def fit_add(zF_fill, zS, y, options=None):
    y = _labels(y)
    res = minimize(lambda w: bce(w[0] + w[1] * zF_fill + w[2] * zS, y) + H["add_ridge"] * float(w[1] ** 2 + w[2] ** 2), np.zeros(3),
                   method="L-BFGS-B", options=options or OPTIONS)
    return dict(w=np.array(res.x, float), grad_per_row=float(np.max(np.abs(res.jac))) / len(y), message=_message(res),
                objective=float(res.fun), n=int(len(y)))


def bary_lambda(q, sF, sS, has_f):
    return np.where(has_f, expit(q[0] + q[1] * (np.log(sF) - np.log(sS))), 1.0)


def bary_score(q, mF, sF, mS, sS, has_f):
    lam = bary_lambda(q, sF, sS, has_f)
    return lam * mS + (1.0 - lam) * np.where(has_f, mF, 0.0)


def fit_bary(mF, sF, mS, sS, y, has_f, options=None):
    y, has_f = _labels(y), _mask(has_f, len(mF), "has_f")
    res = minimize(lambda q: bce(bary_score(q, mF, sF, mS, sS, has_f), y) + H["bary_ridge"] * float(np.sum(q ** 2)), np.zeros(2),
                   method="L-BFGS-B", options=options or OPTIONS)
    lam = bary_lambda(res.x, sF, sS, has_f)
    return dict(q=np.array(res.x, float), grad_per_row=float(np.max(np.abs(res.jac))) / len(y), message=_message(res),
                objective=float(res.fun), lambda_S_mean_fmri=float(lam[has_f].mean()), n=int(len(y)))


# ---------- one fold: fit on the pooled rows, apply to every seed's test rows, average over seeds ----------
def fit_heads(val_rows_by_seed):
    r = pool(val_rows_by_seed)
    return dict(MOX_FROZEN=fit_mox(r["zF"], r["zS"], r["y"], r["has_f"]), ADD=fit_add(r["zF_fill"], r["zS"], r["y"]),
                BARY=fit_bary(r["mF"], r["sF"], r["mS"], r["sS"], r["y"], r["has_f"]), n_rows=int(len(r["y"])),
                n_autistic=int(r["y"].sum()), n_without_fmri=int((~r["has_f"]).sum()))


def assert_fit(heads):
    """raise if the MoX fit checks fail"""
    m = heads["MOX_FROZEN"]
    if not m["ok"]:
        raise RuntimeError(f"MoX fit check failed: control mean exp(e) - 1 = {m['dev_F']:.1e} / {m['dev_S']:.1e}, max |gradient| per row "
                           f"{m['grad_per_row']:.1e} (limits {H['fit_checks']}), optimiser: {m['message']}")


def head_scores(heads, r):
    """log-odds of every arm for one seed's rows, and the MoX pattern shares"""
    m, w, q = heads["MOX_FROZEN"], heads["ADD"]["w"], heads["BARY"]["q"]
    s = dict(F_BRANCH=r["lF"], S_BRANCH=r["lS"], EQUAL=(r["zF_fill"] + r["zS"]) / 2.0, ADD=w[0] + w[1] * r["zF_fill"] + w[2] * r["zS"],
             MOX_FROZEN=mox_f(m["p"], m["cF"], m["cS"], r["zF"], r["zS"], r["has_f"]),
             BARY=bary_score(q, r["mF"], r["sF"], r["mS"], r["sS"], r["has_f"]))
    return s, mox_posterior(m["p"], m["cF"], m["cS"], r["zF"], r["zS"], r["has_f"])


def combine_seeds(heads, test_rows_by_seed):
    """each arm's per-seed log-odds averaged over the seeds (the same people in the same order in every seed); refuses failed MoX fits"""
    assert_fit(heads)
    first = test_rows_by_seed[0]
    for r in test_rows_by_seed[1:]:
        if not (np.array_equal(r["ids"], first["ids"]) and np.array_equal(r["y"], first["y"]) and np.array_equal(r["has_f"], first["has_f"])):
            raise ValueError("the seeds' test rows are not the same people in the same order")
    per = [head_scores(heads, r) for r in test_rows_by_seed]
    scores = {a: np.mean([s[a] for s, _ in per], axis=0) for a in ARMS}
    for a, v in scores.items():
        _finite(f"scores of {a}", v)
    return scores, np.mean([p for _, p in per], axis=0)


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


def self_test(zF, zS, y, has_f, fit, heads=None, test_rows=None):
    """invariance check: after a shift of the scores, with the offsets recomputed, the objective, the fused log-odds and (given heads and
    test rows) the applied MoX scores stay the same at the fitted p, and the offsets move by -scale x shift; the refit must pass the checks"""
    dF, dS = H["self_test"]["shift"]
    tol = H["self_test"]["rel_tol"]
    y = _labels(y)
    p = fit["p"]
    o0, o1 = mox_objective(p, zF, zS, y, has_f), mox_objective(p, zF + dF, zS + dS, y, has_f)
    ctrl = y == 0
    f0 = mox_f(p, mox_offset(p[4], zF, ctrl & has_f), mox_offset(p[5], zS, ctrl), zF, zS, has_f)
    cF1, cS1 = mox_offset(p[4], zF + dF, ctrl & has_f), mox_offset(p[5], zS + dS, ctrl)
    obj_dev, f_dev = abs(o1 - o0) / max(1.0, abs(o0)), _rel(mox_f(p, cF1, cS1, zF + dF, zS + dS, has_f), f0)
    offs_dev = (max(abs(cF1 - (fit["cF"] - fit["scale_F"] * dF)), abs(cS1 - (fit["cS"] - fit["scale_S"] * dS)))
                / max(1.0, abs(fit["cF"]), abs(fit["cS"])))
    app_dev = None
    if heads is not None:
        if heads["MOX_FROZEN"] is not fit:
            raise ValueError("self_test: fit must be heads['MOX_FROZEN']")
        moved = [dict(r, zF=r["zF"] + dF, zS=r["zS"] + dS) for r in test_rows]
        app_dev = _rel(combine_seeds(dict(heads, MOX_FROZEN=dict(fit, cF=cF1, cS=cS1)), moved)[0]["MOX_FROZEN"],
                       combine_seeds(heads, test_rows)[0]["MOX_FROZEN"])
    refit = fit_mox(zF + dF, zS + dS, y, has_f)
    return dict(shift=[dF, dS], objective_rel_change=obj_dev, fused_rel_change=f_dev, offset_change_error=float(offs_dev), cF_shifted=cF1, cS_shifted=cS1,
                applied_rel_change=app_dev, ok=bool(obj_dev < tol and f_dev < tol and offs_dev < tol and (app_dev is None or app_dev < tol) and refit["ok"]),
                refit_pi_move=float(np.max(np.abs(refit["pi"] - fit["pi"]))), refit_objective=refit["objective"],
                refit_objective_original_scores=mox_objective(refit["p"], zF, zS, y, has_f), objective=fit["objective"], refit_ok=refit["ok"])


def _clean(v):
    if isinstance(v, (np.ndarray, list, tuple)):
        return [_clean(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    return int(v) if isinstance(v, np.integer) else bool(v) if isinstance(v, np.bool_) else v


def summary(heads):
    """JSON-ready copy of the fitted heads (numbers only; non-finite numbers become null)"""
    return _clean(heads)
