"""Evaluation metrics and statistical tests."""
import numpy as np
from scipy.stats import t as tdist
from sklearn.metrics import average_precision_score, roc_auc_score

METRICS = ("auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "f1")


def metrics(y, score, threshold):
    """Metrics for one test fold. threshold: 0 for log-odds, 0.5 for probabilities."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    pred = (score > threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return dict(auroc=roc_auc_score(y, score), auprc=average_precision_score(y, score),
                balanced_accuracy=(sens + spec) / 2, sensitivity=sens, specificity=spec,
                f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def summarise(per_fold):
    """Mean over the folds of each repeat, then mean and SD over the repeats."""
    rep = per_fold.groupby("repeat")[list(METRICS)].mean()
    return rep.mean(), rep.std(ddof=1)


def corrected_t(diff, ratio=1 / 9):
    """Corrected resampled t-test (Nadeau and Bengio) on per-fold differences; ratio = n_test / n_train."""
    diff = np.asarray(diff, float)
    j, m, v = len(diff), float(diff.mean()), float(diff.var(ddof=1))
    se = float(np.sqrt((1 / j + ratio) * v))
    h = float(tdist.ppf(0.975, j - 1) * se)
    return dict(diff=m, ci_low=m - h, ci_high=m + h, p=float(2 * tdist.sf(abs(m / se), j - 1)))


def holm(ps):
    """Holm-adjusted p-values."""
    ps = np.asarray(ps, float)
    order, out, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out
