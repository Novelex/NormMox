"""Site calibration of the chart residuals (Eq. 1): per site, mean and SD of the residuals of the training controls; the training
controls themselves are scored by 10-fold cross-fitting within their site. In leave-one-site-out the held-out site is calibrated by
its own controls, cross-fitted with the folds of assignments.csv.
The functions read the module variables n, sitenum, y_all, site_fold (set by data.Data) and scheme, rep (set per fold)."""
import zlib

import numpy as np

INNER_K, INNER_SEED = 10, 20261016
n = sitenum = y_all = site_fold = scheme = rep = None


def site_score(R, is_adapter, adapt_fold):
    """z for everyone from R (people x features), rulers per site from the adapters: a person with adapt_fold = j is scored with the
    adapters of the same site outside fold j; a person with adapt_fold = -1 (not an adapter) with all adapters of the site.
    Blanks never enter a ruler; a ruler needs >= 2 values. Returns z, and the adaptation count per person."""
    Z = np.full(R.shape, np.nan)
    K = np.zeros(n, dtype=int)
    for g in np.unique(sitenum):
        in_site = sitenum == g
        for j in np.unique(adapt_fold[in_site]):
            scored = in_site & (adapt_fold == j)
            adapters = in_site & is_adapter & (adapt_fold != j)
            A = R[adapters]
            if A.shape[0] == 0:
                continue
            cnt = np.isfinite(A).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.nanmean(A, axis=0)
                sd = np.nanstd(A, axis=0)                    # numpy std, ddof 0
                z = (R[scored] - mu) / sd
            z[:, cnt < 2] = np.nan
            Z[scored] = z
            K[scored] = int(np.median(cnt)) if cnt.size else 0
    return Z, K


def ruler_setup(train, test, fold_now=0):
    """who adapts whom in this fold: adapters = training controls (inner cross-fit); in leave-one-site-out also the test site's own
    controls, with everyone of the test site keeping their fold from assignments.csv."""
    is_adapter = np.zeros(n, bool)
    adapt_fold = np.full(n, -1, dtype=int)
    tr_ctrl = np.where(train & (y_all == 0))[0]
    rng = np.random.default_rng([INNER_SEED, zlib.crc32(scheme.encode()), rep, int(fold_now)])   # deterministic per scheme, repeat, fold
    for g in np.unique(sitenum[tr_ctrl]):
        idx = tr_ctrl[sitenum[tr_ctrl] == g]
        idx = idx[rng.permutation(len(idx))]
        adapt_fold[idx] = np.arange(len(idx)) % INNER_K
    is_adapter[tr_ctrl] = True
    if scheme == "loso":
        test_sites = np.unique(sitenum[test])
        own = np.isin(sitenum, test_sites) & test
        is_adapter[own & (y_all == 0)] = True
        adapt_fold[own] = site_fold[own]
    return is_adapter, adapt_fold
