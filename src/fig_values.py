"""Figure 2 and 3a values. One quantity throughout, the map of Table 2: GA = mean Integrated-Gradients contribution to the fused
log-odds over the autistic test people (outputs/isbi/per_fold.npz from fig_maps.py; 100 test folds).
  drawn      an input (fMRI pair) or a structural measure (z + |z| channels summed) is drawn when |mean GA| >= 2 x its SD over the 100
             folds; the SD of a structural measure is the SD of its per-fold net value
  direction  of the drawn inputs and measures, the 8 per modality whose contribution differs most between the groups (|autistic minus
             control mean contribution|, DIFF); per person, the in-fold chart z of each test fold averaged over the 10 repeats; per group
             mean, SD, n and 95 % CI over participants (mean +- 1.96 SD / sqrt(n)); open circle = the autistic 95 % CI includes 0
  stability  Table 2's fold stability and top-10 rows on the autistic maps (GA) and on the autistic-minus-control maps (DIFF)
Writes outputs/isbi/fig2_values.csv (one row per drawn input or measure, and per direction feature) and fig2_values_log.txt.
usage: python src/fig_values.py"""
import os

import numpy as np
import pandas as pd

from data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/outputs/isbi"
REL, TOP = 2.0, 8
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def fold_stability(M, cols):
    """Table 2 definitions: mean Spearman rho over all pairs of folds, top-10 overlap with the overall top-10, sign consistency of the
    overall top-10"""
    g = M[:, cols]
    rk = np.argsort(np.argsort(g, axis=1), axis=1).astype(float)
    C = np.corrcoef(rk)
    rho = C[np.triu_indices(len(g), 1)].mean()
    m = g.mean(0)
    top = np.argsort(-np.abs(m))[:10]
    ovl = np.mean([len(set(top) & set(np.argsort(-np.abs(x))[:10])) / 10 for x in g])
    same = (np.sign(g) == np.sign(m)).mean(0)[top].mean()
    return rho, ovl, same


def group_stats(v, mask):
    x = v[mask]
    x = x[np.isfinite(x)]
    n, mu, sd = len(x), float(x.mean()), float(x.std(ddof=1))
    h = 1.96 * sd / np.sqrt(n)
    return n, mu, sd, mu - h, mu + h, float(np.abs(x).mean())


def main():
    P = np.load(f"{OUT}/per_fold.npz", allow_pickle=True)
    GA, DIFF = P["GA"], P["DIFF"]
    fcols, scols = [str(c) for c in P["func_cols"]], [str(c) for c in P["struct_cols"]]
    nF, nS = len(fcols), len(scols)
    say(f"Figure 2 values; GA = mean contribution in the autistic test people (log-odds); {len(GA)} folds, {nF} fMRI pairs, {nS} structural measures")
    # ---------------- Table 2 alignment: the stability rows on the autistic maps (Table 2) and on the DIFF maps (this figure)
    groups = {"fMRI": np.arange(nF), "sMRI": np.arange(nF, nF + 2 * nS), "all": np.arange(nF + 2 * nS)}
    say("\nstability (Table 2 rows): map | group | rho | top-10 overlap | same sign")
    for name, M in (("autistic (GA, Table 2)", GA), ("autistic - control (DIFF)", DIFF)):
        for g, cols in groups.items():
            rho, ovl, same = fold_stability(M, cols)
            say(f"  {name:26s} {g:5s} rho {rho:.3f}  overlap {ovl:.3f}  same sign {same:.3f}")
    # ---------------- drawn elements
    rows = []
    mF, sF = GA[:, :nF].mean(0), GA[:, :nF].std(0, ddof=1)
    for j, c in enumerate(fcols):
        sign = float((np.sign(GA[:, j]) == np.sign(mF[j])).mean())
        rows.append(dict(panel="a", feature=c, kind="fMRI pair", value=mF[j], sd_folds=sF[j], rel=abs(mF[j]) / sF[j], sign_consistency=sign,
                         value_z=np.nan, value_absz=np.nan, value_diff=DIFF[:, j].mean()))
    net = GA[:, nF:nF + nS] + GA[:, nF + nS:]
    mN, sN = net.mean(0), net.std(0, ddof=1)
    for j, c in enumerate(scols):
        kind = "thickness" if c.endswith("_thickness") else "volume"
        sign = float((np.sign(net[:, j]) == np.sign(mN[j])).mean())
        rows.append(dict(panel="b", feature=c, kind=kind, value=mN[j], sd_folds=sN[j], rel=abs(mN[j]) / sN[j], sign_consistency=sign,
                         value_z=GA[:, nF + j].mean(), value_absz=GA[:, nF + nS + j].mean(),
                         value_diff=(DIFF[:, nF + j] + DIFF[:, nF + nS + j]).mean()))
    V = pd.DataFrame(rows)
    V["drawn"] = V["rel"] >= REL
    V["rank_in_modality"] = V.groupby("panel")["value"].transform(lambda s: s.abs().rank(ascending=False, method="first")).astype(int)
    cort = V[(V.panel == "b") & (V.kind == "thickness") & ~V.feature.str.contains("MeanThickness")]
    vol = V[(V.panel == "b") & (V.kind == "volume")]
    drawn_f, drawn_c, drawn_v = V[(V.panel == "a") & V.drawn], cort[cort.drawn], vol[vol.drawn]
    say(f"\n(a) drawn fMRI pairs: {len(drawn_f)} of {nF} ({int((drawn_f.value > 0).sum())} towards autism); |value| "
        f"{drawn_f.value.abs().min():.4f} - {drawn_f.value.abs().max():.4f}")
    say(f"(b) drawn cortical regions: {len(drawn_c)} of {len(cort)} ({int((drawn_c.value > 0).sum())} towards autism); |value| "
        f"{drawn_c.value.abs().min():.4f} - {drawn_c.value.abs().max():.4f}")
    say(f"(b) drawn volumes: {len(drawn_v)} of {len(vol)}: " + ", ".join(f"{r.feature} {r.value:+.4f}" for r in drawn_v.sort_values('value', key=abs, ascending=False).itertuples()))
    allv = pd.concat([drawn_f, drawn_c, drawn_v]).value.abs()
    say(f"shared colour scale: largest |value| {allv.max():.4f}, smallest drawn |value| {allv.min():.4f}")
    nets = set()
    for c in drawn_f.feature:
        nets |= set(c.split("_", 1))
    say(f"chart networks with a drawn pair: {len(nets)}; without: {sorted(set(x for c in fcols for x in c.split('_', 1)) - nets)}")
    # ---------------- direction: top-8 per modality by |DIFF| (drawn ones only), per-person z
    D = Data()
    if [str(c) for c in D.func_cols] != fcols or [str(c) for c in D.struct_cols] != scols:
        raise RuntimeError("column order differs between per_fold.npz and Data")
    sumF, nFz = np.zeros((D.n, nF)), np.zeros((D.n, nF))
    sumS, nSz = np.zeros((D.n, nS)), np.zeros((D.n, nS))
    folds = D.fold_list()
    if len(folds) != 100:
        raise RuntimeError(f"{len(folds)} folds")
    for r, k in folds:
        zi = D.fold_z(r, k)
        t = zi["test"]
        for Z, s_, n_ in ((zi["zf"], sumF, nFz), (zi["zs"], sumS, nSz)):
            ok = np.isfinite(Z) & t[:, None]
            s_[ok] += Z[ok]
            n_[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        ZF, ZS = sumF / nFz, sumS / nSz
    aut, ctl = D.y == 1, D.y == 0
    say(f"\nper-person z: every person tested in {int(np.nanmax(nSz))} folds (min {int(np.nanmin(nSz.max(1)))}); "
        f"fMRI available for {int(np.isfinite(ZF).any(1).sum())} of {D.n}")
    say(f"drawn volumes, all: " + ", ".join(f"{r.feature} {r.value:+.4f} (rel {r.rel:.1f}, DIFF {r.value_diff:+.4f})"
                                             for r in drawn_v.sort_values('value', key=abs, ascending=False).itertuples()))
    top_f = V[(V.panel == "a") & V.drawn].sort_values("value_diff", key=abs, ascending=False).head(TOP)
    top_s = V[(V.panel == "b") & V.drawn & ~V.feature.str.contains("MeanThickness")].sort_values("value_diff", key=abs, ascending=False).head(TOP)
    out = []
    say(f"\n(d) top-{TOP} per modality by |DIFF| among the drawn; z: mean [95 % CI] autistic | control; SD ratio; open = autistic CI includes 0")
    for blk, T, Z, cols in (("fMRI", top_f, ZF, fcols), ("sMRI", top_s, ZS, scols)):
        for rank, rr in enumerate(T.itertuples(), 1):
            j = cols.index(rr.feature)
            na, ma, sa, la, ha, aa = group_stats(Z[:, j], aut)
            nc, mc, sc, lc, hc, ac = group_stats(Z[:, j], ctl)
            open_ = bool(la <= 0 <= ha)
            out.append(dict(panel="d", block=blk, rank=rank, feature=rr.feature, kind=rr.kind, value=rr.value, value_diff=rr.value_diff,
                            sign_consistency=rr.sign_consistency, value_z=rr.value_z, value_absz=rr.value_absz, open_circle=open_, n_aut=na,
                            z_aut=ma, z_aut_lo=la, z_aut_hi=ha, sd_aut=sa, absz_aut=aa, n_ctl=nc, z_ctl=mc, z_ctl_lo=lc, z_ctl_hi=hc, sd_ctl=sc,
                            absz_ctl=ac, sd_ratio=sa / sc))
            say(f"  {blk} {rank} {rr.feature:38s} GA {rr.value:+.4f} (z {rr.value_z:+.4f}, |z| {rr.value_absz:+.4f}; DIFF {rr.value_diff:+.4f}) "
                f"sign {rr.sign_consistency:.2f} | aut {ma:+.3f} [{la:+.3f}, {ha:+.3f}] n {na} | ctl {mc:+.3f} [{lc:+.3f}, {hc:+.3f}] n {nc} | "
                f"SD ratio {sa / sc:.2f} | mean|z| {aa:.3f} vs {ac:.3f}" + ("  OPEN" if open_ else ""))
    res = pd.concat([V[V.drawn].assign(display_rank=lambda d: d.rank_in_modality), pd.DataFrame(out)], ignore_index=True)
    res.to_csv(f"{OUT}/fig2_values.csv", index=False)
    open(f"{OUT}/fig2_values_log.txt", "w").write("\n".join(LINES) + "\n")
    say(f"\nwritten {OUT}/fig2_values.csv ({len(res)} rows)")


if __name__ == "__main__":
    main()
