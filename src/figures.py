"""Figure 2 panels (a)-(d): four separate panels of the same size (3.45 x 2.8 in), no subtitles (definitions in the caption).
(a) and (b) show the map of Table 2: the mean Integrated-Gradients contribution to the fused log-odds in the autistic test people, drawn
when its mean is at least 2 SD over the 100 folds.
  (a) fMRI: the consistent network pairs; arcs = network systems (Yeo colours), ribbon width and colour = contribution
  (b) sMRI: the consistent cortical regions (Destrieux, inflated fsaverage5, light curvature shading), on the colour scale of (a)
  (c) per person: share of autistic people and controls whose control-calibrated evidence exceeds the training controls' 95th
      percentile, by pattern; autistic / control ratio from the printed shares; Fisher's exact test per repeat, largest p of the 10
  (d) direction: of the inputs drawn in (a) and (b), the 8 per modality whose contribution differs most between the groups; mean in-fold
      chart z with 95 % CI over participants, autistic and controls; open circle = autistic CI includes 0 (no mean shift)
The fMRI columns carry the braincharts chart network names (label k -> k-th chart name); TO_OFFICIAL gives the official CBIG Yeo-17
name of the same label.
Values: outputs/isbi/fig2_values.csv (fig_values.py) and fig2_patterns.json (fig_stats.py).
Writes outputs/figures/Fig2{a,b,c,d}.{png (1200 dpi),pdf,svg} and Fig2_preview_2x2.png.
usage: python src/figures.py [a b c d]   (default: all four, then the preview)"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import matplotlib                                                                        # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                          # noqa: E402
import numpy as np                                                                       # noqa: E402
import pandas as pd                                                                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb, rgb_to_hsv            # noqa: E402
from matplotlib.lines import Line2D                                                      # noqa: E402
from matplotlib.patches import PathPatch, Rectangle, Wedge                               # noqa: E402
from matplotlib.path import Path                                                         # noqa: E402
from matplotlib.text import Text                                                         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/outputs"
OUT = f"{RES}/figures"
CHART = ["VisualA", "VisualB", "SensorimotorA", "SensorimotorB", "DorsalAttnA", "DorsalAttnB", "VentralAttnA", "VentralAttnB", "LimbicA",
         "LimbicB", "ControlA", "ControlB", "ControlC", "DefaultA", "DefaultB", "DefaultC", "TemporalParietal"]     # label k -> chart name (fMRI columns)
ORDER = ["VisualA", "VisualB", "SensorimotorA", "SensorimotorB", "DorsalAttnA", "DorsalAttnB", "VentralAttnA", "VentralAttnB", "LimbicA",
         "LimbicB", "ControlC", "ControlA", "ControlB", "TemporalParietal", "DefaultC", "DefaultA", "DefaultB"]     # label k -> OFFICIAL name
TO_OFFICIAL = dict(zip(CHART, ORDER))                                          # chart name -> official name of the same Yeo label
SYSTEM = {"VisualA": "Visual", "VisualB": "Visual", "SensorimotorA": "Somatomotor", "SensorimotorB": "Somatomotor", "DorsalAttnA": "Dorsal attention",
          "DorsalAttnB": "Dorsal attention", "VentralAttnA": "Ventral attention", "VentralAttnB": "Ventral attention", "LimbicA": "Limbic",
          "LimbicB": "Limbic", "ControlA": "Control", "ControlB": "Control", "ControlC": "Control", "DefaultA": "Default", "DefaultB": "Default",
          "DefaultC": "Default", "TemporalParietal": "Temporal-parietal"}
PW, PH = 3.45, 2.8                                           # every panel, inches
FS_HEAD, FS, FS_SMALL, FS_TICK = 7.4, 6.2, 5.8, 5.6          # Figure 1's sizes
INK, INK2, AXIS, GAP = "#111111", "#4d4d4d", "#9a9a9a", "#c8c8c8"
C_AUT, C_CTL = "#5e3c99", "#8c8c8c"                          # autistic = P(autism) purple of Figure 1; controls grey (>= 3:1 on white)
BASE = plt.get_cmap("RdBu_r")                                # blue = towards control, red = towards autism
SAT0 = 0.45                                                  # saturation of the weakest drawn element
SYSTEM_COLOUR = {"Visual": "#781286", "Somatomotor": "#4682B4", "Dorsal attention": "#00760E", "Ventral attention": "#C43AFA",
                 "Limbic": "#DCF8A4", "Control": "#E69422", "Default": "#CD3E4E", "Temporal-parietal": "#CD3E4E"}
# arcs in the Yeo 7-network colours; TempPar takes the colour of its 7-network parent (default), not blue (blue = towards control here)
CBAR = (0.925, 0.30, 1.60, 0.065)                            # colour bar of (a) and (b): x0, y0, width, height in inches
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "pdf.fonttype": 42, "svg.fonttype": "path",
                     "mathtext.fontset": "dejavusans", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.color": INK2, "ytick.color": INK2})


def new_panel(title):
    fig = plt.figure(figsize=(PW, PH), facecolor="white")
    fig.text(0.06 / PW, 1 - 0.05 / PH, title, fontsize=FS_HEAD, fontweight="bold", color=INK, ha="left", va="top")
    return fig


def inch_axes(fig, x0, y0, w, h, **kw):
    return fig.add_axes([x0 / PW, y0 / PH, w / PW, h / PH], **kw)


def check(fig):
    """text overlaps and text outside the canvas"""
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    items = [(t.get_text().replace("\n", "|")[:24], t.get_window_extent(rd)) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    out = [f"outside: {s}" for s, b in items if b.x0 < 0.5 or b.y0 < 0.5 or b.x1 > W - 0.5 or b.y1 > H - 0.5]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5:
                out.append(f"overlap: {items[i][0]} x {items[j][0]}")
    return out


def save(fig, name):
    issues = check(fig)
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in (("png", dict(dpi=1200)), ("pdf", {}), ("svg", {})):
        fig.savefig(f"{OUT}/{name}.{ext}", facecolor="white", **kw)
    print(f"written {OUT}/{name}.png | text issues: {len(issues)}" + "".join("\n   " + s for s in issues), flush=True)
    plt.close(fig)


# ------------------------------------------------------------------ values and the shared contribution scale of (a) and (b)
def load():
    V = pd.read_csv(f"{RES}/isbi/fig2_values.csv")
    fm = V[V["panel"] == "a"].copy()
    cort = V[(V["panel"] == "b") & (V["kind"] == "thickness") & ~V["feature"].str.contains("MeanThickness")].copy()
    d = V[V["panel"] == "d"].copy()
    if not (len(fm) == 28 and len(cort) == 50 and len(d) == 16):
        raise RuntimeError(f"unexpected counts: {len(fm)} pairs, {len(cort)} regions, {len(d)} direction rows")
    vals = pd.concat([fm["value"], cort["value"]]).abs()
    vmax = float(np.ceil(vals.max() * 100) / 100)
    return fm, cort, d, vmax, float(vals.min())


def colour(v, vmax, thr):
    """signed contribution -> colour: hue by sign, saturation from SAT0 (weakest drawn) to 1 (vmax)"""
    s = SAT0 + (1 - SAT0) * min(max((abs(v) - thr) / (vmax - thr), 0.0), 1.0)
    return BASE(0.5 + 0.5 * s) if v > 0 else BASE(0.5 - 0.5 * s)


def gap_cmap(vmax, thr, n=1024):
    """the scale of colour() over [-vmax, vmax], grey inside the gap below the weakest drawn element"""
    vals = np.linspace(-vmax, vmax, n)
    cols = [colour(v, vmax, thr) if abs(v) >= thr else matplotlib.colors.to_rgba(GAP) for v in vals]
    return LinearSegmentedColormap.from_list("contrib", cols, N=n)


def contribution_bar(fig, vmax, thr):
    x0, y0, w, h = CBAR
    cax = inch_axes(fig, x0, y0, w, h)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap=gap_cmap(vmax, thr)), cax=cax, orientation="horizontal")
    cb.set_ticks([-vmax, 0, vmax])
    cb.set_ticklabels([f"−{vmax:.2f}\ntowards control", "0", f"+{vmax:.2f}\ntowards autism"])
    cb.ax.tick_params(labelsize=FS_TICK, colors=INK2, length=2, pad=1, width=0.5)
    for t in cb.ax.get_xticklabels():
        t.set_linespacing(1.05)
    cb.outline.set_linewidth(0.4)
    fig.text((x0 + w / 2) / PW, (y0 + h + 0.035) / PH, "contribution in autistic participants (log-odds)", fontsize=FS_SMALL, color=INK2,
             ha="center", va="bottom")


SHORT = {"Visual": "Vis", "Somatomotor": "SomMot", "Dorsal attention": "DorsAttn", "Ventral attention": "SalVent", "Limbic": "Limbic",
         "Control": "Cont", "Default": "Default", "Temporal-parietal": "TempPar"}


def short(official):
    """standard short Yeo-17 name of an official network name (ORDER entry): 'ControlC' -> 'Cont C', 'TemporalParietal' -> 'TempPar'"""
    s = SHORT[SYSTEM[official]]
    return s if official == "TemporalParietal" else f"{s} {official[-1]}"


def pair_name(chart_col):
    """chart column 'LimbicA_ControlA' -> 'Limbic A – Cont C' (official names, lower Yeo label first)"""
    a, b = (TO_OFFICIAL[x] for x in chart_col.split("_", 1))
    a, b = sorted((a, b), key=ORDER.index)
    return f"{short(a)} – {short(b)}"


# ------------------------------------------------------------------ (a) fMRI: the consistent network pairs
def panel_a(fm, vmax, thr):
    fig = new_panel("(a) fMRI: network pairs")
    ax = inch_axes(fig, 0.0, 0.55, PW, 2.10)
    order, gap, R, w = ORDER, 2.5, 1.0, 0.085
    n = len(order)
    span = (360 - n * gap) / n
    ang = {}
    for i, net in enumerate(order):
        a0 = 90 - i * (span + gap)
        ang[net] = (a0 - span, a0)
        ax.add_patch(Wedge((0, 0), R, a0 - span, a0, width=w, facecolor=SYSTEM_COLOUR[SYSTEM[net]], edgecolor="white", linewidth=0.4))
        mid = np.deg2rad(a0 - span / 2)
        rot, ha = np.rad2deg(mid), "left"
        if np.cos(mid) < 0:
            rot, ha = rot + 180, "right"
        ax.text(1.06 * np.cos(mid), 1.06 * np.sin(mid), short(net), rotation=rot, rotation_mode="anchor", ha=ha, va="center",
                fontsize=FS_SMALL, color=INK)
    r0 = R - w
    vmax_w = float(fm["value"].abs().max())
    for _, row in fm.sort_values("value", key=np.abs).iterrows():          # weakest first, strongest on top
        a, b = (TO_OFFICIAL[x] for x in row["feature"].split("_", 1))
        v = row["value"]
        hw = 0.03 + 0.22 * abs(v) / vmax_w
        pts = []
        for net in (a, b):
            lo, hi = np.deg2rad(ang[net][0]), np.deg2rad(ang[net][1])
            c = (lo + hi) / 2
            pts.append((c - hw * (hi - lo), c + hw * (hi - lo)))
        (a1, a2), (b1, b2) = pts
        P = lambda t: (r0 * np.cos(t), r0 * np.sin(t))  # noqa: E731
        verts = [P(a1), (0, 0), P(b2), P(b1), (0, 0), P(a2), P(a1)]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=colour(v, vmax, thr), edgecolor="white", linewidth=0.2, alpha=0.95))
    half_h = 1.76
    half_w = half_h * PW / 2.10
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(-half_h, half_h)
    ax.set_aspect("equal")
    ax.axis("off")
    contribution_bar(fig, vmax, thr)
    save(fig, "Fig2a")
    print(f"  (a) {len(fm)} of 136 pairs drawn ({int((fm['value'] > 0).sum())} towards autism); scale +-{vmax:.2f}, weakest drawn {thr:.4f}")


# ------------------------------------------------------------------ (b) sMRI: the consistent cortical regions
def soften(im, lo=0.50, span=0.45):
    """lighter curvature shading: grey pixels (saturation < 0.10) mapped to value lo + span * value; coloured pixels unchanged"""
    out = im.copy()
    hsv = rgb_to_hsv(np.clip(im[..., :3], 0, 1))
    grey = hsv[..., 1] < 0.10
    hsv[..., 2] = np.where(grey, lo + span * hsv[..., 2], hsv[..., 2])
    out[..., :3] = hsv_to_rgb(hsv)
    return out


def view_png(fs, vals, hemi, view, path, cmap, vmax, thr):
    from nilearn import plotting
    f = plt.figure(figsize=(3.0, 2.25), facecolor="none")
    a = f.add_axes([0, 0, 1, 1], projection="3d")
    plotting.plot_surf_stat_map(fs[f"infl_{hemi}"], vals, hemi=hemi, view=view, bg_map=fs[f"sulc_{hemi}"], cmap=cmap, vmax=vmax,
                                symmetric_cbar=True, threshold=thr * 0.999, colorbar=False, axes=a, figure=f)
    a.set_title("", pad=0)
    f.savefig(path, dpi=600, transparent=True)
    plt.close(f)
    im = plt.imread(path)
    keep = (im[..., 3] > 0.05) & (im[..., :3].min(-1) < 0.97)
    r, c = np.where(keep.any(1))[0], np.where(keep.any(0))[0]
    return soften(im[r.min(): r.max() + 1, c.min(): c.max() + 1])


def panel_b(cort, vmax, thr):
    from nilearn import datasets
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    de = datasets.fetch_atlas_surf_destrieux()
    labels = [lab.decode() if isinstance(lab, bytes) else lab for lab in de["labels"]]
    cmap = gap_cmap(vmax, thr)
    tmp = f"{OUT}/views"
    os.makedirs(tmp, exist_ok=True)
    ims = {}
    for hemi, key, pre in (("left", "map_left", "lh_"), ("right", "map_right", "rh_")):
        ann = np.asarray(de[key])
        vals = np.zeros(len(ann))
        for r, v in zip(cort["feature"], cort["value"]):
            if r.startswith(pre):
                name = r[3:-len("_thickness")].replace("&", "_and_")
                if name not in labels:
                    raise RuntimeError(f"{r} not in the Destrieux atlas")
                vals[ann == labels.index(name)] = v
        for view in ("lateral", "medial"):
            ims[(hemi, view)] = view_png(fs, vals, hemi, view, f"{tmp}/{hemi}_{view}.png", cmap, vmax, thr)
    fig = new_panel("(b) sMRI: cortical thickness")
    cells = {("left", "lateral"): (0.20, 1.55), ("right", "lateral"): (1.84, 1.55), ("left", "medial"): (0.20, 0.66),
             ("right", "medial"): (1.84, 0.66)}
    cw, ch = 1.56, 0.86
    for k, (x0, y0) in cells.items():
        im = ims[k]
        s = min(cw / im.shape[1], ch / im.shape[0])
        w, h = im.shape[1] * s, im.shape[0] * s
        a = inch_axes(fig, x0 + (cw - w) / 2, y0 + (ch - h) / 2, w, h)
        a.imshow(im)
        a.axis("off")
    for x, t in ((0.20 + cw / 2, "left"), (1.84 + cw / 2, "right")):
        fig.text(x / PW, 2.52 / PH, t, fontsize=FS, color=INK2, ha="center", va="center")
    for y, t in ((1.55 + ch / 2, "lateral"), (0.66 + ch / 2, "medial")):
        fig.text(0.09 / PW, y / PH, t, fontsize=FS_SMALL, color=INK2, ha="center", va="center", rotation=90)
    contribution_bar(fig, vmax, thr)
    save(fig, "Fig2b")
    print(f"  (b) {len(cort)} of 148 regions drawn ({int((cort['value'] > 0).sum())} towards autism); |value| {cort['value'].abs().min():.4f}-"
          f"{cort['value'].abs().max():.4f}")


# ------------------------------------------------------------------ (c) per person: atypical evidence by pattern
def p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def panel_c():
    S = json.load(open(f"{RES}/isbi/fig2_patterns.json"))
    rows = (("fMRI only", "F_only"), ("sMRI only", "S_only"), ("both", "both"))
    fig = new_panel("(c) Atypical evidence per person")
    ax = inch_axes(fig, 0.62, 0.40, 2.73, 2.02)
    bw, off = 0.34, 0.18
    for x, (lab, key) in enumerate(rows):
        va, vc = 100 * S[f"aut_{key}"], 100 * S[f"ctrl_{key}"]
        ax.bar(x - off, va, width=bw, color=C_AUT, edgecolor="none", zorder=2)
        ax.bar(x + off, vc, width=bw, color=C_CTL, edgecolor="none", zorder=2)
        ax.text(x - off, va + 0.35, f"{va:.1f}", ha="center", va="bottom", fontsize=FS_TICK, color=INK)
        ax.text(x + off, vc + 0.35, f"{vc:.1f}", ha="center", va="bottom", fontsize=FS_TICK, color=INK)
        top = va + 2.0
        ax.plot([x - off, x - off, x + off, x + off], [top - 0.45, top, top, top - 0.45], color=INK2, linewidth=0.5, zorder=3)
        ratio = f"×{round(va, 1) / round(vc, 1):.1f}" if vc >= 1.0 else "rare in controls"    # from the printed shares
        ax.text(x, top + 0.35, f"{ratio}\n{p_text(S[f'p_{key}_max'])}", ha="center", va="bottom", fontsize=FS_SMALL, color=INK,
                linespacing=1.1)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 20)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=FS, color=INK)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(axis="y", labelsize=FS_TICK, colors=INK2, length=2, pad=1)
    ax.set_ylabel("participants with evidence above\nthe controls' 95th percentile (%)", fontsize=FS_SMALL, color=INK2, labelpad=3,
                  linespacing=1.1)
    ax.text(1.0, 0.835, "× = autistic ÷ controls", transform=ax.transAxes, ha="right", va="top", fontsize=FS_SMALL, color=INK2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(AXIS)
    hs = [Rectangle((0, 0), 1, 1, color=C_AUT), Rectangle((0, 0), 1, 1, color=C_CTL)]
    ax.legend(hs, [f"autistic (n = {S['n_aut']})", f"controls (n = {S['n_ctrl']})"], loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False,
              fontsize=FS_SMALL, handlelength=1.0, handleheight=0.8, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.1)
    save(fig, "Fig2c")
    print("  (c) " + "; ".join(f"{lab}: {100 * S['aut_' + key]:.1f}% vs {100 * S['ctrl_' + key]:.1f}%, largest p {S['p_' + key + '_max']:.1e}"
                               for lab, key in rows))


# ------------------------------------------------------------------ (d) direction of the top drivers
NAMES = {"3rd-Ventricle": "3rd ventricle", "4th-Ventricle": "4th ventricle", "rh_Pole_occipital_thickness": "R occipital pole",
         "lh_Pole_occipital_thickness": "L occipital pole", "Left-Pallidum": "L pallidum", "Right-Pallidum": "R pallidum",
         "lh_S_oc_middle&Lunatus_thickness": "L mid. occipital sulcus", "lh_G_temp_sup-Lateral_thickness": "L lateral STG",
         "Left-Inf-Lat-Vent": "L temporal horn", "Right-Inf-Lat-Vent": "R temporal horn", "TotalGrayVol": "total grey volume",
         "Left-Lateral-Ventricle": "L lateral ventricle", "Right-Lateral-Ventricle": "R lateral ventricle", "lh_G_cuneus_thickness": "L cuneus",
         "CSF": "CSF", "lh_S_calcarine_thickness": "L calcarine sulcus", "rh_G&S_paracentral_thickness": "R paracentral lobule",
         "lh_G&S_frontomargin_thickness": "L frontomarginal gyrus", "lh_G_rectus_thickness": "L gyrus rectus",
         "rh_G_temp_sup-Plan_polar_thickness": "R planum polare", "rh_G_temp_sup-G_T_transv_thickness": "R Heschl's gyrus",
         "Right-Accumbens-area": "R accumbens", "Left-Amygdala": "L amygdala", "Right-Putamen": "R putamen"}


def panel_d(d):
    fig = new_panel("(d) Direction of the top drivers")
    blocks = (("fMRI", "fMRI network pairs", 2.56, 1.64), ("sMRI", "sMRI regions and volumes", 1.44, 0.52))
    pitch, n = 0.105, 8
    x0, w = 1.20, 2.15
    for blk, head, y_head, y_bottom in blocks:
        T = d[d["block"] == blk].sort_values("rank")
        if len(T) != n:
            raise RuntimeError(f"{blk}: {len(T)} rows")
        ax = inch_axes(fig, x0, y_bottom, w, pitch * n)
        names = []
        for k, (_, r) in enumerate(T.iterrows()):
            y = n - 1 - k
            if blk == "fMRI":
                names.append(pair_name(r["feature"]))
            else:
                if r["feature"] not in NAMES:
                    raise RuntimeError(f"no curated name for {r['feature']}")
                names.append(NAMES[r["feature"]])
            ax.plot([r["z_ctl_lo"], r["z_ctl_hi"]], [y - 0.2] * 2, color=C_CTL, linewidth=0.7, solid_capstyle="butt", zorder=2)
            ax.plot(r["z_ctl"], y - 0.2, "o", markersize=2.6, color=C_CTL, markeredgewidth=0, zorder=3)
            ax.plot([r["z_aut_lo"], r["z_aut_hi"]], [y + 0.2] * 2, color=C_AUT, linewidth=0.8, solid_capstyle="butt", zorder=2)
            op = bool(r["open_circle"])
            ax.plot(r["z_aut"], y + 0.2, "o", markersize=3.4, color="white" if op else C_AUT, markeredgecolor=C_AUT, markeredgewidth=0.8,
                    zorder=4)
        ax.axvline(0, color=AXIS, linewidth=0.5, zorder=1)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks(range(n - 1, -1, -1))
        ax.set_yticklabels(names, fontsize=FS_SMALL, color=INK)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
        ax.set_xticklabels(["−0.5", "", "0", "", "+0.5"], fontsize=FS_TICK)
        ax.tick_params(axis="x", colors=INK2, length=2, pad=1)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        fig.text(0.06 / PW, y_head / PH, head, ha="left", va="center", fontsize=FS, color=INK, fontweight="bold")
        if blk == "sMRI":
            ax.set_xlabel("mean z, 95% CI (0 = typical for age, sex, site)", fontsize=FS_SMALL, color=INK2, labelpad=1.5)
    hs = [Line2D([], [], color=C_AUT, marker="o", markersize=3.4, linewidth=0.8),
          Line2D([], [], color=C_CTL, marker="o", markersize=2.6, linewidth=0.7),
          Line2D([], [], color="white", marker="o", markersize=3.4, markeredgecolor=C_AUT, markeredgewidth=0.8, linewidth=0)]
    fig.legend(hs, ["autistic", "controls", "no mean shift (via |z|)"], loc="center", bbox_to_anchor=(0.5, 0.11 / PH), ncol=3,
               frameon=False, fontsize=FS_SMALL, handlelength=1.4, handletextpad=0.35, columnspacing=0.9, borderaxespad=0)
    save(fig, "Fig2d")
    for blk in ("fMRI", "sMRI"):
        T = d[d["block"] == blk].sort_values("rank")
        print(f"  (d) {blk}: " + "; ".join(f"{pair_name(f) if blk == 'fMRI' else NAMES[f]} {z:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                                         + (" open" if o else "") for f, z, lo, hi, o in
                                         zip(T["feature"], T["z_aut"], T["z_aut_lo"], T["z_aut_hi"], T["open_circle"])))


def preview():
    """the four panels 2 x 2 at 7.0 in wide (0.1 in gutter), 300 dpi, to check that they match"""
    from PIL import Image
    ims = [Image.open(f"{OUT}/Fig2{k}.png") for k in "abcd"]
    dpi = 300
    w, h = int(PW * dpi), int(PH * dpi)
    g = int(0.1 * dpi)
    canvas = Image.new("RGB", (2 * w + g, 2 * h + g), "white")
    for i, im in enumerate(ims):
        canvas.paste(im.convert("RGB").resize((w, h), Image.LANCZOS), ((i % 2) * (w + g), (i // 2) * (h + g)))
    canvas.save(f"{OUT}/Fig2_preview_2x2.png", dpi=(dpi, dpi))
    print(f"written {OUT}/Fig2_preview_2x2.png ({canvas.size[0]} x {canvas.size[1]} px)")


if __name__ == "__main__":
    fm, cort, d, vmax, thr = load()
    todo = sys.argv[1:] or ["a", "b", "c", "d"]
    for k in todo:
        {"a": lambda: panel_a(fm, vmax, thr), "b": lambda: panel_b(cort, vmax, thr), "c": panel_c, "d": lambda: panel_d(d)}[k]()
    if all(os.path.exists(f"{OUT}/Fig2{k}.png") for k in "abcd"):
        preview()
