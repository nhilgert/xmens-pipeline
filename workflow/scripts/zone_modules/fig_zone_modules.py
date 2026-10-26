"""Decidual-zone AUCell scores: spatial panels of one mouse section and subtype x module heatmaps.

Called by rule zone_figures (workflow/rules/16_zone_modules.smk) through render_zone_figures.py,
once per configured figure job. Reads obs of the mouse release object (--obj, only the columns
x_um_qc, y_um_qc, sample, timepoint, cell_type, cell_subtype), the barcode-indexed auc_<MODULE>
table of score_zone_aucell.py (--scores) with the provenance.json beside it, and optionally a
CSV of barcodes to hide from the spatial panels (--hide-barcodes). Writes to --out, for every
figure, a light and a dark variant in each --formats (png, svg) plus a <name>.provenance.json
sidecar. Existing files are never overwritten without --force.

Figure names:
    pdz_sdz_spatial_<tp>[_noapop][_rot<deg>]                 PDZ and SDZ
    zone_modules_spatial_<tp>[_noapop][_rot<deg>][_<n>zones] all 11 modules (PDZ featured),
                                                             or the --modules subset
    zone_modules_heatmap_subtypes[_with_apoptotic][_<tp>][_<n>zones][_fibpooled][_p99]
    plus _<cmap> for a non-magma colour map and --suffix. Heatmap SVG is fully vector. In
    spatial SVG each section's point cloud is one embedded bitmap at --dpi.

Modules: PDZ / TDZ / SDZ = primary / transitional / secondary decidual zone (inner to outer),
VSZ (vascular sinus zone), MD (mesometrial decidua, whose top markers are uterine NK-cell
genes that Li et al. describe as non-cytotoxic), MMy, C (conceptus), FMI (fetal-maternal
interface, endothelial-rich in Li et al.'s deconvolution, which maps myeloid cells to C),
Glan, udStr, AMMy: the 11 uterine microenvironments of Li et al. (2022) Biol Reprod 107:529, Table 1
(top 20 markers each, listed in resources/signatures/aucell_modules_mouse.yaml).

Conventions, fixed for the manuscript:
    * The score table is the only source of scores. The object's own auc_<MODULE> columns were
      scored with a different AUCell window and are never read, so two windows cannot mix in
      one figure. Window, seed, normalization, scored timepoints and gene sets are copied from
      the scorer's provenance.json into every sidecar.
    * The score table may cover only some timepoints. Figures are then restricted to those
      timepoints, every nucleus of them must be scored, and an all-timepoint heatmap is
      refused (pass --heatmap-timepoint).
    * Spatial panels: colour runs from a true 0 to the module's 99th percentile within the
      plotted cells, per module. AUCell at this window never approaches 1, so a fixed 0-1 axis
      would leave the map unused, and a true min-max lets a few extreme cells hide the
      structure. The ceiling is stated on the colour bar. Lanes of one timepoint are one
      section and are pooled; cells without QC'd coordinates are left out; bright cells are
      drawn last.
    * --hide-barcodes removes nuclei from the spatial panels only. Embedding outliers are
      omitted from embedding and spatial plots and kept in every summary, so heatmaps keep them.
    * Heatmap, default --heatmap-scale colmax: colour = subtype mean score from a true 0 to the
      module's largest subtype mean (per column); that maximum is printed under each column so
      absolute levels stay readable. The exploratory notebook used min-max per column instead
      (lowest subtype always black).
    * Heatmap, --heatmap-scale p99 (files *_p99*): colour = subtype mean / 99th percentile of
      the per-cell scores of that module over the plotted subtypes, so the same colour means
      the same score level as in the spatial panels.
    * --pool-fib: the five efib_* rows become one 'endometrial fibroblasts (pooled)' row, the
      cell-weighted mean over all fibroblast cells. The p99 ceiling is unchanged (same cells).
    * Subtypes use the release vocabulary (efib_outer = former efib_myometrial,
      decidual_boundary = former decidual_border). decidual_apoptotic appears only in the
      *_with_apoptotic variant: that subtype (excluded from the pseudotime of record) has the
      highest mean in PDZ/TDZ/SDZ/VSZ/FMI/udStr, so including it rescales those columns and
      hides the decidual_mature PDZ/TDZ signal.
    The original exploratory panel (analysis/notebooks/module_scoring_aucell.ipynb) used the
    2nd-98th percentile per module, re-scored the 3 dpi cells on their own, and applied a
    second spatial-outlier filter. This script does none of those.
"""
import os, sys, json, argparse, datetime
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import h5py
import hashlib
from anndata.io import read_elem

# Constants ----
ORDER = ["PDZ", "TDZ", "SDZ", "VSZ", "MD", "MMy", "C", "FMI", "Glan", "udStr", "AMMy"]
TWO = [("PDZ", "primary decidual zone (PDZ)"), ("SDZ", "secondary decidual zone (SDZ)")]

# Heatmap rows, fibroblast subtypes first (outer to proliferative), then decidual.
SUBTYPES = [
    "efib_outer",
    "efib_middle",
    "efib_middle_induced",
    "efib_subluminal",
    "efib_proliferative",
    "decidual_boundary",
    "decidual_proliferative",
    "decidual_mature",
]
# Heatmap variant: decidual_apoptotic has the highest mean in 6 of 11 modules (module docstring).
SUBTYPES_APOP = SUBTYPES + ["decidual_apoptotic"]
# The first N_FIB rows are endometrial_fibroblast subtypes; a rule separates them from decidual.
N_FIB = 5
CMAP, Q = "magma", 0.99  # colour map (overridden by --cmap) and the per-module ceiling quantile
META_GREY = "#6f6f6f"  # n, column maxima and the footnote on the light background


def style():
    """Set journal-style rcParams; returns the font family actually available."""
    avail = {f.name for f in font_manager.fontManager.ttflist}

    # Helvetica Neue is the manuscript font; the others are fallbacks in order of preference.
    fam = next(
        (
            f
            for f in ("Helvetica Neue", "Helvetica", "Nimbus Sans", "Nimbus Sans L", "DejaVu Sans")
            if f in avail
        ),
        "DejaVu Sans",
    )

    # fonttype 42 keeps text editable in PDF/PS; svg.fonttype none keeps SVG text as text.
    plt.rcParams.update({
        "font.family": [fam, "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    return fam


def _md5(path):
    h = hashlib.md5()
    f = open(path, "rb")

    for b in iter(lambda: f.read(1 << 20), b""):  # 1 MiB blocks
        h.update(b)

    return h.hexdigest()


def _src(path, **extra):
    """Provenance record of one input file."""
    return {
        "path": os.path.abspath(path),
        "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(
            timespec="seconds"
        ),
        "bytes": os.path.getsize(path),
        **extra,
    }


def load(timepoint, OBJ, scores, table=None):
    """obs (from the h5ad or a CSV export) joined with the score table.

    Returns the plotted spatial set g (one timepoint, placed nuclei), D (all nuclei of the
    scored timepoints), score metadata, sources, scored timepoints and all timepoints.
    """
    auc = [f"auc_{m}" for m in ORDER]
    base = ["sample", "timepoint", "cell_type", "cell_subtype", "x_um_qc", "y_um_qc"]

    # Load obs ----
    if table:
        D = pd.read_csv(table, index_col=0)
        D.index = D.index.astype(str)
        miss = [c for c in base if c not in D.columns]
        assert not miss, f"missing columns in {table}: {miss}"
        D = D[base]
        srcs = [_src(table, md5=_md5(table), note=f"CSV export of the obs of {OBJ}")]
    else:
        # obs only; the object's own auc_ columns are deliberately not read
        with h5py.File(OBJ, "r") as h:
            idx = pd.Index(read_elem(h["obs/_index"]).astype(str))
            D = pd.DataFrame({c: np.asarray(read_elem(h["obs/" + c])) for c in base}, index=idx)
        srcs = [_src(OBJ)]

    for c in ("sample", "timepoint", "cell_type", "cell_subtype"):
        D[c] = D[c].astype(str)

    # Join the score table ----
    S = pd.read_csv(scores, index_col=0)
    S.index = S.index.astype(str)
    miss = [c for c in auc if c not in S.columns]
    assert not miss, f"missing columns in {scores}: {miss}"
    extra = S.index.difference(D.index)
    assert len(extra) == 0, f"{len(extra)} scored nuclei are not in the input"

    # Restrict to the timepoints the score table covers; each must be scored completely.
    all_tps = sorted(D["timepoint"].unique())
    tps = sorted(D.loc[S.index, "timepoint"].unique())
    D = D[D["timepoint"].isin(tps)].copy()
    absent = D.index.difference(S.index)
    assert len(absent) == 0, (
        f"{len(absent)} nuclei of the scored timepoints {tps} are not in {scores}"
    )

    for c in auc:
        D[c] = S.loc[D.index, c].to_numpy(float)

    # Score metadata from the scorer's provenance.json, copied into every sidecar ----
    pj = os.path.join(os.path.dirname(os.path.abspath(scores)), "provenance.json")
    P = json.load(open(pj)) if os.path.exists(pj) else {}
    meta = {
        k: str(P.get(k, "?"))
        for k in ("auc_threshold", "rank_cutoff", "seed", "normalization", "timepoints")
    }
    meta["genesets"] = P.get("genesets", {})
    meta["scores_file"] = os.path.abspath(scores)
    srcs.append(_src(scores, md5=_md5(scores), role="auc_<MODULE> scores"))

    if os.path.exists(pj):
        srcs.append(_src(pj, md5=_md5(pj), role="score provenance"))

    # Spatial set: one timepoint, placed nuclei only ----
    assert timepoint in tps, f"timepoint {timepoint!r} is not scored; the score table covers {tps}"
    g = D[(D["timepoint"] == timepoint) & D[["x_um_qc", "y_um_qc"]].notna().all(1)].copy()
    assert len(g) > 0, f"no cells with coordinates at timepoint {timepoint!r}"

    return g, D, meta, srcs, tps, all_tps


def rotate_section(g, deg):
    """Rotate the plotted section by deg (counter-clockwise) about the median centre of its
    placed cells. The lanes of one timepoint are one section in one coordinate frame, so
    the pooled cells rotate about a single shared centre.
    Display only; obs coordinates are not modified."""
    if not deg % 360:
        return g

    r = np.deg2rad(deg)
    cx, cy = float(g.x_um_qc.median()), float(g.y_um_qc.median())
    x, y = g.x_um_qc.values - cx, g.y_um_qc.values - cy

    return g.assign(
        x_um_qc=cx + np.cos(r) * x - np.sin(r) * y,
        y_um_qc=cy + np.sin(r) * x + np.cos(r) * y,
    )


def rc(dark):
    """rcParams for the dark or light variant, and the foreground colour."""
    bg, fg = ("black", "white") if dark else ("white", "black")

    return {
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "text.color": fg,
        "axes.labelcolor": fg,
        "xtick.color": fg,
        "ytick.color": fg,
        "savefig.facecolor": bg,
    }, fg


def panel(ax, g, mod, s):
    """One spatial panel of module mod; s is the marker size. Returns the colour ceiling."""
    # Colour: 0 -> 99th percentile of this module within the plotted cells (module docstring).
    v = g[f"auc_{mod}"].values.astype(float)
    hi = float(np.quantile(v, Q))
    vn = np.clip(v / hi, 0, 1) if hi > 0 else np.zeros_like(v)

    # Bright cells are drawn last so they are not buried under dim neighbours.
    o = np.argsort(vn)
    ax.scatter(
        g.x_um_qc.values[o],
        g.y_um_qc.values[o],
        c=vn[o],
        cmap=CMAP,
        vmin=0,
        vmax=1,
        s=s,
        linewidths=0,
        rasterized=True,  # the point cloud is one bitmap in SVG, text stays vector
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.margins(0.04)

    return hi


def cbar(fig, rect, fg):
    """Shared colour bar of the spatial panels, 0 to the per-module 99th percentile."""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=CMAP),
        cax=cax,
        ticks=[0, 1],
    )
    cb.ax.set_yticklabels(["0", "99th pct"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=fg, labelcolor=fg, length=2)
    cax.set_title(
        "AUCell score\n(0 \u2013 99th pct,\nper module)",
        loc="left",
        fontsize=8,
        color=fg,
        pad=6,
    )


def note(g, tp):
    lanes = sorted(g["sample"].unique())
    tpl = tp.replace("_", " ")

    return f"{tpl}, n = {len(g):,} cells" + (
        f" ({len(lanes)} lanes of one section pooled)" if len(lanes) > 1 else ""
    )


def fig_two(g, tp, dark):
    """PDZ and SDZ side by side."""
    rcd, fg = rc(dark)

    with plt.rc_context(rcd):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))  # inches
        ceil = {}

        for ax, (m, lab) in zip(axes, TWO):
            ceil[m] = panel(ax, g, m, 1.8)
            ax.set_title(lab, loc="left", color=fg)

        # Room on the right for the colour bar.
        fig.subplots_adjust(left=0.02, right=0.84, top=0.90, bottom=0.07, wspace=0.04)
        cbar(fig, [0.865, 0.28, 0.018, 0.40], fg)

    return fig, {"ceiling_99th_pct": {m: round(v, 4) for m, v in ceil.items()}}


def fig_six(g, tp, dark, modules):
    """Equal-size panels for a module subset (e.g. the heatmap's six), 2 rows x ceil(n/2)
    columns."""
    rcd, fg = rc(dark)
    nc = int(np.ceil(len(modules) / 2))

    with plt.rc_context(rcd):
        # 2.55 in per column plus 1.2 in on the right for the colour bar.
        fig = plt.figure(figsize=(2.55 * nc + 1.2, 5.6))
        gs = matplotlib.gridspec.GridSpec(
            2,
            nc,
            figure=fig,
            left=0.01,
            right=1 - 1.2 / (2.55 * nc + 1.2),
            top=0.93,
            bottom=0.06,
            wspace=0.05,
            hspace=0.14,
        )
        ceil = {}

        for i, m in enumerate(modules):
            ax = fig.add_subplot(gs[divmod(i, nc)])
            ceil[m] = panel(ax, g, m, 1.4)
            ax.set_title(m, loc="left", color=fg)

        cbar(fig, [1 - 0.95 / (2.55 * nc + 1.2), 0.32, 0.012, 0.36], fg)

    return fig, {
        "modules": list(modules),
        "ceiling_99th_pct": {m: round(v, 4) for m, v in ceil.items()},
    }


def fig_all(g, tp, dark):
    """All 11 modules: PDZ as a full-height panel on the left, the other ten in a 2 x 5 grid."""
    rcd, fg = rc(dark)

    with plt.rc_context(rcd):
        fig = plt.figure(figsize=(11.0, 4.3))  # inches
        gs = matplotlib.gridspec.GridSpec(
            2,
            6,
            figure=fig,
            left=0.01,
            right=0.90,
            top=0.92,
            bottom=0.06,
            wspace=0.04,
            hspace=0.18,
        )
        ceil = {}

        # PDZ spans both rows of the first column; its larger panel gets a larger marker.
        ax = fig.add_subplot(gs[:, 0])
        ceil["PDZ"] = panel(ax, g, "PDZ", 2.4)
        ax.set_title("PDZ", loc="left", color=fg)

        for i, m in enumerate(ORDER[1:]):
            r, c = divmod(i, 5)
            ax = fig.add_subplot(gs[r, c + 1])
            ceil[m] = panel(ax, g, m, 1.2)
            ax.set_title(m, loc="left", color=fg, fontsize=8)

        cbar(fig, [0.915, 0.32, 0.012, 0.36], fg)

    return fig, {"ceiling_99th_pct": {m: round(v, 4) for m, v in ceil.items()}}


FIB_POOLED = "endometrial fibroblasts (pooled)"


def fig_heatmap(
    D,
    dark,
    subtypes=SUBTYPES,
    scale="colmax",
    pool_fib=False,
    modules=None,
    timepoint=None,
):
    """Mean AUCell per fibroblast/decidual subtype (rows) x module (columns), colour 0 to the
    column ceiling (see the module docstring for the colmax and p99 scales)."""
    # One timepoint only (lanes of that section pooled); the p99 ceiling then uses those cells too.
    if timepoint:
        D = D[D["timepoint"].astype(str) == timepoint]
        assert len(D), timepoint

    ORDER_ = list(modules) if modules else ORDER
    # The bare string below is a misplaced docstring: a no-op expression statement.
    """Mean AUCell per fib/dec subtype (rows) x module (columns); colour 0 -> column max."""
    rcd, fg = rc(dark)
    grey = "#bbbbbb" if dark else META_GREY

    # Rows ----
    sub = D[D["cell_subtype"].isin(subtypes)]
    missing = [s_ for s_ in subtypes if s_ not in set(sub["cell_subtype"])]
    assert not missing, f"subtypes absent from object: {missing}"
    n_fib_rows = N_FIB

    # Pooled row = mean over ALL fibroblast cells (cell-weighted), not the mean of the five
    # subtype means.
    if pool_fib:
        fibs = subtypes[:N_FIB]
        assert all(f_.startswith("efib_") for f_ in fibs), fibs
        sub = sub.assign(
            cell_subtype=np.where(
                sub["cell_subtype"].isin(fibs),
                FIB_POOLED,
                sub["cell_subtype"].astype(str),
            )
        )
        subtypes = [FIB_POOLED] + list(subtypes[N_FIB:])
        n_fib_rows = 1

    M = sub.groupby("cell_subtype")[[f"auc_{m}" for m in ORDER_]].mean().reindex(subtypes)
    n = sub["cell_subtype"].value_counts().reindex(subtypes)

    # Colour ceiling per column ----
    if scale == "p99":
        # 99th percentile of PER-CELL scores over these subtypes: same colour = same score
        # level as the spatial panels.
        cmax = sub[[f"auc_{m}" for m in ORDER_]].quantile(Q)
    else:
        # Largest subtype mean.
        cmax = M.max(axis=0)

    S = (M / cmax.values).clip(0, 1).values

    # Layout in inches: a constant cell size so every variant has the same cell footprint.
    nr, nc = S.shape
    cw = 0.40  # cell size
    # Margins: left for row labels, right for n and the colour bar, top for column labels,
    # bottom for the footnote.
    L, R, T, B = 1.75, 2.25, 0.45, 0.75
    W, H = L + nc * cw + R, T + nr * cw + B

    with plt.rc_context(rcd):
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([L / W, B / H, nc * cw / W, nr * cw / H])

        # Grid ----
        # Vector cells (one path per cell) so the SVG grid is editable. edgecolors="face"
        # avoids hairline seams between cells.
        ax.pcolormesh(
            np.arange(nc + 1) - 0.5,
            np.arange(nr + 1) - 0.5,
            S,
            cmap=CMAP,
            vmin=0,
            vmax=1,
            edgecolors="face",
            linewidth=0.2,
            rasterized=False,
        )
        ax.set_xlim(-0.5, nc - 0.5)
        ax.set_ylim(nr - 0.5, -0.5)  # first subtype on top
        ax.set_xticks(range(nc))
        ax.set_xticklabels(ORDER_)
        ax.xaxis.tick_top()
        ax.set_yticks(range(nr))
        ax.set_yticklabels(subtypes)
        ax.tick_params(length=0, pad=3, colors=fg)

        for sp in ax.spines.values():
            sp.set_visible(False)

        # Fibroblast/decidual divider only when the five fibroblast subtypes are shown separately.
        if not pool_fib:
            ax.axhline(n_fib_rows - 0.5, color=fg, lw=0.8)

        # Annotations: n per row on the right, the colour ceiling under each column ----
        for i, k in enumerate(n.values):
            ax.text(
                nc - 0.5 + 0.15,
                i,
                f"n = {int(k):,}",
                ha="left",
                va="center",
                fontsize=7,
                color=grey,
            )

        for j, v in enumerate(cmax.values):
            ax.text(
                j,
                nr - 0.5 + 0.12,
                f"{v:.2f}" if v >= 0.01 else f"{v:.3f}",
                ha="center",
                va="top",
                fontsize=6.5,
                color=grey,
                rotation=90,
            )

        ax.text(
            -0.5 - 0.15,
            nr - 0.5 + 0.12,
            "99th pct\n(per cell)" if scale == "p99" else "column max\n(mean AUCell)",
            ha="right",
            va="top",
            fontsize=6.5,
            color=grey,
        )
        fig.text(
            0.01,
            0.015,
            f"mouse fibroblast/decidual cells, "
            + (
                f"{timepoint.replace('_', ' ')} only (both lanes of the section pooled)"
                if timepoint
                else "all timepoints pooled"
            )
            + f", n = {int(n.sum()):,}",
            ha="left",
            va="bottom",
            fontsize=7,
            color=grey,
        )

        # Colour bar, vector like the grid ----
        x0 = (L + nc * cw + 0.95) / W
        cax = fig.add_axes([x0, (B + nr * cw * 0.25) / H, 0.12 / W, nr * cw * 0.5 / H])
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=CMAP),
            cax=cax,
            ticks=[0, 1],
        )
        cb.ax.set_yticklabels(["0", "99th pct" if scale == "p99" else "max"])
        cb.outline.set_visible(False)
        cb.ax.tick_params(color=fg, labelcolor=fg, length=2)
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
        cax.set_title(
            "mean AUCell\n(0 \u2013 99th pct\nof cells)"
            if scale == "p99"
            else "mean AUCell\n(0 \u2013 column max)",
            loc="left",
            fontsize=8,
            color=fg,
            pad=6,
        )

    return fig, {
        "heatmap_scale": scale,
        ("ceiling_99th_pct_per_cell" if scale == "p99" else "column_max_mean_auc"): {
            m: round(float(v), 4) for m, v in zip(ORDER_, cmax.values)
        },
        "n_per_subtype": {k: int(v) for k, v in n.items()},
        "mean_auc": {
            s_: {m: round(float(M.loc[s_, f"auc_{m}"]), 4) for m in ORDER_} for s_ in subtypes
        },
    }


def main():
    # Arguments ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timepoint", default="3_dpi")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--formats", default="png,svg")
    ap.add_argument("--cmap", default="magma")
    ap.add_argument("--figures", default="spatial,heatmap")
    ap.add_argument("--heatmap-scale", default="colmax", choices=["colmax", "p99"])  # p99 adds _p99
    # Comma list subset of columns, e.g. PDZ,TDZ,SDZ,VSZ,MD,udStr; adds _<n>zones to names.
    ap.add_argument("--modules", default="")
    # e.g. 3_dpi: heatmap from that timepoint only; adds _<tp> to heatmap names.
    ap.add_argument("--heatmap-timepoint", default="")
    # Spatial panels: degrees CCW about the section centre; adds _rot<deg>.
    ap.add_argument("--rotate", type=float, default=0)
    # Spatial panels: drop decidual_apoptotic before scaling and plotting; adds _noapop.
    ap.add_argument("--drop-apoptotic", action="store_true")
    ap.add_argument("--pool-fib", action="store_true")  # one fibroblast row; adds _fibpooled
    # Barcode-indexed auc_<MODULE> table from score_zone_aucell.py.
    ap.add_argument("--scores", required=True)
    # CSV with a 'barcode' column: hidden from the spatial panels only.
    ap.add_argument("--hide-barcodes", default="")
    ap.add_argument("--table", default="")  # read obs columns from a CSV export instead of the h5ad
    a = ap.parse_args()

    # The colour map is read by panel() and cbar() through the module constant.
    global CMAP
    CMAP = a.cmap
    ctag = "" if a.cmap == "magma" else f"_{a.cmap}"
    want = {f.strip() for f in a.figures.split(",")}

    fam = style()
    os.makedirs(a.out, exist_ok=True)

    # Load inputs ----
    g, D, uns, srcs, scored_tps, all_tps = load(a.timepoint, a.obj, a.scores, a.table or None)
    win = (
        f"AUCell window {uns['auc_threshold']} (top {uns['rank_cutoff']} genes, "
        f"seed {uns['seed']}, {uns['normalization']}; "
        f"scored timepoints {uns['timepoints']}; scores {os.path.basename(uns['scores_file'])})"
    )

    # Spatial set: hidden barcodes, apoptotic cells, rotation ----
    # Hiding applies to the spatial panels only; heatmaps (D) keep these nuclei.
    n_hid, hid_note = 0, ""

    if a.hide_barcodes:
        hb = set(pd.read_csv(a.hide_barcodes)["barcode"].astype(str))
        n_hid = int(g.index.isin(hb).sum())
        g = g[~g.index.isin(hb)]
        hid_note = (
            f"; {n_hid} nuclei listed in {os.path.basename(a.hide_barcodes)} hidden "
            f"(embedding outliers; kept in heatmaps)"
        )
        srcs.append(
            _src(
                a.hide_barcodes,
                md5=_md5(a.hide_barcodes),
                role="barcodes hidden from spatial panels",
                n_listed=len(hb),
            )
        )

    tag = a.timepoint.replace("_", "")  # 3_dpi -> 3dpi
    n_ap = 0

    if a.drop_apoptotic:
        n_ap = int((g["cell_subtype"] == "decidual_apoptotic").sum())
        g = g[g["cell_subtype"] != "decidual_apoptotic"]
        tag += "_noapop"

    if a.rotate % 360:
        g = rotate_section(g, a.rotate)
        tag += f"_rot{a.rotate:g}"

    sp_note = (
        f"AUCell module scores, {win}, timepoint {a.timepoint}, "
        f"lanes pooled {sorted(g['sample'].unique())}, "
        f"{len(g)} cells with QC'd coordinates"
        + hid_note
        + (f" ({n_ap} decidual_apoptotic removed)" if a.drop_apoptotic else "")
        + (
            f"; display rotated {a.rotate:g} deg CCW about the section's median centre"
            if a.rotate % 360
            else ""
        )
        + f"; colour 0 -> per-module 99th percentile; font {fam}"
    )

    # Heatmap options ----
    mods = [m.strip() for m in a.modules.split(",") if m.strip()] or None
    assert not mods or set(mods) <= set(ORDER), mods
    htp = a.heatmap_timepoint or None

    # A partially scored object cannot give an honest "all timepoints pooled" heatmap.
    if "heatmap" in want:
        assert htp is not None or scored_tps == all_tps, (
            "the score table covers timepoints %s only; pass --heatmap-timepoint "
            "(an 'all timepoints pooled' heatmap would be mislabelled)" % scored_tps
        )
        assert htp is None or htp in scored_tps, (
            f"heatmap timepoint {htp!r} is not scored ({scored_tps})"
        )

    ptag = (
        (f"_{htp.replace('_', '')}" if htp else "")
        + (f"_{len(mods)}zones" if mods else "")
        + ("_fibpooled" if a.pool_fib else "")
        + ("_p99" if a.heatmap_scale == "p99" else "")
    )
    Dh = D[D["timepoint"] == htp] if htp else D
    hm_note = lambda st: (
        f"mean AUCell per fibroblast/decidual subtype ({len(st)} subtypes), "
        + (f"timepoint {htp} only" if htp else "all timepoints pooled")
        + ", "
        f"{int(Dh.cell_subtype.isin(st).sum())} cells; {win}; colour 0 -> "
        + ("99th pct of per-cell scores" if a.heatmap_scale == "p99" else "largest subtype mean")
        + f" per module; font {fam}"
    )

    # Figure jobs: (file stem, figure factory, sidecar note) ----
    jobs = [
        (
            "pdz_sdz_spatial_%s" % tag,
            lambda dark: fig_two(g, a.timepoint, dark),
            sp_note,
        ),
        (
            "zone_modules_spatial_%s" % tag,
            lambda dark: fig_all(g, a.timepoint, dark),
            sp_note,
        ),
        *(
            [
                (
                    "zone_modules_spatial_%s_%dzones" % (tag, len(mods)),
                    lambda dark: fig_six(g, a.timepoint, dark, mods),
                    sp_note,
                )
            ]
            if mods
            else []
        ),
        (
            "zone_modules_heatmap_subtypes" + ptag,
            lambda dark: fig_heatmap(D, dark, SUBTYPES, a.heatmap_scale, a.pool_fib, mods, htp),
            hm_note(SUBTYPES),
        ),
        (
            "zone_modules_heatmap_subtypes_with_apoptotic" + ptag,
            lambda dark: fig_heatmap(
                D, dark, SUBTYPES_APOP, a.heatmap_scale, a.pool_fib, mods, htp
            ),
            hm_note(SUBTYPES_APOP),
        ),
    ]
    jobs = [j for j in jobs if ("heatmap" if "heatmap" in j[0] else "spatial") in want]

    # Render every job in a dark and a light variant, each with a provenance sidecar ----
    for stem, fn, jnote in jobs:
        for dark in (True, False):
            base = stem + ctag + a.suffix + ("_dark" if dark else "")
            fmts = [f.strip() for f in a.formats.split(",") if f.strip()]
            paths = {f: os.path.join(a.out, base + "." + f) for f in fmts}

            for path in paths.values():
                if os.path.exists(path) and not a.force:
                    sys.exit(f"refusing to overwrite {path} (use --force)")

            fig, info = fn(dark)

            for f, path in paths.items():
                # SVG: rasterized scatters embed at this dpi, text stays vector.
                fig.savefig(path, dpi=a.dpi)

            plt.close(fig)
            path = paths[fmts[0]]

            side = {
                "figure": os.path.abspath(path),
                "files": [os.path.abspath(p) for p in paths.values()],
                "dpi": a.dpi,
                "written": datetime.datetime.now().isoformat(timespec="seconds"),
                "note": jnote,
                "cmap": a.cmap,
                "aucell_scores": uns,
                "hide_barcodes": (
                    {
                        "file": os.path.abspath(a.hide_barcodes),
                        "applies_to": "spatial panels only",
                        "n_hidden_in_spatial_set": n_hid,
                    }
                    if a.hide_barcodes
                    else None
                ),
                **info,
                "script": os.path.abspath(__file__),
                "sources": srcs,
            }
            json.dump(side, open(os.path.join(a.out, base + ".provenance.json"), "w"), indent=1)

            print("wrote", *paths.values())


if __name__ == "__main__":
    main()
