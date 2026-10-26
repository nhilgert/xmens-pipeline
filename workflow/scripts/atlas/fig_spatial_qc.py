"""Supplementary figure: Slide-tags positions and their spatial QC (rule spatial_qc, 04f).

Run by hand from the command line, as no Snakemake rule calls it. Panel a shows the raw
positions of every section colored by QC outcome with the exclusion regions and the
uninduced section boundary, panel b the QC outcome per 10x lane, panel c the
far-from-neighbors cut per mouse timepoint and panel d the QC outcome per cell type.

Reads only obs: the release all-cells objects (mouse, human) and the 04f mouse object
(data/interim/qc_spatial_mouse.h5ad), whose uns['spatial_qc'] holds the parameters the
QC applied. With --tables it reads CSV exports of the same columns (written by
--export-tables) for local renders. Writes <out>/supp_spatial_qc.{png,pdf,svg},
supp_spatial_qc.provenance.json and one CSV of the counts behind each of panels b-d.
"""

import argparse
import datetime
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, PercentFormatter
from sklearn.neighbors import NearestNeighbors


# Constants ----

STEM = "supp_spatial_qc"

# Full page width (183 mm); 9 in tall leaves room for the legend text on a supplementary page.
FIG_SIZE_IN = (7.2, 9.0)

# Three text sizes, set once in apply_style(): titles and axis labels; legend and annotations;
# tick labels share the annotation size. Panel letters are the one exception.
BASE_PT = 7
SMALL_PT = 6
LETTER_PT = 9
FONT_PREFERENCE = ["Helvetica Neue", "Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"]

REASONS = ["kept", "degenerate", "far", "boxed", "unmapped"]
BLANKED = ["degenerate", "far", "boxed"]

# Kept positions in mid gray and nuclei that never had one in a lighter gray, so the eye goes to
# the three blanking criteria. Those get Okabe-Ito hues, which stay apart under the common
# color-vision deficiencies. The same mapping is used in every panel.
REASON_COLORS = {
    "kept": "#B0B0B0",
    "degenerate": "#D55E00",
    "far": "#0072B2",
    "boxed": "#CC79A7",
    "unmapped": "#E9E9E9",
}

# Plain-language names, with the spatial_qc_reason value of the release objects in parentheses.
REASON_LABELS = {
    "kept": "Kept",
    "degenerate": "Duplicate or crowded (degenerate)",
    "far": "Far from neighbors (far)",
    "boxed": "In an exclusion region (boxed)",
    "unmapped": "No Slide-tags position (unmapped)",
}

# Display order follows the experiment: mouse induction time course, then the human phases.
TIMEPOINTS = {
    "mouse": ["uninduced", "1_dpi_early", "1_dpi_late", "2_dpi", "3_dpi", "4_dpi_bleeding"],
    "human": ["secretory", "menstrual"],
}
TIMEPOINT_LABELS = {
    "uninduced": "Uninduced",
    "1_dpi_early": "1 dpi, early",
    "1_dpi_late": "1 dpi, late",
    "2_dpi": "2 dpi",
    "3_dpi": "3 dpi",
    "4_dpi_bleeding": "4 dpi, bleeding",
    "secretory": "Secretory",
    "menstrual": "Menstrual",
}

SCALE_BAR_UM = 1000
MAP_PAD = 0.04  # margin around each section, as a fraction of its larger extent
KEPT_POINT_SIZE = 0.5  # pt^2, small because a section holds thousands of kept positions

# Duplicate positions stack exactly and crowded ones sit a few µm apart, so at map scale hundreds
# of them would print as one dot. Blanked positions within one 50 µm grid cell (about one marker
# width at this scale) therefore share a marker. Its area grows with the square root of its
# nuclei: counts per marker run from 1 to over 1,000, and a linear area would cover a section.
# Markers of LABEL_MIN_NUCLEI or more also print their count.
BLANKED_BIN_UM = 50
BLANKED_POINT_SIZE = 2.5  # pt^2 for one nucleus
SIZE_KEY_NUCLEI = [1, 10, 100, 1000]
LABEL_MIN_NUCLEI = 100

FAR_BIN_LOG10 = 0.05  # panel c bins: 12% steps in distance, one edge at each cut

# Own logger, so that the font subsetter's INFO messages stay out of the job log.
LOG = logging.getLogger("fig_spatial_qc")

RELEASE_COLUMNS = [
    "sample",
    "timepoint",
    "cell_type",
    "x_um_dbscan",
    "y_um_dbscan",
    "x_um_qc",
    "y_um_qc",
    "spatial_qc_reason",
]
QC_COLUMNS = ["sample", "timepoint", "x_um_dbscan", "y_um_dbscan", "spatial_qc_reason"]
BOX_KEYS = {"timepoint", "x_gt", "x_lt", "y_gt", "y_lt"}


# Inputs ----


def read_h5ad_obs(path: Path, columns: list[str], uns_key: str | None = None):
    """Read obs columns (and one uns entry) of an h5ad without loading X."""
    # Imported here so that --tables renders need neither h5py nor anndata.
    import h5py
    from anndata.io import read_elem

    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        uns = read_elem(f["uns"][uns_key]) if uns_key else None

    missing = sorted(set(columns) - set(obs.columns))
    if missing:
        raise KeyError(f"{path} has no obs columns {missing}")

    obs = obs[columns].copy()
    obs.index.name = "barcode"
    return obs, uns


def normalize(obs: pd.DataFrame) -> pd.DataFrame:
    """Labels as str and coordinates as float, so h5ad and CSV inputs compare equal."""
    obs = obs.copy()
    for col in obs.columns:
        if col.startswith(("x_um", "y_um")):
            obs[col] = obs[col].astype(float)
        else:
            obs[col] = obs[col].astype(str)
    return obs


def plain(value):
    """numpy scalars and arrays from uns as JSON-ready Python values."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    return value.tolist() if hasattr(value, "tolist") else value


def load_inputs(args) -> tuple[dict, dict, pd.DataFrame, dict, list]:
    """Release obs and cell-type display names per species, the 04f mouse obs and its
    uns['spatial_qc'], and a record of every source file."""
    if args.tables:
        release, display_names = {}, {}
        for species in TIMEPOINTS:
            release[species] = pd.read_csv(
                args.tables / f"release_obs_{species}.csv.gz",
                index_col="barcode",
            )
            display_names[species] = json.loads(
                (args.tables / f"display_names_{species}.json").read_text()
            )
        qc_mouse = pd.read_csv(args.tables / "qc_mouse_obs.csv.gz", index_col="barcode")
        uns = json.loads((args.tables / "qc_mouse_uns_spatial_qc.json").read_text())
        sources = [
            file_record(path, role="CSV export (--export-tables)")
            for path in sorted(args.tables.glob("*_*.*"))
        ]
    else:
        objects = {"mouse": args.mouse_obj, "human": args.human_obj}
        sha256 = release_sha256(args.manifest) if args.manifest else {}
        release, display_names, sources = {}, {}, []
        for species, path in objects.items():
            release[species], names = read_h5ad_obs(path, RELEASE_COLUMNS, uns_key="display_names")
            display_names[species] = plain(names)
            sources.append(
                file_record(
                    path,
                    role=f"release object, {species}",
                    sha256=sha256.get(path.name),
                )
            )
        qc_mouse, uns = read_h5ad_obs(args.qc_mouse, QC_COLUMNS, uns_key="spatial_qc")
        uns = plain(uns)
        sources.append(
            file_record(args.qc_mouse, role="04f output: uns['spatial_qc'] and far-step inputs")
        )
        if args.manifest:
            sources.append(file_record(args.manifest, role="release manifest"))

    release = {species: normalize(obs) for species, obs in release.items()}
    return release, display_names, normalize(qc_mouse), uns, sources


def qc_parameters(uns: dict) -> dict:
    """The 04f parameters, with the exclusion regions and section lines decoded from JSON."""
    params = dict(uns)
    params["exclusions"] = json.loads(params.pop("exclusions_json"))
    params["sections"] = json.loads(params.pop("sections_json"))
    params["thresholds_um"] = {tp: float(v) for tp, v in params["thresholds_um"].items()}

    # far_step() below recomputes the cut per timepoint on log1p distances; any other
    # setting in uns means the figure would not reproduce what 04f applied.
    if params["mad_group"] != "timepoint" or not params["log_scale"]:
        raise ValueError(
            f"far step pooled by {params['mad_group']!r}, log_scale={params['log_scale']}"
        )
    for rule in params["exclusions"]:
        if set(rule) - BOX_KEYS:
            raise ValueError(f"exclusion region with keys this figure cannot draw: {rule}")
    return params


def export_tables(
    release: dict,
    display_names: dict,
    qc_mouse: pd.DataFrame,
    uns: dict,
    directory: Path,
) -> None:
    """Write the exports that --tables reads."""
    directory.mkdir(parents=True, exist_ok=True)
    for species, obs in release.items():
        obs.to_csv(directory / f"release_obs_{species}.csv.gz")
        (directory / f"display_names_{species}.json").write_text(
            json.dumps(display_names[species], indent=1)
        )
    qc_mouse.to_csv(directory / "qc_mouse_obs.csv.gz")
    (directory / "qc_mouse_uns_spatial_qc.json").write_text(json.dumps(uns, indent=1))
    LOG.info("Exported input tables to %s", directory)


# Checks ----


def check_release(obs: pd.DataFrame, species: str) -> None:
    """Fail unless every nucleus's coordinates agree with its QC label."""
    reason = obs["spatial_qc_reason"]
    unknown = sorted(set(reason) - set(REASONS))
    if unknown:
        raise ValueError(f"{species}: unknown spatial_qc_reason values {unknown}")

    has_raw = obs[["x_um_dbscan", "y_um_dbscan"]].notna().all(axis=1)
    has_qc = obs[["x_um_qc", "y_um_qc"]].notna().all(axis=1)
    if ((reason == "unmapped") != ~has_raw).any():
        raise ValueError(f"{species}: 'unmapped' does not match missing raw coordinates")
    if ((reason == "kept") != has_qc).any():
        raise ValueError(f"{species}: 'kept' does not match present QC coordinates")
    for axis in ("x", "y"):
        if not np.array_equal(
            obs.loc[has_qc, f"{axis}_um_qc"],
            obs.loc[has_qc, f"{axis}_um_dbscan"],
        ):
            raise ValueError(f"{species}: kept QC {axis} coordinates differ from the raw ones")

    # Panel c and the box outlines come from the mouse 04f object; human positions went through
    # the duplicate/crowding step only (methods, Spatial quality control).
    if species == "human" and reason.isin(["far", "boxed"]).any():
        raise ValueError(
            "human nuclei carry far/boxed labels; panel c and the boxes need a human 04f object"
        )

    LOG.info("%s: %s", species, reason.value_counts().reindex(REASONS, fill_value=0).to_dict())


def check_mouse_against_qc(release_mouse: pd.DataFrame, qc_mouse: pd.DataFrame) -> None:
    """Fail unless every released mouse nucleus has the 04f object's label and raw position."""
    absent = ~release_mouse.index.isin(qc_mouse.index)
    if absent.any():
        raise ValueError(f"{absent.sum()} released mouse barcodes are not in the 04f object")

    qc = qc_mouse.loc[release_mouse.index]
    if (qc["spatial_qc_reason"] != release_mouse["spatial_qc_reason"]).any():
        raise ValueError("released mouse QC labels differ from the 04f object")
    for col in ("x_um_dbscan", "y_um_dbscan"):
        if not np.array_equal(qc[col], release_mouse[col], equal_nan=True):
            raise ValueError(f"released mouse {col} differs from the 04f object")


def check_exclusion_regions(release_mouse: pd.DataFrame, params: dict) -> None:
    """Fail unless every 'boxed' position lies in a drawn exclusion region and no kept one does."""
    positioned = release_mouse[release_mouse["spatial_qc_reason"] != "unmapped"]
    in_region = pd.Series(False, index=positioned.index)
    for rule in params["exclusions"]:
        inside = positioned["timepoint"] == rule["timepoint"]
        if "x_gt" in rule:
            inside &= positioned["x_um_dbscan"] > rule["x_gt"]
        if "x_lt" in rule:
            inside &= positioned["x_um_dbscan"] < rule["x_lt"]
        if "y_gt" in rule:
            inside &= positioned["y_um_dbscan"] > rule["y_gt"]
        if "y_lt" in rule:
            inside &= positioned["y_um_dbscan"] < rule["y_lt"]
        in_region |= inside

    reason = positioned["spatial_qc_reason"]
    kept_inside = int((in_region & (reason == "kept")).sum())
    boxed_outside = int((~in_region & (reason == "boxed")).sum())
    if kept_inside or boxed_outside:
        raise ValueError(
            f"{kept_inside} kept positions inside, "
            f"{boxed_outside} boxed positions outside the regions"
        )


# Far step ----


def far_step(qc_mouse: pd.DataFrame, params: dict) -> tuple[pd.Series, dict]:
    """Recompute the 04f far-step score (µm) and cut per timepoint, as spatial_qc.py does.

    The score is the mean distance to the k nearest positions of the same timepoint among
    nuclei that survived the duplicate/crowding step (lanes pooled). The cut is the median
    plus n_mads * mad_scale MADs of log1p(score). Nuclei that were not scored get NaN.
    """
    k = int(params["k"])
    n_mads = float(params["n_mads"]) * float(params["mad_scale"])
    survived = qc_mouse["spatial_qc_reason"].isin(["kept", "far", "boxed"])

    score = pd.Series(np.nan, index=qc_mouse.index)
    cut = {}
    for timepoint, group in qc_mouse[survived].groupby("timepoint"):
        xy = group[["x_um_dbscan", "y_um_dbscan"]].to_numpy(float)

        # spatial_qc.py scores a group of k + 1 or fewer positions as 0, so none of it is far.
        if len(xy) <= k + 1:
            dist = np.zeros(len(xy))
        else:
            dist_k, _ = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy)
            dist = dist_k[:, 1:].mean(axis=1)

        log_dist = np.log1p(dist)
        median = np.median(log_dist)
        mad = np.median(np.abs(log_dist - median))
        cut[timepoint] = float(np.expm1(median + n_mads * mad))
        score[group.index] = dist

    return score, cut


def check_far_step(qc_mouse: pd.DataFrame, score: pd.Series, cut: dict, params: dict) -> None:
    """Fail unless the recomputed cut reproduces the 04f 'far' labels and stored thresholds."""
    is_far = score > qc_mouse["timepoint"].map(cut)
    mismatched = int((is_far != (qc_mouse["spatial_qc_reason"] == "far")).sum())
    if mismatched:
        raise ValueError(f"recomputed far step disagrees with 04f for {mismatched} nuclei")

    # uns stores the cuts rounded to 0.1 µm.
    stored = params["thresholds_um"]
    off = {
        tp: (round(cut[tp], 2), stored[tp])
        for tp in cut
        if abs(cut[tp] - stored[tp]) > 0.05 + 1e-9  # half the rounding step, plus float slack
    }
    if off or set(cut) != set(stored):
        raise ValueError(
            f"recomputed far cuts differ from uns thresholds_um: {off or set(cut) ^ set(stored)}"
        )
    LOG.info(
        "Far step reproduced: %d far nuclei, cuts %s",
        int(is_far.sum()),
        {tp: round(value, 1) for tp, value in cut.items()},
    )


# Tables ----


def lane_labels(obs: pd.DataFrame, species: str) -> dict:
    """'<timepoint> <letter>' per 10x lane; the letter is the sample name's last character."""
    labels = {}
    for timepoint in TIMEPOINTS[species]:
        lanes = sorted(obs.loc[obs["timepoint"] == timepoint, "sample"].unique())
        for lane in lanes:
            suffix = lane[-1] if len(lanes) > 1 else ""
            labels[lane] = f"{TIMEPOINT_LABELS[timepoint]} {suffix}".strip()

    unlabelled = sorted(set(obs["sample"]) - set(labels))
    if unlabelled:
        raise ValueError(f"{species}: lanes of timepoints missing from TIMEPOINTS: {unlabelled}")
    if len(set(labels.values())) < len(labels):
        raise ValueError(f"{species}: lane labels are not unique: {labels}")
    return labels


def outcome_counts(obs: pd.DataFrame, by: str) -> pd.DataFrame:
    """Nuclei per QC outcome (columns in REASONS order) for each value of `by`."""
    counts = pd.crosstab(obs[by], obs["spatial_qc_reason"])
    return counts.reindex(columns=REASONS, fill_value=0)


def format_count(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def format_tick_count(value: float, _position=None) -> str:
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def percent_range(fractions: pd.Series) -> str:
    return f"{100 * fractions.min():.0f}–{100 * fractions.max():.0f}%"


# Drawing helpers ----


def apply_style(font: str) -> None:
    plt.rcParams.update(
        {
            "font.family": [font, "DejaVu Sans"],
            "font.size": BASE_PT,
            "axes.titlesize": BASE_PT,
            "axes.labelsize": BASE_PT,
            "axes.titlelocation": "left",
            "xtick.labelsize": SMALL_PT,
            "ytick.labelsize": SMALL_PT,
            "legend.fontsize": SMALL_PT,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )


def choose_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next(name for name in FONT_PREFERENCE if name in available)


def panel_header(fig, ax, letter: str, title: str, dx_in: float, dy_in: float = 0.16) -> None:
    """Bold panel letter dx_in left of and dy_in above the top-left corner of `ax`, title beside
    it."""
    box = ax.get_position()
    x = box.x0 - dx_in / fig.get_figwidth()
    y = box.y1 + dy_in / fig.get_figheight()
    fig.text(x, y, letter, fontsize=LETTER_PT, fontweight="bold", va="bottom", ha="left")
    fig.text(x + 0.16 / fig.get_figwidth(), y, title, fontsize=BASE_PT, va="bottom", ha="left")


def fit_limits(x: np.ndarray, y: np.ndarray, box_aspect: float) -> tuple:
    """Equal-scale limits around the points, widened along one axis to fill a box whose
    height/width is box_aspect."""
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    pad = MAP_PAD * max(x1 - x0, y1 - y0)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad

    width, height = x1 - x0, y1 - y0
    if height / width < box_aspect:
        grow = (box_aspect * width - height) / 2
        y0, y1 = y0 - grow, y1 + grow
    else:
        grow = (height / box_aspect - width) / 2
        x0, x1 = x0 - grow, x1 + grow
    return (x0, x1), (y0, y1)


def exclusion_rectangle(rule: dict, xlim: tuple, ylim: tuple) -> Rectangle | None:
    """The part of an exclusion region inside the plotted limits; a missing bound is open."""
    x0 = max(rule.get("x_gt", -np.inf), xlim[0])
    x1 = min(rule.get("x_lt", np.inf), xlim[1])
    y0 = max(rule.get("y_gt", -np.inf), ylim[0])
    y1 = min(rule.get("y_lt", np.inf), ylim[1])
    if x0 >= x1 or y0 >= y1:
        return None
    return Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        edgecolor=REASON_COLORS["boxed"],
        linewidth=0.7,
        linestyle=(0, (3, 2)),
        zorder=4,
    )


def marker_area(n_nuclei):
    """Scatter area (pt^2) of a blanked marker holding n_nuclei; see BLANKED_BIN_UM."""
    return BLANKED_POINT_SIZE * np.sqrt(n_nuclei)


def blanked_markers(positions: pd.DataFrame, reason: str) -> pd.DataFrame:
    """Positions with one QC reason pooled per BLANKED_BIN_UM grid cell: mean x, y and count n."""
    blanked = positions[positions["spatial_qc_reason"] == reason]
    cell_x = (blanked["x_um_dbscan"] // BLANKED_BIN_UM).astype(int)
    cell_y = (blanked["y_um_dbscan"] // BLANKED_BIN_UM).astype(int)
    return blanked.groupby([cell_x, cell_y]).agg(
        x=("x_um_dbscan", "mean"),
        y=("y_um_dbscan", "mean"),
        n=("x_um_dbscan", "size"),
    )


def draw_section(ax, positions: pd.DataFrame, title: str, boxes: list, lines: list) -> dict:
    """Raw positions of one section by QC outcome, its exclusion regions and section boundary."""
    box = ax.get_position()
    box_aspect = (box.height * ax.figure.get_figheight()) / (box.width * ax.figure.get_figwidth())
    x = positions["x_um_dbscan"].to_numpy()
    y = positions["y_um_dbscan"].to_numpy()
    xlim, ylim = fit_limits(x, y, box_aspect)

    # Kept positions first, as the tissue outline. Point layers are bitmaps in the vector outputs.
    kept = positions[positions["spatial_qc_reason"] == "kept"]
    ax.scatter(
        kept["x_um_dbscan"],
        kept["y_um_dbscan"],
        s=KEPT_POINT_SIZE,
        color=REASON_COLORS["kept"],
        linewidths=0,
        rasterized=True,
    )

    # Blanked positions on top, one marker per grid cell sized by its nuclei (see BLANKED_BIN_UM);
    # the largest are drawn first so small markers stay visible, with a thin white rim between
    # overlapping ones. Unclipped, so a marker at the edge of a section is drawn whole.
    for reason in BLANKED:
        markers = blanked_markers(positions, reason).sort_values("n", ascending=False)
        ax.scatter(
            markers["x"],
            markers["y"],
            s=marker_area(markers["n"]),
            color=REASON_COLORS[reason],
            edgecolors="white",
            linewidths=0.2,
            rasterized=True,
            clip_on=False,
            zorder=3,
        )
        # Count label above and right of the marker, clear of markers along a section edge, with
        # a white outline where it crosses positions.
        for _, marker in markers[markers["n"] >= LABEL_MIN_NUCLEI].iterrows():
            ax.annotate(
                f"{int(marker['n']):,} nuclei",
                (marker["x"], marker["y"]),
                xytext=(6, 9),
                textcoords="offset points",
                color=REASON_COLORS[reason],
                fontsize=SMALL_PT,
                va="bottom",
                ha="left",
                path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")],
                zorder=5,
            )

    # Exclusion regions are half-open rectangles in the raw frame; only their plotted part is drawn.
    n_drawn = 0
    for rule in boxes:
        rectangle = exclusion_rectangle(rule, xlim, ylim)
        if rectangle is not None:
            ax.add_patch(rectangle)
            n_drawn += 1

    # Section boundary y = slope * x + intercept of a puck carrying two cross-sections.
    for section in lines:
        slope, intercept = section["line"]["slope"], section["line"]["intercept"]
        ax.plot(
            xlim,
            [slope * xlim[0] + intercept, slope * xlim[1] + intercept],
            color="0.25",
            linewidth=0.7,
            linestyle=(0, (1, 1.5)),
            zorder=4,
        )

    # 1 mm bar just below the map, clear of the positions: sections are drawn at different
    # scales to fill their boxes. x is in µm, y in axes fractions.
    bar_x = xlim[0] + 0.02 * (xlim[1] - xlim[0])
    under_map = ax.get_xaxis_transform()
    ax.plot(
        [bar_x, bar_x + SCALE_BAR_UM],
        [-0.04, -0.04],
        transform=under_map,
        color="0.15",
        linewidth=1.0,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        bar_x + SCALE_BAR_UM + 0.02 * (xlim[1] - xlim[0]),
        -0.04,
        f"{SCALE_BAR_UM / 1000:g} mm",
        transform=under_map,
        ha="left",
        va="center",
        fontsize=SMALL_PT,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=2)
    return {"exclusion_regions_drawn": n_drawn, "exclusion_regions_listed": len(boxes)}


def draw_outcome_bars(
    ax,
    counts: pd.DataFrame,
    labels: list[str],
    headers: dict | None = None,
) -> None:
    """One horizontal bar per row of counts, split into QC-outcome fractions, with n at the right.

    headers maps a row position to a group name drawn as an unbarred, bold row label.
    """
    headers = headers or {}
    n_rows = len(counts) + len(headers)
    bar_rows = [row for row in range(n_rows) if row not in headers]

    # Fractions, not counts, so lanes and cell types of very different size compare directly.
    fractions = counts.div(counts.sum(axis=1), axis=0)
    left = np.zeros(len(fractions))
    for reason in REASONS:
        ax.barh(
            bar_rows,
            fractions[reason],
            left=left,
            height=0.78,
            color=REASON_COLORS[reason],
            linewidth=0,
        )
        left += fractions[reason].to_numpy()

    for row, n in zip(bar_rows, counts.sum(axis=1)):
        ax.text(1.015, row, format_count(int(n)), va="center", ha="left", fontsize=SMALL_PT)

    tick_labels = dict(zip(bar_rows, labels)) | headers
    ax.set_yticks(range(n_rows), labels=[tick_labels[row] for row in range(n_rows)])
    for row, tick in enumerate(ax.get_yticklabels()):
        if row in headers:
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(n_rows - 0.4, -0.6)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1]))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.spines["left"].set_visible(False)


# Figure ----


def draw_figure(
    release: dict,
    display_names: dict,
    params: dict,
    score: pd.Series,
    cut: dict,
    labels: dict,
) -> tuple:
    """Lay out and draw panels a-d. Returns the figure and the counts behind each panel."""
    fig = plt.figure(figsize=FIG_SIZE_IN)
    summary = {}

    # Grids are placed by hand in figure fractions so that the map boxes have a known aspect
    # before the positions are drawn into them.
    maps_grid = fig.add_gridspec(
        2,
        5,
        left=0.035,
        right=0.99,
        top=0.885,
        bottom=0.552,
        width_ratios=[1, 1, 1, 0.08, 1],
        wspace=0.06,
        hspace=0.26,
    )
    middle_grid = fig.add_gridspec(
        1,
        2,
        left=0.105,
        right=0.985,
        top=0.484,
        bottom=0.29,
        width_ratios=[1, 1.12],
        wspace=0.30,
    )
    far_grid = middle_grid[1].subgridspec(2, 3, wspace=0.12, hspace=0.42)
    bottom_grid = fig.add_gridspec(
        1,
        2,
        left=0.205,
        right=0.945,
        top=0.178,
        bottom=0.045,
        wspace=0.92,
    )

    # Legend ----

    # One key for the outcome colors of every panel, plus the two map annotations.
    handles = [
        Patch(color=REASON_COLORS[reason], label=REASON_LABELS[reason]) for reason in REASONS
    ]
    handles += [
        Patch(
            fill=False,
            edgecolor=REASON_COLORS["boxed"],
            linestyle=(0, (3, 2)),
            linewidth=0.7,
            label="Exclusion region",
        ),
        Line2D(
            [],
            [],
            color="0.25",
            linewidth=0.7,
            linestyle=(0, (1, 1.5)),
            label="Section boundary",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.03, 0.995),
        ncol=4,
        handlelength=1.4,
        columnspacing=1.4,
        handletextpad=0.5,
        borderaxespad=0,
    )

    # Panel a: positions by QC outcome ----

    # Mouse sections fill a 3 x 2 block, the two human sections the column on the right; each
    # timepoint's lanes share one puck and coordinate frame, so they are drawn together.
    slots = {
        "mouse": [maps_grid[row, col] for row in range(2) for col in range(3)],
        "human": [maps_grid[row, 4] for row in range(2)],
    }
    exclusion_summary = {}
    first_map = {}
    for species, timepoints in TIMEPOINTS.items():
        obs = release[species]
        for slot, timepoint in zip(slots[species], timepoints):
            ax = fig.add_subplot(slot)
            first_map.setdefault(species, ax)
            positions = obs[
                (obs["timepoint"] == timepoint) & (obs["spatial_qc_reason"] != "unmapped")
            ]
            # Exclusion regions and section lines exist for mouse only (04f ran for mouse).
            boxes, lines = [], []
            if species == "mouse":
                boxes = [r for r in params["exclusions"] if r["timepoint"] == timepoint]
                lines = [s for s in params["sections"] if s["timepoint"] == timepoint]
            exclusion_summary[f"{species}/{timepoint}"] = draw_section(
                ax,
                positions,
                TIMEPOINT_LABELS[timepoint],
                boxes,
                lines,
            )
    summary["exclusion_regions"] = exclusion_summary

    # Species headers sit just above the map titles; the panel title one line higher.
    panel_header(
        fig,
        first_map["mouse"],
        "a",
        "Slide-tags positions by QC outcome, one map per section",
        dx_in=0.08,
        dy_in=0.34,
    )
    for species, ax in first_map.items():
        box = ax.get_position()
        fig.text(
            box.x0,
            box.y1 + 0.17 / fig.get_figheight(),
            species.capitalize(),
            fontsize=BASE_PT,
            fontweight="bold",
            va="bottom",
        )

    # Size key for the pooled blanked markers, at the right end of panel a's title line, the
    # only panel that uses it. Legend marker size is a diameter, the square root of the area.
    size_handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=np.sqrt(marker_area(n)),
            markerfacecolor="0.45",
            markeredgecolor="white",
            markeredgewidth=0.2,
            label=f"{n:,}",
        )
        for n in SIZE_KEY_NUCLEI
    ]
    fig.legend(
        handles=size_handles,
        title="Blanked nuclei per marker (area ∝ √n)",
        title_fontsize=SMALL_PT,
        loc="lower right",
        bbox_to_anchor=(0.99, first_map["human"].get_position().y1 + 0.3 / fig.get_figheight()),
        ncol=len(SIZE_KEY_NUCLEI),
        handletextpad=0.3,
        columnspacing=1.2,
        borderaxespad=0,
    )

    # Panel b: outcome per 10x lane ----

    ax_lanes = fig.add_subplot(middle_grid[0])
    lane_counts, lane_names, headers, row = [], [], {}, 0
    for species in TIMEPOINTS:
        counts = outcome_counts(release[species], "sample").loc[list(labels[species])]
        headers[row] = species.capitalize()
        row += 1 + len(counts)
        lane_counts.append(counts)
        lane_names += [labels[species][lane] for lane in counts.index]
    lane_counts = pd.concat(lane_counts)
    draw_outcome_bars(ax_lanes, lane_counts, lane_names, headers)
    ax_lanes.set_xlabel("Nuclei in lane")

    kept_fraction = lane_counts["kept"] / lane_counts.sum(axis=1)
    panel_header(
        fig,
        ax_lanes,
        "b",
        f"{percent_range(kept_fraction)} of each lane's nuclei keep a position",
        dx_in=0.62,
    )
    summary["lanes"] = lane_counts

    # Panel c: far-from-neighbors cut, mouse ----

    # Only nuclei that survived the duplicate/crowding step are scored. Bins have equal width in
    # log distance with one edge at the cut, so every bin lies wholly on one side of it.
    mouse = release["mouse"].assign(far_score=score.reindex(release["mouse"].index))
    scored = mouse[mouse["spatial_qc_reason"].isin(["kept", "far", "boxed"])]
    log_lo = np.log10(scored["far_score"].min())
    log_hi = np.log10(scored["far_score"].max())
    far_axes, far_rows, y_max = [], [], 1
    for i, timepoint in enumerate(TIMEPOINTS["mouse"]):
        ax = fig.add_subplot(far_grid[i // 3, i % 3])
        far_axes.append(ax)
        dist = scored.loc[scored["timepoint"] == timepoint, "far_score"].to_numpy()
        n_far = int(
            (mouse.loc[mouse["timepoint"] == timepoint, "spatial_qc_reason"] == "far").sum()
        )
        if int((dist > cut[timepoint]).sum()) != n_far:
            raise ValueError(f"{timepoint}: nuclei beyond the cut do not match the 'far' labels")

        steps = np.arange(
            np.floor((log_lo - np.log10(cut[timepoint])) / FAR_BIN_LOG10),
            np.ceil((log_hi - np.log10(cut[timepoint])) / FAR_BIN_LOG10) + 1,
        )
        edges = cut[timepoint] * 10 ** (FAR_BIN_LOG10 * steps)
        counts, _ = np.histogram(dist, bins=edges)
        y_max = max(y_max, counts.max())
        beyond = edges[:-1] >= cut[timepoint]

        # Step outlines, not filled bars: the count axis is logarithmic so that the sparse tail
        # beyond the cut is visible next to the bulk. Empty bins drop below the axis floor.
        for side, color in ((~beyond, "0.35"), (beyond, REASON_COLORS["far"])):
            idx = np.flatnonzero(side)
            if len(idx):
                ax.stairs(
                    np.where(counts[idx] > 0, counts[idx], 0.1),
                    edges[idx[0] : idx[-1] + 2],
                    baseline=0.1,
                    color=color,
                    linewidth=0.8,
                )
        # The cut line stops below the corner label.
        ax.axvline(
            cut[timepoint],
            ymax=0.8,
            color=REASON_COLORS["far"],
            linewidth=0.7,
            linestyle=(0, (3, 2)),
        )

        # Cut and number of far nuclei in the top-right corner, above the sparse tail.
        ax.text(
            0.98,
            0.97,
            f"{n_far} far (> {cut[timepoint]:.0f} µm)",
            transform=ax.transAxes,
            color=REASON_COLORS["far"],
            fontsize=SMALL_PT,
            va="top",
            ha="right",
        )
        ax.set_title(TIMEPOINT_LABELS[timepoint], loc="left", pad=2)
        far_rows.append(
            {
                "timepoint": timepoint,
                "n_scored": len(dist),
                "cut_um": round(cut[timepoint], 2),
                "n_far": n_far,
            }
        )

    # Shared log axes so the timepoints compare directly; tick labels only on the outer panels.
    for i, ax in enumerate(far_axes):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10 ** (log_lo - 0.05), 10 ** (log_hi + 0.05))
        # Headroom above the tallest bin for the corner label.
        ax.set_ylim(0.7, y_max * 8)
        ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.yaxis.set_major_formatter(FuncFormatter(format_tick_count))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if i % 3:
            ax.tick_params(axis="y", labelleft=False)
        if i < 3:
            ax.tick_params(axis="x", labelbottom=False)
    # One y label centered on the two rows, left of the tick labels.
    top_box, bottom_box = far_axes[0].get_position(), far_axes[3].get_position()
    fig.text(
        top_box.x0 - 0.3 / fig.get_figwidth(),
        (top_box.y1 + bottom_box.y0) / 2,
        "Nuclei per bin",
        rotation=90,
        ha="center",
        va="center",
        fontsize=BASE_PT,
    )
    far_axes[4].set_xlabel(f"Mean distance to {int(params['k'])} nearest positions (µm)")

    n_mads = float(params["n_mads"]) * float(params["mad_scale"])
    panel_header(
        fig,
        far_axes[0],
        "c",
        f"Mouse far cut: median + {n_mads:g} MADs of log distance, per timepoint",
        dx_in=0.36,
    )
    summary["far_cut"] = pd.DataFrame(far_rows)

    # Panel d: outcome per cell type ----

    # Cell types sorted by the fraction keeping a position; the dashed line is the species-wide
    # fraction, so a cell type left of it is under-represented in the spatial maps.
    cell_counts = {}
    for i, species in enumerate(TIMEPOINTS):
        ax = fig.add_subplot(bottom_grid[i])
        counts = outcome_counts(release[species], "cell_type")
        kept = counts["kept"] / counts.sum(axis=1)
        counts = counts.loc[kept.sort_values(ascending=False).index]
        unnamed = sorted(set(counts.index) - set(display_names[species]["cell_type"]))
        if unnamed:
            raise KeyError(f"{species}: no display name in uns['display_names'] for {unnamed}")
        draw_outcome_bars(
            ax,
            counts,
            [display_names[species]["cell_type"][c] for c in counts.index],
        )

        # An empty band above the first bar holds the reference line's label.
        overall = counts["kept"].sum() / counts.to_numpy().sum()
        ax.set_ylim(len(counts) - 0.4, -1.5)
        ax.vlines(
            overall,
            len(counts) - 0.4,
            -0.6,
            color="0.15",
            linewidth=0.7,
            linestyle=(0, (2, 2)),
            zorder=3,
        )
        ax.text(overall, -1.0, "all nuclei", ha="center", va="center", fontsize=SMALL_PT)
        ax.set_xlabel("Nuclei of cell type")
        ax.set_title(
            f"{species.capitalize()}: {percent_range(kept.loc[counts.index])} kept "
            f"(all nuclei {100 * overall:.0f}%)",
            loc="left",
            pad=2,
        )
        if i == 0:
            panel_header(
                fig,
                ax,
                "d",
                "Fraction of each cell type keeping a position",
                dx_in=1.36,
                dy_in=0.24,
            )
        cell_counts[species] = counts
    summary["cell_types"] = cell_counts

    return fig, summary


# Provenance ----


def file_record(path: Path, role: str, **extra) -> dict:
    stat = path.stat()
    record = {
        "path": str(path.resolve()),
        "role": role,
        "bytes": stat.st_size,
        "mtime": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }
    return record | {key: value for key, value in extra.items() if value is not None}


def release_sha256(manifest: Path) -> dict:
    """sha256 per release file name, as recorded by finalize_deliverables (04g)."""
    objects = json.loads(manifest.read_text())["objects"]
    return {entry["file"]: entry["sha256"] for entry in objects.values()}


def script_revision() -> dict | None:
    """Commit of this script's checkout and whether the script matches it; None outside git."""
    here = Path(__file__).resolve()
    git = ["git", "-C", str(here.parent)]
    try:
        commit = subprocess.run(
            git + ["rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        status = subprocess.run(
            git + ["status", "--porcelain", "--", here.name],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return {"commit": commit.stdout.strip(), "script_matches_commit": status.stdout.strip() == ""}


# Main ----


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mouse-obj", type=Path, help="release all-cells mouse object (.h5ad)")
    parser.add_argument("--human-obj", type=Path, help="release all-cells human object (.h5ad)")
    parser.add_argument(
        "--qc-mouse", type=Path, help="04f output data/interim/qc_spatial_mouse.h5ad"
    )
    parser.add_argument(
        "--manifest", type=Path, help="release MANIFEST.json, for the objects' sha256"
    )
    parser.add_argument(
        "--tables", type=Path, help="read the CSV exports of --export-tables instead"
    )
    parser.add_argument(
        "--export-tables", type=Path, help="also write the CSV exports to this directory"
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()
    if args.tables is None and None in (args.mouse_obj, args.human_obj, args.qc_mouse):
        parser.error("give --mouse-obj, --human-obj and --qc-mouse, or --tables")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(message)s")
    LOG.setLevel(logging.INFO)

    figure_files = [args.out / f"{STEM}.{ext}" for ext in ("png", "pdf", "svg")]
    sidecar = args.out / f"{STEM}.provenance.json"
    existing = [str(p) for p in figure_files + [sidecar] if p.exists()]
    if existing and not args.force:
        raise SystemExit(f"Refusing to overwrite {existing}; pass --force")
    args.out.mkdir(parents=True, exist_ok=True)

    release, display_names, qc_mouse, uns, sources = load_inputs(args)
    if args.export_tables:
        export_tables(release, display_names, qc_mouse, uns, args.export_tables)

    params = qc_parameters(uns)
    for species, obs in release.items():
        check_release(obs, species)
    check_mouse_against_qc(release["mouse"], qc_mouse)
    check_exclusion_regions(release["mouse"], params)

    score, cut = far_step(qc_mouse, params)
    check_far_step(qc_mouse, score, cut, params)

    labels = {species: lane_labels(obs, species) for species, obs in release.items()}
    font = choose_font()
    apply_style(font)
    fig, summary = draw_figure(release, display_names, params, score, cut, labels)
    for path in figure_files:
        fig.savefig(path)
    plt.close(fig)
    LOG.info("Wrote %s", [str(p) for p in figure_files])

    # Counts behind panels b-d, one CSV each.
    all_labels = {
        lane: label
        for species_labels in labels.values()
        for lane, label in species_labels.items()
    }
    lane_table = summary["lanes"].copy()
    lane_table.insert(0, "lane_label", lane_table.index.map(all_labels))
    lane_table.to_csv(args.out / f"{STEM}_counts_by_lane.csv")
    cell_table = pd.concat(summary["cell_types"], names=["species", "cell_type"])
    cell_table.to_csv(args.out / f"{STEM}_counts_by_cell_type.csv")
    summary["far_cut"].to_csv(args.out / f"{STEM}_far_cut_by_timepoint.csv", index=False)

    record = {
        "figure": str(figure_files[0].resolve()),
        "files": [str(p.resolve()) for p in figure_files],
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
        "git": script_revision(),
        "font": font,
        "dpi": plt.rcParams["savefig.dpi"],
        "sources": sources,
        "note": (
            "Raw Slide-tags positions (x_um_dbscan, y_um_dbscan) in their own unrotated frame, "
            "in which the exclusion regions and section boundary are defined. Lanes of one "
            "timepoint share a puck and frame and are pooled. All nuclei of the release objects "
            "are included; none are hidden, including the embedding outliers of 2026-10-22. "
            "Human positions went through the duplicate/crowding step only."
        ),
        "qc_parameters": params,
        "far_step_recomputed": {"cut_um": cut, "reproduces_04f_far_labels": True},
        "exclusion_regions": summary["exclusion_regions"],
        "lane_labels": labels,
        "counts_by_reason": {
            species: (
                obs["spatial_qc_reason"].value_counts().reindex(REASONS, fill_value=0).to_dict()
            )
            for species, obs in release.items()
        },
    }
    sidecar.write_text(json.dumps(record, indent=1, default=str))
    LOG.info("Wrote %s", sidecar)


if __name__ == "__main__":
    main()
