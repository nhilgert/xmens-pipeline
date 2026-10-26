"""
spatial_qc.py — "blank, don't drop" spatial-coordinate QC on the concatenated
per-species object (stage 01d).

Faithful port of notebooks/01b_spatial_qc_mouse.ipynb (mouse path). PURELY
spatial: only coordinates drive it (kNN distance, exact duplicates, hand-drawn
boxes). Cell-type annotations ride along on the object but are NOT an input to
any decision. A cell with an untrustworthy coordinate has its coordinate set to
NaN; NO cell is removed, so expression/annotation/DE and all cell counts are
untouched. Only positions change.

Placed AFTER annotation: reads the annotated object so the QC'd coordinates land
on the same object the downstream spatial arms consume.

Reads : data/interim/annotated_{species}.h5ad   (post-annotation object; carries
        obs x_um_dbscan / y_um_dbscan / sample / timepoint + obsm['spatial'])
Writes: data/interim/qc_spatial_{species}.h5ad  (original coords preserved;
        QC'd + oriented coords + tissue_section added alongside)
        results/tables/spatial_qc_{species}.csv  (per-timepoint tally)

Stages, in order (each reported):
  unmapped   no coordinate to begin with            -> kept as-is
  degenerate exact dup coords (per sample) OR kNN mean dist < floor_um
             -> runs FIRST so pile-ups cannot corrupt the MAD downstream
  far        kNN mean dist > median + n_mads*mad_scale*MAD on log1p, per mad_group,
             recomputed on degenerate-survivors
  boxed      hand-drawn exclusion regions the statistics miss
  sections   label each kept cell's physical cross-section: '{sample}__A' by default;
             timepoints listed in config sections get split by a straight line
             (y > slope*x + intercept -> 'above' label, else 'below')
  blank      everything not 'kept' loses its coordinate
  rotate     per-puck rigid rotation (rotations config; 0 = identity)

All parameters come from config['spatial_qc']; nothing is hardcoded here.
"""
import json
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

# ---- snakemake plumbing ----------------------------------------------------
IN     = snakemake.input.h5ad                       # noqa: F821
OUT    = snakemake.output.h5ad                      # noqa: F821
TALLY  = snakemake.output.tally                      # noqa: F821
P      = dict(snakemake.params.sqc)                  # noqa: F821
SPECIES = snakemake.wildcards.species                # noqa: F821
LOG = open(snakemake.log[0], "w")                    # noqa: F821


def say(*a):
    print(*a, file=LOG, flush=True)
    print(*a, flush=True)


# ---- config ----------------------------------------------------------------
X, Y         = P.get("x_key", "x_um_dbscan"), P.get("y_key", "y_um_dbscan")
SAMPLE       = P.get("sample_key", "sample")
TIMEPOINT    = P.get("timepoint_key", "timepoint")
K            = int(P.get("k", 5))
FLOOR_UM     = float(P.get("floor_um", 10.0))
MAX_AT_COORD = int(P.get("max_at_coord", 1))
N_MADS       = float(P.get("n_mads", 3.0))
MAD_SCALE    = float(P.get("mad_scale", 1.4826))   # 1.4826 = 1/Phi^-1(0.75): n_mads in sigma units; 1.0 = plain MADs
MAD_GROUP    = P.get("mad_group", TIMEPOINT)
if MAD_GROUP in ("timepoint", "TIMEPOINT"):
    MAD_GROUP = TIMEPOINT
elif MAD_GROUP in ("sample", "SAMPLE"):
    MAD_GROUP = SAMPLE
TP_ORDER = P.get("tp_order", {}).get(SPECIES, [])
EXCL     = P.get("exclusions", {}).get(SPECIES, []) or []
ROT      = P.get("rotations", {}).get(SPECIES, {}) or {}
SECT     = P.get("sections", {}).get(SPECIES, []) or []   # line splits for multi-section pucks

say(f"[spatial_qc] species={SPECIES}  X={X} Y={Y}  k={K} floor={FLOOR_UM} "
    f"max_at_coord={MAX_AT_COORD} n_mads={N_MADS} mad_scale={MAD_SCALE} mad_group={MAD_GROUP}")
say(f"[spatial_qc] {len(EXCL)} exclusion box(es); rotations="
    f"{ {k: v for k, v in ROT.items()} }; {len(SECT)} section split(s)")

# ---- load ------------------------------------------------------------------
adata = sc.read_h5ad(IN)
obs = adata.obs
for col in (X, Y, SAMPLE, TIMEPOINT):
    if col not in obs.columns:
        raise KeyError(f"obs missing required column {col!r}; have {list(obs.columns)[:40]}")
say(f"{SPECIES}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")


def knn_mean_dist(xy, k=K):
    if xy.shape[0] <= k + 1:
        return np.zeros(xy.shape[0])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    d, _ = nn.kneighbors(xy)
    return d[:, 1:].mean(axis=1)


# ---- stage: unmapped -------------------------------------------------------
reason = pd.Series("kept", index=obs.index, dtype=object)
mapped = (obs[X].notna() & obs[Y].notna()).values
reason[~mapped] = "unmapped"
say(f"unmapped {int((~mapped).sum()):,} / {len(obs):,} ({100*(~mapped).mean():.1f}%)")

# ---- stage: degenerate (per sample; runs first) ----------------------------
score0 = pd.Series(np.nan, index=obs.index)
for s in obs.loc[mapped, SAMPLE].astype(str).unique():
    m = mapped & (obs[SAMPLE].astype(str) == s).values
    score0.loc[m] = knn_mean_dist(obs.loc[m, [X, Y]].to_numpy(float))

key = (obs[SAMPLE].astype(str) + "|" + obs[X].astype(str) + "|" + obs[Y].astype(str))
n_at = key.map(key[mapped].value_counts()).fillna(0)
degen = mapped & ((n_at > MAX_AT_COORD).values | (score0 < FLOOR_UM).values)
reason[degen] = "degenerate"
say(f"degenerate {int(degen.sum()):,}  "
    f"(exact dups {int((mapped & (n_at > MAX_AT_COORD).values).sum()):,}, "
    f"< {FLOOR_UM}um {int((mapped & (score0 < FLOOR_UM).values).sum()):,})")

# ---- stage: far (per mad_group; recomputed on survivors) -------------------
surv = mapped & ~degen
score = pd.Series(np.nan, index=obs.index)
thr = {}
for g in obs.loc[surv, MAD_GROUP].astype(str).unique():
    m = surv & (obs[MAD_GROUP].astype(str) == g).values
    d = knn_mean_dist(obs.loc[m, [X, Y]].to_numpy(float))
    score.loc[m] = d
    lv = np.log1p(d)
    med = np.median(lv)
    mad = np.median(np.abs(lv - med))
    thr[g] = float(np.expm1(med + N_MADS * MAD_SCALE * mad))
    reason[m & (score > thr[g]).values] = "far"
say("far thresholds (um): " + json.dumps({k: round(v, 1) for k, v in thr.items()}))
say(f"far {int((reason == 'far').sum()):,}")


# ---- stage: boxed (hand-drawn regions) -------------------------------------
def rule_mask(r, base):
    m = base.copy()
    if "timepoint" in r:
        m &= (obs[TIMEPOINT].astype(str) == r["timepoint"]).values
    if "sample" in r:
        m &= (obs[SAMPLE].astype(str) == r["sample"]).values
    for k_, col, op in [("x_gt", X, ">"), ("x_lt", X, "<"), ("y_gt", Y, ">"), ("y_lt", Y, "<")]:
        if k_ in r:
            m &= (obs[col] > r[k_]).values if op == ">" else (obs[col] < r[k_]).values
    return m


kept0 = (reason == "kept").values
boxed = np.zeros(len(obs), bool)
for r in EXCL:
    m = rule_mask(dict(r), kept0)
    boxed |= m
    say(f"  box {dict(r)} -> {int(m.sum())} cells")
reason[boxed] = "boxed"
say(f"boxed {int(boxed.sum()):,}")

# ---- blank -----------------------------------------------------------------
ORDER = ["kept", "far", "degenerate", "boxed", "unmapped"]
kept = (reason == "kept").values
x_qc = np.where(kept, obs[X].astype(float), np.nan)
y_qc = np.where(kept, obs[Y].astype(float), np.nan)

tal = reason.value_counts().reindex(ORDER).fillna(0).astype(int)
say("\ntally:\n" + pd.DataFrame({"n": tal, "%": (100 * tal / len(obs)).round(1)}).to_string())
say(f"\nusable coordinates: {int(kept.sum()):,} "
    f"({100*kept.sum()/max(mapped.sum(),1):.1f}% of originally-mapped); "
    f"object unchanged at {len(obs):,} cells")


# ---- stage: sections (physical cross-sections per puck) --------------------
# Default: one section per sample -> "{sample}__A". Timepoints with several sections on one
# puck are split by a straight line in the QC'd frame: y > slope*x + intercept -> `above`
# label, else `below`. Blanked cells get no section. Deterministic, one parameter, plots.
section = pd.Series(pd.NA, index=obs.index, dtype=object)
section[kept] = obs.loc[kept, SAMPLE].astype(str) + "__A"
for r in SECT:
    r = dict(r); ln = dict(r["line"])
    m = kept & (obs[TIMEPOINT].astype(str) == str(r["timepoint"])).values
    above = (obs[Y].astype(float) > float(ln["slope"]) * obs[X].astype(float) + float(ln["intercept"])).values
    la, lb = str(r.get("above", "A")), str(r.get("below", "B"))
    section[m & above]  = obs.loc[m & above,  SAMPLE].astype(str) + "__" + la
    section[m & ~above] = obs.loc[m & ~above, SAMPLE].astype(str) + "__" + lb
    say(f"  section split {r['timepoint']}: y > {ln['slope']}*x + {ln['intercept']} -> "
        f"{la}={int((m & above).sum()):,}  {lb}={int((m & ~above).sum()):,}")
sec_counts = section.dropna().value_counts().sort_index()
small = sec_counts[sec_counts < 200]
if len(small):
    say(f"!! sections with < 200 cells (check the split line): {small.to_dict()}")
say("sections: " + json.dumps({k: int(v) for k, v in sec_counts.items()}))

# ---- rotate (per puck; 0 = identity) ---------------------------------------
miss = set(obs.loc[mapped, TIMEPOINT].astype(str).unique()) - set(ROT)
if miss:
    say(f"!! no rotation angle for {sorted(miss)} -- defaulting to 0")


def rotate_about(x, y, deg, cx, cy):
    if deg % 360 == 0:
        return x.copy(), y.copy()
    r = np.radians(deg)
    ca, sa = np.cos(r), np.sin(r)
    dx, dy = x - cx, y - cy
    return cx + ca * dx - sa * dy, cy + sa * dx + ca * dy


x_rot, y_rot = x_qc.copy(), y_qc.copy()
for s in obs.loc[mapped, SAMPLE].astype(str).unique():
    m = mapped & (obs[SAMPLE].astype(str) == s).values
    mk = m & kept
    tp = obs.loc[m, TIMEPOINT].astype(str).iloc[0]
    deg = float(ROT.get(tp, 0.0))
    src = mk if mk.sum() >= 20 else m
    cx, cy = float(np.nanmedian(obs.loc[src, X])), float(np.nanmedian(obs.loc[src, Y]))
    nx, ny = rotate_about(x_qc[m], y_qc[m], deg, cx, cy)
    x_rot[m], y_rot[m] = nx, ny

# ---- write (original coords preserved; QC'd + rotated added) ---------------
adata.obs["spatial_qc_reason"] = pd.Categorical(reason.values, categories=ORDER)
adata.obs["x_um_qc"], adata.obs["y_um_qc"] = x_qc, y_qc
adata.obs["x_um_rot"], adata.obs["y_um_rot"] = x_rot, y_rot
adata.obs["tissue_section"] = pd.Categorical(section.astype(object).where(section.notna(), None))
adata.obsm["spatial"] = np.c_[x_qc, y_qc]        # QC'd, unrotated (overwrites raw obsm)
adata.obsm["spatial_rot"] = np.c_[x_rot, y_rot]  # QC'd + oriented (= QC'd when all angles 0)
adata.uns["spatial_qc"] = {
    "k": K, "floor_um": FLOOR_UM, "max_at_coord": MAX_AT_COORD, "n_mads": N_MADS, "mad_scale": MAD_SCALE,
    "mad_group": MAD_GROUP, "log_scale": True, "action": "blank",
    "drop_unmapped": False, "x_key": X, "y_key": Y,
    "thresholds_um": {k_: round(v, 1) for k_, v in thr.items()},
    "exclusions_json": json.dumps([dict(r) for r in EXCL]),
    "rotations_json": json.dumps({k: float(v) for k, v in ROT.items()}),
    "sections_json": json.dumps([dict(r) for r in SECT], default=str),
}
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
adata.write_h5ad(OUT)

# ---- tally CSV -------------------------------------------------------------
ct = (pd.crosstab(obs[TIMEPOINT].astype(str), reason)
        .reindex(columns=ORDER, fill_value=0))
ct.loc["__total__"] = ct.sum(axis=0)
os.makedirs(os.path.dirname(TALLY), exist_ok=True)
ct.to_csv(TALLY)
say(f"\nwrote {OUT}")
say(f"wrote {TALLY}")
say("  obs: spatial_qc_reason, x_um_qc/y_um_qc, x_um_rot/y_um_rot, tissue_section")
say("  obsm: spatial (QC'd), spatial_rot (QC'd + oriented)")
say("  original x_um_dbscan/y_um_dbscan UNCHANGED")
LOG.close()
