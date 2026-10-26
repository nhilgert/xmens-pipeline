"""Stage 00b — build a per-sample AnnData from CellBender counts + Slide-tags coords.

Reads the CellBender-corrected count matrix (.h5, no metadata), writes
species/timepoint from the sample sheet into obs (the .h5 carries neither, and
human objects had timepoint=NaN — the sheet is authoritative), joins the DBSCAN
spatial coordinates from the fork's barcode-keyed coords CSV (Source 1), and
keeps ALL cells (coords are nullable obs columns; unplaced cells stay).

See DATA_PROVENANCE.md sections 2-3 for source/format details.
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc

counts_path = snakemake.input.counts          # noqa: F821 (snakemake-injected)
out_path = snakemake.output.h5ad               # noqa: F821
sample = snakemake.wildcards.sample            # noqa: F821
species = snakemake.params.species             # noqa: F821
timepoint = snakemake.params.timepoint         # noqa: F821
modality = snakemake.params.modality           # noqa: F821
coords_id = snakemake.params.coords_id         # noqa: F821
coords_dir = snakemake.params.coords_dir       # noqa: F821
coords_pattern = snakemake.params.coords_pattern   # noqa: F821
coords_x = snakemake.params.coords_x           # noqa: F821 (e.g. "dbscan_1")
coords_y = snakemake.params.coords_y           # noqa: F821 (e.g. "dbscan_2")
logfile = snakemake.log[0]                      # noqa: F821


with open(logfile, "w") as log:
    def say(msg):
        print(f"[ingest:{sample}] {msg}", file=log, flush=True)

    # --- counts (CellBender .h5) ---
    say(f"reading CellBender counts: {counts_path}")
    adata = sc.read_10x_h5(counts_path)
    adata.var_names_make_unique()   # gene symbols have duplicates
    n0 = adata.n_obs
    say(f"loaded {n0} cells x {adata.n_vars} genes")

    # Preserve corrected counts as a layer so downstream (scVI) has raw input
    # even after any future normalization of X.
    adata.layers["counts"] = adata.X.copy()

    # --- metadata from the sample sheet (authoritative) ---
    adata.obs["sample"] = sample
    adata.obs["species"] = species
    adata.obs["timepoint"] = timepoint
    adata.obs["modality"] = modality
    adata.obs["coords_id"] = coords_id
    for c in ["sample", "species", "timepoint", "modality", "coords_id"]:
        adata.obs[c] = adata.obs[c].astype("category")

    # --- coords (Source 1: barcode-keyed fork export) ---
    coords_path = os.path.join(coords_dir, coords_pattern.format(coords_id=coords_id))
    say(f"reading coords: {coords_path} (x={coords_x}, y={coords_y})")
    co = pd.read_csv(coords_path)

    if "barcode" not in co.columns:
        raise KeyError(f"coords CSV {coords_path} has no 'barcode' column; got {list(co.columns)}")
    for col in (coords_x, coords_y):
        if col not in co.columns:
            raise KeyError(f"coords CSV {coords_path} missing '{col}'; got {list(co.columns)}")

    co = co.set_index("barcode")

    # Direct barcode join (suffixes already match, verified in provenance). If a
    # future sample mismatches, the overlap count below will reveal it.
    common = adata.obs_names.intersection(co.index)
    say(f"barcode overlap: {len(common)}/{n0} cells have a coords row")
    if len(common) == 0:
        raise ValueError(
            f"no barcode overlap between counts and coords for {sample}. "
            f"counts e.g. {list(adata.obs_names[:3])}; coords e.g. {list(co.index[:3])}"
        )

    # Attach DBSCAN coords as nullable obs columns; unplaced cells stay as NaN.
    x = pd.Series(np.nan, index=adata.obs_names, dtype=float)
    y = pd.Series(np.nan, index=adata.obs_names, dtype=float)
    x.loc[common] = pd.to_numeric(co.loc[common, coords_x], errors="coerce")
    y.loc[common] = pd.to_numeric(co.loc[common, coords_y], errors="coerce")
    adata.obs["x_um_dbscan"] = x.to_numpy()
    adata.obs["y_um_dbscan"] = y.to_numpy()

    n_placed = int((~adata.obs["x_um_dbscan"].isna() & ~adata.obs["y_um_dbscan"].isna()).sum())
    say(f"placed cells (non-null dbscan x AND y): {n_placed}/{n0} "
        f"({100 * n_placed / n0:.1f}%)")

    # Also populate obsm['spatial'] for convenience (NaN for unplaced).
    # Spatial rules subset to placed cells before using this.
    adata.obsm["spatial"] = adata.obs[["x_um_dbscan", "y_um_dbscan"]].to_numpy(dtype=float)

    # --- write (ALL cells kept) ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    adata.write_h5ad(out_path)
    say(f"wrote {adata.n_obs} cells x {adata.n_vars} genes -> {out_path}")
