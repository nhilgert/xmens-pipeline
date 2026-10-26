"""Stage 01c - concatenate per-sample QC'd objects into a per-species object.

Keeps ALL cells (placed + unplaced). Harmonizes obs to use 'sample' (tolerant of
either 'sample' or legacy 'sample_id' on the inputs), adds a clean 'cell_barcode'
with the trailing -1 suffix stripped (for joining to collaborator objects on
sample+barcode), and TRIMS obs to a meaningful whitelist (drops redundant /
constant / vestigial columns - see KEEP_OBS). Preserves layers['counts'] and
obsm['spatial'].

Writes data/interim/qc_passed_{species}.h5ad.
"""
import re
import anndata as ad
import scanpy as sc

inputs = snakemake.input             # noqa: F821 (list of per-sample h5ads)
out = snakemake.output.h5ad           # noqa: F821
species = snakemake.params.species    # noqa: F821
logfile = snakemake.log[0]            # noqa: F821

# Whitelist of obs columns to keep on the concatenated object. Everything else
# is dropped: n_counts (== total_counts), modality/coords_id (constant/derivable
# from sample), the *_outlier flags (all False post-QC), and log1p_* duplicates.
# 'cell_barcode' is added by this script (below) and always kept.
KEEP_OBS = [
    "sample", "timepoint", "species",
    "x_um_dbscan", "y_um_dbscan",
    "total_counts", "n_genes_by_counts",
    "pct_counts_mt", "pct_counts_ribo",
    "doubletdetection_predicted_doublet", "doubletdetection_doublet_score",
]


def strip_suffix(barcodes):
    """Strip a trailing -<digits> (the 10x GEM-well suffix, e.g. -1)."""
    return barcodes.str.replace(r"-\d+$", "", regex=True)


with open(logfile, "w") as log:
    def say(msg):
        print(f"[concat:{species}] {msg}", file=log, flush=True)

    say(f"concatenating {len(inputs)} samples")

    objs = {}
    for path in inputs:
        a = sc.read_h5ad(path)
        # standardize the sample key: prefer 'sample', fall back to legacy 'sample_id'
        if "sample" not in a.obs.columns:
            if "sample_id" in a.obs.columns:
                a.obs["sample"] = a.obs["sample_id"].astype(str)
            else:
                raise KeyError(f"{path}: no 'sample' or 'sample_id' in obs")
        # drop the legacy column so the concatenated object is clean
        if "sample_id" in a.obs.columns:
            del a.obs["sample_id"]
        sample = str(a.obs["sample"].iloc[0])
        # clean barcode (suffix stripped) for joining to collaborator objects
        a.obs["cell_barcode"] = strip_suffix(a.obs_names.to_series()).values
        say(f"  {sample}: {a.n_obs} cells")
        objs[sample] = a

    # concat keeps all cells; index made unique by appending the sample key so
    # stripped barcodes (unique only within a sample) don't collide.
    adata = ad.concat(
        objs,
        join="outer",
        label="sample_concat",     # records which input each cell came from
        index_unique="-",
        merge="same",              # keep var/obsm columns identical across inputs
    )
    # 'sample' is the authoritative grouping; 'sample_concat' duplicates it - drop.
    if "sample_concat" in adata.obs.columns:
        del adata.obs["sample_concat"]

    adata.obs["species"] = species

    # --- trim obs to the meaningful whitelist (+ cell_barcode, always kept) ---
    keep = [c for c in KEEP_OBS if c in adata.obs.columns]
    keep_set = set(keep) | {"cell_barcode"}
    dropped = [c for c in adata.obs.columns if c not in keep_set]
    adata.obs = adata.obs[[c for c in adata.obs.columns if c in keep_set]].copy()
    say(f"trimmed obs: kept {sorted(keep_set & set(adata.obs.columns))}")
    say(f"trimmed obs: dropped {dropped}")

    say(f"concatenated: {adata.n_obs} cells x {adata.n_vars} genes")
    say(f"obs columns: {list(adata.obs.columns)}")
    say(f"layers: {list(adata.layers.keys())} | obsm: {list(adata.obsm.keys())}")

    # sanity: counts layer + spatial present (scVI + spatial steps need them)
    if "counts" not in adata.layers:
        say("WARNING: no 'counts' layer after concat - scVI needs raw counts")
    if "spatial" not in adata.obsm:
        say("WARNING: no obsm['spatial'] after concat")

    n_placed = int((~adata.obs["x_um_dbscan"].isna()
                    & ~adata.obs["y_um_dbscan"].isna()).sum())
    say(f"placed cells: {n_placed}/{adata.n_obs} "
        f"({100 * n_placed / adata.n_obs:.1f}%)")

    adata.write_h5ad(out)
    say(f"wrote -> {out}")
