"""Per-sample quality control, following the project QC notebook (Cevrim et al. 2025).

Called by rule `qc` (workflow/rules/01_qc.smk). Reads the per-sample AnnData from
rule `ingest` (X = CellBender-corrected counts, all cells present, coordinates
nullable). Applies, in the notebook's order: a hard floor on total counts, a fixed
upper bound on mitochondrial percentage, 3-MAD outlier bounds on log10(x+1) of
total_counts and n_genes_by_counts, then DoubletDetection (BoostClassifier) with
doublets removed. Thresholds come from config['qc'] and config['doublet'].
DoubletDetection lives in a separate env (01_qc.smk `conda:`). If the package is
missing, doublet removal is skipped with a warning in the log.

Writes data/interim/qc/{sample}.h5ad with the corrected counts kept in layers['counts'].
"""
import numpy as np
import scanpy as sc
from scipy.stats import median_abs_deviation

inp = snakemake.input.h5ad           # noqa: F821 (snakemake-injected)
out = snakemake.output.h5ad           # noqa: F821
sample = snakemake.wildcards.sample   # noqa: F821
species = snakemake.params.species    # noqa: F821
cfg = snakemake.config                # noqa: F821
logfile = snakemake.log[0]            # noqa: F821

qc = cfg["qc"]
seed = cfg.get("seed", 0)
dd_cfg = cfg.get("doublet", {})

# Metrics MAD-thresholded on a log10(x+1) scale (matches notebook).
LOG_METRICS = ("total_counts", "n_genes_by_counts")


def _log10p(x):
    return np.log10(x + 1)


def _inv_log10p(y):
    return np.power(10, y) - 1


def outlier_bounds(values, nmads):
    """Raw-MAD bounds at +/- nmads from the median."""
    med = np.nanmedian(values)
    mad = median_abs_deviation(values, nan_policy="omit")  # scale=1.0 (raw)
    return med - nmads * mad, med + nmads * mad, mad


with open(logfile, "w") as log:
    def say(msg):
        print(f"[qc:{sample}] {msg}", file=log, flush=True)

    # Global seed from config; BoostClassifier below is called without random_state.
    np.random.seed(seed)

    adata = sc.read_h5ad(inp)
    n0 = adata.n_obs
    say(f"loaded {n0} cells x {adata.n_vars} genes (species={species})")

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # QC metrics ----
    # Both the mouse (mt-) and human (MT-) prefixes are matched so one script
    # serves both species, as in the notebook.
    adata.var["mt"] = adata.var_names.str.startswith(("mt-", "MT-"))
    adata.var["ribo"] = adata.var_names.str.startswith((
        "rps", "rpl", "mrps", "mrpl",
        "Rps", "Rpl", "Mrps", "Mrpl",
        "RPS", "RPL", "MRPS", "MRPL",
    ))
    adata.var["hb"] = (
        adata.var_names.str.contains("^Hb[^(p)]")
        | adata.var_names.str.contains("^HB[^(P)]")
    )
    n_mt = int(adata.var["mt"].sum())
    say(f"flagged {n_mt} mito genes (prefixes mt-/MT-)")
    if n_mt == 0:
        say("WARNING: no mito genes matched - check gene naming for this species")

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        percent_top=[20],  # pct_counts_in_top_20_genes, as in the notebook
        log1p=True,
        inplace=True,
    )

    # Filtering is sequential, matching the QC notebook exactly: the count floor
    # is applied first (so junk barcodes do not distort the MAD bounds), then the
    # mito and MAD outlier flags are evaluated on the surviving cells and applied
    # in the notebook's order (mito -> total_counts -> n_genes_by_counts).

    # Count floor ----
    min_counts = qc["min_counts"]
    n_low = int((adata.obs["total_counts"] < min_counts).sum())
    sc.pp.filter_cells(adata, min_counts=min_counts)
    say(f"min {min_counts} counts: removed {n_low}, {adata.n_obs}/{n0} remain")

    # MAD outlier flags ----
    # Computed on the floored object, so the bounds reflect the real cell
    # population rather than the junk below the floor.
    n_mads = qc["n_mads"]
    for metric in LOG_METRICS:
        vals = _log10p(adata.obs[metric].to_numpy())
        lo, hi, mad = outlier_bounds(vals, n_mads)
        if mad == 0:
            say(f"  WARNING: MAD is zero for {metric}; bounds collapse to the median")
        adata.obs[f"{metric}_outlier"] = (vals < lo) | (vals > hi)
        say(f"{n_mads}-MAD {metric} (log10): "
            f"bounds=[{_inv_log10p(lo):.1f}, {_inv_log10p(hi):.1f}], "
            f"flag {int(adata.obs[f'{metric}_outlier'].sum())}")

    # Mito uses a constant percentage cap from config, not a MAD bound.
    max_mito = qc["max_pct_mito"]
    adata.obs["pct_counts_mt_outlier"] = adata.obs["pct_counts_mt"] > max_mito
    say(f"mito > {max_mito}%: flag {int(adata.obs['pct_counts_mt_outlier'].sum())}")

    # Apply the flags ----
    # One metric at a time, in the notebook's order, so the per-step counts in
    # the log line up with the notebook's.
    for metric in ("pct_counts_mt", "total_counts", "n_genes_by_counts"):
        n_before = adata.n_obs
        adata = adata[~adata.obs[f"{metric}_outlier"]].copy()
        say(f"  filter {metric}: removed {n_before - adata.n_obs}, {adata.n_obs} remain")

    say(f"after threshold QC: {adata.n_obs}/{n0} cells")

    # Doublet removal ----
    # The fallback defaults below equal config/config.yaml (the paper's settings);
    # config is the source of truth when it sets them.
    try:
        import doubletdetection
        ver = getattr(doubletdetection, "__version__", "?")
        n_iters = dd_cfg.get("n_iters", 40)
        n_top = dd_cfg.get("n_top_var_genes", 3000)
        p_thresh = dd_cfg.get("p_thresh", 1e-7)
        voter_thresh = dd_cfg.get("voter_thresh", 0.5)
        say(f"DoubletDetection {ver}: {n_iters} iters, n_top_var_genes={n_top}, "
            f"voter_thresh={voter_thresh}")
        clf = doubletdetection.BoostClassifier(
            n_iters=n_iters,
            clustering_algorithm="leiden",
            standard_scaling=True,
            n_top_var_genes=n_top,
            n_jobs=-1,
        )

        # Fit on the corrected counts, as the notebook does (clf.fit(adata.X)).
        preds = clf.fit(adata.layers["counts"]).predict(
            p_thresh=p_thresh, voter_thresh=voter_thresh
        )
        adata.obs["doubletdetection_predicted_doublet"] = preds.astype(bool)
        adata.obs["doubletdetection_doublet_score"] = clf.doublet_score()
        n_doub = int(adata.obs["doubletdetection_predicted_doublet"].sum())
        say(f"flagged {n_doub} doublets; removing")
        adata = adata[~adata.obs["doubletdetection_predicted_doublet"].to_numpy()].copy()
    except ImportError:
        say("WARNING: doubletdetection not available in this env - SKIPPING doublet "
            "removal. Set config['envs']['doubletdetection'] to an env with it "
            "(==4.2 to match the paper).")
    # A DoubletDetection failure is logged and the object is still written, so
    # dry runs and partial environments complete (see the 01_qc.smk header).
    except Exception as e:  # noqa: BLE001
        say(f"WARNING: DoubletDetection failed ({e}); proceeding without doublet removal")

    say(f"final: {adata.n_obs} cells x {adata.n_vars} genes "
        f"({n0 - adata.n_obs} removed total)")
    adata.write_h5ad(out)
    say(f"wrote -> {out}")
