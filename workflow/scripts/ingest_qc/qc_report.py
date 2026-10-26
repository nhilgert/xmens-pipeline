"""Stage 01b - QC diagnostic figures, mirroring the project QC notebook.

Reads the PRE-QC raw object (full distributions + threshold lines) and the
POST-QC object (survival + doublet calls), writes figures to params.outdir.

Recomputes the same metrics and bounds that 01_qc uses (log10 scale, raw MAD,
nmads from config; 500 floor; 8% mito) so the threshold lines on the plots
exactly match what the filter applied. Filtering happens in 01_qc; this only
visualizes.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.stats import median_abs_deviation

raw_path = snakemake.input.raw        # noqa: F821 (snakemake-injected)
qcd_path = snakemake.input.qcd        # noqa: F821
sample = snakemake.wildcards.sample   # noqa: F821
species = snakemake.params.species    # noqa: F821
outdir = snakemake.params.outdir      # noqa: F821
cfg = snakemake.config                # noqa: F821
logfile = snakemake.log[0]            # noqa: F821

qc = cfg["qc"]
min_counts = qc["min_counts"]
max_mito = qc["max_pct_mito"]
n_mads = qc["n_mads"]
LOG_METRICS = ("total_counts", "n_genes_by_counts")

sns.set_context("notebook", font_scale=1)
sns.set_style("ticks")
blues = sns.color_palette("Blues", 9)
reds = sns.color_palette("Reds", 9)


def log10p(x):
    return np.log10(x + 1)


def mad_bounds(values, nmads):
    med = np.nanmedian(values)
    mad = median_abs_deviation(values, nan_policy="omit")  # raw MAD, matches 01_qc
    return med - nmads * mad, med + nmads * mad


with open(logfile, "w") as log:
    def say(msg):
        print(f"[qc_report:{sample}] {msg}", file=log, flush=True)

    os.makedirs(outdir, exist_ok=True)

    # --- pre-QC object: recompute metrics for plotting ---
    adata = sc.read_h5ad(raw_path)
    n_raw = adata.n_obs
    say(f"raw: {n_raw} cells")
    adata.var["mt"] = adata.var_names.str.startswith(("mt-", "MT-"))
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True
    )

    # --- post-QC object: survival + doublets ---
    try:
        qcd = sc.read_h5ad(qcd_path)
        n_qcd = qcd.n_obs
    except Exception as e:  # noqa: BLE001
        say(f"could not read QC'd object ({e}); plotting raw only")
        qcd = None
        n_qcd = None

    # --- log10 MAD-metric histograms with bounds ---
    for metric in LOG_METRICS:
        vals = log10p(adata.obs[metric].to_numpy())
        lo, hi = mad_bounds(vals, n_mads)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(adata.obs[metric], bins=50, color=blues[6],
                     log_scale=True, kde=True, ax=ax)
        ax.axvline(10 ** lo - 1, color=reds[5], linestyle="--")
        ax.axvline(10 ** hi - 1, color=reds[5], linestyle="--")
        ax.set_title(f"{sample}: {metric}")
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("# barcodes")
        ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        say(f"wrote {metric}.png (bounds [{10**lo-1:.0f}, {10**hi-1:.0f}])")

    # --- mito % histogram with fixed bound ---
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(adata.obs["pct_counts_mt"], bins=50, color=blues[6], kde=True, ax=ax)
    ax.axvline(max_mito, color=reds[5], linestyle="--")
    ax.set_title(f"{sample}: pct_counts_mt")
    ax.set_xlabel("% Mitochondrial")
    ax.set_ylabel("# barcodes")
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pct_counts_mt.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    say(f"wrote pct_counts_mt.png (bound {max_mito}%)")

    # --- min-counts line on a total_counts view (informational) ---
    say(f"min_counts floor = {min_counts}")

    # --- doublet score distribution (post-QC, if present) ---
    if qcd is not None and "doubletdetection_doublet_score" in qcd.obs:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(qcd.obs["doubletdetection_doublet_score"], bins=50,
                     color=blues[6], ax=ax)
        ax.set_title(f"{sample}: doublet score (post-threshold QC)")
        ax.set_xlabel("DoubletDetection score")
        ax.set_ylabel("# cells")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "doublet_score.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        say("wrote doublet_score.png")

    # --- survival summary ---
    if n_qcd is not None:
        frac = 100 * n_qcd / n_raw
        say(f"survival: {n_qcd}/{n_raw} cells ({frac:.1f}%) passed QC")
        with open(os.path.join(outdir, "summary.txt"), "w") as f:
            f.write(f"sample\t{sample}\n")
            f.write(f"species\t{species}\n")
            f.write(f"n_raw\t{n_raw}\n")
            f.write(f"n_qc\t{n_qcd}\n")
            f.write(f"pct_retained\t{frac:.1f}\n")

    say("done")
