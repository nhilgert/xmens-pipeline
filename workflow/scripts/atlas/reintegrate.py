"""Per-lineage scVI reintegration and Leiden ladder, rule `reintegrate_lineage` (04b).

Reads one lineage subset, data/interim/lineages/{species}_{lineage}.h5ad, with raw
counts in layers[params.counts_layer] and writes data/interim/latent_{species}_{lineage}.h5ad
with obsm['X_scVI'], obsm['X_umap_scVI'] and the leiden_scVI_<res> ladder the clustree
step consumes. No cell is removed. The scVI settings are those of integrate.py; HVGs
are reselected on the pooled subset (no per-sample batch in the selection, see below)
to recover within-lineage variation the global fit blurred.
"""
import time
import numpy as np
import scanpy as sc
import scvi

# Snakemake plumbing ----

inp = snakemake.input.h5ad  # noqa: F821
out = snakemake.output.h5ad  # noqa: F821
species = snakemake.wildcards.species  # noqa: F821
lineage = snakemake.wildcards.lineage  # noqa: F821
cfg = snakemake.config  # noqa: F821
logfile = snakemake.log[0]  # noqa: F821
counts_layer = snakemake.params.counts_layer  # noqa: F821

# Parameters ----

# config['scvi'] and config['leiden'] override these defaults.
SEED = cfg.get("seed", 0)
scvi_cfg = cfg.get("scvi", {})
leiden_cfg = cfg.get("leiden", {})

N_HVG = scvi_cfg.get("n_hvg", 3000)
HVG_FLAVOR = scvi_cfg.get("hvg_flavor", "seurat_v3")
N_LATENT = scvi_cfg.get("n_latent", 30)
N_LAYERS = scvi_cfg.get("n_layers", 2)
BATCH_KEY = scvi_cfg.get("batch_key", "sample")
GENE_LIKELIHOOD = scvi_cfg.get("gene_likelihood", "nb")
EXCLUDE_MITO_RIBO = scvi_cfg.get("exclude_mito_ribo", True)
N_NEIGHBORS = leiden_cfg.get("knn", 100)
RESOLUTIONS = scvi_cfg.get(
    "resolutions",
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0,
     4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
)

SCVI_LATENT_KEY = "X_scVI"
SCVI_NEIGHBORS_KEY = f"neighbors_scVI_{N_NEIGHBORS}"
SCVI_UMAP_KEY = "X_umap_scVI"


with open(logfile, "w") as log:
    def say(msg):
        print(f"[reintegrate:{species}:{lineage}] {msg}", file=log, flush=True)

    scvi.settings.seed = SEED
    np.random.seed(SEED)  # global seed for any call made without an explicit random_state

    adata = sc.read_h5ad(inp)
    say(f"loaded {adata.n_obs} cells x {adata.n_vars} genes")

    if counts_layer not in adata.layers:
        raise KeyError(
            f"counts layer '{counts_layer}' not in layers "
            f"(available: {list(adata.layers.keys())}); scVI needs raw counts. "
            f"If the lineage subset dropped it, carry layers through the 04 apply step."
        )

    if adata.n_obs < N_NEIGHBORS + 1:
        say(f"WARNING only {adata.n_obs} cells < knn {N_NEIGHBORS}; clamping knn")

    # Label mito and ribosomal genes ----

    # One prefix list serves both species: mouse symbols are capitalised, human upper-case.
    adata.var["mito"] = adata.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    adata.var["ribo"] = adata.var_names.str.startswith((
        "rps", "rpl", "mrps", "mrpl",
        "Rps", "Rpl", "Mrps", "Mrpl",
        "RPS", "RPL", "MRPS", "MRPL",
    ))
    say(f"mito {int(adata.var['mito'].sum())} | ribo {int(adata.var['ribo'].sum())}")

    # HVG reselection on the subset ----

    # No batch_key here, unlike integrate.py: seurat_v3 fits a per-batch loess, and
    # samples with only tens of nuclei in a lineage make it singular. HVGs come from the pooled subset; scVI below still corrects by batch.
    if EXCLUDE_MITO_RIBO:
        adata_hvg = adata[:, (~adata.var["mito"]) & (~adata.var["ribo"])].copy()
    else:
        adata_hvg = adata.copy()

    sc.pp.filter_genes(adata_hvg, min_cells=3)  # genes detected in < 3 cells are dropped
    sc.pp.highly_variable_genes(
        adata_hvg, flavor=HVG_FLAVOR, layer=counts_layer,
        n_top_genes=N_HVG, subset=True,
    )
    say(f"reselected {int(adata_hvg.var['highly_variable'].sum())} HVGs on the "
        f"pooled subset ({adata_hvg.n_vars} genes in after min_cells=3)")

    adata.uns["hvg"] = adata_hvg.uns["hvg"].copy()
    hvg_names = set(adata_hvg.var_names[adata_hvg.var["highly_variable"]])
    adata.var["highly_variable"] = adata.var_names.isin(hvg_names)

    # Normalized expression for downstream DE and plots ----

    adata.X = adata.layers[counts_layer].copy()
    sc.pp.normalize_total(
        adata,
        target_sum=1e4,  # counts per cell, as in integrate.py
        exclude_highly_expressed=False,
        inplace=True,
    )
    sc.pp.log1p(adata)

    # scVI ----

    scvi.model.SCVI.setup_anndata(adata_hvg, layer=counts_layer, batch_key=BATCH_KEY)
    model = scvi.model.SCVI(adata_hvg, n_layers=N_LAYERS, n_latent=N_LATENT,
                            gene_likelihood=GENE_LIKELIHOOD)
    say(f"training scVI (n_layers={N_LAYERS}, n_latent={N_LATENT}, "
        f"gene_likelihood={GENE_LIKELIHOOD}, early_stopping=True)...")
    t0 = time.time()
    model.train(early_stopping=True)
    say(f"  done in {time.time() - t0:.1f}s")

    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

    # Neighbors and the Leiden ladder ----

    # knn is clamped so neighbors() is valid on a lineage smaller than N_NEIGHBORS.
    knn = min(N_NEIGHBORS, adata.n_obs - 1)
    sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY, n_neighbors=knn,
                    key_added=SCVI_NEIGHBORS_KEY)

    for res in RESOLUTIONS:
        key = f"leiden_scVI_{res}"
        sc.tl.leiden(adata, resolution=res, random_state=SEED, flavor="igraph",
                     neighbors_key=SCVI_NEIGHBORS_KEY, key_added=key)
        say(f"  leiden res={res}: {adata.obs[key].nunique()} clusters")

    # UMAP with scanpy defaults (min_dist 0.5, spread 1.0), moved to a key naming its graph ----

    sc.tl.umap(adata, neighbors_key=SCVI_NEIGHBORS_KEY)
    adata.obsm[SCVI_UMAP_KEY] = adata.obsm.pop("X_umap")

    adata.write_h5ad(out)
    say(f"wrote -> {out} ({adata.n_obs} cells, obsm={list(adata.obsm.keys())})")
