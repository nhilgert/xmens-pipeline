"""Stage 05 - per-lineage scVI reintegration.

Sibling of integrate.py (stage 02): identical scVI procedure, applied to one
lineage sub-object. "Re-integrated using previous scVI model parameters"
(Cevrim et al. Methods) => same hyperparameters, HVGs RESELECTED on the subset, pooled (no batch_key -- see section 2 note).
(The point of reintegrating is to capture within-lineage variation the global
fit blurred.)

Input : data/interim/lineages/{species}_{lineage}.h5ad
          - carries layers['counts'] (raw counts; required by scVI) and obs
            (sample, cell_barcode, timepoint, lineage) inherited from stage 02/04.
Output: data/interim/latent_{species}_{lineage}.h5ad
          - obsm['X_scVI'], the Leiden ladder, obsm['X_umap_scVI'].

NO cell removal here. The Leiden ladder is what the later clustree fine-structure
step consumes. Frozen stochastic checkpoint, same status as latent_{species}.
"""
import time
import numpy as np
import scanpy as sc
import scvi

inp = snakemake.input.h5ad             # noqa: F821
out = snakemake.output.h5ad            # noqa: F821
species = snakemake.wildcards.species  # noqa: F821
lineage = snakemake.wildcards.lineage  # noqa: F821
cfg = snakemake.config                 # noqa: F821
logfile = snakemake.log[0]             # noqa: F821
counts_layer = snakemake.params.counts_layer  # noqa: F821  ('counts')

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
    np.random.seed(SEED)

    adata = sc.read_h5ad(inp)
    say(f"loaded {adata.n_obs} cells x {adata.n_vars} genes")
    if counts_layer not in adata.layers:
        raise KeyError(
            f"counts layer '{counts_layer}' not in layers "
            f"(available: {list(adata.layers.keys())}); scVI needs raw counts. "
            f"If the lineage subset dropped it, carry layers through the 04 apply step."
        )
    if adata.n_obs < N_NEIGHBORS + 1:
        # tiny lineage (e.g. a small immune split): shrink knn so neighbors() is valid
        say(f"WARNING only {adata.n_obs} cells < knn {N_NEIGHBORS}; clamping knn")

    # --- 1. label mito/ribo genes (same prefix lists as stage 02) ---
    adata.var["mito"] = adata.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    adata.var["ribo"] = adata.var_names.str.startswith((
        "rps", "rpl", "mrps", "mrpl",
        "Rps", "Rpl", "Mrps", "Mrpl",
        "RPS", "RPL", "MRPS", "MRPL",
    ))
    say(f"mito {int(adata.var['mito'].sum())} | ribo {int(adata.var['ribo'].sum())}")

    # --- 2. HVG reselection on this subset (mito/ribo excluded, raw counts) ---
    # NO batch_key here (unlike stage 02). seurat_v3 fits a per-batch loess of
    # variance vs mean; within a lineage subset some samples have only tens of
    # cells (e.g. 35 in mouse epithelial), so the per-batch fit hits a loess
    # singularity ("near singularities ... 0.090619") for EVERY lineage, not
    # just small ones. HVGs are selected on the pooled subset instead. scVI
    # still uses batch_key=sample for the integration below (setup_anndata), so
    # batch correction is unaffected -- only which genes are fed to it. The
    # min_cells filter drops genes too sparse in the subset for the loess.
    if EXCLUDE_MITO_RIBO:
        adata_hvg = adata[:, (~adata.var["mito"]) & (~adata.var["ribo"])].copy()
    else:
        adata_hvg = adata.copy()
    sc.pp.filter_genes(adata_hvg, min_cells=3)
    sc.pp.highly_variable_genes(
        adata_hvg, flavor=HVG_FLAVOR, layer=counts_layer,
        n_top_genes=N_HVG, subset=True,          # pooled: no batch_key
    )
    say(f"reselected {int(adata_hvg.var['highly_variable'].sum())} HVGs on the "
        f"pooled subset ({adata_hvg.n_vars} genes in after min_cells=3)")

    adata.uns["hvg"] = adata_hvg.uns["hvg"].copy()
    hvg_names = set(adata_hvg.var_names[adata_hvg.var["highly_variable"]])
    adata.var["highly_variable"] = adata.var_names.isin(hvg_names)

    # --- 3. re-derive .X = log1p(normalize_total(counts, 1e4)) from raw counts ---
    adata.X = adata.layers[counts_layer].copy()
    sc.pp.normalize_total(adata, target_sum=1e4, exclude_highly_expressed=False,
                          inplace=True)
    sc.pp.log1p(adata)

    # --- 4. scVI on the subset's HVGs ---
    scvi.model.SCVI.setup_anndata(adata_hvg, layer=counts_layer, batch_key=BATCH_KEY)
    model = scvi.model.SCVI(adata_hvg, n_layers=N_LAYERS, n_latent=N_LATENT,
                            gene_likelihood=GENE_LIKELIHOOD)
    say(f"training scVI (n_layers={N_LAYERS}, n_latent={N_LATENT}, "
        f"gene_likelihood={GENE_LIKELIHOOD}, early_stopping=True)...")
    t0 = time.time()
    model.train(early_stopping=True)
    say(f"  done in {time.time() - t0:.1f}s")
    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

    # --- 5. neighbors + Leiden ladder (feeds clustree later) ---
    knn = min(N_NEIGHBORS, adata.n_obs - 1)
    sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY, n_neighbors=knn,
                    key_added=SCVI_NEIGHBORS_KEY)
    for res in RESOLUTIONS:
        key = f"leiden_scVI_{res}"
        sc.tl.leiden(adata, resolution=res, random_state=SEED, flavor="igraph",
                     neighbors_key=SCVI_NEIGHBORS_KEY, key_added=key)
        say(f"  leiden res={res}: {adata.obs[key].nunique()} clusters")

    # --- 6. UMAP on the scVI graph (scanpy defaults = min_dist 0.5, spread 1.0) ---
    sc.tl.umap(adata, neighbors_key=SCVI_NEIGHBORS_KEY)
    adata.obsm[SCVI_UMAP_KEY] = adata.obsm.pop("X_umap")

    adata.write_h5ad(out)
    say(f"wrote -> {out} ({adata.n_obs} cells, obsm={list(adata.obsm.keys())})")
