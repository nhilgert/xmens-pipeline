"""scVI integration and a Leiden ladder for one species.

Called by rule `integrate` (02_integrate.smk) on the QC-passed object and by rule
`reembed` (04e_reembed.smk) on the refined object. Reads the input h5ad with raw
counts in layers[params.counts_layer] and writes data/interim/latent_{species}.h5ad
(or final_{species}.h5ad) carrying obsm['X_scVI'], obsm['X_umap_scVI'] and one
leiden_scVI_<res> obs column per resolution.

Follows integrate_and_cluster.ipynb and Cevrim et al. Methods: HVGs selected with
seurat_v3 per sample on genes other than mito/ribo, scVI trained on raw counts of the
HVG subset with sample as batch, kNN graph and Leiden on the latent, UMAP on that graph.
The notebook read raw counts from a layer named 'raw'; in this pipeline the layer is
'counts' and its name arrives as params.counts_layer.
"""
import time
import numpy as np
import scanpy as sc
import scvi

# Snakemake plumbing ----

inp = snakemake.input.h5ad  # noqa: F821
out = snakemake.output.h5ad  # noqa: F821
species = snakemake.wildcards.species  # noqa: F821
cfg = snakemake.config  # noqa: F821
logfile = snakemake.log[0]  # noqa: F821
counts_layer = snakemake.params.counts_layer  # noqa: F821

# Parameters ----

# Defaults are the values used in Cevrim et al. Methods; config['scvi'] and
# config['leiden'] override them.
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
    [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    ],
)

SCVI_LATENT_KEY = "X_scVI"
SCVI_NEIGHBORS_KEY = f"neighbors_scVI_{N_NEIGHBORS}"
SCVI_UMAP_KEY = "X_umap_scVI"


# The log directory does not exist on a fresh checkout.
from pathlib import Path as _Path
_Path(logfile).parent.mkdir(parents=True, exist_ok=True)

with open(logfile, "w") as log:
    def say(msg):
        print(f"[integrate:{species}] {msg}", file=log, flush=True)

    scvi.settings.seed = SEED
    np.random.seed(SEED)  # global seed for any call made without an explicit random_state

    # Load ----

    adata = sc.read_h5ad(inp)
    say(f"loaded {adata.n_obs} cells x {adata.n_vars} genes")

    if counts_layer not in adata.layers:
        raise KeyError(
            f"counts layer '{counts_layer}' not in layers "
            f"(available: {list(adata.layers.keys())}); scVI needs raw counts"
        )

    # Label mito and ribosomal genes ----

    # Prefix lists cover mouse (mt-, Rps), human (MT-, RPS) and lower-case symbols.
    mito_mask = adata.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    adata.var["mito"] = mito_mask
    ribo_mask = adata.var_names.str.startswith((
        "rps", "rpl", "mrps", "mrpl",
        "Rps", "Rpl", "Mrps", "Mrpl",
        "RPS", "RPL", "MRPS", "MRPL",
    ))
    adata.var["ribo"] = ribo_mask
    say(f"mito genes: {int(mito_mask.sum())} | ribo genes: {int(ribo_mask.sum())}")

    # HVG selection ----

    # Mito and ribosomal genes are excluded from the HVG candidate pool (selection runs on a
    # separate object; flags are copied back to the full object). seurat_v3 expects raw counts.
    if EXCLUDE_MITO_RIBO:
        adata_hvg = adata[:, (~adata.var["mito"]) & (~adata.var["ribo"])].copy()
        say(f"HVG candidate genes (mito/ribo excluded): {adata_hvg.n_vars}")
    else:
        adata_hvg = adata.copy()

    sc.pp.highly_variable_genes(
        adata_hvg,
        flavor=HVG_FLAVOR,
        layer=counts_layer,
        batch_key=BATCH_KEY,
        n_top_genes=N_HVG,
        subset=True,
    )
    say(f"selected {int(adata_hvg.var['highly_variable'].sum())} HVGs "
        f"(flavor={HVG_FLAVOR}, batch_key={BATCH_KEY})")

    # HVG flags and ranks are copied to the full object; withheld genes get NaN ranks.
    adata.uns["hvg"] = adata_hvg.uns["hvg"].copy()
    hvg_names = set(adata_hvg.var_names[adata_hvg.var["highly_variable"]])
    adata.var["highly_variable"] = adata.var_names.isin(hvg_names)
    adata.var["highly_variable_rank"] = adata.var_names.map(
        dict(zip(adata_hvg.var_names, adata_hvg.var["highly_variable_rank"]))
    )
    adata.var["highly_variable_nbatches"] = adata.var_names.map(
        dict(zip(adata_hvg.var_names, adata_hvg.var["highly_variable_nbatches"]))
    )

    # Normalized expression for downstream DE and plots ----

    # .X is rebuilt from the counts layer so that it does not depend on what the
    # upstream object held in .X.
    adata.X = adata.layers[counts_layer].copy()
    sc.pp.normalize_total(
        adata,
        target_sum=1e4,  # counts per cell, as in the notebook
        exclude_highly_expressed=False,
        inplace=True,
    )
    sc.pp.log1p(adata)
    say("set .X = log1p(normalize_total(counts, 1e4))")

    # scVI ----

    # Trained on raw counts of the HVG subset; the latent is copied onto the full object.
    say("setting up scVI...")
    scvi.model.SCVI.setup_anndata(adata_hvg, layer=counts_layer, batch_key=BATCH_KEY)
    model = scvi.model.SCVI(
        adata_hvg, n_layers=N_LAYERS, n_latent=N_LATENT,
        gene_likelihood=GENE_LIKELIHOOD,
    )
    say(f"training scVI (n_layers={N_LAYERS}, n_latent={N_LATENT}, "
        f"gene_likelihood={GENE_LIKELIHOOD}, early_stopping=True)...")
    t0 = time.time()
    model.train(early_stopping=True)
    say(f"  training done in {time.time() - t0:.1f}s")

    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
    say(f"latent -> obsm['{SCVI_LATENT_KEY}'] shape {adata.obsm[SCVI_LATENT_KEY].shape}")

    # Neighbors and the Leiden ladder ----

    # The graph is stored under its own key so a later neighbors() call cannot overwrite it.
    say(f"computing neighbors (k={N_NEIGHBORS}) on {SCVI_LATENT_KEY}...")
    sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY, n_neighbors=N_NEIGHBORS,
                    key_added=SCVI_NEIGHBORS_KEY)

    for res in RESOLUTIONS:
        key = f"leiden_scVI_{res}"
        t0 = time.time()
        sc.tl.leiden(adata, resolution=res, random_state=SEED, flavor="igraph",
                     neighbors_key=SCVI_NEIGHBORS_KEY, key_added=key)
        n_clusters = adata.obs[key].nunique()
        say(f"  leiden res={res}: {n_clusters} clusters ({time.time() - t0:.1f}s)")

    # UMAP, moved from obsm['X_umap'] to a key that names its graph ----

    say("computing UMAP on scVI graph...")
    sc.tl.umap(adata, neighbors_key=SCVI_NEIGHBORS_KEY)
    adata.obsm[SCVI_UMAP_KEY] = adata.obsm["X_umap"].copy()
    del adata.obsm["X_umap"]

    # Write ----

    say(f"final: {adata.n_obs} cells, obsm={list(adata.obsm.keys())}")
    import os
    from pathlib import Path
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Existing objects under a 'migrated' directory are never overwritten.
    if out_path.exists() and "migrated" in out_path.parts:
        raise FileExistsError(f"refusing to overwrite existing object: {out}")

    # Write to a temporary name and rename, so a killed job cannot leave a partial h5ad
    # that Snakemake would treat as complete.
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp.h5ad")
    adata.write_h5ad(tmp_path)
    os.replace(tmp_path, out_path)
    say(f"wrote -> {out}")
