"""Stage 02 - scVI integration + Leiden clustering, per species.

Faithful port of integrate_and_cluster.ipynb (= Cevrim et al. Methods):
  1. Label mito/ribo genes; split off a mito/ribo-EXCLUDED object for HVG selection.
  2. HVG selection: seurat_v3, n_top_genes=3000, batch_key='sample', on the raw
     counts layer, subset=True. Map HVG flags back to the full object.
  3. Set .X to raw counts, normalize_total(1e4), log1p (for downstream DE/plots).
  4. scVI: setup_anndata(layer=raw counts, batch_key='sample');
     SCVI(n_layers=2, n_latent=30, gene_likelihood='nb'); train(early_stopping=True).
     Latent -> obsm['X_scVI'] on the full object.
  5. neighbors(k=100) on X_scVI; Leiden at many resolutions (igraph, seed 0).
  6. UMAP on the scVI graph -> obsm['X_umap_scVI'].

NOTE: the notebook reads its raw counts from a layer named 'raw'. In this
pipeline that layer is 'counts' (params.counts_layer). scVI requires raw counts.

Writes data/interim/latent_{species}.h5ad.
"""
import time
import numpy as np
import scanpy as sc
import scvi

inp = snakemake.input.h5ad            # noqa: F821
out = snakemake.output.h5ad           # noqa: F821
species = snakemake.wildcards.species # noqa: F821
cfg = snakemake.config                # noqa: F821
logfile = snakemake.log[0]            # noqa: F821
counts_layer = snakemake.params.counts_layer  # noqa: F821  (this pipeline: 'counts')

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
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
)

SCVI_LATENT_KEY = "X_scVI"
SCVI_NEIGHBORS_KEY = f"neighbors_scVI_{N_NEIGHBORS}"
SCVI_UMAP_KEY = "X_umap_scVI"


from pathlib import Path as _Path
_Path(logfile).parent.mkdir(parents=True, exist_ok=True)
with open(logfile, "w") as log:
    def say(msg):
        print(f"[integrate:{species}] {msg}", file=log, flush=True)

    scvi.settings.seed = SEED
    np.random.seed(SEED)

    adata = sc.read_h5ad(inp)
    say(f"loaded {adata.n_obs} cells x {adata.n_vars} genes")
    if counts_layer not in adata.layers:
        raise KeyError(
            f"counts layer '{counts_layer}' not in layers "
            f"(available: {list(adata.layers.keys())}); scVI needs raw counts"
        )

    # --- 1. label mito/ribo genes (matches notebook prefix lists) ---
    mito_mask = adata.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    adata.var["mito"] = mito_mask
    ribo_mask = adata.var_names.str.startswith((
        "rps", "rpl", "mrps", "mrpl",
        "Rps", "Rpl", "Mrps", "Mrpl",
        "RPS", "RPL", "MRPS", "MRPL",
    ))
    adata.var["ribo"] = ribo_mask
    say(f"mito genes: {int(mito_mask.sum())} | ribo genes: {int(ribo_mask.sum())}")

    # --- 2. HVG selection on mito/ribo-excluded genes (raw counts layer) ---
    if EXCLUDE_MITO_RIBO:
        adata_hvg = adata[:, (~adata.var["mito"]) & (~adata.var["ribo"])].copy()
        say(f"HVG candidate genes (mito/ribo excluded): {adata_hvg.n_vars}")
    else:
        adata_hvg = adata.copy()

    sc.pp.highly_variable_genes(
        adata_hvg,
        flavor=HVG_FLAVOR,
        layer=counts_layer,        # raw counts (notebook used layer='raw')
        batch_key=BATCH_KEY,
        n_top_genes=N_HVG,
        subset=True,
    )
    say(f"selected {int(adata_hvg.var['highly_variable'].sum())} HVGs "
        f"(flavor={HVG_FLAVOR}, batch_key={BATCH_KEY})")

    # map HVG info back to the full object (for provenance / downstream)
    adata.uns["hvg"] = adata_hvg.uns["hvg"].copy()
    hvg_names = set(adata_hvg.var_names[adata_hvg.var["highly_variable"]])
    adata.var["highly_variable"] = adata.var_names.isin(hvg_names)
    adata.var["highly_variable_rank"] = adata.var_names.map(
        dict(zip(adata_hvg.var_names, adata_hvg.var["highly_variable_rank"]))
    )
    adata.var["highly_variable_nbatches"] = adata.var_names.map(
        dict(zip(adata_hvg.var_names, adata_hvg.var["highly_variable_nbatches"]))
    )

    # --- 3. set .X to raw counts, normalize + log1p (for downstream DE/plots) ---
    adata.X = adata.layers[counts_layer].copy()
    sc.pp.normalize_total(adata, target_sum=1e4, exclude_highly_expressed=False,
                          inplace=True)
    sc.pp.log1p(adata)
    say("set .X = log1p(normalize_total(counts, 1e4))")

    # --- 4. scVI on the HVG subset (raw counts), latent -> full object ---
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

    # --- 5. neighbors on latent + Leiden at many resolutions ---
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

    # --- 6. UMAP on the scVI graph ---
    say("computing UMAP on scVI graph...")
    sc.tl.umap(adata, neighbors_key=SCVI_NEIGHBORS_KEY)
    adata.obsm[SCVI_UMAP_KEY] = adata.obsm["X_umap"].copy()
    del adata.obsm["X_umap"]

    say(f"final: {adata.n_obs} cells, obsm={list(adata.obsm.keys())}")
    import os
    from pathlib import Path
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and "migrated" in out_path.parts:
        raise FileExistsError(f"refusing to overwrite existing object: {out}")
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp.h5ad")
    adata.write_h5ad(tmp_path)
    os.replace(tmp_path, out_path)
    say(f"wrote -> {out}")
