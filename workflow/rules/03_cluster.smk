# Stage 03 — Leiden clustering + clustree + UMAP (per species).
rule cluster:
    input:  "data/interim/latent_{species}.h5ad"
    output: "data/interim/clustered_{species}.h5ad"
    params: leiden=config["leiden"]
    log:    "logs/cluster_{species}.log"
    script: "../scripts/atlas/cluster.py"
