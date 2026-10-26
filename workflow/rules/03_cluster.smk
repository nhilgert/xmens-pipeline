# Stage 03 — stub. Leiden and UMAP are computed in integrate.py (rule integrate);
# cluster.py exits with an error and clustered_{species}.h5ad is consumed by nothing.
rule cluster:
    input:  "data/interim/latent_{species}.h5ad"
    output: "data/interim/clustered_{species}.h5ad"
    params: leiden=config["leiden"]
    log:    "logs/cluster_{species}.log"
    script: "../scripts/atlas/cluster.py"
