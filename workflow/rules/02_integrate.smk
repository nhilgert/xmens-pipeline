# Stage 02 - scVI integration + Leiden clustering, per species.
#
# Ported from integrate_and_cluster.ipynb (faithful to Cevrim et al. Methods):
#   - HVG selection (seurat_v3, 3000, batch_key=sample) on mito/ribo-EXCLUDED
#     genes, computed on the raw-counts layer
#   - scVI (n_layers=2, n_latent=30, gene_likelihood=nb) trained on the raw
#     counts of the HVG subset, with sample as batch key; early stopping
#   - neighbors (k=100) on the scVI latent
#   - Leiden at many resolutions (igraph, seed 0)
#   - UMAP on the scVI graph
#
# GPU rule: profiles/slurm routes `integrate` to a GPU node (--gres=gpu:1).
# Input is the per-species concatenated object from 01c_concat.

rule integrate:
    input:
        h5ad="data/interim/qc_passed_{species}.h5ad",
    output:
        h5ad="data/interim/latent_{species}.h5ad",
    params:
        counts_layer="counts",          # this pipeline's raw-counts layer (= notebook 'raw')
    log:
        "logs/02_integrate_{species}.log",
    script:
        "../scripts/atlas/integrate.py"
