# 04e_reembed -- fresh whole-tissue scVI on the refined object. Removing EXCLUDE
# cells and moving throwbacks changed the composition, and the per-lineage
# latents are separate fits in different spaces, so the embedding is rebuilt.
#
# This is the stage-02 script pointed at a different input -- batch_key HVG works
# again here because it's the whole tissue, not a thin lineage subset.
#
# NOTE: `reembed` is a GPU rule -- add it to profiles/slurm set-resources by NAME
# (same as `integrate` and `reintegrate_lineage`).

rule reembed:
    input:
        h5ad="data/interim/refined_{species}.h5ad",
    output:
        h5ad="data/interim/final_{species}.h5ad",
    params:
        counts_layer="counts",          # integrate.py reads snakemake.params.counts_layer
    log:
        "logs/04e_reembed_{species}.log",
    script:
        "../scripts/atlas/integrate.py"
