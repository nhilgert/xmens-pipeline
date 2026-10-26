# 04e_reembed -- fresh whole-tissue scVI on the refined object. Removing EXCLUDE
# cells and moving throwbacks changed the composition, and the per-lineage
# latents are separate fits in different spaces, so the embedding is rebuilt.
#
# This is the stage-02 script pointed at a different input -- batch_key HVG works
# again here because it's the whole tissue, not a thin lineage subset.
#
# `reembed` is a GPU rule: profiles/slurm set-resources must list it by name, as it
# lists `integrate` and `reintegrate_lineage`.

# Defined only when config scvi.retrain is true. scVI fits are deterministic only
# within one GPU class and the annotation tables refer to the July 2026 fits, so by
# default data/interim/final_{species}.h5ad is an input the pipeline reads,
# never an output it rewrites.
if config["scvi"].get("retrain", False):

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
