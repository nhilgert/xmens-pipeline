# Clustree -- off the critical path (not in rule all).
#
#     snakemake clustree                      # every object in CLUSTREE_TARGETS
#     snakemake results/figures/clustree/mouse_stromal.png
#
# Nothing in rule all depends on these: a figure is terminal, so a plotting
# failure (or a missing R install) can never stall real computation. But they
# ARE in the DAG, so they regenerate when the underlying object changes -- the
# figure can't silently go stale relative to the data.
#
# Requires R with clustree + readr + ggplot2 on PATH (module load / renv).

CLUSTREE = config.get("clustree", {})
CLUSTREE_TARGETS = CLUSTREE.get("targets", ["mouse_stromal"])
# Leiden resolutions shown in the clustree figure.
CLUSTREE_RANGE = CLUSTREE.get("range", {"start": 0.1, "stop": 0.8, "step": 0.1})


# Leiden labels over the plotting resolution range, as the CSV clustree reads.
rule clustree_export:
    input:
        h5ad="data/interim/latent_{species}_{lineage}.h5ad",
    output:
        csv="results/tables/clustree/{species}_{lineage}.csv",
    params:
        start=CLUSTREE_RANGE["start"],
        stop=CLUSTREE_RANGE["stop"],
        step=CLUSTREE_RANGE["step"],
    log:
        "logs/98_clustree_export_{species}_{lineage}.log",
    script:
        "../scripts/atlas/export_clustree.py"


# The clustree figure for one lineage latent (R).
rule clustree_plot:
    input:
        csv="results/tables/clustree/{species}_{lineage}.csv",
    output:
        png="results/figures/clustree/{species}_{lineage}.png",
    params:
        title=lambda w: f"{w.species} {w.lineage} -- Leiden ladder",
    log:
        "logs/98_clustree_plot_{species}_{lineage}.log",
    script:
        "../scripts/clustree_plot.R"


# Target: clustree figures for the configured objects.
rule clustree:
    input:
        expand("results/figures/clustree/{target}.png", target=CLUSTREE_TARGETS),


# Reference-free resolution diagnostics ----
# Stability of cluster membership across the FULL Leiden ladder stored on each
# lineage latent (all leiden_scVI_* columns, not the clustree plotting range).
# Reports whether the production resolution sits below the churn elbow (the largest
# step in the fraction of clusters with a minor parent). Pure obs computation; runs
# for every lineage. No z-scores.

# Membership stability between adjacent resolutions for one lineage latent.
rule ladder_stability:
    input:
        h5ad="data/interim/latent_{species}_{lineage}.h5ad",
    output:
        tsv="results/tables/clustree/{species}_{lineage}_stability.tsv",
        summary="results/tables/clustree/{species}_{lineage}_stability_summary.tsv",
    params:
        res=lambda w: LINEAGE_RES(w.species, w.lineage),
    log:
        "logs/98_ladder_stability_{species}_{lineage}.log",
    script:
        "../scripts/atlas/ladder_stability.py"


# One table of the per-lineage stability summaries across both species.
rule stability_summary:
    input:
        [f"results/tables/clustree/{sp}_{lin}_stability_summary.tsv"
         for sp, lins in LINEAGES.items() for lin in lins],
    output:
        "results/tables/clustree/stability_summary_all_lineages.tsv",
    run:
        import pandas as pd
        pd.concat([pd.read_csv(f, sep="\t") for f in input]).to_csv(output[0], sep="\t", index=False)
