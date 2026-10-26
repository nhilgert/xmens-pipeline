# Stage 04b - per-lineage scVI reintegration.
#
# Sibling of 02_integrate: identical scVI procedure ("re-integrated using
# previous scVI model parameters", Cevrim et al. Methods), applied to each
# lineage sub-object written by stage 04. HVGs are RESELECTED on each subset --
# that is the point of reintegrating, since the global fit blurs within-lineage
# structure.
#
# NO cell removal here. The Methods' UMAP-kNN MAD-outlier drop is not
# reproduced (computed on a superset; superseded by the clustree fine-structure
# step, which consumes the Leiden ladder this rule emits).
#
# GPU rule: profiles/slurm must route `reintegrate_lineage` to a GPU node the
# same way it routes `integrate` (--gres=gpu:1). Add the rule name there.
#
# Valid (species, lineage) pairs are species-specific -- mouse has perimetrium,
# human has perivascular -- so DON'T expand a naive product. config["lineages"]
# carries the per-species slugs.

LINEAGES = config["lineages"]

def lineage_latents():
    """All (species, lineage) latents. Add to rule all's input."""
    return [f"data/interim/latent_{sp}_{lin}.h5ad"
            for sp, lins in LINEAGES.items() for lin in lins]

rule reintegrate_lineage:
    input:
        h5ad="data/interim/lineages/{species}_{lineage}.h5ad",
    output:
        h5ad="data/interim/latent_{species}_{lineage}.h5ad",
    params:
        counts_layer="counts",          # same raw-counts layer as 02_integrate
    log:
        "logs/04b_reintegrate_{species}_{lineage}.log",
    script:
        "../scripts/atlas/reintegrate.py"
