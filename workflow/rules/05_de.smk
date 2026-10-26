# Stage 05 — marker DE (Wilcoxon).
rule de:
    input:  "data/interim/annotated_{species}.h5ad"
    output: "results/tables/markers_{species}.csv"
    log:    "logs/de_{species}.log"
    script: "../scripts/de/de.py"



# ---- per-lineage marker tables (2026-09-02) ---------------------------------
# clusters : one-vs-rest on the spec resolution of the lineage latent -- the
#            over-clustered evidence behind "all panel markers were recovered de novo".
# subtypes : one-vs-rest on cell_subtype of the REANNOTATED object, excluded cells
#            dropped -- the supplementary marker table for the final annotation.
#            Downstream of resources/annotations/{species}_{lineage}.csv, so it
#            regenerates whenever a merge call changes.

DE_TOP_N = config.get("de", {}).get("top_n", 200)

rule de_lineage_clusters:
    input:
        h5ad="data/interim/latent_{species}_{lineage}.h5ad",
    output:
        tsv="results/tables/de/{species}_{lineage}_clusters_top{n}.tsv",
    wildcard_constraints:
        n=r"\d+",
    params:
        level="clusters",
        res=lambda w: LINEAGE_RES(w.species, w.lineage),
        top_n=lambda w: w.n,
    log:
        "logs/05_de_{species}_{lineage}_clusters_top{n}.log",
    script:
        "../scripts/de/de_lineage.py"


rule de_lineage_subtypes:
    input:
        h5ad="data/interim/reannotated/{species}_{lineage}.h5ad",
    output:
        tsv="results/tables/de/{species}_{lineage}_subtypes_top{n}.tsv",
    wildcard_constraints:
        n=r"\d+",
    params:
        level="subtypes",
        res=lambda w: LINEAGE_RES(w.species, w.lineage),
        top_n=lambda w: w.n,
    log:
        "logs/05_de_{species}_{lineage}_subtypes_top{n}.log",
    script:
        "../scripts/de/de_lineage.py"


rule de_lineages:
    input:
        [f"results/tables/de/{sp}_{lin}_{lvl}_top{DE_TOP_N}.tsv"
         for sp, lins in LINEAGES.items() for lin in lins for lvl in ("clusters", "subtypes")],
