# 04d_merge -- concatenate the cleaned per-lineage objects into one per-species
# object. A cluster whose table assigns it another lineage's cell type (a throwback,
# e.g. human stromal cluster 13 -> perivascular_cell) already carries that label from
# 04c, so nothing moves here.
# Output has NO embedding: composition changed, so 04e re-embeds from scratch.

rule merge_reannotated:
    input:
        parts=lambda w: expand(
            "data/interim/reannotated/{species}_{lineage}.h5ad",
            species=w.species, lineage=config["lineages"][w.species]),
        vocab="resources/annotation_vocabulary.yaml",
    output:
        h5ad="data/interim/refined_{species}.h5ad",
        report="results/tables/refined_{species}_guardrail.tsv",
    params:
        species=lambda w: w.species,
    log:
        "logs/04d_merge_{species}.log",
    script:
        "../scripts/atlas/merge_reannotated.py"
