# 04d_merge -- concatenate the cleaned per-lineage objects into one per-species
# object. Throwbacks already carry their final label, so nothing moves here.
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
        target_n=lambda w: config.get("published_n", {}).get(w.species, 0),
    log:
        "logs/04d_merge_{species}.log",
    script:
        "../scripts/atlas/merge_reannotated.py"
