# 04c_reannotate -- apply a committed per-lineage choices CSV to a reintegrated
# object. Same contract as 04_annotate: the CSV is an input, so the pipeline
# stops (loudly, by name) on any lineage whose call hasn't been made yet.

import yaml as _yaml
_SPEC = _yaml.safe_load(open("resources/lineage_spec.yaml"))

def LINEAGE_RES(species, lineage):
    """Resolution for this lineage, from the spec (falls back to defaults)."""
    lin = _SPEC["species"][species]["lineages"][lineage]
    return lin.get("res", _SPEC.get("defaults", {}).get("res", "leiden_scVI_2.0"))


rule reannotate_lineage:
    input:
        h5ad="data/interim/latent_{species}_{lineage}.h5ad",
        csv="resources/annotations/{species}_{lineage}.csv",
        spec="resources/lineage_spec.yaml",
        vocab="resources/annotation_vocabulary.yaml",
    output:
        h5ad="data/interim/reannotated/{species}_{lineage}.h5ad",
    params:
        level="fine",
        species=lambda w: w.species,
        lineage=lambda w: w.lineage,
        res=lambda w: LINEAGE_RES(w.species, w.lineage),
        repo=config.get("repo", "."),
    log:
        "logs/04c_reannotate_{species}_{lineage}.log",
    script:
        "../scripts/atlas/apply_annotation.py"
