# 04_annotate -- apply the committed atlas choices CSV.
#
# The CSV is an INPUT: judgment enters the DAG here. Missing it stops the run
# with "MissingInputException" naming the decision you haven't made yet; editing
# it re-runs everything downstream on the next invocation.
#
# One rule per species so the lineage subsets can be declared as real outputs
# (they're what 04b_reintegrate consumes).

import yaml as _yaml
_SPEC = _yaml.safe_load(open("resources/lineage_spec.yaml"))


for _sp, _lins in config["lineages"].items():

    rule:
        name:
            f"annotate_{_sp}"
        input:
            h5ad="data/interim/latent_{sp}.h5ad".format(sp=_sp),
            csv="resources/annotations/{sp}_lineage.csv".format(sp=_sp),
            spec="resources/lineage_spec.yaml",
            vocab="resources/annotation_vocabulary.yaml",
        output:
            h5ad="data/interim/annotated_{sp}.h5ad".format(sp=_sp),
            subsets=["data/interim/lineages/{sp}_{lin}.h5ad".format(sp=_sp, lin=_l)
                     for _l in _lins],
            ladder="results/tables/clusters/{sp}/leiden_all_resolutions.csv".format(sp=_sp),
        params:
            level="atlas",
            species=_sp,
            res=_SPEC["atlas"][_sp]["res"],   # single source of truth = the spec
            repo=config.get("repo", "."),
        log:
            "logs/04_annotate_{sp}.log".format(sp=_sp),
        script:
            "../scripts/atlas/apply_annotation.py"
