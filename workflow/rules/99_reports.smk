import os

# 99_reports -- OFF the critical path on purpose.
#
# Executes the public de-novo notebooks with papermill to refresh figures and
# produce an executed copy. Nothing in rule all depends on this, so a broken
# dotplot can never stall the pipeline:
#
#     snakemake --profile profiles/slurm reports
#
# The pipeline and the notebooks share workflow/scripts/atlas/apply_annotation.py, so
# they cannot drift -- this only re-renders the human-readable record.

REPORT_NOTEBOOKS = [f"04_annotate_{sp}_denovo" for sp in config["lineages"]] + [
    f"04c_annotate_{sp}_{lin}_denovo"
    for sp, lins in config["lineages"].items() for lin in lins
]

rule render_notebook:
    input:
        nb="notebooks/public/{name}.ipynb",
    output:
        nb="results/notebooks/{name}.executed.ipynb",
    params:
        repo=config.get("repo", "."),
    log:
        "logs/99_report_{name}.log",
    shell:
        "papermill {input.nb} {output.nb} -p REPO {params.repo} "
        "--log-output --cwd {params.repo} > {log} 2>&1"

rule reports:
    input:
        expand("results/notebooks/{name}.executed.ipynb",
               name=[n for n in REPORT_NOTEBOOKS
                     if os.path.exists(f"notebooks/public/{n}.ipynb")]),
