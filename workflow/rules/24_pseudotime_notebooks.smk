# Pseudotime notebooks as DAG rules (dependency-free papermill-style runner).
#
# Not included in the Snakefile. To enable, add
#     include: "rules/24_pseudotime_notebooks.smk"
# Standalone dry-run:
#     snakemake -n -s workflow/rules/24_pseudotime_notebooks.smk --directory . pt_nb_all
#
# Executing these rules trains scVI (GPU) into results/pt_notebook_rules/<v>/, new
# directories only; deliverables, RTK objects and existing caches are inputs, never
# written. The embedding cache lives inside the rule's own OUT_DIR, so snakemake's
# mtime model is the only caching layer.

# Loaded only when this file runs standalone; the Snakefile has already set config.
if not config:
    configfile: "config/config.yaml"

PTNB = {
  "mouse_with_uninduced": dict(nb="notebooks/pseudotime_papermill/mouse_with_uninduced_pm.ipynb",
                               src="data/deliverables/2026_09_15/Cevrim_XMens_Only_Fib_Dec_Cells.h5ad"),
  "mouse_no_uninduced":   dict(nb="notebooks/pseudotime_papermill/mouse_no_uninduced_pm.ipynb",
                               src="data/deliverables/2026_09_15/Cevrim_XMens_Only_Fib_Dec_Cells.h5ad"),
  "human":                dict(nb="notebooks/pseudotime_papermill/human_pm.ipynb",
                               src="data/deliverables/2026_09_15/Cevrim_Human_Only_Fib_Dec_Cells.h5ad"),
}


# Target: every notebook variant executed.
rule pt_nb_all:
    input: expand("results/pt_notebook_rules/{v}/executed.ipynb", v=PTNB)


# One executed notebook per variant; SRC_PATH and OUT_DIR are papermill parameters.
rule pt_notebook:
    input:
        nb=lambda wc: PTNB[wc.v]["nb"],
        src=lambda wc: PTNB[wc.v]["src"],
        runner="workflow/scripts/pseudotime/run_notebook.py",
    output:
        executed="results/pt_notebook_rules/{v}/executed.ipynb",
    threads: 8
    resources: gpu=1, mem_mb=200000, runtime=480   # MB; minutes
    shell:
        "python {input.runner} --nb {input.nb} --out {output.executed} "
        "-p SRC_PATH {input.src} -p OUT_DIR results/pt_notebook_rules/{wildcards.v}"
