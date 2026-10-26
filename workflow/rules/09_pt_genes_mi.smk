# Stage 09 — stub: pt_genes_mi.py exits with an error. The MI arm is 11_mi_pseudotime.smk.
rule pt_genes_mi:
    input:  "data/interim/annotated_{species}.h5ad"
    output: "results/tables/mi_{species}.csv"
    params: mi=config["mi"]
    log:    "logs/pt_genes_mi_{species}.log"
    script: "../scripts/mi/pt_genes_mi.py"
