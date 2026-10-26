# Stage 07 — per-sample spatial pseudotime + GPR gradient field (slide-tags samples).
rule spatial_pseudotime:
    input:  coords="data/interim/coords/{sample}.csv"
    output: "data/interim/pt/spatial_{sample}.h5ad"
    log:    "logs/spatial_pt_{sample}.log"
    script: "../scripts/pseudotime/spatial_pseudotime.py"
