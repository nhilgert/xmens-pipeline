# Stage 23 — in-vitro decidualization vs in-vivo comparison.
rule invitro:
    input:  "data/interim/annotated_human.h5ad"
    output: "results/tables/invitro_comparison.csv"
    log:    "logs/invitro.log"
    script: "../scripts/crossspecies/invitro.py"
