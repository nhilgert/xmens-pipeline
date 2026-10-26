# Stage 23 — stub: invitro.py exits with an error.
rule invitro:
    input:  "data/interim/annotated_human.h5ad"
    output: "results/tables/invitro_comparison.csv"
    log:    "logs/invitro.log"
    script: "../scripts/crossspecies/invitro.py"
