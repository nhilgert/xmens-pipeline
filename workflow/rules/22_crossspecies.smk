# Stage 22 — cross-species conservation: orthogroups + threshold-free rank comparison.
rule crossspecies:
    input:  human="results/tables/mi_human.csv",
            mouse="results/tables/mi_mouse.csv"
    output: "results/tables/conservation.csv"
    log:    "logs/crossspecies.log"
    script: "../scripts/crossspecies/crossspecies.py"
