# Stage 22 — stub: crossspecies.py exits with an error. Cross-species transfer is
# 10_samap.smk / 10b_samap_raw.smk.
rule crossspecies:
    input:  human="results/tables/mi_human.csv",
            mouse="results/tables/mi_mouse.csv"
    output: "results/tables/conservation.csv"
    log:    "logs/crossspecies.log"
    script: "../scripts/crossspecies/crossspecies.py"
