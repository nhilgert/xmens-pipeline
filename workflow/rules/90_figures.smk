# Stage 90 — stub: figures.py exits with an error. Figure scripts live beside their
# arms (fig_spatial_qc.py, fig_zone_modules.py).
rule figures:
    input:
        expand("results/tables/markers_{sp}.csv", sp=SPECIES),
        expand("results/tables/mi_{sp}.csv", sp=SPECIES),
        "results/tables/conservation.csv",
    output: "results/figures/.done"
    log:    "logs/figures.log"
    script: "../scripts/crossspecies/figures.py"
