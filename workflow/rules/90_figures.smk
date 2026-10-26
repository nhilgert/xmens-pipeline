# Stage 90 — assemble figure panels from the tables above.
rule figures:
    input:
        expand("results/tables/markers_{sp}.csv", sp=SPECIES),
        expand("results/tables/mi_{sp}.csv", sp=SPECIES),
        "results/tables/conservation.csv",
    output: "results/figures/.done"
    log:    "logs/figures.log"
    script: "../scripts/crossspecies/figures.py"
