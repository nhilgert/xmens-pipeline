# Stage 21 — SCENIC gene regulatory networks / TF activity (per species).
rule scenic:
    input:  "data/interim/annotated_{species}.h5ad"
    output: "results/tables/scenic_{species}.csv"
    log:    "logs/scenic_{species}.log"
    script: "../scripts/scenic/scenic.py"     # uses envs/scenic
