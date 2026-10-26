# Stage 21 — stub: scenic.py exits with an error. The SCENIC arm is 13_scenic.smk.
rule scenic:
    input:  "data/interim/annotated_{species}.h5ad"
    output: "results/tables/scenic_{species}.csv"
    log:    "logs/scenic_{species}.log"
    script: "../scripts/scenic/scenic.py"     # uses envs/scenic
