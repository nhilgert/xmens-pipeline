# Stage 08 — stub: cellrank.py exits with an error.
rule cellrank:
    input:  "data/interim/pt/global_mouse.h5ad"
    output: "data/interim/pt/cellrank_mouse.h5ad"
    log:    "logs/cellrank.log"
    script: "../scripts/pseudotime/cellrank.py"   # uses envs/cellrank
