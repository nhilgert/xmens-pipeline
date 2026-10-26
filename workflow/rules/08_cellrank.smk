# Stage 08 — CellRank fate / probabilistic layer (on the global pseudotime).
rule cellrank:
    input:  "data/interim/pt/global_mouse.h5ad"
    output: "data/interim/pt/cellrank_mouse.h5ad"
    log:    "logs/cellrank.log"
    script: "../scripts/pseudotime/cellrank.py"   # uses envs/cellrank
