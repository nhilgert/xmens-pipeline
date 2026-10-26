# Stage 06 — diffusion pseudotime (global, mouse fibroblasts + decidual cells).
rule pseudotime_global:
    input:  "data/interim/annotated_mouse.h5ad"
    output: "data/interim/pt/global_mouse.h5ad"
    params: pt=config["pseudotime"]
    log:    "logs/pseudotime_global.log"
    script: "../scripts/pseudotime/pseudotime.py"
