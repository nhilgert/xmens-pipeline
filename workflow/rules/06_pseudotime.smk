# Stage 06 — stub: pseudotime.py exits with an error. Pseudotime is computed in
# 06b_timepoint_pseudotime, 15_pt_per_timepoint and 24_pseudotime_notebooks.
rule pseudotime_global:
    input:  "data/interim/annotated_mouse.h5ad"
    output: "data/interim/pt/global_mouse.h5ad"
    params: pt=config["pseudotime"]
    log:    "logs/pseudotime_global.log"
    script: "../scripts/pseudotime/pseudotime.py"
