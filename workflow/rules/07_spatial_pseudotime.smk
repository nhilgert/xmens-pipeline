# Stage 07 — stub: spatial_pseudotime.py exits with an error. The spatial GP arm is
# 12_spatial_gp.smk.
rule spatial_pseudotime:
    input:  coords="data/interim/coords/{sample}.csv"
    output: "data/interim/pt/spatial_{sample}.h5ad"
    log:    "logs/spatial_pt_{sample}.log"
    script: "../scripts/pseudotime/spatial_pseudotime.py"
