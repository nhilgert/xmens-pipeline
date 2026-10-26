# Stage 01 — quality control (Cevrim et al. 2025 Methods).
#
# 500-UMI floor, 3-MAD adaptive filters on log1p(counts) and log1p(n_genes),
# mito > 8% excluded, DoubletDetection (40 iters, voter_thresh 0.5). Operates on
# all cells from 00b. DoubletDetection lives in a separate (CUDA) env; point
# config['envs']['doubletdetection'] at it. If unset/missing, qc.py skips
# doublet removal with a warning (so dry-runs and partial setups still work).

rule qc:
    input:
        h5ad="data/interim/raw/{sample}.h5ad",
    output:
        h5ad="data/interim/qc/{sample}.h5ad",
    params:
        species=lambda w: samples.loc[w.sample, "species"],
    conda:
        config.get("envs", {}).get("doubletdetection", None)
    log:
        "logs/01_qc_{sample}.log",
    script:
        "../scripts/ingest_qc/qc.py"
