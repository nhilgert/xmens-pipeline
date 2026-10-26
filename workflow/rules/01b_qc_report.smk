# Stage 01b - QC diagnostic figures (decoupled from the filtering rule 01_qc).
#
# Reads the PRE-QC raw object (to show full distributions with threshold lines)
# and the POST-QC object (to report survival), and writes per-sample figures to
# results/qc/figures/{sample}/. Mirrors the project QC notebook's plots:
#   - total_counts, n_genes_by_counts histograms (log10) with MAD bounds
#   - pct_counts_mt histogram with the 8% mito bound
#   - doublet score distribution (if doublet calls present in the QC'd object)
#
# Filtering lives in 01_qc; this rule only visualizes. Runs in the same env as
# the spine (scanpy + matplotlib); does NOT need DoubletDetection.

rule qc_report:
    input:
        raw="data/interim/raw/{sample}.h5ad",
        qcd="data/interim/qc/{sample}.h5ad",
    output:
        # a sentinel so Snakemake tracks completion; figures go alongside it
        done=touch("results/qc/figures/{sample}/.done"),
    params:
        species=lambda w: samples.loc[w.sample, "species"],
        outdir=lambda w: f"results/qc/figures/{w.sample}",
    log:
        "logs/01b_qc_report_{sample}.log",
    script:
        "../scripts/ingest_qc/qc_report.py"
