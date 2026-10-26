# Stage 01c - concatenate per-sample QC'd objects into a per-species object.
#
# Reads all QC'd samples for one species, concatenates (keeping ALL cells),
# harmonizes obs (standardize to 'sample', add clean 'cell_barcode' with the
# -1 suffix stripped for joining to collaborator objects), and preserves the
# counts layer, spatial coords, and all CellBender/DoubletDetection/QC columns.
#
# Output feeds 02_integrate (scVI). CPU rule (no GPU) - runs cheaply; kept
# separate from integrate so the GPU job isn't doing file IO and so this
# checkpoint is cached/reusable (e.g. for the colleague-comparison notebook).

# QC'd per-sample objects of one species: every sample of that species in the sheet.
def species_samples(wildcards):
    sub = samples[samples["species"] == wildcards.species]
    return [f"data/interim/qc/{s}.h5ad" for s in sub.index]


# One concatenated object per species; the input of 02_integrate.
rule concat:
    input:
        species_samples,
    output:
        h5ad="data/interim/qc_passed_{species}.h5ad",
    params:
        species=lambda w: w.species,
    log:
        "logs/01c_concat_{species}.log",
    script:
        "../scripts/ingest_qc/concat.py"
