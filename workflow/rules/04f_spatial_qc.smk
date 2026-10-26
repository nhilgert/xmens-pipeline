# 04f_spatial_qc -- "blank, don't drop" spatial-coordinate QC, applied AFTER
# annotation.
#
# Purely spatial: only coordinates drive it (kNN distance, exact duplicates,
# hand-drawn exclusion boxes) and labels each kept cell's physical cross-section
# (`tissue_section`, split by a config line for multi-section pucks). Annotations
# are present on the input object but
# are never an input to a QC decision. No cell is removed -- an untrustworthy
# coordinate is set to NaN, so cell counts, expression, and every annotation are
# byte-identical to the input; only positions change.
#
# Reads the annotated per-species object so the QC'd coordinates land on the same
# object the downstream spatial arms consume. Produces a NEW object; the raw
# x_um_dbscan/y_um_dbscan columns are preserved untouched. Nothing downstream is
# repointed by this rule -- switching a spatial consumer from x_um_dbscan to the
# QC'd column is a separate, deliberate edit.
#
# Scoped to config['spatial_qc']['species'] (mouse only for now; human deferred
# until the slide-tags positioning rerun). Rotations default to 0 (identity).

SQC          = config["spatial_qc"]
SQC_SPECIES  = [s for s in SQC.get("species", []) if s in SPECIES]


def spatial_qc_targets():
    return expand("data/interim/qc_spatial_{s}.h5ad", s=SQC_SPECIES)


rule spatial_qc:
    input:
        h5ad="data/interim/annotated_{species}.h5ad",
    output:
        h5ad="data/interim/qc_spatial_{species}.h5ad",
        tally="results/tables/spatial_qc_{species}.csv",
    params:
        sqc=SQC,
    wildcard_constraints:
        species="|".join(SQC_SPECIES) if SQC_SPECIES else "__nospatialqc__",
    log:
        "logs/04f_spatial_qc_{species}.log",
    script:
        "../scripts/atlas/spatial_qc.py"


rule spatial_qc_all:
    input:
        spatial_qc_targets(),
