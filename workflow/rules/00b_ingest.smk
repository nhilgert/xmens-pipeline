# Stage 00b — ingest CellBender-corrected counts + Slide-tags coords -> AnnData
#
# Reads the per-sample CellBender output (.h5, no metadata), attaches species/
# timepoint from the sample sheet (authoritative — the .h5 carries neither),
# joins the DBSCAN spatial coordinates from the fork's coords CSV (Source 1),
# keeps ALL cells (coords nullable), and writes data/interim/raw/{sample}.h5ad.
#
# See DATA_PROVENANCE.md for why we ingest from the .h5 rather than Andy's
# *_with_metadata.h5ad, and why coords come from Source 1 (barcode-keyed).

rule ingest:
    input:
        counts=lambda w: samples.loc[w.sample, "path"],
    output:
        h5ad="data/interim/raw/{sample}.h5ad",
    params:
        species=lambda w: samples.loc[w.sample, "species"],
        timepoint=lambda w: samples.loc[w.sample, "timepoint"],
        modality=lambda w: samples.loc[w.sample, "modality"],
        coords_id=lambda w: samples.loc[w.sample, "coords_id"],
        coords_dir=lambda w: config["coords"]["dir"],
        coords_pattern=lambda w: config["coords"]["pattern"],
        coords_x=lambda w: config["coords"]["x_col"],
        coords_y=lambda w: config["coords"]["y_col"],
    log:
        "logs/00b_ingest_{sample}.log",
    script:
        "../scripts/ingest_qc/ingest.py"
