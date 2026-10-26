# Stage 00a — Slide-tags spatial coordinates (per-cell x,y)
#
# The Slide-tags mapping pipeline is vendored as a pinned submodule at
# external/slidetags (nhilgert/slide-tags @ v0.1-fasrc). It is a separate
# pipeline: own conda env (`slidetags`) + Julia project, and it submits its
# own SLURM jobs via slidetag_pipe.sh. We do NOT execute it from inside this
# DAG (that would nest cluster-job submission and tangle env systems). The
# boundary between the two pipelines is the per-sample coords CSV.
#
# Run the fork separately (see external/slidetags/FASRC_SETUP.md) to produce a
# barcode,x,y CSV per sample; this rule normalizes it into the xmens layout.

# CONFIRMED 2026-09-14: the fork's per-sample coords live where config["coords"]
# already points (the same source stage 00b reads via params) -- dir + pattern
# keyed by coords_id from the sample sheet. The old default here was a
# placeholder path that never existed; see .pre_coords_fix.bak for the original.


rule slidetags:
    input:
        fork_coords=lambda w: config["coords"]["dir"] + "/"
            + config["coords"]["pattern"].format(coords_id=samples.loc[w.sample, "coords_id"]),
    output:
        coords="data/interim/coords/{sample}.csv",
    log:
        "logs/00a_slidetags_{sample}.log",
    script:
        "../scripts/ingest_qc/slidetags.py"
