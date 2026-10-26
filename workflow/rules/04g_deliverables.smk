# 04g_deliverables -- finalize_deliverables: promote the canonical source
# objects into an immutable, dated, sha256-manifested release under
# data/deliverables/<release>/.
#
# This is the pipeline's OUTBOUND boundary: everything downstream of the spine
# (SAMap, SCENIC, LIANA, pt-per-timepoint, the papermill notebooks) reads
# data/deliverables/<release>/<name>.h5ad, never a hand-copied file. Cutting a
# release is deliberate: bump config[deliverables][release], point the sources at
# the new objects, then
#     snakemake --rerun-triggers mtime finalize_deliverables
#
# Sources are ancient(): a release is cut on purpose, never re-cut because a
# source mtime moved. Releases are immutable -- the script refuses to overwrite
# a release file whose sha256 differs. NOT in rule all.

DELIV     = config["deliverables"]
DELIV_REL = DELIV["release"]
DELIV_DIR = f"data/deliverables/{DELIV_REL}"
DELIV_SRC = dict(DELIV["sources"])            # name -> absolute source path
MIGRATE   = config["vocab_migration"]
SPLITS    = MIGRATE.get("subtype_splits", {})


# Relabels the frozen source release to the current annotation vocabulary
# (resources/annotation_vocabulary.yaml legacy_aliases), then applies any
# cell-level subtype splits (config vocab_migration.subtype_splits), so a
# vocabulary change reaches a new release through this rule and
# finalize_deliverables alone.
# The vocabulary and the decision CSVs are also inputs of 04/04c/04d, so run it
# targeted, never as part of a full run (that would re-annotate against
# data/interim/latent_*.h5ad):
#     snakemake finalize_deliverables \
#         --allowed-rules migrate_release_vocabulary finalize_deliverables
rule migrate_release_vocabulary:
    input:
        objects=[ancient(f"data/deliverables/{MIGRATE['from_release']}/{name}.h5ad")
                 for name in DELIV_SRC],
        vocab="resources/annotation_vocabulary.yaml",
        # Split inputs: the clustering is ancient() (pinned by sha256 in the script),
        # the decision CSVs are not -- editing a call re-cuts the release on purpose.
        split_clusters=[ancient(s["clusters"]) for s in SPLITS.values()],
        split_decisions=[s["decisions"] for s in SPLITS.values()],
    output:
        objects=[f"{MIGRATE['outdir']}/{name}.h5ad" for name in DELIV_SRC],
        report=f"results/tables/vocab_migration_{DELIV_REL}.tsv",
    params:
        splits=SPLITS,
    log:
        f"logs/04g_migrate_release_vocabulary_{DELIV_REL}.log",
    resources:
        mem_mb=8000,   # objects are migrated one at a time
        runtime=30,    # measured 3 min
    script:
        "../scripts/deliverables/migrate_vocabulary.py"


# Copies the sources into the release directory and writes the sha256 manifest.
rule finalize_deliverables:
    input:
        sources=[ancient(p) for p in DELIV_SRC.values()],
        vocab="resources/annotation_vocabulary.yaml",
    output:
        manifest=f"{DELIV_DIR}/MANIFEST.json",
        objects=[f"{DELIV_DIR}/{name}.h5ad" for name in DELIV_SRC],
    params:
        sources=DELIV_SRC,
        release=DELIV_REL,
        note=DELIV.get("note", ""),
    log:
        f"logs/04g_finalize_deliverables_{DELIV_REL}.log",
    resources:
        mem_mb=32000,
        runtime=120,    # minutes
    script:
        "../scripts/deliverables/finalize_deliverables.py"
