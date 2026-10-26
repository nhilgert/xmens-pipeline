# Scratch-only, seeded per-timepoint mouse pseudotime tests.
#
# Deliberately NOT included in rule all or samap_all.  The input is the staged
# migration and every output is under config[pseudotime_tests][test_output_dir].

PT_CFG = config["pseudotime_tests"]
PT_STAGES = PT_CFG["timepoints"]
PT_TEST_DIR = PT_CFG["test_output_dir"]
# Keep cross-timepoint settings in one immutable mapping, but do not attach the
# entire timepoint table to every job's provenance.  A 4dpi root correction
# must not make the independent 1dpi or 3dpi artifacts stale.
PT_GLOBAL_CFG = {key: value for key, value in PT_CFG.items() if key != "timepoints"}


rule compute_pseudotime:
    input:
        h5ad=PT_CFG["input_h5ad"],
    output:
        h5ad=f"{PT_TEST_DIR}/adata_{{pt_stage}}.h5ad",
        metrics=f"{PT_TEST_DIR}/{{pt_stage}}_metrics.json",
        subtypes=f"{PT_TEST_DIR}/{{pt_stage}}_subtype_pseudotime.tsv",
    params:
        pt_config=PT_GLOBAL_CFG,
        stage_config=lambda wildcards: PT_STAGES[wildcards.pt_stage],
        diagnostic_only=False,
    log:
        f"{PT_TEST_DIR}/logs/compute_{{pt_stage}}.log",
    wildcard_constraints:
        pt_stage="1dpi_late|4dpi|3dpi",
    script:
        "../scripts/pseudotime/compute_pseudotime.py"


rule diagnose_3dpi_pseudotime_roots:
    input:
        h5ad=PT_CFG["input_h5ad"],
    output:
        candidates=f"{PT_TEST_DIR}/3dpi_root_candidates.tsv",
        subtypes=f"{PT_TEST_DIR}/3dpi_candidate_subtype_pseudotime.tsv",
        summary=f"{PT_TEST_DIR}/3dpi_diagnostic_summary.json",
    params:
        pt_config=PT_GLOBAL_CFG,
        stage_config=PT_STAGES["3dpi"],
        diagnostic_only=True,
    log:
        f"{PT_TEST_DIR}/logs/diagnose_3dpi_roots.log",
    script:
        "../scripts/pseudotime/compute_pseudotime.py"
