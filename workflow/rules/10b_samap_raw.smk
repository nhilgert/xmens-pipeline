# COMPLEMENTARY to 10_samap.smk: this file is the VANILLA/RAW arm (the paper arm); 10 is the transfer arm. Both are live. (2026-09-13)
# workflow/rules/10b_samap_raw.smk
# =============================================================================
# SAMap RAW-COUNTS arm — the all-timepoint ("full object") fib/dec integration
# on RAW INTEGER COUNTS, wrapping the standalone work in
# /n/lab_storage/.../code/samap-scvi/full_fibdec into the DAG.
#
#   v2 deliverables --(samap_raw_prep, --scale raw)--> inputs_raw/{hs,mm}_fib_dec_all_raw.h5ad
#     --(samap_raw_fit x3 arms)--> runs_raw/<arm>/{samap.pkl, eval.json}
#     --(samap_raw_stain)-->       runs_raw/<arm>/staining/*.csv
#     --(samap_raw_transferred)--> runs_raw/<arm>/samap_transferred.h5ad
#     --(samap_raw_pt_transfer)--> runs_raw/<arm>/pt_transfer/pt_transferred_hs.csv
#     --(samap_raw_attach_pt)-->   runs_raw/<arm>/samap_transferred_pt.h5ad
#     --(samap_raw_compare)-->     runs_raw/summary.csv
#
# INDEPENDENT of 10_samap.smk (the CP10K per-timepoint-pair paper arm): different
# root, different scripts, different env. Nothing here reads or writes 10_samap paths.
#
# All heavy outputs already exist under ROOT (fits 2026-09-01, stains/transfers
# 2026-09-02), so `snakemake samap_raw_all` only builds what is missing —
# currently the pt-attach step. Scripts are ancient() where outputs pre-exist,
# mirroring 10_samap's "fit is expensive, don't re-fit on edit" convention.
#
# The 2026_09_02_samap_raw_all.h5ad on netscratch is a dated DEPLOY COPY of
# runs_raw/vanilla_raw/samap_transferred(_pt).h5ad (+ synced QC coords); deploying
# to netscratch stays manual on purpose.
#
# ENV — samap_harmony_uv (uv env on lab_storage): stock sc-samap v3 with
# harmonypy PINNED to 0.0.10 (the >=2.0 API breaks SAMap's built-in batch_key
# hook; see code/samap-scvi/harmonypy2-fix.patch).
#
# config.yaml is NOT required — every key below has an inline default. To
# override, add a block:
#   samap_raw:
#     root: ...          # external project root (holds inputs_raw/, runs_raw/)
#     python: ...        # samap_harmony_uv interpreter, ABSOLUTE
#     maps_dir: ...      # parent of hsmm/ (SAMap convention)
#     n_iter: 3
#     pt_h5ad: ...       # mouse object carrying pseudotime
#     pt_col: dpt_pseudotime
#     pt_arms: [vanilla_raw]
# Snakefile:  include: "rules/10b_samap_raw.smk"
# =============================================================================
import re as _sr_re

_SR       = config.get("samap_raw", {})
SR_ROOT   = _SR.get("root", "/n/lab_storage/eddy_lab/users/nhilgert/code/samap-scvi/full_fibdec")
SR_PY     = _SR.get("python", "/n/lab_storage/eddy_lab/users/nhilgert/.uv/envs/samap_harmony_uv/bin/python")
SR_MAPS   = _SR.get("maps_dir", "/n/lab_storage/eddy_lab/users/nhilgert/decidualization/revisions/samap_integrate/homology/hsmm_hmmer")
SR_NITER  = _SR.get("n_iter", 3)

# same v2 deliverables 10_samap reads; ancient() so the annotation DAG stays isolated
# 2026-09-15: repointed at the finalized release (04g finalize_deliverables)
_SR_SRC = {
    "hs": "data/deliverables/2026_09_15/Cevrim_Human_Only_Fib_Dec_Cells.h5ad",
    "mm": "data/deliverables/2026_09_15/Cevrim_XMens_Only_Fib_Dec_Cells.h5ad",
}
SR_PT_H5AD = _SR.get("pt_h5ad", "data/deliverables/2026_09_15/Cevrim_XMens_Only_Fib_Dec_Cells.h5ad")
SR_PT_COL  = _SR.get("pt_col", "dpt_pseudotime")
# pt-transfer variants (2026-09-15): suffix -> obs column. "" keeps the legacy
# path AND column, so existing outputs are never re-pointed; new variants land in
# sibling directories. Override in config[samap_raw][pt_variants].
SR_PT_VARIANTS = _SR.get("pt_variants", {"": SR_PT_COL,
                                         "_noapop": "dpt_pseudotime_no_apoptotic"})

# arm name -> (run_samap_arm.py --arm, --min-expression)
SR_ARMS = {
    "vanilla_raw":          ("vanilla", 1),  # by-the-book: raw counts, SAM defaults
    "harmony_raw":          ("harmony", 1),  # + harmonypy inside each SAM iteration
    "vanilla_raw_nofilter": ("vanilla", 0),  # isolates the min_expression cut
}
SR_PT_ARMS = _SR.get("pt_arms", ["vanilla_raw"])
assert all(a in SR_ARMS for a in SR_PT_ARMS), f"unknown pt_arm in {SR_PT_ARMS}"

_SR_STAIN_CSVS = ["hs_from_mm_coarse", "hs_from_mm_fine", "mm_from_hs_coarse"]

wildcard_constraints:
    arm   = "|".join(_sr_re.escape(a) for a in SR_ARMS),
    ptsuf = "(" + "|".join(_sr_re.escape(v) for v in sorted(SR_PT_VARIANTS, key=len)) + ")",
    rawsp = "hs|mm",

# -----------------------------------------------------------------------------
# PREP — deliverable -> one all-timepoint RAW-counts SAMap input. Guards assert
# integer values, count-scale max, variable library sizes (CV>0.05), .raw deleted.
rule samap_raw_prep:
    input:
        src    = lambda wc: ancient(_SR_SRC[wc.rawsp]),
        script = ancient("workflow/scripts/samap/prep_samap_full_input.py"),
    output:
        h5ad = f"{SR_ROOT}/inputs_raw/{{rawsp}}_fib_dec_all_raw.h5ad",
    params:
        py = SR_PY,
    resources:
        mem_mb  = 96000,
        runtime = 60,
    log:
        "logs/samap_raw/prep_{rawsp}.log",
    shell:
        r"""
        module load gcc/14.2.0-fasrc01
        export HDF5_USE_FILE_LOCKING=FALSE
        {params.py} {input.script} \
            --src {input.src} --species {wildcards.rawsp} \
            --coarse-col cell_type --fine-col cell_subtype \
            --scale raw --out {output.h5ad} \
            > {log} 2>&1
        """

# -----------------------------------------------------------------------------
# FIT + EVAL — one arm: SAM per species (input scale re-verified and logged at
# load time), SAMAP stitch, then the mixing/purity/alignment evaluation.
rule samap_raw_fit:
    input:
        hs     = f"{SR_ROOT}/inputs_raw/hs_fib_dec_all_raw.h5ad",
        mm     = f"{SR_ROOT}/inputs_raw/mm_fib_dec_all_raw.h5ad",
        script = ancient("workflow/scripts/samap/run_samap_arm.py"),
    output:
        pkl  = f"{SR_ROOT}/runs_raw/{{arm}}/samap.pkl",
        eval = f"{SR_ROOT}/runs_raw/{{arm}}/eval.json",
    params:
        py      = SR_PY,
        maps    = SR_MAPS,
        base    = lambda wc: SR_ARMS[wc.arm][0],
        minexp  = lambda wc: SR_ARMS[wc.arm][1],
        out_dir = f"{SR_ROOT}/runs_raw/{{arm}}",
        n_iter  = SR_NITER,
    threads: 8
    resources:
        mem_mb  = 128000,
        runtime = 600,
    log:
        "logs/samap_raw/fit_{arm}.log",
    shell:
        r"""
        module load gcc/14.2.0-fasrc01
        export HDF5_USE_FILE_LOCKING=FALSE
        export PYTHONPATH=workflow/scripts    # samap_scvi.py (scvi arm only)
        {params.py} {input.script} \
            --hs {input.hs} --mm {input.mm} --maps-dir {params.maps} \
            --arm {params.base} --out-dir {params.out_dir} \
            --batch-key sample --min-expression {params.minexp} \
            --n-iter {params.n_iter} --ncpus {threads} \
            > {log} 2>&1
        """

# -----------------------------------------------------------------------------
# LABEL TRANSFER — argmax connectivity-weighted vote of cross-species neighbors,
# per cell; CSVs + spatial stain panels under runs_raw/<arm>/staining/.
rule samap_raw_stain:
    input:
        pkl    = f"{SR_ROOT}/runs_raw/{{arm}}/samap.pkl",
        script = ancient("workflow/scripts/samap/samap_label_stain_full.py"),
    output:
        csvs = expand(f"{SR_ROOT}/runs_raw/{{{{arm}}}}/staining/{{stem}}.csv",
                      stem=_SR_STAIN_CSVS),
    params:
        py      = SR_PY,
        out_dir = f"{SR_ROOT}/runs_raw/{{arm}}/staining",
    resources:
        mem_mb  = 128000,
        runtime = 120,
    log:
        "logs/samap_raw/stain_{arm}.log",
    shell:
        r"""
        module load gcc/14.2.0-fasrc01
        export HDF5_USE_FILE_LOCKING=FALSE
        {params.py} {input.script} \
            --samap-pkl {input.pkl} --out-dir {params.out_dir} \
            > {log} 2>&1
        """

# -----------------------------------------------------------------------------
# EXPORT — collaborator-ready joint h5ad: stitched object + unified obs +
# transferred labels/confidences + uns['provenance'].
rule samap_raw_transferred:
    input:
        pkl    = f"{SR_ROOT}/runs_raw/{{arm}}/samap.pkl",
        csvs   = expand(f"{SR_ROOT}/runs_raw/{{{{arm}}}}/staining/{{stem}}.csv",
                        stem=_SR_STAIN_CSVS),
        script = ancient("workflow/scripts/samap/make_samap_transferred_h5ad.py"),
    output:
        h5ad = f"{SR_ROOT}/runs_raw/{{arm}}/samap_transferred.h5ad",
    params:
        py      = SR_PY,
        arm_dir = f"{SR_ROOT}/runs_raw/{{arm}}",
    resources:
        mem_mb  = 160000,
        runtime = 120,
    log:
        "logs/samap_raw/transferred_{arm}.log",
    shell:
        r"""
        module load gcc/14.2.0-fasrc01
        export HDF5_USE_FILE_LOCKING=FALSE
        {params.py} {input.script} --arm-dir {params.arm_dir} > {log} 2>&1
        """

# -----------------------------------------------------------------------------
# PT TRANSFER — mouse dpt_pseudotime -> human cells, Nadaraya-Watson over the
# cross-species block of the stitched graph; three variants side by side.
rule samap_raw_pt_transfer:
    input:
        joint  = f"{SR_ROOT}/runs_raw/{{arm}}/samap_transferred.h5ad",
        pt     = ancient(SR_PT_H5AD),
        script = ancient("workflow/scripts/samap/samap_transfer_pt.py"),
    output:
        csv     = f"{SR_ROOT}/runs_raw/{{arm}}/pt_transfer{{ptsuf}}/pt_transferred_hs.csv",
        summary = f"{SR_ROOT}/runs_raw/{{arm}}/pt_transfer{{ptsuf}}/summary.json",
    params:
        py      = SR_PY,
        pt_col  = lambda wc: SR_PT_VARIANTS[wc.ptsuf],
        out_dir = f"{SR_ROOT}/runs_raw/{{arm}}/pt_transfer{{ptsuf}}",
    resources:
        mem_mb  = 128000,
        runtime = 120,
    log:
        "logs/samap_raw/pt_transfer_{arm}{ptsuf}.log",
    shell:
        r"""
        module load gcc/14.2.0-fasrc01
        export HDF5_USE_FILE_LOCKING=FALSE
        {params.py} {input.script} \
            --joint-h5ad {input.joint} --pt-h5ad {input.pt} \
            --pt-col {params.pt_col} --out-dir {params.out_dir} \
            > {log} 2>&1
        """

# -----------------------------------------------------------------------------
# ATTACH — pt CSV -> obs columns (transferred_mm_dpt_pseudotime{,_ranknorm,
# _ranknorm_insample,_weight}) on a NEW file; provenance updated.
rule samap_raw_attach_pt:
    input:
        joint  = f"{SR_ROOT}/runs_raw/{{arm}}/samap_transferred.h5ad",
        csv    = f"{SR_ROOT}/runs_raw/{{arm}}/pt_transfer{{ptsuf}}/pt_transferred_hs.csv",
        script = "workflow/scripts/samap/samap_attach_pt.py",
    output:
        h5ad = f"{SR_ROOT}/runs_raw/{{arm}}/samap_transferred_pt{{ptsuf}}.h5ad",
    params:
        py = SR_PY,
    resources:
        mem_mb  = 128000,
        runtime = 90,
    log:
        "logs/samap_raw/attach_pt_{arm}{ptsuf}.log",
    shell:
        r"""
        module load gcc/14.2.0-fasrc01
        export HDF5_USE_FILE_LOCKING=FALSE
        {params.py} {input.script} \
            --joint {input.joint} --pt-csv {input.csv} --out {output.h5ad} \
            > {log} 2>&1
        """

# -----------------------------------------------------------------------------
# COMPARE — collate the arms' eval.json into runs_raw/summary.csv
rule samap_raw_compare:
    input:
        evals  = expand(f"{SR_ROOT}/runs_raw/{{arm}}/eval.json", arm=SR_ARMS),
        script = ancient("workflow/scripts/samap/compare_samap_arms.py"),
    output:
        csv = f"{SR_ROOT}/runs_raw/summary.csv",
    params:
        py       = SR_PY,
        arm_dirs = expand(f"{SR_ROOT}/runs_raw/{{arm}}", arm=SR_ARMS),
    resources:
        mem_mb  = 8000,
        runtime = 15,
    log:
        "logs/samap_raw/compare.log",
    shell:
        r"""
        module load gcc/14.2.0-fasrc01
        {params.py} {input.script} {params.arm_dirs} > {log} 2>&1
        """

# -----------------------------------------------------------------------------
rule samap_raw_all:
    input:
        expand(f"{SR_ROOT}/runs_raw/{{arm}}/samap_transferred.h5ad", arm=SR_ARMS),
        expand(f"{SR_ROOT}/runs_raw/{{arm}}/samap_transferred_pt.h5ad", arm=SR_PT_ARMS),
        f"{SR_ROOT}/runs_raw/summary.csv",
