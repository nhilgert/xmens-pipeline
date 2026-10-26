# workflow/rules/11_mi_pseudotime.smk
# =============================================================================
# Mutual information between gene expression and pseudotime, per (species, pt-key).
#
#   rooted pseudotime h5ad --(mi_prep)--> mi/<run>/{expr.npz, pseudotime.npy, genes.npy, manifest.json}
#     --(mi_worker[array 0..N-1])--> mi/<run>/results/mi_###.csv
#       --(mi_merge)--> mi/<run>/mi_all.csv   (gene, mi, p, q)
#
# Driven by mi_pseudotime.py (prep/worker/merge/diff). One "run" = one h5ad crossed
# with one pseudotime obs column, so the same object can be scored against DPT,
# Palantir-on-RTK-basis, and Palantir-standard as three independent runs.
#
# config.yaml:
#   mi_pseudotime:
#     python: /n/lab_storage/eddy_lab/users/nhilgert/.uv/envs/rsc_uv/bin/python
#     n_chunks: 200
#     perms: 1001
#     min_counts: 10
#     layer: log1p
#     runs:
#       mouse_palantir_rtk:
#         h5ad: results/.../mouse_fibdec_pseudotime_rooted.h5ad
#         pt_key: palantir_pt_rtk
#       mouse_palantir_std:
#         h5ad: results/.../mouse_fibdec_pseudotime_rooted.h5ad
#         pt_key: palantir_pt_std
#       human_palantir_rtk:
#         h5ad: results/.../human_fibdec_pseudotime_rooted.h5ad
#         pt_key: palantir_pt_rtk
# Snakefile:  include: "rules/11_mi_pseudotime.smk"
# =============================================================================
import re

MI_PY       = config["mi_pseudotime"]["python"]
MI_CHUNKS   = int(config["mi_pseudotime"].get("n_chunks", 200))
MI_PERMS    = int(config["mi_pseudotime"].get("perms", 1001))
MI_MINCOUNT = int(config["mi_pseudotime"].get("min_counts", 10))
MI_LAYER    = config["mi_pseudotime"].get("layer", "log1p")
MI_RUNS     = config["mi_pseudotime"]["runs"]          # run_name -> {h5ad, pt_key}
MI_ROOT     = "results/mi"

def _mi_h5ad(run):   return MI_RUNS[run]["h5ad"]
def _mi_ptkey(run):  return MI_RUNS[run]["pt_key"]

wildcard_constraints:
    mi_run = "|".join(re.escape(r) for r in MI_RUNS),

# ---- prep: matrices + manifest (one job per run) ----------------------------
rule mi_prep:
    input:
        h5ad   = lambda wc: _mi_h5ad(wc.mi_run),
        script = "workflow/scripts/mi/mi_pseudotime.py",
    output:
        expr     = f"{MI_ROOT}/{{mi_run}}/expr.npz",
        pt       = f"{MI_ROOT}/{{mi_run}}/pseudotime.npy",
        genes    = f"{MI_ROOT}/{{mi_run}}/genes.npy",
        manifest = f"{MI_ROOT}/{{mi_run}}/manifest.json",
    params:
        py      = MI_PY,
        pt_key  = lambda wc: _mi_ptkey(wc.mi_run),
        outdir  = lambda wc: f"{MI_ROOT}/{wc.mi_run}",
        chunks  = MI_CHUNKS, min_counts = MI_MINCOUNT, layer = MI_LAYER,
    resources:
        mem_mb = 48000, runtime = 60,
    log:
        "logs/mi/{mi_run}_prep.log",
    shell:
        r"""
        {params.py} {input.script} prep \
            --h5ad {input.h5ad} --pt-key {params.pt_key} \
            --layer {params.layer} --min-counts {params.min_counts} \
            --chunks {params.chunks} --outdir {params.outdir} > {log} 2>&1
        """

# ---- worker: one chunk (checkpoint fans out the array over n_chunks) ---------
# n_chunks is fixed in config, so we can expand the chunk results directly.
rule mi_worker:
    input:
        manifest = f"{MI_ROOT}/{{mi_run}}/manifest.json",
        script   = "workflow/scripts/mi/mi_pseudotime.py",
    output:
        csv = f"{MI_ROOT}/{{mi_run}}/results/mi_{{chunk}}.csv",
    params:
        py     = MI_PY,
        outdir = lambda wc: f"{MI_ROOT}/{wc.mi_run}",
        chunk  = lambda wc: int(wc.chunk),
        perms  = MI_PERMS,
    resources:
        mem_mb = 16000, runtime = 240,     # MI + 1000 perms per gene is CPU-heavy
    threads: 1
    log:
        "logs/mi/{mi_run}_worker_{chunk}.log",
    shell:
        r"""
        {params.py} {input.script} worker \
            --outdir {params.outdir} --chunk {params.chunk} \
            --perms {params.perms} > {log} 2>&1
        """

def _mi_chunk_csvs(wc):
    return [f"{MI_ROOT}/{wc.mi_run}/results/mi_{c:03d}.csv" for c in range(MI_CHUNKS)]

# ---- merge: concat + BH FDR -------------------------------------------------
rule mi_merge:
    input:
        chunks = _mi_chunk_csvs,
        script = "workflow/scripts/mi/mi_pseudotime.py",
    output:
        mi_all = f"{MI_ROOT}/{{mi_run}}/mi_all.csv",
    params:
        py     = MI_PY,
        outdir = lambda wc: f"{MI_ROOT}/{wc.mi_run}",
    resources:
        mem_mb = 16000, runtime = 30,
    log:
        "logs/mi/{mi_run}_merge.log",
    shell:
        r"""
        {params.py} {input.script} merge --outdir {params.outdir} > {log} 2>&1
        """

# ---- target -----------------------------------------------------------------
rule mi_all:
    input:
        expand(f"{MI_ROOT}/{{mi_run}}/mi_all.csv", mi_run=list(MI_RUNS)),
