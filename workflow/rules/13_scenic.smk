# workflow/rules/13_scenic.smk
# =============================================================================
# Multi-seed SCENIC: loom build -> GRNBoost2 (seed sweep) -> cisTarget -> consensus AUCell.
#
#   deliverable h5ad --(scenic_pooled)--> {sp}_dsc_pooled.h5ad + loom + manifest  (Stage 0a)
#     --(scenic_sketch x SEEDS, if sketch != full)--> looms_{tag}/..._seed{S}.loom (Stage 0b)
#     --(scenic_grn x SEEDS)--> adjacencies_seed{S}.tsv                            (Stage 1, stochastic)
#       --(scenic_ctx x SEEDS)--> regulons_seed{S}.pkl                             (Stage 2, deterministic)
#         --(scenic_consensus)--> consensus_{tag}_{mode}/  (regulons + AUCell)     (Stage 3, deterministic)
#
# Stages 1-3 wrap the existing scripts under {rev}/scripts/ (grn_only.py / ctx_only.py /
# build_consensus_aucell.py) UNCHANGED -- they are idempotent + atomic, which composes
# cleanly with snakemake retries. Stage 0 is workflow/scripts/scenic/scenic_make_loom.py, a script
# port of {rev}/scripts/prep_fib_inputs.ipynb (2026-09-07) so the loom is no longer an
# ancient() hand-made input.
#
# 2026-09-07 changes:
#   * Stage 0 rule added; source h5ad per species comes from config (scenic.source_h5ad).
#   * consensus paths use SCENIC_TAG ("full" | "sketch15k"), mirroring lane_tag() in
#     build_consensus_aucell.py. The previous f"consensus_{SCENIC_SKETCH}_..." produced
#     consensus_15k_... which the script never writes (MissingOutputException).
#   * optional --timepoint-col forwarded to build_consensus_aucell.py (scenic.timepoint_col).
#
# config.yaml:
#   scenic:
#     python:     <pyscenic env python>            # NOT rsc_uv (anndata 0.11 / numpy 1.26)
#     rev:        <lane root>                      # holds scripts/, databases/, scenic_seeds/
#     n_seeds:    100
#     base_seed:  1000
#     sketch:     "full"                           # 5k | 15k | full
#     mode:       "weighted_soft"                  # or hard_filter
#     species:    ["mouse", "human"]
#     source_h5ad: {mouse: <path>, human: <path>}  # fib/dec deliverable objects
#     raw_layer:  counts                           # layer holding raw counts
#     timepoint_col: timepoint                     # obs column for auc_by_timepoint.tsv (optional)
#     sketch_per_seed: true                        # new geosketch per GRN seed (default true)
#     sketch_seed: 0                               # shared-sketch seed when sketch_per_seed is false
# =============================================================================
import re
SCENIC_PY    = config["scenic"]["python"]
SCENIC_REV   = config["scenic"]["rev"]
SCENIC_SEEDS = [config["scenic"].get("base_seed", 1000) + i
                for i in range(config["scenic"].get("n_seeds", 25))]
SCENIC_SKETCH  = str(config["scenic"].get("sketch", "15k"))
SCENIC_MODE    = config["scenic"].get("mode", "weighted_soft")
SCENIC_SPECIES = config["scenic"].get("species", ["mouse", "human"])
SCENIC_SRC     = config["scenic"]["source_h5ad"]          # {species: h5ad path}
SCENIC_RAW     = config["scenic"].get("raw_layer", "counts")
SCENIC_TP_COL  = config["scenic"].get("timepoint_col", None)
SCENIC_CT_COL  = config["scenic"].get("celltype_col", None)   # optional, loom-log composition table
# lane tag exactly as build_consensus_aucell.py::lane_tag()
SCENIC_TAG     = "full" if SCENIC_SKETCH == "full" else f"sketch{SCENIC_SKETCH}"
# geosketch size for the loom rule (0 = full object)
if SCENIC_SKETCH != "full" and not re.fullmatch(r"[0-9]+(\.[0-9]+)?k", SCENIC_SKETCH):
    raise ValueError(f"scenic.sketch must be 'full' or '<N>k' (e.g. 15k), got {SCENIC_SKETCH!r}")
SCENIC_SKETCH_N = 0 if SCENIC_SKETCH == "full" else int(float(SCENIC_SKETCH.rstrip("k")) * 1000)
# GRNBoost2 walltime scales with cells: 15k sketch ~7 h/seed, full (43-48k cells) 9-24 h/seed
SCENIC_GRN_RUNTIME = 4320 if SCENIC_TAG == "full" else 720

SCENIC_PER_SEED = bool(config["scenic"].get("sketch_per_seed", True)) and SCENIC_TAG != "full"
SCENIC_SKETCH_SEED = int(config["scenic"].get("sketch_seed", 0))   # only used when sketch_per_seed is false

def _sc_root(sp):   return f"{SCENIC_REV}/scenic_seeds/{sp}"
def _sc_pooled_loom(sp):     return f"{_sc_root(sp)}/{sp}_dsc_pooled.loom"
def _sc_pooled_h5ad(sp):     return f"{_sc_root(sp)}/{sp}_dsc_pooled.h5ad"
def _sc_pooled_manifest(sp): return f"{_sc_root(sp)}/{sp}_dsc_pooled.manifest.json"
def _sc_loom(sp, seed=None):
    """Loom a given GRN seed trains on: pooled (full lane), one shared sketch, or a per-seed sketch."""
    if SCENIC_TAG == "full":
        return _sc_pooled_loom(sp)
    if SCENIC_PER_SEED:
        return f"{_sc_root(sp)}/looms_{SCENIC_TAG}/{sp}_dsc_{SCENIC_TAG}_seed{seed}.loom"
    return f"{_sc_root(sp)}/{sp}_dsc_{SCENIC_TAG}.loom"
def _sc_sketch_manifest(sp, seed=None):
    return _sc_loom(sp, seed).replace(".loom", ".manifest.json")
def _sc_adj_dir(sp): return f"{_sc_root(sp)}/adjacencies_{SCENIC_TAG}"
def _sc_reg_dir(sp): return f"{_sc_root(sp)}/regulons_{SCENIC_TAG}"
def _sc_cons_dir(sp): return f"{_sc_root(sp)}/consensus_{SCENIC_TAG}_{SCENIC_MODE}"
def _sc_tfs(sp):
    return f"{SCENIC_REV}/databases/" + ("allTFs_mm.txt" if sp == "mouse" else "allTFs_hg38.txt")

wildcard_constraints:
    sc_species = "|".join(SCENIC_SPECIES),
    sc_seed    = "|".join(str(s) for s in SCENIC_SEEDS),

# ---- Stage 0a: deliverable h5ad -> pooled h5ad (AUCell) + pooled loom + manifest ----
# Runs once per species. The pooled loom is the GRN input only when sketch == full.
rule scenic_pooled:
    input:
        h5ad   = lambda wc: SCENIC_SRC[wc.sc_species],
        tfs    = lambda wc: ancient(_sc_tfs(wc.sc_species)),
        script = "workflow/scripts/scenic/scenic_make_loom.py",
    output:
        loom     = _sc_pooled_loom("{sc_species}"),
        h5ad     = _sc_pooled_h5ad("{sc_species}"),
        manifest = _sc_pooled_manifest("{sc_species}"),
    params:
        py = SCENIC_PY, raw = SCENIC_RAW,
        out_dir = lambda wc: _sc_root(wc.sc_species),
        ct = (f"--celltype-col {SCENIC_CT_COL}" if SCENIC_CT_COL else ""),
    threads: 4
    resources:
        mem_mb = 64000, runtime = 120,
    log:
        "logs/scenic/pooled_{sc_species}.log",
    shell:
        r"""
        {params.py} {input.script} \
            --h5ad {input.h5ad} --species {wildcards.sc_species} \
            --tf-list {input.tfs} --out-dir {params.out_dir} \
            --raw-layer {params.raw} --sketch-n 0 {params.ct} \
            --out-loom {output.loom} --out-h5ad {output.h5ad} --out-manifest {output.manifest} \
            > {log} 2>&1
        """

# ---- Stage 0b: geosketch of the pooled object -> GRN loom ---------------------
# sketch_per_seed: true  -> one sketch per GRN seed (sketch_seed = GRN seed), so the seed
#                           ensemble bootstraps over cells as well as over boosting randomness.
# sketch_per_seed: false -> one shared sketch (sketch_seed from config), as in the June 2026 lane.
if SCENIC_TAG != "full":
    rule scenic_sketch:
        input:
            h5ad   = _sc_pooled_h5ad("{sc_species}"),
            tfs    = lambda wc: ancient(_sc_tfs(wc.sc_species)),
            script = "workflow/scripts/scenic/scenic_make_loom.py",
        output:
            loom     = _sc_loom("{sc_species}", "{sc_seed}"),
            manifest = _sc_sketch_manifest("{sc_species}", "{sc_seed}"),
        params:
            py = SCENIC_PY, raw = "raw", sketch_n = SCENIC_SKETCH_N,
            sketch_seed = lambda wc: getattr(wc, "sc_seed", SCENIC_SKETCH_SEED),
            out_dir = lambda wc: _sc_root(wc.sc_species),
            ct = (f"--celltype-col {SCENIC_CT_COL}" if SCENIC_CT_COL else ""),
        threads: 2
        resources:
            mem_mb = 32000, runtime = 45,
        log:
            (f"logs/scenic/sketch_{{sc_species}}_seed{{sc_seed}}.log" if SCENIC_PER_SEED
             else "logs/scenic/sketch_{sc_species}.log"),
        shell:
            r"""
            {params.py} {input.script} \
                --h5ad {input.h5ad} --species {wildcards.sc_species} \
                --tf-list {input.tfs} --out-dir {params.out_dir} --skip-h5ad \
                --raw-layer {params.raw} --sketch-n {params.sketch_n} --sketch-seed {params.sketch_seed} {params.ct} \
                --out-loom {output.loom} --out-manifest {output.manifest} \
                > {log} 2>&1
            """

# ---- Stage 1: seeded GRNBoost2 (the only stochastic stage) -------------------
rule scenic_grn:
    input:
        loom   = lambda wc: _sc_loom(wc.sc_species, wc.sc_seed),
        tfs    = lambda wc: ancient(_sc_tfs(wc.sc_species)),
        script = f"{SCENIC_REV}/scripts/grn_only.py",
    output:
        adj = _sc_adj_dir("{sc_species}") + "/adjacencies_seed{sc_seed}.tsv",
    params:
        py = SCENIC_PY,
    threads: 16
    resources:
        mem_mb = 64000, runtime = SCENIC_GRN_RUNTIME,
    log:
        "logs/scenic/grn_{sc_species}_seed{sc_seed}.log",
    shell:
        r"""
        {params.py} {input.script} \
            --loom {input.loom} --tf-list {input.tfs} \
            --out {output.adj} --seed {wildcards.sc_seed} \
            --n-cpu {threads} > {log} 2>&1
        """

# ---- Stage 2: cisTarget (deterministic; memory-heavy) ------------------------
rule scenic_ctx:
    input:
        adj    = _sc_adj_dir("{sc_species}") + "/adjacencies_seed{sc_seed}.tsv",
        loom   = lambda wc: _sc_loom(wc.sc_species, wc.sc_seed),
        script = f"{SCENIC_REV}/scripts/ctx_only.py",
    output:
        pkl = _sc_reg_dir("{sc_species}") + "/regulons_seed{sc_seed}.pkl",
    params:
        py     = SCENIC_PY,
        db_dir = f"{SCENIC_REV}/databases",
    threads: 8
    resources:
        mem_mb = 96000, runtime = 480,
    log:
        "logs/scenic/ctx_{sc_species}_seed{sc_seed}.log",
    shell:
        r"""
        {params.py} {input.script} \
            --adjacencies {input.adj} --loom {input.loom} \
            --db-dir {params.db_dir} --species {wildcards.sc_species} \
            --out {output.pkl} --n-cpu {threads} > {log} 2>&1
        """

# ---- Stage 3: consensus + AUCell (gathers all seeds) --------------------------
rule scenic_consensus:
    input:
        pkls = lambda wc: [_sc_reg_dir(wc.sc_species) + f"/regulons_seed{s}.pkl"
                           for s in SCENIC_SEEDS],
        h5ad   = _sc_pooled_h5ad("{sc_species}"),
        script = f"{SCENIC_REV}/scripts/build_consensus_aucell.py",
    output:
        # build_consensus_aucell.py writes consensus_{lane_tag}_{mode}/ under {rev}/scenic_seeds/{sp};
        # deliverables are the AUCell matrix and the pickled consensus regulons
        auc  = _sc_cons_dir("{sc_species}") + "/auc_mtx.csv",
        pkl  = _sc_cons_dir("{sc_species}") + "/consensus_regulons.pkl",
    params:
        py = SCENIC_PY, sketch = SCENIC_SKETCH, mode = SCENIC_MODE, rev = SCENIC_REV,
        tp = (f"--timepoint-col {SCENIC_TP_COL}" if SCENIC_TP_COL else ""),
    threads: 8
    resources:
        mem_mb = 128000, runtime = 240,
    log:
        "logs/scenic/consensus_{sc_species}.log",
    shell:
        r"""
        {params.py} {input.script} \
            --species {wildcards.sc_species} --sketch {params.sketch} \
            --mode {params.mode} --rev {params.rev} --h5ad {input.h5ad} {params.tp} \
            --n-cpu {threads} > {log} 2>&1
        """

rule scenic_all:
    input:
        expand(_sc_cons_dir("{sp}") + "/auc_mtx.csv", sp=SCENIC_SPECIES),
