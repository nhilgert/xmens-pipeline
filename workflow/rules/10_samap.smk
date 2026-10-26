# SAMap transfer arm: cross-species label transfer between one human phase and one
# mouse timepoint. Complementary to 10b_samap_raw.smk (the raw-counts arm); both are live.
#
#   fib/dec deliverables --(prep_samap_input)--> inputs/h5ads_samap/*.h5ad
#     --(run_samap)--> runs/<run>/samap.pkl --(samap_label_stain)--> results/.../*.png
#
# Env is stock sc-samap v3. Everything hangs off SAMAP_ROOT; rules run from the
# xmens_final root. Target: samap_all.
#
# Inputs are the fib/dec deliverables cut by 04g: they carry pseudotime, the final
# labels and the fib/dec subset, so prep_samap_input subsets one timepoint, renames
# columns to the shared schema and validates the scale.
#
# Input scale is CP10K (library-size normalized to 1e4, not logged). This retains
# 99.7% of nonzeros under SAMap's min_expression=1 threshold, vs 71% for raw counts.
# The guard requires rowsums == 1e4 and .X max > 20, which blocks any pre-logged matrix.
#
# run_samap's fit_script is ancient(), so editing it never refits; to refit, delete
# runs/<run>/samap.pkl. Pickles written by older samap versions cannot be opened by
# anndata 0.12 and must be deleted by hand.
#
# config[samap][python] is the stock-v3 venv interpreter (absolute path).
# run_samap.py, prep_samap_input.py and transfer_label_staining.py live in workflow/scripts/samap/.
#
# config.yaml:
#   samap:
#     root:        <samap_integrate project root: inputs/, runs/, homology/, slurm/>
#     coords_root: <crossspecies_radial_gp root: isodepth and radius npz files>
#     python:      <stock sc-samap v3 python>
#     n_iter:      3
#     pairs: ["menstrual_x_4dpi_hmmer", "secretory_x_1dpi_late_hmmer"]
# Snakefile:  include: "rules/10_samap.smk"

import csv, re

SAMAP_ROOT  = config["samap"]["root"]
COORDS_ROOT = config["samap"]["coords_root"]
SAMAP_PY    = config["samap"]["python"]
N_ITER      = config["samap"].get("n_iter", 3)
SAMAP_PAIRS = config["samap"]["pairs"]
# run_name -> mouse per-timepoint pseudotime h5ad (optional)
PT_H5AD     = config["samap"].get("pt_h5ad", {})

# manifest = source of truth for run_name -> {hs_h5ad, mm_h5ad, homology}
_manifest = {}
with open(f"{SAMAP_ROOT}/slurm/pairs.csv", newline="") as fh:
    for row in csv.DictReader(fh):
        _manifest[row["run_name"].strip()] = {k: (v.strip() if isinstance(v, str) else v)
                                              for k, v in row.items()}
_missing = [r for r in SAMAP_PAIRS if r not in _manifest]
assert not _missing, f"run_name(s) not in pairs.csv: {_missing}"
assert all(_manifest[r]["homology"] == "hmmer" for r in SAMAP_PAIRS), \
    "paper deliverables are scoped to hmmer; a requested pair is blast"
PAIRS = {r: _manifest[r] for r in SAMAP_PAIRS}

# Per-species source object (the 2026_09_15 release, config vocab_migration.from_release),
# its coarse/fine annotation columns and the expected fib/dec cell count, which
# prep_samap_input compares and warns on (it does not stop).
_REFINED = {
    "hs": dict(path="data/deliverables/2026_09_15/Cevrim_Human_Only_Fib_Dec_Cells.h5ad",
                coarse="cell_type", fine="cell_subtype", expect=47533),
    "mm": dict(path="data/deliverables/2026_09_15/Cevrim_XMens_Only_Fib_Dec_Cells.h5ad",
                coarse="cell_type", fine="cell_subtype", expect=43197),
}

# SAMap input basenames the pairings need, e.g. hs_fib_dec_secretory, mm_fib_dec_4_dpi_bleeding
_NEEDED_INPUTS = set()
for _r in PAIRS:
    _NEEDED_INPUTS |= {PAIRS[_r]["hs_h5ad"], PAIRS[_r]["mm_h5ad"]}

def _parse_input_basename(bn):
    # "<sp>_fib_dec_<timepoint>" -> (sp, timepoint); the slug comes from pairs.csv, not run_name
    for sp in ("hs", "mm"):
        pre = f"{sp}_fib_dec_"
        if bn.startswith(pre):
            return sp, bn[len(pre):]
    raise ValueError(f"unexpected samap input basename: {bn}")

# "<phase>_x_<stage>_hmmer" -> (human phase, mouse stage)
def _phase_stage(run):
    assert run.endswith("_hmmer"), run
    phase, _, stage = run[:-len("_hmmer")].partition("_x_")
    return phase, stage

def _hs_coords(run): return f"{COORDS_ROOT}/isodepth_{_phase_stage(run)[0]}.npz"
def _mm_coords(run): return f"{COORDS_ROOT}/transfer_all/radius_{_phase_stage(run)[1]}.npz"
def _maps_dir(run):  return f"{SAMAP_ROOT}/homology/hsmm_{PAIRS[run]['homology']}"
def _hs_input(run):  return f"{SAMAP_ROOT}/inputs/h5ads_samap/{PAIRS[run]['hs_h5ad']}.h5ad"
def _mm_input(run):  return f"{SAMAP_ROOT}/inputs/h5ads_samap/{PAIRS[run]['mm_h5ad']}.h5ad"

wildcard_constraints:
    run      = "|".join(re.escape(r) for r in PAIRS),
    basename = "|".join(re.escape(b) for b in _NEEDED_INPUTS),

# Bridge: deliverable -> one per-timepoint SAMap input. Subset to the timepoint, rename
# columns to the shared schema, validate the CP10K scale.
rule prep_samap_input:
    input:
        refined = lambda wc: _REFINED[_parse_input_basename(wc.basename)[0]]["path"],
        script  = "workflow/scripts/samap/prep_samap_input.py",
    output:
        h5ad = f"{SAMAP_ROOT}/inputs/h5ads_samap/{{basename}}.h5ad",
    params:
        py      = SAMAP_PY,
        species = lambda wc: _parse_input_basename(wc.basename)[0],
        tp      = lambda wc: _parse_input_basename(wc.basename)[1],
        coarse  = lambda wc: _REFINED[_parse_input_basename(wc.basename)[0]]["coarse"],
        fine    = lambda wc: _REFINED[_parse_input_basename(wc.basename)[0]]["fine"],
        expect  = lambda wc: _REFINED[_parse_input_basename(wc.basename)[0]]["expect"],
    log:
        "logs/prep_samap_input/{basename}.log",
    shell:
        r"""
        module load gcc/14.2.0-fasrc01
        {params.py} {input.script} \
            --refined       {input.refined} \
            --species       {params.species} \
            --coarse-col    {params.coarse} \
            --fine-col      {params.fine} \
            --timepoint     {params.tp} \
            --expect-fibdec {params.expect} \
            --out           {output.h5ad} \
            > {log} 2>&1
        """

# Fit: runs/<run>/samap.pkl (stock v3). The hs/mm inputs come from prep_samap_input, so
# SAMap sits downstream of the annotation.
rule run_samap:
    input:
        hs         = lambda wc: _hs_input(wc.run),
        mm         = lambda wc: _mm_input(wc.run),
        map_hm     = lambda wc: ancient(f"{_maps_dir(wc.run)}/hsmm/hs_to_mm.txt"),
        map_mh     = lambda wc: ancient(f"{_maps_dir(wc.run)}/hsmm/mm_to_hs.txt"),
        # ancient: the fit is expensive, so a script edit never refits
        fit_script = ancient("workflow/scripts/samap/run_samap.py"),
    output:
        # meta.json and the umap are written best-effort, so they are undeclared
        pkl = f"{SAMAP_ROOT}/runs/{{run}}/samap.pkl",
    params:
        py       = SAMAP_PY,
        maps_dir = lambda wc: _maps_dir(wc.run),
        out_dir  = lambda wc: f"{SAMAP_ROOT}/runs/{wc.run}",
        n_iter   = N_ITER,
    log:
        f"{SAMAP_ROOT}/slurm/logs/run_samap_{{run}}.log",
    threads: 8
    resources:
        mem_mb  = 128000,   # MB
        runtime = 360,      # minutes
    shell:
        r"""
        module load gcc/14.2.0-fasrc01
        {params.py} {input.fit_script} \
            --hs-h5ad  {input.hs} \
            --mm-h5ad  {input.mm} \
            --maps-dir {params.maps_dir} \
            --out-dir  {params.out_dir} \
            --n-iter   {params.n_iter} \
            --ncpus    {threads} \
            > {log} 2>&1
        """

# Consumer: coarse dec/fib label-transfer staining on the spatial coordinates.
rule samap_label_stain:
    input:
        pkl       = f"{SAMAP_ROOT}/runs/{{run}}/samap.pkl",
        hs_coords = lambda wc: ancient(_hs_coords(wc.run)),     # external isodepth npz, frozen
        mm_coords = lambda wc: ancient(_mm_coords(wc.run)),
        script    = "workflow/scripts/samap/transfer_label_staining.py",
    output:
        png   = "results/figures/samap_transfer/{run}_labelstain.png",
        stats = "results/tables/samap_transfer/{run}_labelstain.json",
    params:
        py  = SAMAP_PY,     # same stock-v3 env (unpickling the pkl needs samap importable)
        run = "{run}",
    log:
        "logs/samap_label_stain/{run}.log",
    shell:
        # If pickle.load cannot read the pkl, transfer_label_staining.py has to load it
        # with samap's load_samap instead.
        r"""
        module load gcc/14.2.0-fasrc01
        {params.py} {input.script} \
            --run-name  {params.run} \
            --samap-pkl {input.pkl} \
            --hs-coords {input.hs_coords} \
            --mm-coords {input.mm_coords} \
            --out-png   {output.png} \
            --out-stats {output.stats} \
            > {log} 2>&1
        """

# Deliverable object: the joint post-integration object with transferred labels, for a
# collaborator to open and plot (joint UMAP for both species, or per-species spatial).
# Same transfer computation as the stain (make_transferred_object.py imports it);
# pseudotime is one more obs column on this object.
rule samap_transferred_object:
    input:
        pkl       = f"{SAMAP_ROOT}/runs/{{run}}/samap.pkl",
        hs_coords = lambda wc: ancient(_hs_coords(wc.run)),
        mm_coords = lambda wc: ancient(_mm_coords(wc.run)),
        pt        = lambda wc: ancient(PT_H5AD[wc.run]),   # mouse per-timepoint pseudotime
        script    = "workflow/scripts/samap/make_transferred_object.py",
        core      = "workflow/scripts/samap/transfer_label_staining.py",   # imported by the script
    output:
        h5ad = "results/samap_transfer/{run}_transferred.h5ad",
    params:
        py  = SAMAP_PY,     # same stock-v3 env
        run = "{run}",
    log:
        "logs/samap_transferred_object/{run}.log",
    shell:
        r"""
        module load gcc/14.2.0-fasrc01
        {params.py} {input.script} \
            --samap-pkl     {input.pkl} \
            --run-name      {params.run} \
            --hs-coords     {input.hs_coords} \
            --mm-coords     {input.mm_coords} \
            --transfer-fine \
            --pt-h5ad       {input.pt} \
            --out           {output.h5ad} \
            > {log} 2>&1
        """

# Paper target. For every run in config samap.pairs: prep_samap_input per distinct input,
# run_samap and samap_label_stain per run, and a transferred object for the runs in pt_h5ad.
rule samap_all:
    input:
        expand("results/figures/samap_transfer/{run}_labelstain.png", run=list(PAIRS)),
        expand("results/tables/samap_transfer/{run}_labelstain.json", run=list(PAIRS)),
        # Transferred-object deliverables only exist for runs with a pt_h5ad entry (the
        # two paper pairings); expanding over PAIRS would KeyError at DAG time for the
        # sweep pairs that have none.
        expand("results/samap_transfer/{run}_transferred.h5ad", run=list(PT_H5AD)),
