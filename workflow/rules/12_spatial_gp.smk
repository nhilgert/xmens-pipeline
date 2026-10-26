# Spatial GP fit of pseudotime + gradient field (single species, native pt;
# the cross-species transfer arm is deferred, see the spatial_pt_gp.py docstring).
#
# Reads one rooted-pseudotime h5ad per run and writes a fit summary and the
# gradient field under results/spatial_gp/<run>/. Target: spatial_gp_all.
#
# config.yaml:
#   spatial_gp:
#     python: <rsc_uv python>          # torch + gpytorch (+ geosketch)
#     runs:
#       mouse_palantir_rtk_1dpi_late:
#         h5ad: pseudotime_mouse_v2_induced_no_apop/2026_08_28_mouse_fibdec_pseudotime_rooted.h5ad
#         pt_key: palantir_pt_rtk
#         timepoint: 1_dpi_late

import re

SGP_PY   = config["spatial_gp"]["python"]
SGP_RUNS = config["spatial_gp"]["runs"]   # run_name -> {h5ad, pt_key, timepoint}

wildcard_constraints:
    sgp_run = "|".join(re.escape(r) for r in SGP_RUNS),


# GP fit and gradient field for one run, optionally restricted to one timepoint.
rule spatial_gp:
    input:
        h5ad   = lambda wc: SGP_RUNS[wc.sgp_run]["h5ad"],
        script = "workflow/scripts/pseudotime/spatial_pt_gp.py",
    output:
        summary = "results/spatial_gp/{sgp_run}/gp_fit_summary.json",
        field   = "results/spatial_gp/{sgp_run}/gp_field.npz",
    params:
        py     = SGP_PY,
        pt_key = lambda wc: SGP_RUNS[wc.sgp_run]["pt_key"],
        tp     = lambda wc: SGP_RUNS[wc.sgp_run].get("timepoint"),
        tp_arg = lambda wc: (f"--timepoint {SGP_RUNS[wc.sgp_run]['timepoint']}"
                             if SGP_RUNS[wc.sgp_run].get("timepoint") else ""),
        outdir = lambda wc: f"results/spatial_gp/{wc.sgp_run}",
    log:
        "logs/spatial_gp/{sgp_run}.log",
    threads: 4
    resources:
        mem_mb = 32000, runtime = 120,     # MB; minutes. CPU exact GP at 5k sketch
    shell:
        r"""
        {params.py} {input.script} \
            --h5ad {input.h5ad} --pt-key {params.pt_key} {params.tp_arg} \
            --outdir {params.outdir} > {log} 2>&1
        """


# Target: every configured run.
rule spatial_gp_all:
    input:
        expand("results/spatial_gp/{r}/gp_fit_summary.json", r=list(SGP_RUNS)),
