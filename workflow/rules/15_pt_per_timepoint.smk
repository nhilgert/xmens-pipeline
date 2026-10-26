# workflow/rules/15_pt_per_timepoint.smk
import re

# =============================================================================
# Per-timepoint pseudotime: subset the fib/dec deliverable to ONE timepoint,
# retrain scVI on those cells, DPT with an explicit root. No moscot/RTK (needs
# >=2 timepoints); orientation gate = decidual median > fibroblast median.
# GPU per job (scVI retrain).
#
# config.yaml:
#   pt_per_timepoint:
#     python: <rsc_uv python>
#     drop_subtypes:
#       mouse: ["decidual_apoptotic"]
#       human: []
#     h5ad:
#       mouse: data/deliverables/2026_08_28_Cevrim_XMens_Only_Fib_Dec_Cells.h5ad
#       human: data/deliverables/2026_08_28_Cevrim_Human_Only_Fib_Dec_Cells.h5ad
#     timepoints:
#       mouse: ["uninduced", "1_dpi_early", "1_dpi_late", "2_dpi", "3_dpi", "4_dpi_bleeding"]
#       human: ["secretory", "menstrual"]
# =============================================================================
PTT     = config["pt_per_timepoint"]
PTT_PY  = PTT["python"]
PTT_RUNS = [(sp, tp) for sp in PTT["timepoints"] for tp in PTT["timepoints"][sp]]

wildcard_constraints:
    ptt_species = "|".join(PTT["timepoints"]),
    ptt_tp      = "|".join(sorted({re.escape(tp) for _, tp in PTT_RUNS})),

rule pt_per_timepoint:
    input:
        h5ad   = lambda wc: PTT["h5ad"][wc.ptt_species],
        script = "workflow/scripts/pseudotime/pt_per_timepoint.py",
    output:
        h5   = "results/pt_per_timepoint/{ptt_species}_{ptt_tp}/{ptt_species}_{ptt_tp}_pt.h5ad",
        rep  = "results/pt_per_timepoint/{ptt_species}_{ptt_tp}/{ptt_species}_{ptt_tp}_report.json",
    params:
        py    = PTT_PY,
        drops = lambda wc: " ".join(PTT.get("drop_subtypes", {}).get(wc.ptt_species, [])),
        drop_arg = lambda wc: ("--drop-subtypes " + " ".join(PTT["drop_subtypes"][wc.ptt_species])
                               if PTT.get("drop_subtypes", {}).get(wc.ptt_species) else ""),
        outdir = lambda wc: f"results/pt_per_timepoint/{wc.ptt_species}_{wc.ptt_tp}",
    threads: 8
    resources:
        mem_mb = 96000, runtime = 240,
        slurm_extra = "--gres=gpu:1",     # scVI retrain per timepoint
    log:
        "logs/pt_per_timepoint/{ptt_species}_{ptt_tp}.log",
    shell:
        r"""
        {params.py} {input.script} \
            --h5ad {input.h5ad} --species {wildcards.ptt_species} \
            --timepoint {wildcards.ptt_tp} {params.drop_arg} \
            --outdir {params.outdir} > {log} 2>&1
        """

rule pt_per_timepoint_all:
    input:
        [f"results/pt_per_timepoint/{sp}_{tp}/{sp}_{tp}_report.json" for sp, tp in PTT_RUNS],
