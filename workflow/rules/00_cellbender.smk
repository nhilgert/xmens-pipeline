# Stage 00 - CellBender ambient-RNA removal (UPSTREAM; not in the default DAG).
#
# Reproduces the ambient-RNA correction that produced the pipeline's inputs.
# NOT in the main DAG (pipeline starts from already-corrected xmens_inputs/).
# Output -> config["cellbender"]["output_dir"] (PARALLEL regen dir, never the
# real inputs). Params match Methods: v0.2.2, --total-droplets-included, default
# epochs (150 for 0.2.x), --cuda. Input = CellRanger raw under velocity/{lane}/;
# lane == coords_id.

rule cellbender:
    input:
        raw=lambda w: config["cellbender"]["cellranger_raw"].format(
            lane=samples.loc[w.sample, "coords_id"]
        ),
    output:
        h5=config["cellbender"]["output_dir"] + "/{sample}/{sample}_cellbender.h5",
    params:
        total_droplets=lambda w: config["cellbender"]["total_droplets_included"],
        env=lambda w: config["cellbender"]["env"],
    log:
        "logs/00_cellbender_{sample}.log",
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.h5})"

        # Activate the cellbender_022 conda env inside a bare SLURM job shell.
        # The job shell does NOT inherit the orchestrator's PATH, so put the
        # conda binary on PATH FIRST, then source conda.sh, then activate. The
        # PATH export is the load-bearing bit (conda is otherwise 'command not
        # found' in the non-interactive job shell).
        export PATH="/n/sw/Miniforge3-25.3.1-0/bin:$PATH"
        source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
        conda activate {params.env}

        cellbender remove-background \
            --input {input.raw} \
            --output {output.h5} \
            --total-droplets-included {params.total_droplets} \
            --cuda \
            > {log} 2>&1
        """
