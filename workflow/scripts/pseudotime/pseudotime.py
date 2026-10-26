"""Placeholder for the unimplemented `pseudotime_global` rule (06_pseudotime.smk).

Reads nothing and writes nothing: it exits with an error that names the rule and its
declared outputs, so Snakemake never records empty result files for this stage. The
implemented pseudotime lives in 06b_timepoint_pseudotime.smk, 15_pt_per_timepoint.smk and
the notebook rules of 24_pseudotime_notebooks.smk (not included in the Snakefile).
"""
import sys

snk = snakemake  # noqa: F821  (injected by Snakemake)

sys.exit(
    f"STUB RULE '{snk.rule}' is not implemented; refusing to fabricate "
    f"outputs: {[str(f) for f in snk.output]}"
)
