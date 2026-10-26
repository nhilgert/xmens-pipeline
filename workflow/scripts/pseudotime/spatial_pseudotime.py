"""Placeholder for the unimplemented `spatial_pseudotime` rule (07_spatial_pseudotime.smk).

Reads nothing and writes nothing: it exits with an error that names the rule and its
declared outputs, so Snakemake never records empty result files for this stage. The
implemented analyses live in the numbered arms (10_samap through 15_pt_per_timepoint).
"""
import sys

snk = snakemake  # noqa: F821  (injected by Snakemake)

sys.exit(
    f"STUB RULE '{snk.rule}' is not implemented; refusing to fabricate "
    f"outputs: {[str(f) for f in snk.output]}"
)
