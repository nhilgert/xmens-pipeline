"""Stub for rule figures in workflow/rules/90_figures.smk.

Reads nothing and writes nothing: it exits with an error. This scaffold stage
was never implemented. Figure rules live in the stage files that produce them
(e.g. zone_figures in 16_zone_modules.smk, clustree_plot in 98_clustree.smk).
Touching the declared outputs would
fabricate empty result files, so the rule fails loudly instead.
"""
import sys

snk = snakemake  # noqa: F821  (injected by Snakemake)
sys.exit(
    f"STUB RULE '{snk.rule}' is not implemented; refusing to fabricate "
    f"outputs: {[str(f) for f in snk.output]}"
)
