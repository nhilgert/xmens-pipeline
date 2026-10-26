"""Stub for rule crossspecies in workflow/rules/22_crossspecies.smk.

Reads nothing and writes nothing: it exits with an error. This scaffold stage
was never implemented. The cross-species analyses that exist are the SAMap arms
(10_samap.smk, 10b_samap_raw.smk) and the standalone crossspecies/fig3d_conservation.py.
Touching the declared outputs would
fabricate empty result files, so the rule fails loudly instead.
"""
import sys

snk = snakemake  # noqa: F821  (injected by Snakemake)
sys.exit(
    f"STUB RULE '{snk.rule}' is not implemented; refusing to fabricate "
    f"outputs: {[str(f) for f in snk.output]}"
)
