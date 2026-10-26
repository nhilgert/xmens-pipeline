"""Stub for rule `scenic` (workflow/rules/21_scenic.smk).

Reads nothing and writes nothing. This scaffold stage was never implemented and the
real SCENIC analyses live in 13_scenic.smk. The stub exits with an error that names
the rule's declared outputs instead of touching empty files, which downstream rules
could otherwise mistake for results.
"""
import sys

snk = snakemake  # noqa: F821  (injected by Snakemake)
sys.exit(
    f"STUB RULE '{snk.rule}' is not implemented; refusing to fabricate "
    f"outputs: {[str(f) for f in snk.output]}"
)
