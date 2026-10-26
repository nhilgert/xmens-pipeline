"""Stub for rule invitro in workflow/rules/23_invitro.smk.

Reads nothing and writes nothing: it exits with an error. This scaffold stage
was never implemented, and no in-vitro comparison exists elsewhere in this
workflow. Touching the declared outputs would
fabricate empty result files, so the rule fails loudly instead.
"""
import sys

snk = snakemake  # noqa: F821  (injected by Snakemake)
sys.exit(
    f"STUB RULE '{snk.rule}' is not implemented; refusing to fabricate "
    f"outputs: {[str(f) for f in snk.output]}"
)
