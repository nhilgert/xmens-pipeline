"""Placeholder for the unimplemented `cellrank` rule (08_cellrank.smk).

Reads nothing and writes nothing: it exits with an error that names the rule and its
declared outputs, so Snakemake never records empty result files for this stage. No
CellRank step is implemented in this workflow.
"""
import sys

snk = snakemake  # noqa: F821  (injected by Snakemake)

sys.exit(
    f"STUB RULE '{snk.rule}' is not implemented; refusing to fabricate "
    f"outputs: {[str(f) for f in snk.output]}"
)
