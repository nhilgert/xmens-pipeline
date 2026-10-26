"""Placeholder for the unimplemented `de` rule (05_de.smk).

Reads nothing and writes nothing: it exits with an error that names the rule and its
declared outputs, so Snakemake never records empty result files for this stage. The
implemented DE lives in rules de_lineage_clusters and de_lineage_subtypes of the same file.
"""
import sys

snk = snakemake  # noqa: F821  (injected by Snakemake)

sys.exit(
    f"STUB RULE '{snk.rule}' is not implemented; refusing to fabricate "
    f"outputs: {[str(f) for f in snk.output]}"
)
