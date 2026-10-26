"""Placeholder for rule `cluster` (03_cluster.smk): exits with an error on purpose.

Reads nothing and writes nothing. Leiden and UMAP are computed by integrate.py (rule
integrate), so this stage was never implemented. Touching the declared output
(data/interim/clustered_{species}.h5ad) would leave an empty file that looks like a
result, so the script fails loudly instead.
"""
import sys

snk = snakemake  # noqa: F821  (injected by Snakemake)

sys.exit(
    f"STUB RULE '{snk.rule}' is not implemented; refusing to fabricate "
    f"outputs: {[str(f) for f in snk.output]}"
)
