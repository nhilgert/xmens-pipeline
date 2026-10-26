"""STUB — intentionally disabled (2026-09-01 cleanup).

This scaffold stage was never implemented; the real analyses live in the
numbered arms (10_samap ... 15_pt_per_timepoint). This stub previously
`touch`ed its declared outputs, which could fabricate empty result files
(e.g. results/tables/mi_*.csv). It now fails loudly instead.
"""
import sys

snk = snakemake  # noqa: F821  (injected by Snakemake)
sys.exit(
    f"STUB RULE '{snk.rule}' is not implemented; refusing to fabricate "
    f"outputs: {[str(f) for f in snk.output]}"
)
