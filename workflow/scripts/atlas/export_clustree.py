#!/usr/bin/env python3
"""Export a Leiden-ladder slice for clustree.

Reads a reintegrated lineage object (or the whole-tissue latent) and writes a CSV
with one column per resolution -- exactly the shape clustree() wants, with the
`leiden_scVI_` prefix it keys on. No judgment here; it's a column subset.

snakemake `script:` directive, or CLI:
    python export_clustree.py --in-h5ad data/interim/latent_mouse_stromal.h5ad \
        --out results/tables/clustree/mouse_stromal.csv --start 0.1 --stop 0.8 --step 0.1
"""
import sys
from pathlib import Path
import pandas as pd


def export(in_h5ad, out_csv, resolutions, extra_cols=("sample", "cell_barcode", "timepoint")):
    import scanpy as sc
    adata = sc.read_h5ad(in_h5ad, backed="r")
    cols = [f"leiden_scVI_{r}" for r in resolutions]
    missing = [c for c in cols if c not in adata.obs]
    if missing:
        have = sorted([c for c in adata.obs if c.startswith("leiden_scVI_")],
                      key=lambda c: float(c.rsplit("_", 1)[1]))
        raise KeyError(f"missing ladder columns {missing}. Object has: {have}")

    df = pd.DataFrame(index=adata.obs_names)
    for c in extra_cols:
        if c in adata.obs:
            df[c] = adata.obs[c].astype(str).values
    for c in cols:
        df[c] = adata.obs[c].astype(str).values

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_clus = {c: df[c].nunique() for c in cols}
    print(f"wrote {out_csv}  ({len(df):,} cells x {len(cols)} resolutions)")
    print("  clusters per resolution:", n_clus)
    return df


def _fmt(x):
    """0.1 not 0.1000000001; 1.0 stays 1.0 to match the obs column names."""
    s = f"{x:.10g}"
    return s if "." in s else f"{s}.0"


def _ladder(start, stop, step):
    out, x, i = [], start, 0
    while x <= stop + 1e-9:
        out.append(_fmt(x)); i += 1; x = start + i * step
    return out


if "snakemake" in globals():
    sm = globals()["snakemake"]                                   # noqa: F821
    export(sm.input.h5ad, sm.output.csv,
           _ladder(sm.params.start, sm.params.stop, sm.params.step))
elif __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-h5ad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.1)
    ap.add_argument("--stop", type=float, default=0.8)
    ap.add_argument("--step", type=float, default=0.1)
    a = ap.parse_args()
    export(a.in_h5ad, a.out, _ladder(a.start, a.stop, a.step))
