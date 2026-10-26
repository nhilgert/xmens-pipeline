"""Export a slice of the Leiden ladder as the CSV that clustree() reads.

Called by rule `clustree_export` (98_clustree.smk), or from the command line with
--in-h5ad/--out/--start/--stop/--step. Reads the obs of a reintegrated lineage
object (or the whole-tissue latent) and writes a CSV with one `leiden_scVI_<res>`
column per resolution plus sample, cell_barcode and timepoint where present.
No judgment is applied: the output is a column subset.
"""
import sys
from pathlib import Path

import pandas as pd


def export(in_h5ad, out_csv, resolutions, extra_cols=("sample", "cell_barcode", "timepoint")):
    import scanpy as sc

    # Backed read: only obs is needed, and the lineage objects carry full X.
    adata = sc.read_h5ad(in_h5ad, backed="r")

    cols = [f"leiden_scVI_{r}" for r in resolutions]
    missing = [c for c in cols if c not in adata.obs]
    if missing:
        have = sorted([c for c in adata.obs if c.startswith("leiden_scVI_")],
                      key=lambda c: float(c.rsplit("_", 1)[1]))
        raise KeyError(f"missing ladder columns {missing}. Object has: {have}")

    # Labels go out as strings so clustree treats them as categories, not numbers.
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
    """Resolutions start, start + step, ... up to stop, formatted as column suffixes."""
    out, x, i = [], start, 0

    # Each value is start + i * step rather than a running sum, so float error does not
    # accumulate; the 1e-9 tolerance keeps stop itself in the ladder.
    while x <= stop + 1e-9:
        out.append(_fmt(x))
        i += 1
        x = start + i * step
    return out


if "snakemake" in globals():
    sm = globals()["snakemake"]  # noqa: F821
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
