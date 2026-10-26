#!/usr/bin/env python
"""samap_attach_pt.py — join a transfer_pt CSV into a joint SAMap h5ad as obs columns.

Pipeline port of full_fibdec/attach_pt_0902.py: same columns and provenance, but
parameterized and writing a NEW file (no in-place mutation), so it can be a rule output.

Input CSV (from samap_transfer_pt.py): cell, pt_raw, weight, pt_ranknorm, pt_ranknorm_insample
New obs columns (NaN on mouse cells, matching the transferred_* label columns):
  {prefix}                     <- pt_raw
  {prefix}_ranknorm            <- pt_ranknorm
  {prefix}_ranknorm_insample   <- pt_ranknorm_insample
  {prefix}_weight              <- weight (total cross-species vote mass)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def log(m):
    print(m, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joint", required=True, help="joint samap_transferred.h5ad")
    p.add_argument("--pt-csv", required=True, help="pt_transferred_hs.csv from samap_transfer_pt.py")
    p.add_argument("--prefix", default="transferred_mm_dpt_pseudotime")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    colmap = {
        "pt_raw": args.prefix,
        "pt_ranknorm": f"{args.prefix}_ranknorm",
        "pt_ranknorm_insample": f"{args.prefix}_ranknorm_insample",
        "weight": f"{args.prefix}_weight",
    }

    df = pd.read_csv(args.pt_csv).set_index("cell")
    assert df.index.is_unique, "duplicate cell barcodes in the pt CSV"
    log(f"CSV: {len(df):,} cells, columns {list(df.columns)}")

    log(f"loading {args.joint} ...")
    a = ad.read_h5ad(args.joint)
    log(f"  {a.shape}; {len(a.obs.columns)} obs columns")

    for new in colmap.values():
        assert new not in a.obs.columns, f"{new} already present — refusing to clobber"

    hs_mask = a.obs["species"].astype(str).values == "hs"
    n_hs = int(hs_mask.sum())

    for src, new in colmap.items():
        v = pd.to_numeric(df[src], errors="coerce").reindex(a.obs_names)
        # transfers are defined on human cells only; anything landing on a
        # mouse barcode would mean a barcode collision
        on_mm = int(v.notna()[~hs_mask].sum())
        assert on_mm == 0, f"{on_mm} values joined onto MOUSE cells for {src}"
        a.obs[new] = v.values.astype(float)
        got = int(v.notna().sum())
        log(f"  {new}: {got:,}/{n_hs:,} human cells "
            f"({got / n_hs:.2%}), median {np.nanmedian(v.values):.4f}")

    prov = json.loads(a.uns.get("provenance", "{}"))
    prov["pt_transfer"] = {
        "attached": datetime.now().isoformat(),
        "source_csv": str(Path(args.pt_csv).resolve()),
        "method": ("Nadaraya-Watson weighted mean of mouse pseudotime over the "
                   "cross-species block of the stitched SAMap graph "
                   "(samap_transfer_pt.py)"),
        "columns": {new: src for src, new in colmap.items()},
        "note": ("human cells only, NaN on mouse; _weight = total cross-species "
                 "connectivity mass behind the estimate (low weight = weakly "
                 "anchored); ranknorm = global rank/N over source, "
                 "ranknorm_insample = rank/N within each mouse sample — the two "
                 "ranknorm variants are NOT interchangeable (spearman ~0.61)"),
    }
    a.uns["provenance"] = json.dumps(prov)

    tmp = str(args.out) + ".tmp"
    log(f"writing {args.out} ...")
    a.write_h5ad(tmp)
    os.replace(tmp, args.out)
    log(f"done: {args.out} ({Path(args.out).stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
