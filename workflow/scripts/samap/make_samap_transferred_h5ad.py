#!/usr/bin/env python
"""make_transferred_h5ad.py — collaborator-ready joint object with transfers.

Loads one arm's samap.pkl and writes <arm>/samap_transferred.h5ad containing:
  * the stitched joint object (SAM-processed expression in .X, joint UMAP,
    combined kNN graph, homology graphs in varp)
  * unified per-cell metadata as plain string obs columns (species, sample,
    timepoint, cell_type_annotations, cell_subtype_annotations) — pulled from
    the per-species SAM objects, since the joint copy keeps only
    categorical-dtype columns
  * ALL cross-species transferred labels + vote confidences from
    <arm>/staining/*.csv, as obs columns:
      transferred_mm_coarse / _confidence   (human cells; NaN on mouse)
      transferred_mm_fine   / _confidence   (human cells)
      transferred_hs_coarse / _confidence   (mouse cells)
  * uns['provenance'] describing exactly what .X is and where raw counts live.

Usage: python make_transferred_h5ad.py --arm-dir runs_raw/vanilla_raw
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import anndata._core.file_backing as _fb


def _ps(self, state):
    if "_adata_ref" not in state and "_adata" in state:
        state = dict(state)
        state["_adata_ref"] = state.pop("_adata")
    self.__dict__.update(state)


_fb.AnnDataFileManager.__setstate__ = _ps


def log(m):
    print(m, flush=True)


def load_samap_pkl(path):
    try:
        from samap import load_samap
    except ImportError:
        from samap.utils import load_samap
    return load_samap(str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm-dir", required=True)
    p.add_argument("--out-name", default="samap_transferred.h5ad")
    args = p.parse_args()
    arm = Path(args.arm_dir)

    log(f"loading {arm}/samap.pkl ...")
    sm = load_samap_pkl(arm / "samap.pkl")
    joint = sm.samap.adata
    log(f"joint object: {joint.shape}")

    # --- unified per-species metadata (plain strings; joint copy is lossy) ---
    for col in ("sample", "timepoint", "cell_type_annotations",
                "cell_subtype_annotations"):
        v = pd.Series(index=joint.obs_names, dtype=object)
        for sid in sm.sams:
            so = sm.sams[sid].adata.obs
            common = so.index.intersection(joint.obs_names)
            if col in so.columns:
                v.loc[common] = so.loc[common, col].astype(str).values
        joint.obs[col] = v.values

    # --- transferred labels + confidences from the staining CSVs -------------
    attached = []
    for stem, col in (("hs_from_mm_coarse", "transferred_mm_coarse"),
                      ("hs_from_mm_fine", "transferred_mm_fine"),
                      ("mm_from_hs_coarse", "transferred_hs_coarse")):
        f = arm / "staining" / f"{stem}.csv"
        if not f.exists():
            log(f"  (missing {f.name}; skipped)")
            continue
        df = pd.read_csv(f).set_index("cell")
        lab = df[col].reindex(joint.obs_names)
        conf = df["confidence"].reindex(joint.obs_names)
        joint.obs[col] = lab.values
        joint.obs[f"{col}_confidence"] = conf.values.astype(float)
        n = int(lab.notna().sum())
        attached.append(f"{col} ({n} cells)")
        log(f"  attached {col}: {n} cells, "
            f"median confidence {np.nanmedian(conf.values):.3f}")

    # object-dtype obs columns write poorly with NaN mixtures -> stringify
    for c in joint.obs.columns:
        if joint.obs[c].dtype == object:
            joint.obs[c] = joint.obs[c].astype(str).replace("nan", "NA")

    joint.uns["provenance"] = json.dumps({
        "created": datetime.now().isoformat(),
        "arm": str(arm),
        "X": ("SAM-preprocessed expression: log2(cell_median-normalized + 1) "
              "with the arm's min_expression cut; block-diagonal over species; "
              "NOT raw counts"),
        "raw_counts": "inputs_raw/{hs,mm}_fib_dec_all_raw.h5ad (layers-free, .X = counts)",
        "transfers": attached,
        "transfer_method": ("argmax connectivity-weighted vote of cross-species "
                            "neighbors on the stitched SAMap graph; confidence = "
                            "winning label's fraction of cross-species vote mass"),
        "graphs": "obsp['connectivities'] = stitched joint kNN; varp = homology graphs",
    })

    out = arm / args.out_name
    log(f"writing {out} ...")
    joint.write_h5ad(out)
    log(f"done: {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
