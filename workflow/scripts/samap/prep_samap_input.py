#!/usr/bin/env python
"""
prep_samap_input.py — build ONE SAMap input h5ad from a pipeline object.

    source object
      -> subset to fib/dec on the coarse cell-type column
      -> rename coarse/fine to the SHARED SAMap schema (cell_type_annotations /
         cell_subtype_annotations) that run_samap + the stain expect
      -> put CP10K (library-size normalized to 10,000, NOT log-transformed) in .X
      -> subset to one timepoint
      -> write inputs/h5ads_samap/{sp}_fib_dec_{tp}.h5ad

================================================================================
WHY CP10K IN .X  (changed 2026-08-29; was raw counts)
================================================================================
SAMap's default preprocessing is, from sc_samap 3.0.1 sam/core.py:

    preprocess_data(sum_norm="cell_median", norm="log", min_expression=1)
      1. sum-normalize:  D * (1/rowsum) * median(rowsum)
      2. transform:      log2(x + 1)
      3. zero out:       every value where log2(v+1) <= 1, i.e. v <= 1

Step 1 is a NO-OP on CP10K input: every row already sums to exactly 10,000, so the
scaling factor median(s)/s equals 1.0 for every cell. SAMap therefore applies its
single log2 and nothing is double-normalized.

CP10K in .X is numerically IDENTICAL to passing raw counts with sum_norm=1e4
(verified: max |diff| = 4.8e-07 on a simulated matrix at nucleus-like depth).

It also retains far more signal than the counts default, because min_expression=1
is applied on the post-sum_norm scale:

    input .X                 post-norm median    nonzeros surviving
    raw counts + cell_median        1.48               71.0%
    CP10K      + cell_median        5.03               99.7%
    log1p(CP10K)                    1.80               91.4%   <- DO NOT USE

At ~3,000 UMIs median depth (single-nucleus), the counts default discards ~29% of
nonzero entries. CP10K discards 0.3%.

WHAT MUST NEVER GO IN .X: a LOG-transformed matrix. That is the failure this script
originally guarded against — SAMap would log an already-logged matrix. The guard
below still blocks it; it now checks for the CP10K scale instead of integer counts.

.raw is NOT a place to hide counts: sam/core.py:459-465 copies adata.raw into
adata_raw, uses it only for layers["X_disp"], then `del self.adata.raw`. The
mapping always runs on .X. So .X is the only thing that matters for integration.
================================================================================

Usage:
    python prep_samap_input.py --refined <source.h5ad> --species hs \
        --coarse-col cell_type --fine-col cell_subtype --timepoint secretory \
        --expect-fibdec 47533 --out .../inputs/h5ads_samap/hs_fib_dec_secretory.h5ad
"""
import argparse

import numpy as np
import anndata as ad
import scipy.sparse as sp

FIBDEC = {"decidual_cell", "endometrial_fibroblast"}
TARGET_SUM = 1e4          # CP10K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refined", required=True)
    p.add_argument("--species", choices=["hs", "mm"], required=True)
    p.add_argument("--coarse-col", required=True, help="obs col carrying the fib/dec cell_type")
    p.add_argument("--fine-col", required=True, help="obs col carrying the fine subtypes")
    p.add_argument("--timepoint", required=True)
    p.add_argument("--tp-col", default="timepoint")
    p.add_argument("--counts-layer", default="counts")
    p.add_argument("--expect-fibdec", type=int, default=None,
                   help="expected whole fib/dec count (warns, does not fail, if different)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"Loading {args.refined} ...", flush=True)
    a = ad.read_h5ad(args.refined)
    print(f"  {a.shape}; coarse={args.coarse_col!r} fine={args.fine_col!r} tp={args.tp_col!r}")
    for col in (args.coarse_col, args.fine_col, args.tp_col):
        if col not in a.obs.columns:
            raise SystemExit(f"obs column {col!r} not present. Have: {list(a.obs.columns)}")

    # --- subset to fib/dec on the coarse column ------------------------------------------
    keep = a.obs[args.coarse_col].astype(str).isin(FIBDEC).values
    a = a[keep].copy()
    got = set(a.obs[args.coarse_col].astype(str).unique())
    assert got == FIBDEC, f"after fib/dec subset the coarse labels are {got}, expected {FIBDEC}"
    print(f"  fib/dec subset: {a.n_obs} cells")
    if args.expect_fibdec is not None and a.n_obs != args.expect_fibdec:
        print(f"  WARNING: fib/dec count {a.n_obs} != expected {args.expect_fibdec} "
              f"(annotation may have changed since that number was recorded)")

    # --- shared SAMap schema -------------------------------------------------------------
    obs = a.obs.copy()
    obs["cell_type_annotations"] = obs[args.coarse_col].astype(str)
    obs["cell_subtype_annotations"] = obs[args.fine_col].astype(str)
    a.obs = obs

    # --- CP10K INTO .X -------------------------------------------------------------------
    if args.counts_layer not in a.layers:
        raise SystemExit(
            f"source has no {args.counts_layer!r} layer — cannot build CP10K. "
            f"Layers present: {list(a.layers)}")

    a.layers["lognorm"] = a.X.copy()                     # preserve the pipeline's log matrix
    counts = a.layers[args.counts_layer]

    # verify the counts layer really is counts before normalizing from it
    _c = counts[:500]
    _c = _c.toarray() if sp.issparse(_c) else np.asarray(_c)
    assert np.allclose(_c, np.round(_c)) and _c.max() > 50, (
        f"{args.counts_layer!r} does not look like raw counts "
        f"(max={_c.max():.2f}, integer={np.allclose(_c, np.round(_c))}). Refusing to "
        f"normalize from a non-count matrix.")

    counts = sp.csr_matrix(counts, dtype="float32") if sp.issparse(counts) \
        else sp.csr_matrix(np.asarray(counts, dtype="float32"))
    rowsum = np.asarray(counts.sum(1)).ravel()
    n_empty = int((rowsum == 0).sum())
    if n_empty:
        raise SystemExit(f"{n_empty} cells have zero total counts — cannot CP10K-normalize")

    a.X = counts.multiply(TARGET_SUM / rowsum[:, None]).tocsr()
    del a.layers[args.counts_layer]
    if a.raw is not None:
        del a.raw
    print(f"  .X <- CP10K (library-size normalized to {TARGET_SUM:.0f}, NOT log-transformed)")

    # --- scale guard: .X must be CP10K, and must NOT be log ------------------------------
    rs = np.asarray(a.X.sum(1)).ravel()
    _xs = a.X[:500]
    _xs = _xs.toarray() if sp.issparse(_xs) else np.asarray(_xs)
    xmax = float(_xs.max())

    assert np.allclose(rs, TARGET_SUM, rtol=1e-4), (
        f"CP10K guard: row sums are not {TARGET_SUM:.0f} "
        f"(min={rs.min():.2f}, max={rs.max():.2f})")
    assert xmax > 20, (
        f"CP10K guard: .X max is {xmax:.3f}, too small for CP10K — this looks LOG-scaled. "
        f"SAMap applies log2 itself; a pre-logged matrix gets logged twice.")
    assert (_xs >= 0).all(), "CP10K guard: negative values in .X"
    print(f"  guard OK: row sums == {TARGET_SUM:.0f}, .X max = {xmax:.1f} (not log-scale)")

    # record the choice in the object so a reader does not have to infer it
    a.uns["samap_input_scale"] = "CP10K"
    a.uns["samap_input_scale_note"] = (
        "obs .X is library-size normalized to 10,000 per cell, NOT log-transformed. "
        "SAMap's default sum_norm='cell_median' is a no-op on this scale (all row sums "
        "equal), so SAMap applies exactly one log2(x+1). Numerically identical to passing "
        "raw counts with sum_norm=1e4. Raw counts are recoverable only from the source "
        "pipeline object; layers['lognorm'] holds the pipeline's log-normalized matrix."
    )

    # --- subset to the one timepoint -----------------------------------------------------
    tpmask = (a.obs[args.tp_col].astype(str) == args.timepoint).values
    if int(tpmask.sum()) == 0:
        avail = sorted(a.obs[args.tp_col].astype(str).unique())
        raise SystemExit(f"timepoint {args.timepoint!r} has 0 fib/dec cells. Available: {avail}")
    a = a[tpmask].copy()
    print(f"  timepoint={args.timepoint}: {a.n_obs} cells")

    # post-subset re-check: slicing must not have disturbed the scale
    rs2 = np.asarray(a.X.sum(1)).ravel()
    assert np.allclose(rs2, TARGET_SUM, rtol=1e-4), "row sums drifted after timepoint subset"

    n_fine = int((a.obs["cell_subtype_annotations"].astype(str).str.len() > 0).sum())
    print(f"  cells with a fine subtype: {n_fine}/{a.n_obs}")
    print(f"  carried through: obsm={list(a.obsm.keys())}  "
          f"cell_barcode in obs? {'cell_barcode' in a.obs.columns}")

    print(f"Writing {args.out} ...", flush=True)
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(args.out) + ".tmp"
    a.write_h5ad(tmp)
    import os
    os.replace(tmp, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
