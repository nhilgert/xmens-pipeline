"""Build one per-timepoint SAMap input h5ad from a fib/dec deliverable object.

Reads the per-species deliverable h5ad (--refined) and writes
inputs/h5ads_samap/{sp}_fib_dec_{tp}.h5ad with CP10K in .X, the shared SAMap
annotation columns in obs, and only the requested timepoint. Called by rule
prep_samap_input (workflow/rules/10_samap.smk).

Why CP10K (library-size normalized to 10,000, not log-transformed) goes in .X:
SAMap's default preprocessing (sc_samap 3.0.1 sam/core.py) is
sum_norm="cell_median", then log2(x + 1), then zeroing every value with
log2(v + 1) <= 1. On CP10K input the sum-normalization is a no-op (every row
already sums to 10,000, so median(s)/s == 1), and SAMap applies exactly one
log2, which is numerically identical to raw counts with sum_norm=1e4
(max |diff| = 4.8e-07 on a simulated matrix at nucleus-like depth). Because
min_expression=1 is applied on the post-sum_norm scale, CP10K keeps 99.7% of
nonzeros where raw counts with cell_median keep 71% at ~3,000 UMIs median
depth, and log1p(CP10K) keeps 91.4% while being double-logged by SAMap.
A log-transformed matrix must never go in .X, and the scale guard below blocks it.
.raw is not a place to hide counts: with .raw present, SAM.load_data uses .raw.X as the
source matrix instead of .X (sc-samap 3.0.1 sam/core.py), silently replacing CP10K.
"""
import argparse

import numpy as np
import anndata as ad
import scipy.sparse as sp

FIBDEC = {"decidual_cell", "endometrial_fibroblast"}
TARGET_SUM = 1e4  # CP10K: counts per 10,000 per cell


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

    # Load the deliverable ----
    print(f"Loading {args.refined} ...", flush=True)
    a = ad.read_h5ad(args.refined)
    print(f"  {a.shape}; coarse={args.coarse_col!r} fine={args.fine_col!r} tp={args.tp_col!r}")

    for col in (args.coarse_col, args.fine_col, args.tp_col):
        if col not in a.obs.columns:
            raise SystemExit(f"obs column {col!r} not present. Have: {list(a.obs.columns)}")

    # Subset to fib/dec on the coarse column ----
    # The deliverable is already fib/dec only, so this is an assertion in practice.
    keep = a.obs[args.coarse_col].astype(str).isin(FIBDEC).values
    a = a[keep].copy()

    got = set(a.obs[args.coarse_col].astype(str).unique())
    assert got == FIBDEC, f"after fib/dec subset the coarse labels are {got}, expected {FIBDEC}"
    print(f"  fib/dec subset: {a.n_obs} cells")

    if args.expect_fibdec is not None and a.n_obs != args.expect_fibdec:
        print(f"  WARNING: fib/dec count {a.n_obs} != expected {args.expect_fibdec} "
              f"(annotation may have changed since that number was recorded)")

    # Shared SAMap annotation schema ----
    # The stain and make_transferred_object.py read these two columns (hs_/mm_-prefixed).
    obs = a.obs.copy()
    obs["cell_type_annotations"] = obs[args.coarse_col].astype(str)
    obs["cell_subtype_annotations"] = obs[args.fine_col].astype(str)
    a.obs = obs

    # CP10K into .X ----
    if args.counts_layer not in a.layers:
        raise SystemExit(
            f"source has no {args.counts_layer!r} layer — cannot build CP10K. "
            f"Layers present: {list(a.layers)}")

    a.layers["lognorm"] = a.X.copy()  # keep the pipeline's log-normalized matrix
    counts = a.layers[args.counts_layer]

    # A log-normalized matrix mislabeled as counts would be silently renormalized,
    # so the first 500 cells must be integer-valued with a count-sized maximum.
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

    # SAM.load_data would take .raw.X over .X (see the docstring), so .raw must go.
    if a.raw is not None:
        del a.raw
    print(f"  .X <- CP10K (library-size normalized to {TARGET_SUM:.0f}, NOT log-transformed)")

    # Scale guard: .X must be CP10K and must not be log-scaled ----
    rs = np.asarray(a.X.sum(1)).ravel()
    _xs = a.X[:500]
    _xs = _xs.toarray() if sp.issparse(_xs) else np.asarray(_xs)
    xmax = float(_xs.max())

    assert np.allclose(rs, TARGET_SUM, rtol=1e-4), (
        f"CP10K guard: row sums are not {TARGET_SUM:.0f} "
        f"(min={rs.min():.2f}, max={rs.max():.2f})")
    # A log1p matrix tops out around 10; CP10K maxima are in the hundreds.
    assert xmax > 20, (
        f"CP10K guard: .X max is {xmax:.3f}, too small for CP10K — this looks LOG-scaled. "
        f"SAMap applies log2 itself; a pre-logged matrix gets logged twice.")
    assert (_xs >= 0).all(), "CP10K guard: negative values in .X"
    print(f"  guard OK: row sums == {TARGET_SUM:.0f}, .X max = {xmax:.1f} (not log-scale)")

    # Record the scale choice in the object so a reader does not have to infer it.
    a.uns["samap_input_scale"] = "CP10K"
    a.uns["samap_input_scale_note"] = (
        "obs .X is library-size normalized to 10,000 per cell, NOT log-transformed. "
        "SAMap's default sum_norm='cell_median' is a no-op on this scale (all row sums "
        "equal), so SAMap applies exactly one log2(x+1). Numerically identical to passing "
        "raw counts with sum_norm=1e4. Raw counts are recoverable only from the source "
        "pipeline object; layers['lognorm'] holds the pipeline's log-normalized matrix."
    )

    # Subset to the one timepoint ----
    tpmask = (a.obs[args.tp_col].astype(str) == args.timepoint).values
    if int(tpmask.sum()) == 0:
        avail = sorted(a.obs[args.tp_col].astype(str).unique())
        raise SystemExit(f"timepoint {args.timepoint!r} has 0 fib/dec cells. Available: {avail}")
    a = a[tpmask].copy()
    print(f"  timepoint={args.timepoint}: {a.n_obs} cells")

    # Slicing an AnnData view must not have disturbed the per-cell scale.
    rs2 = np.asarray(a.X.sum(1)).ravel()
    assert np.allclose(rs2, TARGET_SUM, rtol=1e-4), "row sums drifted after timepoint subset"

    n_fine = int((a.obs["cell_subtype_annotations"].astype(str).str.len() > 0).sum())
    print(f"  cells with a fine subtype: {n_fine}/{a.n_obs}")
    print(f"  carried through: obsm={list(a.obsm.keys())}  "
          f"cell_barcode in obs? {'cell_barcode' in a.obs.columns}")

    # Write ----
    # Write to a temp path and rename so a crash never leaves a truncated h5ad.
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
