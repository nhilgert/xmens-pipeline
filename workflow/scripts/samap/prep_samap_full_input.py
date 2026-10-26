"""Build one full (all-timepoint) SAMap input h5ad from a fib/dec deliverable.

Reads the per-species deliverable h5ad (--src) and writes
inputs_raw/{sp}_fib_dec_all_<scale>.h5ad with either CP10K or raw integer
counts in .X, the shared SAMap annotation columns in obs, obsm carried through,
and no layers or .raw. Called by rule samap_raw_prep (workflow/rules/10b_samap_raw.smk).

Sibling of prep_samap_input.py that keeps every timepoint ("all by all"),
carries obsm untouched (the scVI-arm latents ride along) and drops the lognorm
layer copy. .raw is deleted because
sc-samap's load_data uses .raw.X as the source matrix when .raw exists and
discards .X, which would silently flip SAMap back onto raw counts.
"""

from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import scipy.sparse as sp

FIBDEC = {"decidual_cell", "endometrial_fibroblast"}
TARGET_SUM = 1e4  # CP10K: counts per 10,000 per cell


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--species", choices=["hs", "mm"], required=True)
    p.add_argument("--coarse-col", default="cell_type")
    p.add_argument("--fine-col", default="cell_subtype")
    p.add_argument("--counts-layer", default="counts")
    p.add_argument("--scale", choices=["cp10k", "raw"], default="cp10k",
                   help="cp10k = library-normalized to 1e4 (SAMap's sum_norm "
                        "becomes a no-op); raw = untouched integer counts, the "
                        "input SAMap's paper actually specifies")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Load the deliverable ----
    print(f"Loading {args.src} ...", flush=True)
    a = ad.read_h5ad(args.src)
    print(f"  {a.shape}; obs columns: {list(a.obs.columns)}")
    print(f"  obsm keys: {list(a.obsm.keys())}")
    print(f"  layers: {list(a.layers.keys())}")

    for col in (args.coarse_col, args.fine_col, "sample", "timepoint"):
        if col not in a.obs.columns:
            raise SystemExit(f"obs column {col!r} not present. Have: {list(a.obs.columns)}")

    # Subset to fib/dec ----
    # The *_Only_Fib_Dec_Cells deliverables are already fib/dec only, so this is
    # an assertion in practice.
    keep = a.obs[args.coarse_col].astype(str).isin(FIBDEC).values
    if keep.sum() < a.n_obs:
        print(f"  WARNING: {a.n_obs - keep.sum()} non-fib/dec cells dropped")
    a = a[keep].copy()
    print(f"  fib/dec cells: {a.n_obs}")
    print(f"  samples: {sorted(a.obs['sample'].astype(str).unique())}")
    print(f"  timepoints: {sorted(a.obs['timepoint'].astype(str).unique())}")

    # Shared SAMap annotation schema ----
    obs = a.obs.copy()
    obs["cell_type_annotations"] = obs[args.coarse_col].astype(str)
    obs["cell_subtype_annotations"] = obs[args.fine_col].astype(str)
    a.obs = obs

    # Counts layer into .X, CP10K-scaled or raw ----
    if args.counts_layer not in a.layers:
        raise SystemExit(
            f"source has no {args.counts_layer!r} layer. Layers: {list(a.layers)}"
        )
    counts = a.layers[args.counts_layer]

    # A log-normalized matrix mislabeled as counts would be silently renormalized,
    # so the first 500 cells must be integer-valued with a count-sized maximum.
    _c = counts[:500]
    _c = _c.toarray() if sp.issparse(_c) else np.asarray(_c)
    assert np.allclose(_c, np.round(_c)) and _c.max() > 50, (
        f"{args.counts_layer!r} does not look like raw counts "
        f"(max={_c.max():.2f}, integer={np.allclose(_c, np.round(_c))})."
    )

    counts = (
        sp.csr_matrix(counts, dtype="float32")
        if sp.issparse(counts)
        else sp.csr_matrix(np.asarray(counts, dtype="float32"))
    )
    rowsum = np.asarray(counts.sum(1)).ravel()

    n_empty = int((rowsum == 0).sum())
    if n_empty:
        raise SystemExit(f"{n_empty} cells have zero total counts")

    if args.scale == "cp10k":
        a.X = counts.multiply(TARGET_SUM / rowsum[:, None]).tocsr()
    else:
        a.X = counts  # untouched integer counts

    # Layers and .raw only add memory to the fit, and .raw would override .X in SAMap.
    for lay in list(a.layers.keys()):
        del a.layers[lay]
    if a.raw is not None:
        del a.raw
    print(f"  .X <- {args.scale.upper()}; layers dropped; .raw deleted")

    # Scale guards ----
    rs = np.asarray(a.X.sum(1)).ravel()
    _xs = a.X[:2000]
    _xs = _xs.toarray() if sp.issparse(_xs) else np.asarray(_xs)
    xmax = float(_xs.max())
    assert (_xs >= 0).all(), "negative values in .X"

    if args.scale == "cp10k":
        assert np.allclose(rs, TARGET_SUM, rtol=1e-4), (
            f"CP10K guard: row sums not {TARGET_SUM:.0f} "
            f"(min={rs.min():.2f}, max={rs.max():.2f})"
        )
        # A log1p matrix tops out around 10; CP10K maxima are in the hundreds.
        assert xmax > 20, f"CP10K guard: .X max {xmax:.3f} looks LOG-scaled"
        print(f"  guard OK: row sums == {TARGET_SUM:.0f}, .X max = {xmax:.1f}")
    else:
        # Raw counts are integer-valued, count-sized, and have variable library
        # sizes; a pre-normalized matrix would have CV ~ 0 and fail here.
        assert np.allclose(_xs, np.round(_xs)), (
            "RAW guard: .X is not integer-valued — this is not raw counts"
        )
        assert xmax > 50, f"RAW guard: .X max {xmax:.2f} too small to be counts"
        cv = float(rs.std() / rs.mean())
        assert cv > 0.05, (
            f"RAW guard: library sizes have CV={cv:.4f} — matrix looks "
            f"pre-normalized, not raw counts"
        )
        print(f"  guard OK: RAW COUNTS — integer-valued, .X max = {xmax:.0f}, "
              f"library sizes median={np.median(rs):.0f}, CV={cv:.3f} "
              f"(variable, as counts must be)")

    # Record the scale choice in the object so a reader does not have to infer it.
    a.uns["samap_input_scale"] = args.scale.upper()
    a.uns["samap_input_note"] = (
        f"full all-timepoint fib/dec input for species {args.species}; "
        + ("CP10K in .X (SAMap's cell_median sum-norm is a no-op on this scale)"
           if args.scale == "cp10k" else
           "RAW INTEGER COUNTS in .X — the input the SAMap paper specifies; SAM "
           "applies sum_norm='cell_median' -> log2(x+1) -> min_expression cut itself")
        + "; obsm carried through for the scVI slot-1 arm."
    )

    # Write ----
    # Write to a temp path and rename so a crash never leaves a truncated h5ad.
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(args.out) + ".tmp"
    print(f"Writing {args.out} ...", flush=True)
    a.write_h5ad(tmp)
    import os

    os.replace(tmp, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
