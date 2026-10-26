#!/usr/bin/env python
"""
pt_per_timepoint.py — per-timepoint scVI embedding + DPT pseudotime.

One run = one (species deliverable, timepoint): subset the fib/dec object to a single
obs['timepoint'] value, RETRAIN scVI on exactly those cells, then neighbors -> UMAP ->
diffmap -> DPT with an explicit root.

Design constraints that differ from the integrated (all-timepoint) pseudotime:
  * NO moscot / RealTimeKernel — temporal coupling needs >= 2 timepoints. Within one
    timepoint this is plain DPT on the per-timepoint embedding.
  * NO day-orientation gate — day is constant within a run. The orientation check is
    biological: median pt(decidual_cell) must exceed median pt(endometrial_fibroblast)
    (decidualization runs fib -> dec). If ROOT_STRATEGY picks a wrong-end root, this
    gate fails the run rather than writing a backwards column.
  * Root strategies (--root-strategy):
      anchor_far / anchor_near : among --root-anchor cells (default efib_proliferative),
                                 the cell farthest/nearest from the global centroid in
                                 DC2-DC3. Falls back to max_prolif if the anchor subtype
                                 is absent in this timepoint (small timepoints may lack it).
      max_prolif               : the fibroblast with the highest obs['proliferation'].
    The choice + fallback are recorded in the report and in obs['pt_root_desc'].

Outputs under --outdir:
  {species}_{timepoint}_pt.h5ad     subset with X_scVI (retrained), X_umap, X_diffmap,
                                    dpt_pseudotime, pt_root_desc
  {species}_{timepoint}_obs.csv     per-cell pt + labels + spatial coords
  {species}_{timepoint}_report.json full provenance + gates

Usage:
    python pt_per_timepoint.py \
        --h5ad data/deliverables/2026_08_28_Cevrim_XMens_Only_Fib_Dec_Cells.h5ad \
        --species mouse --timepoint 1_dpi_late \
        --drop-subtypes decidual_apoptotic \
        --outdir results/pt_per_timepoint/mouse_1_dpi_late
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Per-timepoint scVI + DPT pseudotime")
    p.add_argument("--h5ad", required=True, help="fib/dec deliverable")
    p.add_argument("--species", choices=["mouse", "human"], required=True)
    p.add_argument("--timepoint", required=True)
    p.add_argument("--cluster-key", default="cell_subtype")
    p.add_argument("--counts-layer", default="counts")
    p.add_argument("--batch-key", default="sample")
    p.add_argument("--drop-subtypes", nargs="*", default=[],
                   help="e.g. decidual_apoptotic (mouse)")
    p.add_argument("--n-hvg", type=int, default=3000)
    p.add_argument("--n-latent", type=int, default=30)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--n-diffmap", type=int, default=15)
    p.add_argument("--root-strategy", choices=["anchor_far", "anchor_near", "max_prolif"],
                   default="anchor_far",
                   help="anchor_far matched the best integrated-run ordering (rho_rank 0.79)")
    p.add_argument("--root-anchor", default="efib_proliferative")
    p.add_argument("--min-cells", type=int, default=500,
                   help="refuse to fit a timepoint smaller than this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=pathlib.Path, required=True)
    return p.parse_args()


FIB_COARSE = "endometrial_fibroblast"
DEC_COARSE = "decidual_cell"

report: dict = {}


def log(m):
    print(m, flush=True)


def main():
    args = parse_args()
    import scanpy as sc
    import anndata as ad
    import scvi

    scvi.settings.seed = args.seed
    np.random.seed(args.seed)
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.species}_{args.timepoint}"
    report.update(species=args.species, timepoint=args.timepoint, seed=args.seed,
                  h5ad=str(args.h5ad), root_strategy=args.root_strategy)

    # ---- load + subset -----------------------------------------------------
    a = sc.read_h5ad(args.h5ad)
    assert args.counts_layer in a.layers, f"no {args.counts_layer!r} layer"
    tp_mask = (a.obs["timepoint"].astype(str) == args.timepoint).values
    if tp_mask.sum() == 0:
        raise SystemExit(f"timepoint {args.timepoint!r} has 0 cells; "
                         f"available: {sorted(a.obs['timepoint'].astype(str).unique())}")
    a = a[tp_mask].copy()
    log(f"[{tag}] timepoint subset: {a.n_obs:,} cells")

    if args.drop_subtypes:
        have = set(a.obs[args.cluster_key].astype(str).unique())
        to_drop = [s for s in args.drop_subtypes if s in have]   # tolerate absent labels
        if to_drop:
            keep = ~a.obs[args.cluster_key].astype(str).isin(to_drop).values
            log(f"[{tag}] dropping {to_drop}: -{int((~keep).sum()):,} cells")
            a = a[keep].copy()
        report["dropped_subtypes"] = to_drop
    report["n_cells"] = int(a.n_obs)

    if a.n_obs < args.min_cells:
        raise SystemExit(f"only {a.n_obs} cells after filtering (< --min-cells "
                         f"{args.min_cells}) — too small for a per-timepoint embedding")

    labels = a.obs[args.cluster_key].astype(str)
    coarse = a.obs["cell_type"].astype(str)
    report["cell_type_counts"] = {k: int(v) for k, v in coarse.value_counts().items()}
    assert {FIB_COARSE, DEC_COARSE} <= set(coarse.unique()), \
        f"need both {FIB_COARSE} and {DEC_COARSE} in this timepoint for the orientation gate"

    # ---- scVI on exactly these cells ----------------------------------------
    sc.pp.highly_variable_genes(a, n_top_genes=args.n_hvg, flavor="seurat_v3",
                                layer=args.counts_layer, batch_key=args.batch_key,
                                subset=False)
    train = a[:, a.var["highly_variable"]].copy()
    train.X = train.layers[args.counts_layer].copy()
    n_batches = train.obs[args.batch_key].nunique()
    batch_key = args.batch_key if n_batches > 1 else None   # single-sample timepoint
    scvi.model.SCVI.setup_anndata(train, batch_key=batch_key)
    model = scvi.model.SCVI(train, n_latent=args.n_latent)
    log(f"[{tag}] training scVI: {train.n_obs:,} cells x {train.n_vars} HVGs "
        f"(batches={n_batches})")
    model.train(max_epochs=args.max_epochs, early_stopping=True,
                early_stopping_patience=20, enable_progress_bar=False)
    a.obsm["X_scVI"] = model.get_latent_representation()
    report["scvi"] = dict(n_hvg=int(train.n_vars), n_latent=args.n_latent,
                          batch_key=batch_key, max_epochs=args.max_epochs)

    # ---- neighbors / UMAP / diffmap / DPT -----------------------------------
    sc.pp.neighbors(a, use_rep="X_scVI", random_state=args.seed)
    sc.tl.umap(a, random_state=args.seed)
    sc.tl.diffmap(a, n_comps=args.n_diffmap)

    dc = a.obsm["X_diffmap"][:, [2, 3]]
    strategy = args.root_strategy
    amask = (labels == args.root_anchor).values
    if strategy in ("anchor_far", "anchor_near") and not amask.any():
        log(f"[{tag}] anchor {args.root_anchor!r} absent in this timepoint — "
            f"falling back to max_prolif")
        strategy = "max_prolif"
        report["root_fallback"] = f"{args.root_strategy} -> max_prolif (anchor absent)"

    if strategy in ("anchor_far", "anchor_near"):
        cand = np.where(amask)[0]
        dists = np.linalg.norm(dc[amask] - dc[~amask].mean(axis=0), axis=1)
        pick = dists.argmax() if strategy == "anchor_far" else dists.argmin()
        root_idx = int(cand[pick])
    else:
        assert "proliferation" in a.obs.columns, "max_prolif needs obs['proliferation']"
        fibmask = (coarse == FIB_COARSE).values
        prolif = a.obs["proliferation"].to_numpy(float)
        pf = np.where(fibmask & np.isfinite(prolif))[0]
        assert pf.size, "no fibroblasts with finite proliferation score"
        root_idx = int(pf[np.argmax(prolif[pf])])

    root_desc = (f"{strategy} (anchor={args.root_anchor})"
                 if strategy.startswith("anchor") else strategy)
    report.update(root_idx=root_idx, root_desc=root_desc,
                  root_subtype=str(labels.iloc[root_idx]),
                  root_obs_name=str(a.obs_names[root_idx]))
    log(f"[{tag}] root: {root_desc} idx={root_idx} subtype={labels.iloc[root_idx]}")

    a.uns["iroot"] = root_idx
    sc.tl.dpt(a)
    pt = a.obs["dpt_pseudotime"].to_numpy(float)
    assert np.isfinite(pt).all(), f"{int((~np.isfinite(pt)).sum())} non-finite pt values"
    report["pt_min"], report["pt_max"] = float(pt.min()), float(pt.max())

    # ---- orientation gate: decidual later than fibroblast --------------------
    med_dec = float(np.median(pt[(coarse == DEC_COARSE).values]))
    med_fib = float(np.median(pt[(coarse == FIB_COARSE).values]))
    report["median_decidual"] = med_dec
    report["median_fibroblast"] = med_fib
    report["orientation_ok"] = bool(med_dec > med_fib)
    log(f"[{tag}] median pt: dec={med_dec:.4f} fib={med_fib:.4f}  "
        f"orientation {'OK' if med_dec > med_fib else 'FAILED'}")
    assert med_dec > med_fib, (
        "REFUSING to write: decidual median <= fibroblast median. Re-run with the "
        "opposite root direction (anchor_far <-> anchor_near) rather than negating pt.")

    # per-subtype table (informative; small timepoints may scramble fine order)
    report["by_subtype"] = {
        s: dict(median=float(np.median(pt[m])), std=float(np.std(pt[m])), n=int(m.sum()))
        for s in sorted(labels.unique())
        if (m := (labels == s).values).sum() > 0
    }

    # ---- write ---------------------------------------------------------------
    a.obs["pt_root_desc"] = root_desc
    import os
    h5 = out / f"{tag}_pt.h5ad"
    tmp = h5.with_suffix(".h5ad.tmp")
    a.write_h5ad(tmp)
    os.replace(tmp, h5)
    cols = [c for c in ("sample", "timepoint", "cell_type", args.cluster_key,
                        "dpt_pseudotime", "x_um_dbscan", "y_um_dbscan") if c in a.obs.columns]
    a.obs[cols].to_csv(out / f"{tag}_obs.csv")
    (out / f"{tag}_report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"[{tag}] wrote {h5.name}, obs csv, report")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        report["FAILED"] = traceback.format_exc()
        try:
            outdir = pathlib.Path(sys.argv[sys.argv.index("--outdir") + 1])
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "FAILED_report.json").write_text(json.dumps(report, indent=2, default=str))
        except Exception:
            pass
        traceback.print_exc()
        sys.exit(1)
