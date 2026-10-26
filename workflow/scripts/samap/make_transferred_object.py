#!/usr/bin/env python
"""
make_transferred_object.py — post-integration object with ALL transferred annotations attached.

Loads a fitted SAMap pkl, reproduces the cross-species transfers (reusing the tested functions the
stain rule uses), attaches them to the joint object, joins spatial coords, and writes ONE h5ad per
run for a collaborator to plot (joint SAMap UMAP, or per-species spatial).

Three transfers, all mouse<->human via the SAMap graph:
  COARSE dec/fib  (binary project, both directions)     -> cell_type_transferred
  FINE subtype    (categorical argmax, mouse -> human)   -> cell_subtype_transferred  [--transfer-fine]
  PSEUDOTIME      (continuous project, mouse -> human)    -> pseudotime               [--pt-h5ad]

obs columns:
  species                        'hs' | 'mm'
  cell_type_independent          own-species coarse (decidual / endometrial fibroblast)
  cell_type_transferred          coarse label voted from the other species
  transferred_frac_decidual      continuous weighted decidual fraction
  cell_subtype                   own-species fine (mouse: dec1/2/3/decAp, eFib1/2/2i/3/P; human: usually "")
  cell_subtype_transferred       [--transfer-fine] mouse fine subtype voted onto HUMAN cells
  cell_subtype_transferred_frac  [--transfer-fine] winning-subtype neighbor fraction (confidence)
  pseudotime                     [--pt-h5ad] mouse: native dpt (joined); human: transferred (projected)
  pseudotime_source              [--pt-h5ad] 'native' | 'transferred'
obsm:
  spatial (per-species coord system; subset before plotting), X_umap (joint, co-registered)

Usage:
  python make_transferred_object.py --samap-pkl PKL --run-name RUN \
      --hs-coords isodepth_<phase>.npz --mm-coords transfer_all/radius_<stage>.npz \
      --out RUN_transferred.h5ad [--transfer-fine] [--pt-h5ad adata_<stage>.h5ad] [--pt-col dpt_pseudotime]
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transfer_label_staining as T           # tested: load_samap, get_combined_adata, species_masks,
                                               # binary_labels, project, join_coords, _core_bc, GRAPH_KEY, *_COL

COARSE = {1: "decidual", 0: "endometrial fibroblast", -1: "unlabeled"}
TRANSF = {1: "decidual", 0: "endometrial fibroblast", -1: "no cross-species neighbor"}
BLANK  = {"", "unassigned", "nan", "none"}


def join_by_bc(joint, bc_col, mask, src_bc, src_val):
    """Join a single per-cell value from a source object onto `mask` cells of the joint, by barcode.
    Two-pass (raw, then -\\d+$-stripped) like the kara_rings figure. Returns (full-length array, rate)."""
    out = np.full(joint.n_obs, np.nan)
    idx = np.where(mask)[0]
    jraw = pd.Series(joint.obs[bc_col].values[idx], index=idx).astype(str)
    best_rate, best = -1, None
    for norm in (False, True):
        jb = np.asarray(T._core_bc(jraw.values)) if norm else jraw.values
        sb = np.asarray(T._core_bc(src_bc)) if norm else np.asarray(src_bc, dtype=object).astype(str)
        lut = pd.Series(np.asarray(src_val, float), index=sb); lut = lut[~lut.index.duplicated()]
        vals = pd.to_numeric(pd.Series(jb, index=idx).map(lut.to_dict()), errors="coerce")
        r = np.isfinite(vals).mean()
        if r > best_rate:
            best_rate, best = r, vals.values
    out[idx] = best
    return out, best_rate


def transfer_categorical(C, src_mask, labels):
    """Categorical label transfer: for each cell, the argmax over per-category weighted neighbor
    fractions among labeled source neighbors. Returns (winner, winning_frac, categories)."""
    valid = src_mask & np.array([str(l).lower() not in BLANK for l in labels])
    cats = list(pd.unique(labels[valid]))
    den = C.dot(valid.astype(float))                                   # constant denominator per cell
    num = np.column_stack([C.dot((valid & (labels == c)).astype(float)) for c in cats])
    winner = np.array([""] * C.shape[0], dtype=object)
    wfrac = np.full(C.shape[0], np.nan)
    idx = np.where(den > 0)[0]
    am = np.argmax(num[idx], axis=1)
    winner[idx] = [cats[k] for k in am]
    wfrac[idx] = num[idx, am] / den[idx]
    return winner, wfrac, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samap-pkl", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--hs-coords", required=True)
    ap.add_argument("--mm-coords", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transfer-fine", action="store_true",
                    help="transfer mouse fine subtypes onto human (categorical)")
    ap.add_argument("--pt-h5ad", default=None,
                    help="mouse per-timepoint object with a pseudotime column (obs_names = mouse barcodes)")
    ap.add_argument("--pt-col", default="dpt_pseudotime")
    ap.add_argument("--drop-X", action="store_true")
    args = ap.parse_args()

    print(f"loading {args.samap_pkl} ...")
    joint = T.get_combined_adata(T.load_samap(args.samap_pkl))
    is_h, is_m = T.species_masks(joint)
    C = joint.obsp[T.GRAPH_KEY].tocsr()
    print(f"  joint {joint.n_obs} | human {int(is_h.sum())}  mouse {int(is_m.sum())}")

    obs = joint.obs

    # --- COARSE dec/fib (binary, both directions) ------------------------------------------
    y = T.binary_labels(obs, is_h)
    dec = np.where(y == 1, 1.0, np.where(y == 0, 0.0, np.nan))
    frac_h, nz_h = T.project(C, is_m, dec)
    frac_m, nz_m = T.project(C, is_h, dec)
    transf = np.full(joint.n_obs, -1)
    transf[is_h & nz_h] = (frac_h[is_h & nz_h] >= 0.5).astype(int)
    transf[is_m & nz_m] = (frac_m[is_m & nz_m] >= 0.5).astype(int)
    obs["species"] = np.where(is_h, "hs", "mm")
    obs["cell_type_independent"] = [COARSE[v] for v in y]
    obs["cell_type_transferred"] = [TRANSF[v] for v in transf]
    obs["transferred_frac_decidual"] = np.where(is_h, frac_h, frac_m)

    # own-species fine subtype (fine, else coarse fallback where fine is blank)
    hs_sub, mm_sub = "hs_cell_subtype_annotations", "mm_cell_subtype_annotations"
    hs_ct,  mm_ct  = "hs_cell_type_annotations",    "mm_cell_type_annotations"
    def fine_or_coarse(sub_col, ct_col):
        s = obs[sub_col].astype(str).values if sub_col in obs.columns else np.array([""] * joint.n_obs)
        c = obs[ct_col].astype(str).values  if ct_col  in obs.columns else np.array([""] * joint.n_obs)
        return np.array([si if str(si).lower() not in BLANK else ci for si, ci in zip(s, c)], dtype=object)
    mm_fine = fine_or_coarse(mm_sub, mm_ct)
    hs_fine = fine_or_coarse(hs_sub, hs_ct)
    obs["cell_subtype"] = np.where(is_h, hs_fine, mm_fine)

    # --- FINE subtype transfer (categorical, mouse -> human) -------------------------------
    if args.transfer_fine:
        winner, wfrac, cats = transfer_categorical(C, is_m, mm_fine)
        ct = np.array([""] * joint.n_obs, dtype=object)
        ct[is_h] = winner[is_h]                                   # only assign the transferred label to HUMAN
        obs["cell_subtype_transferred"] = ct
        obs["cell_subtype_transferred_frac"] = np.where(is_h, wfrac, np.nan)
        vc = pd.Series(ct[is_h]).value_counts()
        print(f"  fine transfer (mouse->human), {len(cats)} source subtypes; human assignments:")
        for k, n in list(vc.items())[:12]:
            print(f"      {k!r}: {n}")

    # --- PSEUDOTIME transfer (continuous, mouse -> human) ----------------------------------
    if args.pt_h5ad:
        import anndata as ad
        a_pt = ad.read_h5ad(args.pt_h5ad, backed="r")
        mm_pt, rate = join_by_bc(joint, T.MM_BC_COL, is_m,
                                 a_pt.obs_names.astype(str).values,
                                 pd.to_numeric(a_pt.obs[args.pt_col], errors="coerce").values)
        tr_pt, _ = T.project(C, is_m, mm_pt)                      # project returns (out, nz) — unpack
        pt = np.where(is_m, mm_pt, tr_pt)
        obs["pseudotime"] = pt
        obs["pseudotime_source"] = np.where(is_m, "native", "transferred")
        n_h = int(np.isfinite(pt[is_h]).sum()); n_m = int(np.isfinite(pt[is_m]).sum())
        print(f"  pseudotime: mouse-barcode join rate {rate:.2f}; finite pt on {n_m} mouse, {n_h} human")
        if rate < 0.5:
            print(f"  WARNING: low pt join rate ({rate:.2f}) — is --pt-h5ad the right timepoint for {args.run_name}?")

    joint.obs = obs

    # --- spatial coords -> obsm['spatial'] -------------------------------------------------
    Xh, Yh, rh = T.join_coords(joint, T.HS_BC_COL, is_h, args.hs_coords)
    Xm, Ym, rm = T.join_coords(joint, T.MM_BC_COL, is_m, args.mm_coords)
    joint.obsm["spatial"] = np.column_stack([np.where(is_h, Xh, Xm), np.where(is_h, Yh, Ym)])
    print(f"  coord join  human {rh:.2f}  mouse {rm:.2f}")

    # --- slim + write ----------------------------------------------------------------------
    if joint.raw is not None:
        del joint.raw
    for k in list(joint.layers.keys()):
        del joint.layers[k]
    if args.drop_X:
        import scipy.sparse as sp
        joint.X = sp.csr_matrix((joint.n_obs, joint.n_vars), dtype="float32")

    print(f"writing {args.out} ...  shape={joint.shape}")
    joint.write_h5ad(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
