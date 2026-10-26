"""Post-integration joint object with every transferred annotation attached.

Reads a fitted samap.pkl, the two coordinate npz files and, optionally, the
mouse per-timepoint pseudotime h5ad (--pt-h5ad). Writes one
results/samap_transfer/<run>_transferred.h5ad per run for a collaborator to
plot (joint SAMap UMAP, or per-species spatial). Called by rule
samap_transferred_object (workflow/rules/10_samap.smk). The transfer functions
are imported from transfer_label_staining.py, so the labels match the stain.

obs columns:
  species                        'hs' | 'mm'
  cell_type_independent          own-species coarse (decidual / endometrial fibroblast)
  cell_type_transferred          coarse label voted from the other species (binary
                                 project, both directions)
  transferred_frac_decidual      continuous weighted decidual fraction
  cell_subtype                   own-species fine label (cell_subtype_annotations of the
                                 input), falling back to the coarse label where blank
  cell_subtype_transferred       [--transfer-fine] mouse fine subtype voted onto human
                                 cells (categorical argmax)
  cell_subtype_transferred_frac  [--transfer-fine] winning-subtype neighbor fraction
  pseudotime                     [--pt-h5ad] mouse: native dpt (joined), human: transferred
  pseudotime_source              [--pt-h5ad] 'native' | 'transferred'
obsm:
  spatial (per-species coord system, so subset before plotting), X_umap (joint)
"""
import argparse, os, sys
import numpy as np
import pandas as pd

# Puts this file's directory on sys.path so the sibling module imports regardless
# of how the file is loaded.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transfer_label_staining as T

COARSE = {1: "decidual", 0: "endometrial fibroblast", -1: "unlabeled"}
TRANSF = {1: "decidual", 0: "endometrial fibroblast", -1: "no cross-species neighbor"}
BLANK  = {"", "unassigned", "nan", "none"}  # fine-subtype values that mean "no label"


def join_by_bc(joint, bc_col, mask, src_bc, src_val):
    """Join one per-cell value from a source object onto `mask` cells of the joint, by barcode.

    Barcodes are matched raw and with the -\\d+$ suffix stripped, and the variant
    that matches more cells wins. Returns (full-length array, match rate).
    """
    out = np.full(joint.n_obs, np.nan)
    idx = np.where(mask)[0]
    jraw = pd.Series(joint.obs[bc_col].values[idx], index=idx).astype(str)

    best_rate, best = -1, None
    for norm in (False, True):
        jb = np.asarray(T._core_bc(jraw.values)) if norm else jraw.values
        sb = (
            np.asarray(T._core_bc(src_bc))
            if norm
            else np.asarray(src_bc, dtype=object).astype(str)
        )
        lut = pd.Series(np.asarray(src_val, float), index=sb)
        lut = lut[~lut.index.duplicated()]
        vals = pd.to_numeric(pd.Series(jb, index=idx).map(lut.to_dict()), errors="coerce")
        r = np.isfinite(vals).mean()
        if r > best_rate:
            best_rate, best = r, vals.values

    out[idx] = best
    return out, best_rate


def transfer_categorical(C, src_mask, labels):
    """Categorical label transfer by argmax of weighted neighbor fraction per category.

    Only labeled source neighbors count. Returns (winner, winning_frac, categories).
    """
    valid = src_mask & np.array([str(l).lower() not in BLANK for l in labels])
    cats = list(pd.unique(labels[valid]))

    den = C.dot(valid.astype(float))  # total labeled-neighbor weight per cell
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
    ap.add_argument(
        "--pt-h5ad",
        default=None,
        help="mouse per-timepoint object with a pseudotime column (obs_names = mouse barcodes)",
    )
    ap.add_argument("--pt-col", default="dpt_pseudotime")
    ap.add_argument("--drop-X", action="store_true")
    args = ap.parse_args()

    # Load the joint object ----
    print(f"loading {args.samap_pkl} ...")
    joint = T.get_combined_adata(T.load_samap(args.samap_pkl))
    is_h, is_m = T.species_masks(joint)
    C = joint.obsp[T.GRAPH_KEY].tocsr()
    print(f"  joint {joint.n_obs} | human {int(is_h.sum())}  mouse {int(is_m.sum())}")

    obs = joint.obs

    # Coarse dec/fib transfer (binary, both directions) ----
    # Same computation as the stain; unlabeled cells carry NaN and do not vote.
    y = T.binary_labels(obs, is_h)
    dec = np.where(y == 1, 1.0, np.where(y == 0, 0.0, np.nan))
    frac_h, nz_h = T.project(C, is_m, dec)
    frac_m, nz_m = T.project(C, is_h, dec)

    transf = np.full(joint.n_obs, -1)
    transf[is_h & nz_h] = (frac_h[is_h & nz_h] >= 0.5).astype(int)  # majority vote
    transf[is_m & nz_m] = (frac_m[is_m & nz_m] >= 0.5).astype(int)

    obs["species"] = np.where(is_h, "hs", "mm")
    obs["cell_type_independent"] = [COARSE[v] for v in y]
    obs["cell_type_transferred"] = [TRANSF[v] for v in transf]
    obs["transferred_frac_decidual"] = np.where(is_h, frac_h, frac_m)

    # Own-species fine subtype, falling back to the coarse label where fine is blank.
    hs_sub, mm_sub = "hs_cell_subtype_annotations", "mm_cell_subtype_annotations"
    hs_ct,  mm_ct  = "hs_cell_type_annotations",    "mm_cell_type_annotations"

    def fine_or_coarse(sub_col, ct_col):
        s = (
            obs[sub_col].astype(str).values
            if sub_col in obs.columns
            else np.array([""] * joint.n_obs)
        )
        c = (
            obs[ct_col].astype(str).values
            if ct_col in obs.columns
            else np.array([""] * joint.n_obs)
        )
        return np.array(
            [si if str(si).lower() not in BLANK else ci for si, ci in zip(s, c)],
            dtype=object,
        )

    mm_fine = fine_or_coarse(mm_sub, mm_ct)
    hs_fine = fine_or_coarse(hs_sub, hs_ct)
    obs["cell_subtype"] = np.where(is_h, hs_fine, mm_fine)

    # Fine subtype transfer (categorical, mouse -> human) ----
    if args.transfer_fine:
        winner, wfrac, cats = transfer_categorical(C, is_m, mm_fine)

        # Only human cells receive the transferred label; mouse cells keep their own.
        ct = np.array([""] * joint.n_obs, dtype=object)
        ct[is_h] = winner[is_h]
        obs["cell_subtype_transferred"] = ct
        obs["cell_subtype_transferred_frac"] = np.where(is_h, wfrac, np.nan)

        vc = pd.Series(ct[is_h]).value_counts()
        print(f"  fine transfer (mouse->human), {len(cats)} source subtypes; human assignments:")
        for k, n in list(vc.items())[:12]:
            print(f"      {k!r}: {n}")

    # Pseudotime transfer (continuous, mouse -> human) ----
    if args.pt_h5ad:
        import anndata as ad
        a_pt = ad.read_h5ad(args.pt_h5ad, backed="r")
        mm_pt, rate = join_by_bc(joint, T.MM_BC_COL, is_m,
                                 a_pt.obs_names.astype(str).values,
                                 pd.to_numeric(a_pt.obs[args.pt_col], errors="coerce").values)

        tr_pt, _ = T.project(C, is_m, mm_pt)
        pt = np.where(is_m, mm_pt, tr_pt)
        obs["pseudotime"] = pt
        obs["pseudotime_source"] = np.where(is_m, "native", "transferred")

        n_h = int(np.isfinite(pt[is_h]).sum())
        n_m = int(np.isfinite(pt[is_m]).sum())
        print(f"  pseudotime: mouse-barcode join rate {rate:.2f}; "
              f"finite pt on {n_m} mouse, {n_h} human")
        # A join rate under half usually means the pt object is for another timepoint.
        if rate < 0.5:
            print(f"  WARNING: low pt join rate ({rate:.2f}) — "
                  f"is --pt-h5ad the right timepoint for {args.run_name}?")

    joint.obs = obs

    # Spatial coordinates into obsm['spatial'] ----
    Xh, Yh, rh = T.join_coords(joint, T.HS_BC_COL, is_h, args.hs_coords)
    Xm, Ym, rm = T.join_coords(joint, T.MM_BC_COL, is_m, args.mm_coords)
    joint.obsm["spatial"] = np.column_stack([np.where(is_h, Xh, Xm), np.where(is_h, Yh, Ym)])
    print(f"  coord join  human {rh:.2f}  mouse {rm:.2f}")

    # Slim and write ----
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
