"""
transfer_label_staining.py   —  run under the SAMap env (was: samap_env_2)
--------------------------------------------------------------------------
Pink-on-grey staining of the cross-species LABEL transfer, one label per panel — the honest
rendering that removes the z-order artifact (no overlaying two colors, so draw order can't change
what you see). Matches the subtype-plot aesthetic: black background, magenta = the label, grey = rest.

Per species (row mm, row hs), four columns:
  independent decidual | TRANSFERRED decidual | independent fibroblast | TRANSFERRED fibroblast

"Independent" = the cell's own-species annotation. "Transferred" = the label voted from the OTHER
species' neighbors through the SAMap graph (weighted fraction decidual, threshold 0.5).

CHANGED vs the original standalone:
  * All paths are now arguments (fixes the /n/eddy_lab -> /n/lab_storage migration by construction).
  * Numeric/plotting helpers below are preserved BYTE-FOR-BYTE from the tested original.
  * Adds an optional JSON stats sidecar (--out-stats): the numbers that used to be print-only,
    PLUS the pkl's own label categories, so a stale-pkl (old `stromal` label) is auditable from a
    file instead of by eyeballing an empty panel.

Runnable by hand:
  python transfer_label_staining.py \
      --run-name secretory_x_1dpi_late_hmmer \
      --samap-pkl  .../runs/secretory_x_1dpi_late_hmmer/samap.pkl \
      --hs-coords  .../crossspecies_radial_gp/isodepth_secretory.npz \
      --mm-coords  .../crossspecies_radial_gp/transfer_all/radius_1dpi_late.npz \
      --out-png    .../secretory_x_1dpi_late_hmmer_labelstain.png \
      --out-stats  .../secretory_x_1dpi_late_hmmer_labelstain.json
or driven by the Snakemake rule that passes the same six values.
"""
import os, pickle, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import anndata._core.file_backing as _fb
def _patched_setstate(self, state):
    if "_adata_ref" not in state and "_adata" in state:
        state = dict(state); state["_adata_ref"] = state.pop("_adata")
    self.__dict__.update(state)
_fb.AnnDataFileManager.__setstate__ = _patched_setstate

# ============================== constants (not paths) ======================
GRAPH_KEY = "connectivities"
HS_BC_COL, MM_BC_COL = "hs_cell_barcode", "mm_cell_barcode"
HS_ANNOT_COL, MM_ANNOT_COL = "hs_cell_type_annotations", "mm_cell_type_annotations"
DECIDUAL_KEYS, FIBRO_KEYS = ("dec",), ("fib", "stroma", "efib")
PINK, GREY = "#FF3DFF", "#5a5a5a"

# ============================== helpers (tested — verbatim) ================
def load_samap(path):
    with open(path, "rb") as f: return pickle.load(f)

def get_combined_adata(obj):
    for chain in (("samap", "adata"), ("adata",)):
        o, ok = obj, True
        for a in chain:
            if hasattr(o, a): o = getattr(o, a)
            else: ok = False; break
        if ok and hasattr(o, "obs"): return o
    raise ValueError("no combined adata")

def species_masks(adata):
    obs = adata.obs
    if "species" in obs.columns and obs["species"].nunique() == 2:
        vals = obs["species"].astype(str); uniq = set(vals.unique())
        if {"hs", "mm"} <= uniq:          ih = (vals == "hs").values
        elif {"human", "mouse"} <= uniq:  ih = (vals == "human").values
        else:
            a, b = sorted(uniq)
            a_is_h = (obs.loc[vals == a, HS_ANNOT_COL].notna().mean() >
                      obs.loc[vals == a, MM_ANNOT_COL].notna().mean())
            ih = (vals == (a if a_is_h else b)).values
        return ih, ~ih
    if HS_BC_COL in obs.columns and MM_BC_COL in obs.columns:
        ih, im = obs[HS_BC_COL].notna().values, obs[MM_BC_COL].notna().values
        if ih.any() and im.any() and not (ih & im).any(): return ih, im
    raise ValueError("could not split species")

def binary_labels(obs, is_h):
    hs = obs[HS_ANNOT_COL].astype(str).values; mm = obs[MM_ANNOT_COL].astype(str).values
    own = np.where(is_h, hs, mm); low = np.char.lower(own.astype(str))
    y = np.full(len(low), -1, dtype=int)
    y[[any(k in s for k in DECIDUAL_KEYS) for s in low]] = 1
    isf = np.array([any(k in s for k in FIBRO_KEYS) for s in low]); y[isf & (y != 1)] = 0
    return y

def _core_bc(s):
    return pd.Series(s, dtype=object).astype(str).str.replace(r"-\d+$", "", regex=True)

def join_coords(joint, joint_bc_col, mask, npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if not ("x" in d.files and "y" in d.files): raise ValueError(f"{npz_path} has no x/y coords")
    src_bc = pd.Series(d[d.files[0] if "barcode" not in d.files[0] else d.files[0]]).astype(str).values \
        if False else pd.Series(d[[k for k in d.files if "barcode" in k][0]]).astype(str).values
    sx = np.asarray(d["x"], float); sy = np.asarray(d["y"], float)
    X = np.full(joint.n_obs, np.nan); Y = np.full(joint.n_obs, np.nan)
    idx = np.where(mask)[0]
    jb_raw = pd.Series(joint.obs[joint_bc_col].values[idx], index=idx).astype(str)
    best_rate, bx, by = -1, None, None
    for norm in (False, True):
        jb = _core_bc(jb_raw.values) if norm else jb_raw.values
        sb = _core_bc(src_bc) if norm else src_bc
        lx = pd.Series(sx, index=sb); lx = lx[~lx.index.duplicated()]
        ly = pd.Series(sy, index=sb); ly = ly[~ly.index.duplicated()]
        vx = pd.Series(jb, index=idx).map(lx.to_dict()).values
        vy = pd.Series(jb, index=idx).map(ly.to_dict()).values
        rate = np.isfinite(pd.to_numeric(vx, errors="coerce")).mean()
        if rate > best_rate: best_rate, bx, by = rate, vx, vy
    X[idx] = pd.to_numeric(bx, errors="coerce"); Y[idx] = pd.to_numeric(by, errors="coerce")
    return X, Y, best_rate

def project(C, source_mask, source_val):
    sv = np.where(source_mask & np.isfinite(source_val), source_val, 0.0)
    si = (source_mask & np.isfinite(source_val)).astype(float)
    num = C.dot(sv); den = C.dot(si)
    out = np.full(C.shape[0], np.nan); nz = den > 0; out[nz] = num[nz] / den[nz]
    return out, nz

# ============================== driver =====================================
def _title_from_run(run_name):
    body = run_name[:-6] if run_name.endswith("_hmmer") else run_name
    phase, _, stage = body.partition("_x_")
    return (f"{phase} x mouse {stage}  —  label transfer "
            f"(magenta = label, grey = rest; per-label = no z-order)")

def run_stain(run_name, samap_pkl, hs_coords, mm_coords, out_png, out_stats=None):
    joint = get_combined_adata(load_samap(samap_pkl))
    C = joint.obsp[GRAPH_KEY].tocsr()
    is_h, is_m = species_masks(joint)
    y = binary_labels(joint.obs, is_h)
    print(f"{run_name}: {joint.n_obs} cells ({is_h.sum()} human, {is_m.sum()} mouse)")
    print(f"labels: dec={int((y==1).sum())} fib={int((y==0).sum())} unlabeled={int((y==-1).sum())}")

    # TRANSFER: weighted fraction decidual among cross-species neighbors, threshold 0.5
    dec_ind = np.where(y == 1, 1.0, np.where(y == 0, 0.0, np.nan))
    frac_h, nz_h = project(C, is_m, dec_ind)     # human <- mouse
    frac_m, nz_m = project(C, is_h, dec_ind)     # mouse <- human
    transf = np.full(joint.n_obs, -1)
    transf[is_h & nz_h] = (frac_h[is_h & nz_h] >= 0.5).astype(int)
    transf[is_m & nz_m] = (frac_m[is_m & nz_m] >= 0.5).astype(int)

    # agreement (transferred vs independent) per species -- the NUMBER behind the picture
    agree = {}
    for mask, nm in ((is_h, "human"), (is_m, "mouse")):
        m = mask & (transf >= 0) & (y >= 0)
        if m.sum():
            a  = float((transf[m] == y[m]).mean())
            dr = float(np.mean(transf[m & (y == 1)] == 1)) if int((m & (y == 1)).sum()) else None
            fr = float(np.mean(transf[m & (y == 0)] == 0)) if int((m & (y == 0)).sum()) else None
            agree[nm] = {"agreement": a, "dec_recall": dr, "fib_recall": fr, "n": int(m.sum())}
            print(f"  {nm}: transferred agrees with independent on {a:.2f} "
                  f"(dec recall={dr}, fib recall={fr})")

    # coords from npz
    Xh, Yh, rh = join_coords(joint, HS_BC_COL, is_h, hs_coords)
    Xm, Ym, rm = join_coords(joint, MM_BC_COL, is_m, mm_coords)
    X = np.where(is_h, Xh, Xm); Y = np.where(is_h, Yh, Ym)
    print(f"  coord join: human {rh:.2f}, mouse {rm:.2f}")

    # STAIN pink-on-grey, one label per panel (no z-order)
    def stain(ax, species_mask, label_mask, title):
        ax.set_facecolor("black")
        m = species_mask & np.isfinite(X) & np.isfinite(Y)
        bg = m & ~label_mask; fg = m & label_mask
        ax.scatter(X[bg], Y[bg], s=5, c=GREY, linewidths=0)
        ax.scatter(X[fg], Y[fg], s=5, c=PINK, linewidths=0)
        if title: ax.set_title(title, color="white", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    fig, ax = plt.subplots(2, 4, figsize=(16, 8)); fig.patch.set_facecolor("black")
    fig.suptitle(_title_from_run(run_name), color="white", fontsize=12, y=1.0)
    cols = [("independent decidual", y == 1), ("TRANSFERRED decidual", transf == 1),
            ("independent fibroblast", y == 0), ("TRANSFERRED fibroblast", transf == 0)]
    for ri, (mask, nm) in enumerate(((is_m, "MOUSE"), (is_h, "HUMAN"))):
        for ci, (ttl, lm) in enumerate(cols):
            stain(ax[ri, ci], mask, lm, ttl if ri == 0 else "")
        ax[ri, 0].set_ylabel(nm, color="white", fontsize=11)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="black"); plt.close(fig)
    print(f"\nsaved -> {out_png}")

    # stats sidecar: numbers + PROVENANCE (what labels the pkl actually carries)
    stats = {
        "run_name": run_name,
        "samap_pkl": os.path.abspath(samap_pkl),
        "n_obs": int(joint.n_obs), "n_human": int(is_h.sum()), "n_mouse": int(is_m.sum()),
        "pkl_label_categories": {
            "hs": sorted({str(v) for v in pd.Series(joint.obs[HS_ANNOT_COL]).dropna().unique()}),
            "mm": sorted({str(v) for v in pd.Series(joint.obs[MM_ANNOT_COL]).dropna().unique()}),
        },
        "labels": {"decidual": int((y == 1).sum()), "fibroblast": int((y == 0).sum()),
                   "unlabeled": int((y == -1).sum())},
        "transfer_agreement": agree,
        "coord_join_rate": {"human": float(rh), "mouse": float(rm)},
    }
    if out_stats:
        with open(out_stats, "w") as fh: json.dump(stats, fh, indent=2)
        print(f"stats -> {out_stats}")
    return stats

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Cross-species SAMap label-transfer staining (coarse dec/fib).")
    ap.add_argument("--run-name",  required=True)
    ap.add_argument("--samap-pkl", required=True)
    ap.add_argument("--hs-coords", required=True, help="human coord npz (isodepth_<phase>.npz)")
    ap.add_argument("--mm-coords", required=True, help="mouse coord npz (transfer_all/radius_<stage>.npz)")
    ap.add_argument("--out-png",   required=True)
    ap.add_argument("--out-stats", default=None, help="optional JSON stats sidecar")
    args = ap.parse_args()
    for p in (args.out_png, args.out_stats):
        if p:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    run_stain(args.run_name, args.samap_pkl, args.hs_coords, args.mm_coords,
              args.out_png, args.out_stats)

if __name__ == "__main__":
    main()
