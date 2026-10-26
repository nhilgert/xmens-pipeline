"""Spatial label-transfer stainings for one full-object SAMap arm.

Reads runs_raw/<arm>/samap.pkl and writes, per direction and granularity,
<tgt>_from_<src>_<tag>.csv (transferred label and confidence per cell) and
stain_<tgt>_from_<src>_<tag>.png into --out-dir. Called by rule samap_raw_stain
(workflow/rules/10b_samap_raw.smk).

Each cell receives the argmax connectivity-weighted vote of its cross-species
neighbors' labels (mouse fine subtypes -> human cells, human coarse labels ->
mouse cells), painted at its spatial position (x_um_dbscan / y_um_dbscan), one
panel per timepoint, with the native labels alongside.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import anndata._core.file_backing as _fb


# Pickles written by an older anndata store the file-manager reference under
# "_adata"; the installed anndata expects "_adata_ref". Without this shim the
# samap.pkl cannot be unpickled.
def _ps(self, state):
    if "_adata_ref" not in state and "_adata" in state:
        state = dict(state)
        state["_adata_ref"] = state.pop("_adata")
    self.__dict__.update(state)


_fb.AnnDataFileManager.__setstate__ = _ps

import matplotlib

# The backend must be set before pyplot is imported; the cluster has no display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def log(m):
    print(m, flush=True)


def load_samap_pkl(path):
    # load_samap moved between samap releases.
    try:
        from samap import load_samap
    except ImportError:
        from samap.utils import load_samap
    return load_samap(str(path))


def vote(conn, src_mask, labels_src):
    """Connectivity-weighted label vote; returns (label, fraction of weight on the winner)."""
    cats = np.unique(labels_src)
    onehot = sp.csr_matrix(
        (np.ones(int(src_mask.sum())),
         (np.arange(int(src_mask.sum())),
          pd.Categorical(labels_src, categories=cats).codes)),
        shape=(int(src_mask.sum()), cats.size))

    V = conn[:, src_mask] @ onehot
    tot = np.asarray(V.sum(1)).ravel()
    pred = np.asarray(cats)[np.asarray(V.argmax(1)).ravel()]

    # Cells with no cross-species edge have zero total weight and no vote.
    pred = np.where(tot > 1e-9, pred, "unmapped")
    conf = np.asarray(V.max(1).todense()).ravel() / np.maximum(tot, 1e-9)
    return pred, conf


def panels(obs, pred, native_col, title, path, max_cols=3):
    """One column per timepoint; native labels on the top row, transferred below."""
    tps = sorted(obs["timepoint"].astype(str).unique())
    n = len(tps)
    ncol = min(max_cols, n)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(4.6 * ncol, 8.4 * nrow),
                             squeeze=False)

    # One shared palette so a label has the same color in the native and transferred rows.
    cats = sorted(set(pred) | set(obs[native_col].astype(str)))
    cmap = plt.get_cmap("tab20")
    cdict = {c: cmap(i % 20) for i, c in enumerate(cats)}
    cdict["unmapped"] = (0.85, 0.85, 0.85, 1.0)  # light gray

    for i, tp in enumerate(tps):
        r, c = divmod(i, ncol)
        m = (obs["timepoint"].astype(str) == tp).values
        x = pd.to_numeric(obs["x_um_dbscan"], errors="coerce").values[m]
        y = pd.to_numeric(obs["y_um_dbscan"], errors="coerce").values[m]
        ok = np.isfinite(x) & np.isfinite(y)  # cells without a spatial position are skipped

        for row_off, (vals, lab) in enumerate(
            [(obs[native_col].astype(str).values[m], "native"),
             (pred[m], "transferred")]):
            ax = axes[r * 2 + row_off][c]
            ax.scatter(x[ok], y[ok], c=[cdict[v] for v in vals[ok]], s=2.2,
                       linewidths=0, rasterized=True)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spn in ax.spines.values():
                spn.set_visible(False)
            ax.set_title(f"{tp} — {lab}", fontsize=9)

    # Blank the unused cells of the grid.
    for j in range(n, nrow * ncol):
        r, c = divmod(j, ncol)
        axes[r * 2][c].axis("off")
        axes[r * 2 + 1][c].axis("off")

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=cdict[c], label=c, markersize=6)
        for c in cats
        if c in set(pred) | set(obs[native_col].astype(str))
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))  # leave room for the legend below the panels

    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samap-pkl", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load the fitted SAMap and split the joint obs back by species ----
    log(f"loading {args.samap_pkl}")
    sm = load_samap_pkl(args.samap_pkl)
    joint = sm.samap.adata
    conn = joint.obsp["connectivities"].tocsr()
    species = joint.obs["species"].astype(str).values

    obs = {}
    masks = {}
    for sid in sm.sams:
        masks[sid] = species == sid
        obs[sid] = sm.sams[sid].adata.obs.loc[joint.obs_names[masks[sid]]]

    def informative(sid, col):
        # A fine column that only repeats the coarse labels carries nothing to transfer.
        return (col in obs[sid].columns and
                obs[sid][col].astype(str).nunique()
                > obs[sid]["cell_type_annotations"].astype(str).nunique())

    # Transfer in both directions at both granularities ----
    NAMES = {"hs": "Human", "mm": "Mouse"}
    combos = [(src, tgt, col, tag)
              for src, tgt in (("mm", "hs"), ("hs", "mm"))
              for col, tag in (("cell_type_annotations", "coarse"),
                               ("cell_subtype_annotations", "fine"))]

    for src, tgt, col, tag in combos:
        if tag == "fine" and not informative(src, col):
            log(f"skip {src}->{tgt} fine: {src} has no fine labels beyond coarse")
            continue

        labels_src = obs[src][col].astype(str).values
        pred, conf = vote(conn[masks[tgt]], masks[src], labels_src)
        stem = f"{tgt}_from_{src}_{tag}"
        pd.DataFrame({"cell": joint.obs_names[masks[tgt]],
                      f"transferred_{src}_{tag}": pred,
                      "confidence": conf}).to_csv(out / f"{stem}.csv", index=False)

        # Native comparator at matching granularity when the target has one.
        native_col = (col if (tag == "coarse" or informative(tgt, col))
                      else "cell_type_annotations")
        panels(obs[tgt], pred,
               native_col if native_col in obs[tgt].columns else "cell_type_annotations",
               f"{NAMES[tgt]} tissue — native vs transferred {NAMES[src].lower()} "
               f"{tag} labels",
               out / f"stain_{stem}.png")

        # Agreement is only meaningful where granularities are directly comparable.
        if native_col == col:
            agree = float((pred == obs[tgt][col].astype(str).values).mean())
            log(f"agreement {src}->{tgt} ({tag}): {agree:.3f}")

    log("done")


if __name__ == "__main__":
    main()
