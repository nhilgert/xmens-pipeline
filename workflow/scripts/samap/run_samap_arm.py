"""One full-object cross-species SAMap fit plus evaluation, for a single arm.

Reads the two all-timepoint SAMap input h5ads (--hs, --mm) and the hsmm/
homology maps under --maps-dir. Writes samap.pkl, meta.json, eval.json,
mapping_{celltype,subtype,timepoint}.csv and figures/umap_*.png under
--out-dir. Called by rule samap_raw_fit (workflow/rules/10b_samap_raw.smk).

Arms (--arm):
  vanilla : stock SAMap, no within-species correction (the baseline)
  harmony : stock SAMap's built-in hook, sam.run(batch_key='sample'), which
            runs harmonypy on the weighted PCA inside every SAM iteration
            (requires harmonypy==0.0.10 because the >=2.0 API breaks the hook)
  scvi    : slot 1, where the per-species manifold is rebuilt after SAM.run
            from a precomputed scVI latent found in obsm
            (samap_scvi.use_embedding_manifold, with SAMap itself stock)

All arms share identical inputs (from prep_samap_full_input.py) and identical
SAMAP settings, so differences are attributable to the within-species manifold.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# SAMAP defaults (samap/core/mapping.py and _constants.py, sc-samap 3.0.1), spelled
# out so a change in a future release cannot silently alter the fit.
PRE = dict(sum_norm="cell_median", norm="log", thresh_low=0.0, thresh_high=0.96,
           min_expression=1)
RUN = dict(preprocessing="StandardScaler", npcs=100, weight_PCs=False, k=20,
           n_genes=3000, weight_mode="rms", verbose=False)

# obsm keys tried in order when --latent-key AUTO; the fib/dec-specific latents first.
LATENT_PRIORITY = [
    "X_scvi_fibdec", "X_scVI_fibdec", "X_scvi_fib_dec", "X_scVI_fib_dec",
    "X_scvi", "X_scVI", "X_scanvi", "X_scANVI", "X_latent", "latent",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_latent_key(obsm_keys: list[str], requested: str) -> str:
    if requested != "AUTO":
        if requested not in obsm_keys:
            raise SystemExit(f"--latent-key {requested!r} not in obsm: {obsm_keys}")
        return requested

    for k in LATENT_PRIORITY:
        if k in obsm_keys:
            return k

    # Fall back to a name match only when it is unambiguous.
    rx = re.compile(r"scvi|scanvi|latent", re.IGNORECASE)
    hits = [k for k in obsm_keys if rx.search(k) and "umap" not in k.lower()]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(
        f"Cannot auto-pick a latent from obsm keys {obsm_keys} (regex hits: {hits}). "
        f"Pass --latent-key explicitly."
    )


def describe_input_scale(sam) -> str:
    """Report what SAM actually loaded, for the record in the job log.

    SAM.load_data prefers .raw.X when .raw exists, so this inspects the matrix
    SAM will preprocess rather than the one the prep script intended to write.
    """
    X = sam.adata.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    head = X[: min(2000, X.shape[0])]
    head = head.toarray()
    rs = np.asarray(X.sum(1)).ravel()

    # Raw counts are integer-valued with variable library sizes; CP10K rows all sum
    # to the same value, so their CV is numerically zero.
    integer = bool(np.allclose(head, np.round(head)))
    cv = float(rs.std() / max(rs.mean(), 1e-12))
    kind = ("RAW COUNTS" if integer and cv > 0.05 else
            "CP10K" if cv < 1e-6 else
            "non-integer / normalized")
    return (f"  INPUT SCALE = {kind}: integer={integer}, max={head.max():.1f}, "
            f"library median={np.median(rs):.0f}, CV={cv:.3f}, "
            f"has_raw={sam.adata.raw is not None}")


def build_sam(path: str, arm: str, batch_key: str, latent_key: str,
              min_expression: float = 1):
    """Load one species, preprocess, and run SAM with the arm's manifold choice."""
    from samap.sam import SAM

    sam = SAM()
    sam.load_data(path)
    log(f"  loaded {path}: {sam.adata.shape}")
    log(describe_input_scale(sam))

    pre = dict(PRE, min_expression=min_expression)
    log(f"  preprocess_data({pre})")
    sam.preprocess_data(**pre)

    run_kw = dict(RUN)
    if arm == "harmony":
        run_kw["batch_key"] = batch_key
        log(f"  SAM.run with batch_key={batch_key!r} (harmonypy inside the loop)")
    t0 = time.time()
    sam.run(**run_kw)
    log(f"  SAM.run done in {(time.time() - t0) / 60:.1f} min")

    # The scvi arm replaces the manifold after SAM.run (samap_scvi.use_embedding_manifold).
    used_key = None
    if arm == "scvi":
        from samap_scvi import use_embedding_manifold

        keys = list(sam.adata.obsm.keys())
        used_key = pick_latent_key(keys, latent_key)
        Z = np.asarray(sam.adata.obsm[used_key])
        if Z.ndim != 2 or not (2 <= Z.shape[1] <= 256):  # a UMAP or a matrix is not a latent
            raise SystemExit(f"obsm[{used_key!r}] has shape {Z.shape}; not a latent")
        log(f"  slot 1: rebuilding manifold from obsm[{used_key!r}] {Z.shape}")
        use_embedding_manifold(sam, Z, key_added="X_slot1", k=RUN["k"])
    return sam, used_key


# Evaluation on the stitched object ----


def sample_mixing(nnm: sp.spmatrix, samples: np.ndarray) -> dict:
    """Observed vs expected fraction of neighbor weight in a different sample."""
    nnm = nnm.tocsr()
    n = nnm.shape[0]
    cats, codes = np.unique(samples, return_inverse=True)
    onehot = sp.csr_matrix(
        (np.ones(n), (np.arange(n), codes)), shape=(n, cats.size)
    )

    per_batch = nnm @ onehot  # (n x n_batches) weight to each sample
    total = np.asarray(per_batch.sum(1)).ravel()
    own = np.asarray(per_batch[np.arange(n), codes]).ravel()
    ok = total > 0
    obs_frac = ((total - own) / np.maximum(total, 1e-12))[ok]

    # Expected under random mixing: the share of all cells not in the cell's own sample.
    counts = np.bincount(codes, minlength=cats.size).astype(float)
    exp_frac = (1.0 - counts[codes] / n)[ok]
    return {
        "observed_other_sample_frac": float(obs_frac.mean()),
        "expected_other_sample_frac": float(exp_frac.mean()),
        "mixing_ratio": float(obs_frac.mean() / max(exp_frac.mean(), 1e-12)),
        "n_samples": int(cats.size),
    }


def knn_label_agreement(nnm: sp.spmatrix, labels: np.ndarray) -> float:
    """Mean fraction of neighbor weight sharing the cell's label (graph purity)."""
    nnm = nnm.tocsr()
    n = nnm.shape[0]
    cats, codes = np.unique(labels, return_inverse=True)
    onehot = sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, cats.size))

    per = nnm @ onehot
    total = np.asarray(per.sum(1)).ravel()
    own = np.asarray(per[np.arange(n), codes]).ravel()
    ok = total > 0
    return float((own[ok] / total[ok]).mean())


def evaluate(sm, out: Path) -> dict:
    """Alignment, within-species mixing/purity, mapping tables and UMAPs for one arm."""
    import scanpy as sc
    from samap.analysis import get_mapping_scores
    from samap.core.mapping import _avg_as
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ad_j = sm.samap.adata
    nnm = ad_j.obsp["connectivities"].tocsr()
    species = ad_j.obs["species"].values

    res: dict = {"n_cells_joint": int(ad_j.n_obs)}

    # Cross-species alignment: SAMap's species-by-species alignment score, off-diagonal mean.
    A = _avg_as(sm.samap)
    res["alignment"] = {r: {c: float(A.loc[r, c]) for c in A.columns} for r in A.index}
    off = A.values[~np.eye(A.shape[0], dtype=bool)]
    res["mean_alignment"] = float(off.mean())

    # Joint leiden once on the symmetrized graph, reused for per-species ARI/NMI.
    t0 = time.time()
    sym = ((nnm + nnm.T) * 0.5).tocsr()
    sc.tl.leiden(ad_j, adjacency=sym, key_added="joint_leiden", resolution=1.0,
                 flavor="igraph", n_iterations=2, directed=False)
    log(f"  joint leiden: {ad_j.obs['joint_leiden'].nunique()} clusters "
        f"({time.time() - t0:.0f}s)")

    # Within-species metrics on each species' block of the joint graph.
    for sid in np.unique(species):
        mask = species == sid
        sub = nnm[mask][:, mask]
        obs_names = ad_j.obs_names[mask]
        src = sm.sams[sid].adata.obs.loc[obs_names]
        samples = src["sample"].astype(str).values
        ctype = src["cell_type_annotations"].astype(str).values
        subtype = src["cell_subtype_annotations"].astype(str).values
        leiden = ad_j.obs["joint_leiden"].values[mask]

        d = {
            "sample_mixing": sample_mixing(sub, samples),
            "knn_celltype_agreement": knn_label_agreement(sub, ctype),
            "leiden_vs_celltype_ARI": float(adjusted_rand_score(ctype, leiden)),
            "leiden_vs_celltype_NMI": float(
                normalized_mutual_info_score(ctype, leiden)
            ),
            "leiden_vs_sample_NMI": float(
                normalized_mutual_info_score(samples, leiden)
            ),
        }

        # Subtype metrics only mean something when the species has more than one subtype.
        if len(set(subtype)) > 1:
            d["knn_subtype_agreement"] = knn_label_agreement(sub, subtype)
            d["leiden_vs_subtype_ARI"] = float(adjusted_rand_score(subtype, leiden))
            d["leiden_vs_subtype_NMI"] = float(
                normalized_mutual_info_score(subtype, leiden)
            )
        res[f"within_{sid}"] = d

    # Mapping tables ----
    # Human has no fine subtypes, so the subtype table pairs human coarse with mouse fine.
    for tag, keys in [
        ("celltype", {"hs": "cell_type_annotations", "mm": "cell_type_annotations"}),
        ("subtype", {"hs": "cell_type_annotations", "mm": "cell_subtype_annotations"}),
        ("timepoint", {"hs": "timepoint", "mm": "timepoint"}),
    ]:
        try:
            _, M = get_mapping_scores(sm, keys, n_top=0)
            M.to_csv(out / f"mapping_{tag}.csv")
            if tag == "celltype":
                hs_rows = [i for i in M.index if i.startswith("hs_")]
                mm_cols = [c for c in M.columns if c.startswith("mm_")]
                res["celltype_mapping"] = {
                    r: {c: float(M.loc[r, c]) for c in mm_cols} for r in hs_rows
                }
        except Exception as e:  # the tables are diagnostics, and eval.json must still be written
            log(f"  mapping table {tag} failed: {e}")

    # Figures ----
    # Diagnostic UMAPs, not rule outputs, so a render failure must not fail the arm.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figdir = out / "figures"
        figdir.mkdir(exist_ok=True)

        # Per-species obs columns live on the SAM objects, not the joint object, so
        # they are copied over with a species prefix for coloring.
        for col in ("timepoint", "sample", "cell_type_annotations"):
            v = pd.Series(index=ad_j.obs_names, dtype=object)
            for sid in sm.sams:
                so = sm.sams[sid].adata.obs
                common = so.index.intersection(ad_j.obs_names)
                v.loc[common] = sid + ":" + so.loc[common, col].astype(str)
            ad_j.obs[f"joint_{col}"] = pd.Categorical(v.values)

        for col, fname in [
            ("species", "umap_species.png"),
            ("joint_timepoint", "umap_timepoint.png"),
            ("joint_sample", "umap_sample.png"),
            ("joint_cell_type_annotations", "umap_celltype.png"),
        ]:
            fig = sc.pl.embedding(ad_j, basis="X_umap", color=col,
                                  return_fig=True, show=False)
            fig.savefig(figdir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        log(f"  figures skipped (non-fatal): {e}")

    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hs", required=True)
    p.add_argument("--mm", required=True)
    p.add_argument("--maps-dir", required=True, help="parent of hsmm/")
    p.add_argument("--arm", choices=["vanilla", "harmony", "scvi"], required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-key", default="sample")
    p.add_argument("--latent-key", default="AUTO")
    p.add_argument("--min-expression", type=float, default=1,
                   help="SAM's post-log low-expression cut. 1 = SAMap default "
                        "(drops <=1 count in a median cell); 0 = disabled.")
    p.add_argument("--n-iter", type=int, default=3)
    p.add_argument("--ncpus", type=int, default=8)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from samap import SAMAP, save_samap

    # Provenance is written before the fit so a crashed run still leaves a trail.
    meta = {
        "arm": args.arm,
        "started_at": datetime.now().isoformat(),
        "hs": args.hs,
        "mm": args.mm,
        "maps_dir": args.maps_dir,
        "batch_key": args.batch_key if args.arm == "harmony" else None,
        "n_iter": args.n_iter,
        "min_expression": args.min_expression,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Per-species SAM ----
    log(f"=== arm: {args.arm} ===")
    sams = {}
    latent_keys = {}
    for sid, path in (("hs", args.hs), ("mm", args.mm)):
        log(f"building SAM for {sid} ...")
        sams[sid], latent_keys[sid] = build_sam(
            path, args.arm, args.batch_key, args.latent_key,
            min_expression=args.min_expression
        )
    meta["latent_keys"] = latent_keys

    # Cross-species SAMAP ----
    log("SAMAP init ...")
    t0 = time.time()
    sm = SAMAP(sams, f_maps=args.maps_dir.rstrip("/") + "/", save_processed=False)
    log(f"SAMAP init done in {(time.time() - t0) / 60:.1f} min")

    log(f"SAMAP run (NUMITERS={args.n_iter}) ...")
    t0 = time.time()
    sm.run(NUMITERS=args.n_iter, ncpus=args.ncpus)
    meta["samap_run_minutes"] = (time.time() - t0) / 60
    log(f"SAMAP run done in {meta['samap_run_minutes']:.1f} min")

    save_samap(sm, str(out / "samap.pkl"))
    log(f"wrote {out / 'samap.pkl'}")

    # Evaluate and record ----
    log("evaluating ...")
    res = evaluate(sm, out)
    res["arm"] = args.arm
    res["latent_keys"] = latent_keys
    (out / "eval.json").write_text(json.dumps(res, indent=2))

    meta["finished_at"] = datetime.now().isoformat()
    meta["mean_alignment"] = res["mean_alignment"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"=== arm {args.arm} complete; mean alignment {res['mean_alignment']:.3f} ===")


if __name__ == "__main__":
    main()
