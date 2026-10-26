"""SCENIC stage 0: fib/dec h5ad -> pooled AUCell h5ad, GRN loom and manifest.

Called by rules `scenic_pooled` (--sketch-n 0, writes the pooled h5ad) and
`scenic_sketch` (--sketch-n N --sketch-seed S --skip-h5ad, one loom per GRN seed)
in workflow/rules/13_scenic.smk. Reads the deliverable fib/dec h5ad (or the pooled
h5ad when sketching) and the TF list. Writes {species}_dsc_pooled.h5ad (raw counts
in X and layers['raw'], the AUCell input), {species}_dsc_{pooled|sketchTAG}.loom
(genes x cells, the GRN input) and a .manifest.json recording the parameters and
the sha256 of the source h5ad.

The procedure follows prep_fib_inputs.ipynb:
  * raw counts come from layers[--raw-layer], else layers['counts'], else
    layers['raw'], else .raw.X restricted to the object's var_names. Anything
    else is an error.
  * genes are kept if detected (count > 0) in >= --frac of all cells of the
    object. The mask is computed on the full object
    even when sketching, so every seed's loom carries the same gene set.
  * optional geosketch (--sketch-n) on obsm[--embedding] with --sketch-seed.
    Per-seed sketches bootstrap over cells as well as over boosting randomness.
  * the loom is genes x cells with the 'Gene' row attribute and the 'CellID',
    'nGene' and 'nUMI' column attributes pySCENIC expects.

Writes are atomic (tmp + rename). Runs under the pyscenic env (anndata 0.11, numpy 1.26;
see 13_scenic.smk).
"""
import argparse, hashlib, json, os, sys
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import loompy as lp


def parse_args():
    p = argparse.ArgumentParser(description="SCENIC Stage 0: build pooled h5ad + GRN loom")
    p.add_argument("--h5ad", required=True, help="source fib/dec h5ad (deliverable object)")
    p.add_argument("--species", required=True, choices=["mouse", "human"])
    p.add_argument(
        "--tf-list",
        required=True,
        help="allTFs_mm.txt / allTFs_hg38.txt (one gene per line)",
    )
    p.add_argument("--out-dir", required=True, help="lane dir, e.g. {rev}/scenic_seeds/{species}")
    p.add_argument(
        "--frac",
        type=float,
        default=0.01,
        help="keep genes detected in >= frac of ALL cells",
    )
    p.add_argument(
        "--raw-layer",
        default="counts",
        help="layer holding raw counts (fallbacks: 'raw', .raw)",
    )
    p.add_argument(
        "--sketch-n",
        type=int,
        default=0,
        help="geosketch to this many cells (0 = full)",
    )
    p.add_argument("--embedding", default="X_scVI", help="obsm key for geosketch")
    p.add_argument("--sketch-seed", type=int, default=0)
    p.add_argument("--celltype-col", default=None, help="obs column for the composition printout")
    p.add_argument(
        "--out-loom",
        default=None,
        help="explicit loom path (default derived from names)",
    )
    p.add_argument("--out-h5ad", default=None, help="explicit pooled h5ad path (default derived)")
    p.add_argument("--out-manifest", default=None)
    p.add_argument(
        "--skip-h5ad",
        action="store_true",
        help="do not (re)write the pooled h5ad; use when --h5ad already IS the pooled "
        "object (per-seed sketches)",
    )
    return p.parse_args()


def sketch_tag(n):
    """Cell count -> file tag: 15000 -> '15k', 2500 -> '2.5k'."""
    return f"{n // 1000}k" if n % 1000 == 0 else f"{n / 1000:g}k"


def sha256(path, chunk=1 << 24):  # 16 MiB per read (the h5ads are multi-GB)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_path(path):
    return f"{path}.partial.{os.getpid()}"


def read_h5ad_compat(path):
    """sc.read_h5ad, falling back to element-wise reads that skip /uns.

    Deliverables written by anndata >= 0.12 carry a 'null' encoding in
    uns/log1p/base that the pyscenic env's anndata 0.11 cannot decode.
    SCENIC needs X/layers/obs/var/obsm only, so uns is dropped in that case."""
    import anndata as ad
    import h5py
    try:
        return sc.read_h5ad(path)
    except Exception as e:
        print(f"[warn] read_h5ad failed ({repr(e).splitlines()[0][:160]}); "
              f"falling back to element-wise read without uns", flush=True)

    # read_elem moved between anndata releases.
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem

    parts, skipped = {}, {}
    with h5py.File(path, "r") as f:
        for key in ("X", "obs", "var", "layers", "obsm", "raw"):
            if key in f:
                try:
                    parts[key] = read_elem(f[key])
                except Exception as e:
                    skipped[key] = repr(e).splitlines()[0][:200]
    if "obs" not in parts or "var" not in parts or "X" not in parts:
        sys.exit(f"element-wise read failed for required parts; skipped={skipped}")

    adata = ad.AnnData(X=parts["X"], obs=parts["obs"], var=parts["var"],
                       layers=parts.get("layers") or None, obsm=parts.get("obsm") or None)
    if "raw" in parts and parts["raw"] is not None:
        try:
            adata.raw = parts["raw"]
        except Exception as e:
            skipped["raw_attach"] = repr(e)[:200]

    # Record what was dropped so the manifest consumer can see it.
    adata.uns["scenic_stage0_read_skipped"] = skipped
    return adata


def main():
    a = parse_args()
    use_sketch = a.sketch_n > 0
    tag = sketch_tag(a.sketch_n) if use_sketch else "full"
    os.makedirs(a.out_dir, exist_ok=True)

    # Output paths ----
    # Explicit --out-* paths win (the Snakemake rules pass them); otherwise the
    # names are derived from species and sketch tag.
    loom_path = a.out_loom or os.path.join(
        a.out_dir,
        f"{a.species}_dsc_sketch{tag}.loom" if use_sketch else f"{a.species}_dsc_pooled.loom",
    )
    h5ad_path = a.out_h5ad or os.path.join(a.out_dir, f"{a.species}_dsc_pooled.h5ad")
    man_path = a.out_manifest or os.path.join(
        a.out_dir,
        f"{a.species}_dsc_sketch{tag}.manifest.json"
        if use_sketch
        else f"{a.species}_dsc_pooled.manifest.json",
    )

    # Load source ----
    print(f"species={a.species} tag={tag} frac={a.frac:.1%} source={a.h5ad}", flush=True)
    src_sha = sha256(a.h5ad)
    print(f"source sha256={src_sha}", flush=True)

    adata = read_h5ad_compat(a.h5ad)
    print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes | layers={list(adata.layers.keys())} "
          f"raw={adata.raw is not None} obsm={list(adata.obsm.keys())}", flush=True)

    # Raw counts into X ----
    # Same fallback order as the notebook: the requested layer, then 'counts',
    # then 'raw', then .raw.X restricted to the object's genes.
    cand = [a.raw_layer] + [L for L in ("counts", "raw") if L != a.raw_layer]
    hit = next((L for L in cand if L in adata.layers), None)
    if hit is not None:
        adata.X = adata.layers[hit].copy()
        src = f"layers['{hit}']"
    elif adata.raw is not None:
        raw_ad = adata.raw.to_adata()
        keep = [g for g in adata.var_names if g in set(raw_ad.var_names)]
        adata = adata[:, keep].copy()
        adata.X = raw_ad[:, keep].X.copy()
        src = ".raw.X"
    else:
        sys.exit("No raw counts found in layers or .raw")

    # Sanity check on the first 50 cells: negative values mean scaled/centered
    # data, non-integers mean normalized or log data.
    X = adata.X
    sample = X[:50].toarray() if sp.issparse(X) else np.asarray(X[:50])
    print(
        f"raw source={src} dtype={X.dtype} min={sample.min():.3g} max={sample.max():.3g}",
        flush=True,
    )
    if (sample < 0).any():
        sys.exit("negative values -> NOT raw counts (scaled/centered). Stop.")
    nonint = float(np.abs(sample - np.round(sample)).max())
    if nonint > 1e-6:  # float tolerance for integer-valued counts
        print(
            f"[warn] non-integer values (max frac dev {nonint:.3g}) -- "
            f"OK if normalized, NOT if log/scaled",
            flush=True,
        )

    # Drop stale layers so the pooled h5ad carries exactly one copy of the counts.
    for L in list(adata.layers.keys()):
        del adata.layers[L]
    adata.layers["raw"] = adata.X.copy()

    # Pooled h5ad for AUCell ----
    if a.skip_h5ad:
        h5ad_path = os.path.abspath(a.h5ad)
        print(f"--skip-h5ad: pooled h5ad not rewritten (using {h5ad_path})", flush=True)
    else:
        tmp = atomic_path(h5ad_path)
        adata.write_h5ad(tmp)
        os.replace(tmp, h5ad_path)
        print(
            f"wrote {h5ad_path}  [{adata.n_obs:,} x {adata.n_vars:,}]  (FULL - for AUCell)",
            flush=True,
        )

    # Optional geosketch ----
    if use_sketch:
        from geosketch import gs
        if a.embedding not in adata.obsm:
            sys.exit(f"{a.embedding} not in obsm: {list(adata.obsm.keys())}")

        # Global seed as well as gs(seed=...): older geosketch releases take no
        # seed argument (the TypeError branch) and draw from numpy's global RNG.
        np.random.seed(a.sketch_seed)
        try:
            idx = gs(adata.obsm[a.embedding], a.sketch_n, replace=False, seed=a.sketch_seed)
        except TypeError:
            idx = gs(adata.obsm[a.embedding], a.sketch_n, replace=False)
        adata_grn_src = adata[idx].copy()
        print(
            f"geosketch: {adata.n_obs:,} -> {adata_grn_src.n_obs:,} cells from {a.embedding}",
            flush=True,
        )

        # Composition table: shows which cell types the sketch over- or under-samples.
        if a.celltype_col and a.celltype_col in adata.obs.columns:
            comp = pd.DataFrame({
                "full": adata.obs[a.celltype_col].astype(str).value_counts(),
                "sketch": adata_grn_src.obs[a.celltype_col].astype(str).value_counts(),
            }).fillna(0).astype(int)
            print(comp.to_string(), flush=True)
    else:
        adata_grn_src = adata

    # Gene filter and TF intersection ----
    # Detection is counted on the full object, not the sketch, so the gene set
    # is identical across seeds.
    with open(a.tf_list) as f:
        tfs = {ln.strip() for ln in f if ln.strip()}
    min_cells = max(1, int(round(a.frac * adata.n_obs)))
    ncpg_full = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    keep = ncpg_full >= min_cells
    adata_grn = adata_grn_src[:, keep].copy()
    n_tf_retained = int(adata_grn.var_names.isin(tfs).sum())
    print(
        f"genes: {adata_grn_src.n_vars:,} -> {adata_grn.n_vars:,} "
        f"(>= {min_cells} of {adata.n_obs:,} cells) | "
        f"TF list {len(tfs):,}, retained {n_tf_retained:,}",
        flush=True,
    )
    if n_tf_retained == 0:
        sys.exit("no TFs survived -- check nomenclature (MGI vs HGNC)")

    # Loom (genes x cells) ----
    # pySCENIC reads a genes x cells loom with these exact attribute names.
    Xg = adata_grn.X
    mat = Xg.T.tocsc() if sp.issparse(Xg) else sp.csc_matrix(np.asarray(Xg).T)
    row_attrs = {"Gene": np.array(adata_grn.var_names)}
    col_attrs = {"CellID": np.array(adata_grn.obs_names),
                 "nGene": np.asarray((Xg > 0).sum(axis=1)).ravel(),
                 "nUMI": np.asarray(Xg.sum(axis=1)).ravel()}
    tmp = atomic_path(loom_path)
    if os.path.exists(tmp):
        os.remove(tmp)
    lp.create(tmp, mat, row_attrs, col_attrs)
    os.replace(tmp, loom_path)
    print(
        f"wrote {loom_path}  [{mat.shape[0]:,} genes x {mat.shape[1]:,} cells]  (GRN input)",
        flush=True,
    )

    # Manifest ----
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "species": a.species,
        "source_h5ad": os.path.abspath(a.h5ad),
        "source_sha256": src_sha,
        "source_size_bytes": os.path.getsize(a.h5ad),
        "raw_source": src,
        "subset_applied": False,
        "use_sketch": use_sketch,
        "sketch_embedding": a.embedding if use_sketch else None,
        "sketch_n": a.sketch_n if use_sketch else None,
        "sketch_tag": tag,
        "sketch_seed": a.sketch_seed if use_sketch else None,
        "gene_filter_frac": a.frac,
        "gene_filter_basis": "full_object",
        "gene_filter_min_cells": min_cells,
        "n_cells_full": int(adata.n_obs),
        "n_cells_grn": int(adata_grn.n_obs),
        "n_genes_full": int(adata.n_vars),
        "n_genes_grn": int(adata_grn.n_vars),
        "n_tfs_in_list": len(tfs),
        "n_tfs_retained": n_tf_retained,
        "loom": loom_path,
        "h5ad": h5ad_path,
        "script": os.path.abspath(__file__),
    }
    tmp = atomic_path(man_path)
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, man_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
