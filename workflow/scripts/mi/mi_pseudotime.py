#!/usr/bin/env python3
"""
mi_pseudotime.py — mutual information between gene expression and pseudotime.

Cleaned, pipeline-ready rewrite of mi_batch/scripts/mi_pipeline_*.py. Same method
(sklearn mutual_info_regression + permutation null + BH FDR), but:

  * the pseudotime column is a PARAMETER (--pt-key), so it runs against any of the
    new Palantir pseudotimes (palantir_pt_rtk, palantir_pt_std) or DPT
    (dpt_pseudotime) without editing code;
  * prep / worker / merge share ONE contract — worker reads prep's matrices, it does
    NOT re-open the h5ad or re-chunk with a different count (the original silently did
    both, so a prep with --chunks 200 was ignored by a worker hardcoded to 500);
  * everything is CLI-driven with deterministic paths, so a Snakemake rule can wire
    prep -> worker[array] -> merge with no hidden state;
  * MI calls take an explicit random_state, so a rerun reproduces the numbers
    (mutual_info_regression uses a kNN estimator with random tie-breaking — without a
    fixed seed the observed MI and the null both wander between runs).

Sub-commands
------------
  prep     read h5ad -> filter genes -> write expr matrix, pseudotime vector(s),
           gene list, and a chunk manifest.  Run once per (dataset, pt-key).
  worker   one SLURM array task: load prep outputs, take one chunk's genes, compute
           MI(gene, pseudotime) + N permutations, write results/mi_<chunk>.csv.
  merge    concat per-chunk CSVs -> Benjamini-Hochberg q -> mi_all.csv (sorted by q).
  diff     OPTIONAL: given two merged mi_all.csv (e.g. two conditions or two pt-keys),
           compute delta-MI per shared gene with a combined permutation z-score.
           This is the "differential MI" comparison the original suite did by hand.

Layout produced under --outdir
------------------------------
  expr.npz            float32 dense expression matrix  (cells x genes_kept)
  pseudotime.npy      1-D pseudotime vector aligned to rows of expr
  genes.npy           gene names aligned to columns of expr
  manifest.json       {pt_key, n_cells, n_genes, n_chunks, min_counts, layer, seed}
  results/mi_###.csv  per-chunk: gene, mi, p  (+ null columns if --keep-null)
  mi_all.csv          merged, with q

Example (matches how the Snakemake rule below drives it)
--------------------------------------------------------
  python mi_pseudotime.py prep \
      --h5ad results/.../mouse_fibdec_pseudotime_rooted.h5ad \
      --pt-key palantir_pt_rtk --layer log1p --min-counts 10 \
      --chunks 200 --outdir results/mi/mouse_palantir_rtk

  # SLURM array 0-199
  python mi_pseudotime.py worker --outdir results/mi/mouse_palantir_rtk \
      --chunk $SLURM_ARRAY_TASK_ID --perms 1001

  python mi_pseudotime.py merge --outdir results/mi/mouse_palantir_rtk
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- utils
def empirical_p(obs: float, null: np.ndarray) -> float:
    """One-sided empirical p: P(null >= observed), with the +1/+1 continuity guard so
    a gene can never get p=0 (that would break BH). n permutations -> min p = 1/(n+1)."""
    return (float((null >= obs).sum()) + 1.0) / (len(null) + 1.0)


def chunk_bounds(n_items: int, n_chunks: int, chunk_id: int) -> tuple[int, int]:
    """Half-open [start, end) for chunk_id, splitting n_items into n_chunks even pieces.
    Deterministic and identical between prep (which writes n_chunks) and worker (which
    reads it back from the manifest) — that is the contract the original broke."""
    if not 0 <= chunk_id < n_chunks:
        raise ValueError(f"chunk_id {chunk_id} out of range [0, {n_chunks})")
    edges = np.linspace(0, n_items, n_chunks + 1, dtype=int)
    return int(edges[chunk_id]), int(edges[chunk_id + 1])


def load_manifest(outdir: pathlib.Path) -> dict:
    mpath = outdir / "manifest.json"
    if not mpath.exists():
        raise SystemExit(f"no manifest.json in {outdir} — run `prep` first")
    return json.loads(mpath.read_text())


# ----------------------------------------------------------------------------- prep
def cmd_prep(args: argparse.Namespace) -> None:
    import scanpy as sc

    out = args.outdir.resolve()
    (out).mkdir(parents=True, exist_ok=True)

    print(f"[prep] reading {args.h5ad}", flush=True)
    adata = sc.read_h5ad(args.h5ad)

    if args.pt_key not in adata.obs.columns:
        raise SystemExit(
            f"[prep] pseudotime key {args.pt_key!r} not in obs. "
            f"Available: {sorted(adata.obs.columns)}")
    pt = adata.obs[args.pt_key].to_numpy(dtype=float)
    n_finite = int(np.isfinite(pt).sum())
    if n_finite < adata.n_obs:
        raise SystemExit(
            f"[prep] {adata.n_obs - n_finite} cells have non-finite {args.pt_key!r}; "
            f"subset or re-root before MI (a NaN pseudotime silently poisons MI).")

    # detection filter (2026-09-16): keep genes detected (count > 0) in at least
    # --min-detect-frac of cells, computed on the counts layer -- the posted
    # methods' "expressed in at least 1% of cells" criterion. Sparse genes have
    # degenerate KSG nulls (tiny MI passes); this removes them explicitly rather
    # than letting subsampling attrition do it implicitly. 0 disables (legacy).
    if args.min_detect_frac > 0:
        import scipy.sparse as _sp
        C = adata.layers["counts"] if "counts" in adata.layers else adata.X
        det = np.asarray((C > 0).mean(axis=0)).ravel()
        keep = det >= args.min_detect_frac
        print(f"[prep] detection filter >= {args.min_detect_frac:.2%} of cells: "
              f"{int(keep.sum()):,}/{adata.n_vars:,} genes kept", flush=True)
        adata = adata[:, keep].copy()

    # gene filter on counts, exactly as the original (keeps informative genes only)
    sc.pp.filter_genes(adata, min_counts=args.min_counts)
    print(f"[prep] {adata.n_vars:,} genes remain after min_counts={args.min_counts}", flush=True)

    layer_used = args.layer
    if args.layer not in adata.layers:
        if "counts" in adata.layers:
            # Rooted objects from the notebook path carry only 'counts'. Rather
            # than fail, derive the standard log1p(CP10K) here — the derivation
            # is deterministic and recorded in the manifest as layer_used.
            print(f"[prep] layer {args.layer!r} absent; deriving log1p(CP10K) "
                  f"from 'counts' (recorded in manifest)", flush=True)
            import scipy.sparse as sp
            C = adata.layers["counts"]
            C = C.tocsr().astype(np.float64) if sp.issparse(C) else np.asarray(C, np.float64)
            rs = np.asarray(C.sum(axis=1)).ravel()
            rs[rs == 0] = 1.0
            C = (sp.diags(1e4 / rs) @ C) if sp.issparse(C) else (C * (1e4 / rs)[:, None])
            X = C.toarray() if sp.issparse(C) else C
            np.log1p(X, out=X)
            layer_used = "log1p_cp10k_derived_from_counts"
        else:
            raise SystemExit(
                f"[prep] layer {args.layer!r} absent and no 'counts' layer to derive "
                f"from. Layers: {list(adata.layers)}.")
    else:
        X = adata.layers[args.layer]
        X = X.toarray() if not isinstance(X, np.ndarray) else X
    X = np.ascontiguousarray(X, dtype=np.float32)

    np.savez_compressed(out / "expr.npz", X=X)
    np.save(out / "pseudotime.npy", pt)
    np.save(out / "genes.npy", adata.var_names.to_numpy())

    manifest = dict(
        pt_key=args.pt_key, layer=layer_used, min_counts=int(args.min_counts),
        min_detect_frac=float(args.min_detect_frac),
        n_cells=int(adata.n_obs), n_genes=int(adata.n_vars),
        n_chunks=int(args.chunks), seed=int(args.seed), h5ad=str(args.h5ad),
        # power settings recorded as provenance; workers default to these values
        # (the 2025-09 era's unrecorded --subsample-cells cost a forensic evening)
        perms=int(args.perms), subsample_cells=int(args.subsample_cells),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[prep] wrote matrices + manifest to {out}  "
          f"({adata.n_obs:,} cells x {adata.n_vars:,} genes, {args.chunks} chunks)", flush=True)


# ----------------------------------------------------------------------------- worker
def cmd_worker(args: argparse.Namespace) -> None:
    from sklearn.feature_selection import mutual_info_regression

    out = args.outdir.resolve()
    man = load_manifest(out)
    n_chunks = man["n_chunks"]
    # power settings: CLI override, else the manifest's recorded values, else
    # legacy defaults (pre-2026-09-16 manifests lack these keys)
    if args.perms is None:
        args.perms = int(man.get("perms", 1001))
    if args.subsample_cells is None:
        args.subsample_cells = int(man.get("subsample_cells", 3000))

    X = np.load(out / "expr.npz")["X"]                 # cells x genes, float32
    y = np.load(out / "pseudotime.npy").astype(float)  # cells
    genes = np.load(out / "genes.npy", allow_pickle=True)
    assert X.shape[0] == y.shape[0] == man["n_cells"], "prep matrices are inconsistent"
    assert X.shape[1] == genes.shape[0] == man["n_genes"], "gene axis mismatch"

    start, end = chunk_bounds(man["n_genes"], n_chunks, args.chunk)
    gidx = range(start, end)
    print(f"[worker {args.chunk:03d}] genes [{start}:{end}) = {end - start}  "
          f"pt_key={man['pt_key']}  perms={args.perms}  "
          f"subsample_cells={args.subsample_cells}", flush=True)

    # one RNG per chunk, seeded off the prep seed so the whole run is reproducible
    rng = np.random.default_rng(man["seed"] * 100_000 + args.chunk)
    # MI's own estimator seed: fixed per worker so observed + null are comparable
    mi_seed = man["seed"]
    n_null = args.perms - 1
    null_cols = [f"mi_null_{i}" for i in range(1, args.perms)]

    # ---- cost control (v2, after the 33k-cell run hit walltime) -------------
    # (a) SUBSAMPLE cells for the estimator: KSG MI on n cells is O(n log n)
    #     per call and stabilizes well below 33k cells. One fixed subsample per
    #     worker (same seed logic) keeps observed and null on identical cells.
    if args.subsample_cells and X.shape[0] > args.subsample_cells:
        sub = np.sort(rng.choice(X.shape[0], size=args.subsample_cells, replace=False))
        X, y = X[sub], y[sub]
        print(f"[worker {args.chunk:03d}] subsampled to {len(sub):,} cells "
              f"for the estimator", flush=True)
    # (b) SHARED PERMUTATIONS across genes in the chunk: draw the shuffled
    #     pseudotime vectors ONCE and reuse them for every gene. The per-gene
    #     empirical p is unchanged in expectation (each gene still sees
    #     args.perms-1 independent shuffles of y); sharing only introduces
    #     dependence BETWEEN genes' p-values, which BH tolerates (PRDS).
    Y_null = np.empty((n_null, y.shape[0]), dtype=y.dtype)
    y_shuf = y.copy()
    for p in range(n_null):
        rng.shuffle(y_shuf)
        Y_null[p] = y_shuf

    records = []
    for i in gidx:
        xi = X[:, [i]]
        mi_obs = float(mutual_info_regression(
            xi, y, discrete_features=False, random_state=mi_seed)[0])

        mi_null = np.empty(n_null, dtype=np.float32)
        for p in range(n_null):
            mi_null[p] = mutual_info_regression(
                xi, Y_null[p], discrete_features=False, random_state=mi_seed)[0]

        rec = {"gene": str(genes[i]), "mi": mi_obs, "p": empirical_p(mi_obs, mi_null)}
        if args.keep_null:
            rec.update(dict(zip(null_cols, mi_null)))
        records.append(rec)

    cols = ["gene", "mi", "p"] + (null_cols if args.keep_null else [])
    rdir = out / "results"
    rdir.mkdir(exist_ok=True)
    dest = rdir / f"mi_{args.chunk:03d}.csv"
    pd.DataFrame(records, columns=cols).to_csv(dest, index=False)
    print(f"[worker {args.chunk:03d}] wrote {len(records)} genes -> {dest}", flush=True)


# ----------------------------------------------------------------------------- merge
def cmd_merge(args: argparse.Namespace) -> None:
    import statsmodels.stats.multitest as smm

    out = args.outdir.resolve()
    man = load_manifest(out)
    files = sorted((out / "results").glob("mi_*.csv"))
    if not files:
        raise SystemExit(f"[merge] no per-chunk CSVs in {out/'results'}")

    n_expected = man["n_chunks"]
    if len(files) != n_expected:
        print(f"[merge] WARNING: {len(files)}/{n_expected} chunk files present — "
              f"merging a PARTIAL run (FDR will be computed on what's here).", flush=True)

    df = pd.concat([pd.read_csv(f, usecols=["gene", "mi", "p"]) for f in files],
                   ignore_index=True)
    df = df.drop_duplicates("gene", keep="first")
    df["q"] = smm.multipletests(df["p"].to_numpy(), method="fdr_bh")[1]
    df = df.sort_values("q").reset_index(drop=True)

    dest = out / "mi_all.csv"
    df.to_csv(dest, index=False)
    n_sig = int((df["q"] < args.fdr).sum())
    print(f"[merge] wrote {dest}  (n={len(df):,}, {n_sig:,} genes at FDR<{args.fdr})", flush=True)


# ----------------------------------------------------------------------------- diff
def cmd_diff(args: argparse.Namespace) -> None:
    """Differential MI between two merged runs (A vs B): same genes, different condition
    or different pt-key. Reports per shared gene: mi_a, mi_b, delta_mi = mi_a - mi_b, and
    each run's BH q (q_a, q_b), sorted by |delta_mi|. Ranking + the two q's flag genes
    whose MI is real in one run and moves between them; it is a descriptive comparison,
    not a significance test on the delta itself. (A permutation test on delta_mi would
    need the per-gene null MI distributions — run `worker --keep-null` on both and add a
    merge that preserves the mi_null_* columns; the current merge drops them, so that
    path is intentionally not wired here.)"""
    a = pd.read_csv(args.a).set_index("gene")
    b = pd.read_csv(args.b).set_index("gene")
    shared = a.index.intersection(b.index)
    if shared.empty:
        raise SystemExit("[diff] no shared genes between the two runs")

    d = pd.DataFrame(index=shared)
    d["mi_a"] = a.loc[shared, "mi"]
    d["mi_b"] = b.loc[shared, "mi"]
    d["delta_mi"] = d["mi_a"] - d["mi_b"]
    d["q_a"] = a.loc[shared, "q"]
    d["q_b"] = b.loc[shared, "q"]
    d = d.sort_values("delta_mi", key=lambda s: s.abs(), ascending=False)

    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest)
    print(f"[diff] wrote {dest}  ({len(d):,} shared genes)", flush=True)
    print("[diff] top |delta MI|:\n" + d.head(10).to_string())


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    P = argparse.ArgumentParser(description="MI(expression, pseudotime): prep / worker / merge / diff")
    sub = P.add_subparsers(dest="mode", required=True)

    pp = sub.add_parser("prep", help="write matrices + manifest")
    pp.add_argument("--h5ad", required=True)
    pp.add_argument("--pt-key", default="palantir_pt_rtk",
                    help="obs column with the pseudotime (e.g. palantir_pt_rtk, "
                         "palantir_pt_std, dpt_pseudotime)")
    pp.add_argument("--layer", default="log1p", help="log-normalized expression layer")
    pp.add_argument("--min-counts", type=int, default=10)
    pp.add_argument("--min-detect-frac", type=float, default=0.0,
                    help="keep genes detected (count>0) in >= this fraction of cells "
                         "(the posted methods used 0.01); 0 = legacy behaviour")
    pp.add_argument("--perms", type=int, default=1001,
                    help="run's intended permutations, recorded in the manifest; "
                         "workers default to this value")
    pp.add_argument("--subsample-cells", type=int, default=3000,
                    help="run's intended estimator subsample (0 = full cells), recorded "
                         "in the manifest; workers default to this value")
    pp.add_argument("--chunks", type=int, default=200)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--outdir", type=pathlib.Path, required=True)
    pp.set_defaults(func=cmd_prep)

    pw = sub.add_parser("worker", help="one array task: MI + permutations for one chunk")
    pw.add_argument("--outdir", type=pathlib.Path, required=True)
    pw.add_argument("--chunk", type=int, required=True,
                    help="chunk index (0-based); wire to $SLURM_ARRAY_TASK_ID")
    pw.add_argument("--perms", type=int, default=None,
                    help="total permutations INCLUDING observed (min p = 1/perms); "
                         "default: the manifest's recorded value")
    pw.add_argument("--subsample-cells", type=int, default=None,
                    help="cells used by the KSG estimator (one fixed per-worker subsample; "
                         "0 = all cells). 33k cells x 1001 perms x 113 genes blew the "
                         "4h walltime; KSG MI is stable well below 3k cells for this use.")
    pw.add_argument("--keep-null", action="store_true",
                    help="also write the per-gene null MI columns (large CSVs)")
    pw.set_defaults(func=cmd_worker)

    pm = sub.add_parser("merge", help="concat chunks + BH FDR")
    pm.add_argument("--outdir", type=pathlib.Path, required=True)
    pm.add_argument("--fdr", type=float, default=0.05, help="FDR threshold for the summary count")
    pm.set_defaults(func=cmd_merge)

    pd_ = sub.add_parser("diff", help="differential MI between two merged mi_all.csv")
    pd_.add_argument("--a", required=True, help="mi_all.csv for condition/pt-key A")
    pd_.add_argument("--b", required=True, help="mi_all.csv for condition/pt-key B")
    pd_.add_argument("--out", required=True, help="output diff CSV")
    pd_.set_defaults(func=cmd_diff)
    return P


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
