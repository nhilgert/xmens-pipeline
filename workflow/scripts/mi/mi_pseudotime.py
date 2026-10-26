"""Mutual information between each gene's expression and pseudotime.

Called by rules `mi_prep`, `mi_worker` and `mi_merge` (11_mi_pseudotime.smk), one
sub-command each. Method: sklearn mutual_info_regression (kNN/KSG estimator) per gene,
a permutation null from shuffled pseudotime, one-sided empirical p and
Benjamini-Hochberg q across genes.

Sub-commands
  prep     read the h5ad, filter genes, write expr.npz (cells x genes, float32),
           pseudotime.npy, genes.npy and manifest.json under --outdir.
  worker   one array task: load prep's matrices, take one chunk of genes, compute
           MI(gene, pseudotime) plus permutations, write results/mi_<chunk>.csv.
  merge    concatenate the chunk CSVs, add BH q, write mi_all.csv sorted by q.
  diff     not wired into the pipeline: delta-MI per shared gene between two
           mi_all.csv files.

The worker reads the chunk count and the power settings (perms, subsample_cells)
from the manifest, so prep and worker can never disagree on the chunking. Every
MI call takes an explicit random_state: the kNN estimator breaks ties at random,
and without a seed both the observed MI and the null drift between runs.

Example:
  python mi_pseudotime.py prep --h5ad <rooted.h5ad> --pt-key palantir_pt_rtk \\
      --layer log1p --min-counts 10 --chunks 200 --outdir results/mi/<run>
  python mi_pseudotime.py worker --outdir results/mi/<run> --chunk $SLURM_ARRAY_TASK_ID
  python mi_pseudotime.py merge --outdir results/mi/<run>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd


# Shared helpers ----
def empirical_p(obs: float, null: np.ndarray) -> float:
    """One-sided empirical p, P(null >= observed), with +1/+1 continuity so that no
    gene gets p = 0 (which would break BH). With n permutations, min p = 1/(n+1)."""
    return (float((null >= obs).sum()) + 1.0) / (len(null) + 1.0)


def chunk_bounds(n_items: int, n_chunks: int, chunk_id: int) -> tuple[int, int]:
    """Half-open [start, end) for chunk_id, splitting n_items into n_chunks even pieces.
    prep writes n_chunks to the manifest and worker reads it back, so both sides
    compute the same bounds."""
    if not 0 <= chunk_id < n_chunks:
        raise ValueError(f"chunk_id {chunk_id} out of range [0, {n_chunks})")
    edges = np.linspace(0, n_items, n_chunks + 1, dtype=int)
    return int(edges[chunk_id]), int(edges[chunk_id + 1])


def load_manifest(outdir: pathlib.Path) -> dict:
    mpath = outdir / "manifest.json"
    if not mpath.exists():
        raise SystemExit(f"no manifest.json in {outdir} — run `prep` first")
    return json.loads(mpath.read_text())


# prep ----
def cmd_prep(args: argparse.Namespace) -> None:
    import scanpy as sc

    out = args.outdir.resolve()
    (out).mkdir(parents=True, exist_ok=True)

    print(f"[prep] reading {args.h5ad}", flush=True)
    adata = sc.read_h5ad(args.h5ad)

    # A NaN pseudotime would silently poison every MI value, so it is an error here.
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

    # Detection filter: keep genes with count > 0 in at least --min-detect-frac of
    # cells, on the counts layer. This is the methods' "expressed in at least 1% of
    # cells" criterion.
    if args.min_detect_frac > 0:
        import scipy.sparse as _sp
        C = adata.layers["counts"] if "counts" in adata.layers else adata.X
        det = np.asarray((C > 0).mean(axis=0)).ravel()
        keep = det >= args.min_detect_frac
        print(f"[prep] detection filter >= {args.min_detect_frac:.2%} of cells: "
              f"{int(keep.sum()):,}/{adata.n_vars:,} genes kept", flush=True)
        adata = adata[:, keep].copy()

    sc.pp.filter_genes(adata, min_counts=args.min_counts)
    print(f"[prep] {adata.n_vars:,} genes remain after min_counts={args.min_counts}", flush=True)

    # Some source objects carry only a 'counts' layer. In that
    # case log1p(CP10K) is derived here, and the manifest records that it was.
    layer_used = args.layer
    if args.layer not in adata.layers:
        if "counts" in adata.layers:
            print(f"[prep] layer {args.layer!r} absent; deriving log1p(CP10K) "
                  f"from 'counts' (recorded in manifest)", flush=True)
            import scipy.sparse as sp
            C = adata.layers["counts"]
            C = C.tocsr().astype(np.float64) if sp.issparse(C) else np.asarray(C, np.float64)
            rs = np.asarray(C.sum(axis=1)).ravel()
            rs[rs == 0] = 1.0  # a cell with zero counts stays all-zero instead of NaN
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

    # Write matrices and manifest ----
    np.savez_compressed(out / "expr.npz", X=X)
    np.save(out / "pseudotime.npy", pt)
    np.save(out / "genes.npy", adata.var_names.to_numpy())

    # perms and subsample_cells are recorded as provenance and become the workers'
    # defaults, so a run's power settings are never left implicit.
    manifest = dict(
        pt_key=args.pt_key, layer=layer_used, min_counts=int(args.min_counts),
        min_detect_frac=float(args.min_detect_frac),
        n_cells=int(adata.n_obs), n_genes=int(adata.n_vars),
        n_chunks=int(args.chunks), seed=int(args.seed), h5ad=str(args.h5ad),
        perms=int(args.perms), subsample_cells=int(args.subsample_cells),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[prep] wrote matrices + manifest to {out}  "
          f"({adata.n_obs:,} cells x {adata.n_vars:,} genes, {args.chunks} chunks)", flush=True)


# worker ----
def cmd_worker(args: argparse.Namespace) -> None:
    from sklearn.feature_selection import mutual_info_regression

    out = args.outdir.resolve()
    man = load_manifest(out)
    n_chunks = man["n_chunks"]

    # Power settings: CLI override, else the manifest, else the defaults that
    # older manifests (which lack these keys) were run with.
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

    # One RNG per chunk, derived from the prep seed (stride 100_000 leaves room
    # for the chunk index), so the whole run is reproducible. The estimator's own
    # seed is the same for observed and null so the two are comparable.
    rng = np.random.default_rng(man["seed"] * 100_000 + args.chunk)
    mi_seed = man["seed"]
    n_null = args.perms - 1
    null_cols = [f"mi_null_{i}" for i in range(1, args.perms)]

    # Cost control ----
    # (a) Subsample cells for the estimator: kNN MI on n cells is O(n log n) per
    #     call and stabilizes well below the full 33k cells. One subsample per
    #     worker keeps observed and null on identical cells.
    if args.subsample_cells and X.shape[0] > args.subsample_cells:
        sub = np.sort(rng.choice(X.shape[0], size=args.subsample_cells, replace=False))
        X, y = X[sub], y[sub]
        print(f"[worker {args.chunk:03d}] subsampled to {len(sub):,} cells "
              f"for the estimator", flush=True)

    # (b) Permutations shared across the chunk's genes: the shuffled pseudotime
    #     vectors are drawn once and reused for every gene. Each gene's empirical
    #     p is unchanged in expectation (it still sees perms-1 independent shuffles
    #     of y); sharing introduces dependence between genes' p-values, and BH is
    #     applied to them as is.
    Y_null = np.empty((n_null, y.shape[0]), dtype=y.dtype)
    y_shuf = y.copy()
    for p in range(n_null):
        rng.shuffle(y_shuf)
        Y_null[p] = y_shuf

    # MI per gene ----
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


# merge ----
def cmd_merge(args: argparse.Namespace) -> None:
    import statsmodels.stats.multitest as smm

    out = args.outdir.resolve()
    man = load_manifest(out)
    files = sorted((out / "results").glob("mi_*.csv"))
    if not files:
        raise SystemExit(f"[merge] no per-chunk CSVs in {out/'results'}")

    # A partial merge is allowed but announced: BH is then computed over fewer genes.
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


# diff ----
def cmd_diff(args: argparse.Namespace) -> None:
    """Differential MI between two merged runs (A vs B) over their shared genes.

    Writes mi_a, mi_b, delta_mi = mi_a - mi_b and each run's BH q, sorted by
    |delta_mi|. This is a descriptive comparison, not a test on the delta: a
    permutation test would need the per-gene null distributions, which merge
    drops (worker --keep-null writes them, but no merge keeps them).
    """
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


# Command line ----
def build_parser() -> argparse.ArgumentParser:
    P = argparse.ArgumentParser(
        description="MI(expression, pseudotime): prep / worker / merge / diff",
    )
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
