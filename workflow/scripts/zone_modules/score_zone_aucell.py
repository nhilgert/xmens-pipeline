"""Score the 11 decidual-zone modules of Li et al. (2022) per nucleus with pySCENIC AUCell.

Called by rule zone_aucell (workflow/rules/16_zone_modules.smk). Reads the mouse release
object (--obj; raw counts in layers['counts'] and the obs columns timepoint and cell_subtype),
its release MANIFEST.json (--manifest) and the module gene sets (--genesets, YAML). Writes to
--out: zone_aucell.csv (one auc_<MODULE> column per gene set, scored nuclei only),
subtype_means.csv (mean score per cell_subtype) and provenance.json. Existing outputs are
never overwritten.

Scoring follows the pySCENIC recipe (create_rankings + aucell4r): every gene of the object is
ranked within each nucleus, gene sets are unweighted (GeneSignature, weight 1 each), and the
AUC window is a fixed fraction of all genes (--auc-threshold) with a fixed seed. Two shortcuts
are exact rather than approximate, because AUCell ranks each nucleus on its own and the seed's
gene-column shuffle depends only on the seed and on the (never reduced) number of genes:

  * only the nuclei of --timepoints are scored (default: all nuclei);
  * nuclei are scored in chunks of --chunk. As a check the run re-scores the first
    <= 2,000 nuclei as their own chunk and asserts identical scores.

Filtering genes on the subset, or deriving the threshold on it, would change the window, so
neither is done. Raw counts and CP10k give identical rankings (CP10k multiplies a nucleus by
one positive constant, ties included), so --normalization cp10k only reproduces the earlier
input and --normalization none is the default, as for the SCENIC regulon scores.
"""
import argparse, datetime, hashlib, json, os, sys, time
import numpy as np, pandas as pd, h5py, yaml, scipy, scipy.sparse as sp
from anndata.io import read_elem
import pyscenic, ctxcore
from ctxcore.genesig import GeneSignature
from pyscenic.aucell import create_rankings, aucell4r

OUTPUTS = ("zone_aucell.csv", "subtype_means.csv", "provenance.json")


def md5(path):
    h = hashlib.md5()
    f = open(path, "rb")

    for b in iter(lambda: f.read(1 << 20), b""):  # 1 MiB blocks
        h.update(b)

    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    f = open(path, "rb")

    for b in iter(lambda: f.read(1 << 24), b""):  # 16 MiB blocks
        h.update(b)

    return h.hexdigest()


def main():
    # Arguments ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--genesets", required=True)
    ap.add_argument("--out", required=True)
    # Window as a fraction of ALL genes of the object; 0.05 is the pySCENIC default.
    ap.add_argument("--auc-threshold", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timepoints", default="")  # comma list of obs['timepoint']; "" = all nuclei
    ap.add_argument("--normalization", default="none", choices=["none", "cp10k"])
    ap.add_argument("--chunk", type=int, default=8000)  # nuclei per chunk; bounds the dense matrix
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest", default="")  # release MANIFEST.json holding the object's sha256
    ap.add_argument("--object-name", default="")  # key under MANIFEST['objects']
    ap.add_argument("--expect-sha256", default="")
    a = ap.parse_args()

    t0 = time.time()

    clash = [f for f in OUTPUTS if os.path.exists(os.path.join(a.out, f))]
    if clash:
        sys.exit(f"refusing to overwrite {clash} in {a.out}")

    os.makedirs(a.out, exist_ok=True)

    # Check the object against the release manifest ----
    # The release object is an ancient() input of the rule, so this hash is the only guard
    # against scoring a re-cut object under the old name.
    sha = sha256(a.obj)
    expect = a.expect_sha256

    if a.manifest:
        man = json.load(open(a.manifest))
        mexp = man["objects"][a.object_name]["sha256"]
        assert not expect or expect == mexp, "--expect-sha256 disagrees with the manifest"
        expect = mexp

    if expect:
        assert sha == expect, f"object sha256 {sha} != expected {expect}"

    G = yaml.safe_load(open(a.genesets))
    tps = [t.strip() for t in a.timepoints.split(",") if t.strip()]

    # Load counts and obs ----
    # Read straight from the h5 groups rather than anndata so only obs columns and the raw
    # counts layer are loaded, not X and the full obs.
    with h5py.File(a.obj, "r") as h:
        obs_names = pd.Index(np.asarray(read_elem(h["obs/_index"])).astype(str))
        var_names = pd.Index(np.asarray(read_elem(h["var/_index"])).astype(str))
        obs = pd.DataFrame(
            {
                c: np.asarray(read_elem(h["obs/" + c])).astype(str)
                for c in ["timepoint", "cell_subtype"]
            },
            index=obs_names,
        )
        g = h["layers/counts"]
        n_obs, n_vars = (int(x) for x in g.attrs["shape"])
        X = sp.csr_matrix(
            (g["data"][()], g["indices"][()], g["indptr"][()]),
            shape=(n_obs, n_vars),
        )

    assert (
        len(obs_names) == n_obs
        and len(var_names) == n_vars
        and var_names.is_unique
        and obs_names.is_unique
    )

    # Restrict to the requested timepoints; genes are never subset (see the module docstring).
    if tps:
        have = set(obs["timepoint"])
        assert set(tps) <= have, (
            f"timepoints {sorted(set(tps) - have)} not in object (have {sorted(have)})"
        )
        keep = np.flatnonzero(obs["timepoint"].isin(tps).values)
    else:
        keep = np.arange(n_obs)

    X = X[keep]
    names = obs_names[keep]
    obs = obs.iloc[keep]

    print(f"[{time.time()-t0:.0f}s] read {n_obs} x {n_vars}; scoring {len(keep)} nuclei"
          + (f" of timepoints {tps}" if tps else " (all)") + f", nnz {X.nnz}", flush=True)

    # Normalization ----
    rs = np.asarray(X.sum(1)).ravel().astype(np.float64)
    assert (rs > 0).all()

    # CP10k gives the same rankings as raw counts; kept only to reproduce the earlier input.
    if a.normalization == "cp10k":
        X = (sp.diags((1e4 / rs).astype(np.float32)) @ X.astype(np.float32)).tocsr()
    else:
        X = X.astype(np.float32).tocsr()

    # Gene sets ----
    # Genes absent from the object are dropped from the signature and listed in the provenance.
    present = {m: [x for x in G[m] if x in var_names] for m in G}
    missing = {m: [x for x in G[m] if x not in var_names] for m in G}
    sigs = [GeneSignature(name=m, gene2weight=present[m]) for m in G]

    # Score in chunks of nuclei ----
    parts, check = [], None

    for i in range(0, len(names), a.chunk):
        j = min(i + a.chunk, len(names))
        ex = pd.DataFrame(X[i:j].toarray(), index=names[i:j], columns=var_names)
        rnk = create_rankings(ex, a.seed)
        parts.append(aucell4r(rnk, sigs, a.auc_threshold, False, False, a.workers))

        # Chunking check: the first <= 2,000 nuclei scored as their own chunk must match exactly.
        if i == 0:
            k = min(2000, j)
            a2 = aucell4r(
                create_rankings(ex.iloc[:k], a.seed),
                sigs,
                a.auc_threshold,
                False,
                False,
                a.workers,
            )
            d = float(np.abs(a2.values - parts[0].iloc[:k].values).max())
            check = {"rows": k, "max_abs_diff": d}
            assert d == 0.0, f"chunking changed scores (max diff {d})"

        del ex, rnk
        print(f"[{time.time()-t0:.0f}s] scored nuclei {i}-{j}", flush=True)

    # Write scores and subtype means ----
    A = pd.concat(parts).reindex(names)
    A.columns = [f"auc_{c}" for c in A.columns]
    A = A[[f"auc_{m}" for m in G]]  # column order of the gene-set file
    assert A.notna().all().all() and len(A) == len(names)
    A.index.name = "barcode"
    A.to_csv(os.path.join(a.out, "zone_aucell.csv"))

    M = A.groupby(obs["cell_subtype"].values).mean()
    M.insert(0, "n", obs["cell_subtype"].value_counts().reindex(M.index).values)
    M.index.name = "cell_subtype"
    M.to_csv(os.path.join(a.out, "subtype_means.csv"))

    # Provenance ----
    prov = {
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": os.path.abspath(__file__),
        "script_md5": md5(os.path.abspath(__file__)),
        "obj": {
            "path": os.path.abspath(a.obj),
            "bytes": os.path.getsize(a.obj),
            "sha256": sha,
            "sha256_checked_against": (
                os.path.abspath(a.manifest)
                if a.manifest
                else ("--expect-sha256" if expect else None)
            ),
            "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(a.obj)).isoformat(
                timespec="seconds"
            ),
        },
        "genesets": {
            "path": os.path.abspath(a.genesets),
            "md5": md5(a.genesets),
            "n_listed": {m: len(G[m]) for m in G},
        },
        "n_obs_object": n_obs,
        "n_vars": n_vars,
        "n_scored": int(len(names)),
        "timepoints": tps or "all",
        "n_scored_per_timepoint": {
            k: int(v) for k, v in obs["timepoint"].value_counts().sort_index().items()
        },
        "genes_present": {m: len(present[m]) for m in G},
        "genes_missing": {m: missing[m] for m in G if missing[m]},
        "normalization": (
            "none (raw layers['counts'])"
            if a.normalization == "none"
            else "CP10k from layers['counts'] (no log)"
        ),
        "auc_threshold": a.auc_threshold,
        "rank_cutoff": int(round(a.auc_threshold * n_vars)),
        "seed": a.seed,
        "chunk": a.chunk,
        "workers": a.workers,
        "chunking_check": check,
        "versions": {
            "pyscenic": pyscenic.__version__,
            "ctxcore": getattr(ctxcore, "__version__", "?"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "python": sys.version.split()[0],
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(prov, open(os.path.join(a.out, "provenance.json"), "w"), indent=1)

    print(
        json.dumps(
            {
                k: prov[k]
                for k in (
                    "n_scored",
                    "timepoints",
                    "genes_missing",
                    "auc_threshold",
                    "rank_cutoff",
                    "seed",
                    "normalization",
                )
            }
        ),
        flush=True,
    )
    print(f"[{time.time()-t0:.0f}s] done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
