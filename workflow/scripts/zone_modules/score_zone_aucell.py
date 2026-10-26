#!/usr/bin/env python
"""AUCell scores for the 11 decidual-zone modules (Li et al. 2022, Table 1) on the mouse release object.

Pipeline version (rule zone_aucell, workflow/rules/16_zone_modules.smk), 2026-10-24. Derived from
results/zone_aucell/2026_10_22_auc005_seed0/score_zone_aucell.py (md5 93b46dc754c5620326c15c8c912fc47f), which made
the 2026-10-22 score table of record. Kept from it: pyscenic create_rankings + aucell4r, all genes of the object
ranked, unweighted gene sets (GeneSignature, weight 1 each), a fixed auc_threshold and seed, chunking over nuclei.
Changed on 2026-10-24 at the user's request:
  * raw counts (layers['counts']) instead of CP10k, as for the SCENIC regulon scores (--normalization none).
    AUCell ranks genes within each nucleus and CP10k multiplies a nucleus by one positive constant, so rankings,
    ties included, and therefore scores are unchanged; --normalization cp10k reproduces the old input exactly.
  * only the nuclei of --timepoints are scored (default: all). This is exact, not an approximation: each nucleus is
    ranked on its own, and the seed's gene-column shuffle depends only on the seed and the number of genes, which
    is never reduced (all n_vars genes are ranked for every nucleus). Do NOT filter genes on the subset or derive
    the threshold on it: either would change the window.
  * the object's sha256 is checked against the release MANIFEST.json (--manifest/--object-name) or --expect-sha256.

Chunking is exact for the same reason as the timepoint subset; the run re-scores the first <=2,000 nuclei as their
own chunk and asserts identical scores.

Outputs (in --out; existing files are never overwritten):
    zone_aucell.csv      barcode-indexed, one column auc_<MODULE> per gene set, the scored nuclei only
    subtype_means.csv    mean score per cell_subtype (scored nuclei)
    provenance.json      inputs (sha256/md5), parameters, versions, checks, timings
"""
import argparse, datetime, hashlib, json, os, sys, time
import numpy as np, pandas as pd, h5py, yaml, scipy, scipy.sparse as sp
from anndata.io import read_elem
import pyscenic, ctxcore
from ctxcore.genesig import GeneSignature
from pyscenic.aucell import create_rankings, aucell4r

OUTPUTS = ("zone_aucell.csv", "subtype_means.csv", "provenance.json")


def md5(path):
    h = hashlib.md5(); f = open(path, "rb")
    for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256(); f = open(path, "rb")
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--genesets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--auc-threshold", type=float, default=0.05)     # fraction of ALL genes (pySCENIC default 0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timepoints", default="")                      # comma list of obs['timepoint'] values; "" = all nuclei
    ap.add_argument("--normalization", default="none", choices=["none", "cp10k"])
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest", default="")                        # release MANIFEST.json holding the object's sha256
    ap.add_argument("--object-name", default="")                     # key under MANIFEST['objects']
    ap.add_argument("--expect-sha256", default="")
    a = ap.parse_args()
    t0 = time.time()
    clash = [f for f in OUTPUTS if os.path.exists(os.path.join(a.out, f))]
    if clash:
        sys.exit(f"refusing to overwrite {clash} in {a.out}")
    os.makedirs(a.out, exist_ok=True)

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

    with h5py.File(a.obj, "r") as h:
        obs_names = pd.Index(np.asarray(read_elem(h["obs/_index"])).astype(str))
        var_names = pd.Index(np.asarray(read_elem(h["var/_index"])).astype(str))
        obs = pd.DataFrame({c: np.asarray(read_elem(h["obs/" + c])).astype(str) for c in ["timepoint", "cell_subtype"]}, index=obs_names)
        g = h["layers/counts"]
        n_obs, n_vars = (int(x) for x in g.attrs["shape"])
        X = sp.csr_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=(n_obs, n_vars))
    assert len(obs_names) == n_obs and len(var_names) == n_vars and var_names.is_unique and obs_names.is_unique
    if tps:
        have = set(obs["timepoint"])
        assert set(tps) <= have, f"timepoints {sorted(set(tps) - have)} not in object (have {sorted(have)})"
        keep = np.flatnonzero(obs["timepoint"].isin(tps).values)
    else:
        keep = np.arange(n_obs)
    X = X[keep]; names = obs_names[keep]; obs = obs.iloc[keep]
    print(f"[{time.time()-t0:.0f}s] read {n_obs} x {n_vars}; scoring {len(keep)} nuclei"
          + (f" of timepoints {tps}" if tps else " (all)") + f", nnz {X.nnz}", flush=True)

    rs = np.asarray(X.sum(1)).ravel().astype(np.float64)
    assert (rs > 0).all()
    if a.normalization == "cp10k":   # the 2026-10-22 input; same rankings as raw counts
        X = (sp.diags((1e4 / rs).astype(np.float32)) @ X.astype(np.float32)).tocsr()
    else:
        X = X.astype(np.float32).tocsr()

    present = {m: [x for x in G[m] if x in var_names] for m in G}
    missing = {m: [x for x in G[m] if x not in var_names] for m in G}
    sigs = [GeneSignature(name=m, gene2weight=present[m]) for m in G]
    parts, check = [], None
    for i in range(0, len(names), a.chunk):
        j = min(i + a.chunk, len(names))
        ex = pd.DataFrame(X[i:j].toarray(), index=names[i:j], columns=var_names)
        rnk = create_rankings(ex, a.seed)
        parts.append(aucell4r(rnk, sigs, a.auc_threshold, False, False, a.workers))
        if i == 0:   # chunking check: the first <=2,000 nuclei scored as their own chunk
            k = min(2000, j)
            a2 = aucell4r(create_rankings(ex.iloc[:k], a.seed), sigs, a.auc_threshold, False, False, a.workers)
            d = float(np.abs(a2.values - parts[0].iloc[:k].values).max())
            check = {"rows": k, "max_abs_diff": d}
            assert d == 0.0, f"chunking changed scores (max diff {d})"
        del ex, rnk
        print(f"[{time.time()-t0:.0f}s] scored nuclei {i}-{j}", flush=True)

    A = pd.concat(parts).reindex(names)
    A.columns = [f"auc_{c}" for c in A.columns]
    A = A[[f"auc_{m}" for m in G]]
    assert A.notna().all().all() and len(A) == len(names)
    A.index.name = "barcode"
    A.to_csv(os.path.join(a.out, "zone_aucell.csv"))
    M = A.groupby(obs["cell_subtype"].values).mean()
    M.insert(0, "n", obs["cell_subtype"].value_counts().reindex(M.index).values)
    M.index.name = "cell_subtype"
    M.to_csv(os.path.join(a.out, "subtype_means.csv"))

    prov = {
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": os.path.abspath(__file__), "script_md5": md5(os.path.abspath(__file__)),
        "obj": {"path": os.path.abspath(a.obj), "bytes": os.path.getsize(a.obj), "sha256": sha,
                "sha256_checked_against": (os.path.abspath(a.manifest) if a.manifest else ("--expect-sha256" if expect else None)),
                "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(a.obj)).isoformat(timespec="seconds")},
        "genesets": {"path": os.path.abspath(a.genesets), "md5": md5(a.genesets), "n_listed": {m: len(G[m]) for m in G}},
        "n_obs_object": n_obs, "n_vars": n_vars, "n_scored": int(len(names)),
        "timepoints": tps or "all",
        "n_scored_per_timepoint": {k: int(v) for k, v in obs["timepoint"].value_counts().sort_index().items()},
        "genes_present": {m: len(present[m]) for m in G}, "genes_missing": {m: missing[m] for m in G if missing[m]},
        "normalization": "none (raw layers['counts'])" if a.normalization == "none" else "CP10k from layers['counts'] (no log)",
        "auc_threshold": a.auc_threshold, "rank_cutoff": int(round(a.auc_threshold * n_vars)),
        "seed": a.seed, "chunk": a.chunk, "workers": a.workers, "chunking_check": check,
        "versions": {"pyscenic": pyscenic.__version__, "ctxcore": getattr(ctxcore, "__version__", "?"), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__, "python": sys.version.split()[0]},
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(prov, open(os.path.join(a.out, "provenance.json"), "w"), indent=1)
    print(json.dumps({k: prov[k] for k in ("n_scored", "timepoints", "genes_missing", "auc_threshold", "rank_cutoff", "seed", "normalization")}), flush=True)
    print(f"[{time.time()-t0:.0f}s] done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
