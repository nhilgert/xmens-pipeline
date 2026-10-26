"""98_clustree: cluster-membership stability across the Leiden ladder of one lineage latent.

Reference-free resolution diagnostics, computed from obs only (backed read):
  * mean_stab            -- per resolution, mean over clusters of the best Jaccard against the
                            next-coarser and next-finer resolution
  * frac_unstable        -- fraction of clusters with best-Jaccard < 0.5
  * frac_clusters_minor_ge10pct -- fraction of clusters drawing >=10% of cells from a second
                            parent at the next-coarser resolution (clustree's in_prop edges)
  * churn_elbow          -- resolution with the largest single-step jump in that fraction (res >= 1.0)
Writes one TSV (all resolutions) and a one-row summary TSV. No z-scores.
"""
import sys, numpy as np, pandas as pd, anndata as ad
snk = snakemake  # noqa: F821
sys.stderr = sys.stdout = open(snk.log[0], "w")

a = ad.read_h5ad(snk.input.h5ad, backed="r")
obs = a.obs
rc = sorted([c for c in obs.columns if c.startswith("leiden_scVI_")], key=lambda c: float(c.split("_")[-1]))
res = [float(c.split("_")[-1]) for c in rc]
labs = {c: obs[c].astype(str).values for c in rc}
print(f"{snk.wildcards.species}_{snk.wildcards.lineage}: {a.n_obs:,} cells, {len(rc)} resolutions", flush=True)

rows = []
for i, c in enumerate(rc):
    lab = labs[c]; cl = np.unique(lab)
    neigh = [rc[j] for j in (i - 1, i + 1) if 0 <= j < len(rc)]
    stabs = []
    for x in cl:
        m = lab == x
        js = []
        for s in neigh:
            ls = labs[s]
            js.append(max((m & (ls == y)).sum() / (m | (ls == y)).sum() for y in np.unique(ls[m])))
        stabs.append(np.mean(js))
    stabs = np.array(stabs)
    row = dict(res=res[i], k=len(cl), mean_stab=float(stabs.mean()), frac_unstable=float((stabs < 0.5).mean()))
    if i > 0:
        parent = labs[rc[i - 1]]
        minor = []
        for x in cl:
            m = lab == x
            vc = pd.Series(parent[m]).value_counts(normalize=True)
            minor.append(1 - vc.iloc[0])
        minor = np.array(minor)
        row.update(frac_clusters_minor_ge10pct=float((minor >= 0.10).mean()),
                   frac_cells_from_minor_parent=float(np.average(minor, weights=[(lab == x).sum() for x in cl])))
    rows.append(row)
tab = pd.DataFrame(rows)
tab.to_csv(snk.output.tsv, sep="\t", index=False)

d = tab.dropna(subset=["frac_clusters_minor_ge10pct"]); d = d[d.res >= 1.0].reset_index(drop=True)
step = d.frac_clusters_minor_ge10pct.diff()
elbow = float(d.res.iloc[int(step.idxmax())]) if len(d) > 1 else np.nan
cur = snk.params.res
cur_f = float(str(cur).split("_")[-1])
summ = pd.DataFrame([dict(object=f"{snk.wildcards.species}_{snk.wildcards.lineage}", n_cells=a.n_obs,
                          production_res=cur_f, k_at_production=int(tab.loc[tab.res == cur_f, "k"].iloc[0]) if (tab.res == cur_f).any() else np.nan,
                          churn_elbow_res=elbow, production_below_elbow=bool(cur_f < elbow) if pd.notna(elbow) else None)])
summ.to_csv(snk.output.summary, sep="\t", index=False)
print(summ.to_string(index=False), flush=True)
