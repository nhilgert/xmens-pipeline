"""One-vs-rest Wilcoxon marker genes within one lineage (stage 05).

Called by rules `de_lineage_clusters` and `de_lineage_subtypes` (05_de.smk). Reads the
lineage h5ad from snakemake.input.h5ad and writes the top-N genes per group to
snakemake.output.tsv. With params.level == "clusters" the groups are the Leiden
clusters at the lineage's spec resolution (params.res), the over-clustered evidence
behind the panel-recovery statement. With "subtypes" the groups are cell_subtype on
the reannotated object with excluded cells dropped first (the supplementary table).
The test runs on log1p CP10K of raw counts over the full gene set.
"""
import sys, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as ssp
snk = snakemake  # noqa: F821
sys.stderr = sys.stdout = open(snk.log[0], "w")

# Load inputs ----
a = sc.read_h5ad(snk.input.h5ad)
level, top_n = snk.params.level, int(snk.params.top_n)

if level == "clusters":
    grp_col = snk.params.res
elif level == "subtypes":
    grp_col = "cell_subtype"
    if "excluded" in a.obs.columns:
        a = a[~a.obs["excluded"].astype(bool)].copy()
else:
    raise ValueError(level)
if grp_col not in a.obs.columns:
    raise KeyError(f"{grp_col} not in obs; have {list(a.obs.columns)[:20]}...")

# Unlabelled cells would otherwise form a spurious "nan" group.
a.obs[grp_col] = a.obs[grp_col].astype(str)
a = a[~a.obs[grp_col].str.lower().isin(["nan", "", "none"])].copy()
print(
    f"{snk.wildcards.species}_{snk.wildcards.lineage} level={level} groups={grp_col}: "
    f"{a.n_obs:,} cells, "
    f"{a.obs[grp_col].nunique()} groups",
    flush=True,
)

# Rank genes ----
# Raw counts come from layers["counts"] when present, else .raw, else .X; the test
# object is a fresh AnnData holding only the group column.
X = a.layers["counts"] if "counts" in a.layers else (a.raw.X if a.raw is not None else a.X)
vn = a.var_names if ("counts" in a.layers or a.raw is None) else a.raw.var_names
b = sc.AnnData(ssp.csr_matrix(X), obs=a.obs[[grp_col]].copy(), var=pd.DataFrame(index=vn))

sc.pp.normalize_total(b, target_sum=1e4)  # CP10K
sc.pp.log1p(b)

b.obs[grp_col] = b.obs[grp_col].astype("category")
sc.tl.rank_genes_groups(b, grp_col, method="wilcoxon", n_genes=b.n_vars, pts=True)

# Write the top-N table ----
frames = []
for g in b.obs[grp_col].cat.categories:
    d = sc.get.rank_genes_groups_df(b, g)
    d.insert(0, "group", g)
    d.insert(1, "rank", np.arange(1, len(d) + 1))
    frames.append(d.head(top_n))

out = (pd.concat(frames)
         .rename(columns={
             "names": "gene",
             "scores": "wilcoxon_score",
             "logfoldchanges": "log2FC",
             "pvals": "p",
             "pvals_adj": "p_adj_BH",
             "pct_nz_group": "frac_detected_in_group",
             "pct_nz_reference": "frac_detected_in_rest",
         }))
out.to_csv(snk.output.tsv, sep="\t", index=False, float_format="%.5g")
print(f"wrote {len(out):,} rows -> {snk.output.tsv}", flush=True)
