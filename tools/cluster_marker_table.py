"""Per-cluster marker evidence for every annotated cluster: the table the `note` column
of resources/annotations/*.csv is written from (tools/write_annotation_notes.py).

Reads resources/lineage_spec.yaml (resolution and marker panel per lineage), the atlas
marker panel it names under resources/markers/, the committed annotation CSVs and the
clustered objects data/interim/latent_{species}.h5ad (atlas level) and
data/interim/latent_{species}_{lineage}.h5ad. Writes to the output directory:

    per_cluster_markers.csv   cluster x panel gene: fraction detected, mean log1p CP10K
    per_cluster_qc.csv        per cluster: n, median doublet score and genes per nucleus,
                              fraction of nuclei positive for >= 2 lineage panels, fraction
                              positive for each lineage panel; one reference row per level
                              ("ALL": doublet score 75th percentile, genes 25th percentile)
    top_markers.json          Wilcoxon top markers, cluster vs the rest of its level

A nucleus is positive for a lineage panel when at least one of the panel's lineage-role
genes is detected; panels nested in another (decidual within fibroblast) count as one.
Not part of the Snakemake DAG; run from the repo root:

    python tools/cluster_marker_table.py <repo> <outdir>
"""
import json
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
import yaml

REPO, OUT = sys.argv[1:3]
N_TOP = 5
MIN_DETECTED = 0.02           # genes below this detection fraction are not ranked
MIN_FOLD_CHANGE = 1.0         # log2, cluster vs the rest of the level
MIN_FRACTION_IN_CLUSTER = 0.25
QC_COLUMNS = ["doubletdetection_doublet_score", "n_genes_by_counts", "total_counts"]
# Pseudogenes, mitochondrial/ribosomal genes, antisense RNAs and sex-linked lncRNAs:
# never informative as markers, and they dominate rank lists by expression level alone.
UNINFORMATIVE = re.compile(
    r"^(Gm\d+|.*Rik\d*$|ENS|LINC|A[CLP]\d{6}|mt-|MT-|Rp[sl]\d|RP[SL]\d"
    r"|.*-AS\d$|Malat1|MALAT1|Xist|Tsix)"
)

spec = yaml.safe_load(open(f"{REPO}/resources/lineage_spec.yaml"))
marker_rows, qc_rows, top_markers = [], [], {}


def lineage_panels(species):
    """Lineage name -> lineage-role genes, from the atlas panel TSV; nested panels merged."""
    path = f"{REPO}/resources/markers/{spec['atlas'][species]['panel']}"
    panel = pd.read_csv(path, sep="\t", comment="#")
    panel = panel[panel.role == "lineage"]
    genes = {}
    for row in panel.itertuples():
        name = row.nested_within if isinstance(row.nested_within, str) else row.celltype
        name = name.replace(" cells", "").replace(" ", "_")
        genes.setdefault(name, []).extend(row.genes.split(","))
    return genes


def panel_genes(species, level):
    """Every gene to tabulate for one level: the dot-plot panel plus the label markers."""
    if level == "lineage":
        groups = list(lineage_panels(species).values())
        label_markers = spec["atlas"][species]["label_markers"]
    else:
        lineage = spec["species"][species]["lineages"][level]
        groups = list(lineage["markers"].values())
        label_markers = lineage["label_markers"]
    genes = [g for group in groups for g in group]
    genes += [g for gs in label_markers.values() for g in gs]
    return list(dict.fromkeys(genes))


def log_normalise(counts):
    """CP10K log1p from raw counts; both matrices returned as CSR."""
    counts = sps.csr_matrix(counts, dtype=np.float32)
    library = np.asarray(counts.sum(1)).ravel()
    norm = sps.diags(1e4 / np.maximum(library, 1)) @ counts
    norm.data = np.log1p(norm.data)
    return counts, sps.csr_matrix(norm)


def cluster_sort_key(cluster):
    return int(cluster) if cluster.isdigit() else cluster


def summarise(species, level, adata, resolution):
    """Marker, QC and top-marker rows for every cluster of one clustered object."""
    labels = pd.read_csv(f"{REPO}/resources/annotations/{species}_{level}.csv",
                         dtype=str).set_index("cluster")["label"]
    clusters = adata.obs[resolution].astype(str)
    if set(clusters) != set(labels.index):
        raise ValueError(f"{species} {level}: clusters in the object {sorted(set(clusters))} "
                         f"differ from the table {sorted(labels.index)}")

    counts, norm = log_normalise(adata.layers["counts"])
    var_index = {gene: i for i, gene in enumerate(adata.var_names)}
    genes = [g for g in panel_genes(species, level) if g in var_index]
    columns = [var_index[g] for g in genes]
    counts_panel, norm_panel = counts[:, columns], norm[:, columns]

    # Lineage-panel positivity per nucleus, for the co-expression statistic.
    panels = {name: [var_index[g] for g in gs if g in var_index]
              for name, gs in lineage_panels(species).items()}
    positive = pd.DataFrame(
        {name: np.asarray((counts[:, cols] > 0).sum(1)).ravel() >= 1
         for name, cols in panels.items()},
        index=adata.obs_names)
    two_or_more = positive.sum(1) >= 2

    for cluster in sorted(set(clusters), key=cluster_sort_key):
        mask = (clusters == cluster).values
        detected = np.asarray((counts_panel[mask] > 0).mean(0)).ravel()
        mean_expr = np.asarray(norm_panel[mask].mean(0)).ravel()
        marker_rows.extend(
            (species, level, cluster, labels[cluster], int(mask.sum()), gene, float(f), float(m))
            for gene, f, m in zip(genes, detected, mean_expr)
        )
        qc_rows.append(dict(
            species=species, level=level, cluster=cluster, label=labels[cluster], n=int(mask.sum()),
            **{c: float(adata.obs.loc[mask, c].median()) for c in QC_COLUMNS},
            frac_coexpress_2plus=float(two_or_more[mask].mean()),
            **{f"frac_pos_{name}": float(positive.loc[mask, name].mean()) for name in panels},
        ))

    # The reference row: the quantiles an EXCLUDE call is read against.
    qc_rows.append(dict(
        species=species, level=level, cluster="ALL", label="", n=adata.n_obs,
        doubletdetection_doublet_score=float(adata.obs[QC_COLUMNS[0]].quantile(0.75)),
        n_genes_by_counts=float(adata.obs[QC_COLUMNS[1]].quantile(0.25)),
        total_counts=float(adata.obs[QC_COLUMNS[2]].median()),
        frac_coexpress_2plus=float(two_or_more.mean()),
        **{f"frac_pos_{name}": float(positive[name].mean()) for name in panels},
    ))

    ranked_input = ad.AnnData(X=norm, obs=pd.DataFrame({"cluster": clusters.values}),
                              var=pd.DataFrame(index=adata.var_names))
    detected = np.asarray((ranked_input.X > 0).mean(0)).ravel()
    keep = (detected >= MIN_DETECTED) & ~adata.var_names.str.match(UNINFORMATIVE)
    ranked_input = ranked_input[:, keep].copy()
    sc.tl.rank_genes_groups(ranked_input, "cluster", method="wilcoxon", use_raw=False, pts=True)
    top_markers[f"{species}|{level}"] = {}
    for cluster in sorted(set(clusters), key=cluster_sort_key):
        ranked = sc.get.rank_genes_groups_df(ranked_input, group=cluster)
        ranked = ranked[(ranked.logfoldchanges > MIN_FOLD_CHANGE)
                        & (ranked.pct_nz_group >= MIN_FRACTION_IN_CLUSTER)]
        ranked = ranked.sort_values("scores", ascending=False).head(N_TOP)
        top_markers[f"{species}|{level}"][cluster] = [
            f"{row.names} {row.pct_nz_group:.2f}" for row in ranked.itertuples()
        ]


for species, species_spec in spec["species"].items():
    atlas = ad.read_h5ad(f"{REPO}/data/interim/latent_{species}.h5ad")
    summarise(species, "lineage", atlas, spec["atlas"][species]["res"])
    del atlas
    for lineage, lineage_spec in species_spec["lineages"].items():
        resolution = lineage_spec.get("res", spec["defaults"]["res"])
        subset = ad.read_h5ad(f"{REPO}/data/interim/latent_{species}_{lineage}.h5ad")
        summarise(species, lineage, subset, resolution)
        del subset

pd.DataFrame(marker_rows, columns=["species", "level", "cluster", "label", "n", "gene",
                                   "frac_detected", "mean_log1p_cp10k"]
             ).to_csv(f"{OUT}/per_cluster_markers.csv", index=False)
pd.DataFrame(qc_rows).to_csv(f"{OUT}/per_cluster_qc.csv", index=False)
json.dump(top_markers, open(f"{OUT}/top_markers.json", "w"), indent=1)
