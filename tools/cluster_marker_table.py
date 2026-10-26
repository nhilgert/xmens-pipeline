"""Per-cluster marker detection for every annotated cluster: the table the `note`
column of resources/annotations/*.csv was written from.

Reads data/interim/annotated_{species}.h5ad (counts, QC columns, leiden_scVI_2.5),
data/interim/latent_{species}_{lineage}.h5ad (leiden_scVI_3.0 per lineage) and the
committed annotation CSVs. Writes, under the output directory, per_cluster_markers.csv
(fraction of nuclei with each panel gene detected, mean log1p CP10K), per_cluster_qc.csv
(median QC per cluster and per lineage) and top_markers.json (Wilcoxon, cluster vs the
rest of its lineage). Not part of the Snakemake DAG; run by hand from the repo root:

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
MIN_FOLD_CHANGE = 1.0         # log2, cluster vs rest of lineage
MIN_FRACTION_IN_CLUSTER = 0.25
# Pseudogenes, mitochondrial/ribosomal genes, antisense RNAs and sex-linked lncRNAs:
# never informative as markers, and they dominate rank lists by expression level alone.
UNINFORMATIVE = re.compile(
    r"^(Gm\d+|.*Rik\d*$|ENS|LINC|A[CLP]\d{6}|mt-|MT-|Rp[sl]\d|RP[SL]\d"
    r"|.*-AS\d$|Malat1|MALAT1|Xist|Tsix)"
)
QC_COLUMNS = ["doubletdetection_doublet_score", "n_genes_by_counts", "total_counts",
              "pct_counts_mt"]

# The genes read off the dot plots when the clusters were labelled: canonical lineage
# markers plus the fine-type markers named in the Methods.
PANEL = {
    "mouse": [
        "Ptprc", "Pdgfra", "Col1a1", "Col3a1", "Dcn", "Lum", "Prl8a2", "Ptgs2", "Bmp2", "Wnt4",
        "Hand2", "Cdh1", "Pax8", "Epcam", "Wnt7a", "Lgr5", "Muc1", "Foxa2", "Prss28", "Prss29",
        "Ihh", "Lif", "Foxj1", "Pecam1", "Flt1", "Cldn5", "Cdh5", "Pdgfrb", "Rgs5", "Acta2",
        "Myh11", "Tagln", "Des", "Cnn1", "Msln", "Upk3b", "Wt1", "Adgre1", "Csf1r", "C1qa",
        "C1qb", "Mrc1", "Lyve1", "Itgax", "Ly6c2", "Ccr2", "Plac8", "Flt3", "Xcr1", "H2-Ab1",
        "Cd79a", "Ms4a1", "Cd19", "Cd3e", "Cd3d", "Cd4", "Cd8a", "Foxp3", "Trdc", "Il7r", "Ncr1",
        "Klrb1c", "Eomes", "Itga1", "Gzma", "Gzmb", "Gzmc", "Prf1", "Nkg7", "Cpa3", "Kit", "Mki67",
        "Top2a", "Sox6", "Cdkn1a", "Bax",
    ],
    "human": [
        "PTPRC", "PDGFRA", "DCN", "LUM", "COL1A1", "PRL", "IGFBP1", "LEFTY2", "FOXO1", "CDH1",
        "PAX8", "EPCAM", "WNT7A", "LGR5", "MUC1", "FOXA2", "LIF", "SCGB1D2", "PAEP", "SPP1",
        "FOXJ1", "PECAM1", "FLT1", "CLDN5", "CDH5", "VWF", "PROX1", "LYVE1", "CCL21", "PDPN",
        "PDGFRB", "RGS5", "NOTCH3", "MCAM", "ACTA2", "MYH11", "TAGLN", "DES", "CNN1", "CD68",
        "C1QA", "MRC1", "CD14", "FCGR3A", "CSF1R", "CD79A", "MS4A1", "CD19", "CD3E", "CD3D", "CD4",
        "CD8A", "FOXP3", "CTLA4", "IL2RA", "TRDC", "IL7R", "KIT", "NCAM1", "NCR1", "KLRD1", "NKG7",
        "GZMA", "GZMB", "PRF1", "ITGAX", "ITGA1", "CD9", "EOMES", "CPA3", "TPSAB1", "MKI67", "TOP2A",
    ],
}

spec = yaml.safe_load(open(f"{REPO}/resources/lineage_spec.yaml"))
marker_rows, qc_rows, top_markers = [], [], {}


def log_normalise(counts):
    """CP10K log1p from raw counts; both matrices returned as CSR."""
    counts = sps.csr_matrix(counts, dtype=np.float32)
    library = np.asarray(counts.sum(1)).ravel()
    norm = sps.diags(1e4 / np.maximum(library, 1)) @ counts
    norm.data = np.log1p(norm.data)
    return counts, sps.csr_matrix(norm)


def cluster_sort_key(cluster):
    return int(cluster) if cluster.isdigit() else cluster


def summarise(species, level, clusters, labels, counts, norm, var_names, obs):
    """One row per (cluster, panel gene), plus QC medians and top markers, for one level.

    `clusters` is the cluster id per nucleus, NaN for nuclei outside the lineage.
    """
    var_index = {gene: i for i, gene in enumerate(var_names)}
    genes = [gene for gene in PANEL[species] if gene in var_index]
    columns = [var_index[gene] for gene in genes]
    counts_panel, norm_panel = counts[:, columns], norm[:, columns]
    in_lineage = clusters.notna().values

    for cluster in sorted(set(clusters[in_lineage].astype(str)), key=cluster_sort_key):
        mask = (clusters.astype(str) == cluster).values & in_lineage
        detected = np.asarray((counts_panel[mask] > 0).mean(0)).ravel()
        mean_expr = np.asarray(norm_panel[mask].mean(0)).ravel()
        label = labels.get(cluster, "?")
        marker_rows.extend(
            (species, level, cluster, label, int(mask.sum()), gene, float(f), float(m))
            for gene, f, m in zip(genes, detected, mean_expr)
        )
        medians = {c: float(obs.loc[mask, c].median()) for c in QC_COLUMNS if c in obs.columns}
        qc_rows.append(dict(species=species, level=level, cluster=cluster, label=label,
                            n=int(mask.sum()), **medians))

    # Lineage-wide medians, the reference the per-cluster QC values are read against.
    medians = {c: float(obs.loc[in_lineage, c].median()) for c in QC_COLUMNS if c in obs.columns}
    qc_rows.append(dict(species=species, level=level, cluster="ALL", label="",
                        n=int(in_lineage.sum()), **medians))

    lineage = ad.AnnData(X=norm[in_lineage],
                         obs=pd.DataFrame({"cluster": clusters[in_lineage].astype(str).values}),
                         var=pd.DataFrame(index=var_names))
    detected = np.asarray((lineage.X > 0).mean(0)).ravel()
    keep = (detected >= MIN_DETECTED) & ~var_names.str.match(UNINFORMATIVE)
    lineage = lineage[:, keep].copy()
    sc.tl.rank_genes_groups(lineage, "cluster", method="wilcoxon", use_raw=False, pts=True)

    top_markers[f"{species}|{level}"] = {}
    for cluster in sorted(lineage.obs.cluster.unique(), key=cluster_sort_key):
        ranked = sc.get.rank_genes_groups_df(lineage, group=cluster)
        ranked = ranked[(ranked.logfoldchanges > MIN_FOLD_CHANGE)
                        & (ranked.pct_nz_group >= MIN_FRACTION_IN_CLUSTER)]
        ranked = ranked.sort_values("scores", ascending=False).head(N_TOP)
        top_markers[f"{species}|{level}"][cluster] = [
            f"{row.names} {row.pct_nz_group:.2f}" for row in ranked.itertuples()
        ]


def read_labels(species, level):
    table = pd.read_csv(f"{REPO}/resources/annotations/{species}_{level}.csv", dtype=str)
    return table.set_index("cluster")["label"]


for species, species_spec in spec["species"].items():
    atlas = ad.read_h5ad(f"{REPO}/data/interim/annotated_{species}.h5ad")
    counts, norm = log_normalise(atlas.layers["counts"])
    atlas_resolution = spec["atlas"][species]["res"]
    summarise(species, "lineage", atlas.obs[atlas_resolution].astype(str),
              read_labels(species, "lineage"), counts, norm, atlas.var_names, atlas.obs)

    for lineage, lineage_spec in species_spec["lineages"].items():
        resolution = lineage_spec.get("res", spec["defaults"]["res"])
        latent = ad.read_h5ad(f"{REPO}/data/interim/latent_{species}_{lineage}.h5ad", backed="r")
        clusters = latent.obs[resolution].astype(str)
        latent.file.close()

        # The lineage object is a subset of the atlas, so the join is by barcode.
        clusters = clusters.reindex(atlas.obs_names)
        if clusters.notna().sum() != len(latent.obs):
            raise ValueError(f"{species} {lineage}: {len(latent.obs)} nuclei in the lineage "
                             f"object, {clusters.notna().sum()} in the atlas")
        summarise(species, lineage, clusters, read_labels(species, lineage),
                  counts, norm, atlas.var_names, atlas.obs)

    del atlas, counts, norm

pd.DataFrame(marker_rows, columns=["species", "level", "cluster", "label", "n", "gene",
                                   "frac_detected", "mean_log1p_cp10k"]
             ).to_csv(f"{OUT}/per_cluster_markers.csv", index=False)
pd.DataFrame(qc_rows).to_csv(f"{OUT}/per_cluster_qc.csv", index=False)
json.dump(top_markers, open(f"{OUT}/top_markers.json", "w"), indent=1)
