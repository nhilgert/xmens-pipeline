"""Write one evidence notebook per annotation table, under notebooks/annotation/.

Reads resources/lineage_spec.yaml and resources/annotations/*.csv; writes
notebooks/annotation/{species}_lineage.ipynb (stage 04, atlas clusters at the atlas
resolution) and notebooks/annotation/{species}_{lineage}.ipynb (stage 04c, one per
lineage). Each notebook shows the committed table, the marker dot plot by cluster, QC per
cluster, the UMAP by cluster and by label, and the contamination markers of the
EXCLUDE clusters; it computes nothing that assigns a label. Run from the repo root:

    python tools/make_annotation_notebooks.py          # write the notebooks
    python tools/make_annotation_notebooks.py --execute   # also execute them in place

Execution needs the clustered objects on disk (data/interim/latent_*.h5ad); every notebook
first checks that the clusters in the object are exactly the clusters in its table.
"""
import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import nbformat
import yaml

OUT_DIR = Path("notebooks/annotation")
SPEC = yaml.safe_load(open("resources/lineage_spec.yaml"))

# Fine-type markers read on the dot plots when the lineage tables were made, by label.
# They are shown next to the lineage panel from lineage_spec.yaml.
FINE_MARKERS = {
    "mouse": {
        "stromal": ["Prl8a2", "Bmp2", "Wnt4", "Hand2", "Ptgs2", "Mki67", "Top2a", "Cdkn1a"],
        "immune": ["Adgre1", "C1qa", "Mrc1", "Csf1r", "Ly6c2", "Ccr2", "Plac8", "Flt3", "Xcr1",
                   "H2-Ab1", "Cd79a", "Ms4a1", "Cd3e", "Cd3d", "Cd4", "Cd8a", "Trdc", "Ncr1",
                   "Eomes", "Itga1", "Gzmb", "Gzmc", "Prf1", "Itgax", "Mki67", "Top2a"],
        "epithelial": ["Wnt7a", "Foxa2", "Prss28", "Prss29", "Ihh", "Lif", "Lgr5", "Foxj1",
                       "Mki67", "Top2a"],
        "vascular": ["Pecam1", "Flt1", "Cldn5", "Cdh5", "Prox1", "Lyve1"],
        "smc": ["Acta2", "Myh11", "Tagln", "Des", "Cnn1"],
        "perimetrium": ["Msln", "Upk3b", "Wt1", "Sox6"],
    },
    "human": {
        "stromal": ["PRL", "IGFBP1", "LEFTY2", "FOXO1", "PDGFRA", "DCN", "LUM", "RGS5", "NOTCH3",
                    "MCAM", "MKI67"],
        "immune": ["CD68", "C1QA", "MRC1", "CD14", "CD3E", "CD8A", "CD4", "IL7R", "FOXP3", "CTLA4",
                   "IL2RA", "TRDC", "KIT", "CPA3", "TPSAB1", "NCAM1", "KLRD1", "GZMB", "ITGAX",
                   "MKI67"],
        "epithelial": ["WNT7A", "FOXA2", "LIF", "SCGB1D2", "PAEP", "FOXJ1", "MKI67"],
        "perivascular": ["PECAM1", "FLT1", "VWF", "PROX1", "LYVE1", "CCL21", "PDPN"],
        "smc": ["ACTA2", "MYH11", "TAGLN", "DES", "CNN1"],
    },
}
ATLAS_MARKERS = {
    "mouse": ["Pdgfra", "Col1a1", "Prl8a2", "Cdh1", "Pax8", "Epcam", "Ptprc", "Pecam1", "Flt1",
              "Acta2", "Myh11", "Msln", "Upk3b", "Wt1"],
    "human": ["PDGFRA", "DCN", "PRL", "IGFBP1", "CDH1", "PAX8", "EPCAM", "PTPRC", "PECAM1", "FLT1",
              "RGS5", "ACTA2", "MYH11"],
}


def markdown(text):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


def notebook(species, level, resolution, object_path, table_path, panel, title, stage_text):
    """The cells shared by every annotation notebook."""
    cells = [
        markdown(f"""
        # {title}

        {stage_text}

        Clustering: `{object_path}`, column `{resolution}`. Decisions: `{table_path}`, one
        row per cluster with the markers that justify the label in `note`. Nothing in this
        notebook assigns a label; it shows the evidence the table was decided from.
        """),
        code(f"""
        import warnings
        from pathlib import Path

        import anndata as ad
        import numpy as np
        import pandas as pd
        import scanpy as sc

        # Notebook kernels start in notebooks/annotation/, so the repo root is two levels up.
        REPO = Path("../..").resolve()
        if not (REPO / "resources/annotations").is_dir():
            raise FileNotFoundError(f"{{REPO}} is not the repository root")
        SPECIES, LEVEL, RESOLUTION = "{species}", "{level}", "{resolution}"
        OBJECT = REPO / "{object_path}"
        TABLE = REPO / "{table_path}"
        PANEL = {panel!r}
        sc.settings.set_figure_params(dpi=80, frameon=False)
        pd.set_option("display.max_colwidth", 200)
        """),
        markdown("## 1. The committed table"),
        code("""
        table = pd.read_csv(TABLE, dtype={"cluster": str})
        table.set_index("cluster")
        """),
        markdown("## 2. The clustered object (must carry exactly the clusters in the table)"),
        code("""
        adata = ad.read_h5ad(OBJECT)
        adata.obs["cluster"] = adata.obs[RESOLUTION].astype(str)
        in_object, in_table = set(adata.obs["cluster"]), set(table["cluster"])
        if in_object != in_table:
            raise ValueError(f"cluster sets differ: object-only {sorted(in_object - in_table)}, "
                             f"table-only {sorted(in_table - in_object)}")
        labels = table.set_index("cluster")["label"]
        adata.obs["label"] = adata.obs["cluster"].map(labels).astype("category")
        print(adata.n_obs, "nuclei;", len(in_object), "clusters")
        print(adata.obs["label"].value_counts().to_dict())
        """),
        code("""
        # Expression for the dot plot: CP10K log1p from raw counts.
        counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
        expr = ad.AnnData(X=counts.copy(), obs=adata.obs[["cluster", "label"]].copy(),
                          var=adata.var[[]].copy())
        sc.pp.normalize_total(expr, target_sum=1e4)
        sc.pp.log1p(expr)
        genes = [g for g in PANEL if g in expr.var_names]
        missing = sorted(set(PANEL) - set(genes))
        if missing:
            print("not in this object:", missing)
        """),
        markdown("""
        ## 3. Marker dot plot by cluster

        Dot size is the fraction of nuclei in which the gene is detected, colour the mean
        log1p CP10K. Clusters are grouped by their label; EXCLUDE clusters are last.
        """),
        code("""
        def sort_key(c):
            return (labels[c] == "EXCLUDE", labels[c], int(c) if c.isdigit() else c)

        order = sorted(in_object, key=sort_key)
        expr.obs["cluster_labelled"] = pd.Categorical(
            [f"{c}  {labels[c]}" for c in expr.obs["cluster"]],
            categories=[f"{c}  {labels[c]}" for c in order])
        sc.pl.dotplot(expr, genes, groupby="cluster_labelled", standard_scale=None,
                      figsize=(0.32 * len(genes) + 3, 0.28 * len(order) + 1.5))
        """),
        markdown("""
        ## 4. QC per cluster

        Median DoubletDetection score and genes per nucleus per cluster, with the lineage
        median for reference. A high doublet score together with markers of another lineage
        (section 6) is what an EXCLUDE call rests on; a moderate score alone is not
        (cycling clusters score around 1 throughout this dataset).
        """),
        code("""
        qc_cols = [c for c in ("doubletdetection_doublet_score", "n_genes_by_counts",
                               "total_counts") if c in adata.obs]
        qc = adata.obs.groupby("cluster", observed=True)[qc_cols].median().loc[order]
        qc.insert(0, "n", adata.obs["cluster"].value_counts().loc[order])
        qc.insert(1, "label", labels.loc[order])
        qc.loc["lineage median"] = [adata.n_obs, ""] + list(adata.obs[qc_cols].median())
        qc.round(2)
        """),
        markdown("## 5. UMAP by cluster and by label"),
        code("""
        umap_key = next((k for k in ("X_umap_scVI", "X_umap") if k in adata.obsm), None)
        if umap_key is None:
            latent_key = next(k for k in adata.obsm if "scVI" in k or "scvi" in k)
            sc.pp.neighbors(adata, use_rep=latent_key, n_neighbors=15, random_state=0)
            sc.tl.umap(adata, random_state=0)
            umap_key = "X_umap"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pl.embedding(adata, basis=umap_key, color=["cluster", "label"], legend_loc="on data",
                            legend_fontsize=7, size=6, ncols=2, wspace=0.25)
        """),
        markdown("""
        ## 6. EXCLUDE clusters

        Detection of each lineage's markers inside the excluded clusters, next to the
        lineage-wide detection; a cluster is excluded when it carries another lineage's
        programme (doublets) or too little RNA to call.
        """),
        code("""
        excluded = [c for c in order if labels[c] == "EXCLUDE"]
        if excluded:
            detected = pd.DataFrame(
                {c: np.asarray((expr[expr.obs["cluster"] == c, genes].X > 0).mean(0)).ravel()
                 for c in excluded},
                index=genes)
            detected["lineage"] = np.asarray((expr[:, genes].X > 0).mean(0)).ravel()
            display(detected.round(2).T)
        else:
            print("no EXCLUDE clusters in this table")
        """),
    ]
    return cells


def mouse_epithelium_extra():
    return [
        markdown("""
        ## 7. Luminal vs glandular (v3)

        The split was called per cluster: Wnt7a detected in >= 14% of nuclei with
        Foxa2/Prss28/Prss29 each < 8% -> luminal; the reverse -> glandular; clusters
        meeting neither rule stay `epithelial_cell`. Detection fractions and the UMAP
        of the two axes:
        """),
        code("""
        axes_genes = [g for g in ["Wnt7a", "Foxa2", "Prss28", "Prss29"] if g in expr.var_names]
        detected = pd.DataFrame(np.asarray((expr[:, axes_genes].X > 0).todense()),
                                columns=axes_genes, index=expr.obs_names)
        frac = detected.groupby(expr.obs["cluster"].values).mean().loc[order]
        frac.insert(0, "label", labels.loc[order])
        display(frac.round(2))
        for g in axes_genes[:2]:
            column = np.asarray(counts[:, list(adata.var_names).index(g)].todense()).ravel()
            adata.obs[f"{g} detected"] = (column > 0).astype(str)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pl.embedding(adata, basis=umap_key, color=[f"{g} detected" for g in axes_genes[:2]],
                            size=6, ncols=2, palette=["#e0e0e0", "#4a1486"])
        """),
    ]


def write_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for species, species_spec in SPEC["species"].items():
        atlas = SPEC["atlas"][species]
        nb = nbformat.v4.new_notebook()
        nb.cells = notebook(
            species, "lineage", atlas["res"], f"data/interim/latent_{species}.h5ad",
            f"resources/annotations/{species}_lineage.csv", ATLAS_MARKERS[species],
            f"{species.capitalize()} atlas: lineage annotation (stage 04)",
            "Every cluster of the whole-tissue scVI clustering is assigned one lineage (or "
            "EXCLUDE); the lineage objects re-integrated in 04b are cut from these labels.")
        path = OUT_DIR / f"{species}_lineage.ipynb"
        nbformat.write(nb, path)
        written.append(path)

        for lineage, lineage_spec in species_spec["lineages"].items():
            resolution = lineage_spec.get("res", SPEC["defaults"]["res"])
            panel = [g for group in lineage_spec["markers"].values() for g in group]
            panel += [g for g in FINE_MARKERS[species].get(lineage, []) if g not in panel]
            nb = nbformat.v4.new_notebook()
            nb.cells = notebook(
                species, lineage, resolution, f"data/interim/latent_{species}_{lineage}.h5ad",
                f"resources/annotations/{species}_{lineage}.csv", panel,
                f"{species.capitalize()} {lineage}: cell-type annotation (stage 04c)",
                f"Every cluster of the re-integrated {lineage} lineage is assigned one cell type "
                "(or EXCLUDE). Labels are the vocabulary in resources/annotation_vocabulary.yaml.")
            if species == "mouse" and lineage == "epithelial":
                nb.cells += mouse_epithelium_extra()
            path = OUT_DIR / f"{species}_{lineage}.ipynb"
            nbformat.write(nb, path)
            written.append(path)
    return written


def execute(paths):
    """Execute every notebook in place; report all failures at the end, then fail."""
    failed = []
    for path in paths:
        print("executing", path, flush=True)
        result = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                                 "--inplace", "--ExecutePreprocessor.timeout=3600", str(path)])
        if result.returncode != 0:
            failed.append(path)
    if failed:
        raise RuntimeError(f"{len(failed)} notebook(s) failed: {[str(p) for p in failed]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--execute", action="store_true", help="execute the notebooks in place")
    args = parser.parse_args()
    paths = write_all()
    print(f"wrote {len(paths)} notebooks under {OUT_DIR}/")
    if args.execute:
        execute(paths)
