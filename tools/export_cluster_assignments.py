"""Write the barcode -> cluster tables that the annotation tables refer to.

Reads resources/lineage_spec.yaml (atlas and lineage resolutions),
resources/annotations/{species}_lineage.csv, data/interim/latent_{species}.h5ad and
data/interim/latent_{species}_{lineage}.h5ad (obs only). Writes
resources/clusters/{species}_clusters.csv.gz: one row per nucleus of the atlas object,
with its atlas cluster and, for nuclei the atlas table keeps, its cluster within its
lineage. Fails unless each lineage object holds exactly the nuclei the atlas table sends
to it. Not part of the Snakemake DAG; run from the repo root:

    python tools/export_cluster_assignments.py
"""
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml

OUT_DIR = Path("resources/clusters")
spec = yaml.safe_load(open("resources/lineage_spec.yaml"))


def read_clusters(path, resolution):
    """obs_name -> cluster id as text, reading only obs from the object."""
    return ad.read_h5ad(path, backed="r").obs[resolution].astype(str)


OUT_DIR.mkdir(exist_ok=True)
for species, species_spec in spec["species"].items():
    atlas_res = spec["atlas"][species]["res"]
    lineage_resolutions = {lineage.get("res", spec["defaults"]["res"])
                           for lineage in species_spec["lineages"].values()}
    if len(lineage_resolutions) != 1:
        raise ValueError(f"{species}: lineages use several resolutions "
                         f"{sorted(lineage_resolutions)}; the table has one lineage column")
    lineage_res = lineage_resolutions.pop()

    atlas = read_clusters(f"data/interim/latent_{species}.h5ad", atlas_res)
    atlas_labels = pd.read_csv(f"resources/annotations/{species}_lineage.csv", dtype=str)
    lineage_of = atlas.map(atlas_labels.set_index("cluster")["label"])
    lineage_of = lineage_of.map(spec["atlas"][species]["labels"])    # EXCLUDE -> NaN

    table = pd.DataFrame({atlas_res: atlas, lineage_res: pd.NA}, index=atlas.index)
    for lineage in species_spec["lineages"]:
        clusters = read_clusters(f"data/interim/latent_{species}_{lineage}.h5ad", lineage_res)
        expected = set(lineage_of.index[lineage_of == lineage])
        if set(clusters.index) != expected:
            raise ValueError(f"{species} {lineage}: the lineage object holds "
                             f"{len(clusters):,} nuclei, the atlas table sends {len(expected):,}")
        table.loc[clusters.index, lineage_res] = clusters

    path = OUT_DIR / f"{species}_clusters.csv.gz"
    table.rename_axis("obs_name").to_csv(path)
    print(f"{path}: {len(table):,} nuclei, "
          f"{int(table[lineage_res].notna().sum()):,} with a lineage cluster")
