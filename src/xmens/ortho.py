"""Ensembl Compara orthogroups with many-to-many relationships (connected components).
Do NOT filter on orthology confidence by default — it deletes expanded families
(e.g. 25/26 prolactin edges are low-confidence)."""
from __future__ import annotations
import networkx as nx, pandas as pd

def load_compara_homology(host="http://www.ensembl.org",
                          dataset="mmusculus_gene_ensembl", target="hsapiens"):
    import pybiomart
    d = pybiomart.Dataset(name=dataset, host=host)
    df = d.query(attributes=["ensembl_gene_id",
        f"{target}_homolog_ensembl_gene", f"{target}_homolog_orthology_type",
        f"{target}_homolog_orthology_confidence"])
    df.columns = ["mm","hs","type","conf"]
    return df.dropna(subset=["hs"])

def orthogroups_from_homology(df, drop_low_confidence=False):
    if drop_low_confidence: df = df[df["conf"] == 1]
    g = nx.Graph(); g.add_edges_from((("mm",a),("hs",b)) for a,b in zip(df["mm"],df["hs"]))
    return list(nx.connected_components(g))

def gene_to_orthogroup(orthogroups):
    return {gene:i for i,og in enumerate(orthogroups) for gene in og}
