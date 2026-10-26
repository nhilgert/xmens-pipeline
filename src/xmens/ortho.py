"""Mouse-human orthogroups from Ensembl Compara, keeping many-to-many families.

Reads the homology table from BioMart (load_compara_homology) and returns
orthogroups as connected components of the mouse-human homology graph, plus a
gene -> orthogroup lookup. Writes nothing.

Orthology confidence is not filtered by default on purpose: the low-confidence
edges are the expanded gene families (25 of the 26 prolactin-family edges are
low confidence), so filtering on confidence deletes those families wholesale.
"""
from __future__ import annotations
import networkx as nx, pandas as pd


def load_compara_homology(host="http://www.ensembl.org",
                          dataset="mmusculus_gene_ensembl", target="hsapiens"):
    """Query BioMart for one row per (mouse gene, target-species homolog) pair."""
    # Lazy import: pybiomart is only needed by this function.
    import pybiomart

    d = pybiomart.Dataset(name=dataset, host=host)
    df = d.query(attributes=["ensembl_gene_id",
        f"{target}_homolog_ensembl_gene", f"{target}_homolog_orthology_type",
        f"{target}_homolog_orthology_confidence"])
    df.columns = ["mm","hs","type","conf"]

    # Genes with no homolog come back as NaN in the homolog column.
    return df.dropna(subset=["hs"])


def orthogroups_from_homology(df, drop_low_confidence=False):
    """Connected components of the bipartite mouse-human graph, as sets of (species, gene)."""
    if drop_low_confidence:
        df = df[df["conf"] == 1]  # Ensembl flags 1 = high confidence, 0 = low

    # Nodes are (species, gene) pairs, so the graph is bipartite mouse <-> human.
    g = nx.Graph()
    g.add_edges_from((("mm",a),("hs",b)) for a,b in zip(df["mm"],df["hs"]))

    return list(nx.connected_components(g))


def gene_to_orthogroup(orthogroups):
    return {gene:i for i,og in enumerate(orthogroups) for gene in og}
