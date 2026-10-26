"""Write the `note` column of resources/annotations/*.csv from the marker evidence.

Reads resources/lineage_spec.yaml (`label_markers` per lineage and for the atlas), the
committed annotation CSVs and the three outputs of tools/cluster_marker_table.py in the
given directory. Rewrites each CSV in place with columns cluster, label, n_cells, res, note;
cluster and label are never changed (the script checks this before writing). A labelled
cluster's note lists the detection fraction of each of its label's markers and the top
markers against the rest of its level; an EXCLUDE note lists the doublet score, genes per
nucleus and lineage-panel co-expression against the level's reference quantiles. Run from
the repo root after cluster_marker_table.py:

    python tools/write_annotation_notes.py <marker_table_dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

TABLE_DIR = Path(sys.argv[1])
ANNOTATIONS = Path("resources/annotations")
N_TOP_IN_NOTE = 3
# Subtypes named by position in the tissue rather than by a marker programme.
POSITIONAL = {"decidual_boundary", "decidual_mature", "efib_outer", "efib_middle",
              "efib_middle_induced", "efib_subluminal"}

spec = yaml.safe_load(open("resources/lineage_spec.yaml"))
markers = pd.read_csv(TABLE_DIR / "per_cluster_markers.csv", dtype={"cluster": str})
detected = markers.set_index(["species", "level", "cluster", "gene"])["frac_detected"]
qc = pd.read_csv(TABLE_DIR / "per_cluster_qc.csv", dtype={"cluster": str})
qc = qc.set_index(["species", "level", "cluster"])
top_markers = json.load(open(TABLE_DIR / "top_markers.json"))


def label_markers(species, level):
    if level == "lineage":
        return spec["atlas"][species]["label_markers"]
    return spec["species"][species]["lineages"][level]["label_markers"]


def resolution(species, level):
    if level == "lineage":
        return spec["atlas"][species]["res"]
    return spec["species"][species]["lineages"][level].get("res", spec["defaults"]["res"])


def exclude_note(species, level, cluster):
    row, reference = qc.loc[(species, level, cluster)], qc.loc[(species, level, "ALL")]
    panels = [c for c in qc.columns if c.startswith("frac_pos_")]
    strongest = sorted(panels, key=lambda c: -row[c])[:3]
    foreign = ", ".join(f"{c[len('frac_pos_'):]} {row[c]:.2f}" for c in strongest)
    return (f"EXCLUDE: doublet score med {row.doubletdetection_doublet_score:.2f} "
            f"(level q75 {reference.doubletdetection_doublet_score:.2f}), genes med "
            f"{row.n_genes_by_counts:.0f} (level q25 {reference.n_genes_by_counts:.0f}), "
            f"{row.frac_coexpress_2plus:.2f} co-express 2+ lineage panels "
            f"(level {reference.frac_coexpress_2plus:.2f}); panels: {foreign}")


def label_note(species, level, cluster, label):
    genes = label_markers(species, level)[label]
    evidence = "; ".join(f"{g} {detected[(species, level, cluster, g)]:.2f}" for g in genes)
    top = [t for t in top_markers[f"{species}|{level}"].get(cluster, [])
           if t.split()[0] not in genes][:N_TOP_IN_NOTE]
    note = evidence + (f" | top vs level: {', '.join(top)}" if top else "")
    if label in POSITIONAL:
        note += " | position-named subtype: confirm on the spatial map"
    return note


for species in spec["species"]:
    for level in ["lineage"] + list(spec["species"][species]["lineages"]):
        path = ANNOTATIONS / f"{species}_{level}.csv"
        table = pd.read_csv(path, dtype=str)
        before = table[["cluster", "label"]].copy()
        table["n_cells"] = [int(qc.loc[(species, level, c), "n"]) for c in table.cluster]
        table["res"] = resolution(species, level)
        table["note"] = [exclude_note(species, level, c) if label == "EXCLUDE"
                         else label_note(species, level, c, label)
                         for c, label in zip(table.cluster, table.label)]
        table = table[["cluster", "label", "n_cells", "res", "note"]]
        if not table[["cluster", "label"]].equals(before):
            raise RuntimeError(f"{path}: cluster/label columns would change; not written")
        table.to_csv(path, index=False)
        print(f"{path}: {len(table)} rows")
