"""Concatenate the reannotated per-lineage objects into one per-species atlas.

Called by rule `merge_reannotated` (04d_merge.smk). Reads every
data/interim/reannotated/{species}_{lineage}.h5ad plus annotation_vocabulary.yaml and
writes data/interim/refined_{species}.h5ad (obs reduced to the canonical columns, no
embedding) and results/tables/refined_{species}_guardrail.tsv. Counts are never changed
to satisfy a gate: expected/actual discrepancies are written as SUSPECT and the object
is retained for human review.
"""
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

# Snakemake plumbing ----

sm = globals().get("snakemake")
parts_in = list(sm.input.parts)
out_h5ad = Path(sm.output.h5ad)
report = Path(sm.output.report)
species = sm.params.species
vocab_path = Path(getattr(sm.input, "vocab", None)
                  or getattr(sm.params, "vocab", "resources/annotation_vocabulary.yaml"))
vocab = yaml.safe_load(vocab_path.read_text())

KEEP_OBS = ["sample", "cell_barcode", "timepoint", "cell_type", "cell_subtype"]


def gate_row(check, expected, actual):
    delta = actual - expected
    return {
        "object": f"refined_{species}",
        "check": check,
        "expected": int(expected),
        "actual": int(actual),
        "delta": int(delta),
        "status": "PASS" if delta == 0 else "SUSPECT",
    }


# Load and check the lineage objects ----

parts = []
first_vars = None
sum_cells = 0
for path_string in parts_in:
    path = Path(path_string)
    obj = sc.read_h5ad(path)

    missing = sorted(set(("cell_type", "cell_subtype", "sample", "cell_barcode")) - set(obj.obs))
    if missing:
        raise ValueError(f"{path}: missing canonical/provenance obs columns {missing}")
    if obj.obs_names.has_duplicates:
        raise ValueError(f"{path}: duplicated obs_names")
    if "counts" not in obj.layers:
        raise ValueError(f"{path}: missing raw counts layer")

    # An inner join on var would silently drop genes if the lineage objects differed.
    if first_vars is None:
        first_vars = obj.var_names.copy()
    elif not obj.var_names.equals(first_vars):
        raise ValueError(f"{path}: var_names/order differs from first lineage object")

    # The file stem is {species}_{lineage}; per-lineage embeddings are separate fits
    # in different spaces, so they do not survive the merge (04e re-embeds).
    obj.obs["origin_lineage"] = path.stem.split("_", 1)[1]
    for embedding in ("X_scVI", "X_umap_scVI"):
        if embedding in obj.obsm:
            del obj.obsm[embedding]
    obj.obs = obj.obs[KEEP_OBS + ["origin_lineage"]].copy()

    print(f"  {path.name:34} {obj.n_obs:>7,} cells")
    sum_cells += obj.n_obs
    parts.append(obj)

# Concatenate ----

merged = ad.concat(parts, axis=0, join="inner", merge="same", uns_merge=None,
                   index_unique=None)
if merged.n_obs != sum_cells:
    raise ValueError(f"concat cell count {merged.n_obs} != sum(parts) {sum_cells}")
if merged.obs_names.has_duplicates:
    raise ValueError("merged object has duplicated obs_names")
if "counts" not in merged.layers:
    raise ValueError("lost layers['counts']; 04e requires raw counts")

merged.obs["cell_subtype"] = merged.obs["cell_subtype"].astype(object).fillna("").astype(str)
merged.uns["annotation_schema_version"] = vocab["schema_version"]
merged.uns["display_names"] = vocab["display_names"]
print(f"\nmerged: {merged.n_obs:,} cells x {merged.n_vars:,} genes")

# Vocabulary checks ----

# The coarse-fill contract from 04c: every cell has a subtype, which is either a
# subtype allowed under its cell_type or the cell_type itself.
types = merged.obs["cell_type"].astype(str)
subtypes = merged.obs["cell_subtype"].astype(str)
empty_count = int((subtypes == "").sum())
if empty_count:
    raise ValueError(f"merged {species}: cell_subtype has {empty_count} empty values")

allowed_types = set(vocab["expected_coarse_counts"][species])
invalid_types = sorted(set(types) - allowed_types)
if invalid_types:
    raise ValueError(f"merged {species}: invalid cell_type values {invalid_types}")

subtype_parent = {
    subtype: parent
    for parent, values in vocab["allowed"].items()
    for subtype in (values or [])
}
invalid_pairs = sorted({
    (parent, subtype)
    for parent, subtype in zip(types, subtypes)
    if subtype != parent and subtype_parent.get(subtype) != parent
})
if invalid_pairs:
    raise ValueError(f"merged {species}: invalid subtype-parent pairs {invalid_pairs}")

# Count gates ----

# Gates are evidence, not mutation instructions: a mismatch is recorded as SUSPECT. The
# expected numbers are the vocabulary's expected_coarse_counts and expected_subtype_counts
# (the latter is the per-label sum of the committed annotation tables).
rows = []
actual_types = merged.obs["cell_type"].astype(str).value_counts()
for label, expected in vocab["expected_coarse_counts"][species].items():
    rows.append(gate_row(f"cell_type:{label}", expected, int(actual_types.get(label, 0))))
rows.append(gate_row("total_cells", vocab["expected_coarse_counts"]["totals"][species],
                     merged.n_obs))
rows.append(gate_row("cell_subtype_empty", 0, empty_count))

actual_subtypes = merged.obs["cell_subtype"].astype(str).value_counts()
for label, expected in vocab["expected_subtype_counts"][species].items():
    rows.append(gate_row(f"cell_subtype:{label}", expected, int(actual_subtypes.get(label, 0))))
for label in set(actual_subtypes.index) - set(vocab["expected_subtype_counts"][species]):
    rows.append(gate_row(f"cell_subtype:{label}", 0, int(actual_subtypes[label])))

# Write the gate table and the merged object ----

# Both are written to a temporary name and renamed, so a killed job cannot leave a
# partial file that Snakemake would treat as complete.
gate_table = pd.DataFrame(rows)
report.parent.mkdir(parents=True, exist_ok=True)
report_tmp = report.with_name(f".{report.name}.{os.getpid()}.tmp")
gate_table.to_csv(report_tmp, sep="\t", index=False)
os.replace(report_tmp, report)
print("\n=== count gates ===")
print(gate_table.to_string(index=False))

out_h5ad.parent.mkdir(parents=True, exist_ok=True)
tmp = out_h5ad.with_name(f".{out_h5ad.name}.{os.getpid()}.tmp.h5ad")
merged.write_h5ad(tmp)
os.replace(tmp, out_h5ad)
print(f"\nwrote {out_h5ad} ({merged.n_obs:,} cells; status retained in {report})")
