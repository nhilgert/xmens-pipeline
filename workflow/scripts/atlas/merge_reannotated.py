#!/usr/bin/env python3
"""Concatenate canonical per-lineage objects into one annotation-complete atlas.

This script is shared by the normal and migration-only rules.  It never changes
counts to satisfy a gate: expected/actual discrepancies are written as SUSPECT
and the staged object is retained for human review.
"""
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml


sm = globals().get("snakemake")
parts_in = list(sm.input.parts)
out_h5ad = Path(sm.output.h5ad)
report = Path(sm.output.report)
species = sm.params.species
ref_path = str(sm.params.ref or "")
vocab_path = Path(getattr(sm.input, "vocab", None)
                  or getattr(sm.params, "vocab", "resources/annotation_vocabulary.yaml"))
vocab = yaml.safe_load(vocab_path.read_text())

KEEP_OBS = ["sample", "cell_barcode", "timepoint", "cell_type", "cell_subtype"]


def canon(series):
    series = series.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return series.str.replace("bleeding", "menstrual", regex=False)


def key(obs):
    return canon(obs["sample"]) + "||" + obs["cell_barcode"].astype(str).str.replace(
        r"-\d+$", "", regex=True)


def gate_row(check, expected, actual):
    delta = actual - expected
    return {
        "object": f"refined_{species}_migrated",
        "check": check,
        "expected": int(expected),
        "actual": int(actual),
        "delta": int(delta),
        "status": "PASS" if delta == 0 else "SUSPECT",
    }


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
    if first_vars is None:
        first_vars = obj.var_names.copy()
    elif not obj.var_names.equals(first_vars):
        raise ValueError(f"{path}: var_names/order differs from first lineage object")
    obj.obs["origin_lineage"] = path.stem.split("_", 1)[1]
    for embedding in ("X_scVI", "X_umap_scVI"):
        if embedding in obj.obsm:
            del obj.obsm[embedding]
    obj.obs = obj.obs[KEEP_OBS + ["origin_lineage"]].copy()
    print(f"  {path.name:34} {obj.n_obs:>7,} cells")
    sum_cells += obj.n_obs
    parts.append(obj)

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

# Enforce the post-04c coarse-fill contract before writing a staged atlas.
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

# Count gates are evidence, not mutation instructions.
rows = []
expected_types = vocab["expected_coarse_counts"][species]
actual_types = merged.obs["cell_type"].astype(str).value_counts()
for label, expected in expected_types.items():
    rows.append(gate_row(f"cell_type:{label}", expected, int(actual_types.get(label, 0))))
rows.append(gate_row("total_cells", vocab["expected_coarse_counts"]["totals"][species],
                     merged.n_obs))
rows.append(gate_row("cell_subtype_empty", 0, empty_count))
if species == "human":
    actual_sub = merged.obs["cell_subtype"].astype(str).value_counts()
    rows.extend([
        gate_row("cell_subtype:endothelial_blood", 828,
                 int(actual_sub.get("endothelial_blood", 0))),
        gate_row("cell_subtype:endothelial_lymph", 506,
                 int(actual_sub.get("endothelial_lymph", 0))),
        gate_row("perivascular_coarse_subtype", 471,
                 int(((merged.obs.cell_type.astype(str) == "perivascular_cell") &
                      (merged.obs.cell_subtype.astype(str) == "perivascular_cell")).sum())),
        gate_row("decidual_coarse_subtype", 21648,
                 int(((types == "decidual_cell") & (subtypes == "decidual_cell")).sum())),
        gate_row("fibroblast_coarse_subtype", 25885,
                 int(((types == "endometrial_fibroblast") &
                      (subtypes == "endometrial_fibroblast")).sum())),
        gate_row("epithelial_coarse_subtype", 10608,
                 int(((types == "epithelial_cell") &
                      (subtypes == "epithelial_cell")).sum())),
        gate_row("smooth_muscle_coarse_subtype", 2084,
                 int(((types == "smooth_muscle_cell") &
                      (subtypes == "smooth_muscle_cell")).sum())),
    ])
else:
    actual_sub = merged.obs["cell_subtype"].astype(str).value_counts()
    epi_split = vocab["content_tasks"]["mouse_epithelial_split"]["expected_counts"]
    rows.extend([
        gate_row("cell_subtype:endothelial_blood", 3487,
                 int(actual_sub.get("endothelial_blood", 0))),
        gate_row("cell_subtype:endothelial_lymph", 0,
                 int(actual_sub.get("endothelial_lymph", 0))),
        # v3: mouse epithelium is split; expected counts live in the vocabulary.
        gate_row("cell_subtype:epithelial_luminal", epi_split["epithelial_luminal"],
                 int(actual_sub.get("epithelial_luminal", 0))),
        gate_row("cell_subtype:epithelial_glandular", epi_split["epithelial_glandular"],
                 int(actual_sub.get("epithelial_glandular", 0))),
        gate_row("epithelial_coarse_subtype", epi_split["epithelial_cell"],
                 int(((types == "epithelial_cell") &
                      (subtypes == "epithelial_cell")).sum())),
        gate_row("smooth_muscle_coarse_subtype", 1443,
                 int(((types == "smooth_muscle_cell") &
                      (subtypes == "smooth_muscle_cell")).sum())),
        gate_row("mesothelial_coarse_subtype", 550,
                 int(((types == "perimetrial_cell") &
                      (subtypes == "perimetrial_cell")).sum())),
    ])

# Reference agreement is partial because legacy human "perivascular cells" is
# deliberately content-ambiguous.  Exclude unmapped reference labels.
if ref_path and Path(ref_path).exists():
    them = sc.read_h5ad(ref_path, backed="r").obs.copy()
    them["k"] = key(them)
    ref_raw = key(merged.obs).map(them.set_index("k")["cell_type_annotations"])
    type_map = vocab["import_mapping"][species]["cell_type"]
    ref_canonical = ref_raw.map(type_map)
    comparable = ref_canonical.notna().values
    agreement = ((merged.obs.cell_type.astype(str).values[comparable] ==
                  ref_canonical.astype(str).values[comparable]).mean()
                 if comparable.sum() else np.nan)
    rows.append({
        "object": f"refined_{species}_migrated",
        "check": "legacy_reference_comparable_agreement_pct",
        "expected": np.nan,
        "actual": round(100 * agreement, 4) if comparable.sum() else np.nan,
        "delta": np.nan,
        "status": "INFO",
    })
    print(f"legacy reference comparable rows: {int(comparable.sum()):,}; "
          f"agreement={100 * agreement:.2f}%")
else:
    print(f"note: reference object not found: {ref_path}")

gate_table = pd.DataFrame(rows)
report.parent.mkdir(parents=True, exist_ok=True)
report_tmp = report.with_name(f".{report.name}.{os.getpid()}.tmp")
gate_table.to_csv(report_tmp, sep="\t", index=False)
os.replace(report_tmp, report)
print("\n=== count gates ===")
print(gate_table.to_string(index=False))

out_h5ad.parent.mkdir(parents=True, exist_ok=True)
if out_h5ad.exists() and "migrated" in out_h5ad.parts:
    raise FileExistsError(f"refusing to overwrite existing staged object: {out_h5ad}")
tmp = out_h5ad.with_name(f".{out_h5ad.name}.{os.getpid()}.tmp.h5ad")
merged.write_h5ad(tmp)
os.replace(tmp, out_h5ad)
print(f"\nwrote {out_h5ad} ({merged.n_obs:,} cells; status retained in {report})")
