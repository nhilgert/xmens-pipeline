#!/usr/bin/env python3
"""Validate frozen decision records or a staged xmens.annotation.v1 object."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


LEGACY_OBS = {
    "coarse_refined", "fine_refined", "subtype", "annotation_level",
    "cell_type_annotations", "cell_subtype_annotations", "cell_type_subannotations",
}
FIBDEC = {"decidual_cell", "endometrial_fibroblast"}
FIBDEC_N = {"human": 47533, "mouse": 43197}


def canonical_sets(vocab):
    types = set(vocab["allowed"])
    subtype_parent = {}
    for cell_type, subtypes in vocab["allowed"].items():
        for subtype in subtypes or []:
            if subtype in subtype_parent and subtype_parent[subtype] != cell_type:
                raise ValueError(f"duplicate subtype parent for {subtype}")
            subtype_parent[subtype] = cell_type
    return types, subtype_parent


def validate_csvs(repo: Path, species: list[str], spec: dict, vocab: dict):
    failures = []
    canonical_types, subtype_parent = canonical_sets(vocab)
    canonical = canonical_types | set(subtype_parent)
    rows_out = []
    for sp in species:
        print(f"=== {sp} decision records ===")
        pairs = [
            ("(atlas)", spec["atlas"][sp]["res"],
             repo / f"resources/annotations/{sp}_lineage.csv",
             repo / f"data/interim/latent_{sp}.h5ad", True),
        ]
        for lineage, cfg in spec["species"][sp]["lineages"].items():
            pairs.append((
                lineage, cfg.get("res", spec.get("defaults", {}).get("res")),
                repo / f"resources/annotations/{sp}_{lineage}.csv",
                repo / f"data/interim/latent_{sp}_{lineage}.h5ad", False,
            ))
        for lineage, resolution, csv_path, h5ad_path, atlas in pairs:
            table = pd.read_csv(csv_path, dtype=str)
            if not {"cluster", "label"} <= set(table):
                failures.append(f"{csv_path}: missing cluster/label")
                continue
            if table.cluster.duplicated().any():
                failures.append(f"{csv_path}: duplicate cluster keys")
            csv_clusters = set(table.cluster.str.strip())
            labels = set(table.label.str.strip()) - {"EXCLUDE"}
            if atlas:
                valid = set(spec["atlas"][sp]["labels"])
            else:
                valid = canonical
            invalid_labels = sorted(labels - valid)
            if invalid_labels:
                failures.append(f"{csv_path}: invalid labels {invalid_labels}")
            if h5ad_path.exists():
                import anndata as ad
                obs = ad.read_h5ad(h5ad_path, backed="r").obs
                if resolution not in obs:
                    failures.append(f"{h5ad_path}: missing {resolution}")
                    object_clusters = set()
                else:
                    object_clusters = set(obs[resolution].astype(str).unique())
                    if object_clusters != csv_clusters:
                        failures.append(
                            f"{csv_path}: cluster set differs from {h5ad_path.name}; "
                            f"missing={sorted(object_clusters-csv_clusters)} "
                            f"extra={sorted(csv_clusters-object_clusters)}")
            rows_out.append({
                "species": sp, "lineage": lineage, "resolution": resolution,
                "csv": str(csv_path), "n_clusters": len(csv_clusters),
                "n_labels": len(labels), "status": "PASS" if not invalid_labels else "FAIL",
            })
            print(f"  {lineage:14} {resolution}: {len(csv_clusters)} clusters; "
                  f"labels={sorted(labels)}")
    return failures, rows_out


def validate_h5ad(path: Path, species: str, role: str, vocab: dict):
    import anndata as ad
    obj = ad.read_h5ad(path, backed="r")
    obs = obj.obs
    failures = []
    gates = []

    def gate(check, expected, actual, *, info=False):
        status = "INFO" if info else ("PASS" if expected == actual else "SUSPECT")
        gates.append({"check": check, "expected": expected, "actual": actual,
                      "delta": None if info else actual - expected, "status": status})

    for col in ("cell_type", "cell_subtype"):
        if col not in obs:
            failures.append(f"missing required obs column {col}")
    present_legacy = sorted(LEGACY_OBS & set(obs.columns))
    if present_legacy:
        failures.append(f"legacy annotation columns remain: {present_legacy}")
    if failures:
        return obj, failures, gates

    types = obs.cell_type.astype(str)
    subtypes = obs.cell_subtype.astype(str)
    if obs.cell_type.isna().any() or obs.cell_subtype.isna().any():
        failures.append("canonical annotations contain NaN")
    for label, values in (("cell_type", types), ("cell_subtype", subtypes)):
        whitespace = sorted({x for x in set(values) if x != x.strip()})
        if whitespace:
            failures.append(f"{label} has surrounding whitespace: {whitespace}")
    missing_literals = sorted(set(subtypes.str.lower()) & {"nan", "none", "na", "<na>"})
    if missing_literals:
        failures.append(f"cell_subtype contains missing-value literals: {missing_literals}")

    allowed_species = set(vocab["expected_coarse_counts"][species])
    invalid_types = sorted(set(types) - allowed_species)
    if invalid_types:
        failures.append(f"invalid {species} cell_type values: {invalid_types}")
    canonical_types, subtype_parent = canonical_sets(vocab)
    empty_count = int((subtypes == "").sum())
    if empty_count:
        failures.append(
            f"cell_subtype contains {empty_count} empty values; expected coarse-fill")
    invalid_subtypes = sorted(set(subtypes) - canonical_types - set(subtype_parent))
    if invalid_subtypes:
        failures.append(f"invalid cell_subtype values: {invalid_subtypes}")
    invalid_pairs = sorted({
        (cell_type, subtype)
        for cell_type, subtype in zip(types, subtypes)
        if subtype != cell_type and subtype_parent.get(subtype) != cell_type
    })
    if invalid_pairs:
        failures.append(f"invalid subtype-parent pairs: {invalid_pairs}")

    if role in {"refined", "final", "all_deposit"}:
        actual_counts = types.value_counts()
        for label, expected in vocab["expected_coarse_counts"][species].items():
            gate(f"cell_type:{label}", int(expected), int(actual_counts.get(label, 0)))
        gate("total_cells", int(vocab["expected_coarse_counts"]["totals"][species]), obj.n_obs)
    if role in {"fibdec", "fibdec_deposit"}:
        if set(types) != FIBDEC:
            failures.append(f"fib/dec object has cell_type values {sorted(set(types))}")
        gate("fibdec_total", FIBDEC_N[species], obj.n_obs)

    if species == "human" and role in {"refined", "final", "all_deposit"}:
        gate("endothelial_blood", 828, int((subtypes == "endothelial_blood").sum()))
        gate("endothelial_lymph", 506, int((subtypes == "endothelial_lymph").sum()))
        gate("cell_subtype_empty", 0, empty_count)
        for label, expected in (("decidual_cell", 21648),
                                ("endometrial_fibroblast", 25885),
                                ("epithelial_cell", 10608),
                                ("perivascular_cell", 471),
                                ("smooth_muscle_cell", 2084)):
            gate(f"{label}_coarse_subtype", expected,
                 int(((types == label) & (subtypes == label)).sum()))
    if species == "mouse" and role in {"refined", "final", "all_deposit"}:
        gate("endothelial_blood", 3487, int((subtypes == "endothelial_blood").sum()))
        gate("endothelial_lymph", 0, int((subtypes == "endothelial_lymph").sum()))
        gate("cell_subtype_empty", 0, empty_count)
        for label, expected in (("epithelial_cell", 6265),
                                ("smooth_muscle_cell", 1443),
                                ("perimetrial_cell", 550)):
            gate(f"{label}_coarse_subtype", expected,
                 int(((types == label) & (subtypes == label)).sum()))

    if role != "reannotated":
        if obj.uns.get("annotation_schema_version") != vocab["schema_version"]:
            failures.append("annotation_schema_version missing or wrong")
        if obj.uns.get("display_names") != vocab["display_names"]:
            failures.append("uns['display_names'] differs from locked vocabulary")
        if "" in obj.uns.get("display_names", {}).get("cell_subtype", {}):
            failures.append("display_names contains forbidden empty-string key")

    return obj, failures, gates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--species", nargs="+", choices=["human", "mouse"],
                    default=["human", "mouse"])
    ap.add_argument("--spec", default="resources/lineage_spec.yaml")
    ap.add_argument("--vocab", default="resources/annotation_vocabulary.yaml")
    ap.add_argument("--h5ad", default=None)
    ap.add_argument("--role", choices=["reannotated", "refined", "final", "fibdec",
                                       "all_deposit", "fibdec_deposit"], default="refined")
    ap.add_argument("--report", default=None)
    ap.add_argument("--nonfatal-counts", action="store_true")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    vocab = yaml.safe_load((repo / args.vocab).read_text())
    spec = yaml.safe_load((repo / args.spec).read_text())

    if args.h5ad:
        if len(args.species) != 1:
            raise SystemExit("--h5ad requires exactly one --species")
        path = Path(args.h5ad)
        obj, failures, gates = validate_h5ad(path, args.species[0], args.role, vocab)
        count_suspects = [row for row in gates if row["status"] == "SUSPECT"]
        payload = {
            "path": str(path), "species": args.species[0], "role": args.role,
            "shape": [obj.n_obs, obj.n_vars], "schema_failures": failures,
            "count_gates": gates,
            "status": "FAIL" if failures else ("SUSPECT" if count_suspects else "PASS"),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.report:
            report = Path(args.report)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if failures:
            return 1
        if count_suspects and not args.nonfatal_counts:
            return 2
        return 0

    failures, rows = validate_csvs(repo, args.species, spec, vocab)
    payload = {"schema_version": vocab["schema_version"], "failures": failures,
               "records": rows, "status": "FAIL" if failures else "PASS"}
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
