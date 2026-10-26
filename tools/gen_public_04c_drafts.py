#!/usr/bin/env python3
"""Generate all read-only DRAFT 04c presentation notebooks from frozen CSVs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import yaml


ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/n/lab_storage/eddy_lab/users/nhilgert/xmens_final")
ap.add_argument("--spec", default="resources/lineage_spec.yaml")
ap.add_argument("--vocab", default="resources/annotation_vocabulary.yaml")
ap.add_argument("--outdir", default="notebooks/public")
args = ap.parse_args()
repo = Path(args.repo)
spec = yaml.safe_load((repo / args.spec).read_text())
outdir = repo / args.outdir
outdir.mkdir(parents=True, exist_ok=True)
manifest_path = outdir / "DRAFT_04c_MANIFEST.json"
if manifest_path.exists():
    raise FileExistsError(f"refusing to overwrite {manifest_path}")

forbidden = ["/n/eddy_lab/", "data/interim/reannotated/", "write_h5ad(", ".to_csv("]
records = []
for species, species_cfg in spec["species"].items():
    for lineage, lineage_cfg in species_cfg["lineages"].items():
        csv_path = repo / f"resources/annotations/{species}_{lineage}.csv"
        out = outdir / f"DRAFT_04c_annotate_{species}_{lineage}_denovo.ipynb"
        if out.exists():
            raise FileExistsError(f"refusing to overwrite {out}")
        command = [
            sys.executable, str(repo / "tools/compile_denovo.py"),
            "--species", species, "--lineage", lineage,
            "--res", lineage_cfg.get("res", spec["defaults"]["res"]),
            "--csv", str(csv_path), "--out", str(out),
            "--repo", str(repo), "--spec", str(repo / args.spec),
            "--vocab", str(repo / args.vocab),
        ]
        subprocess.run(command, check=True, cwd=repo)
        notebook = nbformat.read(out, as_version=4)
        nbformat.validate(notebook)
        source = "\n".join(cell.source for cell in notebook.cells)
        violations = [term for term in forbidden if term in source]
        if violations:
            raise ValueError(f"{out}: unsafe source terms {violations}")
        if notebook.metadata.get("xmens", {}).get("status") != "DRAFT":
            raise ValueError(f"{out}: missing DRAFT metadata")
        records.append({
            "species": species, "lineage": lineage, "path": str(out),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "decision_csv": str(csv_path),
            "decision_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "cells": len(notebook.cells), "status": "DRAFT",
        })

manifest = {"schema_version": "xmens.annotation.v1", "status": "DRAFT",
            "notebooks": records}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
