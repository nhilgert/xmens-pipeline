"""Collate per-arm eval.json files into one SAMap arm comparison table.

Reads <arm_dir>/eval.json for every arm directory given on the command line,
prints the table with a key to its columns, and writes summary.csv next to the
arm directories. Called by rule samap_raw_compare (workflow/rules/10b_samap_raw.smk).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    rows = []
    for d in sys.argv[1:]:
        f = Path(d) / "eval.json"
        if not f.exists():
            print(f"[skip] {f} missing (arm failed?)")
            continue
        e = json.loads(f.read_text())

        row = {"arm": e.get("arm", Path(d).name),
               "mean_alignment": round(e["mean_alignment"], 3)}

        # Within-species metrics; subtype purity is absent for a species without subtypes.
        for sid in ("hs", "mm"):
            w = e.get(f"within_{sid}", {})
            mix = w.get("sample_mixing", {})
            row[f"{sid}_mix_ratio"] = round(mix.get("mixing_ratio", float("nan")), 3)
            row[f"{sid}_knn_celltype"] = round(
                w.get("knn_celltype_agreement", float("nan")), 3)
            if "knn_subtype_agreement" in w:
                row[f"{sid}_knn_subtype"] = round(w["knn_subtype_agreement"], 3)
            row[f"{sid}_NMI_ct"] = round(
                w.get("leiden_vs_celltype_NMI", float("nan")), 3)
            row[f"{sid}_NMI_sample"] = round(
                w.get("leiden_vs_sample_NMI", float("nan")), 3)

        # Cross-species alignment of the matched coarse types only (dec~dec, end~end);
        # keys are "hs_<type>" / "mm_<type>", so [3:] strips the species prefix.
        cm = e.get("celltype_mapping", {})
        for hs_k, mm_d in cm.items():
            for mm_k, v in mm_d.items():
                short = f"{hs_k[3:6]}~{mm_k[3:6]}"
                if hs_k[3:] == mm_k[3:]:
                    row[f"map_{short}"] = round(v, 3)
        rows.append(row)

    if not rows:
        print("No eval.json found in any arm dir.")
        return

    df = pd.DataFrame(rows).set_index("arm")
    print("\n================ ARM COMPARISON ================")
    print(df.to_string())
    print(
        "\nKey: *_mix_ratio = within-species sample mixing, observed/expected "
        "(1.0 = perfectly batch-mixed); *_knn_celltype/subtype = fraction of "
        "within-species neighbor weight sharing the cell's label (higher = "
        "manifold respects cell types); *_NMI_ct vs *_NMI_sample = joint-leiden "
        "agreement with cell types (want HIGH) vs samples (want LOW); "
        "map_dec~dec / map_end~end = cross-species alignment score of matched "
        "coarse cell types; mean_alignment = hs<->mm average."
    )

    out = Path(sys.argv[1]).parent / "summary.csv"
    df.to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
