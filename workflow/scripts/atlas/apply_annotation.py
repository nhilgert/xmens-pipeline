"""Apply a committed cluster-to-label choices CSV to a clustered object.

The single implementation of the apply step, reached three ways on one code path: the
Snakemake rules `annotate_<species>` (04_annotate.smk, level atlas) and `reannotate_lineage`
(04c_reannotate.smk, level fine), the command line (`--level atlas ...`), and the
importable function `apply_map`, so that they cannot drift.

Level atlas reads latent_{sp}.h5ad and {sp}_lineage.csv and writes annotated_{sp}.h5ad,
one lineages/{sp}_{slug}.h5ad per lineage and the Leiden ladder CSVs. Level fine reads
latent_{sp}_{lin}.h5ad and {sp}_{lin}.csv and writes reannotated/{sp}_{lin}.h5ad.
"""
import sys
from pathlib import Path
import pandas as pd, numpy as np

__all__ = [
    "load_choices", "apply_map", "write_atlas", "write_fine", "build_rollup",
    "build_vocabulary_rollup", "validate_canonical_annotations",
]


# Choices ----

def load_choices(csv) -> dict:
    """cluster -> label, from the committed two-column record."""
    df = pd.read_csv(csv, dtype=str)
    need = {"cluster", "label"}
    if not need <= set(df.columns):
        raise ValueError(f"{csv}: needs columns {need}, has {list(df.columns)}")

    cluster = df["cluster"].str.strip()
    label = df["label"].str.strip()
    if cluster.isna().any() or label.isna().any() or (cluster == "").any() or (label == "").any():
        raise ValueError(f"{csv}: blank cluster or label")
    if cluster.duplicated().any():
        raise ValueError(
            f"{csv}: duplicate cluster keys {sorted(set(cluster[cluster.duplicated(False)]))}"
        )
    return dict(zip(cluster, label))


def build_rollup(fine_cfg: dict, coarse: str) -> dict:
    """fine label -> coarse label, from a spec `fine:` block.

    A lineage with no `labels` has no fine labels: its CSV carries coarse labels and
    nothing rolls up. Otherwise every listed label maps to `rollup_all`, to the
    `rollup_prefix` entry it starts with, or to the lineage's coarse label.
    """
    out = {}
    for lab in (fine_cfg or {}).get("labels", []):
        if "rollup_all" in fine_cfg:
            out[lab] = fine_cfg["rollup_all"]
            continue
        for pref, tgt in fine_cfg.get("rollup_prefix", {}).items():
            if str(lab).startswith(pref):
                out[lab] = tgt
                break
        else:
            out[lab] = coarse
    return out


def build_vocabulary_rollup(vocab: dict) -> dict:
    """Return the globally unique canonical subtype -> cell-type mapping."""
    out = {}
    for cell_type, subtypes in vocab["allowed"].items():
        for subtype in subtypes or []:
            if subtype in out and out[subtype] != cell_type:
                raise ValueError(
                    f"cell_subtype {subtype!r} appears under both {out[subtype]!r} "
                    f"and {cell_type!r} in annotation_vocabulary.yaml")
            out[subtype] = cell_type
    return out


def validate_canonical_annotations(adata, vocab: dict, *, context: str = "object"):
    """Fail on missing, legacy, or invalid canonical annotation values."""
    for col in ("cell_type", "cell_subtype"):
        if col not in adata.obs:
            raise ValueError(f"{context}: missing required obs column {col!r}")
        if adata.obs[col].isna().any():
            raise ValueError(f"{context}: {col} contains NaN")

    types = adata.obs["cell_type"].astype(str)
    subtypes = adata.obs["cell_subtype"].astype(str)
    invalid_types = sorted(set(types) - set(vocab["allowed"]))
    if invalid_types:
        raise ValueError(f"{context}: invalid cell_type values {invalid_types}")

    # Missing values that were stringified upstream would pass the NaN check above.
    bad_missing = sorted(set(subtypes.str.lower()) & {"nan", "none", "na", "<na>"})
    if bad_missing:
        raise ValueError(f"{context}: invalid missing-value strings in cell_subtype {bad_missing}")
    if (subtypes == "").any():
        raise ValueError(
            f"{context}: cell_subtype contains {int((subtypes == '').sum())} empty values; "
            "kept cells must use a real subtype or the coarse-fill cell_type")

    # A kept cell_subtype is valid if it is a real subtype allowed under its
    # cell_type or equals its cell_type under the coarse-fill policy.
    bad_pairs = []
    for cell_type, subtype in zip(types, subtypes):
        if (
            subtype
            and subtype != cell_type
            and subtype not in set(vocab["allowed"][cell_type] or [])
        ):
            bad_pairs.append((cell_type, subtype))
    if bad_pairs:
        raise ValueError(
            f"{context}: invalid cell_type/cell_subtype pairs {sorted(set(bad_pairs))}"
        )
    return True


# Apply ----

def apply_map(adata, res: str, choices: dict, rollup: dict | None = None,
              label_col: str = "label", strict: bool = True,
              output_mode: str = "fine"):
    """Apply a frozen map as either internal lineage or canonical annotations.

    Returns a per-cluster summary frame. Pure: no I/O, no plotting -- this is
    what the pipeline and the notebooks share.
    """
    rollup = rollup or {}
    if res not in adata.obs:
        raise KeyError(f"{res} not in obs; have "
                       f"{[c for c in adata.obs if c.startswith('leiden')]}")

    # Every cluster present needs a decision; rows for clusters that no longer exist
    # (a re-run ladder) are tolerated.
    clusters = set(adata.obs[res].astype(str).unique())
    missing = clusters - set(choices)
    extra = set(choices) - clusters
    if missing and strict:
        raise ValueError(f"clusters with no row in the choices CSV: "
                         f"{sorted(missing, key=lambda c: (len(c), c))}")
    if extra:
        print(f"  note: CSV has rows for {len(extra)} absent cluster(s) (ignored)")

    key = adata.obs[res].astype(str)
    adata.obs[label_col] = key.map(choices).astype(str)
    adata.obs["excluded"] = adata.obs[label_col].eq("EXCLUDE").values

    if output_mode == "atlas":
        adata.obs["lineage"] = np.where(
            adata.obs.excluded, "EXCLUDE", adata.obs[label_col])
    elif output_mode == "fine":
        adata.obs["cell_type"] = np.where(
            adata.obs.excluded, "EXCLUDE",
            adata.obs[label_col].map(lambda value: rollup.get(value, value)))

        # cell_subtype policy: a real fine subtype where one exists, otherwise the
        # coarse cell_type rather than an empty string. Excluded cells stay "" because
        # they are dropped downstream.
        adata.obs["cell_subtype"] = np.where(
            adata.obs.excluded,
            "",
            np.where(
                adata.obs[label_col].isin(rollup),
                adata.obs[label_col],
                adata.obs["cell_type"],
            ),
        )
    else:
        raise ValueError(f"unknown output_mode {output_mode!r}")

    # Clusters sort numerically ("2" before "10") via the (length, text) key.
    summary = pd.DataFrame({
        "n": key.value_counts(),
        "label": pd.Series(choices).reindex(sorted(clusters, key=lambda c: (len(c), c))),
    }).dropna(subset=["label"])
    summary["excluded"] = summary["label"].eq("EXCLUDE")
    summary["coarse"] = summary["label"].map(lambda l: rollup.get(l, l))
    summary.index.name = "cluster"

    n_ex = int(adata.obs.excluded.sum())
    print(f"  applied {len(choices)} choices at {res}: "
          f"{adata.n_obs - n_ex:,} kept, {n_ex:,} EXCLUDE")
    return summary


# Write ----

def write_atlas(adata, out_h5ad, lineage_dir, ladder_dir, slug: dict, species: str,
                label_col: str = "lineage"):
    """Annotated object + the lineage subsets 04b consumes + the leiden ladder."""
    out_h5ad = Path(out_h5ad)
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    lineage_dir, ladder_dir = Path(lineage_dir), Path(ladder_dir)
    lineage_dir.mkdir(parents=True, exist_ok=True)
    ladder_dir.mkdir(parents=True, exist_ok=True)

    # Each lineage label must map to a file slug in the spec, or 04b has no input name.
    labels = set(adata.obs.loc[~adata.obs.excluded, label_col])
    unknown = labels - set(slug)
    if unknown:
        raise ValueError(f"labels in the CSV with no slug in the spec: {sorted(unknown)}. "
                         f"Add them to atlas.<species>.labels or fix the CSV.")

    # The annotated object drops EXCLUDE cells and the notebook scoring columns;
    # lineage categories follow the spec's order.
    ann = adata[~adata.obs.excluded].copy()
    for c in [c for c in ann.obs if c.startswith(("score_", "ref_"))] + ["label"]:
        if c in ann.obs:
            del ann.obs[c]
    ann.obs[label_col] = pd.Categorical(
        ann.obs[label_col], categories=[l for l in slug if l in labels])
    ann.write_h5ad(out_h5ad)
    print(f"  wrote {out_h5ad}  ({ann.n_obs:,} cells)")

    # One subset per lineage, named by slug, for reintegrate_lineage.
    rows = []
    for lin in ann.obs[label_col].cat.categories:
        sub = ann[ann.obs[label_col] == lin].copy()
        if sub.n_obs == 0:
            continue
        p = lineage_dir / f"{species}_{slug[lin]}.h5ad"
        sub.write_h5ad(p)
        rows.append({"lineage": lin, "n": sub.n_obs, "file": p.name})
    print(pd.DataFrame(rows).set_index("lineage").to_string() if rows else "  (no subsets)")

    # The Leiden ladder as CSV, one file per resolution and one wide file with every
    # resolution plus the lineage label, written for all cells including EXCLUDE.
    base = pd.DataFrame({"cell": adata.obs_names})
    for c in ("sample", "cell_barcode", "timepoint"):
        if c in adata.obs:
            base[c] = adata.obs[c].astype(str).values
    res_cols = sorted([c for c in adata.obs if c.startswith("leiden_scVI_")],
                      key=lambda c: float(c.rsplit("_", 1)[1]))
    for col in res_cols:
        base.assign(cluster=adata.obs[col].astype(str).values).to_csv(
            ladder_dir / f"leiden_res{col.rsplit('_', 1)[1]}.csv", index=False)
    wide = base.copy()
    for col in res_cols:
        wide[col] = adata.obs[col].astype(str).values
    wide[label_col] = adata.obs[label_col].astype(str).values
    wide.to_csv(ladder_dir / "leiden_all_resolutions.csv", index=False)
    print(f"  wrote {len(res_cols)} ladder files -> {ladder_dir}")
    return ann


def write_fine(adata, out_h5ad):
    """Cleaned per-lineage object for 04d_merge (EXCLUDE dropped)."""
    import os
    out_h5ad = Path(out_h5ad)
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)

    # Existing objects under a 'migrated' directory are never overwritten.
    if out_h5ad.exists() and "migrated" in out_h5ad.parts:
        raise FileExistsError(f"refusing to overwrite staged annotation object: {out_h5ad}")

    # Only the canonical columns leave this step; the notebook scoring columns and
    # the internal working labels stay behind.
    out = adata[~adata.obs.excluded].copy()
    internal = [
        c for c in out.obs
        if c.startswith(("ref_", "score_"))
        or c in {"celltype_ref", "lineage", "excluded", "label", "asn_lbl",
                 "cluster_lbl", "coarse_refined", "fine_refined", "subtype",
                 "annotation_level"}
    ]
    for c in internal:
        if c in out.obs:
            del out.obs[c]

    # Write to a temporary name and rename, so a killed job cannot leave a partial
    # h5ad that Snakemake would treat as complete.
    tmp = out_h5ad.with_name(f".{out_h5ad.name}.{os.getpid()}.tmp.h5ad")
    out.write_h5ad(tmp)
    os.replace(tmp, out_h5ad)
    print(f"  wrote {out_h5ad}  ({out.n_obs:,} cells)")
    return out


# Entry points ----

def _run(level, species, lineage, res, in_h5ad, csv, spec_path, vocab_path,
         repo, out_h5ad):
    import scanpy as sc, yaml
    spec = yaml.safe_load(open(spec_path))
    vocab = yaml.safe_load(open(vocab_path))
    repo = Path(repo)
    print(f"apply_annotation [{level}] {species}"
          f"{'/' + lineage if lineage else ''} | {Path(in_h5ad).name} | {res}")

    adata = sc.read_h5ad(in_h5ad)
    choices = load_choices(csv)

    # Atlas level labels lineages and has no rollup; fine level writes canonical
    # cell_type/cell_subtype through the vocabulary rollup and validates them.
    if level == "atlas":
        cfg = spec["atlas"][species]
        apply_map(adata, res, choices, rollup={}, output_mode="atlas")
        write_atlas(adata, out_h5ad,
                    repo / "data/interim/lineages",
                    repo / "results/tables/clusters" / species,
                    slug=cfg["labels"], species=species)
    else:
        rollup = build_vocabulary_rollup(vocab)
        apply_map(adata, res, choices, rollup=rollup, output_mode="fine")
        validate_canonical_annotations(adata[~adata.obs.excluded], vocab,
                                       context=f"{species}/{lineage}")
        adata.uns["annotation_schema_version"] = vocab["schema_version"]
        adata.uns["display_names"] = vocab["display_names"]
        write_fine(adata, out_h5ad)


def _from_snakemake(sm):
    p = sm.params
    _run(level=p.level, species=p.species, lineage=getattr(p, "lineage", None),
         res=p.res, in_h5ad=sm.input.h5ad, csv=sm.input.csv,
         spec_path=getattr(sm.input, "spec", None)
                   or "resources/lineage_spec.yaml",
         vocab_path=getattr(sm.input, "vocab", None)
                    or "resources/annotation_vocabulary.yaml",
         repo=p.repo, out_h5ad=sm.output.h5ad)


def _from_cli(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["atlas", "fine"], required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--lineage")
    ap.add_argument("--res", required=True)
    ap.add_argument("--in-h5ad", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--spec", default="resources/lineage_spec.yaml")
    ap.add_argument("--vocab", default="resources/annotation_vocabulary.yaml")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    _run(a.level, a.species, a.lineage, a.res, a.in_h5ad, a.csv, a.spec,
         a.vocab, a.repo, a.out)


if "snakemake" in globals():
    _from_snakemake(globals()["snakemake"])  # noqa: F821
elif __name__ == "__main__":
    _from_cli(sys.argv[1:])
