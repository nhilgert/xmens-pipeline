"""migrate_release_vocabulary (04g): relabel a frozen release forward to the
current annotation vocabulary, so a vocabulary change reaches a new release
without re-running annotation (04 to 04e).

Reads the objects of config vocab_migration.from_release, annotation_vocabulary.yaml
and, for each subtype split in config vocab_migration.subtype_splits, the pinned
clustering and its decision CSV. Writes the relabeled objects and a TSV of every
label value changed with the number of cells. Expression, embeddings and every other
obs column are left as they are; the script fails if anything besides the label
columns differs, or if a split does not reproduce the counts recorded in the
vocabulary's content task.

Outside Snakemake, --check prints the report without writing objects:
    python migrate_vocabulary.py --vocab V [--config config/config.yaml] \\
        [--report R.tsv] --check SRC.h5ad [SRC.h5ad ...]
"""
import argparse
import hashlib
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml

# The annotation checker and CSV loader live in workflow/scripts/atlas/; Snakemake
# script directives cannot import sibling directories without extending the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atlas"))
from apply_annotation import load_choices, validate_canonical_annotations  # noqa: E402

LABEL_COLUMNS = ("cell_type", "cell_subtype")
HASH_BLOCK_BYTES = 1 << 24  # 16 MiB per read while hashing a multi-GB h5ad


def rename_column(adata, column: str, aliases: dict) -> list[dict]:
    """Rename superseded values in one obs column; return one report row per value."""
    values = adata.obs[column]
    counts = values.astype(str).value_counts()
    renamed = [
        {"column": column, "old": old, "new": new, "n_cells": int(counts[old])}
        for old, new in aliases.items()
        if old in counts.index
    ]
    if not renamed:
        return []

    # Renaming categories keeps their order, so positional uns colors stay aligned.
    # A many-to-one alias would merge two categories and silently drop a color.
    if isinstance(values.dtype, pd.CategoricalDtype):
        new_categories = [aliases.get(c, c) for c in values.cat.categories]
        if len(set(new_categories)) != len(new_categories):
            raise ValueError(f"{column}: aliases would merge categories {new_categories}")
        adata.obs[column] = values.cat.rename_categories(new_categories)
    else:
        adata.obs[column] = values.astype(str).replace(aliases)

    return renamed


def rename_uns_keys(adata, aliases: dict) -> list[str]:
    """Rename superseded label keys inside dict-valued uns entries (per-label colors)."""
    touched = []
    for key, value in adata.uns.items():
        if isinstance(value, dict) and any(old in value for old in aliases):
            adata.uns[key] = {aliases.get(k, k): v for k, v in value.items()}
            touched.append(key)
    return touched


def find_superseded_elsewhere(adata, aliases: dict) -> dict:
    """Count superseded strings in obs columns this script does not rename."""
    found = {}
    for column in adata.obs.columns:
        dtype = adata.obs[column].dtype
        is_text = dtype == object or isinstance(dtype, pd.CategoricalDtype)
        if column in LABEL_COLUMNS or not is_text:
            continue
        hits = adata.obs[column].astype(str).isin(list(aliases)).sum()
        if hits:
            found[column] = int(hits)
    return found


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_key(obs: pd.DataFrame) -> pd.Series:
    """sample||barcode, normalized as in merge_reannotated.py (bleeding == menstrual)."""
    sample = obs["sample"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    sample = sample.str.replace("bleeding", "menstrual", regex=False)
    barcode = obs["cell_barcode"].astype(str).str.replace(r"-\d+$", "", regex=True)
    return sample + "||" + barcode


def load_pinned_clustering(task: str, split: dict) -> pd.Series:
    """Cluster id per cell key from the clustering the split's decisions were made on.

    Refuses any file whose sha256 differs from the pinned value, so the split can
    never be applied to a re-trained embedding.
    """
    clusters_path = Path(split["clusters"])
    found_sha = sha256_of(clusters_path)
    if found_sha != split["clusters_sha256"]:
        raise ValueError(
            f"{task}: {clusters_path} sha256 {found_sha} is not the pinned clustering"
        )

    reference = ad.read_h5ad(clusters_path, backed="r")
    ref_obs = reference.obs[["sample", "cell_barcode", split["cluster_column"]]].copy()
    reference.file.close()

    ref_key = cell_key(ref_obs)
    if ref_key.duplicated().any():
        n_dup = int(ref_key.duplicated().sum())
        raise ValueError(f"{task}: {n_dup} duplicate cell keys in {clusters_path}")

    return pd.Series(ref_obs[split["cluster_column"]].astype(str).values, index=ref_key.values)


def split_labels(adata, task: str, split: dict, vocab: dict) -> pd.Series:
    """New cell_subtype per coarse-filled cell of split['cell_type'], from decisions[cluster].

    Fails unless every target cell maps to a kept cluster, every kept cell of the
    clustering is a target cell, and the counts equal the vocabulary content task.
    """
    cell_type = split["cell_type"]
    cluster_of = load_pinned_clustering(task, split)

    decisions = load_choices(split["decisions"])
    allowed = set(vocab["allowed"][cell_type] or []) | {cell_type}
    bad_labels = sorted(set(decisions.values()) - allowed - {"EXCLUDE"})
    if bad_labels:
        raise ValueError(f"{task}: decision labels not allowed under {cell_type}: {bad_labels}")

    # Target cells: still carrying the coarse fill, i.e. not yet given a finer subtype.
    types = adata.obs["cell_type"].astype(str)
    subtypes = adata.obs["cell_subtype"].astype(str)
    target = (types == cell_type) & (subtypes == cell_type)
    target_key = cell_key(adata.obs.loc[target])

    cluster = target_key.map(cluster_of)
    if cluster.isna().any():
        n_missing = int(cluster.isna().sum())
        raise ValueError(
            f"{task}: {n_missing} {cell_type} cells have no cluster in the clustering"
        )

    new_label = cluster.map(decisions)
    n_unkept = int(new_label.isna().sum() + new_label.eq("EXCLUDE").sum())
    if n_unkept:
        raise ValueError(
            f"{task}: {n_unkept} {cell_type} cells sit in clusters without a kept decision"
        )

    # The clustering and the object must cover the same cells, in both directions.
    kept_in_reference = cluster_of.map(decisions).ne("EXCLUDE")
    missing_from_object = set(cluster_of.index[kept_in_reference.values]) - set(target_key.values)
    if missing_from_object:
        raise ValueError(
            f"{task}: {len(missing_from_object)} kept cells of the clustering are "
            f"not coarse-filled {cell_type} cells in the object"
        )

    counts = {label: int(n) for label, n in new_label.value_counts().items()}
    expected_counts = vocab["content_tasks"][task]["expected_counts"]
    expected = {label: int(n) for label, n in expected_counts.items()}
    if counts != expected:
        raise ValueError(f"{task}: split counts {counts} != vocabulary expected_counts {expected}")

    return new_label


def apply_split(adata, task: str, split: dict, vocab: dict) -> list[dict]:
    """Write the split labels into cell_subtype; return one report row per new label."""
    new_label = split_labels(adata, task, split, vocab)

    # New categories are appended; existing category order (and any positional
    # uns colors) is left as it was.
    column = adata.obs["cell_subtype"].copy()
    if isinstance(column.dtype, pd.CategoricalDtype):
        added = sorted(set(new_label) - set(column.cat.categories))
        column = column.cat.add_categories(added)
    column.loc[new_label.index] = new_label.values
    adata.obs["cell_subtype"] = column

    cell_type = split["cell_type"]
    return [
        {"column": "cell_subtype", "old": cell_type, "new": label, "n_cells": int(n)}
        for label, n in sorted(new_label.value_counts().items())
        if label != cell_type
    ]


def check_only_labels_changed(
    adata, obs_before: pd.DataFrame, shape_before, report: list[dict], name: str
):
    """Fail if anything besides the label columns changed, or the report miscounts."""
    if adata.shape != shape_before or not adata.obs_names.equals(obs_before.index):
        raise ValueError(f"{name}: cells or genes changed during migration")

    other = [c for c in obs_before.columns if c not in LABEL_COLUMNS]
    if not adata.obs[other].equals(obs_before[other]):
        raise ValueError(f"{name}: obs columns other than {LABEL_COLUMNS} changed")

    for column in LABEL_COLUMNS:
        n_changed = int((adata.obs[column].astype(str) != obs_before[column].astype(str)).sum())
        n_reported = sum(r["n_cells"] for r in report if r["column"] == column)
        if n_changed != n_reported:
            raise ValueError(
                f"{name}: {column} changed {n_changed} cells, report says {n_reported}"
            )


def migrate(src: Path, vocab: dict, splits: dict) -> tuple[ad.AnnData, list[dict]]:
    """Relabel one release object in memory; return it with the rows of the change table."""
    adata = ad.read_h5ad(src)
    obs_before = adata.obs.copy()
    shape_before = adata.shape

    report = []
    for column in LABEL_COLUMNS:
        aliases = vocab["legacy_aliases"].get(column, {})
        for row in rename_column(adata, column, aliases):
            report.append({"object": src.stem, **row})

    for task, split in splits.items():
        if split["object"] == src.stem:
            for row in apply_split(adata, task, split, vocab):
                report.append({"object": src.stem, **row})

    subtype_aliases = vocab["legacy_aliases"].get("cell_subtype", {})
    uns_touched = rename_uns_keys(adata, subtype_aliases)
    adata.uns["annotation_schema_version"] = vocab["schema_version"]
    if "display_names" in adata.uns:
        adata.uns["display_names"] = vocab["display_names"]

    check_only_labels_changed(adata, obs_before, shape_before, report, src.name)
    validate_canonical_annotations(adata, vocab, context=f"migrated/{src.stem}")

    elsewhere = find_superseded_elsewhere(adata, subtype_aliases)
    n_renamed = sum(r["n_cells"] for r in report)
    print(
        f"{src.stem}: {adata.n_obs:,} cells; renamed {n_renamed:,} label values; "
        f"uns keys renamed {uns_touched}; "
        f"superseded strings left in unmigrated obs columns {elsewhere}",
        flush=True,
    )
    return adata, report


def main(
    sources: list[Path],
    vocab_path: Path,
    splits: dict,
    outputs: list[Path] | None,
    report_path: Path | None,
):
    vocab = yaml.safe_load(open(vocab_path))
    print(
        f"vocabulary {vocab['schema_version']} from {vocab_path}; splits {list(splits)}",
        flush=True,
    )

    unknown = {split["object"] for split in splits.values()} - {src.stem for src in sources}
    if unknown:
        raise ValueError(f"subtype_splits name objects not in this release: {sorted(unknown)}")

    rows = []
    for i, src in enumerate(sources):
        adata, report = migrate(src, vocab, splits)
        rows.extend(report)

        if outputs is not None:
            outputs[i].parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(outputs[i])
            print(f"wrote {outputs[i]}", flush=True)
        del adata

    table = pd.DataFrame(rows, columns=["object", "column", "old", "new", "n_cells"])
    print(table.to_string(index=False), flush=True)
    if report_path is not None:
        table.to_csv(report_path, sep="\t", index=False)


if "snakemake" in globals():
    snk = snakemake  # noqa: F821 (injected by Snakemake)
    sys.stdout = sys.stderr = open(snk.log[0], "w")
    main(
        sources=[Path(p) for p in snk.input.objects],
        vocab_path=Path(snk.input.vocab),
        splits=dict(snk.params.splits),
        outputs=[Path(p) for p in snk.output.objects],
        report_path=Path(snk.output.report),
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, help="config.yaml with vocab_migration.subtype_splits"
    )
    parser.add_argument(
        "--report", type=Path, help="write the change table here; objects are never written"
    )
    parser.add_argument("--check", action="store_true", help="report only; write no objects")
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    if not args.check:
        parser.error("outside Snakemake only --check is supported")

    config = yaml.safe_load(open(args.config)) if args.config else {}
    splits = config.get("vocab_migration", {}).get("subtype_splits", {})
    main(args.sources, args.vocab, splits, outputs=None, report_path=args.report)
