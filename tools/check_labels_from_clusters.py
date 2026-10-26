"""Check that the committed tables, applied to the committed clusters, give the release labels.

Reads resources/clusters/{species}_clusters.csv.gz, resources/annotations/*.csv,
resources/lineage_spec.yaml, resources/annotation_vocabulary.yaml and, from the given
release directory, each species' All_Cells object (obs only). Applies the atlas table to
send every nucleus to its lineage, then each lineage table, with the pipeline's own apply
step (workflow/scripts/atlas/apply_annotation.py), and compares cell_type and cell_subtype
nucleus by nucleus. Writes nothing; prints one line per species and exits 1 on any
difference. Run from the repo root:

    python tools/check_labels_from_clusters.py data/deliverables/<release>
"""
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml

# apply_annotation.py is a Snakemake script, not an installed module; importing it from its
# directory makes this check run the pipeline's apply step rather than a copy of it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow" / "scripts" / "atlas"))
from apply_annotation import apply_map, build_vocabulary_rollup, load_choices  # noqa: E402

RELEASE_DIR = Path(sys.argv[1])
# The All_Cells object of each species, as named in config/config.yaml deliverables.sources.
ALL_CELLS = {"mouse": "Cevrim_XMens_All_Cells.h5ad", "human": "Cevrim_Human_All_Cells.h5ad"}

spec = yaml.safe_load(open("resources/lineage_spec.yaml"))
vocab = yaml.safe_load(open("resources/annotation_vocabulary.yaml"))
rollup = build_vocabulary_rollup(vocab)


def table(species, level, resolution):
    return load_choices(f"resources/annotations/{species}_{level}.csv", resolution)


def labels_from_tables(species):
    """cell_type and cell_subtype of every nucleus the tables keep."""
    clusters = pd.read_csv(f"resources/clusters/{species}_clusters.csv.gz", dtype=str,
                           index_col="obs_name")
    atlas_res = spec["atlas"][species]["res"]
    atlas = ad.AnnData(obs=clusters[[atlas_res]].copy())
    apply_map(atlas, atlas_res, table(species, "lineage", atlas_res), output_mode="atlas")
    lineage_of = atlas.obs["lineage"].map(spec["atlas"][species]["labels"])

    kept = []
    for lineage, lineage_spec in spec["species"][species]["lineages"].items():
        res = lineage_spec.get("res", spec["defaults"]["res"])
        members = lineage_of.index[lineage_of == lineage]
        subset = ad.AnnData(obs=clusters.loc[members, [res]].copy())
        apply_map(subset, res, table(species, lineage, res), rollup=rollup, output_mode="fine")
        kept.append(subset.obs.loc[~subset.obs["excluded"], ["cell_type", "cell_subtype"]])
    return pd.concat(kept)


failed = False
for species in spec["species"]:
    derived = labels_from_tables(species)
    released = ad.read_h5ad(RELEASE_DIR / ALL_CELLS[species], backed="r").obs
    released = released[["cell_type", "cell_subtype"]].astype(str)

    only_released = released.index.difference(derived.index)
    only_derived = derived.index.difference(released.index)
    common = released.index.intersection(derived.index)
    differ = {column: int((derived.loc[common, column] != released.loc[common, column]).sum())
              for column in ("cell_type", "cell_subtype")}
    print(f"{species}: {len(released):,} nuclei in the release, {len(derived):,} from the "
          f"tables; {len(only_released)} only in the release, {len(only_derived)} only from "
          f"the tables; cell_type differs for {differ['cell_type']}, cell_subtype for "
          f"{differ['cell_subtype']}")
    failed |= bool(len(only_released) or len(only_derived) or any(differ.values()))

sys.exit(1 if failed else 0)
