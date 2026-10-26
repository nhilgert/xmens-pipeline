# xmens-pipeline

Snakemake pipeline for the computational analysis in Çevrim et al.,
*Induction of menstruation in mice reveals the regulation of menstrual shedding*
(bioRxiv 2025.10.08.681007; revision in preparation).

Slide-tags single-nucleus spatial transcriptomics of the endometrium in a mouse
model of induced menstruation (six timepoints) and in human endometrium (secretory
and menstrual phases). The pipeline takes CellBender-corrected counts and
Slide-tags coordinates through QC, scVI integration, clustering, annotation,
pseudotime and cross-species comparison. Data objects are not included.

## Layout

```
workflow/Snakefile          entry point
workflow/rules/             one file per stage, numbered in run order
workflow/scripts/           the code each rule runs
config/config.yaml          samples, references, parameters, cluster paths
resources/annotations/      cluster -> label tables (the annotation record)
resources/lineage_spec.yaml resolution, markers and label provenance per cell class
resources/annotation_vocabulary.yaml   allowed labels, display names, expected counts
profiles/slurm/             SLURM executor settings
envs/                       conda environments for SCENIC, SAMap, DoubletDetection
```

## Stages

| Stage | Rules | Step |
|---|---|---|
| 00–01 | `00*`, `01*` | CellBender, Slide-tags coordinates, per-sample QC, concatenation per species |
| 02 | `02_integrate` | scVI integration, kNN graph, Leiden at 21 resolutions, UMAP |
| 04 | `04_annotate` | Major cell classes: committed table applied at Leiden resolution 2.5 |
| 04b–04d | `04b`–`04d` | Per-class scVI re-integration, subtypes from committed tables at resolution 3.0, merge |
| 04e–04g | `04e`–`04g` | Final embedding, spatial-coordinate QC, promotion to a dated release with sha256 manifest |
| 05–24 | `05`–`24` | Marker DE, time-informed diffusion pseudotime (moscot + CellRank), pseudotime genes (mutual information), SAMap alignment and transfer, SCENIC regulons, decidual-zone scores, in vitro comparison, figures |
| 98 | `98_clustree` | Off the critical path: clustering-stability tables |

The spine (00–04g) produces one frozen release per species; every analysis arm
reads only that release.

## How annotation works

No label is computed inside the pipeline. Each annotation step looks up a committed
table in `resources/annotations/` (one row per Leiden cluster: cluster, label,
cell count, resolution, curator note) with `workflow/scripts/atlas/apply_annotation.py`.
The tables came from marker review of deliberately over-clustered data, with the
marker panels recorded in `resources/lineage_spec.yaml`.

## Running

```bash
uv sync                                  # Python environment from uv.lock
snakemake --profile profiles/slurm -n    # dry run
snakemake --profile profiles/slurm       # full run on SLURM
```

Paths in `config/config.yaml` point at the authors' cluster and must be edited for
another site. scVI training is deterministic for a fixed seed only within one GPU
class (the released clusterings were made on NVIDIA A40s); the cluster assignments
and annotation tables are deposited with the data, so labels do not depend on
retraining.

## Data

Raw data and processed objects will be deposited at GEO (mouse) and Harvard
Dataverse (human) with the published paper.

## Citation

Çevrim Ç. et al. Induction of menstruation in mice reveals the regulation of
menstrual shedding. bioRxiv (2025). https://doi.org/10.1101/2025.10.08.681007

MIT license.
