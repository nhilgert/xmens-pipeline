# resources/clusters

The cluster assignments the annotation tables refer to, one file per species
(`{species}_clusters.csv.gz`, one row per nucleus that passed QC).

- `obs_name`: the nucleus, as named in the deposited objects.
- `leiden_scVI_2.5`: the whole-tissue cluster. `resources/annotations/{species}_lineage.csv`
  maps it to a lineage or EXCLUDE.
- `leiden_scVI_3.0`: the cluster within that lineage, read by
  `resources/annotations/{species}_{lineage}.csv`; empty for nuclei whose whole-tissue
  cluster is EXCLUDE. Cluster ids are only meaningful within one lineage.

Both come from the July 2026 scVI fits (seed 0): the whole-tissue fits as regenerated
bit-for-bit on NVIDIA A40, the lineage fits as originally written. scVI is deterministic
only within one GPU class, so these files, not a retrain, are what the tables apply to.
Written by `tools/export_cluster_assignments.py`; `tools/check_labels_from_clusters.py`
applies the tables to them and compares the result with a release.
