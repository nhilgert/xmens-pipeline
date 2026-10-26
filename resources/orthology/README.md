Compara homology dump goes here (record Ensembl release + download date).

## ensembl_human_mouse_components.tsv
- Source: Ensembl 116 Compara human-mouse homologies (protein-coding), as built in
  /n/lab_storage/eddy_lab/users/nhilgert/human_ivd_repro/orthogroups/ (see ORTHOGROUP_DEFINITION.md there).
- File date 2026-08-07; copied here 2026-09-10 (byte-identical, sha256 recorded in git).
- 36,334 rows = 17,688 human + 18,646 mouse genes; homology_type: one2one 32,628 / one2many 2,638 / many2many 1,068.
- Pipeline use: LIANA cross-species comparison (rule liana_crossspecies) and mi_dpt overlap both restrict to
  homology_type == ortholog_one2one -> 16,275 human-mouse pairs. No manual overrides (decision 2026-09-10).
