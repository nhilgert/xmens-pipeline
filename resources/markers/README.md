# resources/markers

The marker panels the atlas-level clusters were read against.

## `panel_{species}.tsv` — inputs (hand-curated)

The published coarse cell-type marker panels from Cevrim et al. 2025
(bioRxiv 2025.10.08.681007), as drawn in the Figure 1 dotplots. The underlying
markers are canonical endometrial markers drawn from Kirkwood et al. 2021,
Kirkwood et al. 2022, Wang et al. 2020, and Winkler et al. 2024 (the sources
cited in the Methods). Named by `atlas.<species>.panel` in `resources/lineage_spec.yaml`.

Columns: `celltype`, `role` (`lineage` | `context`), `nested_within`, `genes`
(comma-separated). Row order sets dotplot row order.

Two things these encode that a flat gene list cannot:

- **`role`** separates markers used for *assignment* from markers used only for
  *display*. `PGR`/`ESR1` are expressed across epithelium, fibroblasts, decidual
  cells, SMC and perimetrium; they appear in the published figure for context and
  must never contribute to a cell-type call.
- **`nested_within`** records that decidual cells are a differentiated state of
  endometrial fibroblasts and therefore express the fibroblast markers *as well
  as* their own. Naive argmax over marker-set scores mislabels them, and which
  set wins depends on set size rather than biology.

The mouse and human panels are **not** translations of each other — the
published human panel uses different markers, and the mouse decidual marker
`Prl8a2` has no human ortholog (rodent-expanded placental prolactin family).
