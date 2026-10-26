#!/usr/bin/env python3
"""Generate the private stage-04c (per-lineage) re-annotation notebooks.

Reads the `species:` section of resources/lineage_spec.yaml and writes one
notebooks/curation/04c_reannotate_{species}_{lineage}.ipynb per (species,
lineage), assembled from the code-cell bodies defined below.

Each generated notebook is private scaffolding: the analyst runs it, reads the
diagnostic table and dotplots, edits only the DECISION cell, and the notebook
then writes
    resources/annotations/{species}_{lineage}.csv     cluster,label (committed)
    data/interim/reannotated/{species}_{lineage}.h5ad cleaned object for 04d_merge
The CSV is the durable record, and tools/compile_denovo.py turns it into the
public de-novo notebook. The scaffolding notebooks are regenerable and gitignored.

Usage:
    python tools/gen_04c_notebooks.py                 # create missing only
    python tools/gen_04c_notebooks.py --force         # overwrite existing
    python tools/gen_04c_notebooks.py --only mouse:epithelial
"""
import argparse, json, sys, datetime
from pathlib import Path
import yaml, nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

ap = argparse.ArgumentParser()
ap.add_argument("--spec", default="resources/lineage_spec.yaml")
ap.add_argument("--outdir", default="notebooks/curation")
ap.add_argument("--repo", default="/n/eddy_lab/users/nhilgert/xmens_final")
ap.add_argument("--force", action="store_true", help="overwrite existing notebooks")
ap.add_argument("--only", default=None, help="species:lineage")
a = ap.parse_args()

spec = yaml.safe_load(open(a.spec))
DEF, OUT = spec.get("defaults", {}), Path(a.outdir)
OUT.mkdir(parents=True, exist_ok=True)
STAMP = datetime.date.today().isoformat()  # written into each notebook's config cell


# Notebook cell bodies ----
# Each string below is one code cell of the generated notebook, in the order
# they appear there. The config cell that build() writes first defines SPECIES,
# LINEAGE, RES, COARSE, REF, REPO, REPO_SUB, MARKERS, UMAP_GENES, QC_COLS, AUTO,
# FINE, COLLAPSE, LABELS and ROLLUP, which these bodies reference as globals.
# Anything that looks like a comment inside the strings belongs to the notebook.

BODY_IMPORTS = '''import warnings; warnings.filterwarnings("ignore")
import scanpy as sc, pandas as pd, numpy as np
from pathlib import Path
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 140)

REPO, REPO_SUB = Path(REPO), Path(REPO_SUB)
LATENT = REPO/"data/interim"/f"latent_{SPECIES}_{LINEAGE}.h5ad"
ANNDIR = REPO/"resources/annotations";            ANNDIR.mkdir(parents=True, exist_ok=True)
OUTDIR = REPO/"data/interim/reannotated";         OUTDIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = ANNDIR/f"{SPECIES}_{LINEAGE}.csv"       # <- the committed record
H5_OUT  = OUTDIR/f"{SPECIES}_{LINEAGE}.h5ad"
print(f"{SPECIES} {LINEAGE} | {LATENT.name} | {RES} | fine={FINE['mode']}")
print(f"  choices -> {CSV_OUT}")
print(f"  object  -> {H5_OUT}")'''

BODY_LOAD = '''adata = sc.read_h5ad(LATENT)
assert RES in adata.obs, f"{RES} not in obs; have {[c for c in adata.obs if c.startswith('leiden')]}"
adata.obs[RES] = adata.obs[RES].astype("category")
adata.obs[RES] = adata.obs[RES].cat.reorder_categories(
    sorted(adata.obs[RES].cat.categories, key=int))
print(f"{adata.n_obs:,} cells | {adata.obs[RES].nunique()} clusters at {RES}")'''

BODY_JOIN = '''def _canon(s):
    s = s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return s.str.replace("bleeding", "menstrual", regex=False)   # 4dpi naming quirk
def _key(o):
    return _canon(o["sample"]) + "||" + \\
           o["cell_barcode"].astype(str).str.replace(r"-\\d+$", "", regex=True)

mykey = _key(adata.obs)
them = sc.read_h5ad(REPO_SUB/REF, backed="r").obs.copy(); them["k"] = _key(them)
r = mykey.map(them.set_index("k")["cell_type_annotations"])
adata.obs["ref_type"] = r.astype(object).where(r.notna(), "unlabeled").values
# collapsed into OUR lineage vocabulary -- the vote uses this, agreement uses ref_type
adata.obs["ref_lineage"] = adata.obs["ref_type"].replace(COLLAPSE).values
nj = int((adata.obs.ref_type != "unlabeled").sum())
print(f"coarse join: {nj:,}/{adata.n_obs:,} ({100*nj/adata.n_obs:.1f}%)")
print(adata.obs.ref_type.value_counts().to_string())

adata.obs["ref_fine"] = "unlabeled"
if FINE["mode"] == "csv":
    p = ANNDIR/FINE["csv"]
    if p.exists():
        f = pd.read_csv(p); f["k"] = _key(f)
        rf = mykey.map(f.set_index("k")[FINE["col"]])
        adata.obs["ref_fine"] = rf.astype(object).where(rf.notna(), "unlabeled").values
        nf = int((adata.obs.ref_fine != "unlabeled").sum())
        print(f"\\nfine join ({FINE['csv']}): {nf:,}/{adata.n_obs:,} ({100*nf/adata.n_obs:.1f}%)")
        print(adata.obs.ref_fine.value_counts().to_string())
    else:
        print(f"\\nNOTE fine source {p.name} not on disk -- suggestions fall back to coarse.")'''

BODY_TABLE = '''genes = [g for grp in MARKERS.values() for g in grp if g in adata.var_names]
X = pd.DataFrame(adata[:, genes].X.toarray(), columns=genes, index=adata.obs.index)
gb = adata.obs[RES]

n     = gb.value_counts().rename("n")
dfrac = (adata.obs.groupby(RES, observed=True)["doubletdetection_predicted_doublet"]
         .mean().round(2).rename("dbl_frac")
         if "doubletdetection_predicted_doublet" in adata.obs else None)
qc    = adata.obs.groupby(RES, observed=True)[[c for c in QC_COLS if c in adata.obs]].median().round(2)
tp    = (adata.obs.groupby(RES, observed=True)["timepoint"]
         .agg(lambda s: f"{s.value_counts(normalize=True).idxmax()} "
                        f"{s.value_counts(normalize=True).max():.0%}").rename("top_tp")
         if "timepoint" in adata.obs else None)

def _top(s):
    v = s.value_counts(normalize=True)
    return f"{v.index[0]} {v.iloc[0]:.2f}" if len(v) else ""
vote  = adata.obs.groupby(RES, observed=True)["ref_lineage"].agg(_top).rename("coarse_vote")
fvote = (adata.obs.groupby(RES, observed=True)["ref_fine"].agg(_top).rename("fine_vote")
         if (adata.obs.ref_fine != "unlabeled").any() else None)

# per-cluster mean of each marker GROUP (this is what the fine=markers rule uses)
grp = pd.DataFrame({k: X[[g for g in v if g in X]].mean(axis=1)
                    for k, v in MARKERS.items() if any(g in X for g in v)}).join(gb) \\
        .groupby(RES, observed=True).mean().round(2)
grp.columns = [f"[{c}]" for c in grp.columns]
mk = X.join(gb).groupby(RES, observed=True)[genes].mean().round(2)

TAB = pd.concat([x for x in [n, dfrac, qc, tp, vote, fvote, grp, mk] if x is not None], axis=1)
print("marker groups:", {k: v for k, v in MARKERS.items()}); print()
print(TAB.to_string())'''

BODY_PLOTS = '''plot_sets = {k: [g for g in v if g in adata.var_names] for k, v in MARKERS.items()}
plot_sets = {k: v for k, v in plot_sets.items() if v}
sc.pl.dotplot(adata, plot_sets, groupby=RES, standard_scale="var",
              figsize=(10, max(6, 0.22*adata.obs[RES].nunique())), show=True)
cols = [g for g in UMAP_GENES if g in adata.var_names]
cols += ["timepoint"] if "timepoint" in adata.obs else []
sc.pl.embedding(adata, "X_umap_scVI", color=cols + [RES], ncols=3,
                legend_loc="on data", legend_fontsize=5, show=True)'''

BODY_SUGGEST = '''def suggest_labels():
    """Auto-suggest one label per cluster: majority vote + fine rule + auto-EXCLUDE.
    Nothing here is final -- it only prefills CHOICES below."""
    out = {}
    for c in TAB.index:
        row = TAB.loc[c]
        # (1) auto-EXCLUDE on QC evidence, never on UMAP position
        ds = row.get("doubletdetection_doublet_score", 0)
        df = row.get("dbl_frac", 0)
        if (ds is not None and ds >= AUTO["doublet_score"]) or \\
           (df is not None and df >= AUTO["doublet_frac"]):
            out[c] = "EXCLUDE"; continue
        # (2) throwback: deposited vote says a different coarse lineage
        cv = str(row.get("coarse_vote", "")).rsplit(" ", 1)
        vote_lab, vote_p = (cv[0], float(cv[1])) if len(cv) == 2 else ("", 0.0)
        if vote_lab == "unlabeled" and vote_p >= 0.5:
            out[c] = "EXCLUDE"; continue
        if vote_lab and vote_lab != COARSE and vote_p >= 0.5:
            out[c] = vote_lab; continue
        # (3) fine label
        if FINE["mode"] == "markers":
            sc_ = {lab: row.get(f"[{g}]", 0) for lab, g in FINE["from_groups"].items()}
            best = max(sc_, key=sc_.get)
            out[c] = best if sc_[best] > 0 else COARSE
        elif FINE["mode"] == "csv" and "fine_vote" in TAB.columns:
            fv = str(row["fine_vote"]).rsplit(" ", 1)
            out[c] = fv[0] if len(fv) == 2 and fv[0] != "unlabeled" and float(fv[1]) >= 0.5 else COARSE
        else:
            out[c] = COARSE
    return out

SUGGESTED = suggest_labels()
print("suggested (edit via OVERRIDES in the next cell):")
for c in TAB.index:
    print(f'    "{c}": "{SUGGESTED[c]}",'
          + ("   # <- auto-EXCLUDE" if SUGGESTED[c] == "EXCLUDE" else ""))'''

BODY_DECISION = '''# =============================================================================
#  >>>  THE DECISION CELL  --  this is the only cell you edit  <<<
# =============================================================================
# CHOICES is the exact data structure written to
#     resources/annotations/{SPECIES}_{LINEAGE}.csv   (columns: cluster,label)
# It starts from SUGGESTED (majority vote + marker/CSV fine rule + auto-EXCLUDE).
# Put every hand call in OVERRIDES -- that keeps your judgment explicit and
# reviewable. Read them off the diagnostic table, never off UMAP position.
#
#   label must be one of LABELS, or "EXCLUDE".
#   LABELS = see the config cell.
# =============================================================================

CHOICES = dict(SUGGESTED)

OVERRIDES = {
    # "6":  "EXCLUDE",              # Flt1+Pecam1 -> vascular doublet
    # "22": "glandular epithelium", # transitional, clean QC, faint Foxa2
}

CHOICES.update(OVERRIDES)

# ---- validate ---------------------------------------------------------------
_bad = {c: v for c, v in CHOICES.items() if v not in LABELS + ["EXCLUDE"]}
assert not _bad, f"labels not in LABELS + EXCLUDE: {_bad}"
_missing = set(TAB.index) - set(CHOICES)
assert not _missing, f"clusters with no choice: {sorted(_missing, key=int)}"
print(f"{len(CHOICES)} clusters | {sum(v=='EXCLUDE' for v in CHOICES.values())} EXCLUDE "
      f"| {len(OVERRIDES)} manual override(s)")
print(pd.Series(CHOICES).value_counts().to_string())'''

BODY_APPLY = '''adata.obs["label"]    = adata.obs[RES].map(CHOICES).astype(str)
adata.obs["excluded"] = adata.obs["label"].eq("EXCLUDE").values
adata.obs["coarse_refined"] = np.where(adata.obs.excluded, "EXCLUDE",
                                       adata.obs["label"].map(lambda l: ROLLUP.get(l, l)))
adata.obs["fine_refined"] = np.where(adata.obs.excluded, "",
                              np.where(adata.obs["label"].isin(ROLLUP), adata.obs["label"], ""))

kept = ~adata.obs.excluded.values
cr = adata.obs.coarse_refined.astype(str); rt = adata.obs.ref_type.astype(str)
tr = rt.eq("unlabeled").values
print(f"kept {int(kept.sum()):,}/{adata.n_obs:,} ({100*kept.mean():.1f}%) | "
      f"EXCLUDE {int((~kept).sum()):,}")
both = kept & ~tr
if both.sum():
    print(f"coarse agreement vs deposited: {100*(cr[both].values==rt[both].values).mean():.2f}% "
          f"(on {int(both.sum()):,} cells)")
tp_ = int(((~kept) & tr).sum()); fp = int(((~kept) & ~tr).sum()); fn = int((kept & tr).sum())
if tp_+fp: print(f"EXCLUDE precision {tp_/(tp_+fp):.3f}", end="   ")
if tp_+fn: print(f"recall {tp_/(tp_+fn):.3f}  (low recall is fine if their removals were diffuse)")
if adata.obs.fine_refined.ne("").any():
    print("\\nfine labels:"); print(adata.obs.loc[kept, "fine_refined"].value_counts().to_string())
sc.pl.embedding(adata, "X_umap_scVI", color="label", show=True)'''

BODY_WRITE = '''# 1. the committed record -- minimal on purpose (cluster,label)
pd.DataFrame({"cluster": list(CHOICES), "label": list(CHOICES.values())}) \\
  .sort_values("cluster", key=lambda s: s.astype(int)) \\
  .to_csv(CSV_OUT, index=False)
print(f"wrote {CSV_OUT}  ({len(CHOICES)} rows)")

# 2. the cleaned object for 04d_merge
out = adata[~adata.obs.excluded].copy()
for c in ["ref_type", "ref_lineage", "ref_fine", "label"]:
    if c in out.obs: del out.obs[c]
out.write_h5ad(H5_OUT)
print(f"wrote {H5_OUT}  ({out.n_obs:,} cells)")
print(f"\\nnext:  python tools/compile_denovo.py --species {SPECIES} --res {RES} \\\\\\n"
      f"           --csv resources/annotations/{SPECIES}_{LINEAGE}.csv \\\\\\n"
      f"           --out notebooks/public/04c_annotate_{SPECIES}_{LINEAGE}_denovo.ipynb \\\\\\n"
      f"           --level fine --lineage {LINEAGE}")'''



# Assemble one notebook ----
def build(species, lineage, sp_cfg, lin):
    """Notebook for one (species, lineage) from its spec entry; lineage keys override defaults."""
    res   = lin.get("res", DEF.get("res"))
    auto  = {**DEF.get("auto_exclude", {}), **lin.get("auto_exclude", {})}
    qc    = lin.get("qc_cols", DEF.get("qc_cols", []))
    fine  = lin.get("fine", {"mode": "none"})
    collapse = {**DEF.get("collapse", {}), **lin.get("collapse", {})}
    coarse = lin["coarse"]

    # Fine labels and how each rolls up to a coarse label. Marker-rule lineages
    # roll every fine label up to this lineage; CSV-transferred lineages may send
    # a fine label to another coarse label (rollup_all, or by name prefix).
    if fine["mode"] == "markers":
        fine_labels = list(fine.get("from_groups", {}))
        rollup = {lab: coarse for lab in fine_labels}
    elif fine["mode"] == "csv":
        fine_labels = list(fine.get("labels", []))
        rollup = {}
        for lab in fine_labels:
            if "rollup_all" in fine:
                rollup[lab] = fine["rollup_all"]
                continue
            for pref, tgt in fine.get("rollup_prefix", {}).items():
                if str(lab).startswith(pref):
                    rollup[lab] = tgt
                    break
            else:
                rollup[lab] = coarse
    else:
        fine_labels, rollup = [], {}

    # A cluster whose deposited vote belongs to another lineage ("throwback") may
    # be labelled with any coarse label of the species, so all of them are legal.
    others = sorted({d["coarse"] for d in sp_cfg["lineages"].values()} - {coarse})
    labels = [coarse] + fine_labels + sorted(set(rollup.values()) - {coarse}) + others

    # The config cell: every per-lineage value the cell bodies above refer to.
    cfg = f'''# ---- generated {STAMP} from resources/lineage_spec.yaml -- edit the SPEC, not this
SPECIES, LINEAGE = {json.dumps(species)}, {json.dumps(lineage)}
RES    = {json.dumps(res)}
COARSE = {json.dumps(coarse)}
REF    = {json.dumps(sp_cfg["ref"])}
REPO     = {json.dumps(a.repo)}
REPO_SUB = "/n/eddy_lab/users/nhilgert/decidualization/revisions/adatas/submission"

MARKERS = {json.dumps(lin["markers"], indent=4)}
UMAP_GENES = {json.dumps(lin.get("umap_genes", []))}
QC_COLS = {json.dumps(qc)}
AUTO    = {json.dumps(auto)}
FINE    = {json.dumps(fine)}
COLLAPSE = {json.dumps(collapse, indent=4)}

# every label CHOICES may use (fine labels first, then coarse throwback targets)
LABELS = {json.dumps(labels, indent=4)}
# fine label -> coarse label (for the rolled-up coarse_refined column)
ROLLUP = {json.dumps(rollup, indent=4)}'''

    # Markdown headers interleaved with the code cells, in notebook order.
    fine_desc = {"none": "coarse only", "markers": "fine labels by marker rule",
                 "csv": f"fine labels transferred from {fine.get('csv','?')}"}[fine["mode"]]
    C = [
      md(f"""# 04c — {species} {lineage}: inspect & annotate  *(private scaffolding)*

Generated {STAMP} from `resources/lineage_spec.yaml` — **regenerable, gitignored.**
Edit the spec and re-run `tools/gen_04c_notebooks.py` rather than hand-editing this.

Resolution `{res}` · {fine_desc}.

**How to use:** run everything, study the diagnostic table (§4) and dotplots (§5),
then edit **only the DECISION cell (§7)**. Exclusions come off doublet score /
contamination — *not* UMAP position. Running §9 writes the two artifacts:
`resources/annotations/{species}_{lineage}.csv` (committed) and the cleaned
`.h5ad` for `04d_merge`."""),
      md("## 1. Config *(generated)*"), code(cfg),
      code("from pathlib import Path\n" + BODY_IMPORTS),
      md("## 2. Load the reintegrated object"), code(BODY_LOAD),
      md("## 3. Join the deposited labels"), code(BODY_JOIN),
      md("""## 4. Diagnostic table — the decision surface

Per cluster: size, doublet fraction/score, QC medians, top timepoint, the
deposited coarse (and fine) vote, `[group]` marker-group means, then individual
markers. Everything you need to fill the DECISION cell."""), code(BODY_TABLE),
      md("## 5. Dotplot + UMAP"), code(BODY_PLOTS),
      md("""## 6. Suggested labels *(prefill only)*

Majority vote + the lineage's fine rule + auto-EXCLUDE on doublet evidence.
Printed paste-ready; nothing is committed until §7."""), code(BODY_SUGGEST),
      md("""## 7. ★ DECISION — the only cell you edit ★

`CHOICES` is exactly what lands in the CSV. Start from `SUGGESTED`, put every
hand call in `OVERRIDES`."""), code(BODY_DECISION),
      md("## 8. Apply + check against the deposited object"), code(BODY_APPLY),
      md("## 9. Write the choices CSV *(committed)* + cleaned object"), code(BODY_WRITE),
    ]

    nb = new_notebook(cells=C)
    nb.metadata["kernelspec"] = {"display_name":"Python 3","language":"python","name":"python3"}
    nb.metadata["language_info"] = {"name":"python"}
    return nb


# Write the notebooks ----
made = skipped = 0
for species, sp_cfg in spec["species"].items():
    for lineage, lin in sp_cfg["lineages"].items():
        if a.only and a.only != f"{species}:{lineage}":
            continue
        p = OUT/f"04c_reannotate_{species}_{lineage}.ipynb"
        if p.exists() and not a.force:
            print(f"  skip (exists)  {p.name}")
            skipped += 1
            continue

        nb = build(species, lineage, sp_cfg, lin)
        nbf.validate(nb)
        nbf.write(nb, str(p))
        print(f"  {'overwrote' if a.force else 'created  '}    {p.name}")
        made += 1

print(f"\n{made} written, {skipped} skipped -> {OUT}")
