#!/usr/bin/env python3
"""Generate the private stage-04 (whole-tissue lineage) annotation notebooks.

Reads the `atlas:` section of resources/lineage_spec.yaml and writes one
notebooks/curation/04_annotate_{species}.ipynb per species, assembled from the
code-cell bodies defined below. Sibling of gen_04c_notebooks.py, which does the
same per lineage.

Each generated notebook is private scaffolding: the analyst runs it, edits only
the DECISION cell, and the notebook then writes
    resources/annotations/{species}_lineage.csv     cluster,label (committed)
    data/interim/annotated_{species}.h5ad
    data/interim/lineages/{species}_{slug}.h5ad     (input to 04b_reintegrate)
    results/tables/clusters/{species}/leiden_*.csv  (input to the R clustree)
tools/compile_denovo.py --level lineage then compiles the CSV into the public
de-novo notebook.

Usage:
    python tools/gen_04_atlas.py            # create missing only
    python tools/gen_04_atlas.py --force    # overwrite
"""
import argparse, json, datetime, sys
from pathlib import Path
import yaml, nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

ap = argparse.ArgumentParser()
ap.add_argument("--spec", default="resources/lineage_spec.yaml")
ap.add_argument("--outdir", default="notebooks/curation")
ap.add_argument("--repo", default="/n/eddy_lab/users/nhilgert/xmens_final")
ap.add_argument("--force", action="store_true")
ap.add_argument("--only", default=None, help="species")
a = ap.parse_args()

spec = yaml.safe_load(open(a.spec))
if "atlas" not in spec:
    sys.exit("spec has no `atlas:` section")
DEF = spec.get("defaults", {})

OUT = Path(a.outdir)
OUT.mkdir(parents=True, exist_ok=True)
STAMP = datetime.date.today().isoformat()  # written into each notebook's config cell


# Notebook cell bodies ----
# Each string below is one code cell of the generated notebook, in the order
# they appear there. The config cell (built per species in the loop at the
# bottom) defines SPECIES, RES, REF, PANEL_FILE, REPO, REPO_SUB, COLLAPSE,
# LABELS, SLUG, QC_COLS and AUTO, which these bodies reference as globals.
# Anything that looks like a comment inside the strings belongs to the notebook.

IMPORTS = '''from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import scanpy as sc, pandas as pd, numpy as np
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 140)

REPO, REPO_SUB = Path(REPO), Path(REPO_SUB)
LATENT   = REPO/"data/interim"/f"latent_{SPECIES}.h5ad"
PANEL    = REPO/"resources/markers"/PANEL_FILE
ANNDIR   = REPO/"resources/annotations";       ANNDIR.mkdir(parents=True, exist_ok=True)
CSV_OUT  = ANNDIR/f"{SPECIES}_lineage.csv"     # <- the committed record
OUT_FULL = REPO/"data/interim"/f"annotated_{SPECIES}.h5ad"
OUT_LIN  = REPO/"data/interim/lineages"
OUT_CSV  = REPO/"results/tables/clusters"/SPECIES
OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
print(f"{SPECIES} atlas | {LATENT.name} | {RES} | {len(LABELS)} lineages")
print(f"  choices  -> {CSV_OUT}")
print(f"  object   -> {OUT_FULL}")
print(f"  subsets  -> {OUT_LIN}/{SPECIES}_*.h5ad   (feed 04b_reintegrate)")'''

LOAD = '''adata = sc.read_h5ad(LATENT)
assert RES in adata.obs, f"{RES} not in obs"
adata.obs[RES] = adata.obs[RES].astype("category")
adata.obs[RES] = adata.obs[RES].cat.reorder_categories(
    sorted(adata.obs[RES].cat.categories, key=int))
sizes = adata.obs[RES].value_counts()
print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"{RES}: {adata.obs[RES].nunique()} clusters | median {int(sizes.median())} | "
      f"smallest {int(sizes.min())} | largest {int(sizes.max())}")
xmax = float(adata.X.max())
print(f".X max {xmax:.1f} ->", "log1p-normalised (good)" if xmax < 50
      else "LOOKS LIKE RAW COUNTS -- score_genes needs normalised .X")'''

PANEL = '''panel = pd.read_csv(PANEL, sep="\\t", comment="#")
panel["genes"] = panel["genes"].str.split(",")
lineage_sets, context_sets, missing = {}, {}, {}
for _, r in panel.iterrows():
    have = [g for g in r.genes if g in adata.var_names]
    gone = [g for g in r.genes if g not in adata.var_names]
    if gone: missing[r.celltype] = gone
    (lineage_sets if r.role == "lineage" else context_sets)[r.celltype] = have
plot_sets = {k: v for k, v in {**lineage_sets, **context_sets}.items() if v}
if missing: print("MARKERS NOT FOUND:", missing)
print("scoring sets:", {k: len(v) for k, v in lineage_sets.items()})
print("context (plotted, never used to assign):", {k: len(v) for k, v in context_sets.items()})
print("collapsed for assignment:", COLLAPSE)'''

SCORE = '''score_cols = []
for ct, genes in lineage_sets.items():
    if not genes: print(f"SKIP {ct}: no markers present"); continue
    sc.tl.score_genes(adata, genes, score_name=f"score_{ct}", ctrl_size=50, random_state=0)
    score_cols.append(f"score_{ct}")
cs = adata.obs.groupby(RES, observed=True)[score_cols].mean()
cs.columns = [c.replace("score_", "", 1) for c in cs.columns]
cs_lin = cs.rename(columns=COLLAPSE).T.groupby(level=0).max().T
srt = np.sort(cs_lin.values, axis=1)
marker = pd.DataFrame({
    "n": sizes.reindex(cs.index).astype(int),
    "marker_call": cs_lin.idxmax(axis=1),
    "margin": (srt[:, -1] - srt[:, -2]).round(3),
})
print(f"{len(marker)} clusters scored across {cs_lin.shape[1]} lineages\\n")
print(marker.marker_call.value_counts().to_string())'''

JOIN = '''def _canon(s):
    s = s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return s.str.replace("bleeding", "menstrual", regex=False)
def _key(o):
    return _canon(o["sample"]) + "||" + \\
           o["cell_barcode"].astype(str).str.replace(r"-\\d+$", "", regex=True)
them = sc.read_h5ad(REPO_SUB/REF, backed="r").obs.copy(); them["k"] = _key(them)
r = _key(adata.obs).map(them.set_index("k")["cell_type_annotations"])
adata.obs["ref_type"] = r.astype(object).where(r.notna(), "unlabeled").values
adata.obs["ref_lineage"] = adata.obs["ref_type"].replace(COLLAPSE).values
nj = int((adata.obs.ref_type != "unlabeled").sum())
print(f"deposited join: {nj:,}/{adata.n_obs:,} ({100*nj/adata.n_obs:.1f}%)")
print(adata.obs.ref_lineage.value_counts().to_string())

def _top(s):
    v = s.value_counts(normalize=True)
    return f"{v.index[0]} {v.iloc[0]:.2f}" if len(v) else ""
vote = adata.obs.groupby(RES, observed=True)["ref_lineage"].agg(_top).rename("vote")
qc_cols = [c for c in QC_COLS if c in adata.obs]
qc  = adata.obs.groupby(RES, observed=True)[qc_cols].median().round(2)
dfr = (adata.obs.groupby(RES, observed=True)["doubletdetection_predicted_doublet"]
       .mean().round(2).rename("dbl_frac") if "doubletdetection_predicted_doublet" in adata.obs else None)
TAB = pd.concat([x for x in [marker, dfr, qc, vote] if x is not None], axis=1)
print(); print(TAB.to_string())'''

DOTPLOT = '''sc.pl.dotplot(adata, plot_sets, groupby=RES, standard_scale="var",
              figsize=(8, max(6, 0.22*adata.obs[RES].nunique())), show=True)'''

COVERAGE = '''called = set(TAB.marker_call)
absent = [l for l in LABELS if l not in called]
print("lineages with no cluster called by markers:", absent or "none")
for l in absent:
    if l in cs_lin.columns:
        c = cs_lin[l].idxmax()
        print(f"   {l:<22} best cluster {c} (score {cs_lin.loc[c,l]:.2f}, "
              f"markers say {TAB.loc[c,'marker_call']}, n={int(TAB.loc[c,'n'])})")
med = int(TAB.n.median())
print(f"\\ncells called per lineage (small ones risk not resolving at {RES}):")
for l in LABELS:
    got = int(TAB.loc[TAB.marker_call == l, "n"].sum())
    print(f"  {l:<22} {got:>7,}  {'ABSENT' if got==0 else ('AT RISK' if got<med else '')}")
amb = TAB[TAB.margin < 0.05]
print(f"\\nlow-margin clusters (<0.05): {len(amb)}")
if len(amb): print(amb.to_string())'''

SUGGEST = '''def suggest_labels():
    """Majority vote on the deposited lineage + auto-EXCLUDE. Prefill only."""
    out = {}
    for c in TAB.index:
        row = TAB.loc[c]
        ds = row.get("doubletdetection_doublet_score", 0)
        df = row.get("dbl_frac", 0)
        if (ds is not None and ds >= AUTO["doublet_score"]) or \\
           (df is not None and df >= AUTO["doublet_frac"]):
            out[c] = "EXCLUDE"; continue
        v = str(row.get("vote", "")).rsplit(" ", 1)
        lab, p = (v[0], float(v[1])) if len(v) == 2 else ("", 0.0)
        if lab == "unlabeled" or not lab or p < 0.5:
            out[c] = "EXCLUDE" if (lab == "unlabeled" and p >= 0.5) else str(row.marker_call)
        else:
            out[c] = lab
    return out

SUGGESTED = suggest_labels()
dis = [c for c in TAB.index if SUGGESTED[c] != TAB.loc[c, "marker_call"]
       and SUGGESTED[c] != "EXCLUDE"]
print("suggested (vote-driven; markers shown where they disagree):")
for c in TAB.index:
    tag = ("   # <- auto-EXCLUDE" if SUGGESTED[c] == "EXCLUDE"
           else (f"   # markers say {TAB.loc[c,'marker_call']}" if c in dis else ""))
    print(f'    "{c}": "{SUGGESTED[c]}",{tag}')'''

DECISION = '''# =============================================================================
#  >>>  THE DECISION CELL  --  this is the only cell you edit  <<<
# =============================================================================
# CHOICES is written verbatim to resources/annotations/{SPECIES}_lineage.csv
# (cluster,label) -- the committed record the public de-novo notebook compiles
# from. Start from SUGGESTED (deposited majority vote + auto-EXCLUDE), and put
# every hand call in OVERRIDES so your judgment stays explicit.
#
#   label must be one of LABELS, or "EXCLUDE".
# =============================================================================

CHOICES = dict(SUGGESTED)

OVERRIDES = {
    # "4": "EXCLUDE",              # mixed marker profile on the dotplot
    # "9": "perimetrium cells",    # Sox6 + Msln/Upk3b/Wt1 trio
}

CHOICES.update(OVERRIDES)

_bad = {c: v for c, v in CHOICES.items() if v not in LABELS + ["EXCLUDE"]}
assert not _bad, f"labels not in LABELS + EXCLUDE: {_bad}"
_missing = set(TAB.index) - set(CHOICES)
assert not _missing, f"clusters with no choice: {sorted(_missing, key=int)}"
print(f"{len(CHOICES)} clusters | {sum(v=='EXCLUDE' for v in CHOICES.values())} EXCLUDE "
      f"| {len(OVERRIDES)} manual override(s)")
print(pd.Series(CHOICES).value_counts().to_string())'''

APPLY = '''final = TAB.copy()
final["lineage"] = pd.Series(CHOICES).reindex(final.index)
final["exclude"] = final.lineage.eq("EXCLUDE")
final["marker_agrees"] = final.exclude | (final.lineage == final.marker_call)
adata.obs["lineage"]  = adata.obs[RES].map(final.lineage).astype(str)
adata.obs["excluded"] = adata.obs[RES].map(final.exclude).values

kept = final[~final.exclude]
print(f"{len(final)} clusters | {int(final.exclude.sum())} EXCLUDE "
      f"({int(final.loc[final.exclude,'n'].sum()):,} cells) | retained {int(kept.n.sum()):,}")
print(f"marker agreement: {int(kept.marker_agrees.sum())}/{len(kept)} clusters, "
      f"{100*kept.loc[kept.marker_agrees,'n'].sum()/kept.n.sum():.1f}% of cells\\n")

# validation vs the deposited object
mine_ex = adata.obs.excluded.values
their_ex = adata.obs.ref_type.eq("unlabeled").values
tp = int((mine_ex & their_ex).sum()); fp = int((mine_ex & ~their_ex).sum())
fn = int((~mine_ex & their_ex).sum())
print("=== EXCLUDE vs the cells they removed ===")
if tp+fp: print(f"  precision {tp/(tp+fp):.3f}", end="   ")
if tp+fn: print(f"recall {tp/(tp+fn):.3f}")
both = ~mine_ex & ~their_ex
ag = (adata.obs.lineage.astype(str)[both].values ==
      adata.obs.ref_lineage.astype(str)[both].values)
print(f"=== LINEAGE agreement (cells we both keep): {100*ag.mean():.2f}% "
      f"on {int(both.sum()):,} cells ===")
sc.pl.embedding(adata, "X_umap_scVI", color="lineage", show=True)'''

WRITE = '''# 1. the committed record
pd.DataFrame({"cluster": list(CHOICES), "label": list(CHOICES.values())}) \\
  .sort_values("cluster", key=lambda s: s.astype(int)).to_csv(CSV_OUT, index=False)
print(f"wrote {CSV_OUT}  ({len(CHOICES)} rows)")

# 2. annotated object
ann = adata[~adata.obs.excluded].copy()
for c in ["ref_type", "ref_lineage"] + [c for c in ann.obs if c.startswith("score_")]:
    if c in ann.obs: del ann.obs[c]
ann.obs["lineage"] = pd.Categorical(ann.obs.lineage,
    categories=[l for l in LABELS if l in set(ann.obs.lineage)])
ann.write(OUT_FULL); print(f"wrote {OUT_FULL}: {ann.n_obs:,} cells")

# 3. lineage subsets -- these are what 04b_reintegrate consumes
OUT_LIN.mkdir(parents=True, exist_ok=True)
rows = []
for lin in ann.obs.lineage.cat.categories:
    sub = ann[ann.obs.lineage == lin].copy()
    if sub.n_obs == 0: continue
    p = OUT_LIN/f"{SPECIES}_{SLUG[lin]}.h5ad"; sub.write(p)
    rows.append({"lineage": lin, "n": sub.n_obs, "file": p.name})
print(pd.DataFrame(rows).set_index("lineage").to_string())

# 4. leiden ladder -- the R clustree script reads leiden_all_resolutions.csv
OUT_CSV.mkdir(parents=True, exist_ok=True)
base = pd.DataFrame({"cell": adata.obs_names,
                     "sample": adata.obs["sample"].astype(str).values,
                     "cell_barcode": adata.obs["cell_barcode"].astype(str).values})
if "timepoint" in adata.obs: base["timepoint"] = adata.obs["timepoint"].astype(str).values
res_cols = sorted([c for c in adata.obs if c.startswith("leiden_scVI_")],
                  key=lambda c: float(c.rsplit("_", 1)[1]))
for col in res_cols:
    base.assign(cluster=adata.obs[col].astype(str).values) \\
        .to_csv(OUT_CSV/f"leiden_res{col.rsplit('_',1)[1]}.csv", index=False)
wide = base.copy()
for col in res_cols: wide[col] = adata.obs[col].astype(str).values
wide["lineage"] = adata.obs.lineage.astype(str).values
wide.to_csv(OUT_CSV/"leiden_all_resolutions.csv", index=False)
print(f"wrote {len(res_cols)} ladder files -> {OUT_CSV}")
print(f"\\nnext:  python tools/compile_denovo.py --species {SPECIES} --res {RES} \\\\\\n"
      f"           --csv resources/annotations/{SPECIES}_lineage.csv \\\\\\n"
      f"           --out notebooks/public/04_annotate_{SPECIES}_denovo.ipynb --level lineage")'''



# Assemble one notebook per species ----
made = skipped = 0
for species, cfg in spec["atlas"].items():
    if a.only and a.only != species:
        continue
    p = OUT/f"04_annotate_{species}.ipynb"
    if p.exists() and not a.force:
        print(f"  skip (exists)  {p.name}")
        skipped += 1
        continue

    # The config cell: every per-species value the cell bodies above refer to.
    # Per-species auto_exclude thresholds override the spec defaults.
    labels = list(cfg["labels"])
    conf = f'''# ---- generated {STAMP} from resources/lineage_spec.yaml (`atlas:`) -- edit the SPEC
SPECIES = {json.dumps(species)}
RES     = {json.dumps(cfg["res"])}
REF     = {json.dumps(cfg["ref"])}
PANEL_FILE = {json.dumps(cfg["panel"])}
REPO     = {json.dumps(a.repo)}
REPO_SUB = "/n/eddy_lab/users/nhilgert/decidualization/revisions/adatas/submission"

# deposited fine type -> our lineage (fib + decidual are one continuum; split at 04c)
COLLAPSE = {json.dumps(cfg["collapse"], indent=4)}
LABELS = {json.dumps(labels, indent=4)}
SLUG   = {json.dumps(cfg["labels"], indent=4)}

QC_COLS = {json.dumps(cfg.get("qc_cols", DEF.get("qc_cols", [])))}
AUTO    = {json.dumps({**DEF.get("auto_exclude", {}), **cfg.get("auto_exclude", {})})}'''

    # Markdown headers interleaved with the code cells, in notebook order.
    C = [md(f"""# 04 — {species} coarse lineage annotation  *(private scaffolding)*

Generated {STAMP} from `resources/lineage_spec.yaml` (`atlas:` section) —
**regenerable, gitignored.** Edit the spec, re-run `tools/gen_04_atlas.py`.

Whole-tissue annotation at `{cfg['res']}` into {len(labels)} lineages. Run
everything, study §5/§6/§7, then edit **only the DECISION cell (§8)**. §10 writes
the committed CSV, the annotated object, the **lineage subsets that 04b
reintegrates**, and the leiden ladder the R clustree script reads."""),
        md("## 1. Config *(generated)*"), code(conf), code(IMPORTS),
        md("## 2. Load"), code(LOAD),
        md("## 3. Marker panel"), code(PANEL),
        md("## 4. Score each marker set"), code(SCORE),
        md("## 5. Deposited labels + diagnostic table"), code(JOIN),
        md("## 6. The dotplot — raw clusters, no hints"), code(DOTPLOT),
        md("## 7. Coverage check — before writing the map"), code(COVERAGE),
        md("## 8a. Suggested labels *(prefill only)*"), code(SUGGEST),
        md("""## 8. ★ DECISION — the only cell you edit ★"""), code(DECISION),
        md("## 9. Apply + validate against the deposited object"), code(APPLY),
        md("## 10. Write the record, the object, the lineage subsets, the ladder"), code(WRITE)]

    nb = new_notebook(cells=C)
    nb.metadata["kernelspec"] = {"display_name":"Python 3","language":"python","name":"python3"}
    nb.metadata["language_info"] = {"name":"python"}

    nbf.validate(nb)
    nbf.write(nb, str(p))
    print(f"  {'overwrote' if a.force else 'created  '}    {p.name}")
    made += 1

print(f"\n{made} written, {skipped} skipped -> {OUT}")
