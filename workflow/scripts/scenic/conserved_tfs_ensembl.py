#!/usr/bin/env python
"""Conserved decidual TFs across species -- Ensembl one2one join.

Script port of {scenic rev}/scripts/decidual_tf_list.ipynb (cells 2-17), with
three deliberate changes and no other logic drift:
  * cross-species join through Ensembl 116 Compara ortholog_one2one
    (resources/orthology/ensembl_human_mouse_components.tsv) instead of the
    notebook's `s.upper()` case-mapping -- project orthology policy 2026-09-15;
  * reads the CURRENT SCENIC lane (config scenic.rev, v2 2026_08_28) instead of
    the notebook's hardcoded dead-mount June lane;
  * additionally reports the upper-case join side by side, so the effect of the
    orthology switch on the conserved set is auditable.

Per species: Mann-Whitney decidual-vs-background per consensus regulon ->
Cliff's delta + BH FDR; candidates = delta >= MIN_DELTA & fdr < FDR minus the
cell-cycle / non-TF blocklists (verbatim from the notebook). On the v2 fib/dec
lane the background is the fibroblasts, i.e. the notebook pair's precursor
contrast.

Outputs under --outdir:
  diff_activity_{species}.csv       per-regulon delta/means/fdr
  conserved_tfs_ensembl.csv         one2one-joined conserved decidual-up TFs
  tiers_ensembl.tsv                 tier table (1 known / 2 novel / 3 one-species)
  join_comparison.json              ensembl vs upper join: sizes, entered/exited,
                                    candidates without a one2one partner
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ----- notebook constants, verbatim ------------------------------------------
DECIDUAL_KEYS = ["decid", "dsc"]
CELLCYCLE_TFS = ["E2f1", "E2f2", "E2f3", "E2f4", "E2f5", "E2f6", "E2f7", "E2f8",
                 "Foxm1", "Mybl1", "Mybl2", "Tfdp1", "Tfdp2", "Brca1"]
NON_TF_BLOCKLIST = ["Ube2k", "Ep300", "Tbp", "Taf6", "Ilf2", "Chd1", "Chd2",
                    "Bptf", "Setdb1", "Phf20", "Ing4", "Mbd2"]
KNOWN_DECIDUAL = ["Pgr", "Esr1", "Hand2", "Foxo1", "Cebpb", "Cebpa", "Nr2f2",
                  "Prdm1", "Gata2", "Hoxa10", "Hoxa11", "Sox17", "Klf5", "Wt1"]
_BLOCK = {x.casefold() for x in CELLCYCLE_TFS + NON_TF_BLOCKLIST}


def base(name: str) -> str:
    """Regulon name -> TF symbol (strip the '(+)' style suffix)."""
    return str(name).split("(")[0]


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def load_one2one(tsv: Path, homology_type: str = "ortholog_one2one"):
    """Strict symbol-level 1:1 map, same discipline as liana_crossspecies.py."""
    d = pd.read_csv(tsv, sep="\t")
    o = d[d["homology_type"] == homology_type].dropna(subset=["gene_symbol"])
    piv = o.pivot_table(index="component_id", columns="species",
                        values="gene_symbol", aggfunc=list)
    pairs = []
    for _, row in piv.iterrows():
        hs, ms = row.get("human"), row.get("mouse")
        if isinstance(hs, list) and isinstance(ms, list) and len(hs) == 1 and len(ms) == 1:
            pairs.append((ms[0], hs[0]))
    mc = pd.Series([m for m, _ in pairs]).value_counts()
    hc = pd.Series([h for _, h in pairs]).value_counts()
    m2h = {m: h for m, h in pairs if mc[m] == 1 and hc[h] == 1}
    return m2h


def diff_activity(rev: str, species: str, tag: str, mode: str, ct_col: str) -> pd.DataFrame:
    cdir = Path(rev) / "scenic_seeds" / species / f"consensus_{tag}_{mode}"
    auc = pd.read_csv(cdir / "auc_mtx.csv", index_col=0)
    a = ad.read_h5ad(Path(rev) / "scenic_seeds" / species / f"{species}_dsc_pooled.h5ad",
                     backed="r")
    labels = a.obs.reindex(auc.index)[ct_col].astype(str)
    if labels.isna().any():
        raise SystemExit(f"[{species}] {int(labels.isna().sum())} AUCell cells "
                         f"missing from the pooled object -- index mismatch")
    cats = sorted(labels.unique())
    dec_lbls = [c for c in cats if any(k in c.lower() for k in DECIDUAL_KEYS)]
    bg_lbls = [c for c in cats if c not in dec_lbls]
    if not dec_lbls or not bg_lbls:
        raise SystemExit(f"[{species}] decidual/background split failed: {cats}")
    is_dec = labels.isin(dec_lbls).values
    is_bg = labels.isin(bg_lbls).values
    print(f"[{species}] DECIDUAL={dec_lbls} ({int(is_dec.sum()):,}) vs "
          f"BACKGROUND={bg_lbls} ({int(is_bg.sum()):,}); "
          f"{auc.shape[1]} regulons", flush=True)
    n_dec, n_bg = int(is_dec.sum()), int(is_bg.sum())
    rows = []
    A = auc.values
    for j, reg in enumerate(auc.columns):
        aa, bb = A[is_dec, j], A[is_bg, j]
        U, p = mannwhitneyu(aa, bb, alternative="two-sided")
        rows.append(dict(reg=reg, tf=base(reg),
                         delta=2 * U / (n_dec * n_bg) - 1,
                         mean_dec=float(aa.mean()), mean_bg=float(bb.mean()),
                         p=float(p)))
    df = pd.DataFrame(rows)
    df["fdr"] = bh_fdr(df["p"])
    return df.sort_values("delta", ascending=False).reset_index(drop=True)


def candidates(df: pd.DataFrame, min_delta: float, fdr: float) -> pd.DataFrame:
    keep = df[(df.delta >= min_delta) & (df.fdr < fdr)].copy()
    keep = keep[~keep.tf.str.casefold().isin(_BLOCK)]
    return keep.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rev", required=True, help="SCENIC lane root (config scenic.rev)")
    p.add_argument("--orthology", required=True,
                   help="resources/orthology/ensembl_human_mouse_components.tsv")
    p.add_argument("--sketch", default="15k")
    p.add_argument("--mode", default="weighted_soft")
    p.add_argument("--celltype-col", default="cell_type",
                   help="obs column on the pooled objects (v2 lane: cell_type)")
    p.add_argument("--min-delta", type=float, default=0.10)
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()

    tag = "full" if a.sketch == "full" else f"sketch{a.sketch}"
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    m2h = load_one2one(Path(a.orthology))
    print(f"orthology: {len(m2h):,} symbol-unique one2one pairs", flush=True)

    res, cand = {}, {}
    for sp in ("mouse", "human"):
        res[sp] = diff_activity(a.rev, sp, tag, a.mode, a.celltype_col)
        res[sp].to_csv(out / f"diff_activity_{sp}.csv", index=False)
        cand[sp] = candidates(res[sp], a.min_delta, a.fdr)
        print(f"[{sp}] candidates after delta/FDR/blocklist: {len(cand[sp])}", flush=True)

    # --- Ensembl one2one join (the fix) --------------------------------------
    mouse_tfs = cand["mouse"].set_index("tf")
    human_tfs = cand["human"].set_index("tf")
    ens_rows, unmapped_mouse = [], []
    for mtf in mouse_tfs.index:
        htf = m2h.get(mtf)
        if htf is None:
            unmapped_mouse.append(mtf)
        elif htf in human_tfs.index:
            ens_rows.append(dict(tf_mouse=mtf, tf_human=htf,
                                 delta_mouse=float(mouse_tfs.loc[mtf, "delta"]),
                                 delta_human=float(human_tfs.loc[htf, "delta"]),
                                 fdr_mouse=float(mouse_tfs.loc[mtf, "fdr"]),
                                 fdr_human=float(human_tfs.loc[htf, "fdr"])))
    cons = (pd.DataFrame(ens_rows)
            .sort_values("delta_mouse", ascending=False)
            .reset_index(drop=True))
    cons.to_csv(out / "conserved_tfs_ensembl.csv", index=False)
    print(f"conserved decidual-up TFs (Ensembl one2one): {len(cons)}", flush=True)

    # --- legacy upper-case join, for the audit -------------------------------
    up_m = {t.upper(): t for t in mouse_tfs.index}
    up_h = {t.upper(): t for t in human_tfs.index}
    upper_set = set(up_m) & set(up_h)
    ens_set = {r["tf_human"].upper() for r in ens_rows}
    human_unmapped = [t for t in human_tfs.index
                      if t not in set(m2h.values())]
    comparison = {
        "n_candidates_mouse": len(mouse_tfs),
        "n_candidates_human": len(human_tfs),
        "n_conserved_ensembl": len(cons),
        "n_conserved_upper": len(upper_set),
        "in_upper_not_ensembl": sorted(upper_set - ens_set),
        "in_ensembl_not_upper": sorted(ens_set - upper_set),
        "mouse_candidates_without_one2one": sorted(unmapped_mouse),
        "human_candidates_without_one2one": sorted(human_unmapped),
        "params": dict(min_delta=a.min_delta, fdr=a.fdr, sketch=a.sketch,
                       mode=a.mode, rev=a.rev, orthology=a.orthology,
                       homology_type="ortholog_one2one"),
    }
    (out / "join_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({k: v for k, v in comparison.items()
                      if k.startswith("n_") or "not_" in k}, indent=1), flush=True)

    # --- tier table (notebook cell 17, Ensembl-joined) ------------------------
    known = {k.upper() for k in KNOWN_DECIDUAL}
    cons_upper = {r["tf_human"].upper() for r in ens_rows}
    tier_rows = []
    for r in ens_rows:
        ku = r["tf_human"].upper()
        tier_rows.append(dict(tf_mouse=r["tf_mouse"], tf_human=r["tf_human"],
                              tier=1 if ku in known else 2,
                              label="conserved+known" if ku in known else "conserved+novel",
                              delta_mouse=r["delta_mouse"], delta_human=r["delta_human"]))
    matched_mouse = {r["tf_mouse"] for r in ens_rows}
    matched_human = {r["tf_human"] for r in ens_rows}
    for t in sorted(set(mouse_tfs.index) - matched_mouse):
        tier_rows.append(dict(tf_mouse=t, tf_human="", tier=3, label="mouse-only"))
    for t in sorted(set(human_tfs.index) - matched_human):
        tier_rows.append(dict(tf_mouse="", tf_human=t, tier=3, label="human-only"))
    tiers = pd.DataFrame(tier_rows).sort_values(["tier", "delta_mouse"],
                                                ascending=[True, False])
    tiers.to_csv(out / "tiers_ensembl.tsv", sep="\t", index=False)
    n_cons_known = sum(1 for r in tier_rows if r["label"] == "conserved+known")
    print(f"tiers written; known decidual TFs conserved: {n_cons_known}", flush=True)


if __name__ == "__main__":
    main()
