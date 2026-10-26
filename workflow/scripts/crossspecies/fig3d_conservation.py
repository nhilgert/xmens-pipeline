"""Fig 3D conservation statistic, restated on Ensembl one2one orthology.

Reads the per-species MI tables (--mi-mouse, --mi-human, each an mi_all.csv
with columns gene, mi, q) and the Ensembl orthology components TSV (--orthology).
Writes one JSON (--out) that accumulates runs keyed by --label, plus
fig3d_shared_genes_<label>.csv beside it. Standalone CLI, not wired to a rule.

The statistic follows the preprint's definition: "top pseudotime-associated
genes" are MI-significant (BH q < --q) with MI at or above the
--top-percentile-th percentile among significant genes, per species, and
conservation is the hypergeometric overlap of the two top sets through strict
1:1 orthologs. The orthology source is the project's settled one (Ensembl 116
Compara ortholog_one2one) rather than the draft-era JAX table.

Universe N = one2one pairs where both genes were MI-tested. K = mouse top
genes in the universe, n = human top genes in the universe, k = pairs top in
both. Reported in the draft's shape (pct of human top shared, expected = K/N,
fold, hypergeom p) so Methods can quote one coherent number.

Draft reference values (cevrim.pdf Fig 3D, source universe likely JAX):
N=9,731  K=1,182  n=970  k=302  (31% shared vs 12% expected, p ~ 3.2e-63).
Run once per mouse MI table (dpt_noapop_mouse, dpt_full_mouse) for the
decomposition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

# Values as reported in the preprint draft (Fig 3D), kept in the output for comparison.
DRAFT_REFERENCE = dict(N=9731, K=1182, n=970, k=302,
                       pct_human_shared=31.1, pct_expected=12.1,
                       p="~3.2e-63",
                       note="as reported in the preprint draft (Fig 3D); "
                            "source universe likely JAX 1:1 + curated overrides")


def load_one2one(tsv: Path, homology_type: str):
    """Symbol-unique (mouse, human) pairs: one gene per species per component, each symbol once."""
    d = pd.read_csv(tsv, sep="\t")
    o = d[d["homology_type"] == homology_type].dropna(subset=["gene_symbol"])
    piv = o.pivot_table(index="component_id", columns="species",
                        values="gene_symbol", aggfunc=list)

    pairs = []
    for _, row in piv.iterrows():
        hs, ms = row.get("human"), row.get("mouse")
        if isinstance(hs, list) and isinstance(ms, list) and len(hs) == 1 and len(ms) == 1:
            pairs.append((ms[0], hs[0]))

    # A symbol in two components would count a gene twice in the universe.
    mc = pd.Series([m for m, _ in pairs]).value_counts()
    hc = pd.Series([h for _, h in pairs]).value_counts()
    return [(m, h) for m, h in pairs if mc[m] == 1 and hc[h] == 1]


def top_set(mi_csv: Path, q_thresh: float, top_percentile: float):
    """Return (all tested genes, top genes, metadata) for one species' MI table."""
    df = pd.read_csv(mi_csv)
    sig = df[df["q"] < q_thresh]
    if sig.empty:
        raise SystemExit(f"{mi_csv}: no genes at q < {q_thresh}")

    # The percentile is taken among significant genes only, as in the preprint.
    cut = float(np.percentile(sig["mi"], top_percentile))
    top = set(sig.loc[sig["mi"] >= cut, "gene"].astype(str))
    return set(df["gene"].astype(str)), top, dict(
        n_tested=len(df), n_significant=len(sig), mi_cutoff=round(cut, 5),
        n_top=len(top))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mi-mouse", required=True)
    p.add_argument("--mi-human", required=True)
    p.add_argument("--orthology", required=True)
    p.add_argument("--homology-type", default="ortholog_one2one")
    p.add_argument("--q", type=float, default=0.05)
    p.add_argument("--top-percentile", type=float, default=90.0,
                   help="percentile of MI among significant genes (config mi.top_percentile)")
    p.add_argument("--label", required=True, help="key for this run in the output JSON")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    # Load inputs ----
    pairs = load_one2one(Path(a.orthology), a.homology_type)
    print(f"orthology: {len(pairs):,} symbol-unique one2one pairs", flush=True)

    m_tested, m_top, m_meta = top_set(Path(a.mi_mouse), a.q, a.top_percentile)
    h_tested, h_top, h_meta = top_set(Path(a.mi_human), a.q, a.top_percentile)
    print(f"mouse: {m_meta} | human: {h_meta}", flush=True)

    # Hypergeometric overlap ----
    universe = [(m, h) for m, h in pairs if m in m_tested and h in h_tested]
    N = len(universe)
    K = sum(1 for m, _ in universe if m in m_top)
    n = sum(1 for _, h in universe if h in h_top)
    shared = [(m, h) for m, h in universe if m in m_top and h in h_top]
    k = len(shared)
    p_val = float(hypergeom.sf(k - 1, N, K, n))  # P(X >= k)

    result = dict(
        label=a.label,
        mi_mouse=str(a.mi_mouse), mi_human=str(a.mi_human),
        orthology=str(a.orthology), homology_type=a.homology_type,
        q_thresh=a.q, top_percentile=a.top_percentile,
        mouse=m_meta, human=h_meta,
        N_universe=N, K_mouse_top=K, n_human_top=n, k_shared=k,
        pct_human_shared=round(100 * k / n, 2) if n else None,
        pct_mouse_shared=round(100 * k / K, 2) if K else None,
        pct_expected=round(100 * K / N, 2) if N else None,
        fold=round((k / n) / (K / N), 3) if n and K else None,
        p_hypergeom=p_val,
    )
    print(json.dumps({x: result[x] for x in
                      ("N_universe", "K_mouse_top", "n_human_top", "k_shared",
                       "pct_human_shared", "pct_expected", "fold", "p_hypergeom")},
                     indent=1), flush=True)

    # Write ----
    # The JSON accumulates across runs so both mouse variants end up in one file.
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.loads(out.read_text()) if out.exists() else {
        "definition": ("paper-shape statistic: top = MI-significant (BH q<q_thresh) "
                       "AND MI >= top_percentile among significant; hypergeometric "
                       "overlap through symbol-unique Ensembl one2one pairs tested "
                       "in both species"),
        "draft_reference": DRAFT_REFERENCE,
        "runs": {},
    }
    blob["runs"][a.label] = result
    out.write_text(json.dumps(blob, indent=2) + "\n")

    pd.DataFrame(shared, columns=["mouse", "human"]).to_csv(
        out.with_name(f"fig3d_shared_genes_{a.label}.csv"), index=False)
    print(f"wrote {out} (+ shared-genes CSV)", flush=True)


if __name__ == "__main__":
    main()
