"""Mouse -> human pseudotime transfer through the stitched SAMap graph.

Reads the joint samap_transferred.h5ad (--joint-h5ad, for obsp connectivities)
and a mouse object carrying pseudotime (--pt-h5ad). Writes
<out-dir>/pt_transferred_hs.csv (per human cell: every variant plus the total
cross-species vote weight), summary.json and pt_hist.png. Called by rule
samap_raw_pt_transfer (workflow/rules/10b_samap_raw.smk).

Operator: Nadaraya-Watson over the joint graph's cross-species block,
  pt_hs = (C_hs,mm @ pt_mm) / (C_hs,mm @ 1),
with pt_mm joined from the mouse object by barcode (exact, then a canonical
14-20 bp core fallback).

Three source variants are transferred side by side:
  pt_raw               : obs[pt_col] as-is
  pt_ranknorm          : global rank / N over finite source values
  pt_ranknorm_insample : rank / N within each source sample (the isodepth
                         within-sample convention), when a sample column exists
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def log(m):
    print(m, flush=True)


def canon(bc: str) -> str:
    """The bare 14-20 bp barcode core, so prefixes and suffixes added by different tools match."""
    m = re.search(r"[ACGTN]{14,20}", str(bc).upper())
    return m.group(0) if m else str(bc)


def barcode_join(barcodes, names, values):
    """Look up `values` (indexed by `names`) for each barcode, exact match first.

    The canonical-core fallback is used only when a barcode is absent from the
    source, never when it is present with a NaN value (cells excluded from a
    pseudotime variant by design), and never through a core shared by more
    than one source cell (cores collide across samples). Returns the values and
    a count per match tier.
    """
    vals = pd.to_numeric(pd.Series(values), errors="coerce").values
    lut = dict(zip(names, vals))
    cores = pd.Series([canon(k) for k in names])
    amb = set(cores[cores.duplicated(keep=False)])
    lut_c = {k: v for k, v in zip(cores, vals) if k not in amb}

    out = np.full(len(barcodes), np.nan)
    tiers = {"exact": 0, "exact_nan": 0, "canon": 0, "canon_ambiguous": 0, "none": 0}
    for i, bc in enumerate(barcodes):
        if bc in lut:
            v = lut[bc]
            if np.isfinite(v):
                out[i] = v
                tiers["exact"] += 1
            else:
                tiers["exact_nan"] += 1
            continue

        k = canon(bc)
        if k in amb:
            tiers["canon_ambiguous"] += 1
            continue
        v = lut_c.get(k, np.nan)
        if np.isfinite(v):
            out[i] = v
            tiers["canon"] += 1
        else:
            tiers["none"] += 1
    return out, tiers


def ranknorm(v):
    """Average rank / N over the finite values, NaN elsewhere."""
    out = np.full_like(v, np.nan, dtype=float)
    ok = np.isfinite(v)
    r = pd.Series(v[ok]).rank(method="average").values
    out[ok] = r / ok.sum()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joint-h5ad", required=True,
                   help="samap_transferred.h5ad (carries obsp connectivities)")
    p.add_argument("--pt-h5ad", required=True, help="mouse object with pseudotime")
    p.add_argument("--pt-col", default="dpt_pseudotime")
    p.add_argument("--sample-col", default="sample")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import anndata as ad

    # Load inputs ----
    # Backed mode: only obs and obsp are needed, not the expression matrix.
    log(f"loading joint {args.joint_h5ad} (backed) ...")
    joint = ad.read_h5ad(args.joint_h5ad, backed="r")
    conn = joint.obsp["connectivities"].tocsr()
    species = joint.obs["species"].astype(str).values
    hs_m, mm_m = species == "hs", species == "mm"
    log(f"joint: {joint.shape}; hs={hs_m.sum()}, mm={mm_m.sum()}")

    log(f"loading pt source {args.pt_h5ad} ...")
    src = ad.read_h5ad(args.pt_h5ad, backed="r")
    if args.pt_col not in src.obs.columns:
        raise SystemExit(f"{args.pt_col!r} not in source obs: {list(src.obs.columns)}")
    names = src.obs_names.astype(str).values
    pt0 = pd.to_numeric(src.obs[args.pt_col], errors="coerce").values
    log(f"source: {src.n_obs} cells, finite pt: {int(np.isfinite(pt0).sum())}, "
        f"range [{np.nanmin(pt0):.3f}, {np.nanmax(pt0):.3f}]")

    # Source variants ----
    variants = {"pt_raw": pt0, "pt_ranknorm": ranknorm(pt0)}
    if args.sample_col in src.obs.columns:
        ws = np.full(len(pt0), np.nan)
        samples = src.obs[args.sample_col].astype(str).values
        for s in np.unique(samples):
            m = samples == s
            ws[m] = ranknorm(pt0[m])
        variants["pt_ranknorm_insample"] = ws
        log(f"within-sample variant over {len(np.unique(samples))} samples")

    # Transfer each variant over the human x mouse block ----
    mm_bcs = joint.obs_names[mm_m].astype(str).values
    C = conn[hs_m][:, mm_m].tocsr()
    res = pd.DataFrame(index=joint.obs_names[hs_m])
    summary = {"joint": args.joint_h5ad, "pt_source": args.pt_h5ad,
               "pt_col": args.pt_col}

    for vname, vals in variants.items():
        pt_mm, tiers = barcode_join(mm_bcs, names, vals)

        # Mouse cells without a pseudotime drop out of both numerator and denominator.
        valid = np.isfinite(pt_mm)
        pt_fill = np.where(valid, pt_mm, 0.0)
        num = np.asarray(C @ pt_fill).ravel()
        den = np.asarray(C @ valid.astype(float)).ravel()
        tr = np.full(C.shape[0], np.nan)
        ok = den > 1e-10
        tr[ok] = num[ok] / den[ok]
        res[vname] = tr

        # The weight depends only on the graph and the valid mask, so it is the same
        # for every variant when all source cells carry every variant.
        if "weight" not in res.columns:
            res["weight"] = den

        summary[vname] = {
            "source_join": tiers,
            "source_cells_with_pt_on_graph": int(valid.sum()),
            "hs_cells_covered": int(ok.sum()),
            "hs_coverage": float(ok.mean()),
            "transferred_range": [float(np.nanmin(tr)), float(np.nanmax(tr))],
        }
        log(f"{vname}: join={tiers}, mm-on-graph={int(valid.sum())}, "
            f"hs covered={ok.mean():.3f}")

    # Cross-variant agreement, as a sanity check on the rank transforms ----
    from scipy.stats import spearmanr

    keys = [k for k in variants if res[k].notna().sum() > 100]  # too few cells gives a noisy rho
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            m = res[keys[i]].notna() & res[keys[j]].notna()
            rho = float(spearmanr(res.loc[m, keys[i]], res.loc[m, keys[j]]).statistic)
            summary[f"spearman[{keys[i]}~{keys[j]}]"] = round(rho, 4)
            log(f"spearman {keys[i]} ~ {keys[j]}: {rho:.4f}")

    # Write ----
    res.index.name = "cell"
    res.to_csv(out / "pt_transferred_hs.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Per-timepoint histograms of the transferred values; a diagnostic, not a rule output.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tps = joint.obs["timepoint"].astype(str).values[hs_m]
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4),
                                 squeeze=False)
        for a, k in zip(axes[0], keys):
            for tp in sorted(set(tps)):
                m = (tps == tp) & res[k].notna().values
                a.hist(res[k].values[m], bins=60, histtype="step", lw=1.5,
                       density=True, label=tp)
            a.set_title(k, fontsize=10)
            a.set_xlabel("transferred pseudotime")
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "pt_hist.png", dpi=150)
        log("wrote pt_hist.png")
    except Exception as e:
        log(f"figure skipped: {e}")
    log("done")


if __name__ == "__main__":
    main()
