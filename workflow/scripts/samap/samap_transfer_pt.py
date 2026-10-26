#!/usr/bin/env python
"""transfer_pt_0902.py — mouse -> human pseudotime transfer through the raw
vanilla SAMap graph.

Operator: Nadaraya-Watson over the stitched joint graph's cross-species block,
  pt_hs = (C_hs,mm @ pt_mm) / (C_hs,mm @ 1),
with pt_mm joined from an external mouse object by barcode (exact ->
canonical-16-mer fallback).

Three source variants are transferred side by side:
  pt_raw               : obs[pt_col] as-is
  pt_ranknorm          : global rank / N over finite source values (Nick's
                         "quantile-normalized ... rank divided by N")
  pt_ranknorm_insample : rank / N within each source sample (the isodepth
                         session's within-sample convention), when a sample
                         column exists

Outputs: <out>/pt_transferred_hs.csv (per human cell: all variants + total
cross-species vote weight), <out>/summary.json, <out>/pt_hist.png.
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
    m = re.search(r"[ACGTN]{14,20}", str(bc).upper())
    return m.group(0) if m else str(bc)


def barcode_join(barcodes, names, values):
    """Exact barcode match first. The canonical (bare 14-20 bp core) fallback is used ONLY when a
    barcode is ABSENT from the source - never when it is present with a NaN value (cells excluded
    from a pseudotime variant by design) - and never through a core shared by more than one source
    cell (cores collide across samples). Fixed 2026-09-25: the previous version fell back whenever
    the exact value was NaN, so 9 apoptotic cells borrowed another sample's pseudotime."""
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

    variants = {"pt_raw": pt0, "pt_ranknorm": ranknorm(pt0)}
    if args.sample_col in src.obs.columns:
        ws = np.full(len(pt0), np.nan)
        samples = src.obs[args.sample_col].astype(str).values
        for s in np.unique(samples):
            m = samples == s
            ws[m] = ranknorm(pt0[m])
        variants["pt_ranknorm_insample"] = ws
        log(f"within-sample variant over {len(np.unique(samples))} samples")

    mm_bcs = joint.obs_names[mm_m].astype(str).values
    C = conn[hs_m][:, mm_m].tocsr()
    res = pd.DataFrame(index=joint.obs_names[hs_m])
    summary = {"joint": args.joint_h5ad, "pt_source": args.pt_h5ad,
               "pt_col": args.pt_col}

    for vname, vals in variants.items():
        pt_mm, tiers = barcode_join(mm_bcs, names, vals)
        valid = np.isfinite(pt_mm)
        pt_fill = np.where(valid, pt_mm, 0.0)
        num = np.asarray(C @ pt_fill).ravel()
        den = np.asarray(C @ valid.astype(float)).ravel()
        tr = np.full(C.shape[0], np.nan)
        ok = den > 1e-10
        tr[ok] = num[ok] / den[ok]
        res[vname] = tr
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

    # cross-variant agreement (sanity)
    from scipy.stats import spearmanr

    keys = [k for k in variants if res[k].notna().sum() > 100]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            m = res[keys[i]].notna() & res[keys[j]].notna()
            rho = float(spearmanr(res.loc[m, keys[i]], res.loc[m, keys[j]]).statistic)
            summary[f"spearman[{keys[i]}~{keys[j]}]"] = round(rho, 4)
            log(f"spearman {keys[i]} ~ {keys[j]}: {rho:.4f}")

    res.index.name = "cell"
    res.to_csv(out / "pt_transferred_hs.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

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
