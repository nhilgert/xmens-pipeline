"""Seeded diffusion pseudotime for mouse fibroblast/decidual cells at one timepoint.

Called by rules `compute_pseudotime` and `diagnose_3dpi_pseudotime_roots`
(06b_timepoint_pseudotime.smk). Reads the staged annotation h5ad (snakemake.input.h5ad)
plus the pseudotime-test config, subsets to one timepoint, retrains scVI, builds a
diffusion map and roots DPT. A tuned stage writes a transfer-ready h5ad, a metrics JSON
and a per-subtype TSV. A stage with no root runs only in diagnostic mode and writes
candidate-root tables, never a pseudotime object. Kept apart from the canonical
pseudotime and SAMap outputs on purpose: every output goes to the scratch test dir.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import scipy.sparse as sp
import torch


# Snakemake inputs and settings ----
snk = snakemake  # noqa: F821 - injected by Snakemake
inp = Path(str(snk.input.h5ad))
log_path = Path(str(snk.log[0]))
global_cfg = dict(snk.params.pt_config)
stage_cfg = dict(snk.params.stage_config)
diagnostic_only = bool(snk.params.diagnostic_only)


def setting(name, default=None):
    """Resolve a per-stage override, then a pseudotime-wide setting."""
    value = stage_cfg.get(name, global_cfg.get(name, default))
    return value


timepoint = str(setting("timepoint"))
stage_tag = str(setting("stage_tag"))
root_dc = setting("root_dc")
root_sign = setting("root_sign")
root_subtype = setting("root_subtype")
n_hvg = int(setting("n_hvg", 3000))  # seurat_v3 HVGs (config pseudotime_tests.n_hvg)
n_latent = int(setting("n_latent", 30))  # scVI latent dimensions
n_neighbors = int(setting("n_neighbors", 15))  # kNN graph size for the diffusion map
n_diffmap = int(setting("n_diffmap", 15))  # diffusion components computed
max_epochs = int(setting("max_epochs", 400))  # scVI upper bound; early stopping is on
apop_quantile = setting("apop_quantile")
exclude_apoptotic_subtype = bool(setting("exclude_apoptotic_subtype", False))
seed = int(setting("seed", 0))

# A root is committed by a person after reading the diagnostic tables, so a tuned
# stage must carry one and a diagnostic run must not.
if not diagnostic_only and (root_dc is None or root_sign is None):
    raise ValueError(f"{stage_tag} root not chosen; awaiting human")
if diagnostic_only and (root_dc is not None or root_sign is not None):
    raise ValueError(
        f"diagnostic mode requires an unset root, but {stage_tag} has "
        f"root_dc={root_dc!r}, root_sign={root_sign!r}"
    )
if root_sign is not None and int(root_sign) not in (-1, 1):
    raise ValueError(f"root_sign must be +1 or -1, got {root_sign!r}")
if apop_quantile is not None:
    apop_quantile = float(apop_quantile)
    if not 0.0 < apop_quantile < 1.0:
        raise ValueError(f"apop_quantile must be null or between 0 and 1, got {apop_quantile}")

log_path.parent.mkdir(parents=True, exist_ok=True)
log = log_path.open("w")


def say(message):
    print(f"[compute_pseudotime:{stage_tag}] {message}", file=log, flush=True)


# Outputs are written to a temp file and renamed so that a job killed mid-write
# never leaves a truncated file that Snakemake would treat as complete.
def atomic_tsv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_h5ad(adata, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.h5ad")
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def subtype_summary(adata, candidate):
    """Mean, median and count of dpt_pseudotime per (cell_type, cell_subtype)."""
    frame = pd.DataFrame(
        {
            "cell_type": adata.obs["cell_type"].astype(str).to_numpy(),
            "cell_subtype": adata.obs["cell_subtype"].astype(str).to_numpy(),
            "dpt_pseudotime": adata.obs["dpt_pseudotime"].to_numpy(dtype=float),
        },
        index=adata.obs_names,
    )
    out = (
        frame.groupby(["cell_type", "cell_subtype"], observed=True)["dpt_pseudotime"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .sort_values("mean")
    )
    out.insert(0, "candidate", candidate)
    return out


def evaluate_root(adata, root_ix, candidate):
    """Run DPT from root_ix and score orientation against the decidual marker score.

    Decidualization runs fibroblast -> decidual, so a correctly oriented pseudotime
    correlates positively with decidual_score and puts fibroblasts before decidual
    cells. Returns (metrics dict, per-subtype table).
    """
    adata.uns["iroot"] = int(root_ix)
    sc.tl.dpt(adata)

    pt = adata.obs["dpt_pseudotime"].to_numpy(dtype=float)
    score = adata.obs["decidual_score"].to_numpy(dtype=float)
    if not np.isfinite(pt).all():
        raise ValueError(f"{candidate} produced non-finite dpt_pseudotime values")
    if float(pt.min()) < -1e-7 or float(pt.max()) > 1.0 + 1e-7:  # float tolerance on [0, 1]
        raise ValueError(
            f"{candidate} produced dpt_pseudotime outside [0,1]: "
            f"{float(pt.min())}..{float(pt.max())}"
        )
    if not np.isfinite(score).all():
        raise ValueError("decidual_score contains non-finite values")

    corr = float(np.corrcoef(pt, score)[0, 1])
    types = adata.obs["cell_type"].astype(str).to_numpy()
    fib_mean = float(pt[types == "endometrial_fibroblast"].mean())
    dec_mean = float(pt[types == "decidual_cell"].mean())
    if not np.isfinite([corr, fib_mean, dec_mean]).all():
        raise ValueError(
            f"{candidate} produced non-finite orientation metrics: "
            f"corr={corr}, fibroblast_mean={fib_mean}, decidual_mean={dec_mean}"
        )

    # X_diffmap column 0 is the trivial constant component, so DC1 is column 1.
    metrics = {
        "candidate": candidate,
        "root_index": int(root_ix),
        "root_obs_name": str(adata.obs_names[root_ix]),
        "root_cell_barcode": str(adata.obs["cell_barcode"].iloc[root_ix]),
        "root_cell_subtype": str(adata.obs["cell_subtype"].iloc[root_ix]),
        "root_decidual_score": float(score[root_ix]),
        "root_dc1": float(adata.obsm["X_diffmap"][root_ix, 1]),
        "corr_pseudotime_decidual_score": corr,
        "mean_pseudotime_fibroblast": fib_mean,
        "mean_pseudotime_decidual": dec_mean,
        "fibroblast_mean_below_decidual_mean": bool(fib_mean < dec_mean),
    }
    return metrics, subtype_summary(adata, candidate)


try:
    # Seed every RNG that scVI training and root selection touch (Python, numpy,
    # torch CPU and CUDA, scvi) so that the chosen root is reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    scvi.settings.seed = seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this scratch scVI test")

    # Load and validate the staged object ----
    say(f"input={inp} | timepoint={timepoint} | diagnostic_only={diagnostic_only} | seed={seed}")
    adata = sc.read_h5ad(inp)

    required_obs = {"cell_type", "cell_subtype", "timepoint", "cell_barcode"}
    missing_obs = sorted(required_obs.difference(adata.obs.columns))
    if missing_obs:
        raise KeyError(f"missing canonical obs columns: {missing_obs}")
    if "counts" not in adata.layers:
        raise KeyError("staged input has no raw-count layers['counts']")

    # The staged migration carries only the canonical cell_type/cell_subtype columns;
    # the pre-migration label columns would mean the wrong object was passed in.
    forbidden = {"coarse_refined", "fine_refined"}.intersection(adata.obs.columns)
    if forbidden:
        raise ValueError(f"legacy annotation columns are forbidden here: {sorted(forbidden)}")

    # Subset to fibroblast/decidual cells at this timepoint ----
    fibdec_types = {"decidual_cell", "endometrial_fibroblast"}
    keep = adata.obs["cell_type"].astype(str).isin(fibdec_types)
    keep &= adata.obs["timepoint"].astype(str).eq(timepoint)
    work = adata[keep.to_numpy()].copy()
    del adata

    n_before_apop = int(work.n_obs)
    if n_before_apop == 0:
        raise ValueError(f"no fib/dec cells found for timepoint={timepoint!r}")
    if exclude_apoptotic_subtype:
        work = work[work.obs["cell_subtype"].astype(str).ne("decidual_apoptotic").to_numpy()].copy()
    n_after_subtype_filter = int(work.n_obs)
    say(
        f"subset={n_before_apop} cells; exclude_apoptotic_subtype={exclude_apoptotic_subtype} "
        f"-> {n_after_subtype_filter}"
    )

    # scVI needs integer counts in X.
    work.X = work.layers["counts"].copy()
    count_values = work.X.data if sp.issparse(work.X) else np.asarray(work.X).ravel()
    if not np.isfinite(count_values).all() or (count_values < 0).any():
        raise ValueError("counts are non-finite or negative")
    if not np.allclose(count_values, np.rint(count_values), atol=1e-6):
        raise ValueError("counts are not integer-like")

    # Apoptosis and decidual marker scores ----
    # Scored on a temporary log-normalized copy so that X stays on the count scale.
    scored = work.copy()
    sc.pp.normalize_total(scored, target_sum=1e4)  # CP10K
    sc.pp.log1p(scored)

    apoptosis_markers = [
        g for g in ["Casp3", "Casp8", "Casp9", "Bax", "Bak1", "Bbc3", "Pmaip1",
                    "Cycs", "Apaf1", "Bid", "Fas", "Tnfrsf10b", "Gadd45a", "Ddit3"]
        if g in scored.var_names
    ]
    decidual_markers = [
        g for g in ["Prl8a2", "Prl3c1", "Hand2", "Bmp2", "Wnt4", "Ptgs2", "Igfbp1", "Dkk1"]
        if g in scored.var_names
    ]
    if not apoptosis_markers or len(decidual_markers) < 4:
        raise ValueError(
            f"insufficient marker coverage: apoptosis={apoptosis_markers}, "
            f"decidual={decidual_markers}"
        )

    sc.tl.score_genes(
        scored, apoptosis_markers, score_name="apop_score", random_state=seed, use_raw=False
    )
    sc.tl.score_genes(
        scored, decidual_markers, score_name="decidual_score", random_state=seed, use_raw=False
    )
    work.obs["apop_score"] = scored.obs["apop_score"].to_numpy()
    work.obs["decidual_score"] = scored.obs["decidual_score"].to_numpy()

    # Optional expression-based apoptotic filter: drop cells above the configured
    # quantile of apop_score, in addition to the labelled apoptotic subtype.
    if apop_quantile is not None:
        threshold = float(np.quantile(work.obs["apop_score"], apop_quantile))
        work = work[work.obs["apop_score"].to_numpy() < threshold].copy()
        say(f"apop_quantile={apop_quantile} threshold={threshold:.6g} -> {work.n_obs} cells")
    else:
        threshold = None
        say("apop_quantile=null; no expression-score quantile filter")
    del scored

    # scVI embedding ----
    # Retrained from scratch on the subset, on seurat_v3 HVGs.
    sc.pp.highly_variable_genes(
        work, n_top_genes=n_hvg, flavor="seurat_v3", subset=False
    )
    hvg = work[:, work.var["highly_variable"].to_numpy()].copy()
    if hvg.n_vars != n_hvg:
        raise ValueError(f"expected {n_hvg} HVGs, selected {hvg.n_vars}")

    scvi.model.SCVI.setup_anndata(hvg, layer=None)
    model = scvi.model.SCVI(hvg, n_latent=n_latent)
    say(f"training scVI on {hvg.n_obs} cells x {hvg.n_vars} HVGs; n_latent={n_latent}")
    t0 = time.time()
    model.train(max_epochs=max_epochs, early_stopping=True)
    say(f"scVI complete in {time.time() - t0:.1f}s")
    work.obsm["X_scVI"] = model.get_latent_representation(hvg)
    del hvg, model

    # Diffusion map ----
    # X is put back on the log1p CP10K scale for the written object, and the
    # kNN graph is built on the scVI latent space.
    work.X = work.layers["counts"].copy()
    sc.pp.normalize_total(work, target_sum=1e4)  # CP10K
    sc.pp.log1p(work)

    sc.pp.neighbors(work, use_rep="X_scVI", n_neighbors=n_neighbors, random_state=seed)
    sc.tl.diffmap(work, n_comps=n_diffmap)
    say(f"computed neighbors k={n_neighbors} and diffmap n_comps={n_diffmap}")

    if diagnostic_only:
        # Candidate roots for human review ----
        # Three candidates are evaluated: the two DC1 extremes and the cell of the
        # most fibroblast-like subtype that lies farthest from the DC1 median.
        dc1 = np.asarray(work.obsm["X_diffmap"][:, 1])
        cell_types = work.obs["cell_type"].astype(str)

        fib = work.obs.loc[cell_types.eq("endometrial_fibroblast")].copy()
        fib_means = (
            fib.groupby(fib["cell_subtype"].astype(str), observed=True)["decidual_score"].mean()
        )
        most_fib_subtype = str(fib_means.idxmin())
        subtype_ix = np.flatnonzero(
            work.obs["cell_subtype"].astype(str).eq(most_fib_subtype).to_numpy()
        )
        dc_center = float(np.median(dc1))
        subtype_extreme_ix = int(subtype_ix[np.argmax(np.abs(dc1[subtype_ix] - dc_center))])

        candidates = [
            ("DC1_argmin", int(np.argmin(dc1))),
            ("DC1_argmax", int(np.argmax(dc1))),
            (f"most_extreme_{most_fib_subtype}", subtype_extreme_ix),
        ]
        metric_rows = []
        subtype_rows = []
        for candidate, root_ix in candidates:
            metrics, subtype_table = evaluate_root(work, root_ix, candidate)
            metrics["most_fibroblast_like_subtype"] = most_fib_subtype
            metrics["subtype_selection_rule"] = (
                "lowest mean decidual_score among endometrial_fibroblast subtypes"
            )
            metrics["extreme_selection_rule"] = (
                "largest absolute DC1 deviation from global DC1 median within subtype"
            )
            metric_rows.append(metrics)
            subtype_rows.append(subtype_table)
            say(
                f"candidate={candidate} root={metrics['root_obs_name']} "
                f"corr={metrics['corr_pseudotime_decidual_score']:+.6f}"
            )

        atomic_tsv(pd.DataFrame(metric_rows), snk.output.candidates)
        atomic_tsv(pd.concat(subtype_rows, ignore_index=True), snk.output.subtypes)
        atomic_json(
            {
                "status": "AWAITING_HUMAN_ROOT_CHOICE",
                "statement": "3dpi root awaiting human choice — candidates evaluated below.",
                "timepoint": timepoint,
                "stage_tag": stage_tag,
                "input": str(inp),
                "seed": seed,
                "n_cells": int(work.n_obs),
                "most_fibroblast_like_subtype": most_fib_subtype,
                "candidate_count": len(metric_rows),
                "final_h5ad_written": False,
            },
            snk.output.summary,
        )
        say("3dpi root awaiting human choice; no H5AD written")
    else:
        # Root from the configured diffusion component ----
        # The root is the cell at the root_sign end of DC{root_dc}, optionally
        # restricted to root_subtype.
        root_dc = int(root_dc)
        root_sign = int(root_sign)
        if root_dc <= 0 or root_dc >= work.obsm["X_diffmap"].shape[1]:
            raise ValueError(f"root_dc={root_dc} outside computed diffusion components")

        eligible = np.arange(work.n_obs)
        if root_subtype is not None:
            eligible = np.flatnonzero(
                work.obs["cell_subtype"].astype(str).eq(str(root_subtype)).to_numpy()
            )
            if eligible.size == 0:
                raise ValueError(f"root_subtype={root_subtype!r} has no cells")

        signed_dc = np.asarray(work.obsm["X_diffmap"][:, root_dc]) * root_sign
        root_ix = int(eligible[np.argmin(signed_dc[eligible])])
        candidate = f"DC{root_dc}_{'argmin' if root_sign == 1 else 'argmax'}"

        metrics, subtype_table = evaluate_root(work, root_ix, candidate)
        metrics.update(
            {
                "status": "PASS" if metrics["corr_pseudotime_decidual_score"] > 0 else "FAIL",
                "input": str(inp),
                "timepoint": timepoint,
                "stage_tag": stage_tag,
                "seed": seed,
                "root_dc": root_dc,
                "root_sign": root_sign,
                "root_subtype": root_subtype,
                "n_cells_before_apoptotic_filters": n_before_apop,
                "n_cells_after_subtype_filter": n_after_subtype_filter,
                "n_cells_final": int(work.n_obs),
                "exclude_apoptotic_subtype": exclude_apoptotic_subtype,
                "apop_quantile": apop_quantile,
                "apop_threshold": threshold,
                "n_hvg": n_hvg,
                "n_latent": n_latent,
                "n_neighbors": n_neighbors,
                "max_epochs": max_epochs,
            }
        )
        say(
            f"root={metrics['root_obs_name']} subtype={metrics['root_cell_subtype']} "
            f"corr={metrics['corr_pseudotime_decidual_score']:+.6f}"
        )

        # Orientation gate: a wrong-end root fails the run instead of being flipped,
        # so the config has to be corrected rather than the sign patched downstream.
        if metrics["corr_pseudotime_decidual_score"] <= 0:
            raise ValueError(
                f"orientation failed for {stage_tag}: corr(pseudotime, decidual_score)="
                f"{metrics['corr_pseudotime_decidual_score']:+.6f}; root was not flipped"
            )
        if not metrics["fibroblast_mean_below_decidual_mean"]:
            raise ValueError(
                f"orientation failed for {stage_tag}: "
                f"mean fibroblast pseudotime is not below decidual"
            )

        # Transfer-ready object ----
        # The transfer convention is bare-barcode obs_names, so collisions across
        # samples are expected and only logged.
        output = work.copy()
        output.obs["source_obs_name"] = output.obs_names.astype(str)
        output.obs_names = pd.Index(output.obs["cell_barcode"].astype(str).to_numpy())
        if not output.obs_names.is_unique:
            say(
                f"WARN bare-barcode obs_names are non-unique: "
                f"{int(output.obs_names.duplicated(keep=False).sum())} rows "
                f"participate in collisions"
            )

        keep_obs = [
            "dpt_pseudotime", "decidual_score", "apop_score", "cell_subtype",
            "cell_type", "timepoint", "cell_barcode", "source_obs_name",
        ]
        output.obs = output.obs[keep_obs].copy()

        atomic_h5ad(output, snk.output.h5ad)
        atomic_tsv(subtype_table, snk.output.subtypes)
        atomic_json(metrics, snk.output.metrics)
        say(f"wrote scratch transfer object -> {snk.output.h5ad}")
finally:
    log.close()
