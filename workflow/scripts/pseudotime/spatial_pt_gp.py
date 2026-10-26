#!/usr/bin/env python
"""
spatial_pt_gp.py — spatial Gaussian-process fit of pseudotime, single species.

Standalone module distilled from revisions/gaston/cross_species_radial_gp.py.
The cross-species part (SAMap load, barcode alignment, Nadaraya-Watson mm->hs
transfer) is REMOVED — this fits the GP natively: pseudotime and spatial coords
live on the same cells, read from one h5ad. The transfer + cross-species gradient
comparison is deferred to a later module.

What it does, per (h5ad, pt-key, timepoint):
  1. subset to the timepoint, keep cells with finite pseudotime + spatial coords
  2. optional geosketch to --sketch-n cells (default 5000; exact GP is O(n^3))
  3. one-sided MAD outlier filter on pseudotime (drops upper-tail scatter)
  4. fit ExactGP: ScaleKernel(Matern nu=2.5, ard_num_dims=2) over (x, y) microns,
     GaussianLikelihood, Adam, --train-iters steps
  5. lengthscale sweep (fixed values + learned), chosen by log marginal likelihood
  6. gradient of the GP posterior mean via autograd -> grad field on a grid
     (kept from the original: it is one function and the quiver/streamline figures
      are the readout — the DEFERRED part is the cross-species transfer, not this)
  7. permutation null (shuffled pt, same GP) -> MLL gap as the signal check
  8. outputs under --outdir:
       gp_fit_summary.json     chosen lengthscale, MLL per candidate, null MLL, n cells
       gp_field.npz            grid xy, posterior mean, gradient field, cell xy + pt
       figures/{01_pt_spatial, 02_lengthscale_sweep, 03_streamlines, 04_null}.png

Usage:
    python spatial_pt_gp.py \
        --h5ad   pseudotime_mouse_v2_induced_no_apop/2026_08_28_mouse_fibdec_pseudotime_rooted.h5ad \
        --pt-key palantir_pt_rtk \
        --timepoint 1_dpi_late \
        --x-key x_um_dbscan --y-key y_um_dbscan \
        --outdir results/spatial_gp/mouse_palantir_rtk_1_dpi_late

Env: needs torch + gpytorch (+ geosketch if --sketch-n). CPU is fine at 5k cells;
CUDA is used automatically when available.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ CLI
def parse_args():
    p = argparse.ArgumentParser(description="Spatial GP fit of pseudotime (single species, no transfer)")
    p.add_argument("--h5ad", required=True, help="object carrying pseudotime AND spatial coords")
    p.add_argument("--pt-key", default="dpt_pseudotime",
                   help="obs column: dpt_pseudotime, palantir_pt_rtk, palantir_pt_std, ...")
    p.add_argument("--timepoint", default=None,
                   help="subset obs['timepoint'] to this value (default: all cells)")
    p.add_argument("--x-key", default="x_um_dbscan")
    p.add_argument("--y-key", default="y_um_dbscan")
    p.add_argument("--sample-key", default="sample",
                   help="obs column for per-sample facets in figures (optional)")
    p.add_argument("--sketch-n", type=int, default=5000,
                   help="geosketch target size; 0 disables sketching")
    p.add_argument("--outlier-mad", type=float, default=3.0,
                   help="one-sided upper-tail pt outlier threshold in local MADs; 0 disables")
    p.add_argument("--lengthscales", type=float, nargs="*", default=[500.0, 750.0, 1250.0, 1500.0],
                   help="fixed lengthscale candidates (microns); a learned-ls fit is always added")
    p.add_argument("--train-iters", type=int, default=150)
    p.add_argument("--grid-res", type=int, default=200)
    p.add_argument("--run-null", action="store_true", default=True)
    p.add_argument("--no-null", dest="run_null", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=pathlib.Path, required=True)
    return p.parse_args()


# ------------------------------------------------------------------ GP pieces
# (verbatim structure from cross_species_radial_gp.py, minus the transfer)
def make_gp_classes():
    import torch
    import gpytorch

    class ExactGP(gpytorch.models.ExactGP):
        def __init__(self, tx, ty, lik, kernel):
            super().__init__(tx, ty, lik)
            self.mean = gpytorch.means.ConstantMean()
            self.cov = kernel

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean(x), self.cov(x))

    return torch, gpytorch, ExactGP


def fit_gp(torch, gpytorch, ExactGP, device, X, y, lengthscale=None, train_iters=150):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)).to(device)
    if lengthscale is not None:
        kernel.base_kernel.lengthscale = torch.tensor([[lengthscale, lengthscale]], device=device)
        kernel.base_kernel.raw_lengthscale.requires_grad_(False)
    model = ExactGP(X, y, likelihood, kernel).to(device)
    model.train(); likelihood.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(train_iters):
        opt.zero_grad()
        loss = -mll(model(X), y)
        loss.backward()
        opt.step()
    model.eval(); likelihood.eval()
    return model, likelihood


def log_marginal_likelihood(torch, gpytorch, model, likelihood, X, y):
    model.train(); likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    with torch.no_grad():
        return mll(model(X), y).item() * len(y)


def gp_gradient(torch, gpytorch, model, lik, X_query):
    X_query = X_query.clone().detach().requires_grad_(True)
    with gpytorch.settings.fast_pred_var():
        mean = lik(model(X_query)).mean
    (grads,) = torch.autograd.grad(mean.sum(), X_query)
    return grads.detach().cpu().numpy()


def gp_mean(torch, gpytorch, model, lik, X_query):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return lik(model(X_query)).mean.cpu().numpy()


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    import scanpy as sc

    out = args.outdir.resolve()
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary = {"h5ad": str(args.h5ad), "pt_key": args.pt_key, "timepoint": args.timepoint,
               "seed": args.seed}

    # ---- 1. load + subset ---------------------------------------------------
    a = sc.read_h5ad(args.h5ad)
    if args.pt_key not in a.obs.columns:
        raise SystemExit(f"pt key {args.pt_key!r} not in obs. Have: {sorted(a.obs.columns)}")
    if args.timepoint is not None:
        m = (a.obs["timepoint"].astype(str) == args.timepoint).values
        if m.sum() == 0:
            raise SystemExit(f"timepoint {args.timepoint!r} has 0 cells. "
                             f"Available: {sorted(a.obs['timepoint'].astype(str).unique())}")
        a = a[m].copy()

    pt = a.obs[args.pt_key].to_numpy(float)
    x = a.obs[args.x_key].to_numpy(float)
    y = a.obs[args.y_key].to_numpy(float)
    ok = np.isfinite(pt) & np.isfinite(x) & np.isfinite(y)
    summary["n_cells_subset"] = int(a.n_obs)
    summary["n_cells_with_pt_and_coords"] = int(ok.sum())
    print(f"cells: {a.n_obs:,} in subset; {ok.sum():,} with finite pt + coords", flush=True)
    if ok.sum() < 200:
        raise SystemExit("fewer than 200 usable cells — refusing a meaningless GP fit")

    X_all = np.stack([x[ok], y[ok]], axis=1)
    y_all = pt[ok]
    samples = a.obs[args.sample_key].astype(str).to_numpy()[ok] \
        if args.sample_key in a.obs.columns else np.repeat("all", ok.sum())

    # ---- 2. sketch ----------------------------------------------------------
    if args.sketch_n and len(y_all) > args.sketch_n:
        try:
            from geosketch import gs
            idx = np.array(gs(X_all, args.sketch_n, replace=False))
        except ImportError:
            print("geosketch not importable — falling back to uniform subsample", flush=True)
            idx = rng.choice(len(y_all), args.sketch_n, replace=False)
        X_fit, y_fit = X_all[idx], y_all[idx]
    else:
        X_fit, y_fit = X_all, y_all
    summary["n_cells_fit_pre_outlier"] = int(len(y_fit))

    # ---- 3. one-sided MAD outlier filter ------------------------------------
    if args.outlier_mad > 0:
        med = np.median(y_fit)
        mad = np.median(np.abs(y_fit - med)) or 1e-12
        keep = (y_fit - med) / mad <= args.outlier_mad     # upper tail only
        n_drop = int((~keep).sum())
        X_fit, y_fit = X_fit[keep], y_fit[keep]
        summary["n_outliers_dropped"] = n_drop
        print(f"outlier filter: dropped {n_drop} upper-tail cells", flush=True)

    summary["n_cells_fit"] = int(len(y_fit))

    # ---- 4-5. fit + lengthscale sweep ---------------------------------------
    torch, gpytorch, ExactGP = make_gp_classes()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary["device"] = device
    tX = torch.tensor(X_fit, dtype=torch.float32, device=device)
    ty = torch.tensor(y_fit, dtype=torch.float32, device=device)

    candidates = list(args.lengthscales) + [None]     # None = learned
    fits, mlls = {}, {}
    for ls in candidates:
        tag = f"{ls:g}um" if ls is not None else "learned"
        model, lik = fit_gp(torch, gpytorch, ExactGP, device, tX, ty,
                            lengthscale=ls, train_iters=args.train_iters)
        mll = log_marginal_likelihood(torch, gpytorch, model, lik, tX, ty)
        fits[tag] = (model, lik)
        mlls[tag] = mll
        print(f"  lengthscale {tag:>9s}: MLL = {mll:.1f}", flush=True)
    best = max(mlls, key=mlls.get)
    model, lik = fits[best]
    learned_ls = model.cov.base_kernel.lengthscale.detach().cpu().numpy().ravel().tolist()
    summary["mll_per_lengthscale"] = {k: round(v, 2) for k, v in mlls.items()}
    summary["chosen_lengthscale"] = best
    summary["chosen_lengthscale_values_um"] = [round(v, 1) for v in learned_ls]
    print(f"chosen: {best} (ls = {learned_ls})", flush=True)

    # ---- 6. posterior mean + gradient on a grid ------------------------------
    pad = 0.03 * max(X_fit[:, 0].ptp(), X_fit[:, 1].ptp())
    gx = np.linspace(X_fit[:, 0].min() - pad, X_fit[:, 0].max() + pad, args.grid_res)
    gy = np.linspace(X_fit[:, 1].min() - pad, X_fit[:, 1].max() + pad, args.grid_res)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
    tgrid = torch.tensor(grid, dtype=torch.float32, device=device)
    mean_grid = gp_mean(torch, gpytorch, model, lik, tgrid).reshape(GX.shape)
    grad_grid = gp_gradient(torch, gpytorch, model, lik, tgrid).reshape(*GX.shape, 2)

    np.savez_compressed(out / "gp_field.npz",
                        grid_x=GX, grid_y=GY, mean=mean_grid,
                        grad_x=grad_grid[..., 0], grad_y=grad_grid[..., 1],
                        cell_xy=X_fit, cell_pt=y_fit)

    # ---- 7. permutation null --------------------------------------------------
    if args.run_null:
        y_null = y_fit.copy()
        rng.shuffle(y_null)
        tn = torch.tensor(y_null, dtype=torch.float32, device=device)
        ls_fixed = None if best == "learned" else float(best.rstrip("um"))
        nmodel, nlik = fit_gp(torch, gpytorch, ExactGP, device, tX, tn,
                              lengthscale=ls_fixed, train_iters=args.train_iters)
        null_mll = log_marginal_likelihood(torch, gpytorch, nmodel, nlik, tX, tn)
        summary["null_mll"] = round(null_mll, 2)
        summary["mll_gap_over_null"] = round(mlls[best] - null_mll, 2)
        print(f"null MLL = {null_mll:.1f}  (gap = {mlls[best] - null_mll:.1f})", flush=True)

    # ---- 8. figures -----------------------------------------------------------
    f1, ax = plt.subplots(figsize=(7, 6))
    s = ax.scatter(X_fit[:, 0], X_fit[:, 1], c=y_fit, s=4, cmap="viridis")
    plt.colorbar(s, ax=ax, label=args.pt_key)
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_title(f"{args.pt_key} in space  ({args.timepoint or 'all'}; n={len(y_fit):,})")
    ax.set_aspect("equal")
    f1.tight_layout(); f1.savefig(fig / "01_pt_spatial.png", dpi=150); plt.close(f1)

    f2, ax = plt.subplots(figsize=(6, 4))
    tags = list(mlls); vals = [mlls[t] for t in tags]
    ax.bar(range(len(tags)), vals, color=["#C44E52" if t == best else "#4C72B0" for t in tags])
    ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.set_ylabel("log marginal likelihood"); ax.set_title("lengthscale sweep (red = chosen)")
    f2.tight_layout(); f2.savefig(fig / "02_lengthscale_sweep.png", dpi=150); plt.close(f2)

    f3, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(GX, GY, mean_grid, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax, label=f"GP mean {args.pt_key}")
    step = max(1, args.grid_res // 30)
    ax.streamplot(gx, gy, grad_grid[..., 0], grad_grid[..., 1],
                  color="white", density=1.2, linewidth=0.7, arrowsize=0.8)
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_title(f"GP posterior mean + gradient streamlines ({best})")
    ax.set_aspect("equal")
    f3.tight_layout(); f3.savefig(fig / "03_streamlines.png", dpi=150); plt.close(f3)

    if args.run_null:
        f4, ax = plt.subplots(figsize=(5, 4))
        ax.bar([0, 1], [mlls[best], summary["null_mll"]], color=["#4C72B0", "#999999"])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["observed", "shuffled pt"])
        ax.set_ylabel("log marginal likelihood")
        ax.set_title(f"signal vs permutation null (gap {summary['mll_gap_over_null']:.0f})")
        f4.tight_layout(); f4.savefig(fig / "04_null.png", dpi=150); plt.close(f4)

    (out / "gp_fit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out/'gp_fit_summary.json'}\n      {out/'gp_field.npz'}\n      figures/ (4)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
