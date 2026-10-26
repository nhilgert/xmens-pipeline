"""Fit one SAMap pairing with stock sc-samap v3 (3.0.1).

Reads the two per-timepoint SAMap input h5ads (--hs-h5ad, --mm-h5ad) and the
hsmm/{hs_to_mm.txt, mm_to_hs.txt} homology maps under --maps-dir (its parent, by
SAMap convention). Writes samap.pkl, meta.json and, best-effort,
figures/joint_umap_species.png under --out-dir. Called by rule run_samap
(workflow/rules/10_samap.smk).

Only the final fitted object is saved: stock v3 exposes no per-iteration hook.
NUMITERS is a deprecated alias of n_iterations in v3 (it warns), and ncpus, f_maps
and save_processed are v3 kwargs (samap/core/mapping.py).
"""
import argparse, json, time
from datetime import datetime
from pathlib import Path

from samap import SAMAP, save_samap  # v3 exports both at top level, not samap.mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hs-h5ad", required=True)
    p.add_argument("--mm-h5ad", required=True)
    p.add_argument("--maps-dir", required=True, help="Parent of hsmm/ (SAMap convention)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-iter", type=int, default=3)
    p.add_argument("--ncpus", type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SAMap resolves the maps from <maps_dir>/<sp1><sp2>/ and fails late otherwise.
    maps_dir = Path(args.maps_dir)
    pair = maps_dir / "hsmm"
    if not pair.is_dir():
        raise SystemExit(
            f"Expected {maps_dir}/hsmm/{{hs_to_mm.txt, mm_to_hs.txt}} — {pair} missing."
        )
    for fn in ("hs_to_mm.txt", "mm_to_hs.txt"):
        if not (pair / fn).is_file():
            raise SystemExit(f"Missing maps file: {pair / fn}")

    # Provenance is written before the fit so a crashed run still leaves a trail.
    meta = {
        "started_at": datetime.now().isoformat(),
        "samap": "stock sc-samap v3",
        "hs_h5ad": str(Path(args.hs_h5ad).resolve()),
        "mm_h5ad": str(Path(args.mm_h5ad).resolve()),
        "maps_dir": str(maps_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_iter": args.n_iter, "ncpus": args.ncpus,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    import samap
    print(f"samap module: {samap.__file__}")  # confirms the stock install in the log

    # Fit ----
    filenames = {"hs": args.hs_h5ad, "mm": args.mm_h5ad}
    print("Initializing SAMAP...")
    t0 = time.time()
    sm = SAMAP(filenames, f_maps=str(maps_dir) + "/", save_processed=False)
    print(f"  initialized in {time.time() - t0:.1f}s")

    print(f"Running SAMap (NUMITERS={args.n_iter}, ncpus={args.ncpus})...")
    t0 = time.time()
    sm.run(NUMITERS=args.n_iter, ncpus=args.ncpus)
    elapsed = time.time() - t0
    print(f"  done in {elapsed/60:.1f} min")

    # Save ----
    samap_path = out_dir / "samap.pkl"
    save_samap(sm, str(samap_path))
    print(f"  wrote {samap_path} ({samap_path.stat().st_size/1e6:.1f} MB)")

    meta.update(finished_at=datetime.now().isoformat(),
                wall_time_seconds=elapsed,
                n_cells_joint=int(sm.samap.adata.n_obs))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # The species UMAP is a diagnostic, not a declared rule output, so a render
    # failure must not fail the fit.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt, scanpy as sc
        figdir = out_dir / "figures"
        figdir.mkdir(exist_ok=True)
        fig = sc.pl.embedding(sm.samap.adata, basis="X_umap", color="species",
                              return_fig=True, show=False)
        fig.savefig(figdir / "joint_umap_species.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {figdir/'joint_umap_species.png'}")
    except Exception as e:
        print(f"  UMAP render skipped (non-fatal): {e}")

    print("Done.")


if __name__ == "__main__":
    main()
