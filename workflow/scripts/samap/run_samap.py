#!/usr/bin/env python
"""
run_samap.py  —  STOCK SAMap v3 (sc-samap 3.0.1). No instrumented fork.

Ported from the fork version. The fork-only machinery is REMOVED because it raises TypeError
under stock v3:
  * `iteration_callback=` kwarg to .run()               -> gone in v3
  * per-iteration capture (gnnm_pruned / X_wPCA /        -> fork-only; not available
     per-iter alignment_scores) and the resulting snapshots.pkl
So this writes the FINAL fitted object only. THERE IS NO snapshots.pkl anymore — the fit rule's
outputs were updated to match.

Usage:
    python run_samap.py --hs-h5ad H --mm-h5ad M --maps-dir DIR --out-dir OUT [--n-iter 3] [--ncpus 8]
--maps-dir is the PARENT of hsmm/{hs_to_mm.txt, mm_to_hs.txt} (SAMap convention).

Outputs under --out-dir:  samap.pkl (via save_samap), meta.json, figures/joint_umap_species.png (best-effort)

⚠ VERIFY BEFORE THE FIRST FIT — the v3 *constructor* signature was not confirmed upstream
   (only .run() was). One command settles it:
     python -c "import inspect, samap; \
       print('init', inspect.signature(samap.SAMAP.__init__)); \
       print('run ', inspect.signature(samap.SAMAP.run))"
   If SAMAP.__init__ does NOT accept `f_maps` / `save_processed`, adjust the SAMAP(...) call below.
   (.run() legacy alias NUMITERS is confirmed accepted; `ncpus` is a confirmed v3 kwarg.)
"""
import argparse, json, time
from datetime import datetime
from pathlib import Path

from samap import SAMAP, save_samap      # v3: both top-level (NOT samap.mapping)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hs-h5ad", required=True)
    p.add_argument("--mm-h5ad", required=True)
    p.add_argument("--maps-dir", required=True, help="Parent of hsmm/ (SAMap convention)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-iter", type=int, default=3)
    p.add_argument("--ncpus", type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # verify SAMap-expected maps layout
    maps_dir = Path(args.maps_dir); pair = maps_dir / "hsmm"
    if not pair.is_dir():
        raise SystemExit(f"Expected {maps_dir}/hsmm/{{hs_to_mm.txt, mm_to_hs.txt}} — {pair} missing.")
    for fn in ("hs_to_mm.txt", "mm_to_hs.txt"):
        if not (pair / fn).is_file():
            raise SystemExit(f"Missing maps file: {pair / fn}")

    # provenance written early (crash trail)
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
    print(f"samap module: {samap.__file__}")          # confirms STOCK samap (no fork path)

    filenames = {"hs": args.hs_h5ad, "mm": args.mm_h5ad}
    print("Initializing SAMAP...")
    t0 = time.time()
    # kwargs carried over from the fork call; verify against v3 __init__ signature (see docstring).
    sm = SAMAP(filenames, f_maps=str(maps_dir) + "/", save_processed=False)
    print(f"  initialized in {time.time() - t0:.1f}s")

    print(f"Running SAMap (NUMITERS={args.n_iter}, ncpus={args.ncpus})...")
    t0 = time.time()
    sm.run(NUMITERS=args.n_iter, ncpus=args.ncpus)    # iteration_callback REMOVED (fork-only)
    elapsed = time.time() - t0
    print(f"  done in {elapsed/60:.1f} min")

    samap_path = out_dir / "samap.pkl"
    save_samap(sm, str(samap_path))
    print(f"  wrote {samap_path} ({samap_path.stat().st_size/1e6:.1f} MB)")

    meta.update(finished_at=datetime.now().isoformat(),
                wall_time_seconds=elapsed,
                n_cells_joint=int(sm.samap.adata.n_obs))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # best-effort species UMAP (non-fatal)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, scanpy as sc
        figdir = out_dir / "figures"; figdir.mkdir(exist_ok=True)
        fig = sc.pl.embedding(sm.samap.adata, basis="X_umap", color="species",
                              return_fig=True, show=False)
        fig.savefig(figdir / "joint_umap_species.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {figdir/'joint_umap_species.png'}")
    except Exception as e:
        print(f"  UMAP render skipped (non-fatal): {e}")

    print("Done.")


if __name__ == "__main__":
    main()
