"""Stage 00a — normalize fork Slide-tags coords into the xmens layout.

Reads the per-sample coordinate CSV produced by the vendored fork
(external/slidetags, run separately), validates a barcode + numeric x/y, and
writes the canonical data/interim/coords/{sample}.csv that 00b_ingest attaches
as adata.obsm['spatial']. Faithful pass-through: no spatial-outlier filtering
or coordinate transforms — QC owns that downstream.
"""
import os
import pandas as pd

src = snakemake.input.fork_coords      # noqa: F821 (snakemake-injected)
dst = snakemake.output.coords          # noqa: F821
sample = snakemake.wildcards.sample    # noqa: F821
logfile = snakemake.log[0]             # noqa: F821

with open(logfile, "w") as log:
    def say(msg):
        print(f"[00a:{sample}] {msg}", file=log, flush=True)

    say(f"reading {src}")
    df = pd.read_csv(src)
    say(f"loaded {df.shape[0]} rows, columns={list(df.columns)}")

    lower = {c.lower(): c for c in df.columns}
    def pick(name, *alts):
        for key in (name, *alts):
            if key in lower:
                return lower[key]
        raise KeyError(
            f"no '{name}' column in {src}; got {list(df.columns)}. "
            f"Fix the header in export_coords.R or adjust 00a."
        )

    bc = pick("barcode", "cb", "cell", "cell_barcode")
    xc = pick("x", "x_um", "spatial_x")
    yc = pick("y", "y_um", "spatial_y")

    out = df[[bc, xc, yc]].rename(columns={bc: "barcode", xc: "x", yc: "y"})
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")

    n0 = len(out)
    out = out.dropna(subset=["x", "y"])
    if len(out) < n0:
        say(f"dropped {n0 - len(out)} rows with non-numeric/NaN coords")

    dups = int(out["barcode"].duplicated().sum())
    if dups:
        say(f"warning: {dups} duplicate barcodes; keeping first")
        out = out.drop_duplicates(subset="barcode", keep="first")

    if out.empty:
        raise ValueError(f"no valid coordinates remain for {sample} after cleaning {src}")

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out.to_csv(dst, index=False)
    say(f"wrote {len(out)} rows -> {dst}")
