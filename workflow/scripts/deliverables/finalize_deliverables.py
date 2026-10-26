#!/usr/bin/env python
"""finalize_deliverables (04g) -- promote source objects into an immutable,
dated, sha256-manifested release under data/deliverables/<release>/.

Promotion, NOT regeneration: the deliverable objects are built/enriched by the
notebooks; this rule makes the hand-off auditable and single-sourced. Per
object it:
  * streams a sha256 of the source, copies with mtime preserved, verifies the
    copy's sha256 matches (corruption check across filesystems),
  * validates canonical annotations against annotation_vocabulary.yaml using
    the SAME checker the pipeline uses (apply_annotation.validate_canonical_annotations),
  * records structure (n_obs/n_vars, obs columns, layers, obsm, uns keys) and
    source provenance in MANIFEST.json.

Releases are immutable: an existing release file with a matching sha256 is
skipped (idempotent re-runs); a mismatching one is a hard error -- bump
config[deliverables][release] instead of overwriting.
"""
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import yaml

# the shared annotation checker lives in workflow/scripts/atlas/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atlas"))
from apply_annotation import validate_canonical_annotations  # noqa: E402

snk = snakemake  # noqa: F821 (injected by Snakemake)
sources = dict(snk.params.sources)
release = str(snk.params.release)
note = str(snk.params.note or "")
vocab = yaml.safe_load(open(snk.input.vocab))
outdir = Path(str(snk.output.manifest)).parent
outdir.mkdir(parents=True, exist_ok=True)
log = open(snk.log[0], "w")


def say(msg):
    print(f"[finalize:{release}] {msg}", file=log, flush=True)
    print(f"[finalize:{release}] {msg}", flush=True)


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


entries = {}
for name, src in sources.items():
    src = Path(src)
    dst = outdir / f"{name}.h5ad"
    if not src.exists():
        raise SystemExit(f"[{name}] source missing: {src}")
    say(f"{name}: hashing source {src}")
    src_sha = sha256(src)

    if dst.exists():
        if sha256(dst) == src_sha:
            say(f"{name}: already promoted (sha match) -- skipping copy")
        else:
            raise SystemExit(
                f"[{name}] {dst} exists with a DIFFERENT sha256 than the source. "
                f"Releases are immutable: bump config[deliverables][release] "
                f"rather than overwriting.")
    else:
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
        say(f"{name}: copying ({src.stat().st_size / 1e9:.2f} GB)")
        shutil.copy2(src, tmp)          # copy2 preserves the source mtime
        copy_sha = sha256(tmp)
        if copy_sha != src_sha:
            tmp.unlink()
            raise SystemExit(f"[{name}] copy sha256 mismatch -- transfer corrupted")
        os.replace(tmp, dst)
        say(f"{name}: promoted -> {dst}")

    a = ad.read_h5ad(dst, backed="r")
    validate_canonical_annotations(a, vocab, context=f"{release}/{name}")
    say(f"{name}: canonical annotations valid "
        f"({a.n_obs:,} cells x {a.n_vars:,} genes)")
    entries[name] = {
        "file": dst.name,
        "sha256": src_sha,
        "size_bytes": int(dst.stat().st_size),
        "source_path": str(src),
        "source_mtime": datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds"),
        "n_obs": int(a.n_obs),
        "n_vars": int(a.n_vars),
        "obs_columns": sorted(map(str, a.obs.columns)),
        "layers": sorted(map(str, a.layers.keys())),
        "obsm": sorted(map(str, a.obsm.keys())),
        "uns_keys": sorted(map(str, a.uns.keys())),
    }
    del a

manifest = {
    "release": release,
    "created": datetime.now().isoformat(timespec="seconds"),
    "note": note,
    "vocabulary_schema": vocab.get("schema_version"),
    "objects": entries,
}
tmp = Path(str(snk.output.manifest) + f".{os.getpid()}.tmp")
tmp.write_text(json.dumps(manifest, indent=2) + "\n")
os.replace(tmp, str(snk.output.manifest))
say(f"wrote {snk.output.manifest}")
log.close()
