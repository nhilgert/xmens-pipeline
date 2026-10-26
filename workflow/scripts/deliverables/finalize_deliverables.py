"""Promote source objects into an immutable, dated, sha256-manifested release.

Called by rule `finalize_deliverables` (workflow/rules/04g_deliverables.smk). Reads
the source h5ads named in config[deliverables][sources] and annotation_vocabulary.yaml.
Writes data/deliverables/<release>/<name>.h5ad for each source and MANIFEST.json.

This is promotion, not regeneration: the source objects are produced upstream, and this
step makes the hand-off auditable and single-sourced.
Per object it streams a sha256 of the source, copies with mtime preserved, verifies
the copy's sha256 (corruption check across filesystems), validates the canonical
annotations with the same checker the pipeline uses
(apply_annotation.validate_canonical_annotations), and records structure
(n_obs/n_vars, obs columns, layers, obsm, uns keys) and source provenance in the
manifest.

Releases are immutable. An existing release file with a matching sha256 is skipped,
so re-runs are idempotent. A mismatching one is a hard error: bump
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

# The shared annotation checker lives in workflow/scripts/atlas/; Snakemake
# script directives cannot import sibling directories without extending the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "atlas"))
from apply_annotation import validate_canonical_annotations  # noqa: E402

# Snakemake inputs ----
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


def sha256(path, chunk=1 << 24):  # 16 MiB per read (the h5ads are multi-GB)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# Promote each source ----
entries = {}
for name, src in sources.items():
    src = Path(src)
    dst = outdir / f"{name}.h5ad"
    if not src.exists():
        raise SystemExit(f"[{name}] source missing: {src}")
    say(f"{name}: hashing source {src}")
    src_sha = sha256(src)

    # Copy through a temp file and verify the hash before the rename, so a
    # partial or corrupted transfer never appears under the release name.
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
        say(f"{name}: copying ({src.stat().st_size / 1e9:.2f} GB)")  # bytes -> GB
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

# Manifest ----
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
