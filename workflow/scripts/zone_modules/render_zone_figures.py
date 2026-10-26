"""Render the configured decidual-zone figures for one AUCell score run.

Called by rule zone_figures (workflow/rules/16_zone_modules.smk). Runs fig_zone_modules.py
once per entry of --jobs (config zone_modules.figures: a JSON list of {name, args}) with the
shared inputs --obj, --scores and --hide-barcodes, writing every figure into --out. Then writes
MANIFEST.json there: every output file with its md5 and size, the invocations, and the inputs
with md5. --out must not exist or must be empty (Snakemake removes a stale directory output
before re-running the rule).
"""
import argparse, datetime, hashlib, json, os, shlex, subprocess, sys


def md5(path):
    h = hashlib.md5()
    f = open(path, "rb")

    for b in iter(lambda: f.read(1 << 20), b""):  # 1 MiB blocks
        h.update(b)

    return h.hexdigest()


def main():
    # Arguments ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--script", required=True)  # fig_zone_modules.py
    ap.add_argument("--obj", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--hide-barcodes", default="")
    ap.add_argument("--jobs", required=True)  # JSON list of {"name": str, "args": str}
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"refusing to write into non-empty {a.out}")

    jobs = json.loads(a.jobs)
    names = [j["name"] for j in jobs]
    assert len(set(names)) == len(names), f"duplicate figure job names: {names}"

    os.makedirs(a.out, exist_ok=True)

    # Run one figure job per config entry ----
    shared = [
        "--obj",
        a.obj,
        "--scores",
        a.scores,
        "--out",
        a.out,
    ] + (["--hide-barcodes", a.hide_barcodes] if a.hide_barcodes else [])
    runs = []

    for j in jobs:
        cmd = [a.python, a.script] + shared + shlex.split(j["args"])
        print("RUN", j["name"], ":", " ".join(shlex.quote(x) for x in cmd), flush=True)

        r = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)

        if r.returncode != 0:
            sys.exit(f"figure job {j['name']} failed with exit code {r.returncode}")

        # The figure script reports each file it saves on a "wrote <path> ..." line.
        wrote = [
            x
            for line in r.stdout.splitlines()
            if line.startswith("wrote ")
            for x in line.split()[1:]
        ]
        runs.append(
            {
                "name": j["name"],
                "args": j["args"],
                "files": [os.path.basename(x) for x in wrote],
            }
        )

    # Manifest ----
    files = sorted(f for f in os.listdir(a.out) if f != "MANIFEST.json")
    man = {
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "script": {"path": os.path.abspath(a.script), "md5": md5(a.script)},
        "driver": {"path": os.path.abspath(__file__), "md5": md5(os.path.abspath(__file__))},
        "inputs": {
            "obj": {"path": os.path.abspath(a.obj), "bytes": os.path.getsize(a.obj)},
            "scores": {"path": os.path.abspath(a.scores), "md5": md5(a.scores)},
            "hide_barcodes": (
                {"path": os.path.abspath(a.hide_barcodes), "md5": md5(a.hide_barcodes)}
                if a.hide_barcodes
                else None
            ),
        },
        "jobs": runs,
        "files": {
            f: {
                "md5": md5(os.path.join(a.out, f)),
                "bytes": os.path.getsize(os.path.join(a.out, f)),
            }
            for f in files
        },
    }
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)

    print(f"{len(files)} files in {a.out}", flush=True)


if __name__ == "__main__":
    main()
