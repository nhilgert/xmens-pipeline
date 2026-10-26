#!/usr/bin/env python
"""run_notebook.py -- dependency-free papermill equivalent (nbformat + nbclient only).
Injects/overrides a 'parameters'-tagged cell, executes top-to-bottom, saves the executed copy.
--prepare-only writes the parameter-injected notebook without executing (for dry verification)."""
import argparse, ast, sys
import nbformat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("-p", nargs=2, action="append", default=[], metavar=("KEY","VALUE"))
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--timeout", type=int, default=21600)
    a = ap.parse_args()
    nb = nbformat.read(a.nb, as_version=4)
    lines = []
    for k, v in a.p:
        try: ast.literal_eval(v); lines.append(f"{k} = {v}")
        except (ValueError, SyntaxError): lines.append(f"{k} = {v!r}")
    idx = [i for i, c in enumerate(nb.cells)
           if c.cell_type == "code" and "parameters" in c.get("metadata", {}).get("tags", [])]
    inj = nbformat.v4.new_code_cell("# injected-parameters\n" + "\n".join(lines))
    inj.metadata["tags"] = ["injected-parameters"]
    nb.cells.insert(idx[0] + 1 if idx else 0, inj)   # after the defaults cell, papermill-style
    for c in nb.cells:
        if c.cell_type == "code": ast.parse(c.source)
    if not a.prepare_only:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=a.timeout, kernel_name="python3").execute()
    nbformat.write(nb, a.out)
    print(("prepared" if a.prepare_only else "executed") + f": {a.out} ({len(a.p)} params)")

if __name__ == "__main__":
    sys.exit(main())
