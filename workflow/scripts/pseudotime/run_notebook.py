"""Execute a parameterised notebook top to bottom, in the manner of papermill.

Called by rule `pt_notebook` (24_pseudotime_notebooks.smk). Reads the notebook given
by --nb and the KEY VALUE pairs given by -p, injects them as a code cell after the
cell tagged 'parameters', runs every cell with nbclient, and writes the executed
copy to --out. --prepare-only writes the injected notebook without running it.
"""
import argparse, ast, sys
import nbformat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-p", nargs=2, action="append", default=[], metavar=("KEY","VALUE"))
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--timeout", type=int, default=21600)  # seconds per cell (6 h)
    a = ap.parse_args()

    nb = nbformat.read(a.nb, as_version=4)

    # A value that parses as a Python literal is injected verbatim (numbers, lists);
    # anything else is quoted so that paths become string assignments.
    lines = []
    for k, v in a.p:
        try:
            ast.literal_eval(v)
            lines.append(f"{k} = {v}")
        except (ValueError, SyntaxError):
            lines.append(f"{k} = {v!r}")

    # The injected cell goes right after the 'parameters' cell so that its
    # assignments override the notebook defaults, as papermill does.
    idx = [i for i, c in enumerate(nb.cells)
           if c.cell_type == "code" and "parameters" in c.get("metadata", {}).get("tags", [])]
    inj = nbformat.v4.new_code_cell("# injected-parameters\n" + "\n".join(lines))
    inj.metadata["tags"] = ["injected-parameters"]
    nb.cells.insert(idx[0] + 1 if idx else 0, inj)

    # Syntax-check every cell before spending GPU time on execution.
    for c in nb.cells:
        if c.cell_type == "code":
            ast.parse(c.source)

    if not a.prepare_only:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=a.timeout, kernel_name="python3").execute()

    nbformat.write(nb, a.out)
    print(("prepared" if a.prepare_only else "executed") + f": {a.out} ({len(a.p)} params)")

if __name__ == "__main__":
    sys.exit(main())
