"""Smoke-test the notebooks by running their code cells as a script.

No Jupyter kernel is installed in this environment, so the notebooks ship
unexecuted.  This is the substitute: concatenate each notebook's code cells and
``exec`` them top-to-bottom in one namespace, exactly as a kernel would, against
the real pipeline API.  It injects a ``display`` shim (a notebook gets one from
IPython; a plain exec does not) and nothing else, so a NameError here is a real
bug in a cell, not a harness artefact.  Run from the project root so ``import
src`` finds the project package and not the stray site-packages one.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


def _display(*objs, **_kw):
    """Stand in for IPython's display: print enough to prove the value exists."""
    for obj in objs:
        text = obj.to_string() if hasattr(obj, "to_string") else repr(obj)
        print(text if len(text) < 800 else text[:800] + " ...[truncated]")


def run(path: Path) -> bool:
    nb = nbf.read(str(path), as_version=4)
    source = "\n\n".join(c.source for c in nb.cells if c.cell_type == "code")
    ns: dict = {"__name__": "__main__", "display": _display}
    print(f"\n===== {path.name} : {sum(c.cell_type == 'code' for c in nb.cells)} "
          f"code cells =====")
    try:
        exec(compile(source, str(path), "exec"), ns)  # noqa: S102 - intentional
    except Exception:  # noqa: BLE001 - report and keep going to the next notebook
        traceback.print_exc()
        print(f"----- {path.name}: FAILED")
        return False
    print(f"----- {path.name}: ok")
    return True


def main() -> int:
    ok = all(run(p) for p in NOTEBOOKS)
    print("\nALL NOTEBOOKS OK" if ok else "\nSOME NOTEBOOKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
