"""Rebuild the four robustness LaTeX tables from the CSVs experiment 04 wrote.

The run that produced ``results/robustness_*.csv`` died in the table step (a pooled
edge count could not be cast to an integer), so the numbers on disk are correct and
only the rendering needed redoing.  Kept in ``scratch/`` because it is not part of
the pipeline: ``python experiments/04_robustness.py`` writes these tables itself.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str):
    """Import an ``experiments/NN_*.py`` module, registering it before it runs.

    The module goes into :data:`sys.modules` first because it defines a dataclass
    under ``from __future__ import annotations``, and :mod:`dataclasses` resolves
    those string annotations by looking the defining module up there.
    """
    spec = importlib.util.spec_from_file_location(
        f"experiment_{name}", ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    exp04 = _load("04_robustness")
    merged = {arm: pd.read_csv(ROOT / "results" / f"robustness_{arm}.csv")
              for arm in ("scm", "k", "budget", "model", "runs")}
    exp04._write_tables(merged, ("scm", "k", "budget", "model"))
    for problem in exp04._check(merged):
        print(f"finding: {problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
