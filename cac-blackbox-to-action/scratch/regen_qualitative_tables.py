"""Rebuild experiment 05's two LaTeX tables from the CSVs the last run wrote.

Only the *rendering* changed --- the flagship selection rule and the caption ---
so the re-scored numbers in ``results/qualitative_{cases,frequency}.csv`` stand
and re-running the ~15-minute re-score would waste them.  Kept in ``scratch/``:
``python experiments/05_qualitative.py`` writes these tables itself.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"experiment_{name}", ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    exp05 = _load("05_qualitative")
    merged = {
        "cases": pd.read_csv(ROOT / "results" / "qualitative_cases.csv"),
        "frequency": pd.read_csv(ROOT / "results" / "qualitative_frequency.csv"),
    }
    narratives = (ROOT / "results" / "qualitative_narratives.txt").read_text(
        encoding="utf-8").split("\n")
    exp05._write(merged, narratives)
    for problem in exp05._check(merged["cases"], merged["frequency"]):
        print(f"CHECK: {problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
