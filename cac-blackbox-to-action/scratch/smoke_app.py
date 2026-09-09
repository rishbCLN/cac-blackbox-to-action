"""Headless smoke test for app/components.py --- no Streamlit runtime needed.

Streamlit's cache decorators execute the wrapped function directly when there is
no script run (they only warn), so every data/view helper can be exercised as a
plain library call.  This proves the demo's logic works against the real
artifacts before anyone launches ``streamlit run``; the UI wiring in app.py is
thin enough to read by eye.  Run from the project root.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")  # silence the "missing ScriptRunContext" noise

from app import components as C  # noqa: E402


def main() -> int:
    datasets = C.datasets_present()
    print("datasets present:", datasets)
    assert datasets, "no counterfactual joblibs found; run make all"

    headline = C.load_csv("headline")
    assert headline is not None and not headline.empty

    for dataset in datasets:
        frame = C.per_record(dataset)
        assert not frame.empty, f"no per-record rows for {dataset}"
        methods = C.methods_for(dataset, frame)
        records = C.records_for(frame)
        assert "cac" in methods and records
        assert {"valid", "causal", "actionable", "usable"} <= set(frame.columns)

        # the gap chart renders
        fig = C.gap_figure(headline, dataset)
        import matplotlib.pyplot as plt  # noqa: PLC0415
        plt.close(fig)

        # a representative applicant: verdicts, moves, and a live plan figure
        record = records[0]
        row = C.best_row(frame, "cac", record)
        assert row is not None
        _ = C.verdict_markdown(row)
        _ = C.moves_table(row)
        png = C.recourse_png(dataset, record, frame)
        assert png is not None and png.exists(), f"no plan png for {dataset}#{record}"
        print(f"  {dataset}: {len(methods)} methods, {len(records)} applicants, "
              f"plan -> {png.name}")

    print("APP COMPONENTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
