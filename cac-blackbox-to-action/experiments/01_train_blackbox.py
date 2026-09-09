"""Experiment 01 — train the black boxes and check the acceptance criteria.

    python experiments/01_train_blackbox.py                 # all datasets, xgb + mlp
    python experiments/01_train_blackbox.py -d adult -m xgb
    python experiments/01_train_blackbox.py --force          # rebuild processed data too

Outputs
-------
``models/<type>_<dataset>.joblib``, ``models/metrics.json``,
``results/blackbox_performance.csv`` and ``paper/tables/blackbox_performance.tex``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python experiments/01_...` without installing
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, DATASETS, MODEL_TYPES  # noqa: E402
from src.model.train import METRICS_PATH, train_dataset  # noqa: E402
from src.utils.io import save_dataframe, save_latex_table  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger("experiments.01_train_blackbox")

_LABELS = {"adult": "Adult", "german": "German", "heloc": "HELOC",
           "diabetes": "Diabetes"}


def _summary_frame(reports: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for report in reports.values():
        test = report["test"]
        rows.append({
            "Dataset": _LABELS.get(report["dataset"], report["dataset"]),
            "Model": report["model_type"].upper(),
            "n": test["n"],
            "Accuracy": test["accuracy"],
            "Precision": test["precision"],
            "Recall": test["recall"],
            "F1": test["f1"],
            "AUC": test["roc_auc"],
            "AP": test["average_precision"],
            "Brier": test["brier"],
            "Target AUC": report["target_auc"],
            "Meets target": "yes" if report["meets_target"] else "NO",
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the black-box classifiers.")
    parser.add_argument("-d", "--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("-m", "--models", nargs="+", default=list(MODEL_TYPES),
                        choices=list(MODEL_TYPES))
    parser.add_argument("--force", action="store_true",
                        help="Rebuild the processed datasets from the raw sources.")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip RandomizedSearchCV even if the AUC target is missed.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if a primary (XGB) model misses its AUC target.")
    args = parser.parse_args(argv)

    setup_logging("01_train_blackbox")
    LOGGER.info("Training %s x %s (force=%s, tune=%s)", args.datasets, args.models,
                args.force, not args.no_tune)

    reports: dict[str, dict] = {}
    for dataset in args.datasets:
        for model_type in args.models:
            reports[f"{model_type}_{dataset}"] = train_dataset(
                dataset,
                model_type,
                tune_if_below_target=not args.no_tune,
                force=args.force,
            )

    summary = _summary_frame(reports)
    LOGGER.info("Black-box performance:\n%s", summary.to_string(index=False))
    save_dataframe(summary, CONFIG.RESULTS_DIR / "blackbox_performance.csv")
    save_latex_table(
        summary.drop(columns=["Meets target"]),
        "blackbox_performance",
        caption=("Black-box predictive performance on the held-out 20\\% test split. "
                 "AUC targets follow the pre-registered acceptance criteria "
                 "(0.90 for Adult, 0.80 for the Pima diabetes set, and 0.75 for "
                 "German Credit and HELOC)."),
        label="tab:blackbox_performance",
        column_format="llrrrrrrrrr",
        escape=False,
    )
    LOGGER.info("Metrics written to %s", METRICS_PATH)

    missed = [k for k, r in reports.items()
              if not r["meets_target"] and r["model_type"] == CONFIG.BLACKBOX_TYPE]
    if missed:
        LOGGER.error("Acceptance criteria NOT met for: %s", ", ".join(sorted(missed)))
        if args.strict:
            return 1
    else:
        LOGGER.info("All primary black boxes meet their AUC targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
