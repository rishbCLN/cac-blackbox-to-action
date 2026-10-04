"""Experiment 02 — every method on the same records, judged by the same rules.

    python experiments/02_compare_methods.py                  # all datasets, all methods
    python experiments/02_compare_methods.py -d german -n 10
    python experiments/02_compare_methods.py -m cac oracle -k 3

This is §9's main run.  One query set per dataset — the test records the black box
rejects, picked by :func:`~src.evaluation.metrics.query_instances` — every method
asked for the same ``k`` counterfactuals on every one of them, and every number in
the output tables recomputed by :mod:`src.evaluation` rather than reported by the
method that produced the row.

Two decisions worth stating, because they change what the tables mean.  A method
that crashes or gives up on a record is not dropped: it scores zero coverage
there, which is what the person in front of it would experience.  And all methods
share one black-box instance, so the query counts they report are differences
taken inside :meth:`~src.methods.base.CounterfactualMethod.generate`, and the
queries the audit spends afterwards are never charged to them.

Outputs
-------
``results/counterfactuals.csv``    one row per proposal, every metric recomputed
``results/method_comparison.csv``  §9's per-method headline
``results/headline.csv``           the valid -> causal -> actionable funnel
``results/faithfulness*.csv``      does the flip survive the world it implies
``results/feasibility*.csv``       could a person carry the plan out
``results/diversity.csv``          point diversity against distinct routes
``results/cost_ratio.csv``         effort over the oracle's proven optimum
``results/disagreements.csv``      self-reported vs. recomputed (expected empty)
``results/counterfactuals_<dataset>.joblib``  the objects, for 05_qualitative
and ``paper/tables/{main_results,faithfulness,feasibility}.tex``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python experiments/02_...` without installing
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, DATASETS  # noqa: E402
from src.data.preprocess import load_dataset  # noqa: E402
from src.evaluation import faithfulness, feasibility, metrics  # noqa: E402
from src.methods.base import Counterfactual, get_method  # noqa: E402
from src.model.blackbox import load_blackbox  # noqa: E402
from src.utils.io import save_dataframe, save_joblib, save_latex_table  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger("experiments.02_compare_methods")

#: Table order: the literature first, then the two references the paper adds, so
#: a reader meets the baselines before the thing being argued for.
METHOD_ORDER: tuple[str, ...] = ("random", "wachter", "dice", "oracle", "cac")

#: Which result goes in which file.  Named once so the writer, the log and the
#: docstring above cannot drift apart.
OUTPUTS: dict[str, str] = {
    "counterfactuals": "counterfactuals.csv",
    "summary": "method_comparison.csv",
    "headline": "headline.csv",
    "faithfulness_summary": "faithfulness.csv",
    "faithfulness": "faithfulness_per_cf.csv",
    "blamed_features": "blamed_features.csv",
    "feasibility_summary": "feasibility.csv",
    "feasibility": "feasibility_per_cf.csv",
    "blocked_features": "blocked_features.csv",
    "diversity": "diversity.csv",
    "cost_ratio": "cost_ratio.csv",
    "disagreements": "disagreements.csv",
    "runs": "method_runs.csv",
}

_DATASET_LABELS = {"adult": "Adult", "german": "German", "heloc": "HELOC",
                   "diabetes": "Diabetes"}
_METHOD_LABELS = {"random": "Random", "wachter": "Wachter", "dice": "DiCE",
                  "oracle": "Oracle", "cac": "CAC"}

def _run_method(evaluator: metrics.Evaluator, name: str, records: np.ndarray,
                X: np.ndarray, k: int, y_target: int
                ) -> tuple[dict[int, list[Counterfactual]], dict[str, Any]]:
    """Ask one method for ``k`` counterfactuals on every record of the query set.

    Exceptions are caught per record rather than allowed to end the run.  A
    baseline that raises on one applicant — dice-ml does, on rows where its
    sampling finds nothing — has failed to provide recourse for that person, and
    the honest way to record that is an empty result the metrics layer counts as
    zero coverage.  Letting it abort the experiment instead would mean the only
    published numbers are the ones from the runs where nothing went wrong.
    """
    method = get_method(name, evaluator.dataset, model=evaluator.model)
    results: dict[int, list[Counterfactual]] = {}
    failures: list[int] = []
    every = max(1, records.size // 10)
    started = time.perf_counter()
    for i, record in enumerate(records):
        try:
            results[int(record)] = method.generate(X[i], y_target=y_target, k=k)
        except Exception as exc:  # noqa: BLE001 - a crash is a failure, not a gap
            LOGGER.warning("%s/%s record %d: %s: %s", evaluator.dataset, name,
                           int(record), type(exc).__name__, exc)
            results[int(record)] = []
            failures.append(int(record))
        if (i + 1) % every == 0 or i + 1 == records.size:
            elapsed = time.perf_counter() - started
            LOGGER.info("%s/%s: %d/%d records, %.1fs (%.2fs/record)",
                        evaluator.dataset, name, i + 1, records.size, elapsed,
                        elapsed / (i + 1))
    elapsed = time.perf_counter() - started
    produced = sum(len(v) for v in results.values())
    report = {
        "dataset": evaluator.dataset,
        "method": name,
        "n_records": int(records.size),
        "n_answered": int(sum(1 for v in results.values() if v)),
        "n_failed": int(len(failures)),
        "n_cfs": int(produced),
        "wall_s": float(elapsed),
        "s_per_record": float(elapsed / max(records.size, 1)),
    }
    LOGGER.info("%s/%s: %d counterfactuals for %d/%d records in %.1fs%s",
                evaluator.dataset, name, produced, report["n_answered"],
                records.size, elapsed,
                f", {len(failures)} raised" if failures else "")
    return results, report

def _run_dataset(dataset: str, method_names: list[str], n: int | None, k: int,
                 y_target: int) -> dict[str, pd.DataFrame]:
    """Run every method on one dataset and build all of §9's tables for it.

    The :class:`~src.evaluation.metrics.Evaluator` is built once and handed to
    every method as their black box, which is the point: "the model", "the SCM"
    and "the cost model" name the same three objects in every row of every table
    produced here, so a difference between two methods cannot be an artefact of
    two differently-fitted artefacts.
    """
    bundle = load_dataset(dataset)
    model = load_blackbox(dataset, CONFIG.BLACKBOX_TYPE)
    evaluator = metrics.Evaluator.build(dataset, bundle=bundle, model=model)
    records = metrics.query_instances(dataset, model=model, bundle=bundle, n=n,
                                      y_target=y_target)
    if records.size == 0:
        LOGGER.error("%s: the black box rejects nobody in the test split; "
                     "there is nothing to explain", dataset)
        return {}
    X = bundle.X_test[records]

    results: dict[str, dict[int, list[Counterfactual]]] = {}
    runs: list[dict[str, Any]] = []
    tables: list[pd.DataFrame] = []
    faith: list[pd.DataFrame] = []
    feas: list[pd.DataFrame] = []
    div: list[pd.DataFrame] = []
    for name in method_names:
        per_record, report = _run_method(evaluator, name, records, X, k, y_target)
        results[name] = per_record
        runs.append(report)
        # Scored immediately, while the objects are in hand: the audit is the
        # same three calls for every method, in the same order, from the same
        # evaluator.
        tables.append(evaluator.table(per_record, method=name))
        faith.append(faithfulness.assess(evaluator, per_record, method=name))
        feas.append(feasibility.assess(evaluator, per_record, method=name))
        div.append(metrics.diversity(evaluator, per_record, method=name))

    frame = pd.concat(tables, ignore_index=True)
    faith_frame = pd.concat(faith, ignore_index=True)
    feas_frame = pd.concat(feas, ignore_index=True)
    # Two ratios, on purpose.  The headline computes its own over the plans a
    # person could actually take, so its cost column and its ratio column
    # describe the same suggestions.  This one prices every valid plan, which
    # isolates how much of a method's premium buys usability rather than a flip.
    ratios = metrics.cost_ratio(frame)
    # The objects, not just their numbers: 05_qualitative reads plans and
    # explanations back out of here rather than regenerating them, so the case
    # studies in the paper are the same suggestions these tables scored.
    save_joblib(results, CONFIG.RESULTS_DIR / f"counterfactuals_{dataset}.joblib")
    return {
        "counterfactuals": frame,
        "summary": metrics.summarise(frame),
        "headline": feasibility.headline(frame, faith_frame, feas_frame),
        "faithfulness": faith_frame,
        "faithfulness_summary": faithfulness.summarise(faith_frame),
        "blamed_features": faithfulness.blamed_features(faith_frame),
        "feasibility": feas_frame,
        "feasibility_summary": feasibility.summarise(feas_frame),
        "blocked_features": feasibility.blocked_features(feas_frame),
        "diversity": pd.concat(div, ignore_index=True),
        "cost_ratio": ratios,
        "disagreements": metrics.disagreements(frame, results),
        "runs": pd.DataFrame(runs),
    }

def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by dataset, then :data:`METHOD_ORDER`, then record — stable across runs."""
    if frame.empty or "method" not in frame.columns:
        return frame
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    out = frame.copy()
    out["_order"] = out["method"].map(lambda m: rank.get(str(m), len(rank)))
    keys = [c for c in ("dataset", "_order", "record", "rank") if c in out.columns]
    return (out.sort_values(keys, kind="stable")
               .drop(columns="_order").reset_index(drop=True))


def _labelled(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Project and rename for LaTeX, spelling the dataset and method names out."""
    present = {k: v for k, v in columns.items() if k in frame.columns}
    out = frame[list(present)].rename(columns=present)
    if "Dataset" in out.columns:
        out["Dataset"] = out["Dataset"].map(lambda d: _DATASET_LABELS.get(d, d))
    if "Method" in out.columns:
        out["Method"] = out["Method"].map(lambda m: _METHOD_LABELS.get(m, m))
    return out


_MAIN_COLUMNS = {
    "dataset": "Dataset", "method": "Method", "n_records": "n",
    "coverage": "Cov.", "validity": "Valid", "causal_validity": "Causal",
    "actionability": "Action.", "usable": "Usable", "cost_mean": "Cost",
    "cost_ratio_median": "Cost/Oracle", "plausibility_mean": "Plaus.",
    "n_queries_mean": "Queries",
}

_FAITH_COLUMNS = {
    "dataset": "Dataset", "method": "Method", "n_valid": "$n$ valid",
    "causal_validity": "Causal valid", "gap_max_mean": "Gap (mean)",
    "gap_max_p90": "Gap (p90)", "n_off_mean": "Features off",
    "flip_survival": "Flip survives", "promise_kept": "Promise kept",
    "proba_drop_mean": "$\\Delta P$",
}

_FEAS_COLUMNS = {
    "dataset": "Dataset", "method": "Method", "n_valid": "$n$ valid",
    "actionable": "Actionable", "admissible": "Admissible",
    "within_budget": "In budget", "immutable_rate": "Immutable",
    "direction_rate": "Wrong dir.", "over_actions_rate": "Too many",
    "n_actions_mean": "Actions", "ordinary": "Ordinary",
}

def _write_tables(merged: dict[str, pd.DataFrame]) -> None:
    """The three tables §9 prints, from the frames just written to ``results/``."""
    save_latex_table(
        _labelled(merged["headline"], _MAIN_COLUMNS), "main_results",
        caption=("Recourse quality on the shared query set (the test records the "
                 "black box rejects). Rates are fractions of records, not of "
                 "suggestions: a method scores on a record if \\emph{any} of its "
                 "$k$ suggestions passes. \\emph{Causal} additionally requires the "
                 "flip to survive the world the SCM says the plan produces, "
                 "\\emph{Action.} that a person could carry it out under \\S7, and "
                 "\\emph{Usable} both at once. Cost is \\S7's normalised effort for "
                 "the cheapest usable plan, and \\emph{Cost/Oracle} its median ratio "
                 "to the cheapest flip that exists on the same record, so the two "
                 "columns describe the same suggestions."),
        label="tab:main_results", column_format="llrrrrrrrrrr", escape=False)
    save_latex_table(
        _labelled(merged["faithfulness_summary"], _FAITH_COLUMNS), "faithfulness",
        caption=("Causal faithfulness, over the counterfactuals that flip the black "
                 "box. Each proposal is read as the intervention it implies, the "
                 "SCM recomputes the descendants with the factual noise carried "
                 "over, and the gap is the largest normalised disagreement between "
                 "what was proposed and what would result. \\emph{Flip survives} "
                 "re-queries the black box on the repaired record; \\emph{promise "
                 "kept} requires both. $\\Delta P$ is the mean loss in target-class "
                 "probability, so a negative value means the consequences happened "
                 "to help."),
        label="tab:faithfulness", column_format="llrrrrrrrr", escape=False)
    save_latex_table(
        _labelled(merged["feasibility_summary"], _FEAS_COLUMNS), "feasibility",
        caption=("Human feasibility under \\S7, over the counterfactuals that flip "
                 "the black box, evaluated on what each method \\emph{asks for} "
                 "rather than on what the SCM propagates. \\emph{Immutable} and "
                 "\\emph{wrong dir.} are the share of plans touching a feature "
                 "nobody can act on or moving one against its only direction; "
                 "\\emph{too many} the share exceeding the "
                 f"{int(CONFIG.MAX_ACTIONS)}-action budget. \\emph{{Ordinary}} is "
                 "reported separately because an unusual plan is harder to believe "
                 "rather than forbidden."),
        label="tab:feasibility", column_format="llrrrrrrrrr", escape=False)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare every counterfactual method on one shared query set.")
    parser.add_argument("-d", "--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("-m", "--methods", nargs="+", default=list(METHOD_ORDER),
                        choices=list(METHOD_ORDER))
    parser.add_argument("-n", "--n-records", type=int, default=None,
                        help=f"Query-set size per dataset (default "
                             f"{CONFIG.N_QUERY_INSTANCES} from the config).")
    parser.add_argument("-k", type=int, default=int(CONFIG.N_COUNTERFACTUALS),
                        help="Counterfactuals requested per record.")
    parser.add_argument("-t", "--target", type=int, default=1, choices=(0, 1),
                        help="Class to search towards; the query set is the "
                             "records the black box does not already put there.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if a method covers nothing or reports "
                             "a number the audit cannot reproduce.")
    args = parser.parse_args(argv)

    setup_logging("02_compare_methods")
    methods = [m for m in METHOD_ORDER if m in args.methods]
    if "oracle" not in methods:
        LOGGER.warning("oracle excluded: the cost-ratio column has no reference "
                       "and will be empty")
    LOGGER.info("Comparing %s on %s (k=%d, target=%d, n=%s)", methods,
                args.datasets, args.k, args.target,
                args.n_records or CONFIG.N_QUERY_INSTANCES)

    per_dataset: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset in args.datasets:
        out = _run_dataset(dataset, methods, args.n_records, args.k, args.target)
        if out:
            per_dataset[dataset] = out
    if not per_dataset:
        LOGGER.error("No dataset produced any result.")
        return 1

    merged = {key: _ordered(pd.concat([out[key] for out in per_dataset.values()],
                                      ignore_index=True))
              for key in OUTPUTS}
    for key, filename in OUTPUTS.items():
        save_dataframe(merged[key], CONFIG.RESULTS_DIR / filename)
    _write_tables(merged)

    for key in ("headline", "faithfulness_summary", "feasibility_summary",
                "cost_ratio", "runs"):
        LOGGER.info("%s:\n%s", key.replace("_", " "),
                    merged[key].to_string(index=False))

    problems: list[str] = []
    silent = merged["runs"][merged["runs"]["n_answered"] == 0]
    for row in silent.to_dict("records"):
        problems.append(f"{row['dataset']}/{row['method']} answered no record")
    if not merged["disagreements"].empty:
        LOGGER.error("Methods reported %d numbers the audit does not reproduce:\n%s",
                     merged["disagreements"].shape[0],
                     merged["disagreements"].to_string(index=False))
        problems.append(f"{merged['disagreements'].shape[0]} self-report mismatches")
    else:
        LOGGER.info("Every method's self-reported cost and causal gap match the "
                    "audit's recomputation.")

    LOGGER.info("Wrote %d tables to %s and 3 LaTeX tables to %s",
                len(OUTPUTS), CONFIG.RESULTS_DIR, CONFIG.TABLES_DIR)
    if problems:
        LOGGER.error("Problems: %s", "; ".join(problems))
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
