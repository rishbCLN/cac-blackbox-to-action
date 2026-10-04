"""Experiment 03 — which part of CAC's objective is doing the work.

    python experiments/03_ablation.py                       # all datasets, all variants
    python experiments/03_ablation.py -d german -n 20
    python experiments/03_ablation.py -v cac cac_no_causal

The paper claims CAC *jointly* optimises causal validity, cost and human
feasibility.  A joint claim is only worth making if each part is load-bearing, so
this run takes the full objective apart one term at a time and re-scores what is
left through the same audit §9 uses on the baselines.

Six variants, one per row of ``paper/tables/ablation.tex``:

``cac``             the paper's method, all four weights at their §3 defaults
``cac_no_causal``   the SCM removed from the *representation* — same search, same
                    grids, same cost model, but a candidate is scored as a point
                    the way every baseline scores one
``cac_no_valid``    ``CAC_LAMBDA_VALID = 0``: nothing pushes towards the target
``cac_no_cost``     ``CAC_LAMBDA_COST = 0``: any flip will do, at any effort
``cac_no_feas``     ``CAC_LAMBDA_FEAS = 0``: side effects on sensitive features
                    stop being counted against a plan
``cac_no_plaus``    ``CAC_LAMBDA_PLAUS = 0``: the search may leave the data
``cac_valid_only``  all three of the above at once — what CAC degrades to when
                    only flipping the model counts, i.e. Wachter's objective
                    searched in intervention space

``cac_no_causal`` is the one that cannot be written as a weight, and it is the
one the paper's central claim rests on: if causal validity were a penalty term,
zeroing a weight would be enough to remove it.  It is not — it is a property of
what a candidate *is* — so removing it means scoring the same population
differently.  Everything else about the two runs is identical, which is what
makes the gap between their ``causal_gap`` columns attributable.

The reference for the cost ratio here is ``cac`` itself, not the oracle: the
question this experiment asks is what each term buys or costs *relative to the
full objective*, and using the oracle would import experiment 02's reference into
a table that does not run it.

Outputs
-------
``results/ablation.csv``               one row per proposal, every metric recomputed
``results/ablation_headline.csv``      the funnel, one row per variant
``results/ablation_faithfulness.csv``  gap, flip survival, promise kept
``results/ablation_feasibility.csv``   admissibility and budget, per variant
``results/ablation_runs.csv``          wall clock and failures
and ``paper/tables/ablation.tex``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python experiments/03_...` without installing
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, DATASETS  # noqa: E402
from src.data.preprocess import load_dataset  # noqa: E402
from src.evaluation import faithfulness, feasibility, metrics  # noqa: E402
from src.methods.base import Counterfactual  # noqa: E402
from src.methods.cac import CACMethod  # noqa: E402
from src.model.blackbox import load_blackbox  # noqa: E402
from src.utils.io import save_dataframe, save_latex_table  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger("experiments.03_ablation")


@dataclass(frozen=True)
class Variant:
    """One row of the ablation: a label, a caption, and what was taken away."""

    name: str
    caption: str
    lambdas: dict[str, float] = field(default_factory=dict)
    propagate: bool = True

    def build(self, dataset: str, model: Any) -> CACMethod:
        """A CAC instance that reports rows under :attr:`name`.

        The weights go through the constructor rather than through ``CONFIG``,
        which is frozen and process-global: an ablation that mutated it would
        change the objective for every other variant in the same run, and the
        comparison would quietly stop being one.
        """
        return CACMethod(dataset, model=model, label=self.name,
                         lambdas=self.lambdas, propagate=self.propagate)


VARIANTS: tuple[Variant, ...] = (
    Variant("cac", "Full objective"),
    Variant("cac_no_causal", "No SCM in the loop (scored as a point)",
            propagate=False),
    Variant("cac_no_valid", "$\\lambda_{\\mathrm{valid}} = 0$",
            lambdas={"valid": 0.0}),
    Variant("cac_no_cost", "$\\lambda_{\\mathrm{cost}} = 0$",
            lambdas={"cost": 0.0}),
    Variant("cac_no_feas", "$\\lambda_{\\mathrm{feas}} = 0$",
            lambdas={"feas": 0.0}),
    Variant("cac_no_plaus", "$\\lambda_{\\mathrm{plaus}} = 0$",
            lambdas={"plaus": 0.0}),
    Variant("cac_valid_only", "Validity only (cost, feasibility, plausibility off)",
            lambdas={"cost": 0.0, "feas": 0.0, "plaus": 0.0}),
)

#: The variant every other one is read against — the full objective.
REFERENCE = "cac"

OUTPUTS: dict[str, str] = {
    "counterfactuals": "ablation.csv",
    "headline": "ablation_headline.csv",
    "faithfulness": "ablation_faithfulness.csv",
    "feasibility": "ablation_feasibility.csv",
    "runs": "ablation_runs.csv",
}

_DATASET_LABELS = {"adult": "Adult", "german": "German", "heloc": "HELOC",
                   "diabetes": "Diabetes"}
_VARIANT_LABELS = {
    "cac": "CAC (full)",
    "cac_no_causal": "-- SCM",
    "cac_no_valid": "-- validity",
    "cac_no_cost": "-- cost",
    "cac_no_feas": "-- feasibility",
    "cac_no_plaus": "-- plausibility",
    "cac_valid_only": "validity only",
}

def _run_variant(evaluator: metrics.Evaluator, variant: Variant,
                 records: np.ndarray, X: np.ndarray, k: int, y_target: int
                 ) -> tuple[dict[int, list[Counterfactual]], dict[str, Any]]:
    """Ask one variant for ``k`` plans on every record of the shared query set.

    Same contract as experiment 02: a record that raises scores zero coverage
    rather than dropping out, because a variant that crashes on an applicant has
    failed that applicant.  ``cac_no_valid`` is the one where this matters — with
    nothing pushing towards the target it can legitimately return nothing.
    """
    method = variant.build(evaluator.dataset, evaluator.model)
    results: dict[int, list[Counterfactual]] = {}
    failures: list[int] = []
    every = max(1, records.size // 5)
    started = time.perf_counter()
    for i, record in enumerate(records):
        try:
            results[int(record)] = method.generate(X[i], y_target=y_target, k=k)
        except Exception as exc:  # noqa: BLE001 - a crash is a failure, not a gap
            LOGGER.warning("%s/%s record %d: %s: %s", evaluator.dataset,
                           variant.name, int(record), type(exc).__name__, exc)
            results[int(record)] = []
            failures.append(int(record))
        if (i + 1) % every == 0 or i + 1 == records.size:
            elapsed = time.perf_counter() - started
            LOGGER.info("%s/%s: %d/%d records, %.1fs (%.2fs/record)",
                        evaluator.dataset, variant.name, i + 1, records.size,
                        elapsed, elapsed / (i + 1))
    elapsed = time.perf_counter() - started
    report = {
        "dataset": evaluator.dataset,
        "variant": variant.name,
        "lambdas": ",".join(f"{key}={value:g}"
                            for key, value in sorted(variant.lambdas.items())) or "-",
        "propagate": bool(variant.propagate),
        "n_records": int(records.size),
        "n_answered": int(sum(1 for v in results.values() if v)),
        "n_failed": int(len(failures)),
        "n_cfs": int(sum(len(v) for v in results.values())),
        "wall_s": float(elapsed),
        "s_per_record": float(elapsed / max(records.size, 1)),
    }
    LOGGER.info("%s/%s: %d plans for %d/%d records in %.1fs%s", evaluator.dataset,
                variant.name, report["n_cfs"], report["n_answered"], records.size,
                elapsed, f", {len(failures)} raised" if failures else "")
    return results, report


def _run_dataset(dataset: str, variants: list[Variant], n: int | None, k: int,
                 y_target: int) -> dict[str, pd.DataFrame]:
    """Every variant on one dataset, scored by experiment 02's three audits.

    The query set is built the same way and from the same config knob, so a row
    here and a row of ``results/headline.csv`` are averages over the same people.
    """
    bundle = load_dataset(dataset)
    model = load_blackbox(dataset, CONFIG.BLACKBOX_TYPE)
    evaluator = metrics.Evaluator.build(dataset, bundle=bundle, model=model)
    records = metrics.query_instances(dataset, model=model, bundle=bundle, n=n,
                                     y_target=y_target)
    if records.size == 0:
        LOGGER.error("%s: the black box rejects nobody in the test split", dataset)
        return {}
    X = bundle.X_test[records]

    runs: list[dict[str, Any]] = []
    tables: list[pd.DataFrame] = []
    faith: list[pd.DataFrame] = []
    feas: list[pd.DataFrame] = []
    for variant in variants:
        per_record, report = _run_variant(evaluator, variant, records, X, k,
                                          y_target)
        runs.append(report)
        tables.append(evaluator.table(per_record, method=variant.name))
        faith.append(faithfulness.assess(evaluator, per_record, method=variant.name))
        feas.append(feasibility.assess(evaluator, per_record, method=variant.name))

    frame = pd.concat(tables, ignore_index=True)
    faith_frame = pd.concat(faith, ignore_index=True)
    feas_frame = pd.concat(feas, ignore_index=True)
    head = feasibility.headline(frame, faith_frame, feas_frame, reference=REFERENCE)
    # The gap and the flip survival belong in the ablation table itself: they are
    # what `cac_no_causal` is meant to move, and a reader should not have to join
    # two files to see whether it did.
    summary = faithfulness.summarise(faith_frame)
    head = head.merge(summary[["dataset", "method", "gap_max_mean", "gap_max_p90",
                               "flip_survival"]],
                      on=["dataset", "method"], how="left")
    return {
        "counterfactuals": frame,
        "headline": head,
        "faithfulness": summary,
        "feasibility": feasibility.summarise(feas_frame),
        "runs": pd.DataFrame(runs),
    }


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by dataset then by :data:`VARIANTS` order — stable across runs."""
    if frame.empty:
        return frame
    column = "method" if "method" in frame.columns else "variant"
    if column not in frame.columns:
        return frame
    rank = {v.name: i for i, v in enumerate(VARIANTS)}
    out = frame.copy()
    out["_order"] = out[column].map(lambda m: rank.get(str(m), len(rank)))
    keys = [c for c in ("dataset", "_order", "record", "rank") if c in out.columns]
    return (out.sort_values(keys, kind="stable")
               .drop(columns="_order").reset_index(drop=True))


_ABLATION_COLUMNS = {
    "dataset": "Dataset", "method": "Variant", "coverage": "Cov.",
    "validity": "Valid", "causal_validity": "Causal", "actionability": "Action.",
    "usable": "Usable", "gap_max_mean": "Gap", "flip_survival": "Flip survives",
    "cost_mean": "Cost", "cost_ratio_median": "Cost/CAC",
    "plausibility_mean": "Plaus.", "n_queries_mean": "Queries",
}


def _labelled(frame: pd.DataFrame) -> pd.DataFrame:
    """Project and rename for LaTeX, spelling out datasets and variants."""
    present = {k: v for k, v in _ABLATION_COLUMNS.items() if k in frame.columns}
    out = frame[list(present)].rename(columns=present)
    out["Dataset"] = out["Dataset"].map(lambda d: _DATASET_LABELS.get(d, d))
    out["Variant"] = out["Variant"].map(lambda m: _VARIANT_LABELS.get(m, m))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Take CAC's objective apart one term at a time.")
    parser.add_argument("-d", "--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("-v", "--variants", nargs="+",
                        default=[v.name for v in VARIANTS],
                        choices=[v.name for v in VARIANTS])
    parser.add_argument("-n", "--n-records", type=int, default=None,
                        help=f"Query-set size per dataset (default "
                             f"{CONFIG.N_QUERY_INSTANCES} from the config).")
    parser.add_argument("-k", type=int, default=int(CONFIG.N_COUNTERFACTUALS),
                        help="Counterfactuals requested per record.")
    parser.add_argument("-t", "--target", type=int, default=1, choices=(0, 1))
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if a variant answers no record, or if "
                             "the full objective is not the causally faithful one.")
    args = parser.parse_args(argv)

    setup_logging("03_ablation")
    variants = [v for v in VARIANTS if v.name in args.variants]
    if REFERENCE not in {v.name for v in variants}:
        LOGGER.warning("%s excluded: the cost-ratio column has no reference and "
                       "will be empty", REFERENCE)
    LOGGER.info("Ablating %s on %s (k=%d, target=%d, n=%s)",
                [v.name for v in variants], args.datasets, args.k, args.target,
                args.n_records or CONFIG.N_QUERY_INSTANCES)

    per_dataset: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset in args.datasets:
        out = _run_dataset(dataset, variants, args.n_records, args.k, args.target)
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
    save_latex_table(
        _labelled(merged["headline"]), "ablation",
        caption=("Objective ablation on the same query set as "
                 "Table~\\ref{tab:main_results}. Each row removes one term and "
                 "re-scores what is left through the same audit. "
                 "\\emph{-- SCM} is not a weight: it scores the identical "
                 "population as a point rather than as an intervention, which is "
                 "the only way to remove causal validity from a method that "
                 "represents it structurally rather than as a penalty. "
                 "\\emph{Gap} is the mean largest normalised disagreement between "
                 "what a plan proposes and what the SCM says would result, and "
                 "\\emph{Cost/CAC} the median effort ratio against the full "
                 f"objective (tolerance {CONFIG.CAUSAL_TOLERANCE:g})."),
        label="tab:ablation", column_format="ll" + "r" * 11, escape=False)

    for key in ("headline", "faithfulness", "runs"):
        LOGGER.info("%s:\n%s", key, merged[key].to_string(index=False))

    problems: list[str] = []
    silent = merged["runs"][merged["runs"]["n_answered"] == 0]
    for row in silent.to_dict("records"):
        problems.append(f"{row['dataset']}/{row['variant']} answered no record")
    # The claim this experiment exists to check: the full objective is the one
    # whose plans the SCM signs off on.  If a variant beats it on causal
    # faithfulness, the paper's argument does not hold and saying so here is
    # cheaper than finding out in review.
    head = merged["headline"]
    if REFERENCE in set(head["method"]):
        for dataset, group in head.groupby("dataset"):
            ref = group[group["method"] == REFERENCE]["causal_validity"].iloc[0]
            better = group[(group["method"] != REFERENCE)
                           & (group["causal_validity"] > float(ref) + 1e-9)]
            for row in better.to_dict("records"):
                problems.append(f"{dataset}: {row['method']} is more causally valid "
                                f"than {REFERENCE} ({row['causal_validity']:.3f} > "
                                f"{float(ref):.3f})")

    LOGGER.info("Wrote %d tables to %s and 1 LaTeX table to %s", len(OUTPUTS),
                CONFIG.RESULTS_DIR, CONFIG.TABLES_DIR)
    if problems:
        LOGGER.error("Problems: %s", "; ".join(problems))
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
