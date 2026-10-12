"""Experiment 08 — known-ground-truth audit: is causal validity real, or an echo?

    python experiments/08_known_scm.py                 # both worlds, all methods
    python experiments/08_known_scm.py -d synth_block -n 20
    python experiments/08_known_scm.py -m cac mint oracle

Experiment 07 breaks the SCM circularity on *real* data by re-fitting the auditor
FCM on a disjoint fold — an independent estimate, but still an estimate.  Here the
data-generating equations are known exactly (``src/data/load_synth_reach`` and
``load_synth_block``), so the auditor need not be estimated at all:
:class:`~src.scm.synthetic_truth.TrueSCM` propagates a method's declared actions
through the *real* structural laws (§6's three steps, closed form).  A method still
*generates* under its own fitted FCM — the circular reading it optimised against —
and the same plans are then scored twice: under that ``own`` FCM and under the
``truth``.  The black box is identical throughout, so any gap is a fact about
causal validity, not about the decision function.

Two worlds, one honest contrast:

``synth_reach``   age has no edge into approval; every rejected applicant has a
                  reachable, budget-feasible flip.  A faithful method should keep a
                  high ``usable`` rate under the truth.

``synth_block``   age enters approval directly and is immutable, so a known share
                  of young rejected applicants have *honest non-recourse*: no
                  admissible income/savings move can lift the true score over the
                  threshold.  A method that still claims to flip them is caught by
                  the truth auditor — ``usable`` under truth falls below one, and
                  that shortfall is the honest failure the paper wants to measure
                  rather than paper over with hallucinated recourse.

Outputs
-------
``results/known_scm.csv``        one row per (dataset, method, auditor)
``results/known_scm_runs.csv``   generation wall clock and failures
``paper/tables/known_scm.tex``   own vs. truth, side by side, both worlds
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python experiments/08_...` without installing
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.cost.cost_functions import get_cost_model  # noqa: E402
from src.data import load_synth_block, load_synth_reach  # noqa: E402
from src.data.preprocess import DatasetBundle, load_dataset  # noqa: E402
from src.evaluation import faithfulness, feasibility, metrics  # noqa: E402
from src.methods.base import Counterfactual, get_method  # noqa: E402
from src.methods.cac import CACMethod  # noqa: E402
from src.methods.mint import MINTMethod  # noqa: E402
from src.model.blackbox import load_blackbox, model_path  # noqa: E402
from src.model.train import train_dataset  # noqa: E402
from src.scm.define_scm import get_scm  # noqa: E402
from src.scm.fcm import FCM, get_fcm  # noqa: E402
from src.scm.synthetic_truth import true_scm  # noqa: E402
from src.utils.io import save_dataframe, save_latex_table  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger("experiments.08_known_scm")

#: The two known-SCM worlds.  Registered as first-class datasets but kept out of
#: ``CONFIG.DATASETS`` so the §9 headline stays the four real datasets; this
#: experiment is the only one that exercises them.
SYNTH_WORLDS: tuple[str, ...] = ("synth_reach", "synth_block")

#: The world modules, for the closed-form intrinsic-reachability diagnostic.
_WORLD = {"synth_reach": load_synth_reach, "synth_block": load_synth_block}

#: Reading order inside a table: acausal baselines first, then the causal pair.
METHOD_ORDER: tuple[str, ...] = ("random", "wachter", "dice", "oracle", "mint", "cac")

#: The methods whose generation reads an SCM, so the own->truth drop is about them;
#: a baseline never consults an FCM, but its declared action is still audited through
#: the truth (an acausal flip should fail a real causal law, and here we can prove it).
CAUSAL_METHODS: tuple[str, ...] = ("cac", "mint")
REFERENCE = "cac"

#: Quantities the circularity shows up in, pulled from the two audit summaries.
_METRIC_COLUMNS: tuple[str, ...] = ("causal_validity", "flip_survival", "usable",
                                    "gap_max_mean", "cost_mean")

OUTPUTS: dict[str, str] = {
    "audit": "known_scm.csv",
    "runs": "known_scm_runs.csv",
}

_DATASET_LABELS = {"synth_reach": "Synth-Reach", "synth_block": "Synth-Block"}
_METHOD_LABELS = {"random": "Random", "wachter": "Wachter", "dice": "DiCE",
                  "oracle": "Oracle", "mint": "MINT", "cac": "CAC"}


# ------------------------------------------------------------------- provisioning

def _ensure_blackbox(dataset: str) -> Any:
    """Load the world's black box, training it on first use.

    The synthetic worlds are not part of ``make all``'s model stage, so their box
    may not exist yet.  Training is additive — it writes only
    ``models/xgb_<world>.joblib`` and one key in ``models/metrics.json`` — and the
    known rule is near-separable, so the default XGB clears the 0.75 AUC floor
    without any search.
    """
    if not model_path(dataset, CONFIG.BLACKBOX_TYPE).exists():
        LOGGER.info("%s: no %s black box yet — training one (additive)", dataset,
                    CONFIG.BLACKBOX_TYPE)
        report = train_dataset(dataset, CONFIG.BLACKBOX_TYPE)
        LOGGER.info("%s: trained, test AUC=%.3f acc=%.3f (target %.2f, %s)", dataset,
                    report["test"]["roc_auc"], report["test"]["accuracy"],
                    report["target_auc"], "PASS" if report["meets_target"] else "BELOW")
    return load_blackbox(dataset, CONFIG.BLACKBOX_TYPE)


def _build_method(name: str, dataset: str, model: Any, scm: Any, own_fcm: FCM) -> Any:
    """Construct one method, injecting the own fitted FCM into the causal pair.

    CAC and MINT generate *under* ``own_fcm`` (the circular reading the truth
    auditor is about to break); the baselines take only the black box, exactly as
    in experiment 02's :func:`get_method` path.
    """
    if name == "cac":
        return CACMethod(dataset, model=model, scm=scm, fcm=own_fcm)
    if name == "mint":
        return MINTMethod(dataset, model=model, fcm=own_fcm)
    return get_method(name, dataset, model=model)


# --------------------------------------------------------------------- the audit

def _funnel(evaluator: metrics.Evaluator,
            results: dict[str, dict[int, list[Counterfactual]]]) -> pd.DataFrame:
    """Put each method's plans through §9's three audits, under *this* evaluator.

    The same calls experiments 02/07 make, so a row means what a headline row
    means; the only thing that differs is which FCM the evaluator carries, which
    is the whole experiment.  ``flip_survival`` and the gap come from the
    faithfulness summary; ``usable`` and ``causal_validity`` from the funnel.
    """
    tables, faith, feas = [], [], []
    for method, per_record in results.items():
        tables.append(evaluator.table(per_record, method=method))
        faith.append(faithfulness.assess(evaluator, per_record, method=method))
        feas.append(feasibility.assess(evaluator, per_record, method=method))
    frame = pd.concat(tables, ignore_index=True)
    faith_all = pd.concat(faith, ignore_index=True)
    feas_all = pd.concat(feas, ignore_index=True)
    if frame.empty or "method" not in frame.columns:
        return pd.DataFrame()
    reference = REFERENCE if REFERENCE in results else next(iter(results))
    head = feasibility.headline(frame, faith_all, feas_all, reference=reference)
    summary = faithfulness.summarise(faith_all)
    return head.merge(summary[["dataset", "method", "flip_survival", "gap_max_mean"]],
                      on=["dataset", "method"], how="left")


def _tidy(head: pd.DataFrame, auditor: str) -> pd.DataFrame:
    """The funnel reduced to the columns this experiment reports, labelled by auditor."""
    if head.empty:
        return head
    keep = ["dataset", "method", "n_records", *_METRIC_COLUMNS]
    out = head.reindex(columns=keep).copy()
    out.insert(2, "auditor", auditor)
    return out


def _generate(method: Any, records: np.ndarray, X: np.ndarray, k: int,
              y_target: int) -> tuple[dict[int, list[Counterfactual]], dict[str, Any]]:
    """One method over the shared query set — experiment 02/07's contract.

    A record that raises scores zero coverage rather than dropping out, so a
    method that cannot generate for an applicant is counted as having failed that
    applicant rather than quietly shrinking the denominator.
    """
    results: dict[int, list[Counterfactual]] = {}
    failures: list[int] = []
    started = time.perf_counter()
    for i, record in enumerate(records):
        try:
            results[int(record)] = method.generate(X[i], y_target=y_target, k=k)
        except Exception as exc:  # noqa: BLE001 - a crash is a failure, not a gap
            LOGGER.warning("%s record %d: %s: %s", method.name, int(record),
                           type(exc).__name__, exc)
            results[int(record)] = []
            failures.append(int(record))
    elapsed = time.perf_counter() - started
    report = {
        "dataset": method.dataset,
        "method": method.name,
        "n_records": int(records.size),
        "n_answered": int(sum(1 for v in results.values() if v)),
        "n_failed": int(len(failures)),
        "n_cfs": int(sum(len(v) for v in results.values())),
        "wall_s": float(elapsed),
    }
    LOGGER.info("%s/%s: %d plans for %d/%d records in %.1fs%s", method.dataset,
                method.name, report["n_cfs"], report["n_answered"], records.size,
                elapsed, f", {len(failures)} raised" if failures else "")
    return results, report


def _intrinsic_nonrecourse(dataset: str, bundle: DatasetBundle,
                           records: np.ndarray) -> np.ndarray:
    """Which query applicants are structurally unreachable under the *known* rule.

    For a rule monotone increasing in the two direct levers, the best any
    intervention can do is push income and savings to the top of their fitted
    support (experience/training only feed income, already maxed).  So an applicant
    is structurally blocked iff the known score at that ceiling is still below
    ``TAU`` — ignoring the action budget, hence a *lower* bound on true
    non-recourse.  ``synth_reach`` has no age term in approval, so this is empty by
    construction; ``synth_block`` is where the honest failures live.
    """
    mod = _WORLD[dataset]
    names = list(bundle.feature_names)
    bounds = bundle.transformer.bounds
    inc_hi = float(bounds[names.index("income"), 1])
    sav_hi = float(bounds[names.index("savings"), 1])
    age = bundle.X_test[records][:, names.index("age")]
    if dataset == "synth_block":
        ceiling = mod.approval_score(age, inc_hi, sav_hi)  # noise=0: deterministic ceiling
    else:
        ceiling = np.full(records.shape, float(mod.approval_score(inc_hi, sav_hi)))
    return np.asarray(ceiling < mod.TAU)


def _run_dataset(dataset: str, method_names: list[str], n: int | None, k: int,
                 y_target: int, runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Generate every method under the own FCM, then audit under own and truth."""
    bundle = load_dataset(dataset)
    model = _ensure_blackbox(dataset)
    scm = get_scm(dataset)
    own_fcm = get_fcm(dataset)        # the method-facing fitted equations (circular auditor)
    truth = true_scm(dataset)         # the analytic data-generating SCM (honest auditor)

    records = metrics.query_instances(dataset, model=model, bundle=bundle, n=n,
                                      y_target=y_target)
    if records.size == 0:
        LOGGER.error("%s: the black box rejects nobody in the test split", dataset)
        return pd.DataFrame()
    X = bundle.X_test[records]

    blocked = _intrinsic_nonrecourse(dataset, bundle, records)
    LOGGER.info("%s: %d query applicants, %d structurally unreachable under the known "
                "rule (%.1f%%)", dataset, records.size, int(blocked.sum()),
                100.0 * float(blocked.mean()))

    plans: dict[str, dict[int, list[Counterfactual]]] = {}
    for name in method_names:
        method = _build_method(name, dataset, model, scm, own_fcm)
        per_record, report = _generate(method, records, X, k, y_target)
        runs.append({**report, "n_blocked": int(blocked.sum())})
        plans[name] = per_record

    cost_model = get_cost_model(dataset)
    own = metrics.Evaluator(dataset=dataset, bundle=bundle, model=model, fcm=own_fcm,
                            cost_model=cost_model)
    truth_eval = metrics.Evaluator(dataset=dataset, bundle=bundle, model=model,
                                   fcm=truth, cost_model=cost_model)
    return pd.concat([_tidy(_funnel(own, plans), "own"),
                      _tidy(_funnel(truth_eval, plans), "truth")], ignore_index=True)


# ------------------------------------------------------------------ own vs truth

def _contrast(audit: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, method): the four quantities, own beside truth."""
    rows: list[dict[str, Any]] = []
    for (dataset, method), group in audit.groupby(["dataset", "method"], sort=False):
        own = group[group["auditor"] == "own"]
        tru = group[group["auditor"] == "truth"]

        def pick(frame: pd.DataFrame, column: str) -> float:
            if len(frame) and pd.notna(frame[column].iloc[0]):
                return float(frame[column].iloc[0])
            return float("nan")

        present = tru if tru.shape[0] else own
        rows.append({
            "dataset": dataset, "method": method,
            "n_records": int(pick(present, "n_records"))
            if np.isfinite(pick(present, "n_records")) else 0,
            "causal_own": pick(own, "causal_validity"),
            "causal_truth": pick(tru, "causal_validity"),
            "usable_own": pick(own, "usable"),
            "usable_truth": pick(tru, "usable"),
            "gap_own": pick(own, "gap_max_mean"),
            "gap_truth": pick(tru, "gap_max_mean"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    data_rank = {d: i for i, d in enumerate(SYNTH_WORLDS)}
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    out["_d"] = out["dataset"].map(lambda d: data_rank.get(d, len(SYNTH_WORLDS)))
    out["_m"] = out["method"].map(lambda m: method_rank.get(m, len(METHOD_ORDER)))
    return (out.sort_values(["_d", "_m"], kind="stable")
               .drop(columns=["_d", "_m"]).reset_index(drop=True))


def _fmt(value: Any, digits: int = 2) -> str:
    """A number, or ``--`` — a table that prints ``nan`` reads as a bug."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(number) else f"{number:.{digits}f}"


def _arrow(own: Any, other: Any, digits: int = 2) -> str:
    """``own -> truth`` in one cell, so the drop is read without hunting two columns."""
    return f"${_fmt(own, digits)} \\rightarrow {_fmt(other, digits)}$"


def _contrast_table(contrast: pd.DataFrame) -> pd.DataFrame:
    """The LaTeX table: causal validity, usable, and gap, each own beside truth."""
    rows: list[dict[str, Any]] = []
    for _, row in contrast.iterrows():
        rows.append({
            "World": _DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"])),
            "Method": _METHOD_LABELS.get(str(row["method"]), str(row["method"])),
            "N": int(row["n_records"]),
            "Causal (own $\\rightarrow$ true)": _arrow(row["causal_own"], row["causal_truth"]),
            "Usable (own $\\rightarrow$ true)": _arrow(row["usable_own"], row["usable_truth"]),
            "Gap (own $\\rightarrow$ true)": _arrow(row["gap_own"], row["gap_truth"], 3),
        })
    return pd.DataFrame(rows)


_CAPTION = (
    "Known-ground-truth causal audit. Two synthetic worlds are generated from an "
    "explicit SCM, so a method's declared actions can be scored through the "
    "\\emph{true} structural equations (\\S6, closed form) rather than through an "
    "FCM the method also optimised against. Each method generates under its own "
    "fitted FCM (\\emph{own}); the same plans are then re-scored under the truth "
    "(\\emph{true}). The black box is identical throughout, so a drop is a fact "
    "about causal validity, not the decision function. In \\textsc{Synth-Reach} "
    "every rejected applicant has a reachable flip; in \\textsc{Synth-Block} the "
    "immutable age enters approval directly, so a known share of applicants have "
    "\\emph{honest non-recourse} and \\emph{usable} under the truth must fall "
    "below one for any method that respects the real causal laws."
)


def _check(contrast: pd.DataFrame, blocked_by_world: dict[str, int]) -> list[str]:
    """The two claims this experiment exists to test, stated as ways they can fail.

    (1) De-circularisation is visible: for a causal method, moving from its own FCM
    to the truth should either drop causal validity or — since record-level causal
    validity is a saturating ``any``-flip-survives rate — widen the mean gap. Only
    when *neither* moves is there a problem (the truth found no echo at all).

    (2) Honest non-recourse surfaces: ``synth_block`` is engineered to contain
    structurally unreachable applicants, so no method should post a perfect
    ``usable`` rate under the truth there. A perfect score means recourse was
    hallucinated for an applicant the known rule cannot satisfy.
    """
    problems: list[str] = []
    if contrast.empty:
        return ["no (dataset, method) produced a result"]

    for _, row in contrast.iterrows():
        if str(row["method"]) not in CAUSAL_METHODS:
            continue
        own, tru = row["causal_own"], row["causal_truth"]
        gap_own, gap_tru = row["gap_own"], row["gap_truth"]
        causal_known = bool(np.isfinite(own) and np.isfinite(tru))
        gap_known = bool(np.isfinite(gap_own) and np.isfinite(gap_tru))
        if not causal_known and not gap_known:
            continue
        causal_dropped = causal_known and tru <= own - 1e-9
        gap_widened = gap_known and gap_tru >= gap_own + 1e-9
        if not causal_dropped and not gap_widened:
            problems.append(
                f"{row['dataset']}/{row['method']}: the true SCM moved neither causal "
                f"validity ({_fmt(own, 3)} -> {_fmt(tru, 3)}) nor the mean gap "
                f"({_fmt(gap_own, 3)} -> {_fmt(gap_tru, 3)}); no circularity surfaced")

    if "synth_block" in contrast["dataset"].values:
        block = contrast[contrast["dataset"] == "synth_block"]
        usable_truth = block["usable_truth"].to_numpy(dtype=float)
        finite = usable_truth[np.isfinite(usable_truth)]
        if blocked_by_world.get("synth_block", 0) == 0:
            problems.append("synth_block query set contained no structurally "
                            "unreachable applicant; the honest non-recourse case "
                            "cannot be demonstrated (raise -n or reseed)")
        elif finite.size and float(np.max(finite)) >= 1.0 - 1e-9:
            worst = block.iloc[int(np.nanargmax(usable_truth))]
            problems.append(
                f"synth_block/{worst['method']}: usable under the truth is "
                f"{_fmt(worst['usable_truth'], 3)} despite "
                f"{blocked_by_world['synth_block']} structurally unreachable "
                "applicants — recourse was hallucinated past the known rule")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit causal recourse on known-SCM synthetic worlds through "
                    "the true structural equations, breaking the FCM circularity "
                    "with ground truth rather than a held-out estimate.")
    parser.add_argument("-d", "--datasets", nargs="+", default=list(SYNTH_WORLDS),
                        choices=list(SYNTH_WORLDS), help="worlds to audit (default: both)")
    parser.add_argument("-m", "--methods", nargs="+", default=list(METHOD_ORDER),
                        choices=list(METHOD_ORDER), help="methods to run (default: all)")
    parser.add_argument("-n", "--n-records", type=int, default=None,
                        help=f"query-set size per world (default "
                             f"{CONFIG.N_QUERY_INSTANCES} from the config).")
    parser.add_argument("-k", type=int, default=int(CONFIG.N_COUNTERFACTUALS),
                        help="counterfactuals requested per record.")
    parser.add_argument("-t", "--target", type=int, default=1, choices=(0, 1))
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if de-circularisation does not surface "
                             "or synth_block shows no honest non-recourse.")
    args = parser.parse_args(argv)

    setup_logging("08_known_scm")
    methods = [m for m in METHOD_ORDER if m in args.methods]
    n = None if (args.n_records is None or args.n_records <= 0) else int(args.n_records)
    LOGGER.info("Known-SCM audit of %s on %s (n=%s, k=%d, target=%d)", methods,
                args.datasets, n or CONFIG.N_QUERY_INSTANCES, args.k, args.target)

    audits: list[pd.DataFrame] = []
    runs: list[dict[str, Any]] = []
    for dataset in args.datasets:
        LOGGER.info("=== %s ===", dataset)
        part = _run_dataset(dataset, methods, n, args.k, args.target, runs)
        if not part.empty:
            audits.append(part)

    if not audits:
        LOGGER.error("No world produced any result.")
        return 1

    audit = pd.concat(audits, ignore_index=True)
    contrast = _contrast(audit)
    runs_frame = pd.DataFrame(runs)
    save_dataframe(audit, CONFIG.RESULTS_DIR / OUTPUTS["audit"])
    save_dataframe(runs_frame, CONFIG.RESULTS_DIR / OUTPUTS["runs"])
    LOGGER.info("Wrote results/%s (%d rows) and results/%s (%d rows)",
                OUTPUTS["audit"], audit.shape[0], OUTPUTS["runs"], len(runs))

    blocked_by_world = {}
    if not runs_frame.empty and "n_blocked" in runs_frame.columns:
        blocked_by_world = (runs_frame.groupby("dataset")["n_blocked"].max()
                            .astype(int).to_dict())

    if not contrast.empty:
        save_latex_table(_contrast_table(contrast), "known_scm", caption=_CAPTION,
                         label="tab:known_scm", column_format="ll" + "r" + "ccc",
                         escape=False)
        LOGGER.info("known_scm:\n%s", contrast.to_string(index=False))

    problems = _check(contrast, blocked_by_world)
    if problems:
        LOGGER.warning("CHECK: %s", "; ".join(problems))
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
