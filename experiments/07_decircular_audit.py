"""Experiment 07 — is CAC's causal validity real, or an echo of its own SCM?

    python experiments/07_decircular_audit.py                     # every dataset, both arms
    python experiments/07_decircular_audit.py -d adult -a rescore # free: re-audit saved plans
    python experiments/07_decircular_audit.py -d adult -n 20      # bound the regenerate arm

§9 audits every causal proposal through the *same* fitted structural equations
the method optimised against.  For a causal method that is circular: CAC proposes
a record its own FCM says is self-consistent, and the audit asks that same FCM
whether it is, so ``causal_valid`` is close to 1.0 almost by construction (and for
MINT it is 1.0 *by* construction).  A number a method cannot fail to earn is not
evidence, and a referee is right to distrust it.  This experiment breaks the
circle by auditing through structural equations the method never saw.

The black box is untouched throughout — only the SCM used to *judge* the
consequences changes — so a drop between the two columns is a fact about causal
validity, not about the decision function.

Two arms, cheap first:

``rescore``   The headline, and free.  The main-run plans in
              ``results/counterfactuals_<dataset>.joblib`` are re-audited under an
              FCM fitted on a disjoint half of the training rows (fold B).  Every
              method appears, because a baseline's flips should survive an
              independent SCM too; the question is only sharpest for CAC, the one
              that read the SCM to begin with.  ``own`` is §9's number (the shared,
              full-data FCM); ``independent`` is fold B.  Nothing is regenerated.

``crossfit``  The stronger test, and the only one that can speak for MINT (the
              saved plans predate it).  CAC and MINT *generate* under an FCM
              fitted on fold A, then the exact same plans are audited under ``own``
              (fold A, the circular reading) and ``independent`` (fold B).  Same
              plans, swapped auditor, so the gap isolates the circularity and
              nothing else.

Expected honest story: CAC/MINT ``causal_valid`` sits near 1.0 under their own FCM
and falls under the independent one.  How far it falls is the size of the echo.

**Caveat (in the caption too).** Cross-fit halves the FCM's training data; on the
small-train datasets (german, diabetes) fold FCMs are noisy, so the ``rescore``
arm is the more trustworthy reading there and ``crossfit`` is reported beside it
rather than instead of it.

Outputs
-------
``results/decircular_audit.csv``   one row per (arm, dataset, method, auditor)
``results/decircular_runs.csv``    wall clock and failures for the regenerate arm
``paper/tables/decircular_audit.tex``   own vs. independent, side by side
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
if str(ROOT) not in sys.path:  # allow `python experiments/07_...` without installing
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, DATASETS  # noqa: E402
from src.cost.cost_functions import get_cost_model  # noqa: E402
from src.data.preprocess import DatasetBundle, load_dataset  # noqa: E402
from src.evaluation import faithfulness, feasibility, metrics  # noqa: E402
from src.methods.base import Counterfactual  # noqa: E402
from src.methods.cac import CACMethod  # noqa: E402
from src.methods.mint import MINTMethod  # noqa: E402
from src.model.blackbox import load_blackbox  # noqa: E402
from src.scm.define_scm import get_scm  # noqa: E402
from src.scm.fcm import FCM  # noqa: E402
from src.utils.io import load_joblib, save_dataframe, save_latex_table  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger("experiments.07_decircular_audit")

#: The arms, cheapest first — the order the paper reads them in.
ARMS: tuple[str, ...] = ("rescore", "crossfit")

#: The methods the regenerate arm can speak for: the ones that read an SCM to
#: generate.  A baseline does not consult the FCM, so cross-fitting its generation
#: would change nothing — it only appears in the (free) ``rescore`` arm.
CAUSAL_METHODS: tuple[str, ...] = ("cac", "mint")

#: Reading order for methods inside a table; also the cost-ratio reference, which
#: this experiment does not report but ``headline`` requires to be present.
METHOD_ORDER: tuple[str, ...] = ("cac", "mint", "oracle", "dice", "wachter", "random")
REFERENCE = "cac"

#: The quantities the circularity shows up in, pulled from the two audit summaries.
_METRIC_COLUMNS: tuple[str, ...] = ("causal_validity", "flip_survival", "usable",
                                    "gap_max_mean", "cost_mean")

OUTPUTS: dict[str, str] = {
    "audit": "decircular_audit.csv",
    "runs": "decircular_runs.csv",
}

_DATASET_LABELS = {"adult": "Adult", "german": "German", "heloc": "HELOC",
                   "diabetes": "Diabetes"}
_METHOD_LABELS = {"random": "Random", "wachter": "Wachter", "dice": "DiCE",
                  "oracle": "Oracle", "mint": "MINT", "cac": "CAC"}


# ----------------------------------------------------------------- cross-fit FCMs

def _folds(bundle: DatasetBundle, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the training rows into two disjoint, seeded halves A and B.

    Seeded from the config and the arm's own seed (not ``hash``, which is salted
    per process), so the two FCMs are reproducible from the command alone.  The
    halves are index sets into ``X_train``; the black box, fitted on the whole
    split, is never touched.
    """
    rng = np.random.default_rng([int(CONFIG.RANDOM_STATE), int(seed)])
    n = int(bundle.X_train.shape[0])
    perm = rng.permutation(n)
    half = n // 2
    return perm[:half], perm[half:]


def _fit_fcm(dataset: str, X: np.ndarray) -> FCM:
    """The true graph, re-fitted on ``X`` only — an independent set of equations.

    The *structure* is always ``data/scm/``'s: cross-fitting is about the fitted
    conditionals, not the graph (a referee who doubts the graph is answered by
    experiment 04, not here).  Two FCMs with the same edges and different data are
    exactly what an independent auditor is.
    """
    return FCM(dataset, scm=get_scm(dataset)).fit(X)


# --------------------------------------------------------------------- the audit

def _funnel(evaluator: metrics.Evaluator,
            results: dict[str, dict[int, list[Counterfactual]]]) -> pd.DataFrame:
    """Put a method's plans through §9's three audits, under *this* evaluator.

    The same calls experiments 02/04/05 make, so a row here means what a row of
    ``results/headline.csv`` means — the only thing different is which FCM the
    evaluator carries, which is the whole experiment.  ``flip_survival`` and the
    gap come from the faithfulness summary (both move with the FCM); ``usable`` and
    ``causal_validity`` come from the funnel.
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


def _tidy(head: pd.DataFrame, arm: str, auditor: str) -> pd.DataFrame:
    """The funnel reduced to the columns this experiment reports, labelled by arm."""
    if head.empty:
        return head
    keep = ["dataset", "method", "n_records", *_METRIC_COLUMNS]
    out = head.reindex(columns=keep).copy()
    out.insert(2, "arm", arm)
    out.insert(3, "auditor", auditor)
    return out


def _generate(method: Any, records: np.ndarray, X: np.ndarray, k: int,
              y_target: int) -> tuple[dict[int, list[Counterfactual]], dict[str, Any]]:
    """One method over the shared query set — experiment 02/04's contract.

    A record that raises scores zero coverage rather than dropping out, so a
    method that cannot generate under a half-data FCM is counted as having failed
    that applicant rather than quietly shrinking the denominator.
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
    LOGGER.info("%s/%s [crossfit]: %d plans for %d/%d records in %.1fs%s",
                method.dataset, method.name, report["n_cfs"], report["n_answered"],
                records.size, elapsed, f", {len(failures)} raised" if failures else "")
    return results, report


def _rescore(dataset: str, bundle: DatasetBundle, model: Any,
             fcm_independent: FCM) -> pd.DataFrame:
    """Re-audit §9's saved plans under the shared FCM and under an independent one.

    Free: nothing is generated.  ``own`` rebuilds the §9 evaluator (the shared,
    full-data FCM) so its column reproduces ``results/headline.csv`` on these
    plans; ``independent`` swaps in the fold-B FCM and nothing else.
    """
    path = CONFIG.RESULTS_DIR / f"counterfactuals_{dataset}.joblib"
    if not path.exists():
        LOGGER.error("%s: %s is missing — run experiments/02_compare_methods.py -d %s "
                     "first; skipping the rescore arm", dataset, path, dataset)
        return pd.DataFrame()
    results: dict[str, dict[int, list[Counterfactual]]] = load_joblib(path)
    own = metrics.Evaluator.build(dataset, bundle=bundle, model=model)
    independent = metrics.Evaluator(dataset=dataset, bundle=bundle, model=model,
                                    fcm=fcm_independent, cost_model=get_cost_model(dataset))
    LOGGER.info("%s/rescore: re-auditing %d methods x %d records under an independent FCM",
                dataset, len(results), max((len(v) for v in results.values()), default=0))
    return pd.concat([_tidy(_funnel(own, results), "rescore", "own"),
                      _tidy(_funnel(independent, results), "rescore", "independent")],
                     ignore_index=True)


def _crossfit(dataset: str, bundle: DatasetBundle, model: Any, records: np.ndarray,
              X: np.ndarray, k: int, y_target: int, fcm_a: FCM, fcm_b: FCM,
              runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Generate CAC/MINT under fold A, then audit the same plans under A and B.

    Same plans, swapped auditor: ``own`` reads them through the fold-A FCM they
    were built against (the circular reading — MINT scores a perfect gap here by
    construction), ``independent`` reads them through fold B.  The gap between the
    two columns is the circularity, with the black box and the plans held fixed.
    """
    scm = get_scm(dataset)
    builders = {
        "cac": lambda: CACMethod(dataset, model=model, scm=scm, fcm=fcm_a),
        "mint": lambda: MINTMethod(dataset, model=model, fcm=fcm_a),
    }
    plans: dict[str, dict[int, list[Counterfactual]]] = {}
    for name in CAUSAL_METHODS:
        per_record, run = _generate(builders[name](), records, X, k, y_target)
        runs.append({**run, "arm": "crossfit"})
        plans[name] = per_record

    own = metrics.Evaluator(dataset=dataset, bundle=bundle, model=model, fcm=fcm_a,
                            cost_model=get_cost_model(dataset))
    independent = metrics.Evaluator(dataset=dataset, bundle=bundle, model=model,
                                    fcm=fcm_b, cost_model=get_cost_model(dataset))
    return pd.concat([_tidy(_funnel(own, plans), "crossfit", "own"),
                      _tidy(_funnel(independent, plans), "crossfit", "independent")],
                     ignore_index=True)


def _run_dataset(dataset: str, arms: Sequence[str], n: int | None, k: int,
                 y_target: int, seed: int,
                 runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Both requested arms on one dataset, scored by §9's three audits."""
    bundle = load_dataset(dataset)
    model = load_blackbox(dataset, CONFIG.BLACKBOX_TYPE)
    fold_a, fold_b = _folds(bundle, seed)
    LOGGER.info("%s: cross-fit folds A=%d, B=%d rows of %d train", dataset,
                fold_a.size, fold_b.size, bundle.X_train.shape[0])
    fcm_b = _fit_fcm(dataset, bundle.X_train[fold_b])

    parts: list[pd.DataFrame] = []
    if "rescore" in arms:
        parts.append(_rescore(dataset, bundle, model, fcm_b))
    if "crossfit" in arms:
        records = metrics.query_instances(dataset, model=model, bundle=bundle, n=n,
                                          y_target=y_target)
        if records.size == 0:
            LOGGER.error("%s: the black box rejects nobody in the test split", dataset)
        else:
            fcm_a = _fit_fcm(dataset, bundle.X_train[fold_a])
            parts.append(_crossfit(dataset, bundle, model, records,
                                   bundle.X_test[records], k, y_target, fcm_a, fcm_b,
                                   runs))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ------------------------------------------------------------- own vs independent

def _contrast(audit: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, dataset, method): the four quantities, own beside independent."""
    rows: list[dict[str, Any]] = []
    for (arm, dataset, method), group in audit.groupby(["arm", "dataset", "method"],
                                                        sort=False):
        own = group[group["auditor"] == "own"]
        ind = group[group["auditor"] == "independent"]

        def pick(frame: pd.DataFrame, column: str) -> float:
            if len(frame) and pd.notna(frame[column].iloc[0]):
                return float(frame[column].iloc[0])
            return float("nan")

        present = ind if ind.shape[0] else own
        rows.append({
            "arm": arm, "dataset": dataset, "method": method,
            "n_records": int(pick(present, "n_records"))
            if np.isfinite(pick(present, "n_records")) else 0,
            "causal_own": pick(own, "causal_validity"),
            "causal_ind": pick(ind, "causal_validity"),
            "usable_own": pick(own, "usable"),
            "usable_ind": pick(ind, "usable"),
            "gap_own": pick(own, "gap_max_mean"),
            "gap_ind": pick(ind, "gap_max_mean"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    arm_rank = {a: i for i, a in enumerate(ARMS)}
    data_rank = {d: i for i, d in enumerate(DATASETS)}
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    out["_a"] = out["arm"].map(lambda a: arm_rank.get(a, len(ARMS)))
    out["_d"] = out["dataset"].map(lambda d: data_rank.get(d, len(DATASETS)))
    out["_m"] = out["method"].map(lambda m: method_rank.get(m, len(METHOD_ORDER)))
    return (out.sort_values(["_a", "_d", "_m"], kind="stable")
               .drop(columns=["_a", "_d", "_m"]).reset_index(drop=True))


def _fmt(value: Any, digits: int = 2) -> str:
    """A number, or ``--`` — a table that prints ``nan`` reads as a bug."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(number) else f"{number:.{digits}f}"


def _arrow(own: Any, ind: Any, digits: int = 2) -> str:
    """``own -> ind`` in one cell, so the drop is read without hunting two columns."""
    return f"${_fmt(own, digits)} \\rightarrow {_fmt(ind, digits)}$"


def _contrast_table(contrast: pd.DataFrame) -> pd.DataFrame:
    """The LaTeX table: causal validity, usable, and gap, each own beside independent."""
    rows: list[dict[str, Any]] = []
    for _, row in contrast.iterrows():
        rows.append({
            "Dataset": _DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"])),
            "Method": _METHOD_LABELS.get(str(row["method"]), str(row["method"])),
            "Arm": str(row["arm"]),
            "N": int(row["n_records"]),
            "Causal (own $\\rightarrow$ indep.)": _arrow(row["causal_own"], row["causal_ind"]),
            "Usable (own $\\rightarrow$ indep.)": _arrow(row["usable_own"], row["usable_ind"]),
            "Gap (own $\\rightarrow$ indep.)": _arrow(row["gap_own"], row["gap_ind"], 3),
        })
    return pd.DataFrame(rows)


_CAPTION = (
    "De-circularized causal audit. Every causal proposal in \\S9 is scored through "
    "the same fitted SCM the method optimised against, which makes CAC's and MINT's "
    "causal validity close to one almost by construction. Here the plans are "
    "re-scored through structural equations the method never saw: the \\emph{own} "
    "number uses the shared, full-data FCM (\\emph{rescore}) or the fold the method "
    "generated under (\\emph{crossfit}), and the \\emph{indep.} number a disjoint "
    "half of the training rows. The black box is identical throughout, so a drop is "
    "a fact about causal validity, not the decision function. \\emph{rescore} "
    "re-audits \\S9's saved plans (every method, no regeneration); \\emph{crossfit} "
    "regenerates CAC/MINT under fold~A and audits the same plans under fold~B. "
    "Cross-fit halves the FCM's training data, so on the small-train datasets "
    "(German, Diabetes) the \\emph{rescore} arm is the more trustworthy reading."
)


def _check(contrast: pd.DataFrame) -> list[str]:
    """The claim this experiment exists to test, stated as ways it can fail to hold.

    The honest expectation is that an independent FCM *moves* a causal method's
    numbers: causal validity drops, or — since record-level causal validity is a
    saturating ``any``-flip-survives rate that often stays pinned at 1.0 — the mean
    gap widens. Circularity surfacing in the gap is the experiment succeeding, not a
    failure, so a row is surfaced (and gates ``--strict``) only when *neither* moves:
    the one case where the independent FCM found no echo at all. Keying the gate on
    causal validity alone would mis-fire on exactly the datasets where the echo lives
    in the gap rather than in the saturated validity rate.
    """
    problems: list[str] = []
    if contrast.empty:
        return ["no (arm, dataset, method) produced a result"]
    for _, row in contrast.iterrows():
        if str(row["method"]) not in CAUSAL_METHODS:
            continue
        own, ind = row["causal_own"], row["causal_ind"]
        gap_own, gap_ind = row["gap_own"], row["gap_ind"]
        causal_known = bool(np.isfinite(own) and np.isfinite(ind))
        gap_known = bool(np.isfinite(gap_own) and np.isfinite(gap_ind))
        if not causal_known and not gap_known:
            continue  # the method is absent from this cell — nothing to judge
        causal_dropped = causal_known and ind <= own - 1e-9
        gap_widened = gap_known and gap_ind >= gap_own + 1e-9
        if not causal_dropped and not gap_widened:
            problems.append(
                f"{row['dataset']}/{row['arm']}/{row['method']}: the independent FCM "
                f"moved neither causal validity ({_fmt(own, 3)} -> {_fmt(ind, 3)}) nor "
                f"the mean gap ({_fmt(gap_own, 3)} -> {_fmt(gap_ind, 3)}); no "
                f"circularity surfaced here")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-audit §9's causal proposals through structural equations the "
                    "method never saw, breaking the SCM circularity.")
    parser.add_argument("-d", "--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS), help="datasets to audit (default: all)")
    parser.add_argument("-a", "--arms", nargs="+", default=list(ARMS),
                        choices=list(ARMS),
                        help="rescore (free, every method) and/or crossfit (regenerate "
                             "cac+mint)")
    parser.add_argument("-n", "--n-records", type=int, default=20,
                        help="query-set size for the crossfit regenerate arm. Defaults "
                             f"to 20 rather than §9's {CONFIG.N_QUERY_INSTANCES} to keep "
                             "a default run bounded; the rescore arm reads every saved "
                             "plan regardless. The query set is a prefix of §9's.")
    parser.add_argument("-k", type=int, default=int(CONFIG.N_COUNTERFACTUALS),
                        help="counterfactuals per record for the crossfit arm.")
    parser.add_argument("-t", "--target", type=int, default=1, choices=(0, 1))
    parser.add_argument("--seed", type=int, default=int(CONFIG.RANDOM_STATE),
                        help="seed for the disjoint cross-fit fold split.")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if an independent audit fails to reduce a "
                             "causal method's validity.")
    args = parser.parse_args(argv)

    setup_logging("07_decircular_audit")
    LOGGER.info("Arms %s on %s (n=%d, k=%d, target=%d, fold seed=%d)", args.arms,
                args.datasets, args.n_records, args.k, args.target, args.seed)

    audits: list[pd.DataFrame] = []
    runs: list[dict[str, Any]] = []
    n = None if args.n_records <= 0 else int(args.n_records)
    for dataset in args.datasets:
        LOGGER.info("=== %s ===", dataset)
        part = _run_dataset(dataset, args.arms, n, args.k, args.target, args.seed, runs)
        if not part.empty:
            audits.append(part)

    if not audits:
        LOGGER.error("No dataset produced any result.")
        return 1

    audit = pd.concat(audits, ignore_index=True)
    contrast = _contrast(audit)
    save_dataframe(audit, CONFIG.RESULTS_DIR / OUTPUTS["audit"])
    save_dataframe(pd.DataFrame(runs), CONFIG.RESULTS_DIR / OUTPUTS["runs"])
    LOGGER.info("Wrote results/%s (%d rows) and results/%s (%d rows)",
                OUTPUTS["audit"], audit.shape[0], OUTPUTS["runs"], len(runs))

    if not contrast.empty:
        save_latex_table(_contrast_table(contrast), "decircular_audit",
                         caption=_CAPTION, label="tab:decircular",
                         column_format="lll" + "r" + "ccc", escape=False)
        LOGGER.info("decircular_audit:\n%s", contrast.to_string(index=False))

    problems = _check(contrast)
    if problems:
        LOGGER.warning("CHECK: %s", "; ".join(problems))
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
