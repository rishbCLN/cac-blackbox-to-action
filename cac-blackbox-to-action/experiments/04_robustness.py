"""Experiment 04 — does anything in §9 depend on a setting we happened to choose?

    python experiments/04_robustness.py                        # every arm, every dataset
    python experiments/04_robustness.py -a scm -d german -n 10
    python experiments/04_robustness.py -a model              # free: re-audits, no search

§9's tables fix four things by decree: the causal graph is the one in
``data/scm/``, ``k`` is 5, a plan may contain 3 actions, and the black box is the
XGBoost model in ``models/``.  CAC is the only method that reads the first of
those, so the first is also the one a referee is entitled to be suspicious about:
a method that needs the right graph to work is a method that needs something
nobody has.  This experiment varies all four and re-scores through the same audit.

Four arms, and they are not equally expensive:

``scm``     CAC searches with a *wrong* graph and is judged against the right
            one.  Six perturbations — drop one edge, drop two, cut every
            consequence of one lever, add an edge that is not there, reverse an
            edge, and delete the feature-to-feature structure entirely — each at
            :data:`~src.config.Config.ROBUSTNESS_SEEDS` (the last is
            deterministic).  Only the propagation changes: the intervention set,
            the grids, the cost model and the audit are the true ones, so the
            gap between a row and its reference is attributable to the graph.
``k``       ``k`` in :data:`K_VALUES` for CAC and for the strongest baseline, so
            the headline cannot be an artefact of asking for five routes.
``budget``  §7.2's action budget in :data:`BUDGETS`, moved through the cost model
            so the search ceiling, the cost denominator and the audit's
            "too many actions" rule all move together.
``model``   the reference plans, re-audited under a different black box: the
            secondary architecture, and three refits of the deployed one on
            bootstrap resamples of its own training split.  Nothing is
            regenerated, so this arm costs three model fits and no search.

The ``empty`` perturbation is the link back to §9.3: an FCM with no
feature-to-feature edges propagates nothing, so ``cac/empty`` should land near
``cac_no_causal`` in ``results/ablation.csv``.  Two experiments reaching the same
number by different routes is worth more than either alone.

Outputs
-------
``results/robustness.csv``                one row per proposal, every arm
``results/robustness_headline.csv``       the funnel, per arm and cell
``results/robustness_{scm,k,budget,model}.csv``   one table per arm
``results/robustness_graphs.csv``         what each perturbation did to the graph
``results/robustness_disagreements.csv``  CAC's self-report vs. the audit
``results/robustness_runs.csv``           wall clock and failures
and ``paper/tables/robustness_{scm,k,budget,model}.tex``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python experiments/04_...` without installing
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, DATASETS, MODEL_TYPES  # noqa: E402
from src.cost.cost_functions import get_cost_model  # noqa: E402
from src.data.preprocess import DatasetBundle, load_dataset  # noqa: E402
from src.evaluation import faithfulness, feasibility, metrics  # noqa: E402
from src.methods.base import Counterfactual, get_method  # noqa: E402
from src.methods.cac import CACMethod  # noqa: E402
from src.model.blackbox import BlackBoxModel, load_blackbox, model_path  # noqa: E402
from src.model.train import evaluate_predictions  # noqa: E402
from src.scm.define_scm import SCM, get_scm  # noqa: E402
from src.scm.fcm import FCM, get_fcm  # noqa: E402
from src.utils.io import save_dataframe, save_latex_table  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger("experiments.04_robustness")

#: The arms, in the order the paper reads them.
ARMS: tuple[str, ...] = ("scm", "k", "budget", "model")

#: How wrong the graph gets.  ``true`` is the reference row, ``empty`` the bound.
PERTURBATIONS: tuple[str, ...] = ("drop1", "drop2", "cut_lever", "add1",
                                  "reverse1", "empty")

#: Perturbations that do not depend on a seed — run once, not once per seed.
DETERMINISTIC: frozenset[str] = frozenset({"empty"})

#: Counterfactuals requested, for the ``k`` arm.
K_VALUES: tuple[int, ...] = (1, 3, 5, 10)

#: Simultaneous actions allowed, for the ``budget`` arm (§7.2).
BUDGETS: tuple[int, ...] = (1, 2, 3)

#: The baseline the ``k`` arm carries alongside CAC — the one §9 finds strongest.
K_BASELINE = "dice"

OUTPUTS: dict[str, str] = {
    "counterfactuals": "robustness.csv",
    "headline": "robustness_headline.csv",
    "scm": "robustness_scm.csv",
    "k": "robustness_k.csv",
    "budget": "robustness_budget.csv",
    "model": "robustness_model.csv",
    "graphs": "robustness_graphs.csv",
    "disagreements": "robustness_disagreements.csv",
    "runs": "robustness_runs.csv",
}

_DATASET_LABELS = {"adult": "Adult", "german": "German", "heloc": "HELOC",
                   "diabetes": "Diabetes"}

_PERTURBATION_LABELS = {
    "true": "the graph in \\texttt{data/scm/}",
    "drop1": "one edge missing",
    "drop2": "two edges missing",
    "cut_lever": "one lever believed inert",
    "add1": "one edge that is not there",
    "reverse1": "one edge pointing the wrong way",
    "empty": "no structure at all",
}


def _feature_edges(scm: SCM) -> list[tuple[str, str]]:
    """Edges between *features*, sorted for reproducibility.

    Edges into the target are excluded throughout this arm, and not for
    convenience: the target is not a column of ``X``, so :class:`~src.scm.fcm.FCM`
    never fits an equation for it and never propagates through it.  Perturbing
    those edges would change the picture in §6 and nothing a counterfactual
    method computes, which would understate the graph's importance by padding the
    arm with rows that cannot move.
    """
    return sorted((u, v) for u, v in scm.graph.edges() if v != scm.target)


def _perturb(scm: SCM, kind: str, seed: int) -> tuple[SCM, dict[str, Any]]:
    """A deliberately wrong version of ``scm``, plus a note of what was done.

    Only the edges move.  ``non_manipulable``, ``sensitive`` and the
    interventional costs are properties of the world rather than of the graph —
    nobody's age becomes actionable because an analyst drew an arrow wrongly — so
    they are carried over untouched, which is also what keeps CAC's intervention
    set identical across this whole arm.
    """
    graph: nx.DiGraph = scm.graph.copy()
    edges = _feature_edges(scm)
    # Seeded from the config and the arm's own seed only: `hash` is salted per
    # process, so a graph chosen with it would differ between two runs of the
    # same command.  The edge list already differs per dataset.
    rng = np.random.default_rng([int(CONFIG.RANDOM_STATE), int(seed)])
    removed: list[tuple[str, str]] = []
    added: list[tuple[str, str]] = []

    if kind == "true":
        pass
    elif kind in ("drop1", "drop2"):
        want = min(int(kind[-1]), len(edges))
        removed = [edges[i] for i in rng.choice(len(edges), size=want, replace=False)]
    elif kind == "cut_lever":
        levers = [f for f in scm.actionable
                  if any(v != scm.target for v in graph.successors(f))]
        if levers:
            lever = sorted(levers)[int(rng.integers(len(levers)))]
            removed = [(lever, v) for v in sorted(graph.successors(lever))
                       if v != scm.target]
    elif kind == "empty":
        removed = list(edges)
    elif kind == "add1":
        absent = [(u, v) for u in scm.features for v in scm.features
                  if u != v and not graph.has_edge(u, v)
                  and not nx.has_path(graph, v, u)]
        if absent:
            added = [sorted(absent)[int(rng.integers(len(absent)))]]
    elif kind == "reverse1":
        for i in rng.permutation(len(edges)):
            u, v = edges[int(i)]
            probe = graph.copy()
            probe.remove_edge(u, v)
            if not nx.has_path(probe, u, v):  # reversing would not close a cycle
                removed, added = [(u, v)], [(v, u)]
                break
    else:
        raise ValueError(f"unknown perturbation {kind!r}; expected one of "
                         f"{('true',) + PERTURBATIONS}")

    graph.remove_edges_from(removed)
    graph.add_edges_from(added, type="direct")
    if not nx.is_directed_acyclic_graph(graph):  # pragma: no cover - guarded above
        raise ValueError(f"{scm.dataset}/{kind}: perturbation created a cycle")
    detail = "; ".join([f"-{u}->{v}" for u, v in removed]
                       + [f"+{u}->{v}" for u, v in added]) or "-"
    report = {
        "dataset": scm.dataset,
        "perturbation": kind,
        "seed": int(seed),
        "n_removed": len(removed),
        "n_added": len(added),
        "shd": len(removed) + len(added),
        "n_edges": int(graph.number_of_edges()),
        "detail": detail,
    }
    return dataclasses.replace(scm, graph=graph), report


def _generate(method: Any, records: np.ndarray, X: np.ndarray, k: int,
              y_target: int, arm: str, cell: str) -> tuple[dict[int, list[Counterfactual]],
                                                           dict[str, Any]]:
    """One method over the shared query set — experiments 02 and 03's contract.

    A record that raises scores zero coverage rather than dropping out: this arm
    exists to measure degradation, and a variant that crashes on an applicant has
    degraded as far as it is possible to degrade.
    """
    results: dict[int, list[Counterfactual]] = {}
    failures: list[int] = []
    started = time.perf_counter()
    for i, record in enumerate(records):
        try:
            results[int(record)] = method.generate(X[i], y_target=y_target, k=k)
        except Exception as exc:  # noqa: BLE001 - a crash is a failure, not a gap
            LOGGER.warning("%s/%s record %d: %s: %s", method.name, cell,
                           int(record), type(exc).__name__, exc)
            results[int(record)] = []
            failures.append(int(record))
    elapsed = time.perf_counter() - started
    report = {
        "dataset": method.dataset,
        "arm": arm,
        "cell": cell,
        "method": method.name,
        "n_records": int(records.size),
        "n_answered": int(sum(1 for v in results.values() if v)),
        "n_failed": int(len(failures)),
        "n_cfs": int(sum(len(v) for v in results.values())),
        "wall_s": float(elapsed),
        "s_per_record": float(elapsed / max(records.size, 1)),
    }
    LOGGER.info("%s/%s [%s]: %d plans for %d/%d records in %.1fs%s", method.dataset,
                method.name, cell, report["n_cfs"], report["n_answered"],
                records.size, elapsed,
                f", {len(failures)} raised" if failures else "")
    return results, report


@dataclasses.dataclass
class _Arm:
    """The frames one arm produced, and the cell everything is read against."""

    name: str
    reference: str
    metrics: list[pd.DataFrame] = dataclasses.field(default_factory=list)
    faith: list[pd.DataFrame] = dataclasses.field(default_factory=list)
    feas: list[pd.DataFrame] = dataclasses.field(default_factory=list)
    results: dict[str, dict[int, list[Counterfactual]]] = dataclasses.field(
        default_factory=dict)
    columns: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    #: Maps a cell to the cell it should be read against, when that is not
    #: :attr:`reference` for the whole arm.  The ``k`` arm needs it: a cost ratio
    #: against CAC asked for a *different* number of routes would answer a
    #: question nobody asked.
    reference_of: Callable[[str], str] | None = None

    def add(self, evaluator: metrics.Evaluator,
            per_record: dict[int, list[Counterfactual]], label: str,
            **extra: Any) -> None:
        """Score one cell through the same three audits §9 uses on the baselines.

        The ``method`` column is then forced to ``label``, because a cell is not
        always a method: the ``model`` arm re-audits one set of plans under
        several black boxes, so the same ``Counterfactual`` objects have to appear
        under several row names, and only this frame knows which one it is.
        """
        for frame, out in ((evaluator.table(per_record, method=label), self.metrics),
                           (faithfulness.assess(evaluator, per_record, method=label),
                            self.faith),
                           (feasibility.assess(evaluator, per_record, method=label),
                            self.feas)):
            if not frame.empty:
                frame["method"] = label
            out.append(frame)
        self.results[label] = per_record
        self.columns.append({"dataset": evaluator.dataset, "method": label,
                             "arm": self.name, **extra})

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (pd.concat(self.metrics, ignore_index=True),
                pd.concat(self.faith, ignore_index=True),
                pd.concat(self.feas, ignore_index=True))

    def headline(self) -> pd.DataFrame:
        """The funnel for this arm, with the causal columns joined on.

        ``reference`` is the arm's own default setting rather than the oracle:
        every question here is "what does changing this cost, relative to leaving
        it alone", and importing experiment 02's reference would answer a
        different one.  Only the cost ratio depends on it, so when
        :attr:`reference_of` splits the arm into slices the funnel is computed
        once per slice and the rates are unaffected — they are per record and per
        method, and no method's rate can be changed by which other rows it is
        tabulated beside.
        """
        frame, faith, feas = self.frames()
        if frame.empty or "method" not in frame.columns:
            return pd.DataFrame()
        groups: dict[str, list[str]] = {}
        for label in self.results:
            reference = (self.reference if self.reference_of is None
                         else self.reference_of(label))
            groups.setdefault(reference, []).append(label)
        heads = [feasibility.headline(frame[frame["method"].isin(labels)],
                                      faith[faith["method"].isin(labels)],
                                      feas[feas["method"].isin(labels)],
                                      reference=reference)
                 for reference, labels in groups.items()]
        head = pd.concat(heads, ignore_index=True)
        summary = faithfulness.summarise(faith)
        head = head.merge(summary[["dataset", "method", "gap_max_mean",
                                   "gap_max_p90", "flip_survival", "promise_kept"]],
                          on=["dataset", "method"], how="left")
        head = head.merge(pd.DataFrame(self.columns), on=["dataset", "method"],
                          how="left")
        head["arm"] = self.name
        return head


def _relabel(method: Any, label: str) -> Any:
    """Make ``method`` sign its rows ``label``.

    :meth:`~src.evaluation.metrics.Evaluator.row` reads ``Counterfactual.method``
    rather than the argument it was handed, which is the right way round — the
    audit should take the label from the object that produced the plan.  So the
    cell name goes on the instance, shadowing the class attribute exactly as
    CAC's own ``label`` knob does.  Nothing in :mod:`src.methods` branches on
    ``name``; it reaches the dataclass field and the log line and stops there.
    """
    method.name = label
    return method


def _scm_label(kind: str, seed: int | None) -> str:
    return f"cac/{kind}" if seed is None else f"cac/{kind}#{seed}"


def _arm_scm(dataset: str, bundle: DatasetBundle, model: Any,
             evaluator: metrics.Evaluator, records: np.ndarray, X: np.ndarray,
             k: int, y_target: int, seeds: Sequence[int],
             runs: list[dict[str, Any]], graphs: list[dict[str, Any]]) -> _Arm:
    """CAC with a wrong graph, judged against the right one.

    ``evaluator`` is built from the true SCM and never told about the
    perturbation, which is the whole design: the method's own ``causal_gap`` is
    measured against the graph it was given, so a misspecified run reports
    consistency it does not have and only the audit can see the difference.  That
    divergence is written to ``results/robustness_disagreements.csv``.
    """
    true_scm = get_scm(dataset)
    arm = _Arm("scm", reference=_scm_label("true", None))
    cells: list[tuple[str, int | None]] = [("true", None)]
    for kind in PERTURBATIONS:
        cells.extend([(kind, None)] if kind in DETERMINISTIC
                     else [(kind, int(s)) for s in seeds])

    for kind, seed in cells:
        label = _scm_label(kind, seed)
        scm, report = _perturb(true_scm, kind, 0 if seed is None else seed)
        graphs.append({**report, "label": label})
        if kind != "true" and report["shd"] == 0:
            LOGGER.warning("%s/%s: perturbation changed nothing, skipping", dataset,
                           label)
            continue
        if kind == "true":
            method = CACMethod(dataset, model=model, label=label)
        else:
            fcm = FCM(dataset, scm=scm).fit(bundle.X_train)
            method = CACMethod(dataset, model=model, scm=scm, fcm=fcm, label=label)
        per_record, run = _generate(method, records, X, k, y_target, "scm", label)
        runs.append(run)
        arm.add(evaluator, per_record, label, cell=label,
                perturbation=kind, seed=-1 if seed is None else int(seed),
                shd=int(report["shd"]), n_edges=int(report["n_edges"]),
                detail=report["detail"])
    return arm


def _arm_k(dataset: str, model: Any, evaluator: metrics.Evaluator,
           records: np.ndarray, X: np.ndarray, y_target: int,
           runs: list[dict[str, Any]]) -> _Arm:
    """How many routes each method is asked for.

    CAC and the baseline §9 finds strongest, at four values of ``k``.  Asking for
    more suggestions can only help a per-record rate — the rate counts a record
    once if *any* suggestion passes — so the question is not whether the numbers
    rise but whether the *ordering* between the two methods survives, and whether
    the extra suggestions are extra routes or the same route restated.

    Effort is read two ways, because one number cannot answer both questions:
    ``cost_mean`` is comparable straight down the column (same denominator, same
    query set, so it shows what another route buys), while ``cost_ratio_median``
    is a paired per-record median against CAC *at the same* ``k``, which is the
    only fair way to put the two methods beside each other.
    """
    arm = _Arm("k", reference=f"cac/k={int(CONFIG.N_COUNTERFACTUALS)}",
               reference_of=lambda label: f"cac/k={label.rsplit('=', 1)[-1]}")
    for value in K_VALUES:
        for name in ("cac", K_BASELINE):
            label = f"{name}/k={value}"
            method = _relabel(get_method(name, dataset, model=model), label)
            per_record, run = _generate(method, records, X, value, y_target, "k",
                                        label)
            runs.append(run)
            arm.add(evaluator, per_record, label, cell=f"k={value}", k=int(value),
                    base_method=name)
    return arm


def _arm_budget(dataset: str, bundle: DatasetBundle, model: Any,
                records: np.ndarray, X: np.ndarray, k: int, y_target: int,
                runs: list[dict[str, Any]]) -> _Arm:
    """§7.2's action budget, moved everywhere at once.

    The budget is not only a ceiling on the search: §7 normalises effort by the
    cost of the ``max_actions`` dearest single actions, and the audit calls a plan
    over-long by the same number.  Passing one :class:`~src.cost.cost_functions.
    CostModel` to the method *and* to the evaluator keeps all three definitions
    in step — but it also means the ``cost`` column of each cell is divided by a
    different denominator, so the last column of the table reports what that
    denominator was and the costs are not comparable across rows.
    """
    arm = _Arm("budget", reference=f"cac/actions={int(CONFIG.MAX_ACTIONS)}")
    fcm = get_fcm(dataset)
    for budget in BUDGETS:
        label = f"cac/actions={budget}"
        cost_model = get_cost_model(dataset, max_actions=int(budget))
        evaluator = metrics.Evaluator(dataset=dataset, bundle=bundle, model=model,
                                     fcm=fcm, cost_model=cost_model)
        method = CACMethod(dataset, model=model, cost_model=cost_model, label=label)
        per_record, run = _generate(method, records, X, k, y_target, "budget", label)
        runs.append(run)
        arm.add(evaluator, per_record, label, cell=f"actions={budget}",
                max_actions=int(budget), budget_effort=float(cost_model.budget))
    return arm


def _refit(dataset: str, bundle: DatasetBundle, seed: int) -> tuple[Any, dict[str, Any]]:
    """The deployed architecture, refitted on a bootstrap resample of its own split.

    Reseeding alone would not do: the deployed XGBoost draws no subsample, so
    ``random_state`` changes nothing about the fitted trees and the arm would
    report perfect survival by construction.  Resampling the training rows is the
    scenario recourse actually has to survive — the same bank, the same
    architecture, next quarter's refit — and it is the protocol the robustness
    literature uses for exactly that reason.
    """
    rng = np.random.default_rng([int(CONFIG.RANDOM_STATE), int(seed)])
    n = bundle.X_train.shape[0]
    idx = rng.integers(0, n, size=n)
    fitted = BlackBoxModel(CONFIG.BLACKBOX_TYPE, transformer=bundle.transformer,
                           dataset=dataset, random_state=int(seed))
    fitted.fit(bundle.X_train[idx], bundle.y_train[idx])
    scored = evaluate_predictions(bundle.y_test,
                                  fitted.proba_positive(bundle.X_test))
    return fitted, {"auc": float(scored["roc_auc"])}


def _audit_models(dataset: str, bundle: DatasetBundle, deployed: Any,
                  seeds: Sequence[int]) -> list[tuple[str, Any, dict[str, Any]]]:
    """Every black box the reference plans get re-checked against.

    The deployed model comes first, so its row reproduces §9's numbers on this
    query set and the rest of the column can be read as a drop from something
    familiar.  ``agreement`` is how often the audit model and the deployed one
    label the same test record identically: a plan that stops working under a
    model that disagrees about half the data has not told us much about recourse.
    """
    reference = np.asarray(deployed.predict(bundle.X_test)).ravel().astype(int)
    scored = evaluate_predictions(bundle.y_test,
                                  deployed.proba_positive(bundle.X_test))
    out: list[tuple[str, Any, dict[str, Any]]] = [
        (CONFIG.BLACKBOX_TYPE, deployed,
         {"audit_auc": float(scored["roc_auc"]), "agreement": 1.0}),
    ]
    for other in (t for t in MODEL_TYPES if t != CONFIG.BLACKBOX_TYPE):
        if not model_path(dataset, other).exists():
            LOGGER.warning("%s: no %s black box on disk, skipping that audit",
                           dataset, other)
            continue
        out.append((other, load_blackbox(dataset, other), {}))
    for seed in seeds:
        fitted, report = _refit(dataset, bundle, int(seed))
        out.append((f"refit{int(seed)}", fitted, {"audit_auc": report["auc"]}))

    for tag, audit, extra in out:
        if "audit_auc" not in extra:
            scored = evaluate_predictions(bundle.y_test,
                                         audit.proba_positive(bundle.X_test))
            extra["audit_auc"] = float(scored["roc_auc"])
        if "agreement" not in extra:
            predicted = np.asarray(audit.predict(bundle.X_test)).ravel().astype(int)
            extra["agreement"] = float(np.mean(predicted == reference))
        LOGGER.info("%s: audit model %s — AUC %.4f, agrees with the deployed box on "
                    "%.1f%% of the test split", dataset, tag, extra["audit_auc"],
                    100 * extra["agreement"])
    return out


def _arm_model(dataset: str, bundle: DatasetBundle, model: Any,
               records: np.ndarray, X: np.ndarray, k: int, y_target: int,
               seeds: Sequence[int], runs: list[dict[str, Any]]) -> _Arm:
    """Every method's plans, re-audited under a black box that is not the one they beat.

    Nothing is regenerated: these are the same suggestions, re-scored against a
    different decision function, which is what a person experiences when the bank
    refits between the advice and the application.  ``already`` is the share of
    the query set the audit model was going to accept anyway — without it a high
    survival rate cannot be distinguished from a more generous model, since the
    query set was chosen by the deployed one.
    """
    arm = _Arm("model", reference=f"cac@{CONFIG.BLACKBOX_TYPE}")
    fcm, cost_model = get_fcm(dataset), get_cost_model(dataset)
    generated: dict[str, dict[int, list[Counterfactual]]] = {}
    for name in ("random", "wachter", "dice", "oracle", "cac"):
        method = get_method(name, dataset, model=model)
        per_record, run = _generate(method, records, X, k, y_target, "model",
                                    f"generated@{CONFIG.BLACKBOX_TYPE}")
        runs.append(run)
        generated[name] = per_record

    for tag, audit, extra in _audit_models(dataset, bundle, model, seeds):
        evaluator = metrics.Evaluator(dataset=dataset, bundle=bundle, model=audit,
                                     fcm=fcm, cost_model=cost_model)
        accepted = np.asarray(audit.predict(X)).ravel().astype(int)
        already = float(np.mean(accepted == int(y_target)))
        for name, per_record in generated.items():
            label = f"{name}@{tag}"
            arm.add(evaluator, per_record, label, cell=tag, audit=tag,
                    base_method=name, already=already, **extra)
    return arm


#: Columns of the funnel that are averaged over :data:`~src.config.Config.
#: ROBUSTNESS_SEEDS` when several draws of the same perturbation are pooled.
_POOLED = ("coverage", "validity", "causal_validity", "actionability", "usable",
           "gap_max_mean", "gap_max_p90", "flip_survival", "promise_kept",
           "cost_mean", "cost_ratio_median", "plausibility_mean", "shd")


def _pool_seeds(head: pd.DataFrame) -> pd.DataFrame:
    """Pool the ``scm`` arm's seeds: one row per perturbation, not per draw.

    Three draws of "drop one edge" are three samples of the same question, and the
    spread between them is the answer — a method that survives the average
    misspecification but collapses on one draw is not robust.  So the mean goes in
    the table and ``causal_validity_std`` goes beside it, over however many draws
    actually ran (``n_seeds``: a draw that changed nothing was skipped upstream,
    and pretending otherwise would dilute the mean towards the true graph).
    """
    if head.empty or "perturbation" not in head.columns:
        return head
    pooled = [c for c in _POOLED if c in head.columns]
    grouped = head.groupby(["dataset", "perturbation"], sort=False)
    out = grouped[pooled].mean().reset_index()
    out["causal_validity_std"] = grouped["causal_validity"].std().fillna(0.0).values
    out["usable_std"] = grouped["usable"].std().fillna(0.0).values
    out["n_seeds"] = grouped.size().values
    out["n_records"] = grouped["n_records"].max().values
    rank = {kind: i for i, kind in enumerate(("true",) + PERTURBATIONS)}
    out["_order"] = out["perturbation"].map(lambda p: rank.get(str(p), len(rank)))
    return (out.sort_values(["dataset", "_order"], kind="stable")
               .drop(columns="_order").reset_index(drop=True))


def _run_dataset(dataset: str, arms: Sequence[str], n: int | None, k: int,
                 y_target: int, seeds: Sequence[int]) -> dict[str, pd.DataFrame]:
    """Every requested arm on one dataset, scored by experiment 02's three audits.

    The query set is built the same way and from the same knob as §9's, so a row
    here and a row of ``results/headline.csv`` are averages over the same people —
    and with a smaller ``-n`` over a strict prefix of them, since
    :func:`~src.evaluation.metrics.query_instances` selects in index order.
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
    LOGGER.info("%s: %d records, arms %s, seeds %s", dataset, records.size,
                list(arms), list(seeds))

    runs: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    built: dict[str, _Arm] = {}
    if "scm" in arms:
        built["scm"] = _arm_scm(dataset, bundle, model, evaluator, records, X, k,
                                y_target, seeds, runs, graphs)
    if "k" in arms:
        built["k"] = _arm_k(dataset, model, evaluator, records, X, y_target, runs)
    if "budget" in arms:
        built["budget"] = _arm_budget(dataset, bundle, model, records, X, k,
                                      y_target, runs)
    if "model" in arms:
        built["model"] = _arm_model(dataset, bundle, model, records, X, k, y_target,
                                    seeds, runs)
    if not built:
        return {}

    rows: list[pd.DataFrame] = []
    heads: list[pd.DataFrame] = []
    for arm in built.values():
        frame = arm.frames()[0]
        if not frame.empty:
            frame["arm"] = arm.name
        rows.append(frame)
        heads.append(arm.headline())
    head = pd.concat(heads, ignore_index=True)

    out: dict[str, pd.DataFrame] = {
        "counterfactuals": pd.concat(rows, ignore_index=True),
        "headline": head,
        "graphs": pd.DataFrame(graphs),
        "runs": pd.DataFrame(runs),
        "disagreements": pd.DataFrame(),
    }
    for name, arm in built.items():
        part = head[head["arm"] == name].dropna(axis=1, how="all")
        out[name] = _pool_seeds(part) if name == "scm" else part
    # The misspecified runs are the one place a self-report and the audit are
    # *expected* to diverge: CAC measures its gap against the graph it was given,
    # and the audit measures it against the graph that is true.  The size of that
    # divergence is how misleading a wrong graph is to the person holding it.
    if "scm" in built:
        frame = built["scm"].frames()[0]
        out["disagreements"] = metrics.disagreements(frame, built["scm"].results)
    return out


_FUNNEL = {"coverage": "Cov.", "validity": "Valid", "causal_validity": "Causal",
           "actionability": "Action.", "usable": "Usable"}

_SCM_COLUMNS = {"dataset": "Dataset", "perturbation": "Graph given to CAC",
                "shd": "SHD", "n_seeds": "Draws", **_FUNNEL,
                "gap_max_mean": "Gap", "flip_survival": "Flip survives",
                "cost_ratio_median": "Cost/true"}

_K_COLUMNS = {"dataset": "Dataset", "base_method": "Method", "k": "$k$", **_FUNNEL,
              "cost_mean": "Cost", "cost_ratio_median": "Cost/CAC",
              "n_queries_mean": "Queries"}

_BUDGET_COLUMNS = {"dataset": "Dataset", "max_actions": "Actions",
                   "budget_effort": "Denom.", **_FUNNEL, "cost_mean": "Cost",
                   "plausibility_mean": "Plaus."}

_MODEL_COLUMNS = {"dataset": "Dataset", "base_method": "Method",
                  "audit": "Audit box", "audit_auc": "AUC",
                  "agreement": "Agrees", "already": "Already in",
                  "validity": "Valid", "causal_validity": "Causal",
                  "usable": "Usable", "cost_mean": "Cost"}


def _labelled(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Project onto ``columns``, in that order, and spell the codes out.

    ``causal_validity`` picks up its across-seed spread when one is present, which
    is the only column that differs between the CSV and the LaTeX: a reader of the
    table needs the spread beside the mean, and a reader of the CSV would rather
    have two numbers than a string.
    """
    present = {k: v for k, v in columns.items() if k in frame.columns}
    out = frame[list(present)].rename(columns=present).copy()
    if "Dataset" in out.columns:
        out["Dataset"] = out["Dataset"].map(lambda d: _DATASET_LABELS.get(d, d))
    if "Graph given to CAC" in out.columns:
        out["Graph given to CAC"] = out["Graph given to CAC"].map(
            lambda p: _PERTURBATION_LABELS.get(str(p), str(p)))
    if "Causal" in out.columns and "causal_validity_std" in frame.columns:
        out["Causal"] = [f"${m:.2f} \\pm {s:.2f}$" for m, s
                         in zip(out["Causal"], frame["causal_validity_std"])]
    for column in ("SHD", "Draws", "$k$", "Actions"):
        if column in out.columns:
            out[column] = _count(out[column])
    return out


def _count(values: "pd.Series[Any]") -> "pd.Series[Any]":
    """A column of counts, printed per cell rather than per column.

    ``SHD`` is a count of edge edits everywhere except in the pooled ``scm`` arm,
    where it is a mean over draws: "drop one edge" averaged over three graphs is
    1.33 edits, which cannot be cast to an integer and must not be rounded to one,
    since that would name a perturbation nobody made.  Formatting the column as a
    whole would then print ``0.000`` and ``2.000`` for the rows that really are
    counts, so each cell is rendered on its own terms and the result is text.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if bool(((numeric.dropna() % 1) == 0).all()):
        return numeric.astype("Int64")
    return numeric.map(lambda v: "" if pd.isna(v)
                       else f"{int(v)}" if float(v).is_integer() else f"{v:.2f}")


_CAPTIONS: dict[str, str] = {
    "scm": ("Graph misspecification. CAC searches with the graph named in the "
            "second column and is scored against the one in "
            "\\texttt{data/scm/}, on the same query set as "
            "Table~\\ref{tab:main_results}. Only the propagation changes: the "
            "intervention set, the grids, the cost model and the audit are the "
            "true ones, so a drop is attributable to the structure alone. "
            "\\emph{SHD} is the number of edge edits, averaged over the draws and "
            "so fractional where they differed, \\emph{Draws} how many "
            "random perturbations of that kind were averaged (the spread on "
            "\\emph{Causal} is across those draws), and \\emph{Cost/true} the "
            "median effort ratio against CAC with the true graph. The last row "
            "removes the feature-to-feature structure entirely and should "
            "reproduce \\emph{-- SCM} in Table~\\ref{tab:ablation}."),
    "k": ("Sensitivity to $k$, the number of routes each method is asked for. "
          "Every rate is per record and counts a record once if any of its $k$ "
          "suggestions passes, so all of them can only rise with $k$; the "
          "question is whether the ordering between CAC and the strongest "
          "baseline survives, and whether the extra suggestions buy anything "
          "after the first. \\emph{Cost} is comparable down the column, and "
          "\\emph{Cost/CAC} is the median per-record effort ratio against CAC "
          "asked for the same number of routes."),
    "budget": ("Sensitivity to the action budget of \\S\\ref{sec:cost}. The "
               "budget moves the search ceiling, the denominator effort is "
               "normalised by, and the audit's \\emph{too many actions} rule "
               "together, so \\emph{Cost} is divided by a different quantity in "
               "each row and is not comparable down the column --- "
               "\\emph{Denom.} reports what that quantity was. The rates are "
               "comparable, since a record either has a plan that survives all "
               "three audits within the budget or it does not."),
    "model": ("Model transfer. The plans of Table~\\ref{tab:main_results} are "
              "not regenerated, only re-scored against a black box that is not "
              "the one they were built against: the secondary architecture, and "
              "refits of the deployed one on bootstrap resamples of its own "
              "training split. \\emph{AUC} is the audit model's test AUC, "
              "\\emph{Agrees} how often it labels a test record as the deployed "
              "box does, and \\emph{Already in} the share of the query set it "
              "would have accepted with no action taken --- a survival rate is "
              "only informative next to that last column, since the query set "
              "was selected by the deployed box. \\emph{Cost} is the mean effort "
              "of each method's cheapest usable plan, and is therefore empty on "
              "the rows where no plan was usable."),
}

_FORMATS: dict[str, str] = {
    "scm": "ll" + "r" * 10, "k": "llr" + "r" * 8,
    "budget": "lrr" + "r" * 7, "model": "lll" + "r" * 7,
}

_METHOD_RANK = {"cac": 0, "oracle": 1, "dice": 2, "wachter": 3, "random": 4}


def _audit_rank(tag: str) -> int:
    """Deployed box first, then the other architectures, then the refits."""
    text = str(tag)
    if text == CONFIG.BLACKBOX_TYPE:
        return 0
    if text in MODEL_TYPES:
        return 1 + list(MODEL_TYPES).index(text)
    return 100 + (int(text[5:]) if text.startswith("refit") and text[5:].isdigit()
                  else 0)


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort so the tables read in the order the paper discusses them.

    Alphabetical would put ``k=10`` between ``k=1`` and ``k=3`` and the secondary
    black box above the deployed one, so the cells are ranked by what they *mean*
    rather than by how they are spelled.
    """
    if frame.empty:
        return frame
    out = frame.copy()
    keys: list[str] = []
    ranks: dict[str, Any] = {
        "arm": lambda s: s.map(lambda a: ARMS.index(a) if a in ARMS else len(ARMS)),
        "perturbation": lambda s: s.map(
            {k: i for i, k in enumerate(("true",) + PERTURBATIONS)}).fillna(99),
        "audit": lambda s: s.map(_audit_rank),
        "base_method": lambda s: s.map(lambda m: _METHOD_RANK.get(str(m), 99)),
        "k": lambda s: pd.to_numeric(s, errors="coerce"),
        "max_actions": lambda s: pd.to_numeric(s, errors="coerce"),
        "seed": lambda s: pd.to_numeric(s, errors="coerce"),
    }
    for column, rank in ranks.items():
        if column in out.columns:
            out[f"_{column}"] = rank(out[column])
            keys.append(f"_{column}")
    keys = ["dataset"] + keys if "dataset" in out.columns else keys
    keys += [c for c in ("method", "record", "rank") if c in out.columns]
    if not keys:
        return frame
    return (out.sort_values(keys, kind="stable")
               .drop(columns=[c for c in out.columns if c.startswith("_")])
               .reset_index(drop=True))


def _write_tables(merged: dict[str, pd.DataFrame], arms: Sequence[str]) -> None:
    """One LaTeX table per arm that ran, each with its own caption and reference."""
    for arm, columns in (("scm", _SCM_COLUMNS), ("k", _K_COLUMNS),
                         ("budget", _BUDGET_COLUMNS), ("model", _MODEL_COLUMNS)):
        if arm not in arms or merged.get(arm) is None or merged[arm].empty:
            continue
        save_latex_table(_labelled(merged[arm], columns), f"robustness_{arm}",
                         caption=_CAPTIONS[arm], label=f"tab:robustness_{arm}",
                         column_format=_FORMATS[arm], escape=False)


def _check(merged: dict[str, pd.DataFrame]) -> list[str]:
    """The claims this experiment exists to test, stated as things that can fail.

    Saying them here is cheaper than finding them out in review.  None of these is
    a bug in the code: they are ways the paper's argument could turn out not to
    hold, and each one would need reporting rather than fixing.
    """
    problems: list[str] = []
    runs = merged.get("runs")
    if runs is not None and not runs.empty:
        for row in runs[runs["n_answered"] == 0].to_dict("records"):
            problems.append(f"{row['dataset']}/{row['arm']}: {row['method']} "
                            f"answered no record")

    scm = merged.get("scm")
    if scm is not None and not scm.empty and "perturbation" in scm.columns:
        for dataset, group in scm.groupby("dataset"):
            true = group[group["perturbation"] == "true"]
            if true.empty:
                continue
            ref = float(true["causal_validity"].iloc[0])
            better = group[(group["perturbation"] != "true")
                           & (group["causal_validity"] > ref + 1e-9)]
            for row in better.to_dict("records"):
                problems.append(
                    f"{dataset}: the {row['perturbation']} graph is more causally "
                    f"valid than the true one ({row['causal_validity']:.3f} > "
                    f"{ref:.3f})")
            empty = group[group["perturbation"] == "empty"]
            if not empty.empty and float(empty["causal_validity"].iloc[0]) >= ref:
                problems.append(f"{dataset}: deleting the graph cost nothing, so the "
                                f"query set may not need the structure at all")

    k_arm = merged.get("k")
    if k_arm is not None and not k_arm.empty and "base_method" in k_arm.columns:
        for (dataset, value), group in k_arm.groupby(["dataset", "k"]):
            ours = group[group["base_method"] == "cac"]["usable"]
            theirs = group[group["base_method"] == K_BASELINE]["usable"]
            if not ours.empty and not theirs.empty \
                    and float(ours.iloc[0]) < float(theirs.iloc[0]) - 1e-9:
                problems.append(f"{dataset}: at k={int(value)} {K_BASELINE} is more "
                                f"usable than CAC ({float(theirs.iloc[0]):.3f} > "
                                f"{float(ours.iloc[0]):.3f})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Vary what §9 fixed by decree and re-score through the same "
                    "audit.")
    parser.add_argument("-d", "--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("-a", "--arms", nargs="+", default=list(ARMS),
                        choices=list(ARMS))
    parser.add_argument("-n", "--n-records", type=int, default=20,
                        help="Query-set size per dataset. Defaults to 20 rather "
                             f"than §9's {CONFIG.N_QUERY_INSTANCES}: this run is "
                             "~35 searches per dataset, and the query set is a "
                             "prefix of §9's either way.")
    parser.add_argument("-k", type=int, default=int(CONFIG.N_COUNTERFACTUALS),
                        help="Counterfactuals per record, for every arm but `k`.")
    parser.add_argument("-t", "--target", type=int, default=1, choices=(0, 1))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(CONFIG.ROBUSTNESS_SEEDS),
                        help="Draws per random perturbation, and refits of the "
                             "black box for the `model` arm.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if a cell answers no record, if a "
                             "wrong graph beats the true one, or if the baseline "
                             "overtakes CAC at some k.")
    args = parser.parse_args(argv)

    setup_logging("04_robustness")
    LOGGER.info("Arms %s on %s (n=%d, k=%d, target=%d, seeds=%s)", args.arms,
                args.datasets, args.n_records, args.k, args.target, args.seeds)

    per_dataset: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset in args.datasets:
        out = _run_dataset(dataset, args.arms, args.n_records, args.k, args.target,
                           args.seeds)
        if out:
            per_dataset[dataset] = out
    if not per_dataset:
        LOGGER.error("No dataset produced any result.")
        return 1

    merged: dict[str, pd.DataFrame] = {}
    for key in OUTPUTS:
        parts = [out[key] for out in per_dataset.values()
                 if key in out and not out[key].empty]
        merged[key] = (_ordered(pd.concat(parts, ignore_index=True)) if parts
                       else pd.DataFrame())
    for key, filename in OUTPUTS.items():
        save_dataframe(merged[key], CONFIG.RESULTS_DIR / filename)
    _write_tables(merged, args.arms)

    for key in ("scm", "k", "budget", "model", "runs"):
        if not merged[key].empty:
            LOGGER.info("%s:\n%s", key, merged[key].to_string(index=False))
    if not merged["disagreements"].empty:
        # Expected, and the point: a misspecified CAC reports a gap of zero
        # against the graph it was handed while the true SCM says otherwise.
        LOGGER.info("%d self-report/audit disagreements, all from the scm arm — "
                    "that divergence is what a wrong graph hides from the person "
                    "holding it", len(merged["disagreements"]))

    LOGGER.info("Wrote %d tables to %s and %d LaTeX tables to %s",
                sum(1 for f in merged.values() if not f.empty), CONFIG.RESULTS_DIR,
                sum(1 for a in args.arms if not merged[a].empty), CONFIG.TABLES_DIR)
    problems = _check(merged)
    if problems:
        LOGGER.error("Problems: %s", "; ".join(problems))
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
