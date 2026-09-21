"""§9's numbers, defined once and applied to every method from the outside.

The whole comparison turns on this module being indifferent to who produced a
counterfactual.  A method that scores its own homework can define *validity* as
whatever it happens to achieve and *cost* as whatever it happens to minimise —
so nothing here asks a method for a metric.  Given a factual row and a proposed
row, the functions below re-derive every quantity from the shared artefacts: the
black box for validity, the fitted SCM for the causal gap, §7's cost model for
effort, the training split for plausibility.  Where a method also reports a
number of its own (CAC fills ``causal_gap``), :func:`evaluate` recomputes it and
:func:`disagreements` will say so if the two ever differ.

Three conventions make the tables comparable:

*One query set.*  :func:`query_instances` picks the records every method must
explain — the first ``N_QUERY_INSTANCES`` test rows the black box currently
rejects, in index order.  Deterministic, so run *n* and run *n+1* compare the
same people, and shared, so no method gets an easier sample.  The queries that
selection costs belong to the fixture, not to any method, and are excluded from
the query counts each method reports.

*One notion of failure.*  A method that returns nothing for a record is not
silently dropped: it scores zero coverage on that record.  Averages over "the
records where the method worked" flatter whichever method gave up most often,
so every rate here has the *query set* as its denominator and
:func:`summarise` reports ``coverage`` alongside so a reader can see how much of
each average is real.

*One reference.*  Costs are reported both absolutely and as a ratio to the
oracle's proven optimum on the same record, which is the only way to read "0.03"
as either good or wasteful.  Records the oracle could not solve are excluded
from the ratio and counted in ``n_ratio``, never quietly averaged in at 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.cost.cost_functions import CostModel, get_cost_model
from src.data.preprocess import DatasetBundle, load_dataset
from src.methods.base import MOVE_ATOL, Counterfactual
from src.model.blackbox import BlackBoxModel, load_blackbox
from src.scm.fcm import FCM, get_fcm
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: Columns :func:`evaluate` always produces, in table order.
BASE_COLUMNS: tuple[str, ...] = (
    "dataset", "method", "record", "rank", "y_original", "y_target",
    "y_predicted", "valid", "proba_target", "cost", "raw_cost", "state_cost",
    "n_changed", "n_actions", "l0", "l1", "l2", "causal_gap", "causal_valid",
    "feasible", "n_violations", "violations", "side_effects", "plausibility",
    "sensitive_shift", "changed_features", "runtime_s", "n_queries",
)

#: Columns a method may add on its own, kept when present.  Anything a method
#: puts in ``extras`` under one of these names reaches the table; anything else
#: stays in the objects, because a column that exists for one method and not
#: another cannot be part of a comparison.
EXTRA_COLUMNS: tuple[str, ...] = (
    "n_generations", "objective", "term_validity", "term_cost",
    "term_feasibility", "term_plausibility", "proven_optimal", "n_evaluated",
)


def query_instances(
    dataset: str,
    model: BlackBoxModel | None = None,
    bundle: DatasetBundle | None = None,
    n: int | None = None,
    y_target: int = 1,
) -> np.ndarray:
    """Indices into ``X_test`` of the records every method must explain.

    The people who need recourse are the ones the black box currently rejects,
    so the query set is the test rows it predicts *away* from ``y_target``.
    Selection is by index order rather than by sampling: a reader can reproduce
    the exact set from the dataset alone, without knowing a seed.
    """
    bundle = bundle or load_dataset(dataset)
    model = model or load_blackbox(dataset)
    n = int(CONFIG.N_QUERY_INSTANCES if n is None else n)
    predicted = np.asarray(model.predict(bundle.X_test)).ravel().astype(int)
    idx = np.flatnonzero(predicted != int(y_target))[:n]
    LOGGER.info("%s: query set = %d of %d test records the black box rejects "
                "(asked for %d)", dataset, idx.size, bundle.X_test.shape[0], n)
    return idx


@dataclass(slots=True)
class Evaluator:
    """The shared artefacts every metric is derived from, loaded once.

    One instance per dataset, reused across methods — partly for speed, mostly
    so that "the SCM" and "the cost model" mean the same objects in every row of
    every table.  A metric computed against a differently-fitted FCM would not
    be comparable, and nothing in the results would show it.
    """

    dataset: str
    bundle: DatasetBundle
    model: BlackBoxModel
    fcm: FCM
    cost_model: CostModel

    @classmethod
    def build(cls, dataset: str, bundle: DatasetBundle | None = None,
              model: BlackBoxModel | None = None) -> "Evaluator":
        return cls(
            dataset=dataset,
            bundle=bundle or load_dataset(dataset),
            model=model or load_blackbox(dataset),
            fcm=get_fcm(dataset),
            cost_model=get_cost_model(dataset),
        )

    # ------------------------------------------------------------------ pieces
    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.bundle.feature_names)

    @property
    def span(self) -> np.ndarray:
        """Encoded width of each feature, for scale-free distances."""
        bounds = self.cost_model.bounds
        return np.maximum(bounds[:, 1] - bounds[:, 0], 1e-12)

    def is_valid(self, cf: Counterfactual) -> bool:
        """Did the black box actually change its mind?  Asked here, not there.

        One query per call, charged to the evaluation rather than to the method —
        the method's own ``n_queries`` was fixed when it finished searching, so
        auditing it afterwards cannot inflate the cost it reports.
        """
        proba = float(self.model.predict_proba(
            np.asarray(cf.x_cf, dtype=float).reshape(1, -1))[0, 1])
        return int(proba >= float(CONFIG.DECISION_THRESHOLD)) == int(cf.y_target)

    def causal_gap(self, X: np.ndarray, X_cf: np.ndarray) -> np.ndarray:
        """How far each proposal is from the world its own changes imply.

        The rule §9 applies to everybody: read the proposal as the intervention
        ``do(X_S = x'_S)`` on the features it moved, let the SCM recompute the
        rest, and measure the largest normalised disagreement.  A method that
        thought in intervention space scores zero by construction; one that moved
        coordinates independently scores whatever the graph says it got wrong.
        """
        return self.fcm.violation(X, X_cf).max(axis=1)

    def plausibility(self, X_cf: np.ndarray, y_target: int) -> np.ndarray:
        """Distance to the nearest ``CONFIG.KNN_K`` accepted training records.

        Normalised by the median such distance among those records, so ``1.0``
        is "as ordinary as a typical member of the target class" and ``3.0`` is
        "three times further out than typical".  Labelled from ``y_train``, not
        from the black box, so measuring plausibility costs no queries and stays
        a statement about the data rather than about the model.
        """
        R = self.bundle.X_train[self.bundle.y_train == int(y_target)]
        if R.size == 0:
            return np.full(np.atleast_2d(X_cf).shape[0], float("nan"))
        scale = _knn_distance(R[:200], R, self.span, drop_self=True)
        median = float(np.median(scale)) if scale.size else 1.0
        return _knn_distance(np.atleast_2d(X_cf), R, self.span) / max(median, 1e-12)

    def sensitive_shift(self, X: np.ndarray, X_cf: np.ndarray) -> np.ndarray:
        """Total normalised movement in features the SCM marks sensitive.

        Not an error and not a violation — a number.  Recourse that improves an
        outcome by shifting a protected attribute is a different kind of advice
        from recourse that does not, and §9 should be able to say which methods
        do it and how much.
        """
        sensitive = np.array([n in set(self.fcm.scm.sensitive)
                              for n in self.feature_names], dtype=bool)
        if not sensitive.any():
            return np.zeros(np.atleast_2d(X_cf).shape[0])
        diff = np.abs(np.atleast_2d(X_cf) - np.atleast_2d(X)) / self.span
        return diff[:, sensitive].sum(axis=1)

    # -------------------------------------------------------------------- rows
    def row(self, cf: Counterfactual, record: int, rank: int) -> dict[str, Any]:
        """One counterfactual as one table row, every number recomputed here.

        The method's own diagnostics are read only where they are facts about the
        run that nobody else can observe — how long it took, how many queries it
        spent, which features it *asked* for — and never where they are claims
        about quality.  Cost in particular is re-derived from the declared plan
        through §7's model, so a method cannot report a cheaper number than the
        one its own suggestion implies.
        """
        x = np.asarray(cf.x_original, dtype=float).reshape(1, -1)
        x_cf = np.asarray(cf.x_cf, dtype=float).reshape(1, -1)
        # What the method asks the person to do.  For every method but CAC this
        # is the whole move; CAC declares the subset it calls an intervention and
        # leaves the rest to the SCM, so pricing the declaration keeps one rule.
        asked = set(cf.changed_features)
        acted = np.array([n in asked for n in self.feature_names], dtype=bool)
        plan = self.cost_model.breakdown(x.ravel(), np.where(acted, x_cf, x).ravel())
        state, side = self.cost_model.as_effort(x.ravel(), x_cf.ravel())
        proba = float(self.model.predict_proba(x_cf)[0, 1])
        p_target = proba if int(cf.y_target) == 1 else 1.0 - proba
        predicted = int(proba >= float(CONFIG.DECISION_THRESHOLD))
        gap = float(self.causal_gap(x, x_cf)[0])
        diff = (x_cf - x).ravel()
        return {
            "dataset": self.dataset,
            "method": cf.method,
            "record": int(record),
            "rank": int(rank),
            "y_original": int(cf.y_original),
            "y_target": int(cf.y_target),
            "y_predicted": predicted,
            "valid": bool(predicted == int(cf.y_target)),
            "proba_target": p_target,
            "cost": float(plan.total),
            "raw_cost": float(plan.raw),
            "state_cost": float(state.total),
            "n_changed": int(plan.n_changed),
            "n_actions": int(cf.extras.get("n_actions", plan.n_changed)),
            "l0": int(np.count_nonzero(np.abs(diff) > MOVE_ATOL)),
            "l1": float(np.abs(diff / self.span).sum()),
            "l2": float(np.sqrt(((diff / self.span) ** 2).sum())),
            "causal_gap": gap,
            "causal_valid": bool(gap <= float(CONFIG.CAUSAL_TOLERANCE)),
            "feasible": bool(plan.feasible),
            "n_violations": int(len(plan.violations)),
            "violations": ";".join(f"{n}:{c}" for n, c in plan.violations),
            "side_effects": ";".join(f"{n}:{c}" for n, c in side),
            "plausibility": float(self.plausibility(x_cf, int(cf.y_target))[0]),
            "sensitive_shift": float(self.sensitive_shift(x, x_cf)[0]),
            "changed_features": ";".join(cf.changed_features),
            "runtime_s": float(cf.runtime_s),
            "n_queries": int(cf.n_queries),
        }

    def table(self, results: Mapping[int, Sequence[Counterfactual]],
              method: str | None = None) -> pd.DataFrame:
        """``{record: [counterfactuals]}`` as a long table, one row per proposal.

        A record that produced nothing still appears, as a single row with
        ``rank = -1`` and ``valid = False``, so the denominator of every rate is
        the query set rather than the set of successes.  §9 can drop those rows
        for a per-counterfactual average and keep them for a per-record one; it
        cannot accidentally lose them.
        """
        rows: list[dict[str, Any]] = []
        for record, cfs in results.items():
            if not cfs:
                rows.append(_empty_row(self.dataset, method or "?", int(record)))
                continue
            for rank, cf in enumerate(cfs):
                row = self.row(cf, int(record), rank)
                row.update({k: cf.extras[k] for k in EXTRA_COLUMNS
                            if k in cf.extras})
                rows.append(row)
        frame = pd.DataFrame(rows)
        ordered = [c for c in BASE_COLUMNS if c in frame.columns]
        return frame[ordered + [c for c in frame.columns if c not in ordered]]


def _knn_distance(X: np.ndarray, R: np.ndarray, span: np.ndarray,
                  drop_self: bool = False) -> np.ndarray:
    """Mean scaled L1 distance from each row of ``X`` to its ``k`` nearest in ``R``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    k = int(min(max(CONFIG.KNN_K, 1) + int(drop_self), R.shape[0]))
    out = np.empty(X.shape[0], dtype=float)
    for i, row in enumerate(X):
        d = np.abs((R - row) / span).sum(axis=1)
        near = np.partition(d, k - 1)[:k]
        out[i] = float(np.mean(np.sort(near)[1:]) if drop_self and k > 1
                       else np.mean(near))
    return out


def _empty_row(dataset: str, method: str, record: int) -> dict[str, Any]:
    """The row for a record a method could not explain at all.

    Not ``NaN`` everywhere: ``valid`` and ``causal_valid`` are ``False``, because
    "no answer" is a failure to provide recourse and should be counted as one.
    The quality columns *are* ``NaN``, because there is no suggestion to measure
    and a zero there would read as a free one.
    """
    row: dict[str, Any] = {c: float("nan") for c in BASE_COLUMNS}
    row.update({"dataset": dataset, "method": method, "record": record,
                "rank": -1, "valid": False, "causal_valid": False,
                "feasible": False, "violations": "", "side_effects": "",
                "changed_features": ""})
    return row


# ------------------------------------------------------------------- diversity
def diversity(evaluator: "Evaluator",
              results: Mapping[int, Sequence[Counterfactual]],
              method: str | None = None) -> pd.DataFrame:
    """Per record: how *different* a method's suggestions are from each other.

    Two numbers, because they answer different questions and can disagree
    sharply.  ``spread`` is the mean pairwise scaled-L1 distance between the
    valid proposals — the usual definition, and the one a method optimising
    diversity directly will do well on.  ``routes`` counts *distinct sets of
    changed features*: five different amounts of capital gain are one route and
    five pieces of advice differing only in degree, whereas "study", "change
    occupation" and "work more hours" are three genuinely different things a
    person could go and do.  The gap between the two columns is the contrast §9
    draws between point diversity and actionable diversity.

    Validity is re-checked here rather than taken from the method, so a diverse
    set of suggestions that do not work cannot score as diversity.
    """
    out: list[dict[str, Any]] = []
    span = evaluator.span
    for record, cfs in results.items():
        keep = [cf for cf in cfs if evaluator.is_valid(cf)]
        rows = np.atleast_2d([np.asarray(cf.x_cf, dtype=float) for cf in keep]) \
            if keep else np.empty((0, span.size))
        pairs = [float(np.abs((rows[i] - rows[j]) / span).sum())
                 for i in range(rows.shape[0]) for j in range(i + 1, rows.shape[0])]
        sets = {tuple(sorted(cf.changed_features)) for cf in keep}
        out.append({
            "dataset": evaluator.dataset,
            "method": method or (cfs[0].method if cfs else "?"),
            "record": int(record),
            "n_valid": int(len(keep)),
            "spread": float(np.mean(pairs)) if pairs else 0.0,
            "routes": int(len(sets)),
        })
    return pd.DataFrame(out)


def cost_ratio(frame: pd.DataFrame, reference: str = "oracle",
               only: str | None = None) -> pd.DataFrame:
    """Each method's cheapest plan per record, over the oracle's proven optimum.

    Only records where *both* the method and the reference found something enter
    the ratio; the rest are counted in ``n_missing`` and left out of the average
    rather than imputed.  A ratio of 1.0 means the method found the cheapest plan
    that exists on the oracle's grid; 3.0 means it asked for three times the
    effort §7 says was necessary.

    ``only`` names a boolean column restricting which of the method's plans are
    priced — ``"usable"`` for "the advice a person could actually act on".  The
    denominator never moves with it: the reference is always the reference
    method's cheapest *valid* plan, because the question is how much a method
    charges above the cheapest flip that exists, not above the cheapest flip that
    happens to survive the same filter.  The ``subset`` column records which rows
    were priced, so two tables computed with different filters cannot be read as
    the same number.

    A ratio *below* 1.0 is possible and is not a bug: the oracle's optimum is
    proven on its own discretisation of the feature box, so a continuous method
    can undercut it by a sub-grid nudge that lands on the decision boundary.  Read
    such a row together with ``plausibility`` and the robustness experiment — a
    plan that costs 3e-5 has not found cheap recourse, it has found the knife
    edge of the classifier.
    """
    valid = frame[frame["valid"].fillna(False).astype(bool)]
    ref = (valid[valid["method"] == reference]
           .groupby(["dataset", "record"], as_index=False)["cost"].min()
           .rename(columns={"cost": "reference_cost"}))
    priced = valid if only is None else valid[valid[only].fillna(False).astype(bool)]
    best = priced.groupby(["dataset", "method", "record"],
                          as_index=False)["cost"].min()
    merged = best.merge(ref, on=["dataset", "record"], how="left")
    merged["ratio"] = merged["cost"] / merged["reference_cost"].replace(0.0, np.nan)
    rows: list[dict[str, Any]] = []
    for (dataset, method), group in merged.groupby(["dataset", "method"], sort=True):
        ratios = group["ratio"].dropna()
        rows.append({
            "dataset": dataset, "method": method, "subset": only or "valid",
            "n_ratio": int(ratios.size),
            "n_missing": int(group.shape[0] - ratios.size),
            "cost_ratio_mean": float(ratios.mean()) if ratios.size else float("nan"),
            "cost_ratio_median": float(ratios.median()) if ratios.size else float("nan"),
            "cost_ratio_max": float(ratios.max()) if ratios.size else float("nan"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- summary
#: The headline table's columns, in the order §9 reads them: does the suggestion
#: work, would it survive contact with the world, what does it cost, how much is
#: being asked of the person, and what did it take to produce.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "dataset", "method", "n_records", "coverage", "validity", "causal_validity",
    "causal_gap_mean", "causal_gap_p90", "feasibility", "cost_mean",
    "cost_median", "state_cost_mean", "n_changed_mean", "l1_mean",
    "plausibility_mean", "sensitive_shift_mean", "n_cfs", "n_queries_mean",
    "runtime_s_mean",
)


def summarise(frame: pd.DataFrame, per_record: bool = True) -> pd.DataFrame:
    """One row per (dataset, method): §9's headline table.

    ``per_record=True`` reduces each record to its *best valid* suggestion before
    averaging, which is how recourse is actually consumed — a person is given a
    set of options and picks one, so the honest question is "what is the best
    thing this method offered you", not "what is the average of everything it
    listed".  Rates are computed over records either way:

    * ``coverage`` — records where the method returned anything at all;
    * ``validity`` — records where at least one suggestion flips the decision;
    * ``causal_validity`` — records where at least one *valid* suggestion also
      survives the SCM, i.e. is what it claims to be;
    * ``feasibility`` — records where at least one valid suggestion breaks none
      of §7's admissibility rules.

    Each is a fraction of the query set, so a method that answers half the
    records cannot post a better rate than one that answers all of them by being
    selective about which half it answers.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    frame = frame.copy()
    for column in ("valid", "causal_valid", "feasible"):
        frame[column] = frame[column].fillna(False).astype(bool)
    frame["answered"] = frame["rank"] >= 0

    rows: list[dict[str, Any]] = []
    for (dataset, method), group in frame.groupby(["dataset", "method"], sort=True):
        records = group.groupby("record")
        n_records = int(records.ngroups)
        valid = group[group["valid"]]
        best = _best_per_record(valid) if per_record else valid
        rows.append({
            "dataset": dataset,
            "method": method,
            "n_records": n_records,
            "coverage": float(records["answered"].any().mean()),
            "validity": float(records["valid"].any().mean()),
            "causal_validity": float(
                group.assign(both=group["valid"] & group["causal_valid"])
                .groupby("record")["both"].any().mean()),
            "causal_gap_mean": _mean(best["causal_gap"]),
            "causal_gap_p90": float(best["causal_gap"].quantile(0.90))
                              if not best.empty else float("nan"),
            "feasibility": float(
                group.assign(both=group["valid"] & group["feasible"])
                .groupby("record")["both"].any().mean()),
            "cost_mean": _mean(best["cost"]),
            "cost_median": float(best["cost"].median()) if not best.empty
                           else float("nan"),
            "state_cost_mean": _mean(best["state_cost"]),
            "n_changed_mean": _mean(best["n_changed"]),
            "l1_mean": _mean(best["l1"]),
            "plausibility_mean": _mean(best["plausibility"]),
            "sensitive_shift_mean": _mean(best["sensitive_shift"]),
            "n_cfs": int(valid.shape[0]),
            "n_queries_mean": _mean(records["n_queries"].max()),
            "runtime_s_mean": _mean(records["runtime_s"].max()),
        })
    return pd.DataFrame(rows)[list(SUMMARY_COLUMNS)]


def _best_per_record(valid: pd.DataFrame) -> pd.DataFrame:
    """The cheapest valid suggestion per record — what a person would take.

    Ties break on the causal gap and then on sparsity, so "cheapest" never
    silently means "cheapest among the ones that do not work as described".
    """
    if valid.empty:
        return valid
    ordered = valid.sort_values(["record", "cost", "causal_gap", "n_changed"],
                                kind="stable")
    return ordered.groupby("record", as_index=False).first()


def _mean(series: "pd.Series[Any]") -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if values.size else float("nan")


def disagreements(frame: pd.DataFrame, results: Mapping[str, Mapping[int,
                  Sequence[Counterfactual]]], tolerance: float = 1e-6
                  ) -> pd.DataFrame:
    """Where a method's self-reported numbers differ from the recomputed ones.

    Expected to be empty, and worth checking anyway: it is the audit that makes
    "every metric is computed from the outside" a verified statement rather than
    a claim in a docstring.  A non-empty result is not necessarily a bug — a
    method may legitimately define its own cost over a subset of what it changed,
    which is exactly what CAC does — but it is something §9 must explain rather
    than discover.
    """
    lookup = {(m, int(r), i): cf
              for m, per_record in results.items()
              for r, cfs in per_record.items()
              for i, cf in enumerate(cfs)}
    out: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        cf = lookup.get((row["method"], int(row["record"]), int(row["rank"])))
        if cf is None:
            continue
        for column, claimed in (("cost", cf.cost), ("causal_gap", cf.causal_gap)):
            recomputed = row[column]
            if not np.isfinite(claimed) or not np.isfinite(recomputed):
                continue
            if abs(float(claimed) - float(recomputed)) > tolerance:
                out.append({"dataset": row["dataset"], "method": row["method"],
                            "record": row["record"], "rank": row["rank"],
                            "field": column, "reported": float(claimed),
                            "recomputed": float(recomputed)})
    return pd.DataFrame(out)


__all__ = [
    "BASE_COLUMNS",
    "EXTRA_COLUMNS",
    "SUMMARY_COLUMNS",
    "Evaluator",
    "cost_ratio",
    "disagreements",
    "diversity",
    "query_instances",
    "summarise",
]
