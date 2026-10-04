"""Could a person actually do this?  §7's rules, read as questions about people.

Validity says the model would change its mind.  Faithfulness says the world
agrees.  Neither says the applicant can act: a suggestion that asks somebody to
be younger, to undo a delinquency, to change nine things at once, or to spend
more effort than §7's whole budget allows is a *description* of a better record
rather than advice.  This module turns each of §7's rules into a column and each
column into a rate, so "inactionable" becomes a number the paper can attribute to
a specific rule rather than an adjective it applies to the baselines.

Four rules, from hardest to softest:

*Immutability* — the feature cannot be acted on at all (age, gender, a past
delinquency).  A method that moves one has not produced expensive advice; it has
produced no advice.

*Direction* — the feature moves only one way in practice.  Nobody's account
history gets shorter, and a suggestion that shortens it is asking for time
travel, not for effort.

*Budget* — §7.2 caps a plan at ``MAX_ACTIONS`` simultaneous changes and
normalises cost so that ``1.0`` is the most an admissible plan of that size can
demand.  A plan above either line is outside the effort model the paper priced,
so its cost number cannot be compared with anything.

*Ordinariness* — the resulting record should look like someone who already gets
the favourable decision.  Softer than the others and reported separately for that
reason: an unusual plan is not forbidden, it is just harder to believe, and where
the other three give a verdict this one gives a distance.

Every rule is evaluated on what the method *asks for*, never on what the SCM
propagates afterwards; the same distinction :meth:`~src.cost.cost_functions.CostModel.as_effort`
draws, for the same reason.  A consequence is not a request, and charging one as
if it were would make CAC look infeasible for being the only method that reports
its consequences at all.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.evaluation.metrics import Evaluator
from src.methods.base import Counterfactual
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: Columns :func:`assess` produces, in table order.
FEASIBILITY_COLUMNS: tuple[str, ...] = (
    "dataset", "method", "record", "rank", "valid", "n_actions", "cost",
    "immutable", "wrong_direction", "over_actions", "over_budget", "admissible",
    "within_budget", "actionable", "plausibility", "ordinary",
    "sensitive_shift", "sensitive_features", "blocked_features", "actions",
)

#: A plan is "ordinary" when its resulting record is no further from the accepted
#: class than a typical member of that class already is.  ``1.0`` because
#: :meth:`~src.evaluation.metrics.Evaluator.plausibility` is normalised by that
#: median distance, so the line is the data's own scale rather than a knob.
ORDINARY_AT = 1.0


def assess(evaluator: Evaluator,
           results: Mapping[int, Sequence[Counterfactual]],
           method: str | None = None) -> pd.DataFrame:
    """Audit one method's proposals against §7's rules about what people can do."""
    flat: list[tuple[int, int, Counterfactual]] = [
        (int(record), rank, cf)
        for record, cfs in results.items()
        for rank, cf in enumerate(cfs)
    ]
    if not flat:
        return pd.DataFrame(columns=list(FEASIBILITY_COLUMNS))

    cost_model = evaluator.cost_model
    names = evaluator.feature_names
    sensitive = set(evaluator.fcm.scm.sensitive)
    max_actions = int(cost_model.max_actions)
    rows: list[dict[str, Any]] = []

    for record, rank, cf in flat:
        x = np.asarray(cf.x_original, dtype=float)
        x_cf = np.asarray(cf.x_cf, dtype=float)
        # The plan is what the method asked for; everything else is consequence.
        asked = set(cf.changed_features)
        acted = np.array([n in asked for n in names], dtype=bool)
        plan = cost_model.breakdown(x, np.where(acted, x_cf, x))
        codes = {name: code for name, code in plan.violations}
        immutable = sum(1 for c in codes.values() if c == "immutable")
        direction = sum(1 for c in codes.values() if c == "monotone")
        over_actions = plan.n_changed > max_actions
        over_budget = plan.normalised > 1.0
        admissible = not codes
        moved = ~np.isclose(x, x_cf, rtol=0.0, atol=1e-9)
        plausibility = float(evaluator.plausibility(x_cf.reshape(1, -1),
                                                    int(cf.y_target))[0])
        rows.append({
            "dataset": evaluator.dataset,
            "method": method or cf.method,
            "record": record,
            "rank": rank,
            "valid": bool(evaluator.is_valid(cf)),
            "n_actions": int(plan.n_changed),
            "cost": float(plan.total),
            "immutable": int(immutable),
            "wrong_direction": int(direction),
            "over_actions": bool(over_actions),
            "over_budget": bool(over_budget),
            "admissible": bool(admissible),
            "within_budget": bool(not over_actions and not over_budget),
            # The hard verdict: nothing forbidden, nothing beyond the budget §7
            # priced.  Ordinariness stays out of it — see the module docstring.
            "actionable": bool(admissible and not over_actions and not over_budget),
            "plausibility": plausibility,
            "ordinary": bool(plausibility <= ORDINARY_AT),
            "sensitive_shift": float(evaluator.sensitive_shift(
                x.reshape(1, -1), x_cf.reshape(1, -1))[0]),
            "sensitive_features": ";".join(
                n for n, m in zip(names, moved) if m and n in sensitive),
            "blocked_features": ";".join(f"{n}:{c}" for n, c in codes.items()),
            "actions": ";".join(a.feature for a in plan.actions),
        })
    return pd.DataFrame(rows)[list(FEASIBILITY_COLUMNS)]


#: The per-method actionability table §9 reports.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "dataset", "method", "n_cfs", "n_valid", "actionable", "admissible",
    "within_budget", "immutable_rate", "direction_rate", "over_actions_rate",
    "ordinary", "plausibility_mean", "n_actions_mean", "sensitive_rate",
    "sensitive_shift_mean",
)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, method), over the *valid* counterfactuals.

    Restricting to valid ones keeps the reading honest in both directions: a
    method is not credited for the admissibility of suggestions that do not work,
    and it is not blamed for the inadmissibility of them either.  ``n_valid`` is
    printed so the rates can be weighed, and coverage — how many records got any
    answer at all — stays in :func:`~src.evaluation.metrics.summarise`, which is
    where the query set is the denominator.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    rows: list[dict[str, Any]] = []
    for (dataset, method), group in frame.groupby(["dataset", "method"], sort=True):
        valid = group[group["valid"].astype(bool)]
        n = int(valid.shape[0])
        rows.append({
            "dataset": dataset,
            "method": method,
            "n_cfs": int(group.shape[0]),
            "n_valid": n,
            "actionable": _rate(valid["actionable"], n),
            "admissible": _rate(valid["admissible"], n),
            "within_budget": _rate(valid["within_budget"], n),
            "immutable_rate": _rate(valid["immutable"] > 0, n),
            "direction_rate": _rate(valid["wrong_direction"] > 0, n),
            "over_actions_rate": _rate(valid["over_actions"], n),
            "ordinary": _rate(valid["ordinary"], n),
            "plausibility_mean": _mean(valid["plausibility"]),
            "n_actions_mean": _mean(valid["n_actions"]),
            "sensitive_rate": _rate(valid["sensitive_features"].astype(str) != "", n),
            "sensitive_shift_mean": _mean(valid["sensitive_shift"]),
        })
    return pd.DataFrame(rows)[list(SUMMARY_COLUMNS)]


def blocked_features(frame: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Which rule each method breaks, and on which feature.

    The companion to :func:`~src.evaluation.faithfulness.blamed_features`: that
    one names the features whose *consequences* get ignored, this one the features
    whose *constraints* do.  Together they say what a method's failures are made
    of, which is more useful to a reader than a single actionability rate.
    """
    valid = frame[frame["valid"].astype(bool)]
    rows: list[dict[str, Any]] = []
    for (dataset, method), group in valid.groupby(["dataset", "method"], sort=True):
        counts: dict[tuple[str, str], int] = {}
        for listing in group["blocked_features"]:
            for item in str(listing).split(";"):
                if not item:
                    continue
                name, _, code = item.partition(":")
                counts[(name, code)] = counts.get((name, code), 0) + 1
        total = max(int(group.shape[0]), 1)
        for (name, code), count in sorted(counts.items(), key=lambda kv: -kv[1])[:top]:
            rows.append({"dataset": dataset, "method": method, "feature": name,
                         "rule": code, "n": count, "share": count / total})
    return pd.DataFrame(rows)


#: §9's headline: the primary endpoint and the quantities reported beside it.
HEADLINE_COLUMNS: tuple[str, ...] = (
    "dataset", "method", "n_records", "coverage", "validity", "causal_validity",
    "state_fidelity", "actionability", "usable", "cost_mean", "cost_ratio_median",
    "plausibility_mean", "n_queries_mean",
)

#: Keys that identify one suggestion across all three audit frames.
JOIN_KEYS: tuple[str, ...] = ("dataset", "method", "record", "rank")


def join_audits(metrics: pd.DataFrame, faith: pd.DataFrame, feas: pd.DataFrame,
                columns: bool = False) -> pd.DataFrame:
    """The three audits on one row per suggestion, plus ``answered`` and ``usable``.

    ``usable`` is the primary endpoint the paper leads with, and it is defined
    *here*, once, because every table and figure that claims a method is usable
    has to mean the same thing by it.  A caller that re-derived it from its own
    columns would eventually mean something slightly different and nobody would
    notice.

    The endpoint is **replayed-action success and action feasibility**: a person
    could carry the plan out (``actionable``), and once they do and the SCM fills
    in the consequences the black box still accepts the resulting record
    (``flip_survives``).  Deliberately *not* folded into the gate are two things
    the project reports beside it rather than inside it:

    * *proposed-state validity* (``valid``) — the box flips at the point the
      method named, before the world reacts.  This is weaker than replayed
      success and is kept as its own column, not a precondition, so a proposal
      that only works once consequences are filled in is not silently dropped.
    * *state fidelity* (``gap_max`` ≤ ``CAUSAL_TOLERANCE``) — whether the proposed
      state and the replayed one agree.  A low-fidelity proposal can still be
      usable (its replayed world is accepted and feasible); fidelity measures how
      honestly the method described that world, which is a quality of the
      explanation, not a requirement of the recourse.  ``promise_kept`` keeps the
      strict conjunction (valid *and* survives *and* faithful) for readers who
      want the old all-three bar, but it no longer gates ``usable``.

    Pass ``columns=True`` to keep everything the three frames carry (the named
    off-target features, the rendered actions, the violations) rather than only
    what the funnel needs; that is what the qualitative case studies read.
    """
    if metrics.empty:
        return metrics
    keys = list(JOIN_KEYS)
    extra_faith = ["gap_max", "flip_survives", "promise_kept"]
    extra_feas = ["actionable"]
    if columns:
        extra_faith = [c for c in faith.columns if c not in keys + ["valid"]]
        extra_feas = [c for c in feas.columns
                      if c not in keys + ["valid", "cost"] + list(metrics.columns)]
    joined = (metrics.merge(faith[keys + extra_faith], on=keys, how="left")
                     .merge(feas[keys + extra_feas], on=keys, how="left"))
    joined["answered"] = joined["rank"] >= 0
    for column in ("valid", "causal_valid", "promise_kept", "actionable",
                   "flip_survives"):
        joined[column] = joined[column].fillna(False).astype(bool)
    # Finding 3: the primary endpoint is replayed-action success AND action
    # feasibility.  Proposed-state validity (``valid``) and state fidelity
    # (``promise_kept``/``gap_max``) are reported alongside, never as gates.
    joined["usable"] = joined["flip_survives"] & joined["actionable"]
    return joined


def headline(metrics: pd.DataFrame, faith: pd.DataFrame, feas: pd.DataFrame,
             ratios: pd.DataFrame | None = None,
             reference: str = "oracle") -> pd.DataFrame:
    """Join the three audits into the one table the paper leads with.

    Read left to right it reports four distinct quantities and the endpoint that
    combines two of them (Finding 3):

    * ``validity`` — proposed-state validity: the box flips at the point the
      method named, before the world reacts.
    * ``causal_validity`` — replayed-action success: the person carries out the
      declared action, the SCM fills in the consequences, and the box still
      accepts the resulting record.  This is the definition the manuscript's
      recourse equation states.
    * ``state_fidelity`` — whether the proposed state and the replayed one agree
      (gap ≤ ``CAUSAL_TOLERANCE``).  Reported *beside* the endpoint, not folded
      into it: it grades how honestly a method described its own world.
    * ``actionability`` — action feasibility: a person could carry the plan out
      without breaking §7's immutability, direction, or budget rules.
    * ``usable`` — the primary endpoint: replayed-action success **and** action
      feasibility, per record (one suggestion that does both).

    The funnel is computed per *record*, not per suggestion, because five
    suggestions for one person are one piece of advice with options.

    Everything to the right of ``usable`` describes the plan a person would
    actually take: the cheapest usable suggestion for each record, so ``cost_mean``
    and ``cost_ratio_median`` are averages over the same rows and can be read
    against each other.  Pass ``ratios`` to substitute a differently-filtered
    ratio; the default computes it here, from the usable plans, against the
    reference method's cheapest valid one.
    """
    if metrics.empty:
        return pd.DataFrame(columns=list(HEADLINE_COLUMNS))
    joined = join_audits(metrics, faith, feas)
    tolerance = float(CONFIG.CAUSAL_TOLERANCE)

    rows: list[dict[str, Any]] = []
    for (dataset, method), group in joined.groupby(["dataset", "method"], sort=True):
        records = group.groupby("record")
        best = group[group["usable"]].sort_values("cost", kind="stable") \
            .groupby("record", as_index=False).first()
        rows.append({
            "dataset": dataset,
            "method": method,
            "n_records": int(records.ngroups),
            "coverage": float(records["answered"].any().mean()),
            "validity": float(records["valid"].any().mean()),
            # Replayed-action success — the manuscript's recourse definition.
            "causal_validity": float(records["flip_survives"].any().mean()),
            # State fidelity, reported separately: among valid proposals, how many
            # described a world the SCM actually agrees with.
            "state_fidelity": float(
                group.assign(f=group["valid"]
                             & (group["gap_max"] <= tolerance))
                .groupby("record")["f"].any().mean()),
            # Action feasibility, as its own axis.
            "actionability": float(records["actionable"].any().mean()),
            "usable": float(records["usable"].any().mean()),
            "cost_mean": _mean(best["cost"]),
            "plausibility_mean": _mean(best["plausibility"]),
            "n_queries_mean": _mean(records["n_queries"].max()),
        })
    table = pd.DataFrame(rows)
    if ratios is None:
        from src.evaluation.metrics import cost_ratio  # noqa: PLC0415

        ratios = cost_ratio(joined, reference=reference, only="usable")
    if not ratios.empty:
        table = table.merge(ratios[["dataset", "method", "cost_ratio_median"]],
                            on=["dataset", "method"], how="left")
    else:
        table["cost_ratio_median"] = float("nan")
    return table[list(HEADLINE_COLUMNS)]


def _rate(flags: "pd.Series[Any]", denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(np.asarray(flags, dtype=bool).sum()) / float(denominator)


def _mean(series: "pd.Series[Any]") -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if values.size else float("nan")


__all__ = [
    "FEASIBILITY_COLUMNS",
    "HEADLINE_COLUMNS",
    "JOIN_KEYS",
    "ORDINARY_AT",
    "SUMMARY_COLUMNS",
    "assess",
    "blocked_features",
    "headline",
    "join_audits",
    "summarise",
]
