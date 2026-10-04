"""Tests for the evaluation layer (§9, global rule #6).

The evaluation layer is the referee, so these tests are mostly about whether it
can be fooled.  Three things have to hold or none of §9's tables mean anything:

*It prices what the method asked for.*  A suggestion that declares one action and
moves two coordinates must be charged for the action and credited with the
consequence — separately, and in the right columns.

*It counts the records nobody answered.*  Every rate has the query set as its
denominator, so a method cannot improve its numbers by declining to answer.

*It recomputes rather than believes.*  The audit derives cost and causal gap from
the shared artefacts; :func:`~src.evaluation.metrics.disagreements` exists to
prove it, so there is a test that hands it a lie and checks that it notices.

Most counterfactuals here are *forged* rather than generated.  A real method only
ever emits rows it already agrees with, so everything the auditor is supposed to
catch has to be built by hand; the one test that runs real searches is marked
``slow`` and does the integration check.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG
from src.evaluation import faithfulness, feasibility, metrics
from src.methods.base import Counterfactual

#: The smallest of the three datasets, so a module-scoped evaluator is cheap.
DATASET = "german"

@pytest.fixture(scope="module")
def ev() -> metrics.Evaluator:
    return metrics.Evaluator.build(DATASET)


@pytest.fixture(scope="module")
def records(ev: metrics.Evaluator) -> np.ndarray:
    return metrics.query_instances(DATASET, model=ev.model, bundle=ev.bundle, n=4)


@pytest.fixture(scope="module")
def x(ev: metrics.Evaluator, records: np.ndarray) -> np.ndarray:
    return ev.bundle.X_test[records[0]].astype(float).copy()


def _forge(ev: metrics.Evaluator, x: np.ndarray, x_cf: np.ndarray,
           changed: Sequence[str] | None = None, method: str = "forged",
           y_target: int = 1, cost: float = float("nan"),
           **extras: Any) -> Counterfactual:
    """A counterfactual saying whatever the caller wants it to say.

    ``changed`` defaults to the coordinates that actually moved, which is what
    every baseline reports; passing it explicitly is how a test builds the
    interesting cases — a plan that declares less than it moves, or more.
    """
    x = np.asarray(x, dtype=float)
    x_cf = np.asarray(x_cf, dtype=float)
    moved = ~np.isclose(x, x_cf, rtol=0.0, atol=1e-9)
    names = (tuple(n for n, m in zip(ev.feature_names, moved) if m)
             if changed is None else tuple(changed))
    return Counterfactual(
        x_original=x, x_cf=x_cf, y_original=1 - int(y_target),
        y_target=int(y_target), y_predicted=int(y_target), cost=float(cost),
        changed_features=names, causal_path=(), method=method,
        dataset=ev.dataset, feature_names=ev.feature_names, extras=dict(extras))


def _mask(ev: metrics.Evaluator, names: Sequence[str]) -> np.ndarray:
    wanted = set(names)
    return np.array([n in wanted for n in ev.feature_names], dtype=bool)


def _step(ev: metrics.Evaluator, x: np.ndarray, j: int) -> float:
    """A value for feature ``j`` that §7 would let this person move to.

    Borrowed from an accepted training row rather than computed, so it is inside
    the feature box and on the discrete grid, and filtered by the feature's
    monotone direction so the plan does not accidentally break a *different* rule
    than the one under test.
    """
    direction = ev.cost_model.monotone[j]
    for row in ev.bundle.X_train[ev.bundle.y_train == 1]:
        delta = float(row[j]) - float(x[j])
        if abs(delta) <= 1e-6:
            continue
        if direction == "increase" and delta < 0:
            continue
        if direction == "decrease" and delta > 0:
            continue
        return float(row[j])
    raise AssertionError(f"no admissible move found for {ev.feature_names[j]!r}")


# ------------------------------------------------------------------ query set
def test_query_instances_are_the_rejected_records_in_index_order(
        ev: metrics.Evaluator) -> None:
    idx = metrics.query_instances(DATASET, model=ev.model, bundle=ev.bundle, n=6)
    assert idx.size and idx.size <= 6
    assert list(idx) == sorted(idx), "sampling would make the set unreproducible"
    predicted = ev.model.predict(ev.bundle.X_test[idx])
    assert set(np.unique(predicted)) == {0}, "recourse is for the rejected"


def test_query_instances_is_deterministic(ev: metrics.Evaluator) -> None:
    first = metrics.query_instances(DATASET, model=ev.model, bundle=ev.bundle, n=6)
    second = metrics.query_instances(DATASET, model=ev.model, bundle=ev.bundle, n=6)
    assert np.array_equal(first, second)


# ----------------------------------------------------------- action vs. state
def test_row_prices_the_declared_plan_not_the_whole_move(
        ev: metrics.Evaluator, x: np.ndarray) -> None:
    """The rule the whole comparison rests on: a person pays for what they do."""
    j, k = [int(i) for i in np.flatnonzero(ev.cost_model.admissible)][:2]
    x_cf = x.copy()
    x_cf[j], x_cf[k] = _step(ev, x, j), _step(ev, x, k)
    declared = (ev.feature_names[j],)

    row = ev.row(_forge(ev, x, x_cf, changed=declared), record=0, rank=0)
    action_only = np.where(_mask(ev, declared), x_cf, x)
    assert row["cost"] == pytest.approx(
        ev.cost_model.breakdown(x, action_only).total)
    assert row["n_changed"] == 1, "the plan is one action"
    assert row["l0"] == 2, "the move is still two coordinates wide"
    # And the consequence is not lost, just kept in its own column.
    assert row["state_cost"] == pytest.approx(
        ev.cost_model.as_effort(x, x_cf)[0].total)
    assert row["state_cost"] >= row["cost"]


def test_row_recomputes_the_causal_gap_from_the_scm(
        ev: metrics.Evaluator, x: np.ndarray) -> None:
    j = int(np.flatnonzero(ev.cost_model.admissible)[0])
    raw = x.copy()
    raw[j] = _step(ev, x, j)
    row = ev.row(_forge(ev, x, raw), record=0, rank=0)
    assert row["causal_gap"] == pytest.approx(
        float(ev.fcm.violation(x.reshape(1, -1), raw.reshape(1, -1)).max()))

    # A row the SCM itself produced is causally exact by construction, which is
    # the only reason CAC's zero is a meaningful number rather than a tolerance.
    repaired = ev.fcm.repair(x.reshape(1, -1), raw.reshape(1, -1))[0]
    exact = ev.row(_forge(ev, x, repaired, changed=(ev.feature_names[j],)),
                   record=0, rank=0)
    assert exact["causal_gap"] == pytest.approx(0.0, abs=1e-9)
    assert exact["causal_valid"] is True

# --------------------------------------------------------------- denominators
def test_table_keeps_the_records_nobody_answered(
        ev: metrics.Evaluator, x: np.ndarray) -> None:
    frame = ev.table({0: [], 1: [_forge(ev, x, x + 0.0)]}, method="forged")
    assert frame.shape[0] == 2, "a record without an answer is still a record"
    empty = frame[frame["record"] == 0].iloc[0]
    assert empty["rank"] == -1
    assert bool(empty["valid"]) is False
    assert bool(empty["causal_valid"]) is False
    assert pd.isna(empty["cost"]), "a zero here would read as free recourse"


def test_summarise_divides_by_the_query_set(ev: metrics.Evaluator,
                                            x: np.ndarray) -> None:
    donor = ev.bundle.X_train[ev.bundle.y_train == 1][0].astype(float)
    frame = ev.table({0: [], 1: [_forge(ev, x, donor)]}, method="forged")
    summary = metrics.summarise(frame)
    assert summary.loc[0, "n_records"] == 2
    assert summary.loc[0, "coverage"] == pytest.approx(0.5)
    # Whether that one answer is valid is the model's business; what matters here
    # is that the record it never answered counts against it either way.
    assert summary.loc[0, "validity"] <= 0.5


def test_cost_ratio_labels_its_subset_and_keeps_the_reference_fixed() -> None:
    frame = pd.DataFrame([
        # record 0: the oracle's cheapest valid plan is 0.1, but the plan a
        # person could use costs 0.4 — the premium usability charges.
        {"dataset": "d", "method": "oracle", "record": 0, "rank": 0,
         "cost": 0.1, "valid": True, "usable": False},
        {"dataset": "d", "method": "oracle", "record": 0, "rank": 1,
         "cost": 0.4, "valid": True, "usable": True},
        {"dataset": "d", "method": "cac", "record": 0, "rank": 0,
         "cost": 0.2, "valid": True, "usable": True},
        {"dataset": "d", "method": "cac", "record": 1, "rank": 0,
         "cost": 9.9, "valid": True, "usable": True},
    ])
    over_valid = metrics.cost_ratio(frame).set_index("method")
    assert set(over_valid["subset"]) == {"valid"}
    assert over_valid.loc["oracle", "cost_ratio_median"] == pytest.approx(1.0)
    assert over_valid.loc["cac", "cost_ratio_median"] == pytest.approx(2.0)
    # Record 1 has no oracle plan, so it is reported missing, never imputed at 1.
    assert over_valid.loc["cac", "n_ratio"] == 1
    assert over_valid.loc["cac", "n_missing"] == 1

    over_usable = metrics.cost_ratio(frame, only="usable").set_index("method")
    assert set(over_usable["subset"]) == {"usable"}
    assert over_usable.loc["oracle", "cost_ratio_median"] == pytest.approx(4.0), \
        "the reference must stay the cheapest valid plan, filter or no filter"
    assert over_usable.loc["cac", "cost_ratio_median"] == pytest.approx(2.0)

# ------------------------------------------------------------------ diversity
def test_diversity_separates_routes_from_points(ev: metrics.Evaluator,
                                                x: np.ndarray) -> None:
    """Five amounts of the same thing are one piece of advice, not five."""
    accepted = ev.bundle.X_train[ev.bundle.y_train == 1]
    keep = [row for row in accepted
            if ev.model.predict(row.reshape(1, -1))[0] == 1][:2]
    assert len(keep) == 2, "need two rows the black box accepts"
    names = ev.feature_names
    j, k = [int(i) for i in np.flatnonzero(ev.cost_model.admissible)][:2]

    same_route = [_forge(ev, x, row, changed=(names[j],)) for row in keep]
    two_routes = [_forge(ev, x, keep[0], changed=(names[j],)),
                  _forge(ev, x, keep[1], changed=(names[k],))]
    one = metrics.diversity(ev, {0: same_route}, method="forged").iloc[0]
    two = metrics.diversity(ev, {0: two_routes}, method="forged").iloc[0]

    assert one["n_valid"] == 2 and two["n_valid"] == 2
    assert one["routes"] == 1 and two["routes"] == 2
    # Same points, so the point-diversity number cannot tell the two apart.
    assert one["spread"] == pytest.approx(two["spread"])
    assert one["spread"] > 0.0


def test_diversity_ignores_suggestions_that_do_not_work(
        ev: metrics.Evaluator, x: np.ndarray) -> None:
    identical = [_forge(ev, x, x + 0.0), _forge(ev, x, x + 0.0)]
    row = metrics.diversity(ev, {0: identical}, method="forged").iloc[0]
    assert row["n_valid"] == 0, "x itself is rejected; nothing here is valid"
    assert row["spread"] == pytest.approx(0.0)
    assert row["routes"] == 0

# --------------------------------------------------------------- faithfulness
def _with_children(ev: metrics.Evaluator) -> int:
    """Index of the admissible feature the graph gives the most consequences."""
    scm = ev.fcm.scm
    scored = [(len(scm.descendants(n)), j) for j, n in enumerate(ev.feature_names)
              if ev.cost_model.admissible[j]]
    best = max(scored)
    assert best[0] > 0, "an SCM where nothing has consequences proves nothing"
    return int(best[1])


def test_faithfulness_names_the_features_the_graph_contradicts(
        ev: metrics.Evaluator, x: np.ndarray) -> None:
    j = _with_children(ev)
    proposed = x.copy()
    proposed[j] = _step(ev, x, j)  # moved, every consequence left unpropagated
    frame = faithfulness.assess(ev, {0: [_forge(ev, x, proposed)]}, method="forged")
    row = frame.iloc[0]
    assert row["gap_max"] > 0.0
    assert row["n_off"] >= 1
    assert row["worst_feature"] in ev.feature_names
    assert bool(row["promise_kept"]) is False

    repaired = ev.fcm.repair(x.reshape(1, -1), proposed.reshape(1, -1))[0]
    exact = faithfulness.assess(
        ev, {0: [_forge(ev, x, repaired, changed=(ev.feature_names[j],))]},
        method="forged").iloc[0]
    assert exact["gap_max"] == pytest.approx(0.0, abs=1e-9)
    assert exact["n_off"] == 0
    # promise_kept still needs the flip; the point here is only that the state
    # the SCM produced is not the thing standing in the way.
    assert bool(exact["promise_kept"]) == bool(exact["valid"])


def test_faithfulness_summary_reports_its_own_denominator(
        ev: metrics.Evaluator, x: np.ndarray) -> None:
    frame = faithfulness.assess(ev, {0: [_forge(ev, x, x + 0.0)]}, method="forged")
    summary = faithfulness.summarise(frame)
    assert summary.loc[0, "n_cfs"] == 1
    assert summary.loc[0, "n_valid"] == 0
    # Nothing valid to be wrong about, so every rate is NaN rather than 1.0.
    assert np.isnan(summary.loc[0, "flip_survival"])
    assert np.isnan(summary.loc[0, "causal_validity"])

# ----------------------------------------------------------------- feasibility
def test_feasibility_refuses_a_plan_that_acts_on_an_immutable_feature(
        ev: metrics.Evaluator, x: np.ndarray) -> None:
    blocked = np.flatnonzero(~ev.cost_model.admissible)
    if blocked.size == 0:  # pragma: no cover - every dataset here has one
        pytest.skip("this dataset marks nothing immutable")
    j = int(blocked[0])
    proposed = x.copy()
    proposed[j] = x[j] + 1.0
    row = feasibility.assess(ev, {0: [_forge(ev, x, proposed)]},
                             method="forged").iloc[0]
    assert row["immutable"] >= 1
    assert bool(row["admissible"]) is False
    assert bool(row["actionable"]) is False
    assert ev.feature_names[j] in str(row["blocked_features"])


def test_feasibility_refuses_a_plan_over_the_action_budget(
        ev: metrics.Evaluator, x: np.ndarray) -> None:
    budget = int(ev.cost_model.max_actions)
    usable = [int(i) for i in np.flatnonzero(ev.cost_model.admissible)]
    assert len(usable) > budget, "the budget has to be reachable to be tested"
    picked = usable[:budget + 1]
    proposed = x.copy()
    for j in picked:
        proposed[j] = _step(ev, x, j)
    row = feasibility.assess(ev, {0: [_forge(ev, x, proposed)]},
                             method="forged").iloc[0]
    assert row["n_actions"] == budget + 1
    assert bool(row["over_actions"]) is True
    assert bool(row["actionable"]) is False
    # Admissibility and budget are different objections and stay separate.
    assert bool(row["admissible"]) is True


def test_feasibility_judges_the_plan_not_the_consequences(
        ev: metrics.Evaluator, x: np.ndarray) -> None:
    """A record that moved because the graph said so is not a request.

    The mirror image of :func:`test_row_prices_the_declared_plan_not_the_whole_move`:
    an immutable coordinate that the SCM shifted must not be read as somebody
    asking to change it, or CAC — the only method that propagates consequences at
    all — would be the only one punished for doing so.
    """
    blocked = np.flatnonzero(~ev.cost_model.admissible)
    if blocked.size == 0:  # pragma: no cover
        pytest.skip("this dataset marks nothing immutable")
    j, m = int(blocked[0]), int(np.flatnonzero(ev.cost_model.admissible)[0])
    proposed = x.copy()
    proposed[m] = _step(ev, x, m)
    proposed[j] = x[j] + 1.0  # the consequence, undeclared
    row = feasibility.assess(
        ev, {0: [_forge(ev, x, proposed, changed=(ev.feature_names[m],))]},
        method="forged").iloc[0]
    assert bool(row["admissible"]) is True
    assert bool(row["actionable"]) is True
    assert row["immutable"] == 0

# -------------------------------------------------------------------- headline
def _funnel_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One record, two suggestions, each passing a different test.

    The case the funnel exists to catch: rank 0 flips the decision but nobody
    could carry it out, rank 1 is easy to carry out but does not flip anything.
    Read per record with an ``any()`` per column, this method looks like it does
    both; read as a conjunction per *suggestion*, it offers nothing.
    """
    keys = {"dataset": "d", "method": "m", "record": 0}
    metrics_frame = pd.DataFrame([
        {**keys, "rank": 0, "valid": True, "causal_valid": True, "cost": 0.2,
         "plausibility": 1.0, "n_queries": 10},
        {**keys, "rank": 1, "valid": False, "causal_valid": False, "cost": 0.1,
         "plausibility": 1.0, "n_queries": 10},
    ])
    faith = pd.DataFrame([
        {**keys, "rank": 0, "gap_max": 0.0, "flip_survives": True,
         "promise_kept": True},
        {**keys, "rank": 1, "gap_max": 0.0, "flip_survives": False,
         "promise_kept": False},
    ])
    feas = pd.DataFrame([
        {**keys, "rank": 0, "actionable": False},
        {**keys, "rank": 1, "actionable": True},
    ])
    return metrics_frame, faith, feas


def test_headline_requires_one_suggestion_to_pass_everything() -> None:
    table = feasibility.headline(*_funnel_frames())
    row = table.iloc[0]
    assert row["n_records"] == 1
    assert row["coverage"] == pytest.approx(1.0)
    assert row["validity"] == pytest.approx(1.0)
    assert row["causal_validity"] == pytest.approx(1.0)
    # Rank 1 is actionable but invalid, so no *valid* suggestion is actionable.
    assert row["actionability"] == pytest.approx(0.0)
    assert row["usable"] == pytest.approx(0.0)
    assert np.isnan(row["cost_mean"]), "there is no usable plan to price"


def test_headline_funnel_never_widens() -> None:
    metrics_frame, faith, feas = _funnel_frames()
    feas.loc[feas["rank"] == 0, "actionable"] = True  # now rank 0 passes all three
    row = feasibility.headline(metrics_frame, faith, feas).iloc[0]
    assert row["usable"] == pytest.approx(1.0)
    assert row["usable"] <= row["actionability"] + 1e-12
    assert row["actionability"] <= row["validity"] + 1e-12
    assert row["validity"] <= row["coverage"] + 1e-12
    assert row["cost_mean"] == pytest.approx(0.2), "the usable plan, not the cheap one"


def test_join_audits_defines_usable_once_for_everybody() -> None:
    """``usable`` is a conjunction, and it is this module's to define.

    Every table and figure that calls a method usable has to mean the same thing
    by it.  A caller that re-derived the conjunction from its own three columns
    would eventually mean something slightly different — and because each of the
    three is true of *some* suggestion here, nothing in the output would show it.
    """
    joined = feasibility.join_audits(*_funnel_frames())
    assert list(joined["usable"]) == [False, False]
    assert list(joined["answered"]) == [True, True]
    expected = joined["valid"] & joined["promise_kept"] & joined["actionable"]
    assert list(joined["usable"]) == list(expected)
    for column in ("valid", "promise_kept", "actionable", "flip_survives"):
        assert joined[column].dtype == bool, f"{column} must be a verdict, not a NaN"


def test_join_audits_keeps_the_narrative_columns_on_request() -> None:
    """``columns=True`` is what the case studies read, and it must add, not replace.

    A narrative needs the named off-target features and the rendered actions; the
    funnel needs neither.  The wide form has to be a superset of the narrow one,
    or experiment 05 would be selecting cases on one set of numbers and printing
    another.
    """
    metrics_frame, faith, feas = _funnel_frames()
    faith["off_features"] = ["", "savings_status"]
    feas["blocked_features"] = ["age:immutable", ""]
    feas["actions"] = ["age", "savings_status"]
    narrow = feasibility.join_audits(metrics_frame, faith, feas)
    wide = feasibility.join_audits(metrics_frame, faith, feas, columns=True)
    assert set(narrow.columns) <= set(wide.columns)
    assert list(wide["usable"]) == list(narrow["usable"])
    assert list(wide["off_features"]) == ["", "savings_status"]
    assert list(wide["blocked_features"]) == ["age:immutable", ""]
    assert list(wide["actions"]) == ["age", "savings_status"]
    assert list(wide["cost"]) == list(metrics_frame["cost"]), \
        "cost belongs to the metrics frame; the feasibility copy must not shadow it"

# ----------------------------------------------------------------------- audit
def test_disagreements_catches_a_method_understating_its_cost(
        ev: metrics.Evaluator, x: np.ndarray) -> None:
    j = int(np.flatnonzero(ev.cost_model.admissible)[0])
    proposed = x.copy()
    proposed[j] = _step(ev, x, j)
    liar = _forge(ev, x, proposed, method="liar", cost=0.0)
    frame = ev.table({0: [liar]}, method="liar")
    found = metrics.disagreements(frame, {"liar": {0: [liar]}})
    assert not found.empty and set(found["field"]) == {"cost"}
    assert found.iloc[0]["reported"] == pytest.approx(0.0)
    assert found.iloc[0]["recomputed"] > 0.0

    honest = _forge(ev, x, proposed, method="honest",
                    cost=float(ev.cost_model.breakdown(x, proposed).total))
    honest_frame = ev.table({0: [honest]}, method="honest")
    assert metrics.disagreements(honest_frame, {"honest": {0: [honest]}}).empty


@pytest.mark.slow
def test_real_methods_survive_their_own_audit(ev: metrics.Evaluator,
                                              records: np.ndarray) -> None:
    """The integration check: two real searches, scored from the outside.

    Deliberately the only test here that runs a method.  It asserts the three
    claims §9 makes about the pipeline rather than about any one method — the
    oracle is the cost reference, CAC's causal gap is zero because it searched in
    intervention space, and neither of them reports a number the audit cannot
    reproduce.
    """
    from src.methods.base import get_method  # noqa: PLC0415

    idx = records[:2]
    X = ev.bundle.X_test[idx]
    results: dict[str, dict[int, list[Counterfactual]]] = {}
    frames = []
    for name in ("oracle", "cac"):
        method = get_method(name, DATASET, model=ev.model)
        per_record = {int(r): method.generate(X[i], y_target=1, k=2)
                      for i, r in enumerate(idx)}
        results[name] = per_record
        frames.append(ev.table(per_record, method=name))
    frame = pd.concat(frames, ignore_index=True)

    assert metrics.disagreements(frame, results).empty
    cac = frame[(frame["method"] == "cac") & frame["valid"].astype(bool)]
    assert not cac.empty
    assert (cac["causal_gap"] <= float(CONFIG.CAUSAL_TOLERANCE)).all()
    assert cac["causal_gap"].max() == pytest.approx(0.0, abs=1e-6)

    ratio = metrics.cost_ratio(frame).set_index("method")
    assert ratio.loc["oracle", "cost_ratio_median"] == pytest.approx(1.0)
    faith = faithfulness.assess(ev, results["cac"], method="cac")
    assert bool(faith["flip_survives"].all())
    assert bool(faith["promise_kept"].all())
