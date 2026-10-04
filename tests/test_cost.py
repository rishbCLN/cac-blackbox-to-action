"""Tests for the cost layer (§7, global rule #6).

The cost model is hand-crafted domain knowledge, so these tests check the two
things that hand-crafted numbers get wrong: the *algebra* (an identity move
costs nothing, paths compose, the normaliser really does bound the total) and
the *contract* with the other layers (every feature priced, every immutable
feature penalised, the SCM's labels used as weights).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG, DATASETS
from src.cost.cost_functions import (
    CHANGE_ATOL,
    INFINITE_COST,
    ConstantCost,
    CostModel,
    LadderCost,
    LinearCost,
    PiecewiseLinearCost,
    SwitchCost,
    all_cost_models,
    categorical_cost,
    constant_cost,
    get_cost_model,
    linear_cost,
    piecewise_linear,
    total_cost,
)
from src.data.preprocess import dataset_spec
from src.scm.define_scm import get_scm


@pytest.fixture(scope="module")
def models() -> dict[str, CostModel]:
    return all_cost_models()


# ------------------------------------------------------------ building blocks
def test_constant_cost_is_a_step() -> None:
    fn = constant_cost(10.0)
    assert fn(30.0, 30.0) == pytest.approx(0.0)
    assert fn(30.0, 31.0) == pytest.approx(10.0)
    assert fn(31.0, 30.0) == pytest.approx(10.0)  # symmetric: any change is the event
    assert fn.maximum(0.0, 100.0) == pytest.approx(10.0)


def test_linear_cost_direction_is_asymmetric() -> None:
    fn = linear_cost(0.5, up=1.0, down=0.2)
    assert fn(10.0, 12.0) == pytest.approx(1.0)
    assert fn(12.0, 10.0) == pytest.approx(0.2)
    # maximum() must price the dearer direction, or the normaliser under-divides
    assert fn.maximum(0.0, 10.0) == pytest.approx(5.0)


def test_piecewise_linear_matches_the_instruction_example() -> None:
    """§7.1: ``piecewise_linear(0, 40, 60, 99, slopes=[0.01, 0.03, 0.10])``."""
    fn = piecewise_linear(0, 40, 60, 99, slopes=[0.01, 0.03, 0.10])
    assert fn(0.0, 40.0) == pytest.approx(0.40)
    assert fn(40.0, 60.0) == pytest.approx(0.60)
    assert fn(60.0, 99.0) == pytest.approx(3.90)
    assert fn(0.0, 99.0) == pytest.approx(4.90)


def test_piecewise_linear_is_path_independent_and_symmetric() -> None:
    fn = piecewise_linear(0, 40, 60, 99, slopes=[0.01, 0.03, 0.10])
    assert fn(35.0, 70.0) == pytest.approx(fn(35.0, 50.0) + fn(50.0, 70.0))
    assert fn(35.0, 70.0) == pytest.approx(fn(70.0, 35.0))


def test_piecewise_linear_validates_its_schedule() -> None:
    with pytest.raises(ValueError, match="two knots"):
        piecewise_linear(0, slopes=[])
    with pytest.raises(ValueError, match="slopes"):
        piecewise_linear(0, 10, 20, slopes=[0.1])
    with pytest.raises(ValueError, match="increasing"):
        piecewise_linear(0, 20, 10, slopes=[0.1, 0.1])
    with pytest.raises(ValueError, match="non-negative"):
        piecewise_linear(0, 10, slopes=[-0.1])


def test_ladder_pays_every_rung() -> None:
    fn = categorical_cost(levels=list("abcd"), jump_cost=[1.0, 2.0, 4.0], ordered=True)
    assert isinstance(fn, LadderCost)
    assert fn(0, 3) == pytest.approx(7.0)
    assert fn(1, 3) == pytest.approx(6.0)
    assert fn(3, 1) == pytest.approx(6.0)
    assert fn(2, 2) == pytest.approx(0.0)
    # a value between levels is rounded to the nearest level, never interpolated
    assert fn(0.4, 1.0) == pytest.approx(1.0)


def test_switch_cost_prices_arrival_only() -> None:
    fn = categorical_cost(levels=list("abc"), jump_cost=[0.1, 0.5, 2.0])
    assert isinstance(fn, SwitchCost)
    assert fn(0, 2) == pytest.approx(2.0)
    assert fn(1, 2) == pytest.approx(2.0)  # entering 'c' costs the same from anywhere
    assert fn(2, 0) == pytest.approx(0.1)  # asymmetric by construction
    assert fn(1, 1) == pytest.approx(0.0)


def test_categorical_cost_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="rung costs"):
        categorical_cost(levels=list("abcd"), jump_cost=[1.0, 2.0], ordered=True)
    with pytest.raises(ValueError, match="entry costs"):
        categorical_cost(levels=list("abcd"), jump_cost=[1.0, 2.0])


def test_cost_functions_are_vectorised() -> None:
    fn = piecewise_linear(0, 40, 99, slopes=[0.01, 0.05])
    old = np.array([10.0, 40.0, 60.0])
    new = np.array([20.0, 50.0, 60.0])
    got = fn(old, new)
    assert got.shape == (3,)
    assert got == pytest.approx([0.10, 0.50, 0.0])


# ------------------------------------------------------------------ contracts
@pytest.mark.parametrize("dataset", DATASETS)
def test_every_feature_is_priced(dataset: str, models) -> None:
    """A silently unpriced feature would be a free action (rule #10)."""
    model, spec = models[dataset], dataset_spec(dataset)
    assert set(model.functions) == set(spec.feature_names)
    assert model.table()["max_raw"].notna().all()


@pytest.mark.parametrize("dataset", DATASETS)
def test_weights_come_from_the_scm(dataset: str, models) -> None:
    """The graph and the cost model are separate artefacts that must agree."""
    model, scm = models[dataset], get_scm(dataset)
    for j, name in enumerate(model.feature_names):
        assert model.weights[j] == pytest.approx(scm.cost(name))
        assert model.admissible[j] == (name in scm.actionable)


@pytest.mark.parametrize("dataset", DATASETS)
def test_actionable_features_cost_something(dataset: str, models) -> None:
    model = models[dataset]
    assert model.max_raw[model.admissible].min() > 0.0
    assert model.budget > 0.0


# ------------------------------------------------------------------- algebra
@pytest.mark.parametrize("dataset", DATASETS)
def test_identity_costs_nothing(dataset: str, models) -> None:
    """The zero of the cost model — every metric in §9 depends on it."""
    model = models[dataset]
    x = np.array([lo for lo, _ in model.bounds], dtype=float)
    assert model.cost(x, x) == pytest.approx(0.0)
    assert model.raw_cost(x, x) == pytest.approx(0.0)
    assert model.changed_features(x, x) == []
    assert model.breakdown(x, x).actions == ()
    assert model.is_feasible(x, x)


@pytest.mark.parametrize("dataset", DATASETS)
def test_sub_resolution_moves_are_not_changes(dataset: str, models) -> None:
    """A categorical nudged by a third of a level has not changed."""
    model = models[dataset]
    x = np.array([lo for lo, _ in model.bounds], dtype=float)
    y = x.copy()
    discrete = np.where(model.is_discrete)[0]
    y[discrete] += 0.3
    assert model.changed_mask(x, y)[0][discrete].sum() == 0
    assert model.raw_cost(x, y) == pytest.approx(0.0)


def _dearest_move(model: CostModel, j: int) -> tuple[float, float]:
    """The ``(origin, destination)`` pair inside the bounds that costs the most.

    Found by brute force over the endpoints — enough for every shape whose price
    grows with distance — plus every level of a categorical, because a switch's
    dearest level is usually an interior one.  Searching instead of reusing
    ``CostFunction.maximum`` is the point: it checks that claim rather than
    restating it.
    """
    lo, hi = model.bounds[j]
    feat = model.spec.features[j]
    values = {float(lo), float(hi)}
    if feat.levels is not None:
        values |= {float(v) for v in range(feat.n_levels)}
    grid = sorted(v for v in values if lo - CHANGE_ATOL <= v <= hi + CHANGE_ATOL)
    fn = model.functions[model.feature_names[j]]
    pairs = [(a, b) for a in grid for b in grid if a != b]
    return max(pairs, key=lambda ab: float(fn(*ab)))


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_budget_is_the_cost_of_a_plan_that_exists(dataset: str, models) -> None:
    """The dearest permitted plan must normalise to exactly 1, not to 0.86.

    ``maximum()`` prices the dearer *direction*, so the maximal plan does not
    always start at the lower bound — a feature that is cheap to raise and dear
    to lower has to begin at its ceiling.
    """
    model = models[dataset]
    order = np.argsort(np.where(model.admissible, model.weighted_max, -np.inf))[::-1]
    chosen = [int(j) for j in order[: model.max_actions] if model.admissible[j]]
    x = np.array([lo for lo, _ in model.bounds], dtype=float)
    y = x.copy()
    for j in chosen:
        origin, dest = _dearest_move(model, j)
        fn = model.functions[model.feature_names[j]]
        assert float(fn(origin, dest)) == pytest.approx(model.max_raw[j]), (
            f"{model.feature_names[j]}: maximum() over-claims")
        x[j], y[j] = origin, dest
    assert model.normalised_cost(x, y) == pytest.approx(1.0, abs=1e-9)
    assert int(model.batch(x, y)["n_changed"][0]) == len(chosen)


@pytest.mark.parametrize("dataset", DATASETS)
def test_capped_plans_stay_inside_the_unit_interval(dataset: str, models) -> None:
    """The [0, 1] promise, sampled: any plan touching at most MAX_ACTIONS features."""
    model = models[dataset]
    rng = np.random.default_rng(CONFIG.RANDOM_STATE)
    lo, hi = model.bounds[:, 0], model.bounds[:, 1]
    actionable = np.where(model.admissible)[0]
    for _ in range(200):
        x = rng.uniform(lo, hi)
        y = x.copy()
        k = min(int(rng.integers(1, model.max_actions + 1)), actionable.size)
        for j in rng.choice(actionable, size=k, replace=False):
            y[j] = rng.uniform(lo[j], hi[j])
        assert 0.0 <= model.normalised_cost(x, y) <= 1.0 + 1e-9


@pytest.mark.parametrize("dataset", DATASETS)
def test_raw_total_is_the_sum_over_changed_features(dataset: str, models) -> None:
    """§7.2: total cost is the sum over changed features, and nothing else."""
    model = models[dataset]
    rng = np.random.default_rng(CONFIG.RANDOM_STATE)
    x = np.array([lo for lo, _ in model.bounds], dtype=float)
    y = np.array([rng.uniform(lo, hi) for lo, hi in model.bounds], dtype=float)
    breakdown = model.breakdown(x, y)
    assert breakdown.raw == pytest.approx(sum(a.raw for a in breakdown.actions))
    assert breakdown.n_changed == len(breakdown.actions)
    assert set(breakdown.changed) == set(model.changed_features(x, y))


@pytest.mark.parametrize("dataset", DATASETS)
def test_batch_agrees_with_the_scalar_path(dataset: str, models) -> None:
    """CAC evaluates a population at a time; it must not get a different answer."""
    model = models[dataset]
    rng = np.random.default_rng(CONFIG.RANDOM_STATE)
    lo, hi = model.bounds[:, 0], model.bounds[:, 1]
    X0 = rng.uniform(lo, hi, size=(24, len(lo)))
    X1 = rng.uniform(lo, hi, size=(24, len(lo)))
    batch = model.batch(X0, X1)
    assert batch["total"] == pytest.approx([model.cost(a, b) for a, b in zip(X0, X1)])
    assert batch["raw"] == pytest.approx([model.raw_cost(a, b) for a, b in zip(X0, X1)])


@pytest.mark.parametrize("dataset", DATASETS)
def test_batch_broadcasts_one_original_over_many_candidates(dataset: str, models) -> None:
    model = models[dataset]
    rng = np.random.default_rng(CONFIG.RANDOM_STATE)
    lo, hi = model.bounds[:, 0], model.bounds[:, 1]
    x = rng.uniform(lo, hi)
    X = rng.uniform(lo, hi, size=(8, len(lo)))
    assert model.batch(x, X)["total"].shape == (8,)


# ---------------------------------------------------------------- feasibility
@pytest.mark.parametrize("dataset", DATASETS)
def test_immutable_features_draw_the_hard_penalty(dataset: str, models) -> None:
    """§7.2's ``infinite_cost``: touching a non-manipulable feature is not a plan."""
    model = models[dataset]
    fixed = [j for j, ok in enumerate(model.admissible) if not ok]
    assert fixed, f"{dataset} has no immutable feature to test"
    for j in fixed:
        x = np.array([lo for lo, _ in model.bounds], dtype=float)
        y = x.copy()
        y[j] = model.bounds[j, 1] if y[j] != model.bounds[j, 1] else model.bounds[j, 0]
        breakdown = model.breakdown(x, y)
        assert breakdown.penalty >= INFINITE_COST
        assert not breakdown.feasible
        assert breakdown.violations == ((model.feature_names[j], "immutable"),)


@pytest.mark.parametrize("dataset", DATASETS)
def test_monotone_features_reject_the_wrong_direction(dataset: str, models) -> None:
    model = models[dataset]
    for j, direction in enumerate(model.monotone):
        if direction == "any" or not model.admissible[j]:
            continue
        lo, hi = model.bounds[j]
        x = np.array([b for b, _ in model.bounds], dtype=float)
        x[j] = hi if direction == "increase" else lo
        y = x.copy()
        y[j] = lo if direction == "increase" else hi
        breakdown = model.breakdown(x, y)
        assert breakdown.penalty >= INFINITE_COST, model.feature_names[j]
        assert breakdown.violations == ((model.feature_names[j], "monotone"),)


@pytest.mark.parametrize("dataset", DATASETS)
def test_leaving_the_bounds_is_a_violation(dataset: str, models) -> None:
    model = models[dataset]
    j = int(np.argmax(model.admissible))
    x = np.array([lo for lo, _ in model.bounds], dtype=float)
    y = x.copy()
    y[j] = model.bounds[j, 1] + 10.0
    codes = model.violation_codes(x, y)[0]
    assert codes[j] in ("bounds", "monotone")  # both are inadmissible; bounds wins
    assert model.breakdown(x, y).penalty >= INFINITE_COST


def test_penalties_accumulate_per_violating_feature() -> None:
    model = get_cost_model("adult")
    names = model.feature_names
    x = np.array([lo for lo, _ in model.bounds], dtype=float)
    x[names.index("education")] = 8.0
    y = x.copy()
    y[names.index("gender")] = 1.0        # immutable
    y[names.index("age")] = 40.0          # immutable
    y[names.index("education")] = 2.0     # monotone=increase, moved down
    assert model.breakdown(x, y).penalty == pytest.approx(3 * INFINITE_COST)


# ------------------------------------------------------------------- reading
def test_explain_is_human_readable() -> None:
    model = get_cost_model("adult")
    names = model.feature_names
    x = np.array([55, 8, 7, 45, 1, 1, 2, 0, 0], dtype=float)
    y = x.copy()
    y[names.index("education")] = 12      # HS-grad -> Bachelors
    y[names.index("hours_per_week")] = 50
    lines = model.explain(x, y)
    assert len(lines) == 2
    assert any("HS-grad -> Bachelors" in line for line in lines)
    assert any("hours/week" in line for line in lines)
    # education 8 -> 12 costs 0.50 + 0.45 + 0.30 + 0.90 by EDUCATION_RUNGS
    assert model.feature_cost("education", 8, 12) == pytest.approx(2.15)


def test_pretty_decodes_levels_and_units() -> None:
    model = get_cost_model("adult")
    assert model.pretty("education", 12) == "Bachelors"
    assert model.pretty("education", 11.6) == "Bachelors"  # rounds to the level
    assert model.pretty("hours_per_week", 40) == "40 hours/week"
    assert model.pretty("gender", 0) == "Female"


def test_feature_cost_rejects_an_unknown_feature() -> None:
    with pytest.raises(KeyError, match="not a feature"):
        get_cost_model("adult").feature_cost("nonexistent", 0, 1)


def test_total_cost_wrapper_matches_the_model() -> None:
    model = get_cost_model("adult")
    x = np.array([55, 8, 7, 45, 1, 1, 2, 0, 0], dtype=float)
    y = x.copy()
    y[model.feature_names.index("hours_per_week")] = 50
    assert total_cost("adult", x, y) == pytest.approx(model.cost(x, y))
    assert total_cost("adult", x, y, normalised=False) == pytest.approx(
        model.raw_cost(x, y))


# -------------------------------------------------------------------- exports
@pytest.mark.parametrize("dataset", DATASETS)
def test_table_and_dict_round_trip(dataset: str, models) -> None:
    model = models[dataset]
    table = model.table()
    assert list(table["feature"]) == model.feature_names
    assert int(table["in_budget"].sum()) == min(
        model.max_actions, int(model.admissible.sum()))
    payload = model.to_dict()
    assert payload["infinite_cost"] == INFINITE_COST
    assert payload["budget"] == pytest.approx(model.budget)
    assert set(payload["features"]) == set(model.feature_names)


def test_get_cost_model_is_cached() -> None:
    assert get_cost_model("adult") is get_cost_model("adult")


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown dataset"):
        CostModel("nope")


def test_a_dataset_without_prices_is_rejected() -> None:
    """A schema can exist before its cost model does; that must not pass silently."""
    with pytest.raises(KeyError, match="no cost model"):
        CostModel("adult_v2", spec=dataset_spec("adult"), scm=get_scm("adult"))


def test_shape_mismatch_is_rejected() -> None:
    model = get_cost_model("adult")
    with pytest.raises(ValueError, match="columns"):
        model.cost(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="rows"):
        model.batch(np.zeros((2, 9)), np.zeros((3, 9)))


def test_breakdown_refuses_a_batch() -> None:
    model = get_cost_model("adult")
    with pytest.raises(ValueError, match="single row"):
        model.breakdown(np.zeros((2, 9)), np.ones((2, 9)))


def test_change_atol_is_tight_enough_to_be_a_tolerance() -> None:
    """A "tolerance" larger than a rounding error would hide real changes."""
    assert 0 < CHANGE_ATOL < 1e-6


def test_cost_function_base_is_abstract() -> None:
    from src.cost.cost_functions import CostFunction

    with pytest.raises(NotImplementedError):
        CostFunction()(0.0, 1.0)
    with pytest.raises(NotImplementedError):
        CostFunction().maximum(0.0, 1.0)


def test_describe_mentions_the_shape() -> None:
    assert "constant" in ConstantCost(1.0).describe()
    assert "linear" in LinearCost(1.0).describe()
    assert "piecewise" in PiecewiseLinearCost((0.0, 1.0), (1.0,)).describe()
    assert "ladder" in LadderCost((1.0,)).describe()
    assert "switch" in SwitchCost((1.0, 2.0)).describe()


def test_export_writes_nothing_when_asked_not_to(tmp_path) -> None:  # noqa: ARG001
    from src.cost.cost_functions import export_cost_models

    models = export_cost_models(["adult"], write=False)
    assert set(models) == {"adult"}
    assert isinstance(models["adult"].table(), pd.DataFrame)
