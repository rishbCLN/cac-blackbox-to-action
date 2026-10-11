"""Do the methods keep §8's contract, and does each keep its own promise?

Two kinds of test live here, and the difference matters.  The first kind is the
contract every method signs by subclassing
:class:`~src.methods.base.CounterfactualMethod`: nine fields, at most ``k``
suggestions, no duplicates, ranked, on the grid, inside the box, and never a
plan that asks somebody to change something §7 says nobody can change.  Those
are the same assertions for all six methods, because the point of the base
class is that §9 can compare them at all.

The second kind is each method's own claim.  The oracle says its answer is
optimal and proves it by enumeration, so its cost has to be reproducible from
the cost model alone.  CAC says it thinks in interventions, so the record it
proposes has to be one the SCM would actually produce and its cost has to price
the *action* rather than the consequences.  The baselines claim neither, and
:func:`test_baselines_ignore_the_graph` checks that this shows up as a measured
disagreement rather than as an adjective in the paper — it is the experiment's
central claim, reduced to one record and one assertion.

Nothing here is forged: unlike ``test_metrics``, whose job is to catch a referee
that can be fooled, this module runs the real searches on a real rejected
applicant.  One record, ``k=3``, German (the smallest dataset), all six methods
generated once in a module-scoped fixture — a few seconds in total, marked
``slow`` so it can be deselected, and the cheap contract checks that need no
search are left unmarked.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from src.config import CONFIG
from src.data.preprocess import load_dataset
from src.evaluation.metrics import Evaluator, query_instances
from src.methods.base import (MOVE_ATOL, Counterfactual, available_methods,
                              get_method)
from src.methods.cac import CACMethod
from src.model.blackbox import BlackBoxOpacityError, load_blackbox

DATASET = "german"
K = 3
#: The methods that do not consult the graph — §9's comparison group.
BASELINES: tuple[str, ...] = ("random", "wachter", "dice")
#: The methods that score through the SCM, so the base class fills their
#: ``causal_path``: CAC and the MINT causal baseline.
CAUSAL: tuple[str, ...] = ("cac", "mint")


@pytest.fixture(scope="module")
def bundle() -> Any:
    return load_dataset(DATASET)


@pytest.fixture(scope="module")
def model(bundle: Any) -> Any:
    return load_blackbox(DATASET, CONFIG.BLACKBOX_TYPE)


@pytest.fixture(scope="module")
def ev(bundle: Any, model: Any) -> Evaluator:
    """The audit, built from the same artefacts the methods get."""
    return Evaluator.build(DATASET, bundle=bundle, model=model)


@pytest.fixture(scope="module")
def x(bundle: Any, model: Any) -> np.ndarray:
    """One applicant the black box rejects — the only case recourse is about."""
    records = query_instances(DATASET, model=model, bundle=bundle, n=1, y_target=1)
    assert records.size == 1, "the black box rejects nobody; nothing to explain"
    return np.asarray(bundle.X_test[int(records[0])], dtype=float)


@pytest.fixture(scope="module")
def plans(ev: Evaluator, x: np.ndarray) -> tuple[dict[str, list[Counterfactual]],
                                                 dict[str, Exception]]:
    """Every method's answer for that one applicant, generated once.

    A method that raises is recorded rather than allowed to error out of every
    test that touches the fixture: ``test_no_method_raises`` reports it once,
    with the exception, and the rest of the module can still say something about
    the methods that did answer.
    """
    out: dict[str, list[Counterfactual]] = {}
    errors: dict[str, Exception] = {}
    for name in available_methods():
        try:
            method = get_method(name, DATASET, model=ev.model)
            out[name] = method.generate(x, y_target=1, k=K)
        except Exception as exc:  # noqa: BLE001 - reported by a test, not swallowed
            out[name], errors[name] = [], exc
    return out, errors


@pytest.fixture(scope="module")
def cfs(plans: Any) -> Mapping[str, Sequence[Counterfactual]]:
    return plans[0]


# --------------------------------------------------------------- the registry


def test_registry_offers_exactly_the_six_methods() -> None:
    """§9's table has six rows, and they come from here rather than a list."""
    assert set(available_methods()) == {"random", "wachter", "dice", "oracle",
                                        "mint", "cac"}


def test_get_method_builds_what_it_was_asked_for(model: Any) -> None:
    for name in available_methods():
        method = get_method(name, DATASET, model=model)
        assert method.name == name
        assert method.dataset == DATASET
        # Same black box, so §9's query counts are differences on one counter.
        assert method.model is model


def test_unknown_method_is_a_keyerror(model: Any) -> None:
    with pytest.raises(KeyError, match="unknown method"):
        get_method("gradient_descent_on_the_weights", DATASET, model=model)


# ------------------------------------------------------- rule #8: opacity


def test_the_black_box_refuses_to_be_read(model: Any) -> None:
    """Global rule #8, enforced rather than promised.

    Every one of these attributes is how a real explainer would cheat: XGBoost's
    booster, sklearn's coefficients, the fitted tree, the number of features it
    was trained on.  The guard raises for all of them, so a method that tries
    cannot silently succeed on one model type and fail on another.
    """
    for attribute in ("feature_importances_", "coef_", "get_booster", "tree_",
                      "estimators_", "decision_function", "get_params",
                      "n_features_in_", "coefs_"):
        with pytest.raises(BlackBoxOpacityError, match="rule #8"):
            getattr(model, attribute)
    # An honest typo still reads as a typo, not as a policy violation.
    with pytest.raises(AttributeError, match="has no attribute"):
        model.predict_probaa  # noqa: B018


class _SealedModel:
    """The black box with the lid welded shut.

    The real wrapper raises on :data:`~src.model.blackbox._FORBIDDEN`, but a
    determined method could still reach past it for ``_estimator`` — the actual
    fitted booster — and read anything it liked.  This proxy forwards the four
    names §8 allows a search to use and treats every other access as a failed
    test, so "no method peeks" becomes something a search either survives or
    does not.
    """

    _ALLOWED = frozenset({"predict", "predict_proba", "proba_positive",
                          "query_count"})

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, item: str) -> Any:
        inner = object.__getattribute__(self, "_inner")
        if item in _SealedModel._ALLOWED:
            return getattr(inner, item)
        raise AssertionError(f"the search reached for {item!r} on the black box "
                             "(global rule #8: predict / predict_proba only)")


@pytest.mark.slow
def test_a_search_survives_a_sealed_black_box(model: Any, x: np.ndarray) -> None:
    """Run the real searches against a model that cannot be inspected at all.

    ``cac`` is included because it is the method being argued for: if the
    contribution needed the gradients rule #8 forbids, the paper's premise would
    be false, and this is where that would show.
    """
    sealed = _SealedModel(model)
    for name in ("random", "wachter", "cac"):
        found = get_method(name, DATASET, model=sealed).generate(x, y_target=1, k=1)
        assert found, f"{name} produced nothing through the sealed model"


# ------------------------------------------------------------ what §8 refuses


def test_generate_refuses_the_answer_the_model_already_gives(model: Any,
                                                             x: np.ndarray) -> None:
    """Asking for the decision you already have is not recourse, it is identity."""
    method = get_method("random", DATASET, model=model)
    assert int(method.predict(x)[0]) == 0
    with pytest.raises(ValueError, match="already the model's answer"):
        method.generate(x, y_target=0)


def test_generate_refuses_a_nonsense_request(model: Any, x: np.ndarray) -> None:
    method = get_method("random", DATASET, model=model)
    with pytest.raises(ValueError, match="k must be positive"):
        method.generate(x, y_target=1, k=0)
    with pytest.raises(ValueError, match="features, expected"):
        method.generate(x[:-1], y_target=1, k=1)


# ------------------------------------------------- the contract all six sign


@pytest.mark.slow
def test_no_method_raises(plans: Any) -> None:
    """    Reported once, with the exception, rather than as six broken tests.

    §9 scores a crash as zero coverage on purpose — a person who gets an error
    got no recourse — but in the test suite a crash is a regression, so it fails
    here and nowhere else.
    """
    _, errors = plans
    assert not errors, "; ".join(f"{name}: {type(exc).__name__}: {exc}"
                                 for name, exc in errors.items())


@pytest.mark.slow
def test_every_method_answers_with_at_most_k_ranked_unique_plans(
        cfs: Mapping[str, Sequence[Counterfactual]]) -> None:
    for name, found in cfs.items():
        assert 0 < len(found) <= K, f"{name} returned {len(found)} plans for k={K}"
        keys = [cf.key() for cf in found]
        assert len(set(keys)) == len(keys), f"{name} returned the same plan twice"
        ranks = [cf.rank() for cf in found]
        assert ranks == sorted(ranks), f"{name} returned its plans out of order"


@pytest.mark.slow
def test_every_plan_fills_in_the_nine_fields(cfs: Mapping[str, Sequence[Counterfactual]],
                                             x: np.ndarray, ev: Evaluator) -> None:
    """§8.1's dataclass, field by field, on every plan every method produced."""
    for name, found in cfs.items():
        for cf in found:
            assert cf.method == name
            assert cf.dataset == DATASET
            assert cf.feature_names == ev.feature_names
            assert np.allclose(cf.x_original, x)
            assert cf.x_cf.shape == x.shape
            assert (cf.y_original, cf.y_target) == (0, 1)
            assert cf.y_predicted in (0, 1)
            assert cf.valid == (cf.y_predicted == cf.y_target)
            assert np.isfinite(cf.cost) and cf.cost >= 0.0
            assert set(cf.changed_features) <= set(ev.feature_names)
            # Both stamps describe the call, so they are shared by the list.
            assert cf.n_queries > 0 and cf.runtime_s > 0.0
            assert 0.0 <= cf.proba_target <= 1.0


@pytest.mark.slow
def test_every_plan_lands_on_the_grid_and_inside_the_box(
        cfs: Mapping[str, Sequence[Counterfactual]], ev: Evaluator) -> None:
    """A recommendation of 12.7 years of education is not a recommendation.

    Snapping happens in :meth:`~src.methods.base.CounterfactualMethod.audit`, so
    it holds for every method whatever its search does internally — which is the
    reason it lives there and not in each search.
    """
    lo, hi = ev.cost_model.bounds[:, 0], ev.cost_model.bounds[:, 1]
    discrete = ev.cost_model.is_discrete
    for name, found in cfs.items():
        for cf in found:
            assert np.all(cf.x_cf >= lo - 1e-9), f"{name} left the box below"
            assert np.all(cf.x_cf <= hi + 1e-9), f"{name} left the box above"
            fractional = np.abs(cf.x_cf[discrete] - np.rint(cf.x_cf[discrete]))
            assert np.all(fractional <= 1e-9), f"{name} left a discrete feature off-grid"


@pytest.mark.slow
def test_no_plan_asks_for_something_nobody_can_do(
        cfs: Mapping[str, Sequence[Counterfactual]], ev: Evaluator) -> None:
    """§7's hardest rule, on the plans rather than on the rates.

    ``BASELINES_RESPECT_IMMUTABLE`` is on, so this holds for the baselines too —
    deliberately, because a baseline left free to make the applicant younger is
    a strawman rather than a comparison.
    """
    assert CONFIG.BASELINES_RESPECT_IMMUTABLE
    forbidden = {n for n, ok in zip(ev.feature_names, ev.cost_model.admissible)
                 if not ok}
    assert forbidden, "german declares nothing immutable; the test proves nothing"
    for name, found in cfs.items():
        for cf in found:
            asked = set(cf.changed_features) & forbidden
            assert not asked, f"{name} asks the applicant to change {sorted(asked)}"


@pytest.mark.slow
def test_only_the_causal_methods_report_a_causal_path(
        cfs: Mapping[str, Sequence[Counterfactual]], ev: Evaluator) -> None:
    """§8.1 fills ``causal_path`` only for a method that scores through the SCM.

    The field is not decoration: an acausal method that filled it in would be
    claiming to have reasoned about consequences it never consulted, and §9's
    causal columns would then be scoring a claim nobody made.  CAC and MINT both
    propagate through the fitted SCM, so both earn a path; the point baselines
    and the acausal oracle do not.  Each path is checked against the graph rather
    than against itself — the features the method moved, plus everything
    downstream of them.
    """
    for name, found in cfs.items():
        if name in CAUSAL:
            continue
        for cf in found:
            assert cf.causal_path == (), f"{name} claims a causal path"

    for name in CAUSAL:
        for cf in cfs[name]:
            moved = [n for n, m in zip(ev.feature_names,
                                       ~np.isclose(cf.x_original, cf.x_cf,
                                                   rtol=0.0, atol=MOVE_ATOL)) if m]
            assert cf.causal_path, f"{name} moved something but reports no path"
            expected = set(moved) | set(ev.fcm.scm.downstream(moved))
            assert set(cf.causal_path) <= expected
            assert set(cf.changed_features) <= set(cf.causal_path)


@pytest.mark.slow
def test_the_query_budget_is_charged_to_the_search(model: Any, x: np.ndarray) -> None:
    """``n_queries`` is a difference on the shared counter, not a self-report.

    §9 compares methods on this number, so it has to mean the same thing for all
    of them: rows scored between entering ``generate`` and leaving it.  Anything
    the audit spends afterwards belongs to the audit.
    """
    method = get_method("random", DATASET, model=model)
    before = int(model.query_count)
    found = method.generate(x, y_target=1, k=2)
    spent = int(model.query_count) - before
    assert spent > 0
    assert {cf.n_queries for cf in found} == {spent}
    # Reading a plan back out costs the audit queries, not the method.
    after = int(model.query_count)
    method.audit(x, np.array([cf.x_cf for cf in found]), y_target=1)
    assert int(model.query_count) > after
    assert {cf.n_queries for cf in found} == {spent}


# ------------------------------------------------- what the oracle promises


@pytest.mark.slow
def test_the_oracle_proves_what_it_claims(cfs: Mapping[str, Sequence[Counterfactual]],
                                          ev: Evaluator, x: np.ndarray) -> None:
    """The reference for §9's cost ratio has to be a reference.

    Three things make it one.  It only ever returns plans the black box accepts,
    so a ratio against it is a ratio against real recourse.  Its cost is the cost
    model's own number for the whole move, so nothing about its own search enters
    the denominator.  And it says whether the enumeration finished — an answer
    found after the evaluation cap is the best seen, not the best, and the flag
    is how the paper would know.
    """
    found = cfs["oracle"]
    for cf in found:
        assert cf.valid, "the oracle returned a plan that does not flip"
        assert cf.feasible and not cf.violations
        assert cf.extras["proven_optimal"] is True
        assert cf.extras["n_evaluated"] > 0
        priced = ev.cost_model.breakdown(x, cf.x_cf)
        assert cf.cost == pytest.approx(float(priced.total))
        assert cf.n_changed == priced.n_changed


@pytest.mark.slow
def test_the_oracles_cheapest_plan_has_nothing_to_spare(
        cfs: Mapping[str, Sequence[Counterfactual]], ev: Evaluator,
        x: np.ndarray) -> None:
    """Optimality implies irredundancy, and irredundancy is checkable.

    Costs are positive and additive over actions, so any sub-plan of an optimal
    plan is strictly cheaper: if one of them also flipped, best-first search
    would have reached it first and returned *that*.  Dropping any single action
    must therefore lose the flip — one batched query, and it holds without
    reference to the grid the oracle enumerated.
    """
    best = cfs["oracle"][0]
    moved = np.flatnonzero(~np.isclose(x, best.x_cf, rtol=0.0, atol=MOVE_ATOL))
    assert moved.size >= 1
    if moved.size == 1:
        pytest.skip("single-action plan: there is no sub-plan to drop")
    rows = np.repeat(best.x_cf.reshape(1, -1), moved.size, axis=0)
    for i, j in enumerate(moved):
        rows[i, j] = x[j]
    flipped = ev.model.predict(rows) == best.y_target
    assert not flipped.any(), (
        f"dropping {[ev.feature_names[j] for j in moved[flipped]]} still flips: "
        "the oracle's 'optimum' contains an action it did not need")


# ---------------------------------------------------- what CAC promises


@pytest.mark.slow
def test_cac_proposes_a_world_the_scm_would_actually_produce(
        cfs: Mapping[str, Sequence[Counterfactual]], ev: Evaluator,
        x: np.ndarray) -> None:
    """The contribution's whole claim, checked against the SCM and not itself.

    CAC's plan is its ``changed_features``.  Handing *only* those to the fitted
    FCM and letting it recompute the descendants has to reproduce the record CAC
    proposed — otherwise the proposal is a point in feature space that happens to
    flip, which is what the baselines already do.  The row is compared with the
    snapping tolerance the audit applies (discrete features live on integers), so
    the test measures the causal claim rather than the rounding.
    """
    names = ev.feature_names
    discrete = ev.cost_model.is_discrete
    for cf in cfs["cac"]:
        mask = np.array([n in set(cf.changed_features) for n in names], dtype=bool)
        assert mask.any(), "cac proposed a plan with no actions in it"
        actions = np.where(mask, cf.x_cf, x)
        implied = ev.fcm.repair(x.reshape(1, -1), actions.reshape(1, -1))[0]
        tol = np.where(discrete, 0.5 + 1e-9, 1e-6)
        assert np.all(np.abs(implied - cf.x_cf) <= tol), (
            "cac's proposal is not what its own plan implies: "
            f"{[names[j] for j in np.flatnonzero(np.abs(implied - cf.x_cf) > tol)]}")
        gap = float(np.max(ev.fcm.violation(x.reshape(1, -1), cf.x_cf.reshape(1, -1))))
        assert gap == pytest.approx(cf.causal_gap, abs=1e-9)
        assert gap <= 1e-6, f"cac left a causal gap of {gap:.3g}"


@pytest.mark.slow
def test_cac_prices_the_action_and_reports_the_consequence(
        cfs: Mapping[str, Sequence[Counterfactual]], ev: Evaluator,
        x: np.ndarray) -> None:
    """§7's line between what is asked for and what follows, on real output.

    ``cost`` prices the intervention, because that is what a person can be held
    to; ``extras["state_cost"]`` prices the whole resulting record, and it is the
    evaluation layer's own :meth:`CostModel.as_effort` number so the two cannot
    drift apart between here and the paper's tables.  The consequences are never
    free, so the second is never below the first.
    """
    for cf in cfs["cac"]:
        assert cf.n_changed <= int(CONFIG.MAX_ACTIONS)
        mask = np.array([n in set(cf.changed_features) for n in ev.feature_names],
                        dtype=bool)
        actions = np.where(mask, cf.x_cf, x)
        assert cf.cost == pytest.approx(float(ev.cost_model.breakdown(x, actions).total))
        state, _ = ev.cost_model.as_effort(x, cf.x_cf)
        assert cf.extras["state_cost"] == pytest.approx(float(state.total))
        assert cf.extras["state_cost"] >= cf.cost - 1e-9
        assert set(cf.changed_features) <= set(cf.extras["state_changed"])


# ---------------------------------------------------- what MINT promises


@pytest.mark.slow
def test_mint_proposes_a_causal_intervention_priced_as_an_action(
        cfs: Mapping[str, Sequence[Counterfactual]], ev: Evaluator,
        x: np.ndarray) -> None:
    """MINT is CAC's cost opponent, so it has to keep CAC's accounting.

    Its answer is a causal intervention: handing *only* ``changed_features`` to
    the fitted FCM has to reproduce the record it proposed, so ``causal_gap`` is
    zero like CAC's and unlike any baseline's.  And its ``cost`` prices that
    action, not the state the SCM drags along behind it — otherwise a Cost column
    comparing MINT and CAC would be comparing two different measurements.  What
    MINT does *not* carry is what CAC adds: no ``state_cost``, no term breakdown,
    because not having them is the baseline.
    """
    found = cfs["mint"]
    assert found, "mint found no causal intervention on this applicant"
    discrete = ev.cost_model.is_discrete
    for cf in found:
        assert cf.valid, "mint returned a plan that does not flip"
        assert cf.n_changed <= int(CONFIG.MAX_ACTIONS)
        mask = np.array([n in set(cf.changed_features) for n in ev.feature_names],
                        dtype=bool)
        assert mask.any(), "mint proposed a plan with no actions in it"
        actions = np.where(mask, cf.x_cf, x)
        implied = ev.fcm.repair(x.reshape(1, -1), actions.reshape(1, -1))[0]
        tol = np.where(discrete, 0.5 + 1e-9, 1e-6)
        assert np.all(np.abs(implied - cf.x_cf) <= tol), (
            "mint's proposal is not what its own intervention implies: "
            f"{[ev.feature_names[j] for j in np.flatnonzero(np.abs(implied - cf.x_cf) > tol)]}")
        assert cf.cost == pytest.approx(
            float(ev.cost_model.breakdown(x, actions).total))
        gap = float(np.max(ev.fcm.violation(x.reshape(1, -1), cf.x_cf.reshape(1, -1))))
        assert gap == pytest.approx(cf.causal_gap, abs=1e-9)
        assert gap <= 1e-6, f"mint left a causal gap of {gap:.3g}"
        assert "state_cost" not in cf.extras and "term_cost" not in cf.extras


# ------------------------------------------------------ what the paper claims


@pytest.mark.slow
def test_baselines_ignore_the_graph(cfs: Mapping[str, Sequence[Counterfactual]],
                                   ev: Evaluator, x: np.ndarray) -> None:
    """§9's premise, reduced to one applicant and one number.

    Read every baseline suggestion as the intervention it implies and ask the SCM
    what would follow.  At least one of them describes a record the world would
    not produce — that is the gap the paper is about, and if it were absent here
    the contribution would have nothing to fix.  The assertion is deliberately
    weak (one plan, one baseline, on one record): the *rates* belong to
    ``experiments/02_compare_methods.py``, which measures them over the whole
    query set instead of asserting them.
    """
    gaps = {name: max((float(np.max(ev.fcm.violation(x.reshape(1, -1),
                                                     cf.x_cf.reshape(1, -1))))
                       for cf in found if cf.valid), default=0.0)
            for name, found in cfs.items() if name in BASELINES}
    assert gaps, "no baseline produced a valid suggestion to read as an intervention"
    assert max(gaps.values()) > 1e-3, (
        "every baseline happened to propose a causally consistent record on this "
        f"applicant: {gaps} — the claim needs a different record, not a weaker test")


# --------------------------------------------------------- rule #2: determinism


@pytest.mark.slow
def test_the_same_seed_gives_the_same_advice(model: Any, x: np.ndarray) -> None:
    """Rule #2, where it matters most: two runs of one search agree exactly.

    ``dice`` is left out because its randomness lives in ``dice-ml``, which the
    project does not seed; that is a documented deviation rather than something
    this test can fix.
    """
    for name in ("random", "wachter", "oracle", "mint", "cac"):
        first = get_method(name, DATASET, model=model).generate(x, y_target=1, k=2)
        second = get_method(name, DATASET, model=model).generate(x, y_target=1, k=2)
        assert [cf.key() for cf in first] == [cf.key() for cf in second], name
        assert [cf.cost for cf in first] == [cf.cost for cf in second], name


# ------------------------------------------- the knobs §9.3's ablation turns


def test_an_unknown_objective_weight_is_refused(model: Any) -> None:
    """``lambdas`` names the four terms of §3, and nothing else.

    A typo that was accepted silently would produce a variant identical to full
    CAC, published in the ablation table as though a term had been removed —
    which is the one failure mode of an ablation that a reader cannot detect.
    """
    with pytest.raises(KeyError, match="unknown objective weight"):
        CACMethod(DATASET, model=model, lambdas={"validity": 0.0})
    method = CACMethod(DATASET, model=model, lambdas={"valid": 0.0})
    assert method.lambdas["valid"] == 0.0
    assert method.lambdas["cost"] == pytest.approx(CONFIG.CAC_LAMBDA_COST)


def test_the_defaults_are_the_papers_method(model: Any) -> None:
    """Constructed with no knobs, CAC is exactly what §3 describes.

    The weights are read per instance rather than from ``CONFIG`` at scoring
    time, so this asserts the two have not drifted apart — an ablation that
    started from the wrong baseline would make every row of its table wrong by
    the same unknown amount.
    """
    method = CACMethod(DATASET, model=model)
    assert method.lambdas == {"valid": CONFIG.CAC_LAMBDA_VALID,
                              "cost": CONFIG.CAC_LAMBDA_COST,
                              "feas": CONFIG.CAC_LAMBDA_FEAS,
                              "plaus": CONFIG.CAC_LAMBDA_PLAUS}
    assert method.propagate is True
    assert method.name == "cac"
    assert set(method.lambdas) == set(CACMethod.LAMBDAS)


def test_a_variant_signs_its_own_rows(model: Any, x: np.ndarray) -> None:
    """``label`` has to reach ``Counterfactual.method``, or the table mislabels itself.

    ``Evaluator.row`` reads the dataclass field, not the argument it was called
    with, so a variant whose plans still say ``cac`` would be averaged into the
    reference row and the ablation would compare the full objective with itself.
    """
    method = CACMethod(DATASET, model=model, label="cac_no_cost",
                       lambdas={"cost": 0.0})
    assert method.name == "cac_no_cost"
    plans = method.generate(x, y_target=1, k=1)
    assert plans and all(cf.method == "cac_no_cost" for cf in plans)


@pytest.mark.slow
def test_removing_the_scm_from_the_representation_breaks_the_promise(
        model: Any, x: np.ndarray) -> None:
    """The claim the ablation exists to test, on one applicant.

    ``propagate=False`` is the only variant that cannot be written as a weight:
    it scores the identical candidate population as a *point*, the way every
    baseline does, and leaves the rest of CAC untouched.  If causal validity were
    a penalty term, this would change nothing.  It is not — so the plans stop
    being worlds the SCM would produce, and ``causal_gap`` is the measurement of
    that.
    """
    full = CACMethod(DATASET, model=model).generate(x, y_target=1, k=K)
    point = CACMethod(DATASET, model=model, label="cac_no_causal",
                      propagate=False).generate(x, y_target=1, k=K)
    assert full and point
    assert max(cf.causal_gap for cf in full) == pytest.approx(0.0, abs=1e-9)
    assert max(cf.causal_gap for cf in point) > 1e-6, (
        "scoring an intervention as a point left the SCM satisfied, which would "
        "mean the graph has nothing to say about this applicant"
    )


@pytest.mark.slow
def test_the_objective_is_the_weighted_sum_it_claims_to_be(model: Any,
                                                          x: np.ndarray) -> None:
    """§3's objective, read back off the plans that were selected by it.

    Every archived plan carries the four term values that produced its score, so
    the identity can be checked without re-implementing the terms here — which
    would only assert that two copies of the same arithmetic agree.  The second
    half is the part that matters for the ablation: with three weights at zero the
    objective must collapse to the surviving term, so a row labelled ``-- cost``
    really is ranking by nothing but validity, feasibility and plausibility.
    """
    method = CACMethod(DATASET, model=model)
    for cf in method.generate(x, y_target=1, k=K):
        terms = {"valid": cf.extras["term_validity"],
                 "cost": cf.extras["term_cost"],
                 "feas": cf.extras["term_feasibility"],
                 "plaus": cf.extras["term_plausibility"]}
        expected = sum(method.lambdas[key] * float(value)
                       for key, value in terms.items())
        assert float(cf.extras["objective"]) == pytest.approx(expected, abs=1e-9)

    only_cost = CACMethod(DATASET, model=model, label="cac_cost_only",
                          lambdas={"valid": 0.0, "feas": 0.0, "plaus": 0.0})
    plans = only_cost.generate(x, y_target=1, k=K)
    assert plans
    for cf in plans:
        assert float(cf.extras["objective"]) == pytest.approx(
            CONFIG.CAC_LAMBDA_COST * float(cf.extras["term_cost"]), abs=1e-9)
    scores = [float(cf.extras["objective"]) for cf in plans]
    assert scores == sorted(scores), "the selection did not rank by the objective"
