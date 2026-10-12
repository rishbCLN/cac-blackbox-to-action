"""The two known-SCM synthetic worlds, and the ground-truth auditor behind them.

Workstream C (Finding 6b) adds two in-process worlds — :mod:`src.data.load_synth_reach`
and :mod:`src.data.load_synth_block` — whose structural equations and target rule are
written down in closed form, so experiment 08 can audit a method's declared actions
through the *true* SCM rather than through an FCM the method also optimised against.
That de-circularisation argument only holds if three things stay true, and nothing
else in the suite would notice if they drifted:

* the two worlds are *identical except for one edge*.  They share the income
  equation, the feature schema, the bounds and the cost model; they differ only in
  the savings pass-through coefficient and in whether the immutable ``age`` reaches
  ``approved``.  If a refactor let their schemas drift apart, the "same world minus
  one edge" claim — the whole point of the foil — would quietly become false.
* ``synth_reach`` has *no* ``age -> approved`` edge and ``synth_block`` *has* it.
  This single structural difference is what turns guaranteed recourse into honest
  non-recourse; it is asserted here on the graph itself, not inferred from a run.
* :class:`~src.scm.synthetic_truth.TrueSCM` reproduces the generative equations
  exactly.  The auditor is not fitted, so a wiring bug would not show up as a bad
  score — it would show up as the *wrong* score, silently, which is the one failure
  mode a de-circularised audit cannot tolerate.

These worlds are deliberately *not* in :data:`~src.config.DATASETS`: they are
first-class registered datasets exercised only by experiment 08, so §9's headline
table is unchanged.  The first test pins that down.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.config import DATASETS
from src.data import load_synth_block as block
from src.data import load_synth_reach as reach
from src.scm.define_scm import get_scm, load_scm_json
from src.scm.fcm import FCM
from src.scm.synthetic_truth import TrueSCM, true_scm

WORLDS: tuple[str, ...] = ("synth_reach", "synth_block")


# ------------------------------------------------- the worlds stay off the roster
def test_the_synthetic_worlds_are_not_in_the_headline_roster() -> None:
    """§9's comparison roster is the four real datasets, and only those.

    The synthetic worlds are registered loaders with their own SCM JSON, so every
    layer *can* build them — but if they leaked into :data:`DATASETS` the main
    comparison table (and every ``params=list(DATASETS)`` fixture in the suite)
    would silently start reporting a known-SCM toy world as if it were a finding.
    """
    for world in WORLDS:
        assert world not in DATASETS


# -------------------------------------------------------------------- the loaders
@pytest.fixture(scope="module")
def reach_frame():
    return reach.load_synth_reach()


@pytest.fixture(scope="module")
def block_frame():
    return block.load_synth_block()


def test_both_loaders_are_seeded_and_reproducible() -> None:
    """A ground-truth world that changed shape run-to-run would be no ground truth.

    Experiment 08 trains a black box on these rows and audits against the analytic
    SCM; if the draw were not seeded, the "truth" would move under the auditor.
    """
    assert reach.load_synth_reach()[0].equals(reach.load_synth_reach()[0])
    assert block.load_synth_block()[0].equals(block.load_synth_block()[0])


@pytest.mark.parametrize("fixture_name", ["reach_frame", "block_frame"])
def test_the_frame_matches_the_spec_it_ships_with(
        fixture_name: str, request: pytest.FixtureRequest) -> None:
    frame, spec = request.getfixturevalue(fixture_name)
    assert list(frame.columns) == [*spec.feature_names, spec.target]
    assert len(frame) == reach.N_ROWS
    # training is the one discrete column; its labels must be exactly the levels.
    assert set(frame["training"]) <= set(reach.TRAINING_LEVELS)
    # every numeric feature stays inside the box the spec promises the optimiser.
    for feat in spec.features:
        if feat.kind == "numeric":
            lo, hi = feat.bounds
            col = frame[feat.name].to_numpy(dtype=float)
            assert col.min() >= lo and col.max() <= hi, feat.name
    # both classes are present and the world is not degenerate — there have to be
    # rejected applicants for recourse to be about, and approved ones to flip to.
    rate = float(frame[spec.target].mean())
    assert 0.1 < rate < 0.9, f"{spec.name}: degenerate target rate {rate:.3f}"


def test_the_stored_label_follows_the_written_rule(
        reach_frame, block_frame) -> None:
    """The target column must be the known rule applied to the known columns.

    The draw adds a small score noise, so agreement is not 1.0, but a wiring bug
    (scoring the wrong column, dropping a term) collapses it toward chance — the
    gap between ~0.93 and 0.5 is the whole signal, so a conservative floor catches
    it without being flaky on a seeded draw.
    """
    fr, _ = reach_frame
    pred_r = (reach.approval_score(fr["income"].to_numpy(float),
                                   fr["savings"].to_numpy(float)) >= reach.TAU)
    assert (pred_r == fr["approved"].to_numpy(int)).mean() >= 0.90

    fb, _ = block_frame
    pred_b = (block.approval_score(fb["age"].to_numpy(float),
                                   fb["income"].to_numpy(float),
                                   fb["savings"].to_numpy(float)) >= block.TAU)
    assert (pred_b == fb["approved"].to_numpy(int)).mean() >= 0.90


def test_the_two_worlds_differ_only_where_they_are_meant_to(
        reach_frame, block_frame) -> None:
    """Same seed, same income equation — so income is identical and savings is not.

    This is the data-level face of "identical except for one coefficient and one
    edge": the shared ``income_mean`` draw must coincide exactly, while the weaker
    savings pass-through (0.4 vs 0.5) must pull the savings columns apart.
    """
    fr, _ = reach_frame
    fb, _ = block_frame
    assert np.allclose(fr["income"].to_numpy(float), fb["income"].to_numpy(float))
    assert not np.allclose(fr["savings"].to_numpy(float), fb["savings"].to_numpy(float))


# ----------------------------------------------------------- the generative rule
def test_the_shared_income_equation_is_the_one_both_worlds_wrote_down() -> None:
    """``income = 30 + 0.8*exp + 6*train + 0.15*age`` — a contract, not a guess."""
    expected = 30.0 + 0.8 * 20 + 6.0 * 2 + 0.15 * 40  # = 64.0
    assert float(reach.income_mean(40, 20, 2, 0.0)) == pytest.approx(expected)
    assert float(block.income_mean(40, 20, 2, 0.0)) == pytest.approx(expected)
    # savings is where the worlds first diverge: 0.5x vs 0.4x of income.
    assert float(reach.savings_mean(64.0, 0.0)) == pytest.approx(32.0)
    assert float(block.savings_mean(64.0, 0.0)) == pytest.approx(25.6)


def test_only_synth_block_lets_age_touch_approval() -> None:
    """The decisive asymmetry, read straight off the two score functions.

    ``reach``'s rule cannot even *name* ``age`` (so recourse is always reachable);
    ``block``'s can, and because the age coefficient is positive, a younger
    applicant scores strictly lower at fixed income/savings — the engine of honest
    non-recourse.
    """
    assert "age" not in inspect.signature(reach.approval_score).parameters
    assert "age" in inspect.signature(block.approval_score).parameters
    # at the pivots the score is exactly zero in both worlds.
    assert float(reach.approval_score(60.0, 30.0, 0.0)) == pytest.approx(0.0)
    assert float(block.approval_score(44.0, 60.0, 22.0, 0.0)) == pytest.approx(0.0)
    # younger scores lower at fixed, identical levers.
    assert block.approval_score(20.0, 70.0, 30.0) < block.approval_score(60.0, 70.0, 30.0)


def test_synth_block_has_honest_non_recourse_that_synth_reach_cannot(
        reach_frame, block_frame) -> None:
    """A rejected young applicant stays rejected even pushed to the observed ceiling.

    Over-approximate anyone's reachable set by the *dataset-wide* maximum income
    and savings — more than any admissible, budgeted move could grant — and show
    that in ``block`` the youngest applicants still fall short, while in ``reach``
    that same ceiling always clears the bar.  This is the honest-failure vs
    always-reachable contrast experiment 08 reports, reduced to one assertion.
    """
    fr, _ = reach_frame
    fb, _ = block_frame
    youngest = 18.0
    reach_ceiling = float(reach.approval_score(fr["income"].max(), fr["savings"].max()))
    block_ceiling = float(block.approval_score(youngest, fb["income"].max(),
                                               fb["savings"].max()))
    assert reach_ceiling >= reach.TAU, "synth_reach must always offer a reachable flip"
    assert block_ceiling < block.TAU, "synth_block must contain honest non-recourse"


# ---------------------------------------------------------------- the SCM on disk
@pytest.mark.parametrize("world", WORLDS)
def test_each_world_graph_validates_against_its_loader(world: str) -> None:
    """``get_scm`` runs ``_check_against_spec``; a graph/loader drift fails here.

    So this single build is also the regression test that the SCM JSON and the
    loader keep agreeing about features, target and the immutable set.
    """
    scm = get_scm(world)
    assert scm.target == "approved"
    assert set(scm.non_manipulable) == {"age"}
    assert scm.spec is not None and set(scm.features) == set(scm.spec.feature_names)
    assert scm.graph.out_degree("approved") == 0, "approved must be a sink"
    # the JSON on disk and the assembled graph cannot disagree about the edges.
    on_disk = {(e["from"], e["to"]) for e in load_scm_json(world)["edges"]}
    assert on_disk == set(scm.edges())


def test_the_age_to_approved_edge_is_the_only_structural_difference() -> None:
    """``synth_block`` is ``synth_reach`` plus exactly one edge: ``age -> approved``.

    If the two graphs ever differed by anything else, "same world, one edge" — the
    sentence the de-circularisation result leans on — would be false.
    """
    reach_edges = set(get_scm("synth_reach").edges())
    block_edges = set(get_scm("synth_block").edges())
    assert ("age", "approved") not in reach_edges
    assert ("age", "approved") in block_edges
    assert block_edges - reach_edges == {("age", "approved")}
    assert reach_edges - block_edges == set()


# ------------------------------------------------------ the analytic truth auditor
@pytest.mark.parametrize("world", WORLDS)
def test_true_scm_is_an_fcm_with_only_the_known_equations_installed(world: str) -> None:
    """The auditor is an :class:`FCM`, so the evaluator's whole surface still works,
    but its two non-root nodes carry the generative truth and its roots carry none."""
    t = true_scm(world)
    assert isinstance(t, (TrueSCM, FCM))
    assert true_scm(world) is t, "truth is process-cached: one auditor per world"
    non_root = {"income", "savings"}
    for node, eq in t.equations.items():
        if node in non_root:
            assert eq.model is not None and type(eq.model).__name__ == "_ClosedForm"
            assert eq.score_name == "exact"
        else:
            assert eq.model is None, f"{node} is a root and must carry no equation"


@pytest.mark.parametrize("world", WORLDS)
def test_true_scm_refuses_to_be_fitted_or_persisted(world: str) -> None:
    """It is analytic: fitting is a no-op and there is nothing to save or load."""
    t = true_scm(world)
    assert t.fit() is t
    with pytest.raises(NotImplementedError):
        t.save("ignored")
    with pytest.raises(NotImplementedError):
        t.load("ignored")


def test_true_scm_only_exists_for_worlds_with_a_closed_form() -> None:
    with pytest.raises(KeyError):
        true_scm("adult")


def test_true_scm_propagates_the_exact_generative_equations() -> None:
    """Intervene through the truth and the descendants must land on ``f(pa)`` exactly.

    Start from a residual-free factual row (built from the equations themselves),
    so abduction carries zero noise and the counterfactual is the pure structural
    response — which must equal the loader's own ``income_mean`` / ``savings_mean``.
    A fitted auditor could only approximate this; the analytic one has to be exact.
    """
    tr = true_scm("synth_reach")
    inc = float(reach.income_mean(40, 20, 2, 0.0))
    x = np.array([40.0, 20.0, 2.0, inc, float(reach.savings_mean(inc, 0.0))])
    cf = tr.do(x, {"experience": 30.0})
    exp_inc = float(reach.income_mean(40, 30, 2, 0.0))
    assert cf[tr.index["income"]] == pytest.approx(exp_inc)
    assert cf[tr.index["savings"]] == pytest.approx(reach.savings_mean(exp_inc, 0.0))
    assert cf[tr.index["age"]] == 40.0 and cf[tr.index["training"]] == 2.0

    tb = true_scm("synth_block")
    inc_b = float(block.income_mean(40, 20, 2, 0.0))
    xb = np.array([40.0, 20.0, 2.0, inc_b, float(block.savings_mean(inc_b, 0.0))])
    cfb = tb.do(xb, {"income": 100.0})
    assert cfb[tb.index["income"]] == pytest.approx(100.0)
    assert cfb[tb.index["savings"]] == pytest.approx(block.savings_mean(100.0, 0.0))


@pytest.mark.parametrize("world", WORLDS)
def test_repairing_an_untouched_row_through_the_truth_is_the_identity(world: str) -> None:
    """Pearl's three steps with an empty intervention set return the factual record —
    the same invariant ``test_scm`` pins on the fitted FCM, here on the analytic one."""
    t = true_scm(world)
    rng = np.random.default_rng(0)
    rows = t.project(rng.uniform(t.bounds[:, 0], t.bounds[:, 1], size=(4, t.n_features)))
    assert np.allclose(t.repair(rows, rows), rows, atol=1e-9)
