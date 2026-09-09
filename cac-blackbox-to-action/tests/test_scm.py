"""The causal layer: is the graph the one §6 wrote, and does the FCM obey Pearl?

Three modules, three kinds of question, and they fail in very different ways.

:mod:`src.scm.define_scm` is a *specification*.  §6.1 lists Adult's nodes and
fourteen edges literally, so the test for it is literal too: a graph that quietly
loses ``gender -> occupation`` would still validate, still fit, still produce
counterfactuals, and would silently answer a different research question.
:func:`test_adult_is_exactly_the_graph_section_6_specifies` is the only place
that would notice.

:mod:`src.scm.fcm` is *machinery*, and its correctness is Pearl's three steps.
Two properties pin it down.  A counterfactual with no intervention has to return
the factual record — abduction recovers the noise, prediction puts it back — and
that is the identity a bug hides behind.  A repaired record has to be a fixed
point: repair it again and nothing moves, which is precisely what §9 means by
"causally valid".  Both are checked on batches *and* row by row, because a
per-row loop and a vectorised path are two implementations of one function and
the second is where the noise of row *i* gets applied to row *j*.

:mod:`src.scm.validate_scm` is a *critic*, so it is tested on graphs that
deserve criticism: a cycle, a target nothing can reach, a world where every
effective action is priced at infinity.  Its conditional-independence machinery
is exercised on real data but asserted loosely — real data disagrees with any
hand-drawn graph, which is why §6.4 warns instead of failing, and a test that
demanded agreement would be asserting something the paper explicitly denies.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.config import DATASETS
from src.data.preprocess import load_dataset
from src.scm.define_scm import (SCM, SCMError, all_scms, build_scm, get_scm,
                                load_scm_json, scm_path)
from src.scm.fcm import FCM, get_fcm
from src.scm.validate_scm import (check_acyclic, check_reachability,
                                  check_semantics, edge_constraints,
                                  run_ci_tests, training_frame, validate_scm)

#: §6.1, transcribed.  Fourteen edges, in the order the specification lists them.
ADULT_EDGES: tuple[tuple[str, str], ...] = (
    ("age", "education"), ("age", "occupation"), ("age", "hours_per_week"),
    ("age", "marital_status"), ("education", "occupation"), ("education", "income"),
    ("education", "hours_per_week"), ("occupation", "income"),
    ("occupation", "hours_per_week"), ("hours_per_week", "income"),
    ("marital_status", "income"), ("gender", "occupation"),
    ("gender", "hours_per_week"), ("gender", "income"),
)

ADULT_NODES: tuple[str, ...] = (
    "age", "education", "occupation", "hours_per_week", "marital_status",
    "gender", "income",
)

#: §6.1's ``interventional_costs``, as labels rather than numbers — the numbers
#: are §7's business, the ordering is §6's.
ADULT_COST_LABELS: dict[str, str] = {
    "age": "infinite", "gender": "infinite", "marital_status": "very_high",
    "education": "high", "occupation": "high", "hours_per_week": "medium",
    "income": "low",
}


@pytest.fixture(scope="module")
def adult() -> SCM:
    return get_scm("adult")


@pytest.fixture(scope="module", params=list(DATASETS))
def scm(request: Any) -> SCM:
    """Every dataset's graph, so the structural rules are checked on all three."""
    return get_scm(request.param)


# ------------------------------------------------------ §6.1 as a specification


def test_adult_contains_exactly_the_graph_section_6_specifies(adult: SCM) -> None:
    """§6.1's fourteen edges, edge for edge — plus the ones the data forces.

    A deviation worth stating rather than hiding: §6.1's literal JSON names seven
    nodes, but the processed Adult data has ``workclass``, ``capital_gain`` and
    ``capital_loss`` as well, and an SCM that omitted them would leave three
    features outside the causal model — the FCM would treat them as roots, and a
    plan that moved education would silently claim they stay put.  So the graph
    is a strict *superset*: every asserted edge of §6.1 is present, unchanged,
    and the additions are documented in the JSON's own ``notes``.

    Nothing else in the pipeline would notice a missing edge: the FCM would fit,
    the cost model would price, CAC would propagate, and every number in the
    paper would be about a graph nobody wrote down.
    """
    assert set(ADULT_NODES) <= set(adult.nodes)
    assert set(ADULT_EDGES) <= set(adult.edges())
    assert len(ADULT_EDGES) == 14
    assert adult.target == "income"
    assert set(adult.non_manipulable) == {"age", "gender"}
    assert set(adult.sensitive) == {"gender", "age", "marital_status"}
    for node, label in ADULT_COST_LABELS.items():
        assert adult.cost_labels[node] == label, node
    assert any("Pearl" in r for r in adult.references) \
        and any("Jung" in r for r in adult.references), \
        "§6.1 cites Kuroki & Pearl (2014) and Jung et al. (2021)"
    extra = set(adult.nodes) - set(ADULT_NODES)
    assert extra == {"workclass", "capital_gain", "capital_loss"}
    assert adult.notes, "the additions to §6.1's graph have to be justified in the JSON"


def test_every_graph_covers_exactly_its_dataset(scm: SCM) -> None:
    """Why §6.1's example had to grow: the nodes are the columns, all of them.

    A feature outside the graph is a feature the FCM treats as an exogenous root,
    which is a causal claim ("nothing affects it") made by omission rather than
    on purpose.
    """
    assert scm.spec is not None
    assert set(scm.features) == set(scm.spec.feature_names)
    assert scm.target == scm.spec.target


def test_the_graph_survives_a_round_trip_through_its_own_schema(scm: SCM) -> None:
    """``to_dict`` writes the schema ``load_scm_json`` reads, so the JSON on disk
    and the object in memory cannot drift into two different graphs."""
    payload = scm.to_dict()
    assert payload["dataset"] == scm.dataset
    assert payload["target"] == scm.target
    assert {(e["from"], e["to"]) for e in payload["edges"]} == set(scm.edges())
    on_disk = load_scm_json(scm.dataset)
    assert {(e["from"], e["to"]) for e in on_disk["edges"]} == set(scm.edges())
    assert scm_path(scm.dataset).exists()
    mermaid = scm.to_mermaid()
    for parent, child in scm.edges():
        assert parent in mermaid and child in mermaid


def test_get_scm_is_cached_and_all_scms_covers_the_paper(scm: SCM) -> None:
    """One object per dataset, so ``method.scm is evaluator.scm`` throughout §9."""
    assert get_scm(scm.dataset) is get_scm(scm.dataset)
    assert build_scm(scm.dataset) is not scm  # a fresh build, for comparison
    assert set(all_scms()) == set(DATASETS)


def test_a_missing_scm_is_a_hard_error() -> None:
    """§6 makes the graph a required artefact, not an optional refinement."""
    with pytest.raises(FileNotFoundError, match="required artefact"):
        get_scm("mnist")


def test_a_graph_that_contradicts_its_own_declarations_is_rejected(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_scm`` validates; :class:`SCM` itself is only a container.

    Three failure modes, all silent without these guards.  A typo in
    ``non_manipulable`` would leave a protected attribute actionable — the
    baselines mask by that list, so ``gendre`` would make them free to change
    ``gender``.  A cycle would make abduction meaningless.  And an edge *out of*
    the target would let a plan flip the decision by changing something the
    decision causes.

    The third one needs a fresh node to reach at all: on Adult every feature is
    an ancestor of ``income``, so any edge out of the target is caught as a cycle
    first — which is itself worth knowing, since it means the sink rule can only
    be broken by a graph that also invents a variable.
    """
    import src.scm.define_scm as define_scm

    payload = load_scm_json("adult")
    typo = {**payload, "non_manipulable": [*payload["non_manipulable"], "gendre"]}
    monkeypatch.setattr(define_scm, "load_scm_json", lambda dataset: typo)
    with pytest.raises(SCMError, match="unknown node"):
        build_scm("adult")

    cyclic = {**payload,
              "edges": [*payload["edges"], {"from": "income", "to": "education"}]}
    monkeypatch.setattr(define_scm, "load_scm_json", lambda dataset: cyclic)
    with pytest.raises(SCMError, match="cycle"):
        build_scm("adult")

    not_a_sink = {**payload,
                  "nodes": [*payload["nodes"], "commute_time"],
                  "edges": [*payload["edges"],
                            {"from": "income", "to": "commute_time"}]}
    monkeypatch.setattr(define_scm, "load_scm_json", lambda dataset: not_a_sink)
    with pytest.raises(SCMError, match="must be a sink"):
        build_scm("adult")


# ---------------------------------------------------- the FCM, and Pearl's steps


FCM_DATASET = "german"


@pytest.fixture(scope="module")
def fcm() -> FCM:
    return get_fcm(FCM_DATASET)


@pytest.fixture(scope="module")
def rows() -> np.ndarray:
    """Eight real records — a batch, because one row hides every batching bug."""
    bundle = load_dataset(FCM_DATASET)
    return np.asarray(bundle.X_test[:8], dtype=float)


def test_a_counterfactual_with_no_intervention_is_the_factual_record(
        fcm: FCM, rows: np.ndarray) -> None:
    """Pearl's three steps, composed, must be the identity.

    Abduction recovers each record's noise, prediction puts it back through the
    same equations, and with an empty intervention set nothing else changes — so
    the result is the record you started with.  Every fitted equation is
    approximate, but this property does not depend on how good the fit is: it is
    the noise being carried, not the mean being right.  A residual here is a bug
    in the machinery rather than error in the model.

    Two versions of "no intervention", because only the second one actually runs
    the equations.  An empty mask recomputes nothing and would pass even if
    abduction were broken; a mask that *does* fire but sets each feature to the
    value it already had forces every descendant through ``f(pa) + u`` (or the
    inverse-CDF coupling) and must still land back on the factual level.

    Note the shape contract: :meth:`FCM.counterfactual` takes **one** factual row
    and ``m`` candidate interventions on that row.  Populations go through
    :meth:`FCM.repair`.
    """
    m = 3
    for i, row in enumerate(rows):
        idle = fcm.counterfactual(
            row,
            np.repeat(row.reshape(1, -1), m, axis=0),
            np.zeros((m, fcm.n_features), dtype=bool),
        )
        assert idle.shape == (m, fcm.n_features)
        assert np.allclose(idle, row, atol=1e-9), f"row {i}: the empty mask moved something"

    assert np.allclose(fcm.repair(rows, rows), rows, atol=1e-9), (
        "repairing an untouched population is not the identity")

    # A do() that asks for the status quo: the mask fires, the descendants are
    # recomputed, and the carried noise has to put them back where they were.
    with_children = [f for f in fcm.scm.actionable
                     if set(fcm.scm.downstream(f)) & set(fcm.index)]
    assert with_children, f"{FCM_DATASET}: no actionable feature has a child to recompute"
    for name in with_children:
        same = fcm.do(rows[0], {name: float(rows[0, fcm.index[name]])})
        moved = [fcm.feature_names[j]
                 for j in np.flatnonzero(~np.isclose(same, rows[0], atol=1e-6))]
        assert not moved, f"do({name}=status quo) moved {moved}"


def test_the_noise_of_one_row_never_reaches_another(fcm: FCM,
                                                    rows: np.ndarray) -> None:
    """The batching regression, kept as a test.

    :meth:`FCM.counterfactual` is vectorised over records, and the bug it is easy
    to write is applying row *i*'s abducted noise to row *j* — which produces
    plausible-looking output, propagates into every causal number in §9, and
    would have made the paper's central claim unverifiable.  Batched and row-by-row
    have to agree exactly.
    """
    proposals = rows.copy()
    j = fcm.index[fcm.scm.actionable[0]]
    proposals[:, j] = proposals[:, j] + 1.0
    batched = fcm.repair(rows, proposals)
    for i in range(rows.shape[0]):
        alone = fcm.repair(rows[i].reshape(1, -1), proposals[i].reshape(1, -1))
        assert np.allclose(batched[i], alone[0], atol=1e-9), f"row {i} differs in batch"


def test_each_row_recomputes_only_what_its_own_action_implies(
        fcm: FCM, rows: np.ndarray) -> None:
    """One batch, two different interventions, two different recompute sets.

    This is what :attr:`FCM.reach` is for, and it has to be tested through
    :meth:`FCM.counterfactual`: that is the only method vectorised over
    interventions, so it is the only place the bug can live (``repair`` loops
    row by row and would pass either way).

    The probe is built to catch the union implementation in both directions the
    code's own comment names.  Row 0 acts on a feature ``f``; row 1 acts on one of
    ``f``'s children ``c``.  Taking the union of the batch's masks and subtracting
    it — ``(mask.any(0) @ reach) & ~mask.any(0)`` — freezes ``c`` for row 0, which
    never asked for it, and would rewrite ``f``'s descendants for row 1, which
    never touched ``f``.  Batched and solo must agree exactly.
    """
    delta = 1.0
    x = rows[0]
    pair: tuple[str, str] | None = None
    for f in fcm.scm.actionable:
        children = [c for c in fcm.scm.children(f) if c in fcm.index]
        for c in children:
            solo = fcm.do(x, {f: x[fcm.index[f]] + delta})
            if not np.isclose(solo[fcm.index[c]], x[fcm.index[c]], atol=1e-6):
                pair = (f, c)
                break
        if pair:
            break
    if pair is None:
        pytest.skip(f"{FCM_DATASET}: no actionable feature visibly moves a child")
    f, c = pair

    values = np.repeat(x.reshape(1, -1), 2, axis=0)
    mask = np.zeros((2, fcm.n_features), dtype=bool)
    values[0, fcm.index[f]] += delta
    mask[0, fcm.index[f]] = True
    values[1, fcm.index[c]] += delta
    mask[1, fcm.index[c]] = True

    batched = fcm.counterfactual(x, values, mask)
    for i, acted in enumerate((f, c)):
        alone = fcm.counterfactual(x, values[i:i + 1], mask[i:i + 1])
        assert np.allclose(batched[i], alone[0], atol=1e-9), (
            f"row {i} (do({acted})) changed when batched with the other row")
        changed = {fcm.feature_names[j]
                   for j in np.flatnonzero(~np.isclose(batched[i], x, atol=1e-9))}
        allowed = {acted} | (set(fcm.scm.downstream(acted, include_target=True))
                             & set(fcm.index))
        assert changed <= allowed, (
            f"row {i} acted on {acted!r} but {sorted(changed - allowed)} moved too")

    # The two directions, spelled out: row 0 must recompute ``c`` even though row 1
    # pinned it, and row 1 must leave ``f`` alone even though row 0 moved it.
    assert not np.isclose(batched[0, fcm.index[c]], x[fcm.index[c]], atol=1e-6), (
        f"{c!r} was frozen for a row that never intervened on it")
    assert np.isclose(batched[1, fcm.index[f]], x[fcm.index[f]], atol=1e-9), (
        f"{f!r} moved for a row that never intervened on it")


def test_reach_is_the_descendant_relation(fcm: FCM) -> None:
    """``reach[a, b]`` is checked against the graph, not against itself."""
    for name, j in fcm.index.items():
        expected = {fcm.index[c] for c in fcm.scm.descendants(name) if c in fcm.index}
        assert set(np.flatnonzero(fcm.reach[j])) == expected, name
        assert not fcm.reach[j, j]


def test_a_repaired_record_is_a_fixed_point(fcm: FCM, rows: np.ndarray) -> None:
    """What §9 means by "causally valid", stated as a property of ``repair``.

    Repairing a proposal gives the world its actions imply; repairing *that* has
    to change nothing, because every coordinate is now the value the equations
    themselves produce.  If it moved, "causal validity" would be a moving target
    and no method could reach it — including CAC, which is built by construction
    to land on this fixed point.
    """
    proposals = rows.copy()
    proposals[:, fcm.index[fcm.scm.actionable[0]]] += 1.0
    repaired = fcm.repair(rows, proposals)
    assert float(np.max(fcm.violation(rows, repaired))) <= 1e-6
    assert np.allclose(fcm.repair(rows, repaired), repaired, atol=1e-9)


def test_do_is_counterfactual_with_one_column_masked(fcm: FCM,
                                                     rows: np.ndarray) -> None:
    """The convenience API and the general one have to be the same operation."""
    name = fcm.scm.actionable[0]
    j = fcm.index[name]
    x = rows[0]
    value = float(x[j]) + 1.0
    mask = np.zeros((1, fcm.n_features), dtype=bool)
    mask[0, j] = True
    values = x.reshape(1, -1).copy()
    values[0, j] = value
    assert np.allclose(fcm.do(x, {name: value}),
                       fcm.counterfactual(x.reshape(1, -1), values, mask)[0])
    with pytest.raises(KeyError):
        fcm.do(x, {"not_a_feature": 0.0})


def test_real_records_are_already_projected(fcm: FCM, rows: np.ndarray) -> None:
    """``project`` clips and rounds; on real data it must be the identity.

    Otherwise every propagated record would be silently nudged, and the
    difference would show up in §9 as a causal gap the method never caused.
    """
    assert np.allclose(fcm.project(rows), rows, atol=1e-9)


def test_a_saved_fcm_makes_the_same_predictions(fcm: FCM, rows: np.ndarray,
                                                tmp_path: Any) -> None:
    """The FCM is cached to disk between experiments, so §9's causal numbers come
    from a *reloaded* object — it has to be the same function."""
    path = fcm.save(tmp_path / "fcm.joblib")
    reloaded = FCM(FCM_DATASET).load(path)
    proposals = rows.copy()
    proposals[:, fcm.index[fcm.scm.actionable[0]]] += 1.0
    assert np.allclose(reloaded.repair(rows, proposals), fcm.repair(rows, proposals))


# ------------------------------------------------------------- §6.4's validator


def _toy(edges: list[tuple[str, str]], target: str = "y",
         non_manipulable: tuple[str, ...] = (),
         labels: dict[str, str] | None = None) -> SCM:
    """A hand-built SCM, so the critic can be shown something worth criticising.

    ``build_scm`` refuses most of these outright, which is the point: the checks
    are a second line of defence for graphs that arrive some other way, and they
    cannot be tested through a constructor that rejects the input first.
    """
    graph = nx.DiGraph(edges)
    labels = labels or {}
    full = {n: labels.get(n, "medium") for n in graph.nodes()}
    return SCM(dataset="toy", graph=graph, target=target,
               non_manipulable=non_manipulable, sensitive=(),
               costs={n: 1.0 for n in full}, cost_labels=full)


def test_the_validator_reports_a_cycle_as_an_error() -> None:
    cyclic = _toy([("a", "b"), ("b", "a"), ("b", "y")])
    issues = check_acyclic(cyclic)
    assert [i.level for i in issues] == ["error"]
    assert issues[0].code == "cycle"
    assert check_acyclic(_toy([("a", "b"), ("b", "y")]))[0].level == "info"


def test_the_validator_reports_a_target_nothing_can_reach() -> None:
    """§6.4's check 2: if no manipulable feature reaches the decision, the whole
    experiment is asking for recourse the graph forbids."""
    stranded = _toy([("a", "y"), ("b", "c")], non_manipulable=("a",))
    codes = {i.code for i in check_reachability(stranded) if i.level == "error"}
    assert "unreachable" in codes


def test_the_validator_reports_a_decision_only_the_unaffordable_can_change() -> None:
    """Reachable but unaffordable is a different failure, and a subtler one: the
    graph permits recourse, §7's prices make it impossible."""
    priced_out = _toy([("a", "y"), ("b", "c")], labels={"a": "infinite"})
    issues = check_reachability(priced_out)
    assert {i.code for i in issues if i.level == "error"} == {"no_effective_action"}
    assert any(i.code == "inert_action" and i.level == "warning" for i in issues), \
        "'b' cannot influence the target and the reader should be told"


def test_the_validator_passes_the_graphs_the_paper_ships(scm: SCM) -> None:
    """The structural checks, on the real graphs: no errors, whatever the data says."""
    issues = [*check_acyclic(scm), *check_reachability(scm), *check_semantics(scm)]
    errors = [str(i) for i in issues if i.level == "error"]
    assert not errors, "; ".join(errors)


@pytest.mark.slow
def test_the_ci_tests_are_labelled_rather_than_enforced() -> None:
    """§6.4's check 3 warns; it does not fail, and this test does not either.

    Real data disagrees with any hand-drawn graph — German has 800 training rows
    and a target with ten parents, so several asserted edges come back
    contradicted, and the paper reports that as a limitation.  What has to hold is
    that every test is *labelled*: a verdict from the fixed vocabulary, a q-value
    on the unit interval, and one edge test per asserted edge, so a reader can see
    which claims the data supports instead of taking the graph on faith.
    """
    scm = get_scm("german")
    frame = training_frame("german")
    tests, n_dsep_total = run_ci_tests(scm, frame, max_tests=40)
    assert tests, "no conditional-independence test ran at all"
    assert n_dsep_total >= 0
    assert {t.verdict for t in tests} <= {"supported", "weak", "contradicted",
                                          "skipped"}
    for test in tests:
        assert test.family in ("edge", "dsep")
        assert 0.0 <= test.q_value <= 1.0 or np.isnan(test.q_value)
        assert test.n > 0
    edges = {(c, p) for c, p, _ in edge_constraints(scm)}
    tested = {(t.candidate, t.child) for t in tests if t.family == "edge"}
    assert tested == edges, "every asserted edge gets its own test"
    verdicts = pd.Series([t.verdict for t in tests]).value_counts().to_dict()
    assert sum(verdicts.values()) == len(tests)


@pytest.mark.slow
def test_validate_scm_produces_a_report_that_can_be_written_down() -> None:
    """The entry point ``run_all.sh`` calls, end to end on one dataset.

    ``figures=False`` because the figure is a §6.4.4 deliverable rather than a
    property — it is produced by the pipeline run, and drawing it here would only
    test matplotlib.  What is asserted is that the report is serialisable, since
    it is written to disk and read back by the paper's tables.
    """
    import json

    report = validate_scm("german", run_ci=True, figures=False, max_tests=20)
    assert report.dataset == "german"
    assert report.ok, [str(i) for i in report.of_level("error")]
    assert report.summary["acyclic"] is True
    assert not report.frame().empty
    assert set(report.frame().columns) >= {"relation", "verdict", "q_value"}
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["ok"] is True
    assert payload["issues"], "a validation report with no findings tells nobody anything"


# ------------------------------------------------- rules every graph must obey


def test_the_graph_is_a_dag_whose_target_is_a_sink(scm: SCM) -> None:
    """A cycle would make abduction meaningless; a non-sink target would let a
    plan change the decision by changing something the decision causes."""
    assert nx.is_directed_acyclic_graph(scm.graph)
    assert scm.children(scm.target) == ()
    assert scm.parents(scm.target), f"{scm.target!r} has no causes at all"


def test_recourse_is_possible_under_the_graph(scm: SCM) -> None:
    """At least one affordable action has a directed path to the decision.

    §6.4's check 2.  If it failed, every method would be searching for something
    the graph says cannot exist, and CAC — the only one that reads the graph —
    would be the only one to report the impossibility.
    """
    reaching = [f for f in scm.actionable if nx.has_path(scm.graph, f, scm.target)]
    assert reaching, "no affordable feature can influence the target"


def test_features_nobody_can_change_are_priced_that_way(scm: SCM) -> None:
    """§6's ``non_manipulable`` and §7's infinite cost have to agree.

    They are two declarations of one fact, in two files, and a method reads
    whichever is nearer.  If they disagreed, the baselines (which mask by
    ``non_manipulable``) and the oracle (which enumerates by cost) would be
    solving different problems.
    """
    for node in scm.non_manipulable:
        assert scm.is_infinite(node), f"{node!r} is non-manipulable but affordable"
        assert node not in scm.actionable
    assert set(scm.manipulable) == set(scm.features) - set(scm.non_manipulable)
    for node in scm.actionable:
        assert not scm.is_infinite(node)


def test_downstream_is_the_set_that_must_be_recomputed(scm: SCM) -> None:
    """``downstream`` is what CAC propagates and what §9 audits, so it is checked
    against ``networkx`` rather than against itself."""
    for node in scm.manipulable:
        expected = nx.descendants(scm.graph, node) - {node, scm.target}
        assert set(scm.downstream(node)) == expected
        assert set(scm.downstream(node, include_target=True)) == expected | (
            {scm.target} if nx.has_path(scm.graph, node, scm.target) else set())
        order = list(scm.nodes)
        got = list(scm.downstream(node))
        assert got == sorted(got, key=order.index), "not in topological order"
