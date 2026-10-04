"""The experiment drivers' own helpers — the code no other module can check.

Everything in ``experiments/`` is glue: it loads a dataset, asks a method for
plans, and hands them to :mod:`src.evaluation`, all of which is tested elsewhere.
Three things in there are not glue: two in experiment 04, one in experiment 05.

:func:`_perturb` *manufactures a wrong graph*, and it is the only place in the
project that constructs an SCM nobody wrote by hand.  If it silently produced the
true graph the misspecification arm would report perfect robustness, and the
number would be indistinguishable from a real finding.  So the tests pin down
what each perturbation is supposed to change (the edit list, its size) and what
it must never change: the node set, the target, and above all who can act on
what — a wrongly drawn arrow does not make anybody's age mutable, and if it did,
the arm would be varying two things at once and its conclusion would mean
nothing.  One test spends two interpreters checking that the *same* wrong graph
comes back next time, because the first version of this code seeded itself from
:func:`hash`, which Python salts per process.

:func:`_pool_seeds` *averages the draws*, and an average that silently counted a
skipped draw, or that took the mean of a column it should have taken the max of,
would put a number in the paper that no single run produced.

Experiment 05's ``_pick_*`` rules *decide which applicant the paper prints*, and
they are the only code here whose output is a sentence rather than a number.  A
rule that selected on one property and narrated another would make the case study
a misquotation of its own table — which happened twice while the file was written
(a rule read the worst of five suggestions and the prose printed the cheapest; a
rule accused a random-search control of undercutting CAC).  The tests below build
one synthetic query set where each rule has exactly one right answer, so a rule
that drifts back to the wrong column names the wrong record and fails.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import pytest

from src.config import CONFIG
from src.scm.define_scm import SCM, get_scm

ROOT = Path(__file__).resolve().parents[1]

#: Cheap graph, dense enough that every perturbation has something to bite on.
DATASET = "german"


def _load(name: str) -> Any:
    """Import an ``experiments/NN_*.py`` module, whose name is not an identifier.

    The module goes into :data:`sys.modules` before it is executed, which is not
    optional: it defines a dataclass under ``from __future__ import annotations``,
    and :mod:`dataclasses` resolves those string annotations by looking its own
    module up there.
    """
    path = ROOT / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"experiment_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exp04() -> Any:
    return _load("04_robustness")


@pytest.fixture(scope="module")
def scm() -> SCM:
    return get_scm(DATASET)


# --------------------------------------------------- what a perturbation may do

def test_every_perturbation_leaves_a_dag_over_the_same_world(exp04: Any,
                                                             scm: SCM) -> None:
    """The graph may be wrong; the world may not.

    The arm's whole claim to attribution is that only the propagation changed.
    If a perturbation also moved the node set, the target, or who may act on
    what, then a drop in causal validity could be a drop in the size of the
    search space instead, and the row would be unreadable.

    Compared as sets, not as tuples: ``SCM.features`` and ``SCM.actionable`` are
    reported in topological order, so a perturbed graph *should* reorder them —
    that ordering is the propagation order and it is the thing being varied.  The
    column layout of ``X`` comes from the dataset spec and is untouched by any of
    this.
    """
    for kind in ("true",) + exp04.PERTURBATIONS:
        for seed in CONFIG.ROBUSTNESS_SEEDS:
            wrong, report = exp04._perturb(scm, kind, int(seed))
            assert nx.is_directed_acyclic_graph(wrong.graph), f"{kind}#{seed}: cycle"
            assert set(wrong.graph.nodes) == set(scm.graph.nodes)
            assert wrong.target == scm.target
            assert set(wrong.features) == set(scm.features)
            assert set(wrong.actionable) == set(scm.actionable)
            assert wrong.non_manipulable == scm.non_manipulable
            assert wrong.sensitive == scm.sensitive
            assert wrong.costs == scm.costs
            assert report["perturbation"] == kind and report["seed"] == int(seed)


def test_the_target_keeps_its_parents(exp04: Any, scm: SCM) -> None:
    """No perturbation touches an edge into the target.

    Those edges are the only ones with no effect on anything a method computes:
    the target is not a column of ``X``, so the FCM neither fits nor propagates
    through it.  Perturbing them would pad the arm with rows that cannot move and
    understate how much the structure matters.
    """
    before = set(scm.graph.in_edges(scm.target))
    for kind in ("true",) + exp04.PERTURBATIONS:
        for seed in CONFIG.ROBUSTNESS_SEEDS:
            wrong, _ = exp04._perturb(scm, kind, int(seed))
            assert set(wrong.graph.in_edges(scm.target)) == before, kind


def test_the_edit_distance_is_the_edit_that_was_made(exp04: Any, scm: SCM) -> None:
    """``shd`` counts the edits, and each kind makes the number of them it says.

    A perturbation that reported two edits and made one would put CAC's
    robustness on the wrong row of the table — the reader would read the
    resilience of a one-edge error as resilience to a two-edge one.
    """
    expected = {"true": (0, 0), "drop1": (1, 0), "drop2": (2, 0),
                "add1": (0, 1), "reverse1": (1, 1)}
    for kind, (n_removed, n_added) in expected.items():
        for seed in CONFIG.ROBUSTNESS_SEEDS:
            wrong, report = exp04._perturb(scm, kind, int(seed))
            assert (report["n_removed"], report["n_added"]) == (n_removed, n_added)
            assert report["shd"] == n_removed + n_added
            assert report["n_edges"] == wrong.graph.number_of_edges()
            assert report["n_edges"] == scm.graph.number_of_edges() \
                                       - n_removed + n_added
            edits = [] if report["detail"] == "-" else report["detail"].split("; ")
            assert len(edits) == n_removed + n_added
            assert sum(1 for e in edits if e.startswith("-")) == n_removed
            assert sum(1 for e in edits if e.startswith("+")) == n_added


def test_cutting_a_lever_believes_an_action_has_no_consequences(exp04: Any,
                                                               scm: SCM) -> None:
    """``cut_lever`` silences one feature the search is allowed to use.

    This is the perturbation the arm needs.  Dropping a uniformly chosen edge
    usually misses the recourse-relevant subgraph entirely and changes nothing
    measurable, which flatters the method; cutting every consequence of an
    *actionable* feature is the analyst error that actually costs an applicant
    something — a lever the model believes is inert.  The lever has to stay
    actionable, or the search would simply stop using it and the plan would be
    right by omission.
    """
    for seed in CONFIG.ROBUSTNESS_SEEDS:
        wrong, report = exp04._perturb(scm, "cut_lever", int(seed))
        assert report["n_added"] == 0 and report["n_removed"] > 0
        lever = report["detail"].split("->")[0].lstrip("-")
        assert lever in scm.actionable
        assert lever in wrong.actionable
        assert [v for v in wrong.graph.successors(lever) if v != scm.target] == []
        assert [v for v in scm.graph.successors(lever) if v != scm.target] != []


def test_emptying_the_graph_leaves_the_features_but_no_structure(exp04: Any,
                                                                scm: SCM) -> None:
    """``empty`` is the bound: nothing propagates, so CAC degrades to a point method.

    It is the row that should reproduce ``cac_no_causal`` in
    ``results/ablation.csv``, and it only does so if every feature-to-feature edge
    is gone while the target's parents and the node set stay — an FCM over this
    graph must fit zero equations and treat every feature as a root.
    """
    wrong, report = exp04._perturb(scm, "empty", 0)
    assert exp04._feature_edges(wrong) == []
    assert report["n_added"] == 0
    assert report["n_removed"] == len(exp04._feature_edges(scm))
    assert wrong.graph.number_of_edges() == scm.graph.in_degree(scm.target)


_PROBE = """
import sys
sys.path.insert(0, {root!r})
import importlib.util
spec = importlib.util.spec_from_file_location("exp04", {path!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from src.scm.define_scm import get_scm
scm = get_scm({dataset!r})
print("|".join(module._perturb(scm, kind, seed)[1]["detail"]
               for kind in module.PERTURBATIONS for seed in (0, 1, 2)))
"""


def test_the_same_wrong_graph_comes_back_in_a_fresh_interpreter(exp04: Any) -> None:
    """Rule #2, in the one place it was actually broken.

    The first version of :func:`_perturb` mixed the dataset into its seed with
    :func:`hash`, which Python salts per process: the arm would have reported a
    different set of misspecified graphs on every run, and two rows of the paper
    would not have described the same experiment.  Two interpreters with
    different hash seeds are the only test that would have caught it.
    """
    script = _PROBE.format(root=str(ROOT),
                           path=str(ROOT / "experiments" / "04_robustness.py"),
                           dataset=DATASET)
    out = []
    for hash_seed in ("1", "2"):
        done = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, cwd=str(ROOT), timeout=600,
                              env={**dict(__import__("os").environ),
                                   "PYTHONHASHSEED": hash_seed,
                                   "PYTHONIOENCODING": "utf-8"})
        assert done.returncode == 0, done.stderr[-2000:]
        out.append(done.stdout.strip().splitlines()[-1])
    assert out[0] == out[1], "the perturbations depend on something process-local"
    assert out[0].count("|") == 3 * len(exp04.PERTURBATIONS) - 1


# ------------------------------------------------ how the draws reach the table

def _pooled_frame() -> pd.DataFrame:
    """Two draws of one perturbation that disagree, and a reference row."""
    return pd.DataFrame([
        {"dataset": "german", "method": "cac/drop1#0", "perturbation": "drop1",
         "n_records": 20, "coverage": 1.0, "causal_validity": 1.0, "usable": 1.0,
         "shd": 1},
        {"dataset": "german", "method": "cac/drop1#1", "perturbation": "drop1",
         "n_records": 20, "coverage": 1.0, "causal_validity": 0.5, "usable": 0.5,
         "shd": 1},
        {"dataset": "german", "method": "cac/true", "perturbation": "true",
         "n_records": 20, "coverage": 1.0, "causal_validity": 1.0, "usable": 1.0,
         "shd": 0},
    ])


def test_pooling_reports_the_mean_the_spread_and_how_many_draws(exp04: Any) -> None:
    """Three draws of one misspecification are three samples, not three rows.

    The spread is the finding: a method that survives the average wrong graph but
    collapses on one draw is not robust, and a table showing only the mean would
    hide exactly that.  ``n_seeds`` has to be the number of draws that *ran* —
    a perturbation that changed nothing is skipped upstream, and counting it
    would pull the mean towards the true graph.
    """
    out = exp04._pool_seeds(_pooled_frame())
    assert list(out["perturbation"]) == ["true", "drop1"], "reference row first"
    drop1 = out[out["perturbation"] == "drop1"].iloc[0]
    assert drop1["causal_validity"] == pytest.approx(0.75)
    assert drop1["causal_validity_std"] == pytest.approx(0.3535533906, abs=1e-6)
    assert int(drop1["n_seeds"]) == 2
    assert int(drop1["n_records"]) == 20, "records are a count, not something to mean"
    true = out[out["perturbation"] == "true"].iloc[0]
    assert true["causal_validity_std"] == 0.0, "one draw has no spread, not NaN"


def test_the_spread_reaches_the_latex_beside_the_mean(exp04: Any) -> None:
    """The one place the CSV and the table are allowed to differ.

    A reader of the table needs ``0.75 ± 0.35`` in one cell; a reader of the CSV
    would rather have two columns.  If the formatting silently dropped the
    spread, the table would claim a precision the experiment does not have.
    """
    pooled = exp04._pool_seeds(_pooled_frame())
    table = exp04._labelled(pooled, exp04._SCM_COLUMNS)
    assert list(table["Dataset"]) == ["German", "German"]
    assert table["Graph given to CAC"].iloc[1] == exp04._PERTURBATION_LABELS["drop1"]
    assert table["Causal"].iloc[1] == "$0.75 \\pm 0.35$"
    assert "causal_validity_std" not in table.columns


def test_a_pooled_edit_count_is_not_rounded_to_a_graph_nobody_made(exp04: Any
                                                                  ) -> None:
    """``SHD`` prints as a count where it is one and as a mean where it is not.

    Pooling three draws of "drop one edge" can average to 1.33 edits, and casting
    that to an integer both raises and — if it did not — would name a perturbation
    that never happened.  The whole-number case still has to print bare, because a
    column reading ``2.000 edges`` is noise.
    """
    frame = pd.DataFrame([{"dataset": "german", "perturbation": "true", "shd": 0.0,
                           "n_seeds": 1, "causal_validity": 1.0},
                          {"dataset": "german", "perturbation": "drop_edge",
                           "shd": 4 / 3, "n_seeds": 3, "causal_validity": 0.9}])
    columns = {"shd": "SHD", "n_seeds": "Draws", "causal_validity": "Causal"}
    out = exp04._labelled(frame, columns)
    assert list(out["SHD"]) == ["0", "1.33"], "a mean of draws, and a true graph"
    whole = exp04._labelled(frame.assign(shd=[0.0, 2.0]), columns)
    assert str(whole["SHD"].dtype) == "Int64" and list(whole["SHD"]) == [0, 2]
    assert list(out["Draws"]) == [1, 3] and str(out["Draws"].dtype) == "Int64"


def test_the_cells_are_ordered_by_what_they_mean(exp04: Any) -> None:
    """``k=10`` after ``k=3``, and the deployed black box above its replacements.

    Alphabetical order is the default and it is wrong in both arms: it puts
    ``k=10`` second and the secondary architecture first, so a reader scanning
    down the table reads a trend that is an artefact of spelling.
    """
    frame = pd.DataFrame([{"dataset": "german", "arm": "k", "base_method": m,
                           "k": k, "method": f"{m}/k={k}"}
                          for m in ("dice", "cac") for k in (10, 3, 1, 5)])
    out = exp04._ordered(frame)
    assert list(out["method"]) == ["cac/k=1", "cac/k=3", "cac/k=5", "cac/k=10",
                                  "dice/k=1", "dice/k=3", "dice/k=5", "dice/k=10"]
    audits = ["refit2", "mlp", CONFIG.BLACKBOX_TYPE, "refit0"]
    assert sorted(audits, key=exp04._audit_rank)[0] == CONFIG.BLACKBOX_TYPE
    assert sorted(audits, key=exp04._audit_rank)[-1] == "refit2"


# --------------------------------------------------- which applicant gets printed

@pytest.fixture(scope="module")
def exp05() -> Any:
    return _load("05_qualitative")


#: One suggestion per row, in the shape :func:`feasibility.join_audits` hands over.
#: Five applicants, three methods, and each of the five cases has exactly one right
#: answer — see :func:`joined` for what each record is built to catch.
_SUGGESTIONS: tuple[dict[str, Any], ...] = (
    # Record 0 — CAC only.  ``dice`` passes all three audits, but with two
    # different suggestions: the cheap one flips without the SCM's agreement, the
    # dear one keeps its promise by asking for something immutable.  Neither is
    # usable, and the plan it would ship (the cheap one) has no side effects at
    # all, while the one it would not ship disagrees with the SCM everywhere.
    dict(record=0, method="cac", rank=0, valid=1, promise_kept=1, actionable=1,
         cost=0.30, gap_max=0.00, immutable=0),
    dict(record=0, method="dice", rank=0, valid=1, promise_kept=0, actionable=1,
         cost=0.35, gap_max=0.00, immutable=0),
    dict(record=0, method="dice", rank=1, valid=1, promise_kept=1, actionable=0,
         cost=0.40, gap_max=0.90, immutable=1),
    dict(record=0, method="random", rank=0, valid=0, promise_kept=0, actionable=0,
         cost=0.05, gap_max=0.00, immutable=0),
    # Record 1 — cheaper but wrong.  ``dice`` undercuts CAC with a bargain the SCM
    # rejects and also offers a dearer plan that is fine, so the case has to read
    # the cheapest plan's verdict rather than the method's.  ``random`` is cheaper
    # still and causally sound: if the accusation ranged over it, the undercutting
    # plan would be a sound one and the case would not fire at all.
    dict(record=1, method="cac", rank=0, valid=1, promise_kept=1, actionable=1,
         cost=0.50, gap_max=0.00, immutable=0),
    dict(record=1, method="dice", rank=0, valid=1, promise_kept=0, actionable=1,
         cost=0.20, gap_max=0.00, immutable=0),
    dict(record=1, method="dice", rank=1, valid=1, promise_kept=1, actionable=1,
         cost=0.60, gap_max=0.00, immutable=0),
    dict(record=1, method="random", rank=0, valid=1, promise_kept=1, actionable=0,
         cost=0.05, gap_max=0.00, immutable=0),
    # Record 2 — everybody serves this applicant.
    dict(record=2, method="cac", rank=0, valid=1, promise_kept=1, actionable=1,
         cost=0.10, gap_max=0.00, immutable=0),
    dict(record=2, method="dice", rank=0, valid=1, promise_kept=1, actionable=1,
         cost=0.15, gap_max=0.00, immutable=0),
    dict(record=2, method="random", rank=0, valid=1, promise_kept=1, actionable=1,
         cost=0.90, gap_max=0.00, immutable=0),
    # Record 3 — side effect: the plan ``dice`` would hand over is the one the SCM
    # disagrees with, so selection and narration point at the same sentence.
    dict(record=3, method="cac", rank=0, valid=1, promise_kept=1, actionable=1,
         cost=0.20, gap_max=0.00, immutable=0),
    dict(record=3, method="dice", rank=0, valid=1, promise_kept=0, actionable=1,
         cost=0.20, gap_max=0.55, immutable=0),
    dict(record=3, method="random", rank=0, valid=1, promise_kept=1, actionable=0,
         cost=0.30, gap_max=0.00, immutable=0),
    # Record 4 — CAC and dice both serve this applicant and ``random`` does not, so
    # unanimity must fail: it is the control's presence in the denominator that
    # makes "every method" mean every method.
    dict(record=4, method="cac", rank=0, valid=1, promise_kept=1, actionable=1,
         cost=0.10, gap_max=0.00, immutable=0),
    dict(record=4, method="dice", rank=0, valid=1, promise_kept=1, actionable=1,
         cost=0.12, gap_max=0.00, immutable=0),
    dict(record=4, method="random", rank=0, valid=1, promise_kept=1, actionable=0,
         cost=0.11, gap_max=0.00, immutable=0),
    # Record 5 — the accusation record 1 only looks like.  ``dice`` has nothing
    # usable, so the plan it hands over is the cheap one, it costs less than CAC's,
    # and the SCM does not sign it off — here because the flip does not survive
    # repair rather than because the plan strays, which is why its gap is zero.
    # ``random`` is cheaper still and sound: counted, it would clear the case.
    dict(record=5, method="cac", rank=0, valid=1, promise_kept=1, actionable=1,
         cost=0.40, gap_max=0.00, immutable=0),
    dict(record=5, method="dice", rank=0, valid=1, promise_kept=0, actionable=1,
         cost=0.15, gap_max=0.00, immutable=0),
    dict(record=5, method="random", rank=0, valid=1, promise_kept=1, actionable=1,
         cost=0.05, gap_max=0.00, immutable=0),
)


@pytest.fixture(scope="module")
def joined() -> pd.DataFrame:
    """:data:`_SUGGESTIONS` as a frame, with the funnel's conjunction applied.

    ``usable`` is derived here the way :func:`~src.evaluation.feasibility.join_audits`
    derives it, and that function's own tests are what keep the two definitions the
    same; a fixture that hard-coded the column could tell experiment 05 a record is
    usable that the headline table counts as a failure.
    """
    frame = pd.DataFrame(list(_SUGGESTIONS)).astype({
        "valid": bool, "promise_kept": bool, "actionable": bool})
    frame["usable"] = frame["valid"] & frame["promise_kept"] & frame["actionable"]
    frame["sensitive_shift"] = 0.0
    return frame


@pytest.fixture(scope="module")
def comparison(exp05: Any, joined: pd.DataFrame) -> pd.DataFrame:
    return exp05._comparison(exp05._per_record(joined))


def test_a_record_is_scored_by_the_plan_the_method_would_ship(exp05: Any,
                                                             joined: pd.DataFrame
                                                             ) -> None:
    """``dice`` on record 0: three audits passed, no single suggestion passing all.

    The per-record funnel is a conjunction over suggestions, not a conjunction of
    per-method rates, and this is the record that tells them apart.  ``gap_shown``
    is the same distinction again in the other direction: the plan being narrated
    has no side effects, and the one with all of them is a plan ``dice`` would
    never hand over.
    """
    per = exp05._per_record(joined)
    dice = per[(per["record"] == 0) & (per["method"] == "dice")].iloc[0]
    assert (bool(dice["valid"]), bool(dice["causal"]), bool(dice["actionable"])) \
        == (True, True, True)
    assert not bool(dice["usable"]), "no one suggestion passed all three"
    assert dice["cost_valid"] == pytest.approx(0.35)
    assert pd.isna(dice["cost_usable"])
    assert dice["gap_max"] == pytest.approx(0.90)
    assert dice["gap_shown"] == pytest.approx(0.00)
    assert int(dice["immutable"]) == 1
    cac = per[(per["record"] == 0) & (per["method"] == "cac")].iloc[0]
    assert bool(cac["usable"]) and cac["cost_usable"] == pytest.approx(0.30)


def test_the_shipped_plan_carries_its_own_verdict(exp05: Any,
                                                  joined: pd.DataFrame) -> None:
    """Record 1's ``dice`` is causally sound, ships a sound plan, and has a bad bargain.

    Three different answers to "is dice's advice real here", and the effort
    comparison needs the third one.  ``causal`` asks whether any plan keeps its
    promise; ``causal_shown`` asks it of the plan the applicant is handed; the
    cheapest flipping plan is neither, and a rule that priced *it* would accuse a
    method over a suggestion it would not have given.
    """
    per = exp05._per_record(joined)
    dice = per[(per["record"] == 1) & (per["method"] == "dice")].iloc[0]
    assert bool(dice["causal"]) and bool(dice["usable"])
    assert bool(dice["causal_shown"]), "the plan it ships is the sound one"
    assert dice["cost_valid"] == pytest.approx(0.20), "the bargain that is not real"
    assert dice["cost_shown"] == pytest.approx(0.60)
    assert dice["cost_usable"] == pytest.approx(0.60)
    failed = per[(per["record"] == 0) & (per["method"] == "random")].iloc[0]
    assert pd.isna(failed["cost_shown"]) and not bool(failed["causal_shown"]), \
        "the price of a plan that does not flip is not a price"


def test_only_a_library_can_be_accused_of_undercutting(comparison: pd.DataFrame
                                                       ) -> None:
    """Record 5: ``random`` ships the cheapest plan of all and must not be the winner.

    The ``best_``/``worst_`` families range over :data:`LIBRARY` alone, and this
    record is built so the exclusion is load-bearing twice over — ``random`` is
    cheaper than the accused plan, and causally sound, so counting it would both
    misname the undercutter and clear it.
    """
    row = comparison[comparison["record"] == 5].iloc[0]
    assert row["best_cost_shown"] == pytest.approx(0.15), "random must not compete"
    assert row["best_cost_shown_by"] == "dice"
    assert not bool(row["undercut_causal"])
    # Record 1's undercutting plan is one dice would not hand over, and the same
    # columns have to say so or the accusation lands on the wrong suggestion.
    sound = comparison[comparison["record"] == 1].iloc[0]
    assert sound["best_cost_shown"] == pytest.approx(0.60)
    assert bool(sound["undercut_causal"])


def test_unanimity_is_asked_of_every_method_including_the_control(
        comparison: pd.DataFrame) -> None:
    """The ``any_``/``all_`` families keep ``random``, so "everyone" means everyone."""
    unanimous = comparison[comparison["record"] == 2].iloc[0]
    assert bool(unanimous["all_usable"]) and bool(unanimous["any_usable"])
    partial = comparison[comparison["record"] == 4].iloc[0]
    assert bool(partial["any_usable"]) and not bool(partial["all_usable"]), \
        "cac and dice serve record 4; the control does not"


def test_each_case_selects_the_record_it_describes(exp05: Any,
                                                   comparison: pd.DataFrame) -> None:
    """The five rules against a query set built to have one right answer each.

    Written as whole hit sets rather than "the top pick is right", because the
    count is what the frequency table publishes: a rule that names the right
    applicant and the wrong denominator still misstates how often the phenomenon
    happens.
    """
    hits = {case.name: [int(r) for r in case.pick(comparison)] for case in exp05.CASES}
    # Records 0, 3 and 5 all have CAC usable where ``dice`` is not, and the wider
    # side effect goes first: the case is illustrated best by its worst example.
    assert hits["cac_only"] == [0, 3, 5]
    assert hits["cheaper_but_wrong"] == [5], "record 1's bargain is not what dice ships"
    assert hits["immutable_advice"] == [0]
    assert hits["side_effect"] == [3], "record 0's disagreement is in a plan " \
                                       "dice would not hand over"
    assert hits["everyone_agrees"] == [2]


def test_the_rules_survive_a_run_with_no_baselines(exp05: Any,
                                                   joined: pd.DataFrame) -> None:
    """A CAC-only results file leaves every comparison column missing.

    Not a hypothetical: the rules read columns that only exist when somebody was
    there to compare against, and each of them has to answer "no such case" rather
    than raise on a run that asked for one method.
    """
    alone = exp05._comparison(exp05._per_record(joined[joined["method"] == "cac"]))
    for case in exp05.CASES:
        assert exp05._records(case.pick(alone)) == [], case.name


def test_a_case_with_no_examples_reports_none_of_them(exp05: Any,
                                                      joined: pd.DataFrame) -> None:
    """An empty hit set stays empty through the ranking that orders it.

    On adult no applicant is undercut by a baseline's shipped advice, and the rule
    ranked its (empty) hits by a saving computed over the whole table — which put
    all 100 records back, so a case with no examples published the query set as its
    own frequency.  Every rule is checked, because each of them ranks.
    """
    sound = joined[~((joined["method"] == "dice") & ~joined["promise_kept"])]
    comparison = exp05._comparison(exp05._per_record(sound))
    for case in exp05.CASES:
        hits = exp05._records(case.pick(comparison))
        assert len(hits) <= int(comparison.shape[0]), case.name
        assert set(hits) <= set(int(r) for r in comparison["record"]), case.name
    assert exp05._records(exp05._pick_cheaper_but_wrong(comparison)) == [], \
        "dice ships nothing the SCM rejects here"


def test_the_narrative_shows_the_suggestion_the_case_is_about(exp05: Any,
                                                             joined: pd.DataFrame
                                                             ) -> None:
    """``_prefer`` ships the usable plan, except where the case is about a rule.

    ``immutable_advice`` counts records where *one of* the plans a method offers is
    impossible, so it narrates that plan — all ``k`` are put to the applicant, and
    a page printing the method's best suggestion under that heading would be
    describing a different one.
    """
    dice1 = joined[(joined["record"] == 1) & (joined["method"] == "dice")]
    assert int(exp05._prefer(dice1, "usable").iloc[0]["rank"]) == 1, \
        "the cheaper plan flips but the SCM rejects it"
    assert int(exp05._prefer(dice1, "immutable").iloc[0]["rank"]) == 1, \
        "nothing immutable here — fall back to what would be shipped"
    dice0 = joined[(joined["record"] == 0) & (joined["method"] == "dice")]
    assert int(exp05._prefer(dice0, "usable").iloc[0]["rank"]) == 0
    assert int(exp05._prefer(dice0, "immutable").iloc[0]["rank"]) == 1


def test_a_method_that_never_flipped_still_has_something_to_show(exp05: Any,
                                                                joined: pd.DataFrame
                                                                ) -> None:
    """With nothing valid, the cheapest attempt is narrated rather than nothing.

    The alternative is a case study with a blank row where a method's advice should
    be, which reads as missing data rather than as the finding it is: the method
    answered, and its answer does not work.
    """
    failed = joined[(joined["record"] == 0) & (joined["method"] == "random")]
    dearer = failed.assign(rank=1, cost=0.50)
    pool = pd.concat([dearer, failed], ignore_index=True)
    assert not bool(pool["valid"].any())
    assert int(exp05._prefer(pool, "usable").iloc[0]["rank"]) == 0
    assert exp05._prefer(pool, "usable").iloc[0]["cost"] == pytest.approx(0.05)


def test_the_advice_cell_counts_what_it_cannot_print(exp05: Any) -> None:
    """A table cell holds a line of a plan; dropping the rest would understate it."""
    assert exp05._advice([]) == "nothing"
    assert exp05._advice(["age: 30 -> 40 (cost 0.25)"]) == "age: 30 -> 40"
    assert exp05._advice(["age: 30 -> 40", "job: skilled", "savings: rich"]) \
        == "age: 30 -> 40; job: skilled (+1 more)"
    long = ["duration: " + "9" * 90, "job: skilled", "savings: rich"]
    shortened = exp05._advice(long)
    assert shortened.endswith("... (+1 more)") and len(shortened) < len(long[0])


def _cases_frame(pairs: "list[tuple[str, str]]") -> pd.DataFrame:
    """A minimal ``cases`` frame: one method row per (dataset, case) pair."""
    return pd.DataFrame([{"dataset": d, "case": c, "record": i, "method": "cac"}
                         for i, (d, c) in enumerate(pairs)])


def test_the_flagship_is_chosen_by_a_rule_not_by_the_number_it_prints(
        exp05: Any) -> None:
    """The one case printed in full is picked by a fixed walk, blind to its values.

    Dataset order before case order: the primary dataset's *weakest* contrastive
    case outranks the secondary dataset's strongest, so the table cannot drift to
    whichever dataset happens to flatter CAC most.  When the primary supplies
    nothing --- on Adult every method is usable and all four accusing cases are
    empty --- the walk falls through to the next dataset, and the fall-through is
    a warning rather than a silent swap, because the caption names the case.
    """
    first = exp05.CASES[0].name
    order = [exp05.FLAGSHIP] + [d for d in exp05.DATASETS if d != exp05.FLAGSHIP]
    second = order[1]

    # The declared best: primary dataset, first case.
    picked = exp05._flagship(_cases_frame([(second, first), (exp05.FLAGSHIP, first)]))
    assert (picked["dataset"].iloc[0], picked["case"].iloc[0]) == (exp05.FLAGSHIP, first)

    # Primary dataset present but only via a later case beats any other dataset's
    # first case --- dataset priority dominates case priority.
    later = exp05.CASES[-1].name
    picked = exp05._flagship(_cases_frame([(second, first), (exp05.FLAGSHIP, later)]))
    assert (picked["dataset"].iloc[0], picked["case"].iloc[0]) == (exp05.FLAGSHIP, later)

    # Nothing from the primary dataset: fall through in dataset order.
    picked = exp05._flagship(_cases_frame([(order[2], first), (second, later)]))
    assert (picked["dataset"].iloc[0], picked["case"].iloc[0]) == (second, later)

    assert exp05._flagship(pd.DataFrame(columns=["dataset", "case", "record"])).empty

