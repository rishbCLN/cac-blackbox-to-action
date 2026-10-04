"""Tests for the plotting layer (``src/viz``) --- global rule #6.

The figures themselves are judged by eye, but the logic *around* them is not:
which method a legend orders first, which applicant the flagship rule picks
without reading its own numbers, how a plan's asked moves are told from the ones
the SCM propagated.  Those are the claims a wrong figure would quietly misstate,
so they are the ones pinned here.  Rendering is exercised only as a smoke test ---
that a real figure comes out at 300 DPI in both formats (rule #12) --- driven by
synthetic frames and stubs so nothing here touches a black box or the disk.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG, DATASETS
from src.viz import (figure, label_of, method_colour, set_style)
import src.viz.plot_counterfactuals as pc
import src.viz.plot_results as pr


# --------------------------------------------------------------------- fixtures

@pytest.fixture
def figures_dir(tmp_path, monkeypatch):
    """Redirect every saved figure into a tmp dir by patching the frozen CONFIG."""
    out = tmp_path / "figs"
    monkeypatch.setattr("src.utils.io.CONFIG",
                        CONFIG.model_copy(update={"FIGURES_DIR": out}))
    return out


class StubCF:
    """The five fields the recourse plot reads off a :class:`Counterfactual`."""

    def __init__(self, feature_names, x_original, x_cf, changed_features):
        self.feature_names = tuple(feature_names)
        self.x_original = np.asarray(x_original, dtype=float)
        self.x_cf = np.asarray(x_cf, dtype=float)
        self.changed_features = tuple(changed_features)


class StubCostModel:
    """The bounds and ``pretty`` labels the plot needs, and nothing black-box."""

    def __init__(self, feature_names, bounds):
        self.feature_names = tuple(feature_names)
        self.bounds = np.asarray(bounds, dtype=float)

    def pretty(self, name, value):  # noqa: D102
        return f"{float(value):g}u"


def _headline_frame() -> pd.DataFrame:
    """A minimal headline table with every column the result figures read."""
    rows = []
    for dataset in ("adult", "heloc"):
        for method, val, cau, act, use, cost in [
            ("cac", 1.0, 1.0, 1.0, 1.0, 0.30),
            ("dice", 1.0, 0.6, 1.0, 0.55, 0.10),
            ("wachter", 0.98, 0.3, 0.9, 0.20, 0.08),
        ]:
            rows.append({"dataset": dataset, "method": method, "validity": val,
                         "causal_validity": cau, "actionability": act,
                         "usable": use, "cost_mean": cost})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- __init__ helpers

def test_method_colour_is_case_and_suffix_insensitive():
    from src.viz import PALETTE, NEUTRAL

    assert method_colour("cac") == PALETTE["cac"]
    assert method_colour("CAC") == PALETTE["cac"]
    assert method_colour("cac-nocausal") == PALETTE["cac_nocausal"]
    # An unknown suffix falls back to the nearest registered prefix, then neutral.
    assert method_colour("cac_budget2") == PALETTE["cac"]
    assert method_colour("totally_unknown") == NEUTRAL


def test_label_of_maps_known_keys_and_passes_the_rest_through():
    assert label_of("cac") == "CAC"
    assert label_of("heloc") == "HELOC"
    assert label_of("diabetes") == "Diabetes"
    assert label_of("Made up") == "Made up"


def test_set_style_latches_after_one_call():
    import src.viz as viz

    viz._STYLED = False
    set_style()
    assert viz._STYLED is True
    set_style()  # idempotent, must not raise
    fig, ax = figure(1, 1)
    assert ax is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


# ------------------------------------------------------------- plot_results logic

def test_methods_in_orders_by_the_fixed_list_then_alphabetical():
    frame = pd.DataFrame({"method": ["wachter", "zeta", "cac", "alpha"]})
    # cac first (it is in METHODS), then the extras alphabetically.
    assert pr._methods_in(frame) == ["cac", "wachter", "alpha", "zeta"]


def test_datasets_in_follows_config_order():
    frame = pd.DataFrame({"dataset": ["heloc", "adult"]})
    assert pr._datasets_in(frame) == ["adult", "heloc"]


@pytest.mark.parametrize("n,expected", [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)),
                                        (4, (2, 2)), (5, (3, 2))])
def test_grid_is_at_most_two_columns(n, expected):
    assert pr._grid(n) == expected


def test_flat_handles_scalar_1d_and_2d_axes():
    fig, ax = figure(1, 1)
    assert pr._flat(ax, 1) == [ax]
    import matplotlib.pyplot as plt

    plt.close(fig)
    fig, axes = figure(2, 2, squeeze=False)
    assert len(pr._flat(axes, 4)) == 4
    plt.close(fig)


def test_read_names_the_experiment_to_run_when_a_csv_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.viz.plot_results.CONFIG",
                        CONFIG.model_copy(update={"RESULTS_DIR": tmp_path}))
    with pytest.raises(FileNotFoundError, match="run the experiment"):
        pr._read("headline")


def test_result_figures_render_to_both_formats(figures_dir):
    headline = _headline_frame()
    for fn in (pr.plot_usable_gap, pr.plot_recourse_funnel, pr.plot_cost_of_validity):
        paths = fn(headline)
        assert {p.suffix for p in paths} == {".png", ".pdf"}
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)


# ------------------------------------------------- plot_counterfactuals: flagship

def _cases(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame([{"dataset": d, "case": c, "record": r} for d, c, r in rows])


def test_flagship_prefers_the_primary_dataset_over_an_earlier_case_elsewhere():
    # adult only has the *last* case; german has the *first*. Datasets are the
    # outer loop, so the primary dataset wins regardless of the case rank.
    cases = _cases([("german", "cac_only", 3), ("adult", "everyone_agrees", 9)])
    assert pc.default_flagship(cases) == ("adult", "everyone_agrees", 9)


def test_flagship_uses_case_order_within_a_dataset():
    cases = _cases([("german", "side_effect", 5), ("german", "cac_only", 8)])
    assert pc.default_flagship(cases) == ("german", "cac_only", 8)


def test_flagship_takes_the_lowest_record_when_a_pair_has_several():
    cases = _cases([("adult", "cac_only", 40), ("adult", "cac_only", 7)])
    assert pc.default_flagship(cases) == ("adult", "cac_only", 7)


def test_flagship_falls_back_to_the_first_row_for_an_unknown_case():
    cases = _cases([("adult", "brand_new_case", 2)])
    assert pc.default_flagship(cases) == ("adult", "brand_new_case", 2)


def test_flagship_is_none_for_no_cases():
    assert pc.default_flagship(_cases([])) is None


def test_present_cases_is_stable_and_complete():
    cases = _cases([("heloc", "side_effect", 1), ("adult", "everyone_agrees", 2),
                    ("heloc", "cac_only", 3)])
    got = pc._present_cases(cases)
    # adult (primary) first; within heloc, cac_only precedes side_effect.
    assert got == [("adult", "everyone_agrees", 2), ("heloc", "cac_only", 3),
                   ("heloc", "side_effect", 1)]


def test_flagship_rule_mirrors_experiment_05():
    """The viz rule and the paper table's rule must name the same applicant.

    ``plot_counterfactuals`` copies experiment 05's ``FLAGSHIP``/``CASES`` rather
    than importing an experiments module; this test fails loudly if the two ever
    drift, which is the only way they could quietly disagree.
    """
    path = Path(__file__).resolve().parents[1] / "experiments" / "05_qualitative.py"
    spec = importlib.util.spec_from_file_location("experiment_05_qualitative", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.FLAGSHIP == pc.FLAGSHIP_DATASET
    assert tuple(c.name for c in module.CASES) == pc.CASE_ORDER


# ------------------------------------------------- plot_counterfactuals: plan logic

def test_flagged_splits_both_separators_and_ignores_nan():
    assert pc._flagged(pd.Series({"off_features": "a, b", "blocked_features": "c"})) \
        == {"a", "b", "c"}
    assert pc._flagged(pd.Series({"off_features": "x|y", "blocked_features": ""})) \
        == {"x", "y"}
    assert pc._flagged(pd.Series({"off_features": "nan", "blocked_features": np.nan})) \
        == set()


def test_span_guards_zero_width_features():
    cm = StubCostModel(["a", "b"], [[0.0, 4.0], [2.0, 2.0]])
    assert list(pc._span(cm)) == [4.0, 1.0]  # the constant feature becomes 1, not 0


def test_moves_splits_asks_from_scm_consequences_and_normalises():
    cm = StubCostModel(["a", "b", "c"], [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    cf = StubCF(["a", "b", "c"], x_original=[0, 0, 5], x_cf=[5, 2, 5],
                changed_features=["a"])  # a asked; b moved via SCM; c unchanged
    moves = pc._moves(cf, pc._span(cm))
    kinds = {cm.feature_names[j]: (kind, nd) for j, kind, _b, _a, nd in moves}
    assert set(kinds) == {"a", "b"}                 # c did not move, so it is absent
    assert kinds["a"] == ("ask", pytest.approx(0.5))
    assert kinds["b"] == ("follow", pytest.approx(0.2))
    # asks are listed before consequences
    assert moves[0][1] == "ask"


def test_ytick_label_carries_the_value_only_for_an_ask():
    cm = StubCostModel(["a"], [[0.0, 10.0]])
    ask = pc._ytick_label("a", "ask", 0.0, 5.0, cm)
    assert "a" in ask and "0u" in ask and "5u" in ask and "→" in ask
    assert pc._ytick_label("a", "follow", 0.0, 5.0, cm) == "a"


def test_recourse_plan_renders_from_stubs(figures_dir):
    dataset = DATASETS[0]
    names = ["a", "b", "c"]
    cm = StubCostModel(names, [[0, 10], [0, 10], [0, 10]])
    cac = StubCF(names, [0, 0, 5], [3, 1, 5], ["a"])   # asks a, SCM moves b
    dice = StubCF(names, [0, 0, 5], [8, 0, 5], ["a"])  # asks a only, no follow
    cases = pd.DataFrame([
        {"dataset": dataset, "case": "demo", "record": 0, "method": "cac",
         "answered": True, "rank": 0, "valid": True, "causal": True,
         "actionable": True, "usable": True, "cost": 0.30,
         "off_features": "", "blocked_features": ""},
        {"dataset": dataset, "case": "demo", "record": 0, "method": "dice",
         "answered": True, "rank": 0, "valid": True, "causal": False,
         "actionable": True, "usable": False, "cost": 0.10,
         "off_features": "b", "blocked_features": ""},
    ])
    paths = pc.plot_recourse_plans(
        dataset, "demo", 0, cases=cases,
        results_cache={dataset: {"cac": {0: [cac]}, "dice": {0: [dice]}}},
        cost_cache={dataset: cm})
    assert {p.suffix for p in paths} == {".png", ".pdf"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_recourse_plan_rejects_an_absent_case(figures_dir):
    cases = _cases([("adult", "demo", 0)])
    with pytest.raises(ValueError, match="no adult/missing"):
        pc.plot_recourse_plans("adult", "missing", cases=cases,
                               results_cache={"adult": {}}, cost_cache={"adult": None})
