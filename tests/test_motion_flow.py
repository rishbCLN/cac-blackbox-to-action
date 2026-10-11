"""Tests for the demo's motion + flow layer (``app/ui/motion.py``, ``app/ui/flow.py``).

The figures and page choreography are judged by eye, but the invariants the live
show rests on are not: the three speed presets and the reduced-motion override,
that every animation builder embeds its *live* final value (so a frozen/Instant
frame is still numerically correct), and that the eight-act step registry is
well-formed and every page it names imports with a callable ``render``.  Those
are pinned here, with no Streamlit runtime --- both modules are import-safe and
their public surface is pure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui import flow as F  # noqa: E402
from app.ui import motion as M  # noqa: E402


# --------------------------------------------------------------- motion: presets
def test_presets_cover_all_speeds() -> None:
    assert set(M.SPEED_PRESETS) == set(M.SPEEDS)
    assert M.DEFAULT_SPEED in M.SPEEDS


@pytest.mark.parametrize("junk", ["", None, "fast", "turbo", "  "])
def test_resolve_speed_falls_back_to_default(junk) -> None:
    assert M.resolve_speed(junk) == M.DEFAULT_SPEED


def test_resolve_speed_is_case_and_space_insensitive() -> None:
    assert M.resolve_speed("  Cinematic ") == "cinematic"


@pytest.mark.parametrize("speed", list(M.SPEEDS))
def test_reduced_motion_always_forces_instant(speed) -> None:
    assert M.resolve_speed(speed, reduced=True) == "instant"
    assert M.root_class(speed, reduced=True) == "motion-instant"


def test_instant_preset_is_zero_duration() -> None:
    p = M.preset("instant")
    assert p["entrance"] == 0 and p["bar"] == 0 and p["stagger"] == 0


def test_confetti_only_fires_on_cinematic() -> None:
    assert M.confetti_enabled("cinematic")
    assert not M.confetti_enabled("smooth")
    assert not M.confetti_enabled("instant")
    assert not M.confetti_enabled("cinematic", reduced=True)


# ------------------------------------------------------------------ motion: css
@pytest.mark.parametrize("speed", list(M.SPEEDS))
def test_motion_css_defines_every_keyframe_and_root(speed) -> None:
    css = M.motion_css(speed)
    for keyframe in M.KEYFRAMES:
        assert keyframe in css, f"{speed} css missing {keyframe}"
    assert M.root_class(speed) in css
    assert "prefers-reduced-motion" in css


def test_reduced_motion_css_uses_instant_class() -> None:
    assert "motion-instant" in M.motion_css("cinematic", reduced=True)


# ---------------------------------------------- motion: builders embed live data
def test_count_up_embeds_the_final_value() -> None:
    assert "82%" in M.count_up(0.82, "pct")
    assert "1.50" in M.count_up(1.5, "float2")
    assert "7" in M.count_up(7, "int")


def test_count_up_instant_is_zero_duration() -> None:
    assert "var dur = 0" in M.count_up(0.82, "pct", speed="instant")


def test_count_up_tolerates_non_finite() -> None:
    # NaN must not crash and must not emit an invalid JS target (bare `nan`).
    html = M.count_up(float("nan"), "pct")
    assert "var target = nan" not in html


def test_gap_bars_embeds_values_and_flags() -> None:
    rows = [{"label": "CAC", "validity": 1.0, "usable": 1.0, "is_cac": True},
            {"label": "Wachter", "validity": 0.85, "usable": 0.13, "is_worst": True}]
    gap = M.gap_bars_html(rows, "smooth")
    assert "100%" in gap and "13%" in gap
    assert "worst baseline" in gap
    assert M.COL_CAC in gap  # the CAC row is emphasised in the CAC colour


def test_gap_bars_instant_renders_final_frame() -> None:
    rows = [{"label": "CAC", "validity": 1.0, "usable": 1.0}]
    assert "scaleX(1)" in M.gap_bars_html(rows, "instant")


def test_verdict_stamp_text() -> None:
    assert "USABLE" in M.verdict_stamp_html(True)
    assert "NOT USABLE" in M.verdict_stamp_html(False)


@pytest.mark.parametrize("kind", ["card", "bars", "line", "unknown"])
def test_skeleton_always_renders_bars(kind) -> None:
    assert "sk-bar" in M.skeleton(kind)


def test_confetti_is_self_contained() -> None:
    html = M.confetti_html()
    assert "<canvas" in html and "requestAnimationFrame" in html


# ----------------------------------------------------------------- flow registry
def test_step_registry_is_well_formed() -> None:
    steps = F.STEPS
    assert len(steps) == 8
    assert [s.index for s in steps] == list(range(8))
    assert len({s.key for s in steps}) == 8
    assert len({s.url_path for s in steps}) == 8
    assert F.N_STEPS == 8 and F.LAST_STEP == 7
    for step in steps:
        assert step.title and step.label and step.icon and step.module


@pytest.mark.parametrize("step", list(F.STEPS), ids=[s.key for s in F.STEPS])
def test_every_registered_page_imports_with_callable_render(step) -> None:
    fn = F.render_callable(step)
    assert callable(fn)


def test_step_by_key_round_trips() -> None:
    for step in F.STEPS:
        assert F.step_by_key(step.key) is step
    assert F.step_by_key("does-not-exist") is None


# ----------------------------------------------------- evidence: ablation hero
def test_ablation_hero_is_flip_survival() -> None:
    """The corrected ablation isolates flip survival, not the near-flat usable."""
    from app.ui import page_evidence as E  # noqa: PLC0415

    assert E._ABLATION_METRIC == "flip_survival"
    assert E._ABLATION_SECONDARY == "state_fidelity"


def test_ablation_flip_survival_is_monotone() -> None:
    """Removing the causal term must never *raise* flip survival (the thesis)."""
    from app import components as C  # noqa: PLC0415
    from app.ui import page_evidence as E  # noqa: PLC0415

    abl = C.load_csv("ablation_headline")
    if abl is None:
        pytest.skip("ablation_headline.csv not present")
    seen = False
    for dataset in sorted(set(abl["dataset"])):
        block = abl[abl["dataset"] == dataset]
        full = E._val(block, "cac", "flip_survival")
        noc = E._val(block, "cac_no_causal", "flip_survival")
        if full == full and noc == noc:  # both non-NaN
            seen = True
            assert full >= noc, f"{dataset}: removing causal raised flip survival"
    assert seen, "no cac/cac_no_causal flip_survival pairs found"
