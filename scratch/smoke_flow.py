"""Headless smoke test for the motion + flow layer --- no Streamlit runtime.

``app.ui.motion`` is pure (string/CSS builders) and ``app.ui.flow``'s step
registry and speed resolution are too, so both run as plain library calls.  This
also *imports every page the registry names* and checks each exposes a callable
``render`` --- the cheapest proof that the eight-act wiring in ``app.py`` resolves
before anyone launches ``streamlit run``.  Run from the project root.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")  # silence "missing ScriptRunContext" noise

from app.ui import flow as F  # noqa: E402
from app.ui import motion as M  # noqa: E402


def check_motion() -> None:
    # presets + reduced-motion override
    assert set(M.SPEED_PRESETS) == set(M.SPEEDS)
    assert M.resolve_speed("nonsense") == M.DEFAULT_SPEED
    assert M.resolve_speed("Cinematic") == "cinematic"
    for sp in M.SPEEDS:
        assert M.resolve_speed(sp, reduced=True) == "instant"
    p0 = M.preset("instant")
    assert p0["entrance"] == 0 and p0["bar"] == 0 and p0["stagger"] == 0
    assert M.confetti_enabled("cinematic")
    assert not M.confetti_enabled("smooth")
    assert not M.confetti_enabled("cinematic", reduced=True)

    # every keyframe + root class present; reduced forces the instant class
    for sp in M.SPEEDS:
        css = M.motion_css(sp)
        for kf in M.KEYFRAMES:
            assert kf in css, f"{sp}: missing keyframe {kf}"
        assert M.root_class(sp) in css
        assert "prefers-reduced-motion" in css
    assert "motion-instant" in M.motion_css("smooth", reduced=True)

    # builders embed their *live* final value and collapse cleanly at instant
    assert "82%" in M.count_up(0.82, "pct")
    assert "var dur = 0" in M.count_up(0.82, "pct", speed="instant")
    assert "cu" in M.count_up(float("nan"), "pct")  # must not emit invalid JS target

    rows = [{"label": "CAC", "validity": 1.0, "usable": 1.0, "is_cac": True},
            {"label": "Wachter", "validity": 0.85, "usable": 0.13, "is_worst": True}]
    gap = M.gap_bars_html(rows, "smooth")
    assert "100%" in gap and "13%" in gap
    assert "worst baseline" in gap and M.COL_CAC in gap
    assert "scaleX(1)" in M.gap_bars_html(rows, "instant")  # final frame, no grow

    assert "USABLE" in M.verdict_stamp_html(True)
    assert "NOT USABLE" in M.verdict_stamp_html(False)
    assert "<canvas" in M.confetti_html()
    for kind in ("card", "bars", "line", "bogus"):
        assert "sk-bar" in M.skeleton(kind)
    print("MOTION OK")


def check_flow() -> None:
    steps = F.STEPS
    assert len(steps) == 8
    assert [s.index for s in steps] == list(range(8))
    assert len({s.key for s in steps}) == 8, "duplicate step keys"
    assert len({s.url_path for s in steps}) == 8, "duplicate url paths"
    assert F.N_STEPS == 8 and F.LAST_STEP == 7

    for step in steps:
        assert step.title and step.label and step.icon and step.module
        fn = F.render_callable(step)  # imports the page module (cover..takeaway)
        assert callable(fn), f"{step.module}.{step.fn} is not callable"
        assert F.step_by_key(step.key) is step
    assert F.step_by_key("missing") is None
    print(f"FLOW OK ({len(steps)} acts: {', '.join(s.key for s in steps)})")


def main() -> int:
    check_motion()
    check_flow()
    print("MOTION+FLOW OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
