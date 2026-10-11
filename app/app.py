"""Black Box -> Actionable Insight --- the interactive demo entrypoint.

An eight-act walkthrough that carries a non-expert judge from the hook to the
evidence, wired through :mod:`app.ui.flow` (the step registry + the global
stepper) and :mod:`app.ui.motion` (fast-by-default animation, cinematic on cue):

    0 Cover         the title + the presenter's control room (speed, HUD)
    1 Home          the 30-second hook: one applicant, one failing baseline
    2 The Gap       the thesis chart across the whole population
    3 Head-to-head  every method's plan for one applicant, audited
    4 Live sandbox  the judge drives: type a plan, watch the world react
    5 Causal map    the world-model doing the judging
    6 Is this real? significance, robustness and the causal ablation
    7 Takeaway      the one sentence + the one number to remember

The page list, titles, icons, urls and ordering all come from
:data:`app.ui.flow.STEPS`, so the sidebar nav, the progress rail and the
Back/Next buttons can never disagree.

Run with::

    streamlit run app/app.py        # or: make app

The app reads what the experiments already wrote (``make all``) and, on the
sandbox page, *queries* the trained model live (never trains, never mutates it).
Every page degrades to a friendly banner when its artefacts are missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Launchable from anywhere: make the project root win over any stray
# site-packages ``src`` before the first-party imports run, and remove
# the script's directory from sys.path so 'import app' resolves to the package.
ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent

while str(APP_DIR) in sys.path:
    sys.path.remove(str(APP_DIR))
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

from app.ui import (  # noqa: E402
    page_applicant,
    page_cover,
    page_evidence,
    page_gap,
    page_home,
    page_map,
    page_sandbox,
    page_takeaway,
    theme,
)
from app.ui import flow as F  # noqa: E402
from app.ui import state as S  # noqa: E402

st.set_page_config(page_title="Black Box -> Actionable Insight",
                   page_icon="\U0001F9ED", layout="wide")

#: The render function behind each Act, keyed by its ``flow.STEPS`` key, so the
#: shell and the step registry share one source of truth for ordering.
_RENDERERS = {
    "cover": page_cover.render,
    "problem": page_home.render,
    "gap": page_gap.render,
    "head": page_applicant.render,
    "sandbox": page_sandbox.render,
    "map": page_map.render,
    "evidence": page_evidence.render,
    "takeaway": page_takeaway.render,
}


def _warm_up(avail: S.Availability) -> None:
    """Pre-load the default dataset's engine, a CAC plan and the causal-map figure.

    One-shot per session and fully best-effort: every branch is guarded so a
    missing live engine or SCM never blocks the app --- it just skips that warm-up.
    The payoff is that the judge's first stop on the live sandbox (needs the
    engine + a CAC plan) and the causal map (needs the slow-to-build SCM figure)
    both land on already-cached work.
    """
    if st.session_state.get("_warmed"):
        return
    st.session_state["_warmed"] = True
    dataset = S.current_dataset()
    if not dataset:
        return

    # Live engine + the lead CAC plan for the first rejected applicant, so the
    # sandbox opens instantly.
    if dataset in avail.live:
        try:
            from app import live as L  # noqa: PLC0415

            eng = L.get_engine(dataset)
            records = L.rejected_records(dataset, 5)
            if records:
                L.generate_plans(dataset, "cac", int(records[0]), 3)
            _ = eng  # touched so the engine is cached
        except Exception:  # noqa: BLE001 - warm-up is best-effort
            pass

    # The causal-map figure (matplotlib is slow to rebuild) for datasets with
    # stored results, so Act 5 is instant on arrival.
    if dataset in avail.results:
        try:
            from app import components as C  # noqa: PLC0415

            C.scm_figure(dataset)
        except Exception:  # noqa: BLE001 - a dataset may have no SCM
            pass


def _build_pages() -> tuple[list, dict[int, object]]:
    """The ordered ``st.Page`` list and a ``{step index: page}`` map, from the registry.

    Both views are built in one pass so navigation and ``flow.goto`` (which calls
    ``st.switch_page`` on the mapped page) always reference the same objects.
    """
    pages: list = []
    by_step: dict[int, object] = {}
    for step in F.STEPS:
        page = st.Page(_RENDERERS[step.key], title=step.title, icon=step.icon,
                       url_path=step.url_path, default=(step.index == 0))
        pages.append(page)
        by_step[step.index] = page
    return pages, by_step


def main() -> None:
    avail = S.availability()
    theme.inject_css(S.demo_mode())
    F.inject_motion()
    S.sidebar(avail)

    if not (avail.any_results or avail.any_live):
        st.title("Black Box -> Actionable Insight")
        st.error("No artefacts found. Run `make all` in the project root "
                 "(`cac-blackbox-to-action/`) to generate results and models, "
                 "then reload.")
        return

    _warm_up(avail)

    pages, by_step = _build_pages()
    F.set_pages(by_step)
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
