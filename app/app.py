"""Black Box -> Actionable Insight --- the interactive demo entrypoint.

A multipage Streamlit app that walks a non-expert judge from the problem to the
evidence in six pages:

    Home       the 30-second hook: one applicant, one failing baseline, CAC holds
    The Gap    the thesis chart across the whole population
    Head-to-head   every method's plan for one applicant, audited
    Live sandbox   the judge drives: type a plan, watch the world react
    Causal map     the world-model doing the judging
    Is this real?  significance, robustness and the causal ablation

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
# site-packages ``src`` before the first-party imports run.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui import (  # noqa: E402
    page_applicant,
    page_evidence,
    page_gap,
    page_home,
    page_map,
    page_sandbox,
    theme,  # noqa: E402
)
from app.ui import state as S  # noqa: E402

st.set_page_config(page_title="Black Box -> Actionable Insight",
                   page_icon="\U0001F9ED", layout="wide")


def _warm_up(avail: S.Availability) -> None:
    """Pre-load the default dataset's engine + a CAC plan so the first click is fast.

    One-shot per session, guarded so a missing live engine never blocks the app.
    """
    if st.session_state.get("_warmed"):
        return
    st.session_state["_warmed"] = True
    dataset = S.current_dataset()
    if not dataset or dataset not in avail.live:
        return
    try:
        from app import live as L  # noqa: PLC0415

        eng = L.get_engine(dataset)
        records = L.rejected_records(dataset, 5)
        if records:
            L.generate_plans(dataset, "cac", int(records[0]), 3)
            _ = eng  # touched so the engine is cached
    except Exception:  # noqa: BLE001 - warm-up is best-effort
        pass


def main() -> None:
    avail = S.availability()
    theme.inject_css(S.demo_mode())
    S.sidebar(avail)

    if not (avail.any_results or avail.any_live):
        st.title("Black Box -> Actionable Insight")
        st.error("No artefacts found. Run `make all` in the project root "
                 "(`cac-blackbox-to-action/`) to generate results and models, "
                 "then reload.")
        return

    _warm_up(avail)

    pages = [
        st.Page(page_home.render, title="Home", icon="\U0001F3E0", url_path="home",
                default=True),
        st.Page(page_gap.render, title="The Gap", icon="\U0001F4CA", url_path="gap"),
        st.Page(page_applicant.render, title="Head to head", icon="\u2696\uFE0F",
                url_path="applicant"),
        st.Page(page_sandbox.render, title="Live sandbox", icon="\U0001F39B\uFE0F",
                url_path="sandbox"),
        st.Page(page_map.render, title="Causal map", icon="\U0001F578\uFE0F",
                url_path="map"),
        st.Page(page_evidence.render, title="Is this real?", icon="\U0001F52C",
                url_path="evidence"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
