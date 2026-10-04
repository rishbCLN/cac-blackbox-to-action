"""The Causal Map --- the world-model that does the judging (Phase 5).

Renders the dataset's structural causal graph and lets the judge pick an action
set to see everything it drags along (the downstream cascade).  Also shows the
graph-validation summary honestly, so the 'your graph is wrong' objection is met
head-on and handed to the robustness evidence.
"""

from __future__ import annotations

import streamlit as st

from app import components as C
from app.ui import state as S
from app.ui import theme
from src.viz import label_of


def render() -> None:
    st.title("The causal map: what causes what")
    st.markdown(
        "This is the world-model that scores every plan. When someone changes a "
        "feature, the arrows show what else must move --- and a plan that ignores "
        "those knock-on effects is what our **causal** test catches.")

    dataset = S.current_dataset()
    if dataset is None:
        return

    try:
        from src.scm.define_scm import get_scm  # noqa: PLC0415
        from src.viz import plot_graphs  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the causal tooling: {exc}")
        return

    try:
        scm = get_scm(dataset)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No causal graph available for {label_of(dataset)}: {exc}")
        return

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### The graph")
        try:
            fig = plot_graphs.plot_scm(scm, show_costs=True)
            st.pyplot(fig, width="stretch")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"(graph figure unavailable: {exc})")
    with right:
        st.markdown("#### Trace an intervention")
        actionable = list(scm.manipulable)
        picked = st.multiselect(
            "Act on these features", actionable,
            help="Choose what an applicant changes; the map highlights every "
                 "variable the change forces to move.")
        if picked:
            try:
                fig2 = plot_graphs.plot_intervention_paths(scm, picked)
                st.pyplot(fig2, width="stretch")
            except Exception as exc:  # noqa: BLE001
                st.caption(f"(intervention figure unavailable: {exc})")
            downstream = scm.downstream(picked, include_target=True)
            if downstream:
                st.markdown("**Forced to move:** " + ", ".join(downstream))
            record = st.session_state.get(S.K_RECORD)
            if record is not None:
                st.button("Try these in the sandbox \u2192",
                          on_click=S.set_seed_features,
                          args=(dataset, int(record), picked),
                          help="Pre-selects these features on the Live sandbox "
                               "page for the current applicant.")

    st.divider()
    _validation(dataset)

    theme.presenter_note(
        "Pick one feature and show the cascade highlight. Then scroll to the "
        "validation table and say: 'our graphs are hand-drawn and imperfect --- "
        "here's exactly how imperfect --- and the Evidence page shows CAC still "
        "wins even when the graph is wrong.'")


def _validation(dataset: str) -> None:
    st.markdown("#### How trustworthy is this graph?")
    summary = C.load_csv("scm_validation_summary")
    if summary is None:
        st.caption("No validation summary found (`scm_validation_summary.csv`).")
        return
    block = summary[summary["Dataset"].str.lower() == dataset.lower()]
    if block.empty:
        st.caption("No validation row for this dataset.")
        return
    st.dataframe(block, hide_index=True, width="stretch")
    st.caption(
        "We report the graph's imperfections rather than hiding them: 'edges "
        "supported' counts asserted arrows the data backs up. German and HELOC "
        "credit features are densely correlated, so a parsimonious hand-drawn "
        "graph misses real dependences --- the robustness evidence shows CAC "
        "degrades gracefully under a perturbed graph.")
