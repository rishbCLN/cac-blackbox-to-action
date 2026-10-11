"""Act 5 --- The Causal Map: the world-model that does the judging.

Renders the dataset's structural causal graph and lets the judge pick an action
set to see everything it drags along (the downstream cascade).  The honesty panel
shows the graph-validation summary so the "your graph is wrong" objection is met
head-on and handed to the robustness evidence.

The graph figures are built once and cached in :mod:`app.components`
(``scm_figure`` / ``intervention_figure``); the intervention trace lives in an
``st.fragment`` so picking features is instant.  The only motion is the entrance
fade-up and a ripple on the features you choose to act on --- the deliberate
"these are the nodes you touched" cue, not a per-keystroke animation.
"""

from __future__ import annotations

import streamlit as st

from app import components as C
from app.ui import flow as F
from app.ui import motion as Mo
from app.ui import state as S
from app.ui import theme
from src.viz import label_of


def render() -> None:
    F.stepper(5)
    F.keyboard_nav()

    speed = F.speed()
    reduced = F.reduced()
    cls = Mo.root_class(speed, reduced=reduced)

    # --- hero: eyebrow, title, framing, faded up once on entrance ---
    st.markdown(
        f"<div class='{cls}' style='padding:0.2rem 0 0.2rem;'>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:0.9rem;font-weight:700;"
        f"letter-spacing:0.16em;text-transform:uppercase;color:{theme.PALETTE_PRIMARY};'>"
        f"Act 5 &middot; The causal map</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:2.1rem;font-weight:800;"
        f"line-height:1.2;letter-spacing:-0.01em;color:{theme.PALETTE_TEXT};"
        f"margin:0.25rem 0 0.3rem;'>What causes what &mdash; and what it drags</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:1.05rem;line-height:1.55;"
        f"color:{theme.PALETTE_TEXT_MUTED};max-width:820px;'>"
        f"This is the world-model that scores every plan. Pick a feature and the "
        f"graph lights up everything that must move with it &mdash; the knock-on "
        f"effects an ordinary tool ignores. We also show, honestly, how imperfect "
        f"this hand-drawn graph is.</div>"
        f"</div>",
        unsafe_allow_html=True)

    dataset = S.current_dataset()
    if dataset is None:
        return

    try:
        scm = C.scm_object(dataset)
    except Exception as exc:  # noqa: BLE001 - a missing graph must not crash the page
        F.missing_card(
            f"The causal graph for {label_of(dataset)}",
            how=f"Could not load the structural model ({exc}). The other pages "
                "work from stored results.")
        return

    F.banner("This graph is the 'world' every plan is replayed through: change "
             "one feature and the arrows show what else must move. Ignoring those "
             "is exactly what the causal test catches.")

    _map_fragment(dataset, scm, cls=cls)

    st.divider()
    _validation(dataset)

    _present_hud(scm)


# --------------------------------------------------------------- the fragment
@st.fragment
def _map_fragment(dataset: str, scm, *, cls: str) -> None:
    """Graph + intervention trace, isolated so picking features is instant.

    The SCM figure comes from the process-wide cache, so re-rendering it on each
    selection only re-sends a PNG; nothing is rebuilt.
    """
    left, right = st.columns([3, 2])
    with left:
        _graph_panel(dataset)
    with right:
        _trace_panel(dataset, scm, cls)


def _graph_panel(dataset: str) -> None:
    st.markdown("#### The graph")
    F.safe_pyplot(
        C.scm_figure, dataset,
        caption="Nodes are features, coloured by role (see legend); arrows are "
                "direct causes; the value under each node is how costly that "
                "feature is to move. The highlighted node is the model's "
                "decision target.")


def _trace_panel(dataset: str, scm, cls: str) -> None:
    st.markdown("#### Trace an intervention")
    actionable = list(scm.manipulable)
    if not actionable:
        st.caption("This graph has no manipulable features to act on.")
        return

    picked = st.multiselect(
        "Act on these features", actionable, key=f"_map_pick_{dataset}",
        help="Choose what an applicant changes; the map highlights every "
             "variable the change forces to move.")

    if not picked:
        st.caption("Pick one or more features \u2014 the map lights up every "
                   "variable they force to move, and lists them below.")
        return

    # The chosen nodes ripple once per selection (the "you touched these" cue),
    # tinted to match the acted-node colour in the figure.
    chips = " ".join(
        f"<span class='mo-ripple' style='display:inline-block;"
        f"margin:0.15rem 0.25rem 0.15rem 0;padding:0.15rem 0.6rem;"
        f"border-radius:999px;background:#bcd9ee;border:1px solid {Mo.COL_CAC};"
        f"color:{Mo.COL_CAC};font-size:0.82rem;font-weight:700;'>{f}</span>"
        for f in picked)
    st.markdown(f"<div class='{cls}' style='margin:0.1rem 0 0.5rem;'>"
                f"You act on: {chips}</div>", unsafe_allow_html=True)

    F.safe_pyplot(
        C.intervention_figure, dataset, tuple(picked),
        caption="Dark = what you change, light = what the world moves in "
                "response, grey = untouched. The light nodes are the plan's "
                "hidden side effects.")

    downstream = [d for d in scm.downstream(picked, include_target=True)
                  if d not in set(picked)]
    if downstream:
        forced = " ".join(
            f"<span style='display:inline-block;margin:0.1rem 0.2rem 0.1rem 0;"
            f"padding:0.1rem 0.5rem;border-radius:999px;background:#e8f1f8;"
            f"border:1px solid #4a86b0;color:#2B5C6F;font-size:0.8rem;'>{d}</span>"
            for d in downstream)
        st.markdown(
            f"<div style='margin:0.2rem 0 0.4rem;'><b>Forced to move "
            f"({len(downstream)}):</b> {forced}</div>", unsafe_allow_html=True)
    else:
        st.caption("Nothing downstream moves \u2014 these features are leaves in "
                   "the graph.")

    record = st.session_state.get(S.K_RECORD)
    if record is not None:
        st.button(
            "Try these in the sandbox \u2192", key="_map_handoff",
            on_click=S.set_seed_features, args=(dataset, int(record), picked),
            help="Pre-selects these features on the Live sandbox page for the "
                 "current applicant, so you can price the whole cascade.")
    else:
        st.caption("Pick an applicant on the sandbox page first to hand these "
                   "features over to it.")


# ------------------------------------------------------------- honesty panel
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
        "graph misses real dependences --- the robustness evidence (next) shows "
        "CAC degrades gracefully even under a perturbed graph.")


def _present_hud(scm) -> None:
    """Presenter script for Act 5, built from the live graph (no hard-coding)."""
    example = next(iter(scm.manipulable), None)
    watch_value = None
    if example is not None:
        forced = [d for d in scm.downstream([example], include_target=True)
                  if d != example]
        watch_value = f"{example} \u2192 {len(forced)} move(s)"
    F.presenter_hud(
        5,
        script=("'This is the world-model doing the judging. Pick one feature on "
                "the right and watch the map light up everything it drags with it "
                "\u2014 those light nodes are the side effects an ordinary tool "
                "ignores. Then the honesty panel: our graph is hand-drawn and "
                "imperfect, and here is exactly how imperfect.'"),
        clicks=["pick a feature", "(watch the cascade light up)",
                "Try these in the sandbox", "(scroll to validation)",
                "Next \u25B6"],
        watch="Forced to move (downstream count)",
        watch_value=watch_value,
        budget_s=55,
        next_cue="Act 6 \u2014 is this real? significance, robustness, ablation.")
