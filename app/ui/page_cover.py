"""Act 0 --- Cover / green room: the title slide and the presenter's control room.

Artefact-free so it always renders.  It sets the stage (an animated title, the
one-line thesis, a six-beat agenda built straight from :data:`app.ui.flow.STEPS`)
and hosts the controls the rest of the show reads: the motion-speed preset,
reduced-motion, and the presenter-HUD switch.  A single Start button hands off to
Act 1.  Nothing here touches the models or the results.
"""

from __future__ import annotations

import streamlit as st

from app.ui import flow as F
from app.ui import motion as Mo
from app.ui import state as S
from app.ui import theme
from src.viz import label_of

#: One-line framing under each act label on the agenda cards.
_AGENDA_SUB: dict[str, str] = {
    "problem": "one applicant, one failing plan",
    "gap": "the whole population",
    "head": "every method, audited",
    "sandbox": "you drive the plan",
    "map": "the world-model that judges",
    "evidence": "answer the skeptic",
}


def render() -> None:
    F.stepper(0)
    F.keyboard_nav()

    speed = F.speed()
    cls = Mo.root_class(speed, reduced=F.reduced())
    p = theme  # palette shorthand

    # --- hero: eyebrow, title, thesis, staggered in on the current preset ---
    st.markdown(
        f"<div class='{cls}' style='text-align:center;padding:1.6rem 0 0.4rem;'>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:0.9rem;font-weight:700;"
        f"letter-spacing:0.18em;text-transform:uppercase;color:{p.PALETTE_PRIMARY};'>"
        f"Black Box &rarr; Actionable Insight</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:2.6rem;font-weight:800;"
        f"line-height:1.18;letter-spacing:-0.02em;color:{p.PALETTE_TEXT};margin:0.5rem 0;'>"
        f"From a black-box &ldquo;no&rdquo; to a plan a person<br>can actually use</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:1.12rem;line-height:1.55;"
        f"color:{p.PALETTE_TEXT_MUTED};max-width:760px;margin:0.3rem auto 0;'>"
        f"{theme.THESIS}</div>"
        f"</div>",
        unsafe_allow_html=True)

    # --- agenda: one card per Act 1..6, same source of truth as the nav ---
    cards = []
    for step in F.STEPS[1:F.LAST_STEP]:  # acts 1..6 (skip Cover and Takeaway)
        sub = _AGENDA_SUB.get(step.key, "")
        cards.append(
            f"<div class='mo-fade-up mo-stagger' style='flex:1 1 150px;min-width:150px;"
            f"background:{p.PALETTE_SURFACE_CARD if hasattr(p,'PALETTE_SURFACE_CARD') else '#FFFFFF'};"
            f"border:1px solid {p.PALETTE_BORDER};border-radius:14px;padding:0.8rem 0.9rem;"
            f"box-shadow:0 2px 8px rgba(36,34,32,0.04);'>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:0.8rem;"
            f"font-weight:700;color:{p.PALETTE_PRIMARY};'>{step.icon} ACT {step.index}</div>"
            f"<div style='font-size:1.02rem;font-weight:700;color:{p.PALETTE_TEXT};"
            f"margin:0.15rem 0 0.1rem;'>{step.title}</div>"
            f"<div style='font-size:0.85rem;color:{p.PALETTE_TEXT_MUTED};'>{sub}</div>"
            f"</div>")
    st.markdown(
        f"<div class='{cls}' style='display:flex;flex-wrap:wrap;gap:0.6rem;"
        f"margin:1.4rem 0 0.6rem;'>{''.join(cards)}</div>",
        unsafe_allow_html=True)

    st.divider()

    # --- the start button, centred and prominent ---
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        if st.button("Start the walkthrough  \u25B6", type="primary",
                     width="stretch", key="_cover_start"):
            F.goto(1)

    # --- control room: motion speed, reduced motion, presenter HUD ---
    with st.expander("\U0001F39B\uFE0F  Control room \u2014 motion & presenter", expanded=False):
        left, right = st.columns(2)
        with left:
            F.speed_control(container=st, key="_cover_speed")
        with right:
            st.toggle("Presenter mode", key=F.K_PRESENTER,
                      help="Show a per-act script, the exact number to point at, "
                           "the clicks to make, and a stopwatch with a time budget.")
            st.caption("Fast by default. **Cinematic** adds a confetti beat on a "
                       "USABLE verdict; **Instant** (and your OS reduced-motion "
                       "setting) skips every animation for Q&A.")

        dataset = S.current_dataset()
        if dataset:
            st.caption(f"Loaded dataset: **{label_of(dataset)}** "
                       f"(switch it any time in the sidebar).")

    F.presenter_hud(
        0,
        script=("Set the frame: 'A model says no. The usual explanation is the "
                "nearest change that flips it \u2014 which is often impossible or "
                "quietly backfires. We'll separate flipping the model from a plan "
                "a person can really use.'"),
        clicks=["Start the walkthrough"],
        budget_s=30,
        next_cue="Act 1 \u2014 one real rejected applicant.")
