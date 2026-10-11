"""Act 7 --- Takeaway: the one sentence and the one number to leave in the room.

Closes the loop opened on the Cover.  Reads the stored ``headline.csv`` for the
current dataset to show CAC's usable rate as an animated count-up against the
worst ordinary tool --- real numbers only; when the table is missing it degrades
to the qualitative thesis with no fabricated figure.  Ends with the restart and
"re-open the evidence" handoffs.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app import components as C
from app.ui import flow as F
from app.ui import motion as Mo
from app.ui import state as S
from app.ui import theme
from src.viz import label_of


def render() -> None:
    F.stepper(7)
    F.keyboard_nav()

    speed = F.speed()
    cls = Mo.root_class(speed, reduced=F.reduced())
    dataset = S.current_dataset()
    nums = _headline_numbers(dataset) if dataset else None

    # --- the one sentence ---
    st.markdown(
        f"<div class='{cls}' style='text-align:center;padding:1.4rem 0 0.2rem;'>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:0.9rem;font-weight:700;"
        f"letter-spacing:0.16em;text-transform:uppercase;color:{theme.PALETTE_PRIMARY};'>"
        f"The takeaway</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:1.9rem;font-weight:800;"
        f"line-height:1.3;letter-spacing:-0.01em;color:{theme.PALETTE_TEXT};"
        f"max-width:860px;margin:0.5rem auto 0;'>"
        f"Flipping the model is the floor. <span style='color:{theme.PALETTE_PRIMARY};'>"
        f"Usable</span> recourse must survive the world&rsquo;s reaction "
        f"<i>and</i> be something a person can do.</div>"
        f"</div>",
        unsafe_allow_html=True)

    # --- the one number (real, from headline.csv), or an honest fallback ---
    if nums is not None:
        cac_col, gap_col, worst_col = st.columns([1, 0.5, 1])
        with cac_col:
            st.markdown(
                f"<div style='text-align:center;color:{theme.PALETTE_TEXT_MUTED};"
                f"font-weight:700;font-size:0.9rem;text-transform:uppercase;"
                f"letter-spacing:0.05em;'>CAC usable</div>", unsafe_allow_html=True)
            F.safe_html(
                _centered(Mo.count_up(nums["cac"], fmt="pct", color=Mo.COL_CAC,
                                      size="3.4rem", speed=speed,
                                      reduced=F.reduced())),
                height=84)
        with gap_col:
            st.markdown(
                f"<div style='text-align:center;font-size:2.2rem;"
                f"color:{theme.PALETTE_BORDER};padding-top:1.4rem;'>vs</div>",
                unsafe_allow_html=True)
        with worst_col:
            st.markdown(
                f"<div style='text-align:center;color:{theme.PALETTE_TEXT_MUTED};"
                f"font-weight:700;font-size:0.9rem;text-transform:uppercase;"
                f"letter-spacing:0.05em;'>{label_of(nums['worst_method'])} usable</div>",
                unsafe_allow_html=True)
            F.safe_html(
                _centered(Mo.count_up(nums["worst"], fmt="pct",
                                      color=theme.PALETTE_WARN, size="3.4rem",
                                      speed=speed, reduced=F.reduced())),
                height=84)
        st.markdown(
            f"<div style='text-align:center;color:{theme.PALETTE_TEXT_MUTED};"
            f"font-size:0.95rem;margin-top:0.2rem;'>on {label_of(dataset)} "
            f"\u2014 same applicants, same black box, same budget.</div>",
            unsafe_allow_html=True)
    else:
        st.info("Load a dataset with a stored `headline.csv` to show the "
                "usable-rate gap here; the thesis above holds regardless.")

    st.divider()

    # --- three things to remember ---
    beats = [
        ("Flipping \u2260 fixing",
         "Every method can flip the model on paper. That is the easy part."),
        ("The world reacts",
         "A real plan must still hold once one change forces others through the "
         "causal model."),
        ("A person must be able to do it",
         "No impossible asks, within a realistic effort budget and few enough "
         "steps."),
    ]
    cards = "".join(
        f"<div class='mo-fade-up mo-stagger' style='flex:1 1 220px;min-width:220px;"
        f"background:#FFFFFF;border:1px solid {theme.PALETTE_BORDER};"
        f"border-left:4px solid {theme.PALETTE_PRIMARY};border-radius:14px;"
        f"padding:0.9rem 1.05rem;box-shadow:0 2px 8px rgba(36,34,32,0.04);'>"
        f"<div style='font-size:1.05rem;font-weight:800;color:{theme.PALETTE_TEXT};'>"
        f"{title}</div>"
        f"<div style='font-size:0.92rem;line-height:1.5;color:{theme.PALETTE_TEXT_MUTED};"
        f"margin-top:0.25rem;'>{body}</div></div>"
        for title, body in beats)
    st.markdown(
        f"<div class='{cls}' style='display:flex;flex-wrap:wrap;gap:0.7rem;"
        f"margin:0.4rem 0 0.8rem;'>{cards}</div>",
        unsafe_allow_html=True)

    st.divider()

    left, right = st.columns(2)
    with left:
        if st.button("\u21BA  Restart the walkthrough", width="stretch",
                     key="_takeaway_restart"):
            F.reset_demo()
            F.goto(0)
    with right:
        if st.button("Re-open the evidence  \U0001F52C", width="stretch",
                     key="_takeaway_evidence"):
            F.goto(6)

    F.presenter_hud(
        7,
        script=("Land it: 'One number \u2014 CAC is usable for "
                f"{_say_pct(nums['cac']) if nums else 'the whole cohort'}, "
                "the best ordinary tool for far fewer \u2014 because it plans "
                "through the world model, not around it.'"),
        clicks=["Restart", "or take questions"],
        watch="CAC usable rate",
        watch_value=_say_pct(nums["cac"]) if nums else None,
        budget_s=30,
        next_cue="Questions, or restart for the next judge.")


def _headline_numbers(dataset: str) -> dict[str, Any] | None:
    """CAC vs the worst *ordinary* tool's usable rate for one dataset, or None.

    Reads stored ``headline.csv`` only; ``oracle`` is excluded (it is a skyline,
    not a tool a user could run).  Returns ``None`` when the table or the needed
    rows are missing so the page can fall back to the qualitative thesis.
    """
    frame = C.load_csv("headline")
    if frame is None or not {"dataset", "method", "usable"} <= set(frame.columns):
        return None
    block = frame[frame["dataset"] == dataset]
    if block.empty:
        return None

    def usable(method: str) -> float | None:
        sub = block[block["method"] == method]
        return float(sub["usable"].iloc[0]) if not sub.empty else None

    cac = usable("cac")
    baselines = [(m, usable(m)) for m in C.METHODS
                 if m not in ("cac", "oracle") and usable(m) is not None]
    if cac is None or not baselines:
        return None
    worst_method, worst = min(baselines, key=lambda pair: pair[1])
    return {"cac": cac, "worst_method": worst_method, "worst": worst}


def _centered(html: str) -> str:
    """Wrap a count-up fragment so it sits centred in its column."""
    return (f"<div style='display:flex;justify-content:center;align-items:center;'>"
            f"{html}</div>")


def _say_pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.0f}%"
