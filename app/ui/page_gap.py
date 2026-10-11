"""Act 2 --- The Gap: the thesis across the whole population.

Every method clears the validity bar; the bar that matters is *usable*, and it
collapses for the ordinary tools while CAC stays full.  The hero is the animated
"money slide" (:func:`app.ui.motion.gap_bars_html`): a faint validity track with
a solid usable fill that grows per method, the worst baseline flagged and CAC
emphasised.  Reads only ``results/headline.csv`` --- every number on screen is
read live from that table, nothing is hard-coded.
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
    F.stepper(2)
    F.keyboard_nav()

    speed = F.speed()
    reduced = F.reduced()
    cls = Mo.root_class(speed, reduced=reduced)

    # --- hero: eyebrow, title, thesis, faded up on the current preset ---
    st.markdown(
        f"<div class='{cls}' style='padding:0.2rem 0 0.2rem;'>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:0.9rem;font-weight:700;"
        f"letter-spacing:0.16em;text-transform:uppercase;color:{theme.PALETTE_PRIMARY};'>"
        f"Act 2 &middot; The gap</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:2.1rem;font-weight:800;"
        f"line-height:1.2;letter-spacing:-0.01em;color:{theme.PALETTE_TEXT};"
        f"margin:0.25rem 0 0.3rem;'>The gap validity never sees</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:1.05rem;line-height:1.55;"
        f"color:{theme.PALETTE_TEXT_MUTED};max-width:820px;'>"
        f"Every method flips the model on paper. The bar that matters is "
        f"<b>usable</b> &mdash; the plan also survives the world&rsquo;s reaction "
        f"<i>and</i> a real person can carry it out. Here is that gap across the "
        f"<b>whole population</b>, not just one applicant.</div>"
        f"</div>",
        unsafe_allow_html=True)

    dataset = S.current_dataset()
    if dataset is None:
        return

    headline = C.load_csv("headline")
    if headline is None or headline.empty or "dataset" not in headline.columns:
        F.missing_card("The population table (headline.csv)",
                       how="Run experiments 02 and 04 to populate "
                           "`results/headline.csv`, then reload.")
        return

    # --- prominent in-page dataset switch (global state; an instant flip) ---
    present = list(headline["dataset"].unique())
    from src.config import DATASETS  # noqa: PLC0415

    ordered = [d for d in DATASETS if d in present] or present
    idx = ordered.index(dataset) if dataset in ordered else 0
    picked = st.radio("Dataset", ordered, index=idx, horizontal=True,
                      format_func=label_of, key="_gap_dataset")
    if picked != dataset:
        st.session_state[S.K_DATASET] = picked
        dataset = picked
    st.caption(theme.DATASET_BLURB.get(dataset, ""))

    block = headline[headline["dataset"] == dataset].set_index("method")
    nums = _numbers(block)

    # --- the one-line takeaway + the single hero number (gap CAC closes) ---
    if nums["gap"] is not None:
        F.banner(
            "Every method clears the validity bar. Only CAC's advice is "
            "*usable* for (nearly) everyone \u2014 the ordinary tools fall away.",
            number=nums["gap"], fmt="pct", number_label="gap CAC closes",
            color=Mo.COL_CAC)
    else:
        F.banner("Every method clears the validity bar. Only CAC's advice is "
                 "usable once the world reacts and the plan must be doable.")

    # --- the money slide: animated validity-vs-usable bars, per method ---
    st.markdown("#### Validity vs usable, per method")
    order = [m for m in C.METHODS if m in set(block.index)]
    rows = [_gap_row(block, m, nums["worst_method"]) for m in order]
    bars_h = 34 + 56 * max(len(rows), 1)
    F.safe_html(Mo.gap_bars_html(rows, speed=speed, reduced=reduced,
                                 height=bars_h), height=bars_h)

    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"<div class='{cls}'><div class='mo-fade-up' style='font-size:0.98rem;"
            f"line-height:1.6;color:{theme.PALETTE_TEXT_MUTED};'>"
            f"<b style='color:{theme.PALETTE_TEXT};'>How to read this.</b> "
            f"The <b>faint</b> track is validity &mdash; every method flips the "
            f"box on paper. The <b>solid</b> fill is <i>usable</i> &mdash; the "
            f"advice also survives the causal model and is something a person "
            f"could do. The distance between them is the gap. "
            f"<b style='color:{Mo.COL_CAC};'>CAC</b> is built to close it; the "
            f"baselines are not.</div></div>",
            unsafe_allow_html=True)
    with right:
        st.markdown(theme.audit_legend(), unsafe_allow_html=True)

    st.divider()

    # --- supporting detail: where exactly each method drops out ---
    st.markdown("#### Where each method drops out (F \u2192 S \u2192 A \u2192 usable)")
    F.safe_pyplot(
        C.funnel_figure, headline, dataset,
        caption="Each method's four bars, left to right: flips the model, "
                "survives the causal model, is doable, and the conjunction "
                "(usable). A short final bar shows exactly which test fails.")

    _present_hud(nums)


# ----------------------------------------------------------------- helpers
def _numbers(block: Any) -> dict[str, Any]:
    """CAC's usable rate, the worst *ordinary* baseline, and the gap --- live.

    ``oracle`` is excluded from "worst baseline" (it is a skyline, not a tool a
    user could run), matching the Takeaway.  Any missing value degrades to
    ``None`` so the page never shows a fabricated figure.
    """
    def usable(method: str) -> float | None:
        if method not in block.index or "usable" not in block.columns:
            return None
        try:
            return float(block.loc[method, "usable"])
        except (TypeError, ValueError):
            return None

    cac = usable("cac")
    baselines = [(m, usable(m)) for m in block.index
                 if m not in ("cac", "oracle") and usable(m) is not None]
    if baselines:
        worst_method, worst = min(baselines, key=lambda pair: pair[1])
    else:
        worst_method, worst = None, None
    gap = (cac - worst) if (cac is not None and worst is not None) else None
    return {"cac": cac, "worst_method": worst_method, "worst": worst, "gap": gap}


def _gap_row(block: Any, method: str, worst_method: str | None) -> dict[str, Any]:
    """One :func:`gap_bars_html` row (label, validity, usable, flags) for a method."""
    def val(col: str) -> float:
        if col not in block.columns:
            return 0.0
        try:
            return float(block.loc[method, col])
        except (TypeError, ValueError):
            return 0.0

    return {
        "label": label_of(method),
        "validity": val("validity"),
        "usable": val("usable"),
        "is_cac": method == "cac",
        "is_worst": method == worst_method,
    }


def _present_hud(nums: dict[str, Any]) -> None:
    """Presenter script for Act 2, built from the live numbers (no hard-coding)."""
    worst_method = nums["worst_method"]
    worst = nums["worst"]
    worst_name = label_of(worst_method) if worst_method else "the worst baseline"
    watch_value = f"{worst * 100:.0f}%" if worst is not None else None
    F.presenter_hud(
        2,
        script=("Land the thesis across everyone: 'Every method flips the box "
                "\u2014 the faint bars are all full. Now watch the solid "
                f"*usable* bars: {worst_name} drops to the floor while CAC "
                "stays full. Flip the dataset and the gap shrinks \u2014 we "
                "don't hide the easy cases.'"),
        clicks=["(point at the worst usable bar)", "switch Dataset",
                "Next \u25B6"],
        watch=f"{worst_name} usable",
        watch_value=watch_value,
        budget_s=40,
        next_cue="Act 3 \u2014 every method on one applicant, audited.")
