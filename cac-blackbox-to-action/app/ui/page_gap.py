"""The Gap --- the thesis chart across the whole population (Phase 2).

Every method clears the validity bar; the *usable* bar is what counts, and it
ranges from 13% to 100% depending on the method.  Reads only
``results/headline.csv``.
"""

from __future__ import annotations

import streamlit as st

from app import components as C
from app.ui import state as S
from app.ui import theme
from src.viz import label_of

_METHOD_ORDER = ("cac", "oracle", "dice", "wachter", "random")


def render() -> None:
    st.title("The gap validity never sees")
    st.markdown(theme.THESIS)

    dataset = S.current_dataset()
    if dataset is None:
        return

    headline = C.load_csv("headline")
    if headline is None or headline.empty:
        st.info("No headline table yet --- run experiments 02/04 to populate "
                "`results/headline.csv`.")
        return

    # Prominent in-page dataset switch so the presenter can flip HELOC -> Adult.
    present = [d for d in headline["dataset"].unique()]
    from src.config import DATASETS  # noqa: PLC0415

    ordered = [d for d in DATASETS if d in present]
    idx = ordered.index(dataset) if dataset in ordered else 0
    picked = st.radio("Dataset", ordered, index=idx, horizontal=True,
                      format_func=label_of, key="_gap_dataset")
    if picked != dataset:
        st.session_state[S.K_DATASET] = picked
        dataset = picked
    st.caption(theme.DATASET_BLURB.get(dataset, ""))

    _metrics(headline, dataset)
    st.divider()

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Validity vs usable, per method")
        st.pyplot(C.gap_figure(headline, dataset), width="stretch")
    with right:
        st.markdown("#### How to read this")
        st.markdown(
            "- The **faint** bar is validity --- every method flips the model on "
            "paper.\n"
            "- The **solid** bar is *usable* --- the advice also survives the "
            "causal model and is something a person could do.\n"
            "- The distance between them is the gap. **CAC** is built to close "
            "it; the baselines are not.")
        st.markdown(theme.audit_legend(), unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Where each method drops out (F \u2192 S \u2192 A \u2192 usable)")
    st.pyplot(C.funnel_figure(headline, dataset), width="stretch")
    st.caption("Each method's four bars, left to right: flips the model, survives "
               "the causal model, is doable, and the conjunction (usable). A short "
               "final bar shows exactly which test the method fails.")

    theme.presenter_note(
        "Land the thesis here. On HELOC point at Wachter: 100% valid, 13% usable. "
        "Then flip to Adult and say 'we don't hide the easy cases --- here the gap "
        "is small.' The funnel shows *which* test each baseline fails.")


def _metrics(headline, dataset: str) -> None:
    block = headline[headline["dataset"] == dataset].set_index("method")
    cac = float(block.loc["cac", "usable"]) if "cac" in block.index else float("nan")
    baselines = block.drop(index=[m for m in ("cac",) if m in block.index])
    if not baselines.empty:
        worst_method = baselines["usable"].astype(float).idxmin()
        worst = float(baselines.loc[worst_method, "usable"])
    else:
        worst_method, worst = "-", float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("CAC usable recourse", f"{cac:.0%}" if cac == cac else "-")
    c2.metric(f"Worst baseline ({label_of(worst_method)})",
              f"{worst:.0%}" if worst == worst else "-")
    if cac == cac and worst == worst:
        c3.metric("Gap CAC closes", f"+{(cac - worst):.0%}")
