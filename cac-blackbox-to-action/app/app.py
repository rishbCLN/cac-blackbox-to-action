"""Black Box → Actionable Insight — an interactive tour of the recourse gap.

Run with::

    streamlit run app/app.py

The app reads only what the experiments already wrote (``make all``); it trains
nothing and never calls the black box. Pick a dataset to see the validity–usable
gap, then pick an applicant to see what each method would actually tell them to
do and which of the three audits each plan survives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# The app may be launched from anywhere; put the project root on the path so
# ``import src`` finds the project package (and not a stray site-packages one).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import components as C  # noqa: E402
from src.viz import label_of  # noqa: E402

st.set_page_config(page_title="Black Box → Actionable Insight",
                   page_icon="🧭", layout="wide")


def _sidebar() -> tuple[str, str, int]:
    """Dataset, applicant-source mode, and the chosen record."""
    st.sidebar.title("🧭 Recourse explorer")
    datasets = C.datasets_present()
    if not datasets:
        st.sidebar.error("No stored results. Run `make all` first.")
        st.stop()

    dataset = st.sidebar.selectbox("Dataset", datasets,
                                   format_func=label_of)
    frame = C.per_record(dataset)
    if frame.empty:
        st.sidebar.error("results/counterfactuals.csv is missing; run experiment 02.")
        st.stop()

    cases = C.load_csv("qualitative_cases")
    dcases = cases[cases["dataset"] == dataset] if cases is not None else None
    mode = st.sidebar.radio(
        "Choose an applicant by", ["Curated case study", "Any rejected applicant"],
        help="Curated cases are the ones the paper walks through; or browse the "
             "whole evaluation set.")

    if mode.startswith("Curated") and dcases is not None and not dcases.empty:
        case = st.sidebar.selectbox("Case", sorted(dcases["case"].unique()))
        record = int(dcases[dcases["case"] == case]["record"].iloc[0])
        st.sidebar.caption(f"applicant #{record}")
    else:
        if mode.startswith("Curated"):
            st.sidebar.info("No curated cases for this dataset; browsing all.")
        records = C.records_for(frame)
        record = st.sidebar.select_slider("Applicant #", options=records,
                                           value=records[0])
    return dataset, mode, record


def _gap(dataset: str) -> None:
    headline = C.load_csv("headline")
    if headline is None or headline.empty:
        st.info("No headline table yet — run experiment 04 to see the gap chart.")
        return
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(C.gap_figure(headline, dataset), width="stretch")
    with right:
        st.markdown("#### How to read this")
        st.markdown(
            "- Every method clears a high **validity** bar — they all flip the "
            "model on paper.\n"
            "- Only the **usable** bar counts advice that also *survives the "
            "causal model* and is *something a person could do*.\n"
            "- The distance between the two bars is the gap. **CAC** is built to "
            "close it; the baselines are not.")
        block = headline[headline["dataset"] == dataset].set_index("method")
        if "cac" in block.index:
            st.metric("CAC usable recourse", f"{float(block.loc['cac','usable']):.0%}",
                      help="share of applicants CAC gives a usable plan")


def _applicant(dataset: str, mode: str, record: int) -> None:
    frame = C.per_record(dataset)
    st.subheader(f"Applicant #{record} — what each method advises")

    png = C.recourse_png(dataset, record, frame)
    if png is not None and png.exists():
        st.image(str(png), width="stretch")
    else:
        st.warning("Could not render the plan figure for this applicant.")

    st.markdown("#### The verdict, method by method")
    for method in C.methods_for(dataset, frame):
        row = C.best_row(frame, method, record)
        cols = st.columns([1, 4])
        cols[0].markdown(f"**{label_of(method)}**")
        cols[1].markdown(C.verdict_markdown(row), unsafe_allow_html=True)
        if row is not None:
            table = C.moves_table(row)
            if not table.empty:
                with cols[1].expander(
                        f"plan detail · effort {float(row.get('cost', float('nan'))):.2f}"):
                    st.dataframe(table, hide_index=True, width="stretch")

    if mode.startswith("Curated"):
        _narrative(dataset, record)


def _narrative(dataset: str, record: int) -> None:
    """The curated case's own advice/consequences text, if experiment 05 wrote it."""
    cases = C.load_csv("qualitative_cases")
    if cases is None:
        return
    block = cases[(cases["dataset"] == dataset) & (cases["record"] == record)]
    block = block[block["method"] == "cac"]
    if block.empty:
        return
    row = block.iloc[0]
    advice, conseq = row.get("advice"), row.get("consequences")
    if isinstance(advice, str) and advice.strip():
        st.markdown("#### CAC's plan in words")
        st.markdown(f"**Do:** {advice}")
        if isinstance(conseq, str) and conseq.strip() and conseq.lower() != "nan":
            st.markdown(f"**Which causes:** {conseq}")


def main() -> None:
    st.title("Black Box → Actionable Insight")
    st.caption("Counterfactual explanations that are causally valid, affordable, "
               "and actually actionable — compared against the methods that are not.")
    dataset, mode, record = _sidebar()
    _gap(dataset)
    st.divider()
    _applicant(dataset, mode, record)
    st.divider()
    st.caption("Read-only demo over stored results — no model is trained or queried "
               "here. Regenerate the underlying artifacts with `make all`.")


if __name__ == "__main__":
    main()
