"""Head-to-head --- every method's plan for one applicant, side by side (Phase 3).

Reuses the tested recourse-plan figure and the per-record audit join, then lays
out one verdict card per method with the three F/S/A chips, the usable badge, the
effort, and the plans-vs-consequences detail.  Read-only over stored results.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import components as C
from app.ui import state as S
from app.ui import theme
from src.viz import label_of


def render() -> None:
    st.title("Head to head: what each method advises")
    dataset = S.current_dataset()
    if dataset is None:
        return

    frame = C.per_record(dataset)
    if frame.empty:
        st.info("`results/counterfactuals.csv` is missing for this dataset "
                "--- run experiment 02.")
        return

    record = S.applicant_picker(dataset)
    if record is None:
        return

    st.markdown(theme.audit_legend(), unsafe_allow_html=True)
    st.divider()

    st.subheader(f"Applicant #{record}")
    _figure(dataset, record, frame)

    st.markdown("#### The verdict, method by method")
    methods = C.methods_for(dataset, frame)
    _cost_bar(frame, methods, record)
    for method in methods:
        _verdict_card(dataset, record, method, frame)

    _narrative(dataset, record)

    theme.presenter_note(
        "Pick 'A baseline is cheaper but wrong' or 'A baseline asks the "
        "impossible'. Walk down the cards: CAC is first and USABLE; point at the "
        "red chip on the failing baseline and read the plain-language reason.")


def _figure(dataset: str, record: int, frame: pd.DataFrame) -> None:
    try:
        png = C.recourse_png(dataset, record, frame)
    except Exception as exc:  # noqa: BLE001 - the figure is optional
        png = None
        st.caption(f"(plan figure unavailable: {exc})")
    if png is not None and png.exists():
        st.image(str(png), width="stretch")
        st.caption("Each panel is one method's plan. Solid bars = what it asked "
                   "for; hatched = what the causal model adds; a ring marks a "
                   "change that fails an audit.")


def _cost_bar(frame: pd.DataFrame, methods: list[str], record: int) -> None:
    rows = []
    for m in methods:
        r = C.best_row(frame, m, record)
        if r is not None and r.get("cost") == r.get("cost"):  # not NaN
            rows.append({"method": label_of(m), "effort": float(r.get("cost"))})
    if len(rows) >= 2:
        df = pd.DataFrame(rows).set_index("method")
        with st.expander("Effort per method (cheaper is not always better)"):
            st.bar_chart(df, height=220)
            st.caption("A baseline that undercuts CAC on effort is often cheaper "
                       "*because* its plan is not usable.")


def _verdict_card(dataset: str, record: int, method: str,
                  frame: pd.DataFrame) -> None:
    row = C.best_row(frame, method, record)
    with st.container(border=True):
        head, badge = st.columns([1.5, 1])
        head.markdown(f"### {label_of(method)}")
        if row is None:
            head.markdown("*no plan produced*")
            return
        badge.markdown(f"<div style='text-align:right;'>{theme.verdict_badge(theme.is_usable(row))}</div>",
                       unsafe_allow_html=True)
        st.markdown(theme.chips(row), unsafe_allow_html=True)

        cost = row.get("cost")
        cost_txt = f"{float(cost):.3f}" if cost == cost else "n/a"
        st.caption(f"effort {cost_txt} \u00b7 "
                   f"changes {int(row.get('n_changed') or 0)}")

        reason = _reason(row)
        if reason:
            st.markdown(reason, unsafe_allow_html=True)

        table = C.moves_table(row)
        if not table.empty:
            with st.expander("What the plan asks vs. what the world adds"):
                st.dataframe(table, hide_index=True, width="stretch")


def _reason(row: pd.Series) -> str:
    if theme.is_usable(row):
        return ""
    if not theme._truthy(row.get("actionable")):
        blocked = str(row.get("blocked_features") or "").replace(";", ", ")
        detail = (f"asks to change <b>{blocked}</b> (immutable/protected feature)"
                  if blocked and blocked.lower() != "nan" else
                  "exceeds the allowed effort budget or requires too many simultaneous changes")
        prefix = "Not doable"
    elif not theme._truthy(row.get("causal")):
        off = str(row.get("off_features") or "").replace(";", ", ")
        extra = f" (breaks <b>{off}</b>)" if off and off.lower() != "nan" else ""
        detail = f"The causal model propagates changes and the decision flips back{extra}"
        prefix = "Breaks under causal model"
    elif not theme._truthy(row.get("valid")):
        prefix = "Invalid"
        detail = "Does not even flip the black-box model's decision"
    else:
        prefix = "Not usable"
        detail = "Fails the combined causal and actionability criteria"

    return (
        f"<div style='background:{theme.PALETTE_WARN_BG};border:1px solid {theme.PALETTE_WARN_BORDER};"
        f"color:{theme.PALETTE_WARN};border-radius:10px;padding:0.6rem 0.9rem;margin:0.5rem 0;"
        f"font-size:0.88rem;line-height:1.45;'>"
        f"<strong>{prefix}:</strong> {detail}.</div>"
    )


def _narrative(dataset: str, record: int) -> None:
    cases = C.load_csv("qualitative_cases")
    if cases is None:
        return
    block = cases[(cases["dataset"] == dataset) & (cases["record"] == record)
                  & (cases["method"] == "cac")]
    if block.empty:
        return
    row = block.iloc[0]
    advice = str(row.get("advice") or "").strip()
    conseq = str(row.get("consequences") or "").strip()
    if advice:
        st.divider()
        st.markdown("#### CAC's plan in words")
        st.markdown(f"**Do:** {advice}")
        if conseq and conseq.lower() != "nan":
            st.markdown(f"**Which causes:** {conseq}")
