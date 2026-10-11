"""Act 3 --- Head-to-head: every method's plan for one applicant, audited.

Reuses the tested recourse-plan figure and the per-record audit join, then lays
out one verdict card per method with the three F/S/A chips, the usable verdict,
the effort, and the plans-vs-consequences detail.  CAC leads and gets the
animated USABLE stamp; each baseline card names the exact test it fails.  The
applicant chooser and the whole comparison live in an ``st.fragment`` so flipping
applicants reruns only this block, not the entire page.  Read-only over stored
results.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import components as C
from app.ui import flow as F
from app.ui import motion as Mo
from app.ui import state as S
from app.ui import theme
from src.viz import label_of


def render() -> None:
    F.stepper(3)
    F.keyboard_nav()

    speed = F.speed()
    reduced = F.reduced()
    cls = Mo.root_class(speed, reduced=reduced)
    demo = S.demo_mode()

    # --- hero: eyebrow, title, framing, faded up on the current preset ---
    st.markdown(
        f"<div class='{cls}' style='padding:0.2rem 0 0.2rem;'>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:0.9rem;font-weight:700;"
        f"letter-spacing:0.16em;text-transform:uppercase;color:{theme.PALETTE_PRIMARY};'>"
        f"Act 3 &middot; Head to head</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:2.1rem;font-weight:800;"
        f"line-height:1.2;letter-spacing:-0.01em;color:{theme.PALETTE_TEXT};"
        f"margin:0.25rem 0 0.3rem;'>What each method advises</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:1.05rem;line-height:1.55;"
        f"color:{theme.PALETTE_TEXT_MUTED};max-width:820px;'>"
        f"One real applicant, every method&rsquo;s plan side by side, judged by "
        f"the same three tests. CAC leads and is usable; each baseline fails a "
        f"<i>specific</i> test &mdash; and the card says which.</div>"
        f"</div>",
        unsafe_allow_html=True)

    dataset = S.current_dataset()
    if dataset is None:
        return

    frame = C.per_record(dataset)
    if frame.empty:
        F.missing_card("The per-applicant plans (counterfactuals.csv)",
                       how="Run experiment 02 for this dataset, then reload.")
        return

    st.markdown(theme.audit_legend(), unsafe_allow_html=True)

    _comparison(dataset, frame, speed=speed, reduced=reduced, cls=cls, demo=demo)

    F.presenter_hud(
        3,
        script=("'Same applicant, every method, same three tests. CAC is first "
                "and stamped USABLE. Now walk down: each baseline trips one "
                "specific test \u2014 point at the red chip and read the "
                "plain-language reason.'"),
        clicks=["pick a curated case", "(point at CAC's USABLE stamp)",
                "(point at a baseline's red chip)", "Next \u25B6"],
        watch="CAC verdict",
        watch_value="USABLE",
        budget_s=50,
        next_cue="Act 4 \u2014 you drive the plan in the live sandbox.")


@st.fragment
def _comparison(dataset: str, frame: pd.DataFrame, *, speed: str, reduced: bool,
                cls: str, demo: bool) -> None:
    """The applicant chooser + side-by-side verdicts, isolated for snappy flips."""
    record = S.applicant_picker(dataset)
    if record is None:
        return

    st.divider()

    methods = C.methods_for(dataset, frame)
    cac_row = C.best_row(frame, "cac", record)
    cac_usable = cac_row is not None and theme.is_usable(cac_row)
    n_fail = sum(1 for m in methods if m != "cac"
                 and (C.best_row(frame, m, record) is None
                      or not theme.is_usable(C.best_row(frame, m, record))))

    if cac_usable:
        F.banner("CAC's plan is usable here. Watch which single test each "
                 "ordinary tool trips.", number=float(n_fail), fmt="int",
                 number_label="baselines fail here", color=Mo.COL_CAC)
    else:
        F.banner("Every method, audited the same way on this applicant \u2014 "
                 "including where CAC itself is stretched.")

    st.subheader(f"Applicant #{record} \u2014 {label_of(dataset)}")
    _figure(dataset, record, frame)

    st.markdown("#### The verdict, method by method")
    _cost_bar(frame, methods, record)
    for method in methods:
        _verdict_card(dataset, record, method, frame,
                      speed=speed, reduced=reduced, cls=cls, demo=demo)

    _narrative(dataset, record)


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


def _verdict_card(dataset: str, record: int, method: str, frame: pd.DataFrame,
                  *, speed: str = Mo.DEFAULT_SPEED, reduced: bool = False,
                  cls: str = "", demo: bool = False) -> None:
    row = C.best_row(frame, method, record)
    is_cac = method == "cac"
    with st.container(border=True):
        head, badge = st.columns([1.5, 1])
        head.markdown(f"### {label_of(method)}")
        if row is None:
            head.markdown("*no plan produced*")
            return
        usable = theme.is_usable(row)
        # CAC (the winner) earns the animated stamp; baselines get a light badge.
        if is_cac:
            with badge:
                F.safe_html(Mo.verdict_stamp_html(usable, speed=speed,
                                                  reduced=reduced), height=76)
        else:
            badge.markdown(
                f"<div style='text-align:right;'>{theme.verdict_badge(usable)}</div>",
                unsafe_allow_html=True)

        # chips fade up; a failing plan gives a small shake on a talk preset.
        st.markdown(f"<div class='{cls}'><div class='mo-fade-up'>"
                    f"{theme.chips(row)}</div></div>", unsafe_allow_html=True)

        cost = row.get("cost")
        cost_txt = f"{float(cost):.3f}" if cost == cost else "n/a"
        st.caption(f"effort {cost_txt} \u00b7 "
                   f"changes {int(row.get('n_changed') or 0)}")

        reason = _reason(row)
        if reason:
            fail_cls = "mo-shake" if speed != "instant" else "mo-fade-up"
            st.markdown(f"<div class='{cls}'><div class='{fail_cls}'>"
                        f"{reason}</div></div>", unsafe_allow_html=True)

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
