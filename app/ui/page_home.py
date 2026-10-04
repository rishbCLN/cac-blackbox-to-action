"""Home / 'The Problem' --- the 30-second hook (Phase 1).

Shows one real rejected applicant, the advice a typical tool gives (and why it
fails one of the three tests), and CAC's plan (which holds).  Read-only over
``results/qualitative_cases.csv``; the black box is never queried here.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import components as C
from app.ui import state as S
from app.ui import theme
from src.viz import label_of

#: Cases where a baseline visibly fails, best for the hook, in preference order.
_HOOK_CASES = ("cheaper_but_wrong", "cac_only", "side_effect")


def render() -> None:
    st.title("Black Box \u2192 Actionable Insight")
    st.markdown(
        "When a model rejects someone, the usual explanation names the nearest "
        "change that would flip the answer. That advice can be **impossible**, or "
        "it can **quietly backfire** once the world reacts. This demo separates "
        "the three things recourse actually needs.")

    st.markdown("### The three tests every plan must pass")
    st.markdown(theme.audit_legend(), unsafe_allow_html=True)

    dataset = S.current_dataset()
    if dataset is None:
        return

    cases = C.load_csv("qualitative_cases")
    if cases is None or cases.empty:
        st.info("No curated cases yet --- run experiment 05. Head to **The Gap** "
                "for the population picture.")
        _next_button()
        return

    block = cases[cases["dataset"] == dataset]
    chosen_case = _pick_case(block)
    story_dataset = dataset
    borrowed = False
    if chosen_case is None or _first_failing(block[block["case"] == chosen_case]) is None:
        # This dataset has no case that contrasts a failing baseline against a
        # usable CAC (e.g. adult, whose only case is all-usable). Borrow the
        # strongest hook from another dataset so the 30-second story is never
        # empty, and tell the judge where it came from.
        alt_dataset, alt_case = _best_hook_elsewhere(cases, exclude=dataset)
        if alt_case is not None:
            story_dataset, chosen_case, borrowed = alt_dataset, alt_case, True

    if chosen_case is None:
        st.info("No curated case with a failing baseline yet; open **The Gap** "
                "for the population picture.")
        _next_button()
        return

    st.divider()
    if borrowed:
        st.caption(f"The clearest example is on **{label_of(story_dataset)}** "
                   f"(**{label_of(dataset)}** has no failing-baseline case). "
                   "Switch datasets in the sidebar to explore others.")
    _tell_story(story_dataset,
                cases[(cases["dataset"] == story_dataset)
                      & (cases["case"] == chosen_case)])
    st.divider()
    _next_button()

    theme.presenter_note(
        "Open on HELOC. Say: 'Here's one rejected applicant. A normal tool tells "
        "them to do something that flips the score on paper, but it fails one of "
        "our three tests. CAC's plan passes all three.' Then click 'See the whole "
        "picture'.")


def _pick_case(block: pd.DataFrame) -> str | None:
    """The best hook case for one dataset, or ``None`` if none contrasts.

    A hook needs both a failing baseline *and* a usable CAC row; the preferred
    order in ``_HOOK_CASES`` wins, then any other qualifying case.  Returns
    ``None`` when the dataset has no contrasting case (e.g. adult's all-usable
    ``everyone_agrees``), so the caller can borrow one from another dataset.
    """
    have = set(block["case"].unique())
    ordered = [c for c in _HOOK_CASES if c in have]
    ordered += [c for c in sorted(have) if c not in _HOOK_CASES]
    for case in ordered:
        if _has_contrast(block[block["case"] == case]):
            return case
    return None


def _has_contrast(sub: pd.DataFrame) -> bool:
    """True when this case has a failing baseline and a usable CAC plan."""
    if sub.empty:
        return False
    usable = sub["usable"].astype(bool)
    cac = sub[sub["method"] == "cac"]
    cac_ok = bool(cac["usable"].astype(bool).iloc[0]) if not cac.empty else False
    return bool((~usable).any()) and cac_ok


def _best_hook_elsewhere(cases: pd.DataFrame,
                         exclude: str) -> tuple[str | None, str | None]:
    """The strongest hook (dataset, case) among datasets other than ``exclude``.

    Scans in ``_HOOK_CASES`` preference order across the remaining datasets so a
    dataset with no contrasting case still shows a real failing-vs-CAC story.
    """
    from src.config import DATASETS  # noqa: PLC0415

    others = [d for d in DATASETS if d in set(cases["dataset"]) and d != exclude]
    for case in _HOOK_CASES:
        for dataset in others:
            sub = cases[(cases["dataset"] == dataset) & (cases["case"] == case)]
            if _has_contrast(sub):
                return dataset, case
    return None, None


def _tell_story(dataset: str, rows: pd.DataFrame) -> None:
    cac = _row(rows, "cac")
    failing = _first_failing(rows)

    st.subheader(f"Meet applicant #{int(rows['record'].iloc[0])} "
                 f"\u2014 {label_of(dataset)}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### A typical tool says")
        if failing is not None:
            _advice_card(failing, tone="warn")
            st.markdown(theme.chips(failing), unsafe_allow_html=True)
            st.markdown(_why_fails(failing))
        else:
            st.markdown("*(Every baseline happened to succeed on this record; "
                        "browse other cases on the Head-to-head page.)*")
    with right:
        st.markdown("#### CAC says")
        if cac is not None:
            _advice_card(cac, tone="ok")
            st.markdown(theme.chips(cac), unsafe_allow_html=True)
            st.markdown(theme.verdict_badge(theme.is_usable(cac)),
                        unsafe_allow_html=True)


def _advice_card(row: pd.Series, tone: str) -> None:
    advice = str(row.get("advice") or "").strip() or "(no plan produced)"
    method = label_of(str(row.get("method")))
    conseq = str(row.get("consequences") or "").strip()
    body = f"**{method}:** {advice}"
    st.markdown(f"<div class='story-card'>{body}</div>", unsafe_allow_html=True)
    if conseq and conseq.lower() != "nan":
        st.caption(f"Knock-on effects the causal model fills in: {conseq}")


def _why_fails(row: pd.Series) -> str:
    if not theme._truthy(row.get("actionable")):
        blocked = str(row.get("blocked_features") or "").replace(";", ", ")
        why = (f"asks for something nobody can change (**{blocked}**)"
               if blocked and blocked.lower() != "nan" else
               "is over the effort budget or asks too many changes")
        return f"**Why it's not usable:** it {why}."
    if not theme._truthy(row.get("causal")):
        off = str(row.get("off_features") or "").replace(";", ", ")
        extra = f" (breaks **{off}**)" if off and off.lower() != "nan" else ""
        return ("**Why it's not usable:** once the world reacts, the change "
                f"forces other features to move and the 'yes' no longer holds{extra}.")
    if not theme._truthy(row.get("valid")):
        return "**Why it's not usable:** it does not even flip the model."
    return "**Why it's not usable:** it fails one of the three tests."


def _first_failing(rows: pd.DataFrame) -> pd.Series | None:
    for _, r in rows.iterrows():
        if r.get("method") == "cac":
            continue
        if not theme.is_usable(r):
            return r
    return None


def _row(rows: pd.DataFrame, method: str) -> pd.Series | None:
    sub = rows[rows["method"] == method]
    return None if sub.empty else sub.iloc[0]


def _next_button() -> None:
    st.info("Next: open **The Gap** in the left navigation to see this across "
            "everyone, not just one applicant.")
