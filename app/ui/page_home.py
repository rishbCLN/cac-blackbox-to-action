"""Home / 'The Problem' --- the 30-second hook (Phase 1).

Shows one real rejected applicant, the advice a typical tool gives (and why it
fails one of the three tests), and CAC's plan (which holds).  Read-only over
``results/qualitative_cases.csv``; the black box is never queried here.
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

#: Cases where a baseline visibly fails, best for the hook, in preference order.
_HOOK_CASES = ("cheaper_but_wrong", "cac_only", "side_effect")


def render() -> None:
    F.stepper(1)
    F.keyboard_nav()

    speed = F.speed()
    reduced = F.reduced()
    cls = Mo.root_class(speed, reduced=reduced)
    demo = S.demo_mode()

    # Hook framing, faded up on the current preset (Instant collapses to 0ms).
    st.markdown(
        f"<div class='{cls}' style='padding:0.2rem 0 0.2rem;'>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:0.9rem;font-weight:700;"
        f"letter-spacing:0.16em;text-transform:uppercase;color:{theme.PALETTE_PRIMARY};'>"
        f"Act 1 &middot; The problem</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:2.1rem;font-weight:800;"
        f"line-height:1.2;letter-spacing:-0.01em;color:{theme.PALETTE_TEXT};"
        f"margin:0.25rem 0 0.3rem;'>Black Box &rarr; Actionable Insight</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:1.05rem;line-height:1.55;"
        f"color:{theme.PALETTE_TEXT_MUTED};max-width:820px;'>"
        f"When a model rejects someone, the usual explanation names the nearest change "
        f"that would flip the answer. That advice can be <b>impossible</b>, or it can "
        f"<b>quietly backfire</b> once the world reacts. A real plan must "
        f"<b>survive the world&rsquo;s reaction</b> and be <b>something a person can "
        f"do</b> &mdash; flipping the model is only the floor.</div>"
        f"</div>",
        unsafe_allow_html=True)

    F.banner("One rejected applicant: a typical tool's advice fails the usable "
             "test, while CAC's plan holds.")

    st.markdown("### What every plan is checked against")
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
                      & (cases["case"] == chosen_case)],
                speed=speed, reduced=reduced, cls=cls, demo=demo)
    st.divider()
    _next_button()

    F.presenter_hud(
        1,
        script=("'Here's one rejected applicant. A normal tool tells them to do "
                "something that flips the score on paper, but it fails the usable "
                "test \u2014 it snaps back once the world reacts, or asks the "
                "impossible. CAC's plan is usable.'"),
        clicks=["(point at the failing chip)", "Next \u25B6"],
        watch="CAC verdict",
        watch_value="USABLE",
        budget_s=30,
        next_cue="Act 2 \u2014 the same gap across the whole population.")


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


def _tell_story(dataset: str, rows: pd.DataFrame, *, speed: str = Mo.DEFAULT_SPEED,
                reduced: bool = False, cls: str = "", demo: bool = False) -> None:
    cac = _row(rows, "cac")
    failing = _first_failing(rows)
    cls = cls or Mo.root_class(speed, reduced=reduced)
    # A failing plan shakes on a talk preset; Instant/reduced motion just fades.
    fail_cls = "mo-shake" if speed != "instant" else "mo-fade-up"

    st.subheader(f"Meet applicant #{int(rows['record'].iloc[0])} "
                 f"\u2014 {label_of(dataset)}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### A typical tool says")
        if failing is not None:
            _advice_card(failing, tone="warn", cls=cls)
            st.markdown(f"<div class='{cls}'><div class='mo-fade-up'>"
                        f"{theme.chips(failing)}</div></div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='{cls}'><div class='{fail_cls}'>"
                        f"{_why_fails(failing)}</div></div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("*(Every baseline happened to succeed on this record; "
                        "browse other cases on the Head-to-head page.)*")
    with right:
        st.markdown("#### CAC says")
        if cac is not None:
            _advice_card(cac, tone="ok", cls=cls)
            st.markdown(f"<div class='{cls}'><div class='mo-fade-up'>"
                        f"{theme.chips(cac)}</div></div>",
                        unsafe_allow_html=True)
            usable = theme.is_usable(cac)
            F.safe_html(Mo.verdict_stamp_html(usable, speed=speed,
                                              reduced=reduced), height=84)
            if usable and demo and Mo.confetti_enabled(speed, reduced):
                F.safe_html(Mo.confetti_html(height=180), height=180)
            st.markdown(
                f"<div class='{cls}'><div class='mo-fade-up' style='background:{theme.PALETTE_OK_BG};"
                f"border:1px solid {theme.PALETTE_OK_BORDER};color:{theme.PALETTE_OK};"
                f"border-radius:12px;padding:0.75rem 1rem;margin:0.6rem 0;font-size:0.92rem;line-height:1.45;'>"
                f"<b>Why CAC succeeds:</b> It optimizes actions through the causal model &mdash; "
                f"no broken downstream promises, and stays within realistic human effort.</div></div>",
                unsafe_allow_html=True
            )


def _advice_card(row: pd.Series, tone: str, *, cls: str = "") -> None:
    advice = str(row.get("advice") or "").strip() or "(no plan produced)"
    method = label_of(str(row.get("method")))
    conseq = str(row.get("consequences") or "").strip()
    accent = theme.PALETTE_OK if tone == "ok" else theme.PALETTE_WARN
    open_root = f"<div class='{cls}'>" if cls else ""
    close_root = "</div>" if cls else ""
    fade = "mo-fade-up " if cls else ""

    st.markdown(
        f"{open_root}<div class='{fade}story-card' style='border-left: 4px solid {accent};'>"
        f"<div style='font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:{accent};margin-bottom:0.35rem;'>"
        f"{method}</div>"
        f"<div style='font-size:1.05rem;font-weight:500;line-height:1.5;color:#242220;'>{advice}</div>"
        f"</div>{close_root}",
        unsafe_allow_html=True
    )
    if conseq and conseq.lower() != "nan":
        st.markdown(
            f"{open_root}<div class='{fade}' style='background:{theme.PALETTE_SURFACE_HIGH};border:1px solid {theme.PALETTE_BORDER};border-radius:10px;padding:0.55rem 0.85rem;margin:0.4rem 0 0.6rem;font-size:0.86rem;color:{theme.PALETTE_TEXT_MUTED};'>"
            f"<strong>Knock-on effects (SCM):</strong> {conseq}</div>{close_root}",
            unsafe_allow_html=True
        )


def _why_fails(row: pd.Series) -> str:
    if not theme._truthy(row.get("actionable")):
        blocked = str(row.get("blocked_features") or "").replace(";", ", ")
        why = (f"asks for something nobody can change (<b>{blocked}</b>)"
               if blocked and blocked.lower() != "nan" else
               "is over the effort budget or asks too many changes")
        body = f"<b>Why it's not usable:</b> It {why}."
    elif not theme._truthy(row.get("causal")):
        off = str(row.get("off_features") or "").replace(";", ", ")
        extra = f" (breaks <b>{off}</b>)" if off and off.lower() != "nan" else ""
        body = (f"<b>Why it's not usable:</b> Once the world reacts, the change forces other features "
                f"to move and the 'yes' no longer holds{extra}.")
    elif not theme._truthy(row.get("valid")):
        body = "<b>Why it's not usable:</b> It does not even flip the model."
    else:
        body = "<b>Why it's not usable:</b> It fails the usable test (survives the world's reaction and is doable)."

    return (
        f"<div style='background:{theme.PALETTE_WARN_BG};border:1px solid {theme.PALETTE_WARN_BORDER};"
        f"color:{theme.PALETTE_WARN};border-radius:12px;padding:0.75rem 1rem;margin:0.6rem 0;font-size:0.92rem;line-height:1.45;'>"
        f"{body}</div>"
    )


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
    st.info("Next \u25B6 **The Gap** \u2014 the same contrast across everyone, not "
            "just one applicant (use the Next button at the top).")
