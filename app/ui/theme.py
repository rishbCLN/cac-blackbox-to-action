"""Presentation layer for the demo: plain-language strings, badges, CSS.

Everything a non-expert judge reads is defined here once, so the six pages stay
thin and the vocabulary is consistent.  Nothing here touches the models or the
results --- it is pure view.  The three audits the whole project is about get a
fixed name, colour and one-line gloss (:data:`AUDIT_META`), rendered either as
small inline chips (:func:`chips`) or big demo-mode banners (:func:`big_verdict`).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.viz import NEUTRAL, OK, WARN, method_colour  # noqa: F401  (re-export)

# --------------------------------------------------------------- the three audits
#: (key on the audit row/object, short name, one-line gloss).  The reading order
#: F -> S -> A -> Usable is the story order used on every page.
AUDIT_META: tuple[tuple[str, str, str], ...] = (
    ("valid", "Flips the decision",
     "The model changes its 'no' to a 'yes' when the plan is applied."),
    ("causal", "Survives real-world knock-on effects",
     "Once the world reacts (one change forces others), the 'yes' still holds."),
    ("actionable", "A real person could do it",
     "No impossible asks, within a sensible effort budget and few enough steps."),
)

#: The one sentence the whole demo exists to land.
THESIS = ("Every method can flip the model on paper. Only a plan that also "
          "survives the causal model and that a real person could actually do "
          "is *usable* --- and CAC is the only method that is usable every time.")

#: Human labels for the four datasets, with a one-line framing for the judge.
DATASET_BLURB: dict[str, str] = {
    "adult": "Income (>$50k?). An honest case where the gap is small.",
    "german": "German credit approval. A clear usable-recourse gap.",
    "heloc": "Home-equity credit. The dramatic case: one tool helps only 13%.",
    "diabetes": "Diabetes onset risk. Favourable class is 'no onset'.",
}

#: One-line glossary shown in the sidebar.
GLOSSARY: tuple[tuple[str, str], ...] = (
    ("Black box", "The opaque model that says yes/no; we only ask it to predict."),
    ("Recourse", "Advice telling a rejected person what to change to get a yes."),
    ("Validity", "The plan flips the model's decision."),
    ("Causal validity", "The flip survives once the world's knock-on effects fill in."),
    ("Actionability", "A real person could carry the plan out (no impossible asks)."),
    ("Usable", "The primary endpoint: the flip survives the world's knock-on "
     "effects AND a real person could carry it out (state fidelity is reported "
     "separately)."),
    ("State fidelity", "How closely the plan's proposed state matches the one the "
     "SCM actually produces --- reported beside the endpoint, not a gate."),
    ("SCM", "Structural causal model --- a graph of what causes what."),
    ("Side effect", "A feature the SCM moves because you changed something upstream."),
    ("Cost / effort", "How hard the plan is, normalised so 1.0 is the allowed maximum."),
    ("Plausibility", "How ordinary the resulting record looks vs real applicants."),
)


# --------------------------------------------------------------------- helpers
def _truthy(value: Any) -> bool:
    """Robust bool for pandas/numpy nullable values used across the results."""
    try:
        import numpy as np  # noqa: PLC0415

        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
    except Exception:  # noqa: BLE001
        pass
    return bool(value)


def audit_states(row: Any) -> list[tuple[str, str, bool]]:
    """(short name, gloss, passed) for each of the three audits, from a row/obj.

    Accepts a pandas Series, a mapping, or a live ``Audit`` (which exposes
    ``valid`` / ``promise_kept`` / ``causal_valid`` / ``actionable``).  Missing
    fields read False.

    The ``causal`` audit is replayed-action success (``flip_survives``) --- the
    flip survives re-running the SCM-repaired world through the box --- matching
    the headline's ``causal_validity``.  State fidelity (``promise_kept``) is a
    separate quantity reported beside the endpoint, not one of these three chips
    (Finding 3); see :func:`app.components.per_record`.
    """
    out: list[tuple[str, str, bool]] = []
    for key, name, gloss in AUDIT_META:
        val = _read(row, key)
        if val is None and key == "causal":
            # live Audit / stored rows expose replayed success as flip_survives;
            # fall back to the stricter promise_kept, then causal_valid.
            val = _read(row, "flip_survives")
            if val is None:
                val = _read(row, "promise_kept")
            if val is None:
                val = _read(row, "causal_valid")
        out.append((name, gloss, _truthy(val)))
    return out


def _read(row: Any, key: str) -> Any:
    if row is None:
        return None
    if hasattr(row, "get"):
        try:
            return row.get(key)
        except Exception:  # noqa: BLE001
            pass
    return getattr(row, key, None)


def is_usable(row: Any) -> bool:
    """The primary endpoint: replayed-action success AND action feasibility.

    Proposed-state validity (the first chip) is shown but, per Finding 3, does
    not gate the verdict; the endpoint is the ``causal`` (replayed success) and
    ``actionable`` chips together.
    """
    states = [passed for _, _, passed in audit_states(row)]
    # AUDIT_META order is (valid, causal, actionable); the endpoint is the last
    # two: replayed success AND action feasibility.
    return bool(states[1] and states[2]) if len(states) >= 3 else False


# ----------------------------------------------------------------------- chips
def chip(name: str, passed: bool, big: bool = False) -> str:
    colour = OK if passed else WARN
    mark = "\u2713" if passed else "\u2717"
    size = "1.05rem" if big else "0.85rem"
    pad = "0.35rem 0.7rem" if big else "0.15rem 0.5rem"
    return (
        f"<span style='display:inline-block;border:1px solid {colour};"
        f"color:{colour};border-radius:999px;padding:{pad};margin:0.15rem;"
        f"font-size:{size};font-weight:600;white-space:nowrap'>"
        f"{mark}&nbsp;{name}</span>"
    )


def chips(row: Any, big: bool = False) -> str:
    """The three audit chips as one HTML string (use ``unsafe_allow_html``)."""
    parts = [chip(name, passed, big=big) for name, _gloss, passed in audit_states(row)]
    return "".join(parts)


def verdict_badge(usable: bool, big: bool = False) -> str:
    colour = OK if usable else WARN
    text = "USABLE" if usable else "NOT USABLE"
    size = "1.25rem" if big else "0.9rem"
    pad = "0.4rem 0.9rem" if big else "0.15rem 0.6rem"
    return (
        f"<span style='display:inline-block;background:{colour};color:#fff;"
        f"border-radius:8px;padding:{pad};font-size:{size};font-weight:700;"
        f"letter-spacing:0.03em'>{text}</span>"
    )


def big_verdict(row: Any) -> str:
    """Big demo banner: three chips then the arrow then the usable badge."""
    usable = is_usable(row)
    return (chips(row, big=True)
            + "<span style='font-size:1.3rem;margin:0 0.5rem'>&rarr;</span>"
            + verdict_badge(usable, big=True))


def audit_legend() -> str:
    """A compact 'how to read the chips' block for the top of a page."""
    rows = []
    for _key, name, gloss in AUDIT_META:
        rows.append(
            f"<div style='margin:0.2rem 0'><b style='color:{OK}'>&#10003; {name}</b>"
            f" &mdash; <span style='color:{NEUTRAL}'>{gloss}</span></div>")
    rows.append(
        "<div style='margin:0.35rem 0'><b>Usable</b> = all three at once.</div>")
    return "".join(rows)


# ------------------------------------------------------------------------- CSS
_DEMO_CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1250px; }
  h1 { font-size: 2.4rem !important; }
  h2 { font-size: 1.8rem !important; }
  h3 { font-size: 1.35rem !important; }
  .stMarkdown p, .stMarkdown li { font-size: 1.12rem; line-height: 1.55; }
  [data-testid="stMetricValue"] { font-size: 2.1rem; }
  .demo-caption { color: #4a5b6c; font-size: 0.95rem; }
</style>
"""

_BASE_CSS = """
<style>
  .block-container { max-width: 1200px; }
  .demo-caption { color: #4a5b6c; font-size: 0.85rem; }
  .story-card { border:1px solid #d6dee7; border-radius:10px; padding:1rem 1.2rem;
                margin:0.5rem 0; background:#fbfcfe; }
</style>
"""


def inject_css(demo_mode: bool) -> None:
    """Apply the base stylesheet, plus the big-font layer in demo mode."""
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
    if demo_mode:
        st.markdown(_DEMO_CSS, unsafe_allow_html=True)


def presenter_note(text: str) -> None:
    """A collapsible 'what to say / click' panel for the person demoing."""
    with st.expander("Presenter notes", expanded=False):
        st.markdown(f"<span class='demo-caption'>{text}</span>",
                    unsafe_allow_html=True)


__all__ = [
    "AUDIT_META", "THESIS", "DATASET_BLURB", "GLOSSARY",
    "audit_states", "is_usable", "chips", "chip", "verdict_badge",
    "big_verdict", "audit_legend", "inject_css", "presenter_note",
    "NEUTRAL", "OK", "WARN", "method_colour",
]
