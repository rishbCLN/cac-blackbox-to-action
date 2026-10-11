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
    "heloc": "Home-equity credit. The dramatic case: the widest usable-recourse gap.",
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
# ------------------------------------------------------------- warm UI tokens
PALETTE_SURFACE = "#F7F5F0"
PALETTE_SURFACE_CONTAINER = "#FAF8F5"
PALETTE_SURFACE_HIGH = "#F2EFE8"
PALETTE_TEXT = "#242220"
PALETTE_TEXT_MUTED = "#5E5953"
PALETTE_BORDER = "#E2DDD2"

PALETTE_PRIMARY = "#2B5C6F"
PALETTE_PRIMARY_CONTAINER = "#E8F0F3"

PALETTE_OK = "#2D6A4F"           # Soft Pine Green
PALETTE_OK_BG = "#EBF5EE"
PALETTE_OK_BORDER = "#C2E2CE"

PALETTE_WARN = "#9E2A2B"         # Soft Muted Crimson
PALETTE_WARN_BG = "#FDF0EE"
PALETTE_WARN_BORDER = "#F5C7C4"

PALETTE_CAUTION = "#9A5B18"      # Pale Warm Ochre
PALETTE_CAUTION_BG = "#FEF7ED"
PALETTE_CAUTION_BORDER = "#F9DFC0"


# ----------------------------------------------------------------------- chips
def chip(name: str, passed: bool, big: bool = False) -> str:
    colour = PALETTE_OK if passed else PALETTE_WARN
    bg = PALETTE_OK_BG if passed else PALETTE_WARN_BG
    border = PALETTE_OK_BORDER if passed else PALETTE_WARN_BORDER
    mark = "\u2713" if passed else "\u2717"
    size = "0.95rem" if big else "0.82rem"
    pad = "0.35rem 0.75rem" if big else "0.22rem 0.6rem"
    return (
        f"<span style='display:inline-flex;align-items:center;gap:0.35rem;"
        f"background:{bg};color:{colour};border:1px solid {border};"
        f"border-radius:999px;padding:{pad};margin:0.2rem 0.3rem 0.2rem 0;"
        f"font-size:{size};font-weight:600;white-space:nowrap;"
        f"box-shadow:0 1px 2px rgba(36,34,32,0.03);letter-spacing:0.01em;'>"
        f"<span style='font-weight:800;font-size:1.05em'>{mark}</span>&nbsp;{name}</span>"
    )


def chips(row: Any, big: bool = False) -> str:
    """The three audit chips as one HTML string (use ``unsafe_allow_html``)."""
    parts = [chip(name, passed, big=big) for name, _gloss, passed in audit_states(row)]
    return f"<div style='display:inline-flex;flex-wrap:wrap;align-items:center;margin:0.2rem 0;'>{''.join(parts)}</div>"


def verdict_badge(usable: bool, big: bool = False) -> str:
    bg = PALETTE_OK if usable else PALETTE_WARN
    text = "USABLE" if usable else "NOT USABLE"
    icon = "\u2713" if usable else "\u2715"
    size = "1.05rem" if big else "0.85rem"
    pad = "0.45rem 1.05rem" if big else "0.25rem 0.75rem"
    shadow = "rgba(45,106,79,0.22)" if usable else "rgba(158,42,43,0.22)"
    return (
        f"<span style='display:inline-flex;align-items:center;gap:0.4rem;"
        f"background:{bg};color:#ffffff;border-radius:999px;padding:{pad};"
        f"font-size:{size};font-weight:700;letter-spacing:0.04em;"
        f"box-shadow:0 3px 8px {shadow};line-height:1.2;'>"
        f"<span style='font-weight:800'>{icon}</span>{text}</span>"
    )


def big_verdict(row: Any) -> str:
    """Big demo banner: three chips then the arrow then the usable badge."""
    usable = is_usable(row)
    return (
        f"<div style='display:flex;align-items:center;flex-wrap:wrap;gap:0.6rem;margin:0.4rem 0;'>"
        f"{chips(row, big=True)}"
        f"<span style='font-size:1.3rem;color:{PALETTE_TEXT_MUTED};margin:0 0.2rem'>&rarr;</span>"
        f"{verdict_badge(usable, big=True)}"
        f"</div>"
    )


def audit_legend() -> str:
    """A compact 'how to read the chips' block for the top of a page."""
    rows = []
    for _key, name, gloss in AUDIT_META:
        rows.append(
            f"<div style='margin:0.25rem 0;display:flex;align-items:baseline;gap:0.45rem;'>"
            f"<span style='color:{PALETTE_OK};font-weight:700;font-size:0.92rem;min-width:170px;'>&#10003; {name}</span>"
            f"<span style='color:{PALETTE_TEXT_MUTED};font-size:0.9rem;'>&mdash; {gloss}</span></div>")
    rows.append(
        f"<div style='margin:0.5rem 0 0.1rem;padding-top:0.45rem;border-top:1px dashed {PALETTE_BORDER};font-size:0.9rem;color:{PALETTE_TEXT};'>"
        f"<strong style='color:{PALETTE_PRIMARY};'>Usable</strong> = survives knock-on effects "
        "<i>and</i> a real person could do it. (Flipping the decision is the floor "
        "every method clears; state fidelity is reported separately.)</div>")
    return (
        f"<div style='background:{PALETTE_SURFACE_CONTAINER};border:1px solid {PALETTE_BORDER};"
        f"border-radius:14px;padding:0.9rem 1.15rem;margin:0.6rem 0 1rem;box-shadow:0 2px 6px rgba(36,34,32,0.02);'>"
        + "".join(rows)
        + "</div>"
    )


# ------------------------------------------------------------------------- CSS
_DEMO_CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1250px; }
  h1 { font-size: 2.6rem !important; }
  h2 { font-size: 1.95rem !important; }
  h3 { font-size: 1.45rem !important; }
  .stMarkdown p, .stMarkdown li { font-size: 1.15rem; line-height: 1.6; }
  [data-testid="stMetricValue"] { font-size: 2.2rem !important; }
  .demo-caption { color: #5E5953; font-size: 0.98rem; }
</style>
"""

_BASE_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300..800;1,300..800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

  html, body, .stApp, .stMarkdown, p, label, input, button {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #242220;
  }

  /* Never override material icons font */
  [data-testid="stIcon"], [data-testid="stIcon"] *, [class*="material-symbols"], [class*="material-icons"], .material-icons, i {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
  }

  code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
  }

  .stApp {
    background-color: #F7F5F0;
    color: #242220;
  }

  .block-container {
    max-width: 1240px;
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
  }

  h1, h2, h3, h4 {
    color: #242220 !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
  }

  h1 {
    font-size: 2.35rem !important;
    line-height: 1.25 !important;
    margin-bottom: 0.5rem !important;
  }

  h2 {
    font-size: 1.7rem !important;
    line-height: 1.3 !important;
    margin-top: 1.2rem !important;
  }

  h3 {
    font-size: 1.25rem !important;
  }

  .story-card {
    background: #FFFFFF;
    border: 1px solid #E2DDD2;
    border-radius: 16px;
    padding: 1.15rem 1.35rem;
    margin: 0.6rem 0;
    box-shadow: 0 4px 14px rgba(36, 34, 32, 0.04), 0 1px 3px rgba(36, 34, 32, 0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  }

  .story-card:hover {
    border-color: #D3CCC0;
    box-shadow: 0 6px 20px rgba(36, 34, 32, 0.07), 0 2px 6px rgba(36, 34, 32, 0.03);
  }

  [data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
    background: #FFFFFF;
    border: 1px solid #E2DDD2 !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    box-shadow: 0 4px 14px rgba(36, 34, 32, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  [data-testid="stVerticalBlock"] > div[data-testid="stContainer"]:hover {
    box-shadow: 0 6px 22px rgba(36, 34, 32, 0.08);
  }

  [data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2DDD2;
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
    box-shadow: 0 2px 8px rgba(36, 34, 32, 0.03);
  }

  [data-testid="stMetricLabel"] p {
    font-size: 0.85rem !important;
    color: #5E5953 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  [data-testid="stMetricValue"] div {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #242220 !important;
    font-family: 'JetBrains Mono', monospace !important;
  }

  .stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.45rem 1.2rem !important;
    border: 1px solid #E2DDD2 !important;
    background: #FFFFFF !important;
    color: #242220 !important;
    transition: all 0.18s ease-in-out !important;
    box-shadow: 0 2px 6px rgba(36, 34, 32, 0.04) !important;
  }

  .stButton > button:hover {
    border-color: #2B5C6F !important;
    color: #2B5C6F !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(43, 92, 111, 0.12) !important;
  }

  .stButton > button[kind="primary"] {
    background: #2B5C6F !important;
    color: #FFFFFF !important;
    border: 1px solid #2B5C6F !important;
  }

  .stButton > button[kind="primary"]:hover {
    background: #234B5B !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(43, 92, 111, 0.25) !important;
  }

  [data-testid="stSidebar"] {
    background-color: #FAF8F5 !important;
    border-right: 1px solid #E2DDD2 !important;
  }

  [data-testid="stSidebarNav"] span {
    font-size: 0.96rem !important;
    font-weight: 500 !important;
  }

  [data-baseweb="select"] > div {
    border-radius: 12px !important;
    border-color: #E2DDD2 !important;
    background-color: #FFFFFF !important;
  }

  [data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2DDD2 !important;
    border-radius: 14px !important;
    overflow: hidden;
  }

  hr {
    border-color: #E2DDD2 !important;
    opacity: 0.8 !important;
    margin: 1.5rem 0 !important;
  }

  .demo-caption {
    color: #5E5953;
    font-size: 0.88rem;
    line-height: 1.45;
  }

  ::-webkit-scrollbar {
    width: 7px;
    height: 7px;
  }
  ::-webkit-scrollbar-track {
    background: #FAF8F5;
  }
  ::-webkit-scrollbar-thumb {
    background: #D3CCC0;
    border-radius: 999px;
  }
  ::-webkit-scrollbar-thumb:hover {
    background: #8E867C;
  }
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
