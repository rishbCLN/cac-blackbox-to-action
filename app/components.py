"""Data access and small view helpers for the Streamlit demo.

The demo is a thin, interactive shell over artifacts the experiments already
wrote --- it never trains, never calls a method, and never touches the black box's
``predict``.  Everything here either reads a CSV/joblib from ``results/`` or reuses
the already-tested plotting layer, so the app is fully runnable straight after
``make all`` and adds no new claims of its own.

Keeping the logic out of ``app.py`` lets it be exercised without a Streamlit
runtime (see ``scratch/smoke_app.py``): the ``st.cache_*`` decorators degrade to
plain calls when there is no script run, so every function here also works as a
library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.config import CONFIG, DATASETS
from src.utils.io import load_joblib
from src.viz import NEUTRAL, OK, WARN, figure, label_of, method_colour

#: Reading order shared with every figure: CAC first, the random floor last.
METHODS = ("cac", "oracle", "dice", "wachter", "random")
#: A move register as real (matches the cost/plot layers).
_ATOL = 1e-8
#: The three audits, as (letter, per-cf column, human name).
AUDITS = (("F", "valid", "flips the box"),
          ("S", "causal", "survives the SCM"),
          ("A", "actionable", "a person could do it"))


# ------------------------------------------------------------------- discovery

def datasets_present() -> list[str]:
    """Datasets that have a stored counterfactual set, in config order."""
    return [d for d in DATASETS
            if (CONFIG.RESULTS_DIR / f"counterfactuals_{d}.joblib").exists()]


# ------------------------------------------------------------------- artifacts

@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    """A results CSV, or ``None`` if the experiment that writes it has not run."""
    path = CONFIG.RESULTS_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def headline_source(dataset: str, live_fallback: bool = True,
                    n: int = 40, k: int = 3) -> tuple[pd.DataFrame | None, str]:
    """The validity/usable table for one dataset, and where it came from.

    Prefers ``results/headline.csv`` (the fast path the experiments wrote); when
    it is missing and ``live_fallback`` is set, computes a bounded cohort live on
    the real models instead.  Returns ``(frame, source)`` where ``source`` is
    ``"results"``, ``"live"`` or ``"none"`` so the caller can label the chart
    honestly rather than implying a full run happened.
    """
    headline = load_csv("headline")
    if headline is not None and dataset in set(headline["dataset"]):
        return headline[headline["dataset"] == dataset].copy(), "results"
    if not live_fallback:
        return None, "none"
    try:
        from app import live as L  # noqa: PLC0415

        methods = tuple(L.live_methods(dataset))
        frame = L.cohort_headline(dataset, n=n, k=k, methods=methods)
        return (frame, "live") if frame is not None and not frame.empty else (None, "none")
    except Exception:  # noqa: BLE001 - the chart is optional, never fatal
        return None, "none"


@st.cache_data(show_spinner=False)
def per_record(dataset: str) -> pd.DataFrame:
    """Every stored plan for one dataset, one row per (method, record, rank).

    Joins the three per-counterfactual audits the experiments wrote --- validity
    and causal validity from ``counterfactuals``, feasibility/actionability from
    ``feasibility_per_cf``, the SCM's off-promise features from
    ``faithfulness_per_cf`` --- into the columns the plot layer and the verdict
    badges read.  ``usable`` is the primary endpoint, computed here so the app and
    the headline table agree by construction rather than by coincidence.

    The ``causal`` column is replayed-action success (``flip_survives``): the plan
    flips the box and the flip survives re-running the SCM-repaired world through
    the box.  State fidelity (the proposed and repaired states agreeing) is a
    separate ``state_fidelity`` column, reported beside the endpoint rather than
    folded into it (Finding 3), so ``usable`` is replayed success *and* action
    feasibility --- the exact definition
    :func:`src.evaluation.feasibility.join_audits` leads with.
    """
    cf = load_csv("counterfactuals")
    if cf is None:
        return pd.DataFrame()
    cf = cf[cf["dataset"] == dataset].copy()
    keys = ["dataset", "method", "record", "rank"]

    feas = load_csv("feasibility_per_cf")
    if feas is not None:
        cols = keys + [c for c in ("actionable", "blocked_features") if c in feas]
        cf = cf.merge(feas[cols], on=keys, how="left")
    faith = load_csv("faithfulness_per_cf")
    if faith is not None:
        extra = [c for c in ("promise_kept", "flip_survives", "off_features")
                 if c in faith]
        cf = cf.merge(faith[keys + extra], on=keys, how="left")

    # ``causal`` is replayed-action success (flip_survives), matching the
    # headline's ``causal_validity``; fall back to causal_valid only when the
    # faithfulness audit is absent.
    if "flip_survives" in cf:
        cf["causal"] = cf["flip_survives"]
    else:
        cf["causal"] = cf.get("causal_valid")
    # State fidelity, reported separately: the strict promise_kept bundle, or the
    # SCM-agreement flag when the full faithfulness audit is unavailable.
    if "promise_kept" in cf:
        cf["state_fidelity"] = cf["promise_kept"]
    else:
        cf["state_fidelity"] = cf.get("causal_valid")
    if "actionable" not in cf:
        cf["actionable"] = cf.get("feasible")
    for col in ("valid", "causal", "state_fidelity", "actionable"):
        cf[col] = cf[col].astype("boolean")
    cf["usable"] = (cf["causal"] & cf["actionable"]).fillna(False)
    return cf


@st.cache_resource(show_spinner=False)
def plans(dataset: str) -> dict[str, dict[int, list]]:
    """Experiment 02's counterfactual objects: ``{method: {record: [cf, ...]}}``."""
    return load_joblib(CONFIG.RESULTS_DIR / f"counterfactuals_{dataset}.joblib")


@st.cache_resource(show_spinner=False)
def cost_model(dataset: str):
    """The dataset's cost model (feature ranges + pretty values); not the box."""
    from src.cost.cost_functions import get_cost_model  # noqa: PLC0415

    return get_cost_model(dataset)


def methods_for(dataset: str, frame: pd.DataFrame) -> list[str]:
    """Methods present for a dataset, CAC first then the rest alphabetically."""
    have = set(frame["method"].unique())
    return [m for m in METHODS if m in have] + sorted(have - set(METHODS))


def records_for(frame: pd.DataFrame) -> list[int]:
    """Sorted applicant ids that have at least one stored plan."""
    return sorted(int(r) for r in frame["record"].unique())


# --------------------------------------------------------------------- the gap

def gap_figure(headline: pd.DataFrame, dataset: str) -> Any:
    """The thesis in one chart, for one dataset: every method flips, few are usable.

    Faint bar = validity (flips the box), solid bar = usable (flips it *and*
    survives the SCM *and* is actionable). The distance between the two, per
    method, is the gap the whole project is about.
    """
    block = headline[headline["dataset"] == dataset]
    order = [m for m in METHODS if m in set(block["method"])]
    block = block.set_index("method").loc[order]
    y = np.arange(len(order))[::-1]

    fig, ax = figure(1, 1, figsize=(6.6, 0.6 + 0.5 * len(order)))
    for yi, method in zip(y, order, strict=False):
        colour = method_colour(method)
        ax.barh(yi, float(block.loc[method, "validity"]), height=0.66,
                color=colour, alpha=0.25, zorder=2)
        ax.barh(yi, float(block.loc[method, "usable"]), height=0.66,
                color=colour, zorder=3)
        ax.text(float(block.loc[method, "usable"]) + 0.015, yi,
                f"{float(block.loc[method, 'usable']):.0%}", va="center",
                fontsize=8, color=NEUTRAL)
    ax.set_yticks(y)
    ax.set_yticklabels([label_of(m) for m in order])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("share of applicants  (faint = flips the box · solid = usable recourse)")
    ax.set_title(f"{label_of(dataset)}: the validity–usable gap", loc="left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def funnel_figure(headline: pd.DataFrame, dataset: str) -> Any:
    """The F/S/A/Usable decomposition per method, for one dataset.

    Four grouped bars per method --- validity, causal validity, actionability,
    usable --- so a reader sees *which* test each method fails, not just that its
    usable rate is low.  Uses the same method colours as every other figure, with
    decreasing opacity across the four stages so 'usable' (the conjunction) reads
    as the solid endpoint.
    """
    block = headline[headline["dataset"] == dataset]
    order = [m for m in METHODS if m in set(block["method"])]
    block = block.set_index("method").loc[order]

    stages = [("validity", "flips"), ("causal_validity", "survives SCM"),
              ("actionability", "doable"), ("usable", "USABLE")]
    alphas = [0.30, 0.5, 0.7, 1.0]
    x = np.arange(len(order))
    width = 0.2

    fig, ax = figure(1, 1, figsize=(1.6 + 1.3 * len(order), 3.6))
    for si, (col, _lab) in enumerate(stages):
        offs = x + (si - 1.5) * width
        for xi, method in zip(offs, order, strict=False):
            val = float(block.loc[method, col]) if col in block else 0.0
            ax.bar(xi, val, width=width, color=method_colour(method),
                   alpha=alphas[si], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([label_of(m) for m in order])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of applicants")
    ax.set_title(f"{label_of(dataset)}: where each method drops out", loc="left")
    ax.grid(axis="y", alpha=0.3)
    # a tiny legend explaining the four bars via opacity
    from matplotlib.patches import Patch  # noqa: PLC0415

    handles = [Patch(facecolor=NEUTRAL, alpha=a, label=lab)
               for a, (_c, lab) in zip(alphas, stages, strict=False)]
    ax.legend(handles=handles, loc="lower left", fontsize=7, ncol=2,
              frameon=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- per-applicant

def best_row(frame: pd.DataFrame, method: str, record: int) -> pd.Series | None:
    """The plan a method leads with for an applicant: its lowest-rank stored cf."""
    block = frame[(frame["method"] == method) & (frame["record"] == record)]
    if block.empty:
        return None
    return block.sort_values("rank").iloc[0]


def synth_cases(dataset: str, record: int, frame: pd.DataFrame) -> pd.DataFrame:
    """A one-applicant cases frame in the shape :func:`plot_recourse_plans` expects.

    Lets the tested recourse plotter draw an arbitrary applicant, not only the
    curated few, by handing it each method's lead plan with the verdict and
    flagged-feature columns it reads. No numbers are invented --- every field is
    copied from the per-record audits.
    """
    rows = []
    for method in methods_for(dataset, frame):
        row = best_row(frame, method, record)
        if row is None:
            continue
        rows.append({
            "dataset": dataset, "case": "app", "record": record, "method": method,
            "answered": True, "rank": int(row["rank"]),
            "valid": _as_bool(row.get("valid")), "causal": _as_bool(row.get("causal")),
            "actionable": _as_bool(row.get("actionable")), "cost": row.get("cost"),
            "off_features": row.get("off_features", ""),
            "blocked_features": row.get("blocked_features", ""),
        })
    return pd.DataFrame(rows)


def recourse_png(dataset: str, record: int, frame: pd.DataFrame) -> Path | None:
    """Render this applicant's side-by-side plans and return the PNG path.

    Reuses :func:`src.viz.plot_counterfactuals.plot_recourse_plans` with a
    synthesized single-applicant frame, so the demo and the paper draw plans the
    identical way. Saved under a distinct ``app_`` name so it never clobbers a
    curated figure.
    """
    from src.viz.plot_counterfactuals import plot_recourse_plans  # noqa: PLC0415

    cases = synth_cases(dataset, record, frame)
    if cases.empty:
        return None
    paths = plot_recourse_plans(
        dataset, "app", record, cases=cases,
        results_cache={dataset: plans(dataset)},
        cost_cache={dataset: cost_model(dataset)},
        name=f"app_{dataset}_{record}")
    return next((p for p in paths if p.suffix == ".png"), None)


def moves_table(row: pd.Series) -> pd.DataFrame:
    """What a plan asks vs. what the SCM makes follow, as a small table.

    Side-effect tokens are stored with an internal ``:immutable``/``:sensitive``
    tag (e.g. ``ExternalRiskEstimate:immutable``); we split that off so the judge
    sees a clean feature name plus a readable role, never the raw encoding.
    """
    asks = _split(row.get("changed_features"))
    ask_set = set(asks)
    rows = [{"feature": f, "move": "asked for", "role": ""} for f in asks]
    for token in _split(row.get("side_effects")):
        name, _, tag = token.partition(":")
        if name in ask_set:
            continue
        rows.append({"feature": name, "move": "follows via the SCM",
                     "role": tag or ""})
    return pd.DataFrame(rows, columns=["feature", "move", "role"])


# ------------------------------------------------------------------- utilities

def _as_bool(value: Any) -> bool:
    return bool(value) and not (isinstance(value, float) and np.isnan(value))


def _split(value: Any) -> list[str]:
    """Split a ``;``/``,``/``|``-joined feature list, tolerating NaN and blanks."""
    text = "" if value is None else str(value)
    if not text or text.lower() == "nan":
        return []
    out: list[str] = []
    for token in text.replace("|", ",").replace(";", ",").split(","):
        name = token.strip()
        if name:
            out.append(name)
    return out


def verdict_markdown(row: pd.Series | None) -> str:
    """The three audits + the usable conjunction as coloured markdown badges."""
    if row is None:
        return f"<span style='color:{NEUTRAL}'>no plan produced</span>"
    parts = []
    for letter, col, name in AUDITS:
        state = row.get(col)
        colour = OK if _as_bool(state) else WARN
        mark = "✓" if _as_bool(state) else "✗"
        parts.append(f"<span style='color:{colour}'><b>{mark} {name}</b></span>")
    usable = _as_bool(row.get("causal")) and _as_bool(row.get("actionable"))
    verdict = (f"<span style='color:{OK}'><b>USABLE</b></span>" if usable
               else f"<span style='color:{WARN}'><b>not usable</b></span>")
    return " &nbsp;·&nbsp; ".join(parts) + " &nbsp;⟶&nbsp; " + verdict
