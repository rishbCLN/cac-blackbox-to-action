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


@st.cache_data(show_spinner=False)
def per_record(dataset: str) -> pd.DataFrame:
    """Every stored plan for one dataset, one row per (method, record, rank).

    Joins the three per-counterfactual audits the experiments wrote --- validity
    and causal validity from ``counterfactuals``, feasibility/actionability from
    ``feasibility_per_cf``, the SCM's off-promise features from
    ``faithfulness_per_cf`` --- into the columns the plot layer and the verdict
    badges read.  ``usable`` is the conjunction, computed here so the app and the
    headline table agree by construction rather than by coincidence.
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
    if faith is not None and "off_features" in faith:
        cf = cf.merge(faith[keys + ["off_features"]], on=keys, how="left")

    cf = cf.rename(columns={"causal_valid": "causal"})
    if "actionable" not in cf:
        cf["actionable"] = cf.get("feasible")
    for col in ("valid", "causal", "actionable"):
        cf[col] = cf[col].astype("boolean")
    cf["usable"] = (cf["valid"] & cf["causal"] & cf["actionable"]).fillna(False)
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
    for yi, method in zip(y, order):
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
    """What a plan asks vs. what the SCM makes follow, as a small table."""
    asks = _split(row.get("changed_features"))
    follows = [f for f in _split(row.get("side_effects")) if f not in set(asks)]
    return pd.DataFrame(
        [{"feature": f, "move": "asked for"} for f in asks]
        + [{"feature": f, "move": "follows via the SCM"} for f in follows]
    )


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
    usable = _as_bool(row.get("valid")) and _as_bool(row.get("causal")) \
        and _as_bool(row.get("actionable"))
    verdict = (f"<span style='color:{OK}'><b>USABLE</b></span>" if usable
               else f"<span style='color:{WARN}'><b>not usable</b></span>")
    return " &nbsp;·&nbsp; ".join(parts) + " &nbsp;⟶&nbsp; " + verdict
