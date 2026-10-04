"""Per-applicant recourse plans: the advice itself, one person at a time.

The result figures in :mod:`src.viz.plot_results` are rates --- the share of
applicants each method serves.  A rate cannot show the one thing this project is
about: that two plans can both flip the box and be *different pieces of advice*,
one a person could act on and one not.  This module draws that for a single
applicant, the way experiment 05's narrative reads it.

Each method's plan is drawn as the moves it makes to the applicant's record, in
one shared unit --- the fraction of each feature's own range it travels, so a
categorical hop and a forty-hour work week sit on the same axis.  Two kinds of
move are told apart, because conflating them is the easiest way to misread CAC:

* what the method *asked for* (``changed_features``), drawn solid; and
* what merely *followed* from it once the SCM propagated the ask, drawn faint
  and hatched --- moves only a causal method reports, and the reason its plans
  look like they touch more than they demand.

A move the audits flagged --- an ask on an immutable feature, a value the SCM
rejects --- is ringed in the warning colour.  Nothing here re-runs a method or
re-scores a plan: the plans come from the joblib experiment 02 wrote and the
verdicts from the CSV experiment 05 wrote, so the figure and the paper's
case-study table name the same applicant and make the same claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import CONFIG, DATASETS
from src.utils.io import load_joblib, save_figure
from src.utils.logging import get_logger, setup_logging
from src.viz import NEUTRAL, OK, WARN, figure, label_of, method_colour, set_style

LOGGER = get_logger(__name__)

#: Reading order, CAC first and the control last --- the same order every other
#: figure uses, so a reader carries one left-to-right story between them.
METHODS = ("cac", "oracle", "dice", "wachter", "random")

#: The flagship applicant is chosen by the rule in ``experiments/05_qualitative.py``:
#: the primary dataset first, then :data:`DATASETS` order, and within a dataset the
#: cases below in this order --- the first pair that a run produced wins.  Mirrored
#: here rather than imported so the viz layer stays independent of ``experiments/``;
#: if that file's ``CASES`` order or ``FLAGSHIP`` changes, change these two to match
#: or the figure and the paper's table will name different people.
FLAGSHIP_DATASET = DATASETS[0]
CASE_ORDER = ("cac_only", "cheaper_but_wrong", "immutable_advice", "side_effect",
              "everyone_agrees")

#: Floor for the shared symmetric x-limit, so a plan of only tiny moves still
#: reads as small rather than being blown up to fill the panel.
_XLIM_FLOOR = 0.08
#: The moved feature moved at all (matches the cost layer's CHANGE_ATOL).
_MOVE_ATOL = 1e-8


# --------------------------------------------------------------------- loading

def _cases(cases: pd.DataFrame | None) -> pd.DataFrame:
    """The qualitative case rows, read from experiment 05's CSV if not supplied."""
    if cases is not None:
        return cases
    path = CONFIG.RESULTS_DIR / "qualitative_cases.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run experiments/05_qualitative.py before plotting")
    return pd.read_csv(path)


def _results(dataset: str, cache: dict[str, Any]) -> dict[str, dict[int, list]]:
    """Experiment 02's plans for one dataset: ``{method: {record: [cf, ...]}}``."""
    if dataset not in cache:
        path = CONFIG.RESULTS_DIR / f"counterfactuals_{dataset}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run experiments/02_compare_methods.py "
                f"-d {dataset} first")
        cache[dataset] = load_joblib(path)
    return cache[dataset]


def _cost_model(dataset: str, cache: dict[str, Any]):
    """The dataset's cost model --- for feature ranges and human-readable values.

    Loaded lazily and cached; it carries the bounds a move is normalised against
    and the ``pretty`` labels the annotations print.  It is not the black box:
    reading a feature's range to draw it is exactly what §7 is for.
    """
    if dataset not in cache:
        from src.cost.cost_functions import get_cost_model  # noqa: PLC0415

        cache[dataset] = get_cost_model(dataset)
    return cache[dataset]


def _span(cost_model: Any) -> np.ndarray:
    """Each feature's range, with zero-width features pinned to 1 to avoid /0."""
    bounds = np.asarray(cost_model.bounds, dtype=float)
    width = bounds[:, 1] - bounds[:, 0]
    return np.where(width > 0, width, 1.0)


def _flagged(row: pd.Series) -> set[str]:
    """Features the audits named on this plan --- asks on immutables, SCM refusals.

    ``off_features`` and ``blocked_features`` are free-text lists; splitting on
    both separators the pipeline has used keeps the ring around a violated
    feature whichever wrote it.
    """
    out: set[str] = set()
    for col in ("off_features", "blocked_features"):
        raw = str(row.get(col) or "")
        if raw and raw.lower() != "nan":
            for token in raw.replace("|", ",").split(","):
                name = token.strip()
                if name:
                    out.add(name)
    return out


# ------------------------------------------------------------ flagship selection

def _default_record(block: pd.DataFrame) -> int:
    """The applicant a (dataset, case) pair prints: the lowest record id present.

    A pair carries one curated applicant in every run seen so far; taking the
    smallest id keeps the choice deterministic if a future run ever curates more.
    """
    return int(np.min(block["record"].to_numpy()))


def default_flagship(cases: pd.DataFrame) -> tuple[str, str, int] | None:
    """The one applicant the paper prints in full, by the declared rule (no metric).

    Walks :data:`FLAGSHIP_DATASET` then :data:`DATASETS`, and within a dataset the
    :data:`CASE_ORDER`, returning the first pair a run produced --- reading none of
    the numbers it will go on to draw, so the case study cannot be chosen to
    flatter any method.
    """
    if cases.empty:
        return None
    order = [FLAGSHIP_DATASET] + [d for d in DATASETS if d != FLAGSHIP_DATASET]
    available = {(str(d), str(c)) for d, c in zip(cases["dataset"], cases["case"])}
    for dataset in order:
        for case in CASE_ORDER:
            if (dataset, case) in available:
                block = cases[(cases["dataset"] == dataset) & (cases["case"] == case)]
                return dataset, case, _default_record(block)
    # No case matched the declared order; fall back to the first row as written.
    first = cases.iloc[0]
    return str(first["dataset"]), str(first["case"]), int(first["record"])


def _present_cases(cases: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Every (dataset, case) a run produced, in a stable order, with its record."""
    out: list[tuple[str, str, int]] = []
    order = [FLAGSHIP_DATASET] + [d for d in DATASETS if d != FLAGSHIP_DATASET]
    seen = {(str(d), str(c)) for d, c in zip(cases["dataset"], cases["case"])}
    for dataset in order + sorted({d for d, _ in seen} - set(order)):
        for case in CASE_ORDER + tuple(sorted({c for d, c in seen if d == dataset}
                                              - set(CASE_ORDER))):
            if (dataset, case) in seen:
                block = cases[(cases["dataset"] == dataset)
                              & (cases["case"] == case)]
                out.append((dataset, case, _default_record(block)))
    return out


# ------------------------------------------------------------------- one plan

def _moves(cf: Any, span: np.ndarray) -> list[tuple[int, str, float, float, float]]:
    """The features this plan moves: ``(index, kind, before, after, normalised)``.

    ``kind`` is ``"ask"`` for a change the method demanded and ``"follow"`` for one
    the SCM propagated from it; both come straight off the stored vectors, so a
    baseline that reports no consequences simply has no ``follow`` rows.
    """
    x0 = np.asarray(cf.x_original, dtype=float).ravel()
    xc = np.asarray(cf.x_cf, dtype=float).ravel()
    asked = set(cf.changed_features)
    names = cf.feature_names
    moves: list[tuple[int, str, float, float, float]] = []
    for j, name in enumerate(names):
        if abs(xc[j] - x0[j]) <= _MOVE_ATOL:
            continue
        kind = "ask" if name in asked else "follow"
        moves.append((j, kind, float(x0[j]), float(xc[j]), float((xc[j] - x0[j]) / span[j])))
    # Asks first, then consequences, each ordered by how far it travels.
    moves.sort(key=lambda m: (0 if m[1] == "ask" else 1, -abs(m[4])))
    return moves


def _verdict_dots(ax: Any, row: pd.Series | None) -> None:
    """Three audits as coloured letters at the panel's top-right: flip, SCM, act.

    Letters rather than glyph ticks so the mark survives font embedding in the
    vector output (rule #12); green passed, red failed, grey not applicable.  The
    title is left-aligned so it never runs into these.
    """
    audits = [("F", "valid"), ("S", "causal"), ("A", "actionable")]
    for i, (letter, col) in enumerate(audits):
        state = None if row is None else row.get(col)
        colour = NEUTRAL if state is None or pd.isna(state) else (OK if bool(state) else WARN)
        ax.text(0.995 - (len(audits) - 1 - i) * 0.055, 1.015, letter,
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=colour)


def _ytick_label(name: str, kind: str, before: float, after: float,
                 cost_model: Any) -> str:
    """A moved feature's tick label: the name, and for an ask the value it moves to.

    The before/after lives in the left margin rather than floating by the bar,
    where a plan with many SCM consequences would pile labels on top of the axis.
    Only asks carry it --- the advice the person is handed --- so the consequence
    bars stay unlabelled and the panel stays legible however long the cascade.
    """
    if kind == "ask":
        return f"{name}\n{cost_model.pretty(name, before)} → " \
               f"{cost_model.pretty(name, after)}"
    return name


def _draw_panel(ax: Any, method: str, cf: Any, row: pd.Series | None,
                cost_model: Any, span: np.ndarray, limit: float) -> None:
    """One method's plan for one applicant, as signed bars per moved feature.

    ``limit`` is shared across the figure's panels so a wide bar means a wide move
    wherever it is read --- the whole comparison would mislead if each method were
    scaled to its own largest ask.
    """
    colour = method_colour(method)
    if cf is None:
        ax.text(0.5, 0.5, "produced nothing\nfor this applicant", ha="center",
                va="center", transform=ax.transAxes, fontsize=8, color=NEUTRAL)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label_of(method), fontsize=9, loc="left", color=NEUTRAL)
        _verdict_dots(ax, row)
        return

    moves = _moves(cf, span)
    flagged = _flagged(row) if row is not None else set()
    names = cf.feature_names
    y = np.arange(len(moves))[::-1]  # first move at the top

    for yi, (j, kind, before, after, nd) in zip(y, moves):
        ringed = names[j] in flagged
        ax.barh(yi, nd, color=colour, height=0.62,
                alpha=1.0 if kind == "ask" else 0.30,
                hatch="" if kind == "ask" else "////",
                edgecolor=WARN if ringed else "white",
                linewidth=1.7 if ringed else 0.6, zorder=3)

    ax.axvline(0, color=NEUTRAL, linewidth=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [_ytick_label(names[j], kind, before, after, cost_model)
         for j, kind, before, after, _ in moves], fontsize=6.6)
    for tick, (j, kind, *_rest) in zip(ax.get_yticklabels(), moves):
        if names[j] in flagged:
            tick.set_color(WARN)
        elif kind == "follow":
            tick.set_color(NEUTRAL)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-0.7, len(moves) - 0.3 if moves else 0.7)
    ax.grid(axis="y", visible=False)
    cost = None if row is None else row.get("cost")
    title = label_of(method)
    if cost is not None and not pd.isna(cost):
        title += f"   effort {float(cost):.2f}"
    ax.set_title(title, fontsize=9, loc="left")
    _verdict_dots(ax, row)


def _legend(fig: Any) -> None:
    """One key for the whole figure: ask vs consequence vs flagged, audit letters."""
    from matplotlib.patches import Patch  # noqa: PLC0415

    handles = [
        Patch(facecolor=NEUTRAL, edgecolor="white", label="asked for"),
        Patch(facecolor=NEUTRAL, alpha=0.32, hatch="////", edgecolor="white",
              label="follows via the SCM"),
        Patch(facecolor="white", edgecolor=WARN, linewidth=1.6,
              label="flagged by an audit"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)


# ------------------------------------------------------------------ the figure

def plot_recourse_plans(
    dataset: str,
    case: str,
    record: int | None = None,
    *,
    cases: pd.DataFrame | None = None,
    results_cache: dict[str, Any] | None = None,
    cost_cache: dict[str, Any] | None = None,
    name: str | None = None,
) -> list[Path]:
    """Every method's plan for one applicant, side by side --- the advice, drawn.

    ``dataset``/``case`` name a curated applicant; ``record`` defaults to the one
    that pair carries.  One panel per method, CAC first, each showing its moves in
    fractions of feature range: solid for what it asked, faint-hatched for what the
    SCM added, ringed where an audit objected.
    """
    set_style()
    frame = _cases(cases)
    results_cache = {} if results_cache is None else results_cache
    cost_cache = {} if cost_cache is None else cost_cache

    block = frame[(frame["dataset"] == dataset) & (frame["case"] == case)]
    if block.empty:
        raise ValueError(f"no {dataset}/{case} rows in the case CSV")
    if record is None:
        record = _default_record(block)
    block = block[block["record"] == record]
    if block.empty:
        raise ValueError(f"no {dataset}/{case} record {record} in the case CSV")

    results = _results(dataset, results_cache)
    cost_model = _cost_model(dataset, cost_cache)
    span = _span(cost_model)

    present = [m for m in METHODS if m in results] + \
              sorted(m for m in results if m not in METHODS)
    by_method = {str(r["method"]): r for _, r in block.iterrows()}

    # Resolve each method's chosen plan first, so the panels can share one x-limit
    # --- a bar's width must mean the same move in every panel or the comparison
    # the figure exists to make is a lie.
    chosen: dict[str, Any] = {}
    reach = 0.0
    for method in present:
        row = by_method.get(method)
        cf = None
        if row is not None and bool(row.get("answered")) and not pd.isna(row.get("rank")):
            plans = results.get(method, {}).get(int(record), [])
            rank = int(row["rank"])
            cf = plans[rank] if 0 <= rank < len(plans) else None
        chosen[method] = cf
        if cf is not None:
            reach = max([reach] + [abs(m[4]) for m in _moves(cf, span)])
    limit = max(reach * 1.12, _XLIM_FLOOR)

    n = len(present)
    cols = 1 if n == 1 else 2
    rows = (n + cols - 1) // cols
    fig, axes = figure(rows, cols, figsize=(5.4 * cols, 1.0 + 2.5 * rows),
                       squeeze=False)
    flat = list(axes.ravel())

    for ax, method in zip(flat, present):
        _draw_panel(ax, method, chosen[method], by_method.get(method),
                    cost_model, span, limit)
    for ax in flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{label_of(dataset)} · {case} · applicant #{int(record)}: "
                 f"what each method tells them to do", fontsize=11, y=1.0)
    fig.text(0.5, 0.955, "bar = change as a fraction of the feature's range "
             "(right = increase, left = decrease)", ha="center", fontsize=8,
             color=NEUTRAL)
    _legend(fig)
    out = name or f"recourse_plan__{dataset}_{case}"
    return _save(fig, out)


def _save(fig: Any, name: str) -> list[Path]:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    paths = save_figure(fig, name, subdir="recourse")
    plt.close(fig)
    return paths


def plot_all_counterfactuals(cases: pd.DataFrame | None = None) -> dict[str, list[Path]]:
    """The flagship applicant, then every curated case a run produced.

    The flagship is written under its own name so the paper can ``\\input`` a
    stable file; the rest land beside it for the appendix and the notebooks. All
    of them share one loaded joblib and cost model per dataset.
    """
    frame = _cases(cases)
    results_cache: dict[str, Any] = {}
    cost_cache: dict[str, Any] = {}
    out: dict[str, list[Path]] = {}

    flagship = default_flagship(frame)
    if flagship is None:
        LOGGER.warning("no qualitative cases to plot")
        return out
    fd, fc, fr = flagship
    out["flagship"] = plot_recourse_plans(
        fd, fc, fr, cases=frame, results_cache=results_cache,
        cost_cache=cost_cache, name="recourse_flagship")
    LOGGER.info("flagship recourse plan: %s/%s record %d", fd, fc, fr)

    for dataset, case, record in _present_cases(frame):
        try:
            out[f"{dataset}_{case}"] = plot_recourse_plans(
                dataset, case, record, cases=frame, results_cache=results_cache,
                cost_cache=cost_cache)
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.warning("skipping %s/%s: %s", dataset, case, exc)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-applicant recourse plans from the qualitative CSV + joblib.")
    parser.add_argument("-d", "--dataset", help="plot only this applicant's dataset")
    parser.add_argument("-c", "--case", help="plot only this case (needs --dataset)")
    parser.add_argument("-r", "--record", type=int, help="a specific record id")
    args = parser.parse_args(argv)
    setup_logging("plot_counterfactuals")

    if args.dataset and args.case:
        paths = plot_recourse_plans(args.dataset, args.case, args.record)
        LOGGER.info("Saved -> %s", ", ".join(p.name for p in paths))
        return 0
    for name, paths in plot_all_counterfactuals().items():
        LOGGER.info("Saved %s -> %s", name, ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
