"""Result figures: the headline comparison, the recourse funnel, the cost of
causal validity, and the ablation (§9).

Every figure is read from the CSVs experiments 02 and 03 already wrote --- this
module never re-runs a method, so a figure can be regenerated in a second without
touching a black box.  The one point they all serve is the gap the paper is about:
a baseline *flips* the box (``validity`` near 1.0) yet the plan it hands over is
not something a person could act on and have come true (``usable`` far below it).
The headline and funnel figures draw that gap directly; the cost figure shows what
closing it costs; the ablation figure shows which CAC term is doing the closing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import CONFIG, DATASETS
from src.utils.io import save_figure
from src.utils.logging import get_logger, setup_logging
from src.viz import PALETTE, WARN, figure, label_of, method_colour, set_style

LOGGER = get_logger(__name__)

#: Reading order for the methods, CAC first and the control last, so a legend and
#: a bar group carry the same left-to-right story wherever they appear.
METHODS = ("cac", "oracle", "dice", "wachter", "random")

#: The four audits, flip first and the conjunction last.  ``usable`` requires one
#: suggestion to pass all three of the others, so it never sits above any of them
#: --- but the two middle audits are *independent* (a record can have an
#: actionable-but-acausal plan and a causal-but-inactionable one), so the line may
#: rise from ``causal_validity`` to ``actionability``.  That rise is not an error:
#: it says the SCM is the audit that binds, which is the paper's point.
FUNNEL_STAGES = (("validity", "Flips"), ("causal_validity", "SCM agrees"),
                 ("actionability", "Actionable"), ("usable", "All three"))


def _read(name: str) -> pd.DataFrame:
    """Load a results CSV, or say which experiment has to run first.

    A figure built from a stale or absent table would misreport the paper's
    numbers, so a missing file is an error with a fix in it rather than an empty
    plot.
    """
    path = CONFIG.RESULTS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run the experiment that writes it before plotting")
    return pd.read_csv(path)


def _methods_in(frame: pd.DataFrame) -> list[str]:
    """The methods present, in :data:`METHODS` order, then any extra alphabetically.

    An ablation table names variants (``-- SCM``) rather than methods, and a run
    restricted to two baselines names only those; ordering by a fixed list and
    appending the rest keeps every figure stable without assuming the roster.
    """
    present = list(dict.fromkeys(frame["method"]))
    ranked = [m for m in METHODS if m in present]
    return ranked + sorted(m for m in present if m not in ranked)


def _datasets_in(frame: pd.DataFrame) -> list[str]:
    present = set(frame["dataset"])
    return [d for d in DATASETS if d in present] + \
           sorted(d for d in present if d not in DATASETS)


def _grid(n: int) -> tuple[int, int]:
    """Rows and columns for ``n`` small multiples, at most two per row."""
    cols = 1 if n == 1 else 2
    return (n + cols - 1) // cols, cols


def _flat(axes: Any, n: int) -> list[Any]:
    """``plt.subplots`` returns a scalar, a 1-D array or a 2-D array; flatten all."""
    if n == 1:
        return [axes]
    return list(np.atleast_1d(axes).ravel())


def plot_usable_gap(headline: pd.DataFrame | None = None) -> list[Path]:
    """The headline: usable rate per method, with the flip rate drawn as a ceiling.

    One panel per dataset.  The solid bar is ``usable`` --- the share of applicants
    a method gives real recourse to --- and the faint bar behind it is ``validity``,
    the share it merely flips.  The empty space between them is the paper's claim
    made visible: advice that changes the model's mind without being advice a
    person could take and have the change hold.
    """
    set_style()
    headline = _read("headline") if headline is None else headline
    datasets = _datasets_in(headline)
    rows, cols = _grid(len(datasets))
    fig, axes = figure(rows, cols, figsize=(5.6 * cols, 3.2 * rows), squeeze=False)
    axes = _flat(axes, rows * cols)

    for ax, dataset in zip(axes, datasets):
        block = headline[headline["dataset"] == dataset].set_index("method")
        methods = [m for m in _methods_in(headline) if m in block.index]
        x = np.arange(len(methods))
        valid = [float(block.loc[m, "validity"]) for m in methods]
        usable = [float(block.loc[m, "usable"]) for m in methods]
        colours = [method_colour(m) for m in methods]
        ax.bar(x, valid, color=colours, alpha=0.22, width=0.72,
               label="flips the model")
        ax.bar(x, usable, color=colours, width=0.72, label="usable recourse")
        for xi, (v, u) in enumerate(zip(valid, usable)):
            if v - u > 0.02:  # annotate the gap only where there is one to read
                ax.annotate(f"-{v - u:.0%}", (xi, v), ha="center", va="bottom",
                            fontsize=6.5, color=WARN)
        ax.set_xticks(x)
        ax.set_xticklabels([label_of(m) for m in methods], rotation=15, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("share of applicants")
        ax.set_title(label_of(dataset))
    for ax in axes[len(datasets):]:
        ax.set_visible(False)
    _shared_legend(fig, axes[0])
    fig.suptitle("Flipping the model is not the same as giving usable recourse",
                 fontsize=11, y=1.0)
    return _save(fig, "results_usable_gap")


def plot_recourse_funnel(headline: pd.DataFrame | None = None) -> list[Path]:
    """Where each method falls off, one line per method per dataset.

    The x-axis is the flip test, the two independent audits, and their conjunction
    (:data:`FUNNEL_STAGES`); a line that stays flat kept every record it flipped,
    one that dives lost them.  Where the line rises from *SCM agrees* to
    *Actionable* and then drops to the conjunction, the SCM is the audit doing the
    rejecting --- drawn as lines rather than bars because that shape is the point.
    """
    set_style()
    headline = _read("headline") if headline is None else headline
    datasets = _datasets_in(headline)
    rows, cols = _grid(len(datasets))
    fig, axes = figure(rows, cols, figsize=(5.2 * cols, 3.2 * rows), squeeze=False)
    axes = _flat(axes, rows * cols)
    xs = np.arange(len(FUNNEL_STAGES))

    for ax, dataset in zip(axes, datasets):
        block = headline[headline["dataset"] == dataset].set_index("method")
        for method in (m for m in _methods_in(headline) if m in block.index):
            ys = [float(block.loc[method, col]) for col, _ in FUNNEL_STAGES]
            ax.plot(xs, ys, marker="o", markersize=4, color=method_colour(method),
                    label=label_of(method),
                    linewidth=2.0 if method == "cac" else 1.3,
                    zorder=3 if method == "cac" else 2)
        ax.set_xticks(xs)
        ax.set_xticklabels([short for _, short in FUNNEL_STAGES], rotation=15,
                           ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("share of applicants")
        ax.set_title(label_of(dataset))
    for ax in axes[len(datasets):]:
        ax.set_visible(False)
    _shared_legend(fig, axes[0])
    fig.suptitle("Four audits, and the usable conjunction: which one each method "
                 "loses people to", fontsize=11, y=1.0)
    return _save(fig, "results_recourse_funnel")


def plot_cost_of_validity(headline: pd.DataFrame | None = None) -> list[Path]:
    """What usable recourse costs: effort against usable rate, per method.

    CAC sits top-right --- the most applicants served, and rarely the cheapest ---
    while a baseline that looks cheap is cheap partly because the plans that would
    have cost more are the ones that did not survive the SCM.  Effort is the mean
    over each method's cheapest usable plan, so the axis and the point are the same
    population.
    """
    set_style()
    headline = _read("headline") if headline is None else headline
    datasets = _datasets_in(headline)
    rows, cols = _grid(len(datasets))
    fig, axes = figure(rows, cols, figsize=(5.0 * cols, 3.4 * rows), squeeze=False)
    axes = _flat(axes, rows * cols)

    for ax, dataset in zip(axes, datasets):
        block = headline[headline["dataset"] == dataset].set_index("method")
        for method in (m for m in _methods_in(headline) if m in block.index):
            cost = float(block.loc[method, "cost_mean"])
            usable = float(block.loc[method, "usable"])
            ax.scatter(cost, usable, s=70, color=method_colour(method),
                       edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(label_of(method), (cost, usable),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("effort of the plan (fraction of budget)")
        ax.set_ylabel("usable rate")
        ax.set_title(label_of(dataset))
    for ax in axes[len(datasets):]:
        ax.set_visible(False)
    fig.suptitle("The cost of usable recourse", fontsize=11, y=1.0)
    return _save(fig, "results_cost_of_validity")


def plot_ablation(ablation: pd.DataFrame | None = None) -> list[Path]:
    """Usable rate per CAC variant --- which term the recourse depends on (§9).

    The ``method`` column here names ablations (``cac``, ``-- SCM`` …).  Full CAC
    is drawn apart in its own colour as the reference the others are read against;
    the term whose removal moves ``usable`` is the term that was doing the work.
    """
    set_style()
    ablation = _read("ablation_headline") if ablation is None else ablation
    datasets = _datasets_in(ablation)
    rows, cols = _grid(len(datasets))
    fig, axes = figure(rows, cols, figsize=(5.6 * cols, 3.4 * rows), squeeze=False)
    axes = _flat(axes, rows * cols)

    for ax, dataset in zip(axes, datasets):
        block = ablation[ablation["dataset"] == dataset].set_index("method")
        variants = _methods_in(ablation)
        variants = [v for v in variants if v in block.index]
        x = np.arange(len(variants))
        usable = [float(block.loc[v, "usable"]) for v in variants]
        colours = [PALETTE["cac"] if v == "cac" else "#9aa7b2" for v in variants]
        ax.bar(x, usable, color=colours, width=0.7)
        ref = float(block.loc["cac", "usable"]) if "cac" in block.index else None
        if ref is not None:
            ax.axhline(ref, color=PALETTE["cac"], linestyle=":", linewidth=1.0,
                       alpha=0.7)
        for xi, u in enumerate(usable):
            ax.annotate(f"{u:.2f}", (xi, u), ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_variant_label(v) for v in variants], rotation=20,
                           ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("usable rate")
        ax.set_title(label_of(dataset))
    for ax in axes[len(datasets):]:
        ax.set_visible(False)
    fig.suptitle("Ablation: the causal term is what makes the advice usable",
                 fontsize=11, y=1.0)
    return _save(fig, "results_ablation")


def _variant_label(name: str) -> str:
    """``cac`` and the ``-- term`` ablations, spelled for an axis tick."""
    if name == "cac":
        return "CAC"
    return str(name).replace("_", " ").replace("--", "− ").strip()


def _shared_legend(fig: Any, ax: Any) -> None:
    """One legend for the whole figure, from the first panel's handles."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
                   bbox_to_anchor=(0.5, -0.02))


def _save(fig: Any, name: str) -> list[Path]:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig.tight_layout()
    paths = save_figure(fig, name, subdir="results")
    plt.close(fig)
    return paths


def plot_all_results() -> dict[str, list[Path]]:
    """Every result figure, each from the CSV its experiment wrote."""
    headline = _read("headline")
    figures = {
        "usable_gap": plot_usable_gap(headline),
        "recourse_funnel": plot_recourse_funnel(headline),
        "cost_of_validity": plot_cost_of_validity(headline),
    }
    if (CONFIG.RESULTS_DIR / "ablation_headline.csv").exists():
        figures["ablation"] = plot_ablation()
    else:
        LOGGER.warning("no ablation_headline.csv; skipping the ablation figure")
    return figures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Result figures from the CSVs.")
    parser.parse_args(argv)
    setup_logging("plot_results")
    for name, paths in plot_all_results().items():
        LOGGER.info("Saved %s -> %s", name, ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
