"""'Is this real?' --- the skeptic-killer evidence page (Phase 6).

Three tabs answer the three attacks a judge makes: is it statistically real, does
it survive when your assumptions are wrong, and is the causal machinery actually
doing the work?  Read-only over the stored significance / robustness / ablation
CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import components as C
from app.ui import flow as F
from app.ui import motion as Mo
from app.ui import state as S
from app.ui import theme
from src.viz import figure, label_of, method_colour

_METHOD_ORDER = ("cac", "oracle", "dice", "wachter", "random")

#: The ablation hero is per-plan flip survival, not usable: on *usable* the
#: ablation barely moves (german/diabetes stay 100%), but turning off the causal
#: term makes plans stop *surviving* the SCM-repaired world --- flip_survival
#: collapses (e.g. adult 1.00 -> 0.60). State fidelity is reported beside it.
_ABLATION_METRIC = "flip_survival"
_ABLATION_SECONDARY = "state_fidelity"


def render() -> None:
    F.stepper(6)
    F.keyboard_nav()

    speed = F.speed()
    reduced = F.reduced()
    cls = Mo.root_class(speed, reduced=reduced)

    # --- hero: eyebrow, title, framing, faded up on the current preset ---
    st.markdown(
        f"<div class='{cls}' style='padding:0.2rem 0 0.2rem;'>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:0.9rem;font-weight:700;"
        f"letter-spacing:0.16em;text-transform:uppercase;color:{theme.PALETTE_PRIMARY};'>"
        f"Act 6 &middot; Is this real?</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:2.1rem;font-weight:800;"
        f"line-height:1.2;letter-spacing:-0.01em;color:{theme.PALETTE_TEXT};"
        f"margin:0.25rem 0 0.3rem;'>The skeptic&rsquo;s three questions</div>"
        f"<div class='mo-fade-up mo-stagger' style='font-size:1.05rem;line-height:1.55;"
        f"color:{theme.PALETTE_TEXT_MUTED};max-width:820px;'>"
        f"A fair judge attacks three ways: is the gap just <b>luck</b>, does it "
        f"vanish when the <b>assumptions</b> are wrong, and is the <b>causal</b> "
        f"machinery actually doing the work? Each tab answers one &mdash; "
        f"read-only over the stored significance, robustness and ablation "
        f"results.</div>"
        f"</div>",
        unsafe_allow_html=True)

    dataset = S.current_dataset()
    if dataset is None:
        return

    drop = _causal_drop(dataset)
    thesis = ("Three attacks, three answers: it is statistically real, it "
              "survives wrong assumptions, and the causal term is doing the work.")
    if drop is not None:
        F.banner(thesis, number=drop, fmt="pct",
                 number_label="flip survival lost without the causal term",
                 color=Mo.COL_CAC)
    else:
        F.banner(thesis)

    tab_sig, tab_rob, tab_abl = st.tabs(
        ["Not luck (significance)", "Survives wrong assumptions (robustness)",
         "The causal term does the work (ablation)"])
    with tab_sig:
        _significance(dataset)
    with tab_rob:
        _robustness(dataset)
    with tab_abl:
        _ablation(dataset)

    _present_hud(dataset, drop)


def _causal_drop(dataset: str) -> float | None:
    """How much plan survival collapses without the causal term --- live from CSV.

    The evidence page's one hero number: CAC's ``flip_survival`` minus the
    ``cac_no_causal`` ablation on this dataset (``ablation_headline``).  Returns
    ``None`` when the table, the rows, or a meaningful drop is absent so the
    banner degrades to the qualitative thesis rather than showing a hollow 0%.
    """
    abl = C.load_csv("ablation_headline")
    if abl is None or "dataset" not in abl.columns:
        return None
    block = abl[abl["dataset"] == dataset]
    full = _val(block, "cac", _ABLATION_METRIC)
    nocausal = _val(block, "cac_no_causal", _ABLATION_METRIC)
    if pd.notna(full) and pd.notna(nocausal) and (full - nocausal) > 0.005:
        return full - nocausal
    return None


def _present_hud(dataset: str, drop: float | None) -> None:
    """Presenter script for Act 6, built from the live ablation drop (no hard-coding)."""
    watch_value = f"\u2212{drop * 100:.0f}%" if drop is not None else None
    F.presenter_hud(
        6,
        script=("Answer the skeptic in order: 'Significance \u2014 confidence "
                "intervals and a Holm-corrected McNemar test, so it is not luck. "
                "Robustness \u2014 wrong graph, a different black box, a tighter "
                "budget, and CAC still lands. Then the kill shot: Ablation "
                "\u2014 turn the causal term off and plans stop surviving the "
                "world. That isolated drop is the whole thesis.'"),
        clicks=["Significance tab", "Robustness tab", "Ablation tab",
                "(point at the flip-survival drop)", "Next \u25B6"],
        watch="Flip survival lost without the causal term",
        watch_value=watch_value,
        budget_s=70,
        next_cue="Act 7 \u2014 the one sentence and the one number to remember.")


# --------------------------------------------------------------- significance
def _significance(dataset: str) -> None:
    agg = C.load_csv("significance_aggregate")
    if agg is None:
        F.missing_card("The significance study (significance_aggregate.csv)",
                       how="Run `experiments/06_significance.py`, then reload.")
        return
    block = agg[agg["dataset"] == dataset]
    if block.empty:
        st.caption("No significance rows for this dataset.")
        return

    st.markdown("#### Usable rate with 95% confidence (5 seeds)")
    _ci_figure(block)
    st.caption("Each bar is the usable rate averaged over 5 random seeds; the "
               "whisker is the 95% confidence interval.")

    tests = C.load_csv("significance_tests")
    if tests is not None:
        tblock = tests[tests["dataset"] == dataset].copy()
        if not tblock.empty:
            st.markdown("#### CAC vs each baseline (paired McNemar, Holm-corrected)")
            show = tblock[["method", "usable_gap", "gap_lo", "gap_hi",
                           "mcnemar_p_holm", "significant"]].copy()
            show.columns = ["baseline", "usable gap (CAC\u2212base)", "gap lo",
                            "gap hi", "p (Holm)", "significant"]
            st.dataframe(show, hide_index=True, width="stretch")
            st.caption("A positive gap means CAC gives more people a usable plan. "
                       "'significant' is after Holm correction across baselines.")


def _ci_figure(block: pd.DataFrame) -> None:
    order = [m for m in _METHOD_ORDER if m in set(block["method"])]
    b = block.set_index("method").loc[order]
    x = np.arange(len(order))
    means = b["usable_mean"].astype(float).to_numpy()
    lo = b["usable_lo"].astype(float).to_numpy()
    hi = b["usable_hi"].astype(float).to_numpy()
    yerr = np.vstack([means - lo, hi - means])

    fig, ax = figure(1, 1, figsize=(1.4 + 1.1 * len(order), 3.4))
    ax.bar(x, means, color=[method_colour(m) for m in order], zorder=3, width=0.6)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor=theme.NEUTRAL,
                capsize=4, lw=1.1, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([label_of(m) for m in order])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("usable rate")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


# ------------------------------------------------------------------ robustness
def _robustness(dataset: str) -> None:
    st.markdown("#### CAC stays usable when the assumptions bend")
    _rob_graph(dataset)
    st.divider()
    _rob_model(dataset)
    st.divider()
    _rob_sweep(dataset, arm="k", label="k", x_label="suggestions per applicant (k)",
               order=("k=1", "k=3", "k=5", "k=10"),
               heading="**More suggestions (k)**: how many tries each method needs",
               caption="CAC is already usable at k=1; baselines need more "
                       "suggestions to catch up --- and some never do.")
    st.divider()
    _rob_sweep(dataset, arm="budget", label="budget",
               x_label="action budget (max features changed)",
               order=("actions=1", "actions=2", "actions=3"),
               heading="**Tighter action budget**: usable rate as plans are capped",
               caption="Tighten the budget and baselines drop first; CAC "
                       "degrades gracefully as the action cap shrinks.")


def _rob_graph(dataset: str) -> None:
    rob = C.load_csv("robustness_headline")
    if rob is None:
        st.caption("No `robustness_headline.csv` yet (run experiment 04).")
        return
    block = rob[(rob["dataset"] == dataset)]
    if "arm" in block:
        block = block[block["arm"] == "scm"]
    if block.empty:
        st.caption("No graph-perturbation rows for this dataset.")
        return
    st.markdown("**Wrong causal graph** (drop/add edges): usable stays high.")
    show = block[["cell", "perturbation", "usable", "causal_validity",
                  "flip_survival"]].copy()
    show.columns = ["variant", "perturbation", "usable", "causal validity",
                    "flip survival"]
    st.dataframe(show.head(12), hide_index=True, width="stretch")
    st.caption("Even with edges dropped or added, CAC's usable rate barely moves "
               "--- it does not depend on a perfect graph.")


def _rob_model(dataset: str) -> None:
    mod = C.load_csv("robustness_model")
    if mod is None:
        return
    block = mod[mod["dataset"] == dataset]
    if block.empty:
        return
    st.markdown("**Different black box** (XGBoost \u2192 MLP): CAC transfers.")
    show = block[["method", "usable", "causal_validity", "actionability"]].copy()
    show.columns = ["method@model", "usable", "causal validity", "actionability"]
    st.dataframe(show.head(12), hide_index=True, width="stretch")


def _rob_sweep(dataset: str, arm: str, label: str, x_label: str,
               order: tuple[str, ...], heading: str, caption: str) -> None:
    """Usable rate vs a swept parameter (k or action budget), grouped by method.

    Reads ``robustness_headline`` rows for one ``arm``; each row is one
    (method, cell) point whose ``method`` column is encoded ``base/cell`` (e.g.
    ``cac/k=1``), so we derive the base method from the prefix and read ``usable``
    per cell directly rather than pivoting.
    """
    rob = C.load_csv("robustness_headline")
    if rob is None or "arm" not in rob:
        return
    block = rob[(rob["dataset"] == dataset) & (rob["arm"] == arm)].copy()
    if block.empty:
        st.caption(f"No {label}-sweep rows for this dataset.")
        return

    st.markdown(heading)
    block["base"] = block["method"].astype(str).str.split("/").str[0]
    cells = [c for c in order if c in set(block["cell"])]
    methods = [m for m in _METHOD_ORDER if m in set(block["base"])]
    if not cells or not methods:
        st.caption(f"No usable {label}-sweep points to plot.")
        return

    x = np.arange(len(cells))
    width = min(0.8 / max(len(methods), 1), 0.22)
    fig, ax = figure(1, 1, figsize=(1.8 + 1.2 * len(cells), 3.4))
    for mi, method in enumerate(methods):
        sub = block[block["base"] == method].set_index("cell")
        vals = [float(sub.loc[c, "usable"]) if c in sub.index else np.nan
                for c in cells]
        offs = x + (mi - (len(methods) - 1) / 2) * width
        ax.bar(offs, vals, width=width, color=method_colour(method),
               label=label_of(method), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cells)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("usable rate")
    ax.set_xlabel(x_label)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, ncol=min(len(methods), 3))
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.caption(caption)


# -------------------------------------------------------------------- ablation
def _ablation(dataset: str) -> None:
    abl = C.load_csv("ablation_headline")
    if abl is None:
        F.missing_card("The ablation study (ablation_headline.csv)",
                       how="Run `experiments/03_ablation.py`, then reload.")
        return
    block = abl[abl["dataset"] == dataset]
    if block.empty:
        st.caption("No ablation rows for this dataset.")
        return

    st.markdown("#### Turn off the causal term and plans stop surviving the world")
    _ablation_figure(block)

    full = _val(block, "cac", _ABLATION_METRIC)
    nocausal = _val(block, "cac_no_causal", _ABLATION_METRIC)
    if pd.notna(full) and pd.notna(nocausal):
        c1, c2, c3 = st.columns(3)
        c1.metric("CAC (full) flip survival", f"{full:.0%}")
        c2.metric("Without causal term", f"{nocausal:.0%}")
        c3.metric("Flip survival lost", f"\u2212{(full - nocausal):.0%}")
    st.caption("Turn off causal scoring and plans stop *surviving the world "
               "model* --- flip survival drops (and state fidelity slips with "
               "it). That isolated drop is the thesis; the usable rate can lag "
               "because a plan that no longer survives still looks actionable on "
               "paper.")

    show = block[block["method"].isin(["cac", "cac_no_causal"])][
        ["method", _ABLATION_METRIC, _ABLATION_SECONDARY, "usable",
         "causal_validity"]].copy()
    if not show.empty:
        show.columns = ["variant", "flip survival", "state fidelity", "usable",
                        "causal validity"]
        with st.expander("The numbers behind the bars"):
            st.dataframe(show, hide_index=True, width="stretch")


def _ablation_figure(block: pd.DataFrame) -> None:
    """Hero: per-plan flip survival collapsing when the causal term is removed.

    Plots ``flip_survival`` (the real signal) for CAC vs the no-causal ablation,
    with ``state_fidelity`` beside it as the lighter second bar.  ``usable`` is
    deliberately *not* the hero here: on usable the ablation barely moves, which
    is exactly why the old 'usable falls' framing was misleading.
    """
    variants = [v for v in ("cac", "cac_no_causal") if v in set(block["method"])]
    if not variants:
        return
    b = block.set_index("method")
    metrics = [(_ABLATION_METRIC, "flip survival"),
               (_ABLATION_SECONDARY, "state fidelity")]
    labels = {"cac": "CAC (full)", "cac_no_causal": "no causal term"}
    base_colour = {"cac": method_colour("cac"), "cac_no_causal": "#6fa8c7"}
    x = np.arange(len(variants))
    width = 0.36

    fig, ax = figure(1, 1, figsize=(4.6, 3.3))
    for mi, (col, _mlab) in enumerate(metrics):
        offs = x + (mi - 0.5) * width
        for xi, v in zip(offs, variants, strict=False):
            val = float(b.loc[v, col]) if col in b.columns else float("nan")
            if not np.isfinite(val):
                continue
            ax.bar(xi, val, width=width, color=base_colour.get(v, theme.NEUTRAL),
                   alpha=1.0 if mi == 0 else 0.4, zorder=3)
            ax.text(xi, val + 0.02, f"{val:.0%}", ha="center", fontsize=8,
                    color=theme.NEUTRAL)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(v, v) for v in variants])
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("rate")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch  # noqa: PLC0415

    handles = [Patch(facecolor=theme.NEUTRAL, alpha=1.0, label="flip survival"),
               Patch(facecolor=theme.NEUTRAL, alpha=0.4, label="state fidelity")]
    ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def _val(block: pd.DataFrame, method: str, col: str) -> float:
    sub = block[block["method"] == method]
    return float(sub[col].iloc[0]) if not sub.empty else float("nan")
