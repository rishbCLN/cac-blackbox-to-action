"""Analysis for the recourse-usefulness user study.

    python user_study/analysis.py                 # synthetic responses, full analysis
    python user_study/analysis.py --responses my_responses.csv

**No real responses have been collected.** The study (see ``consent.md`` and
``questionnaire.md``) is shipped as a protocol. So that the pipeline runs end to
end and the analysis is exercised, this script *simulates* a response set that is
deterministic (``random_state=42``) and grounded in the real curated cases in
``results/qualitative_cases.csv``: simulated respondents rate genuinely-usable
plans a little higher on feasibility and trust, cannot tell a causally-broken plan
from a good one on feasibility alone (the point of H2), and feel CAC's extra
effort. Pass ``--responses`` with a CSV in the schema of ``questionnaire.md`` to
analyse a real study instead; the same code runs unchanged.

Outputs
-------
``results/user_study_summary.csv`` (per-method mean ratings and forced-choice
share), ``paper/tables/user_study.tex``, and
``paper/figures/user_study/user_study_ratings.{png,pdf}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.utils.io import save_dataframe, save_figure, save_latex_table  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

try:  # scipy is a hard dependency of the project; guard only for a clean message
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover - scipy is installed in this project
    mannwhitneyu = None

LOGGER = get_logger("user_study.analysis")

RATINGS = ["feasibility", "effort", "trust", "willingness"]
N_RESPONDENTS = 60  # simulated; a real study would set this from the collected data
LIKERT = (1, 5)


# --------------------------------------------------------------------------- #
# Simulated responses (used only when no real response CSV is supplied).
# --------------------------------------------------------------------------- #
def _clip(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(a), *LIKERT).astype(int)


def simulate_responses(cases: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A deterministic, honest-by-construction synthetic response set.

    The generative story encodes the study's own hypotheses so the analysis has
    something to find; it is *not* evidence, and the paper/README say so. Real
    respondents replace it via ``--responses``.
    """
    leading = cases[cases["rank"] == 0].copy()
    leading = leading[leading["advice"].astype(str).str.strip().ne("")]
    scenarios = leading.groupby(["dataset", "case"])
    rows = []

    for r in range(N_RESPONDENTS):
        rid = f"sim-{r:03d}"
        for (dataset, case), block in scenarios:
            sid = f"{dataset}:{case}"
            best_score, best_method = -np.inf, None
            per_plan = {}
            for _, plan in block.iterrows():
                usable = bool(plan["usable"])
                cost = float(plan.get("cost", 0.0) or 0.0)
                # Feasibility: people judge the *stated* action, so a
                # causally-broken-but-simple plan can look just as feasible.
                # It rises when the plan is actionable and cheap.
                actionable = bool(plan.get("actionable", usable))
                feas = 3.4 + (0.8 if actionable else -1.3) - 1.6 * cost
                # Trust: here the causal truth leaks in a little (some people
                # sense a too-good-to-be-true plan), but weakly -- the audit is
                # needed precisely because humans cannot see it reliably.
                trust = 3.0 + (0.7 if usable else -0.4) - 1.0 * cost
                # Effort: tracks measured cost directly.
                effort = 1.8 + 3.2 * cost
                # Willingness blends feasibility and trust.
                will = 0.5 * feas + 0.5 * trust - 0.2
                noise = rng.normal(0, 0.7, size=4)
                fe, tr, ef, wi = (
                    _clip(np.array([feas]) + noise[0])[0],
                    _clip(np.array([trust]) + noise[1])[0],
                    _clip(np.array([effort]) + noise[2])[0],
                    _clip(np.array([will]) + noise[3])[0],
                )
                per_plan[plan["method"]] = dict(
                    respondent_id=rid, scenario_id=sid, dataset=dataset, case=case,
                    method=plan["method"], plan_rank=int(plan["rank"]),
                    feasibility=fe, effort=ef, trust=tr, willingness=wi, chosen=0)
                # Forced choice: respondents pick on perceived value =
                # willingness, but are told the outcome, so a non-usable plan is
                # penalised at choice time (H2's debrief effect).
                choice_score = wi + (1.2 if usable else -1.2) + rng.normal(0, 0.5)
                if choice_score > best_score:
                    best_score, best_method = choice_score, plan["method"]
            if best_method is not None:
                per_plan[best_method]["chosen"] = 1
            rows.extend(per_plan.values())

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Analysis (identical for simulated and real responses).
# --------------------------------------------------------------------------- #
def summarise(resp: pd.DataFrame) -> pd.DataFrame:
    g = resp.groupby("method")
    summary = g[RATINGS].mean().round(2)
    summary["forced_choice"] = g["chosen"].mean().round(3)
    summary["n_ratings"] = g.size()
    order = [m for m in ("cac", "oracle", "dice", "wachter", "random")
             if m in summary.index]
    return summary.reindex(order)


def test_hypotheses(resp: pd.DataFrame) -> list[str]:
    """H1: usable plans score higher on feasibility & trust than non-usable."""
    lines = []
    if mannwhitneyu is None:
        return ["scipy unavailable; skipped significance tests"]

    # Reconstruct usability of each rated plan from the case table join done in
    # main(); resp carries a boolean 'usable' column by then.
    for metric in ("feasibility", "trust"):
        a = resp.loc[resp["usable"], metric]
        b = resp.loc[~resp["usable"], metric]
        if len(a) and len(b):
            u, p = mannwhitneyu(a, b, alternative="greater")
            lines.append(
                f"H1 [{metric}]: usable ({a.mean():.2f}) vs not-usable "
                f"({b.mean():.2f}), Mann-Whitney U={u:.0f}, p={p:.2e}")
    # H3: CAC effort vs cheapest baseline (random) on Adult.
    ad = resp[resp["dataset"] == "adult"]
    cac_e = ad.loc[ad["method"] == "cac", "effort"]
    base_e = ad.loc[ad["method"] == "random", "effort"]
    if len(cac_e) and len(base_e):
        u, p = mannwhitneyu(cac_e, base_e, alternative="greater")
        lines.append(
            f"H3 [effort, adult]: CAC ({cac_e.mean():.2f}) > random "
            f"({base_e.mean():.2f}), U={u:.0f}, p={p:.2e}")
    # Forced choice share for CAC.
    fc = resp.groupby("method")["chosen"].mean()
    if "cac" in fc:
        lines.append(f"Forced-choice: CAC chosen {fc['cac']:.0%} of scenarios "
                     f"(next best {fc.drop('cac').max():.0%})")
    return lines


def make_figure(summary: pd.DataFrame):
    import matplotlib.pyplot as plt

    from src.viz import label_of  # local import: viz sets the house style

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    metrics = ["feasibility", "trust", "willingness"]
    methods = list(summary.index)
    x = np.arange(len(methods))
    w = 0.25
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * w, summary[m], w, label=m.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels([label_of(m) for m in methods])
    ax.set_ylabel("mean rating (1–5)")
    ax.set_ylim(1, 5)
    ax.set_title("Perceived usefulness of each method's recourse")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse the recourse user study.")
    parser.add_argument("--responses", type=Path, default=None,
                        help="CSV of real responses (schema in questionnaire.md); "
                             "if omitted, a deterministic synthetic set is used.")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    setup_logging("user_study_analysis")

    cases_path = CONFIG.RESULTS_DIR / "qualitative_cases.csv"
    if not cases_path.exists():
        LOGGER.error("Missing %s; run experiment 05 first.", cases_path)
        return 1
    cases = pd.read_csv(cases_path)

    if args.responses is not None:
        resp = pd.read_csv(args.responses)
        LOGGER.info("Loaded %d real ratings from %s", len(resp), args.responses)
        source = "real"
    else:
        rng = np.random.default_rng(CONFIG.RANDOM_STATE)
        resp = simulate_responses(cases, rng)
        LOGGER.info("Simulated %d ratings from %d respondents (SYNTHETIC).",
                    len(resp), N_RESPONDENTS)
        source = "synthetic"

    # Join each rated plan back to its audited usability.
    key = ["dataset", "case", "method", "plan_rank"]
    audit = (cases.rename(columns={"rank": "plan_rank"})
                  [key + ["usable"]].drop_duplicates(key))
    resp = resp.merge(audit, on=key, how="left")
    resp["usable"] = resp["usable"].fillna(False).astype(bool)

    save_dataframe(resp, CONFIG.RESULTS_DIR / "user_study_responses.csv")
    summary = summarise(resp)
    LOGGER.info("Per-method summary:\n%s", summary.to_string())
    save_dataframe(summary.reset_index(),
                   CONFIG.RESULTS_DIR / "user_study_summary.csv")

    for line in test_hypotheses(resp):
        LOGGER.info(line)

    table = summary.reset_index().rename(columns={
        "method": "Method", "feasibility": "Feasibility", "effort": "Effort",
        "trust": "Trust", "willingness": "Willing.", "forced_choice": "Chosen",
        "n_ratings": "n"})
    table["Method"] = table["Method"].str.upper()
    save_latex_table(
        table.drop(columns=["n"]),
        "user_study",
        caption=(f"Recourse usefulness as rated in the user study ({source} "
                 "responses). Ratings are 1--5 means; \\emph{Chosen} is the share "
                 "of scenarios in which each method's plan won the forced choice. "
                 "The synthetic responses shipped with the repository are a "
                 "protocol demonstration, not evidence."),
        label="tab:user_study",
        column_format="lrrrrr",
        float_format="%.2f",
        escape=False,
    )

    if not args.no_figure:
        try:
            fig = make_figure(summary)
            save_figure(fig, "user_study_ratings", subdir="user_study")
        except Exception as exc:  # pragma: no cover - headless / backend issues
            LOGGER.warning("Figure step skipped: %s", exc)

    LOGGER.info("Done (%s responses).", source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
