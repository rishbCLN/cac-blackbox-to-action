"""Experiment 06 --- does the usable gap survive the seed that produced it?

    python experiments/06_significance.py                     # all datasets, 5 seeds
    python experiments/06_significance.py -d german -s 42 0 1 # a quick slice
    python experiments/06_significance.py --aggregate-only    # re-tabulate, no search

Every headline in \\S9 is one run of a stochastic search: \\cac{}, Random, DiCE and
Wachter all depend on a seed (only the Oracle is deterministic).  Table
\\ref{tab:main_results} reports one draw; a referee is entitled to ask whether the
usable gap is an effect or an accident of ``random_state=42``.  This experiment
re-runs the whole \\S9 comparison under several seeds and answers two questions the
single run cannot:

*Is each rate stable?*  Every cell is reported as a mean over seeds with a
Student-$t$ confidence interval (:func:`~src.evaluation.significance.aggregate_seeds`).

*Is the gap significant?*  \\cac{} is compared to each baseline on the same records
with an exact McNemar test (:func:`~src.evaluation.significance.mcnemar`), the
effect sized with a paired bootstrap, and the family of baselines corrected with
Holm's step-down rule.

The audit is experiment 02's exactly --- the same :class:`~src.evaluation.metrics.
Evaluator`, the same three assessments, the same :func:`~src.evaluation.feasibility.
join_audits` definition of ``usable`` --- so a per-seed row here and the headline
row there are the same computation, differing only in the seed handed to the
search.  The black box and every shared artefact are fixed across seeds; only the
method's own randomness moves, which is what isolates search variance from
everything else.

Because the full run is many searches, it is **resumable**: per-seed rows are
appended to ``results/significance_runs.csv`` and ``significance_per_record.csv``,
and a later invocation skips any ``(dataset, method, seed)`` already present.  Run
it dataset by dataset and the aggregate tables rebuild from everything on disk.

Outputs
-------
``results/significance_runs.csv``        one row per (dataset, method, seed): the funnel rates
``results/significance_per_record.csv``  per (dataset, method, seed, record): the four audit bits
``results/significance_aggregate.csv``   one row per (dataset, method): mean + CI over seeds
``results/significance_tests.csv``       one row per (dataset, baseline): McNemar/bootstrap vs CAC
and ``paper/tables/{significance,significance_tests}.tex``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python experiments/06_...` without installing
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, DATASETS  # noqa: E402
from src.data.preprocess import load_dataset  # noqa: E402
from src.evaluation import faithfulness, feasibility, metrics, significance  # noqa: E402
from src.methods.base import Counterfactual, get_method  # noqa: E402
from src.model.blackbox import load_blackbox  # noqa: E402
from src.utils.io import save_dataframe, save_latex_table  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger("experiments.06_significance")

#: The method the paper argues for, and the baselines it is compared against.
REFERENCE = "cac"
METHOD_ORDER: tuple[str, ...] = ("random", "wachter", "dice", "oracle", "mint", "cac")

#: Seeds handed to the search.  The first is \S9's canonical run, so a single-seed
#: table is a strict subset of this one; the rest measure the spread around it.
DEFAULT_SEEDS: tuple[int, ...] = (42, 0, 1, 2, 3)

#: The one seed whose per-applicant outcome is the prespecified binary the McNemar
#: test reads.  It is \S9's canonical run, so the paired test asks exactly the
#: question the headline table answers --- one outcome per applicant, not one per
#: (applicant, seed) --- and no amount of extra seeds can inflate the pair count.
CANONICAL_SEED: int = DEFAULT_SEEDS[0]

#: Rates carried per seed, and the ones aggregated with a CI.  These are exactly
#: the funnel columns of the headline table, so the two cannot drift.
RATE_COLUMNS: tuple[str, ...] = ("coverage", "validity", "causal_validity",
                                 "state_fidelity", "actionability", "usable",
                                 "cost_mean", "plausibility_mean")

#: The per-record audit bits McNemar and the bootstrap consume.
AUDIT_BITS: tuple[str, ...] = ("valid", "causal_valid", "state_fidelity",
                               "actionable", "usable")

OUTPUTS: dict[str, str] = {
    "runs": "significance_runs.csv",
    "per_record": "significance_per_record.csv",
    "aggregate": "significance_aggregate.csv",
    "tests": "significance_tests.csv",
}

_DATASET_LABELS = {"adult": "Adult", "german": "German", "heloc": "HELOC",
                   "diabetes": "Diabetes"}
_METHOD_LABELS = {"random": "Random", "wachter": "Wachter", "dice": "DiCE",
                  "oracle": "Oracle", "mint": "MINT", "cac": "CAC"}


def _per_record_bits(joined: pd.DataFrame, dataset: str, method: str,
                     seed: int) -> pd.DataFrame:
    """Reduce the per-suggestion audit to one row per record: did *any* pass.

    The funnel credits a record to a method if any of its ``k`` suggestions
    clears the bar (\\S4), so the paired unit for a significance test is the
    record, scored by the disjunction over its suggestions.  ``valid`` etc. are
    booleans here, aligned across methods because every method saw the same query
    set under the same seed.  The columns mirror the headline (Finding 3):
    ``causal_valid`` is replayed-action success and ``usable`` is replayed
    success and action feasibility; state fidelity is reported, not gated.
    """
    tolerance = float(CONFIG.CAUSAL_TOLERANCE)
    records = joined.groupby("record")
    out = pd.DataFrame({
        "valid": records["valid"].any(),
        "causal_valid": records["flip_survives"].any(),
        "state_fidelity": (joined.assign(f=joined["valid"]
                                         & (joined["gap_max"] <= tolerance))
                           .groupby("record")["f"].any()),
        "actionable": records["actionable"].any(),
        "usable": records["usable"].any(),
    }).reset_index()
    out.insert(0, "dataset", dataset)
    out.insert(1, "method", method)
    out.insert(2, "seed", int(seed))
    return out


def _run_seed(dataset: str, evaluator: metrics.Evaluator, records: np.ndarray,
              X: np.ndarray, methods: Sequence[str], k: int, y_target: int,
              seed: int) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Every method on one dataset under one seed, audited exactly as \\S9 does.

    The evaluator (black box, SCM, cost model) is the one shared object; each
    method is rebuilt with ``random_state=seed`` so the only thing that changes
    between seeds is the search's own randomness.  Returns the per-seed funnel
    rows and the per-record audit bits.
    """
    rate_rows: list[dict[str, Any]] = []
    record_frames: list[pd.DataFrame] = []
    for name in methods:
        method = get_method(name, dataset, model=evaluator.model, random_state=seed)
        started = time.perf_counter()
        results: dict[int, list[Counterfactual]] = {}
        failures = 0
        for i, record in enumerate(records):
            try:
                results[int(record)] = method.generate(X[i], y_target=y_target, k=k)
            except Exception as exc:  # noqa: BLE001 - a crash is zero coverage
                LOGGER.warning("%s/%s seed %d record %d: %s: %s", dataset, name,
                               seed, int(record), type(exc).__name__, exc)
                results[int(record)] = []
                failures += 1
        elapsed = time.perf_counter() - started

        table = evaluator.table(results, method=name)
        faith = faithfulness.assess(evaluator, results, method=name)
        feas = feasibility.assess(evaluator, results, method=name)
        head = feasibility.headline(table, faith, feas)
        joined = feasibility.join_audits(table, faith, feas)

        row = head.iloc[0].to_dict() if not head.empty else {}
        rate_rows.append({
            "dataset": dataset, "method": name, "seed": int(seed),
            "n_records": int(records.size), "n_failed": int(failures),
            "wall_s": float(elapsed),
            **{c: float(row.get(c, float("nan"))) for c in RATE_COLUMNS},
        })
        record_frames.append(_per_record_bits(joined, dataset, name, seed))
        LOGGER.info("%s/%s seed %d: usable=%.3f causal=%.3f (%.1fs%s)", dataset,
                    name, seed, row.get("usable", float("nan")),
                    row.get("causal_validity", float("nan")), elapsed,
                    f", {failures} raised" if failures else "")
    return rate_rows, record_frames


def _load_existing(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _merge_new(existing: pd.DataFrame, fresh: pd.DataFrame,
               keys: Sequence[str]) -> pd.DataFrame:
    """Append ``fresh`` rows, letting a re-run of a cell overwrite the old one."""
    if existing.empty:
        return fresh
    if fresh.empty:
        return existing
    combined = pd.concat([existing, fresh], ignore_index=True)
    return combined.drop_duplicates(subset=list(keys), keep="last") \
        .reset_index(drop=True)


def _aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, method): the mean and CI of each rate over seeds."""
    agg = significance.aggregate_seeds(
        runs, keys=["dataset", "method"], metrics=list(RATE_COLUMNS))
    return _ordered(agg)


def _canonical_seed(seeds: Sequence[int]) -> int:
    """The prespecified seed for the McNemar binary, robust to a missing \\S9 run.

    The paired binary is one outcome per applicant at :data:`CANONICAL_SEED`.  If
    that seed was not among those actually run (a sliced invocation, or a test
    frame), fall back to the smallest seed present so the choice is deterministic
    and identical no matter how many *other* seeds were added --- the property the
    duplication regression pins down.
    """
    present = sorted({int(s) for s in seeds})
    if CANONICAL_SEED in present:
        return CANONICAL_SEED
    return present[0]


def _paired_tests(per_record: pd.DataFrame) -> pd.DataFrame:
    """McNemar + cluster bootstrap for CAC vs each baseline, Holm-corrected.

    The inferential unit is the **applicant**, not the ``(seed, record)`` cell: a
    record seen under five seeds is one person, and treating its five rows as five
    independent observations fabricates evidence (duplicating identical seeds alone
    drove an exact ``p`` from 0.0625 to 5.96e-8 without adding a single applicant).
    So each question is answered at the applicant level, and the per-seed spread is
    reported separately in the aggregate table rather than folded in here:

    * **Is the gap real?**  McNemar reads one prespecified binary per applicant ---
      ``usable`` at the canonical seed (``random_state=42``) --- so there is exactly
      one pair per applicant and the discordant count is a number of *people*.
    * **How big is it?**  The effect is each applicant's mean ``usable`` rate over
      the seeds, and :func:`~src.evaluation.significance.cluster_bootstrap_diff`
      resamples applicants (the clusters), so the interval rests on the number of
      people, not people times seeds.

    Holm still corrects the family of baselines within a dataset, since every row
    tests the same CAC column.
    """
    rows: list[dict[str, Any]] = []
    for dataset, group in per_record.groupby("dataset", sort=False):
        ref = group[group["method"] == REFERENCE]
        if ref.empty:
            continue
        canonical = _canonical_seed(group["seed"])
        # One prespecified outcome per applicant for the paired McNemar test.
        ref_bit = (ref[ref["seed"] == canonical]
                   .set_index("record")["usable"].sort_index())
        # Every seed's outcome per applicant for the cluster-bootstrap effect size.
        ref_long = ref.set_index(["seed", "record"]).sort_index()
        dataset_rows: list[dict[str, Any]] = []
        for method, sub in group.groupby("method", sort=False):
            if method == REFERENCE:
                continue
            oth_bit = (sub[sub["seed"] == canonical]
                       .set_index("record")["usable"].sort_index())
            applicants = ref_bit.index.intersection(oth_bit.index)
            oth_long = sub.set_index(["seed", "record"]).sort_index()
            shared = ref_long.index.intersection(oth_long.index)
            if applicants.empty and shared.empty:
                continue
            mc = significance.mcnemar(
                ref_bit.loc[applicants].to_numpy(dtype=bool),
                oth_bit.loc[applicants].to_numpy(dtype=bool))
            clusters = [rec for _seed, rec in shared]
            boot = significance.cluster_bootstrap_diff(
                ref_long.loc[shared, "usable"].to_numpy(dtype=float),
                oth_long.loc[shared, "usable"].to_numpy(dtype=float),
                clusters)
            dataset_rows.append({
                "dataset": dataset, "method": method, "reference": REFERENCE,
                "canonical_seed": int(canonical),
                "n_pairs": mc.n_pairs, "n_applicants": boot.n,
                "cac_only": mc.reference_only,
                "baseline_only": mc.other_only, "discordant": mc.discordant,
                "usable_gap": boot.mean, "gap_lo": boot.lo, "gap_hi": boot.hi,
                "mcnemar_p": mc.p_value,
            })
        corrected = significance.holm_bonferroni([r["mcnemar_p"] for r in dataset_rows])
        for row, (_, adj, reject) in zip(dataset_rows, corrected):
            row["mcnemar_p_holm"] = adj
            row["significant"] = bool(reject)
        rows.extend(dataset_rows)
    return _ordered(pd.DataFrame(rows))


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "method" not in frame.columns:
        return frame
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    out = frame.copy()
    out["_o"] = out["method"].map(lambda m: rank.get(str(m), len(rank)))
    keys = [c for c in ("dataset", "_o") if c in out.columns or c == "_o"]
    return out.sort_values(keys, kind="stable").drop(columns="_o") \
        .reset_index(drop=True)


# --------------------------------------------------------------------- tables
def _ci_cell(frame: pd.DataFrame, metric: str) -> list[str]:
    """A ``mean $\\pm$ half`` string per row for one aggregated metric."""
    out: list[str] = []
    for _, row in frame.iterrows():
        ci = significance.CI(
            mean=float(row[f"{metric}_mean"]),
            lo=float(row.get(f"{metric}_lo", float("nan"))),
            hi=float(row.get(f"{metric}_hi", float("nan"))),
            sem=float(row.get(f"{metric}_sem", float("nan"))),
            std=float(row.get(f"{metric}_std", float("nan"))),
            n=int(row["n_seeds"]))
        out.append(ci.plus_minus())
    return out


def _write_tables(aggregate: pd.DataFrame, tests: pd.DataFrame,
                  n_seeds: int) -> None:
    if not aggregate.empty:
        table = pd.DataFrame({
            "Dataset": aggregate["dataset"].map(lambda d: _DATASET_LABELS.get(d, d)),
            "Method": aggregate["method"].map(lambda m: _METHOD_LABELS.get(m, m)),
            "Valid": _ci_cell(aggregate, "validity"),
            "Causal": _ci_cell(aggregate, "causal_validity"),
            "Fid.": _ci_cell(aggregate, "state_fidelity"),
            "Action.": _ci_cell(aggregate, "actionability"),
            "Usable": _ci_cell(aggregate, "usable"),
        })
        save_latex_table(
            table, "significance",
            caption=(f"Recourse quality over {n_seeds} seeds, as "
                     "mean $\\pm$ half the 95\\% Student-$t$ interval across seeds. "
                     "\\emph{Causal} is replayed-action success and \\emph{Usable} "
                     "is replayed success with action feasibility; \\emph{Fid.} "
                     "(state fidelity) is reported beside the endpoint, not folded "
                     "into it. "
                     "Every rate is the per-record funnel of "
                     "Table~\\ref{tab:main_results}, recomputed under each seed of "
                     "the search; the Oracle is deterministic and so has a "
                     "zero-width interval by construction. This interval measures "
                     "the \\emph{optimizer}'s variation over seeds with the "
                     "applicants and the black box held fixed --- it is a statement "
                     "about search stability, not an applicant-level confidence "
                     "interval, and a zero-width cell at $1.00$ does not imply "
                     "certainty about unseen applicants. The applicant-level test "
                     "of the gap is Table~\\ref{tab:significance_tests}. A tight "
                     "interval on \\cac{}'s \\emph{Usable} column is the claim that "
                     "its advantage is not an artefact of one lucky "
                     "\\texttt{random\\_state}."),
            label="tab:significance", column_format="llrrrrr", escape=False)

    if not tests.empty:
        def _stars(p: float) -> str:
            return ("***" if p < 1e-3 else "**" if p < 1e-2
                    else "*" if p < 5e-2 else "n.s.")
        table = pd.DataFrame({
            "Dataset": tests["dataset"].map(lambda d: _DATASET_LABELS.get(d, d)),
            "Baseline": tests["method"].map(lambda m: _METHOD_LABELS.get(m, m)),
            "Applicants": tests["n_pairs"].astype(int),
            "CAC only": tests["cac_only"].astype(int),
            "Base only": tests["baseline_only"].astype(int),
            "Usable gap": [f"${g:.3f}$ [{lo:.3f}, {hi:.3f}]" for g, lo, hi
                           in zip(tests["usable_gap"], tests["gap_lo"],
                                  tests["gap_hi"])],
            "McNemar $p$": [f"${p:.1e}$ {_stars(p)}" for p in tests["mcnemar_p_holm"]],
        })
        save_latex_table(
            table, "significance_tests",
            caption=("\\cac{} against each baseline on the \\emph{usable} rate, "
                     "paired per \\emph{applicant} rather than per (seed, record) "
                     "cell: a record seen under several seeds is one person. The "
                     "\\emph{Applicants} column counts the query records compared. "
                     "\\emph{CAC only} and \\emph{Base only} count the applicants "
                     "exactly one method made usable at the prespecified canonical "
                     "seed (\\texttt{random\\_state}=42) --- the discordant pairs "
                     "the exact McNemar test conditions on. \\emph{Usable gap} is "
                     "the mean over applicants of each applicant's \\emph{usable}"
                     "-rate difference across the seeds, with a 10{,}000-sample "
                     "cluster bootstrap that resamples applicants, so repeating a "
                     "person under more seeds cannot narrow it. \\emph{McNemar $p$} "
                     "is the exact two-sided value after Holm correction across the "
                     "baselines of each dataset (\\texttt{***} $p<10^{-3}$, "
                     "\\texttt{**} $p<10^{-2}$, \\texttt{*} $p<0.05$, "
                     "\\texttt{n.s.} otherwise)."),
            label="tab:significance_tests", column_format="llrrrrr", escape=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repeat \\S9's comparison under several seeds and test the gap.")
    parser.add_argument("-d", "--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("-m", "--methods", nargs="+", default=list(METHOD_ORDER),
                        choices=list(METHOD_ORDER))
    parser.add_argument("-s", "--seeds", nargs="+", type=int,
                        default=list(DEFAULT_SEEDS),
                        help="Seeds handed to each stochastic search.")
    parser.add_argument("-n", "--n-records", type=int, default=None,
                        help=f"Query-set size per dataset (default "
                             f"{CONFIG.N_QUERY_INSTANCES}).")
    parser.add_argument("-k", type=int, default=int(CONFIG.N_COUNTERFACTUALS))
    parser.add_argument("-t", "--target", type=int, default=1, choices=(0, 1))
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip every search; re-tabulate from the CSVs on disk.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if a baseline is not significantly "
                             "beaten where the point estimates say it should be.")
    args = parser.parse_args(argv)

    setup_logging("06_significance")
    methods = [m for m in METHOD_ORDER if m in args.methods]
    if REFERENCE not in methods:
        LOGGER.error("%s must be among the methods for a paired test", REFERENCE)
        return 1

    runs_path = CONFIG.RESULTS_DIR / OUTPUTS["runs"]
    record_path = CONFIG.RESULTS_DIR / OUTPUTS["per_record"]
    runs = _load_existing(runs_path)
    per_record = _load_existing(record_path)

    if not args.aggregate_only:
        done: set[tuple[str, str, int]] = set()
        if not runs.empty:
            done = {(str(r.dataset), str(r.method), int(r.seed))
                    for r in runs.itertuples()}
        LOGGER.info("Seeds %s x datasets %s x methods %s (%d cells already on disk)",
                    args.seeds, args.datasets, methods, len(done))
        for dataset in args.datasets:
            todo_seeds = [s for s in args.seeds
                          if any((dataset, m, int(s)) not in done for m in methods)]
            if not todo_seeds:
                LOGGER.info("%s: every requested cell already present, skipping",
                            dataset)
                continue
            bundle = load_dataset(dataset)
            model = load_blackbox(dataset, CONFIG.BLACKBOX_TYPE)
            evaluator = metrics.Evaluator.build(dataset, bundle=bundle, model=model)
            records = metrics.query_instances(dataset, model=model, bundle=bundle,
                                              n=args.n_records, y_target=args.target)
            if records.size == 0:
                LOGGER.error("%s: the black box rejects nobody; nothing to explain",
                             dataset)
                continue
            X = bundle.X_test[records]
            for seed in todo_seeds:
                pending = [m for m in methods
                           if (dataset, m, int(seed)) not in done]
                rate_rows, record_frames = _run_seed(
                    dataset, evaluator, records, X, pending, args.k, args.target,
                    int(seed))
                runs = _merge_new(runs, pd.DataFrame(rate_rows),
                                  keys=["dataset", "method", "seed"])
                per_record = _merge_new(
                    per_record, pd.concat(record_frames, ignore_index=True),
                    keys=["dataset", "method", "seed", "record"])
                # Persist after every seed so a long run is resumable mid-flight.
                save_dataframe(_ordered(runs), runs_path)
                save_dataframe(_ordered(per_record), record_path)

    if runs.empty:
        LOGGER.error("No per-seed results on disk; run without --aggregate-only first.")
        return 1

    aggregate = _aggregate(runs)
    tests = _paired_tests(per_record)
    save_dataframe(aggregate, CONFIG.RESULTS_DIR / OUTPUTS["aggregate"])
    save_dataframe(tests, CONFIG.RESULTS_DIR / OUTPUTS["tests"])
    n_seeds = int(runs.groupby(["dataset", "method"]).size().max())
    _write_tables(aggregate, tests, n_seeds)

    LOGGER.info("aggregate (usable, mean over seeds):\n%s",
                aggregate[["dataset", "method", "usable_mean", "usable_lo",
                           "usable_hi", "n_seeds"]].to_string(index=False))
    if not tests.empty:
        LOGGER.info("paired tests vs %s:\n%s", REFERENCE,
                    tests[["dataset", "method", "usable_gap", "mcnemar_p_holm",
                           "significant"]].to_string(index=False))

    problems: list[str] = []
    for row in tests.to_dict("records"):
        if row["usable_gap"] > 0.05 and not row["significant"]:
            problems.append(f"{row['dataset']}/{row['method']}: CAC leads by "
                            f"{row['usable_gap']:.3f} but not significantly "
                            f"(Holm p={row['mcnemar_p_holm']:.3f})")
    if problems:
        LOGGER.warning("Point estimate without significance: %s", "; ".join(problems))
        if args.strict:
            return 1
    LOGGER.info("Wrote %d CSVs to %s and 2 LaTeX tables to %s", len(OUTPUTS),
                CONFIG.RESULTS_DIR, CONFIG.TABLES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
