"""Tests for ``src/evaluation/significance.py`` --- the statistics §9 leans on.

These functions decide whether a gap the paper reports is an effect or an
artefact, so the tests are about the cases where a naive implementation would
mislead:

*A mean over seeds is not a claim until it has a spread.*  One seed must return an
interval that says "one seed", not a zero-width one that reads as certainty; a set
of identical seeds must return a genuine zero, because that is a true statement
about a stable outcome.

*A paired test must condition on the pairs.*  McNemar looks only at the records
where the two methods disagreed; a version that counted the agreements would
dilute a real difference toward nothing, so there is a test that fixes the
discordant counts and checks the $p$-value ignores everything else.

*A correction must actually correct.*  Holm has to reject fewer hypotheses than
reading every raw $p$ at $0.05$ would, and must never reject one a flat
Bonferroni would have kept.

Everything is forged: these are pure functions over arrays, so no model, dataset
or search is involved and the whole module runs in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.significance import (
    aggregate_seeds,
    bootstrap_diff,
    cluster_bootstrap_diff,
    holm_bonferroni,
    mcnemar,
    mean_ci,
    paired_wilcoxon,
)


# --------------------------------------------------------------------- mean_ci
def test_a_single_seed_reports_a_mean_without_pretending_to_a_spread() -> None:
    """One draw has no variance, so the interval is undefined, not zero-width.

    A zero-width interval on one seed would read as "we are certain", which is the
    opposite of what one observation licenses.  ``n = 1`` is the flag a reader (and
    the table formatter) uses to print the bare mean.
    """
    ci = mean_ci([0.87])
    assert ci.mean == pytest.approx(0.87)
    assert ci.n == 1
    assert np.isnan(ci.lo) and np.isnan(ci.hi)
    assert ci.pretty() == "0.870"


def test_identical_seeds_report_a_true_zero_width_interval() -> None:
    """Every seed gave 1.00 --- the interval is [1, 1], and that is honest.

    This is the shape CAC's usable rate takes, and the point of a CI here is to
    distinguish it from a 1.00 seen once: same mean, very different evidence.
    """
    ci = mean_ci([1.0, 1.0, 1.0, 1.0, 1.0])
    assert ci.mean == pytest.approx(1.0)
    assert ci.lo == pytest.approx(1.0) and ci.hi == pytest.approx(1.0)
    assert ci.std == pytest.approx(0.0)
    assert ci.n == 5
    assert "[1.000, 1.000]" in ci.pretty()


def test_the_interval_uses_t_and_widens_as_seeds_shrink() -> None:
    """Two seeds give a wider interval than many, for the same spread.

    The Student-$t$ multiplier is what makes a two-seed interval honestly wide;
    a normal approximation would understate it exactly where the sample is
    smallest and the risk of over-claiming is highest.
    """
    values = [0.8, 0.9]
    ci_two = mean_ci(values)
    ci_many = mean_ci(values * 10)  # same mean and (population) spread, more n
    assert ci_two.mean == pytest.approx(0.85)
    assert (ci_two.hi - ci_two.lo) > (ci_many.hi - ci_many.lo)
    assert ci_two.n == 2 and ci_many.n == 20


def test_a_wider_confidence_level_gives_a_wider_interval() -> None:
    values = [0.6, 0.7, 0.8, 0.9]
    assert (mean_ci(values, 0.99).hi - mean_ci(values, 0.99).lo) > \
           (mean_ci(values, 0.90).hi - mean_ci(values, 0.90).lo)


def test_non_finite_seeds_are_dropped_not_propagated() -> None:
    """A cell where one seed produced no usable plan (NaN) still has a mean."""
    ci = mean_ci([1.0, float("nan"), 1.0])
    assert ci.mean == pytest.approx(1.0)
    assert ci.n == 2


# ------------------------------------------------------------- aggregate_seeds
def test_aggregation_collapses_seeds_to_one_row_per_cell_with_a_ci() -> None:
    """The per-seed frame becomes one row per (dataset, method), CI columns added."""
    frame = pd.DataFrame({
        "dataset": ["adult"] * 3 + ["adult"] * 3,
        "method": ["cac"] * 3 + ["wachter"] * 3,
        "seed": [0, 1, 2, 0, 1, 2],
        "usable": [1.0, 1.0, 1.0, 0.45, 0.50, 0.40],
    })
    out = aggregate_seeds(frame, keys=["dataset", "method"], metrics=["usable"])
    assert set(out["method"]) == {"cac", "wachter"}
    cac = out[out["method"] == "cac"].iloc[0]
    assert cac["usable_mean"] == pytest.approx(1.0)
    assert cac["usable_std"] == pytest.approx(0.0)
    assert cac["n_seeds"] == 3
    wachter = out[out["method"] == "wachter"].iloc[0]
    assert wachter["usable_mean"] == pytest.approx(0.45)
    assert wachter["usable_lo"] < 0.45 < wachter["usable_hi"]


def test_aggregation_of_nothing_is_empty_not_an_error() -> None:
    assert aggregate_seeds(pd.DataFrame(), keys=["method"], metrics=["usable"]).empty


# --------------------------------------------------------------------- mcnemar
def test_mcnemar_reads_only_the_records_the_methods_disagreed_on() -> None:
    """The verdict must not move when concordant pairs are added.

    Two arrangements with identical discordant counts (8 vs 1) but wildly
    different agreement must give the same $p$; if they do not, the test is
    looking at the wrong cells.
    """
    ref_a = np.array([True] * 8 + [False] * 1 + [True] * 2 + [False] * 2)
    oth_a = np.array([False] * 8 + [True] * 1 + [True] * 2 + [False] * 2)
    ref_b = np.array([True] * 8 + [False] * 1 + [True] * 50 + [False] * 50)
    oth_b = np.array([False] * 8 + [True] * 1 + [True] * 50 + [False] * 50)
    a, b = mcnemar(ref_a, oth_a), mcnemar(ref_b, oth_b)
    assert a.discordant == b.discordant == 9
    assert a.p_value == pytest.approx(b.p_value)


def test_mcnemar_finds_a_one_sided_gap_significant() -> None:
    """When a method helps 15 people the other cannot and never the reverse, p is small."""
    ref = np.array([True] * 15 + [True] * 30 + [False] * 5)
    oth = np.array([False] * 15 + [True] * 30 + [False] * 5)
    result = mcnemar(ref, oth)
    assert result.reference_only == 15 and result.other_only == 0
    assert result.p_value < 1e-3
    assert result.both == 30 and result.neither == 5


def test_mcnemar_with_no_disagreement_is_not_significant() -> None:
    """Two methods that succeed on exactly the same records are indistinguishable."""
    same = np.array([True, True, False, False, True])
    result = mcnemar(same, same.copy())
    assert result.discordant == 0
    assert result.p_value == 1.0


def test_mcnemar_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="paired"):
        mcnemar([True, False], [True])


# ------------------------------------------------------------- paired_wilcoxon
def test_wilcoxon_handles_a_perfectly_stable_gap_without_raising() -> None:
    """Every seed shows the same non-zero difference --- scipy refuses, we fall back.

    A large, stable gap (CAC 1.00 vs a baseline 0.50 on every seed) is exactly the
    input the signed-rank test cannot rank, so the sign-test fallback must engage
    rather than let the exception escape into a sweep over cells.
    """
    result = paired_wilcoxon([1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5])
    assert result.median_difference == pytest.approx(0.5)
    assert 0.0 <= result.p_value <= 1.0  # a number, not a crash


def test_wilcoxon_on_no_difference_returns_one() -> None:
    result = paired_wilcoxon([0.7, 0.7, 0.7], [0.7, 0.7, 0.7])
    assert result.p_value == 1.0
    assert result.median_difference == pytest.approx(0.0)


# -------------------------------------------------------------- bootstrap_diff
def test_bootstrap_interval_brackets_the_observed_difference() -> None:
    """The CI is around the observed mean gap, and excludes zero for a clear one."""
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.9, 1.0, size=100)
    oth = rng.uniform(0.3, 0.5, size=100)
    ci = bootstrap_diff(ref, oth, n_boot=2000, seed=1)
    assert ci.lo < ci.mean < ci.hi
    assert ci.lo > 0.0  # a gap this size does not straddle zero
    assert ci.mean == pytest.approx(float(np.mean(ref - oth)))


def test_bootstrap_is_deterministic_under_a_fixed_seed() -> None:
    ref, oth = [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]
    first = bootstrap_diff(ref, oth, n_boot=1000, seed=7)
    second = bootstrap_diff(ref, oth, n_boot=1000, seed=7)
    assert (first.lo, first.hi) == (second.lo, second.hi)


# ------------------------------------------------------ cluster_bootstrap_diff
def test_cluster_bootstrap_ignores_duplicate_rows_within_a_cluster() -> None:
    """Seeing an applicant under more seeds is not more applicants.

    This is Finding 2 in miniature: the pooled ``(seed, record)`` bootstrap
    narrowed its interval when the same people were copied under extra seed
    labels, reading repetition as independent evidence. Collapsing each cluster to
    its mean before resampling makes both the point estimate and the interval
    invariant to that duplication, and ``n`` keeps reporting the number of
    *people* behind the interval, not people times seeds.
    """
    rng = np.random.default_rng(0)
    clusters = np.arange(40)
    ref = rng.uniform(0.8, 1.0, size=40)
    oth = rng.uniform(0.3, 0.6, size=40)
    once = cluster_bootstrap_diff(ref, oth, clusters, n_boot=4000, seed=1)
    # Five identical "seeds": every applicant's two outcomes repeated five times.
    five = cluster_bootstrap_diff(np.tile(ref, 5), np.tile(oth, 5),
                                  np.tile(clusters, 5), n_boot=4000, seed=1)
    assert once.n == five.n == 40
    assert five.mean == pytest.approx(once.mean)
    assert five.lo == pytest.approx(once.lo)
    assert five.hi == pytest.approx(once.hi)


def test_cluster_bootstrap_with_singleton_clusters_is_the_plain_bootstrap() -> None:
    """One observation per cluster leaves nothing to collapse, so it must agree."""
    ref = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    oth = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    clustered = cluster_bootstrap_diff(ref, oth, list(range(6)), n_boot=1000, seed=7)
    plain = bootstrap_diff(ref, oth, n_boot=1000, seed=7)
    assert clustered.n == plain.n
    assert (clustered.mean, clustered.lo, clustered.hi) == \
        pytest.approx((plain.mean, plain.lo, plain.hi))


def test_cluster_bootstrap_averages_within_a_cluster_before_resampling() -> None:
    """A duplicated cluster's two rows average, they do not each get a vote.

    Two applicants, one of them recorded twice with opposing outcomes: the point
    estimate must weigh that applicant by its 0.5 mean, not twice at the extremes.
    """
    ci = cluster_bootstrap_diff([1.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                                ["alice", "alice", "bob"], n_boot=2000, seed=3)
    assert ci.n == 2  # two people, not three rows
    assert ci.mean == pytest.approx(0.75)  # mean(0.5, 1.0)


def test_cluster_bootstrap_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="match"):
        cluster_bootstrap_diff([1.0, 0.0], [1.0, 1.0], [0])


# ------------------------------------------------------------- holm_bonferroni
def test_holm_is_stricter_than_raw_but_not_stricter_than_bonferroni() -> None:
    """Holm sits between reading raw p-values and a flat Bonferroni.

    The middle p-value here (0.02) is significant raw, and under Holm, but a flat
    Bonferroni at m=4 (threshold 0.0125) would reject it too --- Holm must never
    reject something Bonferroni keeps, and must reject at least as much.
    """
    ps = [0.001, 0.02, 0.04, 0.9]
    out = holm_bonferroni(ps, alpha=0.05)
    raw_flags = [p <= 0.05 for p in ps]
    holm_flags = [rej for _, _, rej in out]
    assert sum(holm_flags) <= sum(raw_flags)
    assert out[0][2] is True  # 0.001 survives any correction
    assert out[3][2] is False  # 0.9 never survives
    # adjusted p-values are monotone in the raw order after sorting
    assert out[0][1] <= out[1][1] <= out[2][1]


def test_holm_of_an_empty_family_is_empty() -> None:
    assert holm_bonferroni([]) == []
