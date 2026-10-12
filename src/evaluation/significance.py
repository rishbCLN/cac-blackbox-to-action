"""Does the gap survive the randomness that produced it?  (§9's statistics.)

Every headline number in §9 is an average over one query set, produced by a
search that starts from a seed.  Three of the four baselines and \\cac{} itself
are stochastic — random restarts, sampled perturbations, a bred population — so a
single run reports a point where the honest object is a distribution.  A referee
is right to ask two separate questions about it, and this module answers each
with its own test rather than blurring them into one:

*Is the number stable?*  :func:`aggregate_seeds` repeats the whole comparison
under several seeds and reports, per cell, the mean and a Student-$t$ confidence
interval over seeds.  A usable rate of ``1.00`` with a zero-width interval is a
different claim from ``1.00`` seen once, and the table should be able to tell them
apart.

*Is the gap real?*  A difference between two methods can be large and still be
noise if the query set is small or the two methods happen to fail on different
people.  :func:`mcnemar` compares two methods on the *same* records: it looks
only at the applicants where exactly one of them succeeded (the discordant
pairs), which is the paired comparison the design calls for, and computes an
exact binomial $p$-value that needs no large-sample approximation.
:func:`bootstrap_diff` puts a confidence interval on the size of the gap by
resampling the records, so the effect is reported with a magnitude and not only a
verdict.  :func:`cluster_bootstrap_diff` is the same idea when a record appears
more than once --- once per seed --- and resamples the *applicant* rather than the
row, so repeating a person under more seeds cannot masquerade as more evidence.

Nothing here knows what a "method" or a "dataset" is; the functions take arrays
and frames, so they are exercised by unit tests on forged data and reused
unchanged by :mod:`experiments.06_significance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t, wilcoxon

__all__ = [
    "CI",
    "McNemarResult",
    "PairedResult",
    "mean_ci",
    "aggregate_seeds",
    "mcnemar",
    "paired_wilcoxon",
    "bootstrap_diff",
    "cluster_bootstrap_diff",
    "holm_bonferroni",
]


@dataclass(frozen=True, slots=True)
class CI:
    """A mean with a symmetric-in-probability confidence interval over samples."""

    mean: float
    lo: float
    hi: float
    sem: float
    std: float
    n: int

    def pretty(self, fmt: str = "%.3f") -> str:
        """``mean [lo, hi]`` for a table cell, or just the mean when n < 2."""
        if self.n < 2 or not np.isfinite(self.lo):
            return fmt % self.mean
        return f"{fmt % self.mean} [{fmt % self.lo}, {fmt % self.hi}]"

    def plus_minus(self, fmt: str = "%.3f") -> str:
        """``mean +/- half-width`` — the compact form for a wide table."""
        if self.n < 2 or not np.isfinite(self.hi):
            return fmt % self.mean
        half = 0.5 * (self.hi - self.lo)
        return f"{fmt % self.mean} $\\pm$ {fmt % half}"


def mean_ci(values: Sequence[float] | np.ndarray, confidence: float = 0.95) -> CI:
    """Mean and a two-sided Student-$t$ interval over ``values``.

    The $t$ distribution rather than the normal because the number of seeds is
    small and known exactly; with two seeds the interval is wide and honest about
    it, which is the point of reporting one at all.  A single value has no spread
    to report, so the interval collapses to the mean and ``n = 1`` tells the
    caller not to read significance into it.  A set of identical values (every
    seed gave ``1.00``) has zero half-width, which is a true statement about a
    deterministic-looking outcome rather than a degenerate one.
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), float("nan"),
                  float("nan"), 0)
    mean = float(arr.mean())
    if n == 1:
        return CI(mean, float("nan"), float("nan"), float("nan"), 0.0, 1)
    std = float(arr.std(ddof=1))
    sem = std / np.sqrt(n)
    half = float(t.ppf(0.5 + confidence / 2.0, df=n - 1) * sem)
    return CI(mean, mean - half, mean + half, sem, std, n)


def aggregate_seeds(frame: pd.DataFrame, keys: Sequence[str],
                    metrics: Sequence[str], confidence: float = 0.95
                    ) -> pd.DataFrame:
    """Collapse per-seed rows to one row per cell, with a CI on each metric.

    ``frame`` is expected to carry one row per ``(*keys, seed)`` — the shape
    experiment 06 writes when it runs the comparison under several seeds.  For
    every metric named, the returned frame carries ``<metric>_mean``,
    ``<metric>_std``, ``<metric>_sem``, ``<metric>_lo`` and ``<metric>_hi``, plus
    a single ``n_seeds`` per cell so a reader can weigh every interval by how many
    draws stand behind it.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(keys) + ["n_seeds"])
    rows: list[dict[str, object]] = []
    for values, group in frame.groupby(list(keys), sort=False):
        key_values = values if isinstance(values, tuple) else (values,)
        row: dict[str, object] = dict(zip(keys, key_values))
        row["n_seeds"] = int(group.shape[0])
        for metric in metrics:
            if metric not in group.columns:
                continue
            ci = mean_ci(group[metric].to_numpy(), confidence)
            row[f"{metric}_mean"] = ci.mean
            row[f"{metric}_std"] = ci.std
            row[f"{metric}_sem"] = ci.sem
            row[f"{metric}_lo"] = ci.lo
            row[f"{metric}_hi"] = ci.hi
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """The paired binary comparison of two methods on the same records."""

    n_pairs: int
    both: int
    reference_only: int
    other_only: int
    neither: int
    statistic: float
    p_value: float

    @property
    def discordant(self) -> int:
        """Records where exactly one method succeeded — all McNemar reads."""
        return self.reference_only + self.other_only


def mcnemar(reference: Sequence[bool] | np.ndarray,
            other: Sequence[bool] | np.ndarray,
            exact: bool = True) -> McNemarResult:
    """Exact McNemar test that ``reference`` succeeds where ``other`` does not.

    The two arrays are aligned outcomes on the same units (record, or record and
    seed): ``reference[i]`` and ``other[i]`` are two methods judged on applicant
    ``i``.  McNemar conditions on the discordant pairs — the applicants exactly
    one method helped — because the applicants both helped or both failed carry no
    information about which method is better.  Under the null that a discordant
    pair is equally likely to fall either way, the count favouring ``reference``
    is ``Binomial(discordant, 1/2)``, and :func:`scipy.stats.binomtest` gives an
    exact two-sided $p$ with no continuity fudge or minimum-count caveat.

    The statistic returned is the standard $\\chi^2$ with continuity correction,
    reported for readers who expect it; the $p$-value is the exact binomial one
    and does not use it.
    """
    ref = np.asarray(reference, dtype=bool)
    oth = np.asarray(other, dtype=bool)
    if ref.shape != oth.shape:
        raise ValueError(f"paired arrays must match: {ref.shape} vs {oth.shape}")
    both = int(np.sum(ref & oth))
    ref_only = int(np.sum(ref & ~oth))
    oth_only = int(np.sum(~ref & oth))
    neither = int(np.sum(~ref & ~oth))
    b, c = ref_only, oth_only
    if b + c == 0:
        # No discordant pairs: the methods are indistinguishable on these units.
        return McNemarResult(ref.size, both, b, c, neither, 0.0, 1.0)
    if exact:
        p = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    else:
        p = float("nan")
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)  # chi-square, continuity-corrected
    return McNemarResult(ref.size, both, b, c, neither, float(stat), p)


@dataclass(frozen=True, slots=True)
class PairedResult:
    """A paired test over per-seed (or per-record) rates."""

    n: int
    statistic: float
    p_value: float
    median_difference: float


def paired_wilcoxon(reference: Sequence[float] | np.ndarray,
                    other: Sequence[float] | np.ndarray) -> PairedResult:
    """Wilcoxon signed-rank on paired rates (e.g. one usable rate per seed).

    Used for the seed-level view: ``reference[s]`` and ``other[s]`` are the two
    methods' rates under seed ``s``, and the test asks whether their difference is
    centred away from zero.  With only a handful of seeds this test has little
    power and can refuse to run at all when every difference has the same sign and
    magnitude — which is exactly what a large, stable gap looks like — so it is
    reported as a companion to the record-level McNemar rather than as the primary
    verdict.  A degenerate input (all differences zero, or too few pairs) returns
    ``p = 1`` and ``NaN`` statistic rather than raising, so a caller sweeping many
    cells does not have to special-case the easy ones.
    """
    ref = np.asarray(reference, dtype=float)
    oth = np.asarray(other, dtype=float)
    if ref.shape != oth.shape:
        raise ValueError(f"paired arrays must match: {ref.shape} vs {oth.shape}")
    diff = ref - oth
    median = float(np.median(diff)) if diff.size else float("nan")
    nonzero = diff[np.abs(diff) > 1e-12]
    if nonzero.size < 1:
        return PairedResult(int(diff.size), float("nan"), 1.0, median)
    try:
        stat, p = wilcoxon(ref, oth, zero_method="wilcox", correction=False,
                           mode="auto")
    except ValueError:
        # scipy refuses when all nonzero differences are identical in rank;
        # the sign test is the honest fallback and needs no ranking.
        pos = int(np.sum(nonzero > 0))
        p = float(binomtest(pos, nonzero.size, 0.5, alternative="two-sided").pvalue)
        return PairedResult(int(diff.size), float("nan"), p, median)
    return PairedResult(int(diff.size), float(stat), float(p), median)


def bootstrap_diff(reference: Sequence[float] | np.ndarray,
                   other: Sequence[float] | np.ndarray,
                   n_boot: int = 10_000, confidence: float = 0.95,
                   seed: int = 42) -> CI:
    """Bootstrap CI for the mean paired difference ``reference - other``.

    Resamples the paired units (records) with replacement, so it answers "how big
    is the gap, and how sure are we of its size" rather than only "is it nonzero".
    Paired resampling keeps each applicant's two outcomes together, which is what
    makes the interval a statement about the difference rather than about two
    separately-noisy rates.  The mean of the returned :class:`CI` is the observed
    difference; ``lo``/``hi`` are the percentile bounds over the resamples.
    """
    ref = np.asarray(reference, dtype=float)
    oth = np.asarray(other, dtype=float)
    if ref.shape != oth.shape:
        raise ValueError(f"paired arrays must match: {ref.shape} vs {oth.shape}")
    diff = ref - oth
    n = int(diff.size)
    observed = float(diff.mean()) if n else float("nan")
    if n < 2:
        return CI(observed, float("nan"), float("nan"), float("nan"), 0.0, n)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return CI(observed, lo, hi, float(diff.std(ddof=1) / np.sqrt(n)),
              float(diff.std(ddof=1)), n)


def cluster_bootstrap_diff(reference: Sequence[float] | np.ndarray,
                           other: Sequence[float] | np.ndarray,
                           clusters: Sequence[object] | np.ndarray,
                           n_boot: int = 10_000, confidence: float = 0.95,
                           seed: int = 42) -> CI:
    """Bootstrap CI for the paired difference, resampling whole clusters.

    Each observation ``i`` is one outcome of ``reference`` and ``other`` on a unit
    that belongs to ``clusters[i]`` --- an applicant observed under several search
    seeds, say.  Rows sharing a cluster are correlated and carry no independent
    information, so the honest resampling unit is the cluster, not the row: each
    cluster is first collapsed to its mean ``reference`` and mean ``other``
    outcome, and the bootstrap then resamples those per-cluster means.  Copying a
    cluster's rows --- the same applicant seen under more seeds --- therefore
    leaves both the observed difference and the interval unchanged, which is the
    property a pooled ``(seed, record)`` bootstrap silently violated: it treated
    every repetition as a fresh draw and narrowed the interval accordingly.

    With exactly one observation per cluster this reduces to
    :func:`bootstrap_diff`; the returned :class:`CI` has ``n`` equal to the number
    of distinct clusters, so a reader sees how many independent units the interval
    actually rests on.
    """
    ref = np.asarray(reference, dtype=float)
    oth = np.asarray(other, dtype=float)
    cl = np.asarray(clusters).reshape(-1)
    if not (ref.shape == oth.shape == cl.shape):
        raise ValueError(f"reference, other and clusters must match: "
                         f"{ref.shape} vs {oth.shape} vs {cl.shape}")
    if ref.size == 0:
        return CI(float("nan"), float("nan"), float("nan"), float("nan"), 0.0, 0)
    # Collapse each cluster to its within-cluster mean for both methods.  The
    # paired difference of those means is the per-cluster effect the bootstrap
    # resamples; duplicated rows change the counts but not the means.
    _labels, inverse = np.unique(cl, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse)
    ref_by_cluster = np.bincount(inverse, weights=ref) / counts
    oth_by_cluster = np.bincount(inverse, weights=oth) / counts
    return bootstrap_diff(ref_by_cluster, oth_by_cluster, n_boot=n_boot,
                          confidence=confidence, seed=seed)


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05
                    ) -> list[tuple[float, float, bool]]:
    """Holm's step-down correction for a family of ``p_values``.

    Every table here runs one test per baseline against \\cac{}, so the several
    $p$-values on a row are a family and reading each at $0.05$ would inflate the
    chance that one of them is spurious.  Holm controls that without assuming the
    tests are independent (they are not — they share \\cac{}'s column), and it is
    uniformly more powerful than a flat Bonferroni.  Returns, in the input order,
    each raw $p$, its adjusted $p$, and whether it is rejected at ``alpha``.
    """
    arr = np.asarray(list(p_values), dtype=float)
    m = int(arr.size)
    if m == 0:
        return []
    order = np.argsort(arr, kind="stable")
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        value = (m - rank) * arr[i]
        running = max(running, value)  # enforce monotone non-decreasing
        adjusted[i] = min(running, 1.0)
    return [(float(arr[i]), float(adjusted[i]), bool(adjusted[i] <= alpha))
            for i in range(m)]
