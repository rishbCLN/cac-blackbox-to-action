"""Validate the hand-crafted SCMs against the data (§6.4).

Four checks, in the order the instruction lists them:

1. **acyclicity** — a cycle makes every intervention ill-defined;
2. **reachability** — the target must be reachable from at least one
   *manipulable* feature, or recourse is impossible by construction;
3. **conditional independence** — the constraints the graph entails are tested
   on the training rows; a contradiction is a **warning, never a failure**,
   because real data is noisy and the graph is a domain-knowledge artefact;
4. **figure** — ``paper/figures/scm_<dataset>.png`` plus the Mermaid source.

Two CI tests are reported, and they answer the same question differently.

``dowhy`` — the tool the instruction names — gives the headline verdict through
:meth:`dowhy.CausalModel.refute_graph`.  Two version realities shape how it is
called.  dowhy 0.8's own constraint enumeration calls
``networkx.algorithms.d_separated``, removed in networkx >= 3.5, and its
discrete branch (``dowhy.utils.cit.conditional_MI``) indexes ``data[list(x)]``
with a *string* column name, so it raises ``KeyError`` on every discrete
triple.  The constraints are therefore enumerated here with
:func:`networkx.is_d_separator` and handed to dowhy explicitly, on the all-float
view of the data, which routes every triple through its working
partial-correlation test.

That test is linear-Gaussian, so a second, type-aware test runs here and is the
one reported per edge: a nested-model comparison on the *encoded* data — an OLS
``F`` test when the child is numeric/ordinal, a likelihood-ratio test on a
near-unpenalised multinomial logit when the child is categorical/binary.  Each
carries an effect size next to its p-value, and p-values are corrected for
multiplicity with Benjamini-Hochberg: at n = 5,000 a raw 0.05 threshold calls
almost every pair of credit-file variables dependent, which would make the
check vacuous.

Only the training split is used, so the validation never touches test data.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from src.config import CONFIG, DATASETS
from src.data.preprocess import build_dataset
from src.data.schema import DatasetSpec
from src.scm.define_scm import SCM, get_scm
from src.utils.io import save_dataframe, save_json, save_latex_table
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

# --- CI test settings (reported in the JSON so the paper can quote them) -----
#: Significance level of a single test, before multiplicity correction.
CI_ALPHA = 0.05
#: Deterministic sub-sample size; keeps ~250 nested fits per dataset quick.
CI_MAX_ROWS = 5_000
#: Smallest partial R^2 that counts as a real dependence rather than a
#: statistically detectable but practically empty one.
CI_MIN_EFFECT = 0.01
#: Cap on the number of implied d-separations tested per dataset.
CI_MAX_TESTS = 240
#: Polynomial degree used for continuous blocks (1 = linear, 2 = + curvature).
CI_POLY_DEGREE = 2

_CONTINUOUS_KINDS = ("numeric", "ordinal")


@dataclass(frozen=True, slots=True)
class Issue:
    """One validation finding.  ``level`` is ``error``, ``warning`` or ``info``."""

    level: str
    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - logging aid
        return f"[{self.level}] {self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class CITest:
    """Result of one conditional-independence test.

    ``family`` is ``edge`` (the graph asserts ``x -> y``, so the data should show
    a dependence) or ``dsep`` (the graph implies an independence, so the data
    should not).  ``verdict`` is ``supported``, ``contradicted``, ``weak``
    (significant but below :data:`CI_MIN_EFFECT`) or ``skipped``.
    """

    dataset: str
    family: str
    child: str
    candidate: str
    given: tuple[str, ...]
    expected: str
    test: str
    statistic: float
    dof: float
    p_value: float
    q_value: float
    effect: float
    n: int
    verdict: str
    note: str = ""

    @property
    def contradicted(self) -> bool:
        return self.verdict == "contradicted"

    def row(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "family": self.family,
            "relation": (f"{self.candidate} -> {self.child}" if self.family == "edge"
                         else f"{self.candidate} _||_ {self.child}"),
            "child": self.child,
            "candidate": self.candidate,
            "given": ", ".join(self.given) or "-",
            "expected": self.expected,
            "test": self.test,
            "statistic": self.statistic,
            "dof": self.dof,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "effect": self.effect,
            "n": self.n,
            "verdict": self.verdict,
            "note": self.note,
        }


@dataclass(slots=True)
class ValidationReport:
    """Everything the validator learned about one dataset's SCM."""

    dataset: str
    summary: dict[str, Any]
    issues: list[Issue]
    tests: list[CITest]
    dowhy: dict[str, Any]
    figures: list[Path]

    @property
    def ok(self) -> bool:
        """True when nothing structural is broken (warnings do not count)."""
        return not any(i.level == "error" for i in self.issues)

    def of_level(self, level: str) -> list[Issue]:
        return [i for i in self.issues if i.level == level]

    def family(self, family: str) -> list[CITest]:
        return [t for t in self.tests if t.family == family]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.row() for t in self.tests])

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "ok": self.ok,
            "summary": self.summary,
            "issues": [{"level": i.level, "code": i.code, "message": i.message}
                       for i in self.issues],
            "dowhy": self.dowhy,
            "ci_tests": [t.row() for t in self.tests],
            "figures": [str(p) for p in self.figures],
        }


# --------------------------------------------------------------- design blocks
def _kind(spec: DatasetSpec, target: str, name: str) -> str:
    """Feature kind, with the (binary) target treated as a feature."""
    return "binary" if name == target else spec.get(name).kind


def _levels(spec: DatasetSpec, target: str, name: str) -> int:
    return 2 if name == target else max(spec.get(name).n_levels, 2)


def _block(values: np.ndarray, kind: str, n_levels: int) -> np.ndarray:
    """Design block for one variable.

    Continuous (numeric / ordinal) variables enter as a standardised polynomial,
    which matches how the FCM models them and captures monotone plus
    single-curvature dependence.  Categorical and binary variables enter as
    dummy indicators with the first observed level dropped.
    """
    col = np.asarray(values, dtype=float).ravel()
    if kind in _CONTINUOUS_KINDS:
        sd = float(col.std())
        z = (col - col.mean()) / (sd if sd > 0 else 1.0)
        return np.column_stack([z ** k for k in range(1, CI_POLY_DEGREE + 1)])
    codes = np.clip(np.rint(col), 0, n_levels - 1).astype(int)
    observed = np.unique(codes)
    if observed.size <= 2:
        return codes.reshape(-1, 1).astype(float)
    return np.column_stack([(codes == lv).astype(float) for lv in observed[1:]])


def _design(
    frame: pd.DataFrame,
    names: Sequence[str],
    spec: DatasetSpec,
    target: str,
) -> np.ndarray:
    """Intercept plus one block per conditioning variable."""
    blocks = [np.ones((len(frame), 1))]
    for name in names:
        blocks.append(_block(frame[name].to_numpy(),
                             _kind(spec, target, name),
                             _levels(spec, target, name)))
    return np.hstack(blocks)


# ----------------------------------------------------------------- nested tests
def _rss(design: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Residual sum of squares and the *effective* rank of ``design``."""
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return float(resid @ resid), int(rank)


def _f_test(y: np.ndarray, d0: np.ndarray, d1: np.ndarray) -> dict[str, Any]:
    """Nested OLS ``F`` test of the block that ``d1`` adds to ``d0``."""
    rss0, rank0 = _rss(d0, y)
    rss1, rank1 = _rss(d1, y)
    q, resid_df = rank1 - rank0, len(y) - rank1
    if q <= 0 or resid_df <= 0:
        return {"test": "F", "note": "no identifiable added variation"}
    if rss1 <= 0 or rss0 <= 0:
        return {"test": "F", "note": "child is constant or exactly determined"}
    statistic = ((rss0 - rss1) / q) / (rss1 / resid_df)
    return {
        "test": "F",
        "statistic": float(statistic),
        "dof": float(q),
        "p_value": float(stats.f.sf(statistic, q, resid_df)),
        "effect": float(max(rss0 - rss1, 0.0) / rss0),
    }


def _loglik(design: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Log-likelihood of a near-unpenalised multinomial logit, and its rank."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lbfgs iteration cap on flat blocks
        clf = LogisticRegression(C=1e6, fit_intercept=False, max_iter=500,
                                 solver="lbfgs").fit(design, y)
    proba = np.clip(clf.predict_proba(design), 1e-12, 1.0)
    column = {c: j for j, c in enumerate(clf.classes_)}
    picked = proba[np.arange(len(y)), [column[v] for v in y]]
    return float(np.log(picked).sum()), int(np.linalg.matrix_rank(design))


def _lr_test(y: np.ndarray, d0: np.ndarray, d1: np.ndarray) -> dict[str, Any]:
    """Likelihood-ratio test for a discrete child; effect is a Cox-Snell R^2."""
    classes = np.unique(y)
    if classes.size < 2:
        return {"test": "LR", "note": "child is constant in the sub-sample"}
    ll0, rank0 = _loglik(d0, y)
    ll1, rank1 = _loglik(d1, y)
    q = (rank1 - rank0) * (classes.size - 1)
    if q <= 0:
        return {"test": "LR", "note": "no identifiable added variation"}
    statistic = max(2.0 * (ll1 - ll0), 0.0)
    return {
        "test": "LR",
        "statistic": float(statistic),
        "dof": float(q),
        "p_value": float(stats.chi2.sf(statistic, q)),
        "effect": float(1.0 - np.exp(-statistic / len(y))),
    }


def _one_test(
    dataset: str,
    family: str,
    child: str,
    candidate: str,
    given: Sequence[str],
    frame: pd.DataFrame,
    spec: DatasetSpec,
    target: str,
) -> CITest:
    """Test ``candidate`` against ``child`` given ``given``.

    ``child`` is always the node whose parent set defines the conditioning set,
    so the regression that is fitted is the structural equation the graph
    asserts — with and without the variable under test.
    """
    expected = "dependent" if family == "edge" else "independent"
    blank = CITest(dataset=dataset, family=family, child=child, candidate=candidate,
                   given=tuple(given), expected=expected, test="-",
                   statistic=float("nan"), dof=float("nan"), p_value=float("nan"),
                   q_value=float("nan"), effect=float("nan"), n=len(frame),
                   verdict="skipped")

    d0 = _design(frame, given, spec, target)
    block = _block(frame[candidate].to_numpy(),
                   _kind(spec, target, candidate),
                   _levels(spec, target, candidate))
    if float(block.std()) == 0.0:
        return replace(blank, note="candidate is constant in the sub-sample")
    d1 = np.hstack([d0, block])

    child_kind = _kind(spec, target, child)
    if child_kind in _CONTINUOUS_KINDS:
        out = _f_test(frame[child].to_numpy(dtype=float), d0, d1)
    else:
        out = _lr_test(np.rint(frame[child].to_numpy(dtype=float)).astype(int), d0, d1)
    if "p_value" not in out:
        return replace(blank, test=out["test"], note=out.get("note", ""))
    return replace(blank, test=out["test"], statistic=out["statistic"],
                   dof=out["dof"], p_value=out["p_value"], effect=out["effect"])


def _bh_qvalues(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values (NaN passes through)."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return q.tolist()
    order = finite[np.argsort(p[finite], kind="stable")]
    scaled = p[order] * order.size / np.arange(1, order.size + 1)
    q[order] = np.clip(np.minimum.accumulate(scaled[::-1])[::-1], 0.0, 1.0)
    return q.tolist()


def _verdict(test: CITest) -> str:
    """Read a q-value and an effect size into one of four words."""
    if not np.isfinite(test.q_value):
        return "skipped"
    significant = test.q_value <= CI_ALPHA
    material = test.effect >= CI_MIN_EFFECT
    if not significant:
        return "contradicted" if test.expected == "dependent" else "supported"
    if not material:
        return "weak"
    return "supported" if test.expected == "dependent" else "contradicted"


# ------------------------------------------------------------ constraint sets
def _is_d_separator(graph: nx.DiGraph, x: str, y: str, z: Iterable[str]) -> bool:
    """``networkx.is_d_separator`` with a fallback for networkx < 3.4."""
    args = (graph, {x}, {y}, set(z))
    if hasattr(nx, "is_d_separator"):
        return bool(nx.is_d_separator(*args))
    return bool(nx.algorithms.d_separated(*args))  # pragma: no cover - old networkx


def edge_constraints(scm: SCM) -> list[tuple[str, str, tuple[str, ...]]]:
    """``(parent, child, other parents)`` for every asserted edge.

    An edge earns its place if the parent still moves the child once the other
    parents are held fixed; that is exactly the block this triple tests.
    """
    return [(u, v, tuple(p for p in scm.parents(v) if p != u))
            for u, v in scm.graph.edges()]


def dsep_constraints(
    scm: SCM,
    limit: int = CI_MAX_TESTS,
) -> tuple[list[tuple[str, str, tuple[str, ...]]], int]:
    """Local-Markov implications: ``v`` is independent of its non-descendants
    given ``Pa(v)``.  Returns the (possibly capped) list and the total found.
    """
    found: list[tuple[str, str, tuple[str, ...]]] = []
    for node in scm.nodes:
        parents = tuple(scm.parents(node))
        excluded = set(parents) | scm.descendants(node) | {node}
        for other in scm.nodes:
            if other in excluded:
                continue
            if _is_d_separator(scm.graph, node, other, parents):
                found.append((other, node, parents))
    total = len(found)
    if total > limit:
        rng = np.random.default_rng(CONFIG.RANDOM_STATE)
        keep = np.sort(rng.choice(total, size=limit, replace=False))
        found = [found[i] for i in keep]
        LOGGER.info("%s: %d implied independencies, testing a deterministic "
                    "sub-sample of %d", scm.dataset, total, limit)
    return found, total


def training_frame(dataset: str, max_rows: int = CI_MAX_ROWS) -> pd.DataFrame:
    """Encoded training rows plus the target column, sub-sampled deterministically."""
    bundle = build_dataset(dataset)
    frame = pd.DataFrame(bundle.X[bundle.train_idx], columns=bundle.feature_names)
    frame[bundle.spec.target] = bundle.y[bundle.train_idx].astype(float)
    if len(frame) > max_rows:
        rng = np.random.default_rng(CONFIG.RANDOM_STATE)
        rows = np.sort(rng.choice(len(frame), size=max_rows, replace=False))
        frame = frame.iloc[rows].reset_index(drop=True)
    return frame


def run_ci_tests(
    scm: SCM,
    frame: pd.DataFrame,
    max_tests: int = CI_MAX_TESTS,
) -> tuple[list[CITest], int]:
    """Test every asserted edge and (up to ``max_tests``) implied independencies.

    Benjamini-Hochberg is applied **within** each family: the two families ask
    opposite questions, so pooling them into one false-discovery pool would
    let a noisy edge test relax the threshold for a missing-edge test.
    """
    spec = scm.spec if scm.spec is not None else None
    if spec is None:  # pragma: no cover - build_scm always attaches the spec
        raise ValueError(f"{scm.dataset}: SCM has no DatasetSpec attached")

    dsep, n_dsep_total = dsep_constraints(scm, limit=max_tests)
    families = (("edge", edge_constraints(scm)), ("dsep", dsep))
    tests: list[CITest] = []
    for family, constraints in families:
        # both enumerators yield (candidate, child, conditioning set)
        raw = [_one_test(scm.dataset, family, child, candidate, given,
                         frame, spec, scm.target)
               for candidate, child, given in constraints]
        qs = _bh_qvalues([t.p_value for t in raw])
        tests.extend(replace(t, q_value=q, verdict=_verdict(replace(t, q_value=q)))
                     for t, q in zip(raw, qs))
    return tests, n_dsep_total


# ------------------------------------------------------------------- dowhy leg
def to_gml(scm: SCM) -> str:
    """GML source of the graph — the format dowhy's ``CausalModel`` parses."""
    nodes = "".join(f'  node [ id "{n}" label "{n}" ]\n' for n in scm.nodes)
    edges = "".join(f'  edge [ source "{u}" target "{v}" ]\n'
                    for u, v in scm.graph.edges())
    return f"graph [ directed 1\n{nodes}{edges}]"


def refute_with_dowhy(
    scm: SCM,
    frame: pd.DataFrame,
    constraints: Sequence[tuple[str, str, tuple[str, ...]]],
) -> dict[str, Any]:
    """Run dowhy's graph refutation on the implied independencies.

    Only triples with a non-empty conditioning set are passed: dowhy's
    partial-correlation routine expects at least one covariate, matching its own
    ``k >= 1`` enumeration.  Failures are reported, never raised — this leg is a
    cross-check, and the instruction is explicit that a noisy contradiction must
    not stop the pipeline.
    """
    passed = [(a, b, tuple(z)) for a, b, z in constraints if z]
    info: dict[str, Any] = {"requested": len(passed), "skipped_unconditional":
                            len(constraints) - len(passed)}
    if not passed:
        return {**info, "available": False, "reason": "no conditional implication to test"}
    try:
        from dowhy import CausalModel  # noqa: PLC0415 - optional heavy import
    except Exception as exc:  # pragma: no cover - dowhy is a declared dependency
        return {**info, "available": False, "reason": f"{type(exc).__name__}: {exc}"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CausalModel(data=frame.astype("float64"),
                                treatment=scm.actionable[0],
                                outcome=scm.target,
                                graph=to_gml(scm))
            result = model.refute_graph(independence_constraints=list(passed))
    except Exception as exc:
        LOGGER.warning("%s: dowhy refutation unavailable (%s: %s)",
                       scm.dataset, type(exc).__name__, exc)
        return {**info, "available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        **info,
        "available": True,
        "n_constraints": int(result.number_of_constraints_model),
        "n_satisfied": int(result.number_of_constraints_satisfied),
        "refutation_passed": bool(result.refutation_result),
        "test_continuous": str(result.method_name_continuous),
        "test_discrete": str(result.method_name_discrete),
    }


# ------------------------------------------------------------ structural checks
def check_acyclic(scm: SCM) -> list[Issue]:
    """Check 1: no cycles."""
    if nx.is_directed_acyclic_graph(scm.graph):
        return [Issue("info", "acyclic", "the graph is a DAG")]
    cycle = nx.find_cycle(scm.graph, orientation="original")  # pragma: no cover
    return [Issue("error", "cycle", f"the graph contains the cycle {cycle}")]


def check_reachability(scm: SCM) -> list[Issue]:
    """Check 2: the target is reachable from at least one manipulable feature."""
    issues: list[Issue] = []
    reaching = [f for f in scm.manipulable if nx.has_path(scm.graph, f, scm.target)]
    if not reaching:
        return [Issue("error", "unreachable",
                      f"no manipulable feature has a directed path to {scm.target!r}; "
                      "recourse is impossible under this graph")]
    issues.append(Issue("info", "reachable",
                        f"{len(reaching)} of {len(scm.manipulable)} manipulable "
                        f"feature(s) reach {scm.target!r}: {', '.join(reaching)}"))
    for feature in scm.actionable:
        if feature not in reaching:
            issues.append(Issue("warning", "inert_action",
                                f"{feature!r} is actionable but cannot influence "
                                f"{scm.target!r}; acting on it can never change the "
                                "decision"))
    if not any(f in reaching for f in scm.actionable):
        issues.append(Issue("error", "no_effective_action",
                            "every feature that reaches the target is priced as "
                            "infinite, so no affordable recourse exists"))
    return issues


def check_semantics(scm: SCM) -> list[Issue]:
    """Cheap structural sanity checks that would otherwise bite much later."""
    issues: list[Issue] = []
    if scm.graph.out_degree(scm.target) > 0:  # pragma: no cover - build_scm raises
        issues.append(Issue("error", "target_not_sink",
                            f"{scm.target!r} has outgoing edge(s)"))
    if scm.graph.in_degree(scm.target) == 0:  # pragma: no cover - implied by check 2
        issues.append(Issue("error", "target_isolated",
                            f"{scm.target!r} has no parents"))
    for node in scm.non_manipulable:
        if not scm.is_infinite(node):  # pragma: no cover - build_scm forces this
            issues.append(Issue("warning", "cost_mismatch",
                                f"{node!r} is non-manipulable but priced "
                                f"{scm.label(node)!r}"))
    discriminatory = [s for s in scm.sensitive
                      if s != scm.target and nx.has_path(scm.graph, s, scm.target)]
    if discriminatory:
        issues.append(Issue("info", "sensitive_paths",
                            f"sensitive attribute(s) {', '.join(discriminatory)} have a "
                            f"directed path to {scm.target!r}; kept deliberately so the "
                            "evaluation can measure what the black box does with them"))
    return issues


# --------------------------------------------------------------- CI -> issues
#: At most this many individual CI contradictions are reported as warnings; the
#: rest stay in ``results/scm_ci_tests.csv`` so the log remains readable.
MAX_REPORTED = 10


def _describe(test: CITest) -> str:
    given = ", ".join(test.given) or "nothing"
    return (f"{test.candidate} vs {test.child} given {given} "
            f"({test.test}={test.statistic:.1f}, q={test.q_value:.3g}, "
            f"effect={test.effect:.3f})")


def _ci_issues(tests: Sequence[CITest]) -> list[Issue]:
    """Turn contradicted tests into warnings — never errors (§6.4.3)."""
    issues: list[Issue] = []
    for family, code, wording in (
        ("edge", "edge_unsupported",
         "the data show no dependence for the asserted edge"),
        ("dsep", "missing_dependence",
         "the data show a dependence the graph forbids"),
    ):
        bad = sorted((t for t in tests if t.family == family and t.contradicted),
                     key=lambda t: (-t.effect if family == "dsep" else t.p_value))
        for test in bad[:MAX_REPORTED]:
            issues.append(Issue("warning", code, f"{wording}: {_describe(test)}"))
        if len(bad) > MAX_REPORTED:
            issues.append(Issue("info", f"{code}_more",
                                f"{len(bad) - MAX_REPORTED} further {family} "
                                "contradiction(s) in results/scm_ci_tests.csv"))
    return issues


def _counts(tests: Sequence[CITest], family: str) -> dict[str, int]:
    subset = [t for t in tests if t.family == family]
    return {
        "n_tests": len(subset),
        "n_supported": sum(t.verdict == "supported" for t in subset),
        "n_weak": sum(t.verdict == "weak" for t in subset),
        "n_contradicted": sum(t.verdict == "contradicted" for t in subset),
        "n_skipped": sum(t.verdict == "skipped" for t in subset),
    }


# -------------------------------------------------------------------- the check
def _figures(scm: SCM) -> list[Path]:
    """Check 4: the diagram, as PNG + PDF + Mermaid source."""
    from src.viz.plot_graphs import save_scm_figure  # noqa: PLC0415 - breaks a cycle

    return list(save_scm_figure(scm))


def validate_scm(
    dataset: str,
    run_ci: bool = True,
    figures: bool = True,
    max_tests: int = CI_MAX_TESTS,
) -> ValidationReport:
    """Run all four checks of §6.4 on one dataset's SCM."""
    scm = get_scm(dataset)
    issues = [*check_acyclic(scm), *check_reachability(scm), *check_semantics(scm)]

    tests: list[CITest] = []
    n_dsep_total, n_rows = 0, 0
    dowhy_info: dict[str, Any] = {"available": False, "reason": "CI tests not run"}
    if run_ci:
        frame = training_frame(dataset)
        n_rows = len(frame)
        tests, n_dsep_total = run_ci_tests(scm, frame, max_tests=max_tests)
        dowhy_info = refute_with_dowhy(
            scm, frame,
            [(t.candidate, t.child, t.given) for t in tests if t.family == "dsep"],
        )
        issues += _ci_issues(tests)

    paths = _figures(scm) if figures else []

    summary: dict[str, Any] = {
        **scm.summary(),
        "edges": _counts(tests, "edge"),
        "dseparations": {**_counts(tests, "dsep"), "n_implied": n_dsep_total},
        "ci_settings": {
            "alpha": CI_ALPHA,
            "min_effect": CI_MIN_EFFECT,
            "max_rows": CI_MAX_ROWS,
            "max_tests": max_tests,
            "poly_degree": CI_POLY_DEGREE,
            "correction": "benjamini-hochberg, within family",
            "n_rows_used": n_rows,
            "split": "train only",
        },
        "dowhy": dowhy_info,
    }
    report = ValidationReport(dataset=dataset, summary=summary, issues=issues,
                              tests=tests, dowhy=dowhy_info, figures=paths)
    _log_report(report)
    return report


def _log_report(report: ValidationReport) -> None:
    for issue in report.issues:
        getattr(LOGGER, {"error": "error", "warning": "warning"}.get(issue.level, "info"))(
            "%s: %s — %s", report.dataset, issue.code, issue.message)
    edges, dsep = report.summary["edges"], report.summary["dseparations"]
    LOGGER.info(
        "SCM %s: %s | edges supported %d/%d (weak %d, unsupported %d) | "
        "implied independencies upheld %d/%d tested of %d | dowhy: %s",
        report.dataset, "OK" if report.ok else "FAILED",
        edges["n_supported"], edges["n_tests"], edges["n_weak"],
        edges["n_contradicted"], dsep["n_supported"] + dsep["n_weak"],
        dsep["n_tests"], dsep["n_implied"],
        report.dowhy.get("refutation_passed", report.dowhy.get("reason", "n/a")),
    )


# ------------------------------------------------------------------- artefacts
def _summary_frame(reports: Sequence[ValidationReport]) -> pd.DataFrame:
    rows = []
    for r in reports:
        edges, dsep = r.summary["edges"], r.summary["dseparations"]
        rows.append({
            "Dataset": r.dataset,
            "Nodes": r.summary["n_nodes"],
            "Edges": r.summary["n_edges"],
            "Actionable": r.summary["n_actionable"],
            "DAG": "yes" if r.summary["acyclic"] else "no",
            "Target reachable": "yes" if not r.of_level("error") else "no",
            "Edges supported": f"{edges['n_supported']}/{edges['n_tests']}",
            "Edges unsupported": edges["n_contradicted"],
            "Independencies upheld":
                f"{dsep['n_supported'] + dsep['n_weak']}/{dsep['n_tests']}",
            "Warnings": len(r.of_level("warning")),
        })
    return pd.DataFrame(rows)


def _fmt_q(q: float) -> str:
    """q-values span 40 orders of magnitude; a fixed format would print zeros."""
    if not np.isfinite(q):
        return "--"
    return "$<$0.001" if q < 1e-3 else f"{q:.3f}"


def _edge_frame(report: ValidationReport) -> pd.DataFrame:
    rows = []
    for t in report.family("edge"):
        rows.append({
            "Edge": f"{t.candidate} $\\to$ {t.child}".replace("_", r"\_"),
            "Given": ", ".join(t.given).replace("_", r"\_") or "--",
            "Test": t.test,
            "Statistic": t.statistic,
            "df": int(t.dof) if np.isfinite(t.dof) else 0,
            "$q$": _fmt_q(t.q_value),
            "Effect": t.effect,
            "Verdict": t.verdict,
        })
    return pd.DataFrame(rows)



def validate_all(
    datasets: Sequence[str] | None = None,
    run_ci: bool = True,
    figures: bool = True,
    write: bool = True,
) -> dict[str, ValidationReport]:
    """Validate every SCM and write the JSON / CSV / LaTeX artefacts."""
    names = tuple(datasets or DATASETS)
    reports = {d: validate_scm(d, run_ci=run_ci, figures=figures) for d in names}
    if not write:
        return reports

    save_json({d: r.to_dict() for d, r in reports.items()},
              CONFIG.RESULTS_DIR / "scm_validation.json")
    frames = [r.frame() for r in reports.values() if not r.frame().empty]
    if frames:
        save_dataframe(pd.concat(frames, ignore_index=True),
                       CONFIG.RESULTS_DIR / "scm_ci_tests.csv")
    summary = _summary_frame(list(reports.values()))
    save_dataframe(summary, CONFIG.RESULTS_DIR / "scm_validation_summary.csv")
    save_latex_table(
        summary, "scm_validation",
        caption="Validation of the hand-crafted SCMs. Edge tests ask whether each "
                "asserted arrow is still detectable given the child's other parents; "
                "independence tests ask whether the arrows the graph omits are "
                "consistent with the data. Both use Benjamini-Hochberg-corrected "
                f"$q \\le {CI_ALPHA}$ and a minimum partial $R^2$ of {CI_MIN_EFFECT} "
                f"on {CI_MAX_ROWS} training rows.",
        label="tab:scm_validation", column_format="lrrrllrrrr", escape=False,
    )
    for dataset, report in reports.items():
        edges = _edge_frame(report)
        if edges.empty:
            continue
        save_latex_table(
            edges, f"scm_edge_tests_{dataset}",
            caption=f"Edge-level conditional-independence tests for the {dataset} SCM.",
            label=f"tab:scm_edge_tests_{dataset}",
            column_format="lllrrrrl", escape=False,
        )
    return reports


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the hand-crafted SCMs against the data (§6.4).")
    parser.add_argument("-d", "--datasets", nargs="*", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("--no-ci", action="store_true",
                        help="skip the conditional-independence tests")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip the SCM diagrams")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when any CI test is contradicted "
                             "(off by default: real data is noisy, §6.4.3)")
    return parser.parse_args()


def main() -> int:
    from src.utils.logging import setup_logging  # noqa: PLC0415 - CLI only

    setup_logging("validate_scm")
    args = _cli()
    reports = validate_all(args.datasets, run_ci=not args.no_ci,
                           figures=not args.no_figures)
    broken = [d for d, r in reports.items() if not r.ok]
    noisy = [d for d, r in reports.items()
             if any(t.contradicted for t in r.tests)]
    if broken:
        LOGGER.error("structurally invalid SCM(s): %s", ", ".join(broken))
        return 1
    if noisy:
        LOGGER.warning("SCM(s) with at least one contradicted CI test: %s "
                       "(warning only — see results/scm_ci_tests.csv)",
                       ", ".join(noisy))
    return 1 if (args.strict and noisy) else 0


__all__ = [
    "CITest",
    "Issue",
    "ValidationReport",
    "CI_ALPHA",
    "CI_MAX_ROWS",
    "CI_MIN_EFFECT",
    "CI_MAX_TESTS",
    "check_acyclic",
    "check_reachability",
    "check_semantics",
    "edge_constraints",
    "dsep_constraints",
    "training_frame",
    "run_ci_tests",
    "refute_with_dowhy",
    "to_gml",
    "validate_scm",
    "validate_all",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())







