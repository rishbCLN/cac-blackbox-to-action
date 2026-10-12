"""The *true* SCM as an auditor: closed-form structural equations behind the FCM API.

Workstream C (Finding 6b).  Experiment 07 breaks the circularity of auditing a
method against the very FCM it optimised against by *re-fitting* the auditor FCM
on a disjoint fold.  On the two synthetic worlds we can do something stronger:
the data-generating equations are known exactly (:mod:`src.data.load_synth_reach`
and :mod:`src.data.load_synth_block`), so the auditor need not be fitted at all.

:class:`TrueSCM` subclasses :class:`~src.scm.fcm.FCM` and swaps each non-root
node's *fitted* mean for its generative truth, leaving Pearl's three-step
machinery (abduction, action, prediction) and the whole audit surface
(:meth:`~src.scm.fcm.FCM.violation`, ``repair``, ``do``, ``counterfactual``,
``.scm``) untouched.  Passed as ``Evaluator(fcm=...)`` it judges a method's
declared actions through the real causal laws rather than through an estimate the
method also saw -- the de-circularisation experiment 08 relies on.

The only non-root nodes in either world are ``income`` and ``savings`` (both
numeric additive-noise equations); ``age``/``experience``/``training`` are roots
and the target ``approved`` is computed by the black box, never by the FCM, so no
closed form is needed for it.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from src.data import load_synth_block, load_synth_reach
from src.scm.fcm import FCM
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: A world's truth: node -> f(parent_columns) -> structural mean, where
#: ``parent_columns`` is ``{parent_name: ndarray}`` so the equation never depends
#: on the positional order of the design matrix.
TruthEquations = Mapping[str, Callable[[Mapping[str, np.ndarray]], np.ndarray]]


class _ClosedForm:
    """Stand-in for a fitted regressor: returns the KNOWN structural mean.

    Mimics the ``predict(design)`` surface :class:`~src.scm.fcm.NodeEquation`
    calls, mapping the design matrix back to named parent columns.  The design is
    pure pass-through for every synthetic parent (all numeric or ordinal, none
    one-hot-expanded), so column ``k`` is parent ``k``; the assertion guards that
    invariant in case a world is ever given a categorical parent with >2 levels.
    """

    def __init__(self, fn: Callable[[Mapping[str, np.ndarray]], np.ndarray],
                 parents: tuple[str, ...]) -> None:
        self._fn = fn
        self.parents = tuple(parents)

    def predict(self, design: Any) -> np.ndarray:
        design = np.atleast_2d(np.asarray(design, dtype=float))
        if design.shape[1] != len(self.parents):
            raise ValueError(
                f"TrueSCM design has {design.shape[1]} columns for parents "
                f"{self.parents}: a one-hot-expanded parent would break the "
                "positional mapping and the closed-form equation."
            )
        cols = {name: design[:, k] for k, name in enumerate(self.parents)}
        return np.asarray(self._fn(cols), dtype=float)


class TrueSCM(FCM):
    """An :class:`~src.scm.fcm.FCM` whose equations are the generative truth."""

    def __init__(self, dataset: str, equations: TruthEquations,
                 scm: Any = None, spec: Any = None) -> None:
        super().__init__(dataset, scm=scm, spec=spec)
        self._truth = dict(equations)
        self._install()

    def _install(self) -> None:
        installed: list[str] = []
        for node in self.order:
            eq = self._equation_for(node)
            if eq.parents:
                if node not in self._truth:
                    raise KeyError(
                        f"{self.dataset}: no closed-form equation for non-root "
                        f"node {node!r}"
                    )
                eq.model = _ClosedForm(self._truth[node], eq.parents)
                eq.score = 1.0
                eq.score_name = "exact"
                eq.n_train = 0
                installed.append(node)
            self.equations[node] = eq
        unexpected = set(self._truth) - set(installed)
        if unexpected:
            raise KeyError(
                f"{self.dataset}: closed-form equation(s) for {sorted(unexpected)} "
                "do not correspond to a non-root feature node"
            )
        LOGGER.info("TrueSCM %s: installed %d closed-form equation(s) for %s",
                    self.dataset, len(installed), installed)

    def fit(self, X: Any = None) -> "TrueSCM":  # noqa: ARG002 - the truth needs no data
        """No-op: the equations are known exactly, not estimated from data."""
        return self

    def save(self, path: Any = None) -> Any:  # noqa: ARG002
        raise NotImplementedError("TrueSCM is analytic; there is nothing to persist")

    def load(self, path: Any = None) -> "TrueSCM":  # noqa: ARG002
        raise NotImplementedError("TrueSCM is analytic; there is nothing to load")


# ------------------------------------------------------------------- world truths
def _reach_income(cols: Mapping[str, np.ndarray]) -> np.ndarray:
    return load_synth_reach.income_mean(cols["age"], cols["experience"], cols["training"])


def _reach_savings(cols: Mapping[str, np.ndarray]) -> np.ndarray:
    return load_synth_reach.savings_mean(cols["income"])


def _block_income(cols: Mapping[str, np.ndarray]) -> np.ndarray:
    return load_synth_block.income_mean(cols["age"], cols["experience"], cols["training"])


def _block_savings(cols: Mapping[str, np.ndarray]) -> np.ndarray:
    return load_synth_block.savings_mean(cols["income"])


_TRUTH: dict[str, TruthEquations] = {
    "synth_reach": {"income": _reach_income, "savings": _reach_savings},
    "synth_block": {"income": _block_income, "savings": _block_savings},
}

_CACHE: dict[str, TrueSCM] = {}


def true_scm(dataset: str) -> TrueSCM:
    """The analytic ground-truth SCM for a synthetic world, behind the FCM API."""
    if dataset not in _TRUTH:
        raise KeyError(f"no closed-form truth for {dataset!r}; "
                       f"expected one of {sorted(_TRUTH)}")
    if dataset not in _CACHE:
        _CACHE[dataset] = TrueSCM(dataset, _TRUTH[dataset])
    return _CACHE[dataset]


__all__ = ["TrueSCM", "true_scm"]
