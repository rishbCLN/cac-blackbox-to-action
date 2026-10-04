"""Wachter et al. (2017) — the original counterfactual explanation (§8.3).

Minimise

.. math::  \\lambda\\,(f(x') - y_{target})^2 + d(x, x')

over ``x'`` with L-BFGS-B, escalating :math:`\\lambda` until the prediction
actually flips, and take the closest solution.  ``d`` is the L1 distance on
range-normalised features, the same distance every baseline here is judged by.

Two honest adaptations, both forced by this repository's rules rather than by
convenience:

*Gradients.*  Wachter's method assumes a differentiable model; global rule #8
says the explainer may only call ``predict_proba``.  So the gradient is a central
finite difference — but evaluated as one batched query of ``2m`` rows instead of
``2m`` separate calls, which is the same arithmetic at a fraction of the wall
clock and keeps the query count comparable across methods.

*Mixed data.*  The optimiser works in a continuous relaxation, so it will happily
propose 12.7 years of education.  The flip is therefore tested on the *snapped*
record — the thing a person could actually be handed — and :math:`\\lambda` is
escalated when snapping loses the flip.  That is Wachter's own stopping rule
applied to a realisable point, and the flips it still loses are a genuine
property of the method on tabular data, reported in §9 rather than hidden.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.config import CONFIG
from src.methods.base import CounterfactualMethod, register
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@register
class WachterMethod(CounterfactualMethod):
    """Gradient-based closest counterfactual, with a numerical gradient."""

    name = "wachter"

    #: Finite-difference step, as a fraction of each feature's range.
    FD_STEP = 1e-3
    #: How many times to multiply lambda by ten before giving up on a restart.
    LAMBDA_STEPS = 4

    @property
    def _free(self) -> np.ndarray:
        """Coordinates the optimiser may move.

        With ``BASELINES_RESPECT_IMMUTABLE`` the immutable features are simply
        not in the search space, which is both cheaper and more honest than
        letting the optimiser change an age and repairing it afterwards.
        """
        return self.actionable if CONFIG.BASELINES_RESPECT_IMMUTABLE \
            else np.arange(self.d)

    def _embed(self, Z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Free-coordinate vectors back to full encoded rows."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        rows = np.repeat(np.asarray(x, dtype=float).reshape(1, -1), Z.shape[0], axis=0)
        rows[:, self._free] = Z
        return rows

    def _loss(self, Z: np.ndarray, x: np.ndarray, y_target: int,
              lam: float) -> np.ndarray:
        """Wachter's objective, vectorised over as many points as you like."""
        rows = self._embed(Z, x)
        p_pos = self.proba(rows)
        return lam * (p_pos - float(y_target)) ** 2 + self.scaled_l1(rows, x)

    def _minimise(self, x: np.ndarray, y_target: int, lam: float,
                  z0: np.ndarray) -> np.ndarray:
        free = self._free
        m = free.size
        step = np.maximum(self.span[free] * self.FD_STEP, 1e-9)
        idx = np.arange(m)

        def fun(z: np.ndarray) -> float:
            return float(self._loss(z, x, y_target, lam)[0])

        def jac(z: np.ndarray) -> np.ndarray:
            Z = np.repeat(np.asarray(z, dtype=float).reshape(1, -1), 2 * m, axis=0)
            Z[idx, idx] += step
            Z[m + idx, idx] -= step
            values = self._loss(Z, x, y_target, lam)
            return (values[:m] - values[m:]) / (2.0 * step)

        result = minimize(
            fun, np.asarray(z0, dtype=float), jac=jac, method="L-BFGS-B",
            bounds=list(zip(self.bounds[free, 0], self.bounds[free, 1])),
            options={"maxiter": int(CONFIG.WACHTER_MAX_ITERS)},
        )
        return np.asarray(result.x, dtype=float)

    def _starts(self, x: np.ndarray, n: int) -> np.ndarray:
        """Restart points: the factual record first, then uniform draws.

        §8.3 asks for ``k`` counterfactuals from random restarts.  Starting at
        ``x`` gives the closest solution the method can find; the random starts
        are what make the ``k`` distinct rather than ``k`` copies.
        """
        free = self._free
        factual = np.asarray(x, dtype=float)[free]
        if n <= 1:
            return factual.reshape(1, -1)
        random = self.rng.uniform(self.bounds[free, 0], self.bounds[free, 1],
                                  size=(n - 1, free.size))
        return np.vstack([factual, random])

    def _propose(self, x: np.ndarray, y_target: int, k: int) -> np.ndarray:
        lambdas = [float(CONFIG.WACHTER_LAMBDA) * 10.0 ** i
                   for i in range(self.LAMBDA_STEPS)]
        found: list[np.ndarray] = []
        near_misses: list[np.ndarray] = []
        for z0 in self._starts(x, int(CONFIG.WACHTER_RESTARTS)):
            for lam in lambdas:
                row = self.snap(self._embed(
                    self._minimise(x, y_target, lam, z0), x))[0]
                _, flipped = self.score(row, y_target)
                if flipped[0]:
                    found.append(row)
                    break
                near_misses.append(row)
        if not found:
            LOGGER.info("wachter/%s: %d restarts x %d lambdas, no flip survived "
                        "snapping", self.dataset, int(CONFIG.WACHTER_RESTARTS),
                        len(lambdas))
        pool = np.asarray(found if found else near_misses, dtype=float)
        if pool.size == 0:  # pragma: no cover - only if every restart errored
            return np.empty((0, self.d))
        order = np.argsort(self.scaled_l1(pool, x), kind="stable")
        return pool[order[:k]]


__all__ = ["WachterMethod"]
