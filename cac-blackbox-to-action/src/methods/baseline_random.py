"""The random-perturbation baseline (§8.3) — the sanity floor.

The recipe is deliberately dumb: draw ``RANDOM_N_SAMPLES`` perturbations of the
record, keep the ones the black box labels the target class, and return the
``k`` nearest.  It exists because a method that cannot beat it is not explaining
anything, and because it shows what "valid counterfactual" is worth on its own —
random flips are usually valid, occasionally sparse, and almost never something
a person could act on.  That gap is the paper's opening argument.

Nothing here consults the SCM, so ``causal_path`` stays empty (§8.1).  The one
concession is ``BASELINES_RESPECT_IMMUTABLE``: with it on, a perturbation may
not rewrite an applicant's age or sex.  A baseline that is allowed to do that
wins on cost by proposing the impossible, which would be a strawman rather than
a comparison.
"""

from __future__ import annotations

import numpy as np

from src.config import CONFIG
from src.methods.base import CounterfactualMethod, register
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@register
class RandomMethod(CounterfactualMethod):
    """Uniform random search over the feature box, filtered by the black box."""

    name = "random"

    #: Expected number of features a single perturbation touches.  Sampling a
    #: sparse subset rather than the whole vector gives the baseline a fighting
    #: chance on sparsity without giving it any knowledge it should not have.
    DEFAULT_CHANGE_PROB = 0.4

    def _sample(self, x: np.ndarray, n: int) -> np.ndarray:
        """``n`` uniform draws inside the box, each touching a random subset."""
        p_change = float(self.params.get("change_prob", self.DEFAULT_CHANGE_PROB))
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        draws = self.rng.uniform(lo, hi, size=(n, self.d))
        touch = self.rng.random((n, self.d)) < p_change
        if CONFIG.BASELINES_RESPECT_IMMUTABLE:
            touch[:, self.immutable] = False
        out = np.where(touch, draws, x)
        return self.snap(out)

    def _propose(self, x: np.ndarray, y_target: int, k: int) -> np.ndarray:
        n = int(self.params.get("n_samples", CONFIG.RANDOM_N_SAMPLES))
        candidates = self._sample(x, n)
        p_target, flipped = self.score(candidates, y_target)
        if flipped.any():
            pool = candidates[flipped]
            order = np.argsort(self.scaled_l1(pool, x), kind="stable")
        else:
            # Report the near-misses rather than nothing: §9 measures the
            # validity rate, and an empty list would hide the failure.
            LOGGER.info("random/%s: 0/%d draws flipped the prediction",
                        self.dataset, n)
            pool, order = candidates, np.argsort(-p_target, kind="stable")
        return pool[order[:k]]


__all__ = ["RandomMethod"]
