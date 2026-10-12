"""Synthetic world #1: ``synth_reach`` -- a known-SCM credit world where recourse always exists.

Workstream C of the publishability plan (Finding 6b).  This is *not* a downloaded
dataset: every row is sampled, with a fixed seed, from an explicit structural
causal model whose equations and target rule are written down here and mirrored
exactly in ``data/scm/synth_reach_scm.json``.  Because the true SCM is known in
closed form, experiment 08 can audit a method's declared actions through the
*true* equations rather than through an FCM the method also optimised against --
which is the whole point of the de-circularisation (see
``experiments/07_decircular_audit.py`` for the real-data counterpart).

The world is engineered so that **every rejected applicant has a reachable,
budget-feasible flip**: the target depends only on ``income`` and ``savings``,
both of which are directly actionable (and are also mediators of the other
levers), so no applicant is ever blocked by an immutable ancestor.  Its sibling
:mod:`src.data.load_synth_block` keeps the same shape but routes decisive
influence through the immutable ``age``, so some applicants are honestly
unreachable.

Graph (roots first; ``approved`` is the sink/target)::

    age* ------.
    experience +--> income --+--> savings --.
    training --'             '---------------+--> approved

``age`` (starred) is protected and non-manipulable; everything else is
actionable.  Variables are given in natural units; the encoded space is
identical to the semantic one except that ``training`` is the integer index of
its ordinal level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.data.schema import DatasetSpec, FeatureSpec

#: Ordered levels of the one discrete feature (encoded as the index 0..3).
TRAINING_LEVELS: tuple[str, ...] = ("none", "basic", "advanced", "expert")

#: Rows sampled.  Comfortably below ``CONFIG.MAX_ROWS`` so no sub-sampling runs.
N_ROWS: int = 6_000

#: Decision threshold of the known rule: ``approved = 1`` iff ``score >= TAU``.
#: Zero because the score below is centred near the generative means, which keeps
#: the two classes reasonably balanced without a data-dependent cut.
TAU: float = 0.0


# -------------------------------------------------------- structural equations
# Each function is the deterministic part of one structural assignment, written
# so experiment 08's TrueSCM can re-use them verbatim (noise=0 for the mean
# response, or an abducted residual for a counterfactual).  All operate on
# *encoded* values, i.e. ``training`` enters as its integer level.

def income_mean(age: Any, experience: Any, training: Any, noise: Any = 0.0) -> np.ndarray:
    """``income`` (k$) from its parents -- an additive-noise structural equation."""
    value = (30.0
             + 0.8 * np.asarray(experience, dtype=float)
             + 6.0 * np.asarray(training, dtype=float)
             + 0.15 * np.asarray(age, dtype=float)
             + np.asarray(noise, dtype=float))
    return np.clip(value, 0.0, 200.0)


def savings_mean(income: Any, noise: Any = 0.0) -> np.ndarray:
    """``savings`` (k$) from ``income``."""
    return np.clip(0.5 * np.asarray(income, dtype=float)
                   + np.asarray(noise, dtype=float), 0.0, 150.0)


def approval_score(income: Any, savings: Any, noise: Any = 0.0) -> np.ndarray:
    """Latent score; ``approved = 1`` iff this is at least :data:`TAU`.

    Depends on ``income`` and ``savings`` only -- never on the immutable ``age`` --
    which is what guarantees a reachable flip for every rejected applicant.
    """
    return (0.05 * (np.asarray(income, dtype=float) - 60.0)
            + 0.08 * (np.asarray(savings, dtype=float) - 30.0)
            + np.asarray(noise, dtype=float))


def _sample(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Draw ``n`` observational rows in semantic space from the true SCM."""
    age = rng.integers(18, 71, size=n).astype(float)
    experience = rng.integers(0, 41, size=n).astype(float)
    training = rng.choice(len(TRAINING_LEVELS), size=n, p=[0.35, 0.35, 0.20, 0.10])
    income = income_mean(age, experience, training, rng.normal(0.0, 5.0, size=n))
    savings = savings_mean(income, rng.normal(0.0, 6.0, size=n))
    score = approval_score(income, savings, rng.normal(0.0, 0.3, size=n))
    approved = (score >= TAU).astype(int)
    return pd.DataFrame({
        "age": age.astype(int),
        "experience": experience.astype(int),
        "training": np.asarray(TRAINING_LEVELS, dtype=object)[training],
        "income": np.round(income, 2),
        "savings": np.round(savings, 2),
        "approved": approved,
    })


SPEC = DatasetSpec(
    name="synth_reach",
    features=(
        FeatureSpec("age", "numeric", bounds=(18.0, 70.0), integer=True,
                    unit="years", protected=True, manipulable=False,
                    description="Applicant age; immutable ancestor of income."),
        FeatureSpec("experience", "numeric", bounds=(0.0, 40.0), integer=True,
                    unit="years", description="Years of work experience."),
        FeatureSpec("training", "ordinal", levels=TRAINING_LEVELS,
                    description="Highest completed training programme."),
        FeatureSpec("income", "numeric", bounds=(0.0, 200.0), unit="k$",
                    description="Annual income; mediator and a direct lever."),
        FeatureSpec("savings", "numeric", bounds=(0.0, 150.0), unit="k$",
                    description="Liquid savings; mediator and a direct lever."),
    ),
    target="approved",
    positive_meaning="loan approved",
    negative_meaning="loan denied",
    source="synthetic (known structural causal model)",
    citation="",
    notes=(
        "SYNTHETIC ground-truth world generated in-process from a known SCM; "
        "not downloaded and never routed through src.data.sources.",
        "Target rule: approved = 1 iff 0.05*(income-60) + 0.08*(savings-30) + eps >= 0.",
        "Engineered so every rejected applicant has a reachable, budget-feasible "
        "flip: the target depends only on the directly-actionable income/savings.",
    ),
)


def load_synth_reach() -> tuple[pd.DataFrame, DatasetSpec]:
    """Sample the ``synth_reach`` world (seeded) and return ``(semantic_df, SPEC)``."""
    rng = np.random.default_rng(int(CONFIG.RANDOM_STATE))
    frame = _sample(rng, N_ROWS)
    return frame, SPEC


__all__ = [
    "SPEC",
    "load_synth_reach",
    "TRAINING_LEVELS",
    "TAU",
    "N_ROWS",
    "income_mean",
    "savings_mean",
    "approval_score",
]
