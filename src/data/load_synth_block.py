"""Synthetic world #2: ``synth_block`` -- a known-SCM credit world with *honest* non-recourse.

Workstream C of the publishability plan (Finding 6b), and the deliberate foil to
:mod:`src.data.load_synth_reach`.  It keeps that world's features, bounds, level
set and cost model unchanged, so the two worlds differ only in their structural
equations and target rule -- never in how actions are priced.  That isolation is
the point: any non-recourse seen here is a fact about the *graph*, not an
artefact of a harsher cost model.

The decisive change is a direct edge ``age -> approved``.  Because ``age`` is
protected and non-manipulable, the approval rule now has a term no intervention
can move.  The age coefficient is large enough that, for sufficiently young
applicants, **even pushing income and savings to the top of their observed
support cannot lift the score over the threshold**: these applicants have
*honest non-recourse*.  Older rejected applicants keep a reachable, budget-
feasible flip, so the world yields a mix -- which is exactly what experiment 08
needs to tell a faithful auditor (the true SCM) apart from a circular one (an FCM
the method also optimised against), and to measure an honest non-recourse rate
instead of hallucinated recourse.

Graph (roots first; ``approved`` is the sink/target)::

    age* ------+--> income --+--> savings --.
    experience |             '--------------+--> approved
    training --'                            |
    age* ----------------------------------'   (the decisive, immutable edge)

``age`` (starred) is protected and non-manipulable.  Variables are in natural
units; the encoded space is identical except that ``training`` is the integer
index of its ordinal level.  Equations are mirrored exactly in
``data/scm/synth_block_scm.json``.
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
TAU: float = 0.0

# ---------------------------------------------------------- structural coefficients
# ``savings`` leans on ``income`` less than it does in synth_reach (0.4 vs 0.5):
# the two worlds genuinely differ in their generative coefficients, so an FCM
# fitted on one cannot be reused for the other and the true-vs-fitted gap in
# experiment 08 is real rather than cosmetic.
SAVINGS_FROM_INCOME: float = 0.4

#: Approval weights.  ``AGE_W`` is set so that the maximal income/savings headroom
#: (~4.0 at the top of the observed support) is overtaken by the age penalty for
#: applicants below ~30, which puts a meaningful minority of rejected applicants
#: into honest non-recourse while leaving older rejects reachable.  Calibrated
#: empirically in ``scratch/verify_synth_block.py``.
AGE_W: float = 0.28
AGE_PIVOT: float = 44.0
INCOME_W: float = 0.05
INCOME_PIVOT: float = 60.0
SAVINGS_W: float = 0.08
SAVINGS_PIVOT: float = 22.0


# -------------------------------------------------------- structural equations
# As in synth_reach, each function is the deterministic part of one structural
# assignment, written so experiment 08's TrueSCM can re-use it verbatim (noise=0
# for the mean response, or an abducted residual for a counterfactual).  All
# operate on *encoded* values, i.e. ``training`` enters as its integer level.

def income_mean(age: Any, experience: Any, training: Any, noise: Any = 0.0) -> np.ndarray:
    """``income`` (k$) from its parents -- identical form to synth_reach."""
    value = (30.0
             + 0.8 * np.asarray(experience, dtype=float)
             + 6.0 * np.asarray(training, dtype=float)
             + 0.15 * np.asarray(age, dtype=float)
             + np.asarray(noise, dtype=float))
    return np.clip(value, 0.0, 200.0)


def savings_mean(income: Any, noise: Any = 0.0) -> np.ndarray:
    """``savings`` (k$) from ``income`` -- a weaker pass-through than synth_reach."""
    return np.clip(SAVINGS_FROM_INCOME * np.asarray(income, dtype=float)
                   + np.asarray(noise, dtype=float), 0.0, 150.0)


def approval_score(age: Any, income: Any, savings: Any, noise: Any = 0.0) -> np.ndarray:
    """Latent score; ``approved = 1`` iff this is at least :data:`TAU`.

    Unlike synth_reach, this depends on the immutable ``age`` as well as the two
    actionable levers.  The age term is what makes recourse honestly impossible
    for the young: no admissible move on income/savings can offset a large enough
    ``AGE_W * (age - AGE_PIVOT)``.
    """
    return (AGE_W * (np.asarray(age, dtype=float) - AGE_PIVOT)
            + INCOME_W * (np.asarray(income, dtype=float) - INCOME_PIVOT)
            + SAVINGS_W * (np.asarray(savings, dtype=float) - SAVINGS_PIVOT)
            + np.asarray(noise, dtype=float))


def _sample(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Draw ``n`` observational rows in semantic space from the true SCM."""
    age = rng.integers(18, 71, size=n).astype(float)
    experience = rng.integers(0, 41, size=n).astype(float)
    training = rng.choice(len(TRAINING_LEVELS), size=n, p=[0.35, 0.35, 0.20, 0.10])
    income = income_mean(age, experience, training, rng.normal(0.0, 5.0, size=n))
    savings = savings_mean(income, rng.normal(0.0, 6.0, size=n))
    score = approval_score(age, income, savings, rng.normal(0.0, 0.3, size=n))
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
    name="synth_block",
    features=(
        FeatureSpec("age", "numeric", bounds=(18.0, 70.0), integer=True,
                    unit="years", protected=True, manipulable=False,
                    description="Applicant age; immutable, and a direct parent of approval."),
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
        "Target rule: approved = 1 iff 0.28*(age-44) + 0.05*(income-60) "
        "+ 0.08*(savings-22) + eps >= 0.",
        "The immutable age enters approval directly, so young rejected applicants "
        "have HONEST non-recourse: no admissible income/savings move can offset the "
        "age penalty. The sibling world synth_reach removes exactly this edge.",
    ),
)


def load_synth_block() -> tuple[pd.DataFrame, DatasetSpec]:
    """Sample the ``synth_block`` world (seeded) and return ``(semantic_df, SPEC)``."""
    rng = np.random.default_rng(int(CONFIG.RANDOM_STATE))
    frame = _sample(rng, N_ROWS)
    return frame, SPEC


__all__ = [
    "SPEC",
    "load_synth_block",
    "TRAINING_LEVELS",
    "TAU",
    "N_ROWS",
    "SAVINGS_FROM_INCOME",
    "income_mean",
    "savings_mean",
    "approval_score",
]
