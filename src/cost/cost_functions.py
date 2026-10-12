"""Hand-crafted intervention costs: what an action costs *the person*.

Global rule #10: the cost of an intervention is domain knowledge, not something
read off a distance metric.  Every feature of every dataset gets an explicit,
documented ``(x_old, x_new) -> effort`` function, and the whole cost model is
exported as a table (``paper/tables/cost_model_<dataset>.tex``) so a reviewer can
argue with the numbers instead of guessing them.

Three quantities, because §7 asks for three incompatible things at once
--------------------------------------------------------------------
The instruction says per-feature costs look like ``"age": ... 10.0``, that the
total is a *sum over changed features*, that costs are *normalised to* ``[0, 1]``
*per dataset*, and that a non-manipulable feature draws a hard penalty of
``1e6``.  A raw 10.0 is not in ``[0, 1]``, so these are three different numbers
and this module keeps all three:

``raw``
    The sum of the hand-crafted per-feature efforts, in the instruction's own
    magnitudes (``age`` -> 10.0, ``marital_status`` -> 5.0, ``hours_per_week``
    -> ``0.01/h`` below 40h).  Reported in the paper as "domain cost"; it is the
    number a human can reason about.
``normalised``
    ``raw`` re-weighted by the SCM's qualitative effort label and divided by a
    per-dataset *budget*: the cost of the most expensive plan the pipeline
    allows, i.e. the ``CONFIG.MAX_ACTIONS`` most expensive maximal single-feature
    actions.  Any admissible plan therefore lands in ``[0, 1]``, and ``1.0``
    means "as expensive as recourse is permitted to get".  This is what the
    search in :mod:`src.methods.cac` minimises, so ``CAC_LAMBDA_COST`` is
    comparable across datasets.
``penalty``
    ``INFINITE_COST`` (``1e6``) per *inadmissible* action — one that moves a
    non-manipulable feature, violates a declared monotonicity, or leaves the
    feature's bounds.  Deliberately outside ``[0, 1]``: an infeasibility is not
    an expensive plan, it is not a plan.

The two factors of ``normalised`` are separate on purpose.  The per-feature
function answers *how far did you move within this feature* (it is divided by
that feature's own maximum, so its magnitude cancels), and the SCM's
``interventional_costs`` label answers *how hard is this feature to move at all*
(:data:`~src.scm.define_scm.COST_LEVELS`).  The graph and the cost model are
separate artefacts that have to agree, and this is where they are joined.

Everything is vectorised: each cost function takes scalars or whole columns, so
evaluating a 60-candidate CAC population costs one call per feature, not one per
candidate.  Nothing here touches the data — the cost model is built from the
declared schema alone, so it cannot leak the test split (rule #8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import CONFIG, DATASETS
from src.data.preprocess import dataset_spec
from src.data.schema import DatasetSpec
from src.scm.define_scm import SCM, get_scm
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: §7.2's ``infinite_cost``: the hard penalty for an action that is not allowed.
INFINITE_COST = 1e6
#: Numeric slack below which two feature values count as "unchanged".
CHANGE_ATOL = 1e-8

_ARR = np.ndarray


def _as_array(x: Any) -> _ARR:
    return np.asarray(x, dtype=float)


def _levels(x: _ARR, n_levels: int) -> _ARR:
    """Encoded value -> integer level index, clipped to the ladder."""
    return np.clip(np.rint(x).astype(np.int64), 0, max(n_levels - 1, 0))


# --------------------------------------------------------------- cost functions
@dataclass(frozen=True, slots=True)
class CostFunction:
    """A per-feature effort map, vectorised over rows.

    Subclasses implement :meth:`__call__` (``(old, new) -> effort >= 0``) and
    :meth:`maximum` (the largest effort attainable inside ``[lo, hi]``), which is
    what turns a raw effort into a fraction of that feature's range.
    """

    def __call__(self, old: Any, new: Any) -> _ARR:  # pragma: no cover - abstract
        raise NotImplementedError

    def maximum(self, lo: float, hi: float) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def domain(self) -> tuple[float, float] | None:
        """The range this function was calibrated for, if it is bounded."""
        return None

    def describe(self) -> str:
        return type(self).__name__

    def _direction(self, old: _ARR, new: _ARR, up: float, down: float) -> _ARR:
        """Multiplier that prices "up" and "down" moves differently."""
        if up == down:
            return np.full(np.broadcast(old, new).shape, float(up))
        return np.where(new > old, float(up), float(down))


@dataclass(frozen=True, slots=True)
class ConstantCost(CostFunction):
    """``0`` when unchanged, ``value`` otherwise — §7.1's ``age`` form.

    The right shape for a feature with no meaningful notion of distance: any
    change is the same life event (marital status), or none is possible at all
    (age, gender), in which case ``value`` mirrors
    ``COST_LEVELS["infinite"] == 10.0`` and the admissibility penalty adds the
    remaining ``1e6``.
    """

    value: float

    def __call__(self, old: Any, new: Any) -> _ARR:
        old, new = _as_array(old), _as_array(new)
        return np.where(np.isclose(old, new, rtol=0.0, atol=CHANGE_ATOL),
                        0.0, float(self.value))

    def maximum(self, lo: float, hi: float) -> float:  # noqa: ARG002 - range-free
        return float(self.value)

    def describe(self) -> str:
        return f"constant({self.value:g})"


@dataclass(frozen=True, slots=True)
class LinearCost(CostFunction):
    """``per_unit * |new - old|``, optionally asymmetric in direction."""

    per_unit: float
    up: float = 1.0
    down: float = 1.0

    def __call__(self, old: Any, new: Any) -> _ARR:
        old, new = _as_array(old), _as_array(new)
        return (float(self.per_unit) * np.abs(new - old)
                * self._direction(old, new, self.up, self.down))

    def maximum(self, lo: float, hi: float) -> float:
        return float(self.per_unit) * abs(hi - lo) * max(self.up, self.down)

    def describe(self) -> str:
        tail = "" if self.up == self.down else f", up={self.up:g}, down={self.down:g}"
        return f"linear({self.per_unit:g}/unit{tail})"


@dataclass(frozen=True, slots=True)
class PiecewiseLinearCost(CostFunction):
    """Progressive (or regressive) marginal effort — §7.1's ``piecewise_linear``.

    ``slopes[i]`` is the marginal cost per unit between ``knots[i]`` and
    ``knots[i + 1]``, and the cost of a move is the integral of that marginal
    cost between the two values, exactly like a tax schedule.  Working with the
    antiderivative keeps it order-independent and vectorised: the cost of
    ``40 -> 60`` plus ``60 -> 99`` equals the cost of ``40 -> 99``.
    """

    knots: tuple[float, ...]
    slopes: tuple[float, ...]
    up: float = 1.0
    down: float = 1.0
    _acc: _ARR = field(init=False, repr=False, compare=False)

    def domain(self) -> tuple[float, float]:
        """Where the schedule is defined; outside it the integral is clipped."""
        return (float(self.knots[0]), float(self.knots[-1]))

    def __post_init__(self) -> None:
        if len(self.knots) < 2:
            raise ValueError("piecewise_linear needs at least two knots")
        if len(self.slopes) != len(self.knots) - 1:
            raise ValueError(f"{len(self.knots)} knots need {len(self.knots) - 1} "
                             f"slopes, got {len(self.slopes)}")
        if any(b <= a for a, b in zip(self.knots, self.knots[1:])):
            raise ValueError(f"knots must be strictly increasing: {self.knots}")
        if any(s < 0 for s in self.slopes):
            raise ValueError(f"slopes must be non-negative: {self.slopes}")
        widths = np.diff(np.asarray(self.knots, dtype=float))
        acc = np.concatenate([[0.0], np.cumsum(widths * np.asarray(self.slopes))])
        object.__setattr__(self, "_acc", acc)

    def _integral(self, x: _ARR) -> _ARR:
        """Antiderivative of the marginal-cost schedule, clipped to the knots."""
        return np.interp(x, np.asarray(self.knots, dtype=float), self._acc)

    def __call__(self, old: Any, new: Any) -> _ARR:
        old, new = _as_array(old), _as_array(new)
        gross = np.abs(self._integral(new) - self._integral(old))
        return gross * self._direction(old, new, self.up, self.down)

    def maximum(self, lo: float, hi: float) -> float:
        span = abs(float(self._integral(_as_array(hi)) - self._integral(_as_array(lo))))
        return span * max(self.up, self.down)

    def describe(self) -> str:
        pieces = ", ".join(
            f"{a:g}-{b:g}@{s:g}"
            for a, b, s in zip(self.knots, self.knots[1:], self.slopes)
        )
        tail = "" if self.up == self.down else f"; up={self.up:g}, down={self.down:g}"
        return f"piecewise({pieces}{tail})"


@dataclass(frozen=True, slots=True)
class LadderCost(CostFunction):
    """Ordinal ladder: pay for every rung between the two levels.

    ``rungs[i]`` is the effort of climbing from level ``i`` to level ``i + 1``,
    so an ordinal feature can price *unequal* steps — finishing a Bachelors is
    not the same amount of work as moving from 9th to 10th grade.
    """

    rungs: tuple[float, ...]
    up: float = 1.0
    down: float = 1.0
    _acc: _ARR = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ValueError("a ladder needs at least one rung")
        if any(r < 0 for r in self.rungs):
            raise ValueError(f"rung costs must be non-negative: {self.rungs}")
        object.__setattr__(
            self, "_acc",
            np.concatenate([[0.0], np.cumsum(np.asarray(self.rungs, dtype=float))]),
        )

    @property
    def n_levels(self) -> int:
        return len(self.rungs) + 1

    def __call__(self, old: Any, new: Any) -> _ARR:
        old, new = _as_array(old), _as_array(new)
        acc = self._acc
        gross = np.abs(acc[_levels(new, self.n_levels)] - acc[_levels(old, self.n_levels)])
        return gross * self._direction(old, new, self.up, self.down)

    def maximum(self, lo: float, hi: float) -> float:
        acc = self._acc
        span = abs(float(acc[int(_levels(_as_array(hi), self.n_levels))]
                         - acc[int(_levels(_as_array(lo), self.n_levels))]))
        return span * max(self.up, self.down)

    def describe(self) -> str:
        tail = "" if self.up == self.down else f"; up={self.up:g}, down={self.down:g}"
        return f"ladder({self.n_levels} levels, total {self._acc[-1]:g}{tail})"


@dataclass(frozen=True, slots=True)
class SwitchCost(CostFunction):
    """Unordered categorical: pay the cost of *arriving* at the new level.

    Distance between two occupations is not a meaningful quantity, but the
    barrier to entry of each one is: becoming an executive is hard whoever you
    are now.  Asymmetric by construction, which is the point.
    """

    entry: tuple[float, ...]
    base: float = 0.0
    _entry: _ARR = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.entry:
            raise ValueError("a switch cost needs one entry cost per level")
        if any(e < 0 for e in self.entry):
            raise ValueError(f"entry costs must be non-negative: {self.entry}")
        object.__setattr__(self, "_entry", np.asarray(self.entry, dtype=float))

    @property
    def n_levels(self) -> int:
        return len(self.entry)

    def __call__(self, old: Any, new: Any) -> _ARR:
        old, new = _as_array(old), _as_array(new)
        arrival = float(self.base) + self._entry[_levels(new, self.n_levels)]
        return np.where(_levels(old, self.n_levels) == _levels(new, self.n_levels),
                        0.0, arrival)

    def maximum(self, lo: float, hi: float) -> float:
        first = int(_levels(_as_array(lo), self.n_levels))
        last = int(_levels(_as_array(hi), self.n_levels))
        span = self._entry[min(first, last):max(first, last) + 1]
        return float(self.base) + float(np.max(span) if span.size else 0.0)

    def describe(self) -> str:
        return (f"switch({self.n_levels} levels, entry "
                f"{np.min(self._entry):g}-{np.max(self._entry):g})")


# ------------------------------------------------------------------- factories
def constant_cost(value: float) -> ConstantCost:
    """Any change costs ``value`` (§7.1's ``age``/``marital_status`` shape)."""
    return ConstantCost(float(value))


def linear_cost(per_unit: float, up: float = 1.0, down: float = 1.0) -> LinearCost:
    """Constant marginal effort per unit of change."""
    return LinearCost(float(per_unit), float(up), float(down))


def piecewise_linear(*knots: float, slopes: Sequence[float],
                     up: float = 1.0, down: float = 1.0) -> PiecewiseLinearCost:
    """``piecewise_linear(0, 40, 60, 99, slopes=[0.01, 0.03, 0.10])`` — §7.1 verbatim."""
    return PiecewiseLinearCost(tuple(float(k) for k in knots),
                               tuple(float(s) for s in slopes), float(up), float(down))


def categorical_cost(levels: Sequence[str], jump_cost: Sequence[float],
                     ordered: bool = False, base: float = 0.0,
                     up: float = 1.0, down: float = 1.0) -> CostFunction:
    """``categorical_cost(levels=[...], jump_cost=[...])`` — §7.1's signature.

    ``ordered=True`` reads ``jump_cost`` as the ladder of ``len(levels) - 1``
    rung costs (an ordinal feature such as ``education``); ``ordered=False``
    reads it as one entry cost per level (an unordered feature such as
    ``occupation``).  ``levels`` is required in both cases so a mismatch between
    the cost model and the loader's level order fails here rather than silently
    pricing the wrong category.
    """
    expected = len(levels) - 1 if ordered else len(levels)
    if len(jump_cost) != expected:
        kind = "rung costs" if ordered else "entry costs"
        raise ValueError(f"{len(levels)} levels need {expected} {kind}, "
                         f"got {len(jump_cost)}")
    values = tuple(float(c) for c in jump_cost)
    if ordered:
        return LadderCost(values, float(up), float(down))
    return SwitchCost(values, float(base))


# ------------------------------------------------------------------ adult (§7.1)
# The effort scale is anchored by the instruction's own numbers: 10.0 is "a thing
# a person cannot do" (age, gender), 5.0 is a marriage or a divorce, and
# 0.01/hour makes a 40h working week worth 0.40.  Everything below is calibrated
# against those three points, so 1.0 reads as "one hard year of someone's life".

#: ``education`` is a 16-rung ladder; the expensive rungs are the credentials,
#: not the grades.  HS-grad -> Bachelors costs 2.15, Bachelors -> Doctorate 3.80.
EDUCATION_RUNGS = (
    0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  # Preschool .. 11th -> 12th
    0.25,  # 12th -> HS-grad (the diploma, not the year)
    0.50,  # HS-grad -> Some-college
    0.45,  # Some-college -> Assoc-voc
    0.30,  # Assoc-voc -> Assoc-acdm
    0.90,  # Assoc-acdm -> Bachelors
    1.10,  # Bachelors -> Masters
    1.30,  # Masters -> Prof-school
    1.40,  # Prof-school -> Doctorate
)

#: ``occupation`` is unordered: what matters is the barrier to *entry*.  The two
#: imputation levels priced at 2.00 ("Unknown") are not actions a person can
#: take, so they are deliberately dearer than any real occupation.
OCCUPATION_ENTRY = (
    0.20,  # Priv-house-serv
    0.20,  # Other-service
    0.25,  # Handlers-cleaners
    0.45,  # Farming-fishing
    0.45,  # Machine-op-inspct
    0.50,  # Transport-moving
    0.55,  # Adm-clerical
    0.75,  # Craft-repair (apprenticeship)
    0.60,  # Sales
    0.85,  # Protective-serv (vetting, fitness)
    1.10,  # Armed-Forces
    0.95,  # Tech-support (certification)
    1.40,  # Prof-specialty (degree + licence)
    1.50,  # Exec-managerial (degree + tenure)
    2.00,  # Unknown - an imputation artefact, never a plan
)

#: ``workclass``: "Never-worked" and "Unknown" are historical facts rather than
#: states one can move into, hence the prohibitive entry cost.
WORKCLASS_ENTRY = (
    0.15,  # Without-pay
    2.00,  # Never-worked - not reachable from any employed state
    0.30,  # Private
    0.70,  # Self-emp-not-inc
    1.10,  # Self-emp-inc (capital + incorporation)
    0.65,  # Local-gov
    0.70,  # State-gov
    0.85,  # Federal-gov (clearance, competitive hiring)
    2.00,  # Unknown - an imputation artefact
)


def _adult_costs(spec: DatasetSpec) -> dict[str, CostFunction]:
    """§7.1's ``COST_MODELS``, extended to the four features it did not price."""
    return {
        # Immutable: the 10.0 mirrors COST_LEVELS["infinite"]; the 1e6 penalty is
        # added on top by CostModel, because these are not merely expensive.
        "age": constant_cost(10.0),
        "gender": constant_cost(10.0),
        # A marriage or a divorce is one event, not a distance (§7.1: 5.0).
        "marital_status": constant_cost(5.0),
        "education": categorical_cost(
            levels=spec.get("education").levels or (),
            jump_cost=EDUCATION_RUNGS, ordered=True,
        ),
        "occupation": categorical_cost(
            levels=spec.get("occupation").levels or (),
            jump_cost=OCCUPATION_ENTRY,
        ),
        # §7.1 verbatim: overtime is cheap, then unpleasant, then punishing.
        "hours_per_week": piecewise_linear(0, 40, 60, 99, slopes=[0.01, 0.03, 0.10]),
        "workclass": categorical_cost(
            levels=spec.get("workclass").levels or (),
            jump_cost=WORKCLASS_ENTRY,
        ),
        # Capital income is *regressive* in effort: the first investable dollars
        # are the hard ones, after which capital compounds on its own.
        "capital_gain": piecewise_linear(
            0, 5_000, 20_000, 99_999, slopes=[1.0e-4, 6.0e-5, 2.0e-5],
        ),
        # Realising a loss destroys wealth, and large losses disproportionately.
        "capital_loss": piecewise_linear(0, 1_000, 4_356, slopes=[4.0e-4, 8.0e-4]),
    }


# ----------------------------------------------------------------- german (§7.2)
#: Tenure cannot be bought, only waited for, which is why ``employment_since``
#: is the dearest ladder in this dataset despite having only four rungs.
EMPLOYMENT_RUNGS = (0.90, 0.70, 0.90, 0.90)   # unemployed -> ... -> above_7_years
JOB_RUNGS = (0.80, 1.20, 1.60)                # unskilled -> ... -> management
SAVINGS_RUNGS = (0.25, 0.45, 0.55, 0.75)      # none -> ... -> above_1000_DM
CHECKING_RUNGS = (0.10, 0.40, 0.50)           # no_account -> ... -> above_200_DM
PROPERTY_RUNGS = (0.60, 0.80, 1.80)           # none -> car -> savings -> real_estate
HOUSING_ENTRY = (0.20, 0.50, 2.00)            # for_free, rent, own

#: ``purpose`` is what the applicant tells the bank, so it is nearly free — but
#: the three purposes that imply a real commitment are priced like one.
PURPOSE_ENTRY = (
    0.15,  # car_new
    0.12,  # car_used
    0.12,  # furniture_equipment
    0.10,  # radio_tv
    0.10,  # domestic_appliances
    0.15,  # repairs
    0.35,  # education
    0.10,  # vacation
    0.35,  # retraining
    0.45,  # business
    0.10,  # other
)


def _german_costs(spec: DatasetSpec) -> dict[str, CostFunction]:
    """Analogous cost model for German Credit (§7.2), same effort scale."""
    return {
        # Immutable by schema: demography, and a repayment record that cannot be
        # rewritten (which is exactly why credit_history is non-manipulable).
        "age": constant_cost(10.0),
        "sex": constant_cost(10.0),
        "num_dependents": constant_cost(10.0),
        "credit_history": constant_cost(10.0),
        # Opening or closing a credit line is paperwork, not hardship.
        "existing_credits": linear_cost(0.30),
        "employment_since": categorical_cost(
            levels=spec.get("employment_since").levels or (),
            jump_cost=EMPLOYMENT_RUNGS, ordered=True,
        ),
        "job": categorical_cost(
            levels=spec.get("job").levels or (),
            jump_cost=JOB_RUNGS, ordered=True,
        ),
        # Saving is slow; spending what you saved is not, hence down << up.
        "savings_status": categorical_cost(
            levels=spec.get("savings_status").levels or (),
            jump_cost=SAVINGS_RUNGS, ordered=True, down=0.15,
        ),
        "checking_status": categorical_cost(
            levels=spec.get("checking_status").levels or (),
            jump_cost=CHECKING_RUNGS, ordered=True, down=0.15,
        ),
        # Acquiring property is the expensive direction; selling it is not.
        "property_magnitude": categorical_cost(
            levels=spec.get("property_magnitude").levels or (),
            jump_cost=PROPERTY_RUNGS, ordered=True, down=0.20,
        ),
        "housing": categorical_cost(
            levels=spec.get("housing").levels or (),
            jump_cost=HOUSING_ENTRY,
        ),
        "purpose": categorical_cost(
            levels=spec.get("purpose").levels or (),
            jump_cost=PURPOSE_ENTRY,
        ),
        # Asking for a *smaller* loan is the sacrifice; asking for more is free.
        # Calibrated so that giving up the whole 19,750 DM range costs ~1.5, i.e.
        # rather more than a change of job but less than acquiring property.
        "credit_amount": piecewise_linear(
            250, 5_000, 20_000, slopes=[1.5e-4, 5.0e-5], up=0.30,
        ),
        # Shortening the term is what raises the monthly instalment, so both
        # directions are priced: 68 months of swing is worth ~0.8.
        "duration": linear_cost(1.2e-2),
        # Each band commits a further slice of monthly disposable income.
        "installment_rate": linear_cost(0.35),
    }


# ------------------------------------------------------------------ heloc (§7.2)
def _heloc_costs(spec: DatasetSpec) -> dict[str, CostFunction]:  # noqa: ARG001
    """Analogous cost model for FICO HELOC (§7.2) — every feature is numeric.

    The asymmetries carry the domain knowledge here: a credit file gets older
    only by waiting, delinquencies age out rather than being erased, and the one
    genuinely actionable lever is paying down revolving balances.
    """
    return {
        # Immutable by schema: a bureau score and three history summaries.
        "ExternalRiskEstimate": constant_cost(10.0),
        "AverageMInFile": constant_cost(10.0),
        "MaxDelqEver": constant_cost(10.0),
        "PercentTradesWBalance": constant_cost(10.0),
        # Age of the oldest trade line: purely a function of waiting.
        "MSinceOldestTradeOpen": linear_cost(5.0e-3),
        # The first satisfactory accounts are the hard ones; the habit scales.
        "NumSatisfactoryTrades": piecewise_linear(
            0, 10, 30, 80, slopes=[0.10, 0.05, 0.02],
        ),
        # The last few percent mean a spotless file, which is the hardest part.
        "PercentTradesNeverDelq": piecewise_linear(
            0, 80, 95, 100, slopes=[0.01, 0.04, 0.12],
        ),
        "NumTotalTrades": linear_cost(0.03, down=0.60),
        "NumTradesOpeninLast12M": linear_cost(0.05),
        # Applying is free; *un*-applying means waiting out the six-month window.
        "NumInqLast6M": linear_cost(0.02, up=0.40),
        # Paying down revolving debt costs money; borrowing more does not.
        "NetFractionRevolvingBurden": linear_cost(0.01, up=0.25),
        "NumRevolvingTradesWBalance": linear_cost(0.12, up=0.30),
    }


def _diabetes_costs(spec: DatasetSpec) -> dict[str, CostFunction]:  # noqa: ARG001
    """Cost model for Pima diabetes (§7.2) — all numeric, three real levers.

    The shape of clinical effort is not the shape of financial effort, and this
    is the one dataset where that shows.  Two asymmetries carry it: losing weight
    is several times harder than gaining it, and *lowering* a glucose or pressure
    reading takes sustained work while letting it drift up takes none.  The three
    piecewise slopes all steepen towards the healthy end, because the last few
    points of any of these three are the ones the DPP trial found people could
    not hold.
    """
    return {
        # Immutable by schema: history, genetics, and two measurements of state.
        "Age": constant_cost(10.0),
        "Pregnancies": constant_cost(10.0),
        "DiabetesPedigreeFunction": constant_cost(10.0),
        "SkinThickness": constant_cost(10.0),
        "Insulin": constant_cost(10.0),
        # The substantive lever.  Reference: ~7% body weight is the DPP target
        # and about the practical ceiling, roughly 2 BMI points for this cohort,
        # so a 2-point loss should already be a substantial fraction of budget.
        # Below 25 there is nothing left to gain and the effort keeps rising.
        "BMI": piecewise_linear(
            18.0, 25.0, 32.0, 68.0, slopes=[0.50, 0.16, 0.09], down=1.0, up=0.20,
        ),
        # Diet, exercise and medication.  Normal fasting-equivalent range is the
        # cheap stretch; pushing into the non-diabetic range is where it bites.
        "Glucose": piecewise_linear(
            40, 100, 140, 200, slopes=[0.040, 0.014, 0.008], down=1.0, up=0.15,
        ),
        # First-line antihypertensives work within weeks and cost little, which
        # is why this is the low-cost lever despite being a physiological value.
        "BloodPressure": piecewise_linear(
            24, 70, 90, 122, slopes=[0.030, 0.010, 0.006], down=1.0, up=0.15,
        ),
    }


#: Dataset -> builder.  Each builder returns §7.1's ``COST_MODELS`` mapping for
#: that dataset; they take the :class:`DatasetSpec` so the level vectors above
#: are validated against the loader's declared level *order* on every build.
def _synth_reach_costs(spec: DatasetSpec) -> dict[str, CostFunction]:
    """Cost model for the synthetic ``synth_reach`` world (Workstream C).

    Immutable ``age`` carries the 10.0 "infinite" effort (the admissibility
    penalty is added by :class:`CostModel` on top); the four actionable levers
    are priced linearly in their own units, with ``training`` an ordinal ladder.
    This world exists to show reachability, so the levers are deliberately cheap.
    """
    return {
        "age": constant_cost(10.0),
        "experience": linear_cost(0.05),
        "training": categorical_cost(
            levels=spec.get("training").levels or (),
            jump_cost=(0.40, 0.50, 0.60), ordered=True,
        ),
        "income": linear_cost(0.02),
        "savings": linear_cost(0.03),
    }


def _synth_block_costs(spec: DatasetSpec) -> dict[str, CostFunction]:
    """Cost model for the synthetic ``synth_block`` world (Workstream C).

    Same shape and effort scale as :func:`_synth_reach_costs`; the two worlds
    differ in their SCM and target rule, not in how actions are priced, so the
    honest non-recourse cases in ``synth_block`` are a fact about the graph
    rather than an artefact of a harsher cost model.
    """
    return {
        "age": constant_cost(10.0),
        "experience": linear_cost(0.05),
        "training": categorical_cost(
            levels=spec.get("training").levels or (),
            jump_cost=(0.40, 0.50, 0.60), ordered=True,
        ),
        "income": linear_cost(0.02),
        "savings": linear_cost(0.03),
    }


COST_MODELS = {
    "adult": _adult_costs,
    "german": _german_costs,
    "heloc": _heloc_costs,
    "diabetes": _diabetes_costs,
    "synth_reach": _synth_reach_costs,
    "synth_block": _synth_block_costs,
}


def cost_functions(dataset: str) -> dict[str, CostFunction]:
    """The per-feature ``(x_old, x_new) -> cost`` mapping for ``dataset``."""
    if dataset not in COST_MODELS:
        raise KeyError(f"no cost model for {dataset!r}; "
                       f"expected one of {sorted(COST_MODELS)}")
    return COST_MODELS[dataset](dataset_spec(dataset))


# ------------------------------------------------------------------ aggregation
#: Why an action is not merely expensive but disallowed, in reporting order.
VIOLATIONS = ("immutable", "bounds", "monotone")


@dataclass(frozen=True, slots=True)
class ActionCost:
    """One feature's contribution to the cost of a counterfactual."""

    feature: str
    x_old: float
    x_new: float
    raw: float
    weight: float
    #: ``weight * raw / budget`` — this action's share of the normalised total.
    normalised: float
    violation: str = ""

    @property
    def admissible(self) -> bool:
        return not self.violation

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "x_old": self.x_old,
            "x_new": self.x_new,
            "raw": self.raw,
            "weight": self.weight,
            "normalised": self.normalised,
            "violation": self.violation,
        }


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """The full cost of one ``x -> x'`` move, per feature and in total."""

    dataset: str
    #: §7.2's "sum over changed features", in hand-crafted effort units.
    raw: float
    #: ``sum(weight * raw)`` before division by the budget.
    weighted: float
    #: ``weighted / budget`` — in ``[0, 1]`` for any admissible plan.
    normalised: float
    #: ``INFINITE_COST`` per inadmissible action.
    penalty: float
    actions: tuple[ActionCost, ...]

    @property
    def total(self) -> float:
        """What an optimiser should minimise: normalised effort plus penalties."""
        return self.normalised + self.penalty

    @property
    def feasible(self) -> bool:
        return self.penalty == 0.0

    @property
    def changed(self) -> tuple[str, ...]:
        return tuple(a.feature for a in self.actions)

    @property
    def n_changed(self) -> int:
        return len(self.actions)

    @property
    def violations(self) -> tuple[tuple[str, str], ...]:
        return tuple((a.feature, a.violation) for a in self.actions if a.violation)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([a.to_dict() for a in self.actions])

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "raw": self.raw,
            "weighted": self.weighted,
            "normalised": self.normalised,
            "penalty": self.penalty,
            "total": self.total,
            "feasible": self.feasible,
            "n_changed": self.n_changed,
            "changed": list(self.changed),
            "actions": [a.to_dict() for a in self.actions],
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        flag = "" if self.feasible else f", penalty={self.penalty:.0f}"
        return (f"CostBreakdown({self.dataset}, raw={self.raw:.3f}, "
                f"norm={self.normalised:.4f}, changed={list(self.changed)}{flag})")


class CostModel:
    """The dataset's cost model: per-feature functions plus their aggregation.

    Everything operates in *encoded* space (level index for categoricals, original
    units for numerics), which is the space the SCM and every counterfactual
    method share — so a cost is never accidentally measured in standard
    deviations.
    """

    def __init__(self, dataset: str, spec: DatasetSpec | None = None,
                 scm: SCM | None = None, max_actions: int | None = None) -> None:
        self.dataset = dataset
        self.spec = spec if spec is not None else dataset_spec(dataset)
        self.scm = scm if scm is not None else get_scm(dataset)
        self.max_actions = int(max_actions if max_actions is not None
                               else CONFIG.MAX_ACTIONS)
        self.functions = self._build_functions()
        self.feature_names: list[str] = self.spec.feature_names

        feats = self.spec.features
        self.bounds = np.array([f.encoded_bounds() for f in feats], dtype=float)
        self.is_discrete = np.array([f.is_discrete for f in feats], dtype=bool)
        self.monotone = tuple(f.monotone for f in feats)
        #: A feature may be acted on at all: manipulable *and* not priced infinite.
        self.admissible = np.array(
            [f.manipulable and not self.scm.is_infinite(f.name) for f in feats],
            dtype=bool,
        )
        #: The SCM's qualitative effort label, as a multiplier (rule #9 <-> #10).
        self.weights = np.array([self.scm.cost(f.name) for f in feats], dtype=float)
        self.max_raw = np.array(
            [self.functions[f.name].maximum(*self.bounds[j])
             for j, f in enumerate(feats)],
            dtype=float,
        )
        self.weighted_max = self.weights * self.max_raw
        self.budget = self._budget()
        self._check_calibration()
        LOGGER.info("CostModel[%s]: %d features, %d actionable, budget %.3f "
                    "(top-%d weighted actions)", dataset, len(feats),
                    int(self.admissible.sum()), self.budget, self.max_actions)

    # ------------------------------------------------------------------- setup
    def _build_functions(self) -> dict[str, CostFunction]:
        if self.dataset not in COST_MODELS:
            raise KeyError(f"no cost model for {self.dataset!r}; "
                           f"expected one of {sorted(COST_MODELS)}")
        functions = COST_MODELS[self.dataset](self.spec)
        declared, priced = set(self.spec.feature_names), set(functions)
        if declared != priced:
            raise ValueError(
                f"{self.dataset}: cost model / schema mismatch — unpriced "
                f"{sorted(declared - priced)}, unknown {sorted(priced - declared)}"
            )
        return functions

    def _budget(self) -> float:
        """Cost of the most expensive plan the pipeline permits.

        ``CONFIG.MAX_ACTIONS`` bounds how many features a recourse plan may
        touch, so the natural unit of "maximally expensive" is the sum of the
        that-many dearest single-feature actions.  Any admissible plan is then
        at most ``1.0`` after division, and a plan that ignores the action cap
        can exceed it — which is information, not a bug.
        """
        candidates = np.sort(self.weighted_max[self.admissible])[::-1]
        budget = float(np.sum(candidates[: self.max_actions]))
        if budget <= 0.0:
            raise ValueError(f"{self.dataset}: every actionable feature has zero "
                             "maximum cost — the cost model is degenerate")
        return budget

    def _check_calibration(self) -> None:
        """Warn when a hand-crafted function does not cover its feature's range."""
        for j, feat in enumerate(self.spec.features):
            fn = self.functions[feat.name]
            lo, hi = self.bounds[j]
            domain = fn.domain()
            if domain is not None and (domain[0] > lo or domain[1] < hi):
                LOGGER.warning(
                    "%s: %s is priced on %s but the schema allows [%g, %g] — costs "
                    "outside the schedule are clipped", self.dataset, feat.name,
                    domain, lo, hi,
                )
            if self.admissible[j] and self.max_raw[j] <= 0.0:
                LOGGER.warning("%s: %s is actionable but costs nothing to change",
                               self.dataset, feat.name)

    # -------------------------------------------------------------- vectorised
    def _prepare(self, X_old: Any, X_new: Any) -> tuple[_ARR, _ARR]:
        old = np.atleast_2d(_as_array(X_old))
        new = np.atleast_2d(_as_array(X_new))
        d = len(self.feature_names)
        for name, arr in (("x_old", old), ("x_new", new)):
            if arr.shape[1] != d:
                raise ValueError(f"{name} has {arr.shape[1]} columns, "
                                 f"expected {d} for {self.dataset}")
        if old.shape[0] == 1 and new.shape[0] > 1:
            old = np.repeat(old, new.shape[0], axis=0)
        elif new.shape[0] == 1 and old.shape[0] > 1:
            new = np.repeat(new, old.shape[0], axis=0)
        elif old.shape[0] != new.shape[0]:
            raise ValueError(f"{old.shape[0]} original rows vs {new.shape[0]} "
                             "counterfactual rows")
        return old, new

    def changed_mask(self, X_old: Any, X_new: Any) -> _ARR:
        """``(n, d)`` boolean: did this feature actually move?

        Discrete features are compared at level resolution, so a categorical
        nudged by 0.4 of a level is *not* a change — the single definition of
        "changed" used by the raw sum, the L0 count and the violation checks.
        """
        old, new = self._prepare(X_old, X_new)
        disc = self.is_discrete
        old_c = np.where(disc, np.rint(old), old)
        new_c = np.where(disc, np.rint(new), new)
        return ~np.isclose(old_c, new_c, rtol=0.0, atol=CHANGE_ATOL)

    def raw_matrix(self, X_old: Any, X_new: Any) -> _ARR:
        """``(n, d)`` raw effort per feature — one vectorised call per feature."""
        old, new = self._prepare(X_old, X_new)
        out = np.zeros_like(old, dtype=float)
        for j, name in enumerate(self.feature_names):
            out[:, j] = self.functions[name](old[:, j], new[:, j])
        return out * self.changed_mask(old, new)

    def violation_codes(self, X_old: Any, X_new: Any) -> np.ndarray:
        """``(n, d)`` array of ``""`` or a member of :data:`VIOLATIONS`."""
        old, new = self._prepare(X_old, X_new)
        changed = self.changed_mask(old, new)
        codes = np.full(old.shape, "", dtype=object)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        out_of_bounds = (new < lo - CHANGE_ATOL) | (new > hi + CHANGE_ATOL)
        increase_only = np.array([m == "increase" for m in self.monotone])
        decrease_only = np.array([m == "decrease" for m in self.monotone])
        wrong_way = ((increase_only & (new < old)) | (decrease_only & (new > old)))
        # Reported in VIOLATIONS order so the message is deterministic.
        codes[changed & out_of_bounds] = "bounds"
        codes[changed & wrong_way & (codes == "")] = "monotone"
        codes[changed & ~self.admissible] = "immutable"
        return codes

    def batch(self, X_old: Any, X_new: Any) -> dict[str, _ARR]:
        """Every cost quantity for ``n`` rows at once (the CAC inner loop)."""
        old, new = self._prepare(X_old, X_new)
        raw = self.raw_matrix(old, new)
        changed = self.changed_mask(old, new)
        codes = self.violation_codes(old, new)
        n_violations = (codes != "").sum(axis=1)
        weighted = raw @ self.weights
        normalised = weighted / self.budget
        penalty = INFINITE_COST * n_violations
        return {
            "raw": raw.sum(axis=1),
            "weighted": weighted,
            "normalised": normalised,
            "penalty": penalty.astype(float),
            "total": normalised + penalty,
            "n_changed": changed.sum(axis=1),
            "n_violations": n_violations,
            "feasible": n_violations == 0,
        }

    # ------------------------------------------------------------------ scalars
    def feature_cost(self, feature: str, x_old: float, x_new: float) -> float:
        """Raw effort of one feature's move — §7.1's ``(x_old, x_new) -> cost``."""
        if feature not in self.functions:
            raise KeyError(f"{feature!r} is not a feature of {self.dataset!r}")
        j = self.feature_names.index(feature)
        old = np.array([[x_old]], dtype=float)
        new = np.array([[x_new]], dtype=float)
        disc = bool(self.is_discrete[j])
        moved = not np.isclose(np.rint(old) if disc else old,
                               np.rint(new) if disc else new,
                               rtol=0.0, atol=CHANGE_ATOL).item()
        return float(self.functions[feature](old, new).item()) if moved else 0.0

    def raw_cost(self, x_old: Any, x_new: Any) -> float:
        """Total raw effort: the sum over changed features (§7.2)."""
        return float(self.batch(x_old, x_new)["raw"][0])

    def cost(self, x_old: Any, x_new: Any) -> float:
        """Normalised cost in ``[0, 1]`` plus any infeasibility penalty."""
        return float(self.batch(x_old, x_new)["total"][0])

    def normalised_cost(self, x_old: Any, x_new: Any) -> float:
        """Normalised cost only, ignoring admissibility."""
        return float(self.batch(x_old, x_new)["normalised"][0])

    def is_feasible(self, x_old: Any, x_new: Any) -> bool:
        return bool(self.batch(x_old, x_new)["feasible"][0])

    def changed_features(self, x_old: Any, x_new: Any) -> list[str]:
        mask = self.changed_mask(x_old, x_new)[0]
        return [n for n, m in zip(self.feature_names, mask) if m]

    def breakdown(self, x_old: Any, x_new: Any) -> CostBreakdown:
        """Per-feature accounting for a single counterfactual."""
        old, new = self._prepare(x_old, x_new)
        if old.shape[0] != 1:
            raise ValueError("breakdown() takes a single row; use batch() for many")
        raw = self.raw_matrix(old, new)[0]
        changed = self.changed_mask(old, new)[0]
        codes = self.violation_codes(old, new)[0]
        actions = tuple(
            ActionCost(
                feature=name,
                x_old=float(old[0, j]),
                x_new=float(new[0, j]),
                raw=float(raw[j]),
                weight=float(self.weights[j]),
                normalised=float(self.weights[j] * raw[j] / self.budget),
                violation=str(codes[j]),
            )
            for j, name in enumerate(self.feature_names) if changed[j]
        )
        weighted = float(raw @ self.weights)
        return CostBreakdown(
            dataset=self.dataset,
            raw=float(raw.sum()),
            weighted=weighted,
            normalised=weighted / self.budget,
            penalty=INFINITE_COST * float((codes != "").sum()),
            actions=actions,
        )

    def as_effort(self, x_old: Any, x_new: Any) -> tuple[CostBreakdown,
                                                         tuple[tuple[str, str], ...]]:
        """Price a *state* rather than a plan: what it would cost to bring about.

        §7's admissibility rules answer "may this person do that?", so they only
        make sense about deliberate moves.  Ask them about a state the world
        arrived at on its own — an applicant's risk score after they closed an
        account, say — and every one of them misfires: the immutability guard
        prices a consequence at ``INFINITE_COST``, and a monotone rule fires on a
        direction nobody chose.  Both answers are category errors, and both are
        large enough to swamp a table of averages.

        So this prices the coordinates §7 *is* willing to price and hands back
        the refused ones by name.  The result is finite, on the same scale as a
        plan's cost, and comparable across records; the excluded coordinates are
        not hidden but listed, which is what lets §9 report "the SCM moved a
        feature no one may act on" as a frequency instead of as an infinity.
        """
        old, new = self._prepare(x_old, x_new)
        if old.shape[0] != 1:
            raise ValueError("as_effort() takes a single row")
        refused = self.breakdown(old, new).violations
        names = {name for name, _ in refused}
        keep = np.array([n not in names for n in self.feature_names], dtype=bool)
        return self.breakdown(old, np.where(keep, new, old)), refused

    # ----------------------------------------------------------------- reading
    def pretty(self, feature: str, value: float) -> str:
        """Encoded value -> the string a human would recognise."""
        feat = self.spec.get(feature)
        if feat.levels:
            idx = int(np.clip(round(float(value)), 0, len(feat.levels) - 1))
            return feat.levels[idx]
        text = f"{float(value):.0f}" if feat.integer else f"{float(value):.2f}"
        return f"{text} {feat.unit}".strip()

    def explain(self, x_old: Any, x_new: Any) -> list[str]:
        """One human-readable line per action, cheapest first."""
        breakdown = self.breakdown(x_old, x_new)
        lines = []
        for action in sorted(breakdown.actions, key=lambda a: a.raw):
            note = f" [{action.violation}]" if action.violation else ""
            lines.append(
                f"{action.feature}: {self.pretty(action.feature, action.x_old)} -> "
                f"{self.pretty(action.feature, action.x_new)} "
                f"(cost {action.raw:.2f}){note}"
            )
        return lines

    # ----------------------------------------------------------------- exports
    def table(self) -> pd.DataFrame:
        """The documented cost model, one row per feature (rule #10)."""
        in_budget = np.zeros(len(self.feature_names), dtype=bool)
        order = np.argsort(np.where(self.admissible, self.weighted_max, -np.inf))[::-1]
        in_budget[order[: self.max_actions]] = self.admissible[order[: self.max_actions]]
        return pd.DataFrame({
            "feature": self.feature_names,
            "kind": [f.kind for f in self.spec.features],
            "cost_label": [self.scm.label(n) for n in self.feature_names],
            "weight": self.weights,
            "function": [self.functions[n].describe() for n in self.feature_names],
            "max_raw": self.max_raw,
            "weighted_max": self.weighted_max,
            "actionable": self.admissible,
            "monotone": list(self.monotone),
            "in_budget": in_budget,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "budget": self.budget,
            "max_actions": self.max_actions,
            "infinite_cost": INFINITE_COST,
            "n_features": len(self.feature_names),
            "n_actionable": int(self.admissible.sum()),
            "features": {
                name: {
                    "function": self.functions[name].describe(),
                    "cost_label": self.scm.label(name),
                    "weight": float(self.weights[j]),
                    "max_raw": float(self.max_raw[j]),
                    "weighted_max": float(self.weighted_max[j]),
                    "actionable": bool(self.admissible[j]),
                    "monotone": self.monotone[j],
                    "bounds": [float(self.bounds[j, 0]), float(self.bounds[j, 1])],
                }
                for j, name in enumerate(self.feature_names)
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"CostModel({self.dataset!r}, features={len(self.feature_names)}, "
                f"actionable={int(self.admissible.sum())}, budget={self.budget:.3f})")


@lru_cache(maxsize=None)
def get_cost_model(dataset: str, max_actions: int | None = None) -> CostModel:
    """Cached accessor — the cost model is immutable, schema-derived state."""
    return CostModel(dataset, max_actions=max_actions)


def total_cost(dataset: str, x_old: Any, x_new: Any, normalised: bool = True) -> float:
    """Convenience wrapper: the cost of one move under ``dataset``'s model."""
    model = get_cost_model(dataset)
    return model.cost(x_old, x_new) if normalised else model.raw_cost(x_old, x_new)


def all_cost_models(datasets: Sequence[str] | None = None) -> dict[str, CostModel]:
    return {d: get_cost_model(d) for d in (datasets or DATASETS)}


# ------------------------------------------------------------------- documenting
_TABLE_COLUMNS = {
    "feature": "Feature",
    "kind": "Type",
    "cost_label": "SCM cost",
    "weight": "Weight",
    "function": "Cost function",
    "max_raw": "Max raw",
    "monotone": "Direction",
}
#: The cost-function column holds a whole schedule, so let LaTeX wrap it.
_TABLE_FORMAT = r"lllr p{0.28\linewidth} rl"


def export_cost_models(datasets: Sequence[str] | None = None,
                       write: bool = True) -> dict[str, CostModel]:
    """Build every cost model and write the artefacts that document it.

    ``results/cost_models.{json,csv}`` for the pipeline, and one
    ``paper/tables/cost_model_<dataset>.tex`` per dataset for the paper — rule
    #10 asks for the cost functions to be *documented*, so they are exported
    rather than described in prose.
    """
    from src.utils.io import save_dataframe, save_json, save_latex_table  # noqa: PLC0415

    models = all_cost_models(datasets)
    frames = []
    for name, model in models.items():
        frame = model.table()
        frame.insert(0, "dataset", name)
        frames.append(frame)
        priced = (frame[frame["actionable"]]
                  .sort_values("weighted_max", ascending=False)
                  .set_index("feature")["weighted_max"].round(3))
        LOGGER.info("%s: budget %.3f = the %d dearest of %s", name, model.budget,
                    model.max_actions, priced.to_dict())
    combined = pd.concat(frames, ignore_index=True)
    if not write:
        return models

    save_json({name: m.to_dict() for name, m in models.items()},
              CONFIG.RESULTS_DIR / "cost_models.json")
    save_dataframe(combined, CONFIG.RESULTS_DIR / "cost_models.csv")
    for name, model in models.items():
        table = model.table()[list(_TABLE_COLUMNS)].rename(columns=_TABLE_COLUMNS)
        save_latex_table(
            table,
            f"cost_model_{name}",
            caption=(
                f"Hand-crafted intervention costs for {name}.  "
                f"``Max raw'' is the effort of traversing the feature's whole "
                f"admissible range, and ``Weight'' is the SCM's qualitative effort "
                f"label as a multiplier.  Costs are normalised by a budget of "
                f"{model.budget:.2f}, the {model.max_actions} most expensive "
                f"admissible actions, so any permitted plan lies in $[0, 1]$."
            ),
            label=f"tab:cost-model-{name}",
            column_format=_TABLE_FORMAT,
        )
    return models


def _cli() -> Any:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="python -m src.cost.cost_functions",
        description="Build, sanity-check and document the per-dataset cost models.",
    )
    parser.add_argument("-d", "--datasets", nargs="*", default=list(DATASETS),
                        choices=list(DATASETS), help="datasets to document")
    parser.add_argument("--no-write", action="store_true",
                        help="report only, write no artefacts")
    return parser.parse_args()


def main() -> int:
    args = _cli()
    CONFIG.ensure_dirs()
    models = export_cost_models(args.datasets, write=not args.no_write)
    for name, model in models.items():
        table = model.table()
        actionable = table[table["actionable"]]
        LOGGER.info(
            "Cost model %s: %d/%d actionable, budget %.3f, dearest actions %s",
            name, len(actionable), len(table), model.budget,
            list(actionable.nlargest(model.max_actions, "weighted_max")["feature"]),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "INFINITE_COST",
    "CHANGE_ATOL",
    "VIOLATIONS",
    "CostFunction",
    "ConstantCost",
    "LinearCost",
    "PiecewiseLinearCost",
    "LadderCost",
    "SwitchCost",
    "constant_cost",
    "linear_cost",
    "piecewise_linear",
    "categorical_cost",
    "COST_MODELS",
    "cost_functions",
    "ActionCost",
    "CostBreakdown",
    "CostModel",
    "get_cost_model",
    "total_cost",
    "all_cost_models",
    "export_cost_models",
]
