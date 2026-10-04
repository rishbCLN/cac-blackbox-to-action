"""The shared contract of every counterfactual method (§8.1, global rule #11).

Rule #11 says every method returns the same :class:`Counterfactual`.  That is
only worth something if the *fields* are computed the same way too, so this
module does more than declare a dataclass: :class:`CounterfactualMethod` owns
the post-processing no subclass should be free to reinvent — snapping a proposal
onto the feature grid, asking the black box whether it really flipped, pricing
it with the dataset's cost model, deduplicating, ranking.  A subclass supplies
:meth:`~CounterfactualMethod._propose` and inherits an identical notion of
"valid", "cost" and "changed".

Everything happens in *encoded* space (level index for categoricals, original
units for numerics) — the space the SCM and the cost model share.  The black box
standardises internally, so no method here ever sees a z-score, and none can
tell an ordinal from a count except through the schema they are all handed.

``generate`` returns *up to* ``k`` counterfactuals, ranked with the valid ones
first, and does not hide a failure: when a method cannot flip the prediction the
best near-misses come back with ``valid == False``.  §9's validity rate is then
a measurement rather than an artefact of silent filtering.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Sequence

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.cost.cost_functions import CostModel, get_cost_model
from src.data.preprocess import dataset_spec
from src.data.schema import DatasetSpec
from src.model.blackbox import BlackBoxModel, load_blackbox
from src.scm.define_scm import SCM, get_scm
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: Two plans that agree to this many decimals are the same plan.
DEDUP_DECIMALS = 6
#: Tolerance for "this feature moved", matching the cost layer's CHANGE_ATOL.
MOVE_ATOL = 1e-8

__all__ = [
    "Counterfactual",
    "CounterfactualMethod",
    "METHODS",
    "register",
    "get_method",
    "available_methods",
    "counterfactuals_frame",
]


@dataclass(frozen=True, slots=True, eq=False)
class Counterfactual:
    """One suggested change to one individual's record.

    The nine fields §8.1 requires come first and mean exactly what it says.
    Everything after them is diagnostic and defaulted, so a method that cannot
    produce it simply does not.

    ``cost`` is the number a search minimises: the cost model's *normalised*
    cost plus any infeasibility penalty.  ``raw_cost`` keeps the hand-crafted
    effort in the units §7 wrote it in, because a reader wants to hear "four
    extra years of study", not "0.19".
    """

    x_original: np.ndarray
    x_cf: np.ndarray
    y_original: int
    y_target: int
    y_predicted: int
    cost: float
    changed_features: tuple[str, ...]
    causal_path: tuple[str, ...]
    method: str
    # ------------------------------------------------------------ diagnostics
    dataset: str = ""
    feature_names: tuple[str, ...] = ()
    raw_cost: float = float("nan")
    proba_target: float = float("nan")
    feasible: bool = True
    violations: tuple[tuple[str, str], ...] = ()
    causal_gap: float = float("nan")
    explanation: tuple[str, ...] = ()
    n_queries: int = 0
    runtime_s: float = float("nan")
    extras: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- derived
    @property
    def valid(self) -> bool:
        """Did the black box actually change its mind?  (§9's validity.)"""
        return int(self.y_predicted) == int(self.y_target)

    @property
    def n_changed(self) -> int:
        """L0 sparsity — how many things the person is being asked to do."""
        return len(self.changed_features)

    @property
    def deltas(self) -> dict[str, tuple[float, float]]:
        """``{feature: (before, after)}`` for the features that moved."""
        idx = {n: j for j, n in enumerate(self.feature_names)}
        return {name: (float(self.x_original[idx[name]]), float(self.x_cf[idx[name]]))
                for name in self.changed_features if name in idx}

    def norm(self, order: float = 1.0) -> float:
        """Distance in encoded space; ``order=0`` counts features instead."""
        diff = np.asarray(self.x_cf, dtype=float) - np.asarray(self.x_original, float)
        if order == 0:
            return float(np.count_nonzero(np.abs(diff) > MOVE_ATOL))
        return float(np.linalg.norm(diff, ord=order))

    def key(self) -> tuple[float, ...]:
        """Identity of the *plan*, used to deduplicate a method's own output."""
        return tuple(np.round(np.asarray(self.x_cf, dtype=float), DEDUP_DECIMALS))

    def rank(self) -> tuple[int, int, float, int]:
        """Sort key: valid before invalid, feasible before not, then cheapest."""
        return (0 if self.valid else 1, 0 if self.feasible else 1,
                float(self.cost), self.n_changed)

    # ---------------------------------------------------------------- reading
    def with_extras(self, **kwargs: Any) -> "Counterfactual":
        """A copy carrying extra diagnostics — the frozen way to annotate."""
        fields = {k: v for k, v in kwargs.items()
                  if k in Counterfactual.__slots__ and k != "extras"}
        rest = {k: v for k, v in kwargs.items() if k not in fields}
        return replace(self, **fields, extras={**self.extras, **rest})

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record — one row of the results table."""
        return {
            "method": self.method,
            "dataset": self.dataset,
            "y_original": int(self.y_original),
            "y_target": int(self.y_target),
            "y_predicted": int(self.y_predicted),
            "valid": self.valid,
            "feasible": self.feasible,
            "cost": float(self.cost),
            "raw_cost": float(self.raw_cost),
            "proba_target": float(self.proba_target),
            "n_changed": self.n_changed,
            "changed_features": list(self.changed_features),
            "causal_path": list(self.causal_path),
            "causal_gap": float(self.causal_gap),
            "violations": [list(v) for v in self.violations],
            "l1": self.norm(1.0),
            "l2": self.norm(2.0),
            "n_queries": int(self.n_queries),
            "runtime_s": float(self.runtime_s),
            "x_original": [float(v) for v in np.asarray(self.x_original).ravel()],
            "x_cf": [float(v) for v in np.asarray(self.x_cf).ravel()],
            "explanation": list(self.explanation),
            **self.extras,
        }

    def frame(self) -> pd.DataFrame:
        """Per-feature before/after table for the app and the qualitative figures."""
        moved = {n for n in self.changed_features}
        return pd.DataFrame({
            "feature": list(self.feature_names),
            "original": np.asarray(self.x_original, dtype=float).ravel(),
            "counterfactual": np.asarray(self.x_cf, dtype=float).ravel(),
            "changed": [n in moved for n in self.feature_names],
        })

    def __repr__(self) -> str:
        state = "valid" if self.valid else "INVALID"
        if not self.feasible:
            state += "/infeasible"
        return (f"Counterfactual({self.method}, {state}, cost={self.cost:.3f}, "
                f"changes={list(self.changed_features)})")


def counterfactuals_frame(cfs: Sequence[Counterfactual]) -> pd.DataFrame:
    """One row per counterfactual — the shape §9 and §10 both consume."""
    if not cfs:
        return pd.DataFrame(columns=["method", "dataset", "valid", "cost"])
    drop = ("x_original", "x_cf", "explanation")
    return pd.DataFrame([
        {k: v for k, v in cf.to_dict().items() if k not in drop} for cf in cfs
    ])


class CounterfactualMethod(ABC):
    """Everything a counterfactual method needs except the search itself.

    A subclass implements :meth:`_propose`, which may return whatever its search
    produced — sloppy values, out-of-bounds values, duplicates, nothing at all.
    :meth:`generate` turns that into at most ``k`` audited counterfactuals.
    """

    #: Registry key, and the label that appears in every table and figure.
    name: ClassVar[str] = "base"
    #: Only a causal method fills ``causal_path`` (§8.1).
    causal: ClassVar[bool] = False

    def __init__(
        self,
        dataset: str,
        model: BlackBoxModel | None = None,
        spec: DatasetSpec | None = None,
        scm: SCM | None = None,
        cost_model: CostModel | None = None,
        random_state: int | None = None,
        **params: Any,
    ) -> None:
        self.dataset = dataset
        self.spec = spec if spec is not None else dataset_spec(dataset)
        self.scm = scm if scm is not None else get_scm(dataset)
        self.cost_model = cost_model if cost_model is not None else get_cost_model(dataset)
        self._model = model
        self.random_state = int(CONFIG.RANDOM_STATE if random_state is None
                                else random_state)
        self.rng = np.random.default_rng(self.random_state)
        self.params = params
        self._train: tuple[np.ndarray, np.ndarray] | None = None

        self.feature_names: tuple[str, ...] = tuple(self.spec.feature_names)
        self.d = len(self.feature_names)
        self.bounds = self.cost_model.bounds
        self.is_discrete = self.cost_model.is_discrete
        self.monotone = self.cost_model.monotone
        #: Features no plan may touch: non-manipulable, or priced infinite (§6/§7).
        self.immutable = ~self.cost_model.admissible
        self.actionable = np.where(self.cost_model.admissible)[0]
        #: Integer features cannot sit on a fractional bound, so snap to the grid.
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        self._lo = np.where(self.is_discrete, np.ceil(lo), lo)
        self._hi = np.where(self.is_discrete, np.floor(hi), hi)
        #: Feature ranges, for the scale-free distances the baselines minimise.
        self.span = np.maximum(hi - lo, 1e-12)

    def scaled_l1(self, X: Any, x: Any) -> np.ndarray:
        """Range-normalised L1, the distance the baselines actually care about.

        Encoded space mixes 0-15 education levels with a 0-99,999 capital gain,
        so an unscaled L1 would rank every proposal by whichever feature happens
        to have the widest units.  Dividing by the range is the same correction
        DiCE makes with its MAD weighting, applied identically to all methods.
        """
        diff = np.atleast_2d(np.asarray(X, dtype=float)) - np.asarray(x, float).ravel()
        return np.sum(np.abs(diff) / self.span, axis=1)

    # ------------------------------------------------------------- black box
    @property
    def model(self) -> BlackBoxModel:
        """The black box, loaded on first use.  Only ``predict*`` is ever called."""
        if self._model is None:
            self._model = load_blackbox(self.dataset, CONFIG.BLACKBOX_TYPE)
        return self._model

    # ------------------------------------------------------------ empirical
    @property
    def X_train(self) -> np.ndarray:
        """Encoded *training* rows — the empirical distribution, loaded on demand.

        DiCE samples from it, the oracle enumerates it, and CAC's plausibility
        term is measured against it.  The training split only: a method that read
        the test rows would be tuned on the data it is later scored on.
        """
        return self._load_train()[0]

    @property
    def y_train(self) -> np.ndarray:
        return self._load_train()[1]

    def _load_train(self) -> tuple[np.ndarray, np.ndarray]:
        if self._train is None:
            from src.data.preprocess import load_dataset  # noqa: PLC0415

            bundle = load_dataset(self.dataset)
            self._train = (bundle.X_train, bundle.y_train)
        return self._train

    def proba(self, X: Any) -> np.ndarray:
        """``P(y = 1)`` for one or more encoded rows — the whole API (rule #8)."""
        return self.model.proba_positive(np.atleast_2d(np.asarray(X, dtype=float)))

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(np.atleast_2d(np.asarray(X, dtype=float)))

    def target_proba(self, X: Any, y_target: int) -> np.ndarray:
        """Probability of the *desired* class, so search code is direction-free."""
        p = self.proba(X)
        return p if int(y_target) == 1 else 1.0 - p

    def score(self, X: Any, y_target: int) -> tuple[np.ndarray, np.ndarray]:
        """``(P(target), flipped)`` in one query pass.

        The mask is derived the way :meth:`BlackBoxModel.predict` derives it, so
        "flipped" cannot drift from the label the model would report — which it
        would if a search compared ``P(target)`` against the threshold directly
        at any threshold other than 0.5.
        """
        p_pos = self.proba(X)
        labels = (p_pos >= CONFIG.DECISION_THRESHOLD).astype(int)
        p_target = p_pos if int(y_target) == 1 else 1.0 - p_pos
        return p_target, labels == int(y_target)

    # -------------------------------------------------------------- geometry
    def snap(self, X: Any) -> np.ndarray:
        """Project onto the feature box and round discrete features to a level.

        Without this a continuous optimiser "recommends" 12.7 years of education
        and the cost model dutifully prices a move nobody can make.
        """
        arr = np.atleast_2d(np.asarray(X, dtype=float)).copy()
        np.clip(arr, self.bounds[:, 0], self.bounds[:, 1], out=arr)
        disc = self.is_discrete
        arr[:, disc] = np.rint(arr[:, disc])
        return np.clip(arr, self._lo, self._hi)

    def mask_immutable(self, X: Any, x: Any) -> np.ndarray:
        """Restore immutable features to their factual values.

        ``BASELINES_RESPECT_IMMUTABLE`` decides whether the baselines are at
        least *legal*: left alone, several of them "solve" adult by making the
        applicant younger, which is not a comparison but a strawman.
        """
        arr = np.atleast_2d(np.asarray(X, dtype=float)).copy()
        base = np.asarray(x, dtype=float).ravel()
        arr[:, self.immutable] = base[self.immutable]
        return arr

    def enforce_monotone(self, X: Any, x: Any) -> np.ndarray:
        """Clip moves that go the one way a feature is declared not to go."""
        arr = np.atleast_2d(np.asarray(X, dtype=float)).copy()
        base = np.asarray(x, dtype=float).ravel()
        for j, direction in enumerate(self.monotone):
            if direction == "increase":
                arr[:, j] = np.maximum(arr[:, j], base[j])
            elif direction == "decrease":
                arr[:, j] = np.minimum(arr[:, j], base[j])
        return arr

    # ----------------------------------------------------------------- search
    @abstractmethod
    def _propose(self, x: np.ndarray, y_target: int, k: int) -> np.ndarray:
        """Return an ``(m, d)`` array of candidate rows in encoded space.

        ``m`` need not equal ``k``, the rows need not be valid, in bounds, or
        distinct, and an empty array is a legitimate "I failed".  Everything
        after this point is handled by :meth:`generate`.
        """

    def _causal_path(self, x: np.ndarray, x_cf: np.ndarray) -> tuple[str, ...]:
        """Which nodes the intervention travels through — empty for a baseline.

        §8.1 reserves ``causal_path`` for CAC, and this default honours that: a
        method that does not set ``causal = True`` reports no path at all, so an
        empty path in the results is evidence about the method, not a missing
        value.
        """
        if not self.causal:
            return ()
        moved = [n for n, m in zip(self.feature_names,
                                   np.abs(x_cf - x) > MOVE_ATOL) if m]
        return tuple(moved) + tuple(self.scm.downstream(moved)) if moved else ()

    # ------------------------------------------------------------------ audit
    def _build(self, x: np.ndarray, x_cf: np.ndarray, y_original: int,
               y_target: int, proba_target: float) -> Counterfactual:
        """Price and audit one snapped candidate.  The single source of truth."""
        breakdown = self.cost_model.breakdown(x, x_cf)
        # Re-derive P(y=1) so the label matches BlackBoxModel.predict exactly,
        # including at a decision threshold that is not 0.5.
        p_pos = float(proba_target if int(y_target) == 1 else 1.0 - proba_target)
        return Counterfactual(
            x_original=x,
            x_cf=x_cf,
            y_original=int(y_original),
            y_target=int(y_target),
            y_predicted=int(p_pos >= CONFIG.DECISION_THRESHOLD),
            cost=float(breakdown.total),
            changed_features=tuple(breakdown.changed),
            causal_path=self._causal_path(x, x_cf),
            method=self.name,
            dataset=self.dataset,
            feature_names=self.feature_names,
            raw_cost=float(breakdown.raw),
            proba_target=float(proba_target),
            feasible=bool(breakdown.feasible),
            violations=tuple(breakdown.violations),
            explanation=tuple(self.cost_model.explain(x, x_cf)),
        )

    def audit(self, x: Any, X_cf: Any, y_target: int | None = None,
              y_original: int | None = None) -> list[Counterfactual]:
        """Snap, score and price candidate rows without running a search.

        Used by :meth:`generate`, by the evaluation layer to re-audit a method's
        saved output, and by the demo app to price a change the *user* typed.
        """
        x = self._check_row(x)
        X = np.atleast_2d(np.asarray(X_cf, dtype=float))
        if X.shape[1] != self.d:
            raise ValueError(f"candidates have {X.shape[1]} columns, expected {self.d}")
        if y_original is None:
            y_original = int(self.predict(x)[0])
        if y_target is None:
            y_target = int(1 - y_original)
        X = self.snap(X)
        if X.size == 0:
            return []
        p_target = self.target_proba(X, y_target)
        return [self._build(x, X[i], y_original, y_target, float(p_target[i]))
                for i in range(X.shape[0])]

    def _check_row(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float).ravel()
        if arr.size != self.d:
            raise ValueError(f"x has {arr.size} features, expected {self.d} "
                             f"for {self.dataset}")
        return arr

    @staticmethod
    def rank_and_trim(cfs: Sequence[Counterfactual], k: int) -> list[Counterfactual]:
        """Deduplicate identical plans, rank, and keep the best ``k``."""
        seen: dict[tuple[float, ...], Counterfactual] = {}
        for cf in sorted(cfs, key=lambda c: c.rank()):
            seen.setdefault(cf.key(), cf)
        return sorted(seen.values(), key=lambda c: c.rank())[:k]

    # --------------------------------------------------------------- §8.1 API
    def generate(self, x: Any, y_target: int | None = None,
                 k: int | None = None) -> list[Counterfactual]:
        """§8.1's entry point: up to ``k`` counterfactuals for one record.

        ``y_target`` defaults to flipping the black box's current answer, which
        is the only case recourse is about.  The returned list is ranked with
        valid, feasible, cheap plans first and never contains a duplicate.
        """
        x = self._check_row(x)
        k = int(CONFIG.N_COUNTERFACTUALS if k is None else k)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        # Counted from here, so ``n_queries`` is every row this call scored —
        # including the one that reads the current decision, which an explainer
        # has to spend before it can look for a different one.
        queries_before = self.model.query_count
        y_original = int(self.predict(x)[0])
        y_target = int(1 - y_original) if y_target is None else int(y_target)
        if y_target == y_original:
            raise ValueError(
                f"y_target={y_target} is already the model's answer for x; "
                "a counterfactual would be the identity"
            )

        started = time.perf_counter()
        proposals = np.atleast_2d(np.asarray(
            self._propose(x, y_target, k), dtype=float).reshape(-1, self.d))
        cfs = self.audit(x, proposals, y_target=y_target, y_original=y_original)
        elapsed = time.perf_counter() - started
        queries = self.model.query_count - queries_before

        chosen = self.rank_and_trim(cfs, k)
        n_valid = sum(c.valid for c in chosen)
        if not chosen:
            LOGGER.warning("%s/%s: no counterfactual produced", self.name, self.dataset)
        elif n_valid == 0:
            LOGGER.warning("%s/%s: %d candidates, none flipped the prediction",
                           self.name, self.dataset, len(cfs))
        # Both stamps describe the whole call, not a share of it: every plan in
        # the list came out of the same search, so §9 groups by instance and
        # takes one value rather than summing.
        return [c.with_extras(runtime_s=elapsed, n_queries=queries) for c in chosen]

    def generate_many(self, X: Any, k: int | None = None) -> list[list[Counterfactual]]:
        """``generate`` over a population — the experiments' outer loop."""
        rows = np.atleast_2d(np.asarray(X, dtype=float))
        return [self.generate(row, k=k) for row in rows]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dataset={self.dataset!r}, seed={self.random_state})"


# ------------------------------------------------------------------- registry
#: Name -> class, filled by :func:`register` as each method module is imported.
METHODS: dict[str, type[CounterfactualMethod]] = {}

#: Where to find each method, so ``base`` never imports its own subclasses.
_METHOD_MODULES: dict[str, str] = {
    "cac": "src.methods.cac",
    "dice": "src.methods.baseline_dice",
    "wachter": "src.methods.baseline_wachter",
    "random": "src.methods.baseline_random",
    "oracle": "src.methods.oracle",
}


def register(cls: type[CounterfactualMethod]) -> type[CounterfactualMethod]:
    """Class decorator making a method addressable by its ``name``."""
    key = cls.name
    if key in ("", "base"):
        raise ValueError(f"{cls.__name__} must set a distinctive `name`")
    if METHODS.get(key) not in (None, cls):
        raise ValueError(f"two methods claim the name {key!r}: "
                         f"{METHODS[key].__name__} and {cls.__name__}")
    METHODS[key] = cls
    return cls


def get_method(name: str, dataset: str, **kwargs: Any) -> CounterfactualMethod:
    """Build a method by name, importing its module on first use."""
    key = str(name).lower()
    if key not in METHODS:
        module = _METHOD_MODULES.get(key)
        if module is None:
            raise KeyError(f"unknown method {name!r}; expected one of "
                           f"{sorted(_METHOD_MODULES)}")
        import importlib  # noqa: PLC0415

        importlib.import_module(module)
        if key not in METHODS:  # pragma: no cover - a broken module
            raise KeyError(f"{module} did not register {key!r}")
    return METHODS[key](dataset, **kwargs)


def available_methods() -> tuple[str, ...]:
    """Every method name the pipeline knows, whether or not it is imported yet."""
    return tuple(sorted(set(_METHOD_MODULES) | set(METHODS)))

