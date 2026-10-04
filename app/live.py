"""Live recourse engine for the demo --- compute audits on the real models.

Where :mod:`app.components` reads what the experiments already wrote, this module
*computes*: it loads the trained black box, the fitted SCM and the cost model for
a dataset and answers, on demand, the three questions the whole project is about
for a plan the **user** typed:

* **F --- does it flip the box?**  one ``predict_proba`` on the edited record;
* **S --- does it survive the SCM?**  :meth:`~src.scm.fcm.FCM.repair` fills in the
  consequences the plan implies and we measure how far the plan sits from that
  world (and whether the flip survives once it is filled in);
* **A --- could a person do it?**  §7's cost model prices the move and flags any
  action that is immutable, wrong-way, over-budget or over the action cap.

``usable`` is the conjunction, defined here exactly as the pipeline defines it
(:mod:`src.evaluation.feasibility`), so the sandbox and the paper's headline
agree by construction.

Everything heavy is cached: models load once per process (``cache_resource``),
generated plans and cohort tables are memoised by ``(dataset, method, record, k)``
(``cache_data``).  The single-record audit is a couple of model calls and one SCM
repair, so it runs live on every widget change without a cache.

Nothing here trains anything and nothing mutates the black box: the engine only
ever *queries* ``predict`` / ``predict_proba`` (global rule #8), which is exactly
what a real applicant's bank API would expose.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Launchable from anywhere: make sure the project root wins over any stray
# site-packages ``src`` before the first-party imports run.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.cost.cost_functions import get_cost_model  # noqa: E402
from src.data.preprocess import load_dataset  # noqa: E402
from src.methods.base import Counterfactual, get_method  # noqa: E402
from src.model.blackbox import load_blackbox, model_path  # noqa: E402
from src.scm.define_scm import get_scm  # noqa: E402
from src.scm.fcm import fcm_path, get_fcm  # noqa: E402

#: The favourable class every dataset's loader agrees on (diabetes is inverted at
#: load time so ``1`` is "no onset"); recourse always targets it.
FAVOURABLE = 1
#: Reading order shared with the figures: CAC first, the random floor last.
METHOD_ORDER = ("cac", "oracle", "dice", "wachter", "random")
#: A move is "real" at this tolerance (matches the cost/method layers' MOVE_ATOL).
_ATOL = 1e-8


# --------------------------------------------------------------------- discovery

def datasets_live() -> list[str]:
    """Datasets whose model + processed data + FCM are all on disk, in config order.

    The live engine needs the black box (``models/xgb_<ds>.joblib``), the
    processed frame (``data/processed/<ds>.csv``) and the fitted structural
    equations (``models/fcm_<ds>.joblib``).  A dataset missing any of the three is
    dropped from the demo rather than crashing it.
    """
    ready: list[str] = []
    for ds in CONFIG_DATASETS():
        model_ok = model_path(ds, CONFIG.BLACKBOX_TYPE).exists()
        data_ok = (CONFIG.DATA_PROCESSED / f"{ds}.csv").exists()
        if model_ok and data_ok:
            ready.append(ds)
    return ready


def CONFIG_DATASETS() -> tuple[str, ...]:
    """The configured dataset keys (indirection keeps imports lazy for tests)."""
    from src.config import DATASETS  # noqa: PLC0415

    return DATASETS


def has_causal(dataset: str) -> bool:
    """Whether the fitted SCM exists, so the Causal audit/map can run for ``ds``."""
    return fcm_path(dataset).exists()


# ------------------------------------------------------------------- the engine

class LiveEngine:
    """Everything one dataset needs to price and audit a plan, loaded once.

    Built behind :func:`get_engine` so Streamlit shares a single instance across
    reruns and sessions.  Holds the black box, the fitted SCM, the cost model and
    the dataset bundle, plus the schema-derived vectors (bounds, discreteness,
    admissibility) the sandbox's editors read.
    """

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.bundle = load_dataset(dataset)
        self.model = load_blackbox(dataset, CONFIG.BLACKBOX_TYPE)
        self.spec = self.bundle.spec
        self.transformer = self.bundle.transformer
        self.cost_model = get_cost_model(dataset)
        self.scm = get_scm(dataset)
        self._fcm = None  # lazily loaded: a dataset may have no SCM

        self.feature_names: list[str] = list(self.bundle.feature_names)
        self.d = len(self.feature_names)
        self.index = {n: j for j, n in enumerate(self.feature_names)}
        # Bounds/span from the cost model (schema bounds), matching the evaluator
        # so a live gap equals the one the results CSV reports for the same plan.
        self.bounds = np.asarray(self.cost_model.bounds, dtype=float)
        self.span = np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-12)
        self.is_discrete = np.asarray(self.cost_model.is_discrete, dtype=bool)
        self.admissible = np.asarray(self.cost_model.admissible, dtype=bool)
        self.monotone = tuple(self.cost_model.monotone)
        self.sensitive = set(self.scm.sensitive)
        self.max_actions = int(self.cost_model.max_actions)

    # ---------------------------------------------------------------- causal
    @property
    def fcm(self):
        """The fitted structural equations, or ``None`` when the dataset has none."""
        if self._fcm is None and has_causal(self.dataset):
            self._fcm = get_fcm(self.dataset)
        return self._fcm

    @property
    def has_causal(self) -> bool:
        return self.fcm is not None

    # ----------------------------------------------------------------- records
    def factual_row(self, record: int) -> np.ndarray:
        """The encoded factual vector of one test record (the query-set index)."""
        return np.asarray(self.bundle.X_test[int(record)], dtype=float).ravel()

    def n_test(self) -> int:
        return int(self.bundle.X_test.shape[0])

    # ------------------------------------------------------------- black box
    def proba_positive(self, x: np.ndarray) -> float:
        """``P(y = 1)`` for one encoded row."""
        row = np.asarray(x, dtype=float).reshape(1, -1)
        return float(self.model.predict_proba(row)[0, 1])

    def predict_label(self, x: np.ndarray) -> int:
        return int(self.proba_positive(x) >= float(CONFIG.DECISION_THRESHOLD))

    def target_proba(self, x: np.ndarray, y_target: int) -> float:
        p = self.proba_positive(x)
        return p if int(y_target) == FAVOURABLE else 1.0 - p

    def default_target(self, x: np.ndarray) -> int:
        """The class recourse is about: the flip of the model's current answer."""
        return int(1 - self.predict_label(x))

    # ------------------------------------------------------------------ causal
    def repair(self, x: np.ndarray, x_edit: np.ndarray) -> np.ndarray:
        """The world the plan's own changes imply, per the fitted SCM.

        Returns ``x_edit`` unchanged when the dataset has no SCM, so callers can
        stay branch-free; :attr:`has_causal` says whether the number is real.
        """
        x = np.asarray(x, dtype=float).reshape(1, -1)
        x_edit = np.asarray(x_edit, dtype=float).reshape(1, -1)
        if not self.has_causal:
            return x_edit.ravel()
        return np.asarray(self.fcm.repair(x, x_edit), dtype=float).ravel()

    # ------------------------------------------------------------------- audit
    def audit(self, x: np.ndarray, x_edit: np.ndarray,
              y_target: int | None = None,
              asked: Iterable[str] | None = None) -> Audit:
        """The three audits (F/S/A) and their conjunction for one plan.

        ``x_edit`` is the full record the plan *declares* --- the point the black
        box is asked about (F) and the SCM is asked to repair (S).  ``asked`` names
        the features that are genuine *actions*; only those are priced (A), because
        a consequence the SCM fills in is not a request and charging one would make
        a method look infeasible for being honest about its fallout.  Pass ``asked``
        when auditing a method's declared plan (see :meth:`audit_plan`); leave it
        ``None`` for a free-typed plan, where every edited feature *is* an action.

        This mirrors the pipeline exactly (:mod:`src.evaluation.faithfulness` scores
        validity/gap/survival on the declared ``x_cf``; :mod:`src.evaluation.feasibility`
        prices the asked subset ``np.where(acted, x_cf, x)``), so ``usable`` here and
        the paper's headline agree by construction.
        """
        x = np.asarray(x, dtype=float).ravel()
        x_edit = np.asarray(x_edit, dtype=float).ravel()
        y_original = self.predict_label(x)
        y_target = self.default_target(x) if y_target is None else int(y_target)

        # Which coordinates count as *actions*.  A free-typed plan (``asked`` None)
        # treats every edited feature as one; a method's declared plan names its
        # intervention set and leaves the rest to the SCM.
        if asked is None:
            acted = np.abs(x_edit - x) > _ATOL
        else:
            wanted = set(asked)
            acted = np.array([n in wanted for n in self.feature_names], dtype=bool)

        # F --- flips the box, at the full point the plan declares.
        proba_target = self.target_proba(x_edit, y_target)
        valid = bool(proba_target >= float(CONFIG.DECISION_THRESHOLD))

        # A --- a person could do it (priced on the asked move only; consequences
        # the SCM fills in are not charged as if they were requests).
        breakdown = self.cost_model.breakdown(x, np.where(acted, x_edit, x))
        over_actions = bool(breakdown.n_changed > self.max_actions)
        over_budget = bool(breakdown.normalised > 1.0)
        admissible = bool(breakdown.feasible)
        actionable = bool(admissible and not over_actions and not over_budget)

        # S --- survives the SCM.
        tol = float(CONFIG.CAUSAL_TOLERANCE)
        if self.has_causal:
            repaired = self.repair(x, x_edit)
            gaps = np.abs(repaired - x_edit) / self.span
            gap_max = float(gaps.max())
            off = tuple(self.feature_names[j]
                        for j in np.flatnonzero(gaps > tol))
            drifted = tuple(self.feature_names[j]
                            for j in np.flatnonzero(np.abs(repaired - x_edit)
                                                    > _ATOL))
            proba_repaired = self.target_proba(repaired, y_target)
            flip_survives = bool(proba_repaired >= float(CONFIG.DECISION_THRESHOLD))
            causal_valid = bool(gap_max <= tol)
        else:
            repaired = x_edit.copy()
            gap_max, off, drifted = float("nan"), (), ()
            proba_repaired = proba_target
            flip_survives = valid
            causal_valid = True  # no SCM to contradict the plan

        changed = tuple(breakdown.changed)
        promise_kept = bool(valid and flip_survives and causal_valid)
        usable = bool(valid and promise_kept and actionable)
        return Audit(
            dataset=self.dataset, x=x, x_edit=x_edit, repaired=repaired,
            y_original=y_original, y_target=y_target,
            proba_target=proba_target, valid=valid,
            gap_max=gap_max, causal_valid=causal_valid, off_features=off,
            drifted=drifted, proba_repaired_target=proba_repaired,
            flip_survives=flip_survives,
            promise_kept=promise_kept,
            n_changed=int(breakdown.n_changed), normalised=float(breakdown.normalised),
            raw_cost=float(breakdown.raw), cost=float(breakdown.total),
            feasible=admissible, over_actions=over_actions, over_budget=over_budget,
            actionable=actionable,
            violations=tuple(breakdown.violations), changed_features=changed,
            usable=usable, breakdown=breakdown, has_causal=self.has_causal,
        )

    def audit_plan(self, x: np.ndarray, cf: Counterfactual,
                   y_target: int | None = None) -> Audit:
        """Audit a method's *declared* plan exactly as the pipeline scores it.

        The black box and the SCM see the method's full ``x_cf`` (endpoint plus
        the consequences it committed to), while the cost model prices only the
        features it named as its intervention (``cf.changed_features``).  Feeding
        the whole ``x_cf`` to a free-typed :meth:`audit` would instead charge the
        consequences as actions and measure the gap from a consequence-stripped
        point, so CAC --- the only method that declares its fallout --- would look
        both over-budget and causally invalid.  Loading a plan therefore goes
        through here, not through :func:`seed_from_plan`.
        """
        x_cf = np.asarray(cf.x_cf, dtype=float).ravel()
        target = int(cf.y_target) if y_target is None else int(y_target)
        return self.audit(x, x_cf, y_target=target, asked=cf.changed_features)

    # ------------------------------------------------------------------ reading
    def decode(self, x: np.ndarray) -> pd.Series:
        """One encoded row as a semantic (human-readable) series."""
        frame = self.transformer.decode(np.asarray(x, dtype=float).reshape(1, -1))
        return frame.iloc[0]

    def pretty(self, feature: str, value: float) -> str:
        """Encoded value -> the string a person would recognise (units/levels)."""
        return self.cost_model.pretty(feature, float(value))

    def levels(self, feature: str) -> tuple[str, ...] | None:
        levels = self.spec.get(feature).levels
        return tuple(levels) if levels else None

    def is_immutable(self, feature: str) -> bool:
        return not bool(self.admissible[self.index[feature]])

    def feature_meta(self, x: np.ndarray) -> list[dict[str, Any]]:
        """Per-feature descriptors the sandbox's editors need, factual value first."""
        x = np.asarray(x, dtype=float).ravel()
        meta: list[dict[str, Any]] = []
        for j, name in enumerate(self.feature_names):
            feat = self.spec.get(name)
            meta.append({
                "name": name,
                "kind": feat.kind,
                "levels": self.levels(name),
                "lo": float(self.bounds[j, 0]),
                "hi": float(self.bounds[j, 1]),
                "integer": bool(feat.integer or feat.is_discrete),
                "unit": feat.unit,
                "monotone": self.monotone[j],
                "immutable": not bool(self.admissible[j]),
                "sensitive": name in self.sensitive,
                "factual": float(x[j]),
                "pretty": self.pretty(name, float(x[j])),
            })
        return meta

    # -------------------------------------------------------- display frames
    def plan_frame(self, audit: Audit) -> pd.DataFrame:
        """What the plan *asks* for: one row per edited feature, priced."""
        rows: list[dict[str, Any]] = []
        for a in audit.breakdown.actions:
            rows.append({
                "feature": a.feature,
                "from": self.pretty(a.feature, a.x_old),
                "to": self.pretty(a.feature, a.x_new),
                "effort": round(float(a.raw), 3),
                "issue": a.violation or "",
                "sensitive": a.feature in self.sensitive,
            })
        return pd.DataFrame(rows, columns=["feature", "from", "to", "effort",
                                           "issue", "sensitive"])

    def consequence_frame(self, audit: Audit) -> pd.DataFrame:
        """What the SCM makes *follow*: features the world moves that you did not.

        Empty when the plan is causally consistent (or the dataset has no SCM).
        A non-empty ``breaks causal`` column is the heart of the thesis: the plan
        quietly assumed those features would stay put, and the graph says they do
        not.
        """
        if not audit.has_causal:
            return pd.DataFrame(columns=["feature", "you asked", "world result",
                                         "breaks causal"])
        asked = set(audit.changed_features)
        rows: list[dict[str, Any]] = []
        off = set(audit.off_features)
        for name in audit.drifted:
            if name in asked:
                continue  # an asked feature is an action, not a consequence
            j = self.index[name]
            rows.append({
                "feature": name,
                "you asked": self.pretty(name, audit.x_edit[j]),
                "world result": self.pretty(name, audit.repaired[j]),
                "breaks causal": name in off,
            })
        return pd.DataFrame(rows, columns=["feature", "you asked", "world result",
                                           "breaks causal"])


@dataclass(slots=True)
class Audit:
    """The full verdict on one typed plan --- every number recomputed live."""

    dataset: str
    x: np.ndarray
    x_edit: np.ndarray
    repaired: np.ndarray
    y_original: int
    y_target: int
    # F --- flips the box
    proba_target: float
    valid: bool
    # S --- survives the SCM
    gap_max: float
    causal_valid: bool
    off_features: tuple[str, ...]
    drifted: tuple[str, ...]
    proba_repaired_target: float
    flip_survives: bool
    promise_kept: bool
    # A --- a person could do it
    n_changed: int
    normalised: float
    raw_cost: float
    cost: float
    feasible: bool
    over_actions: bool
    over_budget: bool
    actionable: bool
    violations: tuple[tuple[str, str], ...]
    changed_features: tuple[str, ...]
    # conjunction + provenance
    usable: bool
    breakdown: Any
    has_causal: bool

    def as_dict(self) -> dict[str, Any]:
        """A flat, JSON-ready record of the verdict (for the download button)."""
        return {
            "dataset": self.dataset,
            "y_original": int(self.y_original),
            "y_target": int(self.y_target),
            "flips_the_box": bool(self.valid),
            "proba_target": round(float(self.proba_target), 4),
            "survives_scm": bool(self.causal_valid),
            "causal_gap": None if np.isnan(self.gap_max) else round(float(self.gap_max), 4),
            "flip_survives_repair": bool(self.flip_survives),
            "off_features": list(self.off_features),
            "actionable": bool(self.actionable),
            "n_changed": int(self.n_changed),
            "max_actions": int(CONFIG.MAX_ACTIONS),
            "normalised_cost": round(float(self.normalised), 4),
            "over_budget": bool(self.over_budget),
            "violations": [list(v) for v in self.violations],
            "changed_features": list(self.changed_features),
            "usable": bool(self.usable),
        }


# --------------------------------------------------------- cached constructors

@st.cache_resource(show_spinner=False)
def get_engine(dataset: str) -> LiveEngine:
    """Process-wide singleton engine for ``dataset`` (models load once)."""
    return LiveEngine(dataset)


@st.cache_resource(show_spinner=False)
def _method(dataset: str, name: str):
    """A counterfactual method wired to the engine's shared artefacts.

    Passing the loaded model/spec/SCM/cost model avoids reloading the black box
    from disk on every ``generate`` (``load_blackbox`` is not itself cached).
    """
    eng = get_engine(dataset)
    return get_method(name, dataset, model=eng.model, spec=eng.spec,
                      scm=eng.scm, cost_model=eng.cost_model)


@st.cache_resource(show_spinner=False)
def live_methods(dataset: str) -> list[str]:
    """Methods that actually import and build for ``dataset`` (optional deps drop)."""
    ok: list[str] = []
    for name in METHOD_ORDER:
        try:
            _method(dataset, name)
            ok.append(name)
        except Exception:  # noqa: BLE001 - a missing optional dep drops the method
            continue
    return ok


@st.cache_data(show_spinner=False)
def rejected_records(dataset: str, n: int) -> list[int]:
    """Indices into ``X_test`` the black box currently rejects (the query set)."""
    from src.evaluation.metrics import query_instances  # noqa: PLC0415

    eng = get_engine(dataset)
    idx = query_instances(dataset, model=eng.model, bundle=eng.bundle, n=int(n),
                          y_target=FAVOURABLE)
    return [int(i) for i in idx]


@st.cache_data(show_spinner=False)
def generate_plans(dataset: str, method: str, record: int,
                   k: int) -> list[Counterfactual]:
    """Up to ``k`` plans from one method for one record, cached and fail-soft."""
    eng = get_engine(dataset)
    x = eng.factual_row(record)
    y_target = eng.default_target(x)
    try:
        model = _method(dataset, method)
        return list(model.generate(x, y_target=y_target, k=int(k)))
    except Exception:  # noqa: BLE001 - a method that crashes produces no plan
        return []


def seed_from_plan(engine: LiveEngine, x: np.ndarray,
                   cf: Counterfactual) -> np.ndarray:
    """Factual row with only a plan's *asked* features overwritten.

    Seeds the sandbox's editors from a method's recommendation without dragging in
    the SCM consequences the method's ``x_cf`` already baked in --- the editors show
    the actions, and :meth:`LiveEngine.audit` (called with ``asked=None``) then
    re-derives the fallout, so the user sees the *action-only* point a non-causal
    tool would show them and the consequences it hides.  To score the method's
    plan the way the paper does --- endpoint and all --- use
    :meth:`LiveEngine.audit_plan`, not this.
    """
    edit = np.asarray(x, dtype=float).ravel().copy()
    x_cf = np.asarray(cf.x_cf, dtype=float).ravel()
    for name in cf.changed_features:
        j = engine.index.get(name)
        if j is not None:
            edit[j] = x_cf[j]
    return edit


@st.cache_data(show_spinner=False)
def cohort_headline(dataset: str, n: int, k: int,
                    methods: tuple[str, ...]) -> pd.DataFrame:
    """Live fallback for the gap chart: audit ``n`` rejected records per method.

    Only invoked when ``results/headline.csv`` is absent (see
    :func:`app.components.headline_source`).  Deliberately bounded by ``n`` and
    ``k``; the CSV is the fast path.  Rates are per-record (a record counts if any
    of its ``k`` plans passes) and each plan is scored by :meth:`LiveEngine.audit_plan`
    --- the declared endpoint, priced on its asked subset --- so the columns are the
    ones :mod:`src.evaluation.feasibility` computes: ``causal_validity`` is
    ``valid & promise_kept`` and ``usable`` is ``valid & promise_kept & actionable``.
    """
    eng = get_engine(dataset)
    records = rejected_records(dataset, n)
    rows: list[dict[str, Any]] = []
    for method in methods:
        flags = {"validity": [], "causal_validity": [], "actionability": [],
                 "usable": [], "cost": []}
        for record in records:
            x = eng.factual_row(record)
            plans = generate_plans(dataset, method, record, k)
            audits = [eng.audit_plan(x, cf) for cf in plans]
            v = any(a.valid for a in audits)
            c = any(a.valid and a.promise_kept for a in audits)
            act = any(a.valid and a.actionable for a in audits)
            usable = [a for a in audits if a.usable]
            flags["validity"].append(v)
            flags["causal_validity"].append(c)
            flags["actionability"].append(act)
            flags["usable"].append(bool(usable))
            if usable:
                flags["cost"].append(min(a.cost for a in usable))
        n_rec = max(len(records), 1)
        rows.append({
            "dataset": dataset, "method": method, "n_records": len(records),
            "validity": sum(flags["validity"]) / n_rec,
            "causal_validity": sum(flags["causal_validity"]) / n_rec,
            "actionability": sum(flags["actionability"]) / n_rec,
            "usable": sum(flags["usable"]) / n_rec,
            "cost_mean": float(np.mean(flags["cost"])) if flags["cost"] else float("nan"),
        })
    return pd.DataFrame(rows)


__all__ = [
    "FAVOURABLE",
    "METHOD_ORDER",
    "Audit",
    "LiveEngine",
    "datasets_live",
    "has_causal",
    "get_engine",
    "live_methods",
    "rejected_records",
    "generate_plans",
    "seed_from_plan",
    "cohort_headline",
]
