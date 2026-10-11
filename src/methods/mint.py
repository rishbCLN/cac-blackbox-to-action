"""MINT — Minimal-cost causal INTervention.  The causal baseline CAC must beat.

CAC's cost story needs an opponent.  The oracle is cost-optimal but *acausal* —
it moves a point and never asks what the SCM would do next — so a low Cost/Oracle
ratio says only that CAC is close to the cheapest *point*, not the cheapest
*intervention*.  Karimi et al. (FAccT 2021) already optimise interventions
through an SCM, and until a method does that here CAC's "causal recourse is
affordable" claim is uncontested.  MINT is that method, reduced to its bones.

MINT searches the same intervention space CAC does — a set of at most
``MAX_ACTIONS`` hard interventions :math:`do(X_S = v_S)` whose consequences the
SCM recomputes (via :class:`~src.scm.fcm.FCM`, Pearl's three steps) — and keeps
the cheapest set whose *resulting* record the black box accepts.  That is the
whole objective: validity at the decision threshold, and action cost, nothing
else.  No plausibility prior, no soft feasibility term, no side-effect penalty,
no acceptance margin.  Those four are exactly what CAC adds, so the gap between
MINT and CAC on cost is the price of everything CAC does beyond flipping the box
causally — which is a number the paper should state rather than leave to an
acausal oracle to stand in for.

The search is the oracle's, inherited wholesale and for a reason: a weak
baseline would make Finding 5 worse, not better.  Costs are the sum of §7.2's
per-feature efforts over the *acted* features, so action plans can be enumerated
best-first in non-decreasing cost with the same heap, the same cost-ordered
pruning once ``k`` flips are known, and the same evaluation cap the oracle uses
— MINT therefore spends a query budget matched to the oracle's by construction.
The one thing MINT changes is what gets scored: where the oracle queries the
black box on the point ``x`` with its moves applied, MINT first propagates those
moves through the fitted structural equations and queries the box on the record
the SCM says would *result*.  Because the propagation reuses a single abduction
of ``x`` (the exogenous terms do not depend on the action), batching a whole
frontier costs one ``counterfactual`` call and one query, exactly as for CAC.

Scored and reported identically to CAC: :meth:`_build` prices the **action**
(the intervened coordinates, via the shared cost model), not the resulting
state, and records ``causal_gap`` from the same :meth:`FCM.violation` the audit
uses — zero by construction here, because MINT proposes only records its own SCM
produces.  ``causal_path`` comes from the inherited helper, since ``causal`` is
true.  The difference from CAC is the objective, not the accounting, which is
the only way a Cost column comparing the two can mean anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.methods.base import (
    DEDUP_DECIMALS,
    MOVE_ATOL,
    Counterfactual,
    CounterfactualMethod,
    register,
)
from src.methods.oracle import OracleMethod
from src.scm.fcm import FCM, get_fcm
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@register
class MINTMethod(OracleMethod):
    """Cheapest causal intervention that flips the box, scored through the SCM.

    Shares the oracle's cost-ordered best-first enumeration (so the two cost
    references are directly comparable and the query budget is matched) but
    scores the record the SCM *produces* from each candidate action rather than
    the point the action lands on.  Everything the oracle promises about
    optimality on the grid therefore holds for MINT too — the cheapest action
    set, on the grid and under the action budget, whose causal consequence the
    black box accepts — which is what makes it a fair opponent for CAC on cost.
    """

    name = "mint"
    causal = True

    def __init__(self, dataset: str, fcm: FCM | None = None,
                 label: str | None = None, **kwargs: Any) -> None:
        """Mirror CAC's constructor: the SCM is MINT's one extra dependency.

        ``fcm`` is injectable (the de-circularity experiments hand MINT a
        held-out FCM to generate under); left ``None`` it is the dataset's own
        fitted equations, loaded on first use.  ``label`` renames the rows an
        ablation or a cross-fit run emits, the way :class:`CACMethod` allows.
        """
        super().__init__(dataset, **kwargs)
        #: §7.2's action budget, read from the injected cost model so a run with
        #: a different ceiling moves the search, the denominator and the audit
        #: together — the same source the oracle and CAC read it from.
        self.max_actions = int(self.cost_model.max_actions)
        if label:  # shadows the class attribute, so every row it emits says so
            self.name = str(label)
        self._fcm = fcm
        #: Resulting-row key -> the intervention mask that produced it, so
        #: :meth:`_build` prices the action rather than its consequences.
        self._plans: dict[tuple[float, ...], np.ndarray] = {}
        #: Set per record by :meth:`_propose`, read by :meth:`_flips`.
        self._x: np.ndarray = np.empty(0)
        self._noise: dict[str, np.ndarray] = {}

    @property
    def fcm(self) -> FCM:
        """The fitted structural equations — MINT's one dependency the baselines lack."""
        if self._fcm is None:
            self._fcm = get_fcm(self.dataset)
        return self._fcm

    # ------------------------------------------------------------------ search
    def _propose(self, x: np.ndarray, y_target: int, k: int) -> np.ndarray:
        """Abduct once, then let the oracle's enumeration drive :meth:`_flips`."""
        self._plans.clear()  # last record's plans are no longer anybody's actions
        self._x = np.asarray(x, dtype=float).ravel()
        # One abduction per record: the exogenous terms do not depend on the
        # action, so every candidate in the frontier reuses them (as in CAC).
        self._noise = self.fcm.abduct(self._x.reshape(1, -1))
        return super()._propose(self._x, y_target, k)

    def _flips(self, rows: list[np.ndarray], costs: list[float],
               y_target: int) -> list[tuple[float, np.ndarray]]:
        """Propagate one frontier through the SCM, keep the plans the box accepts.

        The oracle hands this method the *point* rows its moves land on; MINT
        reads each as the intervention it implies (every acted coordinate, since
        the enumeration never proposes a move that leaves a feature where it
        was), propagates it, and scores the resulting record.  The returned cost
        is still the oracle's action cost, so the heap's pruning ceiling and the
        ``k`` cheapest selection remain by *action* cost — the point of a
        minimal-cost baseline.
        """
        point = np.asarray(rows, dtype=float)
        mask = ~np.isclose(point, self._x, rtol=0.0, atol=MOVE_ATOL)
        resulting = self.snap(
            self.fcm.counterfactual(self._x, point, mask, noise=self._noise))
        _, flipped = self.score(resulting, y_target)
        out: list[tuple[float, np.ndarray]] = []
        for i in np.flatnonzero(flipped):
            key = tuple(np.round(resulting[i], DEDUP_DECIMALS))
            # Cheapest action wins a tie: the frontier arrives in cost order, so
            # the first mask to reach a resulting row is the cheapest route to it.
            self._plans.setdefault(key, mask[i].copy())
            out.append((float(costs[i]), resulting[i]))
        return out

    def _report(self, found: list[tuple[float, np.ndarray]],
                max_actions: int) -> np.ndarray:
        """The oracle's reporting, logged under MINT's name and causal framing."""
        verdict = "proven on the grid" if self._proven else "unproven (eval cap)"
        if not found:
            LOGGER.info("mint/%s: no causal intervention of at most %d action(s) "
                        "on the grid flips the prediction (%d evaluated, %s)",
                        self.dataset, max_actions, self._n_evals, verdict)
            return np.empty((0, self.d))
        LOGGER.info("mint/%s: cheapest of %d interventions evaluated costs %.4f, "
                    "%d returned, optimality %s", self.dataset, self._n_evals,
                    found[0][0] / float(self.cost_model.budget), len(found), verdict)
        return np.asarray([row for _, row in found], dtype=float)

    # ------------------------------------------------------------------ output
    def _build(self, x: np.ndarray, x_cf: np.ndarray, y_original: int,
               y_target: int, proba_target: float) -> Counterfactual:
        """Price the action, report the causal gap — CAC's accounting, no extras.

        Deliberately skips :class:`OracleMethod`'s ``_build`` (which stamps an
        acausal ``proven_optimal``) and goes straight to the base auditor, then
        re-prices on the intervention the way :class:`CACMethod` does.  ``cost``,
        ``raw_cost``, ``changed_features``, ``feasible`` and ``violations`` all
        describe the *action* a person is asked to take, not the state the SCM
        drags along behind it, so MINT's Cost column and CAC's are the same
        measurement.  What MINT does not report is everything CAC adds —
        plausibility, side effects, the four objective terms — because not having
        them is the baseline.
        """
        # Grandparent, not OracleMethod._build: MINT is not an acausal reference.
        cf = CounterfactualMethod._build(self, x, x_cf, y_original, y_target,
                                         proba_target)
        gap = float(np.max(self.fcm.violation(x.reshape(1, -1), x_cf.reshape(1, -1))))
        found = self._plans.get(tuple(np.round(np.asarray(x_cf, dtype=float),
                                               DEDUP_DECIMALS)))
        if found is None:  # audited from outside the search: price the whole move
            return cf.with_extras(causal_gap=gap)
        actions = np.where(found, x_cf, x)
        breakdown = self.cost_model.breakdown(x, actions)
        return cf.with_extras(
            cost=float(breakdown.total),
            raw_cost=float(breakdown.raw),
            changed_features=tuple(breakdown.changed),
            feasible=bool(breakdown.feasible),
            violations=tuple(breakdown.violations),
            explanation=tuple(self.cost_model.explain(x, actions)),
            causal_gap=gap,
            n_actions=int(breakdown.n_changed),
        )


__all__ = ["MINTMethod"]
