"""The cost-optimal oracle — what the cheapest valid plan actually costs.

Not a method from the literature and not a competitor: a reference point.  For
one record it enumerates admissible action sets in increasing order of cost and
stops at the first one the black box accepts, so its answer is the cheapest
recourse that exists *on its grid* under §7's cost model.  Every other method's
cost can then be read as a ratio to something optimal instead of only relative
to the others — and when CAC costs more than the oracle, the gap is the price of
insisting on causal consistency, which is a number the paper should state rather
than hide.

Three properties make the enumeration honest:

* The plan cost is the sum of its per-feature moves, which is §7.2's definition
  of total cost, so costs can be pre-computed and sorted before a single query.
* Once ``k`` flips are known, every plan costing at least the dearest of them is
  pruned, and the search stops as soon as the frontier passes that ceiling.  It
  therefore *proves* its answer on the grid instead of sampling it.
* The grid for a numeric feature is the training quantiles, so the oracle's
  suggestions are values other people in the data actually hold.  An oracle
  free to invent unprecedented values would be an unfair upper bound.

"Optimal" is bounded three ways and the class says so rather than implying more:
plans touch at most ``MAX_ACTIONS`` features (§7.2's budget, the same ceiling
every other method works under), values come from the grid, and the search is
evaluation-capped — when the cap binds the result carries
``proven_optimal=False``.  A record where nothing on the grid flips returns
nothing at all, because a near-miss dressed up as the reference point would
corrupt every ratio computed against it.
"""

from __future__ import annotations

from heapq import heapify, heappop, heappush

import numpy as np

from src.methods.base import Counterfactual, CounterfactualMethod, register
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@register
class OracleMethod(CounterfactualMethod):
    """Cheapest valid plan by exhaustive cost-ordered enumeration."""

    name = "oracle"

    #: Grid points per numeric feature, taken as training quantiles.
    GRID_POINTS = 12
    #: Hard ceiling on rows scored per record, so one query cannot run away.
    MAX_EVALS = 200_000
    #: Rows per ``predict_proba`` call.
    CHUNK = 4_096
    #: Rows in the *first* call.  The batch grows from here to :attr:`CHUNK`:
    #: small batches early make the pruning ceiling arrive after a few dozen
    #: evaluations rather than a few thousand, and by the time batches are large
    #: the cheap plans have all failed anyway.
    MIN_BATCH = 64

    #: Set by :meth:`_propose`.  ``False`` only when :attr:`MAX_EVALS` cut the
    #: search short, in which case the answer is the best seen, not the best.
    _proven: bool = True
    #: Rows the last search actually scored, reported alongside the plan.
    _n_evals: int = 0

    # ------------------------------------------------------------------- grid
    def _values(self, x: np.ndarray, j: int) -> np.ndarray:
        """Candidate values for feature ``j``: snapped, in bounds, never the factual one."""
        lo, hi = self.bounds[j]
        feat = self.spec.features[j]
        if feat.levels:
            values = np.arange(round(lo), round(hi) + 1, dtype=float)
        else:
            column = self.X_train[:, j]
            values = np.concatenate([
                np.quantile(column, np.linspace(0.0, 1.0, self.GRID_POINTS)), [lo, hi],
            ])
        if self.is_discrete[j]:
            values = np.rint(values)
        values = np.unique(np.clip(values, self._lo[j], self._hi[j]))
        return values[np.abs(values - float(x[j])) > 1e-9]

    def _moves(self, x: np.ndarray) -> list[tuple[float, int, float]]:
        """Every single-feature action, as ``(weighted cost, feature, value)``.

        Monotone-forbidden directions and out-of-bounds values never enter the
        list, so the enumeration never spends a query on a plan §7 would price
        at ``INFINITE_COST``.
        """
        moves: list[tuple[float, int, float]] = []
        for j in self.actionable:
            name = self.feature_names[j]
            direction = self.monotone[j]
            for value in self._values(x, int(j)):
                if direction == "increase" and value < x[j]:
                    continue
                if direction == "decrease" and value > x[j]:
                    continue
                raw = self.cost_model.feature_cost(name, float(x[j]), float(value))
                if raw <= 0.0:
                    continue
                moves.append((raw * float(self.cost_model.weights[j]), int(j),
                              float(value)))
        moves.sort()
        return moves

    # ------------------------------------------------------------- enumeration
    def _children(self, plan: tuple[tuple[int, int], ...], cost: float,
                  options: list[list[tuple[float, float]]],
                  max_actions: int) -> list[tuple[float, tuple[tuple[int, int], ...]]]:
        """The plans reachable from ``plan``, none of them cheaper than it.

        A plan is a tuple of ``(slot, choice)`` pairs with strictly increasing
        slots, and it has exactly two kinds of successor: make its last step
        dearer, or append a step in a higher slot at that feature's cheapest
        value.  Every plan is therefore reached by exactly one path — its parent
        is itself minus the last step, or with that step made cheaper — which is
        what lets the heap pop plans in globally non-decreasing cost order with
        no visited set to maintain.
        """
        slot, choice = plan[-1]
        column = options[slot]
        out: list[tuple[float, tuple[tuple[int, int], ...]]] = []
        if choice + 1 < len(column):
            out.append((cost - column[choice][0] + column[choice + 1][0],
                        plan[:-1] + ((slot, choice + 1),)))
        if len(plan) < max_actions:
            out.extend((cost + options[nxt][0][0], plan + ((nxt, 0),))
                       for nxt in range(slot + 1, len(options)))
        return out

    @staticmethod
    def _row(x: np.ndarray, plan: tuple[tuple[int, int], ...],
             features: list[int],
             options: list[list[tuple[float, float]]]) -> np.ndarray:
        """A plan as an encoded record: ``x`` with each of its moves applied."""
        row = np.array(x, dtype=float)
        for slot, choice in plan:
            row[features[slot]] = options[slot][choice][1]
        return row

    def _flips(self, rows: list[np.ndarray], costs: list[float],
               y_target: int) -> list[tuple[float, np.ndarray]]:
        """Score one batch and keep the plans the black box accepts."""
        X = np.asarray(rows, dtype=float)
        _, flipped = self.score(X, y_target)
        return [(costs[i], X[i]) for i in np.flatnonzero(flipped)]

    # ------------------------------------------------------------------ search
    def _propose(self, x: np.ndarray, y_target: int, k: int) -> np.ndarray:
        """The cheapest ``k`` plans on the grid that the black box accepts."""
        self._proven, self._n_evals = True, 0
        k = max(int(k), 1)
        moves = self._moves(x)
        if not moves:
            LOGGER.info("oracle/%s: this record has no admissible move at all",
                        self.dataset)
            return np.empty((0, self.d))

        grouped: dict[int, list[tuple[float, float]]] = {}
        for weighted, j, value in moves:  # `moves` is cost-sorted, so each
            grouped.setdefault(j, []).append((weighted, value))  # column is too
        features = sorted(grouped)
        options = [grouped[j] for j in features]
        max_actions = min(int(self.cost_model.max_actions), len(features))

        heap = [(column[0][0], ((slot, 0),)) for slot, column in enumerate(options)]
        heapify(heap)

        found: list[tuple[float, np.ndarray]] = []  # (cost, row), cheapest first
        ceiling = float("inf")  # the k-th cheapest flip; nothing dearer can win
        rows: list[np.ndarray] = []
        costs: list[float] = []
        batch = self.MIN_BATCH

        while heap:
            cost, plan = heappop(heap)
            if cost >= ceiling:
                break  # the frontier is already dearer than the answer
            for child in self._children(plan, cost, options, max_actions):
                if child[0] < ceiling:
                    heappush(heap, child)
            costs.append(cost)
            rows.append(self._row(x, plan, features, options))
            room = self.MAX_EVALS - self._n_evals - len(rows)
            if len(rows) < batch and heap and room > 0:
                continue
            if room < 0:  # honour the cap to the row
                del rows[room:]
                del costs[room:]
            self._n_evals += len(rows)
            found.extend(self._flips(rows, costs, y_target))
            found.sort(key=lambda pair: pair[0])
            del found[k:]
            if len(found) >= k:
                ceiling = found[-1][0]
            rows, costs, batch = [], [], min(batch * 4, self.CHUNK)
            if room <= 0:
                self._proven = False
                break

        return self._report(found, max_actions)

    def _report(self, found: list[tuple[float, np.ndarray]],
                max_actions: int) -> np.ndarray:
        """Log what the search proved and hand back the plans as rows."""
        verdict = "proven on the grid" if self._proven else "unproven (eval cap)"
        if not found:
            LOGGER.info("oracle/%s: no plan of at most %d action(s) on the grid "
                        "flips the prediction (%d evaluated, %s)", self.dataset,
                        max_actions, self._n_evals, verdict)
            return np.empty((0, self.d))
        LOGGER.info("oracle/%s: cheapest of %d plans evaluated costs %.4f, "
                    "%d returned, optimality %s", self.dataset, self._n_evals,
                    found[0][0] / float(self.cost_model.budget), len(found),
                    verdict)
        return np.asarray([row for _, row in found], dtype=float)

    # ------------------------------------------------------------------ output
    def _build(self, x: np.ndarray, x_cf: np.ndarray, y_original: int,
               y_target: int, proba_target: float) -> Counterfactual:
        """The base record, plus the three numbers only the oracle can report."""
        cf = super()._build(x, x_cf, y_original, y_target, proba_target)
        return cf.with_extras(proven_optimal=bool(self._proven),
                              n_evaluated=int(self._n_evals),
                              grid_points=int(self.GRID_POINTS))


__all__ = ["OracleMethod"]

