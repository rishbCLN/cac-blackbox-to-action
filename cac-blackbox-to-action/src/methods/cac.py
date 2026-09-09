"""CAC — Causal Actionable Counterfactuals (§8.4).  The paper's contribution.

Every baseline here searches for a *point* ``x'`` that the black box labels
differently.  Nothing in that search asks whether a person could reach the
point, and nothing asks whether the rest of the record still makes sense once
part of it has moved.  §9 shows what that licenses: DiCE asking for $28,377 of
capital gains, Wachter offering more capital *losses* as a route to a higher
income, the random baseline flipping the model by working four hours *less*.
Each is a valid counterfactual.  None is recourse.

CAC searches for an **action**, not a point.  A candidate is a set of at most
``MAX_ACTIONS`` interventions :math:`do(X_S = v_S)`, and the record it is scored
on is the one the SCM says would *result* — every descendant recomputed by
abduction, action, prediction (§6, via :class:`~src.scm.fcm.FCM`).  Causal
validity is therefore a property of the representation rather than a penalty
term that can be traded away, which is the whole design: CAC cannot propose a
world that its own SCM disowns, however much the black box would like it.

Four terms, minimised jointly, weighted by the ``CAC_LAMBDA_*`` knobs of §3:

*Validity* — a hinge on ``P(target)`` that reaches zero once the resulting
record clears the decision threshold by ``CAC_TARGET_MARGIN``.  The margin is
there because a counterfactual sitting at ``p = 0.5001`` is not recourse, it is
a coin flip.

*Cost* — §7's normalised cost of the **actions**, not of the resulting state.
Somebody pays to go back to college; they do not additionally pay for the
promotion that follows.  For a baseline the two are identical — everything it
changed is something it asked for — so the rule is uniform across methods even
though only CAC can tell the difference.  Both numbers are recorded: ``cost`` is
the action, ``extras["state_cost"]`` what the whole resulting record would have
cost to bring about by hand.  The second is reported penalty-free, with any
inadmissible side effect named in ``extras["state_violations"]`` instead: §7's
admissibility rules govern what a person may *do*, and a credit score that moves
because they closed an account is a fact about the world, not a request.

*Feasibility* — §7's ``INFINITE_COST`` guard on the action set, plus any side
effect the SCM predicts on a *sensitive* feature nobody touched.  The first is
zero by construction and stays in as a guard rail; the second is a real cost
that no method ignoring the SCM can even observe.

*Plausibility* — how far the resulting record sits from people the data already
records in the target class, as a k-NN distance normalised by the median such
distance.  It keeps the search out of empty regions that happen to flip.

The search is a population, not a gradient descent, for three reasons: the black
box is opaque (rule #8), so there is no gradient to take; half the search space
is a subset choice, which has no derivative; and the discrete half of the SCM is
piecewise constant, so a finite-difference gradient through it is zero almost
everywhere and misleading where it is not.  ``CAC_LEARNING_RATE`` therefore sets
the mutation scale on continuous coordinates — the same knob doing the same job.

§8.4's pseudocode did not survive the instruction being truncated mid-sentence,
so the algorithm above is reconstructed from what the rest of the spec pins
down: the four ``CAC_LAMBDA_*`` weights, ``MAX_ACTIONS``, ``CAC_POPULATION``,
``CAC_MAX_ITERS``, ``CAC_LEARNING_RATE``, ``CAC_TARGET_MARGIN``, and rule #8.
The reconstruction is listed among the deviations in ``README.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

from src.config import CONFIG
from src.cost.cost_functions import INFINITE_COST
from src.methods.base import (
    DEDUP_DECIMALS,
    MOVE_ATOL,
    Counterfactual,
    CounterfactualMethod,
    register,
)
from src.scm.fcm import FCM, get_fcm
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class _Terms(NamedTuple):
    """The objective and everything it is made of, one entry per candidate."""

    total: np.ndarray
    validity: np.ndarray
    cost: np.ndarray
    feasibility: np.ndarray
    plausibility: np.ndarray
    p_target: np.ndarray
    flipped: np.ndarray


#: One archived plan: ``(objective, resulting row, intervention mask, term values)``.
#: The mask is what lets :meth:`CACMethod._build` price the action rather than the
#: state, and the term values let §9 attribute a choice to the term that drove it.
_Entry = tuple[float, np.ndarray, np.ndarray, dict[str, float]]


@register
class CACMethod(CounterfactualMethod):
    """Cost-aware recourse searched in intervention space, scored through the SCM."""

    name = "cac"
    causal = True

    #: Candidate values per feature: training quantiles, so every proposal is a
    #: value other people in the data actually hold.
    GRID_POINTS = 12
    #: Generations without improvement before the search stops early.
    PATIENCE = 20
    #: Share of the population kept as parents.
    ELITE_FRAC = 0.5
    #: Probability that a child's *intervention set* mutates, not just its values.
    P_MASK_MUTATE = 0.35
    #: Probability that an intervened value is redrawn from the grid outright.
    P_RESAMPLE = 0.25
    #: Training rows sampled once as the plausibility reference.
    PLAUS_SAMPLE = 1_000
    #: A population smaller than this cannot support crossover.
    MIN_POPULATION = 8
    #: Improvement below this does not count as progress.
    TOL = 1e-6

    #: The four objective weights, and where each one's default comes from.  Named
    #: here so an ablation can zero one by key instead of by editing the config,
    #: which is frozen (§3) and shared with every other method in the run.
    LAMBDAS: tuple[str, ...] = ("valid", "cost", "feas", "plaus")

    def __init__(self, dataset: str, fcm: FCM | None = None,
                 lambdas: Mapping[str, float] | None = None,
                 propagate: bool = True, label: str | None = None,
                 **kwargs: Any) -> None:
        """``lambdas``, ``propagate`` and ``label`` exist for §9.3's ablation.

        The defaults are the paper's method: weights from :data:`CONFIG`, the SCM
        in the loop, the registered name.  Overriding them in the constructor
        rather than through the config is deliberate — ``CONFIG`` is frozen and
        global, so an ablation that reached in and changed a weight would change
        it for every method in the same process, and the run would silently stop
        being a comparison.  ``propagate=False`` is the one ablation that cannot
        be expressed as a weight: it scores the intervention as a *point*, the
        way a baseline does, leaving everything else about CAC intact, which is
        what isolates the claim that causal validity comes from the
        representation rather than from a penalty term.
        """
        super().__init__(dataset, **kwargs)
        self.lambdas = {
            "valid": float(CONFIG.CAC_LAMBDA_VALID),
            "cost": float(CONFIG.CAC_LAMBDA_COST),
            "feas": float(CONFIG.CAC_LAMBDA_FEAS),
            "plaus": float(CONFIG.CAC_LAMBDA_PLAUS),
        }
        for key, value in dict(lambdas or {}).items():
            if key not in self.lambdas:
                raise KeyError(f"unknown objective weight {key!r}; "
                               f"expected one of {self.LAMBDAS}")
            self.lambdas[key] = float(value)
        self.propagate = bool(propagate)
        #: §7.2's action budget, taken from the injected cost model rather than
        #: from :data:`CONFIG`, so that a run with a different budget moves the
        #: search ceiling, the cost denominator and the audit's "too many
        #: actions" rule together — the oracle and :mod:`src.evaluation` already
        #: read it from there, and a search that disagreed with the referee about
        #: how many actions are allowed would produce plans the audit rejects on
        #: a rule the search never saw.
        self.max_actions = int(self.cost_model.max_actions)
        if label:  # shadows the class attribute, so every row it emits says so
            self.name = str(label)
        self._fcm = fcm
        self._grids: dict[int, np.ndarray] = {}
        self._reference: dict[int, tuple[np.ndarray, float]] = {}
        #: Row key -> ``(intervention set, term values)`` that produced it, so
        #: :meth:`_build` can price the actions rather than their consequences
        #: and report which term the choice turned on.
        self._plans: dict[tuple[float, ...], tuple[np.ndarray, dict[str, float]]] = {}
        self._sensitive = np.array([n in set(self.scm.sensitive)
                                    for n in self.feature_names])
        self._generations = 0

    @property
    def fcm(self) -> FCM:
        """The fitted structural equations — CAC's one dependency the others lack."""
        if self._fcm is None:
            self._fcm = get_fcm(self.dataset)
        return self._fcm

    # ------------------------------------------------------------------- grids
    def _grid(self, j: int) -> np.ndarray:
        """Every value feature ``j`` may be set to, cached across records."""
        if j not in self._grids:
            feat = self.spec.features[j]
            lo, hi = self.bounds[j]
            if feat.levels:
                values = np.arange(round(lo), round(hi) + 1, dtype=float)
            else:
                column = self.X_train[:, j]
                values = np.concatenate([
                    np.quantile(column, np.linspace(0.0, 1.0, self.GRID_POINTS)),
                    [lo, hi],
                ])
            if self.is_discrete[j]:
                values = np.rint(values)
            self._grids[j] = np.unique(np.clip(values, self._lo[j], self._hi[j]))
        return self._grids[j]

    def _admissible(self, x: np.ndarray, j: int) -> np.ndarray:
        """Grid values feature ``j`` may be moved *to* from this record.

        The factual value is not a move and a monotone-forbidden direction is not
        an option, so neither ever enters the search — CAC spends its whole query
        budget on plans §7 would actually price.
        """
        grid = self._grid(j)
        keep = np.abs(grid - float(x[j])) > 1e-9
        direction = self.monotone[j]
        if direction == "increase":
            keep &= grid >= float(x[j])
        elif direction == "decrease":
            keep &= grid <= float(x[j])
        return grid[keep]

    # ----------------------------------------------------------- plausibility
    def _knn(self, X: np.ndarray, R: np.ndarray, drop_self: bool = False) -> np.ndarray:
        """Mean scaled-L1 distance from each row of ``X`` to its nearest in ``R``."""
        k = min(int(CONFIG.KNN_K) + int(drop_self), R.shape[0])
        D = (np.abs(X[:, None, :] - R[None, :, :]) / self.span).sum(axis=2)
        near = np.take_along_axis(D, np.argpartition(D, k - 1, axis=1)[:, :k], axis=1)
        near.sort(axis=1)
        return near[:, 1:].mean(axis=1) if drop_self and k > 1 else near.mean(axis=1)

    def _reference_set(self, y_target: int) -> tuple[np.ndarray, float]:
        """Target-class training rows, and the k-NN distance a *typical* one shows.

        Labelled by the data rather than by the black box on purpose.  A prior
        built from ``predict`` would spend tens of thousands of queries that §9
        would then have to attribute to whichever record happened to be first,
        and it would make the plausibility term depend on the very model it is
        supposed to constrain.
        """
        if y_target not in self._reference:
            keep = np.flatnonzero(self.y_train == int(y_target))
            if keep.size == 0:  # pragma: no cover - one-class training split
                keep = np.arange(self.X_train.shape[0])
            if keep.size > self.PLAUS_SAMPLE:
                keep = self.rng.choice(keep, size=self.PLAUS_SAMPLE, replace=False)
            R = self.X_train[np.sort(keep)]
            probe = R[self.rng.choice(R.shape[0], size=min(200, R.shape[0]),
                                      replace=False)]
            scale = float(max(np.median(self._knn(probe, R, drop_self=True)), 1e-9))
            self._reference[y_target] = (R, scale)
            LOGGER.debug("%s/%s: plausibility reference %d rows, median kNN %.4f",
                         self.name, self.dataset, R.shape[0], scale)
        return self._reference[y_target]

    def _plausibility(self, X: np.ndarray, y_target: int) -> np.ndarray:
        """``1`` for a record as ordinary as its neighbours, more for an outlier."""
        R, scale = self._reference_set(y_target)
        return self._knn(X, R) / scale

    def _side_effects(self, x: np.ndarray, rows: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
        """How far the SCM drags *sensitive* features nobody chose to touch.

        A plan whose consequence is a change of marital status is not the same
        offer as one that leaves it alone, even when both flip the model, and a
        method that ignores the SCM cannot tell the two apart.
        """
        watch = self._sensitive & ~mask
        if not watch.any():
            return np.zeros(rows.shape[0])
        return ((np.abs(rows - x) / self.span) * watch).sum(axis=1)

    # -------------------------------------------------------------- objective
    def _evaluate(self, x: np.ndarray, values: np.ndarray, mask: np.ndarray,
                  y_target: int, noise: dict[str, np.ndarray]
                  ) -> tuple[np.ndarray, _Terms]:
        """One generation: propagate through the SCM, then one black-box query.

        The whole population goes through :meth:`FCM.counterfactual` and
        ``predict_proba`` in a single call each, which is what keeps CAC's query
        count comparable to the baselines' despite the richer search space.

        With ``propagate=False`` (§9.3's ablation) the same population is scored
        as *points* — the intervened coordinates moved and nothing else — which
        is what every baseline does implicitly.  The search, the grids, the cost
        model and the action budget are untouched, so the difference between the
        two runs is the representation and nothing else.
        """
        if self.propagate:
            rows = self.snap(self.fcm.counterfactual(x, values, mask, noise=noise))
        else:
            rows = self.snap(np.where(mask, values, x))
        # What the applicant actually did: the intervened coordinates only.
        actions = np.where(mask, rows, x)
        cost = self.cost_model.batch(x, actions)
        p_target, flipped = self.score(rows, y_target)

        need = float(CONFIG.DECISION_THRESHOLD) if int(y_target) == 1 \
            else 1.0 - float(CONFIG.DECISION_THRESHOLD)
        need = min(need + float(CONFIG.CAC_TARGET_MARGIN), 1.0)
        validity = np.maximum(0.0, need - p_target) / need
        # `normalised` rather than `total`, so an inadmissible action is charged
        # once, under the feasibility weight, instead of twice at two weights.
        feasibility = np.minimum(cost["penalty"] / INFINITE_COST, 1.0) \
            + self._side_effects(x, rows, mask)
        plausibility = self._plausibility(rows, y_target)
        total = (self.lambdas["valid"] * validity
                 + self.lambdas["cost"] * cost["normalised"]
                 + self.lambdas["feas"] * feasibility
                 + self.lambdas["plaus"] * plausibility)
        return rows, _Terms(total, validity, cost["normalised"], feasibility,
                            plausibility, p_target, flipped)

    # ---------------------------------------------------------------- breeding
    def _seed(self, x: np.ndarray, n: int,
              pool: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Generation zero: every single action at its cheapest, then its dearest.

        Seeding the singletons means generation zero already contains the plans
        an oracle would try first, so the population never has to rediscover
        them; the random multi-feature plans are what give it somewhere else to
        go.  Nothing here is a query — the seeds are priced by §7, not scored.
        """
        values = np.repeat(np.asarray(x, dtype=float).reshape(1, -1), n, axis=0)
        mask = np.zeros((n, self.d), dtype=bool)
        row = 0
        for extreme in (0, -1):
            for j, grid in pool.items():
                if row >= n:
                    break
                order = np.argsort([
                    self.cost_model.feature_cost(self.feature_names[j],
                                                 float(x[j]), float(v)) for v in grid])
                values[row, j] = grid[order[extreme]]
                mask[row, j] = True
                row += 1
        slots = list(pool)
        top = min(self.max_actions, len(slots))
        while row < n:
            size = int(self.rng.integers(1, top + 1))
            for j in self.rng.choice(slots, size=size, replace=False):
                grid = pool[int(j)]
                values[row, int(j)] = grid[int(self.rng.integers(grid.size))]
                mask[row, int(j)] = True
            row += 1
        return values, mask

    def _children(self, values: np.ndarray, mask: np.ndarray, x: np.ndarray,
                  pool: dict[int, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
        """``n`` children of the elite: uniform crossover, jitter, set mutation."""
        rng = self.rng
        e = values.shape[0]
        a, b = rng.integers(0, e, size=n), rng.integers(0, e, size=n)
        take = rng.random((n, self.d)) < 0.5
        child_v = np.where(take, values[a], values[b])
        child_m = np.where(take, mask[a], mask[b])
        # Jitter is what CAC_LEARNING_RATE buys in a gradient-free search: a step
        # proportional to each feature's range, on the coordinates being acted on.
        jitter = rng.normal(0.0, float(CONFIG.CAC_LEARNING_RATE), size=(n, self.d))
        child_v = np.where(child_m, child_v + jitter * self.span, child_v)
        for i in range(n):
            self._mutate(child_v[i], child_m[i], pool, rng)
        return self._legalise(child_v, child_m, x, pool)

    def _mutate(self, v: np.ndarray, m: np.ndarray, pool: dict[int, np.ndarray],
                rng: np.random.Generator) -> None:
        """In place: redraw some values outright, and sometimes change *which*
        features the plan touches at all.

        Jitter alone can only refine a plan it already has.  Adding, dropping and
        swapping actions is how the search reaches a different *route* — which is
        the part of the space the baselines cannot represent, let alone explore.
        """
        for j in [int(j) for j in np.flatnonzero(m)]:
            if rng.random() < self.P_RESAMPLE:
                v[j] = pool[j][int(rng.integers(pool[j].size))]
        if rng.random() >= self.P_MASK_MUTATE:
            return
        on = [int(j) for j in np.flatnonzero(m)]
        off = [j for j in pool if j not in set(on)]
        roll = rng.random()
        if on and (roll < 0.35 or not off):
            m[on[int(rng.integers(len(on)))]] = False
            return
        if off and (roll < 0.70 or not on):
            j = off[int(rng.integers(len(off)))]
            m[j], v[j] = True, pool[j][int(rng.integers(pool[j].size))]
            return
        if on and off:  # swap one action for another
            m[on[int(rng.integers(len(on)))]] = False
            j = off[int(rng.integers(len(off)))]
            m[j], v[j] = True, pool[j][int(rng.integers(pool[j].size))]

    def _legalise(self, values: np.ndarray, mask: np.ndarray, x: np.ndarray,
                  pool: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Force a candidate to be a legal plan before it costs a single query.

        Values land on the feature's own grid, features with nothing admissible
        to offer drop out, un-acted coordinates go back to their factual value,
        and an over-long plan loses actions at random until it fits inside
        the action budget — at random because dropping the dearest action would
        quietly bias the population toward cheap plans that do not work.
        """
        out_v = np.repeat(np.asarray(x, dtype=float).reshape(1, -1),
                          values.shape[0], axis=0)
        out_m = np.zeros_like(mask)
        for i in range(values.shape[0]):
            chosen = [int(j) for j in np.flatnonzero(mask[i]) if int(j) in pool]
            if len(chosen) > self.max_actions:
                chosen = [int(j) for j in self.rng.choice(
                    chosen, size=self.max_actions, replace=False)]
            for j in chosen:
                grid = pool[j]
                out_v[i, j] = grid[int(np.argmin(np.abs(grid - values[i, j])))]
                out_m[i, j] = True
        return out_v, out_m

    # ------------------------------------------------------------------ search
    def _propose(self, x: np.ndarray, y_target: int, k: int) -> np.ndarray:
        pool = {int(j): self._admissible(x, int(j)) for j in self.actionable}
        pool = {j: grid for j, grid in pool.items() if grid.size}
        self._generations = 0
        self._plans.clear()  # last record's plans are no longer anybody's actions
        if not pool:
            LOGGER.info("%s/%s: no admissible action exists for this record",
                        self.name, self.dataset)
            return np.empty((0, self.d))

        n = max(int(CONFIG.CAC_POPULATION), self.MIN_POPULATION)
        n_elite = max(2, int(round(n * self.ELITE_FRAC)))
        noise = self.fcm.abduct(np.asarray(x, dtype=float).reshape(1, -1))

        values, mask = self._seed(x, n, pool)
        rows, terms = self._evaluate(x, values, mask, y_target, noise)
        archive: dict[tuple[float, ...], _Entry] = {}
        self._remember(archive, rows, mask, terms)
        best, stale = float(terms.total.min()), 0

        for gen in range(1, int(CONFIG.CAC_MAX_ITERS) + 1):
            self._generations = gen
            elite = np.argsort(terms.total, kind="stable")[:n_elite]
            child_v, child_m = self._children(values[elite], mask[elite], x, pool,
                                              n - n_elite)
            child_rows, child_terms = self._evaluate(x, child_v, child_m, y_target,
                                                     noise)
            self._remember(archive, child_rows, child_m, child_terms)
            values = np.vstack([values[elite], child_v])
            mask = np.vstack([mask[elite], child_m])
            rows = np.vstack([rows[elite], child_rows])
            terms = _Terms(*(np.concatenate([getattr(terms, f)[elite],
                                             getattr(child_terms, f)])
                             for f in _Terms._fields))
            current = float(terms.total.min())
            if current < best - self.TOL:
                best, stale = current, 0
                continue
            stale += 1
            if stale >= self.PATIENCE:
                break

        return self._select(x, rows, mask, terms, archive, k)

    def _remember(self, archive: dict[tuple[float, ...], "_Entry"],
                  rows: np.ndarray, mask: np.ndarray, terms: _Terms) -> None:
        """Keep every candidate that flipped, best objective winning ties.

        The archive is what makes the answer independent of when the search
        happened to stop: a plan found in generation 3 and bred out of the
        population by generation 40 is still on the table.  Its four term values
        travel with it so §9 can attribute a choice to the term that drove it
        without re-running the search.
        """
        for i in np.flatnonzero(terms.flipped):
            key = tuple(np.round(rows[i], DEDUP_DECIMALS))
            prior = archive.get(key)
            total = float(terms.total[i])
            if prior is not None and prior[0] <= total:
                continue
            archive[key] = (total, rows[i].copy(), mask[i].copy(), {
                "objective": total,
                "term_validity": float(terms.validity[i]),
                "term_cost": float(terms.cost[i]),
                "term_feasibility": float(terms.feasibility[i]),
                "term_plausibility": float(terms.plausibility[i]),
            })

    def _select(self, x: np.ndarray, rows: np.ndarray, mask: np.ndarray,
                terms: _Terms,
                archive: dict[tuple[float, ...], "_Entry"],
                k: int) -> np.ndarray:
        """The ``k`` plans to return, preferring genuinely different routes.

        Diversity here is over *intervention sets*, not over points.  Five ways
        to reach the same decision are only useful if they are five different
        things to do; five neighbouring values of one feature are one suggestion
        reported five times.
        """
        pairs: list[_Entry]
        if archive:
            pairs = sorted(archive.values(), key=lambda item: item[0])
        else:
            order = np.argsort(terms.total, kind="stable")[:k]
            pairs = [(float(terms.total[i]), rows[i], mask[i],
                      {"objective": float(terms.total[i]),
                       "term_validity": float(terms.validity[i]),
                       "term_cost": float(terms.cost[i]),
                       "term_feasibility": float(terms.feasibility[i]),
                       "term_plausibility": float(terms.plausibility[i])})
                     for i in order]
            LOGGER.info("%s/%s: %d generations, nothing flipped; returning the %d "
                        "best near-misses (best P(target)=%.3f)", self.name,
                        self.dataset, self._generations, len(pairs),
                        float(terms.p_target.max()))

        chosen: list[np.ndarray] = []
        routes: set[tuple[int, ...]] = set()
        for _, row, m, values in pairs:
            route = tuple(int(j) for j in np.flatnonzero(m))
            if route in routes:
                continue
            routes.add(route)
            self._plans[tuple(np.round(row, DEDUP_DECIMALS))] = (m, values)
            chosen.append(row)
            if len(chosen) >= k:
                break
        for _, row, m, values in pairs:  # top up when there are fewer routes than k
            if len(chosen) >= k:
                break
            if any(np.array_equal(row, taken) for taken in chosen):
                continue
            self._plans[tuple(np.round(row, DEDUP_DECIMALS))] = (m, values)
            chosen.append(row)
        if archive:
            LOGGER.info("%s/%s: %d generations, %d valid plans found over %d "
                        "routes, best objective %.4f", self.name, self.dataset,
                        self._generations, len(archive), len(routes), pairs[0][0])
        return np.asarray(chosen, dtype=float) if chosen else np.empty((0, self.d))

    # ------------------------------------------------------------------ output
    def _build(self, x: np.ndarray, x_cf: np.ndarray, y_original: int,
               y_target: int, proba_target: float) -> Counterfactual:
        """Price the actions; report the consequences; measure the causal gap.

        Every field that describes *the plan* is priced on the intervention —
        ``cost``, ``raw_cost``, ``changed_features``, ``feasible``,
        ``violations`` — because that is what a person is being asked to do and
        the only thing they can be held to.  Everything about the world that
        results is in ``extras``: ``state_cost`` is what the whole resulting
        record would have cost to bring about by hand, and ``state_violations``
        names the features the SCM moved that no one is allowed to move
        directly.

        That last list is not an error.  Closing revolving accounts *does* move
        an applicant's risk score, and a model of recourse that refuses to say
        so is the one being unrealistic; §7's admissibility rules are about
        actions, so applying them to consequences would price a fact about the
        world as if it were a request.  :meth:`CostModel.as_effort` is what draws
        that line — the same call §9 uses, so the state cost in the paper's
        tables and the state cost here cannot drift apart.  Both numbers stay
        finite and on one scale, so ``state_cost - cost`` is the SCM's own
        contribution in §7's units.
        """
        cf = super()._build(x, x_cf, y_original, y_target, proba_target)
        gap = float(np.max(self.fcm.violation(x.reshape(1, -1), x_cf.reshape(1, -1))))
        moved = ~np.isclose(x, x_cf, rtol=0.0, atol=MOVE_ATOL)
        state, side = self.cost_model.as_effort(x, x_cf)
        found = self._plans.get(tuple(np.round(np.asarray(x_cf, dtype=float),
                                               DEDUP_DECIMALS)))
        state_extras: dict[str, Any] = {
            "causal_gap": gap,
            "state_cost": float(state.total),
            "state_changed": tuple(name for name, m in zip(self.feature_names, moved)
                                   if m),
            "state_violations": side,
            "n_generations": int(self._generations),
        }
        if found is None:  # audited from outside the search: charge everything
            return cf.with_extras(**state_extras)
        mask, terms = found
        actions = np.where(mask, x_cf, x)
        breakdown = self.cost_model.breakdown(x, actions)
        return cf.with_extras(
            cost=float(breakdown.total),
            raw_cost=float(breakdown.raw),
            changed_features=tuple(breakdown.changed),
            feasible=bool(breakdown.feasible),
            violations=tuple(breakdown.violations),
            explanation=self._explain(x, x_cf, actions, mask),
            n_actions=int(breakdown.n_changed),
            **state_extras,
            **terms,
        )

    def _explain(self, x: np.ndarray, x_cf: np.ndarray, actions: np.ndarray,
                 mask: np.ndarray) -> tuple[str, ...]:
        """What to do, and then what follows — never the two in one list.

        §7's own explanation prices every feature that moved, which for a causal
        method would read as an instruction to arrange the consequences too.  The
        consequences are the SCM's side of the bargain, so they are reported and
        not charged.
        """
        lines = list(self.cost_model.explain(x, actions))
        follows = [j for j in range(self.d)
                   if not mask[j] and abs(float(x_cf[j]) - float(x[j])) > MOVE_ATOL]
        if follows:
            names = self.feature_names
            lines.append("...and the SCM then implies: " + ", ".join(
                f"{names[j]} {self.cost_model.pretty(names[j], float(x[j]))}"
                f" -> {self.cost_model.pretty(names[j], float(x_cf[j]))}"
                for j in follows))
        return tuple(lines)


__all__ = ["CACMethod"]
