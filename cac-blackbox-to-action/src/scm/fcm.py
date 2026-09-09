"""Fitted structural equations: the machinery that makes interventions real.

The SCM in :mod:`src.scm.define_scm` says *which* variables react to an action;
this module says *how much*.  Each non-root node gets an additive-noise
equation ``V = f_V(Pa(V)) + U_V`` (numeric / ordinal nodes) or a conditional
categorical distribution ``P(V | Pa(V))`` (categorical / binary nodes), fitted
on the **training split only** and in **encoded** space.

A counterfactual is then computed the textbook way (Pearl's three steps):

1. **abduction** — recover the exogenous term of every node from the factual
   row: ``u_V = v - f_V(pa)`` for numeric nodes, and the mid-quantile of the
   observed category under ``P(V | pa)`` for categorical ones;
2. **action** — overwrite the intervened nodes (a hard ``do``);
3. **prediction** — recompute every strict descendant in topological order,
   re-using the abducted exogenous terms.

Because the exogenous term is carried over, an intervention that changes nothing
upstream of a node leaves that node exactly where it was, which makes the whole
propagation an identity map on the factual row — the property the causal-validity
metric relies on.

Nothing here touches the black box: the equations are learned from data, so a
counterfactual method built on this module still only ever *queries* the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.config import CONFIG
from src.data.preprocess import Transformer, build_dataset, dataset_spec
from src.data.schema import DatasetSpec, FeatureSpec
from src.scm.define_scm import SCM, get_scm
from src.utils.io import load_joblib, save_joblib
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: numeric nodes are modelled with additive noise, discrete ones with a coupling
_ANM_KINDS = ("numeric", "ordinal")
_CATEGORICAL_KINDS = ("categorical", "binary")


def fcm_path(dataset: str) -> Path:
    """``models/fcm_<dataset>.joblib``."""
    return CONFIG.MODELS_DIR / f"fcm_{dataset}.joblib"


@dataclass(slots=True)
class NodeEquation:
    """One structural equation ``V <- f(Pa(V)) + U`` or ``P(V | Pa(V))``."""

    node: str
    kind: str
    parents: tuple[str, ...]
    parent_kinds: tuple[str, ...]
    parent_levels: tuple[int, ...]
    n_levels: int = 0
    model: Any | None = None
    score: float = float("nan")
    score_name: str = ""
    n_train: int = 0

    @property
    def is_root(self) -> bool:
        return not self.parents

    @property
    def is_discrete(self) -> bool:
        return self.kind in _CATEGORICAL_KINDS

    def design(self, parent_values: np.ndarray) -> np.ndarray:
        """One-hot the categorical parents, pass numeric ones through."""
        arr = np.atleast_2d(np.asarray(parent_values, dtype=float))
        blocks: list[np.ndarray] = []
        for j, (kind, n_levels) in enumerate(zip(self.parent_kinds, self.parent_levels)):
            col = arr[:, j]
            if kind in _CATEGORICAL_KINDS and n_levels > 2:
                codes = np.clip(np.rint(col), 0, n_levels - 1).astype(int)
                onehot = np.zeros((arr.shape[0], n_levels), dtype=float)
                onehot[np.arange(arr.shape[0]), codes] = 1.0
                blocks.append(onehot)
            else:
                blocks.append(col.reshape(-1, 1))
        return np.hstack(blocks) if blocks else np.zeros((arr.shape[0], 0))

    def mean(self, parent_values: np.ndarray) -> np.ndarray:
        """Structural mean ``f(Pa)`` for an additive-noise node."""
        if self.model is None:
            raise RuntimeError(f"{self.node}: no fitted equation (root node)")
        return np.asarray(self.model.predict(self.design(parent_values)), dtype=float)

    def proba(self, parent_values: np.ndarray) -> np.ndarray:
        """``P(V = k | Pa)`` as an ``(n, n_levels)`` matrix for a discrete node."""
        if self.model is None:
            raise RuntimeError(f"{self.node}: no fitted equation (root node)")
        design = self.design(parent_values)
        proba = np.zeros((design.shape[0], self.n_levels), dtype=float)
        raw = np.asarray(self.model.predict_proba(design), dtype=float)
        # classes_ may miss levels that never occur in the training split
        for col, level in enumerate(np.asarray(self.model.classes_, dtype=int)):
            proba[:, int(level)] = raw[:, col]
        total = proba.sum(axis=1, keepdims=True)
        return np.divide(proba, np.where(total > 0, total, 1.0))


class FCM:
    """Fitted structural equations for one dataset, in encoded space."""

    def __init__(
        self,
        dataset: str,
        scm: SCM | None = None,
        spec: DatasetSpec | None = None,
    ) -> None:
        self.dataset = dataset
        self.scm = scm or get_scm(dataset)
        self.spec = spec or dataset_spec(dataset)
        self.feature_names: tuple[str, ...] = tuple(self.spec.feature_names)
        self.index: dict[str, int] = {n: i for i, n in enumerate(self.feature_names)}
        #: features in topological order (the target is not a column of X)
        self.order: tuple[str, ...] = tuple(
            n for n in self.scm.nodes if n != self.scm.target
        )
        self.equations: dict[str, NodeEquation] = {}
        feats: tuple[FeatureSpec, ...] = self.spec.features
        self.bounds = np.asarray([f.encoded_bounds() for f in feats], dtype=float)
        self.kinds: tuple[str, ...] = tuple(f.kind for f in feats)
        self.discrete = np.asarray([f.is_discrete for f in feats], dtype=bool)
        self.integral = np.asarray([f.integer or f.is_discrete for f in feats], dtype=bool)
        self.n_features = len(self.feature_names)
        #: ``reach[a, b]``: ``b`` is a strict descendant of ``a``.  Lets one
        #: batched call give every row the recompute set *its own* actions imply,
        #: rather than the set the batch as a whole implies (see
        #: :meth:`counterfactual`).
        self.reach = np.zeros((self.n_features, self.n_features), dtype=bool)
        for name, j in self.index.items():
            for child in self.scm.descendants(name):
                if child in self.index:
                    self.reach[j, self.index[child]] = True

    # ------------------------------------------------------------------ fitting
    def _new_regressor(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False,
            random_state=CONFIG.RANDOM_STATE,
        )

    def _new_classifier(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False,
            random_state=CONFIG.RANDOM_STATE,
        )

    def _equation_for(self, node: str) -> NodeEquation:
        parents = tuple(p for p in self.scm.parents(node) if p != self.scm.target)
        feature = self.spec.get(node)
        return NodeEquation(
            node=node,
            kind=feature.kind,
            parents=parents,
            parent_kinds=tuple(self.spec.get(p).kind for p in parents),
            parent_levels=tuple(self.spec.get(p).n_levels for p in parents),
            n_levels=feature.n_levels,
        )

    def fit(self, X: np.ndarray) -> "FCM":
        """Fit every non-root equation on encoded training rows ``X``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} encoded columns, got {X.shape[1]}")
        for node in self.order:
            eq = self._equation_for(node)
            if eq.parents:
                P = X[:, [self.index[p] for p in eq.parents]]
                y = X[:, self.index[node]]
                design = eq.design(P)
                if eq.is_discrete:
                    eq.model = self._new_classifier()
                    eq.model.fit(design, np.rint(y).astype(int))
                    eq.score = float(eq.model.score(design, np.rint(y).astype(int)))
                    eq.score_name = "accuracy"
                else:
                    eq.model = self._new_regressor()
                    eq.model.fit(design, y)
                    eq.score = float(eq.model.score(design, y))
                    eq.score_name = "r2"
                eq.n_train = int(X.shape[0])
            self.equations[node] = eq
        fitted = [e for e in self.equations.values() if e.model is not None]
        LOGGER.info("FCM %s: fitted %d equation(s), %d root(s); mean fit %.3f",
                    self.dataset, len(fitted), len(self.equations) - len(fitted),
                    float(np.mean([e.score for e in fitted])) if fitted else float("nan"))
        return self

    # --------------------------------------------------------------- projection
    def project(self, X: np.ndarray) -> np.ndarray:
        """Clip to encoded bounds and round the discrete columns."""
        arr = np.atleast_2d(np.asarray(X, dtype=float)).copy()
        np.clip(arr, self.bounds[:, 0], self.bounds[:, 1], out=arr)
        if self.integral.any():
            arr[:, self.integral] = np.rint(arr[:, self.integral])
        return arr

    def _project_column(self, values: np.ndarray, j: int) -> np.ndarray:
        col = np.clip(values, self.bounds[j, 0], self.bounds[j, 1])
        return np.rint(col) if self.integral[j] else col

    # ---------------------------------------------------------------- abduction
    def abduct(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Exogenous term of every non-root node, one value per row of ``X``.

        Additive-noise nodes yield the residual ``v - f(pa)``.  Discrete nodes
        yield the mid-quantile of the observed level under ``P(V | pa)``: the
        monotone coupling that makes a discrete counterfactual deterministic and
        equal to the factual level whenever the parents do not move.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        rows = np.arange(X.shape[0])
        noise: dict[str, np.ndarray] = {}
        for node, eq in self.equations.items():
            if eq.model is None:
                continue
            P = X[:, [self.index[p] for p in eq.parents]]
            v = X[:, self.index[node]]
            if eq.is_discrete:
                proba = eq.proba(P)
                codes = np.clip(np.rint(v), 0, eq.n_levels - 1).astype(int)
                cum = np.cumsum(proba, axis=1)
                below = np.where(codes > 0, cum[rows, np.maximum(codes - 1, 0)], 0.0)
                noise[node] = below + 0.5 * proba[rows, codes]
            else:
                noise[node] = v - eq.mean(P)
        return noise

    @staticmethod
    def _couple(eq: NodeEquation, P: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Inverse-CDF coupling: the level holding quantile ``u`` under ``P(V|pa)``."""
        cum = np.cumsum(eq.proba(P), axis=1)
        cum[:, -1] = 2.0  # guarantee a hit even at u == 1
        return np.argmax(cum > np.clip(u, 0.0, 1.0)[:, None], axis=1).astype(float)

    # ----------------------------------------------------------- interventions
    def counterfactual(
        self,
        x: np.ndarray,
        values: np.ndarray,
        mask: np.ndarray,
        noise: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Three-step counterfactual of one factual row under ``m`` interventions.

        Parameters
        ----------
        x:
            ``(d,)`` factual row in encoded space.
        values:
            ``(m, d)`` candidate values; only read where ``mask`` is ``True``.
        mask:
            ``(m, d)`` boolean; ``True`` marks a hard ``do`` on that feature.
            Each row is treated independently: row ``i`` recomputes exactly the
            descendants of row ``i``'s own interventions, so batching a whole
            population changes nothing but the wall clock.
        noise:
            Pre-computed abduction for ``x`` (saves work across calls).

        Returns
        -------
        ndarray
            ``(m, d)`` counterfactual rows, projected onto the feasible box.
        """
        base = np.asarray(x, dtype=float).ravel()
        values = np.atleast_2d(np.asarray(values, dtype=float))
        mask = np.atleast_2d(np.asarray(mask, dtype=bool))
        m = values.shape[0]
        if noise is None:
            noise = self.abduct(base.reshape(1, -1))

        out = np.broadcast_to(base, (m, self.n_features)).copy()
        np.copyto(out, values, where=mask)
        out = self.project(out)

        # Which cells this row's *own* actions oblige us to recompute.  Taking
        # the union of the batch's interventions instead would be wrong in both
        # directions: a node someone else intervened on would be frozen for
        # everybody (``downstream`` drops its own seeds), and a node nobody's
        # actions reach would still be rewritten.
        recompute = (mask @ self.reach) & ~mask
        for node in self.order:
            eq = self.equations[node]
            if eq.model is None:  # a root has nothing to recompute from
                continue
            j = self.index[node]
            free = recompute[:, j]
            if not free.any():
                continue
            P = out[:, [self.index[p] for p in eq.parents]]
            u = np.broadcast_to(np.asarray(noise[node], dtype=float).ravel(), (m,))
            new = self._couple(eq, P, u) if eq.is_discrete else eq.mean(P) + u
            out[free, j] = self._project_column(new, j)[free]
        return out

    def do(self, x: np.ndarray, actions: Mapping[str, float]) -> np.ndarray:
        """Convenience single intervention from a ``{feature: value}`` mapping."""
        base = np.asarray(x, dtype=float).ravel()
        values = base.reshape(1, -1).copy()
        mask = np.zeros((1, self.n_features), dtype=bool)
        for name, value in actions.items():
            if name not in self.index:
                raise KeyError(f"{name!r} is not a feature of {self.dataset}")
            j = self.index[name]
            values[0, j] = float(value)
            mask[0, j] = True
        return self.counterfactual(base, values, mask)[0]

    def action_mask(self, X_factual: np.ndarray, X_proposed: np.ndarray) -> np.ndarray:
        """Which features a proposal changed (the implied intervention set)."""
        F = np.atleast_2d(np.asarray(X_factual, dtype=float))
        P = np.atleast_2d(np.asarray(X_proposed, dtype=float))
        return ~np.isclose(F, P, rtol=0.0, atol=1e-9)

    def repair(self, X_factual: np.ndarray, X_proposed: np.ndarray) -> np.ndarray:
        """The world that would actually result from a proposal's own changes.

        Takes every feature a method decided to change, treats it as a hard
        intervention, and lets the SCM recompute the consequences.  Comparing the
        result with the proposal is the causal-validity audit.
        """
        F = np.atleast_2d(np.asarray(X_factual, dtype=float))
        P = np.atleast_2d(np.asarray(X_proposed, dtype=float))
        if F.shape != P.shape:
            raise ValueError(f"shape mismatch: factual {F.shape} vs proposed {P.shape}")
        mask = self.action_mask(F, P)
        out = np.empty_like(P)
        for i in range(F.shape[0]):
            out[i] = self.counterfactual(F[i], P[i:i + 1], mask[i:i + 1])[0]
        return out

    def violation(self, X_factual: np.ndarray, X_proposed: np.ndarray) -> np.ndarray:
        """``(n, d)`` normalised gap between a proposal and its causal repair."""
        repaired = self.repair(X_factual, X_proposed)
        span = np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-12)
        return np.abs(repaired - np.atleast_2d(np.asarray(X_proposed, float))) / span

    # ----------------------------------------------------------------- reporting
    def report(self) -> list[dict[str, Any]]:
        """Per-equation fit quality — goes straight into the paper's appendix."""
        rows: list[dict[str, Any]] = []
        for node in self.order:
            eq = self.equations[node]
            rows.append({
                "dataset": self.dataset,
                "node": node,
                "kind": eq.kind,
                "n_parents": len(eq.parents),
                "parents": ", ".join(eq.parents),
                "metric": eq.score_name or "-",
                "score": eq.score,
                "root": eq.is_root,
            })
        return rows

    # --------------------------------------------------------------- persistence
    def save(self, path: str | Path | None = None) -> Path:
        return save_joblib(
            {"dataset": self.dataset, "equations": self.equations,
             "feature_names": self.feature_names},
            path or fcm_path(self.dataset),
        )

    def load(self, path: str | Path | None = None) -> "FCM":
        payload = load_joblib(path or fcm_path(self.dataset))
        if tuple(payload["feature_names"]) != self.feature_names:
            raise ValueError(f"{self.dataset}: cached FCM has stale feature names; "
                             "rebuild it with build_fcm(..., force=True)")
        self.equations = payload["equations"]
        return self


def build_fcm(dataset: str, force: bool = False, X: np.ndarray | None = None) -> FCM:
    """Fit (or load) the structural equations for ``dataset``."""
    fcm = FCM(dataset)
    path = fcm_path(dataset)
    if not force and path.exists() and X is None:
        LOGGER.info("Reusing cached FCM %s", path.name)
        return fcm.load(path)
    if X is None:
        X = build_dataset(dataset).X_train
    fcm.fit(X)
    fcm.save(path)
    return fcm


_CACHE: dict[str, FCM] = {}


def get_fcm(dataset: str, force: bool = False) -> FCM:
    """Process-cached accessor for the fitted structural equations."""
    if force or dataset not in _CACHE:
        _CACHE[dataset] = build_fcm(dataset, force=force)
    return _CACHE[dataset]


__all__ = [
    "FCM",
    "NodeEquation",
    "fcm_path",
    "build_fcm",
    "get_fcm",
]
