"""The black box.  Records in, probabilities out — and nothing else.

Global rule #8 ("the black box is sacred") is *enforced*, not just documented:
:class:`BlackBoxModel` exposes ``fit`` / ``predict`` / ``predict_proba`` /
``save`` / ``load`` and actively refuses any attribute that would leak model
internals (``feature_importances_``, ``coef_``, ``get_booster``, gradients …).
Every counterfactual method in this repository therefore has access to exactly
the same information a real applicant's bank API would expose.

The model consumes *encoded* space (see :mod:`src.data.preprocess`) and applies
its own internal standardisation, so scaling is an implementation detail the
explainers cannot see.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import CONFIG
from src.data.preprocess import Transformer
from src.utils.io import load_joblib, save_joblib
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: Attributes that would let an explainer peek inside the model.
_FORBIDDEN: frozenset[str] = frozenset({
    "feature_importances_", "coef_", "intercept_", "coefs_", "intercepts_",
    "booster", "get_booster", "get_score", "trees_to_dataframe", "apply",
    "estimators_", "tree_", "decision_function", "get_params", "set_params",
    "staged_predict", "partial_fit", "loss_curve_", "n_features_in_",
})


class BlackBoxOpacityError(AttributeError):
    """Raised when code tries to inspect the internals of the black box."""


class BlackBoxModel:
    """Opaque binary classifier over *encoded* feature vectors."""

    def __init__(
        self,
        model_type: str = "xgb",
        transformer: Transformer | None = None,
        dataset: str | None = None,
        **overrides: Any,
    ) -> None:
        if model_type not in ("xgb", "mlp"):
            raise ValueError(f"model_type must be 'xgb' or 'mlp', got {model_type!r}")
        self.model_type = model_type
        self.dataset = dataset
        self._transformer = transformer
        self._estimator = self._build(model_type, **overrides)
        #: number of *rows* scored so far — the query budget an explainer spent
        self.query_count: int = 0

    # ---------------------------------------------------------------- builder
    @staticmethod
    def _build(model_type: str, **overrides: Any) -> Any:
        if model_type == "xgb":
            from xgboost import XGBClassifier  # noqa: PLC0415

            params: dict[str, Any] = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": CONFIG.RANDOM_STATE,
                "eval_metric": "logloss",
                "n_jobs": 4,
                "tree_method": "hist",
            }
            params.update(overrides)
            return XGBClassifier(**params)

        from sklearn.neural_network import MLPClassifier  # noqa: PLC0415

        params = {
            "hidden_layer_sizes": (64, 32),
            "max_iter": 300,
            "random_state": CONFIG.RANDOM_STATE,
            "early_stopping": False,
        }
        params.update(overrides)
        return MLPClassifier(**params)

    # ------------------------------------------------------------- internals
    def _to_model_space(self, X: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        if self._transformer is None:
            return arr
        return self._transformer.to_model_space(arr)

    # ------------------------------------------------------------------- API
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlackBoxModel":
        self._estimator.fit(self._to_model_space(X), np.asarray(y).ravel())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard label at :attr:`Config.DECISION_THRESHOLD`."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= CONFIG.DECISION_THRESHOLD).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        arr = self._to_model_space(X)
        self.query_count += int(arr.shape[0])
        return np.asarray(self._estimator.predict_proba(arr), dtype=float)

    def proba_positive(self, X: np.ndarray) -> np.ndarray:
        """Convenience: ``predict_proba(X)[:, 1]`` as a 1-D array."""
        return self.predict_proba(X)[:, 1]

    def reset_queries(self) -> None:
        self.query_count = 0

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Persist estimator + transformer so the box stays self-contained."""
        return save_joblib(
            {
                "estimator": self._estimator,
                "transformer": self._transformer,
                "model_type": self.model_type,
                "dataset": self.dataset,
            },
            path,
        )

    def load(self, path: str | Path) -> "BlackBoxModel":
        bundle = load_joblib(path)
        if isinstance(bundle, dict) and "estimator" in bundle:
            self._estimator = bundle["estimator"]
            self._transformer = bundle.get("transformer", self._transformer)
            self.model_type = bundle.get("model_type", self.model_type)
            self.dataset = bundle.get("dataset", self.dataset)
        else:  # a bare estimator saved by an older run
            self._estimator = bundle
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "BlackBoxModel":
        bundle = load_joblib(path)
        model = cls(
            model_type=bundle.get("model_type", "xgb") if isinstance(bundle, dict) else "xgb",
            transformer=bundle.get("transformer") if isinstance(bundle, dict) else None,
            dataset=bundle.get("dataset") if isinstance(bundle, dict) else None,
        )
        return model.load(path)

    # ------------------------------------------------------------- opacity
    def __getattr__(self, item: str) -> Any:
        if item in _FORBIDDEN:
            raise BlackBoxOpacityError(
                f"{item!r} is not available: the black box exposes only fit / "
                "predict / predict_proba (global rule #8)."
            )
        raise AttributeError(f"{type(self).__name__!r} has no attribute {item!r}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BlackBoxModel(type={self.model_type!r}, dataset={self.dataset!r})"


def model_path(dataset: str, model_type: str = "xgb") -> Path:
    """``models/<type>_<dataset>.joblib`` — the layout required by the spec."""
    return CONFIG.MODELS_DIR / f"{model_type}_{dataset}.joblib"


def load_blackbox(dataset: str, model_type: str = "xgb") -> BlackBoxModel:
    """Load a trained black box, with an actionable error if it is missing."""
    path = model_path(dataset, model_type)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python experiments/01_train_blackbox.py` first."
        )
    LOGGER.info("Loaded black box %s", path.name)
    return BlackBoxModel.from_file(path)


__all__ = [
    "BlackBoxModel",
    "BlackBoxOpacityError",
    "model_path",
    "load_blackbox",
]
