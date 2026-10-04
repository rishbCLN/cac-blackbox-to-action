"""DiCE — Diverse Counterfactual Explanations (Mothilal et al. 2020), §8.2.

The reference implementation, called through ``dice-ml``, so the comparison is
against the real library rather than my reading of the paper.  DiCE is the
strongest baseline here: it handles categoricals natively, it samples from the
training distribution, and diversity is an explicit objective — everything the
random baseline lacks.  What it still has no notion of is *causality* or
*effort*, which is the point the paper is making.

Two adapters are needed and neither leaks anything (global rule #8):

*The model.*  ``dice-ml`` expects a scikit-learn-style object taking a DataFrame.
:class:`_BlackBoxFacade` provides exactly ``predict`` / ``predict_proba`` /
``classes_`` over the encoded columns and forwards to the black box.  It cannot
expose more than that because it has nothing more to expose.

*The data.*  DiCE wants a training frame to sample from and to compute its MAD
weights.  It gets the encoded training split with the categoricals as level
codes cast to strings, which is how DiCE recognises a categorical — so it does
its own sampling over real levels instead of interpolating between them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.methods.base import CounterfactualMethod, register
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: The outcome column name handed to dice-ml; kept out of the feature namespace.
_OUTCOME = "_dice_outcome"


class _BlackBoxFacade:
    """A scikit-learn shaped view of the black box, and nothing more.

    ``dice-ml``'s sklearn backend calls ``predict_proba`` (and sometimes
    ``predict``) with a DataFrame whose categorical columns hold strings.  This
    casts the frame back to the encoded float matrix the black box expects.
    ``classes_`` is declared here rather than forwarded, so the estimator's own
    attributes stay unreachable.
    """

    def __init__(self, method: "DiCEMethod") -> None:
        self._method = method
        self.classes_ = np.array([0, 1])

    def _encoded(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.reindex(columns=list(self._method.feature_names))
        return np.atleast_2d(np.asarray(X, dtype=float))

    def predict(self, X: Any) -> np.ndarray:
        return self._method.model.predict(self._encoded(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._method.model.predict_proba(self._encoded(X))


@register
class DiCEMethod(CounterfactualMethod):
    """The reference DiCE explainer, wired to the opaque black box."""

    name = "dice"

    def __init__(self, dataset: str, **kwargs: Any) -> None:
        super().__init__(dataset, **kwargs)
        self._explainer: Any = None
        #: Which columns DiCE samples by level.  Only genuine categoricals and
        #: ordinals, not every integer-valued feature: telling DiCE that
        #: ``hours_per_week`` is categorical would let it sample only the 91
        #: values it saw in training and reject any record holding a 92nd.
        self._categorical = [f.name for f in self.spec.features if f.levels]
        self._continuous = [n for n in self.feature_names
                            if n not in set(self._categorical)]

    # ------------------------------------------------------------------ setup
    def _frame(self, X: np.ndarray, y: np.ndarray | None = None) -> pd.DataFrame:
        """Encoded rows as the frame DiCE expects, categoricals as level strings."""
        df = pd.DataFrame(np.atleast_2d(np.asarray(X, dtype=float)),
                          columns=list(self.feature_names))
        for name in self._categorical:
            df[name] = np.rint(df[name].to_numpy()).astype(int).astype(str)
        if y is not None:
            df[_OUTCOME] = np.asarray(y).ravel().astype(int)
        return df

    @property
    def explainer(self) -> Any:
        """The ``dice_ml.Dice`` object, built once per method instance."""
        if self._explainer is None:
            self._explainer = self._build_explainer()
        return self._explainer

    def _build_explainer(self) -> Any:
        import dice_ml  # noqa: PLC0415

        data = dice_ml.Data(
            dataframe=self._frame(self.X_train, self.y_train),
            continuous_features=list(self._continuous),
            outcome_name=_OUTCOME,
        )
        model = dice_ml.Model(model=_BlackBoxFacade(self), backend="sklearn",
                              model_type="classifier")
        LOGGER.info("dice/%s: %s method, %d continuous / %d categorical features",
                    self.dataset, CONFIG.DICE_METHOD, len(self._continuous),
                    len(self._categorical))
        return dice_ml.Dice(data, model, method=CONFIG.DICE_METHOD)

    def _permitted_range(self) -> dict[str, list[Any]]:
        """The schema's box, so DiCE cannot propose 200 hours in a week.

        Categorical ranges are intersected with the levels actually present in
        the training frame, because ``dice-ml`` rejects a permitted category it
        has never seen — a schema level that no training row uses would abort
        the whole call.
        """
        observed = {name: set(np.rint(self.X_train[:, j]).astype(int))
                    for j, name in enumerate(self.feature_names)
                    if name in set(self._categorical)}
        ranges: dict[str, list[Any]] = {}
        for j, name in enumerate(self.feature_names):
            lo, hi = self.bounds[j]
            if name in observed:
                levels = sorted(v for v in observed[name]
                                if round(lo) <= v <= round(hi))
                ranges[name] = [str(v) for v in levels]
            else:
                ranges[name] = [float(lo), float(hi)]
        return ranges

    def _features_to_vary(self) -> list[str] | str:
        if not CONFIG.BASELINES_RESPECT_IMMUTABLE:
            return "all"
        return [self.feature_names[j] for j in self.actionable]

    # ----------------------------------------------------------------- search
    def _propose(self, x: np.ndarray, y_target: int, k: int) -> np.ndarray:
        query = self._frame(x.reshape(1, -1))
        kwargs: dict[str, Any] = {
            "total_CFs": int(k),
            "desired_class": int(y_target),
            "features_to_vary": self._features_to_vary(),
            "permitted_range": self._permitted_range(),
            "verbose": False,
        }
        try:
            try:
                result = self.explainer.generate_counterfactuals(
                    query, random_seed=self.random_state, **kwargs)
            except TypeError:  # `random_seed` is a random/genetic-only knob
                result = self.explainer.generate_counterfactuals(query, **kwargs)
        except Exception as exc:  # noqa: BLE001 - dice raises several own types
            LOGGER.warning("dice/%s: generation failed (%s: %s)",
                           self.dataset, type(exc).__name__, exc)
            return np.empty((0, self.d))
        return self._rows(result)

    def _rows(self, result: Any) -> np.ndarray:
        """Pull the counterfactual frame out of DiCE's result object."""
        examples = getattr(result, "cf_examples_list", None) or []
        frames = [e.final_cfs_df for e in examples
                  if getattr(e, "final_cfs_df", None) is not None]
        if not frames:
            return np.empty((0, self.d))
        df = pd.concat(frames, ignore_index=True)
        df = df.reindex(columns=list(self.feature_names))
        return df.to_numpy(dtype=float)


__all__ = ["DiCEMethod"]




