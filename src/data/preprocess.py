"""Preprocessing: three explicit representations, and the map between them.

============  =====================================================  ==============
space         content                                                consumed by
============  =====================================================  ==============
*semantic*    ``pd.DataFrame``, strings for categories, real units    humans, DiCE
*encoded*     ``np.ndarray[float]``, level index for categories,      SCM, cost model,
              **original units** for numeric features                every CF method
*model*       encoded + ``StandardScaler`` on non-protected numeric   the black box
============  =====================================================  ==============

Only the black box ever sees *model* space; it applies the scaler internally
(global rule #8).  Protected attributes are never rescaled (global rule §4.2),
so ``age`` reads as years everywhere a human or the SCM looks at it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.data.schema import DatasetSpec
from src.utils.io import load_joblib, load_json, save_dataframe, save_joblib, save_json
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class LevelEncoder:
    """Order-preserving categorical encoder (``sklearn``-compatible surface).

    ``sklearn.preprocessing.LabelEncoder`` sorts its classes alphabetically,
    which would silently destroy the semantic order of ordinal features such as
    ``education`` (``Bachelors`` < ``Doctorate`` < ``HS-grad`` alphabetically).
    This encoder keeps the order declared in the :class:`FeatureSpec`.
    """

    def __init__(self, levels: Iterable[str]) -> None:
        self.classes_: np.ndarray = np.asarray(list(levels), dtype=object)
        self._to_code: dict[str, int] = {c: i for i, c in enumerate(self.classes_)}

    def fit(self, y: Iterable[str] | None = None) -> "LevelEncoder":  # noqa: ARG002
        return self

    def transform(self, y: Iterable[str]) -> np.ndarray:
        unknown = sorted({v for v in y if v not in self._to_code})
        if unknown:
            raise ValueError(f"unknown level(s) {unknown}; known: {list(self.classes_)}")
        return np.asarray([self._to_code[v] for v in y], dtype=np.int64)

    def inverse_transform(self, codes: Iterable[float]) -> np.ndarray:
        idx = np.clip(np.rint(np.asarray(list(codes), dtype=float)).astype(int),
                      0, len(self.classes_) - 1)
        return self.classes_[idx]

    def __len__(self) -> int:
        return len(self.classes_)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LevelEncoder({len(self.classes_)} levels)"


class Transformer:
    """Bidirectional map between the semantic, encoded and model spaces."""

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self.feature_names: list[str] = spec.feature_names
        self.encoders: dict[str, LevelEncoder] = {
            f.name: LevelEncoder(f.levels or ())
            for f in spec.features
            if f.kind != "numeric"
        }
        self.scaler: StandardScaler | None = None
        #: numeric, non-protected columns — the only ones the scaler touches
        self.scaled_columns: list[str] = [
            f.name for f in spec.features if f.kind == "numeric" and not f.protected
        ]
        self.bounds: np.ndarray = np.array(
            [f.encoded_bounds() for f in spec.features], dtype=float
        )
        self.is_categorical: np.ndarray = np.array(
            [f.kind != "numeric" for f in spec.features], dtype=bool
        )
        self.is_integer: np.ndarray = np.array(
            [f.is_discrete for f in spec.features], dtype=bool
        )
        self._fitted = False

    # ------------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame) -> "Transformer":
        """Fit the scaler and tighten numeric bounds to the observed support."""
        enc = self._encode_raw(df)
        for j, feat in enumerate(self.spec.features):
            if feat.kind == "numeric":
                lo, hi = float(np.min(enc[:, j])), float(np.max(enc[:, j]))
                spec_lo, spec_hi = feat.encoded_bounds()
                self.bounds[j] = (max(lo, spec_lo), min(hi, spec_hi))
        self.scaler = StandardScaler()
        if self.scaled_columns:
            self.scaler.fit(enc[:, self._scaled_idx()])
        self._fitted = True
        LOGGER.info("Transformer[%s] fitted on %d rows, %d features",
                    self.spec.name, len(df), len(self.feature_names))
        return self

    def _scaled_idx(self) -> list[int]:
        return [self.feature_names.index(c) for c in self.scaled_columns]

    # ---------------------------------------------------------------- encode
    def _encode_raw(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for feat in self.spec.features:
            col = df[feat.name]
            if feat.kind == "numeric":
                cols.append(pd.to_numeric(col, errors="raise").to_numpy(dtype=float))
            else:
                cols.append(
                    self.encoders[feat.name].transform(col.astype(str)).astype(float)
                )
        return np.column_stack(cols)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """semantic -> encoded."""
        return self._encode_raw(df)

    def decode(self, arr: np.ndarray) -> pd.DataFrame:
        """encoded -> semantic."""
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        out: dict[str, np.ndarray] = {}
        for j, feat in enumerate(self.spec.features):
            column = arr[:, j]
            if feat.kind == "numeric":
                if feat.integer:
                    column = np.rint(column)
                out[feat.name] = column
            else:
                out[feat.name] = self.encoders[feat.name].inverse_transform(column)
        return pd.DataFrame(out)

    # ----------------------------------------------------------------- model
    def to_model_space(self, arr: np.ndarray) -> np.ndarray:
        """encoded -> model (used *only* inside :class:`BlackBoxModel`)."""
        if not self._fitted or self.scaler is None:
            raise RuntimeError("Transformer must be fitted first")
        out = np.atleast_2d(np.asarray(arr, dtype=float)).copy()
        idx = self._scaled_idx()
        if idx:
            out[:, idx] = self.scaler.transform(out[:, idx])
        return out

    def from_model_space(self, arr: np.ndarray) -> np.ndarray:
        """model -> encoded (inverse of :meth:`to_model_space`)."""
        if not self._fitted or self.scaler is None:
            raise RuntimeError("Transformer must be fitted first")
        out = np.atleast_2d(np.asarray(arr, dtype=float)).copy()
        idx = self._scaled_idx()
        if idx:
            out[:, idx] = self.scaler.inverse_transform(out[:, idx])
        return out

    # ------------------------------------------------------------- utilities
    def project(self, arr: np.ndarray) -> np.ndarray:
        """Clip to bounds and round discrete features — keeps CFs admissible."""
        out = np.atleast_2d(np.asarray(arr, dtype=float)).copy()
        out = np.clip(out, self.bounds[:, 0], self.bounds[:, 1])
        disc = np.where(self.is_integer)[0]
        if disc.size:
            out[:, disc] = np.rint(out[:, disc])
        return out

    def as_frame(self, arr: np.ndarray) -> pd.DataFrame:
        """encoded ndarray -> encoded DataFrame (numeric, keeps column names)."""
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        return pd.DataFrame(arr, columns=self.feature_names)

    def scaler_params(self) -> dict[str, list[float]]:
        if self.scaler is None or not self.scaled_columns:
            return {"columns": [], "mean": [], "scale": []}
        return {
            "columns": list(self.scaled_columns),
            "mean": [float(v) for v in self.scaler.mean_],
            "scale": [float(v) for v in self.scaler.scale_],
        }


@dataclass(slots=True)
class DatasetBundle:
    """Everything downstream code needs about one prepared dataset."""

    spec: DatasetSpec
    frame: pd.DataFrame  # semantic space, features + target
    X: np.ndarray  # encoded space, all rows
    y: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    transformer: Transformer
    is_synthetic: bool = False

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def feature_names(self) -> list[str]:
        return self.spec.feature_names

    @property
    def X_train(self) -> np.ndarray:
        return self.X[self.train_idx]

    @property
    def X_test(self) -> np.ndarray:
        return self.X[self.test_idx]

    @property
    def y_train(self) -> np.ndarray:
        return self.y[self.train_idx]

    @property
    def y_test(self) -> np.ndarray:
        return self.y[self.test_idx]

    def frame_train(self) -> pd.DataFrame:
        return self.frame.iloc[self.train_idx].reset_index(drop=True)

    def frame_test(self) -> pd.DataFrame:
        return self.frame.iloc[self.test_idx].reset_index(drop=True)


def _loaders() -> Mapping[str, Callable[[], tuple[pd.DataFrame, DatasetSpec]]]:
    """Lazy loader registry (imported here to avoid a circular import)."""
    from src.data.load_adult import load_adult
    from src.data.load_diabetes import load_diabetes
    from src.data.load_german import load_german
    from src.data.load_heloc import load_heloc
    from src.data.load_synth_block import load_synth_block
    from src.data.load_synth_reach import load_synth_reach

    return {"adult": load_adult, "german": load_german, "heloc": load_heloc,
            "diabetes": load_diabetes, "synth_reach": load_synth_reach,
            "synth_block": load_synth_block}


def dataset_spec(dataset: str) -> DatasetSpec:
    """The feature contract of ``dataset`` without loading or downloading any data.

    Used by the SCM and cost layers, which need the schema but not the rows.
    """
    modules = {
        "adult": "src.data.load_adult",
        "german": "src.data.load_german",
        "heloc": "src.data.load_heloc",
        "diabetes": "src.data.load_diabetes",
        "synth_reach": "src.data.load_synth_reach",
        "synth_block": "src.data.load_synth_block",
    }
    if dataset not in modules:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {list(modules)}")
    import importlib  # noqa: PLC0415 - deferred to keep import time low

    return importlib.import_module(modules[dataset]).SPEC


def processed_paths(dataset: str) -> dict[str, "object"]:
    """Canonical on-disk locations for one prepared dataset."""
    return {
        "csv": CONFIG.DATA_PROCESSED / f"{dataset}.csv",
        "metadata": CONFIG.DATA_PROCESSED / f"{dataset}_metadata.json",
        "transformer": CONFIG.MODELS_DIR / f"transformer_{dataset}.joblib",
        "encoders": CONFIG.MODELS_DIR / f"encoders_{dataset}.joblib",
        "scaler": CONFIG.MODELS_DIR / f"scaler_{dataset}.joblib",
    }


def build_dataset(dataset: str, force: bool = False) -> DatasetBundle:
    """Load, clean, split, encode and persist ``dataset``.

    Writes ``data/processed/<dataset>.csv``, ``<dataset>_metadata.json`` and the
    fitted ``Transformer`` / encoders / scaler under ``models/``.
    """
    if dataset not in _loaders():
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {list(_loaders())}")
    paths = processed_paths(dataset)
    if not force and paths["csv"].exists() and paths["transformer"].exists():
        LOGGER.info("Reusing cached processed %s (pass force=True to rebuild)", dataset)
        return load_dataset(dataset)

    frame, spec = _loaders()[dataset]()
    is_synthetic = any("SYNTHETIC" in note for note in spec.notes)

    if len(frame) > CONFIG.MAX_ROWS:
        frame, _ = train_test_split(
            frame, train_size=CONFIG.MAX_ROWS, random_state=CONFIG.RANDOM_STATE,
            stratify=frame[spec.target],
        )
        frame = frame.reset_index(drop=True)
        LOGGER.info("%s sub-sampled to MAX_ROWS=%d", dataset, CONFIG.MAX_ROWS)

    y = frame[spec.target].to_numpy(dtype=int)
    train_idx, test_idx = train_test_split(
        np.arange(len(frame)), test_size=CONFIG.TEST_SIZE,
        random_state=CONFIG.RANDOM_STATE, stratify=y,
    )
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)

    transformer = Transformer(spec).fit(frame.iloc[train_idx])
    X = transformer.encode(frame)

    save_dataframe(frame, paths["csv"])
    save_joblib(transformer, paths["transformer"])
    save_joblib(transformer.encoders, paths["encoders"])
    save_joblib(transformer.scaler, paths["scaler"])
    save_json(
        {
            "dataset": dataset,
            "spec": spec.to_dict(),
            "n_rows": int(len(frame)),
            "n_features": len(spec.features),
            "feature_names": spec.feature_names,
            "feature_types": {f.name: f.kind for f in spec.features},
            "protected_attributes": spec.protected_features,
            "non_manipulable": spec.non_manipulable_features,
            "target": spec.target,
            "positive_rate": float(np.mean(y)),
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "encoded_bounds": transformer.bounds.tolist(),
            "scaler": transformer.scaler_params(),
            "level_maps": {
                name: list(enc.classes_) for name, enc in transformer.encoders.items()
            },
            "is_synthetic": is_synthetic,
            "random_state": CONFIG.RANDOM_STATE,
        },
        paths["metadata"],
    )
    LOGGER.info("Built %s: %d train / %d test rows -> %s",
                dataset, len(train_idx), len(test_idx), paths["csv"])
    return DatasetBundle(spec, frame, X, y, train_idx, test_idx, transformer, is_synthetic)


def load_dataset(dataset: str) -> DatasetBundle:
    """Read a previously prepared dataset back from disk (no recomputation)."""
    paths = processed_paths(dataset)
    if not paths["csv"].exists():
        raise FileNotFoundError(
            f"{paths['csv']} not found — run `python experiments/01_train_blackbox.py` "
            "or `build_dataset()` first."
        )
    meta = load_json(paths["metadata"])
    spec = DatasetSpec.from_dict(meta["spec"])
    frame = pd.read_csv(paths["csv"])
    for feat in spec.features:
        if feat.kind != "numeric":
            frame[feat.name] = frame[feat.name].astype(str)
    transformer: Transformer = load_joblib(paths["transformer"])
    X = transformer.encode(frame)
    y = frame[spec.target].to_numpy(dtype=int)
    return DatasetBundle(
        spec=spec,
        frame=frame,
        X=X,
        y=y,
        train_idx=np.asarray(meta["train_idx"], dtype=int),
        test_idx=np.asarray(meta["test_idx"], dtype=int),
        transformer=transformer,
        is_synthetic=bool(meta.get("is_synthetic", False)),
    )


def get_dataset(dataset: str, force: bool = False) -> DatasetBundle:
    """Build ``dataset`` if the processed files are missing, else read them back.

    A thin alias for :func:`build_dataset`, which reuses the on-disk cache in
    ``data/processed/`` when it exists and only re-fits on ``force=True``.  Unlike
    :func:`~src.scm.define_scm.get_scm` and its siblings there is no in-process
    memoisation here: each call re-reads from disk, which is cheap next to the fit
    it avoids and keeps the bundle free of cross-call shared state.
    """
    return build_dataset(dataset, force=force)


__all__ = [
    "LevelEncoder",
    "Transformer",
    "DatasetBundle",
    "build_dataset",
    "load_dataset",
    "get_dataset",
    "processed_paths",
    "dataset_spec",
]
