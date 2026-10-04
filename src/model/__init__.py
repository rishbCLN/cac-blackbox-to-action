"""Model layer: the opaque black box and its training entry points."""

from __future__ import annotations

from src.model.blackbox import (
    BlackBoxModel,
    BlackBoxOpacityError,
    load_blackbox,
    model_path,
)
from src.model.train import (
    METRICS_PATH,
    evaluate_predictions,
    train_all,
    train_dataset,
)

__all__ = [
    "BlackBoxModel",
    "BlackBoxOpacityError",
    "model_path",
    "load_blackbox",
    "train_dataset",
    "train_all",
    "evaluate_predictions",
    "METRICS_PATH",
]
