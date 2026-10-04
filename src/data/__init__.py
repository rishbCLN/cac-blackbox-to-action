"""Data layer: free dataset loaders, schema and the three-space transformer."""

from __future__ import annotations

from src.data.preprocess import (
    DatasetBundle,
    LevelEncoder,
    Transformer,
    build_dataset,
    dataset_spec,
    get_dataset,
    load_dataset,
    processed_paths,
)
from src.data.schema import DatasetSpec, FeatureSpec

__all__ = [
    "DatasetSpec",
    "FeatureSpec",
    "DatasetBundle",
    "LevelEncoder",
    "Transformer",
    "build_dataset",
    "load_dataset",
    "get_dataset",
    "dataset_spec",
    "processed_paths",
]
