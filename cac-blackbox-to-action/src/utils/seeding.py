"""Deterministic seeding helpers (global rule #2)."""

from __future__ import annotations

import os
import random

import numpy as np

from src.config import CONFIG


def set_global_seed(seed: int | None = None) -> int:
    """Seed every RNG the pipeline touches and return the seed used."""
    s = CONFIG.RANDOM_STATE if seed is None else int(seed)
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    return s


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a fresh :class:`numpy.random.Generator` (never the global one)."""
    return np.random.default_rng(CONFIG.RANDOM_STATE if seed is None else int(seed))


__all__ = ["set_global_seed", "get_rng"]
