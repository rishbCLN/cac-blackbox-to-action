"""The counterfactual methods: one contract, five searches.

:mod:`src.methods.base` defines the :class:`~src.methods.base.Counterfactual`
every method returns (global rule #11) and the auditing every method inherits.
The searches themselves live one module each — three baselines from the
literature, a cost-optimal oracle, and CAC.

Only ``base``'s names are imported eagerly; the methods are pulled in on demand
by :func:`~src.methods.base.get_method`, so importing this package does not drag
in dice-ml, scipy.optimize and a trained black box just to read a dataclass.
"""

from __future__ import annotations

from typing import Any

from src.methods.base import (
    METHODS,
    Counterfactual,
    CounterfactualMethod,
    available_methods,
    counterfactuals_frame,
    get_method,
    register,
)

_LAZY = {
    "CACMethod": "src.methods.cac",
    "DiCEMethod": "src.methods.baseline_dice",
    "WachterMethod": "src.methods.baseline_wachter",
    "RandomMethod": "src.methods.baseline_random",
    "OracleMethod": "src.methods.oracle",
    "MINTMethod": "src.methods.mint",
}


def __getattr__(name: str) -> Any:
    """Import the module that defines ``name`` only when it is actually used."""
    if name in _LAZY:
        import importlib  # noqa: PLC0415

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "Counterfactual",
    "CounterfactualMethod",
    "METHODS",
    "register",
    "get_method",
    "available_methods",
    "counterfactuals_frame",
    *_LAZY,
]
