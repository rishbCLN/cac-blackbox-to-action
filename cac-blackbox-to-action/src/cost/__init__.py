"""The cost layer: how expensive an action is for the person who has to take it.

One module, three quantities — a raw hand-crafted effort, a ``[0, 1]``
normalised cost that the search minimises, and a ``1e6`` penalty for actions
that are not admissible at all.  See :mod:`src.cost.cost_functions` for the
reasoning and :func:`~src.cost.cost_functions.export_cost_models` for the
artefacts that document every number.

The names are re-exported lazily (PEP 562) so that
``python -m src.cost.cost_functions`` does not import that module twice.
"""

from __future__ import annotations

from typing import Any

_EXPORTS = (
    "CHANGE_ATOL",
    "COST_MODELS",
    "INFINITE_COST",
    "VIOLATIONS",
    "ActionCost",
    "ConstantCost",
    "CostBreakdown",
    "CostFunction",
    "CostModel",
    "LadderCost",
    "LinearCost",
    "PiecewiseLinearCost",
    "SwitchCost",
    "all_cost_models",
    "categorical_cost",
    "constant_cost",
    "cost_functions",
    "export_cost_models",
    "get_cost_model",
    "linear_cost",
    "piecewise_linear",
    "total_cost",
)


def __getattr__(name: str) -> Any:
    """Import :mod:`src.cost.cost_functions` on first use of one of its names."""
    if name in _EXPORTS:
        import importlib  # noqa: PLC0415

        return getattr(importlib.import_module("src.cost.cost_functions"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = list(_EXPORTS)
