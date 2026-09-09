"""The evaluation layer: §9's numbers, computed from outside every method.

Three questions, one per module, asked in the order a reader asks them:

* :mod:`src.evaluation.metrics` — did the black box change its mind, what did the
  plan cost, how does that compare with the oracle's proven optimum?
* :mod:`src.evaluation.faithfulness` — if the person does it and the world reacts
  the way the fitted SCM says, does the decision still change?
* :mod:`src.evaluation.feasibility` — could a person have done it at all?

Nothing here asks a method for a metric.  Each function takes the factual row and
the proposed row and re-derives everything from the shared artefacts, so a method
cannot grade its own work and two methods cannot be measured by different rules.
:func:`~src.evaluation.feasibility.headline` joins the three into the funnel the
paper leads with.

The names are re-exported lazily (PEP 562) so importing the package does not drag
in pandas, the black box and the fitted SCM before anything needs them.
"""

from __future__ import annotations

from typing import Any

_MODULES = {
    "metrics": (
        "BASE_COLUMNS",
        "EXTRA_COLUMNS",
        "Evaluator",
        "cost_ratio",
        "disagreements",
        "diversity",
        "query_instances",
    ),
    "faithfulness": (
        "FAITHFULNESS_COLUMNS",
        "blamed_features",
    ),
    "feasibility": (
        "FEASIBILITY_COLUMNS",
        "HEADLINE_COLUMNS",
        "ORDINARY_AT",
        "blocked_features",
        "headline",
    ),
}

#: ``summarise`` lives in all three modules and means something different in each,
#: so it is *not* re-exported bare: ``from src.evaluation import metrics`` and then
#: ``metrics.summarise`` says which table you meant.  The aliases below are the
#: unambiguous spellings.
_ALIASES = {
    "summarise_metrics": ("metrics", "summarise"),
    "summarise_faithfulness": ("faithfulness", "summarise"),
    "summarise_feasibility": ("feasibility", "summarise"),
    "assess_faithfulness": ("faithfulness", "assess"),
    "assess_feasibility": ("feasibility", "assess"),
}

_EXPORTS = tuple(sorted({name for names in _MODULES.values() for name in names}
                        | set(_ALIASES)))


def __getattr__(name: str) -> Any:
    """Import the module that owns ``name`` on first use."""
    import importlib  # noqa: PLC0415

    if name in _ALIASES:
        module, attribute = _ALIASES[name]
        return getattr(importlib.import_module(f"src.evaluation.{module}"), attribute)
    for module, names in _MODULES.items():
        if name in names:
            return getattr(importlib.import_module(f"src.evaluation.{module}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = list(_EXPORTS)
