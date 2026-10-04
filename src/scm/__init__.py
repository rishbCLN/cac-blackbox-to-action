"""The causal layer: the graph, its fitted equations, and its validation.

``define_scm`` says *which* variables react to an intervention, ``fcm`` says
*how much*, and ``validate_scm`` checks both claims against the data.  Nothing
in this package touches the black box (global rule #8) — the structural
equations are learned from the training split alone.

The validation names are re-exported lazily (PEP 562) so that
``python -m src.scm.validate_scm`` does not import that module twice, and so
that importing the graph does not drag in scipy and the plotting stack.
"""

from __future__ import annotations

from typing import Any

from src.scm.define_scm import (
    COST_LEVELS,
    INFINITE_LABEL,
    SCM,
    SCMError,
    all_scms,
    build_scm,
    get_scm,
    load_scm_json,
    scm_path,
)
from src.scm.fcm import FCM, NodeEquation, build_fcm, fcm_path, get_fcm

_LAZY = {
    "CITest": "src.scm.validate_scm",
    "Issue": "src.scm.validate_scm",
    "ValidationReport": "src.scm.validate_scm",
    "validate_scm": "src.scm.validate_scm",
    "validate_all": "src.scm.validate_scm",
}


def __getattr__(name: str) -> Any:
    """Import ``validate_scm`` only when one of its names is actually used."""
    if name in _LAZY:
        import importlib  # noqa: PLC0415

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "SCM",
    "SCMError",
    "COST_LEVELS",
    "INFINITE_LABEL",
    "scm_path",
    "load_scm_json",
    "build_scm",
    "get_scm",
    "all_scms",
    "FCM",
    "NodeEquation",
    "fcm_path",
    "build_fcm",
    "get_fcm",
    "CITest",
    "Issue",
    "ValidationReport",
    "validate_scm",
    "validate_all",
]
