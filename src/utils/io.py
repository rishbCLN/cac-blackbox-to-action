"""Small IO helpers: JSON, joblib, dataframes, figures and LaTeX tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars/arrays and ``Path``."""

    def default(self, o: Any) -> Any:  # noqa: D102
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        return super().default(o)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Write ``obj`` to ``path`` atomically, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=indent, cls=NumpyJSONEncoder, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update_json(path: str | Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into the JSON dict at ``path`` (created if missing)."""
    path = Path(path)
    current: dict[str, Any] = load_json(path) if path.exists() else {}
    current.update(updates)
    save_json(current, path)
    return current


def save_joblib(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_joblib(path: str | Path) -> Any:
    return joblib.load(Path(path))


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_figure(fig, name: str, subdir: str | None = None) -> list[Path]:
    """Save ``fig`` publication-ready (300 DPI + vector PDF).  Global rule #12."""
    out_dir = CONFIG.FIGURES_DIR / subdir if subdir else CONFIG.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in CONFIG.FIGURE_FORMATS:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=CONFIG.FIGURE_DPI, bbox_inches="tight", format=ext)
        paths.append(p)
    LOGGER.info("Saved figure %s -> %s", name, ", ".join(p.name for p in paths))
    return paths


def save_latex_table(
    df: pd.DataFrame,
    name: str,
    caption: str,
    label: str | None = None,
    float_format: str = "%.3f",
    column_format: str | None = None,
    escape: bool = True,
    na_rep: str = "--",
) -> Path:
    """Write a ``booktabs`` table to ``paper/tables/<name>.tex``.

    A missing cell prints as an en dash rather than as ``NaN``.  Every table here
    has quantities that are only defined when something survived --- the mean
    effort of a method's usable plans, the ratio against a reference that has
    none --- and pandas' default would put the word ``NaN`` in a published table,
    which reads as a crash rather than as the absence it is.
    """
    CONFIG.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG.TABLES_DIR / f"{name}.tex"
    body = df.to_latex(
        index=False,
        escape=escape,
        na_rep=na_rep,
        float_format=lambda v: float_format % v,
        column_format=column_format or "l" * df.shape[1],
        caption=caption,
        label=label or f"tab:{name}",
        position="t",
    )
    path.write_text(body.strip() + "\n", encoding="utf-8")
    LOGGER.info("Saved LaTeX table -> %s", path)
    return path


def existing(paths: Sequence[str | Path]) -> list[Path]:
    """Filter ``paths`` down to the ones that exist on disk."""
    return [Path(p) for p in paths if Path(p).exists()]


__all__ = [
    "NumpyJSONEncoder",
    "save_json",
    "load_json",
    "update_json",
    "save_joblib",
    "load_joblib",
    "save_dataframe",
    "save_figure",
    "save_latex_table",
    "existing",
]
