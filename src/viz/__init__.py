"""Publication-ready plotting (global rule #12).

Every figure in the paper goes through :func:`set_style` and
:func:`src.utils.io.save_figure`, so all of them share one font stack, one
palette and one export path (300 DPI PNG + vector PDF).
"""

from __future__ import annotations

from typing import Any

#: One palette for the whole paper.  Method colours are fixed so that a reader
#: can carry the mapping from one figure to the next.
PALETTE: dict[str, str] = {
    "cac": "#1b5e8c",
    "cac_nocausal": "#6fa8c7",
    "dice": "#c0392b",
    "wachter": "#e08c1a",
    "random": "#7f8c8d",
    "oracle": "#2e7d32",
}
NEUTRAL = "#4a5b6c"
ACCENT = "#1b5e8c"
WARN = "#a13333"
OK = "#2f6b32"

#: Node roles in the SCM figures.
ROLE_COLOURS: dict[str, str] = {
    "plain": "#eef2f7",
    "immutable": "#f6d5d5",
    "sensitive": "#fdebc8",
    "target": "#d5e8d4",
}
ROLE_EDGES: dict[str, str] = {
    "plain": NEUTRAL,
    "immutable": WARN,
    "sensitive": "#a5741b",
    "target": OK,
}

_STYLED = False


def set_style(force: bool = False) -> None:
    """Apply the paper's matplotlib style once per process."""
    global _STYLED  # noqa: PLW0603 - a module-level latch is the point
    if _STYLED and not force:
        return
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg", force=False)  # headless by default; notebooks override
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.8,
        "pdf.fonttype": 42,  # editable text in the vector output
        "ps.fonttype": 42,
    })
    _STYLED = True


def method_colour(method: str, default: str = NEUTRAL) -> str:
    """Stable colour for a method name (case- and suffix-insensitive)."""
    key = str(method).strip().lower().replace("-", "_")
    if key in PALETTE:
        return PALETTE[key]
    for name, colour in PALETTE.items():
        if key.startswith(name):
            return colour
    return default


def label_of(name: str) -> str:
    """Display label for a method or dataset key."""
    labels: dict[str, str] = {
        "cac": "CAC",
        "cac_nocausal": "CAC (no causal)",
        "dice": "DiCE",
        "wachter": "Wachter",
        "random": "Random",
        "oracle": "Cost oracle",
        "adult": "Adult",
        "german": "German",
        "heloc": "HELOC",
        "diabetes": "Diabetes",
        "xgb": "XGBoost",
        "mlp": "MLP",
    }
    return labels.get(str(name).strip().lower(), str(name))


def figure(*args: Any, **kwargs: Any):
    """``plt.subplots`` with the paper style already applied."""
    set_style()
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt.subplots(*args, **kwargs)


__all__ = [
    "PALETTE",
    "ROLE_COLOURS",
    "ROLE_EDGES",
    "NEUTRAL",
    "ACCENT",
    "WARN",
    "OK",
    "set_style",
    "method_colour",
    "label_of",
    "figure",
]
