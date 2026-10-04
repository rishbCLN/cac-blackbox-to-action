"""SCM figures: the causal graph as the reader sees it (§6.4.4).

The instruction asks for a Mermaid diagram at ``paper/figures/scm_<dataset>.png``.
Mermaid renders in a browser, and this project must build offline with no Node
toolchain, so :func:`save_scm_figure` writes **both**:

* ``paper/figures/scm_<dataset>.png`` / ``.pdf`` — a matplotlib rendering with the
  same layering and colour code, which is what the paper includes;
* ``paper/figures/scm_<dataset>.mmd`` — the Mermaid source, so the diagram can be
  regenerated in any Mermaid-aware viewer without this repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np

from src.config import CONFIG, DATASETS
from src.scm.define_scm import SCM, get_scm
from src.utils.io import save_figure
from src.utils.logging import get_logger
from src.viz import ROLE_COLOURS, ROLE_EDGES, figure, set_style

LOGGER = get_logger(__name__)


def _role(scm: SCM, node: str) -> str:
    if node == scm.target:
        return "target"
    if node in scm.non_manipulable:
        return "immutable"
    if node in scm.sensitive:
        return "sensitive"
    return "plain"


def _layered_positions(scm: SCM) -> dict[str, tuple[float, float]]:
    """Left-to-right layers from the graph's topological generations."""
    generations = list(nx.topological_generations(scm.graph))
    pos: dict[str, tuple[float, float]] = {}
    height = max(len(layer) for layer in generations)
    for x, layer in enumerate(generations):
        ordered = sorted(layer, key=lambda n: (_role(scm, n) != "immutable", n))
        offset = (height - len(ordered)) / 2.0
        for y, node in enumerate(ordered):
            pos[node] = (float(x), float(y) + offset)
    return pos


def _wrap(name: str, width: int = 13) -> str:
    """Break ``snake_case`` / ``CamelCase`` names onto short lines."""
    parts: list[str] = []
    for chunk in str(name).split("_"):
        start = 0
        for i in range(1, len(chunk)):
            if chunk[i].isupper() and not chunk[i - 1].isupper():
                parts.append(chunk[start:i])
                start = i
        parts.append(chunk[start:])
    lines, current = [], ""
    for part in [p for p in parts if p]:
        candidate = f"{current} {part}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def plot_scm(scm: SCM, ax: Any | None = None, show_costs: bool = True) -> Any:
    """Draw one causal graph; returns the figure."""
    set_style()
    pos = _layered_positions(scm)
    layers = int(max(x for x, _ in pos.values())) + 1
    rows = max(y for _, y in pos.values()) + 1
    if ax is None:
        fig, ax = figure(figsize=(1.9 * layers + 1.2, 0.92 * rows + 1.1))
    else:
        fig = ax.figure

    nx.draw_networkx_edges(
        scm.graph, pos, ax=ax, arrows=True, arrowsize=7, width=0.7,
        edge_color="#8795a3", alpha=0.75, node_size=2400,
        connectionstyle="arc3,rad=0.07", min_source_margin=6, min_target_margin=8,
    )
    for node, (x, y) in pos.items():
        role = _role(scm, node)
        label = _wrap(node)
        if show_costs and node != scm.target:
            label = f"{label}\n[{scm.label(node)}]"
        ax.text(
            x, y, label, ha="center", va="center", fontsize=6.4, zorder=3,
            linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.34", facecolor=ROLE_COLOURS[role],
                      edgecolor=ROLE_EDGES[role], linewidth=0.9),
        )

    ax.set_axis_off()
    ax.set_xlim(-0.6, layers - 0.4)
    ax.set_ylim(-0.75, rows - 0.25)
    ax.invert_yaxis()
    ax.set_title(f"{scm.dataset}: causal graph "
                 f"({scm.graph.number_of_nodes()} nodes, "
                 f"{scm.graph.number_of_edges()} edges)")
    _role_legend(ax)
    return fig


def _role_legend(ax: Any) -> None:
    from matplotlib.patches import Patch  # noqa: PLC0415

    handles = [
        Patch(facecolor=ROLE_COLOURS["plain"], edgecolor=ROLE_EDGES["plain"],
              label="actionable feature"),
        Patch(facecolor=ROLE_COLOURS["immutable"], edgecolor=ROLE_EDGES["immutable"],
              label="non-manipulable"),
        Patch(facecolor=ROLE_COLOURS["sensitive"], edgecolor=ROLE_EDGES["sensitive"],
              label="sensitive"),
        Patch(facecolor=ROLE_COLOURS["target"], edgecolor=ROLE_EDGES["target"],
              label="prediction target"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5,
              bbox_to_anchor=(0.5, -0.06), handlelength=1.1, handleheight=0.9,
              columnspacing=1.0)


def save_scm_figure(dataset: str | SCM, show_costs: bool = True) -> list[Path]:
    """Write ``scm_<dataset>.{png,pdf}`` and the Mermaid source beside them."""
    scm = dataset if isinstance(dataset, SCM) else get_scm(dataset)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig = plot_scm(scm, show_costs=show_costs)
    paths = save_figure(fig, f"scm_{scm.dataset}")
    plt.close(fig)

    mermaid = CONFIG.FIGURES_DIR / f"scm_{scm.dataset}.mmd"
    mermaid.write_text(scm.to_mermaid() + "\n", encoding="utf-8")
    LOGGER.info("Wrote Mermaid source -> %s", mermaid)
    return [*paths, mermaid]


def plot_all_scms(datasets: Sequence[str] | None = None) -> dict[str, list[Path]]:
    return {d: save_scm_figure(d) for d in (datasets or DATASETS)}


def plot_intervention_paths(
    scm: SCM,
    actions: Iterable[str],
    ax: Any | None = None,
) -> Any:
    """Highlight one action set and everything it drags along with it.

    This is the figure that makes the paper's core point visually: the changed
    features are not the whole story, the downstream cascade is.
    """
    set_style()
    acted = tuple(actions)
    affected = set(scm.downstream(acted, include_target=True))
    pos = _layered_positions(scm)
    layers = int(max(x for x, _ in pos.values())) + 1
    rows = max(y for _, y in pos.values()) + 1
    if ax is None:
        fig, ax = figure(figsize=(1.9 * layers + 1.2, 0.92 * rows + 1.1))
    else:
        fig = ax.figure

    live = [(u, v) for u, v in scm.graph.edges()
            if u in set(acted) | affected and v in affected]
    dead = [e for e in scm.graph.edges() if e not in set(live)]
    for edges, colour, width, alpha in ((dead, "#c7ced6", 0.6, 0.6),
                                        (live, "#1b5e8c", 1.5, 0.95)):
        nx.draw_networkx_edges(
            scm.graph, pos, edgelist=edges, ax=ax, arrows=True, arrowsize=8,
            width=width, edge_color=colour, alpha=alpha, node_size=2400,
            connectionstyle="arc3,rad=0.07", min_source_margin=6, min_target_margin=8,
        )
    for node, (x, y) in pos.items():
        if node in acted:
            face, edge, weight = "#bcd9ee", "#1b5e8c", "bold"
        elif node in affected:
            face, edge, weight = "#e8f1f8", "#4a86b0", "normal"
        else:
            face, edge, weight = "#f4f6f8", "#c7ced6", "normal"
        ax.text(x, y, _wrap(node), ha="center", va="center", fontsize=6.4,
                zorder=3, linespacing=1.15, fontweight=weight,
                bbox=dict(boxstyle="round,pad=0.34", facecolor=face,
                          edgecolor=edge, linewidth=0.9))
    ax.set_axis_off()
    ax.set_xlim(-0.6, layers - 0.4)
    ax.set_ylim(-0.75, rows - 0.25)
    ax.invert_yaxis()
    ax.set_title(f"{scm.dataset}: acting on {', '.join(acted)} moves "
                 f"{len(affected)} further variable(s)")
    return fig


__all__ = [
    "plot_scm",
    "save_scm_figure",
    "plot_all_scms",
    "plot_intervention_paths",
]
