"""Load the hand-crafted SCMs and expose them as validated graph objects.

Global rule #9 (*SCM first*): the causal model is an explicit, version-controlled
artefact — ``data/scm/<dataset>_scm.json`` — not something inferred inside a
method.  This module turns that JSON into a :class:`networkx.DiGraph` with node
attributes (interventional cost, manipulability, sensitivity, feature kind) and
cross-checks it against the data layer's :class:`~src.data.schema.DatasetSpec`,
so a graph that disagrees with the loaders fails fast instead of silently
producing nonsense counterfactuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

from src.config import CONFIG, DATASETS
from src.data.preprocess import dataset_spec
from src.data.schema import DatasetSpec
from src.utils.io import load_json
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: Qualitative effort labels used in the JSON, mapped to numeric multipliers.
#: ``infinite`` mirrors the 10.0 sentinel of the instruction's cost model (§7.1):
#: large enough to dominate any objective, finite enough to keep gradients sane.
COST_LEVELS: dict[str, float] = {
    "free": 0.0,
    "low": 0.10,
    "medium": 0.30,
    "high": 0.60,
    "very_high": 0.85,
    "infinite": 10.0,
}
INFINITE_LABEL = "infinite"
DEFAULT_LABEL = "medium"


class SCMError(ValueError):
    """Raised when an SCM JSON is internally inconsistent or contradicts the data."""


@dataclass(frozen=True, slots=True)
class SCM:
    """A directed acyclic causal graph plus its intervention semantics."""

    dataset: str
    graph: nx.DiGraph
    target: str
    non_manipulable: tuple[str, ...]
    sensitive: tuple[str, ...]
    costs: Mapping[str, float]
    cost_labels: Mapping[str, str]
    references: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    provenance: Mapping[str, Any] | None = None
    spec: DatasetSpec | None = None

    # ------------------------------------------------------------------ views
    @property
    def nodes(self) -> tuple[str, ...]:
        """All nodes, parents before children."""
        return tuple(nx.topological_sort(self.graph))

    @property
    def features(self) -> tuple[str, ...]:
        """Every node except the target, in topological order."""
        return tuple(n for n in self.nodes if n != self.target)

    @property
    def manipulable(self) -> tuple[str, ...]:
        """Features an applicant may act on directly."""
        return tuple(n for n in self.features if n not in self.non_manipulable)

    @property
    def actionable(self) -> tuple[str, ...]:
        """Manipulable features whose intervention cost is not infinite."""
        return tuple(n for n in self.manipulable if not self.is_infinite(n))

    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.graph.edges())

    # ------------------------------------------------------------- graph query
    def parents(self, node: str) -> tuple[str, ...]:
        return tuple(self.graph.predecessors(node))

    def children(self, node: str) -> tuple[str, ...]:
        return tuple(self.graph.successors(node))

    def ancestors(self, node: str) -> set[str]:
        return set(nx.ancestors(self.graph, node))

    def descendants(self, node: str) -> set[str]:
        return set(nx.descendants(self.graph, node))

    def downstream(
        self,
        nodes: str | Iterable[str],
        include_target: bool = False,
    ) -> tuple[str, ...]:
        """Strict descendants of ``nodes``, in topological order.

        These are the variables that *must* be recomputed after an intervention;
        a counterfactual that changes ``nodes`` and leaves them untouched is
        causally invalid by construction.
        """
        seeds = (nodes,) if isinstance(nodes, str) else tuple(nodes)
        affected: set[str] = set()
        for seed in seeds:
            affected |= self.descendants(seed)
        affected -= set(seeds)
        if not include_target:
            affected.discard(self.target)
        return tuple(n for n in self.nodes if n in affected)

    def is_root(self, node: str) -> bool:
        return self.graph.in_degree(node) == 0

    def is_derived(self, node: str) -> bool:
        """True when ``node`` has parents, i.e. it is (partly) an output."""
        return self.graph.in_degree(node) > 0

    # -------------------------------------------------------------------- cost
    def cost(self, node: str) -> float:
        return float(self.costs.get(node, COST_LEVELS[DEFAULT_LABEL]))

    def label(self, node: str) -> str:
        return str(self.cost_labels.get(node, DEFAULT_LABEL))

    def is_infinite(self, node: str) -> bool:
        return self.label(node) == INFINITE_LABEL

    # ----------------------------------------------------------------- exports
    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "target": self.target,
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "n_features": len(self.features),
            "n_manipulable": len(self.manipulable),
            "n_actionable": len(self.actionable),
            "n_sensitive": len(self.sensitive),
            "non_manipulable": list(self.non_manipulable),
            "sensitive": list(self.sensitive),
            "roots": [n for n in self.nodes if self.is_root(n)],
            "acyclic": nx.is_directed_acyclic_graph(self.graph),
        }

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable dictionary in the on-disk JSON schema."""
        return {
            "dataset": self.dataset,
            "target": self.target,
            "nodes": list(self.graph.nodes()),
            "edges": [{"from": u, "to": v, "type": d.get("type", "direct")}
                      for u, v, d in self.graph.edges(data=True)],
            "non_manipulable": list(self.non_manipulable),
            "sensitive": list(self.sensitive),
            "interventional_costs": dict(self.cost_labels),
            "references": list(self.references),
            "notes": list(self.notes),
            "provenance": dict(self.provenance or {}),
        }

    def to_mermaid(self, direction: str = "LR") -> str:
        """Mermaid ``graph`` source — the text form of the figure (§6.4.4)."""
        lines = [f"graph {direction}"]
        for node in self.nodes:
            marks = []
            if node in self.non_manipulable:
                marks.append("immutable")
            if node in self.sensitive:
                marks.append("sensitive")
            if node == self.target:
                marks.append("target")
            tag = f"<br/>({', '.join(marks)})" if marks else ""
            css = ("target" if node == self.target
                   else "immutable" if node in self.non_manipulable
                   else "sensitive" if node in self.sensitive
                   else "plain")
            lines.append(f'    {node}["{node}{tag}"]:::{css}')
        for u, v in self.graph.edges():
            lines.append(f"    {u} --> {v}")
        lines += [
            "    classDef plain fill:#eef2f7,stroke:#4a5b6c,color:#111;",
            "    classDef immutable fill:#f6d5d5,stroke:#a13333,color:#111;",
            "    classDef sensitive fill:#fdebc8,stroke:#a5741b,color:#111;",
            "    classDef target fill:#d5e8d4,stroke:#2f6b32,color:#111;",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"SCM(dataset={self.dataset!r}, nodes={self.graph.number_of_nodes()}, "
                f"edges={self.graph.number_of_edges()}, target={self.target!r})")


# --------------------------------------------------------------------- loading
def scm_path(dataset: str) -> Path:
    """``data/scm/<dataset>_scm.json`` — the layout required by the spec."""
    return CONFIG.SCM_DIR / f"{dataset}_scm.json"


def load_scm_json(dataset: str) -> dict[str, Any]:
    path = scm_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — the SCM is a required artefact.")
    return load_json(path)


def _build_graph(payload: Mapping[str, Any], dataset: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(payload["nodes"])
    for edge in payload["edges"]:
        src, dst = edge["from"], edge["to"]
        for end in (src, dst):
            if end not in graph:
                raise SCMError(f"{dataset}: edge endpoint {end!r} is not in 'nodes'")
        graph.add_edge(src, dst, type=edge.get("type", "direct"))
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph, orientation="original")
        raise SCMError(f"{dataset}: the causal graph has a cycle: {cycle}")
    return graph


def _check_against_spec(scm: SCM, spec: DatasetSpec) -> None:
    """Fail fast when the graph and the loaders disagree about the world."""
    graph_features, spec_features = set(scm.features), set(spec.feature_names)
    if graph_features != spec_features:
        raise SCMError(
            f"{scm.dataset}: SCM/loader feature mismatch — "
            f"only in SCM: {sorted(graph_features - spec_features)}, "
            f"only in loader: {sorted(spec_features - graph_features)}"
        )
    if scm.target != spec.target:
        raise SCMError(f"{scm.dataset}: SCM target {scm.target!r} != "
                       f"loader target {spec.target!r}")
    graph_fixed, spec_fixed = set(scm.non_manipulable), set(spec.non_manipulable_features)
    if graph_fixed != spec_fixed:
        raise SCMError(
            f"{scm.dataset}: non-manipulable sets disagree — SCM {sorted(graph_fixed)} "
            f"vs loader {sorted(spec_fixed)}"
        )
    protected = set(spec.protected_features)
    stray = set(scm.sensitive) - protected
    if stray:
        LOGGER.warning("%s: SCM marks %s sensitive but the loader does not flag them "
                       "as protected", scm.dataset, sorted(stray))


def build_scm(dataset: str, check_spec: bool = True) -> SCM:
    """Read, validate and assemble the SCM for ``dataset``."""
    payload = load_scm_json(dataset)
    target = payload["target"]
    graph = _build_graph(payload, dataset)

    if target not in graph:
        raise SCMError(f"{dataset}: target {target!r} is not a node")
    if graph.out_degree(target) > 0:
        raise SCMError(f"{dataset}: the target must be a sink, but {target!r} points to "
                       f"{list(graph.successors(target))}")

    non_manipulable = tuple(payload.get("non_manipulable", ()))
    sensitive = tuple(payload.get("sensitive", ()))
    for group, name in ((non_manipulable, "non_manipulable"), (sensitive, "sensitive")):
        unknown = [n for n in group if n not in graph]
        if unknown:
            raise SCMError(f"{dataset}: {name} lists unknown node(s) {unknown}")

    labels: dict[str, str] = dict(payload.get("interventional_costs", {}))
    unknown_labels = sorted(set(labels.values()) - set(COST_LEVELS))
    if unknown_labels:
        raise SCMError(f"{dataset}: unknown interventional cost label(s) {unknown_labels}; "
                       f"expected one of {sorted(COST_LEVELS)}")
    for node in graph.nodes():
        if node not in labels:
            LOGGER.warning("%s: no interventional cost for %r — defaulting to %r",
                           dataset, node, DEFAULT_LABEL)
            labels[node] = DEFAULT_LABEL
    for node in non_manipulable:
        if labels[node] != INFINITE_LABEL:
            LOGGER.warning("%s: %r is non-manipulable but priced %r — forcing %r",
                           dataset, node, labels[node], INFINITE_LABEL)
            labels[node] = INFINITE_LABEL
    costs = {node: COST_LEVELS[label] for node, label in labels.items()}

    spec = dataset_spec(dataset) if check_spec else None
    for node in graph.nodes():
        feature = spec.get(node) if (spec is not None and node != target) else None
        graph.nodes[node].update(
            cost=costs[node],
            cost_label=labels[node],
            manipulable=node not in non_manipulable and node != target,
            sensitive=node in sensitive,
            is_target=node == target,
            kind=getattr(feature, "kind", "target" if node == target else "unknown"),
            unit=getattr(feature, "unit", ""),
            monotone=getattr(feature, "monotone", "any"),
        )

    scm = SCM(
        dataset=dataset,
        graph=graph,
        target=target,
        non_manipulable=non_manipulable,
        sensitive=sensitive,
        costs=costs,
        cost_labels=labels,
        references=tuple(payload.get("references", ())),
        notes=tuple(payload.get("notes", ())),
        provenance=payload.get("provenance"),
        spec=spec,
    )
    if spec is not None:
        _check_against_spec(scm, spec)
    if not scm.actionable:
        raise SCMError(f"{dataset}: no actionable feature — recourse is impossible")
    LOGGER.info("SCM %s: %d nodes, %d edges, %d actionable of %d features",
                dataset, graph.number_of_nodes(), graph.number_of_edges(),
                len(scm.actionable), len(scm.features))
    return scm


@lru_cache(maxsize=None)
def get_scm(dataset: str, check_spec: bool = True) -> SCM:
    """Returns an SCM object with ``.graph``, ``.non_manipulable``, ``.sensitive``
    and ``.costs`` (§6.3).  Cached: the graph is immutable state."""
    return build_scm(dataset, check_spec=check_spec)


def all_scms(datasets: Sequence[str] | None = None) -> dict[str, SCM]:
    return {d: get_scm(d) for d in (datasets or DATASETS)}


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
]
