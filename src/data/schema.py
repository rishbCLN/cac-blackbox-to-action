"""Declarative feature/dataset schemas shared by loaders, SCM and cost model.

The schema is the contract between the four layers of the pipeline:

* **data**      — which columns exist, their type and admissible range;
* **SCM**       — which nodes are manipulable / protected;
* **cost**      — which direction of change is allowed and how it is priced;
* **methods**   — bounds and integrality constraints of the search space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

FeatureKind = Literal["numeric", "ordinal", "categorical", "binary"]
Monotone = Literal["any", "increase", "decrease"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Everything the pipeline needs to know about one column."""

    name: str
    kind: FeatureKind
    #: Ordered category labels (``ordinal``/``categorical``/``binary`` only).
    levels: tuple[str, ...] | None = None
    #: Closed interval in *encoded* space (numeric features only).
    bounds: tuple[float, float] | None = None
    integer: bool = False
    unit: str = ""
    #: Protected / sensitive attribute (fairness reporting).
    protected: bool = False
    #: Hard actionability constraint: no intervention may set this feature.
    manipulable: bool = True
    #: Admissible direction of change for an intervention.
    monotone: Monotone = "any"
    description: str = ""

    @property
    def is_discrete(self) -> bool:
        return self.kind in ("ordinal", "categorical", "binary") or self.integer

    @property
    def n_levels(self) -> int:
        return len(self.levels) if self.levels else 0

    def encoded_bounds(self) -> tuple[float, float]:
        """Bounds of this feature in the encoded (analysis) space."""
        if self.levels:
            return (0.0, float(len(self.levels) - 1))
        if self.bounds is None:
            raise ValueError(f"feature {self.name!r} has neither levels nor bounds")
        return (float(self.bounds[0]), float(self.bounds[1]))


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Schema of a whole dataset."""

    name: str
    features: tuple[FeatureSpec, ...]
    target: str
    #: Human-readable meaning of ``target == 1``.
    positive_meaning: str
    #: Human-readable meaning of ``target == 0``.
    negative_meaning: str
    source: str = ""
    citation: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ views
    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def numeric_features(self) -> list[str]:
        return [f.name for f in self.features if f.kind == "numeric"]

    @property
    def categorical_features(self) -> list[str]:
        return [f.name for f in self.features if f.kind != "numeric"]

    @property
    def protected_features(self) -> list[str]:
        return [f.name for f in self.features if f.protected]

    @property
    def manipulable_features(self) -> list[str]:
        return [f.name for f in self.features if f.manipulable]

    @property
    def non_manipulable_features(self) -> list[str]:
        return [f.name for f in self.features if not f.manipulable]

    def get(self, name: str) -> FeatureSpec:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(f"{name!r} is not a feature of dataset {self.name!r}")

    def index(self, name: str) -> int:
        return self.feature_names.index(name)

    def subset(self, names: Sequence[str]) -> tuple[FeatureSpec, ...]:
        return tuple(self.get(n) for n in names)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target": self.target,
            "positive_meaning": self.positive_meaning,
            "negative_meaning": self.negative_meaning,
            "source": self.source,
            "citation": self.citation,
            "notes": list(self.notes),
            "features": [
                {
                    "name": f.name,
                    "kind": f.kind,
                    "levels": list(f.levels) if f.levels else None,
                    "bounds": list(f.bounds) if f.bounds else None,
                    "integer": f.integer,
                    "unit": f.unit,
                    "protected": f.protected,
                    "manipulable": f.manipulable,
                    "monotone": f.monotone,
                    "description": f.description,
                }
                for f in self.features
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        feats = tuple(
            FeatureSpec(
                name=f["name"],
                kind=f["kind"],
                levels=tuple(f["levels"]) if f.get("levels") else None,
                bounds=tuple(f["bounds"]) if f.get("bounds") else None,
                integer=bool(f.get("integer", False)),
                unit=f.get("unit", ""),
                protected=bool(f.get("protected", False)),
                manipulable=bool(f.get("manipulable", True)),
                monotone=f.get("monotone", "any"),
                description=f.get("description", ""),
            )
            for f in d["features"]
        )
        return cls(
            name=d["name"],
            features=feats,
            target=d["target"],
            positive_meaning=d.get("positive_meaning", ""),
            negative_meaning=d.get("negative_meaning", ""),
            source=d.get("source", ""),
            citation=d.get("citation", ""),
            notes=tuple(d.get("notes", ())),
        )


__all__ = ["FeatureSpec", "DatasetSpec", "FeatureKind", "Monotone"]
