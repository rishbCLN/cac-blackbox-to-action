"""Does the suggestion survive contact with the world it describes?

A counterfactual is a promise: *change these things and the decision changes.*
Every method here can be checked against the black box, and every method passes
that check by construction — it searched until it did.  This module checks the
other half of the promise, the half no method that ignores causality can even
state: if the person actually does what they were told, and the world reacts the
way the fitted SCM says it does, is the resulting record still accepted?

The audit is Pearl's three steps applied to somebody else's answer.  Read the
proposal as the intervention it implicitly asks for — the features it moved —
recompute every descendant with the factual noise carried over
(:meth:`~src.scm.fcm.FCM.repair`), and you get *the world that would actually
result*.  Two numbers follow:

*The gap* — how far the proposal is from that world, per feature and normalised.
Zero means the method was describing a reachable state.  Non-zero means it was
describing a point in feature space that the causal structure forbids: the
applicant was told to hold education at one value and hours at another when the
graph says the first drags the second along.

*Survival* — whether the black box still returns the target class on the repaired
record.  This is the number that matters to a person, and it is strictly harder
than validity: a suggestion can flip the model at the point it names and lose the
flip once the consequences are filled in.  Nothing about the model changes here;
the record does.

Both are computed for every method by the same code, from the same fitted SCM, so
CAC's advantage — if it has one — shows up as a measurement rather than as an
assumption.  The queries spent on the repaired records are the evaluation's, not
the methods'.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.evaluation.metrics import Evaluator
from src.methods.base import Counterfactual
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: Columns :func:`assess` produces, in table order.
FAITHFULNESS_COLUMNS: tuple[str, ...] = (
    "dataset", "method", "record", "rank", "valid", "gap_max", "gap_mean",
    "n_off", "worst_feature", "proba_proposed", "proba_repaired", "proba_drop",
    "flip_survives", "promise_kept", "off_features",
)


def assess(evaluator: Evaluator,
           results: Mapping[int, Sequence[Counterfactual]],
           method: str | None = None) -> pd.DataFrame:
    """Audit one method's proposals against the world its own changes imply.

    Batched deliberately: every proposal for every record goes through one
    :meth:`~src.scm.fcm.FCM.repair` call and one ``predict_proba`` call, which is
    both faster and safer — per-row repair would make it possible for the audit
    of one record to see a different SCM state than the audit of another, and
    then a difference in the table could be an artefact of the loop.
    """
    flat: list[tuple[int, int, Counterfactual]] = [
        (int(record), rank, cf)
        for record, cfs in results.items()
        for rank, cf in enumerate(cfs)
    ]
    if not flat:
        return pd.DataFrame(columns=list(FAITHFULNESS_COLUMNS))

    X = np.asarray([cf.x_original for _, _, cf in flat], dtype=float)
    X_cf = np.asarray([cf.x_cf for _, _, cf in flat], dtype=float)
    targets = np.asarray([int(cf.y_target) for _, _, cf in flat])

    repaired = evaluator.fcm.repair(X, X_cf)
    gaps = np.abs(repaired - X_cf) / evaluator.span
    p_proposed = _p_target(evaluator, X_cf, targets)
    p_repaired = _p_target(evaluator, repaired, targets)
    threshold = float(CONFIG.DECISION_THRESHOLD)
    tolerance = float(CONFIG.CAUSAL_TOLERANCE)
    names = evaluator.feature_names

    rows: list[dict[str, Any]] = []
    for i, (record, rank, cf) in enumerate(flat):
        off = np.flatnonzero(gaps[i] > tolerance)
        valid = bool(p_proposed[i] >= threshold)
        survives = bool(p_repaired[i] >= threshold)
        rows.append({
            "dataset": evaluator.dataset,
            "method": method or cf.method,
            "record": record,
            "rank": rank,
            "valid": valid,
            "gap_max": float(gaps[i].max()),
            "gap_mean": float(gaps[i].mean()),
            "n_off": int(off.size),
            "worst_feature": names[int(np.argmax(gaps[i]))] if gaps[i].max() > 0 else "",
            "proba_proposed": float(p_proposed[i]),
            "proba_repaired": float(p_repaired[i]),
            "proba_drop": float(p_proposed[i] - p_repaired[i]),
            "flip_survives": survives,
            # The whole promise: it works, and it works for the stated reason.
            "promise_kept": bool(valid and survives and gaps[i].max() <= tolerance),
            "off_features": ";".join(names[int(j)] for j in off),
        })
    return pd.DataFrame(rows)[list(FAITHFULNESS_COLUMNS)]


def _p_target(evaluator: Evaluator, X: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """``P(y = y_target)`` per row, in one call.

    ``predict_proba`` returns ``P(y = 1)`` in column 1, so a record whose target
    is the negative class reads the complement.  Both appear in this project —
    §9's main tables ask for the positive class, the robustness experiment flips
    the question — so the conversion lives in one place.
    """
    proba = np.asarray(evaluator.model.predict_proba(X), dtype=float)[:, 1]
    return np.where(targets == 1, proba, 1.0 - proba)


#: The per-method faithfulness table §9 reports.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "dataset", "method", "n_cfs", "n_valid", "causal_validity", "gap_max_mean",
    "gap_max_p90", "n_off_mean", "flip_survival", "promise_kept",
    "proba_drop_mean", "proba_drop_max",
)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, method) — the paper's faithfulness claim, quantified.

    Rates are over *valid* counterfactuals, not over all of them, and the
    denominator is printed beside each: the question "of the suggestions that
    work, how many still work once the world reacts?" is only interesting once
    the suggestion works at all, and a method that produced few valid ones should
    not look reliable because it has little to be wrong about.  Read
    ``flip_survival`` next to ``n_valid``, never on its own.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    rows: list[dict[str, Any]] = []
    tolerance = float(CONFIG.CAUSAL_TOLERANCE)
    for (dataset, method), group in frame.groupby(["dataset", "method"], sort=True):
        valid = group[group["valid"].astype(bool)]
        rows.append({
            "dataset": dataset,
            "method": method,
            "n_cfs": int(group.shape[0]),
            "n_valid": int(valid.shape[0]),
            "causal_validity": _rate(valid["gap_max"] <= tolerance, valid.shape[0]),
            "gap_max_mean": _mean(valid["gap_max"]),
            "gap_max_p90": float(valid["gap_max"].quantile(0.90))
                           if not valid.empty else float("nan"),
            "n_off_mean": _mean(valid["n_off"]),
            "flip_survival": _rate(valid["flip_survives"], valid.shape[0]),
            "promise_kept": _rate(valid["promise_kept"], valid.shape[0]),
            "proba_drop_mean": _mean(valid["proba_drop"]),
            "proba_drop_max": float(valid["proba_drop"].max())
                              if not valid.empty else float("nan"),
        })
    return pd.DataFrame(rows)[list(SUMMARY_COLUMNS)]


def blamed_features(frame: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Which features methods most often get causally wrong.

    A diagnostic, and a figure: the features that appear here are the ones whose
    consequences the graph considers strong, so a method that treats coordinates
    as independent will keep tripping over the same handful.  Counted over valid
    counterfactuals only, since an invalid one's causal error is moot.
    """
    valid = frame[frame["valid"].astype(bool)]
    rows: list[dict[str, Any]] = []
    for (dataset, method), group in valid.groupby(["dataset", "method"], sort=True):
        counts: dict[str, int] = {}
        for listing in group["off_features"]:
            for name in str(listing).split(";"):
                if name:
                    counts[name] = counts.get(name, 0) + 1
        total = max(int(group.shape[0]), 1)
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1])[:top]:
            rows.append({"dataset": dataset, "method": method, "feature": name,
                         "n_off": count, "share": count / total})
    return pd.DataFrame(rows)


def _rate(flags: "pd.Series[Any]", denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(np.asarray(flags, dtype=bool).sum()) / float(denominator)


def _mean(series: "pd.Series[Any]") -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if values.size else float("nan")


__all__ = [
    "FAITHFULNESS_COLUMNS",
    "SUMMARY_COLUMNS",
    "assess",
    "blamed_features",
    "summarise",
]
