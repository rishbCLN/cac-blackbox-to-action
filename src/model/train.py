"""Train and evaluate the black boxes (experiment 01's engine).

Acceptance criteria from the project spec (§5.2): test AUC ≥ 0.90 on Adult and
≥ 0.75 on German Credit and HELOC.  If the default hyper-parameters miss the
target, a ``RandomizedSearchCV`` (50 draws, 3-fold, fixed seed) is run and the
better model is kept.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from src.config import CONFIG, DATASETS, MODEL_TYPES
from src.data.preprocess import DatasetBundle, build_dataset
from src.model.blackbox import BlackBoxModel, model_path
from src.utils.io import update_json
from src.utils.logging import get_logger
from src.utils.seeding import set_global_seed

LOGGER = get_logger(__name__)

METRICS_PATH = CONFIG.MODELS_DIR / "metrics.json"

XGB_SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "gamma": [0.0, 0.1, 0.5],
}
MLP_SEARCH_SPACE: dict[str, list[Any]] = {
    "hidden_layer_sizes": [(64, 32), (128, 64), (64,), (128,), (32, 16)],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "learning_rate_init": [1e-3, 3e-3, 1e-2],
    "max_iter": [300, 600],
}


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    """Full metric report for one model on one split."""
    pred = (proba >= CONFIG.DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def _cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(CONFIG.TUNE_CV, shuffle=True, random_state=CONFIG.RANDOM_STATE)


def _fit_quietly(model: BlackBoxModel, X: np.ndarray, y: np.ndarray, tag: str) -> BlackBoxModel:
    """Fit, folding sklearn's per-restart ``ConvergenceWarning``s into one log line."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    hit_cap = 0
    for entry in caught:
        if issubclass(entry.category, ConvergenceWarning):
            hit_cap += 1
        else:  # never swallow anything that is not about the iteration cap
            warnings.warn_explicit(entry.message, entry.category, entry.filename, entry.lineno)
    if hit_cap:
        LOGGER.warning("%s: the optimiser hit its iteration cap (%d warning(s)); metrics "
                       "below are for the capped model", tag, hit_cap)
    return model


def _tune(bundle: DatasetBundle, model_type: str) -> tuple[dict[str, Any], float, float]:
    """Randomised search over the raw estimator (training-time, not explanation).

    Returns ``(best_params, cv_auc_tuned, cv_auc_default)``.  Both scores come from
    the *same* cross-validation of the training split, so the choice between the
    default and the tuned configuration never looks at the test set.
    """
    space = XGB_SEARCH_SPACE if model_type == "xgb" else MLP_SEARCH_SPACE
    base = BlackBoxModel._build(model_type)  # noqa: SLF001 - same module family
    if model_type == "xgb":
        base.set_params(n_jobs=2)
    cv = _cv_splitter()
    X_train_model = bundle.transformer.to_model_space(bundle.X_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        default_cv = float(np.mean(cross_val_score(
            base, X_train_model, bundle.y_train, cv=cv, scoring="roc_auc", n_jobs=2,
        )))
        search = RandomizedSearchCV(
            base,
            param_distributions=space,
            n_iter=CONFIG.TUNE_N_ITER,
            cv=cv,
            scoring="roc_auc",
            random_state=CONFIG.RANDOM_STATE,
            n_jobs=2,
            refit=False,
            error_score="raise",
        )
        search.fit(X_train_model, bundle.y_train)

    LOGGER.info("Tuned %s/%s: cv AUC %.4f (default %.4f) with %s", bundle.name, model_type,
                search.best_score_, default_cv, search.best_params_)
    return dict(search.best_params_), float(search.best_score_), default_cv


def train_dataset(
    dataset: str,
    model_type: str = "xgb",
    tune_if_below_target: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Train one black box, persist it, and return its metric report."""
    set_global_seed()
    bundle = build_dataset(dataset, force=force)
    target_auc = CONFIG.auc_target(dataset)
    tag = f"{dataset}/{model_type}"

    model = BlackBoxModel(model_type, transformer=bundle.transformer, dataset=dataset)
    _fit_quietly(model, bundle.X_train, bundle.y_train, tag)
    test = evaluate_predictions(bundle.y_test, model.proba_positive(bundle.X_test))
    train = evaluate_predictions(bundle.y_train, model.proba_positive(bundle.X_train))
    tuned_params: dict[str, Any] = {}
    selection: dict[str, Any] = {}

    if tune_if_below_target and test["roc_auc"] < target_auc:
        LOGGER.warning("%s AUC %.4f < target %.2f — running RandomizedSearchCV "
                       "(%d draws, %d-fold)", tag, test["roc_auc"], target_auc,
                       CONFIG.TUNE_N_ITER, CONFIG.TUNE_CV)
        params, cv_tuned, cv_default = _tune(bundle, model_type)
        selection = {
            "n_iter": CONFIG.TUNE_N_ITER,
            "cv_folds": CONFIG.TUNE_CV,
            "cv_auc_default": cv_default,
            "cv_auc_tuned": cv_tuned,
            "kept": "tuned" if cv_tuned > cv_default else "default",
        }
        if cv_tuned > cv_default:
            candidate = BlackBoxModel(model_type, transformer=bundle.transformer,
                                      dataset=dataset, **params)
            _fit_quietly(candidate, bundle.X_train, bundle.y_train, f"{tag} (tuned)")
            model, tuned_params = candidate, params
            test = evaluate_predictions(bundle.y_test, model.proba_positive(bundle.X_test))
            train = evaluate_predictions(bundle.y_train, model.proba_positive(bundle.X_train))
            LOGGER.info("%s: kept the tuned configuration (cv AUC %.4f > %.4f)",
                        tag, cv_tuned, cv_default)
        else:
            LOGGER.info("%s: search did not beat the defaults on cv AUC "
                        "(%.4f <= %.4f); keeping the defaults", tag, cv_tuned, cv_default)

    path = model.save(model_path(dataset, model_type))
    report = {
        "dataset": dataset,
        "model_type": model_type,
        "path": str(path.relative_to(CONFIG.PROJECT_ROOT)),
        "n_features": len(bundle.feature_names),
        "feature_names": bundle.feature_names,
        "target_auc": target_auc,
        "meets_target": bool(test["roc_auc"] >= target_auc),
        "tuned_params": tuned_params,
        "model_selection": selection,
        "test": test,
        "train": train,
        "is_synthetic_data": bundle.is_synthetic,
        "random_state": CONFIG.RANDOM_STATE,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    status = "PASS" if report["meets_target"] else "BELOW TARGET"
    LOGGER.info("%s: AUC=%.4f F1=%.4f acc=%.4f [%s]", tag,
                test["roc_auc"], test["f1"], test["accuracy"], status)
    update_json(METRICS_PATH, {f"{model_type}_{dataset}": report})
    return report


def train_all(
    datasets: tuple[str, ...] = tuple(DATASETS),
    model_types: tuple[str, ...] = tuple(MODEL_TYPES),
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Train every (dataset, model) pair and collect the reports.

    The defaults come from the config rather than being spelled out here: a
    literal list silently stopped covering :data:`~src.config.DATASETS` the moment
    a fourth dataset was added, and "train everything" that quietly trains most of
    it is worse than one that fails.
    """
    reports: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        for model_type in model_types:
            key = f"{model_type}_{dataset}"
            reports[key] = train_dataset(dataset, model_type, force=force)
    return reports


__all__ = [
    "train_dataset",
    "train_all",
    "evaluate_predictions",
    "METRICS_PATH",
    "XGB_SEARCH_SPACE",
    "MLP_SEARCH_SPACE",
]
