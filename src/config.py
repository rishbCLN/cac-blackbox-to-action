"""Single source of truth for paths, seeds and hyper-parameters.

Every module imports :data:`CONFIG` instead of hard-coding constants or paths
(global rule #3).  Values can be overridden from the environment using the
``CAC_`` prefix, e.g. ``CAC_RANDOM_STATE=7`` or via a local ``.env`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Canonical dataset keys used across the whole code base.
DATASETS: tuple[str, ...] = ("adult", "german", "heloc", "diabetes")

#: Canonical black-box model keys.
MODEL_TYPES: tuple[str, ...] = ("xgb", "mlp")


class Config(BaseSettings):
    """Project configuration.  Immutable at runtime (frozen)."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_prefix="CAC_",
        extra="ignore",
        frozen=True,
    )

    # ------------------------------------------------------------------ paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
    SCM_DIR: Path = PROJECT_ROOT / "data" / "scm"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    FIGURES_DIR: Path = PROJECT_ROOT / "paper" / "figures"
    TABLES_DIR: Path = PROJECT_ROOT / "paper" / "tables"
    LOG_DIR: Path = PROJECT_ROOT / "logs"

    # ----------------------------------------------------------- determinism
    RANDOM_STATE: int = 42

    # ------------------------------------------------------------------ data
    TEST_SIZE: float = 0.2
    #: Rows sub-sampled from very large datasets to keep the pipeline fast.
    MAX_ROWS: int = 20_000

    # ----------------------------------------------------------- black box
    BLACKBOX_TYPE: Literal["xgb", "mlp"] = "xgb"
    #: Minimum test AUC per dataset; below this ``train.py`` runs a search.
    AUC_TARGETS: dict[str, float] = Field(
        default_factory=lambda: {"adult": 0.90, "german": 0.75, "heloc": 0.75,
                                "diabetes": 0.80}
    )
    TUNE_N_ITER: int = 50
    TUNE_CV: int = 3
    #: Probability above which the positive class is predicted.
    DECISION_THRESHOLD: float = 0.5

    # --------------------------------------------------------------- methods
    N_COUNTERFACTUALS: int = 5
    #: Number of negatively-classified test instances explained per dataset.
    N_QUERY_INSTANCES: int = 100
    #: Maximum number of simultaneous *actions* (interventions) CAC may take.
    MAX_ACTIONS: int = 3

    # CAC search budget.  There is no gradient (the model is a black box), so
    # ``CAC_LEARNING_RATE`` is used as the relative mutation step size.
    CAC_MAX_ITERS: int = 200
    CAC_LEARNING_RATE: float = 0.1
    CAC_POPULATION: int = 60
    #: Objective weights: causal validity / cost / feasibility / plausibility.
    CAC_LAMBDA_VALID: float = 1.0
    CAC_LAMBDA_COST: float = 0.6
    CAC_LAMBDA_FEAS: float = 0.3
    CAC_LAMBDA_PLAUS: float = 0.2
    #: Extra probability margin required beyond the decision threshold.
    CAC_TARGET_MARGIN: float = 0.05

    # Baselines
    WACHTER_LAMBDA: float = 1.0
    WACHTER_RESTARTS: int = 8
    WACHTER_MAX_ITERS: int = 300
    RANDOM_N_SAMPLES: int = 1_000
    DICE_METHOD: Literal["random", "genetic", "kdtree"] = "random"
    #: Give baselines the immutability constraint too (strong-baseline setting).
    BASELINES_RESPECT_IMMUTABLE: bool = True

    # ------------------------------------------------------------ evaluation
    #: Tolerance (in std-devs) for judging structural consistency of a CF.
    CAUSAL_TOLERANCE: float = 0.10
    KNN_K: int = 5
    #: Seeds used to retrain the black box for faithfulness/robustness checks.
    ROBUSTNESS_SEEDS: tuple[int, ...] = (0, 1, 2)
    #: Seeds handed to the stochastic searches for the significance study (§9.6).
    #: The first is the canonical run every other table uses, so a single-seed
    #: result is a strict subset of the multi-seed one.
    SIGNIFICANCE_SEEDS: tuple[int, ...] = (42, 0, 1, 2, 3)

    # --------------------------------------------------------------- logging
    LOG_LEVEL: str = "INFO"

    # ----------------------------------------------------------------- figures
    FIGURE_DPI: int = 300
    FIGURE_FORMATS: tuple[str, ...] = ("png", "pdf")

    @field_validator("LOG_LEVEL")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    # ------------------------------------------------------------------ utils
    def ensure_dirs(self) -> None:
        """Create every output directory the pipeline writes to."""
        for p in (
            self.DATA_RAW,
            self.DATA_PROCESSED,
            self.SCM_DIR,
            self.MODELS_DIR,
            self.RESULTS_DIR,
            self.FIGURES_DIR,
            self.TABLES_DIR,
            self.LOG_DIR,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def auc_target(self, dataset: str) -> float:
        return self.AUC_TARGETS.get(dataset, 0.75)


CONFIG = Config()
CONFIG.ensure_dirs()

__all__ = ["CONFIG", "Config", "PROJECT_ROOT", "DATASETS", "MODEL_TYPES"]
