"""Pima Indians Diabetes loader — fourth dataset, the clinical setting.

Sources tried in order: local CSV, Kaggle (``uciml/pima-indians-diabetes-database``,
``mathchi/diabetes-data-set``), a public unauthenticated mirror, and finally a
deterministic synthetic surrogate so the pipeline is *always* runnable offline
(global rule #1/#7).  Whether the surrogate was used is recorded in
``data/processed/diabetes_metadata.json`` as ``is_synthetic``.

Two things about this dataset are worth stating up front, because both bear on
how its numbers should be read.

**The label is inverted.**  Pima's ``Outcome`` is 1 for *diabetes within five
years*, i.e. the bad news.  Every other dataset here uses 1 for the outcome the
applicant wants (``income`` > 50K, ``credit_risk`` good, ``RiskPerformance``
good), and the whole evaluation layer assumes ``y_target=1`` is the favourable
flip.  So the target is stored as ``no_diabetes = 1 - Outcome`` rather than
special-casing the sign downstream.

**Most of what is actionable here is a measurement, not a lever.**  Of the eight
variables, five cannot be intervened on at all: ``Age``, ``Pregnancies`` and
``DiabetesPedigreeFunction`` are history or genetics, and ``SkinThickness`` and
``Insulin`` are *readings* — a triceps fold and a serum concentration are what a
body does, not something a person can be told to do.  They still move, but only
as consequences of ``BMI`` and ``Glucose`` through the SCM.  That makes them the
sharpest causal-validity test cases in the project: a point-based method will
cheerfully advise "reduce your 2-hour serum insulin by 40 µU/ml", which is not
an action, and CAC structurally cannot.

The canonical Pima file encodes missing measurements as 0 (impossible for
glucose, blood pressure, skin fold, insulin and BMI).  ``load_diabetes`` treats
those zeros as missing and median-imputes them, which is the standard treatment.

Most circulating copies — including Kaggle's — are *already* imputed, and several
were imputed with **class-conditional** means or medians, which writes the label
into the feature.  In the copy this project was given, ``Insulin`` is 102.5 on 236
rows (every one of them negative) and 169.5 on 138 rows (every one positive):
between them, 374 of 768 rows carry their own answer in one column, and an
XGBoost trained on it reaches AUC 0.96 against a literature norm of ~0.83.  That
is not a model of diabetes, it is a model of who did the imputing, and every
recourse number computed from it would be fiction.  :func:`_deleak` finds those
fill values and puts them back to missing before imputing unconditionally; what
it removes is recorded in the returned spec's notes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.data import sources
from src.data.schema import DatasetSpec, FeatureSpec
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

RAW_DIR = CONFIG.DATA_RAW / "diabetes"
SOURCE_TARGET = "Outcome"
TARGET = "no_diabetes"
MIRROR = ("https://raw.githubusercontent.com/plotly/datasets/master/"
          "diabetes.csv")
#: Columns where 0 is physiologically impossible and therefore means "missing".
ZERO_IS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
#: Leak detector (see :func:`_deleak`).  A fill value has to clear *both* of the
#: frequency thresholds and the purity one, or be fractional in a whole-number
#: column; the numbers are loose enough that no genuine reading in this file
#: trips them and tight enough to catch all five artefacts.
LEAK_MIN_COUNT = 15
LEAK_SPIKE_FACTOR = 3.0
LEAK_PURITY = 0.15

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("Pregnancies", "numeric", bounds=(0, 20), integer=True,
                unit="births", protected=True, manipulable=False,
                description="Number of times pregnant — reproductive history, and "
                            "in an all-female cohort a proxy for it."),
    FeatureSpec("Glucose", "numeric", bounds=(40, 200), integer=True,
                unit="mg/dl", monotone="decrease",
                description="Plasma glucose at 2h in an oral tolerance test. "
                            "Movable by diet, exercise and medication."),
    FeatureSpec("BloodPressure", "numeric", bounds=(24, 122), integer=True,
                unit="mm Hg", monotone="decrease",
                description="Diastolic blood pressure — treatable, and the one "
                            "lever with a same-week response."),
    FeatureSpec("SkinThickness", "numeric", bounds=(7, 99), integer=True,
                unit="mm", manipulable=False,
                description="Triceps skin-fold thickness: a *measurement* of "
                            "subcutaneous fat, so it follows BMI rather than "
                            "being set."),
    FeatureSpec("Insulin", "numeric", bounds=(14, 846), integer=True,
                unit="mu U/ml", manipulable=False,
                description="2-hour serum insulin: the body's response to the "
                            "glucose load, not an action a person can take."),
    FeatureSpec("BMI", "numeric", bounds=(18.0, 68.0), unit="kg/m^2",
                monotone="decrease",
                description="Body mass index — the substantive intervention, and "
                            "the expensive one."),
    FeatureSpec("DiabetesPedigreeFunction", "numeric", bounds=(0.05, 2.5),
                unit="score", protected=True, manipulable=False,
                description="Family-history score: fixed at birth."),
    FeatureSpec("Age", "numeric", bounds=(21, 81), integer=True, unit="years",
                protected=True, manipulable=False, monotone="increase",
                description="Age in years."),
)

SPEC = DatasetSpec(
    name="diabetes",
    features=FEATURES,
    target=TARGET,
    positive_meaning="no_diabetes = 1 (no onset within five years)",
    negative_meaning="no_diabetes = 0 (diabetes within five years)",
    source="Pima Indians Diabetes (UCI / NIDDK); Kaggle uciml/pima-indians-diabetes-database",
    citation=("Smith, J. W. et al. (1988). Using the ADAP learning algorithm to "
              "forecast the onset of diabetes mellitus. Proc. Symp. Computer "
              "Applications in Medical Care, 261-265."),
    notes=(
        "The label is inverted relative to the source file: no_diabetes = 1 - Outcome, "
        "so that 1 is the favourable class as in every other dataset here.",
        "Age, Pregnancies and DiabetesPedigreeFunction are non-manipulable and "
        "protected; SkinThickness and Insulin are non-manipulable because they are "
        "measurements of state, not actions.",
        "Three features are actionable: BMI, Glucose and BloodPressure.",
        "This is a clinical dataset used here as a recourse benchmark, not a "
        "diagnostic tool; nothing it produces is medical advice.",
    ),
)


def _synthetic(n: int = 3_000) -> pd.DataFrame:
    """Deterministic surrogate with Pima's schema and causal ordering."""
    rng = np.random.default_rng(CONFIG.RANDOM_STATE)
    age = rng.gamma(6.0, 5.5, n).clip(21, 81)
    pedigree = rng.gamma(2.0, 0.24, n).clip(0.05, 2.5)
    pregnancies = (0.11 * (age - 21) * rng.beta(2, 3, n) * 3).round().clip(0, 17)
    bmi = (24.0 + 0.10 * (age - 21) + 0.35 * pregnancies
           + rng.normal(0, 5.5, n)).clip(18.0, 67.1)
    skin = (0.85 * bmi + 0.05 * (age - 21) + rng.normal(0, 6.0, n)).clip(7, 99)
    bp = (52.0 + 0.55 * bmi + 0.22 * (age - 21) + rng.normal(0, 9.0, n)).clip(24, 122)
    insulin = (0.6 * np.exp(0.055 * bmi) * 8.0 + 40.0 * pedigree
               + rng.gamma(2.0, 22.0, n)).clip(14, 846)
    glucose = (60.0 + 1.05 * bmi + 0.32 * (age - 21) + 0.020 * insulin
               + 9.0 * pedigree + 0.6 * pregnancies
               + rng.normal(0, 17.0, n)).clip(44, 199)
    logit = (-9.2 + 0.043 * glucose + 0.052 * bmi + 0.013 * bp
             + 0.0016 * insulin + 0.55 * pedigree + 0.021 * age)
    onset = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))
    LOGGER.warning("Diabetes: using the SYNTHETIC surrogate (no real source reachable)")
    return pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose.round(),
        "BloodPressure": bp.round(), "SkinThickness": skin.round(),
        "Insulin": insulin.round(), "BMI": bmi.round(1),
        "DiabetesPedigreeFunction": pedigree.round(3), "Age": age.round(),
        SOURCE_TARGET: onset.astype(int),
    })


def _usable(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return SOURCE_TARGET in cols and all(f.name in cols for f in FEATURES)


def _fetch_raw() -> tuple[pd.DataFrame, bool]:
    df, label = sources.acquire(
        "diabetes",
        steps=(
            ("local", lambda: sources.local_csv(RAW_DIR)),
            ("kaggle:uciml/pima-indians-diabetes-database",
             lambda: sources.from_kagglehub("uciml/pima-indians-diabetes-database")),
            ("kaggle:mathchi/diabetes-data-set",
             lambda: sources.from_kagglehub("mathchi/diabetes-data-set")),
            ("mirror:plotly/datasets", lambda: sources.from_url(MIRROR)),
            ("synthetic", _synthetic),
        ),
        validate=_usable,
        cache_to=RAW_DIR / "diabetes.csv",
    )
    return df, label == "synthetic"


def _leaked_values(column: pd.Series, y: pd.Series) -> list[float]:
    """Values in ``column`` that look like somebody's imputation, not a reading.

    Two independent signatures, either of which is enough:

    *Fractional in a whole-number column.*  A triceps fold and a serum insulin
    are recorded as integers; 102.5 and 74.5 are means of integers, so they can
    only have been written by a fill step.

    *A frequency spike with a pure label.*  A class-conditional median lands on a
    whole number and hides in plain sight — ``SkinThickness`` is 27 on 162 rows
    and 32 on 119, against a third-place count of 27.  What gives it away is that
    those rows are 4% and 86% positive against a base rate of 35%: no genuine
    skin-fold measurement knows the outcome that well.

    The purity test needs ``y``, which is the one place in this project where a
    label touches a feature before the split.  It is defensible in this
    direction only: the test *removes* information that the redistributor put in,
    and cannot manufacture any.  A detector that ignored the labels would either
    miss the two integral fills or start deleting real readings.
    """
    counts = column.value_counts()
    if counts.empty:
        return []
    integral = float((column.dropna() % 1 == 0).mean())
    third = float(counts.iloc[2]) if counts.size >= 3 else float(counts.iloc[-1])
    base = float(y.mean())
    leaked: list[float] = []
    for value, n in counts.items():
        fractional = integral >= 0.95 and float(value) % 1 != 0
        purity = float(y[column == value].mean())
        extreme = min(purity, 1.0 - purity) <= LEAK_PURITY and abs(purity - base) > 0.15
        spike = n >= LEAK_MIN_COUNT and n >= LEAK_SPIKE_FACTOR * third
        if fractional or (spike and extreme):
            leaked.append(float(value))
            LOGGER.warning("Diabetes: %s = %g on %d row(s), %.0f%% positive vs a "
                           "%.0f%% base rate — treating as an imputation artefact",
                           column.name, float(value), int(n), 100 * purity, 100 * base)
    return leaked


def _deleak(df: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Put class-conditional fill values back to missing.

    Masking a fill value also masks the genuine readings that happen to sit on
    it — some of those 162 skin folds really were 27 mm.  That is the right trade
    at this ratio: 227 of the file's 281 rows at 27 or 32 are Pima's known
    missing skin folds, and keeping them would leave half the dataset able to
    recite its own label.
    """
    removed: dict[str, list[float]] = {}
    out = df.copy()
    for name in ZERO_IS_MISSING:
        values = _leaked_values(out[name], y)
        if values:
            removed[name] = values
            out[name] = out[name].mask(out[name].isin(values))
    return out, removed


def load_diabetes() -> tuple[pd.DataFrame, DatasetSpec]:
    """Return ``(clean_dataframe, spec)``; the target is ``no_diabetes``."""
    raw, is_synthetic = _fetch_raw()
    raw = raw.rename(columns=lambda c: str(c).strip())
    missing = [f.name for f in FEATURES if f.name not in raw.columns]
    if missing:
        raise RuntimeError(f"Diabetes source is missing columns: {missing}")

    # 1 = no onset, so that the favourable class is 1 as in every other dataset.
    y = 1 - pd.to_numeric(raw[SOURCE_TARGET], errors="coerce").fillna(0).astype(int).clip(0, 1)

    df = raw[[f.name for f in FEATURES]].apply(pd.to_numeric, errors="coerce")
    zeros = int(sum(int((df[c] == 0).sum()) for c in ZERO_IS_MISSING))
    df[list(ZERO_IS_MISSING)] = df[list(ZERO_IS_MISSING)].mask(
        df[list(ZERO_IS_MISSING)] == 0)
    LOGGER.info("Diabetes: %d impossible zeros treated as missing across %s",
                zeros, list(ZERO_IS_MISSING))
    df, leaked = _deleak(df, 1 - y)  # purity is measured against Pima's Outcome
    n_missing = int(df.isna().sum().sum())
    df = df.fillna(df.median(numeric_only=True))
    LOGGER.info("Diabetes: median-imputed %d missing cell(s)", n_missing)
    for feat in FEATURES:
        if feat.bounds is not None:
            df[feat.name] = df[feat.name].clip(*feat.bounds)
        if feat.integer:
            df[feat.name] = df[feat.name].round().astype("int64")
    df[TARGET] = y.to_numpy()
    df = df.dropna().reset_index(drop=True)

    notes = list(SPEC.notes)
    if leaked:
        detail = "; ".join(f"{k} at {', '.join(f'{v:g}' for v in vals)}"
                           for k, vals in leaked.items())
        notes.append(
            f"DE-LEAKED SOURCE: this copy arrived pre-imputed with "
            f"class-conditional fill values, which put the label inside the "
            f"features ({detail}). Those cells were restored to missing and "
            f"median-imputed unconditionally; {n_missing} cells in total were "
            f"filled. Without this step the black box reaches AUC 0.96 against a "
            f"literature norm near 0.83, and every recourse number derived from "
            f"it would be an artefact of the redistribution rather than a "
            f"property of the domain."
        )
    source, citation = SPEC.source, SPEC.citation
    if is_synthetic:
        source = "SYNTHETIC surrogate (real Pima source unreachable)"
        notes.append("SYNTHETIC SURROGATE — results are illustrative only.")
    spec = DatasetSpec(
        name=SPEC.name, features=SPEC.features, target=SPEC.target,
        positive_meaning=SPEC.positive_meaning,
        negative_meaning=SPEC.negative_meaning,
        source=source, citation=citation, notes=tuple(notes),
    )
    LOGGER.info("Diabetes: %d rows, positives(no onset)=%.1f%%, synthetic=%s",
                len(df), 100 * df[TARGET].mean(), is_synthetic)
    return df, spec


__all__ = ["load_diabetes", "SPEC", "TARGET", "SOURCE_TARGET", "ZERO_IS_MISSING",
           "MIRROR"]