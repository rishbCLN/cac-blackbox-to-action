"""Adult Income loader (primary dataset).

Sources tried in order: local CSV, Kaggle (``uciml/adult``,
``wenruliu/adult-income-dataset``), UCI ML repository id=2.

The nine retained columns are exactly the nodes of ``data/scm/adult_scm.json``.
``fnlwgt`` (a survey weight, not a property of the person), ``education-num``
(a duplicate of ``education``), ``relationship``, ``race`` and
``native-country`` are dropped: they are either non-causal artefacts or would
add protected attributes we do not model.
"""

from __future__ import annotations

import pandas as pd

from src.config import CONFIG
from src.data import sources
from src.data.schema import DatasetSpec, FeatureSpec
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

RAW_DIR = CONFIG.DATA_RAW / "adult"
TARGET = "income"

EDUCATION_LEVELS: tuple[str, ...] = (
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
    "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors",
    "Masters", "Prof-school", "Doctorate",
)
OCCUPATION_LEVELS: tuple[str, ...] = (
    "Priv-house-serv", "Other-service", "Handlers-cleaners", "Farming-fishing",
    "Machine-op-inspct", "Transport-moving", "Adm-clerical", "Craft-repair",
    "Sales", "Protective-serv", "Armed-Forces", "Tech-support",
    "Prof-specialty", "Exec-managerial", "Unknown",
)
MARITAL_LEVELS: tuple[str, ...] = (
    "Never-married", "Married-civ-spouse", "Married-AF-spouse",
    "Married-spouse-absent", "Separated", "Divorced", "Widowed",
)
WORKCLASS_LEVELS: tuple[str, ...] = (
    "Without-pay", "Never-worked", "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Local-gov", "State-gov", "Federal-gov", "Unknown",
)
GENDER_LEVELS: tuple[str, ...] = ("Female", "Male")

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("age", "numeric", bounds=(17, 90), integer=True, unit="years",
                protected=True, manipulable=False, monotone="increase",
                description="Age of the respondent in years."),
    FeatureSpec("education", "ordinal", levels=EDUCATION_LEVELS, monotone="increase",
                unit="level", description="Highest completed education level."),
    FeatureSpec("occupation", "categorical", levels=OCCUPATION_LEVELS,
                description="Occupation group, ordered by median wage."),
    FeatureSpec("hours_per_week", "numeric", bounds=(1, 99), integer=True,
                unit="hours/week", description="Usual weekly working hours."),
    FeatureSpec("marital_status", "categorical", levels=MARITAL_LEVELS,
                protected=True, description="Marital status (sensitive)."),
    FeatureSpec("gender", "binary", levels=GENDER_LEVELS, protected=True,
                manipulable=False, description="Reported sex (protected)."),
    FeatureSpec("workclass", "categorical", levels=WORKCLASS_LEVELS,
                description="Employer sector."),
    FeatureSpec("capital_gain", "numeric", bounds=(0, 99_999), integer=True,
                unit="USD/year", description="Annual capital gains."),
    FeatureSpec("capital_loss", "numeric", bounds=(0, 4_356), integer=True,
                unit="USD/year", description="Annual capital losses."),
)

SPEC = DatasetSpec(
    name="adult",
    features=FEATURES,
    target=TARGET,
    positive_meaning="income > 50K USD/year",
    negative_meaning="income <= 50K USD/year",
    source="UCI Adult / Census Income (id=2); Kaggle uciml/adult",
    citation="Kohavi, R. (1996). Scaling up the accuracy of Naive-Bayes classifiers.",
    notes=(
        "fnlwgt, education-num, relationship, race and native-country are dropped.",
        "'?' categories are recoded to the explicit level 'Unknown'.",
    ),
)

_RENAME = {
    "sex": "gender",
    "hours-per-week": "hours_per_week",
    "marital-status": "marital_status",
    "capital-gain": "capital_gain",
    "capital-loss": "capital_loss",
    "education-num": "education_num",
    "native-country": "native_country",
    "educational-num": "education_num",
    "capital.gain": "capital_gain",
    "capital.loss": "capital_loss",
    "hours.per.week": "hours_per_week",
    "marital.status": "marital_status",
    "native.country": "native_country",
}


def _ucimlrepo_frame() -> pd.DataFrame | None:
    uci = sources.from_ucimlrepo(2)
    if uci is None:
        return None
    features, targets = uci
    return pd.concat([features, targets], axis=1)


def _fetch_raw() -> pd.DataFrame:
    df, _ = sources.acquire(
        "adult",
        steps=(
            ("local", lambda: sources.local_csv(RAW_DIR)),
            ("kaggle:uciml/adult", lambda: sources.from_kagglehub("uciml/adult")),
            ("kaggle:wenruliu/adult-income-dataset",
             lambda: sources.from_kagglehub("wenruliu/adult-income-dataset")),
            ("uci:2", _ucimlrepo_frame),
        ),
        validate=_usable,
        cache_to=RAW_DIR / "adult.csv",
    )
    return df


def _usable(df: pd.DataFrame) -> bool:
    cols = {c.strip(): c for c in df.columns}
    renamed = {_RENAME.get(k, k) for k in cols}
    needed = {f.name for f in FEATURES}
    has_target = bool(renamed & {"income", "class", "salary", "target"})
    return has_target and needed.issubset(renamed)


def load_adult() -> tuple[pd.DataFrame, DatasetSpec]:
    """Return ``(clean_dataframe, spec)`` in human-readable *semantic* space."""
    raw = _fetch_raw()
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})
    raw = raw.rename(columns=_RENAME)

    target_col = next(
        (c for c in ("income", "class", "salary", "target") if c in raw.columns), None
    )
    if target_col is None:
        raise RuntimeError(f"No target column among {list(raw.columns)}")

    y_raw = raw[target_col].astype(str).str.strip().str.replace(".", "", regex=False)
    y = (y_raw.str.contains(">50K")).astype(int)

    df = pd.DataFrame(index=raw.index)
    for feat in FEATURES:
        if feat.name not in raw.columns:
            raise RuntimeError(f"Adult source is missing column {feat.name!r}")
        col = raw[feat.name]
        if feat.kind == "numeric":
            df[feat.name] = pd.to_numeric(col, errors="coerce")
        else:
            values = col.astype(str).str.strip().replace({"?": "Unknown", "nan": "Unknown"})
            allowed = set(feat.levels or ())
            df[feat.name] = values.where(values.isin(allowed), "Unknown")
    df[TARGET] = y.to_numpy()

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after_na = len(df)
    for feat in FEATURES:
        if feat.kind == "numeric" and feat.bounds is not None:
            df[feat.name] = df[feat.name].clip(*feat.bounds)
            if feat.integer:
                df[feat.name] = df[feat.name].round().astype("int64")
    df = df.drop_duplicates().reset_index(drop=True)
    LOGGER.info(
        "Adult: %d rows kept (%d dropped as missing, %d as duplicate profiles), "
        "positives=%.1f%%",
        len(df), before - after_na, after_na - len(df), 100 * df[TARGET].mean(),
    )
    return df, SPEC


__all__ = ["load_adult", "SPEC", "TARGET", "EDUCATION_LEVELS", "OCCUPATION_LEVELS",
           "MARITAL_LEVELS", "WORKCLASS_LEVELS", "GENDER_LEVELS"]
