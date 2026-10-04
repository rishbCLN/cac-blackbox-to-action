"""German Credit (Statlog) loader — secondary dataset.

Sources tried in order: local CSV, Kaggle (``uciml/german-credit``,
``kamalchhirang/german-credit-data``), UCI ML repository id=144.

The opaque ``AttributeN`` column names of the UCI export and the ``A1x`` value
codes are decoded to human-readable strings using the official
``german.doc`` data dictionary, because every downstream layer (SCM, cost
model, questionnaire) is meant to be read by a human.
"""

from __future__ import annotations

import pandas as pd

from src.config import CONFIG
from src.data import sources
from src.data.schema import DatasetSpec, FeatureSpec
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

RAW_DIR = CONFIG.DATA_RAW / "german_credit"
TARGET = "credit_risk"

ATTRIBUTE_NAMES: dict[str, str] = {
    "Attribute1": "checking_status", "Attribute2": "duration",
    "Attribute3": "credit_history", "Attribute4": "purpose",
    "Attribute5": "credit_amount", "Attribute6": "savings_status",
    "Attribute7": "employment_since", "Attribute8": "installment_rate",
    "Attribute9": "personal_status_sex", "Attribute10": "other_parties",
    "Attribute11": "residence_since", "Attribute12": "property_magnitude",
    "Attribute13": "age", "Attribute14": "other_payment_plans",
    "Attribute15": "housing", "Attribute16": "existing_credits",
    "Attribute17": "job", "Attribute18": "num_dependents",
    "Attribute19": "own_telephone", "Attribute20": "foreign_worker",
}

CHECKING_LEVELS = ("no_account", "below_0_DM", "0_to_200_DM", "above_200_DM")
SAVINGS_LEVELS = ("none_or_unknown", "below_100_DM", "100_to_500_DM",
                  "500_to_1000_DM", "above_1000_DM")
EMPLOYMENT_LEVELS = ("unemployed", "below_1_year", "1_to_4_years",
                     "4_to_7_years", "above_7_years")
JOB_LEVELS = ("unemployed_unskilled_nonresident", "unskilled_resident",
              "skilled_employee", "management_self_employed")
PROPERTY_LEVELS = ("none_or_unknown", "car_or_other", "savings_or_insurance",
                   "real_estate")
HISTORY_LEVELS = ("critical_other_credits", "delay_in_past",
                  "existing_paid_duly", "all_paid_this_bank", "no_credits_all_paid")
PURPOSE_LEVELS = ("car_new", "car_used", "furniture_equipment", "radio_tv",
                  "domestic_appliances", "repairs", "education", "vacation",
                  "retraining", "business", "other")
HOUSING_LEVELS = ("for_free", "rent", "own")
SEX_LEVELS = ("female", "male")

_VALUE_MAPS: dict[str, dict[str, str]] = {
    "checking_status": {"A11": "below_0_DM", "A12": "0_to_200_DM",
                        "A13": "above_200_DM", "A14": "no_account"},
    "credit_history": {"A30": "no_credits_all_paid", "A31": "all_paid_this_bank",
                       "A32": "existing_paid_duly", "A33": "delay_in_past",
                       "A34": "critical_other_credits"},
    "purpose": {"A40": "car_new", "A41": "car_used", "A42": "furniture_equipment",
                "A43": "radio_tv", "A44": "domestic_appliances", "A45": "repairs",
                "A46": "education", "A47": "vacation", "A48": "retraining",
                "A49": "business", "A410": "other"},
    "savings_status": {"A61": "below_100_DM", "A62": "100_to_500_DM",
                       "A63": "500_to_1000_DM", "A64": "above_1000_DM",
                       "A65": "none_or_unknown"},
    "employment_since": {"A71": "unemployed", "A72": "below_1_year",
                         "A73": "1_to_4_years", "A74": "4_to_7_years",
                         "A75": "above_7_years"},
    "property_magnitude": {"A121": "real_estate", "A122": "savings_or_insurance",
                           "A123": "car_or_other", "A124": "none_or_unknown"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for_free"},
    "job": {"A171": "unemployed_unskilled_nonresident", "A172": "unskilled_resident",
            "A173": "skilled_employee", "A174": "management_self_employed"},
}
_SEX_FROM_PERSONAL = {"A91": "male", "A92": "female", "A93": "male",
                      "A94": "male", "A95": "female"}

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("age", "numeric", bounds=(18, 80), integer=True, unit="years",
                protected=True, manipulable=False, monotone="increase",
                description="Applicant age."),
    FeatureSpec("sex", "binary", levels=SEX_LEVELS, protected=True,
                manipulable=False,
                description="Derived from 'personal status and sex' (protected)."),
    FeatureSpec("num_dependents", "numeric", bounds=(1, 2), integer=True,
                unit="people", protected=True, manipulable=False,
                description="People financially dependent on the applicant."),
    FeatureSpec("credit_history", "categorical", levels=HISTORY_LEVELS,
                manipulable=False,
                description="Past repayment record — a fact about the past."),
    FeatureSpec("existing_credits", "numeric", bounds=(1, 4), integer=True,
                unit="accounts", description="Credits already held at this bank."),
    FeatureSpec("employment_since", "ordinal", levels=EMPLOYMENT_LEVELS,
                monotone="increase", unit="tenure band",
                description="Length of current employment."),
    FeatureSpec("job", "ordinal", levels=JOB_LEVELS, monotone="increase",
                unit="skill band", description="Occupational skill level."),
    FeatureSpec("savings_status", "ordinal", levels=SAVINGS_LEVELS,
                unit="DM band", description="Savings account balance band."),
    FeatureSpec("checking_status", "ordinal", levels=CHECKING_LEVELS,
                unit="DM band", description="Checking account balance band."),
    FeatureSpec("property_magnitude", "ordinal", levels=PROPERTY_LEVELS,
                unit="asset band", description="Most valuable asset owned."),
    FeatureSpec("housing", "categorical", levels=HOUSING_LEVELS,
                description="Housing tenure."),
    FeatureSpec("purpose", "categorical", levels=PURPOSE_LEVELS,
                description="Declared purpose of the requested credit."),
    FeatureSpec("credit_amount", "numeric", bounds=(250, 20_000), integer=True,
                unit="DM", description="Requested credit amount."),
    FeatureSpec("duration", "numeric", bounds=(4, 72), integer=True, unit="months",
                description="Requested repayment period."),
    FeatureSpec("installment_rate", "numeric", bounds=(1, 4), integer=True,
                unit="% band of disposable income",
                description="Instalment as a band of disposable income."),
)

SPEC = DatasetSpec(
    name="german",
    features=FEATURES,
    target=TARGET,
    positive_meaning="good credit risk (loan repaid)",
    negative_meaning="bad credit risk (default)",
    source="UCI Statlog German Credit (id=144); Kaggle uciml/german-credit",
    citation="Hofmann, H. (1994). Statlog (German Credit Data). UCI ML Repository.",
    notes=(
        "'personal status and sex' is split: only the unambiguous sex part is kept.",
        "other_parties, residence_since, other_payment_plans, own_telephone and "
        "foreign_worker are dropped (not part of the causal model).",
        "Target recoded: UCI 1='good' -> 1, 2='bad' -> 0.",
    ),
)


def _ucimlrepo_frame() -> pd.DataFrame | None:
    uci = sources.from_ucimlrepo(144)
    if uci is None:
        return None
    features, targets = uci
    targets = targets.rename(columns={targets.columns[0]: "class"})
    return pd.concat([features, targets], axis=1)


def _usable(df: pd.DataFrame) -> bool:
    """Reject the reduced Kaggle variants that ship no risk label."""
    renamed = {ATTRIBUTE_NAMES.get(c, c) for c in df.columns}
    has_target = bool(renamed & {"class", "credit_risk", "Risk", "target", "y"})
    needed = {f.name for f in FEATURES} - {"sex"}
    needed_ok = needed.issubset(renamed)
    sex_ok = bool(renamed & {"sex", "personal_status_sex"})
    return has_target and needed_ok and sex_ok


def _fetch_raw() -> pd.DataFrame:
    df, _ = sources.acquire(
        "german",
        steps=(
            ("local", lambda: sources.local_csv(RAW_DIR)),
            ("kaggle:uciml/german-credit",
             lambda: sources.from_kagglehub("uciml/german-credit")),
            ("kaggle:kamalchhirang/german-credit-data",
             lambda: sources.from_kagglehub("kamalchhirang/german-credit-data")),
            ("uci:144", _ucimlrepo_frame),
        ),
        validate=_usable,
        cache_to=RAW_DIR / "german_credit.csv",
    )
    return df


def load_german() -> tuple[pd.DataFrame, DatasetSpec]:
    """Return ``(clean_dataframe, spec)`` in human-readable *semantic* space."""
    raw = _fetch_raw().rename(columns=lambda c: str(c).strip())
    raw = raw.rename(columns=ATTRIBUTE_NAMES)

    target_col = next(
        (c for c in ("class", "credit_risk", "Risk", "target", "y") if c in raw.columns),
        None,
    )
    if target_col is None:
        raise RuntimeError(f"No target column among {list(raw.columns)}")
    tgt = raw[target_col]
    if tgt.dtype == object:
        y = (tgt.astype(str).str.strip().str.lower().isin({"good", "1"})).astype(int)
    else:
        y = (pd.to_numeric(tgt, errors="coerce") == 1).astype(int)

    if "personal_status_sex" in raw.columns:
        raw["sex"] = raw["personal_status_sex"].astype(str).str.strip().map(
            _SEX_FROM_PERSONAL
        )
    df = pd.DataFrame(index=raw.index)
    for feat in FEATURES:
        if feat.name not in raw.columns:
            raise RuntimeError(f"German source is missing column {feat.name!r}")
        col = raw[feat.name]
        if feat.kind == "numeric":
            df[feat.name] = pd.to_numeric(col, errors="coerce")
        else:
            values = col.astype(str).str.strip()
            mapping = _VALUE_MAPS.get(feat.name)
            if mapping:
                values = values.map(lambda v, m=mapping: m.get(v, v))
            allowed = set(feat.levels or ())
            unknown = ~values.isin(allowed)
            if unknown.any():
                LOGGER.warning("German %s: %d unmapped values %s", feat.name,
                               int(unknown.sum()), sorted(set(values[unknown]))[:5])
            df[feat.name] = values.where(~unknown, feat.levels[0])
    df[TARGET] = y.to_numpy()

    df = df.dropna().reset_index(drop=True)
    for feat in FEATURES:
        if feat.kind == "numeric" and feat.bounds is not None:
            df[feat.name] = df[feat.name].clip(*feat.bounds)
            if feat.integer:
                df[feat.name] = df[feat.name].round().astype("int64")
    LOGGER.info("German: %d rows, positives=%.1f%%", len(df), 100 * df[TARGET].mean())
    return df, SPEC


__all__ = ["load_german", "SPEC", "TARGET", "ATTRIBUTE_NAMES"]
