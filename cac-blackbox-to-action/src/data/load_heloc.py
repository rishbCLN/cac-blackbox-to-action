"""FICO HELOC loader — tertiary dataset.

Sources tried in order: local CSV, Kaggle (``aghilafi/heloc``,
``averkiyoliabev/home-equity-line-of-creditheloc``), a public unauthenticated
mirror, and finally a deterministic synthetic surrogate so that the pipeline
is *always* runnable offline (global rule #1/#7).  Whether the surrogate was
used is recorded in ``data/processed/heloc_metadata.json`` as
``is_synthetic``; every table and figure produced from it is flagged.

HELOC encodes missingness with the special negative values -9 ("no bureau
record"), -8 ("no usable trades/inquiries") and -7 ("condition not met").
Rows that are entirely -9 are dropped; remaining special values are treated as
missing and median-imputed, which is the standard treatment in the FICO
explainability-challenge literature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.data import sources
from src.data.schema import DatasetSpec, FeatureSpec
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

RAW_DIR = CONFIG.DATA_RAW / "heloc"
TARGET = "RiskPerformance"
MIRROR = ("https://raw.githubusercontent.com/seanchen7/FICO-HELOC/master/"
          "heloc_dataset_v1.csv")
SPECIAL_VALUES = (-7, -8, -9)

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("ExternalRiskEstimate", "numeric", bounds=(0, 100), integer=True,
                unit="score", manipulable=False,
                description="Consolidated bureau risk score — an *output* of the "
                            "other bureau variables, not a lever a person can pull."),
    FeatureSpec("MSinceOldestTradeOpen", "numeric", bounds=(0, 900), integer=True,
                unit="months", monotone="increase",
                description="Age of the oldest credit account."),
    FeatureSpec("AverageMInFile", "numeric", bounds=(0, 400), integer=True,
                unit="months", manipulable=False,
                description="Average age of accounts on file (derived aggregate)."),
    FeatureSpec("NumSatisfactoryTrades", "numeric", bounds=(0, 80), integer=True,
                unit="accounts", monotone="increase",
                description="Number of accounts in good standing."),
    FeatureSpec("PercentTradesNeverDelq", "numeric", bounds=(0, 100), integer=True,
                unit="%", description="Share of accounts never delinquent."),
    FeatureSpec("MaxDelqEver", "numeric", bounds=(1, 9), integer=True, unit="code",
                manipulable=False,
                description="Worst delinquency ever recorded — a fact about the past."),
    FeatureSpec("NumTotalTrades", "numeric", bounds=(0, 110), integer=True,
                unit="accounts", description="Total number of credit accounts."),
    FeatureSpec("NumTradesOpeninLast12M", "numeric", bounds=(0, 20), integer=True,
                unit="accounts", description="Accounts opened in the last 12 months."),
    FeatureSpec("NumInqLast6M", "numeric", bounds=(0, 70), integer=True,
                unit="inquiries", description="Credit inquiries in the last 6 months."),
    FeatureSpec("NetFractionRevolvingBurden", "numeric", bounds=(0, 240), integer=True,
                unit="% utilisation",
                description="Revolving balance as a percentage of the credit limit."),
    FeatureSpec("NumRevolvingTradesWBalance", "numeric", bounds=(0, 35), integer=True,
                unit="accounts", description="Revolving accounts carrying a balance."),
    FeatureSpec("PercentTradesWBalance", "numeric", bounds=(0, 100), integer=True,
                unit="%", manipulable=False,
                description="Share of accounts carrying a balance (derived)."),
)

SPEC = DatasetSpec(
    name="heloc",
    features=FEATURES,
    target=TARGET,
    positive_meaning="RiskPerformance = Good (line repaid)",
    negative_meaning="RiskPerformance = Bad (90+ days delinquent)",
    source="FICO HELOC (xML Challenge); Kaggle aghilafi/heloc",
    citation="FICO (2018). Explainable Machine Learning Challenge — HELOC dataset.",
    notes=(
        "12 of the 23 bureau variables are retained; they form the HELOC SCM.",
        "ExternalRiskEstimate, AverageMInFile, PercentTradesWBalance and "
        "MaxDelqEver are non-manipulable (derived scores or historical facts).",
    ),
)


def _synthetic(n: int = 8_000) -> pd.DataFrame:
    """Deterministic surrogate with HELOC's schema and causal ordering."""
    rng = np.random.default_rng(CONFIG.RANDOM_STATE)
    oldest = rng.gamma(4.0, 45.0, n).clip(2, 900)
    inq = rng.poisson(1.1, n).clip(0, 70)
    delq_ever = rng.choice([2, 3, 4, 5, 6, 7, 8], n, p=[.05, .07, .09, .11, .18, .1, .40])
    open12 = (0.35 * inq + rng.poisson(1.0, n)).round().clip(0, 20)
    avg_m = (0.55 * oldest - 3.0 * open12 + rng.normal(0, 12, n)).clip(2, 400)
    total = (0.05 * oldest + 1.6 * open12 + rng.poisson(9, n)).round().clip(1, 110)
    satisfactory = (total * rng.beta(6, 2, n)).round().clip(0, 80)
    never_delq = (100 * satisfactory / np.maximum(total, 1)
                  - (9 - delq_ever) * 2 + rng.normal(0, 5, n)).clip(0, 100)
    rev_with_bal = (total * rng.beta(2, 4, n)).round().clip(0, 35)
    burden = (12 * rev_with_bal + rng.normal(0, 18, n)).clip(0, 240)
    pct_with_bal = (100 * rev_with_bal / np.maximum(total, 1) + rng.normal(0, 6, n)).clip(0, 100)
    score = (55 + 0.06 * avg_m + 0.12 * never_delq - 0.10 * burden
             - 1.6 * inq + 0.9 * delq_ever + rng.normal(0, 4, n)).clip(0, 100)
    logit = (-6.0 + 0.075 * score + 0.010 * avg_m + 0.012 * never_delq
             - 0.018 * burden - 0.18 * inq - 0.05 * pct_with_bal
             + 0.02 * satisfactory)
    good = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))
    frame = {
        TARGET: np.where(good, "Good", "Bad"),
        "ExternalRiskEstimate": score.round(), "MSinceOldestTradeOpen": oldest.round(),
        "AverageMInFile": avg_m.round(), "NumSatisfactoryTrades": satisfactory,
        "PercentTradesNeverDelq": never_delq.round(), "MaxDelqEver": delq_ever,
        "NumTotalTrades": total, "NumTradesOpeninLast12M": open12,
        "NumInqLast6M": inq, "NetFractionRevolvingBurden": burden.round(),
        "NumRevolvingTradesWBalance": rev_with_bal,
        "PercentTradesWBalance": pct_with_bal.round(),
    }
    LOGGER.warning("HELOC: using the SYNTHETIC surrogate (no real source reachable)")
    return pd.DataFrame(frame)


def _usable(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return TARGET in cols and all(f.name in cols for f in FEATURES)


def _fetch_raw() -> tuple[pd.DataFrame, bool]:
    df, label = sources.acquire(
        "heloc",
        steps=(
            ("local", lambda: sources.local_csv(RAW_DIR)),
            ("kaggle:aghilafi/heloc", lambda: sources.from_kagglehub("aghilafi/heloc")),
            ("kaggle:averkiyoliabev/home-equity-line-of-creditheloc",
             lambda: sources.from_kagglehub(
                 "averkiyoliabev/home-equity-line-of-creditheloc")),
            ("mirror:seanchen7/FICO-HELOC", lambda: sources.from_url(MIRROR)),
            ("synthetic", _synthetic),
        ),
        validate=_usable,
        cache_to=RAW_DIR / "heloc_dataset_v1.csv",
    )
    return df, label == "synthetic"


def load_heloc() -> tuple[pd.DataFrame, DatasetSpec]:
    """Return ``(clean_dataframe, spec)``; ``spec.notes`` flags the surrogate."""
    raw, is_synthetic = _fetch_raw()
    raw = raw.rename(columns=lambda c: str(c).strip())
    missing = [f.name for f in FEATURES if f.name not in raw.columns]
    if missing:
        raise RuntimeError(f"HELOC source is missing columns: {missing}")

    y_raw = raw[TARGET].astype(str).str.strip().str.lower()
    y = y_raw.isin({"good", "1"}).astype(int)

    df = raw[[f.name for f in FEATURES]].apply(pd.to_numeric, errors="coerce")
    all_missing = (df == -9).all(axis=1)
    LOGGER.info("HELOC: dropping %d all-missing (-9) rows", int(all_missing.sum()))
    df, y = df[~all_missing].reset_index(drop=True), y[~all_missing].reset_index(drop=True)

    df = df.mask(df.isin(SPECIAL_VALUES))
    df = df.fillna(df.median(numeric_only=True))
    for feat in FEATURES:
        if feat.bounds is not None:
            df[feat.name] = df[feat.name].clip(*feat.bounds)
        if feat.integer:
            df[feat.name] = df[feat.name].round().astype("int64")
    df[TARGET] = y.to_numpy()
    df = df.dropna().reset_index(drop=True)

    spec = SPEC
    if is_synthetic:
        spec = DatasetSpec(
            name=SPEC.name, features=SPEC.features, target=SPEC.target,
            positive_meaning=SPEC.positive_meaning,
            negative_meaning=SPEC.negative_meaning,
            source="SYNTHETIC surrogate (real HELOC source unreachable)",
            citation=SPEC.citation,
            notes=(*SPEC.notes, "SYNTHETIC SURROGATE — results are illustrative only."),
        )
    LOGGER.info("HELOC: %d rows, positives=%.1f%%, synthetic=%s",
                len(df), 100 * df[TARGET].mean(), is_synthetic)
    return df, spec


__all__ = ["load_heloc", "SPEC", "TARGET", "SPECIAL_VALUES", "MIRROR"]
