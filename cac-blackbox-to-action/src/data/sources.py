"""Free, key-less dataset acquisition with a deterministic fallback chain.

Order of preference for every dataset (global rule #1 — no paid APIs):

1. a CSV the user already placed in ``data/raw/<dataset>/``;
2. Kaggle via :mod:`kagglehub` — only if the machine already has Kaggle
   credentials, never prompted for;
3. an unauthenticated public mirror (UCI ML repository / raw GitHub);
4. for HELOC only, a deterministic synthetic surrogate so the pipeline can
   always run offline.

Whatever succeeds is cached back into ``data/raw/<dataset>/`` so subsequent
runs are fully offline and reproducible.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)
_TIMEOUT = 60


def local_csv(directory: Path, patterns: tuple[str, ...] = ("*.csv",)) -> pd.DataFrame | None:
    """Return the first readable CSV inside ``directory``, else ``None``."""
    directory = Path(directory)
    if not directory.exists():
        return None
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            try:
                df = pd.read_csv(path, skipinitialspace=True)
            except Exception as exc:  # noqa: BLE001 - a bad local file must not stop us
                LOGGER.warning("Could not read %s (%s); trying next source", path, exc)
                continue
            if not df.empty:
                LOGGER.info("Loaded local raw CSV %s %s", path.name, df.shape)
                return df
    return None


def from_kagglehub(slug: str, filename: str | None = None) -> pd.DataFrame | None:
    """Download ``slug`` with :mod:`kagglehub`; ``None`` if unavailable."""
    try:
        import kagglehub  # noqa: PLC0415 - optional dependency
    except ImportError:
        LOGGER.info("kagglehub not installed; skipping Kaggle source %s", slug)
        return None
    try:
        path = Path(kagglehub.dataset_download(slug))
    except Exception as exc:  # noqa: BLE001 - missing credentials, offline, 403...
        LOGGER.info("Kaggle source %s unavailable (%s)", slug, type(exc).__name__)
        return None
    candidates = sorted(path.rglob(filename or "*.csv"))
    if not candidates:
        return None
    df = pd.read_csv(candidates[0], skipinitialspace=True)
    LOGGER.info("Loaded Kaggle dataset %s %s", slug, df.shape)
    return df


def from_ucimlrepo(uci_id: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fetch ``(features, targets)`` from the free UCI ML repository API."""
    try:
        from ucimlrepo import fetch_ucirepo  # noqa: PLC0415 - optional dependency
    except ImportError:
        LOGGER.warning("ucimlrepo not installed; cannot use UCI source %s", uci_id)
        return None
    try:
        bundle = fetch_ucirepo(id=uci_id)
    except Exception as exc:  # noqa: BLE001 - offline / API change
        LOGGER.warning("UCI id=%s unavailable (%s: %s)", uci_id, type(exc).__name__, exc)
        return None
    LOGGER.info("Loaded UCI id=%s %s", uci_id, bundle.data.features.shape)
    return bundle.data.features.copy(), bundle.data.targets.copy()


def from_url(url: str) -> pd.DataFrame | None:
    """Read a CSV straight from a public URL (no credentials)."""
    request = urllib.request.Request(url, headers={"User-Agent": "cac-pipeline/0.1"})
    try:
        payload = urllib.request.urlopen(request, timeout=_TIMEOUT).read()  # noqa: S310
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        LOGGER.info("Mirror unreachable %s (%s)", url.rsplit("/", 1)[-1], type(exc).__name__)
        return None
    try:
        df = pd.read_csv(io.BytesIO(payload), skipinitialspace=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.info("Mirror payload unparseable %s (%s)", url, exc)
        return None
    LOGGER.info("Loaded mirror %s %s", url.rsplit("/", 1)[-1], df.shape)
    return df


def cache_raw(df: pd.DataFrame, path: Path) -> Path:
    """Persist a freshly downloaded raw frame so later runs work offline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df.to_csv(path, index=False)
        LOGGER.info("Cached raw data -> %s", path)
    return path


def acquire(
    dataset: str,
    steps: "Sequence[tuple[str, Callable[[], pd.DataFrame | None]]]",
    validate: "Callable[[pd.DataFrame], bool]",
    cache_to: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Walk the source chain and return the first frame that ``validate`` accepts.

    Several public copies of these datasets are *reduced* variants (for example
    Kaggle's ``uciml/german-credit`` ships 10 columns and no risk label), so a
    frame is only accepted once its columns have been checked.

    Raises
    ------
    RuntimeError
        If no source produced a usable frame.
    """
    for label, step in steps:
        try:
            df = step()
        except Exception as exc:  # noqa: BLE001 - never let one source break the chain
            LOGGER.info("[%s] source %r failed (%s)", dataset, label, exc)
            continue
        if df is None or df.empty:
            continue
        df = df.rename(columns=lambda c: str(c).strip())
        if not validate(df):
            LOGGER.info("[%s] source %r rejected: unusable columns %s",
                        dataset, label, list(df.columns)[:12])
            continue
        LOGGER.info("[%s] using source %r %s", dataset, label, df.shape)
        if cache_to is not None and label != "local":
            cache_raw(df, cache_to)
        return df, label
    raise RuntimeError(
        f"Could not obtain the {dataset} dataset from any free source. "
        f"Place a CSV in data/raw/{dataset}/ (see README, section 'Data')."
    )


def has_columns(*required: str) -> "Callable[[pd.DataFrame], bool]":
    """Validator factory: every ``required`` column must be present."""

    def _check(df: pd.DataFrame) -> bool:
        cols = set(df.columns)
        return all(any(r == c for c in cols) for r in required)

    return _check


__all__ = [
    "local_csv",
    "from_kagglehub",
    "from_ucimlrepo",
    "from_url",
    "cache_raw",
    "acquire",
    "has_columns",
]
