"""Logging that always goes to stdout *and* ``logs/<script>_<timestamp>.log``.

Global rule #4.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config import CONFIG

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED: dict[str, Path] = {}


def setup_logging(script_name: str, level: str | None = None) -> Path:
    """Attach a stdout handler and a timestamped file handler to the root logger.

    Parameters
    ----------
    script_name:
        Used for the log file name, e.g. ``01_train_blackbox``.
    level:
        Overrides :attr:`Config.LOG_LEVEL`.

    Returns
    -------
    Path
        Location of the log file.
    """
    if script_name in _CONFIGURED:
        return _CONFIGURED[script_name]

    CONFIG.LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = CONFIG.LOG_DIR / f"{script_name}_{stamp}.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, (level or CONFIG.LOG_LEVEL).upper(), logging.INFO))
    for h in list(root.handlers):  # avoid duplicate output under pytest / notebooks
        root.removeHandler(h)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    # Windows consoles default to cp1252 and would mangle any non-ASCII log text.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - redirected/captured stdout
            pass

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Third-party noise.
    for noisy in ("matplotlib", "dowhy", "numexpr", "PIL", "urllib3", "sklearn",
                  "fontTools", "fsspec", "kagglehub", "httpx", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED[script_name] = log_path
    logging.getLogger(__name__).info("Logging to %s", log_path)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; falls back to a bare stdout handler if needed."""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(
            getattr(logging, CONFIG.LOG_LEVEL.upper(), logging.INFO)
        )
    return logger


__all__ = ["setup_logging", "get_logger"]
