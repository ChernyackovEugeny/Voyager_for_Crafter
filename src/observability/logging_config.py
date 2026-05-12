"""Terminal logging setup.

Mirrors the Banking project's stdlib-logging pattern: configure once at boot,
every module uses `logging.getLogger(__name__)` thereafter.

Database logging is a separate concern — see analytics/run_logger.py.
"""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger. Call once at process start.

    Idempotent: subsequent calls replace handlers rather than stacking them,
    so re-importing main during tests does not duplicate log lines.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party loggers — they're verbose at INFO and
    # rarely useful unless we're debugging them specifically.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
