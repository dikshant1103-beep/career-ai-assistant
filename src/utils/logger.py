"""Structured logging using the `rich` handler for pretty terminal output.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("hello")
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

from src.utils.config import get_settings

_INITIALISED = False


def _init_root_logger() -> None:
    """Configure the root logger exactly once."""
    global _INITIALISED
    if _INITIALISED:
        return

    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL.upper())

    for h in list(root.handlers):
        root.removeHandler(h)

    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    root.addHandler(rich_handler)

    if settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        if not log_path.is_absolute():
            log_path = settings.project_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        )
        root.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    _INITIALISED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger."""
    _init_root_logger()
    return logging.getLogger(name if name else "career_ai")


def excepthook(exc_type, exc_value, exc_traceback):
    """Route uncaught exceptions through the logger."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    log = get_logger("uncaught")
    log.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = excepthook
