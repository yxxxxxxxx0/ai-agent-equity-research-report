"""Structured logging for the pipeline.

Every log record carries the ``report_run_id`` and the pipeline ``stage`` so a
single run can be grepped out of a shared log. Logging is configured once, at
the entry point; libraries in this package only ever call ``get_logger``.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

_run_id: ContextVar[str] = ContextVar("report_run_id", default="-")
_stage: ContextVar[str] = ContextVar("stage", default="-")

_configured = False


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.report_run_id = _run_id.get()
        record.stage = _stage.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "report_run_id": getattr(record, "report_run_id", "-"),
            "stage": getattr(record, "stage", "-"),
            "message": record.getMessage(),
        }
        extra = getattr(record, "context", None)
        if extra:
            payload["context"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, as_json: bool = False) -> None:
    """Install handlers on the package logger. Idempotent."""
    global _configured
    root = logging.getLogger("eq_report")
    if _configured:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    if as_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s [%(report_run_id)s|%(stage)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    handler.addFilter(_ContextFilter())
    root.handlers = [handler]
    root.setLevel(level)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the package root."""
    return logging.getLogger(f"eq_report.{name}")


def set_run_id(report_run_id: str) -> None:
    _run_id.set(report_run_id)


class stage_context:
    """Context manager tagging every log record inside it with a stage name."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self._token = None

    def __enter__(self) -> "stage_context":
        self._token = _stage.set(self.stage)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _stage.reset(self._token)


def log_event(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    """Log a message with a structured context payload attached."""
    logger.log(level, message, extra={"context": context})
