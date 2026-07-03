"""Structured logging utilities for the application.

Provides a small JSON formatter and helpers to configure and obtain
loggers consistently across the codebase. Uses `%s`-style logging
semantics via the standard library and emits JSON to stdout.

Follow the project's logging rules: structured output, no prints,
and use `logger.exception()` to attach stack traces when catching.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit logs as compact JSON objects.

    The formatter includes a UTC ISO-8601 timestamp, level, logger
    name and the formatted message. Any non-standard LogRecord
    attributes passed via ``extra=...`` are included when serialisable.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # include any extra fields provided via `extra=` (skip standard attrs)
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }

        for key, value in record.__dict__.items():
            if key in standard_attrs:
                continue
            try:
                # ensure value is JSON serialisable, fall back to repr()
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)

        return json.dumps(payload, ensure_ascii=False)


# Loggers whose records should honour the app level; everything else
# (third-party libraries) is capped at the third-party level.
_APP_PREFIXES = ("app", "main", "__main__", "uvicorn.error")

# Internal loggers of driven libraries that warn constantly during normal
# operation — floored at ERROR so only genuine failures show.
_VERY_NOISY_PREFIXES = ("bubus", "cdp_use", "browser_use.BrowserSession")


class _LevelFilter(logging.Filter):
    """Gate records: app loggers at app level, third-party at a higher floor.

    Applied on the handler so it holds regardless of what levels noisy
    libraries (browser_use, cdp_use, crawl4ai, aiosqlite, …) set on their
    own loggers after import.
    """

    def __init__(self, app_level: int, third_party_level: int) -> None:
        super().__init__()
        self._app = app_level
        self._third_party = third_party_level

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        if any(name == p or name.startswith(p) for p in _VERY_NOISY_PREFIXES):
            return record.levelno >= logging.ERROR
        is_app = any(
            name == p or name.startswith(p + ".") for p in _APP_PREFIXES
        )
        floor = self._app if is_app else self._third_party
        return record.levelno >= floor


def _coerce_level(level: int | str) -> int:
    if isinstance(level, str):
        return logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    return level


def configure_logging(
    level: int | str = logging.INFO,
    third_party_level: int | str = logging.WARNING,
) -> None:
    """Configure a single JSON stream handler with level-aware filtering.

    App loggers (`app.*`, `main`) emit at ``level``; third-party libraries
    are floored at ``third_party_level`` (default WARNING) so their DEBUG
    chatter never reaches the console even when the app runs in debug mode.
    Idempotent: the handler is added once and its filter is refreshed.
    """

    app_level = _coerce_level(level)
    tp_level = _coerce_level(third_party_level)

    root = logging.getLogger()
    # Root passes everything at or above the lower of the two floors; the
    # handler filter makes the finer-grained decision.
    root.setLevel(min(app_level, tp_level))

    if root.handlers:
        for handler in root.handlers:
            handler.filters = [
                f for f in handler.filters if not isinstance(f, _LevelFilter)
            ]
            handler.addFilter(_LevelFilter(app_level, tp_level))
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(_LevelFilter(app_level, tp_level))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``.

    Ensures logging is configured, but never overrides a level already
    set by an explicit `configure_logging(level)` call.
    """

    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)


__all__ = ["get_logger", "configure_logging", "JsonFormatter"]
