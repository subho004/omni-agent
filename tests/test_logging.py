"""Tests for the level-aware logging filter."""

from __future__ import annotations

import logging

from app.core.logging import _LevelFilter


def _record(name: str, level: int) -> logging.LogRecord:
    return logging.LogRecord(name, level, "f", 1, "msg", None, None)


def test_app_logs_honour_app_level() -> None:
    f = _LevelFilter(logging.DEBUG, logging.WARNING)
    assert f.filter(_record("app.services.x", logging.DEBUG)) is True
    assert f.filter(_record("main", logging.INFO)) is True


def test_third_party_debug_is_suppressed_but_warning_passes() -> None:
    f = _LevelFilter(logging.DEBUG, logging.WARNING)
    assert f.filter(_record("cdp_use.client", logging.DEBUG)) is False
    assert f.filter(_record("browser_use.Agent", logging.INFO)) is False
    assert f.filter(_record("browser_use.Agent", logging.WARNING)) is True


def test_app_prefix_not_confused_with_similar_name() -> None:
    f = _LevelFilter(logging.INFO, logging.WARNING)
    # "application" is not the "app" package — treated as third-party.
    assert f.filter(_record("application_lib", logging.INFO)) is False
