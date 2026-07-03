"""Per-session circuit breaker for flaky tools.

Tracks how many times a breakable tool (e.g. browser_use) has failed within
a session and short-circuits further calls once it trips, so the agent stops
wasting time on a tool that clearly is not working and routes around it.
State is in-memory and keyed by (session_id, tool_name).
"""

from __future__ import annotations

from uuid import UUID

from app.core.config import settings

_failures: dict[tuple[UUID, str], int] = {}


def is_tripped(session_id: UUID, tool_name: str) -> bool:
    return _failures.get((session_id, tool_name), 0) >= settings.circuit_trip_threshold


def record_failure(session_id: UUID, tool_name: str) -> None:
    _failures[(session_id, tool_name)] = _failures.get((session_id, tool_name), 0) + 1


def record_success(session_id: UUID, tool_name: str) -> None:
    _failures.pop((session_id, tool_name), None)


def reset(session_id: UUID) -> None:
    for key in [k for k in _failures if k[0] == session_id]:
        _failures.pop(key, None)


__all__ = ["is_tripped", "record_failure", "record_success", "reset"]
