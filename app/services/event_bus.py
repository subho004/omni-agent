"""In-memory run registry for streaming and force-stop.

Each orchestrated run gets a `RunHandle` holding an event queue (for SSE)
and a stop flag the orchestrator checks at safe points (docs/
implementation-plan.md Phases 11-12). A no-queue handle is a silent no-op,
so the blocking /research path and tests need no streaming infrastructure.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

# Sentinel pushed onto the queue to tell an SSE generator the run has ended.
STREAM_DONE = object()


@dataclass
class RunHandle:
    """Per-run streaming + cancellation handle."""

    queue: asyncio.Queue[Any] | None = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event)

    async def emit(self, event_type: str, **data: Any) -> None:
        if self.queue is not None:
            await self.queue.put({"type": event_type, **data})

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()


# Active runs keyed by session id, so /stop can find the run to signal.
_RUNS: dict[UUID, RunHandle] = {}


def register_run(session_id: UUID, handle: RunHandle) -> None:
    _RUNS[session_id] = handle


def unregister_run(session_id: UUID) -> None:
    _RUNS.pop(session_id, None)


def request_stop(session_id: UUID) -> bool:
    """Signal the active run for a session to stop. Returns False if none."""

    handle = _RUNS.get(session_id)
    if handle is None:
        return False
    handle.request_stop()
    return True


__all__ = [
    "RunHandle",
    "STREAM_DONE",
    "register_run",
    "unregister_run",
    "request_stop",
]
