"""Uniform tool abstraction for the agent executor.

Every tool declares a Gemini-compatible function schema and an async
handler. Handlers receive a `ToolContext` (session id + artifact repo +
data dir, plus an optional LLM client and ledger for tools that call the
model, e.g. vision) and return a JSON-serialisable dict. Large payloads
must be written as artifacts and referenced by id, never inlined
(docs/implementation-plan.md §6).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from google.genai import types

from app.repositories.artifact_repository import ArtifactRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.repositories.ledger_repository import LedgerRepository
    from app.services.llm_client import LlmClient

ToolHandler = Callable[["ToolContext", dict[str, Any]], Awaitable[dict[str, Any]]]

# Extensions treated as images for multimodal handling.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def image_mime_for(name: str) -> str | None:
    """Return the image MIME type for a filename, or None if not an image."""

    return _MIME_BY_EXT.get(Path(name).suffix.lower())


@dataclass
class ToolContext:
    """Per-request context handed to every tool handler."""

    session_id: UUID
    artifacts: ArtifactRepository
    data_dir: Path
    llm: LlmClient | None = None
    ledger: LedgerRepository | None = None
    # Set for orchestrated sub-agents so the spawn_subagents tool can open a
    # fresh DB session per child and fan out safely. `depth` is this agent's
    # nesting level (top-level chat = 0, orchestrator sub-agents = 1); children
    # increment it and the spawn tool refuses past settings.max_subagent_depth.
    session_factory: async_sessionmaker[Any] | None = None
    depth: int = 0
    # Optional live-event sink (RunHandle.emit-style) so tools that spawn child
    # agents can stream their activity to the UI. Set for orchestrated
    # sub-agents; the injected turn/step_number make child events nest under the
    # right plan step. None on paths that don't stream.
    on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None


@dataclass
class Tool:
    """A callable tool exposed to the model via function calling."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    # Wall-clock limit for one invocation; the executor kills it past this.
    timeout: float = 120.0
    # Whether repeated failures should disable this tool for the session.
    breakable: bool = False

    def declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self.parameters,
        )


__all__ = [
    "Tool",
    "ToolContext",
    "ToolHandler",
    "IMAGE_EXTENSIONS",
    "image_mime_for",
]
