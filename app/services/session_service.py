"""Session use-cases: create/list sessions, history, artifacts, uploads."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from app.core.config import settings
from app.core.exceptions import (
    BadRequestError,
    NotFoundError,
    SessionNotFoundError,
)
from app.core.logging import get_logger
from app.repositories.agent_session_repository import AgentSessionRepository
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.ledger_repository import LedgerRepository
from app.repositories.message_repository import MessageRepository
from app.schemas.agent_session import (
    ArtifactResponse,
    MessageResponse,
    PaginatedMessages,
    PaginatedSessions,
    SessionResponse,
)
from app.services.llm_client import LlmClient
from app.tools.base import image_mime_for, is_supported_upload

logger = get_logger(__name__)

_IMAGE_DESCRIBE_PROMPT = (
    "Describe this image in 2-3 sentences: what it shows and any visible text "
    "or notable details. This will be stored as a searchable summary."
)


class SessionService:
    def __init__(
        self,
        sessions: AgentSessionRepository,
        messages: MessageRepository,
        artifacts: ArtifactRepository,
        llm: LlmClient | None = None,
        ledger: LedgerRepository | None = None,
    ) -> None:
        self._sessions = sessions
        self._messages = messages
        self._artifacts = artifacts
        self._llm = llm
        self._ledger = ledger

    async def create(
        self,
        title: str,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> SessionResponse:
        session = await self._sessions.create(title, model, thinking_level)
        return SessionResponse.model_validate(session)

    async def delete_session(self, session_id: UUID) -> None:
        """Delete a session, its DB rows, and its on-disk files."""

        await self._require_session(session_id)
        await self._sessions.delete(session_id)
        # Best-effort removal of any files this session produced/uploaded.
        import shutil

        root = Path(settings.data_dir)
        for sub in ("uploads", "parsed", "crawled", "pyexec", "bashexec"):
            shutil.rmtree(root / sub / str(session_id), ignore_errors=True)

    async def list_sessions(self, page: int, page_size: int) -> PaginatedSessions:
        items, total = await self._sessions.list_page(page, page_size)
        return PaginatedSessions(
            items=[SessionResponse.model_validate(s) for s in items],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def get_messages(
        self, session_id: UUID, page: int, page_size: int
    ) -> PaginatedMessages:
        await self._require_session(session_id)
        items, total = await self._messages.list_page(session_id, page, page_size)
        return PaginatedMessages(
            items=[MessageResponse.model_validate(m) for m in items],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def list_artifacts(self, session_id: UUID) -> list[ArtifactResponse]:
        await self._require_session(session_id)
        artifacts = await self._artifacts.list_by_session(session_id)
        return [ArtifactResponse.model_validate(a) for a in artifacts]

    async def get_artifact_file(
        self, session_id: UUID, artifact_id: UUID
    ) -> tuple[Path, str]:
        """Resolve an artifact's on-disk path for download.

        Validates the artifact belongs to the session (no cross-session access)
        and that its backing file still exists. Returns (path, download name).
        """

        await self._require_session(session_id)
        artifact = await self._artifacts.find_by_id(artifact_id)
        if artifact is None or artifact.session_id != session_id:
            raise NotFoundError(f"Artifact '{artifact_id}' not found")
        path = Path(artifact.uri)
        if not path.is_file():
            raise NotFoundError(f"File for artifact '{artifact_id}' is unavailable")
        return path, artifact.name

    async def save_upload(
        self, session_id: UUID, filename: str, content: bytes
    ) -> ArtifactResponse:
        """Persist one uploaded file (from a file or folder pick).

        ``filename`` may be a relative path (e.g. ``docs/report.pdf`` from a
        folder upload); its directory structure is preserved on disk and kept as
        the artifact name so the Files tree shows where it came from. Unsupported
        file types are rejected — folder uploads should filter to supported files
        before sending, and this is the backstop.
        """

        await self._require_session(session_id)
        rel_name = self._sanitize_relpath(filename)
        if not is_supported_upload(rel_name):
            raise BadRequestError(f"Unsupported file type: {rel_name}")

        uploads_dir = Path(settings.data_dir) / "uploads" / str(session_id)
        file_path = uploads_dir / rel_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

        summary = await self._describe_if_image(
            session_id, rel_name, content, len(content)
        )
        artifact = await self._artifacts.create(
            session_id=session_id,
            kind="upload",
            name=rel_name,
            uri=str(file_path),
            summary=summary,
        )
        return ArtifactResponse.model_validate(artifact)

    @staticmethod
    def _sanitize_relpath(filename: str) -> str:
        """Reduce an upload's (possibly relative) name to a safe relative path.

        Normalises separators, drops empty / ``.`` / ``..`` segments and reduces
        each segment to its basename, so a folder upload keeps its structure
        (``docs/a.pdf``) while path-traversal (``../``, absolute paths) can't
        escape the session's uploads directory.
        """

        parts = [
            Path(segment).name
            for segment in filename.replace("\\", "/").split("/")
            if segment and segment not in (".", "..")
        ]
        return "/".join(p for p in parts if p) or "upload.bin"

    async def _describe_if_image(
        self, session_id: UUID, name: str, content: bytes, size: int
    ) -> str:
        """Best-effort vision description for image uploads (Phase 10).

        Cached as the artifact summary so it can be reused without another
        vision call. Falls back to a plain summary on any failure.
        """

        mime = image_mime_for(name)
        if mime is None or self._llm is None:
            return f"User upload ({size} bytes)"
        try:
            description, in_tok, out_tok = await self._llm.describe_image(
                _IMAGE_DESCRIBE_PROMPT, content, mime
            )
        except Exception:
            logger.exception("Failed to auto-describe image %s", name)
            return f"Image upload ({size} bytes)"
        if self._ledger is not None:
            await self._ledger.create(
                session_id, self._llm.model, in_tok, out_tok
            )
        return description or f"Image upload ({size} bytes)"

    async def _require_session(self, session_id: UUID) -> None:
        if await self._sessions.find_by_id(session_id) is None:
            raise SessionNotFoundError(session_id)
