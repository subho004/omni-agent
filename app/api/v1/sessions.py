"""Session & chat routes: create/list sessions, chat, history, uploads."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BadRequestError
from app.core.models import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_THINKING_LEVEL,
    THINKING_LEVELS,
)
from app.db.database import async_session_factory, get_db_session
from app.repositories.agent_session_repository import AgentSessionRepository
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.ledger_repository import LedgerRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.plan_node_repository import PlanNodeRepository
from app.repositories.run_event_repository import RunEventRepository
from app.schemas.agent_session import ChatRequest, ModelChoice, SessionCreate
from app.schemas.research import ResearchRequest, ReviseRequest
from app.services.agent_service import AgentService
from app.services.event_bus import (
    STREAM_DONE,
    RunHandle,
    register_run,
    request_stop,
    unregister_run,
)
from app.services.llm_client import LlmClient
from app.services.orchestrator import OrchestratorService, _to_node_response
from app.services.session_service import SessionService
from app.core.logging import get_logger
from utils.response import error_response, success_response

logger = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])

_llm_client: LlmClient | None = None


def get_llm_client() -> LlmClient:
    """Lazily create the shared Gemini client (needs the API key at runtime)."""

    global _llm_client
    if _llm_client is None:
        _llm_client = LlmClient()
    return _llm_client


def get_session_service(
    db: AsyncSession = Depends(get_db_session),
    llm: LlmClient = Depends(get_llm_client),
) -> SessionService:
    return SessionService(
        sessions=AgentSessionRepository(db),
        messages=MessageRepository(db),
        artifacts=ArtifactRepository(db),
        llm=llm,
        ledger=LedgerRepository(db),
    )


def get_agent_service(
    db: AsyncSession = Depends(get_db_session),
    llm: LlmClient = Depends(get_llm_client),
) -> AgentService:
    return AgentService(
        llm=llm,
        sessions=AgentSessionRepository(db),
        messages=MessageRepository(db),
        artifacts=ArtifactRepository(db),
        ledger=LedgerRepository(db),
    )


def get_orchestrator_service(
    llm: LlmClient = Depends(get_llm_client),
) -> OrchestratorService:
    # Owns its own sessions (parallel sub-agents each need one), so it takes
    # the session factory rather than a single request-scoped session.
    return OrchestratorService(llm=llm, session_factory=async_session_factory)


@router.post("")
async def create_session(
    payload: SessionCreate,
    service: SessionService = Depends(get_session_service),
) -> JSONResponse:
    session = await service.create(
        payload.title, model=payload.model, thinking_level=payload.thinking_level
    )
    return success_response(
        data=session, message="Session created", status_code=201
    )


@router.get("/options/models")
async def list_models() -> JSONResponse:
    """Selectable models + thinking levels for the UI dropdowns."""

    return success_response(
        data={
            "models": [
                {
                    "id": m.id,
                    "label": m.label,
                    "supports_thinking": m.supports_thinking,
                    "context_window_tokens": m.context_window_tokens,
                    "max_output_tokens": m.max_output_tokens,
                }
                for m in AVAILABLE_MODELS
            ],
            "thinking_levels": list(THINKING_LEVELS),
            "default_model": DEFAULT_MODEL,
            "default_thinking_level": DEFAULT_THINKING_LEVEL,
        },
        message="Model options",
    )


@router.get("")
async def list_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: SessionService = Depends(get_session_service),
) -> JSONResponse:
    result = await service.list_sessions(page, page_size)
    return success_response(data=result, message="Sessions retrieved")


@router.delete("/{session_id}")
async def delete_session(
    session_id: UUID,
    service: SessionService = Depends(get_session_service),
) -> JSONResponse:
    request_stop(session_id)  # stop any in-flight run before removing its data
    await service.delete_session(session_id)
    return success_response(message="Session deleted")


@router.post("/{session_id}/chat")
async def chat(
    session_id: UUID,
    payload: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> JSONResponse:
    result = await service.chat(
        session_id, payload.message,
        model=payload.model, thinking_level=payload.thinking_level,
    )
    return success_response(data=result, message="Chat turn completed")


def _sse_response(
    session_id: UUID, work: Callable[[RunHandle], Awaitable[object]]
) -> StreamingResponse:
    """Run `work` in the background, streaming its emitted events as SSE."""

    handle = RunHandle(queue=asyncio.Queue())
    register_run(session_id, handle)

    async def runner() -> None:
        try:
            await work(handle)
        except Exception as exc:  # surface as a terminal error event
            await handle.emit("error", message=str(exc))
        finally:
            assert handle.queue is not None
            await handle.queue.put(STREAM_DONE)

    async def event_stream() -> AsyncIterator[str]:
        task = asyncio.create_task(runner())
        assert handle.queue is not None
        # Persist the activity trace so it survives beyond the live stream
        # (replay on reopen + full export). Transient token deltas are skipped.
        async with async_session_factory() as db:
            events = RunEventRepository(db)
            try:
                while True:
                    event = await handle.queue.get()
                    if event is STREAM_DONE:
                        break
                    if event.get("type") not in ("answer_delta", "answer"):
                        try:
                            await events.append(
                                session_id,
                                str(event["type"]),
                                {k: v for k, v in event.items() if k != "type"},
                            )
                        except Exception:  # never break the stream on a write hiccup
                            logger.warning("failed to persist run event", exc_info=True)
                    yield f"data: {json.dumps(event, default=str)}\n\n"
            finally:
                await task
                unregister_run(session_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{session_id}/research")
async def research(
    session_id: UUID,
    payload: ResearchRequest,
    service: OrchestratorService = Depends(get_orchestrator_service),
) -> JSONResponse:
    result = await service.research(
        session_id, payload.query,
        model=payload.model, thinking_level=payload.thinking_level,
    )
    return success_response(data=result, message="Research completed")


@router.post("/{session_id}/research/stream")
async def research_stream(
    session_id: UUID,
    payload: ResearchRequest,
    service: OrchestratorService = Depends(get_orchestrator_service),
) -> StreamingResponse:
    """Run research in the background, streaming progress events as SSE."""

    return _sse_response(
        session_id,
        lambda h: service.research(
            session_id, payload.query, h,
            model=payload.model, thinking_level=payload.thinking_level,
        ),
    )


@router.post("/{session_id}/revise")
async def revise(
    session_id: UUID,
    payload: ReviseRequest,
    service: OrchestratorService = Depends(get_orchestrator_service),
) -> JSONResponse:
    result = await service.revise(
        session_id, payload.instruction,
        model=payload.model, thinking_level=payload.thinking_level,
    )
    return success_response(data=result, message="Revision completed")


@router.post("/{session_id}/revise/stream")
async def revise_stream(
    session_id: UUID,
    payload: ReviseRequest,
    service: OrchestratorService = Depends(get_orchestrator_service),
) -> StreamingResponse:
    """Re-plan a stopped run with a new instruction, streaming events."""

    return _sse_response(
        session_id,
        lambda h: service.revise(
            session_id, payload.instruction, h,
            model=payload.model, thinking_level=payload.thinking_level,
        ),
    )


@router.post("/{session_id}/stop")
async def stop_research(session_id: UUID) -> JSONResponse:
    if not request_stop(session_id):
        return error_response(
            message="No active run for this session", status_code=404
        )
    return success_response(message="Stop requested")


@router.post("/{session_id}/settings")
async def update_settings(
    session_id: UUID,
    payload: ModelChoice,
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """Persist a model / thinking-level change for the session."""

    repo = AgentSessionRepository(db)
    if await repo.find_by_id(session_id) is None:
        return error_response(message="Session not found", status_code=404)
    await repo.update_model(session_id, payload.model, payload.thinking_level)
    session = await repo.find_by_id(session_id)
    assert session is not None
    return success_response(
        data={"model": session.model, "thinking_level": session.thinking_level},
        message="Settings updated",
    )


@router.get("/{session_id}/plan")
async def get_plan(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    plans = PlanNodeRepository(db)
    # Return every turn's plan (ordered by turn, then step) for full
    # traceability across a multi-turn session. Step numbers restart per turn,
    # so each node carries its `turn` and the client groups by it.
    nodes = await plans.list_by_session(session_id)
    return success_response(
        data=[_to_node_response(n) for n in nodes], message="Plan retrieved"
    )


@router.get("/{session_id}/events")
async def get_events(
    session_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """The persisted activity trace (tools, reflections, evaluator, …)."""

    events = await RunEventRepository(db).list_by_session(session_id)
    return success_response(
        data=[
            {"seq": e.seq, "type": e.type, "data": e.data, "created_at": e.created_at}
            for e in events
        ],
        message="Events retrieved",
    )


@router.get("/{session_id}/messages")
async def get_messages(
    session_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    service: SessionService = Depends(get_session_service),
) -> JSONResponse:
    result = await service.get_messages(session_id, page, page_size)
    return success_response(data=result, message="Messages retrieved")


@router.get("/{session_id}/artifacts")
async def list_artifacts(
    session_id: UUID,
    service: SessionService = Depends(get_session_service),
) -> JSONResponse:
    result = await service.list_artifacts(session_id)
    return success_response(data=result, message="Artifacts retrieved")


@router.get("/{session_id}/artifacts/{artifact_id}/content")
async def download_artifact(
    session_id: UUID,
    artifact_id: UUID,
    service: SessionService = Depends(get_session_service),
) -> FileResponse:
    """Stream an artifact's raw file back as a download."""

    path, name = await service.get_artifact_file(session_id, artifact_id)
    return FileResponse(path, filename=name)


@router.post("/{session_id}/files")
async def upload_file(
    session_id: UUID,
    file: UploadFile,
    service: SessionService = Depends(get_session_service),
) -> JSONResponse:
    content = await file.read()
    if not content:
        raise BadRequestError("Uploaded file is empty")
    artifact = await service.save_upload(
        session_id, file.filename or "upload.bin", content
    )
    return success_response(
        data=artifact, message="File uploaded", status_code=201
    )
