"""Single-agent chat: one turn of the shared tool-calling loop.

Rebuilds model context from the session's stored user/assistant turns, runs
the loop (app/services/agent_loop.py), and persists the answer + token usage.
For multi-step planning across sub-agents, see OrchestratorService.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from google.genai import types

from app.core.config import settings
from app.core.exceptions import SessionNotFoundError
from app.repositories.agent_session_repository import AgentSessionRepository
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.ledger_repository import LedgerRepository
from app.repositories.message_repository import MessageRepository
from app.schemas.agent_session import ChatResponse
from app.services.agent_loop import (
    build_model_content,
    build_user_content,
    format_artifacts,
    remaining_token_budget,
    run_agent_loop,
)
from app.services.llm_client import LlmClient
from app.tools.base import ToolContext

SYSTEM_PROMPT = (
    "You are a research assistant with tools. Work step by step: search to "
    "discover sources, download and parse documents to read them, and keep "
    "going until you can answer the user's question with evidence. Cite the "
    "URLs or documents you used. If a tool returns an error, adapt and try "
    "another approach. Answer directly once you have enough information."
)


class AgentService:
    """Runs one chat turn for a session via the tool-calling loop."""

    def __init__(
        self,
        llm: LlmClient,
        sessions: AgentSessionRepository,
        messages: MessageRepository,
        artifacts: ArtifactRepository,
        ledger: LedgerRepository,
    ) -> None:
        self._llm = llm
        self._sessions = sessions
        self._messages = messages
        self._artifacts = artifacts
        self._ledger = ledger

    async def chat(
        self,
        session_id: UUID,
        user_message: str,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> ChatResponse:
        session = await self._sessions.find_by_id(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)

        # Apply any model/thinking override, then run with the session's choice.
        if model is not None or thinking_level is not None:
            await self._sessions.update_model(session_id, model, thinking_level)
            session = await self._sessions.find_by_id(session_id)
            assert session is not None
        if hasattr(self._llm, "configured_for"):
            self._llm = self._llm.configured_for(
                session.model, session.thinking_level
            )

        await self._messages.create(session_id, "user", user_message)
        contents = await self._build_contents(session_id)

        # Surface any uploaded/downloaded files so the agent can use them.
        note = format_artifacts(await self._artifacts.list_by_session(session_id))
        if note:
            contents.insert(0, build_user_content(note))

        tool_ctx = ToolContext(
            session_id=session_id,
            artifacts=self._artifacts,
            data_dir=Path(settings.data_dir),
            llm=self._llm,
            ledger=self._ledger,
        )
        result = await run_agent_loop(
            self._llm,
            tool_ctx,
            self._messages,
            self._ledger,
            contents,
            SYSTEM_PROMPT,
            token_budget=remaining_token_budget(session.tokens_used),
        )

        await self._messages.create(session_id, "assistant", result.answer)
        await self._sessions.add_tokens_used(
            session_id, result.input_tokens, result.output_tokens
        )

        return ChatResponse(
            session_id=session_id,
            answer=result.answer,
            tool_calls=result.traces,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    async def _build_contents(self, session_id: UUID) -> list[types.Content]:
        """Rebuild model context from stored user/assistant turns.

        Tool traces are kept in the DB for replay/UI but not resent to the
        model — completed turns already summarise their findings.
        """

        contents: list[types.Content] = []
        for message in await self._messages.list_by_session(session_id):
            if message.role == "user":
                contents.append(build_user_content(message.content))
            elif message.role == "assistant":
                contents.append(build_model_content(message.content))
        return contents


__all__ = ["AgentService", "SYSTEM_PROMPT"]
