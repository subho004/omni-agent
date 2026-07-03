"""Tests for session, chat, upload, and history routes.

The Gemini client is replaced with a fake via dependency override, so no
network calls are made. The fake first emits a tool call to an unknown
tool (exercising the executor's error path and trace persistence), then a
final text answer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from google.genai import types
from httpx import AsyncClient

from app.api.v1.sessions import get_llm_client
from app.services.llm_client import LlmResult
from main import app


class FakeLlmClient:
    """Scripted LLM: one unknown tool call, then a final answer."""

    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        contents: list[types.Content],
        system_instruction: str,
        function_declarations: list[types.FunctionDeclaration] | None = None,
    ) -> LlmResult:
        self.calls += 1
        if self.calls == 1:
            call = types.FunctionCall(name="no_such_tool", args={"x": 1})
            return LlmResult(
                text="",
                function_calls=[call],
                content=types.Content(
                    role="model",
                    parts=[types.Part(function_call=call)],
                ),
                input_tokens=10,
                output_tokens=5,
            )
        return LlmResult(
            text="Final answer.", input_tokens=20, output_tokens=7
        )


@pytest_asyncio.fixture
async def fake_llm() -> AsyncIterator[FakeLlmClient]:
    fake = FakeLlmClient()
    app.dependency_overrides[get_llm_client] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_llm_client, None)


async def _create_session(client: AsyncClient) -> str:
    response = await client.post("/sessions", json={"title": "Test session"})
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "success"
    return str(body["data"]["id"])


@pytest.mark.asyncio
async def test_create_and_list_sessions(client: AsyncClient) -> None:
    session_id = await _create_session(client)

    response = await client.get("/sessions")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 1
    assert data["items"][0]["id"] == session_id


@pytest.mark.asyncio
async def test_chat_runs_tool_loop_and_persists_history(
    client: AsyncClient, fake_llm: FakeLlmClient
) -> None:
    session_id = await _create_session(client)

    response = await client.post(
        f"/sessions/{session_id}/chat", json={"message": "Hello"}
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"] == "Final answer."
    assert data["input_tokens"] == 30
    assert data["output_tokens"] == 12
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["tool"] == "no_such_tool"
    assert "Unknown tool" in data["tool_calls"][0]["result_summary"]
    assert fake_llm.calls == 2

    # History: user turn, tool trace, assistant answer — in order.
    response = await client.get(f"/sessions/{session_id}/messages")
    assert response.status_code == 200
    messages = response.json()["data"]["items"]
    assert [m["role"] for m in messages] == ["user", "tool", "assistant"]

    # Token usage rolled up onto the session.
    response = await client.get("/sessions")
    assert response.json()["data"]["items"][0]["tokens_used"] == 42


@pytest.mark.asyncio
async def test_chat_unknown_session_returns_404(
    client: AsyncClient, fake_llm: FakeLlmClient
) -> None:
    response = await client.post(
        "/sessions/00000000-0000-0000-0000-000000000000/chat",
        json={"message": "Hello"},
    )
    assert response.status_code == 404
    assert response.json()["status"] == "error"


@pytest.mark.asyncio
async def test_upload_and_list_artifacts(client: AsyncClient) -> None:
    session_id = await _create_session(client)

    response = await client.post(
        f"/sessions/{session_id}/files",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 201
    artifact = response.json()["data"]
    assert artifact["kind"] == "upload"
    assert artifact["name"] == "notes.txt"

    response = await client.get(f"/sessions/{session_id}/artifacts")
    assert response.status_code == 200
    artifacts = response.json()["data"]
    assert len(artifacts) == 1
    assert artifacts[0]["id"] == artifact["id"]


@pytest.mark.asyncio
async def test_empty_upload_rejected(client: AsyncClient) -> None:
    session_id = await _create_session(client)
    response = await client.post(
        f"/sessions/{session_id}/files",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_delete_session_removes_it_and_its_data(client: AsyncClient) -> None:
    session_id = await _create_session(client)
    # Attach a file so there is child data to cascade-delete.
    await client.post(
        f"/sessions/{session_id}/files",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    response = await client.delete(f"/sessions/{session_id}")
    assert response.status_code == 200

    # Gone from the list and its sub-resources 404.
    listing = (await client.get("/sessions")).json()["data"]["items"]
    assert all(s["id"] != session_id for s in listing)
    assert (await client.get(f"/sessions/{session_id}/messages")).status_code == 404
    assert (await client.get(f"/sessions/{session_id}/artifacts")).status_code == 404


@pytest.mark.asyncio
async def test_delete_unknown_session_404(client: AsyncClient) -> None:
    response = await client.delete(
        "/sessions/00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404
