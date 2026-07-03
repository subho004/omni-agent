"""Tests for dynamic plan reshaping and recursive sub-agent fan-out.

Covers the two mechanisms that let the orchestrator escape its initially
decided steps: the evaluator's ability to drop/reorder/insert steps
(`_apply_plan_delta`) and the `spawn_subagents` tool's depth/fan-out guards.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from google.genai import types
from httpx import AsyncClient

from app.api.v1.sessions import get_llm_client
from app.db.database import async_session_factory
from app.repositories.agent_session_repository import AgentSessionRepository
from app.repositories.plan_node_repository import PlanNodeRepository
from app.schemas.research import EvalDecision, Plan, PlanStep, StepDependency, StepReflection
from app.services.llm_client import LlmResult
from app.services.orchestrator import OrchestratorService
from app.tools.base import ToolContext
from app.tools.spawn_subagents import _handle as spawn_handle
from main import app


class _FakeLlm:
    model = "fake-model"


@pytest.mark.asyncio
async def test_apply_plan_delta_removes_reorders_and_inserts(
    client: AsyncClient,
) -> None:
    """Evaluator drops a pending step, adds one, and injects a prerequisite.

    The retired step is marked 'replaced', the new step is added, the target
    step gains the new prerequisite, and the dangling edge to the retired
    step is stripped so the DAG cannot deadlock.
    """

    async with async_session_factory() as db:
        session = await AgentSessionRepository(db).create("t")
        sid = session.id
        plans = PlanNodeRepository(db)
        await plans.create(sid, 1, "A", "a", [], turn=1)
        await plans.create(sid, 2, "B", "b", [], turn=1)  # will be removed
        await plans.create(sid, 3, "C", "c", [2], turn=1)  # depends on removed 2

    orch = OrchestratorService(_FakeLlm(), async_session_factory)
    orch._turn = 1
    decision = EvalDecision(
        done=False,
        reason="reshape",
        new_steps=[PlanStep(step_number=4, title="D", description="d", depends_on=[])],
        remove_step_numbers=[2],
        new_dependencies=[StepDependency(step_number=3, add_depends_on=[4])],
    )

    changed = await orch._apply_plan_delta(sid, decision)
    assert changed is True

    async with async_session_factory() as db:
        nodes = {
            n.step_number: n
            for n in await PlanNodeRepository(db).list_by_turn(sid, 1)
        }
    assert nodes[2].status == "replaced"          # dropped pending step
    assert nodes[4].status == "pending"           # newly added step
    # Step 3 now waits on the new prerequisite (4), and the dead edge to the
    # retired step (2) has been stripped so it can still become ready.
    assert nodes[3].depends_on == [4]


@pytest.mark.asyncio
async def test_apply_plan_delta_never_touches_completed_steps(
    client: AsyncClient,
) -> None:
    """A 'done' step listed for removal is left intact (only pending drops)."""

    async with async_session_factory() as db:
        session = await AgentSessionRepository(db).create("t")
        sid = session.id
        plans = PlanNodeRepository(db)
        node = await plans.create(sid, 1, "A", "a", [], turn=1)
        await plans.set_result(node.id, "done", "finished")

    orch = OrchestratorService(_FakeLlm(), async_session_factory)
    orch._turn = 1
    decision = EvalDecision(
        done=False, reason="x", new_steps=[], remove_step_numbers=[1]
    )
    changed = await orch._apply_plan_delta(sid, decision)

    assert changed is False  # nothing eligible to change
    async with async_session_factory() as db:
        nodes = await PlanNodeRepository(db).list_by_turn(sid, 1)
    assert nodes[0].status == "done"


@pytest.mark.asyncio
async def test_spawn_subagents_requires_session_factory() -> None:
    """Without a session_factory (chat path) the tool refuses cleanly."""

    ctx = ToolContext(
        session_id=__import__("uuid").uuid4(),
        artifacts=None,  # type: ignore[arg-type]
        data_dir=__import__("pathlib").Path("."),
        llm=_FakeLlm(),  # type: ignore[arg-type]
    )
    result = await spawn_handle(ctx, {"tasks": [{"task": "x"}]})
    assert "error" in result
    assert "not available" in result["error"]


@pytest.mark.asyncio
async def test_spawn_subagents_enforces_depth_cap() -> None:
    """At the max nesting depth the tool refuses rather than recursing on."""

    from app.core.config import settings

    ctx = ToolContext(
        session_id=__import__("uuid").uuid4(),
        artifacts=None,  # type: ignore[arg-type]
        data_dir=__import__("pathlib").Path("."),
        llm=_FakeLlm(),  # type: ignore[arg-type]
        session_factory=async_session_factory,
        depth=settings.max_subagent_depth,
    )
    result = await spawn_handle(ctx, {"tasks": [{"task": "x"}]})
    assert "error" in result
    assert "depth" in result["error"].lower()


class FakeFanoutLlm:
    """One plan step; the sub-agent calls spawn_subagents once, its two
    children answer directly, then the evaluator finishes."""

    model = "fake-model"

    def __init__(self) -> None:
        self.spawned = False
        self.child_tasks: list[str] = []

    async def generate_structured(
        self, prompt: str, system_instruction: str, response_schema: type
    ) -> tuple[object, int, int]:
        if response_schema is Plan:
            return (
                Plan(steps=[PlanStep(step_number=1, title="A", description="a", depends_on=[])]),
                5,
                1,
            )
        if response_schema is StepReflection:
            return (StepReflection(sufficient=True, reason="ok", next_actions=""), 1, 1)
        return (EvalDecision(done=True, reason="done", new_steps=[]), 3, 1)

    async def generate(
        self,
        contents: list[types.Content],
        system_instruction: str,
        function_declarations: list[types.FunctionDeclaration] | None = None,
    ) -> LlmResult:
        text = contents[0].parts[0].text if contents and contents[0].parts else ""
        if "Your focused piece:" in (text or ""):  # a child agent
            self.child_tasks.append(text or "")
            return LlmResult(text="Child gathered its part.", input_tokens=2, output_tokens=1)
        if "Your step:" in (text or "") and not self.spawned:  # top sub-agent, 1st pass
            self.spawned = True
            call = types.FunctionCall(
                name="spawn_subagents",
                args={"tasks": [
                    {"title": "Alpha", "task": "gather alpha"},
                    {"title": "Beta", "task": "gather beta"},
                ]},
            )
            return LlmResult(
                text="",
                function_calls=[call],
                content=types.Content(role="model", parts=[types.Part(function_call=call)]),
                input_tokens=2,
                output_tokens=1,
            )
        if "Your step:" in (text or ""):  # top sub-agent, after children returned
            return LlmResult(text="Combined the child results.", input_tokens=2, output_tokens=1)
        return LlmResult(text="Final answer.", input_tokens=2, output_tokens=1)


@pytest_asyncio.fixture
async def fake_fanout_llm():
    fake = FakeFanoutLlm()
    app.dependency_overrides[get_llm_client] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_llm_client, None)


def _parse_sse(text: str) -> list[dict]:
    events = []
    for frame in text.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


@pytest.mark.asyncio
async def test_spawn_subagents_streams_nested_events(
    client: AsyncClient, fake_fanout_llm: FakeFanoutLlm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fan-out surfaces per-child lifecycle events tagged for UI nesting."""

    from app.services import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.settings, "subagent_max_reflections", 0)

    r = await client.post("/sessions", json={"title": "fanout"})
    session_id = r.json()["data"]["id"]
    async with client.stream(
        "POST",
        f"/sessions/{session_id}/research/stream",
        json={"query": "compare alpha and beta"},
    ) as response:
        body = ""
        async for chunk in response.aiter_text():
            body += chunk

    events = _parse_sse(body)
    types_seen = [e["type"] for e in events]
    assert "subagents_spawned" in types_seen
    started = [e for e in events if e["type"] == "subagent_started"]
    completed = [e for e in events if e["type"] == "subagent_completed"]
    # Two children ran, each with a stable index/title for UI grouping and the
    # parent step number so the UI nests them under step 1.
    assert {e["child_title"] for e in started} == {"Alpha", "Beta"}
    assert {e["child_index"] for e in started} == {0, 1}
    assert all(e["step_number"] == 1 for e in started)
    assert {e["child_title"] for e in completed} == {"Alpha", "Beta"}
    assert all(e["status"] == "done" for e in completed)
    assert len(fake_fanout_llm.child_tasks) == 2
