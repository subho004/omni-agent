"""Orchestrated research tests with a scripted fake LLM.

The fake drives the full loop offline: plan (1 step) → run sub-agent →
evaluate (adds a dependent step, not done) → run it → evaluate (done) →
synthesize. Asserts DAG execution, replanning, node persistence, and
token roll-up.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from google.genai import types
from httpx import AsyncClient

from app.api.v1.sessions import get_llm_client
from app.schemas.research import (
    EvalDecision,
    Plan,
    PlanStep,
    RevisePlan,
    StepReflection,
)
from app.services.llm_client import LlmResult
from main import app


class FakeOrchestratorLlm:
    model = "fake-model"

    def __init__(self) -> None:
        self.eval_calls = 0
        self.subagent_tasks: list[str] = []
        self.plan_prompts: list[str] = []

    async def generate_structured(
        self, prompt: str, system_instruction: str, response_schema: type
    ) -> tuple[object, int, int]:
        if response_schema is Plan:
            self.plan_prompts.append(prompt)
            return (
                Plan(
                    steps=[
                        PlanStep(
                            step_number=1,
                            title="Gather",
                            description="Find the base facts.",
                            depends_on=[],
                        )
                    ]
                ),
                12,
                4,
            )
        if response_schema is StepReflection:
            return (
                StepReflection(sufficient=True, reason="ok", next_actions=""),
                1,
                1,
            )
        # EvalDecision: first call requests one more (dependent) step, then done.
        self.eval_calls += 1
        if self.eval_calls == 1:
            return (
                EvalDecision(
                    done=False,
                    reason="Need a follow-up.",
                    new_steps=[
                        PlanStep(
                            step_number=2,
                            title="Refine",
                            description="Use step 1's output.",
                            depends_on=[1],
                        )
                    ],
                ),
                8,
                3,
            )
        return (EvalDecision(done=True, reason="Enough.", new_steps=[]), 8, 2)

    async def generate(
        self,
        contents: list[types.Content],
        system_instruction: str,
        function_declarations: list[types.FunctionDeclaration] | None = None,
    ) -> LlmResult:
        text = contents[0].parts[0].text if contents and contents[0].parts else ""
        if "Your step:" in (text or ""):
            self.subagent_tasks.append(text or "")
            return LlmResult(
                text="Completed the step and found the requested information.",
                input_tokens=5,
                output_tokens=2,
            )
        # Synthesis call (no per-step framing).
        return LlmResult(
            text="Final synthesized answer.", input_tokens=6, output_tokens=3
        )


@pytest_asyncio.fixture
async def fake_orch_llm() -> AsyncIterator[FakeOrchestratorLlm]:
    fake = FakeOrchestratorLlm()
    app.dependency_overrides[get_llm_client] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_llm_client, None)


async def _create_session(client: AsyncClient) -> str:
    response = await client.post("/sessions", json={"title": "research"})
    return str(response.json()["data"]["id"])


@pytest.mark.asyncio
async def test_research_plans_replans_and_synthesizes(
    client: AsyncClient, fake_orch_llm: FakeOrchestratorLlm
) -> None:
    session_id = await _create_session(client)

    response = await client.post(
        f"/sessions/{session_id}/research", json={"query": "How does X work?"}
    )
    assert response.status_code == 200
    data = response.json()["data"]

    assert data["answer"] == "Final synthesized answer."
    assert data["iterations"] == 2  # first eval replanned, second finished

    # Plan grew to two nodes, both completed, dependency recorded.
    plan = data["plan"]
    assert [n["step_number"] for n in plan] == [1, 2]
    assert all(n["status"] == "done" for n in plan)
    assert plan[1]["depends_on"] == [1]

    # Two sub-agents ran; the dependent one saw step 1's result.
    assert len(fake_orch_llm.subagent_tasks) == 2
    assert "found the requested information" in fake_orch_llm.subagent_tasks[1]

    # Tokens rolled up onto the session.
    assert data["input_tokens"] > 0
    sessions = await client.get("/sessions")
    tokens_used = sessions.json()["data"]["items"][0]["tokens_used"]
    assert tokens_used == data["input_tokens"] + data["output_tokens"]


@pytest.mark.asyncio
async def test_second_research_call_includes_prior_history(
    client: AsyncClient, fake_orch_llm: FakeOrchestratorLlm
) -> None:
    session_id = await _create_session(client)
    await client.post(
        f"/sessions/{session_id}/research", json={"query": "first question"}
    )
    await client.post(
        f"/sessions/{session_id}/research", json={"query": "follow up"}
    )
    # The planner prompt for the 2nd turn carries the earlier conversation.
    assert len(fake_orch_llm.plan_prompts) == 2
    assert "Earlier in this conversation" not in fake_orch_llm.plan_prompts[0]
    second = fake_orch_llm.plan_prompts[1]
    assert "Earlier in this conversation" in second
    assert "first question" in second
    assert "Final synthesized answer." in second  # the prior assistant turn


@pytest.mark.asyncio
async def test_followup_turn_runs_its_own_plan(
    client: AsyncClient, fake_orch_llm: FakeOrchestratorLlm
) -> None:
    """A second query in the same session must plan+execute a fresh turn,
    not short-circuit on the prior turn's completed step numbers."""

    session_id = await _create_session(client)
    await client.post(
        f"/sessions/{session_id}/research", json={"query": "first"}
    )
    ran_after_first = len(fake_orch_llm.subagent_tasks)

    await client.post(
        f"/sessions/{session_id}/research", json={"query": "second"}
    )
    # The follow-up dispatched at least one more sub-agent (it actually ran).
    assert len(fake_orch_llm.subagent_tasks) > ran_after_first

    # The plan endpoint returns every turn (full traceability), all executed.
    plan = (await client.get(f"/sessions/{session_id}/plan")).json()["data"]
    assert plan and all(n["status"] == "done" for n in plan)
    assert {n["turn"] for n in plan} == {1, 2}  # both turns are present
    # Step numbers restart per turn (no collision with turn 1's numbering).
    assert plan[0]["turn"] == 1 and plan[0]["step_number"] == 1


@pytest.mark.asyncio
async def test_research_unknown_session_404(
    client: AsyncClient, fake_orch_llm: FakeOrchestratorLlm
) -> None:
    response = await client.post(
        "/sessions/00000000-0000-0000-0000-000000000000/research",
        json={"query": "hi"},
    )
    assert response.status_code == 404


def _parse_sse(text: str) -> list[dict]:
    events = []
    for frame in text.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


@pytest.mark.asyncio
async def test_research_stream_emits_events(
    client: AsyncClient, fake_orch_llm: FakeOrchestratorLlm
) -> None:
    session_id = await _create_session(client)
    async with client.stream(
        "POST",
        f"/sessions/{session_id}/research/stream",
        json={"query": "How does X work?"},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = ""
        async for chunk in response.aiter_text():
            body += chunk

    events = _parse_sse(body)
    types_seen = [e["type"] for e in events]
    assert "plan_created" in types_seen
    assert "node_started" in types_seen
    assert "node_completed" in types_seen
    # Terminal answer event carries the final answer.
    answer_event = next(e for e in events if e["type"] == "answer")
    assert answer_event["answer"] == "Final synthesized answer."


@pytest.mark.asyncio
async def test_stream_persists_activity_trace(
    client: AsyncClient, fake_orch_llm: FakeOrchestratorLlm
) -> None:
    session_id = await _create_session(client)
    async with client.stream(
        "POST",
        f"/sessions/{session_id}/research/stream",
        json={"query": "How does X work?"},
    ) as response:
        async for _ in response.aiter_text():
            pass

    events = (await client.get(f"/sessions/{session_id}/events")).json()["data"]
    types_seen = {e["type"] for e in events}
    assert {"plan_created", "node_started", "node_completed"} <= types_seen
    # Transient token deltas and the final answer are not stored as trace events.
    assert "answer_delta" not in types_seen
    assert "answer" not in types_seen
    # Sequence numbers are monotonic and payloads are preserved.
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    node = next(e for e in events if e["type"] == "node_completed")
    assert "status" in node["data"]


@pytest.mark.asyncio
async def test_stop_with_no_active_run_returns_404(client: AsyncClient) -> None:
    session_id = await _create_session(client)
    response = await client.post(f"/sessions/{session_id}/stop")
    assert response.status_code == 404


class FakeReviseLlm:
    """Plan (2 steps) → eval done; on revise keep step 1, add step 3."""

    model = "fake-model"

    async def generate_structured(
        self, prompt: str, system_instruction: str, response_schema: type
    ) -> tuple[object, int, int]:
        if response_schema is Plan:
            return (
                Plan(
                    steps=[
                        PlanStep(step_number=1, title="A", description="a", depends_on=[]),
                        PlanStep(step_number=2, title="B", description="b", depends_on=[]),
                    ]
                ),
                10,
                3,
            )
        if response_schema is StepReflection:
            return (
                StepReflection(sufficient=True, reason="ok", next_actions=""),
                1,
                1,
            )
        if response_schema is RevisePlan:
            return (
                RevisePlan(
                    keep_step_numbers=[1],
                    new_steps=[
                        PlanStep(step_number=3, title="C", description="c", depends_on=[1])
                    ],
                ),
                7,
                2,
            )
        return (EvalDecision(done=True, reason="done", new_steps=[]), 5, 1)

    async def generate(
        self,
        contents: list[types.Content],
        system_instruction: str,
        function_declarations: list[types.FunctionDeclaration] | None = None,
    ) -> LlmResult:
        return LlmResult(
            text="Completed the requested step with results.",
            input_tokens=4,
            output_tokens=2,
        )


@pytest_asyncio.fixture
async def fake_revise_llm() -> AsyncIterator[FakeReviseLlm]:
    fake = FakeReviseLlm()
    app.dependency_overrides[get_llm_client] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_llm_client, None)


@pytest.mark.asyncio
async def test_revise_keeps_and_replaces_nodes(
    client: AsyncClient, fake_revise_llm: FakeReviseLlm
) -> None:
    session_id = await _create_session(client)
    await client.post(
        f"/sessions/{session_id}/research", json={"query": "original goal"}
    )

    response = await client.post(
        f"/sessions/{session_id}/revise",
        json={"instruction": "change the goal"},
    )
    assert response.status_code == 200

    plan = (await client.get(f"/sessions/{session_id}/plan")).json()["data"]
    by_step = {n["step_number"]: n for n in plan}
    assert by_step[1]["status"] == "done"       # kept
    assert by_step[2]["status"] == "replaced"   # not kept
    assert by_step[3]["status"] == "done"       # newly added + run
    assert by_step[3]["depends_on"] == [1]


class FakeReflectLlm:
    """One step; the sub-agent's first result is judged insufficient, so it
    runs a second pass (gather-more loop) before the evaluator finishes."""

    model = "fake-model"

    def __init__(self) -> None:
        self.reflect_calls = 0
        self.subagent_passes = 0

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
            self.reflect_calls += 1
            # First review: not enough → ask for more; second: sufficient.
            if self.reflect_calls == 1:
                return (
                    StepReflection(
                        sufficient=False,
                        reason="Only a partial answer so far.",
                        next_actions="Use web_search to get the concrete figure.",
                    ),
                    2,
                    1,
                )
            return (StepReflection(sufficient=True, reason="ok", next_actions=""), 2, 1)
        return (EvalDecision(done=True, reason="done", new_steps=[]), 3, 1)

    async def generate(
        self,
        contents: list[types.Content],
        system_instruction: str,
        function_declarations: list[types.FunctionDeclaration] | None = None,
    ) -> LlmResult:
        text = contents[0].parts[0].text if contents and contents[0].parts else ""
        if "Your step:" in (text or ""):
            self.subagent_passes += 1
            return LlmResult(
                text=f"Gathered result on pass {self.subagent_passes} with details.",
                input_tokens=4,
                output_tokens=2,
            )
        return LlmResult(text="Final answer.", input_tokens=3, output_tokens=1)


@pytest.mark.asyncio
async def test_subagent_reflects_and_gathers_more(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.settings, "subagent_max_reflections", 2)
    fake = FakeReflectLlm()
    app.dependency_overrides[get_llm_client] = lambda: fake
    try:
        session_id = await _create_session(client)
        async with client.stream(
            "POST",
            f"/sessions/{session_id}/research/stream",
            json={"query": "What is the figure?"},
        ) as response:
            body = ""
            async for chunk in response.aiter_text():
                body += chunk
    finally:
        app.dependency_overrides.pop(get_llm_client, None)

    # The sub-agent ran twice: first pass judged insufficient, then improved.
    assert fake.subagent_passes == 2
    assert fake.reflect_calls == 2
    # A reflection event surfaced the "insufficient" verdict to the UI.
    events = _parse_sse(body)
    reflections = [e for e in events if e["type"] == "reflection"]
    assert reflections and reflections[0]["sufficient"] is False


class FakeStallLlm:
    """Sub-agents never finish (always call a tool) and the evaluator keeps
    proposing new steps — so nothing ever completes."""

    model = "fake-model"

    def __init__(self) -> None:
        self.next_step = 2

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
            return (
                StepReflection(sufficient=True, reason="ok", next_actions=""),
                1,
                1,
            )
        step = self.next_step
        self.next_step += 1
        return (
            EvalDecision(
                done=False,
                reason="still trying",
                new_steps=[PlanStep(step_number=step, title="X", description="x", depends_on=[])],
            ),
            3,
            1,
        )

    async def generate(
        self,
        contents: list[types.Content],
        system_instruction: str,
        function_declarations: list[types.FunctionDeclaration] | None = None,
    ) -> LlmResult:
        # Always request a tool call → sub-agent loop never terminates → fails.
        call = types.FunctionCall(name="no_such_tool", args={})
        return LlmResult(
            text="",
            function_calls=[call],
            content=types.Content(role="model", parts=[types.Part(function_call=call)]),
            input_tokens=1,
            output_tokens=1,
        )


@pytest.mark.asyncio
async def test_no_progress_guard_stops_replanning(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.settings, "max_plan_iterations", 8)
    monkeypatch.setattr(orch_mod.settings, "subagent_max_iterations", 2)
    fake = FakeStallLlm()
    app.dependency_overrides[get_llm_client] = lambda: fake
    try:
        session_id = await _create_session(client)
        response = await client.post(
            f"/sessions/{session_id}/research", json={"query": "impossible task"}
        )
    finally:
        app.dependency_overrides.pop(get_llm_client, None)

    assert response.status_code == 200
    # Guard trips after 2 stalled rounds — far short of the 8 allowed.
    assert response.json()["data"]["iterations"] <= 3


@pytest.mark.asyncio
async def test_get_plan_endpoint(
    client: AsyncClient, fake_orch_llm: FakeOrchestratorLlm
) -> None:
    session_id = await _create_session(client)
    await client.post(
        f"/sessions/{session_id}/research", json={"query": "How does X work?"}
    )
    response = await client.get(f"/sessions/{session_id}/plan")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 2
