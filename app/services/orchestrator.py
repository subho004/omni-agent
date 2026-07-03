"""Orchestrator: plan → schedule → sub-agents → evaluate → synthesize.

Implements the closed loop from docs/implementation-plan.md Phases 7-9. The
PlannerService supplies reasoning (plan/evaluate/synthesize); this service
owns control flow, the NetworkX DAG, and persistence. Ready plan nodes (all
dependencies complete) run as parallel sub-agents, each in its own DB session
so concurrent writes are safe.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from pathlib import Path
from typing import cast
from uuid import UUID

import networkx as nx
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import settings
from app.core.exceptions import SessionNotFoundError
from app.core.logging import get_logger
from app.models.plan_node import PlanNode
from app.repositories.agent_session_repository import AgentSessionRepository
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.ledger_repository import LedgerRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.plan_node_repository import PlanNodeRepository
from app.schemas.research import (
    EvalDecision,
    PlanNodeResponse,
    PlanStep,
    ResearchResponse,
)
from app.services.agent_loop import (
    build_model_content,
    build_user_content,
    format_artifacts,
    format_history,
    remaining_token_budget,
    run_agent_loop,
)
from app.services.event_bus import RunHandle
from app.services.llm_client import LlmClient
from app.services.planner import PlannerService
from app.tools.base import ToolContext

# Phrases that signal a sub-agent gave up rather than produced the data.
_REFUSAL_MARKERS = (
    "unable to",
    "could not",
    "couldn't",
    "was not able",
    "were not able",
    "not able to",
    "no results found",
    "no relevant information",
    "failed to retrieve",
    "failed to access",
    "access denied",
    "i cannot",
    "i can't",
    "cannot access",
    "cannot retrieve",
)


def _looks_like_failure(answer: str) -> bool:
    """Cheap check that a sub-agent answer is a refusal/empty, not real data."""

    text = answer.strip()
    if len(text) < 15:
        return True
    low = text.lower()
    # Only treat short answers as failures — a long, detailed answer that
    # merely mentions "could not find X" is still a useful partial result.
    return len(text) < 600 and any(m in low for m in _REFUSAL_MARKERS)

logger = get_logger(__name__)

_SUBAGENT_SYSTEM = (
    "You are a focused, resourceful sub-agent. Complete ONLY the assigned step "
    "using the available tools, then report the concrete information you found "
    "(with source URLs or document ids). Do not attempt the whole task.\n\n"
    "Be persistent: if one tool fails or is blocked, try another before giving "
    "up — e.g. if crawl_url is blocked try browser_use with explicit "
    "navigation; if a page is dynamic, fetch its JSON/API endpoint via "
    "download_file; if one source fails, find an alternative via web_search or "
    "gemini_search. Treat gemini_search as a starting point to verify and "
    "extend, not as the final word. Only report that something could not be "
    "done after you have genuinely tried the relevant tools, and always report "
    "the actual data you obtained rather than just where to find it."
)


class OrchestratorService:
    """Runs an orchestrated, multi-step research task for a session."""

    def __init__(
        self, llm: LlmClient, session_factory: async_sessionmaker
    ) -> None:
        self._base_llm = llm  # unconfigured; per-session model applied per run
        self._llm = llm
        self._planner = PlannerService(llm)
        self._sf = session_factory
        self._in_tokens = 0
        self._out_tokens = 0
        self._started_at = 0.0  # perf_counter at run start; set per research/revise
        self._artifacts_note = ""
        self._history = ""
        # Each user query starts a new research turn; nodes are scoped to it so
        # step numbers can restart at 1 without colliding with prior turns.
        self._turn = 1

    async def _configure_llm(
        self,
        db: object,
        session_id: UUID,
        model: str | None,
        thinking_level: str | None,
    ) -> None:
        """Apply any model/thinking override and point this run's client (and
        planner + sub-agents, which reuse ``self._llm``) at the session's
        chosen model."""

        from sqlalchemy.ext.asyncio import AsyncSession

        assert isinstance(db, AsyncSession)
        repo = AgentSessionRepository(db)
        if model is not None or thinking_level is not None:
            await repo.update_model(session_id, model, thinking_level)
        session = await repo.find_by_id(session_id)
        # configured_for exists on the real client; test fakes are duck-typed
        # and skip reconfiguration (they already are the model under test).
        if session is not None and hasattr(self._base_llm, "configured_for"):
            self._llm = self._base_llm.configured_for(
                session.model, session.thinking_level
            )
            self._planner = PlannerService(self._llm)

    async def research(
        self,
        session_id: UUID,
        query: str,
        run: RunHandle | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> ResearchResponse:
        self._in_tokens = 0
        self._out_tokens = 0
        self._started_at = time.perf_counter()
        run = run or RunHandle()

        async with self._sf() as db:
            repo = AgentSessionRepository(db)
            if await repo.find_by_id(session_id) is None:
                raise SessionNotFoundError(session_id)
            await self._configure_llm(db, session_id, model, thinking_level)
            messages = MessageRepository(db)
            # Build history from prior turns BEFORE recording this query.
            self._history = format_history(
                await messages.list_by_session(session_id)
            )
            await messages.create(session_id, "user", query)
            self._artifacts_note = format_artifacts(
                await ArtifactRepository(db).list_by_session(session_id)
            )
            # Start a fresh turn so this query's plan is scheduled on its own.
            self._turn = await PlanNodeRepository(db).latest_turn(session_id) + 1

        # 1. Plan (aware of prior conversation + any files in the session).
        plan_parts = [p for p in (self._history, query, self._artifacts_note) if p]
        plan, i_tok, o_tok = await self._planner.create_plan("\n\n".join(plan_parts))
        await self._account(session_id, i_tok, o_tok)
        await self._persist_steps(session_id, plan.steps)
        logger.info("planned %d step(s) for session %s", len(plan.steps), session_id)
        await run.emit(
            "plan_created",
            turn=self._turn,
            steps=[
                {
                    "step_number": s.step_number,
                    "title": s.title,
                    "depends_on": list(s.depends_on),
                }
                for s in plan.steps
            ],
        )

        return await self._drive_to_answer(session_id, query, run)

    async def revise(
        self,
        session_id: UUID,
        instruction: str,
        run: RunHandle | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> ResearchResponse:
        """Re-plan a stopped run with a new instruction (idea.md §26).

        Keeps the completed steps the re-planner deems still useful, marks the
        rest as replaced, adds new steps for the revised goal, then re-runs.
        """

        self._in_tokens = 0
        self._out_tokens = 0
        self._started_at = time.perf_counter()
        run = run or RunHandle()

        async with self._sf() as db:
            if await AgentSessionRepository(db).find_by_id(session_id) is None:
                raise SessionNotFoundError(session_id)
            await self._configure_llm(db, session_id, model, thinking_level)
            msgs = MessageRepository(db)
            self._history = format_history(await msgs.list_by_session(session_id))
            await msgs.create(session_id, "user", f"[revise] {instruction}")
            self._artifacts_note = format_artifacts(
                await ArtifactRepository(db).list_by_session(session_id)
            )
            plans = PlanNodeRepository(db)
            # Re-plan the current turn: keep useful nodes, replace the rest.
            self._turn = max(1, await plans.latest_turn(session_id))
            nodes = await plans.list_by_turn(session_id, self._turn)
            original_query = await self._first_user_query(db, session_id)

        existing_summary = "\n".join(
            f"- Step {n.step_number} [{n.status}]: {n.title}" for n in nodes
        )
        completed_digest = await self._results_digest(session_id)
        revised, i_tok, o_tok = await self._planner.revise(
            original_query, completed_digest, existing_summary, instruction
        )
        await self._account(session_id, i_tok, o_tok)

        # Retire every node the re-planner did not keep so it neither runs
        # again nor pollutes the synthesis digest.
        keep = set(revised.keep_step_numbers)
        async with self._sf() as db:
            plans = PlanNodeRepository(db)
            for node in nodes:
                if node.step_number not in keep:
                    await plans.set_status(node.id, "replaced")
        await self._persist_steps(session_id, revised.new_steps)
        await run.emit(
            "plan_created",
            turn=self._turn,
            steps=[
                {
                    "step_number": s.step_number,
                    "title": s.title,
                    "depends_on": list(s.depends_on),
                }
                for s in revised.new_steps
            ],
            revised=True,
            kept=sorted(keep),
        )

        effective_query = f"{original_query}\n\nRevised instruction: {instruction}"
        return await self._drive_to_answer(session_id, effective_query, run)

    async def _drive_to_answer(
        self, session_id: UUID, query: str, run: RunHandle
    ) -> ResearchResponse:
        """Execute ready nodes, evaluate/replan, then synthesize the answer.

        Shared by research() and revise(): the plan nodes are already
        persisted; this drives them to a final answer, honouring stop.
        """

        iterations = 0
        stopped = False
        stalls = 0
        done_count = await self._count_done(session_id)
        # A non-positive cap means "no limit" — replan until done, stopped, or
        # the no-progress guard trips.
        plan_rounds = (
            itertools.count()
            if settings.max_plan_iterations <= 0
            else iter(range(settings.max_plan_iterations))
        )
        for _ in plan_rounds:
            if run.stopped:
                stopped = True
                break
            if await self._budget_reached(session_id):
                stopped = True
                await run.emit("budget_exceeded", reason="Token budget reached")
                break
            executed = await self._run_ready_rounds(session_id, query, run)
            iterations += 1
            if not executed or run.stopped:
                stopped = run.stopped
                break

            # No-progress guard: if replanning stops yielding newly-completed
            # steps for two rounds running, further retries are unlikely to
            # help (e.g. a site keeps blocking) — stop burning budget.
            done_now = await self._count_done(session_id)
            if done_now <= done_count:
                stalls += 1
            else:
                stalls = 0
            done_count = done_now
            if stalls >= 2:
                logger.info("no progress for 2 rounds; stopping replanning")
                await run.emit(
                    "no_progress", reason="No new results after repeated attempts"
                )
                break

            digest = await self._results_digest(session_id)
            decision, i_tok, o_tok = await self._planner.evaluate(query, digest)
            await self._account(session_id, i_tok, o_tok)
            await run.emit(
                "evaluator",
                done=decision.done,
                reason=decision.reason,
                added=[s.step_number for s in decision.new_steps],
                removed=list(decision.remove_step_numbers),
                repointed=[d.step_number for d in decision.new_dependencies],
            )
            if decision.done:
                break
            changed = await self._apply_plan_delta(session_id, decision)
            # Evaluator says "not done" but proposes no change to try — further
            # replanning would repeat itself, so stop and synthesize.
            if not changed:
                break
            logger.info(
                "evaluator reshaped plan (+%d/-%d steps, %d repointed): %s",
                len(decision.new_steps),
                len(decision.remove_step_numbers),
                len(decision.new_dependencies),
                decision.reason,
            )

        if stopped:
            await run.emit("stopped", reason="Stopped by user request")

        digest = await self._results_digest(session_id)
        usage: dict[str, int] = {}
        answer_parts: list[str] = []
        async for chunk in self._planner.synthesize_stream(
            query, digest, self._history, usage
        ):
            answer_parts.append(chunk)
            await run.emit("answer_delta", text=chunk)
        answer = "".join(answer_parts)
        await self._account(session_id, usage.get("in", 0), usage.get("out", 0))

        elapsed_seconds = round(time.perf_counter() - self._started_at, 1)

        async with self._sf() as db:
            await MessageRepository(db).create(session_id, "assistant", answer)
            await AgentSessionRepository(db).add_tokens_used(
                session_id, self._in_tokens, self._out_tokens
            )
            nodes = await PlanNodeRepository(db).list_by_turn(session_id, self._turn)

        await run.emit(
            "answer",
            answer=answer,
            iterations=iterations,
            input_tokens=self._in_tokens,
            output_tokens=self._out_tokens,
            elapsed_seconds=elapsed_seconds,
            stopped=stopped,
        )

        return ResearchResponse(
            session_id=session_id,
            answer=answer,
            plan=[_to_node_response(n) for n in nodes],
            iterations=iterations,
            input_tokens=self._in_tokens,
            output_tokens=self._out_tokens,
            elapsed_seconds=elapsed_seconds,
        )

    async def _first_user_query(self, db: object, session_id: UUID) -> str:
        """The original (non-revise) user message that started the session."""

        from sqlalchemy.ext.asyncio import AsyncSession

        assert isinstance(db, AsyncSession)
        for message in await MessageRepository(db).list_by_session(session_id):
            if message.role == "user" and not message.content.startswith("[revise]"):
                return message.content
        return ""

    async def _run_ready_rounds(
        self, session_id: UUID, query: str, run: RunHandle
    ) -> int:
        """Run all currently-runnable nodes in dependency order.

        Loops within one plan iteration: after each parallel round, newly
        unblocked nodes become ready. Returns the count of nodes executed.
        Stops launching new rounds once a stop has been requested.
        """

        executed = 0
        while not run.stopped:
            ready = await self._ready_nodes(session_id)
            if not ready:
                return executed
            results = await asyncio.gather(
                *(self._run_node(session_id, node, query, run) for node in ready)
            )
            executed += len(results)
        return executed

    async def _ready_nodes(self, session_id: UUID) -> list[PlanNode]:
        """Pending nodes whose dependencies are all completed (via DAG)."""

        async with self._sf() as db:
            nodes = await PlanNodeRepository(db).list_by_turn(session_id, self._turn)

        done = {n.step_number for n in nodes if n.status == "done"}
        graph = nx.DiGraph()
        for node in nodes:
            graph.add_node(node.step_number)
            for dep in node.depends_on:
                graph.add_edge(dep, node.step_number)
        if not nx.is_directed_acyclic_graph(graph):
            logger.warning("plan for %s has a cycle; running independents only", session_id)

        return [
            node
            for node in nodes
            if node.status == "pending"
            and all(dep in done for dep in node.depends_on)
        ]

    async def _run_node(
        self, session_id: UUID, node: PlanNode, query: str, run: RunHandle
    ) -> None:
        """Execute one plan node as a sub-agent in its own DB session."""

        node_id = node.id
        async with self._sf() as db:
            plans = PlanNodeRepository(db)
            await plans.set_status(node_id, "running")
            dep_results = await self._dependency_digest(db, session_id, node)
        await run.emit(
            "node_started",
            turn=self._turn,
            step_number=node.step_number,
            title=node.title,
        )

        # Shared session context first (prior conversation + files), then the
        # concrete step. This is what makes follow-up turns build on — rather
        # than lose — everything gathered earlier in the session.
        task_parts: list[str] = []
        if self._history:
            task_parts.append(self._history)
        if self._artifacts_note:
            task_parts.append(self._artifacts_note)
        task_parts.append(
            f"Overall goal: {query}\n\nYour step: {node.title}\n{node.description}"
        )
        if dep_results:
            task_parts.append(f"Results from previous steps:\n{dep_results}")
        task = "\n\n".join(task_parts)

        async def node_event(event_type: str, data: dict[str, object]) -> None:
            await run.emit(
                event_type, turn=self._turn, step_number=node.step_number, **data
            )

        async with self._sf() as db:
            answer, status = "", "failed"
            try:
                answer, status = await self._run_subagent_with_reflection(
                    db, session_id, node, query, task, node_event, run
                )
            except Exception as exc:
                logger.exception("sub-agent for step %d failed", node.step_number)
                answer, status = f"Sub-agent error: {exc}", "failed"
            await PlanNodeRepository(db).set_result(node_id, status, answer)

        await run.emit(
            "node_completed",
            turn=self._turn,
            step_number=node.step_number,
            title=node.title,
            status=status,
            result=answer,
        )

    async def _run_subagent_with_reflection(
        self,
        db: object,
        session_id: UUID,
        node: PlanNode,
        query: str,
        task: str,
        node_event: object,
        run: RunHandle,
    ) -> tuple[str, str]:
        """Run the sub-agent, then let it self-critique and keep gathering.

        After each tool-loop pass the result is judged ("sufficient, or what
        more to do?"). If not sufficient (and budget/stop allow), the critique
        is fed back and the sub-agent continues — a fully dynamic loop that
        decides for itself when it has gathered enough. Returns (answer, status).
        """

        from collections.abc import Awaitable, Callable

        from sqlalchemy.ext.asyncio import AsyncSession

        assert isinstance(db, AsyncSession)
        emit = cast("Callable[[str, dict[str, object]], Awaitable[None]]", node_event)

        tool_ctx = ToolContext(
            session_id=session_id,
            artifacts=ArtifactRepository(db),
            data_dir=Path(settings.data_dir),
            llm=self._llm,
            ledger=LedgerRepository(db),
            # Give sub-agents the machinery to fan out into parallel children
            # (spawn_subagents). depth=1: they are one level below the top plan.
            session_factory=self._sf,
            depth=1,
            # Child agents stream their activity through this node's event sink,
            # so their tool calls/answers nest under this plan step in the UI.
            on_event=emit,
        )
        contents = [build_user_content(task)]
        answer = ""
        result = None
        rounds = max(settings.subagent_max_reflections, 0) + 1
        for reflect_round in range(rounds):
            result = await run_agent_loop(
                self._llm,
                tool_ctx,
                MessageRepository(db),
                LedgerRepository(db),
                contents,
                _SUBAGENT_SYSTEM,
                max_iterations=settings.subagent_max_iterations,
                on_event=emit,
                token_budget=await self._remaining_budget(session_id),
            )
            answer = result.answer
            await self._account(
                session_id, result.input_tokens, result.output_tokens
            )

            last_round = reflect_round >= rounds - 1
            if (
                result.hit_limit
                or result.hit_budget
                or run.stopped
                or last_round
                or await self._budget_reached(session_id)
            ):
                break

            # Self-review: sufficient, or gather more?
            reflection, i_tok, o_tok = await self._planner.reflect(
                query, node.title, node.description, answer
            )
            await self._account(session_id, i_tok, o_tok)
            await emit(
                "reflection",
                {"sufficient": reflection.sufficient, "reason": reflection.reason},
            )
            failure = settings.subagent_verify and _looks_like_failure(answer)
            if reflection.sufficient and not failure:
                break
            # Feed the critique back and continue in the same context.
            contents.append(build_model_content(answer))
            contents.append(
                build_user_content(
                    "A reviewer judged your result INSUFFICIENT for this step.\n"
                    f"Why: {reflection.reason}\n"
                    f"Do this next: {reflection.next_actions}\n"
                    "Use the tools to gather what is missing, then give a "
                    "complete, concrete answer with sources."
                )
            )

        assert result is not None
        if result.hit_limit or result.hit_budget:
            return answer, "failed"
        if settings.subagent_verify and _looks_like_failure(answer):
            return answer, "failed"  # refusal/empty — let the evaluator retry
        return answer, "done"

    async def _dependency_digest(
        self, db: object, session_id: UUID, node: PlanNode
    ) -> str:
        if not node.depends_on:
            return ""
        from sqlalchemy.ext.asyncio import AsyncSession

        assert isinstance(db, AsyncSession)
        all_nodes = await PlanNodeRepository(db).list_by_turn(session_id, self._turn)
        by_step = {n.step_number: n for n in all_nodes}
        parts = [
            f"[Step {dep}: {by_step[dep].title}]\n{by_step[dep].result}"
            for dep in node.depends_on
            if dep in by_step and by_step[dep].result
        ]
        return "\n\n".join(parts)

    async def _count_done(self, session_id: UUID) -> int:
        async with self._sf() as db:
            nodes = await PlanNodeRepository(db).list_by_turn(session_id, self._turn)
        return sum(1 for n in nodes if n.status == "done")

    async def _results_digest(self, session_id: UUID) -> str:
        async with self._sf() as db:
            nodes = await PlanNodeRepository(db).list_by_turn(session_id, self._turn)
        return "\n\n".join(
            f"[Step {n.step_number}: {n.title} — {n.status}]\n{n.result}"
            for n in nodes
            if n.result and n.status != "replaced"
        )

    async def _apply_plan_delta(
        self, session_id: UUID, decision: "EvalDecision"
    ) -> bool:
        """Apply an evaluator verdict: add, drop, and repoint plan steps.

        Retires dropped pending steps, injects new dependency edges into
        pending steps (reorder / insert-a-prerequisite), and strips retired
        steps from other nodes' deps so the DAG can't deadlock waiting on a
        step that will never complete. Returns True if anything changed.
        """

        # Add new steps first so injected dependencies can reference them.
        await self._persist_steps(session_id, decision.new_steps)

        remove = set(decision.remove_step_numbers)
        add_deps = {d.step_number: d.add_depends_on for d in decision.new_dependencies}
        changed = bool(decision.new_steps)
        if not remove and not add_deps:
            return changed

        async with self._sf() as db:
            plans = PlanNodeRepository(db)
            nodes = await plans.list_by_turn(session_id, self._turn)
            by_step = {n.step_number: n for n in nodes}

            # Retire dropped PENDING steps — never touch running/done ones.
            retired: set[int] = set()
            for step in remove:
                node = by_step.get(step)
                if node is not None and node.status == "pending":
                    await plans.set_status(node.id, "replaced")
                    retired.add(step)

            # Inject dependency edges into pending steps (excluding self/retired).
            for step, extra in add_deps.items():
                node = by_step.get(step)
                if node is None or node.status != "pending":
                    continue
                merged = sorted(
                    set(node.depends_on)
                    | {d for d in extra if d != step and d not in retired}
                )
                if merged != sorted(node.depends_on):
                    await plans.set_depends_on(node.id, merged)
                    changed = True

            # Drop retired steps from any remaining node's dependencies.
            for node in nodes:
                if node.step_number in retired or node.status != "pending":
                    continue
                cleaned = [d for d in node.depends_on if d not in retired]
                if len(cleaned) != len(node.depends_on):
                    await plans.set_depends_on(node.id, cleaned)

        return changed or bool(retired)

    async def _persist_steps(
        self, session_id: UUID, steps: list[PlanStep]
    ) -> None:
        async with self._sf() as db:
            plans = PlanNodeRepository(db)
            existing = await plans.list_by_turn(session_id, self._turn)
            if len(existing) >= settings.max_plan_nodes:
                logger.warning("plan node cap reached for turn %d", self._turn)
                return
            used = {n.step_number for n in existing}
            for step in steps:
                if step.step_number in used:
                    continue
                await plans.create(
                    session_id=session_id,
                    step_number=step.step_number,
                    title=step.title,
                    description=step.description,
                    depends_on=list(step.depends_on),
                    turn=self._turn,
                )
                used.add(step.step_number)

    async def _account(
        self, session_id: UUID, in_tokens: int, out_tokens: int
    ) -> None:
        """Track totals and record the reasoning call in the ledger."""

        self._in_tokens += in_tokens
        self._out_tokens += out_tokens
        async with self._sf() as db:
            await LedgerRepository(db).create(
                session_id, self._llm.model, in_tokens, out_tokens
            )

    async def _remaining_budget(self, session_id: UUID) -> int | None:
        """Tokens left in the session budget, counting this run's spend."""

        async with self._sf() as db:
            session = await AgentSessionRepository(db).find_by_id(session_id)
        prior = session.tokens_used if session else 0
        spent = prior + self._in_tokens + self._out_tokens
        return remaining_token_budget(spent)

    async def _budget_reached(self, session_id: UUID) -> bool:
        remaining = await self._remaining_budget(session_id)
        return remaining is not None and remaining <= 0


def _to_node_response(node: PlanNode) -> PlanNodeResponse:
    return PlanNodeResponse(
        turn=node.turn,
        step_number=node.step_number,
        title=node.title,
        description=node.description,
        status=node.status,
        depends_on=list(node.depends_on),
        result=node.result,
    )


__all__ = ["OrchestratorService"]
