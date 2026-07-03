"""Recursive sub-agent fan-out tool.

Lets an agent that discovers its task is really several *independent*
sub-problems split them into focused child agents that run in parallel,
each with the full tool-calling loop and its own DB session. This is the
mechanism behind dynamic "step 1 → a, b, c in parallel" decomposition:
the top-level planner no longer has to guess every split up front — any
agent can fan out mid-flight when the shape of the work becomes clear.

Nesting is bounded by ``settings.max_subagent_depth`` (orchestrator
sub-agents start at depth 1) and fan-out width by ``settings.max_spawn_fanout``
so recursion cannot run away. Children write to the token ledger via the
shared loop, and their spend is flushed to the session so the budget guard
stays honest. Only available when a session_factory is on the ToolContext
(orchestrated research); the single-turn chat path returns a clear error.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

_CHILD_SYSTEM = (
    "You are a focused child sub-agent handling ONE independent piece of a "
    "larger step. Complete only your assigned piece with the available tools "
    "and report the concrete information you found (with source URLs or "
    "document ids). Be persistent: if one tool fails or is blocked, try "
    "another before giving up. You may spawn your own child agents only if "
    "your piece itself splits into further independent parts."
)


async def _run_child(ctx: ToolContext, index: int, task: dict[str, Any]) -> dict[str, Any]:
    """Run one child agent in its own DB session and return its result.

    Emits ``subagent_started`` / ``subagent_completed`` and forwards the
    child's own tool calls (tagged with this child's index/title) through
    ``ctx.on_event`` so the UI can show the nested activity live.
    """

    # Lazy imports: this tool lives under app.tools, which app.services.agent_loop
    # imports via the registry — importing the loop at module load would be a
    # cycle. Deferring to call-time breaks it.
    from app.repositories.agent_session_repository import AgentSessionRepository
    from app.repositories.artifact_repository import ArtifactRepository
    from app.repositories.ledger_repository import LedgerRepository
    from app.repositories.message_repository import MessageRepository
    from app.services.agent_loop import (
        build_user_content,
        remaining_token_budget,
        run_agent_loop,
    )

    assert ctx.session_factory is not None  # guarded by the handler
    assert ctx.llm is not None  # guarded by the handler
    title = str(task.get("title") or f"subtask {index + 1}")
    description = str(task.get("task") or task.get("description") or "").strip()
    instruction = f"Your focused piece: {title}\n{description}".strip()

    emit = ctx.on_event

    async def child_emit(event_type: str, data: dict[str, Any]) -> None:
        # Tag every event this child (or its descendants) emits so the UI can
        # group it under this child's block beneath the parent plan step.
        if emit is not None:
            await emit(event_type, {**data, "child_index": index, "child_title": title})

    if emit is not None:
        await emit("subagent_started", {"child_index": index, "child_title": title})

    try:
        async with ctx.session_factory() as db:
            session = await AgentSessionRepository(db).find_by_id(ctx.session_id)
            budget = remaining_token_budget(session.tokens_used if session else 0)
            child_ctx = replace(
                ctx,
                artifacts=ArtifactRepository(db),
                ledger=LedgerRepository(db),
                depth=ctx.depth + 1,
                on_event=child_emit,
            )
            result = await run_agent_loop(
                ctx.llm,
                child_ctx,
                MessageRepository(db),
                LedgerRepository(db),
                [build_user_content(instruction)],
                _CHILD_SYSTEM,
                max_iterations=settings.subagent_max_iterations,
                on_event=child_emit,
                token_budget=budget,
            )
            # Flush the child's spend to the session so the outer budget guard
            # sees it (the parent loop never counts tokens spent in a tool call).
            await AgentSessionRepository(db).add_tokens_used(
                ctx.session_id, result.input_tokens, result.output_tokens
            )
    except Exception as exc:
        logger.exception("child sub-agent %d failed", index)
        if emit is not None:
            await emit(
                "subagent_completed",
                {"child_index": index, "child_title": title,
                 "status": "failed", "result": f"Child failed: {exc}"},
            )
        return {"title": title, "answer": f"Child failed: {exc}", "error": True}

    status = "failed" if (result.hit_limit or result.hit_budget) else "done"
    if emit is not None:
        await emit(
            "subagent_completed",
            {"child_index": index, "child_title": title,
             "status": status, "result": result.answer},
        )
    return {
        "title": title,
        "answer": result.answer,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "hit_limit": result.hit_limit or result.hit_budget,
    }


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.session_factory is None:
        return {
            "error": (
                "spawn_subagents is not available in this context. Do the work "
                "directly with the other tools."
            )
        }
    # A non-positive cap means "no limit" (matches the iteration-cap convention).
    if settings.max_subagent_depth > 0 and ctx.depth >= settings.max_subagent_depth:
        return {
            "error": (
                f"Max sub-agent nesting depth ({settings.max_subagent_depth}) "
                "reached. Complete this piece directly with the other tools "
                "instead of spawning more agents."
            )
        }
    if ctx.llm is None:
        return {"error": "No LLM client available to run child agents."}

    tasks = [t for t in (args.get("tasks") or []) if isinstance(t, dict)]
    if not tasks:
        return {"error": "Provide a non-empty 'tasks' list of {title, task} objects."}
    dropped = 0
    if settings.max_spawn_fanout > 0 and len(tasks) > settings.max_spawn_fanout:
        dropped = len(tasks) - settings.max_spawn_fanout
        tasks = tasks[: settings.max_spawn_fanout]

    logger.info("spawn_subagents: fanning out %d child agent(s)", len(tasks))
    if ctx.on_event is not None:
        await ctx.on_event("subagents_spawned", {"count": len(tasks)})

    # _run_child never raises (it catches internally), but guard anyway so one
    # bad child can't sink the whole fan-out.
    results = await asyncio.gather(
        *(_run_child(ctx, i, t) for i, t in enumerate(tasks)),
        return_exceptions=True,
    )
    children = [
        r if isinstance(r, dict) else {"answer": f"Child failed: {r}", "error": True}
        for r in results
    ]

    return {
        "children": children,
        "count": len(children),
        "dropped_for_fanout_cap": dropped,
    }


spawn_subagents_tool = Tool(
    name="spawn_subagents",
    description=(
        "Split the current task into several INDEPENDENT pieces and run each as "
        "its own child agent IN PARALLEL, then get all their results back. Use "
        "this when your step turns out to be several sub-problems that do not "
        "depend on each other (e.g. research three companies, scrape five pages, "
        "analyze four documents) — it is much faster than doing them one by one. "
        "Do NOT use it for a single task or for steps that must run in order; "
        "just use the other tools directly for those. Each child has the full "
        "tool set and reports concrete findings with sources."
    ),
    parameters={
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": (
                    "The independent pieces to run in parallel (2 or more). Each "
                    "is a self-contained instruction for one child agent."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short label for this piece.",
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "Full, self-contained instruction: what to find "
                                "or do and what to report back."
                            ),
                        },
                    },
                    "required": ["task"],
                },
            }
        },
        "required": ["tasks"],
    },
    handler=_handle,
    # Child loops can legitimately take a while; give the fan-out real headroom.
    timeout=1800.0,
    breakable=False,
)
