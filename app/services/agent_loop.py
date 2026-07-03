"""Reusable Gemini function-calling loop.

Shared by the single-agent chat path and by orchestrated sub-agents. Given
a starting list of contents and a system prompt, it runs the model, executes
any tool calls, feeds results back, and repeats until the model answers or
the iteration cap is hit. Every LLM call is written to the ledger; every tool
call is persisted as a `tool` message for replay/UI.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

from app.core.config import settings
from app.core.logging import get_logger
from app.core.models import compact_threshold_chars
from app.repositories.ledger_repository import LedgerRepository
from app.repositories.message_repository import MessageRepository
from app.schemas.agent_session import ToolCallTrace
from app.services.llm_client import LlmClient
from app.tools import tool_cache, tool_guard
from app.tools.base import ToolContext
from app.tools.registry import ALL_TOOLS, TOOLS_BY_NAME

logger = get_logger(__name__)

TRACE_SUMMARY_CHARS = 600

# Optional async callback used to stream events (e.g. tool calls) live.
EventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass
class LoopResult:
    answer: str
    traces: list[ToolCallTrace] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    hit_limit: bool = False
    hit_budget: bool = False


async def run_agent_loop(
    llm: LlmClient,
    tool_ctx: ToolContext,
    messages: MessageRepository,
    ledger: LedgerRepository,
    contents: list[types.Content],
    system_prompt: str,
    max_iterations: int | None = None,
    on_event: EventCallback | None = None,
    token_budget: int | None = None,
) -> LoopResult:
    """Run the tool-calling loop to completion and return the result.

    `token_budget`, when set, is the number of tokens this loop may spend
    before it stops early with a note (cost cap, docs/implementation-plan §4).
    """

    declarations = [tool.declaration() for tool in ALL_TOOLS]
    limit = (
        max_iterations if max_iterations is not None else settings.max_agent_iterations
    )
    # A non-positive limit means "no cap" — loop until the model answers.
    iterations = itertools.count() if limit <= 0 else iter(range(limit))
    result = LoopResult(answer="")

    for iteration in iterations:
        if token_budget is not None and (
            result.input_tokens + result.output_tokens
        ) >= token_budget:
            result.hit_budget = True
            result.answer = (
                "Stopped: token budget for this task was reached. Partial "
                "progress is recorded in the session's tool messages."
            )
            return result

        contents, c_in, c_out = await _maybe_compact(
            llm, contents, ledger, tool_ctx.session_id
        )
        result.input_tokens += c_in
        result.output_tokens += c_out

        turn = await llm.generate(
            contents, system_prompt, function_declarations=declarations
        )
        result.input_tokens += turn.input_tokens
        result.output_tokens += turn.output_tokens
        await ledger.create(
            tool_ctx.session_id, llm.model, turn.input_tokens, turn.output_tokens
        )

        if not turn.function_calls:
            result.answer = turn.text
            return result

        if turn.content is not None:
            contents.append(turn.content)

        response_parts: list[types.Part] = []
        for call in turn.function_calls:
            trace = await _execute_tool(tool_ctx, messages, call, response_parts)
            result.traces.append(trace)
            if on_event is not None:
                await on_event(
                    "tool",
                    {"tool": trace.tool, "result_summary": trace.result_summary},
                )
        contents.append(types.Content(role="user", parts=response_parts))
        logger.info(
            "agent loop iteration %d: %d tool call(s)",
            iteration + 1,
            len(turn.function_calls),
        )

    result.hit_limit = True
    result.answer = (
        "Stopped after reaching the iteration limit. Partial progress is "
        "recorded in the session's tool messages."
    )
    return result


async def _execute_tool(
    ctx: ToolContext,
    messages: MessageRepository,
    call: types.FunctionCall,
    response_parts: list[types.Part],
) -> ToolCallTrace:
    """Run one tool call, append its function response, persist a trace."""

    name = call.name or ""
    args: dict[str, Any] = dict(call.args or {})
    tool = TOOLS_BY_NAME.get(name)
    session_id = ctx.session_id

    if tool is None:
        result: dict[str, Any] = {"error": f"Unknown tool '{name}'"}
    elif tool.breakable and tool_guard.is_tripped(session_id, name):
        result = {
            "error": (
                f"Tool '{name}' is disabled for this session after repeated "
                "failures. Use a different tool or approach."
            )
        }
    elif tool_cache.is_cacheable(name) and (
        cached := tool_cache.get(session_id, name, args)
    ) is not None:
        logger.info("tool cache hit: %s", name)
        result = {**cached, "cached": True}
    else:
        try:
            result = await asyncio.wait_for(
                tool.handler(ctx, args), timeout=tool.timeout
            )
        except TimeoutError:
            logger.warning("Tool '%s' timed out after %ss", name, tool.timeout)
            result = {"error": f"Tool '{name}' timed out after {tool.timeout}s"}
        except Exception as exc:
            logger.exception("Tool '%s' failed", name)
            result = {"error": f"{type(exc).__name__}: {exc}"}

        failed = "error" in result
        if tool.breakable:
            if failed:
                tool_guard.record_failure(session_id, name)
            else:
                tool_guard.record_success(session_id, name)
        if not failed and tool_cache.is_cacheable(name):
            tool_cache.put(session_id, name, args, result)

    response_parts.append(
        types.Part.from_function_response(name=name, response={"result": result})
    )

    summary = json.dumps(result, ensure_ascii=False, default=str)
    if len(summary) > TRACE_SUMMARY_CHARS:
        summary = summary[:TRACE_SUMMARY_CHARS] + "…"
    await messages.create(
        ctx.session_id,
        "tool",
        json.dumps(
            {"tool": name, "args": args, "result_summary": summary},
            ensure_ascii=False,
            default=str,
        ),
    )
    return ToolCallTrace(tool=name, args=args, result_summary=summary)


_COMPACT_SYSTEM = (
    "You compress an agent's intermediate work into a concise, factual summary. "
    "Preserve the concrete findings, data, URLs, artifact ids, and decisions so "
    "the agent can continue without the raw detail. Omit chit-chat."
)


def _content_text(content: types.Content) -> str:
    parts: list[str] = []
    for part in content.parts or []:
        if part.text:
            parts.append(part.text)
        elif part.function_call is not None:
            parts.append(f"[call {part.function_call.name}]")
        elif part.function_response is not None:
            resp = str(part.function_response.response)[:2000]
            parts.append(f"[result {part.function_response.name}: {resp}]")
    return " ".join(parts)


async def _maybe_compact(
    llm: LlmClient,
    contents: list[types.Content],
    ledger: LedgerRepository,
    session_id: Any,
) -> tuple[list[types.Content], int, int]:
    """Summarize old middle rounds when the loop context grows too large.

    Keeps the first message (the task) and the most recent rounds verbatim,
    replacing everything between them with a single LLM-written summary so a
    long tool loop stays within the model window (progressive summarization).
    """

    keep = max(settings.context_keep_last_rounds * 2, 2)
    if len(contents) <= keep + 2:
        return contents, 0, 0
    size = sum(len(_content_text(c)) for c in contents)
    # Threshold scales with the active model's context window (with an optional
    # hard cap from settings to bound memory), so a big-window model actually
    # uses its window instead of compacting early.
    threshold = compact_threshold_chars(
        getattr(llm, "model", None), settings.context_compact_chars
    )
    if size <= threshold:
        return contents, 0, 0

    middle = contents[1:-keep]
    middle_text = "\n".join(_content_text(c) for c in middle)[:40000]
    summary = await llm.generate(
        [build_user_content(f"Summarize this earlier activity:\n\n{middle_text}")],
        _COMPACT_SYSTEM,
    )
    await ledger.create(
        session_id, llm.model, summary.input_tokens, summary.output_tokens
    )
    note = build_user_content(f"[Summary of earlier steps]:\n{summary.text}")
    logger.info("compacted loop context (%d chars) into a summary", size)
    return (
        [contents[0], note, *contents[-keep:]],
        summary.input_tokens,
        summary.output_tokens,
    )


def format_history(messages: list[Any], max_turns: int | None = None) -> str:
    """Render recent user/assistant turns as a conversation preamble.

    Lets a follow-up research run continue with awareness of earlier turns in
    the same session. Tool traces are excluded; the "[revise] " marker is
    stripped. Turn count and per-turn size are configurable so the large model
    context window can be used generously. Returns "" when there is no history.
    """

    limit = max_turns or settings.history_max_turns
    per_turn = settings.history_turn_chars
    turns: list[tuple[str, str]] = []
    for message in messages:
        if message.role == "user":
            turns.append(("User", message.content.removeprefix("[revise] ")))
        elif message.role == "assistant":
            turns.append(("Assistant", message.content))
    turns = turns[-limit * 2 :]
    if not turns:
        return ""
    body = "\n".join(f"{role}: {text[:per_turn]}" for role, text in turns)
    return f"Earlier in this conversation:\n{body}"


def remaining_token_budget(tokens_used: int) -> int | None:
    """Tokens left before the per-session cap, or None if uncapped."""

    if settings.session_token_budget <= 0:
        return None
    return max(settings.session_token_budget - tokens_used, 0)


def format_artifacts(artifacts: list[Any]) -> str:
    """Render a session's artifacts as a note the model can act on.

    Lets the agent discover uploaded/downloaded files (by id, kind, and
    summary) so it can parse, search, or analyze them. Returns "" if none.
    """

    if not artifacts:
        return ""
    lines = [
        f"- {a.name} (artifact_id={a.id}, kind={a.kind}): {a.summary}"
        for a in artifacts
    ]
    return (
        "Files available in this session (reference them by artifact_id):\n"
        + "\n".join(lines)
    )


def build_user_content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def build_model_content(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


__all__ = [
    "LoopResult",
    "run_agent_loop",
    "build_user_content",
    "build_model_content",
]
