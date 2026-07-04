"""Specialized reasoning agent tool: plan / decide / discover.

A pure-thinking tool (no external I/O). When an agent hits a problem that
needs multi-step, deliberate reasoning — devising a non-obvious plan, choosing
between competing approaches, or working out *how* to discover something —
it calls ``deep_think`` to have the model reason it through in one focused,
structured pass and hand back a plan/decision it can then execute with the
other tools.

It runs on the session's OWN model and thinking level: ``ctx.llm`` is already
``configured_for`` the session, so this reasoning inherits the exact model and
low/medium/high thinking budget the user selected — no separate config. Output
is constrained to ``ThinkingResult`` so the model must frame the problem, weigh
distinct options, commit to a recommendation, and lay out next steps rather
than answering in one undirected shot.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.schemas.research import ThinkingResult
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

_SYSTEM = (
    "You are a specialized reasoning agent — a planner, decision-maker, and "
    "discovery strategist. You are handed a hard problem that needs deliberate, "
    "multi-step thinking rather than a quick answer. Do not answer reflexively. "
    "First understand what is REALLY being asked and the crux of the problem. "
    "Surface the facts, constraints, assumptions, and unknowns that matter. "
    "Then generate two or more genuinely DISTINCT approaches — reach for a "
    "non-obvious angle, not just the first one — and weigh their trade-offs "
    "honestly. Commit to the strongest recommendation and justify it, then break "
    "it into concrete, ordered next steps that another agent could execute with "
    "real tools. Be specific and actionable; never hand back vague advice. Flag "
    "what still needs verifying and the main risks. You have no external tools "
    "here — this is pure reasoning; the plan you produce is what gets acted on."
)


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.llm is None or not hasattr(ctx.llm, "generate_structured"):
        return {"error": "Deep reasoning unavailable (no LLM client on context)"}

    problem = str(args.get("problem") or "").strip()
    if not problem:
        return {"error": "Provide a 'problem' describing what to reason about."}

    context = str(args.get("context") or "").strip()
    goal = str(args.get("goal") or "").strip()
    prompt_parts = [f"Problem to reason about:\n{problem}"]
    if context:
        prompt_parts.append(f"Relevant context / what is already known:\n{context}")
    if goal:
        prompt_parts.append(f"Desired outcome / goal:\n{goal}")
    prompt = "\n\n".join(prompt_parts)

    logger.info("deep_think: %s", problem[:120])
    result, in_tok, out_tok = await ctx.llm.generate_structured(
        prompt, _SYSTEM, ThinkingResult
    )
    if ctx.ledger is not None:
        await ctx.ledger.create(ctx.session_id, ctx.llm.model, in_tok, out_tok)

    if isinstance(result, ThinkingResult):
        return result.model_dump()
    # Schema-constrained output should always parse; guard defensively so a rare
    # empty/None response surfaces cleanly instead of crashing the tool loop.
    return {"error": "Deep reasoning produced no structured result; try rephrasing."}


deep_think_tool = Tool(
    name="deep_think",
    description=(
        "Reason through a hard problem in one focused, structured pass to get "
        "back a concrete plan or decision. Use this when the next move is NOT "
        "obvious and needs deliberate, multi-step thinking: devising a strategy, "
        "choosing between competing approaches, deciding, or working out HOW to "
        "discover/verify something. It weighs distinct options, commits to a "
        "recommendation, and returns ordered next steps you then execute with the "
        "other tools. This is pure reasoning — it uses NO web/file/browser tools "
        "itself, so don't use it to look things up; use it to figure out what to "
        "do. Runs on the current model and thinking level."
    ),
    parameters={
        "type": "object",
        "properties": {
            "problem": {
                "type": "string",
                "description": (
                    "The problem, question, or decision to reason through — what "
                    "you are stuck on or need a plan for."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional: relevant facts already known, findings so far, or "
                    "constraints that should shape the reasoning."
                ),
            },
            "goal": {
                "type": "string",
                "description": (
                    "Optional: the desired outcome or what a good answer/plan "
                    "must achieve."
                ),
            },
        },
        "required": ["problem"],
    },
    handler=_handle,
    # Higher thinking levels can reason for a while; give it headroom.
    timeout=300.0,
)
