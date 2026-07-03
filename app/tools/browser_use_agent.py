"""Browser-use agent tool (docs/implementation-plan.md Phase 5).

Prompt-driven web automation: describe what to extract from a site in
natural language and a headless-browser agent navigates and returns the
result. Heavier and slower than crawl_url — use it only for interactive or
JS-driven flows (search boxes, logins, multi-step navigation) where a
plain crawl won't reach the content.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    task = str(args["task"])
    logger.info("browser_use: %s", task)

    # Imported lazily: browser-use pulls in a large dependency tree and drives
    # a real browser, so we only load it when the tool is actually invoked.
    try:
        from browser_use import Agent, ChatGoogle

        from app.tools.browser_session import new_browser_use_profile
    except Exception as exc:  # pragma: no cover - import/environment issue
        return {"error": f"browser-use unavailable: {exc}"}

    try:
        agent: Any = Agent(
            task=task,
            llm=ChatGoogle(
                model=settings.gemini_model, api_key=settings.gemini_api_key
            ),
            # Fresh, randomised browser identity for each run.
            browser_profile=new_browser_use_profile(headless=True),
        )
        history = await agent.run(max_steps=settings.browser_use_max_steps)
        result = history.final_result()
    except Exception as exc:
        logger.exception("browser_use failed")
        return {"error": f"Browser agent failed: {type(exc).__name__}: {exc}"}

    return {"result": result or "No result produced."}


browser_use_tool = Tool(
    name="browser_use",
    description=(
        "Drive a headless browser with a natural-language task to extract "
        "information from a website, including interactive or JavaScript-heavy "
        "pages (search boxes, multi-step navigation). Slower than crawl_url — "
        "prefer crawl_url for static pages and use this only when interaction "
        "is required. Describe exactly what to find and return."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "Natural-language instruction, e.g. 'Go to example.com, "
                    "search for X, and return the first result title.'"
                ),
            },
        },
        "required": ["task"],
    },
    handler=_handle,
    timeout=900.0,
    breakable=True,  # slow + flaky: disable for the session after repeat failures
)
