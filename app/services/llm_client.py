"""Thin async wrapper around the google-genai SDK.

Centralises Gemini access: model/config selection, manual function-calling
(the executor loop owns tool dispatch), and token-usage extraction so every
call can be written to the ledger (docs/implementation-plan.md §4).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from google import genai
from google.genai import errors, types
from tenacity import (
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tenacity import AsyncRetrying as _AsyncRetrying

from app.core.config import settings
from app.core.logging import get_logger
from app.core.models import (
    DEFAULT_THINKING_LEVEL,
    max_output_tokens_for,
    resolve_thinking_level,
)

logger = get_logger(__name__)


def _with_datetime(system_instruction: str) -> str:
    """Prepend the current local date/time to a system prompt.

    Every model turn — planner, evaluator, reviser, reflection, synthesis and
    the tool-calling loop — funnels its system instruction through here, so all
    of them share a consistent, up-to-date "now" for recency/deadline reasoning
    (e.g. "latest", "this year", "as of today") instead of the training cutoff.
    """

    now = datetime.now().astimezone()
    stamp = format(now, "%A, %d %B %Y, %H:%M:%S (UTC%z)")
    header = (
        f"Current date and time: {stamp}. "
        f"Day of week: {now:%A}. Year: {now:%Y}. "
        "Treat this as 'now' for any date/time-relative reasoning."
    )
    return f"{header}\n\n{system_instruction}"


def _is_retryable(exc: BaseException) -> bool:
    """Transient Gemini/network failures worth retrying (429, 5xx, timeouts)."""

    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError):
        return getattr(exc, "code", None) == 429
    return isinstance(exc, (TimeoutError, ConnectionError))


@dataclass
class LlmResult:
    """Result of a single Gemini call."""

    text: str
    function_calls: list[types.FunctionCall] = field(default_factory=list)
    content: types.Content | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class LlmClient:
    """Async Gemini client with manual function-calling support."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> None:
        self._client = genai.Client(api_key=api_key or settings.gemini_api_key)
        self.model = model or settings.gemini_model
        self.thinking_level = thinking_level or DEFAULT_THINKING_LEVEL

    def configured_for(
        self, model: str | None = None, thinking_level: str | None = None
    ) -> LlmClient:
        """A lightweight clone with a different model / thinking level.

        Shares the underlying genai HTTP client (cheap), so per-session model
        selection doesn't re-create a connection each request. Unknown values
        fall back to the catalogue defaults.
        """

        clone = LlmClient.__new__(LlmClient)
        clone._client = self._client
        clone.model = model or self.model
        clone.thinking_level = resolve_thinking_level(
            thinking_level or self.thinking_level
        )
        return clone

    def _thinking_config(self) -> types.ThinkingConfig | None:
        """Map the selected level (low/medium/high) to a Gemini ThinkingConfig."""

        level = resolve_thinking_level(self.thinking_level)
        return types.ThinkingConfig(
            thinking_level=types.ThinkingLevel(level.upper())
        )

    def _max_output_tokens(self) -> int:
        """The model's max output cap — set explicitly so long, complete
        answers are never truncated below what the model supports."""

        return max_output_tokens_for(self.model)

    async def _generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        """Call the SDK with exponential backoff on transient failures."""

        async for attempt in _AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max(settings.llm_max_retries, 1)),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    logger.warning(
                        "Retrying Gemini call (attempt %d)",
                        attempt.retry_state.attempt_number,
                    )
                return await self._client.aio.models.generate_content(**kwargs)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def generate(
        self,
        contents: list[types.Content],
        system_instruction: str,
        function_declarations: list[types.FunctionDeclaration] | None = None,
    ) -> LlmResult:
        """Run one model turn; tool calls are returned, never auto-executed."""

        config = types.GenerateContentConfig(
            system_instruction=_with_datetime(system_instruction),
            thinking_config=self._thinking_config(),
            max_output_tokens=self._max_output_tokens(),
            tools=(
                [types.Tool(function_declarations=function_declarations)]
                if function_declarations
                else None
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        # The SDK's ContentListUnion is a union of list variants that mypy
        # can't match against list[Content] directly; cast is safe here.
        response = await self._generate_content(
            model=self.model,
            contents=cast("types.ContentListUnion", list(contents)),
            config=config,
        )

        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        output_tokens = (usage.candidates_token_count or 0) if usage else 0

        content: types.Content | None = None
        if response.candidates and response.candidates[0].content:
            content = response.candidates[0].content

        return LlmResult(
            text=response.text or "",
            function_calls=list(response.function_calls or []),
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def generate_structured(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: type,
    ) -> tuple[Any, int, int]:
        """Generate a response constrained to a Pydantic schema.

        Returns the parsed object plus (input_tokens, output_tokens). Used by
        the planner and evaluator, which need machine-readable output.
        """

        config = types.GenerateContentConfig(
            system_instruction=_with_datetime(system_instruction),
            thinking_config=self._thinking_config(),
            max_output_tokens=self._max_output_tokens(),
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        response = await self._generate_content(
            model=self.model, contents=prompt, config=config
        )
        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        output_tokens = (usage.candidates_token_count or 0) if usage else 0
        return response.parsed, input_tokens, output_tokens

    async def describe_image(
        self, prompt: str, image_bytes: bytes, mime_type: str
    ) -> tuple[str, int, int]:
        """Run a vision prompt over an image; returns text + token usage.

        Gemini is multimodal, so the same model handles images. Used both to
        auto-describe uploads and to answer image questions (analyze_image).
        """

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
            )
        ]
        response = await self._generate_content(
            model=self.model,
            contents=cast("types.ContentListUnion", contents),
            config=types.GenerateContentConfig(
                thinking_config=self._thinking_config(),
                max_output_tokens=self._max_output_tokens(),
            ),
        )
        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        output_tokens = (usage.candidates_token_count or 0) if usage else 0
        return response.text or "", input_tokens, output_tokens

    async def generate_stream(
        self,
        contents: list[types.Content],
        system_instruction: str,
        usage_sink: dict[str, int],
    ) -> AsyncIterator[str]:
        """Stream a plain-text response chunk by chunk.

        Token usage from the final chunk is written into ``usage_sink`` so the
        caller can ledger it after the stream is exhausted.
        """

        config = types.GenerateContentConfig(
            system_instruction=_with_datetime(system_instruction),
            thinking_config=self._thinking_config(),
            max_output_tokens=self._max_output_tokens(),
        )
        stream = await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=cast("types.ContentListUnion", list(contents)),
            config=config,
        )
        async for chunk in stream:
            usage = chunk.usage_metadata
            if usage:
                usage_sink["in"] = usage.prompt_token_count or 0
                usage_sink["out"] = usage.candidates_token_count or 0
            if chunk.text:
                yield chunk.text

    async def grounded_search(
        self, prompt: str
    ) -> tuple[str, list[str], int, int]:
        """Answer a query grounded in Google Search; returns text + sources.

        Uses Gemini's built-in google_search tool for fast factual discovery
        (docs/implementation-plan.md Phase 8). Source URLs come from the
        response's grounding metadata.
        """

        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
        response = await self._generate_content(
            model=self.model, contents=prompt, config=config
        )
        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        output_tokens = (usage.candidates_token_count or 0) if usage else 0
        return response.text or "", _extract_sources(response), input_tokens, output_tokens


def _extract_sources(response: Any) -> list[str]:
    """Pull deduped grounding source URLs from a response, if any."""

    sources: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        metadata = getattr(candidate, "grounding_metadata", None)
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None)
            if uri and uri not in sources:
                sources.append(uri)
    return sources


__all__ = ["LlmClient", "LlmResult"]
