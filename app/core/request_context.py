"""Request-scoped context (country) shared with deep, layer-crossing code.

The user's country is needed in two places that are far from the request
handler and read a module-level global rather than an injected dependency: the
LLM system-prompt stamp (`LlmClient._with_context`) and the scraping browser
fingerprint (`tools/browser_session.random_fingerprint`). Threading it through
every LLM call and tool would touch many signatures, so instead the API layer
sets it once per request in a `ContextVar` and those call sites read it back.

Resolution order: the per-request value the frontend detects from the visitor's
browser (set from the `X-User-Country` header) overrides `settings.user_country`
(the env global default). `ContextVar` copies into child tasks at creation, so
the value propagates into the streamed research run and its sub-agents.
"""

from __future__ import annotations

from contextvars import ContextVar

from app.core.config import settings

_country_var: ContextVar[str | None] = ContextVar("user_country", default=None)


def set_current_country(value: str | None) -> None:
    """Set (or clear) the current request's country override."""

    _country_var.set(value)


def current_country() -> str:
    """The active country: the request override if present, else the env default."""

    override = _country_var.get()
    if override and override.strip():
        return override.strip()
    return settings.user_country


__all__ = ["set_current_country", "current_country"]
