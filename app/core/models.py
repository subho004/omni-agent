"""Catalogue of selectable Gemini models and thinking levels.

This is the single place to edit when adding a model or thinking level — the
API exposes whatever is listed here, and the UI renders it in its dropdowns.
Add a new ``ModelOption`` to ``AVAILABLE_MODELS`` (or a new string to
``THINKING_LEVELS``) and it shows up automatically; no other code changes
needed. The LLM client validates any incoming choice against these lists and
falls back to the defaults for anything unknown.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True)
class ModelOption:
    """One selectable model: API id, label, and its token limits.

    ``context_window_tokens`` is the max input the model accepts; ``max_output_
    tokens`` is the max it can return. These drive how much context we feed and
    how far we let a loop grow before compacting (see helpers below). Values are
    from Google's Gemini 3 model docs (verified 2026-07: all three are 1M / 64k).
    """

    id: str
    label: str
    context_window_tokens: int = 1_000_000
    max_output_tokens: int = 65_536
    supports_thinking: bool = True


# ── Add / remove models here (set token limits from the model's docs) ─────
AVAILABLE_MODELS: tuple[ModelOption, ...] = (
    ModelOption(
        "gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite (fast, cheap)",
        context_window_tokens=1_000_000, max_output_tokens=65_536,
    ),
    ModelOption(
        "gemini-3.5-flash", "Gemini 3.5 Flash (balanced)",
        context_window_tokens=1_000_000, max_output_tokens=65_536,
    ),
    ModelOption(
        "gemini-3.1-pro-preview", "Gemini 3.1 Pro (preview, strongest)",
        context_window_tokens=1_000_000, max_output_tokens=65_536,
    ),
)

# Rough bytes→tokens ratio for English text; used to translate the token-based
# context window into the char-based budgets the loop/compaction reason about.
CHARS_PER_TOKEN = 4
# Compact a tool loop once it fills this fraction of the model's input window —
# generous enough to exploit the 1M window, short of the hard edge.
COMPACT_CONTEXT_FRACTION = 0.6

# ── Add / rename thinking levels here ─────────────────────────────────────
# Passed to Gemini as ThinkingConfig(thinking_level=...); higher = more
# internal reasoning (slower, pricier, usually more accurate).
THINKING_LEVELS: tuple[str, ...] = ("low", "medium", "high")

# ── Defaults (used for unknown/omitted choices) ───────────────────────────
_MODEL_IDS = {m.id for m in AVAILABLE_MODELS}
DEFAULT_MODEL = (
    settings.gemini_model if settings.gemini_model in _MODEL_IDS
    else AVAILABLE_MODELS[0].id
)
DEFAULT_THINKING_LEVEL = "low"


def resolve_model(model: str | None) -> str:
    """Return a valid model id, falling back to the default."""

    return model if model in _MODEL_IDS else DEFAULT_MODEL


def resolve_thinking_level(level: str | None) -> str:
    """Return a valid thinking level, falling back to the default."""

    return level if level in THINKING_LEVELS else DEFAULT_THINKING_LEVEL


_BY_ID = {m.id: m for m in AVAILABLE_MODELS}
_DEFAULT_OPTION = _BY_ID[DEFAULT_MODEL]


def option_for(model_id: str | None) -> ModelOption:
    """The ModelOption for an id, or the default option for unknown ids."""

    return _BY_ID.get(model_id or "", _DEFAULT_OPTION)


def context_window_tokens(model_id: str | None) -> int:
    return option_for(model_id).context_window_tokens


def max_output_tokens_for(model_id: str | None) -> int:
    return option_for(model_id).max_output_tokens


def compact_threshold_chars(model_id: str | None, hard_cap_chars: int = 0) -> int:
    """Char count at which a loop's context should be compacted for this model.

    Derived from the model's input window (a fraction of it, in chars). An
    optional ``hard_cap_chars`` (> 0) caps it so a huge window can't push memory
    use unbounded.
    """

    model_chars = int(
        context_window_tokens(model_id) * CHARS_PER_TOKEN * COMPACT_CONTEXT_FRACTION
    )
    if hard_cap_chars > 0:
        return min(model_chars, hard_cap_chars)
    return model_chars


__all__ = [
    "ModelOption",
    "AVAILABLE_MODELS",
    "THINKING_LEVELS",
    "DEFAULT_MODEL",
    "DEFAULT_THINKING_LEVEL",
    "CHARS_PER_TOKEN",
    "resolve_model",
    "resolve_thinking_level",
    "option_for",
    "context_window_tokens",
    "max_output_tokens_for",
    "compact_threshold_chars",
]
