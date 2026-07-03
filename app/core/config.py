"""Application configuration.

Exposes a typed `Settings` object loaded from environment variables
(and an optional `.env` file) via `pydantic-settings`.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings.

    Values are read from the environment (case-insensitive) and from a
    `.env` file if present. Unknown keys in `.env` are ignored.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Omni-Agent"
    env: str = "development"
    debug: bool = False
    # Level for noisy third-party libraries (browser_use, crawl4ai, …).
    third_party_log_level: str = "WARNING"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: Annotated[list[str], NoDecode] = ["*"]
    database_url: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    # Tuned for accuracy/robustness — persist through blocks, replan freely.
    # Iteration caps accept 0 (or any non-positive value) to mean "no limit":
    # the loop then runs until the model answers / stop / no-progress guard.
    max_agent_iterations: int = 120
    max_plan_iterations: int = 60
    max_plan_nodes: int = 48
    subagent_max_iterations: int = 60
    # Self-reflection rounds per sub-agent: after producing a result it critiques
    # itself ("is this sufficient? what more can I do?") and keeps gathering.
    subagent_max_reflections: int = 2
    browser_use_max_steps: int = 24
    llm_max_retries: int = 6
    session_token_budget: int = 0  # 0 = unlimited
    # Context sizes — Gemini 3 models have a 1M-token window, so feed generously.
    tool_excerpt_chars: int = 48000  # parse_document / crawl_url / read_artifact
    doc_section_chars: int = 24000  # doc_navigate per-section text
    history_max_turns: int = 24  # conversation turns fed back into planning
    history_turn_chars: int = 8000  # per-turn char cap in the history digest
    # Web search: ddgs region (e.g. wt-wt worldwide, us-en, uk-en, in-en). The
    # default is worldwide/no-country-bias; the model may override per query.
    search_region: str = "wt-wt"
    # Robustness
    tool_default_timeout: float = 360.0  # seconds; per-tool overrides on Tool
    circuit_trip_threshold: int = 3  # tool failures before it's disabled/session
    subagent_verify: bool = True  # flag refusal/empty sub-agent answers as failed
    # Hard cap on a loop's context before compaction; the effective threshold is
    # min(this, a fraction of the active model's window) — see app/core/models.py.
    context_compact_chars: int = 1_500_000
    context_keep_last_rounds: int = 8  # tool-exchange rounds kept verbatim
    data_dir: str = "data"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Allow a comma-separated string for `CORS_ORIGINS` env vars."""

        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


# Single settings instance used across the app
settings = Settings()


__all__ = ["Settings", "settings"]
