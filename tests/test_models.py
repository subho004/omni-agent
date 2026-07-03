"""Tests for selectable models + thinking levels (catalogue, API, persistence)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.core.models import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    compact_threshold_chars,
    context_window_tokens,
    max_output_tokens_for,
    resolve_model,
    resolve_thinking_level,
)
from app.services.llm_client import LlmClient


def test_per_model_token_limits() -> None:
    assert context_window_tokens("gemini-3.1-pro-preview") == 1_000_000
    assert max_output_tokens_for("gemini-3.5-flash") == 65_536
    # Unknown model falls back to the default option's limits.
    assert context_window_tokens("who-knows") == context_window_tokens(DEFAULT_MODEL)


def test_compact_threshold_scales_with_window_and_respects_cap() -> None:
    # No cap: derived purely from the model window (a large number).
    uncapped = compact_threshold_chars("gemini-3.1-flash-lite")
    assert uncapped > 1_000_000
    # A hard cap wins when smaller than the model-derived threshold.
    assert compact_threshold_chars("gemini-3.1-flash-lite", 80_000) == 80_000


def test_resolve_falls_back_to_defaults() -> None:
    assert resolve_model("nonexistent-model") == DEFAULT_MODEL
    assert resolve_model(AVAILABLE_MODELS[-1].id) == AVAILABLE_MODELS[-1].id
    assert resolve_thinking_level("bogus") == "low"
    assert resolve_thinking_level("high") == "high"


def test_thinking_config_maps_level_to_enum() -> None:
    client = LlmClient.__new__(LlmClient)
    client.thinking_level = "medium"
    cfg = client._thinking_config()
    assert cfg is not None
    assert cfg.thinking_level.value == "MEDIUM"


def test_configured_for_clones_without_new_http_client() -> None:
    base = LlmClient(api_key="x")
    clone = base.configured_for("gemini-3.5-flash", "high")
    assert clone.model == "gemini-3.5-flash"
    assert clone.thinking_level == "high"
    assert clone._client is base._client  # shared underlying client
    # Unknown thinking level falls back to the default.
    assert base.configured_for("gemini-3.5-flash", "bogus").thinking_level == "low"


@pytest.mark.asyncio
async def test_model_options_endpoint(client: AsyncClient) -> None:
    data = (await client.get("/sessions/options/models")).json()["data"]
    ids = [m["id"] for m in data["models"]]
    assert "gemini-3.1-flash-lite" in ids
    assert "gemini-3.5-flash" in ids
    assert "gemini-3.1-pro-preview" in ids
    assert data["thinking_levels"] == ["low", "medium", "high"]
    # Per-model token limits are surfaced for the UI.
    assert all(m["context_window_tokens"] > 0 for m in data["models"])
    assert all(m["max_output_tokens"] > 0 for m in data["models"])


@pytest.mark.asyncio
async def test_session_persists_model_choice(client: AsyncClient) -> None:
    created = (
        await client.post(
            "/sessions",
            json={"title": "s", "model": "gemini-3.5-flash", "thinking_level": "high"},
        )
    ).json()["data"]
    assert created["model"] == "gemini-3.5-flash"
    assert created["thinking_level"] == "high"

    # Update via the settings endpoint.
    resp = await client.post(
        f"/sessions/{created['id']}/settings",
        json={"model": "gemini-3.1-pro-preview", "thinking_level": "low"},
    )
    assert resp.status_code == 200
    updated = resp.json()["data"]
    assert updated["model"] == "gemini-3.1-pro-preview"
    assert updated["thinking_level"] == "low"


@pytest.mark.asyncio
async def test_session_create_rejects_unknown_model_gracefully(
    client: AsyncClient,
) -> None:
    # Unknown model falls back to the default rather than erroring.
    created = (
        await client.post("/sessions", json={"title": "s", "model": "made-up"})
    ).json()["data"]
    assert created["model"] == DEFAULT_MODEL
