"""Tests for the shared randomised, stealth-hardened browser-session helper."""

from __future__ import annotations

from app.tools.browser_session import (
    new_browser_use_profile,
    new_crawl4ai_configs,
    random_fingerprint,
    stealth_init_script,
)


def test_random_fingerprint_is_internally_consistent() -> None:
    fp = random_fingerprint()
    assert fp.viewport_width >= 1000 and fp.viewport_height >= 700
    assert "Mozilla/5.0" in fp.user_agent
    assert fp.device_scale_factor >= 1.0
    # Window-size flag matches the chosen viewport.
    assert f"--window-size={fp.viewport_width},{fp.viewport_height}" in fp.extra_args
    assert "--disable-blink-features=AutomationControlled" in fp.extra_args


def test_fingerprints_vary_across_calls() -> None:
    # Over many draws the identities should not all be identical.
    seen = {
        (fp.user_agent, fp.viewport_width, fp.viewport_height, fp.locale)
        for fp in (random_fingerprint() for _ in range(40))
    }
    assert len(seen) > 1


def test_stealth_script_is_available() -> None:
    script = stealth_init_script()
    # playwright-stealth is a project dependency; payload should be non-trivial.
    assert isinstance(script, str)
    assert len(script) > 1000


def test_crawl4ai_configs_apply_fingerprint_and_stealth() -> None:
    browser_config, run_config = new_crawl4ai_configs(
        headless=True, page_timeout_ms=45_000
    )
    assert browser_config.headless is True
    assert browser_config.enable_stealth is True
    assert browser_config.user_agent.startswith("Mozilla/5.0")
    assert browser_config.viewport_width >= 1000
    assert browser_config.init_scripts  # playwright-stealth payload injected
    assert run_config.page_timeout == 45_000
    assert run_config.simulate_user is True
    assert run_config.override_navigator is True


def test_browser_use_profile_builds_with_fingerprint() -> None:
    profile = new_browser_use_profile(headless=True)
    assert profile.headless is True
    assert profile.user_agent.startswith("Mozilla/5.0")
    assert profile.viewport is not None
