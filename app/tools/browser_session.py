"""Shared browser-session factory for the scraping tools.

Both scraping tools (``crawl_url`` via crawl4ai and ``browser_use`` via
browser-use) spin up a *fresh* headless Chromium on every call. To make
scraping robust against bot-detection and fingerprinting, each session gets
a **randomised fingerprint** — a different viewport, user-agent, locale,
timezone, and device-scale-factor every time — plus **playwright-stealth**
evasions injected into the page. Centralising this here keeps the two tools
consistent and gives us one place to tune anti-bot behaviour.

Nothing here launches a browser itself; it only builds the config objects the
tools pass to their respective engines.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.logging import get_logger

if TYPE_CHECKING:
    from browser_use import BrowserProfile
    from crawl4ai import BrowserConfig, CrawlerRunConfig

logger = get_logger(__name__)

# Realistic desktop viewports (width, height). One is picked per session so
# no two crawls share the same window geometry.
_VIEWPORTS: tuple[tuple[int, int], ...] = (
    (1920, 1080),
    (1680, 1050),
    (1600, 900),
    (1536, 864),
    (1512, 982),
    (1470, 956),
    (1440, 900),
    (1366, 768),
    (1280, 800),
    (2560, 1440),
)

# Chrome builds paired with a consistent OS platform + sec-ch-ua hints. Keeping
# UA and platform internally consistent avoids an obvious fingerprint mismatch.
@dataclass(frozen=True)
class _DeviceProfile:
    user_agent: str
    platform: str
    scale_factors: tuple[float, ...]


_DEVICE_PROFILES: tuple[_DeviceProfile, ...] = (
    _DeviceProfile(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
        ),
        platform="Win32",
        scale_factors=(1.0, 1.25, 1.5),
    ),
    _DeviceProfile(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 "
            "Safari/537.36"
        ),
        platform="MacIntel",
        scale_factors=(1.0, 2.0),
    ),
    _DeviceProfile(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
        ),
        platform="Win32",
        scale_factors=(1.0, 1.25),
    ),
    _DeviceProfile(
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
        ),
        platform="Linux x86_64",
        scale_factors=(1.0,),
    ),
)

# Locale paired with a plausible timezone so the two agree.
_LOCALE_TZ: tuple[tuple[str, str], ...] = (
    ("en-US", "America/New_York"),
    ("en-US", "America/Chicago"),
    ("en-US", "America/Los_Angeles"),
    ("en-GB", "Europe/London"),
    ("en-CA", "America/Toronto"),
    ("en-AU", "Australia/Sydney"),
    ("en-IN", "Asia/Kolkata"),
)


@dataclass
class BrowserFingerprint:
    """One randomised browser identity, generated fresh per scraping call."""

    user_agent: str
    platform: str
    viewport_width: int
    viewport_height: int
    device_scale_factor: float
    locale: str
    timezone_id: str
    extra_args: list[str] = field(default_factory=list)


def random_fingerprint() -> BrowserFingerprint:
    """Build a new, internally-consistent randomised fingerprint.

    Called once per browser session so every crawl looks like a different
    real user on different hardware.
    """

    device = random.choice(_DEVICE_PROFILES)
    width, height = random.choice(_VIEWPORTS)
    locale, timezone_id = random.choice(_LOCALE_TZ)
    fp = BrowserFingerprint(
        user_agent=device.user_agent,
        platform=device.platform,
        viewport_width=width,
        viewport_height=height,
        device_scale_factor=random.choice(device.scale_factors),
        locale=locale,
        timezone_id=timezone_id,
        # Flags that reduce automation fingerprints; window-size matches the
        # chosen viewport so the OS-level window and the page agree.
        extra_args=[
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
            f"--window-size={width},{height}",
        ],
    )
    logger.debug(
        "browser fingerprint: %s %dx%d %s/%s",
        fp.platform, fp.viewport_width, fp.viewport_height, fp.locale,
        fp.timezone_id,
    )
    return fp


@lru_cache(maxsize=1)
def stealth_init_script() -> str:
    """playwright-stealth evasion JS to inject on every page (built once).

    Returns an empty string if the library is unavailable, so scraping still
    works (just without the extra evasions).
    """

    try:
        from playwright_stealth import Stealth

        return Stealth().script_payload
    except Exception as exc:  # pragma: no cover - optional dependency/runtime
        logger.warning("playwright-stealth unavailable: %s", exc)
        return ""


def new_crawl4ai_configs(
    *, headless: bool = True, page_timeout_ms: int
) -> tuple[BrowserConfig, CrawlerRunConfig]:
    """Build (BrowserConfig, CrawlerRunConfig) for one fresh crawl4ai crawl.

    Applies a random fingerprint, native + playwright-stealth evasions, and
    light human-like simulation. Returns config objects only — the caller
    owns the AsyncWebCrawler lifecycle (a new instance per call).
    """

    from crawl4ai import BrowserConfig, CrawlerRunConfig

    fp = random_fingerprint()
    init_scripts = [s for s in (stealth_init_script(),) if s]
    browser_config = BrowserConfig(
        headless=headless,
        viewport_width=fp.viewport_width,
        viewport_height=fp.viewport_height,
        user_agent=fp.user_agent,
        device_scale_factor=fp.device_scale_factor,
        enable_stealth=True,  # crawl4ai's native evasions …
        init_scripts=init_scripts,  # … plus playwright-stealth's payload
        extra_args=fp.extra_args,
    )
    run_config = CrawlerRunConfig(
        page_timeout=page_timeout_ms,
        locale=fp.locale,
        timezone_id=fp.timezone_id,
        simulate_user=True,  # small human-like mouse/scroll jitter
        override_navigator=True,  # spoof navigator.* to match the fingerprint
        magic=True,  # crawl4ai's grab-bag of anti-bot handling
    )
    return browser_config, run_config


def new_browser_use_profile(*, headless: bool = True) -> BrowserProfile:
    """Build a fresh, randomised browser-use BrowserProfile.

    browser-use drives a patchright (stealth) Chromium; we add a random
    fingerprint on top so each agent run presents a different identity.
    """

    from browser_use import BrowserProfile
    from browser_use.browser.profile import ViewportSize

    fp = random_fingerprint()
    size = ViewportSize(width=fp.viewport_width, height=fp.viewport_height)
    # BrowserProfile has no locale/timezone_id fields (they'd be ignored), so
    # we only set the fingerprint dimensions it actually supports.
    return BrowserProfile(
        headless=headless,
        user_agent=fp.user_agent,
        viewport=size,
        window_size=size,
        device_scale_factor=fp.device_scale_factor,
        args=fp.extra_args,
    )


__all__ = [
    "BrowserFingerprint",
    "random_fingerprint",
    "stealth_init_script",
    "new_crawl4ai_configs",
    "new_browser_use_profile",
]
