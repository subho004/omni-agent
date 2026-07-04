"""Shared browser-session factory for the scraping tools.

Both scraping tools (``crawl_url`` via crawl4ai and ``browser_use`` via
browser-use) spin up a *fresh* headless Chromium on every call. To make
scraping robust against bot-detection and fingerprinting, each session gets
a **randomised fingerprint** — a different viewport, user-agent, locale,
timezone, and device-scale-factor every time — plus **playwright-stealth**
evasions injected into the page. Centralising this here keeps the two tools
consistent and gives us one place to tune anti-bot behaviour.

When ``settings.user_country`` is configured, the locale/timezone are instead
**pinned** to that country (viewport/user-agent/device still randomised) and a
matching ``Accept-Language`` header + capital geolocation are sent, so crawls
present the user's region and geo-gated content resolves correctly.

Nothing here launches a browser itself; it only builds the config objects the
tools pass to their respective engines.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.core.request_context import current_country
from utils.geo import CountryProfile, resolve_country

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
    accept_language: str
    extra_args: list[str] = field(default_factory=list)
    # Set only when a country is configured; pins the page's reported location.
    latitude: float | None = None
    longitude: float | None = None


def random_fingerprint(country: CountryProfile | None = None) -> BrowserFingerprint:
    """Build a new, internally-consistent randomised fingerprint.

    Called once per browser session so every crawl looks like a different
    real user on different hardware. When a ``country`` is given (or the active
    country resolves — the request's ``X-User-Country`` or the env default), the
    locale/timezone are pinned to it and its capital geolocation +
    ``Accept-Language`` are attached, so scraping presents the user's region;
    otherwise a random locale/timezone is chosen (no geolocation) for
    anti-fingerprinting variety.
    """

    if country is None:
        country = resolve_country(current_country())

    device = random.choice(_DEVICE_PROFILES)
    width, height = random.choice(_VIEWPORTS)
    latitude: float | None = None
    longitude: float | None = None
    if country is not None:
        locale, timezone_id = country.locale, country.timezone_id
        accept_language = country.accept_language
        latitude, longitude = country.latitude, country.longitude
    else:
        locale, timezone_id = random.choice(_LOCALE_TZ)
        accept_language = f"{locale},{locale.split('-')[0]};q=0.9"

    fp = BrowserFingerprint(
        user_agent=device.user_agent,
        platform=device.platform,
        viewport_width=width,
        viewport_height=height,
        device_scale_factor=random.choice(device.scale_factors),
        locale=locale,
        timezone_id=timezone_id,
        accept_language=accept_language,
        latitude=latitude,
        longitude=longitude,
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
        "browser fingerprint: %s %dx%d %s/%s%s",
        fp.platform, fp.viewport_width, fp.viewport_height, fp.locale,
        fp.timezone_id, " (geo-pinned)" if country is not None else "",
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

    from crawl4ai import BrowserConfig, CrawlerRunConfig, GeolocationConfig

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
        headers={"Accept-Language": fp.accept_language},  # signal the region
    )
    # Pin the page's reported geolocation to the configured country's capital so
    # geo-gated content resolves there; omitted (None) when no country is set.
    geolocation = (
        GeolocationConfig(latitude=fp.latitude, longitude=fp.longitude)
        if fp.latitude is not None and fp.longitude is not None
        else None
    )
    run_config = CrawlerRunConfig(
        page_timeout=page_timeout_ms,
        locale=fp.locale,
        timezone_id=fp.timezone_id,
        geolocation=geolocation,
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
    # BrowserProfile has no locale/timezone_id/geolocation fields (they'd be
    # ignored), so we convey the region via the Accept-Language header — the one
    # country signal it does support — plus the dimensions it accepts.
    return BrowserProfile(
        headless=headless,
        user_agent=fp.user_agent,
        viewport=size,
        window_size=size,
        device_scale_factor=fp.device_scale_factor,
        args=fp.extra_args,
        headers={"Accept-Language": fp.accept_language},
    )


__all__ = [
    "BrowserFingerprint",
    "random_fingerprint",
    "stealth_init_script",
    "new_crawl4ai_configs",
    "new_browser_use_profile",
]
