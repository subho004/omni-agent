"""Country/locale context helpers (pure, stateless).

One small source of truth mapping a configured country (``settings.user_country``)
to the regional signals the rest of the app needs:

- the LLM prompt stamp — so every agent shares the user's country alongside the
  current date/day for location-relative reasoning; and
- the scraping browser fingerprint — locale, timezone, capital geolocation and
  an ``Accept-Language`` header so crawls present the right region and geo-gated
  content resolves correctly.

The table is intentionally small (common countries only); an unknown or empty
value resolves to ``None`` and callers fall back to their default behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CountryProfile:
    """Regional context for one country."""

    code: str  # ISO-3166 alpha-2, uppercase
    name: str  # display name
    locale: str  # BCP-47, e.g. "en-IN"
    timezone_id: str  # IANA tz, e.g. "Asia/Kolkata"
    latitude: float  # capital city — approximate geolocation
    longitude: float

    @property
    def accept_language(self) -> str:
        """An ``Accept-Language`` header value for this locale (with base fallback)."""

        base = self.locale.split("-")[0]
        return f"{self.locale},{base};q=0.9" if base != self.locale else self.locale

    @property
    def label(self) -> str:
        """Human-readable one-liner for the prompt stamp."""

        return (
            f"{self.name} ({self.code}); locale {self.locale}; "
            f"timezone {self.timezone_id}"
        )


# Keyed by ISO-2 code. Coordinates are the capital city (approximate).
_COUNTRIES: tuple[CountryProfile, ...] = (
    CountryProfile("US", "United States", "en-US", "America/New_York", 38.8951, -77.0364),
    CountryProfile("GB", "United Kingdom", "en-GB", "Europe/London", 51.5074, -0.1278),
    CountryProfile("IN", "India", "en-IN", "Asia/Kolkata", 28.6139, 77.2090),
    CountryProfile("CA", "Canada", "en-CA", "America/Toronto", 45.4215, -75.6972),
    CountryProfile("AU", "Australia", "en-AU", "Australia/Sydney", -33.8688, 151.2093),
    CountryProfile("DE", "Germany", "de-DE", "Europe/Berlin", 52.5200, 13.4050),
    CountryProfile("FR", "France", "fr-FR", "Europe/Paris", 48.8566, 2.3522),
    CountryProfile("ES", "Spain", "es-ES", "Europe/Madrid", 40.4168, -3.7038),
    CountryProfile("IT", "Italy", "it-IT", "Europe/Rome", 41.9028, 12.4964),
    CountryProfile("NL", "Netherlands", "nl-NL", "Europe/Amsterdam", 52.3676, 4.9041),
    CountryProfile("BR", "Brazil", "pt-BR", "America/Sao_Paulo", -23.5505, -46.6333),
    CountryProfile("JP", "Japan", "ja-JP", "Asia/Tokyo", 35.6762, 139.6503),
    CountryProfile("SG", "Singapore", "en-SG", "Asia/Singapore", 1.3521, 103.8198),
    CountryProfile("AE", "United Arab Emirates", "en-AE", "Asia/Dubai", 25.2048, 55.2708),
    CountryProfile("ZA", "South Africa", "en-ZA", "Africa/Johannesburg", -26.2041, 28.0473),
    CountryProfile("MX", "Mexico", "es-MX", "America/Mexico_City", 19.4326, -99.1332),
    CountryProfile("NG", "Nigeria", "en-NG", "Africa/Lagos", 6.5244, 3.3792),
    CountryProfile("PK", "Pakistan", "en-PK", "Asia/Karachi", 33.6844, 73.0479),
    CountryProfile("BD", "Bangladesh", "en-BD", "Asia/Dhaka", 23.8103, 90.4125),
    CountryProfile("ID", "Indonesia", "id-ID", "Asia/Jakarta", -6.2088, 106.8456),
)

_BY_CODE: dict[str, CountryProfile] = {c.code: c for c in _COUNTRIES}
# Name + common-alias index (all lowercase) for forgiving lookup.
_BY_NAME: dict[str, CountryProfile] = {c.name.lower(): c for c in _COUNTRIES}
_BY_NAME.update(
    {
        "usa": _BY_CODE["US"],
        "united states of america": _BY_CODE["US"],
        "america": _BY_CODE["US"],
        "uk": _BY_CODE["GB"],
        "england": _BY_CODE["GB"],
        "britain": _BY_CODE["GB"],
        "great britain": _BY_CODE["GB"],
        "uae": _BY_CODE["AE"],
        "emirates": _BY_CODE["AE"],
    }
)


def resolve_country(value: str | None) -> CountryProfile | None:
    """Resolve a configured country string to a profile, or ``None``.

    Accepts an ISO-2 code (``"IN"``) or a country name/alias (``"India"``,
    ``"uk"``), case-insensitively. Unknown or empty input returns ``None`` so
    callers keep their default (un-localized) behaviour.
    """

    key = (value or "").strip()
    if not key:
        return None
    if len(key) == 2 and key.upper() in _BY_CODE:
        return _BY_CODE[key.upper()]
    return _BY_NAME.get(key.lower())


__all__ = ["CountryProfile", "resolve_country"]
