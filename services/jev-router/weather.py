"""Deterministic weather sentences for the easy_query weather branch.

Hybrid chain (user decision, 25.09.2026): open-meteo -> Hermes L3 ->
complex_logic. The normal path involves no LLM at all: during E2E the free
model refused («нет доступа к прогнозу») and Hermes named a wrong date, while
a deterministic forecast is instant, honest and cannot invent anything.

Coordinates come from HA /api/config (cached after the first call), so no
city configuration is needed. Any network/parse failure returns [] — the
caller then falls back to Hermes, and if that yields nothing either, the
existing `query_unresolved -> complex_logic` escalation takes over.

Pure helpers (pick_target / build_sentences / _wmo) are stdlib-only and
unit-tested on the host (tests/test_weather.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.request
from datetime import date

import config

logger = logging.getLogger("router.weather")

_OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

# 8 s is a deliberate compromise: a voice turn tolerates ~2-3 s of silence
# before it feels broken, and the caller has the Hermes failover after us —
# hanging longer here would eat the whole budget of the answer.
_TIMEOUT = 8.0

# WMO weather interpretation codes -> short Russian phrases. The phrasing is
# built to fit both «Сейчас на улице X, {desc}.» and «{label} ожидается
# {desc}...», so keep the values nominative and TTS-friendly.
_WMO = {
    0: "ясно",
    1: "малооблачно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозь с туманом",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "ледяная морось",
    57: "сильная ледяная морось",
    61: "небольшой дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "ледяной дождь",
    67: "сильный ледяной дождь",
    71: "небольшой снег",
    73: "снег",
    75: "сильный снег",
    77: "снежная крупа",
    80: "кратковременный ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "снежный заряд",
    86: "сильный снежный заряд",
    95: "гроза",
    96: "гроза с градом",
    99: "сильная гроза с градом",
}

# Weekday stems -> spoken label; iso weekday index (Monday = 0).
# Matched with re.search(r"\b" + stem): only the START of a word, so the
# stem «сред» does not fire inside unrelated words, while «в среду» and
# «среда» are both covered by the single stem. The label is already in the
# right case («В среду», not «В среда») — do not "fix" it to nominative.
_WEEKDAYS = (
    (r"понедельн", "В понедельник", 0),
    (r"вторник", "Во вторник", 1),
    (r"сред", "В среду", 2),       # среда / среду
    (r"четверг", "В четверг", 3),
    (r"пятниц", "В пятницу", 4),
    (r"суббот", "В субботу", 5),
    (r"воскресень", "В воскресенье", 6),
)

# Day-after is checked BEFORE tomorrow because «послезавтра» contains no
# «завтра» as a separate word (\b barrier), but ordering keeps the intent
# obvious; both are one-shot matches on the lowercased utterance.
_RE_DAY_AFTER = re.compile(r"послезавтра")
_RE_TOMORROW = re.compile(r"\bзавтра\b", re.IGNORECASE)


def pick_target(text: str) -> tuple[str, int]:
    """What part of the forecast the question is about.

    -> ("day", 1|2) for tomorrow/day-after,
    -> ("weekday", iso_weekday) for «в пятницу» style questions,
    -> ("now", 0) for current conditions / no day word.
    """
    t = (text or "").lower()
    if _RE_DAY_AFTER.search(t):
        return "day", 2
    if _RE_TOMORROW.search(t):
        return "day", 1
    for stem, _label, iso_wd in _WEEKDAYS:
        if re.search(r"\b" + stem, t):
            return "weekday", iso_wd
    return "now", 0


def _wmo(code) -> str:
    """WMO code -> Russian phrase; anything unparseable/unknown degrades to
    a neutral «условия уточняются» instead of raising — an invented detail
    is worse than a vague one for a spoken forecast."""
    try:
        key = int(code)
    except (TypeError, ValueError):
        return "условия уточняются"
    return _WMO.get(key, "условия уточняются")


def _deg(v) -> str:
    """Round to whole degrees: the TTS reads «минус пять» fine, but
    «минус пять целых четыре десятых» would make the answer useless."""
    return f"{round(float(v))}°"


def build_sentences(text: str, data: dict) -> list[str]:
    """Pure builder from an open-meteo payload. [] means 'cannot answer'.

    No I/O and no clock reads (day offsets come from the payload's own
    `daily.time[0]`, which open-meteo anchors to the requested timezone) —
    that is what makes this function unit-testable on the host and immune
    to "yesterday's forecast" bugs around midnight.
    """
    kind, arg = pick_target(text)

    if kind == "now":
        cur = data.get("current") or {}
        temp, code = cur.get("temperature_2m"), cur.get("weather_code")
        if temp is None or code is None:
            return []
        s = f"Сейчас на улице {_deg(temp)}, {_wmo(code)}"
        wind = cur.get("wind_speed_10m")
        if wind is not None:
            s += f", ветер {round(float(wind))} м/с"
        return [s + "."]

    daily = data.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        return []
    codes = daily.get("weather_code") or []
    t_max = daily.get("temperature_2m_max") or []
    t_min = daily.get("temperature_2m_min") or []
    probs = daily.get("precipitation_probability_max") or []

    if kind == "day":
        idx = arg
        label = "Завтра" if arg == 1 else "Послезавтра"
    else:  # weekday: index relative to the first forecast day (=today)
        today = date.fromisoformat(str(times[0])[:10])
        idx = (arg - today.weekday()) % 7
        label = next(lbl for stem, lbl, wd in _WEEKDAYS if wd == arg)
    if idx >= len(times):
        return []  # beyond the 7-day window -> caller falls back to Hermes

    try:
        code, mx, mn = codes[idx], t_max[idx], t_min[idx]
    except (IndexError, TypeError):
        return []
    if mx is None or code is None:
        return []

    out = [
        f"{label} ожидается {_wmo(code)}: днём до {_deg(mx)}, "
        f"ночью до {_deg(mn)}."
    ]
    try:
        prob = probs[idx]
    except IndexError:
        prob = None
    if prob is not None and float(prob) >= 30:
        out.append(f"Вероятность осадков {round(float(prob))}%.")
    return out


# --- I/O -------------------------------------------------------------------

def _http_json(url: str, headers: dict | None = None) -> dict:
    """Blocking urllib GET -> parsed JSON.

    Runs only inside asyncio.to_thread (see callers): the rest of the
    router is async, and a sync socket here would stall the whole event
    loop for up to _TIMEOUT seconds. urllib instead of aiohttp on purpose —
    this module keeps a stdlib-only import surface so tests/test_weather.py
    can import the pure helpers on the host, where aiohttp may be absent.
    """
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


# Process-lifetime cache of the HA coordinates. Written only from the event
# loop (single writer) and only after a successful load, so a failed first
# attempt is retried on the next request instead of caching None forever.
_coords: tuple[float, float] | None = None


async def _ha_coords() -> tuple[float, float]:
    """Lat/lon from HA /api/config, fetched once per process.

    Using HA's own location keeps "погода" consistent with what the HA
    frontend shows and removes any need for a city env var (README: "no
    city configuration is needed").
    """
    global _coords
    if _coords is not None:
        return _coords

    def _load() -> tuple[float, float]:
        """Blocking HA /api/config read (latitude/longitude) for the thread."""
        cfg = _http_json(
            f"{config.HA_URL}/api/config",
            {"Authorization": f"Bearer {config.HA_TOKEN}"},
        )
        return float(cfg["latitude"]), float(cfg["longitude"])

    _coords = await asyncio.to_thread(_load)
    return _coords


async def weather_sentences(text: str) -> list[str]:
    """Fetch + build. Returns [] on any failure (caller escalates).

    [] is the ONLY failure signal — there is no exception path on purpose:
    the route handler treats "nothing yielded" as «not solved here» and
    tries the Hermes stream next, then query_unresolved -> complex_logic.
    That ordering is the hybrid chain decided by the user (25.09.2026).
    """
    try:
        lat, lon = await _ha_coords()
        url = (
            f"{_OPEN_METEO}?latitude={lat:.6f}&longitude={lon:.6f}"
            "&current=temperature_2m,weather_code,wind_speed_10m"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max"
            "&timezone=auto&forecast_days=7"
        )
        data = await asyncio.to_thread(_http_json, url)
        return build_sentences(text, data)
    except Exception as e:  # noqa: BLE001 — any failure means "not solved here"
        logger.warning("open-meteo unavailable: %s", e)
        return []
