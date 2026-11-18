"""LLM-written greetings for people arriving at the front door.

The arrival path must be FAST: measured door→spoken greeting is already
6-20 s, while a live Hermes generation takes 17.2 s (and OmniRoute's
gemma4_31b_free 11.9 s, with broken Russian). So generation happens in the
BACKGROUND and the arrival picks a phrase from a cache — instantly.

THE ONE RULE THIS MODULE EXISTS FOR
    A phrase generated for the morning must never be spoken at night.
    Phrases are cached by (name, TIME BUCKET), a selection uses the bucket
    of the CURRENT clock only, and `phrase_ok` rejects any phrase carrying
    a time marker from another bucket ("Доброе утро" at 03:00, "Добрый
    вечер" at 08:00, ...). A rejected generation never enters the cache.

Fallback phrases carry NO time marker at all, so they are safe at any hour.
Nothing here imports `main` (the same cycle rule as telegram_client).
"""

import asyncio
import os
import re
import time
import logging

import aiohttp

logger = logging.getLogger("face_greeting")

# MSK, the household clock. The host/container runs UTC (documented in
# AGENTS.md).
TZ_OFFSET_H = 3

# Ordered so that bucket_at() is a single pass. Night wraps around midnight.
BUCKETS = ("night", "morning", "day", "evening")

# A phrase mentioning a DIFFERENT part of the day is invalid for this bucket.
# The night bucket gets extra protection: an outright time-neutral phrase is
# preferred there, and any morning/day/evening marker fails the check.
TIME_MARKERS = {
    "morning": ("доброе утро", "утром", "утренн", "с утра",
                "good morning", "morning"),
    "day": ("добрый день", "днём", "днем", "дневн", "полдень",
            "good afternoon", "afternoon"),
    "evening": ("добрый вечер", "вечером", "вечерн", "к вечеру",
                "good evening", "evening"),
    "night": ("доброй ночи", "спокойной ночи", "ночью", "ночн",
              "полночь", "good night"),
}

# These are spoken when no generated phrase is cached yet. Deliberately
# time-neutral: the previous fixed phrases are fine, they just must never
# claim it is morning at 03:00.
FALLBACK = {
    "alexander": "Здравствуй, мой господин",
    "vladislava": "Здравствуй, маленькая госпожа",
    "marina": "Здравствуй, Марина!",
}
FALLBACK_DEFAULT = "Здравствуй, добро пожаловать домой"

MAX_CHARS = 60
CACHE_TTL_S = 48 * 3600          # a phrase older than two days is re-made
_GEN_TIMEOUT_S = 30.0            # measured Hermes call: 17.2 s
_TAG_RE = re.compile(r"\[[a-z]{2,30}\]")
_MD_RE = re.compile(r"[*`_#>|]")

# Cache keyed by (person, bucket) — NEVER by person alone.
_cache: dict[tuple[str, str], list[str]] = {}
_cached_at: dict[tuple[str, str], float] = {}
_inflight: set[tuple[str, str]] = set()


def bucket_at(now: float | None = None) -> str:
    """Part of the day in MSK: night 23-05, morning 05-12, day 12-17, evening 17-23."""
    if now is None:
        now = time.time()
    hour = time.gmtime(now + TZ_OFFSET_H * 3600).tm_hour
    if hour < 5 or hour >= 23:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "day"
    return "evening"


def _has_foreign_time_marker(text: str, bucket: str) -> bool:
    """A time word belonging to ANY OTHER bucket — the night-good-morning guard."""
    folded = text.casefold()
    for other, markers in TIME_MARKERS.items():
        if other == bucket:
            continue
        if any(m in folded for m in markers):
            return True
    return False


def clean_generated(text: str) -> str:
    """Strip the emotion tag / quotes / markdown an LLM may wrap a phrase in."""
    text = _TAG_RE.sub("", text or "").strip()
    text = _MD_RE.sub("", text).strip().strip("\"'«»").strip()
    return re.sub(r"\s+", " ", text)


def phrase_ok(text: str, bucket: str) -> bool:
    """Accept only a short, speakable, time-correct phrase for THIS bucket."""
    if not text or len(text) > MAX_CHARS or "?" in text or "？" in text:
        return False
    if not re.search(r"[А-Яа-яЁё]", text):   # must be Russian for this TTS
        return False
    if _MD_RE.search(text) or _TAG_RE.search(text):
        return False
    if _has_foreign_time_marker(text, bucket):
        return False
    if len(text.split()) < 2:                # a single word is not a greeting
        return False
    return True


def fallback_for(name: str) -> str:
    return FALLBACK.get((name or "").strip().lower(), FALLBACK_DEFAULT)


def remember(name: str, bucket: str, text: str) -> None:
    """Cache a validated phrase for exactly one person and one time bucket."""
    if not phrase_ok(text, bucket):
        return
    key = ((name or "default").strip().lower(), bucket)
    phrases = _cache.setdefault(key, [])
    if text not in phrases:
        phrases.append(text)
    _cached_at[key] = time.time()


def choose(name: str, bucket: str, now: float | None = None) -> tuple[str, str]:
    """Pick an instantly available phrase. Never borrow from another bucket."""
    if now is None:
        now = time.time()
    key = ((name or "default").strip().lower(), bucket)
    phrases = _cache.get(key) or []
    if phrases and now - _cached_at.get(key, 0.0) < CACHE_TTL_S:
        # Re-validate on the way out: a bad cache entry must not be spoken,
        # even if it got there through a bug in a future writer.
        good = [p for p in phrases if phrase_ok(p, bucket)]
        if good:
            return good[int(now) % len(good)], "llm"
    return fallback_for(name), "fallback"


def _system_prompt(bucket: str) -> str:
    return (
        "Ты голосовой ассистент дома. Напиши ОДНУ короткую фразу на русском "
        f"(до {MAX_CHARS} символов), которой можно поприветствовать человека, "
        "только что вошедшего домой. Обращайся по имени. Без вопросов, без "
        "списков, без разметки, без кавычек и без пояснений."
        f" Сейчас {bucket}. Используй приветствие, подходящее именно для этого "
        "времени суток; ни в коем случае не упоминай другое время суток."
    )


async def generate(name: str, bucket: str) -> str:
    """Ask Hermes for one phrase, validate it, and cache it if it is usable."""
    key = ((name or "default").strip().lower(), bucket)
    if key in _inflight:
        return ""
    url = os.getenv("HERMES_API_URL", "").rstrip("/")
    api_key = os.getenv("HERMES_API_KEY", "")
    if not url or not api_key:
        logger.warning("face greeting: HERMES_API_URL/key unset — no LLM phrases")
        return ""
    _inflight.add(key)
    try:
        payload = {
            "model": "omniroute/oc/hy3-free",
            "messages": [
                {"role": "system", "content": _system_prompt(bucket)},
                {"role": "user", "content": f"Имя: {name or 'друг'}. Сейчас {bucket}."},
            ],
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=_GEN_TIMEOUT_S)
        started = time.monotonic()
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(f"{url}/v1/chat/completions",
                                 json=payload, headers=headers) as resp:
                if resp.status != 200:
                    logger.warning(f"face greeting: Hermes HTTP {resp.status}")
                    return ""
                data = await resp.json()
        raw = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        text = clean_generated(raw)
        took = time.monotonic() - started
        if not phrase_ok(text, bucket):
            logger.warning(f"face greeting: rejected {text!r} for {name}/{bucket}")
            return ""
        remember(name, bucket, text)
        logger.info(f"face greeting: generated {text!r} for {name}/{bucket} "
                    f"in {took:.1f}s")
        return text
    except Exception as exc:
        logger.warning(f"face greeting: generation failed for {name}/{bucket}: {exc!r}")
        return ""
    finally:
        _inflight.discard(key)


def schedule_generation(name: str, bucket: str) -> None:
    """Generate for NEXT arrival — the current one has already been spoken."""
    key = ((name or "default").strip().lower(), bucket)
    if key in _inflight:
        return
    task = asyncio.create_task(generate(name, bucket))
    task.add_done_callback(
        lambda fut: logger.debug(f"face greeting generation done: {fut.exception()!r}")
        if fut.cancelled() is False and fut.exception() else None
    )


async def warm(names: list[str]) -> None:
    """Background startup generation: first arrival then already has an LLM phrase."""
    bucket = bucket_at()
    wanted = [(n.strip().lower(), bucket) for n in names if n and n.strip()]
    if not wanted:
        return
    sem = asyncio.Semaphore(2)

    async def one(name: str, part: str) -> None:
        async with sem:
            await generate(name, part)

    await asyncio.gather(*(one(n, b) for n, b in wanted), return_exceptions=True)


def prepare(names: list[str], now: float | None = None) -> tuple[str, str, str]:
    """Return (text, source, bucket) for an arrival — instant, cache-only."""
    if now is None:
        now = time.time()
    bucket = bucket_at(now)
    people = [n for n in (names or []) if n and n.strip()]
    primary = people[0] if people else "default"
    text, source = choose(primary, bucket, now)
    if source == "fallback":
        schedule_generation(primary, bucket)
    logger.info(f"face greeting: name={primary!r} bucket={bucket} "
                f"source={source} text={text!r}")
    return text, source, bucket
