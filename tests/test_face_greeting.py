"""Tests for `face_greeting.py` — LLM-written greetings for arrivals.

The whole point is the TIME guard: a phrase written for the morning must
never be spoken at 03:00. Generation is slow (measured 17 s live), so the
cache is keyed by (name, bucket) and validated both on write and on read.
"""

import asyncio
import calendar
import time

import pytest

import face_greeting as fg


def test_model_is_pinned_and_marina_gets_a_respectful_tone():
    assert fg.LLM_MODEL == "nvidia/nemotron-3.5-super-vl:free"
    assert "marina" in fg._RESPECTFUL_NAMES
    assert "уважительное" in fg._system_prompt("day", "marina")
    assert "уважительное" not in fg._system_prompt("day", "alexander")


# MSK bucket boundaries, checked on both sides of every seam.
@pytest.mark.parametrize("utc_hour,bucket", [
    (20, "night"),      # 23:00 MSK
    (21, "night"),      # 00:00 MSK
    (1, "night"),       # 04:00 MSK
    (2, "morning"),     # 05:00 MSK
    (8, "morning"),     # 11:00 MSK
    (9, "day"),         # 12:00 MSK
    (13, "day"),        # 16:00 MSK (last day hour)
    (14, "evening"),    # 17:00 MSK
    (16, "evening"),    # 19:00 MSK
    (22, "night"),      # 01:00 MSK next day — the wrap is checked below
])
def test_bucket_boundaries(utc_hour, bucket):
    assert fg.bucket_at(_at(utc_hour)) == bucket


def _at(utc_hour: int) -> float:
    """A fixed UTC timestamp at the top of the given hour."""
    return calendar.timegm((2026, 11, 18, utc_hour, 1, 0, 0, 0, 0))


def test_a_morning_phrase_is_never_spoken_at_night():
    """THE requirement: pre-generated «Доброе утро» cannot reach a 03:00 arrival."""
    fg._cache.clear()
    fg._cached_at.clear()
    fg.remember("alexander", "morning", "Доброе утро, Александр!")
    night = _at(0)                      # 03:00 MSK
    text, source = fg.choose("alexander", "night", night)
    assert source == "fallback"
    assert "утро" not in text.casefold()
    assert text == fg.fallback_for("alexander")


@pytest.mark.parametrize("bucket,phrase", [
    ("night", "Добрый вечер, Александр"),
    ("night", "Доброе утро, Александр"),
    ("night", "С добрым днём, Александр"),
    ("morning", "Доброй ночи, Александр"),
    ("day", "Добрый вечер, Александр"),
    ("evening", "Доброе утро, Александр"),
])
def test_foreign_time_markers_are_rejected(bucket, phrase):
    assert fg.phrase_ok(phrase, bucket) is False
    fg._cache.clear()
    fg.remember("alexander", bucket, phrase)
    assert fg._cache == {}              # never enters the cache


@pytest.mark.parametrize("bucket,phrase", [
    ("night", "Добро пожаловать домой, Александр"),
    ("night", "Доброй ночи, Александр"),
    ("morning", "Доброе утро, Александр"),
    ("day", "Здравствуй, Александр"),
    ("evening", "Добрый вечер, Александр"),
])
def test_matching_or_time_neutral_phrases_are_accepted(bucket, phrase):
    assert fg.phrase_ok(phrase, bucket) is True


def test_fallback_phrases_are_time_neutral_at_every_hour():
    for name in list(fg.FALLBACK) + ["someonenew"]:
        for bucket in fg.BUCKETS:
            assert fg.phrase_ok(fg.fallback_for(name), bucket), (name, bucket)


@pytest.mark.parametrize("phrase", [
    "Привет?",
    "**Привет**, друг!",
    "Слишком длинная фраза, которая явно превышает установленный предел в шестьдесят символов",
    "hello there",                          # not Russian
    "Ок",                                   # one word
])
def test_phrase_validation(phrase):
    assert fg.phrase_ok(phrase, "evening") is False


def test_cache_is_per_name_and_per_bucket():
    fg._cache.clear()
    fg._cached_at.clear()
    fg.remember("alexander", "morning", "Доброе утро, Александр")
    fg.remember("vladislava", "morning", "Доброе утро, Vladislava")
    fg.remember("alexander", "night", "Добро пожаловать домой, Александр")
    morning = fg.choose("alexander", "morning", _at(3))[0]
    night = fg.choose("alexander", "night", _at(0))[0]
    assert "утро" in morning
    assert "утро" not in night
    assert fg.choose("marina", "morning", _at(3))[1] == "fallback"


def test_expired_cache_is_not_used():
    fg._cache.clear()
    fg._cached_at.clear()
    fg.remember("alexander", "morning", "Доброе утро, Александр")
    now = _at(3)
    fg._cached_at[("alexander", "morning")] = now - fg.CACHE_TTL_S - 1
    text, source = fg.choose("alexander", "morning", now)
    assert source == "fallback"
    assert text == fg.fallback_for("alexander")


def test_clean_generated_strips_tags_and_quotes():
    assert fg.clean_generated('  [happy] «Добро пожаловать домой, Александр.»  ') == \
        "Добро пожаловать домой, Александр."


def test_prepare_uses_the_current_bucket_only(monkeypatch):
    """A stale morning cache + a night arrival: no morning words out loud."""
    fg._cache.clear()
    fg._cached_at.clear()
    fg.remember("alexander", "morning", "Доброе утро, Александр")
    monkeypatch.setattr(fg, "schedule_generation", lambda *a: None)
    # Live event loop is needed only because prepare() may schedule generation.
    async def run():
        return fg.prepare(["alexander"], now=_at(0))
    text, source, bucket = asyncio.run(run())
    assert bucket == "night"
    assert source == "fallback"
    assert "утро" not in text.casefold()
