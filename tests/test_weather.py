"""Weather builder tests (hybrid chain, v2.28).

Pure helpers only — no network, runs on host python. The payload below is
a trimmed open-meteo response; daily.time[0] is Friday, so weekday offsets
are checked against it.
"""

import os
import sys

_ROUTER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "jev-router")
)
sys.path.insert(0, _ROUTER)

from weather import _wmo, build_sentences, pick_target  # noqa: E402

DATA = {
    "current": {"temperature_2m": 12.4, "weather_code": 61, "wind_speed_10m": 3.2},
    "daily": {
        # Fri, Sat, Sun
        "time": ["2026-09-25", "2026-09-26", "2026-09-27"],
        "weather_code": [3, 61, 0],
        "temperature_2m_max": [17.2, 15.8, 18.0],
        "temperature_2m_min": [9.1, 11.3, 8.7],
        "precipitation_probability_max": [10, 80, 5],
    },
}


def test_pick_target_day_words():
    assert pick_target("Какая завтра будет погода?") == ("day", 1)
    assert pick_target("А послезавтра?") == ("day", 2)


def test_pick_target_weekday_and_now():
    assert pick_target("погода в субботу") == ("weekday", 5)
    assert pick_target("Какая сейчас погода?") == ("now", 0)
    assert pick_target("что будет с погодой") == ("now", 0)


def test_now_sentence_with_wind():
    out = build_sentences("Какая сейчас погода?", DATA)
    assert out == ["Сейчас на улице 12°, небольшой дождь, ветер 3 м/с."]


def test_tomorrow_sentence_with_probability():
    out = build_sentences("Какая завтра будет погода?", DATA)
    assert out == [
        "Завтра ожидается небольшой дождь: днём до 16°, ночью до 11°.",
        "Вероятность осадков 80%.",
    ]


def test_day_after_skips_low_probability():
    out = build_sentences("что будет послезавтра", DATA)
    assert out == ["Послезавтра ожидается ясно: днём до 18°, ночью до 9°."]


def test_weekday_friday_maps_to_today():
    # 2026-09-25 is a Friday -> index 0
    out = build_sentences("Какая погода в пятницу?", DATA)
    assert out == ["В пятницу ожидается пасмурно: днём до 17°, ночью до 9°."]


def test_weekday_outside_window_falls_back():
    # Wednesday is index 5, only 3 forecast days in DATA -> [] (Hermes next)
    assert build_sentences("а в среду?", DATA) == []


def test_missing_current_returns_empty():
    assert build_sentences("погода сейчас", {"daily": DATA["daily"]}) == []


def test_wmo_mapping_and_fallback():
    assert _wmo(95) == "гроза"
    assert _wmo("61") == "небольшой дождь"
    assert _wmo(12345) == "условия уточняются"
    assert _wmo(None) == "условия уточняются"
