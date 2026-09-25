"""Honesty veto tests: an action claim without a confirmed ha_action must
never be spoken (field case: «Кафеварка включена!» after MatchFailedError).

Runs offline on host python: honesty.py is pure stdlib, no smolagents.
"""

import os
import sys

_WORKER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "smolagents-worker")
)
sys.path.insert(0, _WORKER)

from honesty import vet_answer, vet_weather  # noqa: E402

_FAIL_NAME = [
    {"tool": "intent__HassTurnOn", "ok": False,
     "detail": "Тул вернул ошибку: MatchFailedReason.NAME: 1, name='кафеварка'"}
]
_FAIL_ASSISTANT = [
    {"tool": "intent__HassTurnOn", "ok": False,
     "detail": "Тул вернул ошибку: MatchFailedReason.ASSISTANT: 8"}
]
_OK = [{"tool": "intent__HassTurnOn", "ok": True, "detail": '{"success": true}'}]


def test_veto_replaces_claim_after_failed_action():
    """The exact field regression: claim + only failed actions -> truth."""
    out, replaced = vet_answer("Кафеварка включена!", _FAIL_NAME)
    assert replaced is True
    assert "Не нашла такого устройства" in out


def test_veto_keeps_honest_failure():
    out, replaced = vet_answer("Не получилось включить, устройство не найдено.",
                               _FAIL_NAME)
    assert replaced is False
    assert out == "Не получилось включить, устройство не найдено."


def test_no_veto_when_action_succeeded():
    out, replaced = vet_answer("Кафеварка включена!", _OK)
    assert replaced is False and out == "Кафеварка включена!"


def test_no_veto_without_recorded_events():
    """No ha_action at all (e.g. only ha_read) — nothing to prove."""
    out, replaced = vet_answer("Кафеварка включена!", [])
    assert replaced is False


def test_success_after_earlier_failure_passes():
    events = _FAIL_NAME + _OK
    out, replaced = vet_answer("Включил!", events)
    assert replaced is False


def test_unexposed_device_message():
    out, replaced = vet_answer("Готово, включил!", _FAIL_ASSISTANT)
    assert replaced is True and "ассистент" in out


def test_non_action_answers_untouched():
    out, replaced = vet_answer("Расскажу анекдот: заходит кактус...", _FAIL_NAME)
    assert replaced is False

# --- Weather veto (field case 25.09.2026) ---------------------------------
# Tool recorded the real forecast, the model wrote final_answer in the same
# code block (before reading it) and distorted both cloudiness and numbers.

_FORECAST = [{"tool": "weather_forecast",
              "detail": "Завтра ожидается пасмурно: днём до 16°, ночью до 9°."}]
_LIE = ("Завтра будет переменная облачность, без осадков. Днём температура "
        "составит +22.5°C, а ночью +12.1°C.")
_FAILED = [{"tool": "weather_forecast",
            "detail": "Прогноз погоды недоступен (сетевая ошибка или нет данных)."}]


def test_weather_lie_replaced_with_recorded_forecast():
    out, replaced = vet_weather(_LIE, _FORECAST)
    assert replaced is True
    assert out == _FORECAST[0]["detail"]


def test_faithful_forecast_passes():
    honest = "Завтра пасмурно, днём до 16 градусов, ночью около 9."
    out, replaced = vet_weather(honest, _FORECAST)
    assert replaced is False and out == honest


def test_weather_numbers_must_match_even_if_conditions_do():
    # cloudiness carries over but the numbers are invented -> replace
    wrong_nums = "Завтра пасмурно, днём до 22 градусов, ночью до 12."
    out, replaced = vet_weather(wrong_nums, _FORECAST)
    assert replaced is True and out == _FORECAST[0]["detail"]


def test_failed_fetch_vetoes_forecast_claim():
    out, replaced = vet_weather("Завтра будет солнечно и тепло.", _FAILED)
    assert replaced is True
    assert "недоступен" in out


def test_failed_fetch_admission_passes():
    honest = "Не смогла получить прогноз погоды."
    out, replaced = vet_weather(honest, _FAILED)
    assert replaced is False and out == honest


def test_no_weather_events_no_veto():
    out, replaced = vet_weather(_LIE, [])
    assert replaced is False and out == _LIE


def test_weather_veto_skipped_when_action_attempted():
    """Whole-answer replacement must not wipe an action report."""
    out, replaced = vet_weather(_LIE, _FORECAST, action_attempted=True)
    assert replaced is False and out == _LIE
