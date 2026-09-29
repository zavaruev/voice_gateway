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

from honesty import failure_note, vet_answer, vet_weather  # noqa: E402

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


# --- Vacuum field case 25.09.2026 -------------------------------------------
# «Пусть робот уберется на кухне» -> HassVacuumStart failed with AREA (robot
# is assigned to no area in HA), the model promised «отправляю робота» in the
# same code block — and the old claim regex had no «отправляю»/«запускаю».

_FAIL_AREA = [
    {"tool": "vacuum__HassVacuumStart", "ok": False,
     "detail": "Тул вернул ошибку: MatchFailedReason.AREA: 2, "
               "areas=[AreaEntry(name='Kitchen')]"}
]


def test_veto_catches_vacuum_promise():
    """The exact field case: promise after AREA failure -> truthful refusal."""
    out, replaced = vet_answer("Хорошо, отправляю робота-пылесоса на кухню!",
                               _FAIL_AREA)
    assert replaced is True
    assert "не привязано к этой комнате" in out


def test_veto_catches_start_and_clean_verbs():
    for claim in ("Запустил пылесос!", "Начинаю уборку на кухне!",
                  "Пылесос убирается, всё чисто!"):
        out, replaced = vet_answer(claim, _FAIL_AREA)
        assert replaced is True, claim


def test_negated_status_not_vetoed():
    """«робот не убирается» — status report, not a promise."""
    out, replaced = vet_answer("Робот не убирается — команда не прошла.",
                               _FAIL_AREA)
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


# --- failure_note: what the bounded retry after a veto is told --------------
# Field case 28.09.2026: two of five turns died on a veto with no second
# attempt — the model believed it acted, only the tool disagreed.


def test_failure_note_carries_the_raw_error():
    note = failure_note(_FAIL_NAME)
    assert "MatchFailedReason.NAME" in note and "кафеварка" in note
    # Guard rails: correct the target, do not blind-repeat, do not promise.
    assert "иначе" in note and "не повторяй" in note and "не заявляй" in note


def test_failure_note_is_raw_not_speakable():
    """_truth() would hand the model a polished refusal to repeat."""
    assert failure_note(_FAIL_NAME) != vet_answer("Кафеварка включена!", _FAIL_NAME)[0]


def test_failure_note_empty_when_nothing_failed():
    assert failure_note([]) == ""
    assert failure_note(_OK) == ""


# --- Data-claim veto (field case 29.09.2026) --------------------------------
# «Что там с нашим пылесосом?» -> ha_read returned
# `{"success": false, "error": "No exposed entities matched name 'пылесос'"}`
# and the model still dictated «сейчас работает, уровень заряда сорок пять
# процентов». HA in fact reports `docked` — and that vacuum has no battery
# attribute at all, so BOTH halves were invented. vet_answer() only looked at
# ha_action events, saw an empty log and passed it.

_READ_FAIL = [
    {"tool": "ha_read", "ok": False,
     "detail": '{"success": false, "error": "No exposed entities matched '
               "name 'пылесос'\"}"}
]
_READ_OK = [
    {"tool": "ha_read", "ok": True,
     "detail": "vacuum.valetudo_x: docked (Roborock Robot)"}
]
_READ_MISS = [{"tool": "ha_read", "ok": False,
               "detail": "Ничего не найдено в Home Assistant."}]
_FIELD_LIE = "Пылесос сейчас работает, уровень заряда сорок пять процентов."


def test_data_claim_after_failed_read_is_vetoed():
    """The literal field case: state AND percentage, no successful read."""
    out, replaced = vet_answer(_FIELD_LIE, [], _READ_FAIL)
    assert replaced is True
    assert "не нашла" in out.lower()


def test_data_claim_after_a_miss_is_vetoed():
    """A miss recorded as success would disarm the veto — guard the recorder
    as well as the veto."""
    out, replaced = vet_answer("Пылесос сейчас работает.", [], _READ_MISS)
    assert replaced is True


def test_word_numbers_count_as_claims():
    """This model writes numbers in words («сорок пять», not «45»), so the
    claim is a word list, not a digit regex."""
    assert "сорок пять" in _FIELD_LIE
    assert vet_answer("Пылесос заряжен на сорок пять процентов.", [], _READ_FAIL)[1]


def test_honest_state_passes_when_read_succeeded():
    out, replaced = vet_answer("Пылесос стоит на базе.", [], _READ_OK)
    assert replaced is False and out == "Пылесос стоит на базе."


def test_admission_passes_after_failed_read():
    honest = "Не нашла пылесос — устройства нет в Home Assistant."
    out, replaced = vet_answer(honest, [], _READ_FAIL)
    assert replaced is False and out == honest


def test_no_recorded_events_stays_fail_open():
    out, replaced = vet_answer(_FIELD_LIE, [], [])
    assert replaced is False and out == _FIELD_LIE


def test_failed_read_does_not_veto_non_data_answers():
    out, replaced = vet_answer("Расскажу анекдот: заходит кактус...", [], _READ_FAIL)
    assert replaced is False


def test_negated_state_is_a_report_not_a_claim():
    out, replaced = vet_answer("Пылесос не работает.", [], _READ_FAIL)
    assert replaced is False


def test_successful_read_does_not_disarm_the_action_veto():
    """The two proofs are separate: a state read cannot vouch for a promise."""
    out, replaced = vet_answer("Кафеварка включена!", _FAIL_NAME, _READ_OK)
    assert replaced is True
    assert "не нашла" in out.lower()


def test_failed_read_leaves_the_action_veto_untouched():
    out, replaced = vet_answer("Кафеварка включена!", _FAIL_NAME, _READ_FAIL)
    assert replaced is True


def test_failure_note_carries_a_failed_read():
    """The bounded retry must see the raw read error, not the speakable truth."""
    note = failure_note(_READ_FAIL)
    assert "No exposed entities" in note and "пылесос" in note
