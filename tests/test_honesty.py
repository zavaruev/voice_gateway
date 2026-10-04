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

from honesty import (  # noqa: E402
    failure_note,
    mark_tool_call,
    reset_run_state,
    reset_tool_call,
    same_block_check,
    vet_answer,
    vet_weather,
)

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


# --- GROUNDING: a SUCCESSFUL read still has to carry the claim ---------------
# Second occurrence of the same lie, 29.09.2026 14:17 — «Что у нас с
# роботом?»: ha_read succeeded (ten entity lines), so gate 2 ("one confirmed
# result of that kind") disarmed the data veto completely, and the model
# dictated «Робот сейчас убирает в гостиной, заряд батареи шестьдесят пять
# процентов». Neither was in those ten lines: no battery entity exists for
# that vacuum, and the vacuum's own `docked` had been cut off by match_states'
# cap (fixed in ha_match). A successful read of SOME entities proves nothing
# about ANY entity — the claim's subject must be IN the payload.

_READ_1417 = [
    {"tool": "ha_read", "ok": True,
     "detail": "update.vacuum_card_update: off (Vacuum Card Update)\n"
               "sensor.valetudo_zealouseverlastinggaur_map_segments: 8 "
               "(Roborock Map segments)"},
]
_LIE_1417 = (
    "Робот сейчас убирает в гостиной, заряд батареи шестьдесят пять процентов."
)


def test_ungrounded_state_and_battery_after_a_successful_read_is_vetoed():
    """The exact 14:17 transcript. The EARLIEST fabricated claim picks the
    replacement sentence — the user hears about the first lie."""
    out, replaced = vet_answer(_LIE_1417, [], _READ_1417)
    assert replaced is True
    assert "убирается" in out  # «убирает» стоит в ответе раньше «заряда»


def test_battery_only_claim_is_vetoed():
    out, replaced = vet_answer("Заряд батареи шестьдесят пять процентов.", [],
                               _READ_1417)
    assert replaced is True
    assert "заряд" in out.lower()


def test_grounded_state_passes():
    """`docked` in the payload + «на базе» in the answer: honest, untouched."""
    out, replaced = vet_answer("Робот на базе.", [], _READ_OK)
    assert replaced is False and out == "Робот на базе."


def test_percentage_number_must_appear_in_the_payload():
    """Battery present but the number is not: the digits are invented too."""
    reads = [{"tool": "ha_read", "ok": True,
              "detail": "sensor.valetudo_battery: 82 (%)"}]
    assert vet_answer("Заряд батареи 65 процентов.", [], reads)[1] is True
    # The real number passes (comma/dot and rounding tolerance are handled).
    assert vet_answer("Заряд батареи 82 процента.", [], reads)[1] is False


def test_grounding_is_skipped_after_a_confirmed_action():
    """«включил» -> «включено» is true without a fresh read: a confirmed
    side effect backs the state claim, so the read taken BEFORE the action
    must not veto it."""
    out, replaced = vet_answer("Свет включён.", _OK, _READ_1417)
    assert replaced is False and out == "Свет включён."


def test_weather_numbers_are_not_vetoed_as_ungrounded():
    """«15 градусов» may come from weather_forecast instead of a sensor
    read — the forecast is passed in as evidence for exactly that."""
    weather = [{"tool": "weather_forecast", "detail": "На улице 15 градусов, пасмурно."}]
    assert vet_answer("На улице 15 градусов, пасмурно.", [], _READ_1417, weather)[1] is False
    # Same answer, forecast not recorded -> the read carries no temperature.
    assert vet_answer("На улице 15 градусов.", [], _READ_1417)[1] is True


def test_non_data_answers_are_never_grounded():
    out, replaced = vet_answer("Расскажу анекдот: заходит кактус...", [], _READ_1417)
    assert replaced is False


# --- Same-block guard: the answer must come AFTER the tool output ------------
# smolagents runs `final_answer_checks` before accepting an answer; raising
# there records an error on the step and the run continues, so the model
# gets a fresh step where the tool output is already in its context.


def _refused() -> bool:
    try:
        same_block_check("ответ", None, None)
    except AssertionError:
        return True
    return False


def test_guard_refuses_an_answer_written_with_the_tool():
    reset_run_state()
    mark_tool_call()          # tools.py does this the moment a tool starts
    try:
        same_block_check("ответ", None, None)
        raise AssertionError("the guard must refuse an answer written with a tool")
    except AssertionError as e:
        # …and the message is an instruction for the NEXT step, not a log line.
        assert "прочитай" in str(e) and "final_answer отдельно" in str(e)


def test_guard_lets_a_clean_answer_through():
    reset_run_state()
    assert same_block_check("ответ", None, None) is True


def test_guard_ignores_a_tool_from_a_previous_step():
    """app._end_step clears the flag after every step: a tool in step 1 and
    a lone final_answer in step 2 is the CORRECT flow and must pass."""
    reset_run_state()
    mark_tool_call()
    reset_tool_call()         # step_callbacks does exactly this
    assert same_block_check("ответ", None, None) is True


def test_guard_fires_once_and_rearms_per_run():
    reset_run_state()
    mark_tool_call()
    refusals = sum(1 for _ in range(3) if _refused())
    assert refusals == 1  # a model that repeats itself must not burn MAX_STEPS
    reset_run_state()     # …but the next attempt starts armed again
    mark_tool_call()
    assert _refused() is True


# --- Media transport (03.10.2026) -------------------------------------------
# The turn was honest («К сожалению, не удалось поставить медиаплеер на паузу»)
# only by luck: the model had invented intent__HassMediaPause, spent two steps
# on «Tool … not found» and then reported failure. Two things must hold now —
# a claim about a player is a claim, and the two new blockers have their own
# sentences instead of a generic «Home Assistant отклонил команду».
_FAIL_NO_TOOL = [{"tool": "intent__HassMediaPause", "ok": False,
                  "detail": 'Тул вернул ошибку: в Home Assistant нет тула '
                            '«intent__HassMediaPause». Для паузы, треков и '
                            'громкости тул media_control; …'}]
_FAIL_NO_PLAYER = [{"tool": "media_control", "ok": False,
                    "detail": "Тул вернул ошибку: не нашла медиаплеер, к "
                              "которому это относится"}]
_FAIL_SOME_PLAYERS = [{"tool": "media_control", "ok": False,
                       "detail": "Тул вернул ошибку: подходит несколько "
                                 "медиаплееров (media_player.le_vlada)"}]
_FAIL_MEDIA = [{"tool": "media_control", "ok": False,
                "detail": "Тул вернул ошибку: Home Assistant отклонил "
                          "media_pause (not a valid entity id)"}]


def test_veto_covers_media_promises():
    """«Поставила на паузу» is exactly as much a promise as «включила»: the
    claim list had no transport verb at all, so a failed media_control with
    that sentence was spoken as if it had happened."""
    for claim in ("Поставила на паузу.", "Переключила трек.",
                  "Заглушила телевизор."):
        out, replaced = vet_answer(claim, _FAIL_MEDIA)
        assert replaced is True, claim
        assert out.startswith("Не получилось")


def test_media_status_report_is_not_a_claim():
    """«Коди не на паузе» is a status report, not a promise — the two-word
    guard keeps it out of the claim set (the plain `(?<!не )` cannot see two
    words back)."""
    out, replaced = vet_answer("Коди не на паузе.", _FAIL_MEDIA)
    assert replaced is False and out == "Коди не на паузе."


def test_invented_tool_name_speaks_the_real_blocker():
    """The invented name was the whole story of the lost turn; the spoken
    refusal must name the real blocker AND the tool that does exist."""
    out, replaced = vet_answer("Поставила на паузу.", _FAIL_NO_TOOL)
    assert replaced is True
    assert "не умеет" in out and "media_control" in out


def test_unresolvable_player_and_ambiguity_have_their_own_sentences():
    miss, replaced = vet_answer("Готово, поставила!", _FAIL_NO_PLAYER)
    assert replaced is True and "не нашла медиаплеер" in miss.lower()
    many, replaced = vet_answer("Готово, поставила!", _FAIL_SOME_PLAYERS)
    assert replaced is True and "несколько" in many


def test_confirmed_media_action_arms_the_claim():
    ok = [{"tool": "media_control", "ok": True,
           "detail": "media_player.le_vlada: media_pause выполнено, "
                     "состояние playing -> paused"}]
    out, replaced = vet_answer("Поставила на паузу.", ok)
    assert replaced is False and out == "Поставила на паузу."
    # …and a confirmed action also backs the state claim it produced.
    out, replaced = vet_answer("Коди на паузе.", ok)
    assert replaced is False


def test_failure_note_carries_the_new_blockers():
    note = failure_note(_FAIL_NO_TOOL)
    assert "media_control" in note
    assert note == failure_note(_FAIL_NO_TOOL)


_FAIL_NOTHING_PLAYING = [{"tool": "media_control", "ok": False,
                          "detail": "Тул вернул ошибку: media_player.le_kitchen: "
                                    "media_pause — ничего не играет (состояние idle)."}]
_FAIL_ALREADY_PAUSED = [{"tool": "media_control", "ok": False,
                         "detail": "Тул вернул ошибку: media_player.le_vlada: "
                                   "media_pause — уже на паузе (состояние paused)."}]


def test_a_media_call_that_moved_nothing_is_not_a_confirmed_action():
    """Field check 03.10.2026: the no-change branch was recorded ok=True, which
    armed the veto and let the model say «Переключил трек на кухне» after the
    box had done nothing. No change == no side effect."""
    out, replaced = vet_answer("Переключил трек на кухне.", _FAIL_NOTHING_PLAYING)
    assert replaced is True
    assert out == "Ничего не играет."
    out, replaced = vet_answer("Поставила на паузу.", _FAIL_ALREADY_PAUSED)
    assert replaced is True and out == "Уже на паузе."


_FAIL_WRONG_TITLE = [{"tool": "media_play", "ok": False,
                     "detail": "Тул вернул ошибку: media_player.le_zal_2: Kodi "
                               "открыл «The Simpsons» вместо «Black Mirror» — то, "
                               "что просили, не запустилось."}]


def test_a_wrong_title_is_reported_with_both_names():
    """A `playing` state does not prove the RIGHT thing started: Kodi can open
    something else (plugin redirect, stale queue). Announcing «Black Mirror
    запущено» for The Simpsons is the same confident-wrong the veto exists to
    stop, and the refusal is useless unless it names both titles."""
    out, replaced = vet_answer("Включила Black Mirror в гостиной.",
                               _FAIL_WRONG_TITLE)
    assert replaced is True
    assert out == "Включилось «The Simpsons» вместо «Black Mirror»."
