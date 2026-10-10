"""jev-router unit tests: slot resolver + classifier regex fast-paths.

Runs offline on host python: only pure functions are exercised (no FastAPI,
no Ollama/Qdrant/HA network calls). services/jev-router is put on sys.path
because the host has no fastapi — app.py is never imported here.
"""

import contextlib
import os
import sys

import pytest

_ROUTER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "jev-router")
)
sys.path.insert(0, _ROUTER)

import classifier as clf  # noqa: E402
from resolver import (  # noqa: E402
    device_hint_from,
    area_of,
    resolve_action,
    resolve_query,
    unresolved_hint,
)
from ha_client import (  # noqa: E402
    MEDIA_PREFER,
    MEDIA_STATE_ANSWER,
    _area_phrase,
    confirm_delays,
    describe_entity,
    find_entity,
    find_media_targets,
    media_fingerprint,
    media_no_movement_answer,
    volume_unverifiable,
)


# --- find_entity: bilingual (RU stem -> latin entity registry) --------------

_STATES = [
    {"entity_id": "sensor.bedroom_thermometer_temperature", "state": "22.94",
     "attributes": {"friendly_name": "bedroom_thermometer Temperature",
                    "unit_of_measurement": "°C"}},
    {"entity_id": "number.bedroom_thermometer_temperature_calibration", "state": "0",
     "attributes": {"friendly_name": "bedroom_thermometer Temperature calibration"}},
    {"entity_id": "switch.kitchen_coffee_machine", "state": "on",
     "attributes": {"friendly_name": "kitchen coffee machine"}},
]


def test_find_entity_bilingual_hint_and_area():
    """RU 'температур'+'спальня' must find latin bedroom_thermometer sensor."""
    e = find_entity(_STATES, "температур", "спальня")
    assert e is not None
    assert e["entity_id"] == "sensor.bedroom_thermometer_temperature"


def test_find_entity_prefers_numeric_sensor_over_helpers():
    e = find_entity(_STATES, "температур", "спальня")
    assert e["entity_id"].startswith("sensor.")
    assert "calibration" not in e["entity_id"]


def test_find_entity_missing_device_is_none_not_area_fallback():
    """'чайник' doesn't exist: None (escalate) — never answer with the
    kitchen coffee machine just because the area matched."""
    assert find_entity(_STATES, "чайник", "кухня") is None


def test_describe_entity_ru():
    s = describe_entity(_STATES[0], "спальня")
    # latin friendly_name + known area -> «В спальне: 22,9 градусов»
    assert "В спальне" in s and "22,9" in s and "градусов" in s


def test_area_phrase_prepositions():
    from ha_client import _area_phrase

    assert _area_phrase("кухня") == "На кухне"
    assert _area_phrase("спальня") == "В спальне"
    assert _area_phrase("гостиная") == "В гостиной"
    assert _area_phrase("коридор") == "На коридоре"
    assert _area_phrase("ванная") == "В ванной"
    assert _area_phrase("прихожая") == "В прихожей"  # ж+я -> ей, не «ой»


def test_area_phrase_en_display_names():
    """The resolver now sends registry display names («Living Room»)."""
    from ha_client import _area_phrase

    assert _area_phrase("Living Room") == "В гостиной"
    assert _area_phrase("Kitchen") == "На кухне"
    assert _area_phrase("Corridor") == "На коридоре"


# --- find_action_targets: on/off entity resolution (field E2E: living room) --

_ACTION_STATES = [
    {"entity_id": "switch.living_room_light_swith_relay", "state": "on",
     "attributes": {"friendly_name": "living_room_light_swith Relay"}},
    {"entity_id": "switch.living_room_light_swith_network_led_switch",
     "state": "off",
     "attributes": {"friendly_name": "living_room_light_swith Network led switch"}},
    {"entity_id": "light.wled_living_room", "state": "unavailable",
     "attributes": {"friendly_name": "WLED_living_room"}},
    {"entity_id": "switch.livingroom_camera_recordings", "state": "on",
     "attributes": {"friendly_name": "Livingroom camera Recordings"}},
]
_ACTION_AREA_MAP = {e["entity_id"]: "Living Room" for e in _ACTION_STATES}


def test_action_targets_ru_hint_latin_registry():
    """«свет» must reach the switch relay, skip unavailable lights and the
    camera switches (they live in the same area)."""
    from ha_client import find_action_targets

    got = find_action_targets(_ACTION_STATES, _ACTION_AREA_MAP, "свет", "Living Room")
    ids = {e["entity_id"] for e in got}
    assert "switch.living_room_light_swith_relay" in ids
    assert "switch.living_room_light_swith_network_led_switch" in ids
    assert "light.wled_living_room" not in ids          # unavailable
    assert "switch.livingroom_camera_recordings" not in ids  # not a light


def test_action_targets_area_binding_and_fallback():
    from ha_client import find_action_targets

    # Area is binding: same hint, other room -> nothing.
    assert find_action_targets(
        _ACTION_STATES, _ACTION_AREA_MAP, "свет", "Kitchen"
    ) == []
    # Empty registry map -> latin name-substring fallback still works.
    got = find_action_targets(_ACTION_STATES, {}, "свет", "living room")
    assert any(e["entity_id"].endswith("_relay") for e in got)


# --- resolve_action: happy paths -------------------------------------------


def test_turn_on_light_with_explicit_area():
    r = resolve_action("включи свет на кухне", "device")
    assert r is not None
    assert r.tool == "intent__HassTurnOn"
    assert r.args == {"domain": ["light"], "area": "Kitchen"}


def test_turn_off_uses_stream_default_area():
    r = resolve_action("выключи свет", "corridor")
    assert r is not None
    assert r.tool == "intent__HassTurnOff"
    assert r.args["area"] == "Corridor"


def test_turn_off_carries_spoken_hint_and_registry_area():
    """E2E regression: «гостиная» -> INVALID_AREA. The resolver must send the
    registry display name and keep the spoken word for entity lookup."""
    r = resolve_action("выключи свет в гостиной", "device")
    assert r is not None
    assert r.args["area"] == "Living Room"
    assert r.hint == "свет"


def test_stt_typo_coffemaker_resolves():
    """E2E regression: STT heard «кафеварку» (dropped «о») and the command
    fell through to L2, where the model invented a success."""
    r = resolve_action("Включи кафеварку.", "device")
    assert r is not None
    assert r.tool == "intent__HassTurnOn"
    assert r.args == {"domain": ["switch"], "name": "кофеварка"}
    assert r.hint == "кофеварка"


def test_dedupe_device_facets():
    """switch.coffemaker_child_lock must not ride along with the device root."""
    from ha_client import dedupe_device_facets

    ts = [
        {"entity_id": "switch.coffemaker_child_lock"},
        {"entity_id": "switch.coffemaker"},
        {"entity_id": "switch.living_room_light_swith_relay"},
        {"entity_id": "switch.living_room_light_swith_network_led_switch"},
    ]
    ids = {t["entity_id"] for t in dedupe_device_facets(ts)}
    assert ids == {
        "switch.coffemaker",
        "switch.living_room_light_swith_relay",
        "switch.living_room_light_swith_network_led_switch",
    }


def test_brightness_with_percent():
    r = resolve_action("сделай свет ярче на 50%", "device")
    assert r is not None
    assert r.tool == "light__HassLightSet"
    assert r.args["brightness"] == 50


def test_color_set():
    r = resolve_action("покрась свет в красный", "device")
    assert r is not None
    assert r.args["color"] == "red"


def test_vacuum_start_and_dock():
    r1 = resolve_action("запусти пылесос", "device")
    r2 = resolve_action("верни пылесос на зарядку", "device")
    assert r1 is not None and r1.tool == "vacuum__HassVacuumStart"
    assert r2 is not None and r2.tool == "vacuum__HassVacuumReturnToBase"


def test_vacuum_clean_with_named_room_uses_clean_area():
    """Field regression 25.09.2026: «уберется на кухне» went to HassVacuumStart
    whose area slot filters by the vacuum's LOCATION (robot unassigned in HA ->
    MatchFailedReason.AREA, robot never started). A room named in the utterance
    is the cleaning target -> HassVacuumCleanArea."""
    r = resolve_action("запусти пылесос на кухне", "device")
    assert r is not None
    assert r.tool == "vacuum__HassVacuumCleanArea"
    assert r.args == {"area": "Kitchen"}


def test_vacuum_clean_stream_default_area_keeps_start():
    """The stream's default area is a speaker-location hint, not a cleaning
    target — it must not silently switch a plain Start into CleanArea."""
    r = resolve_action("запусти пылесос", "kitchen")
    assert r is not None
    assert r.tool == "vacuum__HassVacuumStart"
    assert r.args == {"domain": ["vacuum"], "area": "Kitchen"}


def test_cancel_timers():
    r = resolve_action("выключи таймеры", "device")
    assert r is not None
    assert r.tool == "intent__HassCancelAllTimers"


def test_broadcast_extracts_message():
    r = resolve_action("скажи всем обед готов", "device")
    assert r is not None
    assert r.tool == "assist_satellite__HassBroadcast"
    assert r.args["message"] == "обед готов"


# --- resolve_action: escalation cases (None = escalate to L2) ---------------


def test_negated_command_escalates():
    """'не включи свет' must never map to HassTurnOn (wrong side-effect)."""
    assert resolve_action("не включи свет", "device") is None
    assert resolve_action("только не выключи таймеры", "device") is None


def test_ambiguous_bare_verb_escalates():
    assert resolve_action("включи", "device") is None


def test_direction_without_value_escalates():
    # 'сделай ярче' has no % — current brightness unknown offline.
    assert resolve_action("сделай свет ярче", "device") is None


def test_query_text_is_not_an_action():
    assert resolve_action("который час", "device") is None


# --- resolve_query ----------------------------------------------------------


def test_time_query():
    q = resolve_query("который час", "device")
    assert q is not None and q.kind == "datetime"


def test_temperature_query_carries_area_and_hint():
    q = resolve_query("сколько градусов в спальне", "device")
    assert q is not None and q.kind == "state"
    assert q.args["area"] == "Bedroom"
    assert "температур" in q.entity_hint


def test_state_query_onoff_with_stream_default():
    q = resolve_query("включён ли чайник", "kitchen")
    assert q is not None and q.kind == "state"
    assert q.args["name"] == "чайник"
    assert q.args["area"] == "Kitchen"


def test_media_state_query_defaults_to_domain():
    q = resolve_query("что сейчас играет", "device")
    assert q is not None and q.kind == "state"
    assert q.args == {"domain": ["media_player"]}


def test_unrecognized_query_escalates():
    assert resolve_query("привет", "device") is None


# --- classifier regex fast-paths (offline, no embeddings) -------------------


def test_action_fast_path_regexes():
    assert clf.RE_ACTION.match("включи свет на кухне")
    assert clf.RE_ACTION.match("пожалуйста, выключи свет")
    assert clf.RE_ACTION_MID.search("а ну выключи свет")
    # Negation is not an action start — resolver guard is the second line:
    assert not clf.RE_ACTION.match("не включи свет")


def test_query_fast_path_regexes():
    assert clf.RE_QUERY.match("сколько градусов в спальне")
    assert clf.RE_QUERY.match("какая температура на кухне")
    assert clf.RE_QUERY.match("горит ли свет на кухне")
    assert not clf.RE_QUERY.match("расскажи анекдот")


def test_no_tool_regex():
    assert clf.RE_NO_TOOL.search("убавь громкость")
    assert clf.RE_NO_TOOL.search("напомни через 5 минут")
    assert not clf.RE_NO_TOOL.search("выключи свет")


def test_confidence_calibration():
    import pytest

    assert clf.confidence_from_score(0.50) == 0.0
    assert clf.confidence_from_score(0.84) == pytest.approx(0.85)
    assert clf.confidence_from_score(0.90) >= 1.0
    assert clf.confidence_from_score(0.30) == 0.0  # clipped


def test_routes_cover_manifest_five():
    assert set(clf.ROUTES) == {
        "easy_action", "easy_query", "complex_logic", "expert", "general_qa"
    }


# --- warm-up: the 22 s charged to the first command (05.10.2026) --------------
#
# Ollama and Qdrant are containers on the same host, started in PARALLEL with
# this router, so at boot they are normally not listening yet. Both restarts on
# 05.10.2026 logged `Cannot connect to host 192.168.22.102:11434`, the warm-up
# gave up, and the first voice command then took 22.35 s — from 14:52:57.7 to
# 14:53:20.8, with `classifier warmed` logged from inside that request. Two
# separate faults, both pinned here: give up too early, and let a user request
# pay for the retry.


class _FlakyEmbedder:
    """Fails the first `fail_times` calls, then embeds trivially."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("Cannot connect to host 192.168.22.102:11434")
        import numpy as np

        return np.ones((len(texts), 4), dtype=np.float32) / 2.0

    async def close(self):
        return None


@contextlib.contextmanager
def _fast_schedule():
    """Shrink the retry schedule so the tests do not sleep for minutes.

    The real schedule is asserted separately — this only makes the failure paths
    runnable.
    """
    old = (clf._WARMUP_ATTEMPTS, clf._WARMUP_BACKOFF_S, clf._WARMUP_MAX_S)
    clf._WARMUP_ATTEMPTS, clf._WARMUP_BACKOFF_S, clf._WARMUP_MAX_S = 6, 0.0, 0.0
    try:
        yield
    finally:
        (clf._WARMUP_ATTEMPTS,
         clf._WARMUP_BACKOFF_S,
         clf._WARMUP_MAX_S) = old


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_the_warmup_schedule_outlasts_a_container_boot():
    """The whole point of the retry schedule, asserted without running it.

    Before the fix there were two attempts with no delay between them, so both
    hit a socket that had not opened yet. Anything that cannot wait out a
    parallel container start will bring the 22 s first-command cost back.
    """
    total = 0.0
    for attempt in range(1, clf._WARMUP_ATTEMPTS):
        total += min(clf._WARMUP_BACKOFF_S * attempt, clf._WARMUP_MAX_S)
    assert clf._WARMUP_ATTEMPTS >= 5, "too few attempts to survive a slow boot"
    assert total >= 60.0, (
        f"warm-up only waits {total:.0f}s in total — Ollama needs longer"
    )


def test_warmup_retries_a_dependency_that_is_not_up_yet():
    """Two microsecond-apart attempts both hit a socket that had not opened.
    The schedule must outlast a container boot, not just a blip."""
    with _fast_schedule():
        c = clf.Classifier(embedder=_FlakyEmbedder(fail_times=3))
        assert _run(c.warmup()) is True, "gave up while Ollama was still starting"
    assert c.warmed is True


def test_warmup_gives_up_eventually_and_stays_cold():
    with _fast_schedule():
        c = clf.Classifier(embedder=_FlakyEmbedder(fail_times=99))
        assert _run(c.warmup()) is False
    assert c.warmed is False


def test_a_cold_classify_answers_at_once_and_warms_in_the_background():
    """The regression that cost 22 s: `classify()` used to `await self.warmup()`.

    Nothing in the answer needs the matrix — `_regex_action` is the deterministic
    path — so classify must return an easy_action decision immediately and leave
    the warm-up running behind it.
    """
    import asyncio
    import time

    async def scenario():
        c = clf.Classifier(embedder=_FlakyEmbedder(fail_times=1))
        c._matrix = None  # force the cold branch
        started = time.monotonic()
        decision = await c.classify("включи свет")
        elapsed = time.monotonic() - started
        # The warm-up must be in flight, not awaited to completion.
        await asyncio.sleep(0)
        warming = c._warming or c._warming_task is not None
        c._warming_task.cancel()
        return decision, elapsed, warming

    with _fast_schedule():
        decision, elapsed, warming = _run(scenario())

    assert decision.route == "easy_action", (
        "a cold classifier must still answer a plain action"
    )
    assert decision.reason == "", "cold-but-served must not look like a failure"
    assert elapsed < 1.0, f"classify() waited {elapsed:.2f}s on the warm-up"
    assert warming, "warm-up was not left running in the background"


def test_only_one_warmup_runs_at_a_time():
    """Startup and a cold first request must not embed the same 60 utterances
    concurrently against one Ollama."""

    import asyncio

    class _Slow:
        def __init__(self):
            self.max_concurrent = 0
            self.in_flight = 0

        async def embed(self, texts):
            self.in_flight += 1
            self.max_concurrent = max(self.max_concurrent, self.in_flight)
            await asyncio.sleep(0.05)
            self.in_flight -= 1
            raise RuntimeError("nope")

        async def close(self):
            return None

    async def scenario():
        emb = _Slow()
        c = clf.Classifier(embedder=emb)
        await asyncio.gather(c.warmup(), c.warmup(), c.warmup())
        return emb

    with _fast_schedule():
        emb = _run(scenario())
    assert emb.max_concurrent == 1, (
        f"{emb.max_concurrent} concurrent embed batches against one Ollama"
    )


# --- chat sanitizer: markdown must never reach TTS (E2E-verified leaks) -----


def test_speakable_strips_markdown_and_list_markers():
    import chat_proxy

    s = chat_proxy._speakable("### 1. * **Кабель провайдера:** Проверь плотность.")
    assert "#" not in s and "*" not in s
    assert s.startswith("1. Кабель") or s.startswith("Кабель")


def test_speakable_strips_latex_arrows():
    import chat_proxy

    s = chat_proxy._speakable("Красный $\\rightarrow$ нет сигнала от провайдера.")
    assert "$" not in s and "\\rightarrow" not in s
    assert "нет сигнала" in s


def test_fragment_without_letters_is_dropped():
    import chat_proxy

    assert not chat_proxy._has_letters("168.")
    assert not chat_proxy._has_letters("### 2.")
    assert chat_proxy._has_letters("192.168.0.1 или веб-интерфейс")


# --- escalation hint: a corrupted device word becomes a suggestion for L2 ---
# Field case 28.09.2026: «Выключи кашеварку» escalated with NO context, L2
# guessed a name, HA answered MatchFailedError and the turn was lost. The
# hint is a suggestion only — resolve_action must stay conservative.


def test_hint_names_the_stt_corrupted_device():
    hint = unresolved_hint("Выключи кашеварку.")
    assert "кашеварку" in hint and "кофеварка" in hint
    # Conditional wording: a wrong guess must cost one wasted L2 call, not a
    # wrong side effect.
    assert "Если это оно" in hint


def test_hint_silent_for_pronoun():
    """«её» has no device to guess — the dialogue history resolves it."""
    assert unresolved_hint("Выключи её.") == ""


def test_hint_silent_for_known_device():
    """The dictionary knows the word: the fast path failed elsewhere."""
    assert unresolved_hint("Выключи кофеварку.") == ""
    assert unresolved_hint("Выключи чайник.") == ""


def test_hint_silent_for_negation():
    """«не выключи …» must never be nudged toward the refused action."""
    assert unresolved_hint("Не выключи кашеварку.") == ""


def test_hint_silent_without_command_verb():
    """A state or question gets no push to act («кашеварка горит»)."""
    assert unresolved_hint("кашеварка горит") == ""


def test_hint_silent_below_threshold():
    """An unknown device with no near match -> no suggestion at all: a wrong
    name (посудомойку ~ подсветка = 0.44) is worse than none."""
    assert unresolved_hint("Выключи посудомойку.") == ""


def test_hint_never_makes_the_resolver_act():
    """The hint rides along with the escalation, it does not resolve."""
    assert resolve_action("Выключи кашеварку.", "kitchen") is None


# --- status question about a NAMED device (field case 29.09.2026) -----------
# «Что там с нашим пылесосом?» resolved to None, escalated to L2 and the free
# model dictated «сейчас работает, заряда 45 %» while HA reports `docked`.
# The THING dictionary already knows the noun — the answer is one state read.


def test_device_status_query_resolves_to_state():
    q = resolve_query("Что там с нашим пылесосом?", "device")
    assert q is not None and q.kind == "state"
    assert q.args["domain"] == ["vacuum"]
    # RU stem, not the domain: _HINT_LAT expands it to vacuum/roborock/robot
    # and describe_entity() speaks it instead of «Roborock Robot».
    assert q.entity_hint == "пылесос"
    assert q.args["label"] == "пылесос"


def test_device_status_query_short_forms():
    for text in ("что с чайником", "чего с роботом?", "а как пылесос?"):
        q = resolve_query(text, "device")
        assert q is not None and q.kind == "state", text


def test_device_status_query_tolerates_fillers():
    """«Что у нас с роботом?» (29.09.2026, 14:17) — the filler «у нас» was
    not part of the phrasing, resolve_query returned None and the turn went
    to L2, where the free model invented both the state («убирает в
    гостиной») and the battery («шестьдесят пять процентов»). The branch
    only opens on a THING match, so fillers cost nothing elsewhere."""
    for text in ("что у нас с роботом?", "что у нас с пылесосом?",
                 "что вообще с чайником?", "что у нас с роботом на кухне?"):
        q = resolve_query(text, "device")
        assert q is not None and q.kind == "state", text
    q = resolve_query("что у нас с роботом?", "device")
    assert q.args["domain"] == ["vacuum"]
    assert q.args["label"] == "робот"
    assert q.entity_hint == "робот"


def test_device_status_query_keeps_a_named_room():
    q = resolve_query("что там с пылесосом на кухне", "device")
    assert q is not None and q.args["area"] == "Kitchen"


def test_status_question_without_a_device_still_escalates():
    """No THING match -> nothing to read: «как дела?» must not become a
    state query, and the bare greeting stays an escalation."""
    assert resolve_query("привет", "device") is None
    assert resolve_query("как дела?", "device") is None
    assert resolve_query("что нового в мире?", "device") is None


def test_status_question_does_not_shadow_the_earlier_families():
    """Temperature/humidity/battery are decided BEFORE the status rung —
    «что там с температурой?» is a sensor read with a sensor hint."""
    q = resolve_query("что там с температурой в спальне", "device")
    assert q is not None and q.kind == "state"
    assert "температур" in q.entity_hint
    assert q.args.get("label") is None  # sensor stems are never spoken labels


def test_classifier_routes_the_field_phrase_as_easy_query():
    """L1 must even see the question: route=easy_query conf=0.90 was logged
    for exactly this text on 29.09.2026 — and for the 14:17 one too, where
    the resolver (not the classifier) was the part that gave up."""
    assert clf.RE_QUERY.match("что там с нашим пылесосом?")
    assert clf.RE_QUERY.match("что у нас с роботом?")


# --- find_entity domain preference / describe_entity vacuum wording ----------


_VAC_STATES = [
    {"entity_id": "update.vacuum_card_update", "state": "off",
     "attributes": {"friendly_name": "Vacuum Card Update"}},
    {"entity_id": "vacuum.valetudo_zealouseverlastinggaur", "state": "docked",
     "attributes": {"friendly_name": "Roborock Robot"}},
]


def test_find_entity_prefers_the_real_domain():
    """`update.vacuum_card_update` sorts before `vacuum.…` and matches the
    same hint — without a preference the model would read the card helper."""
    e = find_entity(_VAC_STATES, "пылесос", None, domain=["vacuum"])
    assert e is not None and e["entity_id"].startswith("vacuum.")


def test_domain_preference_is_a_preference_not_a_filter():
    """The household lamp is a `switch.*` while the resolver asks for
    domain ['light'] — a filter would answer "device not found"."""
    e = find_entity(_STATES, "кофемашин", None, domain=["light"])
    assert e is not None and e["entity_id"] == "switch.kitchen_coffee_machine"


def test_domain_beats_a_numeric_sensor_with_the_same_hint():
    """Live check 29.09.2026: the router answered «Пылесос: 8» —
    sensor.…_map_segments contains "valetudo" (hint match) AND is a numeric
    sensor, so it outranked the vacuum itself."""
    states = _VAC_STATES + [
        {"entity_id": "sensor.valetudo_zealouseverlastinggaur_map_segments",
         "state": "8",
         "attributes": {"friendly_name": "Roborock Map segments"}},
    ]
    e = find_entity(states, "пылесос", None, domain=["vacuum"])
    assert e is not None and e["entity_id"].startswith("vacuum.")
    assert describe_entity(e, None, label="пылесос") == "Пылесос: на базе"
    # Without a domain the numeric-sensor rule is unchanged — that is exactly
    # what the domain argument is for.
    assert find_entity(states, "пылесос", None)["entity_id"].startswith("sensor.")


def test_describe_vacuum_status_is_spoken_russian():
    vac = _VAC_STATES[1]
    # Without a label the latin friendly name is read out as-is...
    assert describe_entity(vac, None) == "Roborock Robot: на базе"
    # ...with the resolver's RU device word it says what the user asked about.
    assert describe_entity(vac, None, label="пылесос") == "Пылесос: на базе"


def test_describe_label_never_beats_a_russian_name():
    e = {"entity_id": "switch.kettle", "state": "on",
         "attributes": {"friendly_name": "Чайник кухня"}}
    assert describe_entity(e, None, label="чайник").startswith("Чайник кухня")


# --- The battery metric keeps the DEVICE it belongs to -----------------------
# Field case 29.09.2026 (two episodes): the RE_BATT branch dropped the device
# word entirely, so «Сколько заряда у робота?» searched the registry for
# «заряд» globally and answered with the FIRST battery in it — the phone
# («SM-A546E Battery level: 33 процентов») for a question about the robot.

_BAT_STATES = [
    {"entity_id": "sensor.sm_a546e_battery_level", "state": "33",
     "attributes": {"friendly_name": "SM-A546E Battery level",
                    "unit_of_measurement": "%"}},
    {"entity_id": "sensor.valetudo_x_battery_level", "state": "97",
     "attributes": {"friendly_name": "Roborock Battery level",
                    "unit_of_measurement": "%"}},
    {"entity_id": "vacuum.valetudo_x", "state": "docked",
     "attributes": {"friendly_name": "Roborock Robot"}},
]


def test_battery_query_carries_device_label_and_missing_sentence():
    q = resolve_query("Сколько заряда у робота?", "device")
    assert q is not None and q.kind == "state"
    assert q.entity_hint == "заряд"
    assert q.args["device"] == "робот"
    assert q.args["label"] == "робот"
    # Spoken when the named device has no such reading at all («заряд
    # чайника») — the generic «не нашла такого устройства» would deny a
    # device that sits right there in the registry.
    assert "заряде" in q.args["missing"]


def test_battery_query_finds_the_devices_own_sensor():
    q = resolve_query("Сколько заряда у робота?", "device")
    e = find_entity(_BAT_STATES, q.entity_hint, None,
                    domain=q.args.get("domain"), device=q.args.get("device"))
    assert e is not None and e["entity_id"] == "sensor.valetudo_x_battery_level"
    assert describe_entity(e, None, label=q.args["label"]) == "Робот: 97 процентов"
    # The metric arrives through the whole family of spellings the STT
    # produces — «пылесос» stems are the ones the user actually says.
    q2 = resolve_query("Заряд батареи пылесоса", "device")
    e2 = find_entity(_BAT_STATES, q2.entity_hint, None,
                     domain=q2.args.get("domain"), device=q2.args.get("device"))
    assert e2 is not None and e2["entity_id"] == "sensor.valetudo_x_battery_level"
    assert describe_entity(e2, None, label=q2.args["label"]) == "Пылесос: 97 процентов"


def test_battery_query_without_a_device_reads_the_metric_pool():
    """«Сколько заряда в доме?» names no device: unchanged global metric
    lookup (registry order), and the latin name gets a speakable label."""
    q = resolve_query("Сколько заряда в доме?", "device")
    assert q is not None and "device" not in q.args
    e = find_entity(_BAT_STATES, "заряд", None, domain=["sensor"])
    assert e is not None and e["entity_id"] == "sensor.sm_a546e_battery_level"
    assert describe_entity(e, None, label=q.args["label"]) == "Заряд: 33 процентов"


def test_battery_query_never_borrows_another_devices_reading():
    """«Заряд чайника»: the kettle has no battery sensor. Handing over the
    phone's number under the label «Чайник» would be exactly the confident
    fabrication this project keeps fixing — the lookup refuses instead."""
    q = resolve_query("Заряд чайника", "device")
    assert q is not None and q.args.get("device") == "чайник"
    assert find_entity(_BAT_STATES, q.entity_hint, None,
                       domain=q.args.get("domain"),
                       device=q.args.get("device")) is None


# --- Media transport (field case 03.10.2026, «поставь его на паузу») --------
# HA's MCP server exposes ten tools and none of them is a media intent
# (read from tools/list), so before this the turn cost a 7 s L2 round trip and
# still answered «не удалось»: the model had to invent intent__HassMediaPause.
# The services exist, so the fast path now emits a REST call instead.


def test_media_pause_resolves_with_room_and_device_hint():
    call = resolve_action("пауза коди во владиной комнате", "")
    assert call is not None
    assert call.tool == "media__media_pause"
    # The room registry name, NOT the RU word: the REST service filters on
    # entity_id, but the sentence must still be speakable.
    assert call.args["area"] == "Bedroom Vlada"
    assert call.hint == "коди"
    assert call.speak_ok == "Поставила на паузу"


def test_media_pronoun_resolves_without_a_device_word():
    """«Поставь ЕГО на паузу» names its target by a pronoun; the registry
    (which box is playing/paused) decides, so the fast path must NOT bail out
    the way it does for an unknown device word."""
    call = resolve_action("поставь его на паузу", "")
    assert call is not None and call.tool == "media__media_pause"
    assert call.hint == ""
    assert call.args.get("service_data") == {}


def test_media_default_area_of_a_satellite_is_dropped():
    """«Сделай громче» said in the kitchen satellite: the stream's default area
    is a speaker LOCATION, not a claim about which player the user means —
    forwarding it would silence the kitchen box on every «громче»."""
    call = resolve_action("сделай громче", "kitchen")
    assert call is not None and call.tool == "media__volume_up"
    assert "area" not in call.args and call.area_source == "none"
    # …but an explicitly named room is kept.
    named = resolve_action("сделай тише в спальне", "kitchen")
    assert named is not None and named.args["area"] == "Bedroom"


def test_media_volume_set_needs_a_number_it_understands():
    """«Громкость 40» carries no unit at all (the light branch's RE_PCT only
    knows «%»), and this model spells «40 процентов» — both must reach
    volume_set instead of escalating."""
    a = resolve_action("поставь громкость 40", "")
    assert a is not None and a.tool == "media__volume_set"
    assert a.args["service_data"] == {"volume_level": 0.4}
    assert a.speak_ok == "Громкость 40 процентов"
    b = resolve_action("громкость 30 процентов в гостиной", "")
    assert b is not None and b.args["service_data"] == {"volume_level": 0.3}
    assert b.args["area"] == "Living Room"
    # No number at all: a level-less volume_set is a no-op, so escalate.
    assert resolve_action("поставь громкость", "") is None
    # …while «громче» is a step, not a level, and does not need one.
    assert resolve_action("сделай громче погромче", "").tool == "media__volume_up"


def test_media_mute_carries_the_flag_not_just_the_verb():
    """Both directions map to volume_mute; only the flag separates them, and
    «выключи звук» must never unmute the box."""
    off = resolve_action("выключи звук", "")
    assert off is not None and off.tool == "media__volume_mute"
    assert off.args["service_data"] == {"is_volume_muted": True}
    on = resolve_action("включи звук", "")
    assert on is not None and on.args["service_data"] == {"is_volume_muted": False}


def test_media_branch_does_not_steal_the_other_families():
    """The neighbours that MUST keep their proven path: the light dimmer
    («приглуши» means dim in this house), the on/off intents for light/switch
    and a device from another domain entirely («поставь чайник на паузу» is
    not a media command). «Включи телевизор» used to sit here too and moved
    out on 03.10.2026 — see test_media_on_off_is_a_transport_call_not_a_dead_intent."""
    assert resolve_action("приглуши свет", "") is None      # needs a level
    assert resolve_action("приглуши", "") is None          # no object
    light = resolve_action("включи свет", "")
    assert light is not None and light.tool == "intent__HassTurnOn"
    assert resolve_action("выключи свет в спальне", "").tool == "intent__HassTurnOff"
    assert resolve_action("включи чайник", "").tool == "intent__HassTurnOn"
    assert resolve_action("поставь чайник на паузу", "") is None


def test_media_verbs_share_the_classifier_fast_path():
    """«Громкость»/«звук» used to be RE_NO_TOOL («no MCP tool exists») and
    every such turn was escalated; the reminder words must still escalate."""
    assert clf.RE_MEDIA_ACTION.search("поставь его на паузу")
    assert clf.RE_MEDIA_ACTION.search("сделай громче")
    assert clf.RE_MEDIA_ACTION.search("убавь громкость")
    # A reminder is still not a device action, whatever else is in the phrase.
    assert clf.RE_NO_REMINDER.search("напомни отключить звук")
    assert clf.RE_NO_TOOL.search("напомни выключить таймеры на кухне")


_MEDIA_STATES = [
    {"entity_id": "media_player.le_vlada", "state": "paused",
     "attributes": {"friendly_name": "LE-vlada"}},
    {"entity_id": "media_player.le_zal_2", "state": "idle",
     "attributes": {"friendly_name": "LE-zal"}},
    {"entity_id": "media_player.le_spalnya", "state": "idle",
     "attributes": {"friendly_name": "LE-spalnya"}},
    {"entity_id": "media_player.le_kitchen", "state": "idle",
     "attributes": {"friendly_name": "LE-Kitchen"}},
    {"entity_id": "media_player.x96q_pro1_157_dlna", "state": "unavailable",
     "attributes": {"friendly_name": "X96Q X96Q_PRO1-157[DLNA]"}},
    {"entity_id": "media_player.x96q_pro1_157_airplay", "state": "off",
     "attributes": {"friendly_name": "X96Q_PRO1-157[AirPlay]"}},
]
_MEDIA_AREAS = {
    "media_player.le_vlada": "Bedroom Vlada",
    "media_player.le_zal_2": "Living Room",
    "media_player.le_spalnya": "Bedroom",
    "media_player.le_kitchen": "Kitchen",
    "media_player.x96q_pro1_157_dlna": "Living Room",
    "media_player.x96q_pro1_157_airplay": "Living Room",
}


def _media_call(text, stream=""):
    call = resolve_action(text, stream)
    if call is None:
        return None
    return find_media_targets(_MEDIA_STATES, _MEDIA_AREAS, call.hint,
                              call.args.get("area"),
                              MEDIA_PREFER.get(call.tool.split("__", 1)[1],
                                              ("playing", "buffering")))


def test_find_media_targets_answers_the_field_case_on_the_live_registry():
    """The exact turn from the field log: «пауза коди во владиной комнате»
    must reach media_player.le_vlada and nothing else."""
    got = _media_call("пауза коди во владиной комнате")
    assert [e["entity_id"] for e in got] == ["media_player.le_vlada"]


def test_find_media_targets_pronoun_picks_the_one_in_the_requested_state():
    """No device word, no room: the state the request is ABOUT decides. le_vlada
    is the only box that is paused, so «поставь его на паузу» is unambiguous —
    even though nothing is playing."""
    got = _media_call("поставь его на паузу")
    assert [e["entity_id"] for e in got] == ["media_player.le_vlada"]


def test_find_media_targets_intersects_the_device_word_with_the_room():
    """«коди» expands to «le», which every box carries: on its own the name
    would answer for le_vlada in «коди в гостиной». Name AND room are both
    constraints, and an exact room beats a room merely CONTAINED in another."""
    got = _media_call("поставь коди на паузу в гостиной")
    assert [e["entity_id"] for e in got] == ["media_player.le_zal_2"]
    assert [e["entity_id"] for e in
            _media_call("сделай тише в спальне")] == ["media_player.le_spalnya"]


def test_find_media_targets_refuses_instead_of_guessing():
    """Honest refusals: a room that holds no player, and several players with
    nothing to tell them apart."""
    # The corridor holds no player — answering for a box in another room is
    # the wrong-device side effect this project keeps refusing to make.
    assert _media_call("пауза телевизора в коридоре") == []
    # Four idle boxes, no room and no device word: nothing to narrow by.
    assert find_media_targets(_MEDIA_STATES, _MEDIA_AREAS, "", None) == []


def test_find_media_targets_ignores_dead_and_protocol_entities():
    """`unavailable` (the DLNA endpoint) is dropped, and the AirPlay receiver
    is not counted as a second box in the living room — that pair made
    «громкость в гостиной» ambiguous."""
    got = _media_call("громкость 30 процентов в гостиной")
    assert [e["entity_id"] for e in got] == ["media_player.le_zal_2"]
    dead = [e for e in _MEDIA_STATES if e["state"] == "unavailable"]
    assert find_media_targets(dead, _MEDIA_AREAS, "коди", None) == []


def test_media_idle_answers_are_spoken_not_escalated():
    """All four Kodi boxes sit in `idle` most of the day, so «пауза коди на
    кухне» is a very common request whose honest answer is «ничего не
    играет». Escalating it cost 3-6 s of L2 before the same words came back —
    the state table answers it in _execute_media instead."""
    assert MEDIA_STATE_ANSWER["media_pause"]["idle"] == "ничего не играет"
    assert MEDIA_STATE_ANSWER["media_pause"]["off"] == "ничего не играет"
    assert MEDIA_STATE_ANSWER["media_pause"]["paused"] == "Уже на паузе."
    assert MEDIA_STATE_ANSWER["media_play"]["playing"] == "Уже играет."
    # A track switch in a stopped box is about nothing, not a failed command.
    for svc in ("media_next_track", "media_previous_track"):
        assert MEDIA_STATE_ANSWER[svc]["idle"] == "ничего не играет"
    # Volume has no such row on purpose: a muted idle box is still a box whose
    # loudness the user is asking about.
    assert "idle" not in MEDIA_STATE_ANSWER.get("volume_up", {})


def test_area_phrase_speaks_the_kodis_own_room():
    """«В bedroom vlada» at the user is the generic latin fallback; the area
    registry name has a phrase of its own."""
    assert _area_phrase("Bedroom Vlada") == "Во владиной комнате"
    assert _area_phrase("Kitchen") == "На кухне"


def test_media_fingerprint_sees_an_attribute_only_change():
    """HA's service reply said `changed: []` for a volume_up that really did
    turn the box up (0.7 -> 0.8), so the fast path verifies by reading the
    player back. A fingerprint of `state` alone would call that a failure."""
    before = {"entity_id": "media_player.a", "state": "idle",
              "attributes": {"volume_level": 0.7, "is_volume_muted": False}}
    after = {"entity_id": "media_player.a", "state": "idle",
             "attributes": {"volume_level": 0.8, "is_volume_muted": False}}
    ticking = {"entity_id": "media_player.a", "state": "idle",
               "attributes": {"volume_level": 0.7, "is_volume_muted": False,
                              "media_position": 412}}
    assert media_fingerprint(before) != media_fingerprint(after)
    assert media_fingerprint(before) == media_fingerprint(ticking)


def test_media_on_off_is_a_transport_call_not_a_dead_intent():
    """Field case 03.10.2026 18:21: «Ну так найди её и включи» ended in
    intent__HassTurnOn answering MatchFailedReason.INVALID_AREA for the RU room
    and then MatchFailedReason.ASSISTANT for `name='LE-zal'` — HA's MCP server
    has no media intent and the Kodis are not exposed to Assist, so the on/off
    intent can NEVER reach them. Transport services do not care about Assist
    exposure, so «включи коди» is media_play now."""
    on = resolve_action("включи коди в гостиной", "")
    assert on is not None and on.tool == "media__media_play"
    assert on.args["area"] == "Living Room"
    off = resolve_action("выключи коди", "")
    assert off is not None and off.tool == "media__media_stop"
    for text in ("включи телевизор", "включи музыку", "включи колонку на кухне"):
        c = resolve_action(text, "")
        assert c is not None and c.tool == "media__media_play", text
    # …and the OTHER families keep their proven intents.
    for text, tool in (("включи свет в гостиной", "intent__HassTurnOn"),
                       ("выключи свет", "intent__HassTurnOff"),
                       ("включи чайник", "intent__HassTurnOn"),
                       ("включи пылесос", "vacuum__HassVacuumStart")):
        c = resolve_action(text, "")
        assert c is not None and c.tool == tool, text


def test_a_named_episode_is_a_library_request_not_a_track_skip():
    """«Включай следующую серию "Темного зеркала" в гостиной» is a LIBRARY
    item, not «the next track»: routed to media_next_track it switched a track
    on whatever box answered (field check 03.10.2026 19:32 — it reached the
    bedroom). It must escalate to L2, which owns media_search + media_play and
    the room from the escalation context. «Следующий ТРЕК» stays a transport
    command."""
    assert resolve_action("включай следующую серию темного зеркала в гостиной",
                          "") is None
    assert resolve_action("следующая серия чёрного зеркала", "") is None
    track = resolve_action("следующий трек в гостиной", "")
    assert track is not None and track.tool == "media__media_next_track"
    assert track.args["area"] == "Living Room"
    # A pronoun with no device word stays an escalation: L2 owns the history.
    assert resolve_action("ну так найди её и включи", "") is None


def test_area_of_gives_l2_the_canonical_room_name():
    """«гостиная» is not an HA area. Every escalation now carries the display
    name, because an intent fed the RU word answers INVALID_AREA."""
    assert area_of("включи коди в гостиной", "") == ("Living Room", "explicit")
    assert area_of("включи коди во владиной комнате", "") == (
        "Bedroom Vlada", "explicit")
    assert area_of("включи коди", "") == (None, "none")
    # A satellite's default room is NOT «explicit» — only a spoken one is.
    assert area_of("включи коди", "kitchen") == ("Kitchen", "default")


def test_the_regex_action_path_is_the_one_definition():
    """The cosine path and the cold-embedder fallbacks share one test, so a
    cold Ollama can never disagree with a warm one about what an action is."""
    c = clf.Classifier.__new__(clf.Classifier)  # no embedder needed
    for text in ("включи свет", "выключи свет в спальне", "поставь на паузу",
                 "сделай громче", "включай следующую серию в гостиной"):
        assert c._regex_action(text) is True, text
    for text in ("напомни выключить таймеры на кухне", "какая сейчас погода",
                 "привет как дела", "расскажи анекдот"):
        assert c._regex_action(text) is False, text


def test_cold_classifier_still_routes_a_command_it_can_resolve():
    """One cold Ollama used to escalate EVERY voice command (18:38 warmup
    timeout). The media/on-off fast path is pure regex, so it must survive."""
    import asyncio

    class DeadEmbedder:
        async def embed(self, texts):
            raise RuntimeError("ollama down")

    c = clf.Classifier(embedder=DeadEmbedder())
    d = asyncio.run(c.classify("поставь коди на паузу в гостиной"))
    assert d.route == "easy_action" and d.reason == ""
    q = asyncio.run(c.classify("какая сейчас погода"))
    assert q.route == "complex_logic" and q.reason in (
        "classifier_unavailable", "embed_failed")


def test_the_embed_batch_is_chunked_to_fit_the_timeout():
    """Measured 03.10.2026 on this host: 1 utterance ~1.7 s, the whole
    60-utterance warm-up ~102 s against a 30 s client timeout — so the warm-up
    failed on EVERY start and the classifier ran cold. A chunk must stay well
    inside the timeout, and the rows must come back in order."""
    parts = clf.chunked(list(range(60)))
    assert [len(p) for p in parts] == [8] * 7 + [4]
    assert [x for p in parts for x in p] == list(range(60))  # order kept
    assert clf.chunked([]) == []
    # A chunk must not be able to grow past the timeout budget: at the measured
    # per-utteration cost, 8 is ~14 s of a 30 s window.
    assert clf.EMBED_CHUNK * 1.7 < 30


def test_a_playback_question_with_a_room_keeps_the_media_domain():
    """«Что сейчас играет в гостиной» used to carry the area ALONE: the media
    default sat behind `if not args`, so the lookup had no domain, no entity id
    carries «Living Room», and the answer was «Не нашла такого устройства»
    about a TV that was playing (field check 03.10.2026)."""
    q = resolve_query("что сейчас играет в гостиной", "")
    assert q is not None and q.kind == "state"
    assert q.args["domain"] == ["media_player"]
    assert q.args["area"] == "Living Room"
    assert q.entity_hint == "media_player"
    # No room -> the global media pool, unchanged.
    g = resolve_query("что сейчас играет", "")
    assert g is not None and g.args["domain"] == ["media_player"]
    assert "area" not in g.args
    # Another device keeps its own domain.
    v = resolve_query("что там с пылесосом", "")
    assert v is not None and v.args["domain"] == ["vacuum"]


# --- a media call HA accepted that moved nothing -----------------------------
# Field check 04.10.2026 21:08: «сделай громче в спальне» on the idle bedroom
# box. volume_up is no transport service, so the «nothing was playing» branch
# skipped it, the fingerprint had not moved yet inside the 2.1 s window, and the
# call escalated — 29.7 s of LLM plus an httpx retry, >60 s for the turn, for a
# command HA had already accepted. The box went 0.80 -> 0.85 a moment later.


_IDLE_BOX = {"entity_id": "media_player.le_spalnya", "state": "idle",
             "attributes": {"volume_level": 0.85, "friendly_name": "LE-spalnya"}}
_PLAYING_BOX = {"entity_id": "media_player.le_spalnya", "state": "playing",
                 "attributes": {"volume_level": 0.85, "media_title": "Black Mirror"}}


def test_a_volume_call_on_an_idle_box_is_answered_not_escalated():
    """Nothing moved, but the box had nothing to play — that is the answer, and
    it costs 0.2 s instead of an L2 round trip."""
    assert media_no_movement_answer("volume_up", _IDLE_BOX, "спальне") == \
        "В спальне ничего не играет."
    assert media_no_movement_answer("volume_set", _IDLE_BOX, "") == \
        "Ничего не играет."
    assert media_no_movement_answer("media_pause", _IDLE_BOX, "кухне") == \
        "На кухне ничего не играет."


def test_a_playing_box_that_stayed_playing_still_escalates():
    """The rule is about a box that was ALREADY idle. A pause that left a
    playing box playing is a real failure and belongs to L2."""
    assert media_no_movement_answer("media_pause", _PLAYING_BOX, "спальне") is None
    assert media_no_movement_answer("volume_up", _PLAYING_BOX, "спальне") is None


def test_a_service_without_a_play_answer_still_escalates():
    """Only the services that can be «about nothing» get the sentence; anything
    else keeps its old behaviour."""
    for service in ("turn_on", "switch_off", "ha_action", "brightness_set"):
        assert media_no_movement_answer(service, _IDLE_BOX, "спальне") is None


def test_a_room_word_is_declined_however_it_arrives():
    """The resolver hands over the canonical HA name («Bedroom»), so «В
    спальнее» never reached the user — but `_area_phrase` also takes a Russian
    display name or whatever the caller carries, and its generic rule turned
    «спальне» into «В спальнее» in silence."""
    assert _area_phrase("спальня") == "В спальне" == _area_phrase("спальне")
    assert _area_phrase("гостиная") == "В гостиной" == _area_phrase("гостиной")
    assert _area_phrase("прихожая") == "В прихожей" == _area_phrase("прихожей")
    assert _area_phrase("кухня") == "На кухне" == _area_phrase("кухне")
    assert _area_phrase("коридор") == "На коридоре" == _area_phrase("коридоре")
    assert _area_phrase("улица") == "На улице" == _area_phrase("улице")
    assert _area_phrase("детская") == "В детской" == _area_phrase("детской")
    assert _area_phrase("Bedroom Vlada") == "Во владиной комнате"
    assert _area_phrase("") == ""


def test_a_box_without_a_volume_level_is_not_polled_at_all():
    """No level to move means no poll can ever confirm anything — waiting would
    only spend the whole ~6.8 s window to reach the same answer."""
    assert volume_unverifiable({"attributes": {"volume_level": None}})
    assert volume_unverifiable({})
    assert not volume_unverifiable({"attributes": {"volume_level": 0.0}})
    # ...while a box that DOES report a level gets the longer volume window,
    # because that is where the change lands late.
    assert sum(confirm_delays("volume_up")) > sum(confirm_delays("media_pause"))
    assert sum(confirm_delays("volume_set")) == sum(confirm_delays("volume_mute"))


# --- honesty: `ok` is not "the device moved" ---------------------------
# Field case 04.10.2026, living room: «выключи свет» -> the gateway said
# «Выключила» and the lamp stayed on. Three separate holes let that through,
# all reachable from one ordinary utterance, and NONE of them was covered —
# nothing in the suite imported this module.


def _router_app():
    """Load jev-router/app.py under a private name.

    It cannot be imported as `app` (that name belongs to the FastAPI app in
    test_main.py) and it was never imported at all before this, which is why a
    false «Выключила» shipped with a clean log.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "jev_router_app", os.path.join(_ROUTER, "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _DeadHA:
    """HA that answers, but has no state to reason about."""

    def __init__(self, states=None, result=None):
        self._states = states if states is not None else []
        self._result = result or {"ok": True, "result": {}}
        self.calls = []

    async def get_states(self, force=False):
        return self._states

    async def get_entity_areas(self):
        return {}

    async def call_tool(self, name, args, rpc_id=1):
        self.calls.append((name, args))
        return self._result


def _off_call():
    return resolve_action("выключи свет", "livingroom")


def test_unavailable_registry_refuses_instead_of_acting_blind():
    """An empty state list means "HA did not answer", not "no such device".

    `get_states` returns its previous snapshot on failure and that snapshot is
    `[]` on a cold cache. The old code fell straight through to the blind
    intent, which cannot match a `switch.*_relay` with `domain: ["light"]` and
    so reported success for the AREA alone.
    """
    import asyncio

    mod = _router_app()
    ha = _DeadHA()
    mod.ha = ha
    sentence, err = asyncio.run(mod._execute_action(_off_call()))
    assert sentence is None, "claiming success on an unreadable registry is a lie"
    assert err and "registry_unavailable" in str(err.get("error"))
    assert ha.calls == [], "HA must not be called at all in this state"


def test_blind_success_naming_only_an_area_is_refused():
    """The exact 04.10.2026 answer: success for the ROOM, nothing switched."""
    import asyncio

    mod = _router_app()
    # A NON-on/off intent, which is what the blind path is for now: on/off
    # refuses earlier (unreadable registry / empty target set).
    from resolver import ResolvedCall

    call = ResolvedCall(
        tool="intent__HassBroadcast",
        hint="свет",
        args={"domain": ["light"], "area": "Living Room"},
        speak_ok="Включила",
    )
    ha = _DeadHA(
        states=[{"entity_id": "switch.living_room_light_swith_relay", "state": "off"}],
        result={
            "ok": True,
            "result": {"speech": {}, "response_type": "action_done"},
            "claimed": [
                {"name": "Living Room", "type": "area", "id": "living_room"},
                {"name": "WLED_living_room", "type": "entity",
                 "id": "light.wled_living_room"},
            ],
        },
    )
    mod.ha = ha
    sentence, err = asyncio.run(mod._execute_action(call))
    assert sentence is None, "an area is not a device — this is the false «Выключила»"
    assert err and "unverified_side_effect" in str(err.get("error"))


def test_a_real_available_entity_is_a_side_effect():
    mod = _router_app()
    res = {"ok": True, "claimed": [{"name": "living_room_light_swith Relay",
                                    "type": "entity",
                                    "id": "switch.living_room_light_swith_relay"}]}
    states = [{"entity_id": "switch.living_room_light_swith_relay", "state": "off"}]
    assert mod._touched_a_usable_entity(res, states) is True


def test_naming_nothing_is_not_a_side_effect():
    mod = _router_app()
    live = [{"entity_id": "x", "state": "on"}]
    assert mod._touched_a_usable_entity({"ok": True, "claimed": []}, live) is False
    # HA's other payload shape carries a bare bool and names nothing
    assert mod._touched_a_usable_entity({"ok": True}, live) is False


def test_unavailable_entity_is_not_a_side_effect():
    mod = _router_app()
    res = {"ok": True, "claimed": [{"name": "WLED", "type": "entity",
                                    "id": "light.wled_living_room"}]}
    states = [{"entity_id": "light.wled_living_room", "state": "unavailable"}]
    assert mod._touched_a_usable_entity(res, states) is False


def test_claimed_is_parsed_out_of_the_intent_envelope():
    """`call_tool` must surface HA's own account of what it touched."""
    from ha_client import _claimed

    assert _claimed({"data": {"success": [{"id": "a", "type": "entity"}],
                             "failed": [{"id": "b", "type": "entity"}]}}) == {
        "claimed": [{"id": "a", "type": "entity"}],
        "claimed_failed": [{"id": "b", "type": "entity"}],
    }
    assert _claimed({"success": True}) == {"claimed": [], "claimed_failed": []}


def test_a_media_title_is_not_a_missing_device():
    """«включи следующую серию черного зеркала» is not a request for a device.

    Field case 04.10.2026 19:22: `RE_MEDIA_NOUN` matches «серию», but
    `resolve_action` only takes the media branch when the thing resolves to a
    `media_player` — which a title never does — so it fell through to
    `HassTurnOn` with hint «серию», the deterministic refusal fired, and the
    user was told «Не нашла такого устройства» about a request that L2's
    `media_search`/`media_play` exist to serve. Refusing also BLOCKS escalation,
    so the fix is to fail in a way that lets L2 answer.
    """
    import asyncio

    mod = _router_app()
    from resolver import resolve_action

    call = resolve_action("включи следующую серию черного зеркала", "livingroom")
    assert call.hint == "серию", "the noun is carried as a device hint"

    class HA:
        async def get_states(self, force=False):
            return [{"entity_id": "switch.living_room_light_swith_relay",
                     "state": "off", "attributes": {"friendly_name": "Relay"}}]

        async def call_tool(self, *a, **k):
            raise AssertionError("a media title must not reach the intent")

    mod.ha = HA()
    sentence, err = asyncio.run(
        mod._execute_action(call, "включи следующую серию черного зеркала")
    )
    assert sentence is None, "must not answer about devices"
    assert err and "media_content_word" in str(err.get("error")), (
        "the refusal has to name the real reason, or L2 is told nothing useful"
    )


def test_a_genuinely_absent_device_still_refuses_without_escalating():
    """The device case must keep its deterministic, spoken refusal."""
    import asyncio

    mod = _router_app()
    from resolver import resolve_action

    call = resolve_action("включи кафеварку", "livingroom")

    class HA:
        async def get_states(self, force=False):
            return [{"entity_id": "switch.living_room_light_swith_relay",
                     "state": "off", "attributes": {"friendly_name": "Relay"}}]

        async def call_tool(self, *a, **k):
            raise AssertionError("an absent device must not reach the intent")

    mod.ha = HA()
    sentence, err = asyncio.run(
        mod._execute_action(call, "включи кафеварку")
    )
    assert sentence == "Не нашла такого устройства. Может, уточните название?"
    assert err is None


# --- a follow-up carries the action but not the target (05.10.2026) ----------
#
# After «включи свет», «а теперь выключи» is a verb with nothing to apply it to.
# `resolve_action` returns None, the room escalates to L2, and the dialogue looks
# broken because the room does not remember what it was just talking about.


def test_a_followup_borrows_the_device_from_the_previous_turn():
    prev = "включи свет"
    follow = "а теперь выключи"
    hint = device_hint_from(prev)
    assert hint == "свет", hint

    assert resolve_action(follow, "livingroom") is None, (
        "the bare follow-up must NOT resolve on its own — that is the whole gap"
    )
    call = resolve_action(f"{follow} {hint}", "livingroom")
    assert call is not None, "the follow-up did not resolve with the device"
    assert call.tool == "intent__HassTurnOff", call.tool


def test_borrowing_the_device_never_borrows_the_verb():
    """Appending the previous text whole inverts the command.

    Measured: `resolve_action("а теперь выключи включи свет")` returns
    HassTurn**On**, because the resolver scans for any verb and finds «включи» in
    the borrowed words. So the helper must strip the verb, not just trim words.
    """
    assert resolve_action("а теперь выключи включи свет", "livingroom").tool == (
        "intent__HassTurnOn"
    ), "if this ever becomes TurnOff the borrowing needs re-checking, not copying"

    assert device_hint_from("включи свет") == "свет"
    assert device_hint_from("выключи свет в гостиной") == "свет гостиной"
    assert device_hint_from("ну выключи свет") == "свет"


def test_stacked_fillers_are_all_stripped():
    """«а теперь выключи» has two fillers and one pass stripped only «а», leaving
    «теперь» as the device."""
    assert device_hint_from("а теперь выключи") == ""
    assert device_hint_from("а теперь включи") == ""
    assert device_hint_from("включи") == "", "no device named, nothing to borrow"


def test_the_room_is_kept_when_it_was_named():
    """The referent is the device AND its room: «включи свет на кухне» then
    «а теперь выключи» must act on the kitchen, not the satellite's own room."""
    hint = device_hint_from("включи свет на кухне")
    assert "кухн" in hint, hint
    call = resolve_action(f"а теперь выключи {hint}", "kitchen")
    assert call is not None and call.args.get("area"), call


def test_history_exposes_the_users_words_not_the_rooms_reply():
    """The block is prose for the LLM and contains our own REPLY too; feeding
    that to a deterministic resolver would act on what the room said."""
    import history

    history.reset()
    history.push("s", "включи свет", "Включила")
    assert history.last_text("s") == "включи свет"
    assert "Включила" not in history.last_text("s")
    assert history.last_text("missing") == ""


def test_the_time_question_is_phrased_the_way_people_ask_it():
    """Measured 05.10.2026 22:14: «Сколько времени?» took 4.2 s through L2 while
    «который час» took 0.26 s. The second phrasing was in the list and the first,
    which is the more common one, was not."""
    from resolver import RE_TIME_Q

    for phrase in (
        "сколько времени",
        "Сколько сейчас времени",
        "времени сколько",
        "который час",
        "какое сейчас время",
        "какое сегодня число",
        "какая дата",
    ):
        assert RE_TIME_Q.search(phrase), f"{phrase!r} is not a time question"

    for phrase in ("какая температура в гостиной", "что сейчас играет", "кто это"):
        assert not RE_TIME_Q.search(phrase), f"{phrase!r} is not a time question"


# --- Whisper mangles the OFF verb and L2 reverses the action (06.10.2026) ------
#
# Measured end to end, `stream=livingroom`, against the live router:
#
#     'куча свет'             -> L2 conf 0.75 -> «Включила свет»
#     'куча свет в гостиной'  -> L2 conf 0.72 -> «Включила свет в гостиной»
#     'кучи свет'             -> L2 conf 0.74
#     'выкл юч свет'          -> L2 conf 0.86 resolver_ambiguous
#
# Asked to turn the light OFF, the room turned it ON. Whisper writes «выключи» as
# «куча», «ключи» or «выкл юч»; `RE_OFF` matches none of those, the deterministic
# path is skipped, confidence falls under 0.85 and an LLM is handed the garbled text
# with nothing said about the verb — so it picks a direction, and picks it backwards.

def test_the_measured_stt_spellings_of_the_off_verb_resolve():
    """Only the spellings that were actually observed go in. Nothing here is a
    general fuzzy matcher: an unmeasured guess about what Whisper 'might' produce
    is exactly how a wrong action reaches a lamp."""
    from resolver import resolve_action, normalize_stt_verbs

    assert normalize_stt_verbs("куча свет") == "выключи свет"
    assert normalize_stt_verbs("КУЧА свет в гостиной") == "выключи свет в гостиной"
    assert normalize_stt_verbs("кучи свет") == "выключи свет"
    assert normalize_stt_verbs("ключи свет") == "выключи свет"
    assert normalize_stt_verbs("выкл юч свет") == "выключи свет"

    for text in ("куча свет", "куча свет в гостиной", "кучи свет",
                 "ключи свет", "выкл юч свет"):
        call = resolve_action(text, "livingroom")
        assert call is not None, f"{text!r} still escalates"
        assert "TurnOff" in call.tool, f"{text!r} -> {call.tool}, expected TurnOff"
        assert call.args.get("area") == "Living Room", (
            f"{text!r} lost the room: {call.args}"
        )


def test_ordinary_words_that_merely_look_like_the_misheard_verb_are_left_alone():
    """«куча» and «ключи» are ordinary Russian words, which is why the mapping is
    gated on a device noun. Without the gate «куча чая» would switch a lamp off."""
    from resolver import normalize_stt_verbs, resolve_action

    for text in ("куча чая", "ключи от машины", "куча всего"):
        assert normalize_stt_verbs(text) == text, f"{text!r} was rewritten"
    assert resolve_action("куча чая", "livingroom") is None


def test_the_on_verb_is_left_exactly_as_it_is():
    """No ON entry exists because nothing has been observed mangling it — the log
    holds 'включи', 'Включи', 'включи свет'. Adding one would be inventing a rule
    from no measurement, and a wrong ON entry flips a real lamp."""
    from resolver import normalize_stt_verbs

    for text in ("включи свет", "Включи свет в гостиной", "включи свет"):
        assert normalize_stt_verbs(text) == text


def test_a_device_command_with_no_verb_is_flagged_for_the_escalation():
    """`resolve_action` returning None is not protection enough: today's case went
    out through the classifier's `low_confidence`, which never calls the resolver.
    The note has to be attachable for that path too."""
    from resolver import action_verb_missing

    # A MEASURED mangle returns False: normalisation already restored the verb, so
    # the turn does not escalate and there is nothing to warn about. The net is for
    # the mangles nobody has observed yet.
    assert action_verb_missing("куча свет") is False
    assert action_verb_missing("выключи свет") is False
    assert action_verb_missing("что с светом") is True, (
        "a device noun with no verb at all is exactly what must reach L2 as a question"
    )
    assert action_verb_missing("включи свет в гостиной") is False
    assert action_verb_missing("сколько сейчас времени") is False, (
        "no device noun, nothing to be uncertain about"
    )


def test_the_stt_repair_happens_before_the_classifier_decides():
    """Normalising inside the resolver was not enough, and the failure is
    instructive: «куча свет» scored 0.75 (below the 0.85 gate, straight to L2, the
    resolver never consulted) while «выкл юч свет» scored 0.86 and took the
    deterministic path. The same defect, opposite outcome, decided by how close to
    the threshold the embedding happened to land. So the repair has to sit where
    the transcript enters, ahead of the classifier."""
    import app as jev_router_app
    import inspect

    src = inspect.getsource(jev_router_app._handle)
    repair_at = src.index("normalize_stt_verbs(text)")
    classify_at = src.index("classifier.classify(text)")
    assert repair_at < classify_at, (
        "the transcript is classified before it is repaired, so a mangled verb is "
        "judged on its mangled spelling"
    )


def test_the_escalation_note_tells_l2_not_to_guess_the_direction():
    """The note is the whole fix for the unmeasured cases, so its wording is
    load-bearing: it has to forbid the action, not merely mention the ambiguity."""
    import app as jev_router_app
    import inspect

    src = inspect.getsource(jev_router_app._handle)
    assert "action_verb_missing" in src, "the guard is not wired into _handle"
    assert "НЕ угадывай направление" in src
    assert "спроси" in src
    # It must gate on route == complex_logic, or it would fire on resolved actions.
    assert 'route == "complex_logic"' in src


# --- «ok» is not a device that moved, on the FAST path either (06.10.2026) ----
#
# Measured 06.10.2026 18:21 UTC on the kitchen:
#
#     router route=easy_action conf=0.92 text='включи кофеварку'
#     router speaking: 'Включила'
#     switch.coffemaker  off  last_changed=2026-10-06T13:14:53Z   <- hours old
#
# `switch.coffemaker` was `off`, the resolver produced it as the single target,
# HA accepted the call, and the room spoke a success. The verification
# `_touched_a_usable_entity()` existed — and existed precisely because of the
# measured lie of 04.10.2026 — but it was only wired into the BLIND path. The
# per-entity on/off fast path returned `call.speak_ok` on `ok and not fatal`
# without ever re-reading a state.


def test_the_onoff_fast_path_verifies_before_claiming_success():
    """The hole was one missing call, not a missing idea: the same rule already
    guarded the other path."""
    import app as jev_router_app
    import inspect

    src = inspect.getsource(jev_router_app._execute_action)
    # The success return must be behind the verification, not adjacent to it.
    assert src.count("_confirm_on_off_moved") >= 1, (
        "the on/off fast path still returns speak_ok on HA's `ok` alone"
    )
    idx_call = src.index("_confirm_on_off_moved")
    idx_return = src.index("return call.speak_ok, None", idx_call)
    assert idx_return > idx_call, "speak_ok is returned before the verification"
    # And there must be an unverified branch, so a still-false claim escalates
    # with the real blocker instead of speaking it.
    assert "unverified_side_effect" in src


@pytest.mark.asyncio
async def test_a_device_that_did_not_move_is_reported_as_moved(monkeypatch):
    """The verification itself, against the measured failure: HA accepts, the
    relay stays exactly as it was."""
    import app as jev_router_app

    class _HA:
        def __init__(self, seq):
            self.seq = list(seq)
            self.calls = 0

        async def get_states(self, force=False):
            self.calls += 1
            return self.seq[min(self.calls - 1, len(self.seq) - 1)]

    still_off = [{"entity_id": "switch.coffemaker", "state": "off"}]
    monkeypatch.setattr(jev_router_app, "ha", _HA([still_off, still_off, still_off]))

    moved, unmoved = await jev_router_app._confirm_on_off_moved(
        ["switch.coffemaker"], {"switch.coffemaker": "off"}, want_on=True
    )
    assert moved == [], "a relay that stayed off was reported as moved"
    assert unmoved == ["switch.coffemaker"]


@pytest.mark.asyncio
async def test_a_relay_that_answers_on_the_second_read_counts(monkeypatch):
    """A relay is not instantaneous, which is the whole reason for the retry —
    and a single instant read-back would have produced a false refusal."""
    import app as jev_router_app

    class _HA:
        def __init__(self, seq):
            self.seq = list(seq)
            self.calls = 0

        async def get_states(self, force=False):
            self.calls += 1
            return self.seq[min(self.calls - 1, len(self.seq) - 1)]

    monkeypatch.setattr(
        jev_router_app, "ha",
        _HA([[{"entity_id": "switch.coffemaker", "state": "off"}],
             [{"entity_id": "switch.coffemaker", "state": "on"}]]),
    )

    moved, unmoved = await jev_router_app._confirm_on_off_moved(
        ["switch.coffemaker"], {"switch.coffemaker": "off"}, want_on=True
    )
    assert moved == ["switch.coffemaker"]
    assert unmoved == []


@pytest.mark.asyncio
async def test_a_device_that_went_the_OTHER_way_is_not_moved(monkeypatch):
    """The hole in the first version of the check, found by probing the LIVE
    router rather than trusting the unit tests: `before != now` was accepted as
    evidence of movement. For a binary on/off a change means the OPPOSITE of what
    was asked, and an entity missing from the registry reads as an empty state
    that therefore also "changed" — so both of these reported `moved`."""
    import app as jev_router_app

    class _HA:
        def __init__(self, states):
            self.states = states

        async def get_states(self, force=False):
            return self.states

    # Asked to turn OFF, still on: not reached.
    monkeypatch.setattr(jev_router_app, "ha", _HA(
        [{"entity_id": "switch.coffemaker", "state": "on"}]))
    moved, _ = await jev_router_app._confirm_on_off_moved(
        ["switch.coffemaker"], {"switch.coffemaker": "off"}, want_on=False)
    assert moved == [], "a device left in the opposite state was called moved"

    # Absent from the registry entirely: nothing there could have moved.
    monkeypatch.setattr(jev_router_app, "ha", _HA(
        [{"entity_id": "switch.что-то_другое", "state": "on"}]))
    moved, unmoved = await jev_router_app._confirm_on_off_moved(
        ["switch.coffemaker"], {"switch.coffemaker": "off"}, want_on=True)
    assert moved == [], "an entity absent from the registry was called moved"
    assert unmoved == ["switch.coffemaker"]

    # unavailable and unknown are equally not-reached.
    for bad in ("unavailable", "unknown"):
        monkeypatch.setattr(jev_router_app, "ha", _HA(
            [{"entity_id": "switch.coffemaker", "state": bad}]))
        moved, _ = await jev_router_app._confirm_on_off_moved(
            ["switch.coffemaker"], {"switch.coffemaker": "off"}, want_on=True)
        assert moved == [], f"a {bad} device was called moved"


@pytest.mark.asyncio
async def test_a_device_already_in_the_requested_state_counts_as_reached(monkeypatch):
    """The user asked for a state, not for an event — «включи свет» with the lamp
    already on is satisfied, and the caller says so before ever reaching here."""
    import app as jev_router_app

    class _HA:
        async def get_states(self, force=False):
            return [{"entity_id": "switch.entrance_light_switch_relay", "state": "on"}]

    monkeypatch.setattr(jev_router_app, "ha", _HA())
    moved, _ = await jev_router_app._confirm_on_off_moved(
        ["switch.entrance_light_switch_relay"],
        {"switch.entrance_light_switch_relay": "off"}, want_on=True,
    )
    assert moved == ["switch.entrance_light_switch_relay"]


# --- «сними с паузы» was PAUSING, and «пауза» had no room (06.10.2026 20:10) ---
#
# Measured on the living-room camera, two commands in a row:
#
#     20:10:20 'иметь паузу'   -> media_pause on media_player.le_vlada  «Поставила на паузу»
#     20:10:40 'сними из паузы' -> media_target_ambiguous_or_absent — escalating
#
# Two separate faults, and the user described both: the router PAUSED when asked
# to unpause, and it picked a box in somebody else's bedroom and then asked where
# to pause.


def test_removing_the_pause_is_not_a_pause():
    """`\\bпауз\\w*` sat FIRST in RE_MEDIA, matched the bare stem inside «паузы», and
    the loop breaks on the first hit — so every unpause phrasing became a pause.
    Order IS the semantics here: resume phrases contain the word «паузы»."""
    from resolver import resolve_action

    for text in ("сними с паузы", "сними из паузы", "Сними с паузы гостиной",
                 "продолжай", "возобнови"):
        call = resolve_action(text, "livingroom")
        assert call is not None, f"{text!r} does not resolve at all"
        assert call.tool == "media__media_play", (
            f"{text!r} -> {call.tool}: the router PAUSED when asked to unpause"
        )


def test_a_bare_pause_still_pauses():
    """The reorder must not swallow the verb it was protecting."""
    from resolver import resolve_action

    for text in ("поставь на паузу", "пауза", "приостанови", "заморозь"):
        call = resolve_action(text, "livingroom")
        assert call is not None, f"{text!r} does not resolve at all"
        assert call.tool == "media__media_pause", f"{text!r} -> {call.tool}"


def test_a_bare_pause_carries_the_room_the_camera_is_in():
    """Measured: a bare «поставь на паузу» reached the router with
    `args={'service_data': {}}` — no area at all — so `_execute_media` guessed
    among four boxes and chose `media_player.le_vlada`, and the user was asked
    where to pause. The lights take the default area in the same breath
    (`{'domain': ['light'], 'area': 'Living Room'}`), so the asymmetry was here
    alone. "Pause whatever is playing near me" names exactly one player."""
    from resolver import resolve_action

    for room, area in (("livingroom", "Living Room"), ("kitchen", "Kitchen")):
        call = resolve_action("поставь на паузу", room)
        assert call is not None
        assert call.args.get("area") == area, (
            f"{room}: {call.args} — the camera's room was dropped, so the router "
            "guessed a player in someone else's bedroom"
        )


def test_the_relative_volume_case_still_drops_the_room():
    """The original reason for dropping it: «громче» said in a room is about a
    listener, not a claim about which player. That reasoning is untouched — the
    change is scoped to transport verbs."""
    from resolver import resolve_action

    call = resolve_action("громче", "kitchen")
    assert call is not None
    assert "area" not in call.args, (
        f"«громче» must not inherit the room as a player claim: {call.args}"
    )


# --- the on/off happy path was silent, so a false «Включила» was unauditable ---
#
# Kitchen 10.10.2026 11:51: «включи кофеварку» -> route=easy_action conf=0.92 ->
# `speaking: 'Включила'` in 0.32 s, and that was ALL the router logged. HA's own
# history showed `switch.coffemaker` still `off` for the whole window (the next
# change was at 11:54:20, by hand), and `switch.coffemaker_child_lock` never moved
# either. The word «кофеварка» resolves to BOTH, so a partial fan-out was speaking
# as a whole.
#
# Three changes, all about being able to tell afterwards what happened: the target
# list with its states, every per-call answer, and the read-back verdict including
# the entities that did NOT reach the target. A verdict that cannot be audited is
# the same class of problem as a verdict that is wrong.


# --- a partial on/off fan-out must never speak as a whole --------------------
#
# Kitchen 10.10.2026 11:51: «включи кофеварку» -> `speaking: 'Включила'` in
# 0.32 s, and that was ALL the router logged. HA's history showed
# `switch.coffemaker` still `off` for the whole window — the next change was
# 11:54:20, by hand — so the turn claimed a side effect that never happened, and
# nothing in the log recorded the targets, the per-call answers or the read-back.
#
# Two separate fixes are pinned here: the trace, and the refusal to speak for a
# partial fan-out. The partial case needs two entities that are NOT facets of each
# other — `dedupe_device_facets` drops `switch.coffemaker_child_lock` as a facet of
# `switch.coffemaker` (verified live), so the coffee maker alone cannot produce it.
# Two light relays can.


def _two_relays_stub_ha():
    return _DeadHA(
        states=[
            {"entity_id": "switch.living_room_light_swith_relay", "state": "off",
             "attributes": {"friendly_name": "living_room_light_swith Relay"}},
            {"entity_id": "switch.entrance_light_switch_relay", "state": "off",
             "attributes": {"friendly_name": "entrance_light_switch Relay"}},
        ],
        result={"ok": True, "claimed": [{"type": "entity",
                                         "id": "switch.living_room_light_swith_relay"}]},
    )


@pytest.mark.asyncio
async def test_a_partial_on_off_fan_out_never_speaks_as_a_whole():
    """One of two devices answering is not «Включила» for the devices the user
    named. A refusal that names what failed is recoverable; a success nobody can
    check is not."""
    mod = _router_app()
    mod.ha = _two_relays_stub_ha()
    mod.find_action_targets = lambda states, areas, hint, area: [
        {"entity_id": "switch.living_room_light_swith_relay", "state": "off"},
        {"entity_id": "switch.entrance_light_switch_relay", "state": "off"},
    ]

    async def _only_one_moved(ids, before, want_on):
        return [ids[0]], list(ids[1:])

    mod._confirm_on_off_moved = _only_one_moved

    sentence, err = await mod._execute_action(
        resolve_action("включи свет в гостиной", "livingroom")
    )

    assert err is None, f"a partial application must not surface as an error: {err}"
    assert sentence != "Включила", (
        "the room was told the whole command succeeded while half of it did not"
    )
    assert "не сработали" in sentence, (
        f"the reply must name what failed, got {sentence!r}"
    )


@pytest.mark.asyncio
async def test_a_full_on_off_fan_out_still_speaks_plain_success():
    """The partial case must not have broken the ordinary one."""
    mod = _router_app()
    mod.ha = _two_relays_stub_ha()
    mod.find_action_targets = lambda states, areas, hint, area: [
        {"entity_id": "switch.living_room_light_swith_relay", "state": "off"},
        {"entity_id": "switch.entrance_light_switch_relay", "state": "off"},
    ]

    async def _all_moved(ids, before, want_on):
        return list(ids), []

    mod._confirm_on_off_moved = _all_moved

    sentence, err = await mod._execute_action(
        resolve_action("включи свет в гостиной", "livingroom")
    )

    assert err is None
    assert sentence == "Включила", f"got {sentence!r}"


# --- the first read-back must be LATE: HA writes the state before physics does ---
#
# Measured 10.10.2026 on `switch.coffemaker`, with the user's explicit permission
# to toggle it:
#
#     t=+0.01  switch=on     power=1072 W
#     t=+0.26  switch=off    power=1072 W     <- HA already wrote the state
#     t=+0.77  switch=off    power=0 W        <- the physics agrees
#
# `switch.turn_on/off` updates the entity state OPTIMISTICALLY and `/api/states`
# serves it until the integration's next poll. The old schedule read at 0 / 0.25 /
# 0.5 s, i.e. entirely inside that window, so it was reading back the value HA
# had just written and calling it a side effect.


@pytest.mark.asyncio
async def test_the_read_back_waits_longer_than_the_optimistic_window(monkeypatch):
    """Asserted on the SCHEDULE, because that is the defect: a sub-second read
    cannot outlast a state HA writes before the device acts."""
    import inspect
    import re

    import app as jev_router_app

    src = inspect.getsource(jev_router_app._confirm_on_off_moved)
    m = re.search(r"for delay in \(([^)]*)\)", src)
    assert m, "the retry schedule is gone; the check must stay explicit"
    delays = [float(x) for x in m.group(1).split(",")]

    assert delays[0] >= 1.0, (
        f"the first read is at {delays[0]}s, inside the window in which HA's "
        f"optimistic state is still being served — measured divergence is 0.5 s "
        f"(state at +0.26 s, physics at +0.77 s)"
    )
    assert delays == sorted(delays), "the retries must back off, not tighten"
    assert len(delays) >= 2, "a relay is not instantaneous; one read is not a retry"


@pytest.mark.asyncio
async def test_a_device_that_moves_late_is_still_accepted(monkeypatch):
    """Moving the first read later must not turn slow relays into refusals — the
    retry exists precisely for that."""
    import app as jev_router_app

    calls = {"n": 0}

    class _HA:
        async def get_states(self, force=False):
            calls["n"] += 1
            # Still off on the first read, on by the second — a relay that takes
            # a moment, which the old schedule accepted at 0.25 s and this must
            # keep accepting.
            state = "off" if calls["n"] == 1 else "on"
            return [{"entity_id": "switch.relay", "state": state}]

    monkeypatch.setattr(jev_router_app, "ha", _HA())

    moved, unmoved = await jev_router_app._confirm_on_off_moved(
        ["switch.relay"], {"switch.relay": "off"}, want_on=True
    )
    assert moved == ["switch.relay"], (
        f"a relay that settled on the second read was refused: {unmoved}"
    )
