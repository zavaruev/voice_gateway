"""jev-router unit tests: slot resolver + classifier regex fast-paths.

Runs offline on host python: only pure functions are exercised (no FastAPI,
no Ollama/Qdrant/HA network calls). services/jev-router is put on sys.path
because the host has no fastapi — app.py is never imported here.
"""

import os
import sys

_ROUTER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "jev-router")
)
sys.path.insert(0, _ROUTER)

import classifier as clf  # noqa: E402
from resolver import resolve_action, resolve_query  # noqa: E402
from ha_client import find_entity, describe_entity  # noqa: E402


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
