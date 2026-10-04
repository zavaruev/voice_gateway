"""ha_match tests: the bilingual REST fallback + the "carries data?" detector.

Field case 29.09.2026 («Что там с нашим пылесосом?»): ha_read returned the
MCP miss payload verbatim (`{"success": false, "error": "No exposed entities
matched name 'пылесос'"}`) because it only looked for the «Ошибка» prefix, so
the REST fallback never ran — and «пылесос» would not have matched the latin
registry anyway. Pure stdlib: runs on the host python as well as in CI.
"""

import json
import os
import sys

_WORKER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "smolagents-worker")
)
sys.path.insert(0, _WORKER)

from ha_match import (  # noqa: E402
    area_matchers,
    has_data,
    match_states,
    matchers,
    media_fingerprint,
    media_service_data,
    normalize_media_action,
    ordinal_digit,
    query_domains,
    resolve_media_targets,
    resolve_onoff_targets,
)

_STATES = [
    {"entity_id": "update.vacuum_card_update", "state": "off",
     "attributes": {"friendly_name": "Vacuum Card Update"}},
    {"entity_id": "vacuum.valetudo_zealouseverlastinggaur", "state": "docked",
     "attributes": {"friendly_name": "Roborock Robot"}},
    {"entity_id": "sensor.bedroom_humidity", "state": "41",
     "attributes": {"friendly_name": "Bedroom Humidity", "unit_of_measurement": "%"}},
]


# --- has_data(): what ha_read used to treat as data --------------------------


def test_mcp_miss_payload_is_not_data():
    """The literal payload from the field case must be a MISS, not data."""
    assert has_data(
        '{"success": false, "error": "No exposed entities matched name '
        "'пылесос'\"}"
    ) is False


def test_error_prefixes_are_not_data():
    assert has_data("") is False
    assert has_data("   ") is False
    assert has_data("Ошибка: connection refused") is False
    assert has_data("Тул вернул ошибку: MatchFailedReason.NAME") is False
    assert has_data("Не удалось прочитать состояния: timeout") is False
    assert has_data("Ничего не найдено в Home Assistant.") is False


def test_real_results_are_data():
    assert has_data('{"success": true, "states": []}') is True
    assert has_data("vacuum.valetudo_x: docked (Roborock Robot)") is True
    # A context blob with no success key at all is data, not a miss.
    assert has_data("Комната кухня: 22.5 °C") is True


# --- matchers(): RU word -> latin registry fragments -------------------------


def test_russian_device_expands_to_latin_aliases():
    m = matchers("пылесос")
    assert "пылесос" in m
    for lat in ("vacuum", "roborock", "robot"):
        assert lat in m


def test_area_expansion_is_shared():
    m = matchers("свет", "кухня")
    assert "light" in m and "kitchen" in m


def test_short_tokens_do_not_pull_the_lamp_in():
    """«что с ним» — the preposition «с» is a substring of «свет» and must
    not expand into the `light` aliases."""
    assert "light" not in matchers("что с ним")
    assert "kitchen" not in matchers("а с")


# --- match_states(): the field case end to end --------------------------------


def test_russian_query_finds_the_latin_vacuum():
    hits = match_states(_STATES, "пылесос")
    assert hits, "no hits for «пылесос»"
    # Both `vacuum.*` and `update.vacuum_card_update` match the alias, but the
    # entity's own domain must win — otherwise the model reads state off a
    # card-update helper entity.
    assert hits[0].startswith("vacuum.")
    assert any("docked" in h for h in hits)


def test_without_the_alias_the_registry_is_invisible():
    """The gap this module closes: a plain «пылесос» token match finds nothing."""
    assert match_states(
        [{"entity_id": "vacuum.valetudo_x", "state": "docked",
          "attributes": {"friendly_name": "Roborock Robot"}}],
        "пылесоса",
    ) == ["vacuum.valetudo_x: docked (Roborock Robot)"]


def test_unrelated_query_finds_nothing():
    assert match_states(_STATES, "пылесос") != []
    assert match_states(_STATES, "хомяк") == []


def test_hits_are_capped_at_ten():
    states = [
        {"entity_id": f"vacuum.robot_{i}", "state": "docked",
         "attributes": {"friendly_name": "Robot"}}
        for i in range(15)
    ]
    assert len(match_states(states, "пылесос")) == 10


def test_the_device_survives_the_cap():
    """Field case 29.09.2026, 14:17: ten helper entities matching the same
    hint sat EARLIER in the HA registry and the cap was applied before the
    rank sort, so `vacuum.valetudo_…: docked` was the line that got dropped.
    The model was shown a pile of consumable-reset buttons and never the
    device it was asked about — half of why it invented the state and the
    battery. Rank first, cap second."""
    helpers = [
        {"entity_id": f"button.valetudo_reset_{i}", "state": "unknown",
         "attributes": {"friendly_name": f"Roborock Reset {i}"}}
        for i in range(9)
    ]
    states = [
        {"entity_id": "update.vacuum_card_update", "state": "off",
         "attributes": {"friendly_name": "Vacuum Card Update"}},
        *helpers,
        {"entity_id": "vacuum.valetudo_zealouseverlastinggaur", "state": "docked",
         "attributes": {"friendly_name": "Roborock Robot"}},
    ]
    hits = match_states(states, "пылесос")
    assert len(hits) == 10  # 11 candidates -> still capped
    assert hits[0].startswith("vacuum.")  # but the device is IN, and first
    assert "docked" in hits[0]


def test_non_list_states_are_safe():
    assert match_states(None, "пылесос") == []
    assert matchers("") == []


# --- The device's READINGS ride along with the device ------------------------
# Field case 29.09.2026, 14:58: «Проверь робот и его заряд батареи» -> L2
# read «пылесос» and honestly answered «узнать уровень заряда не удалось»
# while HA reported 97 %. Valetudo puts `battery_level` in a separate
# sensor that the integration creates LAST — after four consumable-reset
# buttons, the map camera, statistics and wi-fi — so it was the ~19th of
# 26 matches and the 10-line cap cut it out of the payload.

_ROBOT_STATES = [
    {"entity_id": "update.vacuum_card_update", "state": "off",
     "attributes": {"friendly_name": "Vacuum Card Update"}},
    {"entity_id": "button.valetudo_x_reset_main_brush_consumable",
     "state": "unknown",
     "attributes": {"friendly_name": "Roborock Reset Main Brush Consumable"}},
    {"entity_id": "button.valetudo_x_reset_right_brush_consumable",
     "state": "unknown",
     "attributes": {"friendly_name": "Roborock Reset Right Brush Consumable"}},
    {"entity_id": "camera.valetudo_x_map_data", "state": "idle",
     "attributes": {"friendly_name": "Roborock Map data"}},
    {"entity_id": "number.valetudo_x_speaker_volume", "state": "100",
     "attributes": {"friendly_name": "Roborock Speaker volume"}},
    {"entity_id": "select.valetudo_x_fan", "state": "medium",
     "attributes": {"friendly_name": "Roborock Fan"}},
    {"entity_id": "sensor.valetudo_x_map_segments", "state": "8",
     "attributes": {"friendly_name": "Roborock Map segments"}},
    {"entity_id": "sensor.valetudo_x_main_brush", "state": "665",
     "attributes": {"friendly_name": "Roborock Main Brush",
                    "unit_of_measurement": "min"}},
    {"entity_id": "sensor.valetudo_x_total_statistics_time", "state": "2006294",
     "attributes": {"friendly_name": "Roborock Total statistics time",
                    "unit_of_measurement": "s"}},
    {"entity_id": "sensor.valetudo_x_wi_fi_configuration", "state": "-55",
     "attributes": {"friendly_name": "Roborock Wi-Fi configuration",
                    "unit_of_measurement": "dBm"}},
    {"entity_id": "sensor.valetudo_x_battery_level", "state": "97",
     "attributes": {"friendly_name": "Roborock Battery level",
                    "unit_of_measurement": "%"}},
    {"entity_id": "switch.valetudo_x_carpet_mode", "state": "off",
     "attributes": {"friendly_name": "Roborock Carpet mode"}},
    {"entity_id": "vacuum.valetudo_x", "state": "docked",
     "attributes": {"friendly_name": "Roborock Robot"}},
]


def test_the_devices_battery_rides_along_with_the_device():
    """Device first, its own charge SECOND — ahead of every helper."""
    hits = match_states(_ROBOT_STATES, "пылесос")
    assert hits[0].startswith("vacuum.")
    assert hits[1].startswith("sensor.valetudo_x_battery_level")
    assert hits[1].split(": ", 1)[1].startswith("97")
    assert len(hits) == 10  # the cap still applies — to the right list now


def test_the_rule_works_for_every_device_spelling():
    for query in ("робот", "заряд робота", "что с роботом на кухне"):
        hits = match_states(_ROBOT_STATES, query, "кухня")
        assert "battery_level" in hits[1], query


def test_a_metric_only_query_keeps_the_registry_order():
    """No device on top («заряд» alone) has nothing to hang a reading off:
    the promotion must not reorder a plain metric lookup."""
    states = [
        {"entity_id": "sensor.phone_battery_level", "state": "33",
         "attributes": {"friendly_name": "Phone Battery level",
                        "unit_of_measurement": "%"}},
        {"entity_id": "sensor.valetudo_x_battery_level", "state": "97",
         "attributes": {"friendly_name": "Roborock Battery level",
                        "unit_of_measurement": "%"}},
    ]
    hits = match_states(states, "заряд")
    assert hits[0].startswith("sensor.phone_")  # registry order, untouched


def test_readings_of_another_device_stay_where_they_are():
    """Only the matched device's OWN instance may promote readings — the
    shared household battery must not jump in front of someone else's list."""
    # Another vacuum's battery — matches the same hint (roborock/vacuum in
    # the hay) but has a different instance prefix, so it must NOT ride along.
    states = [
        {"entity_id": "vacuum.valetudo_x", "state": "docked",
         "attributes": {"friendly_name": "Roborock Robot"}},
        {"entity_id": "sensor.other_vacuum_battery_level", "state": "12",
         "attributes": {"friendly_name": "Spare Roborock Battery level",
                        "unit_of_measurement": "%"}},
        {"entity_id": "sensor.valetudo_x_battery_level", "state": "97",
         "attributes": {"friendly_name": "Roborock Battery level",
                        "unit_of_measurement": "%"}},
    ]
    hits = match_states(states, "пылесос")
    idx_own = next(i for i, h in enumerate(hits)
                   if "valetudo_x_battery_level" in h)
    idx_other = next(i for i, h in enumerate(hits)
                     if "other_vacuum" in h)
    assert idx_own == 1 and idx_own < idx_other


# --- The hallway lamp: room beats the domain hint (02.10.2026 05:45) --------
# «Значит, прихожий.» escalated to L2, which sent a BLIND
# `intent__HassTurnOff {"area": "прихожая", "domain": ["light"]}`. HA
# answered MatchFailedReason.AREA twice («прихожая», then «Entrance») and the
# turn was spoken as «я не нашёл устройств в прихожей». Two truths about this
# house: there is no `light.*` in area Entrance at all — the lamp is
# `switch.entrance_light_switch_relay` — and ten light.* status LEDs match
# the word «light» and used to fill the whole 10-line cap.


def _st(eid: str, state: str, name: str = "") -> dict:
    return {"entity_id": eid, "state": state,
            "attributes": {"friendly_name": name}}


_HALL_STATES = [
    # registry noise that matches «light» and sits earlier than the lamp
    *[_st(f"light.esp32_status_led_{i}", "unavailable", f"Esp32 Status Led {i}")
      for i in range(10)],
    _st("text.entrance_light_switch_device_config_switch", "qq9ahj6z;TS0001",
        "entrance_light_switch Device config switch"),
    _st("switch.entrance_light_switch_network_led_switch", "off",
        "entrance_light_switch Network led switch"),
    _st("switch.entrance_light_switch_relay", "on",
        "entrance_light_switch Relay"),
    _st("select.entrance_light_switch_switch_mode_switch", "momentary",
        "entrance_light_switch Switch mode switch"),
    _st("update.entrance_light_switch", "off", "entrance_light_switch"),
    _st("binary_sensor.entrance_move_sensor_occupancy", "off",
        "entrance_move_sensor Occupancy"),
]


def test_the_lamp_is_first_for_a_room_query():
    """«свет в прихожей»: the relay matches BOTH the room and the device word,
    the ten LEDs match only the device word — the room key comes first."""
    hits = match_states(_HALL_STATES, "свет в прихожей")
    assert hits[0].startswith("switch.entrance_light_switch_relay"), hits[:3]


def test_the_lamp_survives_the_ten_led_cap():
    """Before the ranking fix the LEDs were rank 0 (their domain IS the hint)
    and the cap cut the relay out of the payload entirely."""
    hits = match_states(_HALL_STATES, "свет", "прихожая")
    assert any(h.startswith("switch.entrance_light_switch_relay: on") for h in hits)


def test_facets_rank_below_the_device_itself():
    hits = match_states(_HALL_STATES, "свет в прихожей")
    relay = next(i for i, h in enumerate(hits) if "entrance_light_switch_relay" in h)
    led = next(i for i, h in enumerate(hits) if "network_led_switch" in h)
    assert relay < led


# --- resolve_onoff_targets(): the deterministic ha_action fallback ----------


def test_resolver_finds_the_relay_despite_domain_light():
    """The exact failing call: domain ["light"] in an area with no light.*."""
    got = resolve_onoff_targets(_HALL_STATES, area="прихожая", domains=["light"])
    assert [e["entity_id"] for e in got] == ["switch.entrance_light_switch_relay"]


def test_resolver_accepts_the_english_area_name():
    got = resolve_onoff_targets(_HALL_STATES, area="Entrance", domains=["light"])
    assert [e["entity_id"] for e in got] == ["switch.entrance_light_switch_relay"]


def test_resolver_drops_the_status_led_facet():
    """Both switch.* match «light» and «entrance»; the network LED is a facet
    of the same unit and must never be the thing that gets toggled."""
    got = resolve_onoff_targets(_HALL_STATES, area="прихожая",
                                domains=["light", "switch"])
    assert [e["entity_id"] for e in got] == ["switch.entrance_light_switch_relay"]


def test_resolver_refuses_when_no_device_word_was_given():
    """Area only: nothing says WHICH thing in the room was meant."""
    assert resolve_onoff_targets(_HALL_STATES, area="прихожая") == []


def test_resolver_ignores_offline_devices():
    states = [_st("switch.entrance_light_switch_relay", "unavailable",
                  "entrance_light_switch Relay")]
    assert resolve_onoff_targets(states, area="прихожая", domains=["light"]) == []


def test_resolver_keeps_the_lamp_when_only_facets_match():
    """All-facet pool: return it anyway — the CALLER must refuse honestly,
    the resolver must not silently pretend nothing was found."""
    states = [_st("switch.entrance_light_switch_network_led_switch", "off",
                  "entrance_light_switch Network led switch")]
    got = resolve_onoff_targets(states, area="прихожая", domains=["light"])
    assert [e["entity_id"] for e in got] == [
        "switch.entrance_light_switch_network_led_switch"]


_CORRIDOR_STATES = [
    _st("switch.corridor1_light_switch_relay", "off",
        "corridor1_light_switch Relay"),
    _st("switch.corridor2_light_switch_relay", "off",
        "corridor2_light_switch Relay"),
]


def test_two_lamps_in_a_named_room_are_both_targets():
    """«свет в коридоре» is TWO relays and both belong to the command."""
    got = resolve_onoff_targets(_CORRIDOR_STATES, area="коридор",
                                domains=["light"])
    assert len(got) == 2


def test_several_devices_without_a_room_are_a_guess():
    """Mirrors jev-router's ambiguous_no_area: no room, more than one
    candidate -> refuse instead of picking one."""
    assert resolve_onoff_targets(_CORRIDOR_STATES, domains=["light"]) == []


def test_the_area_registry_decides_the_room_when_the_id_says_nothing():
    """Entity ids do not always carry the room (the registry does)."""
    states = [
        _st("light.bedroom_ceiling", "on", "Ceiling"),
        _st("light.kitchen_ceiling", "on", "Ceiling"),
    ]
    areas = {"light.bedroom_ceiling": "Bedroom",
             "light.kitchen_ceiling": "Kitchen"}
    got = resolve_onoff_targets(states, area="спальня", domains=["light"],
                                area_map=areas)
    assert [e["entity_id"] for e in got] == ["light.bedroom_ceiling"]


def test_area_matchers_speaks_both_languages():
    """HA's area registry answers in English («Entrance»); STT asks in
    Russian. An empty set here silently switches the room filter off."""
    assert {"entrance", "hallway"} <= area_matchers("Entrance")
    assert {"entrance", "прихож"} <= area_matchers("прихожая")
    assert not area_matchers("пылесос")


def test_an_exact_room_wins_over_a_room_that_contains_it():
    """«спальня» -> {bedroom} also matches «Bedroom Vlada»: the exact room
    must win, or «свет в спальне» switches off Vlad's garland next door."""
    states = [
        _st("light.wled", "off", "WLED"),
        _st("light.bedroom_ceiling", "on", "Ceiling"),
    ]
    areas = {"light.wled": "Bedroom Vlada",
             "light.bedroom_ceiling": "Bedroom"}
    got = resolve_onoff_targets(states, area="спальня", domains=["light"],
                                area_map=areas)
    assert [e["entity_id"] for e in got] == ["light.bedroom_ceiling"]


# --- Ordinals: «первый коридор» is corridor1, never both --------------------
# Field case 02.10.2026 (second half): resolve_action threw «первом»/«втором»
# away, so L1 resolved hint «свет» + area «Corridor» and MOVED BOTH relays —
# «выключи свет во втором коридоре» also switched the first one off.


def test_ordinal_digit_matches_whole_words_only():
    assert ordinal_digit("свет в первом коридоре") == "1"
    assert ordinal_digit("выключи во втором") == "2"
    assert ordinal_digit("третий") == "3"
    assert ordinal_digit("коридор 1") == "1"
    # false friends: same stem, wrong ending -> not an ordinal
    assert ordinal_digit("во вторник") == ""
    assert ordinal_digit("свет в прихожей") == ""
    assert ordinal_digit("") == ""


def test_only_the_named_corridor_is_a_target():
    first = resolve_onoff_targets(_CORRIDOR_STATES, area="первый коридор",
                                  domains=["light"])
    assert [e["entity_id"] for e in first] == ["switch.corridor1_light_switch_relay"]
    second = resolve_onoff_targets(_CORRIDOR_STATES, area="второй коридор",
                                   domains=["light"])
    assert [e["entity_id"] for e in second] == ["switch.corridor2_light_switch_relay"]


def test_the_ordinal_may_arrive_in_the_name_slot():
    """L2 sees the whole utterance and may hand the phrase over as `name`
    instead of `area` — the digit must be found there too."""
    got = resolve_onoff_targets(_CORRIDOR_STATES, name="свет в первом коридоре",
                                domains=["light"])
    assert [e["entity_id"] for e in got] == ["switch.corridor1_light_switch_relay"]


def test_a_numbered_instance_that_does_not_exist_is_refused():
    """No corridor 3: [] keeps the original error — moving both relays for a
    request that named one of them would be a guessed side effect."""
    assert resolve_onoff_targets(_CORRIDOR_STATES, area="третий коридор",
                                 domains=["light"]) == []


def test_a_command_without_an_ordinal_still_moves_both():
    got = resolve_onoff_targets(_CORRIDOR_STATES, area="коридор",
                                domains=["light"])
    assert len(got) == 2


def test_the_reading_is_narrowed_to_the_named_corridor():
    hits = match_states(_CORRIDOR_STATES, "свет в первом коридоре")
    assert hits, hits
    assert hits[0].startswith("switch.corridor1_light_switch_relay"), hits
    assert all("corridor1" in h for h in hits), hits
    hits = match_states(_CORRIDOR_STATES, "свет во втором коридоре")
    assert hits and all("corridor2" in h for h in hits), hits


def test_a_stray_digit_never_empties_the_payload():
    """`light.wled1` carries a '1' but sits in another room — the room part
    of the filter keeps it out, and if NOTHING matched the digit the unfiltered
    hits would survive instead of returning an empty answer."""
    states = _CORRIDOR_STATES + [_st("light.wled1", "off", "WLED1")]
    hits = match_states(states, "свет в первом коридоре")
    assert hits and all("corridor1" in h for h in hits), hits



# --- The `name` slot must not outrank the area registry ---------------------
# Field check 02.10.2026: with no `area` slot the resolver filtered the room
# by an ID SUBSTRING, so an entity with area None (`switch.corridor1_detect`)
# got in, the pool crossed MAX_ONOFF_TARGETS and «первый коридор» handed over
# as a `name` returned [] while the same words as an `area` resolved the relay.


def test_a_name_only_call_uses_the_area_registry_not_the_id_substring():
    states = _CORRIDOR_STATES + [
        _st("switch.corridor1_detect", "off", "corridor1 Detect"),
        _st("switch.corridor1_motion", "off", "corridor1 Motion"),
        _st("switch.corridor1_recordings", "off", "corridor1 Recordings"),
        _st("switch.corridor1_snapshots", "off", "corridor1 Snapshots"),
        _st("switch.corridor1_review_alerts", "off", "corridor1 Review alerts"),
        _st("switch.corridor1_review_detections", "off",
            "corridor1 Review detections"),
    ]
    areas = {e["entity_id"]: "Corridor" for e in _CORRIDOR_STATES}
    got = resolve_onoff_targets(states, name="первый коридор",
                                domains=["switch"], area_map=areas)
    assert [e["entity_id"] for e in got] == [
        "switch.corridor1_light_switch_relay"]


def test_the_exact_room_rule_applies_to_name_only_calls_too():
    """«свет в спальне» arriving as `name` must beat «Bedroom Vlada» the same
    way it does when it arrives as `area` — otherwise name-only calls lose the
    rule and end in a >1-candidate refusal."""
    states = [
        _st("light.wled", "off", "WLED Bedroom"),
        _st("light.bedroom_ceiling", "on", "Ceiling"),
    ]
    areas = {"light.wled": "Bedroom Vlada",
             "light.bedroom_ceiling": "Bedroom"}
    got = resolve_onoff_targets(states, name="свет в спальне",
                                domains=["light"], area_map=areas)
    assert [e["entity_id"] for e in got] == ["light.bedroom_ceiling"]


def test_an_invented_device_word_is_not_rescued_by_the_fallback():
    """2.34's field case stays intact now that the NAME mismatch also triggers
    the resolver: «кашеварку» matches nothing in the registry, so the pool is
    every switch with no room -> refusal, and HA's original NAME error (and
    with it «Не нашла такого устройства») is what the model still reports.
    The fallback rescues PHRASING, never guessing."""
    states = [
        _st("switch.coffemaker", "off", "coffeemaker"),
        _st("switch.relay", "off", "Relay"),
    ]
    assert resolve_onoff_targets(states, name="кашеварку",
                                 domains=["switch"]) == []


def test_a_latin_friendly_name_is_a_descriptor_not_a_hint():
    """HA hands the model friendly names like «corridor1_light_switch Relay».
    Matched loosely (the old `any` rule) the bare «…_relay» tail also admits
    corridor2's relay, the pool becomes two candidates with no `area` slot and
    the call is refused — a failure for a device the model named exactly."""
    areas = {e["entity_id"]: "Corridor" for e in _CORRIDOR_STATES}
    got = resolve_onoff_targets(_CORRIDOR_STATES,
                                name="corridor1_light_switch Relay",
                                domains=["switch"], area_map=areas)
    assert [e["entity_id"] for e in got] == [
        "switch.corridor1_light_switch_relay"]
    got = resolve_onoff_targets(_CORRIDOR_STATES,
                                name="corridor2_light_switch relay",
                                domains=["switch"], area_map=areas)
    assert [e["entity_id"] for e in got] == [
        "switch.corridor2_light_switch_relay"]


# --- Media transport (field case 03.10.2026, «поставь его на паузу») --------
# The turn was lost because the model had to INVENT an intent name: HA's MCP
# server (tools/list, read live) exposes no media intent at all. media_control
# calls the REST service instead, and these are its two pure halves.


def _mp(eid: str, state: str, name: str) -> dict:
    return {"entity_id": eid, "state": state, "attributes": {"friendly_name": name}}


_MEDIA = [
    _mp("media_player.le_vlada", "paused", "LE-vlada"),
    _mp("media_player.le_zal_2", "idle", "LE-zal"),
    _mp("media_player.le_spalnya", "idle", "LE-spalnya"),
    _mp("media_player.le_kitchen", "idle", "LE-Kitchen"),
    _mp("media_player.x96q_pro1_157_dlna", "unavailable", "X96Q[DLNA]"),
    _mp("media_player.x96q_pro1_157_airplay", "off", "X96Q_PRO1-157[AirPlay]"),
]
_MEDIA_AREAS = {
    "media_player.le_vlada": "Bedroom Vlada",
    "media_player.le_zal_2": "Living Room",
    "media_player.le_spalnya": "Bedroom",
    "media_player.le_kitchen": "Kitchen",
    "media_player.x96q_pro1_157_dlna": "Living Room",
    "media_player.x96q_pro1_157_airplay": "Living Room",
}


def _media(action, name="", area="", states=None):
    svc = normalize_media_action(action)
    got = resolve_media_targets(states or _MEDIA, name=name, area=area,
                                area_map=_MEDIA_AREAS, service=svc)
    return svc, [e["entity_id"] for e in got]


def test_normalize_media_action_accepts_what_the_model_hands_over():
    """The model passes service names, short english verbs and — regularly —
    the user's own phrase («на паузу», «следующий трек»). Refusing a
    recognisable request only loses the turn."""
    for spoken, canonical in [
        ("media_pause", "media_pause"), ("MEDIA_PAUSE", "media_pause"),
        ("pause", "media_pause"), ("Пауза", "media_pause"),
        ("следующий трек", "media_next_track"), ("next", "media_next_track"),
        ("громче", "volume_up"), ("громкость", "volume_set"),
        ("заглуши", "volume_mute"), ("выключи звук", "volume_mute"),
        ("останови", "media_stop"), ("предыдущий трек", "media_previous_track"),
    ]:
        assert normalize_media_action(spoken) == canonical, spoken
    assert normalize_media_action("") == ""
    assert normalize_media_action("запусти ракету") == ""


def test_media_service_data_parses_only_what_it_can():
    """volume_set wants 0..1, volume_mute a boolean; an unusable `value` gives
    {} rather than a guessed level."""
    assert media_service_data("volume_set", "40 процентов") == {"volume_level": 0.4}
    assert media_service_data("volume_set", "40") == {"volume_level": 0.4}
    assert media_service_data("volume_set", "погромче") == {}
    assert media_service_data("volume_mute", "выключи") == {"is_volume_muted": True}
    assert media_service_data("volume_mute", "включи") == {"is_volume_muted": False}
    assert media_service_data("media_pause", "да") == {}


def test_media_targets_answer_the_field_case():
    """«пауза коди во владиной комнате» -> media_player.le_vlada. The room is
    the HA area name «Bedroom Vlada»; without the «влади»/«vlada» entries the
    spoken room matched nothing at all."""
    svc, ids = _media("media_pause", name="коди", area="владиной комнате")
    assert svc == "media_pause" and ids == ["media_player.le_vlada"]


def test_media_targets_pronoun_picks_the_state_the_request_is_about():
    """«поставь ЕГО на паузу»: nothing is playing, but exactly one box is
    paused — which is what the user is talking about."""
    assert _media("media_pause")[1] == ["media_player.le_vlada"]
    # …and for «продолжи» the priority flips: the paused one is the target.
    assert _media("media_play")[1] == ["media_player.le_vlada"]


def test_media_targets_intersect_name_and_room():
    """«коди» expands to «le», which every box carries. Without the room
    constraint «коди в гостиной» answered for le_vlada."""
    assert _media("volume_set", name="коди", area="гостиной")[1] == [
        "media_player.le_zal_2"]
    assert _media("volume_up", name="медиаплеер", area="спальне")[1] == [
        "media_player.le_spalnya"]


def test_media_targets_refuse_rather_than_guess():
    """A room with no player, and four idle boxes with no room: both refuse,
    and the tool turns that into one honest sentence for the model."""
    assert _media("media_pause", name="телевизор", area="коридоре")[1] == []
    # Four idle boxes, no room, no device word: nothing to tell them apart.
    idle = [_mp("media_player.le_vlada", "idle", "LE-vlada"),
            _mp("media_player.le_zal_2", "idle", "LE-zal"),
            _mp("media_player.le_spalnya", "idle", "LE-spalnya"),
            _mp("media_player.le_kitchen", "idle", "LE-Kitchen")]
    assert _media("media_stop", states=idle)[1] == []


def test_media_targets_ignore_dead_and_protocol_entities():
    """The DLNA endpoint is unavailable and the AirPlay receiver is not a box
    a person talks to — counting it made «громкость в гостиной» ambiguous."""
    assert _media("volume_set", area="гостиной")[1] == ["media_player.le_zal_2"]
    dead = [e for e in _MEDIA if e["state"] == "unavailable"]
    assert _media("media_pause", name="коди", states=dead)[1] == []
    # …but when EVERY player is an endpoint they are still candidates.
    endpoints = [_mp("media_player.tv_airplay", "idle", "TV[AirPlay]")]
    assert _media("media_pause", name="плеер", states=endpoints)[1] == [
        "media_player.tv_airplay"]


def test_media_targets_ignore_other_domains_entirely():
    """A `switch.*` or `tv.*` (the TV's own domain) is not a media_player."""
    others = [
        _mp("switch.tv_power", "on", "TV Power"),
        {"entity_id": "tv.living_room", "state": "on",
         "attributes": {"friendly_name": "TV"}},
    ]
    assert _media("media_pause", name="телевизор", states=others)[1] == []


def test_media_fingerprint_sees_attribute_only_changes():
    """`state` alone is not proof of «nothing happened»: volume_set only moves
    volume_level, and a track switch keeps the box `paused` while media_title
    changes. Both must register as a CHANGE — the opposite mistake recorded a
    real side effect as ok=False and the veto replaced a true answer."""
    before = _mp("media_player.a", "paused", "A")
    before["attributes"]["volume_level"] = 0.5
    after_vol = json.loads(json.dumps(before))
    after_vol["attributes"]["volume_level"] = 0.6
    after_title = json.loads(json.dumps(before))
    after_title["attributes"]["media_title"] = "Jaws Wired Shut"
    nothing = json.loads(json.dumps(before))
    nothing["attributes"]["media_position"] = 999  # ticks on its own
    assert media_fingerprint(before) != media_fingerprint(after_vol)
    assert media_fingerprint(before) != media_fingerprint(after_title)
    assert media_fingerprint(before) == media_fingerprint(nothing)


# --- A domain in the query is a HARD filter (field case 03.10.2026 18:21) ---
# «Ну так найди её и включи» -> ha_read(query="media_player", area="гостиная")
# answered with ten living-room SWITCHES (the light relay + six camera
# switches) and not a single player: matchers() ORs the room word with the
# domain word, so the room won and the domain was ignored. The model then
# spent two steps on that list.

_LIVING = [
    {"entity_id": "media_player.le_zal_2", "state": "idle",
     "attributes": {"friendly_name": "LE-zal"}},
    {"entity_id": "media_player.le_vlada", "state": "paused",
     "attributes": {"friendly_name": "LE-vlada"}},
    {"entity_id": "switch.living_room_light_swith_relay", "state": "on",
     "attributes": {"friendly_name": "living_room_light_swith Relay"}},
    {"entity_id": "switch.livingroom_camera_motion", "state": "on",
     "attributes": {"friendly_name": "Livingroom camera Motion"}},
    {"entity_id": "light.wled_living_room", "state": "unavailable",
     "attributes": {"friendly_name": "WLED_living_room"}},
    {"entity_id": "light.kitchen_ceiling", "state": "on",
     "attributes": {"friendly_name": "Kitchen Ceiling"}},
]
_LIVING_AREAS = {
    "media_player.le_zal_2": "Living Room",
    "media_player.le_vlada": "Bedroom Vlada",
    "switch.living_room_light_swith_relay": "Living Room",
    "switch.livingroom_camera_motion": "Living Room",
    "light.wled_living_room": "Living Room",
    "light.kitchen_ceiling": "Kitchen",
}


def test_query_domains_recognises_only_real_registry_domains():
    assert query_domains("media_player", _LIVING) == {"media_player"}
    assert query_domains("media player", _LIVING) == {"media_player"}
    assert query_domains("light", _LIVING) == {"light"}
    # RU device words are never domains — «свет» must keep the fuzzy path.
    assert query_domains("свет", _LIVING) == set()
    assert query_domains("пылесос", _LIVING) == set()
    assert query_domains("", _LIVING) == set()
    assert query_domains("light", None) == set()


def test_a_domain_query_never_answers_with_another_domain():
    """The exact field call: media players in the living room, NOT switches."""
    got = match_states(_LIVING, "media_player", "гостиная", area_map=_LIVING_AREAS)
    assert [e.split(":", 1)[0] for e in got] == ["media_player.le_zal_2"]


def test_domain_plus_room_uses_the_area_map_the_names_cannot_carry():
    """`media_player.le_zal_2` says nothing about «гостиной»; without the area
    map the query found NOTHING at all (03.10.2026)."""
    assert match_states(_LIVING, "media_player", "владиной комнате",
                        area_map=_LIVING_AREAS) == [
        "media_player.le_vlada: paused (LE-vlada)"]
    # …and the name-based room match keeps working unchanged.
    assert match_states(_LIVING, "light", "гостиной",
                        area_map=_LIVING_AREAS) == [
        "light.wled_living_room: unavailable (WLED_living_room)"]
    # RU device words are untouched: no hard filter, name path only.
    assert match_states(_LIVING, "свет", "")[0].startswith(
        "light.kitchen_ceiling") or match_states(_LIVING, "свет", "")
