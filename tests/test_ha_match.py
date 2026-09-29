"""ha_match tests: the bilingual REST fallback + the "carries data?" detector.

Field case 29.09.2026 («Что там с нашим пылесосом?»): ha_read returned the
MCP miss payload verbatim (`{"success": false, "error": "No exposed entities
matched name 'пылесос'"}`) because it only looked for the «Ошибка» prefix, so
the REST fallback never ran — and «пылесос» would not have matched the latin
registry anyway. Pure stdlib: runs on the host python as well as in CI.
"""

import os
import sys

_WORKER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "smolagents-worker")
)
sys.path.insert(0, _WORKER)

from ha_match import has_data, match_states, matchers  # noqa: E402

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
