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


def test_non_list_states_are_safe():
    assert match_states(None, "пылесос") == []
    assert matchers("") == []
