"""L2 on/off verification and the short spoken acknowledgement.

Field case 27.11.2026 16:09 MSK: the worker called a working coffee maker a
failure because it read a cached pre-call snapshot and never looked at the power
sensor. These tests pin the instrument, not the wording of an LLM answer.
"""

import os
import sys

_WORKER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "services", "smolagents-worker")
)
if _WORKER not in sys.path:
    sys.path.insert(0, _WORKER)

import ha_match  # noqa: E402
from ha_match import confirm_on_off_reached, confirm_on_off_verdict  # noqa: E402
from onoff_verify import ack_for_onoff  # noqa: E402


def _fast(monkeypatch):
    """Keep the real read schedule but don't sleep through it in unit tests."""
    monkeypatch.setattr(ha_match._time, "sleep", lambda _secs: None)


def test_fresh_read_sees_the_change_the_cache_hid(monkeypatch):
    _fast(monkeypatch)
    stale = [{"entity_id": "switch.coffemaker", "state": "off"}]
    fresh = [{"entity_id": "switch.coffemaker", "state": "on"}]
    assert stale[0]["state"] == "off"  # the snapshot taken before the call
    assert confirm_on_off_verdict(["switch.coffemaker"], "on", lambda: fresh) == (
        ["switch.coffemaker"], []
    )


def test_optimistic_state_without_power_is_not_reached(monkeypatch):
    _fast(monkeypatch)
    rows = [
        {"entity_id": "switch.coffemaker", "state": "on"},
        {"entity_id": "sensor.coffemaker_power", "state": "0",
         "attributes": {"device_class": "power"}},
    ]
    assert confirm_on_off_verdict(["switch.coffemaker"], "on", lambda: rows) == (
        [], ["switch.coffemaker"]
    )


def test_power_sensor_arriving_after_the_third_read_is_proved(monkeypatch):
    # Measured 27.11: the appliance moved at +2.93 s, after all three base reads.
    _fast(monkeypatch)
    calls = {"n": 0}

    def _feed():
        calls["n"] += 1
        watts = "0" if calls["n"] <= 3 else "1031"
        return [
            {"entity_id": "switch.coffemaker", "state": "on"},
            {"entity_id": "sensor.coffemaker_power", "state": watts,
             "attributes": {"device_class": "power"}},
        ]

    reached, unproven = confirm_on_off_verdict(
        ["switch.coffemaker"], "on", _feed
    )
    assert reached == ["switch.coffemaker"], (
        "the extra read must prove the device actually drawing power"
    )
    assert unproven == []
    assert calls["n"] == 4, "three base reads plus the one measured extra read"


def test_the_old_wrapper_keeps_its_contract(monkeypatch):
    _fast(monkeypatch)
    rows = [{"entity_id": "switch.relay", "state": "on"}]
    assert confirm_on_off_reached(["switch.relay"], "on", lambda: rows) == [
        "switch.relay"
    ]


def test_ack_phrases_are_short_and_gendered_by_the_router():
    assert ack_for_onoff("on") == "Включаю."
    assert ack_for_onoff("off") == "Выключаю."


def test_a_failed_states_fetch_is_unproven_not_a_failure(monkeypatch):
    """No answer from HA is absence of evidence, never a verdict on the device."""
    _fast(monkeypatch)
    calls = {"n": 0}

    def _feed():
        calls["n"] += 1
        return None if calls["n"] == 1 else [
            {"entity_id": "switch.coffemaker", "state": "off"}
        ]

    reached, unproven = confirm_on_off_verdict(
        ["switch.coffemaker"], "on", _feed
    )
    assert reached == []
    assert unproven == ["switch.coffemaker"]


def test_a_sensor_found_after_a_failed_read_is_still_used(monkeypatch):
    """A first read with no rows must not pin sensor pairing to None forever."""
    _fast(monkeypatch)
    calls = {"n": 0}

    def _feed():
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return [
            {"entity_id": "switch.coffemaker", "state": "on"},
            {"entity_id": "sensor.coffemaker_power", "state": "1031",
             "attributes": {"device_class": "power"}},
        ]

    reached, unproven = confirm_on_off_verdict(
        ["switch.coffemaker"], "on", _feed
    )
    assert reached == ["switch.coffemaker"], (
        "an optimistic 'on' with no sensor reading must not be accepted"
    )
    assert unproven == []


def test_unproven_answer_is_not_marked_as_a_tool_error():
    from ha_match import ERROR_PREFIXES
    from onoff_verify import UNPROVEN_ONOFF_TEXT

    for prefix in ERROR_PREFIXES:
        assert not UNPROVEN_ONOFF_TEXT.startswith(prefix), (
            f"an unproven result must not look like a tool error ({prefix})"
        )
