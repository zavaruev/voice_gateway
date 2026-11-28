"""ONE on/off verification rule, used by both L1 (jev-router) and L2.

The two services are separate images, so this file is deliberately identical in
both service directories; tests/test_hint_sync.py is what keeps them one rule.
Field case 27.11.2026 16:09 MSK: L2 called a working coffee maker a failure
because it read a cached state snapshot and never consulted the power sensor.
Do not add service-specific behaviour here.
"""

POWER_SENSOR_SUFFIX = "_power"
# Measured: 1074 W running vs 0 W stopped; the midpoint is not a judgement call.
POWER_ON_W = 5.0


def power_sensor_for(entity_id: str, states: list[dict]) -> str | None:
    """The power sensor paired with this switch, or None.

    HA's /api/states rows carry no device_id, so name pairing is the only
    instrument available. Deliberately conservative: a wrong pairing would mean
    believing a number about the wrong appliance.
    """
    if not entity_id.startswith("switch."):
        return None
    base = entity_id.split(".", 1)[1]
    want = "sensor." + base + POWER_SENSOR_SUFFIX
    for ent in states or []:
        if ent.get("entity_id") != want:
            continue
        attrs = ent.get("attributes") or {}
        if attrs.get("device_class") != "power":
            return None
        return want
    return None


def physics_agrees(ent: dict | None, target: str) -> bool | None:
    """Does the power sensor support `target`? None = no opinion (unproven)."""
    if not ent:
        return None
    raw = str(ent.get("state", "")).strip().lower()
    if raw in ("unknown", "unavailable", "none", ""):
        return None
    try:
        watts = float(raw)
    except ValueError:
        return None
    return watts >= POWER_ON_W if target == "on" else watts < POWER_ON_W


def ack_for_onoff(target: str) -> str:
    """The short confirmation L2 speaks only after a verified action."""
    return "Включаю." if target == "on" else "Выключаю."


# Deliberately NOT starting with any of ha_match.ERROR_PREFIXES: an unproven
# read is not a failed tool call, and the honesty veto must not turn it into a
# refusal of whatever the model says next.
UNPROVEN_ONOFF_TEXT = (
    "Подтверждение пока не пришло. Не называй устройство неисправным: "
    "скажи, что состояние ещё не подтвердилось."
)
