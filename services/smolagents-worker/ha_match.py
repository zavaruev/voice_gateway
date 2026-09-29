"""RU <-> EN entity matching + "no data" detection for L2 (pure stdlib).

Two problems this module solves for `ha_read` (field case 29.09.2026,
«Что там с нашим пылесосом?»):

  1. THE MISS THAT NEVER FALLS THROUGH. The MCP `GetLiveContext` answer was
     treated as data whenever it was non-empty and free of the «Ошибка»
     prefix, so a perfectly valid *failure* payload —
     `{"success": false, "error": "No exposed entities matched name 'пылесос'"}`
     — was returned to the model verbatim and the REST fallback below it
     never ran. `has_data()` is the single definition of "this result
     carries usable data", shared by the REST fallback and by the event
     recorder that feeds the honesty veto.

  2. THE BILINGUAL GAP. The user speaks Russian («пылесос») while the HA
     registry holds latin ids and names (`vacuum.valetudo_…`,
     «Roborock Robot»), so a plain substring match finds nothing. The router
     keeps the same table in `jev-router/ha_client._HINT_LAT`; this is the
     worker's own copy because its image ships only `smolagents-worker/*.py`.
     An alias that matches nothing simply never fires — that is what makes
     extra entries free insurance.

No smolagents/FastAPI/network imports: tests/test_ha_match.py runs on the
host python as well as inside the CI image.
"""

from __future__ import annotations

import json
import re

# Transport/parse error prefixes returned by mcp_call/ha_action/ha_read.
# Byte-level contract with tools._record_action and honesty._truth — do not
# reword (an extra entry is fine, dropping one changes what counts as a
# failure). «Не удалось…» is ha_read's own REST-fallback wording.
ERROR_PREFIXES = (
    "Ошибка",
    "Тул вернул ошибку",
    "Некорректный",
    "Неожиданный",
    "Не удалось",
)

# Russian stems -> latin fragments that occur in entity_ids / friendly_names.
# Keys are stems (substring-matched, so «пылесосом» hits «пылесос») and the
# values are exact substrings of the latin registry.
HINTS: dict[str, list[str]] = {
    "пылесос": ["vacuum", "roborock", "robot", "valetudo"],
    "робот": ["vacuum", "roborock", "robot"],
    "чайник": ["kettle", "boiler"],
    "кофевар": ["coffee"],
    "кафевар": ["coffee"],  # STT hears «кафеварку» (dropped «о»)
    "кофемашин": ["coffee"],
    "свет": ["light"],
    "ламп": ["light"],
    "освещени": ["light"],
    "люстр": ["light"],
    "торшер": ["light"],
    "штор": ["cover", "curtain"],
    "жалюзи": ["blind"],
    "телевизор": ["tv", "television"],
    "розетк": ["switch", "socket"],
    "кондиционер": ["climate", "thermostat"],
    "климат": ["climate"],
    "температур": ["temperature", "temp"],
    "влажн": ["humidity"],
    "заряд": ["battery", "charge"],
    "батаре": ["battery"],
    "громкост": ["volume"],
    "яркост": ["brightness"],
    "музык": ["media_player", "speaker"],
}

# Russian room stem -> latin fragments (the same bilingual problem, for rooms).
AREAS: dict[str, list[str]] = {
    "кухн": ["kitchen"],
    "гостин": ["living"],
    "зал": ["living"],
    "спальн": ["bedroom"],
    "коридор": ["corridor"],
    "прихож": ["hallway", "entrance"],
    "ванн": ["bathroom"],
    "туалет": ["toilet", "wc"],
    "детск": ["kids", "children"],
    "кабинет": ["office"],
    "балкон": ["balcony"],
    "улиц": ["street", "yard"],
    "двор": ["yard"],
    "подвал": ["basement"],
}

# Result payloads that carry no data at all (besides ERROR_PREFIXES).
_RE_NO_DATA = re.compile(
    r"no\s+exposed\s+entities|not\s+found|не\s+найден\w*|ничего\s+не\s+найдено",
    re.IGNORECASE,
)


def has_data(result: str) -> bool:
    """True when a tool/MCP result carries usable data, False on a miss.

    Covers, in order: empty output, the four error prefixes (contract with
    `mcp_call`), a JSON body with `success: false` or an `error` key, and a
    plain-text miss («No exposed entities matched name …»). Anything else
    counts as data — deliberately in the same direction as
    tools._record_action: a wrongly-flagged miss only sends the caller to
    the REST fallback, a wrongly-flagged data would hide a real failure.
    """
    s = (result or "").strip()
    if not s:
        return False
    if any(p in s for p in ERROR_PREFIXES):
        return False
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        if data.get("success") is False or data.get("error"):
            return False
    if _RE_NO_DATA.search(s):
        return False
    return True


def matchers(query: str, area: str = "") -> list[str]:
    """Lowercased substrings to look for in `entity_id friendly_name`.

    Every spoken token is kept as-is and expanded through both tables, so
    «пылесос на кухне» looks for {пылесос, vacuum, roborock, robot, valetudo,
    кухн, kitchen}. Tokens shorter than 3 chars are not reverse-matched
    against the stems («с» is a substring of «свет» and would otherwise pull
    the lamp into every question that contains the preposition).
    """
    out: list[str] = []
    for tok in f"{query} {area}".lower().split():
        if len(tok) < 3:
            continue
        out.append(tok)
        for table in (HINTS, AREAS):
            for stem, lats in table.items():
                if stem in tok or (len(tok) >= 3 and tok in stem):
                    out.extend(lats)
    return list(dict.fromkeys(out))


def match_states(states: list, query: str, area: str = "") -> list[str]:
    """`«entity_id: state (name)»` lines for every registry hit, best first.

    Ranking is a stable sort by "the entity's own domain is one of the
    matchers": for «пылесос» that puts `vacuum.valetudo_…` above
    `update.vacuum_card_update` (both match on "vacuum", but only the first
    is a vacuum), which is what stops the model from reading state off a
    card-update helper entity.

    The sort runs BEFORE the 10-line cap, never after. Observed 29.09.2026
    14:17: ten helper entities (update/button/camera/number/select/sensor)
    matched the same hint and sat earlier in the registry, so the cap cut
    `vacuum.valetudo_…: docked` out of the payload — the model was shown a
    pile of consumable-reset buttons and never the device itself, which is
    half of why it invented «робот убирает, заряд 65 %».

    After the sort the device's own readings are promoted under it
    (_promote_device_readings): see the comment there for the battery case
    of 29.09.2026 14:58 («узнать уровень заряда не удалось» while HA
    reported 97 %).
    """
    m = matchers(query, area)
    if not m:
        return []
    hits: list[tuple[str, str]] = []
    for e in states if isinstance(states, list) else []:
        eid = e.get("entity_id", "")
        name = e.get("attributes", {}).get("friendly_name") or ""
        hay = f"{eid} {name}".lower()
        if any(x in hay for x in m):
            hits.append((eid, f"{eid}: {e.get('state')} ({name})"))
    hits.sort(key=lambda h: 0 if h[0].split(".", 1)[0] in m else 1)
    hits = _promote_device_readings(hits, m)
    # cap: the result goes into the model's context — applied AFTER both
    # orderings so neither the device nor its readings are what gets dropped.
    return [line for _, line in hits[:10]]


# A device's READINGS live in separate `sensor.*` entities that the
# integration creates AFTER its whole army of helpers: for «пылесос» the
# robot's battery is the ~19th match (4 reset buttons, the map camera,
# statistics and wi-fi sit in front of it), so the 10-line cap dropped it
# and the model answered «узнать уровень заряда не удалось» (29.09.2026
# 14:58) while HA reported 97 %. Latin fragments only — the entity ids are
# latin no matter which language the user spoke.
_READING_HINTS = ("battery", "charge", "temperature", "temp", "humidity")
_READING_LEAD = ("battery", "charge")  # asked next almost every time


def _promote_device_readings(
    hits: list[tuple[str, str]], m: list[str]
) -> list[tuple[str, str]]:
    """Put the top device's own `sensor.*` readings directly under it.

    Applies only when the first hit IS a device (its domain is one of the
    matchers): a metric-only query («заряд» with no device named) has no
    device to hang a reading off and keeps the registry order. Battery and
    charge readings sort ahead of the other readings, so the cap sees them
    first; everything the rule does not recognise keeps its position.
    """
    if not hits:
        return hits
    head_eid = hits[0][0]
    if head_eid.split(".", 1)[0] not in m:
        return hits
    device = head_eid.split(".", 1)[1]  # «valetudo_zealouseverlastinggaur»
    ride: list[tuple[str, str]] = []
    rest: list[tuple[str, str]] = []
    for eid, line in hits[1:]:
        bare = eid.split(".", 1)[1] if "." in eid else eid
        reads = eid.startswith("sensor.") and any(h in eid for h in _READING_HINTS)
        if reads and bare.startswith(device):
            ride.append((eid, line))
        else:
            rest.append((eid, line))
    if not ride:
        return hits
    ride.sort(key=lambda p: 0 if any(x in p[0] for x in _READING_LEAD) else 1)
    return [hits[0]] + ride + rest
