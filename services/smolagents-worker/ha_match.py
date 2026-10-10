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
    # Media transport (03.10.2026). «коди» is what the user actually says about
    # the four Kodi boxes (media_player.le_vlada / le_zal_2 / le_spalnya /
    # le_kitchen); without an entry the device word matched nothing in the
    # latin registry and the turn escalated with «не нашла» — field case
    # «пауза коди во владиной комнате». «le» is the shared prefix of every
    # friendly name («LE-vlada», «LE-zal»): it is only ever matched INSIDE
    # media_player pools (resolve_media_targets), where it is precise.
    "коди": ["kodi", "le"],
    "kodi": ["kodi", "le"],
    "медиаплеер": ["media_player", "kodi", "le"],
    "плеер": ["media_player", "speaker", "kodi", "le"],
    "колонк": ["media_player", "speaker", "le"],
}

# Russian room stem -> latin fragments (the same bilingual problem, for rooms).
AREAS: dict[str, list[str]] = {
    "кухн": ["kitchen"],
    "гостин": ["living"],
    "зал": ["living"],
    "спальн": ["bedroom"],
    "коридор": ["corridor", "sorridor"],  # the registry id is misspelled «sorridor»
    "прихож": ["hallway", "entrance"],
    "вход": ["entrance"],  # HA area Entrance has the alias «вход»
    "ванн": ["bathroom"],
    "туалет": ["toilet", "wc"],
    "детск": ["kids", "children"],
    "кабинет": ["office"],
    "балкон": ["balcony"],
    "улиц": ["street", "yard"],
    "двор": ["yard"],
    "подвал": ["basement"],
    # EN display names, mirrored from jev-router/ha_client._AREA_LAT: the
    # HA area registry speaks English («Entrance», «Corridor»), and L2 gets
    # the area argument either way — «прихожая» from STT, «Entrance» from a
    # tool result. Without these keys area_matchers('Entrance') is EMPTY and
    # a room filter silently switches itself off (field case 02.10.2026).
    "living": ["living"],
    "kitchen": ["kitchen"],
    "bedroom": ["bedroom"],
    "corridor": ["corridor", "sorridor"],
    "entrance": ["entrance", "hallway"],
    "bathroom": ["bathroom"],
    "wc": ["wc", "toilet"],
    "balcony": ["balcony"],
    "street": ["street"],
    "yard": ["yard"],
    "basement": ["basement"],
    "office": ["office"],
    "kids": ["kids", "children"],
    # «во владиной комнате» -> the Kodi's HA area «Bedroom Vlada» (03.10.2026).
    # Without it area_matchers() returned nothing for the room and the player
    # was found only by luck (the model picked le_vlada out of a list of eight).
    # The EN key matters just as much: it is what makes area_matchers of the
    # registry side («Bedroom Vlada») intersect with the spoken side.
    "влади": ["vlada"],
    "владин": ["vlada"],
    "vlada": ["vlada"],
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


def area_matchers(text: str) -> set[str]:
    """Room stems (RU and EN) mentioned anywhere in `text`.

    The ROOM half of the bilingual problem, kept separate from `matchers()`:
    ranking needs to know which words say "which room" («прихожая», «вход»,
    «Entrance») so a room hit can outrank a domain hint — «свет в прихожей»
    must surface the relay sitting IN Entrance rather than the ten light.*
    status LEDs that merely contain the word "light".

    Stems are returned instead of the spoken tokens: a stem is always a
    substring of the token it matched («прихож» of «прихожей»), so it hits
    latin ids («entrance_…») and russian friendly names alike.

    A LATIN token is matched FORWARD only. The reverse test (`tok in stem`)
    exists for Russian inflection, but on latin input it manufactures rooms:
    «Living Room» tokenised to «living» + «room», and «room» is a substring of
    the stems «bedroom» and «bathroom», so the area became
    {living, bedroom, bathroom} — the exact-room rule then picked the phantom
    «Bedroom» and `media_play(area="Living Room")` started the show on the
    BEDROOM box (field check 03.10.2026 19:32). Every latin room name in this
    house is a key of its own («bedroom», «living», «kitchen»), so the forward
    test already covers it.
    """
    out: set[str] = set()
    for tok in (text or "").lower().split():
        if len(tok) < 3:
            continue
        latin = all(ord(c) < 128 for c in tok)
        for stem, lats in AREAS.items():
            if len(stem) < 3 and stem != tok:
                continue
            hit = stem in tok or (not latin and tok in stem)
            if hit:
                out.add(stem)
                out.update(lats)
    return out


# Helper/consumer domains: the integration builds an army of them around one
# device (update/button/select/number/text/…), and they match the same words
# the device does. A voice answer must rank the DEVICE above that army —
# same principle as the domain rule below, one level wider.
_HELPER_DOMAINS = frozenset({
    "automation", "script", "scene", "update", "button", "camera", "image",
    "text", "event", "calendar", "todo", "counter", "timer", "select",
    "number", "device_tracker", "person", "zone", "proximity", "stt", "tts",
    "notify", "conversation", "input_boolean", "input_button",
    "input_number", "input_select", "input_text",
})

# Facet words: sub-entities of ONE device (its status LED, network LED,
# config/index buttons, child locks). Word-TOKEN match, not substring: the
# household lamp is «WLED» and a substring test would demote it for holding
# «led» inside its name.
_FACET_WORDS = frozenset({
    "led", "indicator", "status", "network", "config", "child", "index",
    "reset",
})


# Readout domains: what a device REPORTS about itself. A press-action sensor
# and the relay it belongs to both contain «light» and «entrance», and the
# sensor sits earlier in the registry — the thing the user switches must win.
_READOUT_DOMAINS = frozenset({"sensor", "binary_sensor"})


def is_facet(entity_id: str, name: str = "") -> bool:
    """True for a device's auxiliary entity, False for the device itself."""
    toks = re.split(r"[^a-z0-9]+", f"{entity_id} {name}".lower())
    return any(t in _FACET_WORDS for t in toks)



def query_domains(query: str, states: list | None = None) -> set[str]:
    """Registry domains the query NAMES outright: {media_player} for «media_player».

    A domain word is a HARD filter, not a hint. `matchers()` ORs every token,
    so «media_player в гостиной» was answered with the living room's light
    relay plus six camera switches and NOT ONE player (field case 03.10.2026
    18:21, «Ну так найди её и включи»): the room token matched, the domain token
    was ignored, and the model planned its next two steps on a list of
    switches. A token only counts when it really is a domain of an entity in
    this registry — a RU device word («свет») never is.
    """
    if not states:
        return set()
    known = {str(e.get("entity_id", "")).split(".", 1)[0] for e in states
             if isinstance(e, dict)}
    toks = [t.strip(".,!?;:()«»\"'") for t in (query or "").lower().split()]
    out = set()
    for i, tok in enumerate(toks):
        # Underscores are part of the name («media_player»), so they are NOT
        # split away; the spoken form «media player» is recovered from the pair.
        for cand in (tok, "_".join(toks[i:i + 2]) if i + 1 < len(toks) else ""):
            if len(cand) >= 3 and cand in known:
                out.add(cand)
    return out


def match_states(
    states: list,
    query: str,
    area: str = "",
    area_map: dict | None = None,
) -> list[str]:
    """`«entity_id: state (name)»` lines for every registry hit, best first.

    Ranking is a five-key sort applied BEFORE the 10-line cap:

      1. the ROOM the user named \u2014 a hit inside \u00ab\u043f\u0440\u0438\u0445\u043e\u0436\u0435\u0439\u00bb beats entities that
         merely match the device word. Field case 02.10.2026 05:45 (\u00ab\u0441\u0432\u0435\u0442 \u0432
         \u043f\u0440\u0438\u0445\u043e\u0436\u0435\u0439\u00bb): ten `light.*` status LEDs matched \u00ablight\u00bb, ranked
         first (their domain IS the hint) and filled the whole cap, while the
         real lamp \u2014 `switch.entrance_light_switch_relay`, which matches BOTH
         \u00ablight\u00bb and \u00abentrance\u00bb \u2014 was never shown to the model;
      2. how many distinct matchers agree on the entity (a room+device
         combination beats a single generic word);
      3. the device above its facets (status/network LED of the same unit);
      4. the device above its helper army (update/button/select/\u2026);
      5. the thing you SWITCH above the thing that only REPORTS on it
         («\u0441\u0432\u0435\u0442 \u0432 \u043f\u0440\u0438\u0445\u043e\u0436\u0435\u0439\u00bb: the relay above its press-action sensor, and
         the thermometer's own sensor above its `number.*` settings);
      6. the entity's own domain being one of the matchers \u2014 the original
         rule: for \u00ab\u043f\u044b\u043b\u0435\u0441\u043e\u0441\u00bb it puts `vacuum.valetudo_\u2026` above
         `update.vacuum_card_update`, which stops the model reading state off
         a card-update helper entity.

    Keys are applied in that order, the sort is stable, so equal keys keep
    the registry order. The sort runs BEFORE the cap, never after.
    Observed 29.09.2026 14:17: ten helper entities (update/button/camera/
    number/select/sensor) matched the same hint and sat earlier in the
    registry, so the cap cut `vacuum.valetudo_\u2026: docked` out of the payload
    \u2014 the model was shown a pile of consumable-reset buttons and never the
    device itself, which is half of why it invented \u00ab\u0440\u043e\u0431\u043e\u0442 \u0443\u0431\u0438\u0440\u0430\u0435\u0442, \u0437\u0430\u0440\u044f\u0434 65 %\u00bb.

    After the sort the device's own readings are promoted under it
    (_promote_device_readings): see the comment there for the battery case
    of 29.09.2026 14:58 («узнать уровень заряда не удалось» while HA
    reported 97 %).
    """
    m = matchers(query, area)
    if not m:
        return []
    room = area_matchers(f"{query} {area}")
    # A domain named in the query is a hard filter (see query_domains): it must
    # not be diluted by the room token, or «media_player в гостиной» answers
    # with light relays and camera switches.
    hard = query_domains(query, states if isinstance(states, list) else [])
    if hard:
        # The domain token has done its job as a filter; leaving it in the OR
        # set would re-admit every entity of that domain in the house
        # («light» + «гостиной» returned lights from all six rooms).
        m = [x for x in m if x not in hard]
    if not m and not hard:
        return []
    scored: list[tuple[tuple, str, str]] = []
    for e in states if isinstance(states, list) else []:
        eid = e.get("entity_id", "")
        name = e.get("attributes", {}).get("friendly_name") or ""
        hay = f"{eid} {name}".lower()
        dom = eid.split(".", 1)[0]
        if hard and dom not in hard:
            continue
        # A room the NAMES cannot answer for: `media_player.le_zal_2` says
        # nothing about «гостиной», so a domain+room query found nothing at
        # all (03.10.2026 18:21 — the «найди её и включи» turn). The registry
        # area map is the second, authoritative source: an OR with the name
        # test, never a narrowing of it.
        by_area = False
        if room and area_map is not None and not any(w in hay for w in room):
            by_area = bool(room & area_matchers(area_map.get(eid, "")))
        if m and not (any(x in hay for x in m) or by_area):
            continue
        key = (
            0 if by_area or any(w in hay for w in room) else 1,
            -sum(1 for x in m if x in hay),
            1 if is_facet(eid, name) else 0,
            1 if dom in _HELPER_DOMAINS else 0,
            1 if dom in _READOUT_DOMAINS else 0,
            0 if dom in m else 1,
        )
        scored.append((key, eid, f"{eid}: {e.get('state')} ({name})", hay))
    # «свет в первом коридоре»: narrow to the numbered instance, but only
    # when something actually carries the digit AND (if a room was named)
    # sits in that room — a stray '1' in some unrelated id must not empty
    # the payload, and «свет в первом» must not drag in `light.wled1` from
    # another room. Applied as a filter AFTER the ranking, so the relay still
    # leads whatever remains.
    digit = ordinal_digit(f"{query} {area}")
    if digit and scored:
        room_hit = lambda h: (not room) or any(w in h for w in room)
        narrowed = [t for t in scored if digit in t[3] and room_hit(t[3])]
        if narrowed:
            scored = narrowed
    scored.sort(key=lambda t: t[0])
    hits = [(eid, line) for _, eid, line, _ in scored]
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


# --- ha_action's target resolver (field case 02.10.2026 05:45) ---------------
#
# «Значит, прихожий.» (05:45:10) escalated to L2 and the model sent a BLIND
# `intent__HassTurnOff {"area": "прихожая", "domain": ["light"]}`: HA answered
# MatchFailedReason.AREA (states=[]) for «прихожая» AND for «Entrance», and
# the turn was spoken as «я не нашёл устройств в прихожей». Two reasons:
#   * there is no `light.*` in area Entrance at all — the hallway lamp is
#     `switch.entrance_light_switch_relay` (Zigbee relay), so the domain
#     filter ["light"] can never match it;
#   * until 02.10.2026 that relay was `conversation.should_expose: false`,
#     which makes the matcher answer MatchFailedReason.ASSISTANT instead.
# jev-router's L1 resolves the concrete entity from raw /api/states BEFORE
# calling the intent (find_action_targets); this is the same rule for L2,
# kept pure so tests/test_ha_match.py can exercise it without smolagents.

def _norm(s: str) -> str:
    """'Bedroom Vlada' / 'bedroom_vlada' -> 'bedroom vlada' (area compares)."""
    return re.sub(r"[\s_]+", " ", (s or "").strip().lower())


# Deliberately NOT every switch: the household also holds camera_* switches —
# mirroring jev-router's _ONOFF_DOMAINS, a broad domain filter would turn off
# security recording.
ONOFF_DOMAINS = frozenset({"light", "switch"})

# Upper bound of devices one utterance may toggle — same number as
# jev-router's _MAX_ONOFF_TARGETS: «выключи весь свет» is a fan-out, not an
# unbounded series of MCP calls.
MAX_ONOFF_TARGETS = 5

# Spoken ordinal -> the digit it selects inside an entity id: «свет в первом
# коридоре» must reach `corridor1_light_switch_relay`, «во втором» the
# corridor2 one. Duplicate of jev-router/ha_client.py (two images ship two
# copies) — tests/test_hint_sync.py fails if they drift, same contract as
# HINTS. The ENDINGS are the guard against false friends: «вторник» starts
# with «втор» but ends in «ник», so Tuesday never selects digit 2.
ORDINAL_STEMS: dict[str, str] = {"перв": "1", "втор": "2", "трет": "3"}

ORDINAL_ENDINGS: tuple[str, ...] = (
    "ый", "ий", "ой", "ом", "ого", "ому", "ую", "ые", "ых", "ем",
    "ья", "ье", "ей", "яя", "ее",
)


def ordinal_digit(text: str) -> str:
    """'1'..'3' when `text` names a NUMBERED instance of a device, else ''.

    Whole-token match only; a literal 1-3 is honoured too («коридор 1»).
    """
    for raw in (text or "").lower().split():
        tok = raw.strip(".,!?;:()«»\"–-")
        if tok in ("1", "2", "3"):
            return tok
        if len(tok) < 3 or not tok.endswith(ORDINAL_ENDINGS):
            continue
        for stem, digit in ORDINAL_STEMS.items():
            if tok.startswith(stem):
                return digit
    return ""


def resolve_onoff_targets(
    states,
    name: str = "",
    area: str = "",
    domains=None,
    area_map: dict | None = None,
) -> list[dict]:
    """The on/off entities a failed intent call was really aiming at.

    Empty list unless the guess is safe — every branch below keeps the
    ORIGINAL HA error, because an ambiguity must stay an honest refusal and
    an `unavailable` device must escalate instead of being reported as done:

      * no device word at all («area» only): there is no mismatch to fix and
        nothing that says WHICH thing in the room was meant;
      * several targets with NO room in the call: mirroring jev-router's
        `ambiguous_no_area`, picking one room's device would be a guess;
      * more than MAX_ONOFF_TARGETS (a fan-out is not one command);
      * `unavailable`/`unknown` states are filtered out up front.

    Facets are dropped first (the network LED next to the relay belongs to
    the same device and matches every word the relay does); if nothing but
    facets is left they are returned as-is for the caller to refuse.
    """
    if domains is None:
        domains = []
    elif isinstance(domains, str):
        domains = [domains]
    doms = {str(d).lower() for d in domains}

    room_words = area_matchers(f"{name} {area}")
    want_area = area_matchers(area)

    # Device words: the spoken name plus its HINTS expansion, plus whatever
    # domain the model asked for (["light"] has to accept a `switch.*_relay`
    # whose id contains the word light — that mismatch is the whole point).
    dev: set[str] = set()
    name_toks: list[str] = []
    for tok in (name or "").lower().split():
        if len(tok) < 3:
            continue
        name_toks.append(tok)
        dev.add(tok)
        for stem, lats in HINTS.items():
            if len(stem) < 3 and stem != tok:
                continue
            if stem in tok or tok in stem:
                dev.update(lats)
    dev.update(doms)
    if not dev:
        return []
    # A latin/id fragment in `name` — «corridor1_light_switch Relay», the
    # friendly name echoed back from GetLiveContext — is a DESCRIPTOR of one
    # entity, not a hint: every such token must be present, otherwise the bare
    # «…_relay» would admit corridor2's relay as well and the pool would end
    # up refused as ambiguous. Spoken RU words keep the loose rule below («свет»
    # has to match a `switch.*_relay` whose id merely CONTAINS «light» — that
    # mismatch is the whole point of the resolver).
    lit = [t for t in name_toks
           if any(c.isascii() and c.isalnum() for c in t)]

    # 2 = the entity's area IS the requested one, 1 = its name merely
    # CONTAINS it («спальня» -> {bedroom} also matches «Bedroom Vlada»).
    # An exact room wins outright: «свет в спальне» must not switch off
    # Vlad's garland in the next room.
    want_norm = {_norm(x) for x in want_area} | ({_norm(area)} if area else set())
    if not want_area and not area:
        # No `area` slot at all: the room arrived inside `name` («свет в
        # спальне» handed over as one string). Give it the same exact-match
        # treatment an `area` gets, or the exact-room rule below is dead on
        # every name-only call.
        want_norm |= {_norm(x) for x in room_words}
    # When the registry holds an area named EXACTLY like the request, only
    # entities in it count — otherwise «спальня» (-> {bedroom}) would happily
    # accept «Bedroom Vlada» simply because its name contains the room.
    # A household with only «Living Room» for «гостинная» has no exact
    # option, so a name that CONTAINS the room stays acceptable there.
    exact_exists = bool(area_map) and any(
        _norm(d) in want_norm for d in (area_map or {}).values()
    )

    out: list[tuple[dict, int]] = []
    for e in states if isinstance(states, list) else []:
        eid = e.get("entity_id", "")
        dom = eid.split(".", 1)[0]
        if dom not in ONOFF_DOMAINS:
            continue
        if str(e.get("state", "")).lower() not in ("on", "off"):
            continue
        fname = (e.get("attributes") or {}).get("friendly_name") or ""
        hay = f"{eid} {fname}".lower()
        score = 2
        if room_words:
            if area_map is not None:
                # Registry area is authoritative — the rule jev-router's
                # find_action_targets already documents. An entity with NO
                # area must not be admitted by an id substring: a name-only
                # call («первый коридор» sitting in the `name` slot) used to
                # pull in `switch.corridor1_detect` (area None, id says
                # corridor), the pool crossed MAX_ONOFF_TARGETS and the relay
                # the user named was never returned. `want_area` is the `area`
                # slot; with only a name, the room words found in the name
                # filter exactly the same way.
                reg_area = want_area or room_words
                display = area_map.get(eid, "")
                if not (reg_area & area_matchers(display)):
                    continue
                score = 2 if _norm(display) in want_norm else 1
            elif not any(w in hay for w in room_words):
                continue
        if lit:
            if not all(x in hay for x in lit):
                continue
        elif not (any(d in hay for d in dev) or dom in doms):
            continue
        out.append((e, score))

    if not out:
        return []
    # «первый коридор»: only the numbered instance may move. Nothing in the
    # pool carrying the digit == the numbered device does not exist -> [] ,
    # which keeps the caller's ORIGINAL error; toggling BOTH corridor relays
    # for a request that named one of them would be a guessed side effect.
    digit = ordinal_digit(f"{name} {area}")
    if digit:
        out = [
            t for t in out
            if digit in f"{t[0].get('entity_id', '')} "
            f"{(t[0].get('attributes') or {}).get('friendly_name') or ''}".lower()
        ]
        if not out:
            return []
    if exact_exists:
        # An exactly-named room exists in the registry and this entity is
        # not in it: drop it (score 2 == its area name IS the request).
        out = [e for e, sc in out if sc == 2]
    else:
        out = [e for e, _ in out]
    clean = [e for e in out if not is_facet(
        e.get("entity_id", ""),
        (e.get("attributes") or {}).get("friendly_name") or "",
    )]
    pool = clean or out
    if len(pool) > 1 and not (area or "").strip():
        return []  # several devices, no room: acting would be a guess
    if len(pool) > MAX_ONOFF_TARGETS:
        return []
    return pool


# --- Media transport (field case 03.10.2026, «поставь его на паузу») ----------
# The whole turn was lost because HA's MCP server (tools/list, verified live)
# has NO media intent at all: the model invented intent__HassMediaPause,
# got «Tool … not found» twice and reported «не удалось». The services DO
# exist in HA (`media_player.media_pause`), they are simply unreachable over
# MCP — so media_control() calls the REST service endpoint directly, and this
# module holds the two pure halves of that decision: the action table and the
# player lookup. Kept out of tools.py because tests must reach it without
# smolagents (same contract as resolve_onoff_targets).

MEDIA_DOMAIN = "media_player"

# Canonical action -> HA service. The service names ARE the keys so the two
# services cannot drift: jev-router's resolver emits the same names into
# ResolvedCall.tool ("media__" + name) and tests/test_hint_sync.py checks the
# router's table against this one.
MEDIA_SERVICES: dict[str, str] = {
    "media_pause": "media_pause",
    "media_play": "media_play",
    "media_play_pause": "media_play_pause",
    "media_stop": "media_stop",
    "media_next_track": "media_next_track",
    "media_previous_track": "media_previous_track",
    "volume_up": "volume_up",
    "volume_down": "volume_down",
    "volume_set": "volume_set",
    "volume_mute": "volume_mute",
}

# Spoken/short spellings the model reaches for -> canonical action. Russian is
# here too: the model often hands over the user's own verb («на паузу») instead
# of translating it, and refusing that costs the turn.
MEDIA_ALIASES: dict[str, str] = {
    "pause": "media_pause", "пауза": "media_pause", "паузу": "media_pause",
    "приостанови": "media_pause", "stop_pause": "media_pause",
    "play": "media_play", "продолжи": "media_play", "возобнови": "media_play",
    "играй": "media_play", "воспроизведи": "media_play", "unpause": "media_play",
    "toggle": "media_play_pause", "play_pause": "media_play_pause",
    "stop": "media_stop", "останови": "media_stop", "стоп": "media_stop",
    "next": "media_next_track", "следующий": "media_next_track",
    "next_track": "media_next_track", "дальше": "media_next_track",
    "previous": "media_previous_track", "предыдущий": "media_previous_track",
    "prev": "media_previous_track", "назад": "media_previous_track",
    "volume_up": "volume_up", "громче": "volume_up",
    "volume_down": "volume_down", "тише": "volume_down",
    "volume_set": "volume_set", "громкость": "volume_set",
    "mute": "volume_mute", "заглуши": "volume_mute",
    "выключи_звук": "volume_mute", "unmute": "volume_mute",
    "разглуши": "volume_mute", "включи_звук": "volume_mute",
}

# States a media_player can be addressed in. `unavailable`/`unknown` are the
# same signal as for on/off: the box is offline, a command to it cannot be
# confirmed — escalate instead of claiming.
_MEDIA_DEAD = ("unavailable", "unknown", "none", "")

# Which state makes a player the likely target of a given request, in the
# order they are tried. «поставь ЕГО на паузу» names its target by what the
# box is DOING — and when the user repeats the request the one already paused
# is the one meant, so `paused` stays in the chain (live check 03.10.2026:
# le_vlada was paused and a playing-only rule escalated a command whose
# answer is «уже на паузе»). Same table as jev-router/ha_client.MEDIA_PREFER.
MEDIA_PREFER: dict[str, tuple[str, ...]] = {
    "media_pause": ("playing", "paused", "buffering"),
    "media_play": ("paused", "playing", "buffering"),
    "media_play_pause": ("playing", "paused"),
    "media_stop": ("playing", "paused"),
    "media_next_track": ("playing", "paused"),
    "media_previous_track": ("playing", "paused"),
    "volume_up": ("playing", "buffering", "paused"),
    "volume_down": ("playing", "buffering", "paused"),
    "volume_set": ("playing", "buffering", "paused"),
    "volume_mute": ("playing", "buffering", "paused"),
}

def media_fingerprint(state: dict) -> tuple:
    """Everything about a player that a transport/volume call can move.

    Comparing `state` ALONE is not proof of «nothing happened»: `volume_set`
    only changes the `volume_level` ATTRIBUTE, and `media_next_track` changes
    `media_title` while the box stays `paused`. Recorded as ok=True on a
    changed-nothing call, the veto stayed silent and the model said
    «остановила» / «переключил трек» for a command that moved nothing (field
    check 03.10.2026). `media_position` is deliberately absent — it ticks on
    its own and would make every call look like a change.
    """
    a = (state or {}).get("attributes") or {}
    return (
        str((state or {}).get("state", "")).lower(),
        a.get("volume_level"),
        a.get("is_volume_muted"),
        a.get("media_title"),
        a.get("media_content_id"),
        a.get("source"),
    )


# Volume is a transport service for the USER and not for the fingerprint: a
# volume change never reaches HA's `changed` list and lands late on an idle
# Kodi, so it gets a longer re-read window. Same tables as
# jev-router/ha_client (tests/test_hint_sync.py holds the two together) — the
# worker and the fast path must not disagree about how long to wait, or the
# first level calls a working volume change a failure and the second a success.
MEDIA_VOLUME = frozenset({
    "volume_up", "volume_down", "volume_set", "volume_mute",
})

MEDIA_CONFIRM_DELAYS: dict[str, tuple[float, ...]] = {
    "default": (0.0, 0.6, 1.5),                 # ~2.1 s
    "volume": (0.0, 0.8, 1.7, 2.8, 4.0, 5.4),   # ~6.8 s
}


def confirm_delays(service: str) -> tuple[float, ...]:
    """The re-read schedule for `service` — see MEDIA_CONFIRM_DELAYS."""
    return MEDIA_CONFIRM_DELAYS[
        "volume" if service in MEDIA_VOLUME else "default"]


def volume_unverifiable(state: dict) -> bool:
    """The box reports no volume_level at all: there is no level to move, so
    no amount of polling can confirm a volume call (verified live: le_kitchen /
    le_vlada report None while idle, le_spalnya reports a real level)."""
    return ((state or {}).get("attributes") or {}).get("volume_level") is None


# Protocol endpoints are not boxes a person talks to: `media_player.*_airplay`
# and `*_dlna` are receivers the X96Q/Android TV exposes, and they sit in the
# same room as the real player — counting them made «громкость в гостиной»
# ambiguous (le_zal_2 + the AirPlay endpoint, field check 03.10.2026). They
# are dropped when a real player survives, exactly like resolve_onoff_targets
# does with device facets.
_MEDIA_ENDPOINT_WORDS = frozenset({"airplay", "dlna", "chromecast"})

# Final ordering of the survivors (deterministic tie-break by entity_id).
_MEDIA_ORDER = {"playing": 0, "buffering": 1, "paused": 2, "idle": 3, "off": 4}


def normalize_media_action(action: str) -> str:
    """«Пауза» / «next» / «MEDIA_PAUSE» -> «media_pause», "" when unknown.

    A two-word utterance («следующий трек», «громкость 40») is also tried by
    its first word: the model regularly hands over the user's own phrase, and
    refusing a recognisable request only loses the turn.
    """
    a = (action or "").strip().lower().replace(" ", "_").strip(".,!?:")
    if not a:
        return ""
    if a in MEDIA_SERVICES:
        return a
    if a in MEDIA_ALIASES:
        return MEDIA_ALIASES[a]
    head = a.split("_", 1)[0]
    return MEDIA_ALIASES.get(head, "")


def media_service_data(action: str, value: str = "") -> dict:
    """Extra service fields for `action` from a spoken `value` («30 процентов»).

    {} for every action that takes no parameter. `volume_set` wants a 0..1
    level, `volume_mute` a boolean — both parsed from the same free-text
    `value`, and {} (not a guess) when it says nothing usable.
    """
    if action == "volume_set":
        m = re.search(r"(\d{1,3})", value or "")
        if m:
            return {"volume_level": round(min(100, int(m.group(1))) / 100.0, 2)}
        return {}
    if action == "volume_mute":
        v = (value or "").strip().lower()
        if re.search(r"выключ|заглуш|отключ|muted?\b|off", v):
            return {"is_volume_muted": True}
        if re.search(r"включ|разглуш|unmut|on", v):
            return {"is_volume_muted": False}
        return {}
    return {}


def resolve_media_targets(
    states,
    name: str = "",
    area: str = "",
    area_map: dict | None = None,
    service: str = "",
) -> list[dict]:
    """The media_players a transport command may address, most likely first.

    Empty list unless the choice is safe — an ambiguity must stay an honest
    refusal (the caller reports it), never a guessed side effect:

      * `unavailable` players are dropped up front (an offline box cannot
        confirm anything);
      * a NAMED device that matches nothing falls back to the room alone
        («коди» -> le_vlada needs exactly this), but a name that matched
        nothing AND no room leaves nothing to go on;
      * with no device word at all the pool is narrowed by `service`'s
        preferred states — that is what «поставь ЕГО на паузу» means;
      * a room named exactly in the registry wins over a room merely
        CONTAINED in an entity's area (same rule as resolve_onoff_targets);
      * a tie inside the preferred state is a real ambiguity -> [].
    """
    if not isinstance(states, list):
        return []
    pool: list[dict] = []
    for e in states:
        eid = str(e.get("entity_id", ""))
        if eid.split(".", 1)[0] != MEDIA_DOMAIN:
            continue
        if str(e.get("state", "")).lower() in _MEDIA_DEAD:
            continue
        pool.append(e)
    if not pool:
        return []
    real = [
        e for e in pool
        if not any(w in _MEDIA_ENDPOINT_WORDS
                   for w in re.split(r"[^a-z0-9]+", str(e.get("entity_id", "")).lower()))
    ]
    pool = real or pool

    name_toks = [t for t in re.split(r"[^\wЀ-ӿ]+", (name or "").lower())
                 if len(t) >= 3]
    dev: set[str] = set()
    for tok in name_toks:
        dev.add(tok)
        for stem, lats in HINTS.items():
            if stem in tok or tok in stem:
                dev.update(lats)

    def _hay(e: dict) -> str:
        return f"{e.get('entity_id', '')} " \
               f"{(e.get('attributes') or {}).get('friendly_name') or ''}".lower()

    by_name = [e for e in pool if dev and any(d in _hay(e) for d in dev)]

    room_words = area_matchers(f"{name} {area}")
    want_area = area_matchers(area)
    by_area: list[dict] = []
    for e in pool:
        if not room_words and not want_area:
            break
        eid = str(e.get("entity_id", ""))
        if area_map is not None:
            display = _norm(area_map.get(eid, ""))
            if display and not (room_words | want_area) & area_matchers(display):
                continue
            if not display and not any(w in _hay(e) for w in room_words):
                continue
        elif not any(w in _hay(e) for w in room_words | want_area):
            continue
        by_area.append(e)
    if (room_words or want_area) and not by_area:
        # The room the user named holds no player at all: refusing beats
        # answering for a box in another room («телевизор в коридоре» must
        # not reach le_vlada just because «телевизор» matched something).
        return []

    # Name AND room are BOTH constraints, so they intersect. «коди» expands
    # to «le», which every one of the four boxes carries — letting the name
    # win on its own made «коди в гостиной» answer for le_vlada (the paused
    # one) instead of le_zal_2. When the two cannot meet, the room wins: the
    # user named it, and it holds exactly one player.
    if by_name and by_area:
        cand = [e for e in by_name if e in by_area] or by_area
    elif by_name:
        cand = by_name
    elif by_area:
        cand = by_area
    else:
        # Nothing named at all — a pronoun («поставь ЕГО на паузу») — may look
        # at the whole pool, and the preferred states decide.
        cand = pool
    if not cand:
        return []
    if not cand:
        return []

    # Two rules, in this order, and BOTH are needed in this house:
    #  1. HOW MANY named room words the player's area carries. «Bedroom Vlada»
    #     is named twice over («Bedroom» + «Vlada») and «Bedroom» once, so the
    #     two-player room wins — this is the "«спальня» must not reach Bedroom
    #     Vlada" rule of find_action_targets, applied to the room the USER
    #     named rather than to a substring.
    #  2. An EXACTLY named room (the registry name equals a named word) beats a
    #     merely containing one. Without it «спальне» ties 1-to-1 with
    #     «Bedroom Vlada» and the wrong box could win.
    # Equality is against the fragment set, not the raw `area` string, so a RU
    # word works — safe only because area_matchers no longer invents phantom
    # rooms from latin substrings («Living Room» used to expand to
    # {living, bedroom, bathroom} and the phantom «Bedroom» won, 03.10.2026).
    want_norm = {_norm(x) for x in (room_words | want_area)}
    if by_area and area_map is not None:
        counts = [(e, len(room_words & area_matchers(
            area_map.get(str(e.get("entity_id", "")), "")))) for e in by_area]
        top = max((c for _, c in counts), default=0)
        if top:
            by_area = [e for e, c in counts if c == top]
    if want_norm and by_area:
        exact = [e for e in by_area
                 if _norm((area_map or {}).get(str(e.get("entity_id", "")), ""))
                 in want_norm]
        if exact:
            by_area = exact
    cand = [e for e in cand if e in by_area] if by_area else cand

    if len(cand) > 1:
        for want in MEDIA_PREFER.get(service, ("playing", "buffering")):
            narrowed = [e for e in cand
                        if str(e.get("state", "")).lower() == want]
            if narrowed:
                cand = narrowed
                break
    if len(cand) > 1:
        return []
    if len(cand) > MAX_ONOFF_TARGETS:
        return []
    return sorted(
        cand,
        key=lambda e: (_MEDIA_ORDER.get(str(e.get("state", "")).lower(), 9),
                       str(e.get("entity_id", ""))),
    )


def claimed_a_usable_entity(text: str, states: list[dict] | None) -> bool:
    """True only if an intent answer named at least one real, available entity.

    The same rule as the router's `_touched_a_usable_entity`, kept as a second
    copy rather than imported: the two services are deployed as separate images
    (the same reason the media tables are duplicated and held together by
    `tests/test_hint_sync.py`).

    It lives HERE, not in tools.py, because tools.py imports `smolagents` at
    module level and the gateway's test image does not ship it — a function that
    cannot be imported by the suite is a function nobody tests, and the thing it
    guards is a spoken lie.

    `{"type": "area"}` in HA's success list means the intent matched a ROOM, not
    a device — with no `entity` entry nothing was addressed at all. An entity that
    is `unavailable` cannot have changed either, and an entity ABSENT from the live
    registry is not evidence of availability: HA happily names entities filtered
    out of `/api/states` or living in an integration we cannot read, so absent
    means unverified.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Not a parseable intent payload: nothing was claimed, so nothing verified.
        return False
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if not isinstance(data, dict):
        return False
    claimed = data.get("success") or data.get("claimed") or []
    ids = {
        row.get("id")
        for row in claimed
        if isinstance(row, dict) and row.get("type") == "entity" and row.get("id")
    }
    if not ids:
        return False
    if not states:
        # No live registry to check against: refuse rather than assume.
        return False
    live = {
        str(e.get("entity_id")): str(e.get("state", "")).lower()
        for e in states
        if isinstance(e, dict)
    }
    return any(
        live.get(eid) is not None
        and live.get(eid) not in ("unavailable", "unknown", "none", "")
        for eid in ids
    )


def claimed_entity_ids(text: str) -> list[str]:
    """Entity ids HA named as done, from the payload shape
    `claimed_a_usable_entity` reads."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    claimed = data.get("success") or data.get("claimed") or []
    return [
        row.get("id")
        for row in claimed
        if isinstance(row, dict) and row.get("type") == "entity" and row.get("id")
    ]


def confirm_on_off_reached(entity_ids: list[str], target: str,
                           fetch_states) -> list[str]:
    """Which of these entities actually reached `target`, read back LATE.

    `fetch_states` is a zero-argument callable returning the raw HA state list; it
    is passed in rather than imported so this module stays pure stdlib and the
    suite can drive the whole thing without a network — `tools.py` cannot be
    imported at all in the gateway's test image because it needs `smolagents`.

    The delay is not a fudge. Home Assistant updates a `switch` entity
    optimistically on `turn_on`/`turn_off`, so an immediate read returns the value
    HA itself just wrote. Measured 10.10.2026 on `switch.coffemaker`: state `off`
    at t=+0.26 s while `sensor.coffemaker_power` still read 1072 W, and 0 W by
    t=+0.77 s. L1 uses the same schedule for the same reason, and the two must stay
    identical or they will disagree about the same turn.
    """
    import time as _time

    delays = (1.0, 1.5, 2.5)
    reached: list[str] = []
    prev = 0.0
    for delay in delays:
        _time.sleep(delay - prev)
        prev = delay
        live = {
            str(s.get("entity_id")): str(s.get("state", "")).lower()
            for s in (fetch_states() or [])
            if isinstance(s, dict)
        }
        for eid in entity_ids:
            if eid not in reached and live.get(eid) == target:
                reached.append(eid)
        if reached:
            break
    return reached
