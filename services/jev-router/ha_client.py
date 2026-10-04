"""Home Assistant client: MCP JSON-RPC (streamableHttp, stateless) + REST.

PURPOSE
  The ONLY module in jev-router that talks to Home Assistant, and the
  boundary where the L1 fast path actually touches the house (see app.py):
  side effects and answers go through here, so caching, timeouts and the
  ok/error contract live in one place.

  Two protocols, both with the same Bearer auth (HA_TOKEN from env — never
  hardcode tokens; the repo is public):
    * MCP  POST /api/mcp         — jsonrpc `tools/call` / `tools/list` for
      intent execution and read tools (llm__GetDateTime, ...);
    * REST /api/states, /api/history/period/..., /api/template — registry
      snapshots, history and the area map, i.e. everything the MCP intent
      matcher does NOT give us.

  Failure contract: no public method raises. Every error path returns an
  empty list / None / {"ok": False, "error": ...} — app.py turns that into
  a `ha_call_failed:` escalation to L2 (self-healing retry with the decoded
  error) instead of a stack trace in the voice stream.

  Caches: /api/states and the entity->area map are TTL-cached
  (config.STATES_TTL) because every easy_query and every on/off command
  needs them; `force=True` re-reads states before a side effect so the
  router never acts on a stale "off".

Token/source: nanobot_config/config.json `home_assistant` section (Bearer JWT).
Verified Sep 2026: server is stateless — no initialize handshake required;
POST /api/mcp accepts jsonrpc `tools/list` and `tools/call` directly.
"""

import json
import logging
import re
import time

import aiohttp

import config

logger = logging.getLogger("router.ha")

# RU stems (as spoken by resolver.entity_hint / area) -> latin fragments that
# actually occur in HA entity_ids and friendly_names. Verified against the
# live registry: sensor.bedroom_thermometer_temperature | "… Temperature".
_HINT_LAT: dict[str, list[str]] = {
    "температур": ["temperature", "temp"],
    "градус": ["temperature", "temp"],
    "влажн": ["humidity"],
    "заряд": ["battery", "charge"],
    "давлен": ["pressure"],
    "чайник": ["kettle", "boiler"],
    "кофевар": ["coffee"],
    "кофемашин": ["coffee"],
    "свет": ["light"],
    "ламп": ["light"],
    "телевизор": ["tv", "television"],
    "пылесос": ["vacuum", "roborock", "robot"],
    # «робот» was missing here while THING already resolved it (29.09.2026,
    # 14:17): the resolver handed hint="робот" over, nothing in the latin
    # registry contains the RU word, and find_entity returned None —
    # «Не нашла такого устройства» for a vacuum that was right there. Every
    # key of the worker's ha_match.HINTS must exist in this table too (the
    # two copies live in different images, so tests/test_hint_sync.py reads
    # them out of the source and fails if they drift apart).
    "робот": ["vacuum", "roborock", "robot"],
    "батаре": ["battery"],
    "громкост": ["volume"],
    "яркост": ["brightness"],
    "жалюзи": ["blind"],
    "кафевар": ["coffee"],
    "климат": ["climate"],
    "кондиционер": ["climate", "thermostat"],
    "люстр": ["light"],
    "освещени": ["light"],
    "розетк": ["switch", "socket"],
    "торшер": ["light"],
    "штор": ["cover", "curtain", "blind"],
    "музык": ["media_player", "speaker", "receiver"],
    # Media transport (03.10.2026). «коди» is what the user calls the four
    # Kodi boxes (media_player.le_vlada / le_zal_2 / le_spalnya / le_kitchen)
    # and «ле» is the shared prefix of every one of their friendly names
    # («LE-vlada», «LE-zal»): hint «коди» finds nothing without it, which is
    # how «пауза коди во владиной комнате» found no device at all.
    "коди": ["kodi", "le"],
    "kodi": ["kodi", "le"],
    "медиаплеер": ["media_player", "kodi", "le"],
    "плеер": ["media_player", "speaker", "kodi", "le"],
    "колонк": ["media_player", "speaker", "le"],
}

# RU area stem -> latin fragments that occur in entity_ids/friendly_names.
# The same bilingual expansion problem as _HINT_LAT, but for rooms: the
# resolver speaks «кухня», the registry says `kitchen`. Entries are matched
# as substrings, so an alias that matches nothing simply never fires — that
# is what makes the typo-tolerant extras below free insurance.
_AREA_LAT: dict[str, list[str]] = {
    "кухн": ["kitchen"],
    "гостин": ["living"],
    "зал": ["living"],
    "спальн": ["bedroom"],
    "коридор": ["corridor"],
    "прихож": ["hallway", "corridor", "entrance"],
    "ванн": ["bathroom"],
    "туалет": ["toilet", "wc"],
    "детск": ["kids", "children", "child"],
    "кабинет": ["office"],
    "балкон": ["balcony"],
    "улиц": ["street", "yard", "outdoor"],
    "двор": ["yard"],
    "подвал": ["basement"],
    # EN canonical names (the resolver now sends area registry display names)
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
    # «во владиной комнате» -> HA area «Bedroom Vlada» (03.10.2026).
    "влади": ["vlada"],
    "владин": ["vlada"],
    "vlada": ["vlada"],
}




def _claimed(result: dict) -> dict:
    """Pull HA's own account of what it touched out of an intent response.

    HA's MCP intent answers `{"success": bool}` or, for action intents,
    `{"data": {"success": [...], "failed": [...]}}` where each entry is
    `{name, type, id}` and `type` is "entity" or "area". `ok` alone is NOT
    evidence that anything moved: an intent whose domain cannot match the real
    device still returns success for the AREA it matched plus whatever
    `unavailable` entities share the name — measured 04.10.2026 on
    «выключи свет» in the living room: `success: [area "Living Room",
    light.wled_living_room]`, `failed: []`, and the lamp (a `switch.*_relay`,
    invisible to `domain: ["light"]`) stayed on while the gateway said
    «Выключила». Callers must verify against live state before claiming a
    side effect.
    """
    data = result.get("data")
    if not isinstance(data, dict):
        return {"claimed": [], "claimed_failed": []}

    def _ids(key: str) -> list[dict]:
        rows = data.get(key)
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    return {"claimed": _ids("success"), "claimed_failed": _ids("failed")}


class HAClient:
    """Async Home Assistant gateway with TTL caches (module-level singleton).

    Owns one aiohttp session for its whole lifetime (connection reuse), the
    /api/states cache and the entity->area cache. All public methods are
    coroutine and never raise — see the module header for the failure
    contract that lets app.py escalate instead of crashing the voice turn.
    """

    def __init__(
        self,
        url: str = config.HA_URL,
        token: str = config.HA_TOKEN,
        timeout: float = config.HA_TIMEOUT,
    ):
        # Defaults are read from config at *definition* time, which is fine:
        # config itself already resolved the env vars at import time.
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        # (states, fetched_at) and (entity->area, fetched_at) — timestamps are
        # monotonic so a wall-clock jump cannot pin a stale cache forever.
        self._states: list[dict] = []
        self._states_at: float = 0.0
        self._entity_areas: dict[str, str] = {}
        self._entity_areas_at: float = 0.0

    async def _sess(self) -> aiohttp.ClientSession:
        """Lazily created session; recreated if something closed it.

        Re-checking `closed` on every call protects against an exception
        outside this class having torn the session down mid-flight.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    def _headers(self) -> dict:
        """Bearer auth for both REST and MCP.

        The Accept list is why `json(content_type=None)` is used everywhere
        below: HA may answer the MCP endpoint with text/event-stream even
        for a plain POST, so the content type cannot be trusted.
        """
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    async def call_tool(self, name: str, arguments: dict, rpc_id: int = 1) -> dict:
        """Invoke an MCP tool. Returns {"ok": bool, "result": ..., "error": ...}.

        Never raises: transport errors, non-200 HTTP, jsonrpc-level errors
        and tool-level `isError` all come back as {"ok": False, ...} (plus
        "raw" when HA returned a body worth showing L2). The MCP payload
        wraps its own JSON (`{"success": bool, ...}`) inside content[0].text,
        so that inner envelope is decoded and unwrapped here — a non-JSON
        text payload is returned verbatim under "result".
        """
        sess = await self._sess()
        # Stateless server (verified Sep 2026): no initialize handshake, the
        # rpc id only correlates the response.
        payload = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        try:
            async with sess.post(
                f"{self.url}/api/mcp", json=payload, headers=self._headers()
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("HA MCP %s -> HTTP %s: %s", name, resp.status, body[:200])
                    return {"ok": False, "error": f"http_{resp.status}"}
                data = await resp.json(content_type=None)
        except Exception as e:
            logger.error("HA MCP %s transport error: %s", name, e)
            return {"ok": False, "error": str(e)}

        # jsonrpc-level failure: transport worked, the RPC did not (unknown
        # tool, bad params) — distinct from an HTTP error above.
        if "error" in data:
            logger.error("HA MCP %s jsonrpc error: %s", name, data["error"])
            return {"ok": False, "error": str(data["error"])}

        result = data.get("result") or {}
        if result.get("isError"):
            # MCP convention: the call succeeded but the TOOL failed (e.g.
            # MatchFailedError) — `raw` is kept so app.py can build a useful
            # escalation context for L2.
            return {"ok": False, "error": "tool_is_error", "raw": result}

        # Payload shape: content[0].text = JSON string {"success": bool, ...}
        content = result.get("content") or []
        text = content[0].get("text", "") if content else ""
        try:
            inner = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            # Not JSON: a plain-text answer is a legitimate result (read
            # tools return prose) — report ok, hand the text over as-is.
            return {"ok": True, "result": text}

        if isinstance(inner, dict) and inner.get("success") is False:
            return {"ok": False, "error": inner.get("error", "tool_failed"), "raw": inner}
        if isinstance(inner, dict) and "result" in inner:
            out = {"ok": True, "result": inner["result"]}
            if isinstance(out["result"], dict):
                out.update(_claimed(out["result"]))
            return out
        out = {"ok": True, "result": inner}
        if isinstance(inner, dict):
            out.update(_claimed(inner))
        return out

    async def get_states(self, force: bool = False) -> list[dict]:
        """Cached /api/states for topology + easy_query entity lookup.

        `force=True` bypasses the TTL and is used before a SIDE EFFECT (the
        router must see the device's current state, not a 30 s old one).
        On any failure the previous snapshot is returned instead of
        raising — stale-but-valid beats an exception here, and a first-call
        failure simply yields [] (the caller escalates).
        """
        now = time.monotonic()
        if not force and self._states and (now - self._states_at) < config.STATES_TTL:
            return self._states
        sess = await self._sess()
        try:
            async with sess.get(
                f"{self.url}/api/states", headers=self._headers()
            ) as resp:
                if resp.status != 200:
                    logger.error("HA states HTTP %s", resp.status)
                    return self._states
                data = await resp.json(content_type=None)
        except Exception as e:
            logger.error("HA states error: %s", e)
            return self._states
        if isinstance(data, list):
            self._states = data
            # `now` was taken before the request: the TTL ages from the
            # fetch START, not from whenever the response landed.
            self._states_at = now
        return self._states

    async def get_history(self, entity_id: str, hours: int = 24) -> list | None:
        """REST /api/history/period/{t0}?filter_entity_id=... (manifest mapping).

        Returns the raw HA history payload, or None on any failure (no
        caller among the L1 routes today — kept as the history accessor
        for query kinds that need it). `minimal_response` keeps the payload
        to state+timestamp pairs instead of full attribute dumps.
        """
        from datetime import datetime, timedelta, timezone

        t0 = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        sess = await self._sess()
        try:
            async with sess.get(
                f"{self.url}/api/history/period/{t0}",
                params={"filter_entity_id": entity_id, "minimal_response": ""},
                headers=self._headers(),
            ) as resp:
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except Exception as e:
            logger.error("HA history error: %s", e)
            return None

    async def get_entity_areas(self) -> dict[str, str]:
        """entity_id -> area display name via HA template `area_name` (cached).

        /api/states carries no area information, while the intent matcher's
        area resolution is exactly what failed in the field («гостиная» ->
        INVALID_AREA). Targets for on/off commands are therefore resolved
        against the area registry ourselves before calling MCP.
        """
        now = time.monotonic()
        if self._entity_areas and (now - self._entity_areas_at) < config.STATES_TTL:
            return self._entity_areas
        # One line per entity: "entity_id <TAB> area_name"; area_name()
        # renders "None" for entities assigned to no room, hence the filter.
        tpl = (
            "{% for e in states %}"
            "{{ e.entity_id }}\t{{ area_name(e.entity_id) }}\n"
            "{% endfor %}"
        )
        sess = await self._sess()
        try:
            async with sess.post(
                f"{self.url}/api/template",
                json={"template": tpl},
                headers=self._headers(),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("HA area map HTTP %s: %s", resp.status, body[:200])
                    return self._entity_areas
                text = await resp.text()
        except Exception as e:
            logger.error("HA area map error: %s", e)
            return self._entity_areas
        out: dict[str, str] = {}
        for line in text.splitlines():
            if "\t" not in line:
                continue
            eid, area = line.split("\t", 1)
            eid, area = eid.strip(), area.strip()
            if eid and area and area != "None":
                out[eid] = area
        if out:
            # Never replace a good map with an empty one: a truncated or
            # garbled template answer must not disable area matching.
            self._entity_areas = out
            self._entity_areas_at = now
        return self._entity_areas

    async def call_service(
        self, domain: str, service: str, data: dict
    ) -> dict:
        """POST /api/services/{domain}/{service}. {"ok", "changed", "error"}.

        The only way to reach an action HA does NOT expose as an MCP tool.
        Media transport is exactly that case (field case 03.10.2026):
        `media_player.media_pause` exists in HA, the MCP server has no intent
        for it, so «поставь на паузу» had to be invented by the model and
        failed. Never raises.

        `changed` is HA's answer — the list of entities whose state the call
        moved. EMPTY means the service ran but nothing changed (the player was
        already paused, or the box is offline), which the caller must not
        report as success; that decision is left to app.py, which knows which
        of the two it asked for.
        """
        sess = await self._sess()
        try:
            async with sess.post(
                f"{self.url}/api/services/{domain}/{service}",
                json=data,
                headers=self._headers(),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "HA service %s.%s -> HTTP %s: %s",
                        domain, service, resp.status, body[:200],
                    )
                    return {"ok": False, "error": f"http_{resp.status}",
                            "raw": body[:300]}
                changed = await resp.json(content_type=None)
        except Exception as e:
            logger.error("HA service %s.%s transport error: %s",
                         domain, service, e)
            return {"ok": False, "error": str(e)}
        return {"ok": True, "changed": changed if isinstance(changed, list) else []}

    async def get_entity_state(self, entity_id: str) -> dict:
        """One entity's FRESH state object (not the cached list), {} on error.

        Used to confirm a media service call actually moved something: the
        service reply's `changed` list is empty for attribute-only changes
        (volume), so the only honest verification is to read the entity back.
        """
        sess = await self._sess()
        try:
            async with sess.get(
                f"{self.url}/api/states/{entity_id}", headers=self._headers()
            ) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json(content_type=None)
        except Exception as e:
            logger.error("HA entity state %s: %s", entity_id, e)
            return {}
        return data if isinstance(data, dict) else {}

    async def close(self) -> None:
        """Release the aiohttp session (lifespan shutdown hook)."""
        if self._session and not self._session.closed:
            await self._session.close()


def find_entity(
    states: list[dict],
    hint: str,
    area: str | None = None,
    domain: list[str] | None = None,
    device: str | None = None,
) -> dict | None:
    """Fuzzy entity lookup for easy_query.

    HA entity_ids/friendly_names are mostly latin (bedroom_thermometer
    Temperature) while the resolver speaks russian stems (спальня,
    температур) — both are expanded through synonym tables below. A non-empty
    hint that matches NOTHING globally means the entity does not exist:
    return None (escalate) instead of falling back to an area-only match,
    which would answer a completely different question.

    `domain` (optional) is a PREFERENCE, never a filter: it wins whenever a
    candidate with that domain exists, and does nothing otherwise. Needed
    because «пылесос» matches `vacuum.valetudo_…`, `update.vacuum_card_update`
    AND `sensor.…_map_segments` (all contain "valetudo"/"vacuum"), and both
    the registry order and the numeric-sensor rule below would answer with
    something that is not the vacuum — observed live as «Пылесос: 8».
    Filtering outright would be wrong in the other direction: the household
    lamp is a `switch.*` while the resolver asks for domain ["light"], so
    with no `light.*` candidate the search has to fall back to all matches.

    `device` (optional) names the device whose READING is being asked for:
    «сколько заряда у робота» arrives as hint «заряд» plus device «робот»,
    and the metric alone matches every battery in the house — observed live
    as «SM-A546E Battery level: 33 процентов» (the phone; the robot's own
    `sensor.…_battery_level` sits ~40th) for a question about the robot.
    Unlike `domain`, an empty result is a NO rather than a fallback: the
    metric exists, the device exists as a word, but THIS device has no such
    reading — returning a stranger's entity would attribute it to the asked
    device («Чайник: 33 процентов»), so None goes back to the caller, which
    speaks the resolver's `missing` sentence instead.
    """
    hint_l = (hint or "").lower().strip()
    area_l = (area or "").lower().strip()

    # Bidirectional containment: a RU stem in the hint expands to latin
    # fragments, and a latin hint still matches the RU stem (or vice versa)
    # — the STT produces either language depending on the device name.
    hints: set[str] = set()
    if hint_l:
        hints.add(hint_l)
        for stem, lats in _HINT_LAT.items():
            if stem in hint_l or hint_l in stem:
                hints.update(lats)
    areas: set[str] = set()
    if area_l:
        areas.add(area_l)
        # Stem prefix as extra variant: «гостиная» -> «гостин»,
        # «living room» -> «living» (substring match below).
        areas.add(area_l[:5])
        for stem, lats in _AREA_LAT.items():
            if stem in area_l or area_l in stem:
                areas.update(lats)

    def _hay(e: dict) -> str:
        """Lowercased `entity_id friendly_name` — the text both filters scan.

        Built once per entity so hint/area matching share one haystack and
        one casing rule instead of each caller re-deriving it.
        """
        eid = e.get("entity_id", "")
        name = (e.get("attributes", {}).get("friendly_name") or "").lower()
        return f"{eid} {name}".lower()

    def _match(e: dict, use_hint: bool, use_area: bool) -> bool:
        """True when `e` passes every enabled filter (hint and/or area).

        Empty filter sets pass everything; the caller disables one flag at
        a time to implement the hint->area fallback ladder.
        """
        hay = _hay(e)
        if use_hint and hints and not any(h in hay for h in hints):
            return False
        if use_area and areas and not any(a in hay for a in areas):
            return False
        return True

    candidates = [e for e in states if _match(e, True, True)]
    if not candidates and hints:
        # Hint absent across the whole registry: the device is not exposed.
        return None
    if not candidates and areas:
        candidates = [e for e in states if _match(e, False, True)]
    if not candidates:
        return None
    # The named device owns the reading («заряд у робота»): expand its RU
    # stem through the same table the hint uses, then keep only its own
    # entities. Empty here is a NO, not a preference (see the docstring) —
    # falling through would hand the user another device's number under
    # this device's label.
    device_l = (device or "").lower().strip()
    if device_l:
        dev_hints = {device_l}
        for stem, lats in _HINT_LAT.items():
            if stem in device_l or device_l in stem:
                dev_hints.update(lats)
        in_dev = [e for e in candidates if any(d in _hay(e) for d in dev_hints)]
        if not in_dev:
            return None
        candidates = in_dev
    # The requested domain narrows the POOL when it can: if the resolver asked
    # for domain ["vacuum"] and a `vacuum.*` candidate exists, only that group
    # competes — otherwise the numeric-sensor rule below returns
    # sensor.…_map_segments (state "8") for «Пылесос» (live case 29.09.2026).
    # No candidate with that domain -> the pool stays untouched: a
    # preference, never a filter (the household lamp is a switch.* asked as
    # domain ["light"]).
    pool = candidates
    if domain:
        want = {d.lower() for d in domain}
        in_dom = [e for e in candidates if e["entity_id"].split(".", 1)[0] in want]
        if in_dom:
            pool = in_dom
    # Prefer a numeric sensor (temperature/battery readings) over helpers
    # like number.*_calibration or select.*_display_mode.
    ordered = sorted(
        pool, key=lambda e: 0 if e["entity_id"].startswith("sensor.") else 1
    )
    for e in ordered:
        if e["entity_id"].startswith("sensor."):
            try:
                float(e["state"])
                return e
            except (ValueError, TypeError):
                continue
    return ordered[0]


def _norm(s: str) -> str:
    """'Living Room' / 'living_room' -> 'living room' (registry comparisons)."""
    return re.sub(r"[\s_]+", " ", (s or "").strip().lower())


# Domains an on/off command may legally touch. Deliberately NOT every
# switch: the living room also holds camera_* switches — matching them by
# a broad domain filter would turn off security recording.
_ONOFF_DOMAINS = ("light", "switch")

# Spoken ordinal -> the digit it selects inside an entity id: «свет в первом
# коридоре» must reach `corridor1_light_switch_relay` and «во втором» the
# corridor2 one. WITHOUT this the ordinal was dropped in resolve_action and
# both relays moved for a request that named exactly one of them (field case
# 02.10.2026). Duplicated in smolagents-worker/ha_match.py on purpose (two
# images) — tests/test_hint_sync.py reads both copies and fails on drift,
# same contract as _HINT_LAT.
ORDINAL_STEMS: dict[str, str] = {"перв": "1", "втор": "2", "трет": "3"}

# Endings that turn a stem into an ordinal («первый/первом/первых»,
# «второй/втором», «третий/третьем»). The ending is the guard against the
# stems' false friends: «вторник» starts with «втор» but ends in «ник», so
# Tuesday never selects digit 2.
ORDINAL_ENDINGS: tuple[str, ...] = (
    "ый", "ий", "ой", "ом", "ого", "ому", "ую", "ые", "ых", "ем",
    "ья", "ье", "ей", "яя", "ее",
)


def ordinal_digit(text: str) -> str:
    """'1'..'3' when `text` names a NUMBERED instance of a device, else ''.

    Whole-token match only. A literal 1-3 is honoured too («коридор 1»,
    STT likes to spell them out as digits).
    """
    for raw in (text or "").lower().split():
        tok = raw.strip(".,!?;:()«»\"'\u2013-")
        if tok in ("1", "2", "3"):
            return tok
        if len(tok) < 3 or not tok.endswith(ORDINAL_ENDINGS):
            continue
        for stem, digit in ORDINAL_STEMS.items():
            if tok.startswith(stem):
                return digit
    return ""


def find_action_targets(
    states: list[dict],
    area_map: dict[str, str],
    hint: str,
    area: str | None,
) -> list[dict]:
    """Concrete on/off-switchable entities for a spoken device command.

    The household lamp is wired as a switch relay
    (switch.living_room_light_swith_relay) and all names are latin, so a
    blind `domain=["light"] + area` intent match both misses it and can
    no-op on `unavailable` light entities. Matching here is bilingual
    («свет» -> light via _HINT_LAT) and uses the area registry map when
    it is loaded; `unavailable`/`unknown` states are excluded — an
    offline device must escalate, never be reported as done.
    """
    hint_l = (hint or "").lower().strip()
    if not hint_l:
        return []
    hints: set[str] = {hint_l}
    for stem, lats in _HINT_LAT.items():
        if stem in hint_l or hint_l in stem:
            hints.update(lats)

    area_n = _norm(area)
    # Registry map not loaded -> substring fallback on the haystack, in both
    # separator spellings (entity_ids use "_", display names use spaces).
    variants = {area_n, area_n.replace(" ", "_"), area_n.replace("_", " ")} \
        if area_n else set()

    # «в первом коридоре» arrives as hint «свет 1» (resolve_action appends
    # the digit): only the relay whose id carries it may move. Returning an
    # EMPTY list when nothing carries the digit is deliberate — moving BOTH
    # corridor relays for a request that named one of them is worse than an
    # honest escalation, which is what the caller does with [].
    digit = ordinal_digit(hint_l)

    out: list[dict] = []
    for e in states:
        eid = e.get("entity_id", "")
        if eid.split(".", 1)[0] not in _ONOFF_DOMAINS:
            continue
        if str(e.get("state", "")).lower() not in ("on", "off"):
            continue
        name = (e.get("attributes", {}) or {}).get("friendly_name") or ""
        hay = f"{eid} {name}".lower()
        if digit and digit not in hay:
            continue
        if not any(h in hay for h in hints):
            continue
        if area_n:
            if area_map:
                # Registry area is authoritative; entity without an area
                # does not belong to the requested one.
                if _norm(area_map.get(eid, "")) != area_n:
                    continue
            elif not any(v in hay for v in variants):
                continue
        out.append(e)
    return out


def _area_fragments(area: str) -> set[str]:
    """Comparable fragments of a room name, whichever spelling it arrives in.

    The resolver sends the registry DISPLAY name («Bedroom Vlada»), a tool
    result may carry it back, and a caller of this module may just as well pass
    the RU word («гостиной»). Comparing those as strings silently found
    nothing, so the media path filtered every real room out and answered
    «не удалось» (field case 03.10.2026). Normalised whole name, both
    separator spellings, the _AREA_LAT expansion and the individual words.
    """
    n = _norm(area)
    if not n:
        return set()
    out = {n, n.replace(" ", "_"), n.replace("_", " ")}
    for stem, lats in _AREA_LAT.items():
        if stem in n or n in stem:
            out.update(lats)
    out.update(w for w in re.split(r"[^0-9a-zа-яё]+", n) if len(w) >= 3)
    return out


# The state a player is already in for the service we are about to call, and
# the honest answer when nothing has to move. Both directions matter: «уже на
# паузе» is what the user hears when the Kodi is in fact paused, and saying
# «поставила на паузу» there would be a side effect nobody made. The `idle`
# rows exist because they were the EXPENSIVE case: all four boxes sit in
# `idle`, so «пауза коди на кухне» escalated to L2 for 3-6 s to be told
# «не играет» (field check 03.10.2026, all four rooms). Lives here, not in
# app.py, because tests/test_router_resolution.py imports this module and
# never app.py (no fastapi on the host).
MEDIA_STATE_ANSWER: dict[str, dict[str, str]] = {
    "media_pause": {
        "paused": "Уже на паузе.",
        # Sentence FRAGMENTS (no full stop): the caller prefixes the room —
        # «На кухне ничего не играет» reads better than a bare «ничего не
        # играет», and _area_phrase has a phrase for every room in the house.
        "idle": "ничего не играет",
        "off": "ничего не играет",
    },
    "media_play": {"playing": "Уже играет."},
    "media_stop": {"idle": "Уже не играет.", "off": "Уже не играет."},
    # Switching a track in a stopped box is not a failed command, it is a
    # command about nothing — same honest answer as the pause, and it keeps
    # «следующий трек на кухне» out of the L2 round trip (field check
    # 03.10.2026: it escalated and the model then claimed «Переключил трек»).
    "media_next_track": {"idle": "ничего не играет", "off": "ничего не играет"},
    "media_previous_track": {"idle": "ничего не играет", "off": "ничего не играет"},
}

# Transport services answer «nothing is playing»; the volume ones do not (a
# muted idle box is still a box whose loudness the user is asking about).
MEDIA_TRANSPORT = frozenset({
    "media_pause", "media_play", "media_play_pause", "media_stop",
    "media_next_track", "media_previous_track",
})

# ...which is exactly why a volume call on an idle box used to ESCALATE: it is
# no transport service, so the «nothing was playing, so there was nothing to
# do» branch skipped it and the unmoved fingerprint became a failure. Field
# check 04.10.2026 21:08: «сделай громче в спальне» on the idle bedroom box
# (volume_level 0.80 -> 0.85, just after the window) escalated, and the turn
# finished 60+ s later — a 29.7 s LLM step plus an httpx retry — for a command
# HA had already accepted.
MEDIA_VOLUME = frozenset({
    "volume_up", "volume_down", "volume_set", "volume_mute",
})

# How long to re-read a player before calling the call a failure. Transport
# flips `state` within ~2 s; a volume change never reaches HA's `changed` list
# at all and lands late on an idle Kodi, so it needs a longer window. Waiting
# is cheap here (no LLM) — escalating is what cost the minute.
MEDIA_CONFIRM_DELAYS: dict[str, tuple[float, ...]] = {
    "default": (0.0, 0.6, 1.5),                 # ~2.1 s
    "volume": (0.0, 0.8, 1.7, 2.8, 4.0, 5.4),   # ~6.8 s
}

# States in which a player is DOING something — the difference between «which
# one?» (a real ambiguity) and «ничего не играет» (a true answer).
MEDIA_ACTIVE = ("playing", "buffering", "paused")


def confirm_delays(service: str) -> tuple[float, ...]:
    """The re-read schedule for `service` — see MEDIA_CONFIRM_DELAYS."""
    return MEDIA_CONFIRM_DELAYS[
        "volume" if service in MEDIA_VOLUME else "default"]


def volume_unverifiable(state: dict) -> bool:
    """The box reports no volume_level at all: there is no level to move, so
    no amount of polling can confirm a volume call and waiting only delays the
    answer the user is owed (verified live: le_kitchen / le_vlada report None
    while idle, le_spalnya reports 0.85)."""
    return ((state or {}).get("attributes") or {}).get("volume_level") is None


def media_no_movement_answer(service: str, before: dict,
                             area: str = "") -> str | None:
    """The spoken answer for a media call HA ACCEPTED that moved nothing — or
    None when this combination must be escalated instead.

    A box that was idle before the call has nothing to play, so a transport
    command is a command about nothing («ничего не играет») and a volume
    command is about loudness that cannot be heard right now. Both are honest
    answers, not failures, and both are worth 0.2 s instead of an L2 round
    trip. Anything else (a PLAYING box that stayed playing after a pause) is a
    real failure and belongs to L2.
    """
    if str((before or {}).get("state", "")).lower() not in ("idle", "off"):
        return None
    if service not in MEDIA_TRANSPORT and service not in MEDIA_VOLUME:
        return None
    phrase = _area_phrase(area)
    return f"{phrase} ничего не играет." if phrase else "Ничего не играет."


def media_fingerprint(state: dict) -> tuple:
    """Everything about a player that a transport/volume call can move.

    Needed because HA's `/api/services` reply cannot be trusted as proof of a
    side effect: `changed` came back EMPTY for a `volume_up` that really did
    turn the bedroom box up (0.7 -> 0.8, field check 03.10.2026) — the
    attribute-only change does not always reach the service reply, and
    `state` alone never shows it. `media_position` is deliberately absent: it
    ticks on its own and would make every call look like a change. Same
    function as smolagents-worker/ha_match.media_fingerprint (two images).
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

_MEDIA_DEAD = ("unavailable", "unknown", "none", "")

# Protocol endpoints (AirPlay/DLNA receivers of the Android box) are not boxes
# a person talks to and sit in the same room as the real player — counting them
# made «громкость в гостиной» ambiguous (le_zal_2 + the AirPlay endpoint,
# field check 03.10.2026). Dropped when a real player survives.
_MEDIA_ENDPOINT_WORDS = frozenset({"airplay", "dlna", "chromecast"})

# Playing first: a transport command with no room and no device word means
# «поставь ЕГО на паузу», and «его» is the thing that is playing.
_MEDIA_ORDER = {"playing": 0, "buffering": 1, "paused": 2, "idle": 3, "off": 4}


# Which state makes a player the likely target of a given request. «поставь
# ЕГО на паузу» names its target by what the device is DOING: the one that
# plays (or, when the user is repeating the request, the one already paused —
# live check 03.10.2026, where le_vlada was paused and «nothing is playing»
# would have escalated a command whose answer is «уже на паузе»).
MEDIA_PREFER: dict[str, tuple[str, ...]] = {
    "media_pause": ("playing", "paused", "buffering"),
    "media_play": ("paused", "playing", "buffering"),
    "media_play_pause": ("playing", "paused"),
    "media_stop": ("playing", "paused"),
    "media_next_track": ("playing", "paused"),
    "media_previous_track": ("playing", "paused"),
    # Volume is not playback: a paused box is still a box whose loudness the
    # user means, so it stays in the preference chain behind the live one.
    "volume_up": ("playing", "buffering", "paused"),
    "volume_down": ("playing", "buffering", "paused"),
    "volume_set": ("playing", "buffering", "paused"),
    "volume_mute": ("playing", "buffering", "paused"),
}


def find_media_targets(
    states: list[dict],
    area_map: dict[str, str],
    hint: str,
    area: str | None,
    prefer: tuple[str, ...] = ("playing", "buffering"),
) -> list[dict]:
    """Media players a transport command may address, most likely first.

    The media analogue of find_action_targets, and the reason «пауза коди во
    владиной комнате» reaches `media_player.le_vlada` in ~0.1 s instead of
    escalating: HA's MCP server exposes no media intent at all (03.10.2026),
    so this is the only place a player can be picked.

    Same honest contract as the on/off path — an empty list means "escalate",
    never "pick one anyway":
      * `unavailable` players are dropped (an offline box confirms nothing);
      * a NAMED device that matches nothing falls back to the room alone
        (hint «коди» -> «le» via _HINT_LAT, and even without that hint the
        room decides) — an unnamed one may fall back to the whole pool;
      * an exactly named room beats a room merely CONTAINED in a player's
        area, same rule as find_action_targets;
      * several survivors are narrowed by `prefer` (the state the request is
        about); a tie inside that state is a real ambiguity -> [].
    """
    hint_l = (hint or "").lower().strip()
    hints: set[str] = {hint_l} if hint_l else set()
    for stem, lats in _HINT_LAT.items():
        if hint_l and (stem in hint_l or hint_l in stem):
            hints.update(lats)

    pool: list[dict] = []
    for e in states:
        eid = e.get("entity_id", "")
        if eid.split(".", 1)[0] != "media_player":
            continue
        if str(e.get("state", "")).lower() in _MEDIA_DEAD:
            continue
        pool.append(e)
    if not pool:
        return []
    real = [
        e for e in pool
        if not any(w in _MEDIA_ENDPOINT_WORDS for w in
                   re.split(r"[^a-z0-9]+", str(e.get("entity_id", "")).lower()))
    ]
    pool = real or pool

    def _hay(e: dict) -> str:
        return f"{e.get('entity_id', '')} " \
               f"{(e.get('attributes', {}) or {}).get('friendly_name') or ''}".lower()

    by_name = [e for e in pool if hints and any(h in _hay(e) for h in hints)]

    area_n = _norm(area)
    room_frags = _area_fragments(area or "")

    by_area: list[dict] = []
    for e in pool:
        if not room_frags:
            break
        eid = e.get("entity_id", "")
        if area_map:
            # Registry area is authoritative; a player with no area at all is
            # not in the requested room either.
            if not (room_frags & _area_fragments(area_map.get(eid, ""))):
                continue
        elif not any(v in _hay(e) for v in room_frags):
            continue
        by_area.append(e)
    if room_frags and not by_area:
        # The room the user named holds no player: refusing beats answering
        # for a box in another room («телевизор в коридоре» must not reach
        # le_vlada just because «телевизор» matched something).
        return []

    # Name AND room are BOTH constraints, so they intersect. «коди» expands to
    # «le», which every one of the four boxes carries — letting the name win
    # on its own made «коди в гостиной» answer for le_vlada (the paused one)
    # instead of le_zal_2. When the two cannot meet the room wins: the user
    # named it and it holds exactly one player.
    if by_name and by_area:
        cand = [e for e in by_name if e in by_area] or by_area
    elif by_name:
        cand = by_name
    elif by_area:
        cand = by_area
    else:
        # Nothing named at all — a pronoun («поставь ЕГО на паузу») — may look
        # at the whole pool, and `prefer` decides inside it.
        cand = pool
    if not cand:
        return []

    # Same two rules as smolagents-worker/ha_match (the two must never pick
    # different boxes): keep the players whose area carries the MOST named
    # room words — «Bedroom Vlada» is named twice over and «Bedroom» once — and
    # then let an EXACTLY named room beat a merely containing one, so «спальне»
    # never reaches «Bedroom Vlada». Exactness is tested against the fragment
    # set, not the raw word, so a RU room name works too.
    if by_area and area_map is not None:
        counts = [(e, len(room_frags & _area_fragments(
            area_map.get(e.get("entity_id", ""), "")))) for e in by_area]
        top = max((c for _, c in counts), default=0)
        if top:
            by_area = [e for e, c in counts if c == top]
        exact = [e for e in by_area
                 if _norm(area_map.get(e.get("entity_id", ""), "")) in room_frags]
        if exact:
            by_area = exact
        cand = [e for e in cand if e in by_area]

    if len(cand) > 1:
        # Walk the preference order and keep the first state that narrows
        # anything: the player the user means is the one in the state the
        # request is about. A tie INSIDE that state stays ambiguous.
        for want in prefer:
            narrowed = [e for e in cand
                        if str(e.get("state", "")).lower() == want]
            if narrowed:
                cand = narrowed
                break
    if len(cand) > 1:
        return []  # several players and nothing that says which one
    return sorted(
        cand,
        key=lambda e: (_MEDIA_ORDER.get(str(e.get("state", "")).lower(), 9),
                       e.get("entity_id", "")),
    )


def dedupe_device_facets(targets: list[dict]) -> list[dict]:
    """Drop sub-entities of a matched device root.

    «Включи кофеварку» matches both `switch.coffemaker` and
    `switch.coffemaker_child_lock` (the «coffee» fragment is in both ids) —
    toggling every facet would flip the child lock as well. When one entity
    id is a strict prefix of another they belong to the same device: keep the
    root, drop the facets.
    """
    ids = [t.get("entity_id", "").split(".", 1)[1] for t in targets]
    out: list[dict] = []
    for i, base in enumerate(ids):
        if any(
            j != i and base.startswith(ids[j]) and base != ids[j]
            for j in range(len(ids))
        ):
            continue  # facet (coffemaker_child_lock) of a kept root
        out.append(targets[i])
    return out


def describe_entity(e: dict, area: str | None = None, label: str = "") -> str:
    """Human phrase for TTS from an HA state object.

    Latin technical names (bedroom_thermometer Temperature) read terribly
    aloud, so when the spoken area is known the phrase leads with it
    («В спальне: 22,9 градусов»); otherwise underscores are flattened.

    `label` is a RU device word the resolver already extracted («пылесос»):
    when the friendly name is latin, the spoken word beats both the latin
    name («Roborock Robot: на базе») and the room prefix, because the user
    asked about the DEVICE. Ignored when the friendly name is already
    Russian — there it wins — and never used by the sensor families, which
    pass stems like «температур» that would read worse than the real name.
    """
    name = e.get("attributes", {}).get("friendly_name") or e.get("entity_id", "")
    state = str(e.get("state", ""))
    unit = e.get("attributes", {}).get("unit_of_measurement") or ""
    unit_map = {
        "°C": "градусов",
        "%": "процентов",
        "°": "градусов",
        "hPa": "гектопаскалей",
        "µg/m³": "микрограммов на кубический метр",
        "W": "ватт",
        "kWh": "киловатт-часов",
        "lx": "люкс",
        "dB": "децибел",
        "м³": "кубических метров",
    }
    unit_ru = unit_map.get(unit, unit)
    state_map = {
        "on": "включено",
        "off": "выключено",
        "open": "открыто",
        "closed": "закрыто",
        "unavailable": "недоступно",
        "unknown": "неизвестно",
        "home": "дома",
        "not_home": "не дома",
        "idle": "в ожидании",
        "playing": "воспроизводится",
        "paused": "на паузе",
        # vacuum.* states — without them «docked» was read out in English
        # (field case 29.09.2026, «что там с пылесосом?»).
        "docked": "на базе",
        "cleaning": "убирается",
        "spot_cleaning": "убирается",
        "returning": "возвращается на базе",
        "stuck": "застрял",
        "error": "ошибка",
    }
    state_ru = state_map.get(state.lower(), state)
    # Numeric states: '22.94' -> '22,9' (Russian decimal comma) and '22.0'
    # -> '22'; anything non-numeric keeps the state_map word from above.
    try:
        f = float(state)
        if f == int(f):
            state_ru = str(int(f))
        else:
            state_ru = f"{f:.1f}".replace(".", ",")
    except (ValueError, TypeError):
        pass
    # Latin technical names read terribly aloud: lead with the spoken device
    # word when one was given, else with the room when it is known, otherwise
    # flatten underscores; RU names are spoken as-is (a label never overrides
    # a Russian friendly name — it is only a fallback for latin ones).
    if not re.search(r"[А-Яа-я]", name):
        if label:
            head = label[:1].upper() + label[1:]
        elif area:
            head = _area_phrase(area)
        else:
            head = name.replace("_", " ")
        return f"{head}: {state_ru} {unit_ru}".strip()
    return f"{name}: {state_ru} {unit_ru}".strip()


# Stems that take the preposition «на» (Russian collocation, not grammar).
_ON_STEMS = ("кухн", "коридор", "балкон", "двор", "улиц")

# Spoken area (RU word or area-registry display name) -> ready prepositional
# phrase. An explicit table beats on-the-fly declension: «гостиная»/«Kitchen»
# both give «В гостиной», while the ой-rule breaks on «прихожая» (-> прихожей).
_PREP_PHRASE = {
    "кухня": "На кухне", "kitchen": "На кухне",
    "спальня": "В спальне", "bedroom": "В спальне",
    "гостиная": "В гостиной", "living room": "В гостиной",
    "коридор": "На коридоре", "corridor": "На коридоре",
    "прихожая": "В прихожей", "entrance": "В прихожей",
    "ванная": "В ванной", "bathroom": "В ванной",
    "туалет": "В туалете", "wc": "В туалете",
    "балкон": "На балконе", "balcony": "На балконе",
    "улица": "На улице", "street": "На улице",
    "двор": "На дворе", "yard": "На дворе",
    "подвал": "В подвале", "basement": "В подвале",
    "кабинет": "В кабинете", "office": "В кабинете",
    "детская": "В детской", "kids": "В детской",
    # The Kodi's own area (03.10.2026): the generic latin fallback spoke
    # «В bedroom vlada» at the user.
    "bedroom vlada": "Во владиной комнате",
}


def _prep_candidates(a: str) -> list[str]:
    """Nominative and oblique spellings of a room, best first.

    The resolver hands over the CANONICAL HA area name («Bedroom»), which is
    why «В спальнее» never reached the user — but the helper is also called with
    a Russian display name, and anything else that carries the user's own word
    («спальне», «гостиной»). Those fell through to the generic rule and came out
    as «В спальнее», silently. Deriving the nominative is cheaper than a second
    table and never invents a room that is not in `_PREP_PHRASE`.
    """
    out = [a]
    if a.endswith(("ой", "ей")):   # гостиной -> гостиная, прихожей -> прихожая
        out.append(a[:-2] + "ая")
    if a.endswith("е"):           # улице -> улица, спальне -> спальня/спальня
        out.append(a[:-1] + "а")
        out.append(a[:-1] + "я")
        out.append(a[:-1])         # коридоре -> коридор
    return out


def _area_phrase(area: str) -> str:
    """'кухня'/'Kitchen' -> 'На кухне'; 'Living Room' -> 'В гостиной'."""
    a = _norm(area)
    if not a:
        return ""
    for cand in _prep_candidates(a):
        if cand in _PREP_PHRASE:
            return _PREP_PHRASE[cand]
    if not re.search(r"[а-яё]", a):
        return f"В {a}"  # unknown latin area (e.g. 'Bedroom Vlada')
    prep = "На" if any(a.startswith(s) for s in _ON_STEMS) else "В"
    if re.search(r"ая$", a):
        word = a[:-2] + "ой"       # гостиная -> гостиной
    elif re.search(r"[ая]$", a):
        word = a[:-1] + "е"        # кухня -> кухне, улица -> улице
    else:
        word = a + "е"             # коридор -> коридоре
    return f"{prep} {word}"
