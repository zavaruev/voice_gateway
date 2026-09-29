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
    "штор": ["cover", "curtain", "blind"],
    "музык": ["media_player", "speaker", "receiver"],
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
}


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
            return {"ok": True, "result": inner["result"]}
        return {"ok": True, "result": inner}

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

    async def close(self) -> None:
        """Release the aiohttp session (lifespan shutdown hook)."""
        if self._session and not self._session.closed:
            await self._session.close()


def find_entity(
    states: list[dict],
    hint: str,
    area: str | None = None,
    domain: list[str] | None = None,
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

    out: list[dict] = []
    for e in states:
        eid = e.get("entity_id", "")
        if eid.split(".", 1)[0] not in _ONOFF_DOMAINS:
            continue
        if str(e.get("state", "")).lower() not in ("on", "off"):
            continue
        name = (e.get("attributes", {}) or {}).get("friendly_name") or ""
        hay = f"{eid} {name}".lower()
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
}


def _area_phrase(area: str) -> str:
    """'кухня'/'Kitchen' -> 'На кухне'; 'Living Room' -> 'В гостиной'."""
    a = _norm(area)
    if not a:
        return ""
    if a in _PREP_PHRASE:
        return _PREP_PHRASE[a]
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
