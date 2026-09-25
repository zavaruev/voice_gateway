"""Home Assistant client: MCP JSON-RPC (streamableHttp, stateless) + REST.

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
    def __init__(
        self,
        url: str = config.HA_URL,
        token: str = config.HA_TOKEN,
        timeout: float = config.HA_TIMEOUT,
    ):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._states: list[dict] = []
        self._states_at: float = 0.0
        self._entity_areas: dict[str, str] = {}
        self._entity_areas_at: float = 0.0

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    async def call_tool(self, name: str, arguments: dict, rpc_id: int = 1) -> dict:
        """Invoke an MCP tool. Returns {"ok": bool, "result": ..., "error": ...}."""
        sess = await self._sess()
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

        if "error" in data:
            logger.error("HA MCP %s jsonrpc error: %s", name, data["error"])
            return {"ok": False, "error": str(data["error"])}

        result = data.get("result") or {}
        if result.get("isError"):
            return {"ok": False, "error": "tool_is_error", "raw": result}

        # Payload shape: content[0].text = JSON string {"success": bool, ...}
        content = result.get("content") or []
        text = content[0].get("text", "") if content else ""
        try:
            inner = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return {"ok": True, "result": text}

        if isinstance(inner, dict) and inner.get("success") is False:
            return {"ok": False, "error": inner.get("error", "tool_failed"), "raw": inner}
        if isinstance(inner, dict) and "result" in inner:
            return {"ok": True, "result": inner["result"]}
        return {"ok": True, "result": inner}

    async def get_states(self, force: bool = False) -> list[dict]:
        """Cached /api/states for topology + easy_query entity lookup."""
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
            self._states_at = now
        return self._states

    async def get_history(self, entity_id: str, hours: int = 24) -> list | None:
        """REST /api/history/period/{t0}?filter_entity_id=... (manifest mapping)."""
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
            self._entity_areas = out
            self._entity_areas_at = now
        return self._entity_areas

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


def find_entity(states: list[dict], hint: str, area: str | None = None) -> dict | None:
    """Fuzzy entity lookup for easy_query.

    HA entity_ids/friendly_names are mostly latin (bedroom_thermometer
    Temperature) while the resolver speaks russian stems (спальня,
    температур) — both are expanded through synonym tables below. A non-empty
    hint that matches NOTHING globally means the entity does not exist:
    return None (escalate) instead of falling back to an area-only match,
    which would answer a completely different question.
    """
    hint_l = (hint or "").lower().strip()
    area_l = (area or "").lower().strip()

    hints: set[str] = set()
    if hint_l:
        hints.add(hint_l)
        for stem, lats in _HINT_LAT.items():
            if stem in hint_l or hint_l in stem:
                hints.update(lats)
    areas: set[str] = set()
    if area_l:
        areas.add(area_l)
        areas.add(area_l[:5])
        for stem, lats in _AREA_LAT.items():
            if stem in area_l or area_l in stem:
                areas.update(lats)

    def _hay(e: dict) -> str:
        eid = e.get("entity_id", "")
        name = (e.get("attributes", {}).get("friendly_name") or "").lower()
        return f"{eid} {name}".lower()

    def _match(e: dict, use_hint: bool, use_area: bool) -> bool:
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
    # Prefer a numeric sensor (temperature/battery readings) over helpers
    # like number.*_calibration or select.*_display_mode.
    ordered = sorted(
        candidates, key=lambda e: 0 if e["entity_id"].startswith("sensor.") else 1
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


def describe_entity(e: dict, area: str | None = None) -> str:
    """Human phrase for TTS from an HA state object.

    Latin technical names (bedroom_thermometer Temperature) read terribly
    aloud, so when the spoken area is known the phrase leads with it
    («В спальне: 22,9 градусов»); otherwise underscores are flattened.
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
    }
    state_ru = state_map.get(state.lower(), state)
    try:
        f = float(state)
        if f == int(f):
            state_ru = str(int(f))
        else:
            state_ru = f"{f:.1f}".replace(".", ",")
    except (ValueError, TypeError):
        pass
    if not re.search(r"[А-Яа-я]", name):
        if area:
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
