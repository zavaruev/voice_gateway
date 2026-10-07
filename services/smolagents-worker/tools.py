"""Tools exposed to the smolagents CodeAgent (cascade level 2).

All tool functions are synchronous and use stdlib urllib only: smolagents
runs tools inside its thread pool, so blocking HTTP here never stalls the
async SSE heartbeat in app.py, and no extra async dependency is needed.

Tools (manifest mapping):
  ha_action      -> HA MCP side effects (intent/light/vacuum/timers/broadcast)
  ha_read        -> HA MCP GetLiveContext, REST /api/states fallback
  media_control  -> REST media_player transport (pause/tracks/volume): the MCP
                    server exposes NO media intent, so this cannot go through
                    ha_action (field case 03.10.2026)
  media_search   -> Kodi JSON-RPC library search by title (what IS there)
  media_play     -> Kodi JSON-RPC Player.Open for a library item (verified
                    through the HA entity, never through Kodi's own 'OK')
  qdrant_search  -> semantic search over voice_turns + voice_facts (memory)
  hermes_expert  -> delegate hard diagnostics to L3 Hermes Agent
  weather_forecast -> GET jev-router /weather (open-meteo, deterministic)
  get_datetime   -> local date/time for the voice assistant

⚠ PROMPT-TEXT WARNING — read before editing
  The DOCSTRINGS OF THE `@tool` FUNCTIONS (ha_action, ha_read,
  qdrant_search, hermes_expert, get_datetime, weather_forecast) are not
  documentation: smolagents feeds them verbatim to the LLM as tool
  descriptions, so their wording DIRECTLY changes model behaviour. Leave
  them byte-identical; put explanations in `#` comments inside the
  function bodies instead (the model never sees those). The same applies
  to TASK_TEMPLATE in app.py. (media_control is new since 03.10.2026 and
  its docstring IS the model's only description of the capability.)

WHY HONESTY IS ENFORCED HERE AND NOT IN THE PROMPT
  The CodeAgent runs on a FREE LLM that writes `final_answer` in the SAME
  code block as its tool call — announcing success before the tool ran.
  Prompt rules were ignored in 3/3 field cases, so this module records
  what ACTUALLY happened (per-call outcome buffers below) and exposes it
  to the veto in honesty.py, plus `ha_action` carries two deterministic
  fallbacks: the vacuum `area` regression (25.09.2026, «Пусть робот
  уберется на кухне») and the on/off domain mismatch (02.10.2026,
  «Значит, прихожий.» — the hallway lamp is a `switch.*_relay`, not a
  `light.*`): code retries, not prompting tricks.

CONTRACT
  All tools are sync and stdlib-only (urllib): smolagents runs them in
  its own thread pool, so a blocking HTTP call here never stalls the SSE
  heartbeats in app.py and no async dependency is needed. Every tool
  returns a STRING (errors are returned, never raised) because an
  exception inside a tool would surface to the model as an opaque crash;
  env URLs/keys/timeouts come from config.py.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from smolagents import tool

import config
import kodi
from concurrent.futures import ThreadPoolExecutor
from ha_match import (
    MEDIA_DOMAIN,
    claimed_a_usable_entity,
    confirm_delays,
    has_data,
    media_fingerprint,
    match_states,
    media_service_data,
    normalize_media_action,
    resolve_media_targets,
    resolve_onoff_targets,
    volume_unverifiable,
)
# The same-block guard lives in honesty (pure stdlib, so tests import it
# without smolagents): every tool announces itself there the moment it runs,
# and app.py clears the flag at the end of each agent step.
from honesty import mark_tool_call, reset_run_state

logger = logging.getLogger("worker.tools")

# Home timezone for all human-readable times: Qdrant timestamps and
# get_datetime must print home-local wall-clock time for the speaker,
# regardless of the container's TZ.
_MSK = timezone(timedelta(hours=3))  # Moscow fixed UTC+3 (no DST since 2014)

# --- Side-effect recorder (feeds the honesty veto in honesty.py) ------------
# smolagents executes tools inside its thread pool, so the buffer is
# module-level behind a lock: _run_agent resets it before each run and reads
# it after the final answer is produced.
# THIS IS THE CORE OF THE DESIGN: the veto never re-parses MCP payloads or
# trusts the model's narration — it only looks at these recorded outcomes.
_ACTION_EVENTS: list[dict] = []
# READ outcomes live in their own buffer: ha_read has no side effect, so it
# must never count as a "confirmed side effect" (that would disarm the action
# veto in a compound turn), while a failed read IS proof for the data-claim
# veto — field case 29.09.2026, «Пылесос сейчас работает, заряд 45%» was
# spoken with the read sitting right there as `success: false`.
_READ_EVENTS: list[dict] = []
# Weather outcomes are recorded separately from side effects: a successful
# weather_forecast must never count as "confirmed side effect" and suppress
# the action veto in a compound query («включи кофеварку и какая погода»).
_WEATHER_EVENTS: list[dict] = []
_EVENTS_LOCK = threading.Lock()

# Read payloads are kept far longer than action details: honesty's
# GROUNDING greps them for the subject of a claim (battery / state /
# reading) and for the number of a percentage, so truncating at the
# error-marker size would make a TRUE claim look ungrounded. Weather keeps
# the 300-char default — vet_weather compares its numbers verbatim and a
# longer forecast would start flagging honest paraphrases.
_READ_DETAIL_LIMIT = 4000


def reset_action_events() -> None:
    """Clear ALL THREE buffers AND the same-block guard; called by
    app._run_agent at the start of every attempt so events never leak
    between runs (a stale success from the previous turn would silence the
    veto on a fresh failure, a stale guard flag would refuse the first
    honest answer)."""
    with _EVENTS_LOCK:
        _ACTION_EVENTS.clear()
        _READ_EVENTS.clear()
        _WEATHER_EVENTS.clear()
    reset_run_state()


def get_action_events() -> list[dict]:
    """Snapshot copy of this run's ha_action outcomes (lock-protected):
    [{"tool", "ok", "detail"}, ...] in call order. Never blocks the tool
    thread — it is read after agent.run() has returned."""
    with _EVENTS_LOCK:
        return list(_ACTION_EVENTS)


def get_read_events() -> list[dict]:
    """Snapshot copy of this run's ha_read outcomes
    [{"tool": "ha_read", "ok", "detail"}, ...]; consumed by vet_answer's
    data-claim branch (app._run_agent)."""
    with _EVENTS_LOCK:
        return list(_READ_EVENTS)


def get_weather_events() -> list[dict]:
    """Snapshot copy of this run's weather_forecast outputs
    [{"tool", "detail"}, ...]; consumed by vet_weather (app._run_agent)."""
    with _EVENTS_LOCK:
        return list(_WEATHER_EVENTS)


def _record(
    tool: str, bucket: list[dict], result: str, ok: bool | None, limit: int = 300
) -> None:
    """Append one outcome to `bucket` (lock-protected).

    `ok=None` means "decide here" via ha_match.has_data — the same
    definition the REST fallback uses, so a miss can never be recorded as a
    success (a wrongly-flagged success disarms the honesty veto).
    `detail` is truncated to `limit` — 300 chars is enough for the error
    markers the veto greps for, small enough to keep the log tidy; reads
    pass a much larger limit because the grounding check reads them (see
    _READ_DETAIL_LIMIT).
    """
    if ok is None:
        ok = has_data(result)
    with _EVENTS_LOCK:
        bucket.append({"tool": tool, "ok": ok, "detail": result[:limit]})


def _record_action(tool_name: str, result: str) -> None:
    """Append the outcome of one ha_action call (ok is decided here so the
    veto in honesty.py never has to re-parse MCP payloads).

    `ok` follows ha_match.has_data(): False on the transport/parse prefixes
    returned by mcp_call/ha_action, on `"success": false`, on an `error`
    payload and on a plain-text miss; anything else counts as success —
    deliberately conservative in the OTHER direction: a false "ok" would
    disable the veto, but only genuinely successful calls reach here
    unflagged (the vacuum fallback has already retried by then).
    """
    _record(tool_name, _ACTION_EVENTS, result, None)


def _record_read(tool_name: str, result: str, ok: bool) -> None:
    """Append the outcome of one ha_read call. `ok` is decided by the CALLER:
    a read that came back with no entity («Ничего не найдено…») is a miss,
    not data — recording it as a success would let a fabricated status claim
    through the veto (field case 29.09.2026)."""
    _record(tool_name, _READ_EVENTS, result, ok, limit=_READ_DETAIL_LIMIT)


def _http_json(
    url: str,
    payload: dict | None = None,
    headers: dict | None = None,
    timeout: float = config.TOOL_TIMEOUT,
):
    """HTTP GET (payload=None) or POST-with-JSON-body; -> parsed JSON.

    Raises urllib/timeout errors and ValueError("non-JSON response…") on
    garbage — callers are expected to catch and return a Russian error
    string to the model (a tool must never raise into smolagents).
    Non-obvious: streamableHttp MCP endpoints may answer a plain POST with
    SSE frames instead of a JSON body, so a non-JSON body is re-scanned for
    `data:` lines before giving up.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # streamableHttp may answer with SSE frames even for plain POSTs
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                try:
                    return json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"non-JSON response from {url}: {body[:200]}")


def mcp_call(name: str, arguments: dict, timeout: float = config.TOOL_TIMEOUT) -> str:
    """JSON-RPC tools/call against HA MCP (stateless: no initialize).

    Returns the text of the first content block (or the raw result), or a
    Russian error string — NEVER raises. The «Ошибка…», «Неожиданный…» and
    «Тул вернул ошибку» prefixes (plus ha_action's «Некорректный…») form a
    contract with _record_action() and honesty._truth(): they decide
    ok=False and which refusal phrase is spoken, so do not reword them.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    try:
        data = _http_json(
            f"{config.HA_URL}/api/mcp",
            payload,
            headers={
                "Authorization": f"Bearer {config.HA_TOKEN}",
                "Accept": "application/json, text/event-stream",
            },
            timeout=timeout,
        )
    except Exception as e:
        return f"Ошибка MCP-транспорта: {e}"
    if not isinstance(data, dict):
        return f"Неожиданный ответ MCP: {str(data)[:200]}"
    if "error" in data:
        return f"Ошибка MCP: {str(data['error'])[:300]}"
    result = data.get("result") or {}
    if result.get("isError"):
        return f"Тул вернул ошибку: {str(result)[:300]}"
    content = result.get("content") or []
    return content[0].get("text", str(result)) if content else str(result)


# --- Raw registry behind ha_action's deterministic on/off fallback ----------
# /api/states carries no area information, so the room filter also needs one
# `area_name` template render — the same source jev-router's L1 uses. Both
# are fetched ONLY after a blind intent call has already failed (the fast
# path still costs exactly one MCP call) and cached for a few seconds, since
# one turn may retry more than one call.
_RAW_TTL = 10.0
_RAW_LOCK = threading.Lock()
_RAW_STATES: list | None = None
_RAW_AREAS: dict | None = None
_RAW_AT = 0.0


def _http_text(url: str, payload: dict | None = None, timeout: float = config.TOOL_TIMEOUT) -> str:
    """POST/GET returning the BODY VERBATIM — /api/template answers plain
    text («entity_id <TAB> area_name» lines), which _http_json would reject
    as non-JSON."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.HA_TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _raw_registry() -> tuple[list | None, dict | None]:
    """(raw states, entity_id -> HA area display name); (None, None) on error."""
    global _RAW_STATES, _RAW_AREAS, _RAW_AT
    with _RAW_LOCK:
        now = time.monotonic()
        if _RAW_STATES is not None and now - _RAW_AT < _RAW_TTL:
            return _RAW_STATES, _RAW_AREAS
        tpl = "{% for e in states %}{{ e.entity_id }}\t{{ area_name(e.entity_id) }}\n{% endfor %}"
        try:
            states = _http_json(
                f"{config.HA_URL}/api/states",
                headers={"Authorization": f"Bearer {config.HA_TOKEN}"},
            )
            areas: dict[str, str] = {}
            for line in _http_text(f"{config.HA_URL}/api/template", {"template": tpl}).splitlines():
                if "\t" not in line:
                    continue
                eid, area = line.split("\t", 1)
                if area and area != "None":
                    areas[eid.strip()] = area.strip()
        except Exception as e:
            logger.warning("raw HA registry fetch failed: %s", e)
            return None, None
        _RAW_STATES = states if isinstance(states, list) else None
        _RAW_AREAS = areas
        _RAW_AT = now
        return _RAW_STATES, _RAW_AREAS




def _action_ok(text: str) -> bool:
    """True only for an intent answer that CONFIRMED its side effect.

    `action_done` carries both lists; has_data() alone would accept
    `{"data": {"success": [], "failed": [...]}}` as data, and a retry built
    on that would be reported as work that never happened.
    """
    if not has_data(text):
        return False
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return True
    if not isinstance(payload, dict):
        return True
    data = payload.get("data")
    if isinstance(data, dict) and "success" in data:
        return bool(data.get("success"))
    return True


# Every tool HA's MCP server actually exposes, read out of `tools/list` on
# 03.10.2026 (HA 2026.9.4). The docstring above lists EXAMPLES, so the free
# model happily invented `intent__HassMediaPause` / `intent__HassPause` for
# «поставь на паузу» and burned two steps on «Tool … not found» before
# reporting failure — a name that is not here cannot work, so it is refused
# here, in code, with the real list in the message. Cheap insurance the other
# way: a tool HA adds later is refused until this table learns about it, and
# the message tells the model what DOES exist (media_control below).
MCP_TOOLS: frozenset[str] = frozenset({
    "assist_satellite__HassBroadcast",
    "homeassistant__GetLiveContext",
    "intent__HassTurnOn",
    "intent__HassTurnOff",
    "intent__HassCancelAllTimers",
    "light__HassLightSet",
    "llm__GetDateTime",
    "vacuum__HassVacuumStart",
    "vacuum__HassVacuumReturnToBase",
    "vacuum__HassVacuumCleanArea",
})


@tool
def ha_action(tool_name: str, arguments_json: str) -> str:
    """Выполнить действие в Home Assistant через MCP-тул.

    Args:
        tool_name: Имя MCP-тула, напр. intent__HassTurnOn, intent__HassTurnOff,
            light__HassLightSet, vacuum__HassVacuumCleanArea (уборка в комнате),
            vacuum__HassVacuumStart (просто запустить пылесос БЕЗ указания комнаты),
            vacuum__HassVacuumReturnToBase, intent__HassCancelAllTimers,
            assist_satellite__HassBroadcast.
        arguments_json: JSON-строка аргументов, напр. {"area": "кухня", "domain": ["light"]}
            или {"message": "обед готов"}. Ключи: name, area, floor, domain, device_class,
            color, temperature, brightness, message — в зависимости от тула.
            ВАЖНО для пылесоса: уборка в конкретной комнате — только
            vacuum__HassVacuumCleanArea с {"area": "кухня"}; у HassVacuumStart
            area означает ГДЕ стоит пылесос (комната-фильтр), а не цель уборки.
    """
    mark_tool_call()  # honesty.same_block_check: this block now HAS a tool
    # Unknown tool name: never sent (so never even a transport error) and
    # guaranteed to fail. Recorded like any other failure — a claim attached
    # to it must be vetoed.
    if tool_name not in MCP_TOOLS:
        msg = (
            f"Тул вернул ошибку: в Home Assistant нет тула «{tool_name}». "
            "Для паузы, треков и громкости тул media_control; для света, "
            "пылесоса, таймеров и on/off — intent__HassTurnOn, "
            "intent__HassTurnOff, light__HassLightSet, "
            "vacuum__HassVacuumStart, vacuum__HassVacuumCleanArea, "
            "vacuum__HassVacuumReturnToBase, intent__HassCancelAllTimers, "
            "assist_satellite__HassBroadcast."
        )
        _record_action(tool_name, msg)
        return msg
    # A malformed/absent JSON object never reaches HA, but it is still a
    # failed call: record it so the veto can refuse honestly instead of
    # letting the model claim success for a command that was never sent.
    try:
        args = json.loads(arguments_json) if arguments_json.strip() else {}
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        msg = f"Некорректный arguments_json: {e}"
        _record_action(tool_name, msg)
        return msg
    result = mcp_call(tool_name, args)
    # Field regression 25.09.2026 («Пусть робот уберется на кухне»): in
    # HassVacuumStart the area slot filters by the vacuum's LOCATION, and the
    # robot is assigned to no area in HA -> MatchFailedReason.AREA every time,
    # while the user's intent with a room is "clean there". Fallback:
    # Start+area -> CleanArea (area = cleaning target, entity match ignores
    # the robot's area), ReturnToBase+area -> same tool without area (a
    # location filter is meaningless for docking). Original behaviour is kept
    # when Start+area actually matches.
    # Why deterministic code and not a prompt rule: the model ignored the
    # prompt version of this advice, and the fix must hold for a FREE LLM.
    # `tool_name`/`result` are REBOUND so _record_action below logs the
    # call that actually executed — the veto judges the retried outcome.
    if "area" in args and "MatchFailedReason.AREA" in result:
        if tool_name == "vacuum__HassVacuumStart":
            retry_tool = "vacuum__HassVacuumCleanArea"
            retry_args = {k: v for k, v in args.items() if k in ("area", "name")}
        elif tool_name == "vacuum__HassVacuumReturnToBase":
            retry_tool = tool_name
            retry_args = {k: v for k, v in args.items() if k != "area"}
        else:
            # Non-vacuum tool with an AREA mismatch: no safe automatic
            # rewrite exists (intent tools resolve rooms for real devices),
            # so the original error stands and the model must report it.
            retry_tool, retry_args = "", {}
        if retry_tool:
            logger.info("vacuum AREA mismatch -> retry %s %s",
                        retry_tool, retry_args)
            tool_name, result = retry_tool, mcp_call(retry_tool, retry_args)
    # On/off intents whose domain/area filter did not survive HA's matcher —
    # field case 02.10.2026 05:45 («Значит, прихожий.»): the hallway lamp is
    # `switch.entrance_light_switch_relay`, so `domain: ["light"]` answered
    # MatchFailedReason.AREA for «прихожая» AND for «Entrance», and before the
    # relay was exposed to Assist the answer was MatchFailedReason.ASSISTANT.
    # L1 (jev-router) resolves the concrete entity from raw /api/states before
    # calling the intent; this is that rule for L2, and it only fires when the
    # resolver is SURE: a device word to aim at, a room when several devices
    # match, ≤ MAX_ONOFF_TARGETS of them, all on/off right now, facets (the
    # status LED next to the relay) dropped. Anything else — an unreadable
    # registry, an offline device, an ambiguity — keeps the ORIGINAL error:
    # a refusal must stay honest, never a guessed side effect.
    #
    # NAME joins AREA/ASSISTANT because HA answers it for two cases the
    # resolver CAN settle: a phrase in the `name` slot («свет в первом
    # коридоре» — HA matches names, not sentences) and the CONCATENATED
    # friendly name, which HA accepts only for some entities (field check
    # 02.10.2026: «corridor1_light_switch Relay» and «coffemaker» ->
    # MatchFailedReason.NAME while «entrance_light_switch Relay» matched).
    # The honesty contract of 2.34 survives untouched: an INVENTED word
    # («кашеварку») matches nothing in the registry, the resolver returns []
    # and the original NAME error stands — the fallback rescues phrasing,
    # never guessing.
    if tool_name in ("intent__HassTurnOn", "intent__HassTurnOff") and any(
        r in result
        for r in ("MatchFailedReason.AREA", "MatchFailedReason.ASSISTANT",
                  "MatchFailedReason.NAME")
    ):
        states, areas = _raw_registry()
        targets = resolve_onoff_targets(
            states or [],
            name=str(args.get("name") or ""),
            area=str(args.get("area") or ""),
            domains=args.get("domain"),
            area_map=areas,
        )
        oks: list[tuple[str, str]] = []
        fails: list[str] = []
        for ent in targets:
            eid = ent.get("entity_id", "")
            # Name + domain pinned to the resolved entity: the area slot is
            # deliberately DROPPED — it was the filter that just failed, and
            # the entity id is already unique. `name` takes the entity ID
            # rather than the friendly name on purpose: HA's matcher accepts
            # the concatenated friendly name only for some entities — field
            # check 02.10.2026, «corridor1_light_switch Relay» and
            # «coffemaker» answered MatchFailedReason.NAME while
            # «entrance_light_switch Relay» matched, which is what made this
            # retry fail exactly where it was needed. Exposure is still
            # enforced: the intent keeps its assistant='conversation' filter
            # however the entity was named.
            retry_args = {"name": eid, "domain": [eid.split(".", 1)[0]]}
            res2 = mcp_call(tool_name, retry_args)
            (oks if _action_ok(res2) else fails).append((eid, res2))
        if oks and not fails:
            logger.info("on/off matcher miss -> resolved %s, retried %s",
                        ", ".join(e for e, _ in oks), tool_name)
            # One target keeps HA's own confirmation payload; several are
            # summarised so the veto records exactly what moved.
            result = oks[-1][1] if len(oks) == 1 else (
                "Выполнено: " + ", ".join(e for e, _ in oks)
            )
        elif oks:
            # Partial fan-out: the refusal must not hide the side effects
            # that DID happen (same contract as L1's `done` in its error).
            result = (
                f"Тул вернул ошибку: выполнено {len(oks)}/{len(targets)}: "
                f"{', '.join(e for e, _ in oks)}; остальные не сработали"
            )
        # Zero successes: the ORIGINAL error stands — it is what actually
        # happened, and the model must report that instead of a guess.

    # --- AN `ok` THAT NAMED A ROOM IS NOT A DEVICE THAT MOVED -------------------
    # Field case 07.10.2026 11:39, living room: «Выключи свет в гостиной» was
    # transcribed «Выключиться в гостиной», escalated to L2, and L2 called
    #
    #     ha_action("intent__HassTurnOff", {"area": "Living Room", "domain": ["light"]})
    #
    # which answered, with HTTP 200 and `failed: []`:
    #
    #     {"response_type": "action_done",
    #      "data": {"success": [{"type": "area",   "id": "living_room"},
    #                            {"type": "entity", "id": "light.wled_living_room"}],
    #               "failed": []}}
    #
    # and the room SAID «Выключила свет в гостиной.» The lamp did not move: it is
    # `switch.living_room_light_swith_relay`, and `domain: ["light"]` cannot match
    # a `switch` at all. The only entity named was `light.wled_living_room`, which
    # has been `unavailable` since 28.09, and the other entry is the ROOM.
    #
    # `_action_ok` above accepted it because the list is non-empty, `has_data` then
    # recorded `ok=True`, which DISARMS the honesty veto, and the model was free to
    # speak as done. The router has refused this exact shape since 04.10.2026 via
    # `_touched_a_usable_entity` — one path was guarded and the other was not, the
    # same asymmetry already documented for the on/off fast path vs the blind one.
    # Applied HERE as well, and deliberately only to the on/off intents: an area-only
    # match is provably not a lamp for those, whereas a vacuum `CleanArea` may
    # legitimately report a room.
    if tool_name in ("intent__HassTurnOn", "intent__HassTurnOff"):
        states, _areas = _raw_registry()
        if not claimed_a_usable_entity(result, states):
            result = (
                "Тул вернул ошибку: intent__Hass"
                + ("TurnOn" if tool_name.endswith("TurnOn") else "TurnOff")
                + " ответил успехом, но не назвал ни одного рабочего устройства"
                " — в списке только комната и/или недоступные сущности "
                "(проверено по живому реестру HA). Устройство НЕ переключено. "
                "Не говори, что включила или выключила: назови настоящую "
                "причину."
            )
            logger.warning(
                "on/off intent claimed only areas/unavailable entities -> %s",
                result[:120],
            )
    _record_action(tool_name, result)
    return result


@tool
def ha_read(query: str, area: str = "") -> str:
    """Прочитать текущее состояние устройства или датчика из Home Assistant.

    Args:
        query: Что читаем: имя устройства (чайник), тип (свет, датчик температуры),
            домен (light, switch, sensor, media_player) или показатель (температура, влажность, заряд).
        area: Комната по-русски или по-английски (кухня, спальня, kitchen); можно пусто.
    """
    mark_tool_call()  # honesty.same_block_check: this block now HAS a tool
    # Primary: HA MCP GetLiveContext (understands name+area and answers
    # with a curated context blob). A transport/tool error ("Ошибка…"), an
    # EMPTY result and a `{"success": false …}` payload all mean "miss" ->
    # fall through to raw REST states. The old check only looked for the
    # «Ошибка» prefix, so the miss this branch actually produces —
    # `{"success": false, "error": "No exposed entities matched name 'пылесос'"}`
    # (29.09.2026) — was returned verbatim and the fallback never ran.
    args: dict = {"name": query}
    if area:
        args["area"] = area
    text = mcp_call("homeassistant__GetLiveContext", args)
    if has_data(text):
        _record_read("ha_read", text, ok=True)
        return text

    # Fallback: raw REST states, bilingual substring match (MCP context miss).
    # match_states() expands «пылесос» -> vacuum/roborock/robot and «кухня»
    # -> kitchen, because the registry is latin while the user speaks RU.
    try:
        states = _http_json(
            f"{config.HA_URL}/api/states",
            headers={"Authorization": f"Bearer {config.HA_TOKEN}"},
        )
    except Exception as e:
        err = f"Не удалось прочитать состояния: {e}"
        _record_read("ha_read", err, ok=False)
        return err
    # The area map comes from the same cached fetch media_control and the
    # on/off retry use: /api/states carries no room, so a «media_player в
    # гостиной» style query can only be answered with it (03.10.2026).
    try:
        _states, _areas = _raw_registry()
    except Exception:
        _areas = None
    hits = match_states(states, query, area, area_map=_areas)
    if hits:
        out = "\n".join(hits)
        _record_read("ha_read", out, ok=True)
        return out
    out = "Ничего не найдено в Home Assistant."
    # A miss, not data: recorded as such so a fabricated status answer built
    # on top of it is vetoed instead of spoken (honesty.py, data claim).
    _record_read("ha_read", out, ok=False)
    return out


def _entity_snapshot(entity_id: str) -> dict:
    """One entity's fresh state object, {} when HA cannot be read at all."""
    try:
        data = _http_json(
            f"{config.HA_URL}/api/states/{entity_id}",
            headers={"Authorization": f"Bearer {config.HA_TOKEN}"},
        )
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@tool
def media_control(action: str, area: str = "", name: str = "", value: str = "") -> str:
    """Управление медиаплеером: Kodi, телевизор, колонка (media_player).
    Только этот тул умеет паузу, треки и громкость — в Home Assistant для
    них НЕТ тулов intent__*, поэтому через ha_action их сделать нельзя.

    Args:
        action: media_pause, media_play, media_play_pause, media_stop,
            media_next_track, media_previous_track, volume_up, volume_down,
            volume_set, volume_mute.
        area: Комната по-русски («владиная комната», «гостиная»); можно пусто.
        name: Как назвали устройство («коди», «телевизор»); можно пусто.
        value: Для volume_set — «40 процентов»; для volume_mute — «выключи»
            или «включи». Для остальных не нужно.
    """
    mark_tool_call()  # honesty.same_block_check: this block now HAS a tool
    service = normalize_media_action(action)
    if not service:
        msg = ("Тул вернул ошибку: нет такого действия для медиаплеера. "
               "Доступно: пауза, воспроизведение, стоп, следующий/предыдущий "
               "трек, громче, тише, громкость в процентах, заглушить звук.")
        _record("media_control", _ACTION_EVENTS, msg, ok=False)
        return msg

    extra = media_service_data(service, value)
    if service in ("volume_set", "volume_mute") and not extra:
        # Checked here instead of letting HA answer 400: the model handed over
        # no level and no flag, and a «Bad Request» body teaches it nothing
        # about what was missing.
        need = ("уровень громкости в процентах («40 процентов»)"
                if service == "volume_set" else "«выключи» или «включи»")
        msg = (f"Тул вернул ошибку: для {service} нужно указать {need} "
               "в параметре value.")
        _record("media_control", _ACTION_EVENTS, msg, ok=False)
        return msg

    states, areas = _raw_registry()
    if not states:
        msg = ("Тул вернул ошибку: Home Assistant не отвечает, состояние "
               "медиаплееров неизвестно.")
        _record("media_control", _ACTION_EVENTS, msg, ok=False)
        return msg

    targets = resolve_media_targets(states, name=name, area=area, area_map=areas,
                                    service=service)
    if not targets:
        # One reason for every miss (unknown name, unknown room, several
        # players and no room): the model repeats one honest sentence instead
        # of guessing a target on the next step.
        msg = ("Тул вернул ошибку: не нашла медиаплеер, к которому это "
               "относится (нет такого устройства, нет такой комнаты, "
               "или их несколько и комната не названа).")
        _record("media_control", _ACTION_EVENTS, msg, ok=False)
        return msg
    if len(targets) > 1:
        eids = ", ".join(str(e.get("entity_id", "")) for e in targets)
        msg = (f"Тул вернул ошибку: подходит несколько медиаплееров ({eids}) — "
               "назови комнату или устройство.")
        _record("media_control", _ACTION_EVENTS, msg, ok=False)
        return msg

    eid = str(targets[0].get("entity_id", ""))
    before = str(targets[0].get("state", ""))
    payload = {"entity_id": eid, **extra}
    try:
        _http_json(
            f"{config.HA_URL}/api/services/{MEDIA_DOMAIN}/{service}",
            payload,
            headers={"Authorization": f"Bearer {config.HA_TOKEN}"},
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        msg = f"Тул вернул ошибку: Home Assistant отклонил {service} ({body})"
        _record("media_control", _ACTION_EVENTS, msg, ok=False)
        return msg
    except Exception as e:
        msg = f"Тул вернул ошибку: не удалось выполнить {service}: {e}"
        _record("media_control", _ACTION_EVENTS, msg, ok=False)
        return msg

    # The state is re-read instead of assumed, and the WHOLE movable fingerprint
    # is compared — a call that changed nothing is not a side effect, whatever
    # the box happened to report. HA answers the service call BEFORE the Kodi
    # reports back (a pause landed ~2 s later, field check 03.10.2026), so the
    # read is retried for a couple of seconds before giving up: without the
    # wait every working command looked like a failure.
    after_state: dict = {}
    before_fp = media_fingerprint(targets[0])
    # A box that reports NO volume_level is not polled: there is no level to
    # move, so the wait can only end in the honest «ничего не играет» and costs
    # the whole ~6.8 s volume window instead. Otherwise the service decides the
    # schedule — volume lands late on an idle Kodi (le_spalnya 0.80 -> 0.85
    # after the transport window, field check 04.10.2026 21:08) while transport
    # flips `state` within ~2 s.
    if service.startswith("volume") and volume_unverifiable(targets[0]):
        delays: tuple[float, ...] = (0.0,)
    else:
        delays = confirm_delays(service)
    for delay in delays:
        if delay:
            time.sleep(delay)
        after_state = _entity_snapshot(eid)
        if after_state and media_fingerprint(after_state) != before_fp:
            break
    after = str(after_state.get("state", "")) or before
    unchanged = bool(after_state) and media_fingerprint(after_state) == before_fp
    if unchanged:
        # ok=False on purpose: recorded as a success this armed the veto and
        # let the model say «остановила» / «переключил трек» for a command
        # that moved nothing (field check 03.10.2026). honesty._truth turns
        # these markers into the same sentence the fast path speaks in 0.2 s.
        stopped = after in ("idle", "off")
        why = ("уже на паузе" if service == "media_pause" and after == "paused"
               else "уже не играет" if service == "media_stop" and stopped
               # Volume on a stopped Kodi cannot move: it reports no
               # volume_level at all until something plays (verified live,
               # 03.10.2026), so «громкость не изменилась» would blame the
               # command for the box's state.
               else "ничего не играет" if stopped and (
                   service in ("media_pause", "media_next_track",
                               "media_previous_track")
                   or service.startswith("volume"))
               else "громкость не изменилась" if service.startswith("volume")
               else "состояние не изменилось")
        msg = f"Тул вернул ошибку: {eid}: {service} — {why} (состояние {after})."
        _record("media_control", _ACTION_EVENTS, msg, ok=False)
        return msg
    out = f"{eid}: {service} выполнено, состояние {before} -> {after}"
    _record("media_control", _ACTION_EVENTS, out, ok=True)
    return out


def _kodi_ready() -> str:
    """"" for a healthy Kodi setup, or the Russian reason there is none."""
    if not config.KODI_HOSTS:
        return ("ящики Kodi не настроены (нужны KODI_HOSTS, KODI_USER, "
                "KODI_PASS в окружении воркера)")
    if not config.KODI_USER:
        return "ящики Kodi не настроены: нет KODI_USER/KODI_PASS в окружении"
    return ""


def _box_for(area: str = "", name: str = "") -> tuple[dict | None, str, str]:
    """(box, entity_id, why-not) for a room/device the caller named.

    The entity comes from the SAME resolver media_control uses, so a search can
    never land on a different box than a pause would.
    """
    states, areas = _raw_registry()
    if not states:
        return None, "", "Home Assistant не отвечает"
    targets = resolve_media_targets(states, name=name, area=area,
                                    area_map=areas)
    if not targets:
        return None, "", "не нашла медиаплеер, к которому это относится"
    ent = targets[0]
    friendly = (ent.get("attributes") or {}).get("friendly_name") or ""
    box = kodi.match_entity(friendly, _kodi_boxes())
    eid = str(ent.get("entity_id", ""))
    if not box:
        probed = _kodi_boxes()
        # WHY per box: a wrong password on a box that is switched ON is a
        # different fix from an outage, and «не отвечает» for both sent the
        # user looking at the wrong thing (03.10.2026).
        parts = [f"{b['host']} — {b.get('why') or 'не отвечает'}"
                 for b in probed if not b.get("ok")]
        return None, eid, (f"ящик Kodi для «{friendly}» недоступен ("
                           f"{'; '.join(parts) or 'нет ни одного ящика'})")
    return box, eid, ""


def _kodi_client(host: str) -> kodi.Kodi:
    """A client for a box with THAT box's credentials.

    Takes the raw `[user:pass@]host` seed (probe_boxes keeps it in the row) so
    a box whose password differs from the default keeps working — measured
    03.10.2026: LE-vlada answers kodi/kodi while the other three take
    kodi/2441, and a plain host string silently fell back to the wrong pair.
    """
    h, u, p = kodi.split_seed(host, config.KODI_USER, config.KODI_PASS)
    return kodi.Kodi(h, u, p, timeout=config.KODI_TIMEOUT)


def _kodi_boxes() -> list[dict]:
    return kodi.boxes(config.KODI_HOSTS, config.KODI_USER, config.KODI_PASS,
                      timeout=config.KODI_TIMEOUT)


@tool
def media_search(title: str = "", area: str = "", name: str = "") -> str:
    """Что есть в библиотеке Kodi: список сериалов (с первым эпизодом) и
    фильмов. Ничего не запускает. Название — необязательно: с ним в начале
    идёт БЛИЖАЙШЕЕ совпадение, но полный список всё равно печатается —
    «Симпсоны» и «The Simpsons» это разные строки, и по названию из
    библиотеки их свяжет ты, а не тупое сравнение строк.

    Args:
        title: Что ищем, по-русски или по-английски («Темное зеркало»).
        area: Комната по-русски («гостиная»). ОБЯЗАТЕЛЬНО передай комнату из
            запроса: без неё поиск идёт по всем ящикам сразу.
        name: Как назвали устройство («коди»); можно пусто.
    """
    mark_tool_call()  # honesty.same_block_check: this block now HAS a tool
    bad = _kodi_ready()
    if bad:
        return f"Тул вернул ошибку: поиск по библиотеке невозможен — {bad}."
    wanted = (title or "").strip().strip("\u00ab\u00bb\"'`")
    _record("media_search", _READ_EVENTS, "", ok=False)

    if area or name:
        box, eid, why = _box_for(area, name)
        targets = [(box, eid)] if box else []
        if not box:
            return f"Тул вернул ошибку: {why}."
    else:
        targets = [(b, "") for b in _kodi_boxes() if b.get("ok")]
    if not targets:
        return "Тул вернул ошибку: ни один ящик Kodi не отвечает."

    def _scan(b: dict, eid: str) -> str:
        """One box's catalogue block. Blocking JSON-RPC, so the boxes are
        scanned in PARALLEL — a room-less search hits all four and used to pay
        their round trips one after another."""
        label = b["short"] + (f" / {eid}" if eid else "")
        try:
            cli = _kodi_client(b.get("seed") or b["host"])
            shows, movies = cli.tvshows(), cli.movies()
        except kodi.KodiError as e:
            return f"{label}: {e.says} ({e})"
        titles = ([sh.get("title", "") for sh in shows]
                  + [m.get("title", "") for m in movies])
        head = f"[{label}] сериалов {len(shows)}, фильмов {len(movies)}:"
        found = ""
        if wanted:
            best, _score = kodi.best_title(wanted, titles)
            found = best or ""
            head += (f" ближайшее к «{wanted}»: "
                     + (f"{found}" if found else "ничего похожего"))
        rows = []
        seen_titles: set[str] = set()
        dupes = 0
        # Episodes are fetched for the BEST MATCH only: one JSON-RPC round trip
        # per show made a catalogue listing cost 14 of them (and the whole turn
        # 18-22 s, field check 03.10.2026). The ids are printed for every show,
        # so a follow-up media_play can still pick any of them.
        best_norm = kodi.norm(found) if found else ""
        for sh in shows:
            # The library really does carry the same title twice or thrice
            # (measured 03.10.2026: three "The Simpsons", one of them with no
            # episodes at all). List it once — with the FIRST id — and report
            # the duplicates, so the model never chooses between three
            # identical rows.
            key = kodi.norm(sh.get("title", ""))
            if key in seen_titles:
                dupes += 1
                continue
            seen_titles.add(key)
            line = (f"  - сериал «{sh.get('title')}» ({sh.get('year') or '—'}), "
                    f"id={sh['tvshowid']}")
            if key == best_norm:
                try:
                    eps = cli.episodes(sh["tvshowid"], limit=3)
                    line += ", ближайшие: " + (
                        "; ".join(kodi.episode_line(e) for e in eps)
                        or "нет эпизодов")
                except kodi.KodiError as e:
                    line += f", эпизоды не прочитаны ({e})"
            rows.append(line)
        for m in movies[:40]:
            rows.append(f"  - фильм «{m.get('title')}» ({m.get('year') or '—'}), "
                        f"movieid={m.get('movieid')}")
        block = head + "\n" + "\n".join(rows)
        if dupes:
            block += (f"\n  (в библиотеке ещё {dupes} дублей тех же сериалов — "
                     "выше первый, с эпизодами)")
        return block

    with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
        lines = list(pool.map(lambda t: _scan(t[0], t[1]), targets))
    out = "\n".join(lines)
    _record("media_search", _READ_EVENTS, out, ok=True)
    return out


@tool
def media_play(title: str, area: str = "", name: str = "", episode: str = "") -> str:
    """Найти сериал/фильм в библиотеке Kodi и СРАЗУ запустить (пауза и громкость
    тут ни при чём — это запуск конкретной серии).

    Args:
        title: Что включить («Темное зеркало», «Симпсоны»).
        area: Комната по-русски («гостиная»). ОБЯЗАТЕЛЬНО передай комнату из
            запроса: без неё запустится на том ящике, который сейчас играет, а
            это может быть другая комната.
        name: Как назвали устройство («коди»); можно пусто.
        episode: Какой эпизод — «13», «S13E09» или пусто (тогда первый в
            библиотеке, то есть следующий по её порядку).
    """
    mark_tool_call()  # honesty.same_block_check: this block now HAS a tool
    bad = _kodi_ready()
    if bad:
        return f"Тул вернул ошибку: поиск по библиотеке невозможен — {bad}."
    wanted = (title or "").strip().strip("«»\"'")
    if not wanted:
        return "Тул вернул ошибку: не сказано, что включить."
    box, eid, why = _box_for(area, name)
    if not box:
        return f"Тул вернул ошибку: {why}."
    try:
        cli = _kodi_client(box.get("seed") or box["host"])
        shows = cli.tvshows()
        movies = cli.movies()
        titles = [s.get("title", "") for s in shows] + [m.get("title", "")
                                                       for m in movies]
        found, score = kodi.best_title(wanted, titles)
        if not found:
            out = (f"В библиотеке ящика {box['short']} нет такого: {wanted} "
                   f"(всего {len(shows)} сериалов, {len(movies)} фильмов).")
            _record("media_control", _ACTION_EVENTS, f"Тул вернул ошибку: {out}",
                    ok=False)
            return out
        target = None
        for s in shows:
            if kodi.norm(s.get("title", "")) == kodi.norm(found):
                target = ("show", s)
                break
        if target is None:
            for m in movies:
                if kodi.norm(m.get("title", "")) == kodi.norm(found):
                    target = ("movie", m)
                    break
        if target is None:
            out = f"Совпадение «{found}» не найдено в списке ящика {box['short']}."
            _record("media_control", _ACTION_EVENTS, f"Тул вернул ошибку: {out}",
                    ok=False)
            return out
        kind, item = target
        what = ""
        if kind == "show":
            eps = cli.episodes(item["tvshowid"])
            want = (episode or "").strip()
            ep = kodi.pick_episode(eps, want) if want else kodi.next_episode(eps)
            if not ep:
                # The user NAMED an episode that is not in the library. Playing
                # a different one without saying so is the wrong side effect
                # this project refuses (field check 03.10.2026: «13x9» on a
                # season-1 library started S01E01 and said nothing).
                out = (f"У сериала «{found}» нет эпизода «{want}» — в "
                       f"библиотеке {kodi.episode_range(eps)}.")
                _record("media_play", _ACTION_EVENTS,
                        f"Тул вернул ошибку: {out}", ok=False)
                return out
            cli.play_episode(ep["episodeid"])
            what = kodi.episode_line(ep)
        else:
            cli.play_movie(item["movieid"])
            what = str(item.get("year") or "")
        # The RESULT is verified through HA, not through Kodi's own answer:
        # Player.Open returns «OK» long before the box is audibly playing.
    except kodi.KodiError as e:
        out = f"ящик {box['short']} {e.says} ({e})"
        _record("media_control", _ACTION_EVENTS, f"Тул вернул ошибку: {out}",
                ok=False)
        return out
    states, _areas = _raw_registry()
    was = ""
    for e in states:
        if e.get("entity_id") == eid:
            was = str(e.get("state", ""))
    # Verified by STATE, not by "the fingerprint changed": Kodi answers
    # Player.Open long before anything is audible (the Jellyfin plugin needs
    # 8-10 s), and re-opening the episode the box is ALREADY playing changes no
    # field at all — which the fingerprint check then reported as a failed
    # start («не смог открыть файл») for a show that was playing (field check
    # 03.10.2026 19:55).
    playing = False
    snap: dict = {}
    for delay in (0.0, 1.0, 2.5, 4.0, 5.0):
        if delay:
            time.sleep(delay)
        snap = _entity_snapshot(eid)
        if snap and str(snap.get("state", "")).lower() in ("playing", "buffering"):
            playing = True
            break
    if not playing:
        out = (f"{eid}: команду на запуск «{found}» ({what}) ящик принял, но "
               "воспроизведение так и не появилось — Kodi не смог открыть "
               "источник.")
        _record("media_play", _ACTION_EVENTS, f"Тул вернул ошибку: {out}",
                ok=False)
        return out
    # Did it open WHAT we asked for? A `playing` state alone cannot tell:
    # Kodi happily starts something else (a plugin redirect, a stale queue) and
    # the tool would then announce «Black Mirror запущено» for The Simpsons —
    # the same confident-wrong-shape the veto exists to stop, one layer up.
    attrs = snap.get("attributes") or {}
    actual = str(attrs.get("media_series_title") or attrs.get("media_title")
                 or "").strip()
    if actual and not _same_title(actual, found):
        out = (f"{eid}: Kodi открыл «{actual}» вместо «{found}» — то, что "
               "просили, не запустилось.")
        _record("media_play", _ACTION_EVENTS, f"Тул вернул ошибку: {out}",
                ok=False)
        return out
    out = (f"{eid}: «{found}» {what} — "
           + ("уже играло" if was in ("playing", "buffering") else "запущено")
           + ", воспроизведение идёт.")
    _record("media_play", _ACTION_EVENTS, out, ok=True)
    return out


def _same_title(a: str, b: str) -> bool:
    """Do two titles name the same thing? (The show as HA reports it vs the
    library's own spelling.)"""
    na, nb = kodi.norm(a), kodi.norm(b)
    return bool(na) and (na == nb or na in nb or nb in na)


@tool
def qdrant_search(query: str, limit: int = 5) -> str:
    """Поиск в долговременной памяти диалога (Qdrant): прошлые реплики и факты о пользователе.

    Args:
        query: Текстовый запрос по-русски (например, о чём пользователь говорил ранее).
        limit: Максимум результатов (по умолчанию 5).
    """
    mark_tool_call()  # honesty.same_block_check: this block now HAS a tool
    # Embed the query with Ollama first (30 s: cold model load happens
    # here). Any failure is returned as text — the model must read
    # "память недоступна" instead of getting a tool exception.
    try:
        emb = _http_json(
            f"{config.OLLAMA_URL}/api/embed",
            {"model": config.EMBED_MODEL, "input": [query]},
            timeout=30,
        )
        vector = emb["embeddings"][0]
    except Exception as e:
        return f"Не удалось получить эмбеддинг: {e}"

    out: list[str] = []
    # Two collections are searched in one call: raw dialogue turns (context)
    # and distilled facts. Errors are SWALLOWED per collection on purpose —
    # a missing/reindexing collection must not make the whole memory tool
    # fail, partial recall is still useful to the assistant.
    for coll in ("voice_turns", "voice_facts"):
        try:
            res = _http_json(
                f"{config.QDRANT_URL}/collections/{coll}/points/search",
                {"vector": vector, "limit": int(limit), "with_payload": True},
            )
        except Exception:
            continue
        # Format per hit for the LLM: facts as one line, turns as
        # «question | answer» with a home-local timestamp (see _MSK).
        for hit in (res.get("result") or []):
            pl = hit.get("payload") or {}
            ts = pl.get("ts")
            when = (
                datetime.fromtimestamp(ts, _MSK).strftime("%d.%m %H:%M") if ts else "-"
            )
            if pl.get("type") == "fact":
                out.append(f"[факт, {when}] {pl.get('text', '')}")
            else:
                out.append(
                    f"[{when}, рут={pl.get('route', '?')}] "
                    f"Вопрос: {pl.get('text', '')} | Ответ: {pl.get('reply', '')}"
                )
    return "\n".join(out[: max(1, int(limit))]) if out else "В памяти ничего нет."


@tool
def hermes_expert(question: str) -> str:
    """Спросить старшего эксперта (Hermes) по сложной диагностике инфраструктуры: сеть, серверы, Docker, устойчивость сервисов.

    Args:
        question: Вопрос по-русски с контекстом проблемы (что не работает, что уже проверено).
    """
    mark_tool_call()  # honesty.same_block_check: this block now HAS a tool
    # Plain non-streaming chat completion against L3 Hermes — no MCP, no
    # tools: the expert only READS infrastructure and advises. The system
    # message below is PROMPT TEXT (byte-identical, see header).
    payload = {
        "model": config.HERMES_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты — старший инженер инфраструктуры умного дома. Отвечай по-русски, "
                    "по делу, с конкретными причинами и шагами проверки. Без воды."
                ),
            },
            {"role": "user", "content": question},
        ],
        "max_tokens": 900,
        "stream": False,
    }
    try:
        data = _http_json(
            f"{config.HERMES_URL}/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {config.HERMES_API_KEY}"},
            timeout=config.EXPERT_TIMEOUT,
        )
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        # Degraded to a string, not an exception: the CodeAgent must be
        # able to SAY "эксперт недоступен" instead of crashing the run.
        return f"Эксперт недоступен: {e}"


@tool
def get_datetime() -> str:
    """Текущие дата и время (часовой пояс дома, Европа/Москва)."""
    mark_tool_call()  # honesty.same_block_check: this block now HAS a tool
    # strftime has no Russian locale in the slim container, so the string
    # is built in English and the weekday / month names are swapped for
    # their Russian forms via str.replace (the weekday is looked up once,
    # the month likewise — hence the repeated strftime calls).
    now = datetime.now(_MSK)
    return now.strftime("%A, %d %B %Y, %H:%M").replace(
        now.strftime("%A"),
        {
            "Monday": "понедельник",
            "Tuesday": "вторник",
            "Wednesday": "среда",
            "Thursday": "четверг",
            "Friday": "пятница",
            "Saturday": "суббота",
            "Sunday": "воскресенье",
        }[now.strftime("%A")],
    ).replace(
        now.strftime("%B"),
        {
            "January": "января",
            "February": "февраля",
            "March": "марта",
            "April": "апреля",
            "May": "мая",
            "June": "июня",
            "July": "июля",
            "August": "августа",
            "September": "сентября",
            "October": "октября",
            "November": "ноября",
            "December": "декабря",
        }[now.strftime("%B")],
    )


@tool
def weather_forecast(text: str) -> str:
    """Прогноз погоды (open-meteo): текущая погода и прогноз на день.

    Используй для любых вопросов про погоду, температуру на улице, ветер и
    осадки. Возвращённый текст — истина в последней инстанции: в final_answer
    переноси его числа и описание погоды БЕЗ изменений (можно сократить
    фразу, но не факты). Если данных нет — так и верни, не выдумывай.

    Args:
        text: Полный вопрос пользователя про погоду («какая завтра погода»,
            «сколько градусов на улице», «нужен ли зонт в пятницу»).
    """
    mark_tool_call()  # honesty.same_block_check: this block now HAS a tool
    # Deterministic path: L1 router /weather (open-meteo) — the model never
    # computes weather itself, it only relays this text. The question is
    # URL-encoded (RU sentences contain «?», «&», quotes). 15 s timeout;
    # BOTH failure branches below still yield a recordable string, which
    # is what vet_weather later compares the spoken answer against.
    try:
        data = _http_json(
            f"{config.ROUTER_URL}/weather?{urllib.parse.urlencode({'text': text})}",
            timeout=15,
        )
    except Exception as e:
        result = f"Не удалось получить прогноз погоды: {e}"
    else:
        # Empty "sentences" is treated as no-data: the explicit
        # «недоступен» phrasing is what vet_weather's failure detector
        # greps for, so the veto can tell a real forecast from a hole.
        sentences = data.get("sentences") if isinstance(data, dict) else None
        result = (
            " ".join(sentences)
            if sentences
            else "Прогноз погоды недоступен (сетевая ошибка или нет данных)."
        )
    # Recorded for the weather veto in honesty.py: the model writes
    # final_answer before reading this output (field case 25.09), so the
    # spoken answer is checked against the recorded forecast afterwards.
    with _EVENTS_LOCK:
        _WEATHER_EVENTS.append(
            {"tool": "weather_forecast", "detail": result[:400]}
        )
    return result


# Manifest handed to CodeAgent: smolagents derives each tool's JSON schema
# AND its description (the docstring — PROMPT TEXT) from these entries, and
# app.py /health echoes the names for health checks. Adding/removing an
# entry changes what the free model is able (and prompted) to do.
TOOLS = [ha_action, ha_read, media_control, media_search, media_play,
         qdrant_search, hermes_expert, get_datetime, weather_forecast]
