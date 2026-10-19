"""jev-router — cascade level-1 semantic router (SSE /route).

PURPOSE
  This is the FAST path of the three-level voice-control cascade:
  ESP32 satellite -> main.py (root repo, port 6050: audio/VAD/STT/TTS)
    -> jev-router (L1, this service, port 8091)
    -> smolagents-worker (L2, smolagents CodeAgent on a free LLM, port 8092)
    -> Hermes (L3 expert) -> Home Assistant via MCP tools.
  L1 answers deterministic commands and queries directly (~0.1 s, no LLM);
  everything it cannot resolve with certainty is escalated to L2, which may
  in turn delegate to L3. The router never guesses a side-effect.

Pipeline per request:
  classify (Ollama embeddings + calibrated cosine)
    -> easy_action  : regex/lexicon slot resolver -> HA MCP intent call
    -> easy_query   : HA states/history/datetime  -> speakable answer
    -> general_qa   : chat proxy (OmniRoute combo -> Hermes failover)
    -> expert       : chat proxy, expert persona (Hermes-first diagnostics)
    -> complex_logic: forwarded to smolagents-worker (L2 CodeAgent)

Escalation rule: any ambiguity (resolver -> None, HA failure, low confidence)
falls through to complex_logic — never a wrong side-effect.

HTTP contract (FastAPI, uvicorn 0.0.0.0:8091, `network_mode: host`):
  POST /route    {"text": str, "session_id": str, "stream_name": str}
                 -> text/event-stream of the events below (see SSE schema)
  GET  /weather?text=... -> {"sentences": [..], "ok": bool} (open-meteo for L2)
  GET  /health   -> {"status", "classifier_warmed", "routes"}
  Port 8091 carries NO authentication: a deliberate user decision — the
  service is reachable on the home LAN only, and an auth layer would add
  latency to every voice turn without protecting anything outside the house.

SSE event schema (one JSON object per `data:` line):
  {"type":"route",   "route":str,"confidence":float,"reason":str}
  {"type":"sentence","text":str}          # speakable TTS chunk
  {"type":"progress","text":str}          # L2 progress heartbeat
  {"type":"done",    "route":str,"elapsed":float}
  {"type":"error",   "message":str}
  A `route` event may be emitted TWICE: once up front and again with
  "complex_logic" when the fast path fails — the caller (main.py) treats
  the second one as the self-healing retry signal.
"""

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import chat_proxy
import config
import history
import memory
import weather
from classifier import (
    Classifier,
    _WARMUP_ATTEMPTS,
    _WARMUP_BACKOFF_S,
    _WARMUP_MAX_S,
)
from ha_client import (
    HAClient,
    _area_phrase,
    confirm_delays,
    dedupe_device_facets,
    describe_entity,
    find_action_targets,
    find_entity,
    find_media_targets,
    media_fingerprint,
    media_no_movement_answer,
    volume_unverifiable,
    MEDIA_ACTIVE,
    MEDIA_PREFER,
    MEDIA_STATE_ANSWER,
    MEDIA_TRANSPORT,
    MEDIA_VOLUME,
)
from resolver import (
    area_of as resolver_area_of,
    RE_MEDIA_NOUN,
    STREAM_DEFAULT_AREA,
    device_hint_from,
    resolve_action,
    resolve_query,
    unresolved_hint,
    action_verb_missing,
    normalize_stt_verbs,
)

# stdout logging: uvicorn does not configure root logging by default, and the
# route decisions logged here are the main debugging surface in `docker logs`.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("router")

# Process-wide singletons: one embedder connection (Ollama) and one HA client
# (whose /api/states + area-map caches are shared by every request). Both are
# closed in `lifespan` below — creating them per request would lose the cache
# and burn a TCP handshake on every voice turn.
classifier = Classifier()
ha = HAClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook: warm the classifier and the memory collections.

    Warmup failures are deliberately non-fatal — `classify()` falls back to the
    deterministic regex action path and the escalation chain keeps the service
    useful even with Ollama or Qdrant down. On shutdown the embedder session
    and the HA session are closed so uvicorn exits without pending sockets.

    It used to be a single `await classifier.warmup()`. Ollama and Qdrant are
    containers on the same host that come up in PARALLEL with this one, so at
    boot "not listening yet" is the normal case; measured 05.10.2026, both
    restarts failed to connect and the first voice command then paid 22 s to
    warm a classifier that should have been ready. The warm-up is now started in
    the BACKGROUND so the port opens immediately, and it retries on its own
    schedule (~2 min) instead of blocking startup on a dependency that is not
    there yet.
    """
    classifier.start_warmup()
    logger.info(
        "classifier warmup: started in background (retrying up to %d attempts)",
        _WARMUP_ATTEMPTS,
    )
    mem_task = asyncio.ensure_future(memory.ensure_collections())
    mem_ok = await _retry_memory(mem_task)
    logger.info("qdrant collections: %s", "ok" if mem_ok else "FAILED (memory off)")
    yield
    await classifier.embedder.close()
    await ha.close()


async def _retry_memory(task) -> bool:
    """Qdrant has the same startup race as Ollama; give it the same patience.

    Returns whether the collections exist. Never raises: memory is optional and
    a missing Qdrant must not keep the router down.
    """
    for attempt in range(1, _WARMUP_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=_WARMUP_MAX_S)
        except asyncio.TimeoutError:
            logger.warning("qdrant ensure attempt %d/%d timed out",
                           attempt, _WARMUP_ATTEMPTS)
        except Exception as e:
            logger.warning("qdrant ensure attempt %d/%d failed (%s)",
                           attempt, _WARMUP_ATTEMPTS, e or type(e).__name__)
            await asyncio.sleep(min(_WARMUP_BACKOFF_S * attempt, _WARMUP_MAX_S))
    return False


app = FastAPI(title="jev-router", lifespan=lifespan)


class RouteRequest(BaseModel):
    """POST /route body.

    `stream_name` is the physical satellite (kitchen/livingroom/...) — the
    resolver uses it as the default room when the utterance names none;
    `session_id` tags the memory write for L2's qdrant_search.
    `room` is an OPTIONAL default-area hint from a source whose stream_name
    is NOT a room (the Telegram source keys history on "tg:<chat_id>").
    It wins over stream_name for the resolver ONLY — the history/memory key
    stays `stream_name or session_id`, so a chat never shares its turn ring
    with a camera.
    """

    text: str
    session_id: str = ""
    stream_name: str = ""
    room: str = ""


def _sse(obj: dict) -> str:
    """Serialize one SSE `data:` line.

    ensure_ascii=False keeps the Russian TTS sentences readable on the wire;
    the double newline terminates the event for the EventSource reader in
    main.py.
    """
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# RU names for the English fields HA returns from llm__GetDateTime — the
# phrase is spoken aloud, so the translation happens here, not in TTS.
_WEEKDAYS_RU = {
    "Monday": "понедельник", "Tuesday": "вторник", "Wednesday": "среда",
    "Thursday": "четверг", "Friday": "пятница", "Saturday": "суббота",
    "Sunday": "воскресенье",
}
# Keyed by the "MM" slice of the ISO date returned by HA (date[5:7]).
_MONTHS_RU = {
    "01": "января", "02": "февраля", "03": "марта", "04": "апреля",
    "05": "мая", "06": "июня", "07": "июля", "08": "августа",
    "09": "сентября", "10": "октября", "11": "ноября", "12": "декабря",
}


def _datetime_phrase(result) -> str | None:
    """llm__GetDateTime payload -> speakable Russian phrase.

    Shape verified live: {"date": "2026-09-24", "time": "17:52:46",
    "timezone": "MSK", "weekday": "Thursday"} — a plain string is also
    accepted for forward compatibility.
    Returns None when nothing speakable could be built; the caller then
    escalates instead of staying silent (a voice turn must always answer).
    """
    if isinstance(result, str):
        return result.strip() or None
    if not isinstance(result, dict):
        return None
    parts: list[str] = []
    date = str(result.get("date", ""))
    # len/date[4] guards distinguish a real "YYYY-MM-DD" from a short or
    # differently formatted value before slicing it apart.
    if len(date) >= 10 and date[4] == "-":
        month = _MONTHS_RU.get(date[5:7], "")
        date_ru = f"{int(date[8:10])} {month} {date[:4]}".strip()
        wd = _WEEKDAYS_RU.get(str(result.get("weekday", "")), "")
        parts.append(f"Сегодня {wd}, {date_ru}." if wd else f"Сегодня {date_ru}.")
    # "17:52:46" -> "17:52": seconds are noise for speech.
    t = str(result.get("time", ""))[:5]
    if t:
        parts.append(f"Сейчас {t}.")
    return " ".join(parts) or None


# --- easy_action execution --------------------------------------------------
# On/off intents only: a wrong side-effect on other domains (broadcast,
# timers, vacuum) is a different class of mistake and keeps the blind path.
_ONOFF_TOOLS = ("intent__HassTurnOn", "intent__HassTurnOff")
# Upper bound of devices one utterance may toggle: "выключи весь свет" must
# not become an unbounded fan-out of MCP calls with per-call timeouts.
_MAX_ONOFF_TARGETS = 5

# ---------------------------------------------------------------------------
# Manual light control vs. the "nobody is here, switch it off" automations.
# ---------------------------------------------------------------------------
# Reported 14.10.2026: the corridor light is driven by a motion sensor, and a
# person working at a desk does not move enough, so the sensor decides the room
# is empty and switches the light off mid-sentence. Wanted: «включи свет» leaves
# the light on AND stops the automation from fighting it; «выключи свет» turns it
# off AND re-arms the automation.
#
# The link is the DEVICE TOKEN, carried in both names: the relay
# `switch.corridor1_light_switch_relay` and the automation "Corridor1 light OFF"
# both say `corridor1`. Deliberately NOT the room and NOT a substring of the
# English word "off":
#   * by room would tie every lamp in a room to that room's automation, so
#     «включи свет» in the living room would disarm the corridor's sensor;
#   * by "off" is an English word-order guess, and this house has 14
#     automations named "<thing> OFF" of several different kinds.
# The token happens to name the entity id of "Corridor1 light ON" as `corridor1`
# while its friendly name carries no "off" at all, so the test picks the OFF one
# and leaves the ON one alone — verified live on both.
#
# Scoped to named rooms. A room with no such automation simply matches nothing
# and nothing is called, but the default is OFF rather than "every room",
# because disarming a sensor changes a home's behaviour and it was asked for in
# exactly one room.
LIGHT_AUTOMATION_ROOMS: tuple[str, ...] = tuple(
    r.strip().lower()
    for r in os.getenv("LIGHT_AUTOMATION_ROOMS", "").split(",")
    if r.strip()
)

# The token is the middle chunk of a Sonoff entity id:
#   switch.<token>_light_switch_relay
_TOKEN_RE = re.compile(r"^switch\.([a-z0-9]+?)_")


def _device_token(entity_id: str) -> str:
    m = _TOKEN_RE.match(entity_id)
    return m.group(1) if m else ""


async def _tie_light_automation(
    areas: dict[str, str], entities: list[str], want_on: bool
) -> list[str]:
    """Disarm (on) or re-arm (off) the motion automations of these lamps.

    Never raises and never changes what is spoken: the light command's own answer
    is about the light, and a failed automation call must not turn a working
    command into an error the user has to listen to. It is logged at WARNING,
    which is the whole cost of it failing.
    """
    if not LIGHT_AUTOMATION_ROOMS or not entities:
        return []
    wanted_areas = {
        areas.get(e, "").strip().lower()
        for e in entities
        if areas.get(e, "").strip().lower() in LIGHT_AUTOMATION_ROOMS
    }
    if not wanted_areas:
        return []
    tokens = {_device_token(e) for e in entities if _device_token(e)}
    if not tokens:
        logger.info("light automation tie: нет токена устройства в %s", entities)
        return []
    try:
        # The AUTOMATION's area must come from `get_entity_areas()`, not from
        # the state: `/api/states` carries no `area_id` in attributes, so filtering
        # on it matched nothing at all and failed in silence. Measured 14.10.2026
        # on the first live run — the light came on and every automation stayed
        # armed, with no log line to say so.
        auto_areas = await ha.get_entity_areas()
        states = await ha.get_states(force=True)
    except Exception as exc:  # the light already answered; this is secondary
        logger.warning("light automation tie: не удалось прочитать состояния: %r", exc)
        return []
    touched: list[str] = []
    seen_off: list[str] = []
    for s in states or []:
        eid = s.get("entity_id", "")
        if not eid.startswith("automation."):
            continue
        attrs = s.get("attributes") or {}
        name = str(attrs.get("friendly_name") or "")
        # Direction-aware, and this is the whole fix.
        #
        # The rule is: DISARM the automation that would UNDO the command the user
        # just gave, and leave armed the one that agrees with it. «включи свет»
        # wants the motion sensor's "light on" automation live and its "light
        # off" one disarmed; «выключи свет» is the mirror.
        #
        # The first version filtered on `"off" in name` and used
        # `want_enabled = not want_on`, so it only ever considered the OFF-named
        # automations. Measured 19.10.2026 10:32, `camera-corridor1`, after a
        # manual «выключи»:
        #
        #     10:32:31  automations ENABLED for manual control  (the two OFF ones)
        #     10:45:48  corridor1 -> off        (the command, delivered)
        #     10:46:23  corridor1 -> on         <- 35 s later, by the sensor
        #     10:47:44  corridor1 -> off
        #     10:46:23  corridor1 -> on
        #     10:48:22  corridor1 -> off
        #
        # The command worked and the light still came back, every 40-80 s. The
        # automations that turn it ON are `automation.corridor1_light_on` and
        # `automation.corridor2_light_on`, both still armed — and the old code
        # could not have found them, because it required the word "off".
        #
        # Whole-word matching, not substring: "on" is inside "motion", "corridor"
        # and "button", and this house has 14 automations.
        opposing = re.compile(r"\boff\b" if want_on else r"\bon\b", re.IGNORECASE)
        if not opposing.search(name):
            continue
        token_in_name = [t for t in tokens if t in name.lower()]
        if not token_in_name:
            continue
        # The area is authoritative and is not overruled by a name that happens to
        # contain the token. An automation with NO area is the one exception, and
        # only when the token is literally in its name: measured 14.10.2026,
        # "Corridor2 light OFF" is unassigned while its relay is in Corridor, and
        # a strict area test silently left the second corridor lamp still
        # fighting the user. Device tokens do not collide across rooms, so the
        # fallback cannot reach into another one.
        area = str(auto_areas.get(eid, "")).strip().lower()
        if not area:
            if token_in_name:
                logger.info(
                    "light automation tie: %s без Area, беру по токену %s",
                    name, sorted(tokens),
                )
            else:
                continue
        elif area not in wanted_areas:
            logger.info(
                "light automation tie: %s пропущена — Area=%r не в %s",
                name, area, sorted(wanted_areas),
            )
            continue
        seen_off.append(name)
        # Every automation that reached this line is, by construction, one that
        # would UNDO the user's command — `opposing` selected nothing else. So
        # the wanted state is the same in both directions: DISARM IT.
        #
        # Nothing here re-arms anything, and that is the correct shape rather than
        # a missing half. The automation that AGREES with the lamp is simply
        # never selected: «включи свет» leaves the motion-ON sensor armed and
        # disarms the motion-OFF one, «выключи свет» is the mirror. That is the
        # behaviour asked for on 14.10.2026, and it needed no "re-arm" branch —
        # the first version invented one, `want_enabled = not want_on`, and it
        # enabled exactly the automations that switch the lamp back to the state
        # the user just took it out of.
        want_enabled = False
        enabled = str(s.get("state", "")).lower() == "on"
        if enabled == want_enabled:
            continue  # already in the state we are asking for
        try:
            await ha.call_service(
                "automation",
                "turn_on" if want_enabled else "turn_off",
                {"entity_id": eid},
            )
            touched.append(eid)
            logger.info(
                "  light automation tie: %s %s (%s) — ручное управление светом",
                "включена" if want_enabled else "выключена", name, eid,
            )
        except Exception as exc:
            logger.warning(
                "  light automation tie НЕ удалась для %s (%s): %r", eid, name, exc,
            )
    if not touched:
        # Silence here is indistinguishable from "the code never ran", which is
        # how the `area_id` bug above survived a live test.
        logger.info(
            "light automation tie: подходящих автоматик нет или они уже в нужном "
            "состоянии (токены=%s, комнаты=%s, найдено=%s)",
            sorted(tokens), sorted(wanted_areas), seen_off or "—",
        )
    return touched


def _err_text(res: dict | None) -> str:
    """Collapse an HA error dict into one bounded string for logs and for the
    `ha_call_failed:` reason that goes to L2.

    `raw` (the tool's full JSON body) is truncated to 300 chars and the whole
    text to 400 so an SSE `reason` field and a log line stay readable — the
    L2 prompt only needs the failure class (INVALID_AREA, ASSISTANT, ...).
    """
    if not res:
        return "unknown"
    err = str(res.get("error") or "unknown")
    raw = res.get("raw")
    if raw:
        err = f"{err}: {str(raw)[:300]}"
    return err[:400]


def _target_rank(e: dict) -> tuple:
    """Sort key deciding WHICH entity speaks for the device in `speak_ok`.

    A real lamp beats auxiliary switches; shorter name = core device;
    entity_id is the final deterministic tie-break so repeated runs call
    the same entity first. Returns (domain_rank, name_len, entity_id).
    """
    eid = e.get("entity_id", "")
    dom = eid.split(".", 1)[0]
    name = (e.get("attributes", {}) or {}).get("friendly_name") or eid
    return (0 if dom == "light" else 1, len(name), eid)


# The state a player is already in for the service we are about to call, and
# the honest answer when nothing has to move: MEDIA_STATE_ANSWER, which lives
# in ha_client next to MEDIA_PREFER (tests import ha_client, never this module
# — there is no fastapi on the host).


# A media service call is ASYNCHRONOUS from HA's point of view: `media_pause`
# on a Kodi answered `[]` instantly and the entity flipped to `paused` two
# seconds later (field check 03.10.2026, le_vlada). Verifying in the same
# breath therefore declared a working command a failure. The wait only runs
# when the reply listed no change, so the common path stays one call.
_MEDIA_CONFIRM_DELAYS = (0.0, 0.6, 1.5)


async def _media_moved(eid: str, before_fp: tuple,
                       delays: tuple[float, ...] = _MEDIA_CONFIRM_DELAYS
                       ) -> dict | None:
    """Re-read `eid` until its fingerprint differs from `before_fp`.

    Returns the new state object, or None when the box stayed put. The wait is
    spent only on an empty `changed` reply, and volume services get the longer
    schedule (`ha_client.confirm_delays`) because their change lands late.
    """
    for delay in delays:
        if delay:
            await asyncio.sleep(delay)
        state = await ha.get_entity_state(eid)
        if state and media_fingerprint(state) != before_fp:
            return state
    return None


async def _execute_media(call) -> tuple[str | None, dict | None]:
    """Transport command for one media_player -> HA REST service call.

    Separate from `_execute_action`'s on/off branch because it CANNOT go
    through MCP: HA's MCP server exposes ten tools and none of them is a
    media intent (verified 03.10.2026), which is why «поставь его на паузу»
    used to cost a 7 s L2 round trip and still answer «не удалось».

    Escalates rather than guessing, exactly like the on/off branch: an unknown
    device gets a spoken refusal, an ambiguity and a dead player get escalated
    to L2, which owns a `media_control` tool of its own.
    """
    service = call.tool.split("__", 1)[1]
    states = await ha.get_states(force=True)
    if not states:
        return None, {"ok": False, "error": "no_states"}
    targets = find_media_targets(
        states, await ha.get_entity_areas(), call.hint, call.args.get("area"),
        MEDIA_PREFER.get(service, ("playing", "buffering")),
    )
    if not targets:
        # [] covers both «no such player» and «several, none playing»; the
        # named case deserves a spoken question, not an escalation.
        if call.hint and not find_media_targets(states, {}, call.hint, None):
            return "Не нашла такого устройства. Может, уточните название?", None
        # A pronoun with nothing playing anywhere is not an ambiguity, it is
        # the answer — escalating only made L2 repeat it in 4 more seconds.
        if (
            service in MEDIA_TRANSPORT
            and not call.hint
            and not call.args.get("area")
            and not any(str(e.get("state", "")).lower() in MEDIA_ACTIVE
                        for e in states
                        if e.get("entity_id", "").startswith("media_player."))
        ):
            return "Ничего не играет.", None
        return None, {"ok": False, "error": "media_target_ambiguous_or_absent"}
    eid = targets[0]["entity_id"]
    state = str(targets[0].get("state", "")).lower()
    known = MEDIA_STATE_ANSWER.get(service, {}).get(state)
    if known:
        # A fragment gets the room in front of it («На кухне ничего не
        # играет»), a whole sentence is spoken as is.
        if not known.endswith("."):
            phrase = _area_phrase(call.args.get("area") or "")
            return f"{phrase} {known}." if phrase else known.capitalize() + ".", None
        return known, None
    data = {"entity_id": eid, **(call.args.get("service_data") or {})}
    res = await ha.call_service("media_player", service, data)
    if not res.get("ok"):
        return None, {"ok": False, "error": _err_text(res)}
    if not res.get("changed"):
        # Empty reply is NOT proof of failure: HA answers before the box has
        # reported the new state (a pause lands ~2 s later) and an
        # attribute-only change (volume_level) may never appear in the list.
        # Poll the player briefly, then decide — this and the worker's
        # media_control compare the same fingerprint.
        #
        # A box reporting NO volume_level is not polled at all: there is no
        # level to move, so no wait can confirm anything and the answer is the
        # same one every idle box gets.
        if service in MEDIA_VOLUME and volume_unverifiable(targets[0]):
            logger.info("media %s on %s: %s reports no volume_level, "
                        "nothing to confirm", service, eid, state)
        else:
            moved = await _media_moved(eid, media_fingerprint(targets[0]),
                                       confirm_delays(service))
            if moved:
                logger.info("media %s on %s moved after an empty reply",
                            service, eid)
                return call.speak_ok, None
            logger.info("media %s on %s changed nothing", service, eid)
        # Nothing moved: a box that was already idle has nothing to play, so the
        # command was about nothing. Said here in 0.2 s rather than escalated —
        # ha_client.media_no_movement_answer owns the rule (and the volume case
        # that used to cost a minute, field check 04.10.2026 21:08).
        spoken = media_no_movement_answer(service, targets[0],
                                          call.args.get("area") or "")
        if spoken:
            return spoken, None
        return None, {"ok": False, "error": f"no_state_change:{eid}"}
    return call.speak_ok, None


async def _execute_action(
    call, text_hint: str = "", stream_name: str = ""
) -> tuple[str | None, dict | None]:
    """Run one easy_action MCP call. Returns (sentence, None) on success or
    an «already in state» answer, (None, error) otherwise — it never claims
    a side-effect that was not confirmed.

    On/off commands resolve the concrete entity from the live registry
    first: the household lamp is a switch relay (the light.* entities are
    unavailable status LEDs), entity names are latin, and a blind
    domain+area match would either miss it or silently no-op on
    `unavailable` states.

    Media transport (`media__*`) is NOT an MCP call at all and leaves here
    immediately: HA exposes no media intent over MCP, so the resolver's tool
    name is a marker for `_execute_media`'s REST service call.
    """
    if call.tool.startswith("media__"):
        return await _execute_media(call)
    states: list[dict] | None = None
    if call.tool in _ONOFF_TOOLS:
        states = await ha.get_states(force=True)
        if not states:
            # `get_states` returns its previous snapshot on failure, and that
            # snapshot is [] on a cold cache — so an empty list means "HA did
            # not answer", NOT "no such device". Falling through to the blind
            # intent here is what produced the worst lie of 04.10.2026:
            # «выключи свет» -> «Выключила» with the lamp still on, because
            # `domain: ["light"]` cannot match a `switch.*_relay` and HA
            # answered success for the AREA plus an `unavailable` WLED.
            logger.error(
                "HA registry unavailable (%s) — refusing to act blind", call.tool
            )
            return None, {"ok": False, "error": "ha_registry_unavailable"}
        if states:
            area_map = await ha.get_entity_areas()
            targets = dedupe_device_facets(
                find_action_targets(
                    states, area_map, call.hint, call.args.get("area")
                )
            )
            # Deterministic refusal: a device absent from the whole registry
            # must never reach L2 — a free model happily «включает» ghost
            # devices (field case: «Включи кафеварку», STT typo + wrong name).
            #
            # **AND ONLY WHEN THE TARGET FINDER AGREES.** Measured 11.10.2026
            # 18:51: «выключи свет в прихожей» answered «Не нашла такого
            # устройства» while `switch.entrance_light_switch_relay` was right
            # there in area `Entrance`. `find_action_targets` finds it; the
            # gate above used `find_entity`, a GLOBAL name lookup with no area
            # and no domain, and that one lands on whatever entity shares the
            # word — here a press-action BUTTON:
            #
            #     sensor.corridor1_light_switch_switch_press_action_switch
            #
            # So the refusal hinged on which same-named entity the registry
            # happened to hand back, while the real lamp sat unconsidered.
            # **A gate that can veto a device the authoritative lookup found is
            # not a gate, it is a second opinion with worse tools.**
            if call.hint and not targets and find_entity(states, call.hint, None) is None:
                # A media CONTENT word is not a device name. «включи следующую
                # серию черного зеркала» reaches here with hint «серию» — the
                # noun matches `RE_MEDIA_NOUN`, but `resolve_action` only takes
                # the media branch when the thing resolves to a `media_player`,
                # which a title never does, so it falls through to
                # `HassTurnOn`. Field case 04.10.2026 19:22: the answer was
                # «Не нашла такого устройства» for a request the L2 tools
                # (`media_search` / `media_play`) exist to serve. Refusing here
                # also blocks escalation, so route it to L2 instead.
                if RE_MEDIA_NOUN.search(text_hint):
                    return None, {
                        "ok": False,
                        "error": (
                            f"media_content_word:{call.hint}: это название "
                            "контента, а не устройство; поиск и запуск — в L2 "
                            "(media_search / media_play)"
                        ),
                    }
                return "Не нашла такого устройства. Может, уточните название?", None
            if not call.args.get("area") and len(targets) > 1:
                # No room in the utterance and several devices match —
                # acting would mean guessing which one.
                #
                # The refusal itself is right; what was WRONG was that it said
                # nothing about WHY the area was missing. Measured 14.10.2026
                # 16:50:40: «включи свет» in camera-corridor1 escalated with
                # exactly this error, and the room name that would have settled
                # it was one table row away. The satellite's own stream is
                # therefore named in the log, and whether the table knows it —
                # because "no room was named" and "this satellite has no default
                # room" are different faults with different fixes, and the error
                # text alone cannot tell them apart.
                logger.warning(
                    "  команда без комнаты, %d целей, stream=%r%s",
                    len(targets), stream_name,
                    ""
                    if (stream_name or "").lower() in STREAM_DEFAULT_AREA
                    else " — У ЭТОГО ПОТОКА НЕТ Area ПО УМОЛЧАНИЮ "
                         "(STREAM_DEFAULT_AREA в resolver.py)",
                )
                return None, {
                    "ok": False,
                    "error": f"ambiguous_no_area: {len(targets)} targets, команда без комнаты",
                }
            if not targets:
                # The registry answered and holds nothing for this device.
                # Blinding the intent is what turns that into a false
                # «Выключила»: HA matches the AREA and answers success.
                return None, {
                    "ok": False,
                    "error": f"no_target:{call.hint or call.tool}: в реестре HA нет такого устройства",
                }
            want_on = call.tool.endswith("HassTurnOn")
            opposite = "off" if want_on else "on"
            # Manual control takes the lamp away from its motion sensor, and puts
            # it back when the user is done. Placed HERE, above every early
            # return, on purpose: «включи свет» spoken while the lamp is ALREADY
            # on is exactly the case that matters — the light is fine and the
            # automation is the thing that will kill it a minute later — and the
            # "уже включено" branch below returns before any state changes.
            #
            # Runs on `targets`, not `todo`: a lamp already in the requested
            # state still has to lose (or regain) its automation, and `todo` is
            # empty in that case by construction.
            if not LIGHT_AUTOMATION_ROOMS:
                pass
            else:
                try:
                    touched = await _tie_light_automation(
                        area_map,
                        [t.get("entity_id", "") for t in targets if t.get("entity_id")],
                        want_on,
                    )
                    if touched:
                        logger.info(
                            "  automation %s для ручного «%s»",
                            "выключены" if want_on else "включены",
                            ", ".join(touched),
                        )
                except Exception as exc:  # a diagnostic side effect, never fatal
                    logger.warning("light automation tie не отработала: %r", exc)
            # Only devices currently in the OPPOSITE state are worth calling;
            # the rest need no MCP call at all — and if nothing needed
            # changing, the "already ..." branch below says so truthfully.
            todo = [
                e for e in targets
                if str(e.get("state", "")).lower() == opposite
            ]
            if targets and not todo:
                # Every matched device is already in the requested state:
                # tell the truth instead of pretending we changed something.
                word = "включено" if want_on else "выключено"
                area_ph = _area_phrase(call.args.get("area") or "")
                sentence = f"{area_ph} уже {word}." if area_ph else f"Уже {word}."
                return sentence, None
            if len(todo) > _MAX_ONOFF_TARGETS:
                return None, {"ok": False, "error": f"too_many_targets:{len(todo)}"}
            if todo:
                todo.sort(key=_target_rank)
                # Logged because the happy path is otherwise silent, and on
                # 10.10.2026 11:51 the kitchen said «Включила» for «включи
                # кофеварку» with NOTHING in this log: no targets, no per-call
                # answer, no read-back. HA's history showed both the power switch
                # and the child lock still `off` for the whole window, so the turn
                # could not be reconstructed after the fact — the artifacts simply
                # did not exist. A verdict that cannot be audited is the same class
                # of problem as a verdict that is wrong.
                logger.info(
                    "on/off %s %s: %d target(s) %s",
                    call.tool.rsplit("__", 1)[-1],
                    call.hint or call.args.get("area") or "",
                    len(todo),
                    ", ".join(
                        f"{e['entity_id']}={e.get('state')}" for e in todo
                    )[:400],
                )
                ok: list[str] = []
                # State BEFORE the calls, so the read-back can tell a device that
                # moved from one that merely accepted the call.
                before: dict[str, str] = {
                    ent["entity_id"]: str(ent.get("state", "")).lower() for ent in todo
                }
                fatal: list[str] = []
                skipped: list[str] = []
                for ent in todo:
                    eid = ent["entity_id"]
                    # Fresh copy per entity: name/domain are pinned to the
                    # registry entry so the matcher targets exactly what was
                    # resolved, and the next iteration starts from clean args.
                    args = dict(call.args)
                    # `name` gets the ENTITY ID, not the friendly_name: HA's
                    # matcher only accepts the concatenated friendly name for
                    # some entities — field check 02.10.2026, «corridor1_light_
                    # switch Relay» and «coffemaker» both answer
                    # MatchFailedReason.NAME while «entrance_light_switch
                    # Relay» matches, so every fast-path corridor command used
                    # to fail here and escalate. The entity id is unique and
                    # always resolves; exposure is still enforced (the same
                    # `assistant='conversation'` filter applies to it).
                    args["name"] = eid
                    args["domain"] = [eid.split(".", 1)[0]]
                    res = await ha.call_tool(call.tool, args)
                    logger.info(
                        "  call %s -> ok=%s claimed=%s",
                        eid, bool(res.get("ok")), res.get("claimed"),
                    )
                    if res.get("ok"):
                        ok.append(eid)
                        continue
                    et = _err_text(res)
                    # **Case-INSENSITIVE, because HA does not spell it the way
                    # this check was written.** Measured 11.10.2026 14:24:13: HA
                    # answered `No matching entities are exposed to the assistant`
                    # — lower case — so `"ASSISTANT" in et` was False, the
                    # unexposed `switch.living_room_light_swith_network_led_switch`
                    # was recorded as FATAL, and a turn in which the real relay
                    # call had already succeeded (`ok=True`) was thrown away and
                    # escalated to L2. An unexposed device is not a failure, it is
                    # a device that was never ours to control.
                    if "assistant" in et.lower():
                        skipped.append(eid)
                        logger.info("target not exposed to assistant: %s", eid)
                    else:
                        fatal.append(f"{eid}: {et}")
                # `ok` is HA saying the CALL was accepted, not that a device moved — and
                # on this path nothing re-read the state, so «включи кофеварку» answered
                # «Включила» with `switch.coffemaker` untouched (measured 06.10.2026
                # 18:21 UTC, `last_changed` still hours old). The same verification
                # exists on the BLIND path below and was never added here: the same
                # class of lie this file already documents for the other one.
                if ok and not fatal:
                    # **TWO-PART ANSWER, by the user's decision 10.10.2026.**
                    # Measured that day: HA served `off` for five seconds while the
                    # appliance was already on (the user saw the indicator lit), so
                    # a single sentence is either a lie or a five-second silence.
                    # Speaking «Включаю.» at once and confirming afterwards is the
                    # only shape that is honest AND fast.
                    #
                    # The first read is at `_FIRST_READ_S` (0.8 s, measured) rather
                    # than the whole settle, so the common case still answers in one
                    # short utterance — «Включила» — and only the genuinely unsettled
                    # case pays for the two-part form.
                    #
                    # **PROVED devices only.** `confirm_on_off_one_read` returns
                    # two lists and the difference is the whole fix: entities with a
                    # power sensor are proved by what the appliance draws, entities
                    # without one cannot be proved at all and come back as `unproven`.
                    # A one-word answer needs the first list alone — speaking
                    # «Включила» on an unproven read is precisely the echo measured
                    # on `switch.coffemaker` at 16:03:55 on 16.10.2026.
                    ack = "Включаю." if want_on else "Выключаю."
                    target = "on" if want_on else "off"
                    moved, unproven = await confirm_on_off_one_read(ok, target)
                    if moved and not [e for e in ok if e not in moved]:
                        logger.info(
                            "  verify want=%s before=%s moved=%s — proved by the "
                            "power sensor on the first read",
                            target, before, moved,
                        )
                        return call.speak_ok, None
                    logger.info(
                        "  verify want=%s before=%s first read moved=%s unproven=%s — "
                        "speaking the acknowledgement and confirming after the settle",
                        target, before, moved, unproven or [],
                    )
                    return ack, None, _PendingVerify(ok, before, want_on, moved)
                    # Nothing moved. Claiming it did is worse than an honest refusal, so
                    # this escalates with the real blocker instead of speaking a success the
                    # user can plainly see is false.
                    logger.error(
                        "HA accepted %s for %s but no entity reached the requested "
                        "state: before=%s",
                        call.tool, ok, before,
                    )
                    return None, {
                        "ok": False,
                        "error": (
                            "unverified_side_effect: Home Assistant принял вызов "
                            f"{call.tool} для {', '.join(ok)}, но состояние не изменилось"
                        ),
                        "raw": {"claimed": ok, "before": before},
                    }
                if not ok and not fatal and skipped:
                    # Every entity that NEEDED a change was an unexposed facet
                    # (the "Network led switch" of the Sonoff relays), while
                    # the controllable one is already in the requested state
                    # per the registry — say that truth instead of escalating
                    # on «сущность не открыта голосовому ассистенту»: «включи
                    # свет в прихожей» with the light already on used to fail
                    # here, because the LED facet was still in `todo`.
                    if any(str(e.get("state", "")).lower() != opposite
                           for e in targets):
                        word = "включено" if want_on else "выключено"
                        area_ph = _area_phrase(call.args.get("area") or "")
                        sentence = (f"{area_ph} уже {word}." if area_ph
                                    else f"Уже {word}.")
                        return sentence, None
                fails = fatal + [
                    f"{e}: сущность не открыта голосовому ассистенту"
                    for e in skipped
                ]
                errd: dict = {"ok": False, "error": "; ".join(fails)[:400]}
                if ok:
                    # Verified side-effects before the failure — L2 must not
                    # deny them ("never promise" is about unproven actions).
                    errd["done"] = ", ".join(ok)
                return None, errd
    # Blind path: non-on/off intents (timers/vacuum/broadcast/light-set) — the HA
    # matcher decides. On/off never lands here any more: an unreachable registry
    # and an empty target set both refuse above.
    res = await ha.call_tool(call.tool, call.args)
    if res.get("ok"):
        if states is None:
            states = await ha.get_states(force=True)
        if _touched_a_usable_entity(res, states):
            return call.speak_ok, None
        # `ok` is HA saying the CALL was accepted, not that a device moved.
        # A domain that cannot match the real device still yields success for
        # the AREA alone. Claiming the side effect here is how «включи свет»
        # answers «Сделала» with the lamp untouched — measured 04.10.2026.
        logger.error(
            "HA reported success but touched no usable entity: tool=%s claimed=%s",
            call.tool, res.get("claimed"),
        )
        return None, {
            "ok": False,
            "error": "unverified_side_effect: HA принял вызов, но ни одно доступное устройство не изменилось",
            "raw": res.get("claimed"),
        }
    return None, res


def _on_off_verdict(
    moved: list[str], unmoved: list[str], unproven: list[str] | None = None,
) -> str:
    """What to say AFTER the acknowledgement. Empty means say nothing further —
    which is the right answer in the common case, where the device did what was
    asked and a second «Включила» would be noise.

    **`unproven` is a third outcome, not a flavour of failure.** It means there
    was no instrument able to see whether the device moved, so the room cannot
    honestly say either way. Saying «Не получилось» there would invent a failure,
    and saying nothing would imply success — both are claims it cannot support.
    Measured 16.10.2026: HA answers `switch.turn_on` in ~20 ms and the appliance
    acted 2 min 56 s later, so «не получилось» at 2.5 s would have been a lie in
    the other direction.
    """
    unproven = unproven or []
    if unproven and not unmoved:
        # **SAY NOTHING.** This used to answer «Пока не подтвердилось.» and it
        # was wrong twice over — measured 11.10.2026 18:38 on the living-room
        # relay, which has NO power sensor and therefore can never be proved:
        #
        #     HassTurnOn  sent 18:38:27.968 -> relay physically `on`  18:41:23  (2:55)
        #     HassTurnOff sent 18:38:46.345 -> relay physically `off` 18:41:42  (2:56)
        #
        # HA did NOT write the state optimistically here: the relay sat at `off`
        # for the whole three minutes and only read `on` when it physically acted.
        # So `state` is a TRUTHFUL BUT LATE signal for this device, and the room
        # was right to refuse to claim success at 2.5 s — while the sentence it
        # spoke instead told the user something was wrong, for a lamp that was
        # about to work.
        #
        # It also repeated on EVERY light command in the house, which is what
        # «у тебя проблема с методами подтверждения» actually was: not a device
        # fault and not a flaky check, but a verdict phrased as a complaint
        # about a device we simply cannot watch. The acknowledgement «Включаю.»
        # has already been spoken and is true; adding a second, worse sentence
        # on top of it is the only thing to remove. `_late_verify` still re-reads
        # the state afterwards and LOGS what really happened.
        return ""
    if moved and not unmoved:
        return ""
    if moved and unmoved:
        return (
            f"Частично получилось: {', '.join(moved)} — "
            f"{', '.join(unmoved)} не сработали."
        )
    return "Не получилось: устройство не переключилось."


# When HA→device actually takes this long, a 2.5 s window is not a check, it is
# a guess. Measured twice on the living-room relay (18:38 → 18:41:23 and
# 18:38:46 → 18:41:42, 2:55 and 2:56) and twice by hand on the coffee maker
# (16:16:20 → 16:19:16 and 16:16:32 → 16:19:31, 2:56 and 2:59). Four samples,
# one number: **the queue is a property of the HA→device path, not of any one
# appliance**, so it is a module constant rather than a per-device guess.
#
# The re-read at that point is what makes the log trustworthy again — it is the
# only moment at which `state` carries information about this command.
_LATE_VERIFY_S = 185.0


class _PendingVerify:
    """Carried out of `_execute_action` so the caller can speak the
    acknowledgement first and confirm after the remaining retries.

    A plain tuple would do, and this is a tuple: it exists to make the contract
    readable at the call site, because the failure mode being fixed is a verdict
    with no audible counterpart and a three-field return is easy to misread.
    """

    __slots__ = ("ids", "before", "want_on", "moved", "unproven_at_settle")

    def __init__(self, ids, before, want_on, moved):
        self.ids, self.before, self.want_on, self.moved = ids, before, want_on, moved
        self.unproven_at_settle: list[str] = []

    def verify_later(self) -> None:
        """Re-read the state when the answer actually exists, and LOG it.

        **Deliberately detached and silent.** It must not hold the turn open —
        the player waits 120 s for the `None` sentinel and this fires at 185 s —
        and it must not speak: the acknowledgement has already been given, and a
        spoken verdict three minutes later would arrive after the user has moved
        on.

        What it buys is the thing the field lost today: with the state re-read at
        the right moment, the log finally records whether the device moved, which
        is what a later change to the spoken verdict would have to be built on.
        Without it every log ends at 2.5 s with the state still stale, and the
        real outcome is invisible — which is how «не подтвердилось» went on
        claiming failure for lamps that were working.
        """
        asyncio.create_task(self._late_verify())

    async def _late_verify(self) -> None:
        target = "on" if self.want_on else "off"
        try:
            await asyncio.sleep(_LATE_VERIFY_S)
            fresh = await ha.get_states(force=True) or []
            by_id = {x.get("entity_id"): x for x in fresh}
            moved, still = [], []
            for eid in self.ids:
                ent = by_id.get(eid)
                st = str((ent or {}).get("state", "")).lower()
                (moved if st == target else still).append(eid)
            logger.info(
                "  verify late (+%.0fs) want=%s moved=%s not=%s unproven_at_2s=%s%s",
                _LATE_VERIFY_S, target, moved, still, self.unproven_at_settle or [],
                "" if moved else "  ← THE DEVICE NEVER MOVED",
            )
        except Exception as e:
            logger.warning("  verify late failed: %r", e)

    async def settle(self) -> tuple[list[str], list[str], list[str]]:
        target = "on" if self.want_on else "off"
        moved, unproven = await _settle_on_off(
            self.ids, target, list(self.moved), (1.5, 2.5)
        )
        unmoved = [
            e for e in self.ids
            if e not in moved and e not in unproven
        ]
        self.unproven_at_settle = list(unproven)
        logger.info(
            "  verify settled want=%s moved=%s unmoved=%s unproven=%s",
            target, moved, unmoved, unproven,
        )
        return moved, unmoved, unproven


async def _confirm_on_off_moved(
    ids: list[str], before: dict[str, str], want_on: bool
) -> tuple[list[str], list[str]]:
    """Which of these entities actually reached the state that was asked for?

    Measured 06.10.2026 18:21 UTC: `switch.coffemaker` was `off`, «включи
    кофеварку» resolved to it, HA accepted the call, the room said «Включила» —
    and the relay's `last_changed` was hours old. `ok` from HA means the CALL was
    accepted; a relay that does not answer, or a device that ignores the command,
    looks identical from here.

    **Read-back with a short retry**, because a relay is not instantaneous — but the
    FIRST read is deliberately late, see the schedule below.

    **`reached` means exactly `state == target`, and nothing else.** The first
    version also accepted `before != now` "as evidence it moved", which is how a
    safeguard becomes a no-op: for a BINARY on/off a state change means the device
    went the OTHER way, and an entity missing from the registry reads as an empty
    state that therefore also "changed". Probed live against the running router
    before it was trusted:

        coffee maker on -> off, did not move  ->  moved=['switch.coffemaker']  (wrong)
        a device that does not exist           ->  moved=['switch.нет_такого']  (wrong)

    So the first version would have accepted the very lie it was written to catch.
    An entity absent from the registry, `unavailable` or `unknown` is never
    reached — there is nothing there that could have moved.
    """
    target = "on" if want_on else "off"
    moved, _ = await confirm_on_off_one_read(ids, target)
    moved, _ = await _settle_on_off(ids, target, moved, (1.5, 2.5))
    return moved, [e for e in ids if e not in moved]


# How long to wait before the FIRST read-back.
#
# Measured 10.10.2026 on `switch.coffemaker`, with the user's explicit permission
# to toggle it: `switch` state `off` at t=+0.26 s while `sensor.coffemaker_power`
# still read 1072 W, and 0 W by t=+0.77 s. Home Assistant writes the entity state
# OPTIMISTICALLY on `turn_on/off` and `/api/states` serves it until the next poll,
# so a read before ~0.8 s is reading the value HA itself just wrote.
#
# 0.8 s is this device's number, not a constant. It is a compromise with a
# measured cost on each side, and both sides were chosen rather than guessed:
#   - shorter and the common case reads HA's own optimistic write, so a device
#     that did NOT move is reported as having moved — the worse error;
#   - longer and every on/off turn pays silence before it answers.
_FIRST_READ_S = 0.8


# How a device is proved to have moved, when its state cannot be trusted.
#
# **MEASURED 16.10.2026, twice by hand, on `switch.coffemaker`:**
#
#     16:16:20  switch.turn_on   -> HTTP 200 IMMEDIATELY
#     16:19:16  the appliance actually drew current      <- 2 min 56 s later
#     16:16:32  switch.turn_off  -> HTTP 200 IMMEDIATELY
#     16:19:31  the current stopped                      <- 2 min 59 s later
#
# And the same delay explains the field case an hour earlier: the router said
# «Включила» at 16:03:55 and «Выключила» at 16:04:07, while the appliance ran
# 16:06:47-16:06:59 — 2 min 52 s after the room had already answered.
#
# HA's recorder holds NO transition at 16:03:55 or 16:04:07 either, so the value
# the read-back saw was never a recorded state change: it was the value HA writes
# OPTIMISTICALLY when it accepts the call. **Reading `state` back is reading our
# own request echoed.** It returns "the target" for a device that has not moved
# and will not move for minutes, and no retry schedule can fix that — the wrong
# value stays the right value for as long as the device takes.
#
# A POWER sensor is the only instrument that cannot lie this way: it reports what
# the appliance is drawing, not what we asked for. Measured above, it answered in
# ~2 s after the physics, and `sensor.coffemaker_power` read 1074 W on a running
# coffee maker against 0 W on a stopped one.
#
# So: where a power sensor exists, the PHYSICS decides. Where none exists, the
# state read is all there is, and it is logged as unproven rather than believed.
_POWER_SENSOR_SUFFIX = "_power"
# Below this the appliance is drawing nothing measurable. 1074 W on vs 0 W off,
# so the midpoint is not a judgement call — but it is a number, and it is named.
_POWER_ON_W = 5.0


def _power_sensor_for(entity_id: str, states: list[dict]) -> str | None:
    """The power sensor paired with this switch, or None.

    Matched by name (`switch.coffemaker` -> `sensor.coffemaker_power`) and only
    when it really is a power sensor. There is no device_id on a `/api/states`
    row, so name pairing is the instrument available; it is deliberately
    conservative — a wrong pairing would mean believing a number about the wrong
    appliance, which is worse than admitting there is none.
    """
    if not entity_id.startswith("switch."):
        return None
    base = entity_id.split(".", 1)[1]
    want = "sensor." + base + _POWER_SENSOR_SUFFIX
    for ent in states or []:
        if ent.get("entity_id") != want:
            continue
        attrs = ent.get("attributes") or {}
        if attrs.get("device_class") != "power":
            return None       # same name, not a power reading: do not trust it
        return want
    return None


def _physics_agrees(ent: dict | None, target: str) -> bool | None:
    """Does the power sensor support the target state? None = no opinion.

    None is returned whenever the number cannot be read, and the caller must
    treat that as UNPROVEN rather than as success — the whole point of the
    exercise. An `unavailable`/`unknown` sensor is the absence of evidence.
    """
    if not ent:
        return None
    raw = str(ent.get("state", "")).strip().lower()
    if raw in ("unknown", "unavailable", "none", ""):
        return None
    try:
        watts = float(raw)
    except ValueError:
        return None
    return watts >= _POWER_ON_W if target == "on" else watts < _POWER_ON_W


async def confirm_on_off_one_read(
    ids: list[str], target: str, delay: float = _FIRST_READ_S,
) -> tuple[list[str], None]:
    """ONE read-back after `delay`. Split out so the caller can speak before the
    remaining retries — which is the whole point of the two-part answer.

    **Proves devices by their power sensor where one exists** (see the note above
    for the measurement that forced it). Entities without a paired power sensor
    fall back to the state read, and those are returned SEPARATELY so the caller
    can say so out loud instead of passing an echo off as a movement.
    """
    if delay:
        await asyncio.sleep(delay)
    fresh = await ha.get_states(force=True) or []
    by_id = {x.get("entity_id"): x for x in fresh}
    reached: list[str] = []
    unproven: list[str] = []
    for eid in ids:
        sensor_id = _power_sensor_for(eid, fresh)
        if sensor_id:
            if _physics_agrees(by_id.get(sensor_id), target) is True:
                reached.append(eid)
                continue
            # NOT reached, and specifically NOT called a failure either.
            #
            # The measured lag on this appliance is 2 min 56 s, and this read is
            # at most 2.5 s, so a sensor that has not started drawing yet is
            # indistinguishable from a device that never will. Reporting that as
            # «Не получилось» would be the same class of lie as the optimistic
            # read, pointed the other way — and it is the direction a person
            # notices, because the machine starts brewing two minutes later.
            unproven.append(eid)
            continue
        ent = by_id.get(eid)
        if not ent:
            continue              # not in the registry: nothing to have moved
        # **No power sensor, and this is the whole point of this function.**
        #
        # There is no instrument behind this entity, so a read that does not
        # show the target is NOT an observation of failure — it is the absence
        # of one. Measured twice on the living-room relay, 11.10.2026: sent
        # 18:38:27.968, physically `on` at 18:41:23; sent 18:38:46.345,
        # physically `off` at 18:41:42. The relay sat at its old value for the
        # whole three minutes and only read the new one when it physically
        # acted — so for these lamps `state` is truthful but LATE, and HA is
        # not writing anything on our behalf either.
        #
        # The first version only called an un-reached sensorless entity
        # `unproven` when the state happened to already equal the target, and
        # let it fall through to `unmoved` otherwise — which is what produced
        # the complaint report on 19.10.2026, corridor, 10:32:
        #
        #     verify settled want=off moved=[] unmoved=['switch.corridor1_light_switch_relay']
        #     speaking: 'Не получилось: устройство не переключилось.'
        #
        # Both relays were `off` within three minutes (`last_changed` 10:35:54
        # and 10:35:49 against a 10:32:31 call). The room contradicted the
        # switch, in front of the user, about a command that had worked.
        #
        # `_on_off_verdict` already answers `""` for an all-unproven read — the
        # comment there records the same three-minute measurement. The verdict
        # was never the bug; the classification above it was.
        unproven.append(eid)
    return reached, (unproven or None)


async def _settle_on_off(
    ids: list[str], target: str, moved: list[str], delays,
) -> tuple[list[str], list[str]]:
    """The remaining retries, with gaps measured from the previous read.

    Returns `(moved, unproven)` — and the split is the point. An entity with no
    power sensor can never be PROVED, so it is reported apart from the ones that
    were: reporting it as unmoved would be a false alarm, and reporting it as
    moved would be the echo this whole exercise exists to stop.
    """
    prev = _FIRST_READ_S
    unproven: list[str] = []
    for delay in delays:
        await asyncio.sleep(max(0.0, delay - prev))
        prev = delay
        reached, unproved = await confirm_on_off_one_read(ids, target, delay=0.0)
        for eid in reached:
            if eid not in moved:
                moved.append(eid)
        unproven = list(unproved or [])
        if moved:
            break
    # ONLY the entities that had no instrument are unproven. An entity whose
    # sensor contradicted the target, or that is absent from the registry, was
    # observed and did not comply — that is a failure, and calling it "we could
    # not check" would be the soft direction. The `or still` that stood here
    # relabelled every failure as unprovable, which is how a partial fan-out
    # came out as «нечем проверить».
    return moved, unproven


def _touched_a_usable_entity(res: dict, states: list[dict] | None) -> bool:
    """True only if HA named at least one real, available entity as done.

    `{"type": "area"}` in HA's success list means the intent matched a ROOM,
    not a device — with no `entity` entry nothing was addressed at all. An
    entity that is `unavailable` cannot have changed either.
    """
    claimed = res.get("claimed") or []
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
        # Absent from the live registry is NOT evidence of availability: HA
        # happily names entities that are filtered out of /api/states or that
        # live in an integration we cannot read. Absent => unverified.
        live.get(eid) is not None
        and live.get(eid) not in ("unavailable", "unknown", "none", "")
        for eid in ids
    )


async def _run_worker(
    text: str,
    session_id: str,
    stream_name: str,
    context: str = "",
    hist: str = "",
):
    """Forward smolagents-worker (L2) SSE events; yields router-level dicts.

    The worker speaks the same event schema, so its `sentence`/`progress`
    objects are re-emitted verbatim to the client. `context` is the decoded
    failure reason from a fast-path attempt (or the near-miss device hint)
    — that is what makes the L2 retry self-healing instead of a blind
    repeat of the rejected call; `hist` is the formatted ring of the last
    finished turns of this satellite (history.block), which is what lets
    the model resolve «выключи её» without inventing a device. The payload
    field is named `history` — the request schema of the worker.

    Raises RuntimeError on a non-200 response (surfaced as an SSE `error`
    event by _handle); non-JSON `data:` lines are skipped, and `[DONE]`
    terminates the upstream stream.
    """
    # total=WORKER_TIMEOUT caps the whole agent run, sock_connect=5 fails fast
    # when the worker is not listening (it may be restarting between turns).
    timeout = aiohttp.ClientTimeout(total=config.WORKER_TIMEOUT, sock_connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(
            f"{config.WORKER_URL}/invoke",
            json={"text": text, "session_id": session_id,
                  "stream_name": stream_name, "context": context,
                  "history": hist},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"worker http {resp.status}: {body[:200]}")
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue


async def _handle(req: RouteRequest):
    """Async generator of router SSE events for one request.

    Event order: one `route` event (possibly re-emitted as `complex_logic`
    after an escalation), zero or more `sentence`/`progress` events, then a
    final `done` carrying the *effective* route and elapsed seconds. Even an
    unhandled handler exception produces an `error` event and still reaches
    `done` — the caller must never be left waiting on a silent stream.

    `reply_parts` accumulates only what the user will actually hear (no
    progress heartbeats) and is written to memory right before `done`.
    """
    t0 = time.monotonic()
    text = (req.text or "").strip()
    # Everything speakable across all branches; feeds the fire-and-forget
    # memory write (route metadata + what was really said).
    reply_parts: list[str] = []

    if not text:
        # STT can produce an empty string; answer immediately instead of
        # letting classify() burn an embedding round-trip on nothing.
        yield _sse({"type": "error", "message": "empty_text"})
        return

    # --- L0: semantic classification -------------------------------------
    # Repair the transcription BEFORE anything decides anything.
    #
    # Measured 06.10.2026: Whisper writes «выключи» as «куча», «кучи», «ключи»
    # or «выкл юч». Normalising inside `resolve_action` was NOT enough, because
    # the classifier runs first and its confidence decides the route: «куча свет»
    # scored 0.75, fell under the 0.85 gate and went to L2 with the resolver never
    # consulted at all, while «выкл юч свет» happened to score 0.86 and took the
    # deterministic path. Same defect, opposite outcome, decided by how close to the
    # threshold the embedding happened to land.
    #
    # So the repair belongs where the transcript ENTERS. **A transcription fix is
    # not a routing concern and must not sit behind a confidence score.**
    repaired = normalize_stt_verbs(text)
    if repaired != text:
        logger.info(
            "STT repaired before routing: %r -> %r", text[:60], repaired[:60]
        )
        text = repaired

    decision = await classifier.classify(text)
    route, confidence, reason = decision.route, decision.confidence, decision.reason
    logger.info(
        "route=%s conf=%.2f reason=%s stream=%s text=%r",
        route, confidence, reason or "-", req.stream_name, text[:80],
    )

    # --- Escalation hooks before emitting the route event ----------------
    # The resolvers are pure and cheap (regex only), so they run HERE as a
    # pre-flight: a downgrade is visible in the very first route event and
    # the caller never sees an easy_* verdict it would have to roll back.
    # `ctx` explains to L2 WHY L1 could not answer itself and `hist` holds the
    # last finished turns of this satellite; every worker call below passes
    # both, because a bare utterance made the model guess a device name and a
    # pronoun a whole room (field case 28.09.2026).
    ctx = ""
    hist = history.block(req.stream_name or req.session_id)
    # Default-room hint for the resolvers: an explicit /room binding (a
    # source whose stream_name is not a room) beats the stream name. Passed
    # ONLY to the resolvers' default lookup — never used as the history key
    # above (that double duty would merge a Telegram chat's turns into the
    # kitchen camera's ring).
    area_hint = req.room or req.stream_name
    if route == "easy_action":
        call = resolve_action(text, area_hint)
        if call is None:
            # A follow-up carries the ACTION but not the TARGET: after «включи
            # свет», «а теперь выключи» is a verb with nothing to apply it to, and
            # resolve_action returns None — so the room escalated a perfectly
            # clear command. Measured 05.10.2026: exactly that sentence, answered
            # with `resolver_ambiguous` after 6 s of L2. Retry once with the
            # DEVICE the previous turn was about, and only the device.
            #
            # Only the noun is borrowed, never the verb: appending the previous
            # text whole makes `resolve_action("а теперь выключи включи свет")`
            # return HassTurn**On**, because the resolver scans for any verb and
            # finds «включи» in the borrowed words. Measured, not assumed.
            hint = device_hint_from(history.last_text(req.stream_name or req.session_id))
            if hint:
                borrowed = f"{text} {hint}".strip()
                retry = resolve_action(borrowed, area_hint)
                if retry is not None:
                    logger.info(
                        "resolver used the previous turn's device %r: %r -> %s",
                        hint, text[:60], retry.tool,
                    )
                    text, call = borrowed, retry
        if call is None:
            # LAST RESORT, and it lives HERE rather than inside `resolve_action`
            # for a reason that cost a regression test to find: inside the
            # resolver this fires before anything can tell an unknown DEVICE WORD
            # from a sentence with no word at all, and «Выключи кашеварку» —
            # the field case from 28.09.2026 — does not match `_DEVICE_NOUN`, so
            # it looked exactly like «выключи» and the fallback turned a coffee
            # maker into a light.
            #
            # Here, both of those have already been tried and failed:
            #   * the borrow above, which is what a follow-up needs, so
            #     «а теперь выключи» still borrows «кофеварку» from the last turn
            #     and never reaches this line;
            #   * `unresolved_hint`, non-empty precisely when the sentence DOES
            #     name a device the resolver only has a near-miss spelling for.
            #
            # Both empty means the sentence named nothing at all. Measured
            # 19.10.2026 10:17, corridor camera:
            #
            #     10:17:37  Whisper OK: 'выключи'
            #     10:17:38  route=easy_action conf=0.97
            #     10:17:38  resolver ambiguous -> complex_logic: 'выключи'
            #
            # The audio path was clean — pip 0.55 s, utterance ended in 2.48 s via
            # `pause` — and the turn died here anyway. The escalation was worse
            # than slow: L2 spent nine steps and ~45k tokens on a bare verb and
            # ended reading `media_player` in an unrelated room («владиная
            # комната»), having touched no light. Fifty seconds of monologue and
            # no switch.
            #
            # Scoped to lights, and the area is always pinned: a room with no
            # lights produces no targets downstream and the ghost-device
            # protection refuses the turn rather than substituting something.
            # Measured the same day: `find_action_targets('свет', 'Kitchen')` is
            # 0 targets, against Corridor 4, Entrance 2, Living Room 2 — so the
            # kitchen keeps escalating and never guesses the coffee maker.
            near = unresolved_hint(text)
            if not hint and not near and area_hint:
                lit = resolve_action(f"{text} свет", area_hint)
                if lit is not None:
                    logger.info(
                        "resolver last resort: no device named, room known -> "
                        "%s: %r -> %s %s",
                        req.stream_name, text[:60], lit.tool, lit.args,
                    )
                    text, call = f"{text} свет", lit
        if call is None:
            logger.info("resolver ambiguous -> complex_logic: %r", text[:80])
            route, reason = "complex_logic", "resolver_ambiguous"
            ctx = unresolved_hint(text)  # «кашеварку» -> «кофеварка», or ""
    elif route == "easy_query":
        q = resolve_query(text, area_hint)
        if q is None:
            logger.info("query resolver ambiguous -> complex_logic: %r", text[:80])
            route, reason = "complex_logic", "query_resolver_ambiguous"
    # Every escalation carries the canonical room name: the RU word is NOT an
    # HA area, and an intent fed «гостиная» answers INVALID_AREA (field case
    # 03.10.2026 18:21, where the model then spent a step guessing device
    # names instead of reading a real blocker).
    _room, _room_src = resolver_area_of(text, area_hint)
    if _room and _room_src == "explicit":
        ctx = (ctx + " " if ctx else "") + (
            f"Комната в Home Assistant называется «{_room}» — используй это "
            "имя, не переводи."
        )

    # A device command with no verb we recognise is NOT an action for L2 to
    # interpret. Measured 06.10.2026: Whisper wrote «выключи свет» as «куча
    # свет», confidence fell to 0.75, and L2 — handed that with nothing said
    # about the verb — answered «Включила свет». It turned the lamp ON when asked
    # to turn it off, which is the worst thing this system can do to a room.
    #
    # `resolve_action` returning None is not enough protection on its own: this
    # note has to be attached for every escalation, including the classifier's
    # `low_confidence` path, which never calls the resolver at all. **Silence
    # about a missing verb reads as "the direction is yours to choose".**
    if route == "complex_logic" and action_verb_missing(text):
        ctx = (ctx + " " if ctx else "") + (
            "В фразе есть название устройства, но глагол действия не распознан "
            f"(в исходном виде: {text[:60]!r}). НЕ угадывай направление действия и "
            "не вызывай ha_action: спроси, что нужно сделать — включить или "
            "выключить. Если действие нельзя определить, это вопрос, а не команда."
        )
        reason = (reason + "+verb_missing").strip("+")

    # The (usually single) route event: emitted before any execution so the
    # caller can log/telemetry the verdict immediately.
    yield _sse(
        {"type": "route", "route": route, "confidence": round(confidence, 3),
         "reason": reason}
    )

    try:
        # --- easy_action: HA MCP intent call ------------------------------
        if route == "easy_action":
            # Resolved a second time (pure regex, no I/O): the pre-flight
            # already turned this branch off if the resolver said None.
            call = resolve_action(text, area_hint)
            outcome = await _execute_action(call, text, req.stream_name)
            # Three-tuple only when the on/off path could not settle on its first
            # read: the acknowledgement is spoken now and the verdict after the
            # remaining retries, so an unsettled device produces a short utterance
            # and then the truth rather than five seconds of silence.
            pending = outcome[2] if len(outcome) > 2 else None
            sentence, err = outcome[0], outcome[1]
            if sentence:
                # Log what the user will actually hear. A side effect that was
                # claimed but never performed is otherwise invisible: on
                # 04.10.2026 that is exactly how «Выключила» went out with the
                # lamp still on, and the router log showed a clean easy_action
                # with no error anywhere.
                logger.info("speaking: %r", sentence)
                reply_parts.append(sentence)
                yield _sse({"type": "sentence", "text": sentence})
                if pending is not None:
                    moved, unmoved, unproven = await pending.settle()
                    verdict = _on_off_verdict(moved, unmoved, unproven)
                    if unproven:
                        # Nothing to say, but the outcome is still knowable at
                        # +185 s — and today it was not knowable at all, which is
                        # how a working lamp was reported as unconfirmed.
                        pending.verify_later()
                    if verdict:
                        logger.info("speaking: %r (confirmation)", verdict)
                        reply_parts.append(verdict)
                        yield _sse({"type": "sentence", "text": verdict})
            else:
                # Side-effect failed (no matching entity, INVALID_AREA,
                # offline device...) -> escalate with the concrete error
                # rather than lie to the user. L2 gets the context so it
                # can correct the call instead of blind-repeating it.
                err_text = _err_text(err)
                logger.warning(
                    "HA call failed (%s): %s — escalating", call.tool, err_text
                )
                yield _sse({"type": "route", "route": "complex_logic",
                            "confidence": round(confidence, 3),
                            "reason": f"ha_call_failed:{err_text[:160]}"})
                route = "complex_logic"
                ctx = (
                    f"Попытка в Home Assistant не удалась: инструмент {call.tool} "
                    f"с аргументами {call.args} вернул ошибку: {err_text}."
                )
                if isinstance(err, dict) and err.get("done"):
                    ctx += (
                        f" Уже подтверждённо выполнено (не повторяй эти вызовы): "
                        f"{err['done']}."
                    )
                if call.args.get("area"):
                    ctx += (
                        f" Комната в Home Assistant называется "
                        f"«{call.args['area']}» — используй это имя, не переводи."
                    )
                if call.tool.startswith("media__"):
                    # The MCP intent name is meaningless to L2 — it does not
                    # speak `media__*` at all, and its own tool is media_control.
                    # Handing over the resolved room and the service keeps the
                    # retry deterministic instead of a re-invention.
                    ctx += (
                        " Управление плеером в Home Assistant идёт через тул "
                        "media_control (не через ha_action): действие "
                        f"{call.tool.split('__', 1)[1]},"
                        + (f" комната «{call.args['area']}»" if call.args.get("area")
                           else " комната не названа")
                        + (f", устройство «{call.hint}»" if call.hint else "")
                        + "."
                    )
                if "ASSISTANT" in err_text:
                    ctx += (
                        " Причина ASSISTANT — устройство не открыто голосовому "
                        "ассистенту: голосом им управлять нельзя, прямо скажи "
                        "об этом."
                    )
                ctx += (
                    " Не повторяй отклонённый вызов один в один и не обещай "
                    "выполнение без подтверждённого результата тула: либо "
                    "попробуй иначе, либо прямо сообщи, что не получилось."
                )
                async for chunk in _run_worker(
                    text, req.session_id, req.stream_name, context=ctx, hist=hist
                ):
                    if chunk.get("type") == "sentence" and chunk.get("text"):
                        reply_parts.append(chunk["text"])
                    yield _sse(chunk)

        # --- easy_query: states/history/datetime --------------------------
        elif route == "easy_query":
            # Same double-resolve pattern as easy_action (pure, cheap).
            q = resolve_query(text, area_hint)
            sentence = None
            emitted = False  # weather streams its own sentences
            # Three kinds, answered deterministically where possible:
            # datetime (HA-side tool), weather (open-meteo -> Hermes chain),
            # state (live registry lookup with a binding area).
            if q.kind == "datetime":
                # Clock lives in HA, not in this container — timezone and DST
                # are already correct there. On failure `sentence` stays None
                # and the common unresolved branch escalates to L2.
                res = await ha.call_tool("llm__GetDateTime", {})
                if res.get("ok"):
                    sentence = _datetime_phrase(res.get("result"))
            elif q.kind == "weather":
                # Hybrid chain (25.09.2026): open-meteo builds the forecast
                # deterministically (instant, cannot invent dates); Hermes L3
                # is the knowledge fallback; if it also says nothing, fall
                # through to the shared query_unresolved -> complex_logic
                # escalation below — every level gets its chance in order.
                parts = await weather.weather_sentences(text)
                if not parts:
                    try:
                        async for s in chat_proxy.stream_hermes(text):
                            parts.append(s)
                    except Exception as e:
                        logger.warning("weather hermes fallback failed: %s", e)
                for s in parts:
                    emitted = True
                    reply_parts.append(s)
                    yield _sse({"type": "sentence", "text": s})
            elif q.kind == "state":
                states = await ha.get_states()
                area = q.args.get("area")
                hint = q.entity_hint
                # Area is binding: never answer with another room's sensor.
                # (area=None -> unconstrained global match; a room that was
                # named — explicitly or via the satellite default — narrows it.)
                # `domain` ranks a real device above a same-hint helper entity
                # (vacuum.* over update.vacuum_card_update); `label` is the RU
                # device word spoken instead of the latin friendly name.
                ent = find_entity(
                    states, hint, area,
                    domain=q.args.get("domain"),
                    device=q.args.get("device"),
                )
                if ent is None and area and q.args.get("domain"):
                    # A DOMAIN + ROOM query the names cannot answer: no entity
                    # id carries «Living Room», so find_entity filtered the
                    # player out and «что сейчас играет в гостиной» answered
                    # «Не нашла такого устройства» while the TV was playing
                    # (field check 03.10.2026). The area REGISTRY is the
                    # authoritative second source — same shape as the ha_read
                    # fix on the worker side.
                    area_map = await ha.get_entity_areas()
                    wanted = {d.lower() for d in q.args["domain"]}
                    for e in states:
                        eid = str(e.get("entity_id", ""))
                        if eid.split(".", 1)[0] not in wanted:
                            continue
                        if str(e.get("state", "")).lower() in (
                                "unavailable", "unknown"):
                            continue
                        if area_map.get(eid) == area or (
                                area in str(area_map.get(eid, "")).lower()):
                            ent = e
                            break
                if ent is not None:
                    sentence = describe_entity(
                        ent, area, label=q.args.get("label", "")
                    )
                elif states and hint:
                    # The registry is loaded but the hint matches nothing in
                    # RU or EN: deterministic truth from the source of record.
                    # (Escalating here lets free models hallucinate a confident
                    # «да» — verified twice during E2E.) `missing` is the
                    # resolver's wording for the case where the DEVICE is known
                    # and it is the reading that does not exist («заряд
                    # чайника»): the generic sentence would deny a device that
                    # sits right there in the registry.
                    sentence = (
                        q.args.get("missing")
                        or "Не нашла такого устройства. Может, уточните название?"
                    )
            if emitted:
                pass  # already streamed above
            elif sentence:
                reply_parts.append(sentence)
                yield _sse({"type": "sentence", "text": sentence})
            else:
                # Nothing speakable at any step: emit a SECOND route event
                # downgrading to complex_logic (the self-healing retry the
                # caller expects) and hand the text to L2 unchanged.
                logger.info("query unresolved -> complex_logic: %r", text[:80])
                yield _sse({"type": "route", "route": "complex_logic",
                            "confidence": round(confidence, 3),
                            "reason": "query_unresolved"})
                route = "complex_logic"
                async for chunk in _run_worker(
                    text, req.session_id, req.stream_name, context=ctx, hist=hist
                ):
                    if chunk.get("type") == "sentence" and chunk.get("text"):
                        reply_parts.append(chunk["text"])
                    yield _sse(chunk)

        # --- general_qa / expert: chat proxy ------------------------------
        elif route in ("general_qa", "expert"):
            # expert=True flips the failover order inside stream_chat
            # (Hermes-first). No HA side-effects are possible on this path.
            async for s in chat_proxy.stream_chat(text, expert=(route == "expert")):
                reply_parts.append(s)
                yield _sse({"type": "sentence", "text": s})

        # --- complex_logic: L2 CodeAgent ----------------------------------
        else:
            async for chunk in _run_worker(
                text, req.session_id, req.stream_name, context=ctx, hist=hist
            ):
                # Heartbeats are forwarded for UX but only sentences are
                # remembered (memory stores what the user actually heard).
                if chunk.get("type") == "sentence" and chunk.get("text"):
                    reply_parts.append(chunk["text"])
                yield _sse(chunk)

    except Exception as e:
        # Any handler failure degrades to an error event + done instead of a
        # hung stream — a voice turn must always terminate.
        logger.exception("route handler failed")
        yield _sse({"type": "error", "message": str(e)[:300]})

    # --- P4: fire-and-forget memory write (before the final yield so a
    # client disconnect at `done` cannot skip it) --------------------------
    reply = " ".join(reply_parts)
    if reply:
        # Short-term ring FIRST (sync, in-process): the next turn of this
        # satellite must see this one even if the Qdrant write below fails.
        history.push(req.stream_name or req.session_id, text, reply)
        asyncio.create_task(
            memory.save_turn(
                text, reply, route, confidence, req.session_id, req.stream_name
            )
        )

    yield _sse(
        {"type": "done", "route": route,
         "elapsed": round(time.monotonic() - t0, 2)}
    )


@app.post("/route")
async def route_ep(req: RouteRequest):
    """Main SSE endpoint (the one main.py calls for every voice turn).

    Returns a StreamingResponse over the `_handle` generator; no auth on
    port 8091 is a deliberate LAN-only decision (see module header).
    """
    return StreamingResponse(
        _handle(req),
        media_type="text/event-stream",
        # no-cache + X-Accel-Buffering=no: any proxy/buffer in between would
        # hold sentences back and turn a streaming answer into a burst.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/weather")
async def weather_ep(text: str = ""):
    """Deterministic forecast for L2 (smolagents weather_forecast tool).

    Same open-meteo chain the easy_query weather branch uses — the worker
    calls this instead of duplicating WMO codes/coordinates logic. Empty
    sentences = upstream failed: the tool must report «не смог получить»
    rather than guess (L2 hallucinated a forecast on 25.09 before this).
    """
    parts = await weather.weather_sentences(text)
    return {"sentences": parts, "ok": bool(parts)}


@app.get("/health")
async def health():
    """Liveness probe for compose/monitoring and manual checks.

    `classifier_warmed` distinguishes "up but degrading to L2 on every turn"
    from "fully functional"; `routes` is read from the classifier module at
    call time so a routes change needs no restart of this handler.
    """
    return {
        "status": "ok",
        "classifier_warmed": classifier.warmed,
        "routes": list(__import__("classifier").ROUTES.keys()),
    }
