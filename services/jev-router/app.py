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
from classifier import Classifier
from ha_client import (
    HAClient,
    _area_phrase,
    dedupe_device_facets,
    describe_entity,
    find_action_targets,
    find_entity,
)
from resolver import resolve_action, resolve_query, unresolved_hint

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

    Warmup failures are deliberately non-fatal — `classify()` retries lazily
    and the escalation chain (everything -> complex_logic) keeps the service
    useful even with Ollama or Qdrant down. On shutdown the embedder session
    and the HA session are closed so uvicorn exits without pending sockets.
    """
    ok = await classifier.warmup()
    logger.info("classifier warmup: %s", "ok" if ok else "FAILED (escalations active)")
    mem_ok = await memory.ensure_collections()
    logger.info("qdrant collections: %s", "ok" if mem_ok else "FAILED (memory off)")
    yield
    await classifier.embedder.close()
    await ha.close()


app = FastAPI(title="jev-router", lifespan=lifespan)


class RouteRequest(BaseModel):
    """POST /route body.

    `stream_name` is the physical satellite (kitchen/livingroom/...) — the
    resolver uses it as the default room when the utterance names none;
    `session_id` tags the memory write for L2's qdrant_search.
    """

    text: str
    session_id: str = ""
    stream_name: str = ""


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


async def _execute_action(call) -> tuple[str | None, dict | None]:
    """Run one easy_action MCP call. Returns (sentence, None) on success or
    an «already in state» answer, (None, error) otherwise — it never claims
    a side-effect that was not confirmed.

    On/off commands resolve the concrete entity from the live registry
    first: the household lamp is a switch relay (the light.* entities are
    unavailable status LEDs), entity names are latin, and a blind
    domain+area match would either miss it or silently no-op on
    `unavailable` states.
    """
    if call.tool in _ONOFF_TOOLS:
        states = await ha.get_states(force=True)
        if states:
            # Deterministic refusal: a device absent from the whole registry
            # must never reach L2 — a free model happily «включает» ghost
            # devices (field case: «Включи кафеварку», STT typo + wrong name).
            if call.hint and find_entity(states, call.hint, None) is None:
                return "Не нашла такого устройства. Может, уточните название?", None
            area_map = await ha.get_entity_areas()
            targets = dedupe_device_facets(
                find_action_targets(
                    states, area_map, call.hint, call.args.get("area")
                )
            )
            if not call.args.get("area") and len(targets) > 1:
                # No room in the utterance and several devices match —
                # acting would mean guessing which one.
                return None, {
                    "ok": False,
                    "error": f"ambiguous_no_area: {len(targets)} targets, команда без комнаты",
                }
            want_on = call.tool.endswith("HassTurnOn")
            opposite = "off" if want_on else "on"
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
                ok: list[str] = []
                fatal: list[str] = []
                skipped: list[str] = []
                for ent in todo:
                    eid = ent["entity_id"]
                    name = (ent.get("attributes", {}) or {}).get("friendly_name") or eid
                    # Fresh copy per entity: name/domain are pinned to the
                    # registry entry so the matcher targets exactly what was
                    # resolved, and the next iteration starts from clean args.
                    args = dict(call.args)
                    args["name"] = name
                    args["domain"] = [eid.split(".", 1)[0]]
                    res = await ha.call_tool(call.tool, args)
                    if res.get("ok"):
                        ok.append(eid)
                        continue
                    et = _err_text(res)
                    if "ASSISTANT" in et:
                        # Matched, but not exposed to the voice assistant:
                        # not voice-controllable — must not fail the rest
                        # (the living-room aux "Network led switch" is one).
                        skipped.append(eid)
                        logger.info("target not exposed to assistant: %s", eid)
                    else:
                        fatal.append(f"{eid}: {et}")
                # Confirmed success with no fatal error = success; entities
                # merely not exposed to the assistant were skipped, not failed.
                if ok and not fatal:
                    return call.speak_ok, None
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
    # Blind path: non-on/off intents (timers/vacuum/broadcast/light-set) or
    # on/off when the registry gave no target — the HA matcher decides.
    res = await ha.call_tool(call.tool, call.args)
    if res.get("ok"):
        return call.speak_ok, None
    return None, res


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
    if route == "easy_action":
        call = resolve_action(text, req.stream_name)
        if call is None:
            logger.info("resolver ambiguous -> complex_logic: %r", text[:80])
            route, reason = "complex_logic", "resolver_ambiguous"
            ctx = unresolved_hint(text)  # «кашеварку» -> «кофеварка», or ""
    elif route == "easy_query":
        q = resolve_query(text, req.stream_name)
        if q is None:
            logger.info("query resolver ambiguous -> complex_logic: %r", text[:80])
            route, reason = "complex_logic", "query_resolver_ambiguous"

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
            call = resolve_action(text, req.stream_name)
            sentence, err = await _execute_action(call)
            if sentence:
                reply_parts.append(sentence)
                yield _sse({"type": "sentence", "text": sentence})
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
            q = resolve_query(text, req.stream_name)
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
