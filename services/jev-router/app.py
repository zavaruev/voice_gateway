"""jev-router — cascade level-1 semantic router (SSE /route).

Pipeline per request:
  classify (Ollama embeddings + calibrated cosine)
    -> easy_action  : regex/lexicon slot resolver -> HA MCP intent call
    -> easy_query   : HA states/history/datetime  -> speakable answer
    -> general_qa   : chat proxy (OmniRoute combo -> Hermes failover)
    -> expert       : chat proxy, expert persona (Hermes-first diagnostics)
    -> complex_logic: forwarded to smolagents-worker (L2 CodeAgent)

Escalation rule: any ambiguity (resolver -> None, HA failure, low confidence)
falls through to complex_logic — never a wrong side-effect.

SSE event schema (one JSON object per `data:` line):
  {"type":"route",   "route":str,"confidence":float,"reason":str}
  {"type":"sentence","text":str}          # speakable TTS chunk
  {"type":"progress","text":str}          # L2 progress heartbeat
  {"type":"done",    "route":str,"elapsed":float}
  {"type":"error",   "message":str}
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
import memory
from classifier import Classifier
from ha_client import (
    HAClient,
    _area_phrase,
    describe_entity,
    find_action_targets,
    find_entity,
)
from resolver import resolve_action, resolve_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("router")

classifier = Classifier()
ha = HAClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ok = await classifier.warmup()
    logger.info("classifier warmup: %s", "ok" if ok else "FAILED (escalations active)")
    mem_ok = await memory.ensure_collections()
    logger.info("qdrant collections: %s", "ok" if mem_ok else "FAILED (memory off)")
    yield
    await classifier.embedder.close()
    await ha.close()


app = FastAPI(title="jev-router", lifespan=lifespan)


class RouteRequest(BaseModel):
    text: str
    session_id: str = ""
    stream_name: str = ""


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


_WEEKDAYS_RU = {
    "Monday": "понедельник", "Tuesday": "вторник", "Wednesday": "среда",
    "Thursday": "четверг", "Friday": "пятница", "Saturday": "суббота",
    "Sunday": "воскресенье",
}
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
    """
    if isinstance(result, str):
        return result.strip() or None
    if not isinstance(result, dict):
        return None
    parts: list[str] = []
    date = str(result.get("date", ""))
    if len(date) >= 10 and date[4] == "-":
        month = _MONTHS_RU.get(date[5:7], "")
        date_ru = f"{int(date[8:10])} {month} {date[:4]}".strip()
        wd = _WEEKDAYS_RU.get(str(result.get("weekday", "")), "")
        parts.append(f"Сегодня {wd}, {date_ru}." if wd else f"Сегодня {date_ru}.")
    t = str(result.get("time", ""))[:5]
    if t:
        parts.append(f"Сейчас {t}.")
    return " ".join(parts) or None


# --- easy_action execution --------------------------------------------------
# On/off intents only: a wrong side-effect on other domains (broadcast,
# timers, vacuum) is a different class of mistake and keeps the blind path.
_ONOFF_TOOLS = ("intent__HassTurnOn", "intent__HassTurnOff")
_MAX_ONOFF_TARGETS = 5


def _err_text(res: dict | None) -> str:
    if not res:
        return "unknown"
    err = str(res.get("error") or "unknown")
    raw = res.get("raw")
    if raw:
        err = f"{err}: {str(raw)[:300]}"
    return err[:400]


def _target_rank(e: dict) -> tuple:
    """A real lamp beats auxiliary switches; shorter name = core device."""
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
            area_map = await ha.get_entity_areas()
            targets = find_action_targets(
                states, area_map, call.hint, call.args.get("area")
            )
            want_on = call.tool.endswith("HassTurnOn")
            opposite = "off" if want_on else "on"
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


async def _run_worker(text: str, session_id: str, stream_name: str, context: str = ""):
    """Forward smolagents-worker SSE events; yields router-level dicts."""
    timeout = aiohttp.ClientTimeout(total=config.WORKER_TIMEOUT, sock_connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(
            f"{config.WORKER_URL}/invoke",
            json={"text": text, "session_id": session_id,
                  "stream_name": stream_name, "context": context},
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
    """Async generator of router SSE events for one request."""
    t0 = time.monotonic()
    text = (req.text or "").strip()
    reply_parts: list[str] = []

    if not text:
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
    if route == "easy_action":
        call = resolve_action(text, req.stream_name)
        if call is None:
            logger.info("resolver ambiguous -> complex_logic: %r", text[:80])
            route, reason = "complex_logic", "resolver_ambiguous"
    elif route == "easy_query":
        q = resolve_query(text, req.stream_name)
        if q is None:
            logger.info("query resolver ambiguous -> complex_logic: %r", text[:80])
            route, reason = "complex_logic", "query_resolver_ambiguous"

    yield _sse(
        {"type": "route", "route": route, "confidence": round(confidence, 3),
         "reason": reason}
    )

    try:
        # --- easy_action: HA MCP intent call ------------------------------
        if route == "easy_action":
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
                    text, req.session_id, req.stream_name, context=ctx
                ):
                    if chunk.get("type") == "sentence" and chunk.get("text"):
                        reply_parts.append(chunk["text"])
                    yield _sse(chunk)

        # --- easy_query: states/history/datetime --------------------------
        elif route == "easy_query":
            q = resolve_query(text, req.stream_name)
            sentence = None
            emitted = False  # weather streams its own sentences
            if q.kind == "datetime":
                res = await ha.call_tool("llm__GetDateTime", {})
                if res.get("ok"):
                    sentence = _datetime_phrase(res.get("result"))
            elif q.kind == "weather":
                # No guaranteed weather entity: answer via chat (search-capable
                # hermes is the failover target of stream_chat).
                async for s in chat_proxy.stream_chat(text):
                    emitted = True
                    reply_parts.append(s)
                    yield _sse({"type": "sentence", "text": s})
            elif q.kind == "state":
                states = await ha.get_states()
                area = q.args.get("area")
                hint = q.entity_hint
                # Area is binding: never answer with another room's sensor.
                ent = find_entity(states, hint, area) if area else find_entity(states, hint, None)
                if ent is not None:
                    sentence = describe_entity(ent, area)
                elif states and hint:
                    # The registry is loaded but the hint matches nothing in
                    # RU or EN: deterministic truth from the source of record.
                    # (Escalating here lets free models hallucinate a confident
                    # «да» — verified twice during E2E.)
                    sentence = "Не нашла такого устройства. Может, уточните название?"
            if emitted:
                pass  # already streamed above
            elif sentence:
                reply_parts.append(sentence)
                yield _sse({"type": "sentence", "text": sentence})
            else:
                logger.info("query unresolved -> complex_logic: %r", text[:80])
                yield _sse({"type": "route", "route": "complex_logic",
                            "confidence": round(confidence, 3),
                            "reason": "query_unresolved"})
                route = "complex_logic"
                async for chunk in _run_worker(
                    text, req.session_id, req.stream_name
                ):
                    if chunk.get("type") == "sentence" and chunk.get("text"):
                        reply_parts.append(chunk["text"])
                    yield _sse(chunk)

        # --- general_qa / expert: chat proxy ------------------------------
        elif route in ("general_qa", "expert"):
            async for s in chat_proxy.stream_chat(text, expert=(route == "expert")):
                reply_parts.append(s)
                yield _sse({"type": "sentence", "text": s})

        # --- complex_logic: L2 CodeAgent ----------------------------------
        else:
            async for chunk in _run_worker(text, req.session_id, req.stream_name):
                if chunk.get("type") == "sentence" and chunk.get("text"):
                    reply_parts.append(chunk["text"])
                yield _sse(chunk)

    except Exception as e:
        logger.exception("route handler failed")
        yield _sse({"type": "error", "message": str(e)[:300]})

    # --- P4: fire-and-forget memory write (before the final yield so a
    # client disconnect at `done` cannot skip it) --------------------------
    reply = " ".join(reply_parts)
    if reply:
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
    return StreamingResponse(
        _handle(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "classifier_warmed": classifier.warmed,
        "routes": list(__import__("classifier").ROUTES.keys()),
    }
