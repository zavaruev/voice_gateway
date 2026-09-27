"""Tools exposed to the smolagents CodeAgent (cascade level 2).

All tool functions are synchronous and use stdlib urllib only: smolagents
runs tools inside its thread pool, so blocking HTTP here never stalls the
async SSE heartbeat in app.py, and no extra async dependency is needed.

Tools (manifest mapping):
  ha_action      -> HA MCP side effects (intent/light/vacuum/timers/broadcast)
  ha_read        -> HA MCP GetLiveContext, REST /api/states fallback
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
  to TASK_TEMPLATE in app.py.

WHY HONESTY IS ENFORCED HERE AND NOT IN THE PROMPT
  The CodeAgent runs on a FREE LLM that writes `final_answer` in the SAME
  code block as its tool call — announcing success before the tool ran.
  Prompt rules were ignored in 3/3 field cases, so this module records
  what ACTUALLY happened (per-call outcome buffers below) and exposes it
  to the veto in honesty.py, plus `ha_action` carries a deterministic
  fallback for the vacuum `area` regression (25.09.2026, «Пусть робот
  уберется на кухне») — a code retry, not a prompting trick.

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
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from smolagents import tool

import config

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
# Weather outcomes are recorded separately from side effects: a successful
# weather_forecast must never count as "confirmed side effect" and suppress
# the action veto in a compound query («включи кофеварку и какая погода»).
_WEATHER_EVENTS: list[dict] = []
_EVENTS_LOCK = threading.Lock()


def reset_action_events() -> None:
    """Clear BOTH buffers; called by app._run_agent at the start of every
    attempt so events never leak between runs (a stale success from the
    previous turn would silence the veto on a fresh failure)."""
    with _EVENTS_LOCK:
        _ACTION_EVENTS.clear()
        _WEATHER_EVENTS.clear()


def get_action_events() -> list[dict]:
    """Snapshot copy of this run's ha_action outcomes (lock-protected):
    [{"tool", "ok", "detail"}, ...] in call order. Never blocks the tool
    thread — it is read after agent.run() has returned."""
    with _EVENTS_LOCK:
        return list(_ACTION_EVENTS)


def get_weather_events() -> list[dict]:
    """Snapshot copy of this run's weather_forecast outputs
    [{"tool", "detail"}, ...]; consumed by vet_weather (app._run_agent)."""
    with _EVENTS_LOCK:
        return list(_WEATHER_EVENTS)


def _record_action(tool_name: str, result: str) -> None:
    """Append the outcome of one ha_action call (ok is decided here so the
    veto in honesty.py never has to re-parse MCP payloads).

    `ok` is False when the result starts with one of the transport/parse
    error prefixes returned by mcp_call/ha_action, or when a parsed JSON
    body carries "success": false. Anything else counts as success —
    deliberately conservative in the OTHER direction: a false "ok" would
    disable the veto, but only genuinely successful calls reach here
    unflagged (the vacuum fallback has already retried by then).
    `detail` is truncated to 300 chars — enough for the error markers the
    veto greps for, small enough to keep the log tidy.
    """
    head = result.lstrip()
    ok = not head.startswith(
        ("Ошибка", "Тул вернул ошибку", "Некорректный", "Неожиданный")
    )
    if ok:
        try:
            data = json.loads(head)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and data.get("success") is False:
            ok = False
    with _EVENTS_LOCK:
        _ACTION_EVENTS.append(
            {"tool": tool_name, "ok": ok, "detail": result[:300]}
        )


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
    # Primary: HA MCP GetLiveContext (understands name+area and answers
    # with a curated context blob). Any transport/tool error ("Ошибка…")
    # or an EMPTY result means "miss" -> fall through to raw REST states.
    args: dict = {"name": query}
    if area:
        args["area"] = area
    text = mcp_call("homeassistant__GetLiveContext", args)
    if "Ошибка" not in text and text.strip():
        return text

    # Fallback: raw REST states, substring match (MCP context miss)
    try:
        states = _http_json(
            f"{config.HA_URL}/api/states",
            headers={"Authorization": f"Bearer {config.HA_TOKEN}"},
        )
    except Exception as e:
        return f"Не удалось прочитать состояния: {e}"
    # OR-match on every token of "query area" against entity_id +
    # friendly_name: the user speaks Russian while entity ids are English,
    # so a single overlapping word must be enough to surface a candidate.
    q_tokens = [t for t in f"{query} {area}".lower().split() if t]
    hits = []
    for e in states if isinstance(states, list) else []:
        eid = e.get("entity_id", "")
        name = e.get("attributes", {}).get("friendly_name") or ""
        hay = f"{eid} {name}".lower()
        if any(t in hay for t in q_tokens):
            hits.append(f"{eid}: {e.get('state')} ({name})")
        # cap: the result goes into the model's context
        if len(hits) >= 10:
            break
    return "\n".join(hits) if hits else "Ничего не найдено в Home Assistant."


@tool
def qdrant_search(query: str, limit: int = 5) -> str:
    """Поиск в долговременной памяти диалога (Qdrant): прошлые реплики и факты о пользователе.

    Args:
        query: Текстовый запрос по-русски (например, о чём пользователь говорил ранее).
        limit: Максимум результатов (по умолчанию 5).
    """
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
TOOLS = [ha_action, ha_read, qdrant_search, hermes_expert, get_datetime, weather_forecast]
