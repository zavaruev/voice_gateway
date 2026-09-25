"""Tools exposed to the smolagents CodeAgent (cascade level 2).

All tool functions are synchronous and use stdlib urllib only: smolagents
runs tools inside its thread pool, so blocking HTTP here never stalls the
async SSE heartbeat in app.py, and no extra async dependency is needed.

Tools (manifest mapping):
  ha_action      -> HA MCP side effects (intent/light/vacuum/timers/broadcast)
  ha_read        -> HA MCP GetLiveContext, REST /api/states fallback
  qdrant_search  -> semantic search over voice_turns + voice_facts (memory)
  hermes_expert  -> delegate hard diagnostics to L3 Hermes Agent
  get_datetime   -> local date/time for the voice assistant
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from smolagents import tool

import config

logger = logging.getLogger("worker.tools")

_MSK = timezone(timedelta(hours=3))  # Moscow fixed UTC+3 (no DST since 2014)


def _http_json(
    url: str,
    payload: dict | None = None,
    headers: dict | None = None,
    timeout: float = config.TOOL_TIMEOUT,
):
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
    """JSON-RPC tools/call against HA MCP (stateless: no initialize)."""
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
            light__HassLightSet, vacuum__HassVacuumStart, vacuum__HassVacuumReturnToBase,
            intent__HassCancelAllTimers, assist_satellite__HassBroadcast.
        arguments_json: JSON-строка аргументов, напр. {"area": "кухня", "domain": ["light"]}
            или {"message": "обед готов"}. Ключи: name, area, floor, domain, device_class,
            color, temperature, brightness, message — в зависимости от тула.
    """
    try:
        args = json.loads(arguments_json) if arguments_json.strip() else {}
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        return f"Некорректный arguments_json: {e}"
    return mcp_call(tool_name, args)


@tool
def ha_read(query: str, area: str = "") -> str:
    """Прочитать текущее состояние устройства или датчика из Home Assistant.

    Args:
        query: Что читаем: имя устройства (чайник), тип (свет, датчик температуры),
            домен (light, switch, sensor, media_player) или показатель (температура, влажность, заряд).
        area: Комната по-русски или по-английски (кухня, спальня, kitchen); можно пусто.
    """
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
    q_tokens = [t for t in f"{query} {area}".lower().split() if t]
    hits = []
    for e in states if isinstance(states, list) else []:
        eid = e.get("entity_id", "")
        name = e.get("attributes", {}).get("friendly_name") or ""
        hay = f"{eid} {name}".lower()
        if any(t in hay for t in q_tokens):
            hits.append(f"{eid}: {e.get('state')} ({name})")
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
    for coll in ("voice_turns", "voice_facts"):
        try:
            res = _http_json(
                f"{config.QDRANT_URL}/collections/{coll}/points/search",
                {"vector": vector, "limit": int(limit), "with_payload": True},
            )
        except Exception:
            continue
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
        return f"Эксперт недоступен: {e}"


@tool
def get_datetime() -> str:
    """Текущие дата и время (часовой пояс дома, Европа/Москва)."""
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


TOOLS = [ha_action, ha_read, qdrant_search, hermes_expert, get_datetime]
