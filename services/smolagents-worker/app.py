"""smolagents-worker — cascade level 2 (CodeAgent) with SSE /invoke.

Flow per request:
  * CodeAgent.run() executes in a worker thread (smolagents is synchronous);
  * an async poll loop emits SSE progress heartbeats every HEARTBEAT_INTERVAL
    (<=20 s) while the agent works — these replace the old 30 s watchdog and
    keep the voice turn alive end-to-end;
  * hard cap WORKER_TIMEOUT (120 s): on timeout an error event is emitted
    (the thread is bounded by MAX_STEPS and per-call OpenAI timeouts);
  * primary LLM: OmniRoute free combo (cloud), failover: Hermes L3;
  * final answer is split into speakable sentences (verbatim boundary rules
    from voice_gateway/backends.py) for TTS.

SSE event schema (one JSON object per `data:` line):
  {"type":"progress","text":str}   # short RU heartbeat phrase (spoken)
  {"type":"sentence","text":str}   # speakable TTS chunk of the final answer
  {"type":"done","elapsed":float}
  {"type":"error","message":str}
"""

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from smolagents import CodeAgent, OpenAIModel

import config
from honesty import vet_answer, vet_weather
from tools import TOOLS, get_action_events, get_weather_events, reset_action_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("worker")

# --- Sentence splitting: verbatim copy of voice_gateway/backends.py rules ---
_BOUNDARY_NEXT = frozenset(
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789\"«'("
)
_TAG = re.compile(r"\[(happy|neutral|thinking|surprised|sad|angry)\]")

TASK_TEMPLATE = """Пользователь сказал голосом: «{text}»{context}

Ты — голосовой ассистент умного дома (колонка). Решай задачу по шагам, вызывая тулы:
- ha_read — состояние устройства/датчика Home Assistant;
- ha_action — действие в HA (включить/выключить, яркость/цвет света, пылесос, таймеры, громкое сообщение);
- qdrant_search — долговременная память диалога (что пользователь говорил раньше);
- hermes_expert — старший эксперт по инфраструктуре (сложная диагностика сети/серверов);
- weather_forecast — погода и прогноз на улице (для вопросов о погоде/температуре
  вызывай ЕГО, а не ha_read; если вернул «недоступен» — так и скажи, не выдумывай);
- get_datetime — текущие дата и время.

Правила:
- Опирайся на факты из тулов, ничего не выдумывай.
- НИКОГДА не вызывай final_answer в том же блоке кода, что и тул: сначала выполни
  тул и прочитай его вывод, ответ составляй ТОЛЬКО на следующем шаге. Ответ,
  написанный до чтения вывода, — это выдумка, даже если звучит правдоподобно.
- Числа и факты из вывода weather_forecast переноси в ответ БЕЗ изменений
  (температура, осадки, слово о погоде); не округляй и не заменяй их своими.
- Если тул вернул ошибку, «не найдено» или пусто — так и скажи в финальном ответе
  («не нашёл такого устройства» / «данных нет»). НИКОГДА не отвечай «да/включено/работает»
  на основе предположений: выдуманный статус хуже любого отказа.
- НИКОГДА не обещай выполнение действия («выключаю», «включил», «сделал», «хорошо,
  выполняю»), пока тул не подтвердил успех. Если ha_action вернул ошибку — прямо
  скажи, что не удалось и почему; можешь попробовать иначе (другое имя комнаты
  или устройства), но фраза-обещание без реально выполненного вызова запрещена.
- Выполняй только то, что просили; разрушительные/опасные действия — только при явном указании.
- Финальный ответ (final_answer) — 1-3 коротких предложения по-русски, живым разговорным языком,
  пригодные для озвучки колонкой: без markdown, без списков, без слов «код», «шаг», «инструмент».
- Если данных не хватает или действие неоднозначно — коротко попроси уточнение в одном предложении.
"""

PROGRESS_PHRASES = [
    "Анализирую запрос…",
    "Проверяю устройства…",
    "Собираю результат…",
    "Почти готово…",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "smolagents-worker up: combo=%s max_steps=%s timeout=%ss",
        config.OMNIROUTE_COMBO, config.MAX_STEPS, config.WORKER_TIMEOUT,
    )
    yield


app = FastAPI(title="smolagents-worker", lifespan=lifespan)


class InvokeRequest(BaseModel):
    text: str
    session_id: str = ""
    stream_name: str = ""
    context: str = ""  # why L1 escalated (failed HA call args + error)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _sentence_boundary(text: str) -> int:
    """End offset just past the last real sentence boundary, or -1."""
    best = -1
    for m in re.finditer(r"[.!?…]+", text):
        after = text[m.end():]
        if after == "":
            best = m.end()
            continue
        if after[0] in " \t\n":
            rest = after.lstrip()
            if rest == "" or rest[0] in _BOUNDARY_NEXT:
                best = m.end()
    return best


def _speakable(s: str) -> str:
    """Strip tags/markdown and collapse whitespace (TTS-safe); mirrors the
    sanitizer in jev-router/chat_proxy.py (free models leak markdown)."""
    s = _TAG.sub("", s)
    s = re.sub(r"\[[^\]]{0,40}\]", "", s)
    s = re.sub(r"\$[^$]{0,80}\$|\\rightarrow|\\Rightarrow|\\to\b", " ", s)
    s = re.sub(r"[*_`#~]+", "", s)
    s = re.sub(r"^\s*(?:[-–—•]|\d+[.)])\s+", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def split_sentences(text: str) -> list[str]:
    buf = _speakable(text.replace("\n", " "))
    out: list[str] = []
    while True:
        bi = _sentence_boundary(buf)
        if bi == -1:
            break
        head, buf = buf[:bi], buf[bi:]
        s = _speakable(head)
        if s and re.search(r"[A-Za-zА-Яа-яЁё]", s):
            out.append(s)
    tail = _speakable(buf)
    if tail and re.search(r"[A-Za-zА-Яа-яЁё]", tail):
        out.append(tail)
    return out


def _build_model(primary: bool) -> OpenAIModel:
    client_kwargs = {"timeout": config.LLM_CALL_TIMEOUT, "max_retries": 1}
    if primary:
        return OpenAIModel(
            model_id=config.OMNIROUTE_COMBO,
            api_base=f"{config.OMNIROUTE_URL}/v1",
            api_key="none",  # OmniRoute free combo needs no key
            client_kwargs=client_kwargs,
        )
    return OpenAIModel(
        model_id=config.HERMES_MODEL,
        api_base=f"{config.HERMES_URL}/v1",
        api_key=config.HERMES_API_KEY,
        client_kwargs=client_kwargs,
    )


def _run_agent(text: str, context: str = "") -> str:
    """Sync agent run (executed in a thread). OmniRoute combo -> Hermes failover."""
    ctx = f"\nКонтекст эскалации: {context}" if context else ""
    task = TASK_TEMPLATE.format(text=text, context=ctx)
    last_err: Exception | None = None
    for attempt, primary in enumerate((True, False), start=1):
        try:
            reset_action_events()  # per-run side-effect log for the honesty veto
            agent = CodeAgent(
                tools=TOOLS,
                model=_build_model(primary),
                max_steps=config.MAX_STEPS,
            )
            logger.info("agent run start (attempt %d, primary=%s)", attempt, primary)
            t0 = time.monotonic()
            answer = str(agent.run(task, stream=False))
            # Free models claimed «включено» right after a failed ha_action
            # (3 field regressions) — the recorded outcomes decide, not the
            # prompt: a success claim without one confirmed call is replaced.
            events = get_action_events()
            answer, replaced = vet_answer(answer, events)
            if replaced:
                logger.warning("honesty veto: claim replaced with recorded truth")
            # Same pattern for weather: the model wrote final_answer before
            # reading weather_forecast (2 field regressions, 25.09) — the
            # recorded forecast decides instead. Skipped whenever a ha_action
            # was attempted this run: the full-answer replacement must never
            # wipe an action report (true success or honest refusal).
            w_events = get_weather_events()
            answer, w_replaced = vet_weather(answer, w_events, bool(events))
            if w_replaced:
                logger.warning("weather veto: answer replaced with recorded forecast")
            logger.info(
                "agent run done in %.1fs (attempt %d)",
                time.monotonic() - t0, attempt,
            )
            return answer
        except Exception as e:
            last_err = e
            logger.warning("agent run failed (attempt %d): %s", attempt, e)
    raise last_err  # type: ignore[misc]


async def _handle(req: InvokeRequest):
    t0 = time.monotonic()
    box: dict = {}

    def worker() -> None:
        try:
            box["answer"] = _run_agent(req.text, req.context)
        except Exception as e:  # noqa: BLE001 — must surface as SSE error
            box["error"] = str(e)[:300]

    agent_task = asyncio.create_task(asyncio.to_thread(worker))
    last_hb = t0
    hb_idx = 0

    try:
        while "answer" not in box and "error" not in box:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            if now - t0 > config.WORKER_TIMEOUT:
                box["error"] = f"timeout>{int(config.WORKER_TIMEOUT)}s"
                break
            if now - last_hb >= config.HEARTBEAT_INTERVAL:
                last_hb = now
                yield _sse(
                    {"type": "progress", "text": PROGRESS_PHRASES[hb_idx % len(PROGRESS_PHRASES)]}
                )
                hb_idx += 1

        if "error" in box:
            logger.error("invoke error: %s", box["error"])
            yield _sse({"type": "error", "message": box["error"]})
        else:
            sentences = split_sentences(box.get("answer", ""))
            for s in sentences:
                yield _sse({"type": "sentence", "text": s})
            if not sentences:
                yield _sse({"type": "error", "message": "empty_answer"})
    finally:
        # No yield here: on client disconnect GeneratorExit arrives at the
        # await point and a yield in finally would be a RuntimeError.
        if not agent_task.done():
            agent_task.cancel()

    yield _sse({"type": "done", "elapsed": round(time.monotonic() - t0, 2)})


@app.post("/invoke")
async def invoke(req: InvokeRequest):
    return StreamingResponse(
        _handle(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "combo": config.OMNIROUTE_COMBO,
        "max_steps": config.MAX_STEPS,
        "timeout": config.WORKER_TIMEOUT,
        "tools": [t.name for t in TOOLS],
    }
