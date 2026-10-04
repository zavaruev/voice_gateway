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

ROLE IN THE CASCADE (three-level voice control of the smart home)
  ESP32 satellite -> main.py (FastAPI :6050, audio/VAD/STT/TTS)
  -> jev-router (L1, :8091) -> THIS SERVICE (L2, :8092, FastAPI)
  -> Hermes (L3 expert) -> Home Assistant via MCP tools.
  L1 escalates here for complex_logic/expert routes; the `context` field of
  the request explains which HA call failed and why, and `history` holds the
  last finished turns of the same satellite (anaphora like «выключи её»).

CONTRACT
  POST /invoke  {"text": RU utterance, "session_id", "stream_name",
                 "context": escalation reason, "history": recent turns}
             -> text/event-stream: one JSON object per data line, events
                listed above, UTF-8, blank-line terminated.
  GET  /health  -> config snapshot + tool names, for compose/monitoring.
  Ports 8091/8092 are UNAUTHENTICATED BY DELIBERATE DECISION (home LAN
  only) — never expose them, and never put addresses/keys in this file:
  everything comes from config.py / env (the repo is public).

WHY THIS SERVICE IS SHAPED THE WAY IT IS — the single most important fact
  about it: the CodeAgent runs on a FREE LLM that routinely writes
  `final_answer` in the SAME code block as its tool call, i.e. it announces
  success before the tool ever ran. Prompt rules alone were ignored in 4/4
  field cases, so honesty is enforced PROGRAMMATICALLY, twice over:
  BEFORE the answer is accepted, honesty.same_block_check (a smolagents
  `final_answer_checks` hook) refuses an answer written in the same code
  block as a tool call and gives the model a fresh step where the tool
  output is in its context; AFTER agent.run(), vet_answer() cross-checks
  the draft against the recorded ha_action outcomes, against the recorded
  ha_read outcomes (a failed read) and against the payload itself
  (GROUNDING: a claim whose subject is not in any recorded payload is
  invented), vet_weather() against the recorded forecast (honesty.py), and
  tools.ha_action() carries a deterministic vacuum-retry fallback. The
  TASK_TEMPLATE rules below are only a supporting layer — treat them as
  prompt text, never as the enforcement mechanism (they must stay
  byte-identical: they are the model's instructions, not commentary).
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
from honesty import failure_note, reset_tool_call, same_block_check, vet_answer, vet_weather
from tools import (
    TOOLS,
    get_action_events,
    get_read_events,
    get_weather_events,
    reset_action_events,
)

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

# --- PROMPT TEXT -----------------------------------------------------------
# TASK_TEMPLATE is the task string handed to CodeAgent.run(): it is read by
# the FREE model as its instruction sheet, so its Russian wording is
# behaviour-critical and must stay byte-identical (comments go AROUND it,
# never inside). Note that its "never claim before the tool ran" rules are
# only the supporting layer — the model ignored them in 3/3 field cases,
# which is why the hard guarantee lives in honesty.py + tools.py instead.
TASK_TEMPLATE = """Пользователь сказал голосом: «{text}»{context}

Ты — голосовой ассистент умного дома (колонка). Решай задачу по шагам, вызывая тулы:
- ha_read — состояние устройства/датчика Home Assistant;
- ha_action — действие в HA (включить/выключить, яркость/цвет света, пылесос, таймеры, громкое сообщение);
- media_control — плеер, телевизор, колонка, Kodi: пауза, продолжить, стоп, следующий/предыдущий трек, громче, тише, громкость в процентах, заглушить звук;
- media_search — что есть в библиотеке Kodi по названию (ничего не запускает);
- media_play — включить серию/фильм из библиотеки Kodi по названию;
- qdrant_search — долговременная память диалога (что пользователь говорил раньше);
- hermes_expert — старший эксперт по инфраструктуре (сложная диагностика сети/серверов). Вызывай его ТОЛЬКО когда просят ПОЧЕМУ что-то не работает; выполнить команду он не умеет;
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
- Пауза, треки и громкость делай ТОЛЬКО через media_control: в Home Assistant нет тулов
  intent__* для плеера, и выдуманное имя тула (intent__HassMediaPause и подобные) только
  потеряет шаг. Плеер, который играет, — media_control без комнаты и без имени.
- Искать и запускать по НАЗВАНИЮ — через media_search и media_play: сначала media_search
  (что есть в библиотеке), потом media_play с нужным эпизодом («13» или «S13E09»).
  Если media_search сказал, что такого нет — так и скажи пользователю, и НИКОГДА не
  выдумывай причину («не смог связаться», «сломалось», «нет интернета»): причина одна —
  такого названия нет в библиотеке Kodi.
- Если тул ПОДТВЕРДИЛ действие (в его выводе есть «запущено», «выполнено», «состояние
  изменилось») — говори о нём как о сделанном («включила», «идёт»), а не как о будущем
  («включаю», «запускаю»).
- Финальный ответ (final_answer) — 1-3 коротких предложения по-русски, живым разговорным языком,
  пригодные для озвучки колонкой: без markdown, без списков, без слов «код», «шаг», «инструмент».
- Если данных не хватает или действие неоднозначно — коротко попроси уточнение в одном предложении.
"""

# Rotating RU heartbeat phrases: short enough for TTS, neutral enough to
# repeat; emitted by the SSE poll loop while the (synchronous) agent runs
# so the voice turn never goes silent for more than HEARTBEAT_INTERVAL.
PROGRESS_PHRASES = [
    "Анализирую запрос…",
    "Проверяю устройства…",
    "Собираю результат…",
    "Почти готово…",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log the effective model/limits once at startup (uvicorn :8092).
    Nothing is opened here: every HTTP client is created per run, so a
    cold config can never leave a stale connection behind on shutdown."""
    logger.info(
        "smolagents-worker up: combo=%s max_steps=%s timeout=%ss",
        config.OMNIROUTE_COMBO, config.MAX_STEPS, config.WORKER_TIMEOUT,
    )
    yield


# ASGI app served by `uvicorn app:app --port 8092` (see Dockerfile CMD);
# no auth middleware on purpose — home LAN only (see module header).
app = FastAPI(title="smolagents-worker", lifespan=lifespan)


class InvokeRequest(BaseModel):
    """POST /invoke body. `text` is the transcribed user utterance (RU);
    `session_id`/`stream_name` are passed through for logging parity with
    L1; `context` is why L1 escalated (failed HA call args + error) and is
    injected into the task as the «Контекст эскалации» block; `history` is
    the router's ring of the last finished turns («Предыдущие реплики»).

    Both context fields default to "" so an older L1 can still call this
    service: unknown/absent fields simply stay empty.
    """

    text: str
    session_id: str = ""
    stream_name: str = ""
    context: str = ""  # why L1 escalated (failed HA call args + error)
    history: str = ""  # recent turns, one `- пользователь: … — ответ: …` line each


def _sse(obj: dict) -> str:
    """Serialize one SSE event: `data: <json>` + blank line. UTF-8 kept
    readable (ensure_ascii=False) because the payload is Russian text."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _sentence_boundary(text: str) -> int:
    """End offset just past the last real sentence boundary, or -1.

    A [.!?…] run only counts when followed by whitespace and then a
    capital/quote/digit (or end of text): without that, abbreviations
    ("т.д.", "г.", "16.09") would be cut into separate TTS chunks and the
    speaker would stutter. Returns the LAST valid boundary so callers can
    peel the text off head-first; -1 means "no completed sentence yet".
    """
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
    sanitizer in jev-router/chat_proxy.py (free models leak markdown).

    Removes emotion/emphasis tags, [short-bracket] leftovers, $math$ and
    arrow LaTeX, markdown emphasis chars and list bullets — TTS would
    otherwise spell them out ("звёздочка…"). Returns "" for debris."""
    s = _TAG.sub("", s)
    s = re.sub(r"\[[^\]]{0,40}\]", "", s)
    s = re.sub(r"\$[^$]{0,80}\$|\\rightarrow|\\Rightarrow|\\to\b", " ", s)
    s = re.sub(r"[*_`#~]+", "", s)
    s = re.sub(r"^\s*(?:[-–—•]|\d+[.)])\s+", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def split_sentences(text: str) -> list[str]:
    """Split the final answer into speakable TTS chunks.

    Newlines are flattened first (the model likes lists), then complete
    sentences are peeled off via `_sentence_boundary`; whatever is left is
    emitted as a final tail. Chunks that survive sanitizing but contain no
    letters (pure punctuation/emoji) are dropped. Returns [] for an empty
    or fully-symbolic answer — the caller turns that into an SSE error."""
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
    """Build the OpenAI-compatible chat model for one attempt.

    primary=True  -> OmniRoute FREE combo (cloud, no key required): the
                     cheap default that produces most of the answers.
    primary=False -> Hermes (L3) as failover, authenticated via env key.
    Both are bounded by LLM_CALL_TIMEOUT with max_retries=1 so one hung
    completion cannot eat the whole WORKER_TIMEOUT budget; connection
    errors propagate up to _run_agent's retry loop.
    """
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


def _end_step(step) -> None:
    """smolagents `step_callbacks` hook: every code block starts with a clean
    same-block flag (the tools set it while they run, honesty.
    same_block_check reads it when the model tries to answer).

    Signature is deliberately `(step)` only — smolagents passes the extra
    kwargs (agent=…) solely to multi-parameter callbacks, and it inspects
    the signature to decide.
    """
    reset_tool_call()


def _run_agent(
    text: str, context: str = "", hist: str = "", retry: bool = False
) -> str:
    """Sync agent run (executed in a thread). OmniRoute combo -> Hermes failover.

    Params: `text` = transcribed utterance; `context` = why L1 escalated
    (the «Контекст эскалации» block, empty for direct calls); `hist` =
    router-side ring of the last finished turns (the «Предыдущие реплики»
    block — without it an anaphora like «выключи её» has nothing to resolve
    against and the model picks a device at random); `retry` = set by the
    bounded self-healing pass below.
    Returns the final answer AFTER the honesty vetoes. Failure: raises the
    second attempt's exception if both models fail (surfaced as SSE error).

    Non-obvious: the per-run side-effect log is reset inside the loop
    (each attempt starts clean, so a failed first attempt cannot make the
    veto trust stale successes), and the vetoes run on every attempt —
    the failover model is free too and gets the same treatment.

    BOUNDED RETRY: veto fired == the model believed it acted on a target and
    only the tool disagreed, so ONE more pass is given, with the recorded
    error text as context (it can fix the name/room instead of ending the
    turn in a bare refusal). `retry=True` makes a second veto speak the
    recorded truth rather than loop — this is a fix, not a search: field
    case 28.09.2026 lost 2 of 5 turns exactly here, but an unbounded agent
    that keeps trying would eventually act on a wrong target.
    """
    blocks = []
    if context:
        blocks.append(f"Контекст эскалации: {context}")
    if hist:
        blocks.append(
            "Предыдущие реплики этого же разговора (нужны, чтобы понимать "
            f"местоимения вроде «её», «он», «то»):\n{hist}"
        )
    ctx = ("\n" + "\n".join(blocks)) if blocks else ""
    task = TASK_TEMPLATE.format(text=text, context=ctx)
    last_err: Exception | None = None
    for attempt, primary in enumerate((True, False), start=1):
        try:
            reset_action_events()  # per-run outcome log for the honesty veto
            agent = CodeAgent(
                tools=TOOLS,
                model=_build_model(primary),
                max_steps=config.MAX_STEPS,
                # Honesty is enforced by smolagents' own hooks, not by the
                # prompt: `final_answer_checks` refuses an answer written in
                # the same code block as a tool call (the model then gets a
                # fresh step where the tool output IS in its context), and
                # the step callback clears that flag at the end of every
                # step. See honesty.same_block_check.
                final_answer_checks=[same_block_check],
                step_callbacks=[_end_step],
            )
            logger.info("agent run start (attempt %d, primary=%s)", attempt, primary)
            t0 = time.monotonic()
            answer = str(agent.run(task, stream=False))
            # Free models claimed «включено» right after a failed ha_action
            # (3 field regressions) and «заряд 45 %» right after a failed
            # ha_read (29.09.2026) — the recorded outcomes decide, not the
            # prompt: a claim without one confirmed call of its own kind is
            # replaced by the truth.
            events = get_action_events()
            reads = get_read_events()
            w_events = get_weather_events()
            # Third argument is the read log (data claims), fourth the
            # weather output — a degree or a percentage may legitimately
            # come from the forecast, and grounding must not veto it.
            answer, replaced = vet_answer(answer, events, reads, w_events)
            if replaced:
                logger.warning("honesty veto: claim replaced with recorded truth")
            # Same pattern for weather: the model wrote final_answer before
            # reading weather_forecast (2 field regressions, 25.09) — the
            # recorded forecast decides instead. Skipped whenever a ha_action
            # was attempted this run: the full-answer replacement must never
            # wipe an action report (true success or honest refusal).
            answer, w_replaced = vet_weather(answer, w_events, bool(events))
            if w_replaced:
                logger.warning("weather veto: answer replaced with recorded forecast")
            logger.info(
                "agent run done in %.1fs (attempt %d)",
                time.monotonic() - t0, attempt,
            )
            note = failure_note(events + reads) if replaced else ""
            if note and not retry:
                logger.warning("honesty veto -> bounded retry with the recorded errors")
                return _run_agent(
                    text,
                    " ".join(p for p in (context, note) if p),
                    hist,
                    retry=True,
                )
            return answer
        except Exception as e:
            last_err = e
            logger.warning("agent run failed (attempt %d): %s", attempt, e)
    raise last_err  # type: ignore[misc]


async def _handle(req: InvokeRequest):
    """SSE generator for one /invoke call.

    Runs _run_agent in a worker thread (smolagents is synchronous) while
    this coroutine polls a shared `box` dict every second: a "progress"
    heartbeat phrase is emitted each HEARTBEAT_INTERVAL so the upstream
    turn watchdog hears a live gateway, and a "sentence" event per
    speakable chunk once the answer exists. Emits "error" on failure or
    WORKER_TIMEOUT overrun, and always finishes with "done".

    Failure modes: agent exceptions and the timeout both collapse into an
    SSE error event (the HTTP status stays 200 — the stream already
    started); an answer that sanitizes to nothing becomes "empty_answer".
    """
    t0 = time.monotonic()
    # `box` is shared with the worker thread: single-key dict writes are
    # atomic enough under the GIL, and the poll loop only ever READs keys,
    # so no lock is needed (and a lock here would defeat the point — the
    # thread can be stuck inside agent.run() for up to WORKER_TIMEOUT).
    box: dict = {}

    def worker() -> None:
        # Everything, vetoes included, must land in `box`: an unhandled
        # exception here would otherwise hang the poll loop until timeout.
        try:
            box["answer"] = _run_agent(req.text, req.context, req.history)
        except Exception as e:  # noqa: BLE001 — must surface as SSE error
            box["error"] = str(e)[:300]

    agent_task = asyncio.create_task(asyncio.to_thread(worker))
    last_hb = t0
    hb_idx = 0

    try:
        # 1 s poll: responsive enough for heartbeats, cheap enough to run
        # for the whole WORKER_TIMEOUT without burning CPU.
        while "answer" not in box and "error" not in box:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            if now - t0 > config.WORKER_TIMEOUT:
                # The worker thread cannot be interrupted here; its result
                # is simply discarded (the "error" branch below is checked
                # before "answer", so a late write never reaches TTS).
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
    """Start one agent turn. Returns a StreamingResponse immediately; the
    work happens inside the `_handle` generator (200 + text/event-stream —
    errors are reported in-band as SSE "error" events, since the status
    line is already sent by then)."""
    return StreamingResponse(
        _handle(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    """Liveness/config probe for compose and L1: echoes the effective
    combo, step cap, timeout and the tool manifest (tool names only)."""
    return {
        "status": "ok",
        "combo": config.OMNIROUTE_COMBO,
        "max_steps": config.MAX_STEPS,
        "timeout": config.WORKER_TIMEOUT,
        "tools": [t.name for t in TOOLS],
    }
