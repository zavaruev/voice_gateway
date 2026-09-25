"""Streaming chat proxies: OmniRoute combo with failover to Hermes L3.

Used for two routes:
  * general_qa — quick chat that must bypass the L2 CodeAgent entirely
                 (OmniRoute first, Hermes failover);
  * expert     — Hermes L3 FIRST (OmniRoute failover): the escalation
                 «вопрос не решается ниже» lands directly on the agent brain.

Both upstreams are OpenAI-compatible /v1/chat/completions SSE. Output is
pre-split into speakable sentences so voice_gateway receives ready-to-TTS
chunks (contract of CascadeBackend.generate_response).
"""

import asyncio
import json
import logging
import re

import aiohttp

import config

logger = logging.getLogger("router.chat")

# Sentence boundary rules — verbatim copy of voice_gateway/backends.py so
# that "мм рт. ст." / "т.д." are never torn across TTS chunks (bug fixed in
# v2.25; any divergence here would reintroduce audible stutter).
_BOUNDARY_NEXT = frozenset(
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789\"«'("
)
_EMOTION_TAG = re.compile(r"\[(happy|neutral|thinking|surprised|sad|angry)\]")

# Voice budget: ~40 s of speech. Free models love listicles; past this the
# marginal value for a spoken answer is zero.
MAX_SENTENCES = 8


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
    """Strip tags/markdown and collapse whitespace (TTS-safe).

    Free models leak markdown into voice answers (verified during E2E:
    «### 1.», «$\\rightarrow$», «* **Кабель:**»); reading those aloud is
    worse than useless, so everything non-verbal is removed here and
    fragments without a single letter are dropped by the caller.
    """
    s = re.sub(r"\[/?thinking\]", "", s, flags=re.I)
    s = _EMOTION_TAG.sub("", s)
    s = re.sub(r"\[[^\]]{0,40}\]", "", s)
    s = re.sub(r"\$[^$]{0,80}\$|\\rightarrow|\\Rightarrow|\\to\b", " ", s)  # LaTeX
    s = re.sub(r"[*_`#~]+", "", s)                     # bold/italic/code/headings
    s = re.sub(r"^\s*(?:[-–—•]|\d+[.)])\s+", "", s.strip())  # list markers
    return re.sub(r"\s+", " ", s).strip()


def _has_letters(s: str) -> bool:
    return bool(re.search(r"[A-Za-zА-Яа-яЁё]", s))


async def stream_sentences(
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str = "",
    total_timeout: float = config.CHAT_TOTAL_TIMEOUT,
    first_token_timeout: float = config.CHAT_FIRST_TOKEN_TIMEOUT,
    extra_headers: dict | None = None,
):
    """Yield speakable sentences from an OpenAI-compatible SSE endpoint.

    Raises on transport/status errors so the caller can try a failover target.
    Yields nothing (not an error) if the model returned an empty completion.
    """
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    payload = {"model": model, "messages": messages, "stream": True}

    timeout = aiohttp.ClientTimeout(
        total=total_timeout, sock_connect=10, sock_read=first_token_timeout
    )
    buf = ""
    sent = 0
    t_start = asyncio.get_event_loop().time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{base_url}/v1/chat/completions", json=payload, headers=headers
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"chat {resp.status}: {body[:200]}")

            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "") or ""
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if not delta:
                    continue
                # Drop streaming [thinking] blocks before they reach the buffer.
                buf += delta
                # Newlines become spaces BEFORE sentence detection: models put
                # IPs/code on separate lines («192.\n168.») and a `.` + newline
                # would otherwise tear them into spoken fragments.
                if "\n" in buf:
                    buf = buf.replace("\n", " ")
                if re.search(r"\[thinking\]", buf) and not re.search(
                    r"\[/thinking\]", buf
                ):
                    buf = buf[: re.search(r"\[thinking\]", buf).start()]
                while True:
                    mm = re.search(r"\[thinking\](.*?)\[/thinking\]", buf, flags=re.S)
                    if not mm:
                        break
                    buf = buf[: mm.start()] + buf[mm.end():]
                if re.search(r"[.!?…](\s|$)", buf) or len(buf) > 400:
                    bi = _sentence_boundary(buf)
                    while bi != -1:
                        head, buf = buf[:bi], buf[bi:]
                        s = _speakable(head)
                        if s and _has_letters(s):
                            yield s
                            sent += 1
                            if sent >= MAX_SENTENCES:
                                return
                        bi = _sentence_boundary(buf)
                # Hard cap: never let the unsplittable tail grow unbounded.
                if len(buf) > 600:
                    s = _speakable(buf)
                    buf = ""
                    if s and _has_letters(s):
                        yield s
                        sent += 1
                        if sent >= MAX_SENTENCES:
                            return
            # Final flush
            tail = _speakable(buf)
            if tail and _has_letters(tail):
                yield tail
    logger.debug("chat stream done in %.2fs", asyncio.get_event_loop().time() - t_start)


CHAT_SYSTEM = (
    "Ты голосовой ассистент умной колонки. Отвечай ГОЛОСОМ: по-русски, "
    "живо, сразу по существу, 1-3 предложения. Тебе недоступны инструменты — "
    "никогда не озвучивай их вызовы. СТРОГО без markdown: никаких #, *, "
    "списков, нумерации, кода, формул и спецсимволов — только обычные "
    "предложения, пригодные для озвучки."
)

EXPERT_SYSTEM = (
    "Ты — эксперт по диагностике умного дома и домашней инфраструктуры "
    "(сеть, серверы, Home Assistant, устройства). Отвечай ГОЛОСОМ колонки: "
    "по-русски, живо, максимум 5 коротких предложений, сразу по существу: "
    "2-3 самых вероятные причины и один первый шаг проверки. СТРОГО без "
    "markdown: никаких #, *, списков, нумерации, кода, формул, LaTeX и "
    "спецсимволов (→, $, кавычки-звёздочки) — только обычные предложения."
)


async def stream_chat(text: str, expert: bool = False):
    """general_qa: OmniRoute combo → Hermes failover (fast chat brain).
    expert: Hermes L3 FIRST, OmniRoute failover — the expert route is the
    «вопрос не решается ниже → Гермес» path (README promised it; before
    2.28 the free combo answered expert questions too, and its polite
    refusals are exactly what hid the missing escalation)."""
    system = EXPERT_SYSTEM if expert else CHAT_SYSTEM
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    omni = (config.OMNIROUTE_URL, config.OMNIROUTE_COMBO, "")
    hermes = (config.HERMES_URL, "hermes-agent", config.HERMES_API_KEY)
    targets = (hermes, omni) if expert else (omni, hermes)
    logger.info(
        "stream_chat expert=%s primary=%s",
        expert, "hermes" if expert else "omniroute",
    )
    for url, model, api_key in targets:
        try:
            got = False
            async for s in stream_sentences(url, model, messages, api_key=api_key):
                got = True
                yield s
            if got:
                return
            logger.warning("%s returned empty, trying next target", model)
        except Exception as e:
            logger.warning("%s failed (%s), trying next target", model, e)


async def stream_hermes(text: str, system: str = CHAT_SYSTEM):
    """Direct Hermes L3 stream (weather fallback after open-meteo failed).

    Raises on transport/status errors like stream_sentences — the caller
    treats «nothing yielded» as unsolved and escalates to complex_logic.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    logger.info("stream_hermes: direct L3 call (weather fallback)")
    async for s in stream_sentences(
        config.HERMES_URL, "hermes-agent", messages, api_key=config.HERMES_API_KEY
    ):
        yield s
