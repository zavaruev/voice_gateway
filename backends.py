import asyncio
import json
import re
import time
import logging
import aiohttp
from abc import ABC, abstractmethod

logger = logging.getLogger("backends")

class BaseLLMBackend(ABC):
    @abstractmethod
    async def generate_response(self, text: str, session_id: str, stream_name: str, response_queue: asyncio.Queue):
        """
        Generates a response for the given text and pushes sentences into response_queue.
        Should put None into response_queue when finished.
        """
        pass

class NanobotBackend(BaseLLMBackend):
    def __init__(self, url: str, token: str, salt: str):
        self.url = url.rstrip("/")
        self.token = token
        self.salt = salt

    async def generate_response(self, text: str, session_id: str, stream_name: str, response_queue: asyncio.Queue):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"{self.url}?token={self.token}&chat_id={session_id}"
                ) as ws:
                    await ws.send_json({
                        "type": "message",
                        "chat_id": session_id,
                        "content": text,
                        "user_id": "camera",
                        "user_name": stream_name,
                        "voice_reply": True,
                    })
                    
                    _n_sent = 0
                    buf = ""
                    spoke_any = False

                    def speakable(s: str) -> str:
                        s = re.sub(r"\[[a-z]{2,30}\]", "", s)
                        return re.sub(r"\s+", " ", s).strip()

                    def flush(final: bool) -> None:
                        nonlocal buf, spoke_any, _n_sent
                        m = re.search(r"\[thinking\]", buf)
                        if m and not re.search(r"\[/thinking\]", buf[m.start():]):
                            part, holder = (buf[:m.start()], buf[m.start:]) if not final else (buf, "")
                        else:
                            idx = buf.rfind("[")
                            if idx != -1 and not re.search(r"\]", buf[idx:]):
                                part, holder = buf[:idx], buf[idx:]
                            else:
                                part, holder = buf, ""
                        
                        while True:
                            mm = re.search(r"\[thinking\](.*?)\[/thinking\]", part, flags=re.S)
                            if not mm: break
                            part = part[:mm.start()] + part[mm.end():]
                        
                        mm = re.search(r"^(.*[.!?…])([^.!?…]*)$", part, flags=re.S)
                        if mm:
                            done = [mm.group(1)] if mm.group(1).strip() else []
                            tail = mm.group(2)
                        else:
                            done, tail = [], part
                        
                        if final and tail.strip():
                            done.append(tail)
                            tail = ""
                        
                        for s in done:
                            s2 = speakable(s)
                            if s2:
                                spoke_any = True
                                _n_sent += 1
                                response_queue.put_nowait(s2)
                        buf = (tail + " " + holder).strip()

                    try:
                        while True:
                            msg = await asyncio.wait_for(ws.receive(), timeout=45.0)
                            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = json.loads(msg.data)
                            ev = data.get("event")
                            if ev in ("done", "error", "final"):
                                break
                            if ev == "stream_end":
                                break
                            delta = None
                            if "text" in data and data.get("type") not in ("stt", "listen") and ev not in ("reasoning_delta", "thinking", "ready"):
                                delta = str(data["text"])
                            elif "parts" in data:
                                delta = str(data["parts"])
                            if delta:
                                buf += delta
                                if re.search(r"[.!?…](\s|$)", buf) or "\n" in buf or len(buf) > 400:
                                    flush(final=False)
                    finally:
                        flush(final=True)
                        await response_queue.put(None)
        except Exception as e:
            logger.error(f"Nanobot error: {e}")
            await response_queue.put(None)


# Chars that may start a new sentence (after ". " / "? " / "! ").
# A period counts as a sentence boundary only before one of these —
# this keeps abbreviations ("мм рт. ст.", "т.д.", "г.", "16.09")
# from being torn into separate TTS utterances (audible stutter).
_BOUNDARY_NEXT = frozenset(
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789\"«'("
)

# Voice answers MUST start with an emotion tag ([happy]/[neutral]/...).
# Anything before the first tag is internal monologue and must NEVER reach TTS.
_EMOTION_TAG = re.compile(r"\[(happy|neutral|thinking|surprised|sad|angry)\]")

# Fail-open: if the model never emits a tag (prompt not enforcing it),
# don't buffer the whole reply in silence — start streaming as-is after
# this many seconds. Bounded prefix loss beats minutes of dead air
# (which would also trip the 30 s watchdog and nuke the whole reply).
_TAG_WAIT_S = 12.0

# Max chars of pre-tag sentences held back while waiting for the tag.
# Overflow is released (spoken) rather than dropped — silence is worse
# than a leaked fragment.
_HELD_MAX_CHARS = 500


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


class HermesBackend(BaseLLMBackend):
    def __init__(self, url: str, api_key: str = ""):
        self.url = url.rstrip("/")
        self.api_key = api_key

    @staticmethod
    def _speakable(s: str) -> str:
        s = re.sub(r"\[[a-z]{2,30}\]", "", s)
        return re.sub(r"\s+", " ", s).strip()

    async def generate_response(self, text: str, session_id: str, stream_name: str, response_queue: asyncio.Queue):
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            headers["Accept"] = "text/event-stream"
            headers["X-Source"] = "voice_gateway"
            headers["X-Device-MAC"] = session_id
            headers["X-Stream-Name"] = stream_name
            # Hermes API Server: OpenAI-compatible /v1/chat/completions (streaming)
            # System prompt: force a leading emotion tag (the TTS gate keys
            # on it) and forbid verbalized tool-talk — this endpoint offers
            # no tools, so "Need to execute code…" must never be spoken.
            payload = {
                "model": "omniroute/oc/hy3-free",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты голосовой ассистент умной колонки. Начинай КАЖДЫЙ "
                            "ответ ровно с одного тега эмоции из списка [happy] "
                            "[neutral] [surprised] [sad] [angry] — без преамбул. "
                            "Тебе недоступны никакие инструменты: никогда не "
                            "озвучивай вызовы инструментов, служебные рассуждения "
                            "и внутренний монолог, отвечай сразу по существу."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "stream": True,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.url}/v1/chat/completions", json=payload, headers=headers
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Hermes API error: {resp.status}")
                        await response_queue.put(None)
                        return

                    buf = ""
                    holder = ""
                    seen_tag = False
                    held = []  # pre-tag sentences: discarded on tag, spoken on release
                    t_start = time.monotonic()

                    def flush(final: bool) -> None:
                        nonlocal buf, holder, seen_tag, held
                        m = re.search(r"\[thinking\]", buf)
                        if m and not re.search(r"\[/thinking\]", buf[m.start():]):
                            part, holder = (buf[:m.start()], buf[m.start():]) if not final else (buf, "")
                        else:
                            idx = buf.rfind("[")
                            if idx != -1 and not re.search(r"\]", buf[idx:]):
                                part, holder = buf[:idx], buf[idx:]
                            else:
                                part, holder = buf, ""
                        while True:
                            mm = re.search(r"\[thinking\](.*?)\[/thinking\]", part, flags=re.S)
                            if not mm:
                                break
                            part = part[:mm.start()] + part[mm.end():]
                        mtag = _EMOTION_TAG.search(part)
                        if not seen_tag:
                            if mtag:
                                seen_tag = True
                                held = []
                                part = part[mtag.start():]
                                # re-strip thinking blocks after cut
                                while True:
                                    mm0 = re.search(r"\[thinking\](.*?)\[/thinking\]", part, flags=re.S)
                                    if not mm0:
                                        break
                                    part = part[:mm0.start()] + part[mm0.end():]
                            elif not final and time.monotonic() - t_start <= _TAG_WAIT_S and sum(len(h) for h in held) < _HELD_MAX_CHARS:
                                # No tag yet: HOLD completed sentences (don't speak,
                                # don't discard) and keep only the tail buffered.
                                bi0 = _sentence_boundary(part)
                                if bi0 != -1:
                                    if part[:bi0].strip():
                                        held.append(part[:bi0])
                                    part = part[bi0:]
                                buf = (part + " " + holder).strip()
                                return
                            else:
                                # Release: tag wait over, held cap hit, or final
                                # flush — speak the held prefix in order, then
                                # process the rest normally. Never go silent.
                                seen_tag = True
                                if held:
                                    part = " ".join(held) + " " + part
                                    held = []
                        bi = _sentence_boundary(part)
                        if bi != -1:
                            done = [part[:bi]] if part[:bi].strip() else []
                            tail = part[bi:]
                        else:
                            done, tail = [], part
                        if final and tail.strip():
                            # Never speak a dangling "[abc" fragment cut off mid-tag.
                            tail = re.sub(r"\[[^\]]*$", "", tail).strip()
                            if tail:
                                done.append(tail)
                            tail = ""
                        for s in done:
                            s2 = HermesBackend._speakable(s)
                            if s2:
                                response_queue.put_nowait(s2)
                        buf = (tail + " " + holder).strip()

                    async for raw in resp.content:
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            continue
                        if delta:
                            buf += delta
                            if re.search(r"[.!?…](\s|$)", buf) or "\n" in buf or len(buf) > 400:
                                flush(final=False)
                    flush(final=True)
                    await response_queue.put(None)
        except Exception as e:
            logger.error(f"Hermes error: {e}")
            await response_queue.put(None)


class CascadeBackend(BaseLLMBackend):
    """Routes text through jev-router (SSE POST /route).

    Contract-compatible with HermesBackend but without the emotion-tag gate:
    router output is already curated, so we only strip tags and manage the
    *ack* timing the manifest requires for long L2 turns:

      * route complex_logic/expert -> immediate ack «Секунду, занимаюсь…»
        (Level-2 CodeAgent takes 48-120 s; silence would trip the 90 s
        watchdog and cut the reply);
      * route chat/easy            -> 3 s ack timer; cancelled as soon as
        the first real sentence arrives (fast paths never hear the ack);
      * progress events from L2    -> spoken as short sentences (worker
        emits them at most every 20 s), replacing watchdog-driven apologies;
      * error with nothing spoken  -> immediate apology sentence instead of
        dead air.

    The first played audio cancels the gateway watchdog downstream
    (_hermes_player_task.play), so one ack keeps the whole turn alive.
    """

    ACK_COMPLEX = "Секунду, занимаюсь…"
    ACK_THINK = "Секунду, думаю…"
    SORRY = "Простите, что-то пошло не так."

    def __init__(self, router_url: str, ack_delay: float = 3.0,
                 total_timeout: float = 200.0, sock_read_timeout: float = 60.0):
        self.router_url = router_url.rstrip("/")
        self.ack_delay = ack_delay
        self.total_timeout = total_timeout
        self.sock_read_timeout = sock_read_timeout

    async def generate_response(self, text: str, session_id: str, stream_name: str, response_queue: asyncio.Queue):
        spoken = False   # any real sentence reached the queue
        ack_sent = False
        ack_task: asyncio.Task | None = None

        def put(s: str) -> None:
            nonlocal spoken
            s2 = HermesBackend._speakable(s)
            if s2:
                response_queue.put_nowait(s2)
                spoken = True

        def put_ack(phrase: str) -> None:
            nonlocal ack_sent
            if ack_sent or spoken:
                return
            response_queue.put_nowait(phrase)
            ack_sent = True

        async def ack_later(phrase: str) -> None:
            await asyncio.sleep(self.ack_delay)
            put_ack(phrase)

        try:
            headers = {"Accept": "text/event-stream"}
            payload = {
                "text": text,
                "session_id": session_id,
                "stream_name": stream_name,
            }
            timeout = aiohttp.ClientTimeout(
                total=self.total_timeout,
                sock_connect=5,
                sock_read=self.sock_read_timeout,
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.router_url}/route", json=payload, headers=headers
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"router http {resp.status}: {body[:200]}")

                    async for raw in resp.content:
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            ev = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        etype = ev.get("type")

                        if etype == "route":
                            route = ev.get("route", "")
                            logger.info(
                                f"🔀 [Cascade] route={route} conf={ev.get('confidence')} "
                                f"reason={ev.get('reason') or '-'}"
                            )
                            if route in ("complex_logic", "expert"):
                                if ack_task is not None:
                                    ack_task.cancel()
                                    ack_task = None
                                put_ack(self.ACK_COMPLEX)
                            elif not ack_sent and not spoken and ack_task is None:
                                ack_task = asyncio.create_task(
                                    ack_later(self.ACK_THINK)
                                )
                        elif etype == "sentence":
                            if ack_task is not None:
                                ack_task.cancel()
                                ack_task = None
                            if ev.get("text"):
                                put(ev["text"])
                        elif etype == "progress":
                            if ev.get("text"):
                                put(ev["text"])
                        elif etype == "error":
                            logger.error(
                                f"❌ [Cascade] router error: {ev.get('message')}"
                            )
                            # ack alone is not content: after it the watchdog is
                            # already cancelled, so silence here would be final.
                            if not spoken:
                                put(self.SORRY)
                        elif etype == "done":
                            logger.info(
                                f"⏱ [Cascade] done route={ev.get('route')} "
                                f"elapsed={ev.get('elapsed')}s"
                            )
                            break
        except Exception as e:
            logger.error(f"Cascade error: {e}")
            if not spoken:
                put(self.SORRY)
        finally:
            if ack_task is not None:
                ack_task.cancel()
            if not spoken:
                # Stream ended empty without an error event: never leave the
                # caller in silence (watchdog would apologize 90 s later).
                response_queue.put_nowait(self.SORRY)
            response_queue.put_nowait(None)
