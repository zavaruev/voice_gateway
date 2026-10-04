"""Telegram — the gateway's third request source (after ESP32 WS + cameras).

Why a separate module (and why no bot library):
  * the process already speaks aiohttp everywhere — the Bot API is a handful
    of JSON endpoints (getUpdates / sendMessage / editMessageText /
    sendVoice), so a framework would bring its own dispatcher and loop for
    no gain;
  * the reply brain is the SAME `llm_backend` object the devices use, so a
    turn is: voice note -> Whisper -> generate_response() -> text (+ an
    optional TTS voice note). There is no device session, wake word, VAD or
    echo guard here — none of them apply to a person deliberately speaking
    into a phone (which is also why this module has its own lighter
    transcript validation, is_valid_voice_text()).

Injection instead of import: fetch_transcription() and synthesize_tts_mp3()
live in main.py, which imports THIS module — importing main back would be a
cycle. build_telegram_bot() therefore takes them as callables, which also
makes the tests trivial (fake api + fake backend, zero network).

Contract kept identical to the device paths:
  * stream_name = "tg:<chat_id>" — the jev-router history/memory key, so
    every chat owns an isolated turn ring and never shares context with a
    camera (history keys on `stream_name or session_id`);
  * room        = the /room binding — passed through the backend's `room`
    kwarg to the router's RouteRequest.room and used ONLY as the resolver's
    default area (never as the history key — that double duty was the trap
    in the first draft of this plan);
  * queue drain — sentences + exactly one None, backends never raise;
  * turn cap    = TELEGRAM_TURN_TIMEOUT: on expiry the partial text is sent
    (or an apology) — there is no device watchdog to reuse.

Env (read in main.py, values live in .env.voice_gateway — docker-compose
env_file, gitignored):
  TELEGRAM_BOT_TOKEN          empty = source disabled
  TELEGRAM_ALLOWED_CHAT_IDS   comma-separated; EMPTY = nobody is served
  TELEGRAM_REPLY_VOICE        true  = also reply as a voice note
  TELEGRAM_ALLOW_GROUPS       false = private chats only
  TELEGRAM_MAX_VOICE_S        60    = refuse longer voice notes
  TELEGRAM_TURN_TIMEOUT       120   = per-turn cap (matches the device 120 s)
  TELEGRAM_COOLDOWN_S         1.5   = spacing between turn starts in a chat
  TELEGRAM_API_BASE           https://api.telegram.org (overridable for tests)
Module constants (deliberately not env — they are protocol/UX details):
  EDIT_INTERVAL, POLL_TIMEOUT, BACKOFF_MAX, STREAM_LIMIT, VOICE_MAX_CHARS,
  MAX_CONCURRENT_TURNS, MAX_PENDING_PER_CHAT, MAX_TEXT_CHARS.
"""

import asyncio
import io
import json
import os
import re
import subprocess
import time

import aiohttp
from loguru import logger
from pydub import AudioSegment

from audio_utils import SINGLE_WORD_HALLUCINATIONS, WHISPER_HALLUCINATIONS_PATTERN
# Ack phrases the cascade backend speaks while it works — recognised so they
# become a separate status bubble instead of leaking into the final text and,
# worse, into the synthesized voice reply.
from backends import CascadeBackend

API_BASE = "https://api.telegram.org"

# Telegram caps messages at 4096 chars; 4000 leaves headroom for the split
# marker. STREAM_LIMIT is what one reply bubble holds while it is edited.
MSG_LIMIT = 4096
STREAM_LIMIT = 4000

# Minimum spacing between editMessageText calls on one message (the API
# silently throttles ~1 edit/s and answers 400 "message is not modified"
# when the text did not change — TelegramAPI.edit_message maps that to a
# benign no-op).
EDIT_INTERVAL = 1.0

# Long-poll window. Must stay well under 50 s (Bot API max) and be short
# enough that a shutdown cancel is not stuck waiting a whole minute.
POLL_TIMEOUT = 25
BACKOFF_MAX = 60.0

# A typing action expires after ~5 s; refresh it for the whole turn so a
# long L2 wait shows «печатает…» instead of dead silence (the device path
# plays an ack phrase for the same reason).
TYPING_INTERVAL = 4.0

# Concurrent LLM turns from all chats combined — protects the L2 worker
# (WORKER_TIMEOUT 120 s) from a household firing several requests at once.
MAX_CONCURRENT_TURNS = 2
# Per-chat pending queue depth; deeper is a flood, not a conversation.
MAX_PENDING_PER_CHAT = 3
# Typed messages longer than this are refused (a voice assistant, not a
# paste bin; voice notes are already bounded by TELEGRAM_MAX_VOICE_S).
MAX_TEXT_CHARS = 1000

# First N chars of the reply that get synthesized into the voice note — a
# long L2 answer must not become a five-minute recording when the full text
# is already in the message above it.
VOICE_MAX_CHARS = 900

# Voice notes are Ogg/Opus by Telegram spec; other audio attachments (mp3,
# m4a sent as a file) are re-containered to match the Whisper contract.
OGG_EXTS = ("ogg", "oga", "opus")

ACK_PHRASES = frozenset(
    {
        CascadeBackend.ACK_COMPLEX,
        CascadeBackend.ACK_THINK,
        CascadeBackend.SORRY,
    }
)
# Phrases that already close a turn on their own — never double up with a
# second apology when the queue delivered nothing else.
CLOSING_PHRASES = frozenset({CascadeBackend.SORRY})

FALLBACK_TIMEOUT = "Простите, я задумалась. Повторите пожалуйста."
FALLBACK_ERROR = "Простите, что-то пошло не так."
HINT_UNSUPPORTED = "Принимаю голосовые сообщения и текст."
HINT_NOT_HEARD = "Не расслышала. Повторите, пожалуйста, голосом короче."
HINT_VOICE_TOO_LONG = (
    "Слишком длинное голосовое ({n} с, максимум {max} с). Скажите короче."
)
HINT_TEXT_TOO_LONG = (
    "Слишком длинное сообщение ({n} символов, максимум {max}). Сократите, пожалуйста."
)
HINT_DOWNLOAD_FAILED = "Не удалось скачать голосовое. Отправьте ещё раз."

GREETING = (
    "Привет! Я — голосовой ассистент дома.\n"
    "Можно прислать голосовое (распознаю и отвечу) или написать текстом.\n"
    "\n"
    "Команды:\n"
    "/room кухня — комната по умолчанию для включений;\n"
    "/voice on | /voice off — отвечать ли голосом."
)

# Rooms the router can actually default to (STREAM_DEFAULT_AREA keys in
# services/jev-router/resolver.py) plus RU stems a user would type. A room
# outside this set would silently fall back to "no default" and escalate
# every bare «включи свет» to the slow L2 path — so it is refused instead.
KNOWN_ROOMS = ("kitchen", "livingroom", "living_room", "corridor", "hallway", "bedroom")
ROOM_WORDS: tuple[tuple[str, str], ...] = (
    ("кухн", "kitchen"),
    ("гостин", "livingroom"),
    ("гостиная", "livingroom"),
    ("зал", "livingroom"),
    ("комнат", "livingroom"),
    ("коридор", "corridor"),
    ("прихож", "corridor"),
    ("холл", "corridor"),
    ("спальн", "bedroom"),
    ("кроват", "bedroom"),
)
ROOMS_HINT = "кухня, гостиная, коридор, спальня"


def normalize_room(raw: str) -> str | None:
    """User input -> a STREAM_DEFAULT_AREA key, or None if unknown.

    Accepts canonical names (kitchen, corridor...) and Russian stems in any
    inflection («кухне», «кухонный» all map to kitchen via ROOM_WORDS).
    """
    val = (raw or "").strip().lower().replace("ё", "е")
    if not val:
        return None
    if val in KNOWN_ROOMS:
        return val
    # English aliases first (living room with a space, hallway).
    if val.replace(" ", "_") in KNOWN_ROOMS:
        return val.replace(" ", "_")
    if val in ("hallway", "hall"):
        return "corridor"
    for stem, room in ROOM_WORDS:
        if stem in val:
            return room
    return None


def is_valid_voice_text(txt: str) -> bool:
    """Telegram variant of audio_utils.is_valid_text() for deliberate speech.

    The device filter is biased toward false REJECTs because an ambient mic
    hears TTS echoes and silence hallucinations. A phone voice note is
    recorded on purpose, near the mouth, with no echo loop — so the
    echo-specific gates there (SHORT_ECHO_PATTERNS, the <=2-char word rule)
    would reject the most likely answer to the bot's question: «да».
    Kept: Whisper's silence hallucinations, character-run and repetition
    garbage — those come from the model, not from the room.
    """
    clean = (txt or "").strip(" .,?!-").lower()
    if not clean:
        return False
    # Whisper's inventions on silence: «о», «а», «как», «жизнь»...
    if clean in SINGLE_WORD_HALLUCINATIONS:
        return False
    # Long runs of one character («кккккк...»).
    if len(clean) >= 5:
        max_run, cur = 1, 1
        for i in range(1, len(clean)):
            cur = cur + 1 if clean[i] == clean[i - 1] else 1
            max_run = max(max_run, cur)
        if max_run / len(clean) > 0.4:
            return False
    # Whole text being one repeated short substring («каккак», «ататат»).
    for sub_len in (1, 2, 3, 4):
        if len(clean) >= sub_len * 2:
            for start in range(min(sub_len, len(clean) - sub_len + 1)):
                sub = clean[start : start + sub_len]
                reps = len(clean) // len(sub)
                if reps >= 2 and clean == sub * reps:
                    return False
    # «да да да да» — model looping, not a human.
    words = clean.split()
    if len(words) >= 3 and len(set(words)) == 1 and len(words[0]) <= 4:
        return False
    if WHISPER_HALLUCINATIONS_PATTERN.search(clean):
        return False
    return True


def split_reply(text: str, limit: int = STREAM_LIMIT) -> list[str]:
    """Split a long reply into <=limit chunks at sentence/word boundaries.

    Telegram refuses messages over MSG_LIMIT; a hard slice at 4000 would cut
    mid-word. Prefers sentence ends, then any space past the half-way mark,
    and only then a hard cut (for a pathological run-on).
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = 0
        for sep in (". ", "! ", "? ", "… ", "; ", ", ", " "):
            idx = window.rfind(sep)
            if idx > limit // 2:
                cut = idx + len(sep)
                break
        if cut <= 0:
            cut = limit
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        chunks.append(rest)
    return [c for c in chunks if c]


def prepare_stt_audio(
    data: bytes, ext: str, known_duration: float | None
) -> tuple[bytes | None, float]:
    """Telegram audio attachment -> (Ogg/Opus bytes for Whisper, seconds).

    Runs in a worker thread — pydub/ffmpeg are blocking and the loop has a
    stall monitor (main.loop_stall_monitor) that must stay quiet.

    Why re-container non-Ogg input: main.fetch_transcription() labels the
    upload `a.ogg` (the ESP32 path always sends Ogg/Opus), so we match that
    contract exactly instead of trusting the server to sniff an mp3 from an
    .ogg filename. An already-Ogg voice note passes through untouched — no
    double encode, no quality loss — and its duration comes from Telegram's
    `duration` field, no decode needed.
    """
    try:
        if ext in OGG_EXTS:
            if known_duration is not None:
                return data, float(known_duration)
            seg = AudioSegment.from_file(io.BytesIO(data), format="ogg")
            return data, len(seg) / 1000.0
        seg = AudioSegment.from_file(io.BytesIO(data))  # ffmpeg sniffs mp3/m4a/...
        # Opus only accepts 8/12/16/24/48 kHz — normalise before export or
        # a 44.1 kHz mp3 makes the encoder fail.
        seg = seg.set_frame_rate(48000).set_channels(1)
        buf = io.BytesIO()
        seg.export(buf, format="ogg", codec="libopus", bitrate="32k")
        return buf.getvalue(), len(seg) / 1000.0
    except Exception as e:
        logger.warning(f"[Telegram] audio decode failed ({ext}): {e}")
        return None, 0.0


def mp3_to_voice_ogg(mp3: bytes) -> bytes | None:
    """MP3 -> Ogg/Opus — Telegram renders sendVoice as a voice message only
    for Ogg/Opus; an MP3 arrives as a player attachment (or is refused).

    Sync subprocess by design: called via asyncio.to_thread() so the event
    loop never blocks. Same ffmpeg pipe idea as the camera's speaker-id path
    (camera_client._fetch_speaker_id).
    """
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-loglevel", "error",
                "-i", "pipe:0",
                "-c:a", "libopus", "-b:a", "32k", "-ar", "48000", "-ac", "1",
                "-f", "ogg", "pipe:1",
            ],
            input=mp3,
            capture_output=True,
            timeout=20,
        )
        if proc.returncode != 0 or not proc.stdout:
            logger.error(
                f"[Telegram] mp3->ogg failed: {proc.stderr[:200]!r}"
            )
            return None
        return proc.stdout
    except Exception as e:
        logger.error(f"[Telegram] mp3->ogg error: {e}")
        return None


class TelegramAPIError(Exception):
    """A non-fatal Bot API error (HTTP 4xx/5xx other than auth)."""

    def __init__(self, method: str, status: int, description: str):
        super().__init__(f"{method}: HTTP {status} {description}")
        self.method = method
        self.status = status
        self.description = description
        self.retry_after: float | None = None
        if status == 429:
            m = re.search(r"retry after (\d+)", description)
            if m:
                self.retry_after = float(m.group(1))

    @property
    def retryable(self) -> bool:
        """Server errors and rate limits deserve a retry; other 4xx do not."""
        return self.status == 429 or self.status >= 500


class TelegramFatalError(Exception):
    """Auth is broken (401/403) — retrying is pointless, stop the source."""


async def _await_cancelled(task: asyncio.Task) -> None:
    """Await a task we just cancelled WITHOUT swallowing our own cancellation.

    `await gen` on a task we cancelled raises CancelledError in THIS task
    too — but that only means "the producer is gone" and must be absorbed,
    or the chat worker would die on every timeout. If the current task is
    itself being cancelled (shutdown), it must propagate. Task.cancelling()
    (3.11+) tells the two apart; older Pythons fall back to swallowing.
    """
    try:
        await task
    except asyncio.CancelledError:
        cur = asyncio.current_task()
        if cur is not None and getattr(cur, "cancelling", lambda: 0)():
            raise
    except Exception:
        pass


class TelegramAPI:
    """Thin aiohttp wrapper over the Bot API (no library on purpose).

    The owning TelegramBot creates the ClientSession in run() and assigns
    it to `session` — one session for updates, uploads and downloads.
    """

    def __init__(self, token: str, api_base: str = API_BASE):
        self.token = token
        self.base = api_base.rstrip("/")
        self.session: aiohttp.ClientSession | None = None

    @property
    def _root(self) -> str:
        return f"{self.base}/bot{self.token}"

    async def _post(
        self,
        method: str,
        *,
        json_body: dict | None = None,
        form: aiohttp.FormData | None = None,
        params: dict | None = None,
        timeout: float = 30,
    ) -> dict:
        if self.session is None:
            raise RuntimeError("TelegramAPI.session is not set — call run() first")
        url = f"{self._root}/{method}"
        try:
            if form is not None:
                async with self.session.post(
                    url, params=params, data=form, timeout=timeout
                ) as r:
                    body = await r.json(content_type=None)
            else:
                async with self.session.post(
                    url, params=params, json=json_body or {}, timeout=timeout
                ) as r:
                    body = await r.json(content_type=None)
        except aiohttp.ClientError as e:
            raise TelegramAPIError(method, 0, str(e)) from e
        except asyncio.TimeoutError as e:
            raise TelegramAPIError(method, 0, "timeout") from e
        except ValueError as e:
            # Non-JSON body (proxy error page, empty response).
            raise TelegramAPIError(method, 0, f"invalid json: {e}") from e
        status = getattr(r, "status", 0)
        if status in (401, 403):
            raise TelegramFatalError(f"{method}: HTTP {status} {body}")
        if status != 200 or not (isinstance(body, dict) and body.get("ok")):
            desc = ""
            if isinstance(body, dict):
                desc = str(body.get("description", ""))[:200]
            else:
                desc = str(body)[:200]
            raise TelegramAPIError(method, status, desc)
        return body.get("result") if isinstance(body, dict) else None

    async def get_me(self) -> dict:
        return await self._post("getMe", timeout=10)

    async def delete_webhook(self, drop_pending_updates: bool = False) -> bool:
        """A leftover webhook makes getUpdates answer 409 Conflict.

        The startup call also drops everything queued while the container
        was down — answering day-old messages would be bizarre. The 409
        recovery call must NOT drop: those are live messages.
        """
        return await self._post(
            "deleteWebhook",
            json_body={"drop_pending_updates": drop_pending_updates},
            timeout=10,
        )

    async def get_updates(self, offset: int, timeout: int = POLL_TIMEOUT) -> list[dict]:
        return await self._post(
            "getUpdates",
            json_body={
                "offset": offset,
                "timeout": timeout,
                "limit": 100,
                "allowed_updates": ["message"],
            },
            # The HTTP timeout must outlive the long poll itself.
            timeout=timeout + 15,
        )

    async def send_message(
        self, chat_id: str, text: str, reply_to: int | None = None
    ) -> int:
        payload: dict = {"chat_id": chat_id, "text": text[:MSG_LIMIT]}
        if reply_to is not None:
            payload["reply_to_message_id"] = reply_to
        result = await self._post("sendMessage", json_body=payload)
        return int(result.get("message_id", 0)) if isinstance(result, dict) else 0

    async def edit_message(self, chat_id: str, message_id: int, text: str) -> bool:
        """Edit the streamed reply bubble. False = benign no-op (the text did
        not change); anything else raises for the caller to log."""
        try:
            await self._post(
                "editMessageText",
                json_body={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": text[:MSG_LIMIT],
                },
            )
            return True
        except TelegramAPIError as e:
            if "message is not modified" in e.description:
                return False
            raise

    async def send_chat_action(self, chat_id: str) -> bool:
        return bool(
            await self._post(
                "sendChatAction",
                json_body={"chat_id": chat_id, "action": "typing"},
                timeout=10,
            )
        )

    async def get_file(self, file_id: str) -> str:
        result = await self._post("getFile", json_body={"file_id": file_id})
        return str(result.get("file_path", "")) if isinstance(result, dict) else ""

    async def download_file(self, file_path: str) -> bytes:
        url = f"{self.base}/file/bot{self.token}/{file_path}"
        try:
            async with self.session.get(url, timeout=60) as r:
                if r.status != 200:
                    raise TelegramAPIError("downloadFile", r.status, "non-200")
                return await r.read()
        except aiohttp.ClientError as e:
            raise TelegramAPIError("downloadFile", 0, str(e)) from e

    async def send_voice(
        self, chat_id: str, ogg: bytes, reply_to: int | None = None
    ) -> int:
        form = aiohttp.FormData()
        form.add_field("voice", ogg, filename="voice.ogg", content_type="audio/ogg")
        params: dict = {"chat_id": str(chat_id)}
        if reply_to is not None:
            params["reply_to_message_id"] = str(reply_to)
        result = await self._post("sendVoice", form=form, params=params, timeout=60)
        return int(result.get("message_id", 0)) if isinstance(result, dict) else 0


class _ReplySender:
    """Streams one reply into a chat.

    Message 1 carries the reply and is grown with editMessageText (the
    caller throttles); text beyond STREAM_LIMIT becomes extra messages at
    the final flush. `status()` posts an independent bubble (ack phrases).
    """

    def __init__(self, api: TelegramAPI, chat_id: str, reply_to: int | None = None):
        self.api = api
        self.chat_id = chat_id
        self.reply_to = reply_to
        self.msg_id: int | None = None
        self.last_status: str = ""
        self.sent_count = 0

    async def status(self, text: str) -> None:
        """Best-effort side bubble — never fails the turn."""
        try:
            await self.api.send_message(self.chat_id, text, reply_to=self.reply_to)
            self.last_status = text
            self.sent_count += 1
        except TelegramFatalError:
            raise
        except Exception as e:
            logger.warning(f"[Telegram] status send failed: {e}")

    async def flush(self, text: str, final: bool = False) -> None:
        """Push the accumulated text to Telegram (idempotent on errors)."""
        text = (text or "").strip()
        if not text:
            return
        try:
            if final:
                chunks = split_reply(text)
                if not chunks:
                    return
                if self.msg_id is None:
                    self.msg_id = await self.api.send_message(
                        self.chat_id, chunks[0], reply_to=self.reply_to
                    )
                else:
                    await self.api.edit_message(self.chat_id, self.msg_id, chunks[0])
                for extra in chunks[1:]:
                    await self.api.send_message(self.chat_id, extra)
                self.sent_count += len(chunks)
            else:
                display = (
                    text if len(text) <= STREAM_LIMIT else text[: STREAM_LIMIT - 1] + "…"
                )
                if self.msg_id is None:
                    self.msg_id = await self.api.send_message(
                        self.chat_id, display, reply_to=self.reply_to
                    )
                    self.sent_count += 1
                else:
                    await self.api.edit_message(self.chat_id, self.msg_id, display)
        except TelegramFatalError:
            raise
        except Exception as e:
            # A deleted/old message must not kill the turn: log once and
            # keep the text flowing to the final flush attempt.
            logger.warning(f"[Telegram] reply flush failed: {e}")


class TelegramBot:
    """Long-polling bot: updates -> per-chat worker -> shared LLM backend.

    Updates from different chats run in parallel (a slow L2 turn must not
    block another room), updates inside one chat are strictly ordered by
    its worker queue. A turn never touches device state: no wake gate, no
    watchdog task — only the TURN_TIMEOUT cap and the queue contract.
    """

    def __init__(
        self,
        *,
        api: TelegramAPI,
        allowed_chat_ids,
        backend,
        transcribe,
        tts_mp3,
        make_session_id,
        reply_voice: bool = True,
        allow_groups: bool = False,
        max_voice_s: int = 60,
        turn_timeout: float = 120.0,
        cooldown: float = 1.5,
        state_file: str = "",
        log_transcripts: bool = False,
        max_pending: int = MAX_PENDING_PER_CHAT,
        max_text_chars: int = MAX_TEXT_CHARS,
    ):
        self.api = api
        # Empty allowlist denies everyone (a token in a public repo must
        # never buy access to the house by default).
        self.allowed = {str(c).strip() for c in (allowed_chat_ids or []) if str(c).strip()}
        self.backend = backend
        self._transcribe = transcribe          # (ogg_bytes, session) -> str
        self._tts_mp3 = tts_mp3                # (text, session) -> mp3 | None
        self._make_session_id = make_session_id  # chat_id -> stable session id
        self.reply_voice = reply_voice
        self.allow_groups = allow_groups
        self.max_voice_s = max_voice_s
        self.turn_timeout = turn_timeout
        self.cooldown = cooldown
        self.state_file = state_file
        self.log_transcripts = log_transcripts
        self.max_pending = max_pending
        self.max_text_chars = max_text_chars

        self._session: aiohttp.ClientSession | None = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._next_start: dict[str, float] = {}
        self._ignored: set[str] = set()
        self._seen_update_id = -1
        self._turn_sem = asyncio.Semaphore(MAX_CONCURRENT_TURNS)
        self._state: dict = self._load_state()

    # ------------------------------------------------------------------ state

    def _load_state(self) -> dict:
        """Per-chat prefs (room, voice toggle) from disk; missing/broken
        file = empty prefs, never an exception (same rule as devices.json)."""
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.warning(f"[Telegram] state load failed: {e}")
            return {}

    def _write_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)

    async def _save_state(self) -> None:
        try:
            await asyncio.to_thread(self._write_state)
        except Exception as e:
            logger.error(f"[Telegram] state save failed: {e}")

    def chat_room(self, chat_id: str) -> str:
        return str(self._state.get(chat_id, {}).get("room", "") or "")

    def chat_voice(self, chat_id: str) -> bool:
        return bool(self._state.get(chat_id, {}).get("voice", self.reply_voice))

    def _stream_name(self, chat_id: str) -> str:
        """Unique history/memory key per chat — never a bare room name (that
        would merge this chat's turns into the camera's history ring)."""
        return f"tg:{chat_id}"

    def _serves(self, chat_id: str, msg: dict) -> bool:
        """Allowlist + group gate. Checked in dispatch() (before a queue
        even exists) AND in handle_message() so no internal caller can ever
        bypass it — a leaked token must not reach the LLM."""
        if chat_id not in self.allowed:
            if chat_id not in self._ignored:
                self._ignored.add(chat_id)
                logger.warning(
                    f"[Telegram] ignored message from non-allowlisted chat {chat_id}"
                )
            return False
        chat_type = ((msg.get("chat") or {}).get("type")) or "private"
        if chat_type != "private" and not self.allow_groups:
            return False
        return True

    # ------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        """Own the session and poll until cancelled (started in main.on_startup)."""
        timeout = aiohttp.ClientTimeout(total=90, connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            self._session = sess
            self.api.session = sess
            try:
                await self._startup_checks()
                await self._poll_loop()
            except TelegramFatalError as e:
                # Bad token: log here so the task ends cleanly instead of
                # failing later in on_shutdown's await.
                logger.error(f"[Telegram] fatal error, source stopped: {e}")
            finally:
                await self._stop_workers()
                self.api.session = None
                self._session = None

    async def _startup_checks(self) -> None:
        if not self.allowed:
            logger.warning(
                "[Telegram] TELEGRAM_ALLOWED_CHAT_IDS is empty — every chat will be ignored"
            )
        try:
            me = await self.api.get_me()
            logger.info(f"✈️ [Telegram] bot @{me.get('username', '?')} started")
        except TelegramFatalError:
            raise
        except Exception as e:
            # Unreachable right now (boot race, DNS) — the poll loop will
            # retry with backoff; a bad token surfaces there as fatal.
            logger.warning(f"[Telegram] getMe failed, will retry in poll loop: {e}")
        try:
            await self.api.delete_webhook(drop_pending_updates=True)
        except TelegramFatalError:
            raise
        except Exception as e:
            logger.warning(f"[Telegram] deleteWebhook failed: {e}")

    async def _poll_loop(self) -> None:
        offset = 0
        backoff = 1.0
        while True:
            try:
                updates = await self.api.get_updates(offset, timeout=POLL_TIMEOUT)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except TelegramFatalError as e:
                raise  # bad token: run() logs it and stops the source
            except TelegramAPIError as e:
                if e.status == 409:
                    # A webhook reappeared (or was never deleted): clear it
                    # WITHOUT dropping updates — those are live messages.
                    try:
                        await self.api.delete_webhook(drop_pending_updates=False)
                    except Exception:
                        pass
                    await asyncio.sleep(1.0)
                    continue
                delay = e.retry_after or backoff
                logger.warning(f"[Telegram] poll error ({e}); retry in {delay:.0f}s")
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, BACKOFF_MAX)
                continue
            except Exception as e:
                logger.warning(f"[Telegram] poll failed ({e}); retry in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)
                continue

            for update in updates or []:
                uid = update.get("update_id", -1)
                if uid <= self._seen_update_id:
                    continue  # redelivery guard
                self._seen_update_id = uid
                # Acknowledge before handling: a turn lost to a crash is
                # acceptable, answering the same message twice is not.
                offset = uid + 1
                try:
                    self.dispatch(update)
                except Exception as e:
                    logger.error(f"[Telegram] dispatch failed: {e}")

    async def _stop_workers(self) -> None:
        for task in list(self._workers.values()):
            task.cancel()
        for task in list(self._workers.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._workers.clear()
        self._queues.clear()

    def dispatch(self, update: dict) -> None:
        """Allowlist/group gate + per-chat queueing (called from poll loop)."""
        msg = update.get("message")
        if not isinstance(msg, dict):
            return
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if not chat_id:
            return
        if not self._serves(chat_id, msg):
            return
        queue = self._queues.get(chat_id)
        if queue is None:
            queue = asyncio.Queue()
            self._queues[chat_id] = queue
            self._workers[chat_id] = asyncio.create_task(
                self._worker(chat_id), name=f"tg-chat-{chat_id}"
            )
        if queue.qsize() >= self.max_pending:
            logger.warning(
                f"[Telegram] chat {chat_id}: queue full ({self.max_pending}), message dropped"
            )
            return
        queue.put_nowait(msg)

    async def _worker(self, chat_id: str) -> None:
        """Strictly ordered turn processing for one chat."""
        queue = self._queues[chat_id]
        while True:
            msg = await queue.get()
            try:
                await self._throttle(chat_id)
                await self.handle_message(chat_id, msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # One bad turn must not take the chat (or the bot) down.
                logger.error(f"[Telegram] turn failed for {chat_id}: {e}")

    async def _throttle(self, chat_id: str) -> None:
        """Space turn starts within a chat (protects the LLM from floods)."""
        wait = self._next_start.get(chat_id, 0.0) - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)

    # ------------------------------------------------------------ turn entry

    async def _send(self, chat_id: str, text: str, reply_to: int | None = None) -> bool:
        """Best-effort plain message (commands, refusals, hints).

        A send failure must not take a turn down: the Bot API is the ONLY
        channel we have, so there is nothing to tell the user about it — log
        and carry on. `TelegramFatalError` is swallowed too: the poll loop
        hits the same auth error on its next getUpdates and stops the source
        with a clear log, which is the right place to report it.
        """
        try:
            await self.api.send_message(chat_id, text, reply_to=reply_to)
            return True
        except Exception as e:
            logger.warning(f"[Telegram] send to {chat_id} failed: {e}")
            return False

    async def handle_message(self, chat_id: str, msg: dict) -> str:
        """Route one message: command / voice / text / unsupported.

        Returns the final reply text ("" = nothing usable) so callers and
        tests can assert on the outcome.
        """
        if not self._serves(str(chat_id), msg):
            return ""
        chat_id = str(chat_id)
        text = (msg.get("text") or msg.get("caption") or "").strip()
        if text.startswith("/"):
            await self._command(chat_id, msg, text)
            return ""
        if msg.get("voice") or msg.get("audio"):
            return await self._voice_turn(chat_id, msg)
        if text:
            return await self._text_turn(chat_id, msg, text)
        await self._send(chat_id, HINT_UNSUPPORTED, reply_to=msg.get("message_id"))
        return ""

    async def _text_turn(self, chat_id: str, msg: dict, text: str) -> str:
        if len(text) > self.max_text_chars:
            await self._send(
                chat_id,
                HINT_TEXT_TOO_LONG.format(n=len(text), max=self.max_text_chars),
                reply_to=msg.get("message_id"),
            )
            return ""
        return await self._run_turn(chat_id, text, msg.get("message_id"))

    async def _voice_turn(self, chat_id: str, msg: dict) -> str:
        media = msg.get("voice") or msg.get("audio") or {}
        file_id = media.get("file_id")
        reply_to = msg.get("message_id")
        if not file_id:
            return ""
        known_dur = media.get("duration")
        # Cheapest guard first: refuse before downloading anything.
        if known_dur and int(known_dur) > self.max_voice_s:
            await self._send(
                chat_id,
                HINT_VOICE_TOO_LONG.format(n=int(known_dur), max=self.max_voice_s),
                reply_to=reply_to,
            )
            return ""
        try:
            path = await self.api.get_file(file_id)
            if not path:
                raise TelegramAPIError("getFile", 0, "empty file_path")
            data = await self.api.download_file(path)
        except Exception as e:
            logger.warning(f"[Telegram] voice download failed: {e}")
            await self._send(chat_id, HINT_DOWNLOAD_FAILED, reply_to=reply_to)
            return ""

        ext = path.rsplit(".", 1)[-1].lower() if "." in path else "ogg"
        dur = float(known_dur) if known_dur else None
        ogg, duration = await asyncio.to_thread(
            prepare_stt_audio, data, ext, dur
        )
        if not ogg:
            await self._send(chat_id, HINT_DOWNLOAD_FAILED, reply_to=reply_to)
            return ""
        # Authoritative duration (Telegram's field is an int, this is exact).
        if duration > self.max_voice_s:
            await self._send(
                chat_id,
                HINT_VOICE_TOO_LONG.format(n=int(duration), max=self.max_voice_s),
                reply_to=reply_to,
            )
            return ""

        transcript = await self._transcribe(ogg, self._session)
        # Transcript logging mirrors the device path: full text only when
        # opted in, lengths otherwise.
        if self.log_transcripts:
            logger.info(f"✈️ [Telegram:{chat_id}] voice -> '{transcript}'")
        else:
            logger.info(
                f"✈️ [Telegram:{chat_id}] voice -> [REDACTED] ({len(transcript or '')} chars)"
            )
        if not transcript or not is_valid_voice_text(transcript):
            await self._send(chat_id, HINT_NOT_HEARD, reply_to=reply_to)
            return ""
        return await self._run_turn(chat_id, transcript, reply_to)

    async def _run_turn(self, chat_id: str, text: str, reply_to: int | None) -> str:
        """One LLM turn under the concurrency cap, with a typing indicator."""
        async with self._turn_sem:
            self._next_start[chat_id] = time.monotonic() + self.cooldown
            typing = asyncio.create_task(self._typing_loop(chat_id))
            sender = _ReplySender(self.api, chat_id, reply_to)
            try:
                final_text = await self._stream_reply(chat_id, text, sender)
            finally:
                typing.cancel()
                await _await_cancelled(typing)
        if final_text and self.chat_voice(chat_id):
            await self._voice_reply(chat_id, final_text, reply_to)
        return final_text

    async def _typing_loop(self, chat_id: str) -> None:
        while True:
            try:
                await self.api.send_chat_action(chat_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # a failed action is invisible to the user
            await asyncio.sleep(TYPING_INTERVAL)

    async def _stream_reply(self, chat_id: str, text: str, sender: _ReplySender) -> str:
        """Drain generate_response() into Telegram with live edits.

        Returns the final reply text ("" = nothing usable arrived). Ack
        phrases become their own status bubble and never enter the text, so
        the voice reply cannot say «Секунду, занимаюсь…» after the fact.
        """
        queue: asyncio.Queue = asyncio.Queue()
        gen = asyncio.create_task(
            self.backend.generate_response(
                text=text,
                session_id=self._make_session_id(chat_id),
                stream_name=self._stream_name(chat_id),
                response_queue=queue,
                room=self.chat_room(chat_id),
            )
        )
        parts: list[str] = []
        timed_out = False
        errored = False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.turn_timeout
        last_flush = 0.0
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    timed_out = True
                    break
                if item is None:
                    break  # backend contract: exactly one None terminator
                if not isinstance(item, str) or not item.strip():
                    continue
                sentence = item.strip()
                if sentence in ACK_PHRASES:
                    await sender.status(sentence)
                    continue
                parts.append(sentence)
                now = loop.time()
                if now - last_flush >= EDIT_INTERVAL:
                    last_flush = now
                    await sender.flush(" ".join(parts))
        except Exception as e:
            logger.error(f"[Telegram] reply stream failed: {e}")
            errored = True
        finally:
            if gen.done():
                if not gen.cancelled() and gen.exception() is not None:
                    logger.warning(f"[Telegram] backend error: {gen.exception()}")
            else:
                # Timed out or stream broke: stop the producer (backends
                # open their own HTTP request, cancelling ends it).
                gen.cancel()
                await _await_cancelled(gen)

        final_text = " ".join(parts).strip()
        if final_text:
            await sender.flush(final_text, final=True)
            if timed_out:
                logger.warning(
                    f"[Telegram:{chat_id}] turn capped at {self.turn_timeout}s; "
                    "sent the partial reply"
                )
            return final_text
        # Nothing to say: close the turn unless the backend already did.
        if sender.last_status in CLOSING_PHRASES:
            return ""
        if timed_out:
            await sender.status(FALLBACK_TIMEOUT)
        else:
            # Covers both a silent backend and `errored`.
            await sender.status(FALLBACK_ERROR)
        return ""

    async def _voice_reply(self, chat_id: str, text: str, reply_to: int | None) -> None:
        """Text reply is already out — the voice note follows best-effort."""
        try:
            excerpt = self._voice_excerpt(text)
            if not excerpt:
                return
            mp3 = await self._tts_mp3(excerpt, self._session)
            if not mp3:
                logger.warning("[Telegram] TTS returned nothing; voice reply skipped")
                return
            ogg = await asyncio.to_thread(mp3_to_voice_ogg, mp3)
            if not ogg:
                return
            await self.api.send_voice(chat_id, ogg, reply_to=reply_to)
            logger.info(f"✈️ [Telegram:{chat_id}] voice reply ({len(ogg)} bytes)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # The user already has the text — a failed voice note is not a
            # failed turn.
            logger.warning(f"[Telegram] voice reply failed: {e}")

    @staticmethod
    def _voice_excerpt(text: str) -> str:
        """First sentences up to VOICE_MAX_CHARS (never split mid-sentence
        if avoidable)."""
        sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
        out = ""
        for s in sentences:
            if out and len(out) + 1 + len(s) > VOICE_MAX_CHARS:
                break
            out = f"{out} {s}".strip()
            if len(out) >= VOICE_MAX_CHARS:
                break
        return out or text[:VOICE_MAX_CHARS].strip()

    # -------------------------------------------------------------- commands

    async def _command(self, chat_id: str, msg: dict, text: str) -> None:
        parts = text.split()
        cmd = parts[0].split("@")[0].lower()
        args = parts[1:]
        reply_to = msg.get("message_id")
        if cmd in ("/start", "/help"):
            await self._send(chat_id, GREETING, reply_to=reply_to)
        elif cmd == "/room":
            await self._cmd_room(chat_id, reply_to, args)
        elif cmd == "/voice":
            await self._cmd_voice(chat_id, reply_to, args)
        else:
            await self._send(
                chat_id, "Не знаю такой команды. Вот что понимаю:", reply_to=reply_to
            )
            await self._send(chat_id, GREETING)

    async def _cmd_room(self, chat_id: str, reply_to, args: list[str]) -> None:
        current = self.chat_room(chat_id)
        if not args:
            if current:
                await self._send(
                    chat_id,
                    f"Комната по умолчанию: {current}. Сменить: /room {ROOMS_HINT}",
                    reply_to=reply_to,
                )
            else:
                await self._send(
                    chat_id,
                    f"Комната по умолчанию не задана. Указать: /room {ROOMS_HINT}",
                    reply_to=reply_to,
                )
            return
        room = normalize_room(args[0])
        if room is None:
            await self._send(
                chat_id,
                f"Не знаю комнаты «{args[0]}». Доступны: {ROOMS_HINT} "
                f"(или по-английски: {', '.join(KNOWN_ROOMS)}).",
                reply_to=reply_to,
            )
            return
        self._state.setdefault(chat_id, {})["room"] = room
        await self._save_state()
        await self._send(
            chat_id,
            f"Принято: комната по умолчанию — {room}. "
            "Комната в самой фразе («на кухне») всегда важнее.",
            reply_to=reply_to,
        )

    async def _cmd_voice(self, chat_id: str, reply_to, args: list[str]) -> None:
        if args:
            val = args[0].lower()
            if val in ("on", "1", "вкл", "включить", "да"):
                enabled = True
            elif val in ("off", "0", "выкл", "выключить", "нет"):
                enabled = False
            else:
                await self._send(
                    chat_id, "Понимаю /voice on и /voice off.", reply_to=reply_to
                )
                return
            self._state.setdefault(chat_id, {})["voice"] = enabled
            await self._save_state()
            await self._send(
                chat_id,
                "Принято: отвечаю голосом." if enabled else "Принято: отвечаю текстом.",
                reply_to=reply_to,
            )
            return
        current = self.chat_voice(chat_id)
        await self._send(
            chat_id,
            ("Голосовые ответы: вкл." if current else "Голосовые ответы: выкл.")
            + " Переключить: /voice on | /voice off",
            reply_to=reply_to,
        )


def build_telegram_bot(
    *,
    token: str,
    allowed_chat_ids,
    backend,
    transcribe,
    tts_mp3,
    make_session_id,
    reply_voice: bool = True,
    allow_groups: bool = False,
    max_voice_s: int = 60,
    turn_timeout: float = 120.0,
    cooldown: float = 1.5,
    api_base: str = API_BASE,
    state_file: str = "",
    log_transcripts: bool = False,
) -> TelegramBot:
    """Wire the bot to main.py's helpers (the anti-cycle seam).

    `transcribe` is main.fetch_transcription (Ogg -> Whisper text),
    `tts_mp3` adapts main.synthesize_tts_mp3 to a plain session, and
    `make_session_id` gives each chat a stable, salted conversation id
    (make_chat_id("tg:<chat>") in main).
    """
    return TelegramBot(
        api=TelegramAPI(token, api_base=api_base),
        allowed_chat_ids=allowed_chat_ids,
        backend=backend,
        transcribe=transcribe,
        tts_mp3=tts_mp3,
        make_session_id=make_session_id,
        reply_voice=reply_voice,
        allow_groups=allow_groups,
        max_voice_s=max_voice_s,
        turn_timeout=turn_timeout,
        cooldown=cooldown,
        state_file=state_file,
        log_transcripts=log_transcripts,
    )
