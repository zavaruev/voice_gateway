# =====================================================================================
# VOICE GATEWAY — main.py
# =====================================================================================
#
# PURPOSE
#   FastAPI gateway for the smart-home voice stack. It terminates the WebSocket of
#   every ESP32 satellite speaker, ingests their Opus audio, runs server-side VAD +
#   STT, forwards the recognised text into the LLM cascade and streams the spoken
#   answer back as Opus frames. It also serves the operator web UI
#   (templates/index.html), the REST/OTA API used by the firmware, and starts one
#   go2rtc camera session per configured stream (openWakeWord + camera TTS live in
#   camera_client.py).
#
# CASCADE PLACEMENT (this file is the gateway level below the router)
#   ESP32 satellite (device-side wake word, Opus capture)
#        |  WebSocket "/" : binary Opus frames + JSON control events
#        v
#   THIS FILE — silero VAD -> STT (WHISPER_URL) -> speaker ID -> backend dispatch
#        |  LLM_BACKEND=cascade : backends.CascadeBackend -> jev-router (L1,
#        |      ROUTER_URL, default :8091, SSE POST /route) -> smolagents-worker
#        |      (L2 smolagents CodeAgent on a free LLM) -> Hermes (L3 expert)
#        |      -> Home Assistant through MCP tools
#        |  LLM_BACKEND=hermes : backends.HermesBackend -> Hermes (L3) HTTP API
#        |      directly, skipping L1/L2
#        |  LLM_BACKEND=nanobot (default) : legacy backends.NanobotBackend -> the
#        |      Nanobot WebSocket brain (kept for compatibility)
#        v
#   TTS (TTS_URL, OpenAI-compatible) -> pydub resample -> Opus -> satellite speaker
#
# MAIN ENTRY POINTS
#   __main__            uvicorn.run(app, host 0.0.0.0, port 18792, ws_ping disabled —
#                       liveness is the manual ping/pong implemented in voice_ws()).
#   app (FastAPI)       also imported directly by tests (test_ota_auth, test_main).
#   on_startup/shutdown start/stop the go2rtc camera sessions.
#   voice_ws()          the satellite WebSocket; owns per-device session state.
#
# END-TO-END REQUEST / AUDIO LIFECYCLE
#   1. Auth + connect: voice_ws() checks ?token= or "Authorization: Bearer" against
#      NANOBOT_TOKEN with a constant-time compare, accepts, allocates a session_id
#      and a fresh per-session `state` dict (status IDLE), replies `hello` with the
#      audio contract (Opus 16 kHz mono, 60 ms frames) and asks the device for its
#      MCP tool list (tools/list, id 999).
#   2. Wake: device-side wake word -> JSON `listen:start` -> status LISTENING,
#      buffers cleared, VAD reset, screen 100% (ignored during the TTS cooldown so
#      the tail of our own playback cannot re-trigger us).
#   3. Capture: each binary WS message is one 60 ms Opus frame (v1 = raw,
#      v2 = 16-byte header, v3 = 4-byte header); decoded to PCM16@16k and scored by
#      VadEngine (Silero ONNX + adaptive energy floor) for the silence counter.
#   4. Cut: device `listen:stop` or server-side VAD (>VAD_SILENCE_FRAMES quiet
#      frames after real speech) -> status PROCESSING -> process_audio_and_send().
#   5. Gates: echo guard (camera_client.GLOBAL_TTS_UNTIL), min 15 frames,
#      ENERGY_THRESHOLD / MIN_SPEECH_RATIO, cross-device speaker lock — quiet or
#      noisy bursts never reach Whisper.
#   6. STT + identity: pack_ogg() then one round of parallel POSTs to WHISPER_URL
#      and SPEAKER_ID_URL; is_valid_text() (audio_utils.py) filters hallucinations
#      and mic echoes of our own TTS.
#   7. Dispatch: _handle_successful_transcription() echoes the text to the device
#      (`stt` event) and then either _dispatch_hermes() (cascade/hermes backends)
#      or the legacy Nanobot WS path; a WATCHDOG_TIMEOUT timer apologises and
#      re-opens the mic when the upstream AI stalls.
#   8. Speak-back: sentences land on an asyncio.Queue -> _hermes_player_task()
#      (sentence N+1 is synthesised while N plays) -> stream_tts_pcm() -> pydub ->
#      PCM 16 kHz mono -> Opus frames paced in real time -> device, bracketed by
#      `tts start`/`tts stop` control events plus a 1.5 s VAD cooldown afterwards.
#   9. Turn end: the finaliser probes the spoken reply for a question (the
#      HAS_QUESTION_* regexes via _reply_has_question()) and returns the state
#      machine to LISTENING for an adaptive follow-up window —
#      STANDBY_TIMEOUT_QUESTION after a question (screen back to 100%, dialogue
#      mode) or STANDBY_TIMEOUT_STATEMENT after a statement. Both backends share
#      it: NanobotResponseHandler._finalize_response() (legacy) and
#      _finalize_turn_followup() (shared by both — the cascade/hermes path used
#      to drop straight to standby, so a question was never followed up).
#      activity_monitor_task()
#      runs that window's timer and finally calls reset_to_standby().
#
# WEB UI / REST API (HTTP Basic auth via ADMIN_USERNAME/ADMIN_PASSWORD, 5 req/min/IP)
#   GET    /                          operator dashboard (templates/index.html)
#   GET    /health                    liveness probe, unauthenticated
#   GET    /api/devices               live sessions: session_id, mac, status, last_text
#   GET    /api/devices/config        device DB (devices.json) merged with online status
#   POST   /api/devices/config        create device; PUT/DELETE .../{mac} update/remove
#   POST   /mcp/{session_id}          MCP JSON-RPC passthrough to a satellite
#                                     ("latest" resolves to the most recent session)
#   POST   /api/tts                   speak arbitrary text on a satellite session
#   POST   /api/camera/tts            speak text through a go2rtc camera session
#   GET    /api/firmware              current firmware metadata
#   POST   /api/firmware/upload       upload a .bin (size-capped by MAX_FIRMWARE_SIZE)
#   GET|POST /ota                     ESP32 OTA handshake: WS URL, access token, update URL
#   GET    /firmware/*                static firmware files (source for the OTA URL)
#   WS     /                          satellite audio/control channel (token-authenticated)
#
# CONFIG — read from the environment / .env only. The repository is PUBLIC: never
# hardcode IPs, passwords or tokens in this file.
#   Cascade : LLM_BACKEND, ROUTER_URL, ROUTER_ACK_DELAY, HERMES_API_URL, HERMES_API_KEY
#   Legacy  : NANOBOT_WS_URL, NANOBOT_TOKEN, NANOBOT_SESSION_SALT
#   Media   : WHISPER_URL, SPEAKER_ID_URL, TTS_URL, TTS_MODEL, TTS_VOICE, TTS_API_KEY
#   Auth    : ADMIN_USERNAME, ADMIN_PASSWORD (>= 8 chars and != username — enforced
#             at import time, the module refuses to start otherwise)
#   Tuning  : VAD_SILENCE_FRAMES, ENERGY_THRESHOLD, MIN_SPEECH_RATIO, WATCHDOG_TIMEOUT,
#             STANDBY_TIMEOUT_QUESTION, STANDBY_TIMEOUT_STATEMENT, CHAT_ID_TTL,
#             LOG_TRANSCRIPTIONS, MAX_FIRMWARE_SIZE
#   Cameras : DISABLE_CAMERAS, CAMERA_STREAMS, GO2RTC_HOST, GO2RTC_PORT,
#             GO2RTC_SOURCE_URL[_<NAME>], GO2RTC_HEAL_STALLS, WAKE_WORD,
#             WAKE_WORD_MODEL, WAKE_WORD_MODEL_<NAME>
#   Unused here: THINKING_SOUND_PATH, VAD_ADAPTIVE (declared for compatibility).
#
# THREADS / ASYNC INTERPLAY
#   One asyncio event loop owns the process. Blocking work is pushed off-loop:
#   Silero ONNX inference, device-DB writes and firmware file writes go through
#   asyncio.to_thread(), the chat_id cache save through loop.run_in_executor().
#   Watchdogs and buffer-flush timers are loop.call_later() handles stored in the
#   session `state`. Every background task is registered in state["tasks"] by
#   create_tracked_task() and cancelled in the voice_ws() finally block, so a
#   dropped satellite leaves no orphan tasks behind. The cross-session registries
#   (active_sessions, session_states, mcp_futures, _active_speaker_lock) are plain
#   module-level dicts mutated only from the loop — no locks are needed because
#   there is a single writer.
#
# KNOWN GOTCHAS / TODOs
#   * The import block that follows the first config section is duplicated
#     (historical copy-paste). It is harmless — Python caches imported modules —
#     but the redundant block should be merged some day.
#   * During a gateway restart a single `ERROR ... Error in device websocket loop`
#     line (old process logged it as voice_ws:1961) is a benign artifact of the
#     satellite connection dropping mid-shutdown, not a regression.
#   * camera_client.py (2683 lines) is a sibling module that reuses VadEngine,
#     pack_ogg and the TTS backends for go2rtc cameras; it is edited elsewhere —
#     this file only imports it (see start_camera_sessions()).
# =====================================================================================

import asyncio
import json
import os
import time
import uuid
import re
import io
import hashlib
import aiohttp
import secrets
import numpy as np
import onnxruntime as ort
import opuslib
from pydub import AudioSegment
from loguru import logger
from dataclasses import dataclass
from fastapi import (
    FastAPI,
    Request,
    Form,
    WebSocket,
    HTTPException,
    UploadFile,
    File,
    Depends,
)
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import limits
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import camera_client
from camera_client import CameraSession, CameraConfig
import backends

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# Backend selection: one env var switches the whole reply path between the
# three-level cascade (L1 router / L2 CodeAgent / L3 Hermes), a direct Hermes
# call and the legacy Nanobot WebSocket. Read at import time only — flipping
# LLM_BACKEND requires a restart (the global `llm_backend` below is built once).
LLM_BACKEND = os.getenv("LLM_BACKEND", "nanobot").lower()
# Hermes (L3) HTTP API — used when LLM_BACKEND=hermes. Defaults are overridable
# from .env; never hardcode credentials here (public repo).
HERMES_API_URL = os.getenv("HERMES_API_URL", "http://192.168.22.102:8000")
HERMES_API_KEY = os.getenv("HERMES_API_KEY", "")
# Cascade mode (LLM_BACKEND=cascade): jev-router SSE endpoint
# (jev-router = L1 of the cascade; it classifies the utterance and forwards to
# the L2 smolagents worker or the L3 Hermes expert.)
ROUTER_URL = os.getenv("ROUTER_URL", "http://localhost:8091")
# Seconds to wait before speaking the filler ack «Секунду, занимаюсь…» on a
# slow L1/L2 turn — cancelled as soon as the first real sentence arrives, so
# fast paths never hear it (see backends.CascadeBackend).
ROUTER_ACK_DELAY = float(os.getenv("ROUTER_ACK_DELAY", "3"))
# ==========================================

# NOTE: the import block below is a historical duplicate of the one above.
# Modules are cached by Python, so re-importing is a no-op; kept as-is to avoid
# touching executable lines (TODO: merge the two blocks).
import asyncio
import json
import os
import time
import uuid
import re
import io
import hashlib
import aiohttp
import secrets
import numpy as np
import onnxruntime as ort
import opuslib
from pydub import AudioSegment
from loguru import logger
from dataclasses import dataclass
from fastapi import (
    FastAPI,
    Request,
    Form,
    WebSocket,
    HTTPException,
    UploadFile,
    File,
    Depends,
)
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import limits
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import camera_client
from camera_client import CameraSession, CameraConfig

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES (continued)
# Nanobot WebSocket (legacy default backend) — also used as the auth token for
# the satellite WebSocket in voice_ws() and as the OTA access_token.
NANOBOT_WS_URL = os.getenv("NANOBOT_WS_URL", "ws://nanobot:8765/").rstrip("/")
NANOBOT_TOKEN = os.getenv("NANOBOT_TOKEN", "")
# Salt mixed into the deterministic per-MAC chat_id (make_chat_id) so session
# ids cannot be guessed from a device MAC alone.
NANOBOT_SESSION_SALT = os.getenv("NANOBOT_SESSION_SALT", "")
# Speaker-recognition service: POST multipart (file) -> {user_id, confidence}.
SPEAKER_ID_URL = os.getenv("SPEAKER_ID_URL", "http://192.168.22.102:8001/identify")

# Global LLM backend (set at import; HermesBackend, CascadeBackend or NanobotBackend)
# Chosen once at import: every reply path (dispatch + camera sessions) reuses this
# object, so backend config changes need a container restart.
llm_backend = None
if LLM_BACKEND == "hermes":
    llm_backend = backends.HermesBackend(HERMES_API_URL, HERMES_API_KEY)
    logger.info(f"🧠 Global LLM backend: Hermes @ {HERMES_API_URL}")
elif LLM_BACKEND == "cascade":
    llm_backend = backends.CascadeBackend(ROUTER_URL, ack_delay=ROUTER_ACK_DELAY)
    logger.info(f"🔀 Global LLM backend: Cascade @ {ROUTER_URL}")
else:
    # Unknown values silently fall back to Nanobot — the default keeps old
    # deployments working when LLM_BACKEND is unset or mistyped.
    llm_backend = backends.NanobotBackend(NANOBOT_WS_URL, NANOBOT_TOKEN, NANOBOT_SESSION_SALT)
    logger.info(f"🤖 Global LLM backend: Nanobot @ {NANOBOT_WS_URL}")

# OpenAI-compatible STT endpoint (speaches/whisper): multipart POST of the
# packed Ogg file -> {"text": ...}. language/model are set in fetch_transcription().
WHISPER_URL = os.getenv(
    "WHISPER_URL", "http://192.168.22.111:8000/v1/audio/transcriptions"
)

# OpenAI-compatible TTS endpoint (edge_tts sidecar): JSON in -> MP3 out.
TTS_URL = os.getenv("TTS_URL", "http://edge_tts:5050/v1/audio/speech")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "ru-RU-SvetlanaNeural")
TTS_API_KEY = os.getenv("TTS_API_KEY", "")

# Web UI / REST API credentials. Both must be set or verify_auth() rejects every
# request with 401 "Authentication disabled".
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

# Fail fast at import: a weak or identical password would otherwise only be
# discovered on the first login attempt (and the API would happily accept it).
if ADMIN_PASSWORD:
    if len(ADMIN_PASSWORD) < 8:
        raise ValueError(
            "ADMIN_PASSWORD must be at least 8 characters long for security reasons."
        )
    if ADMIN_PASSWORD == ADMIN_USERNAME:
        raise ValueError("ADMIN_PASSWORD cannot be the same as ADMIN_USERNAME.")

# Opt-in transcript logging: by default only redacted lengths are logged so
# spoken content stays out of the (shared) container logs.
LOG_TRANSCRIPTIONS = os.getenv("LOG_TRANSCRIPTIONS", "false").lower() == "true"

DB_FILE = "/app/config/devices.json"
# Server-side VAD: this many consecutive quiet frames cut the utterance.
VAD_SILENCE_FRAMES = int(os.getenv("VAD_SILENCE_FRAMES", 8))
MAX_FIRMWARE_SIZE = int(os.getenv("MAX_FIRMWARE_SIZE", 10 * 1024 * 1024))
# Upstream AI silence limit: after this long without a reply, apologize + re-listen.
WATCHDOG_TIMEOUT = int(os.getenv("WATCHDOG_TIMEOUT", 90))
# Adaptive standby: keep the mic open longer after we asked the user a question
# (they need time to answer) than after a plain statement.
STANDBY_TIMEOUT_QUESTION = int(os.getenv("STANDBY_TIMEOUT_QUESTION", 30))
STANDBY_TIMEOUT_STATEMENT = int(os.getenv("STANDBY_TIMEOUT_STATEMENT", 10))
CHAT_ID_TTL = int(
    os.getenv("CHAT_ID_TTL", 604800)
)  # 7-day sliding window — Nanobot context lives a week
THINKING_SOUND_PATH = os.getenv("THINKING_SOUND_PATH", "")  # declared for compatibility; unused in this file

# VAD gates. ENERGY_THRESHOLD = mean RMS below which Whisper is skipped entirely;
# MIN_SPEECH_RATIO = share of frames Silero must call speech; VAD_ADAPTIVE is
# read for compatibility but not referenced further down.
ENERGY_THRESHOLD = float(os.getenv("ENERGY_THRESHOLD", "0.002"))
MIN_SPEECH_RATIO = float(os.getenv("MIN_SPEECH_RATIO", "0.12"))
VAD_ADAPTIVE = os.getenv("VAD_ADAPTIVE", "true").lower() == "true"

# Shared helpers extracted to audio_utils.py (also reused by camera_client.py):
# pack_ogg() wraps the Opus frames into an Ogg container for the Whisper API,
# is_valid_text() rejects hallucinations and mic echoes of our own TTS.
from audio_utils import pack_ogg, is_valid_text

# Fillers the user may say while deciding what to ask ("wait a second").
# _handle_successful_transcription() detects them in the transcript and only
# extends the listening window, so an utterance starting with a filler is not
# cut off mid-thought. NB: "momento" appears twice in the source (harmless).
HOLD_PHRASES = {
    "подожди",
    "мomento",
    "секундочку",
    "подожди-ка",
    "один момент",
    "мomento",
}

# Question detection driving the adaptive standby window — one implementation
# (_reply_has_question()) behind BOTH turn finalisers: the legacy
# NanobotResponseHandler._finalize_response() and the shared
# _finalize_turn_followup() that the cascade/hermes path calls.
# HAS_QUESTION_RE: does the reply end with "?" (ASCII or fullwidth "？").
HAS_QUESTION_RE = re.compile(r"[?？]\s*$")
# SENTENCE_END_RE: sentence boundary used by NanobotResponseHandler to cut the
# streaming LLM output into speakable chunks ('.', '!', '…'/ellipsis or newline).
SENTENCE_END_RE = re.compile(r"[.!?…](?:\s|$)|[\n]")
# HAS_QUESTION_WORDS_RE: Russian interrogatives and common imperatives — a reply
# phrased without a "?" still counts as a question so the mic stays open for the
# follow-up (applied case-insensitively to the whole reply text).
HAS_QUESTION_WORDS_RE = re.compile(
    r"\b(что|как|где|когда|почему|зачем|сколько|кто|какой|какая|какое|какие|чей|чья|чьё|чьи|куда|откуда|уточни|расскажи|напомни|объясни|повтори|скажи|покажи|подожди|помоги|ответь|напиши|сделай|включи|выключи|открой|закрой|дай|можешь|не знаю|не понимаю)\b",
    re.IGNORECASE,
)


def _reply_has_question(reply_text: str) -> bool:
    """Does the spoken reply ask something? Drives the follow-up window.

    Single source of truth for question detection, reached from both turn
    finalisers: the legacy NanobotResponseHandler._finalize_response() and the
    cascade/hermes path through the shared _finalize_turn_followup(). Keeping
    the verdict in one place is what makes the two backends behave identically
    (they used to diverge — the cascade path never looked at the reply at all
    and dropped straight to standby, so a question was never followed up).

    Args:
        reply_text: the FULL reply as spoken, already concatenated from the
            per-sentence accumulator; case is normalised here so callers may
            pass raw text.

    Returns:
        True when the reply ends with '?' (ASCII or fullwidth), contains the
        «повторите пожалуйста» apology, or matches any Russian interrogative /
        imperative of HAS_QUESTION_WORDS_RE — i.e. the mic should stay open
        for the answer instead of going to standby.
    """
    clean = reply_text.strip().lower()
    return (
        HAS_QUESTION_RE.search(clean) is not None
        or "повторите пожалуйста" in clean
        or HAS_QUESTION_WORDS_RE.search(clean) is not None
    )

SPEAKER_NAME_FILE = "/app/config/speaker_names.json"


def load_speaker_names() -> dict:
    """Load the uid -> display-name map used when labelling speaker IDs.

    Returns:
        dict: {speaker_uid: human-readable name}. Returns {} if the file is
        missing or malformed — a missing names file must never prevent the
        gateway from starting, so every error is swallowed into a warning.
    """
    try:
        with open(SPEAKER_NAME_FILE) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading speaker names: {e}")
        return {}


# Snapshot taken once at import; speaker names change rarely, so the file is
# not re-read per utterance (edit requires a restart).
SPEAKER_NAME_MAP = load_speaker_names()

# chat_id -> {chat_id, ts} cache: keeps each satellite's Nanobot conversation
# context across reconnects and container rebuilds (persists to JSON on disk).
CHAT_ID_CACHE = {}
CHAT_ID_CACHE_FILE = "/app/config/chat_id_cache.json"


def load_chat_id_cache():
    """Loads chat_id cache from disk (survives container rebuild).

    Side effects: replaces the module-level CHAT_ID_CACHE. Entries older than
    CHAT_ID_TTL (7 days) are dropped on load; any read/parse error silently
    resets the cache to empty rather than raising.
    """
    global CHAT_ID_CACHE
    if os.path.exists(CHAT_ID_CACHE_FILE):
        try:
            with open(CHAT_ID_CACHE_FILE, "r") as f:
                data = json.load(f)
            now = time.time()
            # Filter expired entries on load
            CHAT_ID_CACHE = {
                mac: entry
                for mac, entry in data.items()
                if now - entry.get("ts", 0) < CHAT_ID_TTL
            }
        except Exception:
            CHAT_ID_CACHE = {}


def save_chat_id_cache(cache_data: dict = None):
    """Persist the chat_id cache to disk.

    Args:
        cache_data: snapshot to write; when None the live CHAT_ID_CACHE is
            shallow-copied first — the copy avoids "dictionary changed size
            during iteration" if another coroutine mutates the cache while a
            background thread serialises it.

    Failure mode: IO errors are logged, never raised (the cache is a
    performance/continuity aid, losing a write only costs a re-handshake).
    """
    if cache_data is None:
        # Shallow copy to avoid "dictionary changed size during iteration" in bg thread
        cache_data = dict(CHAT_ID_CACHE)
    try:
        with open(CHAT_ID_CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save chat_id cache: {e}")


def get_cached_chat_id(mac: str) -> str | None:
    """Return the still-fresh chat_id for a MAC, or None if absent/expired.

    Params:
        mac: device MAC (lower-cased before lookup).
    Returns:
        str | None: the deterministic chat_id or None outside CHAT_ID_TTL.
    """
    # Adding a small comment to ensure the tests verify the function in the file
    entry = CHAT_ID_CACHE.get(mac.lower())
    if entry and time.time() - entry["ts"] < CHAT_ID_TTL:
        return entry["chat_id"]
    return None


# Timestamp of the last disk write — set_cached_chat_id() throttles persistence
# to at most one write per 5 s (wake words would otherwise hammer the fs).
_chat_id_last_save = 0.0


def set_cached_chat_id(mac: str, chat_id: str):
    """Cache chat_id for a MAC and (throttled) persist it to disk.

    Side effects: mutates CHAT_ID_CACHE; schedules save_chat_id_cache() on the
    running loop's executor at most every 5 s. Falls back to a synchronous
    write when no event loop is running (called from tests / import time).
    """
    global _chat_id_last_save
    CHAT_ID_CACHE[mac.lower()] = {"chat_id": chat_id, "ts": time.time()}
    now = time.time()
    # Rate-limit disk writes: one per 5 s is plenty for a sliding TTL window.
    if now - _chat_id_last_save > 5.0:
        _chat_id_last_save = now
        cache_copy = dict(CHAT_ID_CACHE)
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, save_chat_id_cache, cache_copy)
        except RuntimeError:
            # No running loop (plain sync call): write inline instead.
            save_chat_id_cache(cache_copy)


def make_chat_id(mac: str) -> str:
    """Deterministic chat_id from MAC address in UUID format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx).
    Nanobot reuses the session across reconnects.

    The MAC is salted with NANOBOT_SESSION_SALT and hashed (sha256), so the
    conversation id cannot be forged from a physically visible MAC address.
    """
    h = hashlib.sha256(f"{mac.lower()}{NANOBOT_SESSION_SALT}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


load_chat_id_cache()  # Load on startup (drops entries older than CHAT_ID_TTL)

# slowapi rate limiter keyed by client IP; wired into FastAPI app.state and used
# directly by verify_auth() (which is a dependency, not a route, so it needs the
# manual limiter._limiter.hit() call below).
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# HTTP Basic scheme with auto_error=False: missing credentials produce None
# instead of an automatic 401, letting verify_auth() craft its own responses
# ("auth disabled" vs "auth required").
security = HTTPBasic(auto_error=False)


def verify_auth(
    request: Request, credentials: HTTPBasicCredentials | None = Depends(security)
):
    """FastAPI dependency: HTTP Basic auth for every UI/REST endpoint.

    Args:
        request: used for the per-IP rate limit key.
        credentials: parsed Basic header, or None when absent.
    Returns:
        str: the authenticated username (callers compare it with ADMIN_USERNAME
        for per-device ownership checks).
    Raises:
        HTTPException: 429 when the IP exceeds 5 attempts/minute, 401 when
        credentials are unset/missing/wrong (with WWW-Authenticate so the
        browser prompts).

    Non-obvious behaviour: auth *disabled* (no ADMIN_* configured) still
    raises 401 — the API is never silently left open; the message only tells
    the operator that credentials were not configured.
    """
    limit = limits.parse("5/minute")
    if not limiter._limiter.hit(limit, get_remote_address(request), "verify_auth"):
        raise HTTPException(status_code=429, detail="Too many requests")

    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Authentication disabled (credentials not configured)",
            headers={"WWW-Authenticate": "Basic"},
        )
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    # Constant-time comparison: prevents timing attacks that could reveal
    # how many leading characters of the stored password match.
    is_user_ok = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    is_pass_ok = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (is_user_ok and is_pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# Firmware staging area for OTA updates; /firmware is mounted as a static route
# because the /ota handshake hands the satellite a direct download URL.
FIRMWARE_DIR = "/app/config/firmware"
os.makedirs(FIRMWARE_DIR, exist_ok=True)  # must exist before StaticFiles mounts
FIRMWARE_META = "/app/config/firmware.json"
app.mount("/firmware", StaticFiles(directory=FIRMWARE_DIR), name="firmware")

# Single-page operator dashboard (device list, status badges, MCP console, OTA).
templates = Jinja2Templates(directory="templates")


# ==========================================
# VAD ENGINE (Voice Activity Detection)
# ==========================================
class VadEngine:
    """Server-side voice activity detector: Silero VAD (ONNX) + energy fallback.

    Wraps silero_vad.onnx (16 kHz, 512-sample chunks) and scores PCM16 buffers.
    Two independent signals are combined with OR:
      * ONNX speech probability > onnx_threshold (input pre-amplified by
        onnx_gain to compensate quiet microphones);
      * RMS energy above max(ENERGY_THRESHOLD, adaptive noise floor * 1.2),
        where the floor is an exponential moving average of recent levels.

    Stateful: the recurrent `state`/`context` tensors and the sample `buffer`
    carry history between calls, so reset() MUST be called at every utterance
    boundary (wake, standby, watchdog) — otherwise one utterance's residual
    energy biases the next one's decision.

    Failure modes: exceptions are caught and reported as (False, 0.0) — a VAD
    error should read as "silence", never as a phantom wake.
    """

    def __init__(
        self,
        energy_fallback: bool = True,
        energy_threshold: float = 0.01,
        onnx_threshold: float = 0.02,
        vad_adaptive: bool | None = None,
        rms_noise_floor: float = 0.10,
        rms_alpha: float = 0.05,
        onnx_gain: float = 20.0,
    ):
        """Create the ONNX session and default buffers.

        Args:
            energy_fallback: enable the RMS-based second opinion.
            energy_threshold: static RMS gate (raised to the adaptive floor
                when the floor is higher).
            onnx_threshold: Silero speech-probability cut-off.
            vad_adaptive: accepted for API compatibility, unused (see module
                header "Unused here").
            rms_noise_floor: initial EMA of the ambient noise level.
            rms_alpha: EMA weight for the noise floor (0 = frozen floor —
                used by camera sessions on already-normalised audio).
            onnx_gain: linear pre-gain applied before Silero, because quiet
                camera/satellite mics otherwise score below the threshold.
        """
        logger.info("Loading Silero VAD (ONNX) model...")
        opts = ort.SessionOptions()
        # Single-threaded ONNX: inference runs off-loop via asyncio.to_thread;
        # capping threads keeps latency predictable and avoids starving the
        # event loop on the shared CPU.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            "silero_vad.onnx", sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.buffer = np.array([], dtype=np.float32)
        self.energy_fallback = energy_fallback
        self.energy_threshold = energy_threshold
        self._last_debug_log = 0.0
        self.onnx_gain = onnx_gain

        self.onnx_threshold = onnx_threshold
        self.rms_noise_floor = rms_noise_floor
        self.rms_alpha = rms_alpha
        self._context_size = 64  # official Silero VAD context prefix

        self.reset()

    def reset(self):
        """Zero the recurrent state, context window and sample buffer.

        Call at every utterance boundary. No I/O; safe from any coroutine.
        """
        # Silero's hidden state (2x1x128) and 64-sample context prefix — the
        # shapes are fixed by the ONNX graph, do not change them.
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self.buffer = np.array([], dtype=np.float32)

    def _run_onnx(
        self, chunk: np.ndarray, state: np.ndarray, context: np.ndarray
    ) -> tuple:
        """Run one 512-sample chunk through the model (blocking — call from a
        thread). Returns (probabilities, next_state) as the graph outputs them.

        The context is prepended because Silero expects a (1, 64+512) window,
        not a bare chunk.
        """
        full_input = np.concatenate([context, chunk[np.newaxis, :]], axis=1)  # (1, 576)
        return self.session.run(
            None,
            {
                "input": full_input,
                "state": state,
                "sr": np.array([16000], dtype=np.int64),
            },
        )

    async def is_speech(
        self, pcm: bytes, precomputed_rms: float | None = None
    ) -> tuple[bool, float]:
        """Score one PCM16 chunk.

        Args:
            pcm: raw int16 little-endian samples (one 60 ms frame = 960 samples).
            precomputed_rms: RMS already computed by the caller, saved to avoid
                a second numpy pass over the same buffer.

        Returns:
            (speech_detected, rms): True when either the ONNX probability or
            the energy gate fires.

        Side effects: consumes/extends self.buffer, advances the recurrent
        state, and adapts the noise-floor EMA. Inference is offloaded with
        asyncio.to_thread so the event loop is not blocked.
        Failure mode: any exception -> (False, 0.0) plus an error log.
        """
        try:
            audio_float32 = (
                np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            )
            if precomputed_rms is not None:
                rms = precomputed_rms
            else:
                rms = float(np.sqrt(np.mean(np.square(audio_float32))))

            # ONNX-based detection (apply gain to compensate for quiet camera audio)
            speech_onnx = False
            onnx_max = 0.0
            onnx_input = audio_float32 * self.onnx_gain
            self.buffer = np.concatenate((self.buffer, onnx_input))
            # The model only accepts 512-sample windows: drain whole windows
            # and keep the remainder for the next call (input may be 960 samples).
            while len(self.buffer) >= 512:
                chunk = self.buffer[:512]
                self.buffer = self.buffer[512:]
                out, self._state = await asyncio.to_thread(
                    self._run_onnx, chunk, self._state, self._context
                )
                # Update context: last 64 samples of (context + chunk)
                self._context = np.concatenate(
                    [self._context, chunk[np.newaxis, :]], axis=1
                )[:, -self._context_size :]
                onnx_max = max(onnx_max, out[0][0])
                if out[0][0] > self.onnx_threshold:
                    speech_onnx = True

            # Energy-based detection with adaptive threshold
            speech_energy = False
            if self.energy_fallback:
                # Never let the gate drop below the static threshold, and
                # require ~20% above the learned ambient floor so a slowly
                # rising noise level cannot keep the gate permanently open.
                energy_thresh = max(self.energy_threshold, self.rms_noise_floor * 1.2)
                if rms > energy_thresh:
                    speech_energy = True

            speech_detected = speech_onnx or speech_energy

            # EMA of the ambient level: tracks fans/TV drifting over minutes.
            self.rms_noise_floor = (
                1 - self.rms_alpha
            ) * self.rms_noise_floor + self.rms_alpha * rms

            return speech_detected, rms
        except Exception as e:
            logger.error(f"❌ VAD Error: {e}")
            return False, 0.0

    async def is_speech_batch(
        self, pcms: list[bytes], rms_list: list[float]
    ) -> list[bool]:
        """Process a batch of PCM chunks in a single thread to avoid N+1 async delays.

        Params:
            pcms: PCM16 chunks (None-safe order preserved by the caller).
            rms_list: pre-computed RMS per chunk.
        Returns:
            list[bool]: per-chunk speech flags, same order as pcms.

        Why batch: is_speech() pays one asyncio.to_thread hop per chunk (~1 ms
        each), which added up to a noticeable cut latency on a 100-frame
        utterance; running the whole batch in one thread keeps the model state
        sequential (the ONNX session is stateful and NOT thread-safe) and pays
        a single hop. Per-chunk errors are recorded as False instead of
        aborting the batch.
        """

        def process_all():  # runs entirely in one worker thread (see docstring)
            results = []
            for pcm, precomputed_rms in zip(pcms, rms_list):
                try:
                    audio_float32 = (
                        np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                    rms = precomputed_rms

                    # ONNX-based detection (apply gain to compensate for quiet camera audio)
                    speech_onnx = False
                    onnx_max = 0.0
                    onnx_input = audio_float32 * self.onnx_gain
                    self.buffer = np.concatenate((self.buffer, onnx_input))
                    while len(self.buffer) >= 512:
                        chunk = self.buffer[:512]
                        self.buffer = self.buffer[512:]
                        out, self._state = self._run_onnx(
                            chunk, self._state, self._context
                        )
                        # Update context: last 64 samples of (context + chunk)
                        self._context = np.concatenate(
                            [self._context, chunk[np.newaxis, :]], axis=1
                        )[:, -self._context_size :]
                        onnx_max = max(onnx_max, out[0][0])
                        if out[0][0] > self.onnx_threshold:
                            speech_onnx = True

                    # Energy-based detection with adaptive threshold
                    speech_energy = False
                    if self.energy_fallback:
                        energy_thresh = max(
                            self.energy_threshold, self.rms_noise_floor * 1.2
                        )
                        if rms > energy_thresh:
                            speech_energy = True

                    speech_detected = speech_onnx or speech_energy

                    self.rms_noise_floor = (
                        1 - self.rms_alpha
                    ) * self.rms_noise_floor + self.rms_alpha * rms

                    results.append(speech_detected)
                except Exception as e:
                    logger.error(f"❌ VAD Error in batch: {e}")
                    results.append(False)
            return results

        return await asyncio.to_thread(process_all)


# ==========================================
# UTILS & AUDIO PACKING
# ==========================================
# Module-level cache of devices.json: load_db() is called on every REST request
# (device config, OTA, MCP auth) and save_db() invalidates it. The cache makes
# reads lock-free; all mutation goes through save_db() so there is a single
# writer path (still loop-serialised — no threading lock required).
_DB_CACHE = None


def load_db() -> dict:
    """Return the device DB (devices.json), loading it lazily and caching it.

    Returns:
        dict: {MAC: {friendly_name, ws_url, allowed, owner}}; {} when the file
        is missing or unparseable (a warning is logged, callers treat that as
        an empty DB rather than an error).
    """
    global _DB_CACHE
    if _DB_CACHE is not None:
        return _DB_CACHE
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                _DB_CACHE = json.load(f)
                return _DB_CACHE
        except Exception as e:
            logger.warning(f"⚠️ [DB] Failed to load {DB_FILE}: {e}")
    _DB_CACHE = {}
    return _DB_CACHE


async def save_db(db: dict):
    """Update the in-memory device DB and persist it off-loop.

    Args:
        db: the complete dict to store (callers pass load_db() after mutation).
    Side effects: replaces _DB_CACHE immediately (readable before the write
    finishes), then serialises to disk in a worker thread so a slow fs never
    stalls the event loop.
    """
    global _DB_CACHE
    _DB_CACHE = db
    await asyncio.to_thread(_save_db_sync, db)


def _save_db_sync(db: dict):
    """Blocking JSON write (runs in a worker thread via asyncio.to_thread).

    Creates the config directory on first use; IO errors propagate to save_db's
    caller as an unhandled exception, which REST endpoints surface as a 500 —
    a failed device-DB write should be visible, not silent.
    """
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=4)


# ==========================================
# HARDWARE CONTROL (MCP Tools)
# ==========================================
# Three loop-owned registries (single writer = the event loop, no locks):
#   active_sessions : session_id -> satellite WebSocket (REST/MCP routing)
#   session_states  : session_id -> the per-device `state` dict (status, VAD,
#                     tasks, timings — the state machine of voice_ws())
#   mcp_futures     : request_id -> Future, matching MCP replies to callers
active_sessions = {}
session_states = {}
mcp_futures = {}

# Speaker ID lock: prevents duplicate processing when two devices hear the same speaker
# (a kitchen and a corridor satellite both hear one person; only the first
# may run the turn — the other one drops it as a duplicate.)
_active_speaker_lock: dict | None = None  # {"uid": str, "sid": str, "expires": float}


def _clean_stale_futures():
    """Drop finished or expired entries from mcp_futures.

    Entries are keyed by epoch-ms ids, so anything older than 30 s is assumed
    abandoned (its waiter already timed out) and is cancelled — otherwise a
    chatty device would leak futures forever. Called opportunistically before
    each new MCP exchange instead of on a timer (cheap, no extra task).
    """
    now = time.time()
    stale = [
        rid for rid, f in mcp_futures.items() if f.done() or (rid < (now - 30) * 1000)
    ]
    for rid in stale:
        f = mcp_futures.pop(rid, None)
        if f and not f.done():
            f.cancel()


async def send_mcp_cmd(
    device_ws: WebSocket,
    session_id: str,
    tool_name: str,
    arguments: dict,
    req_id: int = None,
):
    """Send a JSON-RPC tools/call to the satellite (fire-and-forget).

    Args:
        device_ws: the device's WebSocket.
        session_id: satellite session id, echoed so the firmware can route it.
        tool_name: MCP tool, e.g. "self.screen.set_brightness".
        arguments: JSON-serialisable tool arguments.
        req_id: correlation id; defaults to epoch ms. Pass an explicit id when
            a reply is awaited (see handle_device_tool_call / execute_mcp).

    Failure mode: send errors are logged as warnings, never raised — device
    disconnection is routine and the caller (screen dimming, volume set) must
    not crash the session over it.
    """
    if req_id is None:
        req_id = int(time.time() * 1000)

    # JSON-RPC 2.0 envelope wrapped in the gateway's own {"session_id","type"}
    # frame — this is the wire format the ESP32 firmware understands.
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
        "id": req_id,
    }
    try:
        await device_ws.send_json(
            {"session_id": session_id, "type": "mcp", "payload": payload}
        )
    except Exception as e:
        logger.warning(f"⚠️ [MCP] send_mcp_cmd failed: {e}")


async def request_mcp_tools(device_ws: WebSocket, session_id: str):
    """Ask the satellite for its MCP tool catalogue once, right after `hello`.

    Uses the sentinel request id 999: handle_ws_text_message() recognises it
    and stores the returned tool list in state["available_tools"] instead of
    matching it against a pending future (nobody waits for this reply).

    Failure mode: logged, never raised — a device that speaks an older
    protocol may simply ignore tools/list.
    """
    logger.info("🛠 [MCP] Requesting available tools from ESP32...")
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {"cursor": "", "withUserTools": True},
        "id": 999,
    }
    try:
        await device_ws.send_json(
            {"session_id": session_id, "type": "mcp", "payload": payload}
        )
    except Exception as e:
        logger.error(f"❌ [MCP] Failed to request tools: {e}")


async def handle_device_tool_call(
    nano_ws: aiohttp.ClientWebSocketResponse, device_ws: WebSocket, state: dict, d: dict
):
    """Bridge a tool call from the LLM brain to the satellite and back.

    Flow: LLM emits `device_tool_call` -> we forward it as an MCP tools/call
    over the device WS -> await the matching reply (correlated by req_id in
    mcp_futures) -> return the result to the brain as `device_tool_result`.

    Args:
        nano_ws: upstream WS to the brain (carries the result).
        device_ws: the satellite WS (carries the tool call).
        state: session state (provides sid).
        d: the decoded `device_tool_call` event from the brain.

    Failure modes: 10 s timeout -> a structured {"error": ...} result is sent
    so the LLM can react instead of hanging; any other exception is only
    logged (the brain then relies on its own timeout). The future is always
    removed in `finally`.
    """
    tc = d.get("tool_call", {})
    tool_name = tc.get("name", "")
    arguments = tc.get("arguments", {})
    tool_call_id = tc.get("id", "")
    logger.info(f"🔧 [DeviceTool] AI calling ESP32 tool: {tool_name}({arguments})")
    req_id = int(time.time() * 1000)
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    # Register the future BEFORE sending: a reply could otherwise arrive while
    # we are still awaiting and find no waiter. Bound the map at 200 entries
    # (cancel the oldest) so lost replies cannot grow it without limit.
    _clean_stale_futures()
    if len(mcp_futures) > 200:
        oldest = min(mcp_futures.keys())
        old_fut = mcp_futures.pop(oldest, None)
        if old_fut and not old_fut.done():
            old_fut.cancel()
    mcp_futures[req_id] = future
    try:
        await send_mcp_cmd(device_ws, state["sid"], tool_name, arguments, req_id)
        # 10 s cap: on-device tools (relays, sensors) answer in milliseconds;
        # a longer wait would just stall the LLM turn on a dead satellite.
        result = await asyncio.wait_for(future, timeout=10.0)
        await nano_ws.send_json(
            {
                "type": "device_tool_result",
                "chat_id": d.get("chat_id"),
                "tool_call_id": tool_call_id,
                "result": result,
            }
        )
        logger.info(f"✅ [DeviceTool] ESP32 tool {tool_name} completed")
    except asyncio.TimeoutError:
        logger.warning(f"⏱ [DeviceTool] ESP32 tool {tool_name} timed out")
        await nano_ws.send_json(
            {
                "type": "device_tool_result",
                "chat_id": d.get("chat_id"),
                "tool_call_id": tool_call_id,
                "result": {"error": "ESP32 tool call timed out"},
            }
        )
    except Exception as e:
        logger.error(f"❌ [DeviceTool] Error calling ESP32 tool {tool_name}: {e}")
    finally:
        mcp_futures.pop(req_id, None)


def create_tracked_task(coro, state, name=""):
    """Spawn an asyncio task that is registered in state["tasks"] and
    auto-unregistered when it finishes.

    Why: voice_ws()'s finally block cancels every task in that set when the
    satellite disconnects, so background work (pipeline, TTS, monitor) never
    outlives its session or writes to a dead WebSocket.
    """
    task = asyncio.create_task(coro, name=name)
    state["tasks"].add(task)
    task.add_done_callback(lambda _: state["tasks"].discard(task))
    return task


async def activity_monitor_task(device_ws: WebSocket, state: dict):
    """Monitors idle: dims screen but does NOT close the connection.
    Persistent mode — WS/context lives while ESP32 is on.

    Runs for the whole session (1 Hz tick) and implements two behaviours:

    * Adaptive standby: while LISTENING, if nothing arrives within
      STANDBY_TIMEOUT_QUESTION (after an AI question) or
      STANDBY_TIMEOUT_STATEMENT (after a statement), reset_to_standby() is
      scheduled. A wake with no audio at all is logged as a diagnostic.
    * Screen dimming: IDLE for >10 s -> brightness 25%; any active status
      re-arms `dim_sent` so the next idle period dims again.

    Timers are frozen while status is PROCESSING/SPEAKING (last_activity is
    refreshed each tick) so a long TTS answer cannot expire the turn.
    Cancellation: exits silently on CancelledError (session teardown).
    """
    await send_mcp_cmd(
        device_ws, state["sid"], "self.audio_speaker.set_volume", {"volume": 100}
    )
    await send_mcp_cmd(
        device_ws, state["sid"], "self.screen.set_brightness", {"brightness": 25}
    )
    dim_sent = True

    try:
        while state["sid"] in session_states:
            now = time.time()

            # Freeze timer while the speaker is active (thinking/speaking)
            if state["status"] in ["PROCESSING", "SPEAKING"]:
                state["last_activity"] = now

            time_idle = now - state["last_activity"]

            # Adaptive standby: 30s after question, 10s after statement
            if state["status"] == "LISTENING":
                timeout = (
                    STANDBY_TIMEOUT_QUESTION
                    if state.get("last_ai_had_question")
                    else STANDBY_TIMEOUT_STATEMENT
                )
                if time_idle > timeout:
                    if not state.get("_wake_audio_received"):
                        logger.warning(
                            f"🔇 [Diag] Listening window expired — no audio received in {int(time_idle)}s"
                        )
                    logger.info(
                        f"💤 [Timeout] {int(time_idle)}s idle (limit {timeout}s). Standby."
                    )
                    create_tracked_task(reset_to_standby(device_ws, state), state)
            # Dim screen when idle (device powered on but no interaction)
            elif state["status"] == "IDLE" and time_idle > 10:
                if not dim_sent:
                    logger.info(f"💤 [Idle] {int(time_idle)}s idle — dim screen to 25%")
                    await send_mcp_cmd(
                        device_ws,
                        state["sid"],
                        "self.screen.set_brightness",
                        {"brightness": 25},
                    )
                    dim_sent = True
            elif state["status"] not in ("IDLE", "LISTENING"):
                dim_sent = False  # Reset when activity resumes

            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error in monitor task: {e}")


async def reset_to_standby(device_ws: WebSocket, state: dict):
    """Switches the speaker to standby: dims screen, stops listening.
    Keeps WS open — MCP tools (temperature monitoring) continue working.

    Side effects: releases this session's claim on the global speaker lock,
    clears buffers/silence counters, resets the VAD state, forgets the
    "AI asked a question" flag, bumps last_activity and cancels the watchdog
    (nothing is pending anymore). Safe to call from any task of the session.
    """
    # Note: no has_speech guard here — last_activity timer already protects
    # against interrupting active speech. has_speech can be stuck True by
    # VAD false positives on background noise, permanently blocking standby.
    global _active_speaker_lock
    if _active_speaker_lock and _active_speaker_lock["sid"] == state.get("sid"):
        _active_speaker_lock = None
    logger.info("💤 Standby — dim screen, stop listening")
    state["status"] = "IDLE"
    state["frames"] = []
    state["silence"] = 0
    state["has_speech"] = False

    await send_mcp_cmd(
        device_ws, state["sid"], "self.screen.set_brightness", {"brightness": 25}
    )
    state["vad"].reset()

    state["last_ai_had_question"] = False
    state["last_activity"] = time.time()

    if state.get("watchdog"):
        state["watchdog"].cancel()
        state["watchdog"] = None


async def _finalize_turn_followup(
    device_ws: WebSocket, state: dict, spoken_text: str
) -> bool:
    """End of a turn: judge the spoken reply and open the follow-up window.

    Single implementation of the post-turn transition, shared by the legacy
    NanobotResponseHandler._finalize_response() and the cascade/hermes
    _dispatch_hermes(). The two paths used to diverge: the cascade one called
    reset_to_standby() unconditionally, so the satellite went IDLE immediately
    after the reply — including right after the gateway itself asked the
    user a question, which is why the answer never arrived (the firmware's own
    post-TTS `listen:start` was additionally rejected as a false wake).

    Args:
        device_ws: satellite WebSocket, used for the dialogue-mode screen cmd.
        state: session state, mutated in place.
        spoken_text: the FULL reply as spoken; question detection itself is
            case-insensitive and lives in _reply_has_question().

    Returns:
        True when the reply was judged a question (the caller may want it for
        logging; the verdict is also stored in state["last_ai_had_question"]).

    Side effects: status -> LISTENING in BOTH branches, frames/silence cleared,
    TTS cooldown trimmed to 0.2 s (mic ready for the answer, still deaf to the
    tail of our own playback), last_ai_had_question stored for
    activity_monitor_task()'s window choice (STANDBY_TIMEOUT_QUESTION 30 s vs
    STANDBY_TIMEOUT_STATEMENT 10 s), VAD reset. A question additionally lights
    the screen to 100 % (dialogue mode); a statement disarms the pending
    watchdog. The cross-device speaker lock is deliberately NOT touched — as in
    legacy, reset_to_standby() releases it when the window expires.
    """
    has_question = _reply_has_question(spoken_text)
    state["last_ai_had_question"] = has_question
    state.update(
        {"status": "LISTENING", "frames": [], "silence": 0, "has_speech": False}
    )
    state["tts_cooldown_until"] = time.time() + 0.2
    # A fresh listening window opened: no audio received YET. This makes the
    # monitor's "no audio received" warning truthful instead of always-on.
    state["_wake_audio_received"] = False
    if has_question:
        logger.info(f"💡 [Brightness] Dialogue mode — screen 100%")
        await send_mcp_cmd(
            device_ws,
            state["sid"],
            "self.screen.set_brightness",
            {"brightness": 100},
        )
    else:
        if state.get("watchdog"):
            state["watchdog"].cancel()
            state["watchdog"] = None
    state["last_activity"] = time.time()
    state["vad"].reset()
    return has_question


async def watchdog_timeout(device_ws: WebSocket, state: dict):
    """Fired by the WATCHDOG_TIMEOUT timer when the upstream AI goes silent.

    Apologises over TTS, then immediately re-opens the microphone so the user
    can repeat the request without a new wake word. Sets watchdog_fired = True,
    which makes _hermes_player_task() drop any late sentences (they would
    contradict the apology) and makes listen_to_nanobot_task() ignore stray
    late replies.

    Side effects: status -> SPEAKING -> LISTENING, buffers cleared, VAD reset,
    short TTS cooldown (0.3 s) so our own apology cannot re-trigger the VAD,
    screen back to 100%. TTS failure is logged but does not stop the re-arm.
    """
    logger.warning("⏱ [Watchdog] Upstream AI timed out.")
    state["watchdog_fired"] = True
    state["status"] = "SPEAKING"
    try:
        await generate_and_stream_tts(
            "Прости, я затупила. Повтори пожалуйста.", device_ws, state["sid"], state
        )
    except Exception as e:
        logger.error(f"❌ [Watchdog] TTS failed: {e}")
    # If synthesis never produced audio the apology did not close the stream
    # either, so close it here — "keeping mic open" is only true once the
    # satellite has seen `tts stop`. Edge-TTS outage case (2026-09-27 15:11):
    # every synthesis failed, no stop was sent, the device stayed in SPEAKING
    # and the re-opened window expired with "no audio received".
    await send_tts_stop(device_ws, state)

    logger.info("🎤 [Watchdog] Keeping mic open for repeat.")
    state.update(
        {"status": "LISTENING", "frames": [], "silence": 0, "has_speech": False}
    )
    state["last_activity"] = time.time()  # Reset timer
    state["vad"].reset()
    state["tts_cooldown_until"] = time.time() + 0.3  # Short cooldown after apology
    # New listening window: no audio received YET, otherwise the monitor's
    # warning would be silenced by frames from the previous window.
    state["_wake_audio_received"] = False
    await send_mcp_cmd(
        device_ws, state["sid"], "self.screen.set_brightness", {"brightness": 100}
    )


# ==========================================
# ASYNC PIPELINE (STT & SPEAKER ID)
# ==========================================
async def fetch_speaker_id(audio: bytes, sess: aiohttp.ClientSession) -> str:
    """Identify the speaker by voice via the SPEAKER_ID_URL service.

    Args:
        audio: Ogg/Opus utterance (pack_ogg output).
        sess: shared aiohttp session (30 s total timeout from voice_ws).
    Returns:
        str: recognised uid, or "unknown" on low confidence (<= 0.1), HTTP
        errors or service downtime — STT must proceed without identity rather
        than fail (speaker names are optional decoration for the transcript).
    """
    try:
        form = aiohttp.FormData()
        form.add_field("file", audio, filename="audio.ogg", content_type="audio/ogg")
        async with sess.post(SPEAKER_ID_URL, data=form, timeout=10) as r:
            if r.status == 200:
                json_resp = await r.json()
                uid, conf = json_resp.get("user_id", "unknown"), json_resp.get(
                    "confidence", 0.0
                )
                # Confidence floor: below 0.1 the service is guessing, so the
                # uid is discarded instead of attributing the utterance.
                if uid != "unknown" and conf > 0.1:
                    logger.info(f"✅ [SpeakerID] Recognized: {uid} ({conf:.2f})")
                    return uid
                else:
                    logger.debug(f"👤 [SpeakerID] Rejected: {uid} ({conf:.2f})")
    except Exception as e:
        logger.warning(f"⚠️ [SpeakerID] Request failed: {e}")
    return "unknown"


async def fetch_transcription(audio: bytes, sess: aiohttp.ClientSession) -> str:
    """Speech-to-text: POST the Ogg utterance to the Whisper-compatible API.

    Args:
        audio: Ogg/Opus bytes from pack_ogg().
        sess: shared aiohttp session.
    Returns:
        str: transcript text, or "" on HTTP errors / exceptions (callers run
        the result through is_valid_text(), so an empty string simply becomes
        a rejected turn). Non-200 bodies are logged for diagnosis.
    """
    try:
        form = aiohttp.FormData()
        form.add_field("file", audio, filename="a.ogg")

        # REQUIRED field for speaches (OpenAI API), otherwise 422 error
        # Exact downloaded model name
        form.add_field("model", "koekaverna/faster-whisper-podlodka-turbo")
        form.add_field("language", "ru")
        form.add_field("temperature", "0.0")

        async with sess.post(WHISPER_URL, data=form, timeout=30) as r:
            if r.status == 200:
                return (await r.json()).get("text", "").strip()
            else:
                error_body = await r.text()
                logger.error(f"❌ [Whisper] HTTP {r.status}: {error_body}")
    except Exception as e:
        logger.error(f"❌ [Whisper] Error: {e}")
    return ""


def calculate_rms(pcm_data: bytes) -> float:
    """Calculate RMS energy from PCM16 audio data.

    Returns:
        float: RMS in 0..1 (int16 scaled to float); 0.0 for empty or
        malformed buffers — a silent frame must never raise into the pipeline.
    """
    try:
        if not pcm_data:
            return 0.0
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(audio_float32))))
        return rms
    except Exception:
        return 0.0


async def decode_opus_frames(
    frames: list, decoder: opuslib.Decoder, vad: VadEngine
) -> tuple[bytes, list[float], list[bool]]:
    """Decode Opus frames to PCM and return (combined_pcm, rms_list, vad_results).

    Args:
        frames: 60 ms Opus packets as received from the satellite.
        decoder: stateful opuslib decoder (16 kHz mono) — reused across calls
            so packet-loss concealment stays continuous within a session.
        vad: engine that scores every decoded chunk in one batch.
    Returns:
        (combined PCM bytes, per-frame RMS, per-frame speech flag).

    Non-obvious: a frame that fails to decode contributes 0.0/False but keeps
    its position in the result lists (index alignment with `frames` is what
    the caller relies on); the VAD batch only sees the frames that decoded.
    """
    all_pcm = bytearray()
    rms_list = []
    pcm_list = []

    for frame in frames:
        try:
            # 960 samples = 60 ms at 16 kHz — the frame duration negotiated
            # in the `hello` audio_params.
            pcm = decoder.decode(frame, 960)
            all_pcm.extend(pcm)

            rms = calculate_rms(pcm)
            rms_list.append(rms)
            pcm_list.append(pcm)
        except Exception:
            rms_list.append(0.0)
            pcm_list.append(None)

    valid_pcms = []
    valid_rms = []
    for pcm, rms in zip(pcm_list, rms_list):
        if pcm is not None:
            valid_pcms.append(pcm)
            valid_rms.append(rms)

    if valid_pcms:
        batch_results = await vad.is_speech_batch(valid_pcms, valid_rms)
    else:
        batch_results = []

    vad_results = []
    batch_idx = 0
    for pcm in pcm_list:
        if pcm is not None:
            vad_results.append(batch_results[batch_idx])
            batch_idx += 1
        else:
            vad_results.append(False)

    return bytes(all_pcm), rms_list, vad_results


async def _process_audio_metrics_and_gates(
    frames: list, state: dict, decoder: opuslib.Decoder = None
) -> tuple[bool, float, float, opuslib.Decoder]:
    """Decode the utterance and apply the cheap pre-STT gates.

    Returns:
        (passed, avg_rms, speech_ratio, decoder): `passed` is False when the
        audio is too quiet (ENERGY_THRESHOLD) or too little of it is speech
        (MIN_SPEECH_RATIO) — in that case Whisper is skipped entirely, saving
        a network round trip on every background-noise burst.

    The decoder is created lazily and always returned, so the caller can keep
    reusing one decoder per session (opus state must survive between cuts).
    """
    dec = decoder or opuslib.Decoder(16000, 1)
    _, rms_list, vad_results = await decode_opus_frames(frames, dec, state["vad"])

    avg_rms = sum(rms_list) / len(rms_list) if rms_list else 0.0
    speech_frames = sum(1 for v in vad_results if v)
    speech_ratio = speech_frames / len(vad_results) if vad_results else 0.0

    if avg_rms < ENERGY_THRESHOLD:
        logger.info(
            f"🔇 [Energy Gate] RMS={avg_rms:.4f} < {ENERGY_THRESHOLD}, skipping Whisper"
        )
        return False, avg_rms, speech_ratio, dec

    if speech_ratio < MIN_SPEECH_RATIO:
        logger.info(
            f"🔇 [Speech Ratio] {speech_ratio:.1%} < {MIN_SPEECH_RATIO}, skipping Whisper"
        )
        return False, avg_rms, speech_ratio, dec

    return True, avg_rms, speech_ratio, dec


def _check_speaker_lock(uid: str, sid: str) -> bool:
    """Claim the global speaker turn for one session (cross-device de-dup).

    Two satellites often hear the same person at once; without a lock both
    would run the pipeline and the LLM would answer twice.

    Args:
        uid: recognised speaker id ("unknown" counts as its own speaker).
        sid: session that wants to speak.
    Returns:
        bool: True when this session may proceed (it also acquires/refreshes
        the lock, 30 s TTL); False when the same uid is already active on a
        *different* session — the duplicate is dropped.
    Side effects: module-global _active_speaker_lock; expired locks are
    cleared on the way in, so a crashed session cannot block the system.
    """
    global _active_speaker_lock
    now = time.time()
    if _active_speaker_lock and _active_speaker_lock["expires"] < now:
        _active_speaker_lock = None
    if (
        _active_speaker_lock
        and _active_speaker_lock["uid"] == uid
        and _active_speaker_lock["sid"] != sid
    ):
        logger.info(
            f"🔇 [Multi] Speaker '{uid}' already active on {_active_speaker_lock['sid']}, skipping duplicate"
        )
        return False
    _active_speaker_lock = {"uid": uid, "sid": sid, "expires": now + 30}
    return True


async def _handle_successful_transcription(
    txt: str, uid: str, state: dict, device_ws: WebSocket, _t0: float, _t_stt: float
):
    """Run the accepted transcript through the LLM dispatch paths.

    Args:
        txt: validated transcript.
        uid: speaker id from fetch_speaker_id().
        state: session state (mutated: status, last_text, watchdog, timing).
        device_ws: satellite WebSocket.
        _t0: timestamp of the pipeline start (VAD trigger), for latency logs.
        _t_stt: timestamp right after STT returned.

    Behaviour by backend:
      * LLM_BACKEND in (hermes, cascade) -> _dispatch_hermes() streams the
        reply through the queue player.
      * legacy Nanobot -> text pushed over nano_ws, watchdog armed; if the
        brain is connected but has no chat_id yet, the user is told the
        system is not ready; if it is not connected at all, the transcript
        itself is spoken back as a fallback so the turn never ends in silence.
    Early exit: a failed `stt` send to the device (disconnected) aborts
    before any LLM work is done.
    """
    state["_rejected_count"] = 0
    _t_transcribed = time.time()
    state["_stt_time"] = _t_transcribed
    # Transcript logging is opt-in; by default only the length is logged so
    # what the user said stays out of the shared container logs.
    if LOG_TRANSCRIPTIONS:
        logger.info(f"🗣 [User: {uid}] Transcribed: '{txt}'")
    else:
        logger.info(f"🗣 [User: {uid}] Transcribed: [REDACTED] ({len(txt)} chars)")
    state["last_activity"] = _t_transcribed
    state["last_text"] = txt

    if state.get("nanobot_chat_id"):
        set_cached_chat_id(state["mac"].lower(), state["nanobot_chat_id"])

    # Fillers ("подожди", "один момент"...) buy the user time instead of
    # ending the turn — refresh the idle timer so standby does not cut them.
    if any(p in txt.lower() for p in HOLD_PHRASES):
        logger.info(f"🛑 [Hold] Detected hold phrase, extending listening")
        state["last_activity"] = time.time()

    # Show the transcript on the satellite display (subtitle). If the device
    # is gone there is nothing to answer — abort before touching the LLM.
    try:
        await device_ws.send_json(
            {"type": "stt", "text": txt, "session_id": state["sid"]}
        )
    except Exception:
        return

    # --- LLM dispatch: Hermes/Cascade backend or legacy Nanobot ---
    if LLM_BACKEND in ("hermes", "cascade") and llm_backend is not None:
        await _dispatch_hermes(txt, uid, state, device_ws, _t0, _t_stt)
    else:
        nano_ws = state.get("nano_ws")
        if nano_ws and not nano_ws.closed:
            chat_id = state.get("nanobot_chat_id")
            if not chat_id:
                logger.warning(
                    "⚠️ [Pipeline] Nanobot connected but no chat_id yet, waiting..."
                )
                state["status"] = "SPEAKING"
                await generate_and_stream_tts(
                    "Система не готова, повторите.", device_ws, state["sid"], state
                )
                await reset_to_standby(device_ws, state)
                return

            speaker_name = SPEAKER_NAME_MAP.get(uid, uid)
            payload = {
                "type": "message",
                "chat_id": chat_id,
                "content": txt,
                "user_id": uid,
                "user_name": speaker_name,
                "voice_reply": True,
            }
            if state.get("handler"):
                state["handler"].reset_timing()
            await nano_ws.send_json(payload)
            _t_nano = time.time()
            logger.info(
                f"⏱ [Timing] Sent to Nanobot: {_t_nano-_t_stt:.2f}s after STT | "
                f"total={_t_nano-_t0:.2f}s since VAD trigger"
            )

            try:
                await device_ws.send_json(
                    {"type": "tts", "state": "start", "session_id": state["sid"]}
                )
            except Exception:
                return
            state["status"] = "SPEAKING"
            state["tts_started"] = True
            state["watchdog_fired"] = False

            if state.get("watchdog"):
                state["watchdog"].cancel()
            # Arm the watchdog (call_later handle lives in state so any later
            # chunk can cancel it): if the brain stays silent for
            # WATCHDOG_TIMEOUT seconds, watchdog_timeout() apologises and
            # re-opens the mic instead of leaving dead air.
            state["watchdog"] = asyncio.get_event_loop().call_later(
                WATCHDOG_TIMEOUT,
                lambda: create_tracked_task(watchdog_timeout(device_ws, state), state),
            )
        else:
            logger.warning(
                f"⚠️ [Pipeline] Nanobot not connected, falling back to direct TTS"
            )
            state["status"] = "SPEAKING"
            await generate_and_stream_tts(txt, device_ws, state["sid"], state)
            await reset_to_standby(device_ws, state)




async def _dispatch_hermes(txt, uid, state, device_ws, _t0, _t_stt):
    """Send transcribed text to HermesBackend and stream the reply to TTS.

    Used for LLM_BACKEND=hermes and =cascade (backends.CascadeBackend is the
    L1-router entry point of the three-level cascade).

    Steps: notify the device that speech starts, arm the watchdog, create a
    sentence queue + player task, hand the queue to the backend (it pushes
    sentences and finally None), wait for playback to drain (120 s hard cap),
    then open the follow-up window instead of going to standby.

    Turn end parity: this used to end with an unconditional reset_to_standby(),
    which put the satellite IDLE right after the reply — including after the
    gateway itself asked a question, so the answer had nowhere to land (the
    firmware's own post-TTS `listen:start` was in addition rejected as a false
    wake by the 1.5 s cooldown). It now runs _finalize_turn_followup(), the
    same finaliser the legacy Nanobot path uses: question -> 30 s of LISTENING
    with the screen at 100 %, statement -> 10 s of LISTENING.

    Side effects: state status/tts_started/watchdog/reply_sentences are
    mutated; the turn's single closing `tts stop` is emitted from the playback
    `finally` below on every exit path, so the satellite always leaves
    SPEAKING;
    on completion _finalize_turn_followup() always opens a LISTENING window
    (question -> 30 s, statement -> 10 s, post-watchdog apology ->
    10 s), so a reply — or the watchdog's own "repeat please" — is always
    answerable without a fresh wake word.
    Failure modes: a dead device WS is tolerated (send errors swallowed), a
    stalled player is cancelled by the timeout.
    """
    # Fallback chat_id: deterministic per MAC so context survives even when
    # the legacy Nanobot session never announced one.
    chat_id = state.get("nanobot_chat_id") or make_chat_id(state["mac"])
    speaker_name = SPEAKER_NAME_MAP.get(uid, uid)
    try:
        await device_ws.send_json(
            {"type": "tts", "state": "start", "session_id": state["sid"]}
        )
    except Exception:
        pass
    state["status"] = "SPEAKING"
    state["tts_started"] = True
    state["watchdog_fired"] = False
    # Spoken-reply accumulator for question detection: _hermes_player_task()
    # appends every sentence it actually plays, _finalize_turn_followup()
    # consumes it. Initialised per turn so a leftover from a previous turn
    # (e.g. one abandoned by a timeout) can never skew the new verdict.
    state["reply_sentences"] = []
    # Watchdog for the whole turn; the first played sentence cancels it
    # (see _hermes_player_task.play), so a long-but-alive reply never
    # triggers the apology.
    if state.get("watchdog"):
        state["watchdog"].cancel()
    state["watchdog"] = asyncio.get_event_loop().call_later(
        WATCHDOG_TIMEOUT,
        lambda: create_tracked_task(watchdog_timeout(device_ws, state), state),
    )

    # Producer/consumer: the backend pushes sentences into q and finally None;
    # _hermes_player_task consumes them. The player is awaited in `finally`
    # so a backend exception still drains (or cancels) the playback task —
    # otherwise audio could keep streaming after the turn was abandoned.
    q: asyncio.Queue = asyncio.Queue()
    player_task = asyncio.create_task(_hermes_player_task(q, device_ws, state))
    try:
        await llm_backend.generate_response(
            text=txt,
            session_id=chat_id,
            stream_name=state.get("mac", "device"),
            response_queue=q,
        )
    finally:
        # 120 s playback cap: far above any single reply, but bounded so a
        # stuck Opus stream cannot wedge the session forever.
        try:
            await asyncio.wait_for(player_task, timeout=120.0)
        except asyncio.TimeoutError:
            player_task.cancel()
        # Always emit the ONE closing `tts stop` of this turn, on every exit
        # path (normal end, backend exception, playback cancel). It is the
        # signal that lets the satellite re-arm its microphone; the follow-up
        # window opened below is unreachable for the user without it.
        await send_tts_stop(device_ws, state)
    _t_done = time.time()
    logger.info(
        f"⏱ [Timing] Hermes reply done: {_t_done-_t_stt:.2f}s after STT | "
        f"total={_t_done-_t0:.2f}s since VAD trigger"
    )
    # Turn end: open the follow-up window (question -> 30 s, statement -> 10 s)
    # instead of dropping to standby, so the satellite can answer the reply we
    # just produced without a fresh wake word.
    #
    # Runs UNCONDITIONALLY, watchdog apology included. The old code called
    # reset_to_standby() here unconditionally as well, which cut dead the very
    # LISTENING window watchdog_timeout() had just opened for its apology.
    # _finalize_turn_followup() re-opens LISTENING in both of its branches, so
    # that window survives — only the idle timer is refreshed and the question
    # verdict re-judged (post-watchdog it is almost always a statement: play()
    # cancels the watchdog on the FIRST sentence attempt, so the watchdog can
    # only fire before anything was appended to reply_sentences; the apology
    # itself goes through generate_and_stream_tts(), never through the player).
    spoken_reply = "".join(state.pop("reply_sentences", []))
    has_question = await _finalize_turn_followup(device_ws, state, spoken_reply)
    logger.info(
        f"👂 [Hermes] Follow-up window: {'question -> 30s' if has_question else 'statement -> 10s'}"
        + (" (post-watchdog)" if state.get("watchdog_fired") else "")
    )


async def _hermes_player_task(q: asyncio.Queue, device_ws, state):
    """Play reply sentences back-to-back with one-ahead synth prefetch.

    Sentence N+1 is synthesized while sentence N plays, hiding Edge-TTS
    latency (sequential synth-then-play left 1-4 s of silence between
    sentences — heard as stutter). Sentences that arrive after the
    watchdog apology are dropped instead of played stale.

    Every sentence that actually reaches the speaker is also appended to
    state["reply_sentences"] — _finalize_turn_followup() probes that text for
    a question to pick the follow-up window (30 s vs 10 s), mirroring what
    NanobotResponseHandler.full_response_text does for the legacy backend.
    Only played sentences count: dropped/stale ones never reached the user.
    """

    def start_synth(text):
        """Start (do not await) MP3 synthesis for one sentence, returning the task."""
        return asyncio.create_task(synthesize_tts_mp3(text, state))

    async def play(mp3_data, text: str):
        """Stream one sentence to the device and disarm the watchdog.

        The watchdog is cancelled only AFTER audio actually flows: proof that
        the upstream L1/L2/L3 chain is alive, so a slow tail of the reply can
        no longer trigger a duplicate apology. `text` is appended to
        state["reply_sentences"] only when the stream actually succeeded, so
        question detection judges exactly what the user heard.
        """
        streamed = False
        try:
            # send_stop=False: the WHOLE turn is one audio unit, the closing
            # `tts stop` is emitted once by _dispatch_hermes() (send_tts_stop).
            # The satellite re-arms its microphone by itself on `tts stop`
            # (it then sends `listen:start`), but only while it still counts
            # itself as LISTENING — it drops to IDLE after ~10 s with no `stt`
            # reply from us, and a `tts stop` from IDLE does NOT re-arm it.
            # Stopping after every sentence therefore stranded the device in
            # LISTENING between sentences (it timed out, then ignored the
            # FINAL stop as well), so the follow-up window opened by
            # _finalize_turn_followup() received zero frames — observed on
            # 2026-09-27: 1-2-sentence replies followed up fine, the 8-sentence
            # 104 s reply did not. One start / one stop keeps the device in
            # SPEAKING for exactly as long as we are speaking.
            streamed = await stream_tts_pcm(
                mp3_data, device_ws, state["sid"], state, send_stop=False
            )
        except Exception as e:
            logger.error(f"❌ [Hermes] TTS player error: {e}")
        # Audio is flowing — upstream proved alive, drop the guard so a
        # slow tail of the reply can't trigger a duplicate apology.
        wd = state.get("watchdog")
        if wd is not None:
            try:
                wd.cancel()
            except Exception:
                pass
            state["watchdog"] = None
        # Remember what the user actually heard: question detection at turn
        # end reads exactly this, never the raw (unspoken) queue contents.
        if streamed and text:
            state.setdefault("reply_sentences", []).append(text + " ")

    first = await q.get()
    if first is None:
        return  # backend produced nothing (empty reply or early error)
    synth_task = start_synth(first) if not state.get("watchdog_fired") else None
    # Text belonging to the in-flight synth task; travels alongside it because
    # the queue only carries strings, not the (text, mp3) pairing.
    synth_text = first if synth_task is not None else None
    while True:
        # One-ahead pipeline: `synth_task` is always the NEXT sentence's
        # synthesis. Awaiting it here overlaps with `play()` of the current
        # one, so Edge-TTS latency is hidden instead of heard as a gap.
        nxt = await q.get()
        mp3 = await synth_task if synth_task is not None else None
        synth_task, spoken_text = None, synth_text
        synth_text = None
        if state.get("watchdog_fired"):
            if nxt is None:
                return
            continue  # stale: apology already spoken, keep draining
        if nxt is not None:
            synth_task = start_synth(nxt)  # prefetch while mp3 plays
            synth_text = nxt
        if mp3 is not None:
            await play(mp3, spoken_text)
        if nxt is None:
            return  # None is the backend's end-of-stream marker


async def _handle_rejected_transcription(
    txt: str,
    uid: str,
    state: dict,
    device_ws: WebSocket,
    avg_rms: float,
    speech_ratio: float,
    frames_len: int,
):
    """Handle a turn whose transcript failed is_valid_text() (or was empty).

    Args mirror the diagnostics: avg_rms/speech_ratio/frames_len explain WHY
    the cut was suspicious, `txt` shows what Whisper produced (empty text =
    likely noise, gibberish = likely echo/hallucination). The rejection counter
    (_rejected_count) makes consecutive failures visible in the logs.

    Side effect: returns the device to standby — the user must wake it again,
    which is the cheapest way to shake off a false cut.
    """
    _rej_count = state.get("_rejected_count", 0) + 1
    state["_rejected_count"] = _rej_count
    if not txt:
        logger.warning(
            f"🔇 [Diag] Whisper returned EMPTY — RMS={avg_rms:.4f}, speech_ratio={speech_ratio:.2f}, "
            f"frames={frames_len}, rejected#{_rej_count}"
        )
    else:
        logger.warning(
            f"⚠ [Pipeline] Rejected transcription (text='{txt}') from user '{uid}' "
            f"(rejected#{_rej_count})"
        )
    await reset_to_standby(device_ws, state)


async def process_audio_and_send(
    frames: list, state: dict, device_ws: WebSocket, decoder: opuslib.Decoder = None
):
    """The STT pipeline: one cut utterance from VAD to LLM dispatch.

    Args:
        frames: Opus frames accumulated between wake and cut.
        state: session state (status transitions PROCESSING -> LISTENING /
        SPEAKING, timing stamps for the latency logs).
        device_ws: satellite WebSocket.
        decoder: reuse the session's Opus decoder (stateful).

    Gate order (cheapest first, so noise costs nothing):
      1. echo guard — skip while our own TTS is still playing (the mic would
         hear the speaker);
      2. min frame count (15 ≈ 0.9 s) — too short to be a command;
      3. energy + speech-ratio thresholds (_process_audio_metrics_and_gates);
      4. after STT: speaker-lock de-dup, then is_valid_text().
    Side effects: spawns the emotion "thinking" animation plus parallel
    speaker-ID and Whisper tasks; any unexpected exception returns the
    device to standby instead of leaving it stuck in PROCESSING.
    """
    _t0 = time.time()
    if time.time() < camera_client.GLOBAL_TTS_UNTIL:
        # GLOBAL_TTS_UNTIL is set by stream_tts_pcm() for playback + 3 s;
        # the guard keeps the satellite from transcribing its own voice.
        logger.info("🔇 [Pipeline] Skipping — TTS playback active (echo guard)")
        state["status"] = "LISTENING"
        return
    if len(frames) < 15:
        logger.info(
            f"🔇 [Pipeline] Too few frames ({len(frames)}), likely noise — skipping STT"
        )
        state["status"] = "LISTENING"
        return
    try:
        passed, avg_rms, speech_ratio, dec = await _process_audio_metrics_and_gates(
            frames, state, decoder
        )
        logger.info(
            f"📊 [Timing] VAD→ready: {time.time()-_t0:.2f}s | RMS={avg_rms:.4f}, speech_ratio={speech_ratio:.2f}, frames={len(frames)}"
        )
        if not passed:
            state["status"] = "LISTENING"
            return

        audio = pack_ogg(frames)
        _t_packed = time.time()

        # Speaker ID and Whisper run concurrently: two independent HTTP calls
        # whose combined latency dominates the turn, so overlap them.
        create_tracked_task(trigger_emotion("thinking", device_ws, state["sid"]), state)
        sess = state["http_session"]
        uid_task = create_tracked_task(fetch_speaker_id(audio, sess), state)
        stt_task = create_tracked_task(fetch_transcription(audio, sess), state)
        uid, txt = await asyncio.gather(uid_task, stt_task)
        _t_stt = time.time()

        logger.info(
            f"⏱ [Timing] Pack={_t_packed-_t0:.2f}s STT+ID={_t_stt-_t_packed:.2f}s | "
            f"{len(audio)} bytes, rms={avg_rms:.4f}, speech_ratio={speech_ratio:.2f}"
        )

        if not _check_speaker_lock(uid, state["sid"]):
            state["status"] = "LISTENING"
            return

        if is_valid_text(txt):
            await _handle_successful_transcription(
                txt, uid, state, device_ws, _t0, _t_stt
            )
        else:
            await _handle_rejected_transcription(
                txt, uid, state, device_ws, avg_rms, speech_ratio, len(frames)
            )

    except Exception as e:
        logger.error(f"❌ Pipeline Error: {e}")
        await reset_to_standby(device_ws, state)


# ==========================================
# TTS & EMOTION
# ==========================================
async def synthesize_tts_mp3(text: str, state: dict) -> bytes | None:
    """Synthesize text to MP3 via the TTS API (network-bound, no streaming).

    Args:
        text: sentence to speak (already stripped of emotion tags).
        state: session state — only state["http_session"] is used.
    Returns:
        bytes | None: MP3 payload, or None on non-200 responses/errors.

    The OpenAI-style /v1/audio/speech contract (model, voice, response_format)
    is used so any compatible server (edge_tts sidecar here) can be swapped in
    via TTS_URL. "Cannot call" errors are filtered from the log because they
    are the expected noise of a device that disconnected mid-turn.
    """
    logger.info(f"🔊 [TTS] Synthesizing: '{text}'")
    try:
        sess = state["http_session"]
        payload = {
            "model": TTS_MODEL,
            "input": text,
            "voice": TTS_VOICE,
            "response_format": "mp3",
        }
        headers = {"Content-Type": "application/json"}
        if TTS_API_KEY:
            headers["Authorization"] = f"Bearer {TTS_API_KEY}"
        async with sess.post(TTS_URL, json=payload, headers=headers, timeout=30) as r:
            if r.status == 200:
                return await r.read()
            logger.error(f"❌ [TTS] API Error: {await r.text()}")
    except Exception as e:
        if "Cannot call" not in str(e):
            logger.error(f"❌ [TTS] Error: {e}")
    return None


async def stream_tts_pcm(
    mp3_data: bytes,
    device_ws: WebSocket,
    session_id: str,
    state: dict,
    send_stop: bool = True,
) -> bool:
    """Stream pre-synthesized MP3 to the device. Returns True on success.

    Pipeline: pydub decodes the MP3 -> re-samples to 16 kHz mono s16 (the
    contract negotiated in `hello`) -> Opus-encodes 60 ms frames -> sends them
    as binary WS messages, paced against a virtual clock so the speaker plays
    at real time instead of being flooded.

    Args:
        mp3_data: complete MP3 of one utterance.
        session_id: satellite session id echoed in the control events.
        state: session state; only used when truthy (a bare `None` is
            tolerated so shared helpers can call this without a session).
        send_stop: send the closing `tts stop` event — False while more
            segments of the SAME utterance follow (the continuous legacy
            Nanobot stream and the cascade/hermes sentence player), so the
            device keeps one open audio unit and only sees a single `tts stop`
            at the very end of the turn. Only that final stop makes the
            satellite re-arm its microphone for the follow-up window.

    Side effects: sets camera_client.GLOBAL_TTS_UNTIL (echo guard for the
    whole gateway) and state["tts_cooldown_until"] (+1.5 s) so VAD ignores
    the loudness tail of our own playback.
    Failure modes: returns False on send failures or decode errors; "Cannot
    call" errors (device already closed) are suppressed from the logs.
    """
    try:
        if not state or not state.get("tts_started"):
            # Announce the start exactly once per utterance; without it the
            # firmware does not open the speaker path.
            try:
                await device_ws.send_json(
                    {"type": "tts", "state": "start", "session_id": session_id}
                )
            except Exception:
                return False
            if state:
                state["tts_started"] = True

        audio_seg = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
        audio_seg = audio_seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        pcm_data = audio_seg.raw_data

        # Extend the gateway-wide echo guard by the exact playback duration
        # (bytes / (16000 samples/s * 2 bytes/sample)) plus a 3 s margin for
        # the room's reverb tail.
        camera_client.GLOBAL_TTS_UNTIL = time.time() + len(pcm_data) / (16000 * 2) + 3.0

        # Encoder cached in the session (setdefault): creating it per call
        # would reset internal state and cost a fresh codec instance each
        # sentence. "voip" mode trades bitrate for low delay.
        enc = state.setdefault("tts_encoder", opuslib.Encoder(16000, 1, "voip"))
        frame_size = 960          # samples per Opus packet (60 ms @ 16 kHz)
        chunk_size = frame_size * 2  # int16 -> 2 bytes per sample
        _tts_start = time.time()

        start_stream = time.perf_counter()
        next_chunk_time = start_stream

        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk += b"\x00" * (chunk_size - len(chunk))
            opus_frame = enc.encode(chunk, frame_size)

            try:
                await device_ws.send_bytes(opus_frame)
            except Exception as e:
                logger.error(f"❌ [TTS] Send failed mid-stream: {e}")
                return False

            # Real-time pacing: advance a virtual clock by the frame's
            # duration (60 ms) and sleep the remainder, so frames leave the
            # gateway at playback speed. Sending faster would just buffer in
            # the firmware (and defeat the echo guard timing above).
            next_chunk_time += 0.06
            sleep_duration = next_chunk_time - time.perf_counter()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

        _tts_end = time.time()
        logger.info(f"⏱ [Timing] TTS done: {_tts_end - _tts_start:.1f}s playback")
        logger.info("✅ [TTS] Audio stream completed smoothly.")
        # Set TTS cooldown to prevent VAD triggering on our own output
        if state:
            state["tts_cooldown_until"] = time.time() + 1.5
        if send_stop:
            # Each standalone utterance is a self-contained audio unit.
            if state:
                state["tts_started"] = False
            try:
                await device_ws.send_json(
                    {"type": "tts", "state": "stop", "session_id": session_id}
                )
            except Exception:
                pass
        return True
    except Exception as e:
        if "Cannot call" not in str(e):
            logger.error(f"❌ [TTS] Error: {e}")
        return False


async def send_tts_stop(device_ws, state: dict) -> None:
    """Close the turn's open TTS stream (one `tts stop` control event).

    The satellite treats `tts start` -> `tts stop` as a single audio unit: it
    opens the speaker path on start and re-arms the microphone on stop (which
    is what makes the follow-up window answerable without a wake word). The
    closing stop must therefore be sent EXACTLY once per turn and always —
    including turns where nothing reached the speaker at all (every Edge-TTS
    call failed, the backend produced no sentence, playback was cancelled by
    the 120 s cap). Without it the satellite stays in SPEAKING with the mic
    shut and the LISTENING window we open next expires without a single frame,
    as happened on 2026-09-27 15:11 ("no audio received in 10s").

    No-op when no stream is open (tts_started False) — duplicate stops are
    ignored by the firmware, but a stray one would splice the next utterance
    onto a stream the caller believes is closed.

    Failure modes: a dead device WebSocket is swallowed; only the satellite's
    state is stale then, the gateway state is reset regardless.
    """
    if not state.get("tts_started"):
        return
    state["tts_started"] = False
    try:
        await device_ws.send_json(
            {"type": "tts", "state": "stop", "session_id": state["sid"]}
        )
    except Exception:
        pass


async def generate_and_stream_tts(
    text: str, device_ws: WebSocket, session_id: str, state: dict = None
):
    """Compatibility wrapper: synthesize then stream a single utterance.

    Args:
        text: sentence to speak.
        device_ws/session_id: where to send it.
        state: session state; must be provided (it carries the HTTP session
        and the Opus encoder) — without it the call is a logged no-op rather
        than a crash, e.g. during teardown of an already-closed session.
    Silently does nothing when synthesis fails (synthesize_tts_mp3 -> None).
    """
    if not state:
        logger.error("❌ [TTS] generate_and_stream_tts called without state")
        return
    mp3_data = await synthesize_tts_mp3(text, state)
    if mp3_data is not None:
        await stream_tts_pcm(mp3_data, device_ws, session_id, state)


async def trigger_emotion(
    emotion: str, device_ws: WebSocket, session_id: str, logger=logger
):
    """Ask the satellite to switch its display face (`llm` + emotion event).

    The `text: " "` payload is required by the firmware's message schema even
    though no speech follows. Fire-and-forget: a device that already
    disconnected ("Cannot call send") is ignored silently, other failures are
    only warned — cosmetics must never break the audio turn.
    """
    logger.info(f"💡 [Emotion] Setting display face to: '{emotion}'")
    try:
        await device_ws.send_json(
            {"session_id": session_id, "type": "llm", "emotion": emotion, "text": " "}
        )
    except Exception as e:
        if "Cannot call send" not in str(e):
            logger.warning(f"⚠️ [Emotion] Failed to set emotion '{emotion}': {e}")


# ==========================================
# NANOBOT WEBSOCKET RESPONSE HANDLER
# ==========================================
# Emotion tags the brain wraps around sentences ("[neutral]", "[thinking]"...)
# are extracted and sent to the display, then stripped from the spoken text.
EMOTION_REGEX = re.compile(r"\[([a-zA-Z0-9_]+)\]")

class NanobotResponseHandler:
    """Buffers a streaming LLM reply and turns it into speech, sentence by sentence.

    One instance per device connection (stored in state["handler"]). It owns:
      * `buffer` — text chunks not yet spoken;
      * a flush timer (call_later) that bounds how long a partial sentence
        may sit unspoken;
      * a TTS audio queue + single player task, so synthesis of sentence N+1
        overlaps playback of sentence N;
      * `_synth_tasks` — in-flight synthesis jobs awaited before finalising.

    Lifecycle: handle_chunk() per incoming text chunk -> flush() on a complete
    sentence or timer expiry -> _handle_tts() -> after the buffer drains,
    _finalize_response() decides follow-up listening vs standby.

    Failure modes: a dead device WebSocket surfaces as send errors that are
    swallowed per message; if playback fails, _tts_player() stops and the
    next _ensure_tts_player() restarts it.
    """

    def __init__(self, device_ws, state):
        """Bind the handler to one device connection and its session state."""
        self.device_ws = device_ws
        self.state = state
        self.buffer = []           # pending text chunks (not yet spoken)
        self.full_response_text = []  # everything spoken this turn (question detection)
        self.timer = None          # call_later handle for delayed flushes
        self.emotion_regex = EMOTION_REGEX
        self.is_flushing = False   # re-entrancy guard: flush() must not overlap
        self._first_chunk_time = None
        self._last_chunk_time = None
        self._chunk_count = 0
        self.tts_audio_queue: asyncio.Queue = asyncio.Queue()
        self.tts_player_task = None
        self._synth_tasks: set = set()

    async def _tts_player(self):
        """Play pre-synthesized segments back-to-back as one continuous
        stream (no stop/start between segments, so audio is gapless).

        Sends every segment with send_stop=False; the matching single stop is
        emitted later by _await_tts_drained(). Exits on the None sentinel or
        as soon as a send fails (device gone) — _ensure_tts_player() respawns
        it for the next turn.
        """
        while True:
            mp3_data = await self.tts_audio_queue.get()
            if mp3_data is None:
                self.tts_audio_queue.task_done()
                return
            ok = await stream_tts_pcm(
                mp3_data,
                self.device_ws,
                self.state["sid"],
                self.state,
                send_stop=False,
            )
            self.tts_audio_queue.task_done()
            if not ok:
                break
            await asyncio.sleep(0.1)

    async def _send_tts_stop(self):
        """Close the continuous TTS stream (single `tts stop` event).

        Thin wrapper over the shared send_tts_stop() helper — the legacy
        Nanobot path and the cascade/hermes path must agree on the wire
        protocol: no-op when no stream is open (tts_started False), otherwise
        one `tts stop` per audio unit.
        """
        await send_tts_stop(self.device_ws, self.state)

    async def _enqueue_tts(self, text: str):
        """Synthesize in background; player picks it up when ready."""
        mp3_data = await synthesize_tts_mp3(text, self.state)
        if mp3_data is not None:
            await self.tts_audio_queue.put(mp3_data)

    def _ensure_tts_player(self):
        """Start the background player task if it is missing or has finished.

        Lazy start (rather than one per connection) so a session that never
        speaks pays nothing; after a failure the next sentence revives it.
        """
        if self.tts_player_task is None or self.tts_player_task.done():
            self.tts_player_task = create_tracked_task(self._tts_player(), self.state)

    async def _await_tts_drained(self):
        """Wait until all background synthesis and playback has finished,
        then close the continuous TTS stream with a single stop message.

        Order matters: synthesis tasks first (they push into the queue), then
        queue.join() (playback drained), then the stop event. The 30 s timeout
        guarantees a stuck playback cannot block the end of the turn — the
        stop is sent either way.
        """
        while self._synth_tasks:
            await asyncio.sleep(0.05)
        try:
            await asyncio.wait_for(self.tts_audio_queue.join(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("❌ [TTS] Timed out waiting for playback to finish")
        await self._send_tts_stop()

    def reset_timing(self):
        """Clear the per-turn chunk timing stats (called when a new request is
        dispatched so the latency logs describe this turn only)."""
        self._first_chunk_time = None
        self._last_chunk_time = None
        self._chunk_count = 0

    async def handle_chunk(self, chunk: str):
        """Consume one text chunk streamed from the LLM brain.

        Cancels the watchdog (the upstream is demonstrably alive), records
        timing stats and schedules a flush: immediately when the buffer ends
        on a sentence boundary with balanced brackets, otherwise after a
        1.5-2.0 s quiet timer (longer when an emotion tag is still open, so a
        chunk split inside "[neutral]" is never spoken as literal text).
        """
        if self.state.get("watchdog"):
            self.state["watchdog"].cancel()
            self.state["watchdog"] = None
        now = time.time()
        # Log first chunk timing
        if not self._first_chunk_time:
            self._first_chunk_time = now
            self._chunk_count = 0
            t_since_stt = now - self.state.get("_stt_time", now)
            logger.info(
                f"⏱ [Timing] First Nanobot chunk: +{t_since_stt:.1f}s after STT"
            )
        self._chunk_count += 1
        self._last_chunk_time = now
        self.buffer.append(chunk)
        if self.timer:
            self.timer.cancel()
        # Sentence-level flush: start TTS as soon as a complete sentence has
        # arrived instead of waiting for the whole LLM response to finish.
        # Chunks arrive every ~60ms, so a pending timer would be cancelled by
        # the next chunk before it fires; launch the flush immediately instead.
        buffer_str = "".join(self.buffer)
        brackets_balanced = buffer_str.count("[") == buffer_str.count("]")
        delay = 2.0 if not brackets_balanced else 1.5
        self.timer = asyncio.get_event_loop().call_later(
            delay, lambda: create_tracked_task(self.flush(), self.state)
        )
        if brackets_balanced and SENTENCE_END_RE.search(buffer_str) and not self.is_flushing:
            self.timer.cancel()
            self.timer = None
            create_tracked_task(self.flush(), self.state)

    def _process_buffer(self) -> str | None:
        """Take the speakable prefix out of the buffer.

        Returns:
            str | None: the text up to and including the last complete
            sentence (trailing partial text is pushed back into the buffer so
            phrases are never cut mid-thought), or None when an emotion tag
            is still unbalanced — the caller then waits for more chunks.

        Synchronous and deliberately side-effecting: consumes `buffer` and,
        in the None case, re-arms a 0.5 s retry timer.
        """
        text = "".join(self.buffer)
        self.buffer = []

        # If emotion tag is still incomplete, wait for next chunk
        if text.count("[") > text.count("]"):
            self.buffer = [text]
            self.timer = asyncio.get_event_loop().call_later(
                0.5, lambda: create_tracked_task(self.flush(), self.state)
            )
            return None

        # Flush only up to the last complete sentence; any trailing partial
        # text stays in the buffer so phrases are not cut mid-thought and
        # continuation words ("Или", "Если"...) keep flowing into the next
        # segment naturally.
        sentence_matches = list(SENTENCE_END_RE.finditer(text))
        if sentence_matches:
            last_match = sentence_matches[-1]
            trailing = text[last_match.end() :]
            if trailing.strip():
                text = text[: last_match.end()]
                self.buffer = [trailing] + self.buffer

        return text

    async def _handle_disconnect(self, clean_text: str) -> bool:
        """Honour a `[disconnect]` command emitted by the brain.

        The tag means "the user asked to end the session": any text before it
        is spoken first (so the acknowledgement is heard), then the device
        WebSocket is closed.

        Returns:
            bool: True when the tag was present (the caller must stop
            processing this flush); False means normal speech continues.
        """
        if "[disconnect]" not in clean_text:
            return False

        clean_text = clean_text.replace("[disconnect]", "").strip()
        if clean_text:
            logger.info(
                f"📝 [TTS Input] Sending to TTS: '{clean_text[:100]}...' (len={len(clean_text)})"
            )
            try:
                await self.device_ws.send_json(
                    {
                        "type": "tts",
                        "state": "sentence_start",
                        "text": clean_text,
                        "session_id": self.state["sid"],
                    }
                )
            except Exception:
                pass
            mp3_data = await synthesize_tts_mp3(clean_text, self.state)
            if mp3_data is not None:
                await stream_tts_pcm(
                    mp3_data, self.device_ws, self.state["sid"], self.state
                )
        # Close session — user explicitly requested disconnect
        logger.info("🔌 [Disconnect] User requested disconnect — closing session")
        try:
            await self.device_ws.close()
        except Exception:
            pass
        return True

    async def _handle_tts(self, clean_text: str):
        """Speak one sentence chunk.

        Notifies the device (`tts sentence_start` + text for the subtitle),
        accumulates full_response_text (used later for question detection)
        and queues background synthesis while the previous segment is still
        playing — this pipeline is what makes the reply sound continuous.
        Send failures are swallowed: the queue drain is what actually gates
        the end of the turn.
        """
        if not clean_text:
            return

        t_flush = time.time()
        t_since_first = t_flush - (self._first_chunk_time or t_flush)
        t_since_last = t_flush - (self._last_chunk_time or t_flush)
        logger.info(
            f"📝 [TTS Input] Sending to TTS: '{clean_text[:100]}...' (len={len(clean_text)})"
        )
        logger.info(
            f"⏱ [Timing] Flush→TTS: +{t_since_first:.1f}s after first chunk, "
            f"+{t_since_last:.1f}s after last chunk, "
            f"{self._chunk_count} chunks"
        )
        self.full_response_text.append(clean_text + " ")
        try:
            await self.device_ws.send_json(
                {
                    "type": "tts",
                    "state": "sentence_start",
                    "text": clean_text,
                    "session_id": self.state["sid"],
                }
            )
        except Exception:
            pass

        # Pipeline TTS: synthesize in background while the previous
        # segment is still playing, so phrases flow without gaps.
        self._ensure_tts_player()
        # _synth_tasks tracks in-flight synthesis so _await_tts_drained() can
        # wait for it; the done-callback unregisters it (no manual cleanup).
        task = create_tracked_task(self._enqueue_tts(clean_text), self.state)
        self._synth_tasks.add(task)
        task.add_done_callback(self._synth_tasks.discard)

    async def _finalize_response(self):
        """End of a turn: drain audio, then choose follow-up listening or standby.

        Waits for all playback, then delegates to the shared
        _finalize_turn_followup(), which probes the full reply for a question
        (trailing '?', the «повторите пожалуйста» apology, or any Russian
        interrogative / imperative from HAS_QUESTION_WORDS_RE) and stores the
        verdict in state["last_ai_had_question"] — activity_monitor_task()
        reads it to pick the standby timeout (30 s vs 10 s).

        Delegating keeps the legacy Nanobot backend and the cascade backend on
        one code path: they used to each carry their own copy of this
        transition, and only this one had the follow-up window.
        """
        await self._await_tts_drained()

        # Question detection runs on the SPOKEN text (tags already stripped).
        await _finalize_turn_followup(
            self.device_ws, self.state, "".join(self.full_response_text)
        )
        self.full_response_text = []

    async def flush(self):
        """Cut the buffer into speech: emotions, disconnect, TTS, finalisation.

        Re-entrancy: guarded by is_flushing, so a timer firing while a flush
        is in progress exits immediately (the active flush re-schedules
        itself if more text arrived meanwhile).

        Steps: extract speakable text -> fire display emotions for every
        [tag] -> handle [disconnect] -> queue TTS -> if the brain is still
        streaming, schedule the next flush in 150 ms; only when the buffer is
        empty does the turn finalise (question check, standby).
        """
        if self.is_flushing or not "".join(self.buffer).strip():
            return
        self.is_flushing = True
        # Speech means the turn is active again: unpause the idle accounting.
        self.state["status"] = "SPEAKING"

        text = self._process_buffer()
        if text is None:
            self.is_flushing = False
            return

        # Emotion tags are shown on the display and then stripped from the
        # text — the TTS must never read "[neutral]" aloud.
        emotions = self.emotion_regex.findall(text)
        for emotion in emotions:
            create_tracked_task(
                trigger_emotion(emotion, self.device_ws, self.state["sid"]), self.state
            )

        clean_text = self.emotion_regex.sub("", text).strip()

        if await self._handle_disconnect(clean_text):
            self.is_flushing = False
            return

        await self._handle_tts(clean_text)

        # End of flush: if the LLM is still streaming, flush the remaining
        # content shortly after; only finalize (dialogue mode check, return
        # to standby) once the whole response has been produced.
        self.is_flushing = False
        if "".join(self.buffer).strip():
            self.timer = asyncio.get_event_loop().call_later(
                0.15, lambda: create_tracked_task(self.flush(), self.state)
            )
            return

        await self._finalize_response()


# ==========================================
# WEB UI
# ==========================================
@app.get("/health")
async def health():
    """Liveness probe for the container/monitoring — unauthenticated by design
    (it exposes nothing beyond process liveness)."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def web_index(req: Request, username: str = Depends(verify_auth)):
    """Render the operator dashboard (templates/index.html).

    Requires HTTP Basic auth (401 otherwise); `username` is only used by the
    dependency, the template itself fetches the /api/* endpoints with the
    browser's cached credentials.
    """
    return templates.TemplateResponse(req, "index.html")


# ==========================================
# MAIN FASTAPI WEBSOCKET GATEWAY
# ==========================================
async def listen_to_nanobot_task(
    device_ws: WebSocket, state: dict, nano_session: aiohttp.ClientSession
):
    """Maintain the upstream Nanobot WS and dispatch every event it emits.

    Started lazily on the device `hello` (legacy backend only — Hermes/
    Cascade never use nano_ws, see handle_ws_text_message) and loops until
    the session disappears from session_states.

    Reconnect policy: connection failure -> retry after 10 s; connection lost
    mid-read -> retry after 5 s. Each new connection gets a fresh
    NanobotResponseHandler (buffers/queues of the old one are abandoned with
    the socket).

    Event handling:
      * `ready`      -> cache the deterministic chat_id for this MAC;
      * `error`      -> speak the apology and re-arm listening (except the
                        benign "unknown type" protocol warning);
      * `device_tool_call` -> bridge to the satellite (handle_device_tool_call);
      * `text`       -> strip provider errors (apology path) or feed the
                        sentence flusher; replies arriving after the watchdog
                        apology are dropped as stale.
    Every text event re-arms the WATCHDOG_TIMEOUT timer.

    Failure modes: malformed JSON is logged and skipped; any exception in the
    read loop is caught to schedule the reconnect instead of killing the task.
    """
    nano_ws = state.get("nano_ws")
    handler = NanobotResponseHandler(device_ws, state)
    state["handler"] = handler
    while state["sid"] in session_states:
        if nano_ws is None or nano_ws.closed:
            try:
                mac_key = state["mac"].lower()
                det_chat_id = make_chat_id(mac_key)
                # Token + deterministic chat_id go in the query string — that
                # is the auth/routing contract of the Nanobot WS endpoint, and
                # the stable chat_id is what preserves conversation context
                # across reconnects.
                auth_url = (
                    f"{NANOBOT_WS_URL}?token={NANOBOT_TOKEN}&chat_id={det_chat_id}"
                )
                nano_ws = await nano_session.ws_connect(auth_url)
                state["nano_ws"] = nano_ws
                handler = NanobotResponseHandler(device_ws, state)
                state["handler"] = handler
                logger.info(f"✅ [Nanobot] Connected (chat_id={det_chat_id})")
            except Exception as e:
                logger.warning(
                    f"🔁 [Nanobot] Connection failed ({e}), retrying in 10s..."
                )
                if state["sid"] not in session_states:
                    break
                await asyncio.sleep(10)
                continue
        try:
            async for msg in nano_ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Any inbound text proves the upstream chain is alive:
                    # slide the watchdog deadline forward instead of letting
                    # a long-but-streaming reply be interrupted.
                    if state.get("watchdog"):
                        state["watchdog"].cancel()
                        state["watchdog"] = asyncio.get_event_loop().call_later(
                            WATCHDOG_TIMEOUT,
                            lambda: create_tracked_task(
                                watchdog_timeout(device_ws, state), state
                            ),
                        )

                    try:
                        d = json.loads(msg.data)
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"❌ [Nanobot] Invalid JSON from Nanobot: {e}, data={msg.data[:200]}"
                        )
                        continue

                    if d.get("event") == "ready":
                        nano_chat_id = d.get("chat_id")
                        # Ignore the id the brain just minted: ours is
                        # deterministic per MAC, so context survives a gateway
                        # or Nanobot restart (logged alongside for debugging).
                        state["nanobot_chat_id"] = make_chat_id(state["mac"])
                        mac_key = state["mac"].lower()
                        set_cached_chat_id(mac_key, state["nanobot_chat_id"])
                        logger.info(
                            f"💾 [Nanobot] Cached deterministic chat_id for {mac_key}: {state['nanobot_chat_id']} (Nanobot assigned: {nano_chat_id})"
                        )
                        if state["available_tools"]:
                            logger.info(
                                f"🛠 [MCP] Tools available: {len(state['available_tools'])} (managed by nanobot in v0.3.0)"
                            )
                    elif d.get("event") == "error":
                        detail = d.get("detail", "")
                        if "unknown type" in detail:
                            logger.warning(f"⚠️ Nanobot protocol: {detail}")
                        else:
                            logger.error(f"❌ Nanobot Error: {detail}")
                            if state.get("watchdog"):
                                state["watchdog"].cancel()
                                state["watchdog"] = None
                            state["status"] = "SPEAKING"
                            await generate_and_stream_tts(
                                "Прости, я затупила. Повтори пожалуйста.",
                                device_ws,
                                state["sid"],
                                state,
                            )
                            state.update(
                                {
                                    "status": "LISTENING",
                                    "frames": [],
                                    "silence": 0,
                                    "has_speech": False,
                                    "last_activity": time.time(),
                                }
                            )
                            state["vad"].reset()
                            await send_mcp_cmd(
                                device_ws,
                                state["sid"],
                                "self.screen.set_brightness",
                                {"brightness": 100},
                            )
                    elif d.get("event") == "device_tool_call":
                        create_tracked_task(
                            handle_device_tool_call(nano_ws, device_ws, state, d),
                            state,
                        )
                    elif (
                        "text" in d
                        and d.get("type") not in ["stt", "listen"]
                        and d.get("event") != "reasoning_delta"
                    ):
                        text_content = d["text"]
                        if (
                            "Error from provider" in text_content
                            or text_content.startswith("Error:")
                        ):
                            logger.error(f"❌ Nanobot Error in text: {text_content}")
                            if state.get("watchdog"):
                                state["watchdog"].cancel()
                                state["watchdog"] = None
                            state["status"] = "SPEAKING"
                            await generate_and_stream_tts(
                                "Прости, я затупила. Повтори пожалуйста.",
                                device_ws,
                                state["sid"],
                                state,
                            )
                            state.update(
                                {
                                    "status": "LISTENING",
                                    "frames": [],
                                    "silence": 0,
                                    "has_speech": False,
                                    "last_activity": time.time(),
                                }
                            )
                            state["vad"].reset()
                            await send_mcp_cmd(
                                device_ws,
                                state["sid"],
                                "self.screen.set_brightness",
                                {"brightness": 100},
                            )
                        else:
                            if state.get("watchdog_fired"):
                                logger.info(
                                    f"⏱️ [Watchdog] Ignoring late Nanobot response"
                                )
                                continue
                            await handler.handle_chunk(text_content)
        except Exception as e:
            logger.warning(f"🔁 [Nanobot] Connection lost ({e}), reconnecting in 5s...")
            await asyncio.sleep(5)


@dataclass
class WSContext:
    d: dict
    state: dict
    device_ws: WebSocket
    session_id: str
    dec: opuslib.Decoder
    nano_session: aiohttp.ClientSession
    nano_listener_task: asyncio.Task | None


async def handle_ws_text_message(ctx: WSContext):
    d = ctx.d
    state = ctx.state
    device_ws = ctx.device_ws
    session_id = ctx.session_id
    dec = ctx.dec
    nano_session = ctx.nano_session
    nano_listener_task = ctx.nano_listener_task
    if d.get("type") == "mcp":
        _clean_stale_futures()
        payload = d.get("payload", {})
        req_id = payload.get("id")

        if req_id == 999 and "result" in payload and "tools" in payload["result"]:
            state["available_tools"] = payload["result"]["tools"]
            tool_names = [t.get("name") for t in state["available_tools"]]
            logger.info(f"🛠 [MCP] ESP32 returned tools: {tool_names}")
            logger.info(
                f"🛠 [MCP] Tools available: {len(state['available_tools'])}"
            )
            return nano_listener_task, True

        if req_id in mcp_futures and not mcp_futures[req_id].done():
            mcp_futures[req_id].set_result(payload)
        elif req_id is None and payload.get("method"):
            nano_ws = state.get("nano_ws")
            if nano_ws and not nano_ws.closed and state.get("nanobot_chat_id"):
                try:
                    await nano_ws.send_json(
                        {
                            "type": "device_event",
                            "chat_id": state["nanobot_chat_id"],
                            "event": payload["method"],
                            "data": payload.get("params", {}),
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"❌ [MCP] Failed to forward device_event to Nanobot: {e}"
                    )

    elif d.get("type") == "hello":
        state["version"] = d.get("version", 1)
        logger.info(f"🤝 [Device] Hello received (v{state['version']})")
        # In Hermes/Cascade mode the LLM reply path (_dispatch_hermes) never
        # uses nano_ws, so don't open a pointless reconnect loop to Nanobot.
        if nano_listener_task is None and not (
            LLM_BACKEND in ("hermes", "cascade") and llm_backend is not None
        ):
            nano_listener_task = create_tracked_task(
                listen_to_nanobot_task(device_ws, state, nano_session), state
            )

    elif d.get("type") == "listen" and d.get("state") == "start":
        # Skip false wake word immediately after TTS (audio tail)
        if time.time() < state.get("tts_cooldown_until", 0):
            logger.info(
                f"👂 [Wake] Ignoring listen:start during TTS cooldown (false wake)"
            )
            return nano_listener_task, False
        logger.info(f"👂 [Wake] listen:start — wake word detected, entering LISTENING")
        state["status"] = "LISTENING"
        state["frames"] = []
        state["silence"] = 0
        state["has_speech"] = False
        state["last_activity"] = time.time()
        state["vad"].reset()
        state["post_wake_cooldown_until"] = time.time() + 0.3
        # Fresh window: the monitor's "no audio received" warning is only
        # meaningful if the flag is cleared on every new listening session.
        state["_wake_audio_received"] = False
        state["watchdog_fired"] = False
        state["tts_started"] = False
        try:
            await device_ws.send_json(
                {
                    "session_id": state["sid"],
                    "type": "mcp",
                    "payload": {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "self.screen.set_brightness",
                            "arguments": {"brightness": 100},
                        },
                        "id": int(time.time() * 1000),
                    },
                }
            )
        except Exception as e:
            logger.warning(f"⚠️ [Wake] Failed to light screen: {e}")

    elif d.get("type") == "listen" and d.get("state") == "stop":
        if state["status"] == "LISTENING" and len(state["frames"]) >= 10:
            state["status"] = "PROCESSING"
            frames_to_process, state["frames"] = list(state["frames"]), []
            create_tracked_task(
                process_audio_and_send(frames_to_process, state, device_ws, dec), state
            )

    return nano_listener_task, False


async def handle_ws_audio_message(
    byte_data: bytes, state: dict, device_ws: WebSocket, dec: opuslib.Decoder
):
    if state["status"] == "IDLE":
        return
    if state["status"] in ["PROCESSING", "SPEAKING"]:
        return
    if time.time() < state.get("tts_cooldown_until", 0):
        return
    if time.time() < state.get("post_wake_cooldown_until", 0):
        return
    state["status"] = "LISTENING"

    f = (
        byte_data
        if state["version"] == 1
        else (byte_data[16:] if state["version"] == 2 else byte_data[4:])
    )
    state["frames"].append(f)
    # The device IS streaming: silence the monitor's "no audio received"
    # warning for this listening window (it read the flag but nothing ever
    # set it before, so the warning fired unconditionally).
    state["_wake_audio_received"] = True

    try:
        pcm = dec.decode(f, 960)
        audio_int16 = np.frombuffer(pcm, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        frame_rms = float(np.sqrt(np.mean(np.square(audio_float32))))
        if frame_rms < 0.003:
            is_sp = False
        else:
            is_sp, _ = await state["vad"].is_speech(pcm, frame_rms)
        if is_sp:
            state["silence"] = 0
            state["has_speech"] = True
            state["last_activity"] = time.time()
        else:
            state["silence"] += 1
    except Exception:
        state["silence"] += 1

    if state["silence"] > VAD_SILENCE_FRAMES:
        if state["has_speech"]:
            logger.info(f"🔪 Server VAD triggered.")
            state["status"] = "PROCESSING"
            frames_to_process, state["frames"] = list(state["frames"]), []
            state["silence"] = 0
            state["has_speech"] = False
            create_tracked_task(
                process_audio_and_send(frames_to_process, state, device_ws, dec), state
            )
        else:
            state["frames"], state["silence"] = [], 0
            state["vad"].reset()


@app.websocket("/")
async def voice_ws(device_ws: WebSocket):
    global _active_speaker_lock

    token = device_ws.query_params.get("token")
    if not token:
        auth_header = device_ws.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token or not NANOBOT_TOKEN or not isinstance(token, str) or not secrets.compare_digest(token, NANOBOT_TOKEN):
        await device_ws.close(code=1008, reason="Unauthorized")
        return

    await device_ws.accept()

    fwd_headers = {k.lower(): v for k, v in device_ws.headers.items()}
    mac_addr = fwd_headers.get("device-id", fwd_headers.get("mac", "unknown"))
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = device_ws

    state = {
        "status": "IDLE",
        "frames": [],
        "silence": 0,
        "has_speech": False,
        "sid": session_id,
        "mac": mac_addr,
        "version": 1,
        "watchdog": None,
        "nanobot_chat_id": make_chat_id(mac_addr),
        "last_text": "",
        "last_activity": time.time(),
        "available_tools": [],
        "tts_cooldown_until": 0.0,
        "vad": VadEngine(
            rms_noise_floor=0.008, rms_alpha=0.005, energy_threshold=0.008
        ),
        "tasks": set(),
        "last_receive": time.time(),
        "tts_started": False,
        "watchdog_fired": False,
    }
    session_states[session_id] = state
    logger.info(f"🔌 [WS] Device connected. Session: {session_id}")

    monitor_task = asyncio.create_task(activity_monitor_task(device_ws, state))
    state["tasks"].add(monitor_task)
    monitor_task.add_done_callback(lambda _: state["tasks"].discard(monitor_task))

    await device_ws.send_json(
        {
            "type": "hello",
            "transport": "websocket",
            "session_id": session_id,
            "audio_params": {
                "format": "opus",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration": 60,
            },
        }
    )

    await request_mcp_tools(device_ws, session_id)

    dec = opuslib.Decoder(16000, 1)
    state["vad"].reset()
    nano_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    state["http_session"] = http_session
    nano_ws = None
    nano_listener_task = None

    try:
        while True:
            try:
                m = await asyncio.wait_for(device_ws.receive(), timeout=600)
                state["last_receive"] = time.time()
            except asyncio.TimeoutError:
                try:
                    await device_ws.send_json(
                        {"type": "ping", "session_id": session_id}
                    )
                    m = await asyncio.wait_for(device_ws.receive(), timeout=5)
                    state["last_receive"] = time.time()
                    if m.get("text"):
                        d = json.loads(m["text"])
                        if d.get("type") == "pong":
                            continue
                except Exception:
                    pass
                logger.warning(
                    f"🔌 [WS] No data from device for 600s, closing session {session_id}"
                )
                break
            # --- TEXT EVENTS ---
            text_data = m.get("text")
            if text_data:
                state["last_activity"] = time.time()
                d = json.loads(text_data)
                ctx = WSContext(
                    d=d,
                    state=state,
                    device_ws=device_ws,
                    session_id=session_id,
                    dec=dec,
                    nano_session=nano_session,
                    nano_listener_task=nano_listener_task
                )
                nano_listener_task, should_continue = await handle_ws_text_message(ctx)
                if should_continue:
                    continue

            # --- AUDIO BYTES ---
            byte_data = m.get("bytes")
            if byte_data:
                await handle_ws_audio_message(byte_data, state, device_ws, dec)

    except Exception as e:
        logger.error(f"Error in device websocket loop: {e}")
    finally:
        if _active_speaker_lock and _active_speaker_lock["sid"] == session_id:
            _active_speaker_lock = None
        monitor_task.cancel()
        for t in list(state.get("tasks", set())):
            t.cancel()
        if state.get("watchdog"):
            state["watchdog"].cancel()
        if session_id in active_sessions:
            del active_sessions[session_id]
        if session_id in session_states:
            del session_states[session_id]
        if nano_listener_task:
            nano_listener_task.cancel()
        if nano_ws:
            await nano_ws.close()
        await nano_session.close()
        await http_session.close()


# ==========================================
# REST API & OTA
# ==========================================
@app.get("/api/devices")
async def api_get_devices(username: str = Depends(verify_auth)):
    return [
        {
            "session_id": sid,
            "mac": st["mac"],
            "status": st["status"],
            "last_text": st["last_text"],
        }
        for sid, st in session_states.items()
    ]


@app.post("/mcp/{session_id}")
async def execute_mcp(
    session_id: str, req: Request, username: str = Depends(verify_auth)
):
    try:
        data = await req.json()
        req_id = data.get("id")
        if session_id == "latest" and active_sessions:
            session_id = list(active_sessions.keys())[-1]
        if session_id in active_sessions:
            ws = active_sessions[session_id]

            # Authorize MCP command
            state = session_states.get(session_id)
            if state and "mac" in state:
                mac = normalize_mac(state["mac"])
                db = load_db()
                device_config = db.get(mac, {})
                owner = device_config.get("owner")

                # Check authorization
                if username != ADMIN_USERNAME:
                    if not owner or owner != username:
                        return {"error": "Forbidden: Not device owner"}

            if ws.client_state.name == "DISCONNECTED":
                active_sessions.pop(session_id, None)
                return {"error": "Device disconnected"}
            if req_id is not None:
                internal_id = int(time.time() * 1000)
                data["id"] = internal_id
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                mcp_futures[internal_id] = future
                await ws.send_json(
                    {"session_id": session_id, "type": "mcp", "payload": data}
                )
                try:
                    return await asyncio.wait_for(future, timeout=10.0)
                except asyncio.TimeoutError:
                    mcp_futures.pop(internal_id, None)
                    return {"error": "Timeout"}
            else:
                await ws.send_json(
                    {"session_id": session_id, "type": "mcp", "payload": data}
                )
                return {"status": "sent"}
        return {"error": "Offline"}
    except asyncio.CancelledError:
        logger.warning("⏰ [MCP] execute_mcp cancelled (device disconnected)")
        return {"error": "Cancelled"}
    except Exception as e:
        logger.error(f"❌ [MCP] execute_mcp error: {e}")
        return {"error": str(e)}


@app.post("/api/tts")
async def api_tts(req: Request, username: str = Depends(verify_auth)):
    data = await req.json()
    session_id = data.get("session_id", "latest")
    text = data.get("text", "").strip()
    if not text:
        return {"error": "Missing text"}
    if session_id == "latest" and active_sessions:
        session_id = list(active_sessions.keys())[-1]
    if session_id not in active_sessions:
        return {"error": "Offline"}
    state = session_states.get(session_id)
    if not state:
        return {"error": "No state"}

    if "mac" in state:
        mac = normalize_mac(state["mac"])
        db = load_db()
        device_config = db.get(mac, {})
        owner = device_config.get("owner")
        if username != ADMIN_USERNAME:
            if not owner or owner != username:
                return {"error": "Forbidden: Not device owner"}

    device_ws = active_sessions[session_id]

    if state.get("status") == "PLAYING":
        return {"error": "Busy"}

    state["last_activity"] = time.time()

    try:
        await send_mcp_cmd(
            device_ws,
            session_id,
            "self.screen.set_brightness",
            {"brightness": 100},
        )
    except Exception:
        pass

    async def _tts_with_cleanup():
        await generate_and_stream_tts(text, device_ws, session_id, state)
        state["last_activity"] = time.time()

    create_tracked_task(_tts_with_cleanup(), state)
    return {"status": "sent"}


_firmware_meta_cache = None


def load_firmware_meta() -> dict:
    global _firmware_meta_cache
    if _firmware_meta_cache is not None:
        return _firmware_meta_cache
    try:
        with open(FIRMWARE_META) as f:
            meta = json.load(f)
            fpath = os.path.join(FIRMWARE_DIR, meta.get("filename", ""))
            if not os.path.isfile(fpath):
                _firmware_meta_cache = {"version": "", "filename": "", "timestamp": 0}
            else:
                _firmware_meta_cache = meta
            return _firmware_meta_cache
    except Exception:
        _firmware_meta_cache = {"version": "", "filename": "", "timestamp": 0}
        return _firmware_meta_cache


def save_firmware_meta(version: str, filename: str):
    global _firmware_meta_cache
    meta = {
        "version": version,
        "filename": filename,
        "timestamp": int(time.time() * 1000),
    }
    with open(FIRMWARE_META, "w") as f:
        json.dump(meta, f)
    _firmware_meta_cache = meta
    return meta


@app.post("/api/firmware/upload")
async def firmware_upload(
    file: UploadFile = File(...),
    version: str = Form(""),
    username: str = Depends(verify_auth),
):
    if version and not re.match(r"^[a-zA-Z0-9.\-_]+$", version):
        raise HTTPException(400, "Invalid version format")
    if not file.filename or not file.filename.endswith(".bin"):
        raise HTTPException(400, "Only .bin files accepted")
    fname = f"firmware_v{version}.bin" if version else file.filename
    fname = os.path.basename(fname)
    if not re.match(r"^[a-zA-Z0-9_.-]+$", fname):
        raise HTTPException(400, "Invalid filename")
    fpath = os.path.join(FIRMWARE_DIR, fname)

    def write_sync_chunked(path, file_obj, max_size):
        size = 0
        with open(path, "wb") as f:
            while True:
                chunk = file_obj.read(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_size:
                    break
                f.write(chunk)
        if size > max_size:
            os.remove(path)
            raise ValueError("Firmware file too large")

    try:
        await asyncio.to_thread(write_sync_chunked, fpath, file.file, MAX_FIRMWARE_SIZE)
    except ValueError as e:
        raise HTTPException(413, str(e))

    meta = save_firmware_meta(version, fname)
    return {"status": "ok", "version": meta["version"]}


@app.get("/api/firmware")
async def firmware_info(username: str = Depends(verify_auth)):
    meta = load_firmware_meta()
    return meta


@app.api_route("/ota", methods=["GET", "POST"])
async def ota_handler(req: Request, username: str = Depends(verify_auth)):
    db = load_db()
    meta = load_firmware_meta()
    has_update = bool(meta.get("version"))
    url = (
        f"http://{req.url.hostname}:18792/firmware/{meta['filename']}"
        if has_update
        else ""
    )
    return {
        "server_time": {
            "timestamp": int(time.time() * 1000),
            "timeZone": "Europe/Moscow",
            "timezone_offset": 180,
        },
        "protocol": "websocket",
        "websocket": {
            "url": f"ws://{req.url.hostname}:18792/",
            "access_token": NANOBOT_TOKEN,
        },
        "firmware": {
            "has_update": has_update,
            "version": meta.get("version", ""),
            "url": url,
        },
    }


# ==========================================
# DEVICE CONFIG CRUD (Web UI)
# ==========================================
from pydantic import BaseModel


class DeviceCreate(BaseModel):
    mac: str
    friendly_name: str = ""
    ws_url: str = ""
    allowed: bool = True
    owner: str | None = None


class DeviceUpdate(BaseModel):
    friendly_name: str | None = None
    ws_url: str | None = None
    allowed: bool | None = None
    owner: str | None = None


def normalize_mac(mac: str) -> str:
    return mac.strip().upper()


def device_online_status(mac: str) -> str:
    if not mac:
        return "offline"
    for st in session_states.values():
        if st.get("mac", "").lower() == mac.lower():
            return st["status"]
    return "offline"


def device_list_with_status() -> list[dict]:
    db = load_db()
    result = []
    mac_to_status = {}
    for st in session_states.values():
        if "mac" in st:
            mac_to_status[st["mac"].lower()] = st["status"]

    for mac, cfg in db.items():
        status = mac_to_status.get(mac.lower(), "offline")
        entry = {"mac": mac, **cfg, "status": status}
        result.append(entry)
    result.sort(key=lambda x: (x["status"] == "offline", x["mac"]))
    return result


@app.get("/api/devices/config")
async def api_get_device_config(username: str = Depends(verify_auth)):
    return device_list_with_status()


@app.post("/api/devices/config")
async def api_create_device(body: DeviceCreate, username: str = Depends(verify_auth)):
    mac = normalize_mac(body.mac)
    if not mac:
        raise HTTPException(400, "MAC address required")
    db = load_db()
    if mac in db:
        raise HTTPException(409, "Device already exists")
    db[mac] = {
        "friendly_name": body.friendly_name,
        "ws_url": body.ws_url,
        "allowed": body.allowed,
        "owner": body.owner,
    }
    await save_db(db)
    return {"mac": mac, **db[mac], "status": "offline"}


@app.put("/api/devices/config/{mac}")
async def api_update_device(
    mac: str, body: DeviceUpdate, username: str = Depends(verify_auth)
):
    normalized = normalize_mac(mac)
    db = load_db()
    if normalized not in db:
        raise HTTPException(404, "Device not found")
    entry = db[normalized]
    if body.friendly_name is not None:
        entry["friendly_name"] = body.friendly_name
    if body.ws_url is not None:
        entry["ws_url"] = body.ws_url
    if body.allowed is not None:
        entry["allowed"] = body.allowed
    if body.owner is not None:
        entry["owner"] = body.owner
    await save_db(db)
    return {"mac": normalized, **entry, "status": device_online_status(normalized)}


@app.delete("/api/devices/config/{mac}")
async def api_delete_device(mac: str, username: str = Depends(verify_auth)):
    normalized = normalize_mac(mac)
    db = load_db()
    if normalized not in db:
        raise HTTPException(404, "Device not found")
    del db[normalized]
    await save_db(db)
    return {"status": "deleted", "mac": normalized}


# ── Camera sessions (go2rtc WebRTC) ──────────────────────────────────────

_camera_sessions: list[CameraSession] = []


async def start_camera_sessions():
    global _camera_sessions
    if os.getenv("DISABLE_CAMERAS", "").lower() in ("1", "true", "yes"):
        logger.info("📷 Cameras disabled via DISABLE_CAMERAS")
        return
    raw = os.getenv("CAMERA_STREAMS", "")
    streams = [s.strip() for s in raw.split(",") if s.strip()]
    if not streams:
        logger.info("📷 No CAMERA_STREAMS configured")
        return

    logger.info(f"📷 Starting camera sessions: {streams}")
    http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    go2rtc_host = os.getenv("GO2RTC_HOST", "192.168.22.102")
    go2rtc_port = int(os.getenv("GO2RTC_PORT", "1984"))
    # Wake-word switch: WAKE_WORD_MODEL points at any openWakeWord-format
    # head (built-in pretrained names like 'hey_jarvis' resolve inside the
    # package); WAKE_WORD is the spoken phrase used by the command stripper.
    wake_model_env = os.getenv(
        "WAKE_WORD_MODEL",
        # openwakeword.com library model, Creator #7074 (Classic V3):
        # recall 55.6%, measured on our audio — positives 0.64, clean
        # negatives 0.001-0.05 (old custom head: 0.6 vs 0.23 on TV noise).
        "config/computer_20260706_130638.onnx",
    )
    if os.path.sep not in wake_model_env:
        # Bare built-in name -> resolve inside the openwakeword package.
        # Paths containing a separator are filesystem paths used as-is;
        # joining them against the package dir produced nonsense like
        # <pkg>/models/config/computer.onnx (22:51 trace).
        import glob as _glob
        import openwakeword
        res_dir = os.path.join(
            os.path.dirname(openwakeword.__file__), "resources", "models"
        )
        base = wake_model_env if wake_model_env.endswith(".onnx") else wake_model_env + ".onnx"
        hits = sorted(_glob.glob(os.path.join(res_dir, base)))
        if hits:
            wake_model_env = hits[0]
    wake_word_env = os.getenv("WAKE_WORD", "компьютер")

    for name in streams:
        try:
            # Per-room model: the kitchen mic clips hard at close range
            # (ADC rail 32767) and the library model scores clipped audio
            # 0.001 — only the custom head tolerates it (0.6-0.9 there).
            # Clean-mic rooms use the library model (noise <=0.054).
            per_stream_default = (
                "config/computer.onnx" if name == "kitchen" else wake_model_env
            )
            wm = (
                os.getenv(f"WAKE_WORD_MODEL_{name.upper()}")
                or per_stream_default
            )
            config = CameraConfig(
                stream_name=name,
                wakeword_model_path=wm,
                wake_keyword=wake_word_env,
                go2rtc_host=go2rtc_host,
                go2rtc_port=go2rtc_port,
                # Source RTSP URL used by the self-healer to re-register the
                # stream when go2rtc loses its audio track. Per-stream env wins:
                #   GO2RTC_SOURCE_URL_CORRIDOR=rtsp://user:pass@cam/stream=0#backchannel=1
                # fallback: GO2RTC_SOURCE_URL for all streams. Without either,
                # the healer tries to reuse whatever URL go2rtc still lists.
                go2rtc_source_url=(
                    os.getenv(f"GO2RTC_SOURCE_URL_{name.upper()}")
                    or os.getenv("GO2RTC_SOURCE_URL", "")
                ),
                # Consecutive ffmpeg stalls (each ~20s read timeout) before the
                # healer re-registers the stream in go2rtc.
                heal_stalls=int(os.getenv("GO2RTC_HEAL_STALLS", "3")),
                http_session=http,
                nanobot_url=NANOBOT_WS_URL,
                nanobot_token=NANOBOT_TOKEN,
                tts_url=TTS_URL,
                tts_api_key=TTS_API_KEY,
                speaker_id_url=SPEAKER_ID_URL,
                vad=VadEngine(
                    energy_fallback=True,
                    energy_threshold=0.005,
                    rms_noise_floor=0.005,
                    rms_alpha=0.0,
                    onnx_gain=8.0,
                ),
            )
            # Выбор LLM-бэкенда
            if LLM_BACKEND == 'hermes':
                llm_backend = backends.HermesBackend(HERMES_API_URL, HERMES_API_KEY)
                logger.info('Using Hermes backend at ' + HERMES_API_URL)
            elif LLM_BACKEND == 'cascade':
                llm_backend = backends.CascadeBackend(ROUTER_URL, ack_delay=ROUTER_ACK_DELAY)
                logger.info('Using Cascade router at ' + ROUTER_URL)
            else:
                llm_backend = backends.NanobotBackend(NANOBOT_WS_URL, NANOBOT_TOKEN, NANOBOT_SESSION_SALT)
                logger.info('Using Nanobot backend at ' + NANOBOT_WS_URL)
            session = CameraSession(config=config, backend=llm_backend)
            _camera_sessions.append(session)
            await session.start()
            logger.info(f"📷 Camera session started: {name}")
        except Exception as e:
            logger.error(f"📷 Camera session {name} failed: {e}")


@app.post("/api/camera/tts")
async def api_camera_tts(req: Request, username: str = Depends(verify_auth)):
    data = await req.json()
    text = data.get("text", "").strip()
    name = data.get("name", "")
    if not text:
        return {"error": "Missing text"}
    target = [s for s in _camera_sessions if not name or s.stream_name == name]
    if not target:
        return {"error": "Camera not found"}
    for s in target:
        logger.info(f"Dispatching TTS to {s.stream_name}: text={text!r}")
        try:
            t = asyncio.create_task(s._speak(text, text))
            t.add_done_callback(
                lambda fut: logger.info(f"TTS task done, exc={fut.exception()}")
            )
        except Exception as e:
            logger.error(f"TTS create_task failed: {e}", exc_info=True)
    return {"status": "sent", "cameras": [s.stream_name for s in target]}


async def stop_camera_sessions():
    global _camera_sessions
    for s in _camera_sessions:
        await s.stop()
    _camera_sessions.clear()


@app.on_event("startup")
async def on_startup():
    await start_camera_sessions()


@app.on_event("shutdown")
async def on_shutdown():
    await stop_camera_sessions()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=18792,
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )
