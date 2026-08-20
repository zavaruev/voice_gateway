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
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
NANOBOT_WS_URL = os.getenv("NANOBOT_WS_URL", "ws://nanobot:8765/").rstrip("/")
NANOBOT_TOKEN = os.getenv("NANOBOT_TOKEN", "")
NANOBOT_SESSION_SALT = os.getenv("NANOBOT_SESSION_SALT", "")
SPEAKER_ID_URL = os.getenv("SPEAKER_ID_URL", "http://192.168.22.102:8001/identify")

WHISPER_URL = os.getenv(
    "WHISPER_URL", "http://192.168.22.111:8000/v1/audio/transcriptions"
)

TTS_URL = os.getenv("TTS_URL", "http://edge_tts:5050/v1/audio/speech")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "ru-RU-SvetlanaNeural")
TTS_API_KEY = os.getenv("TTS_API_KEY", "")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

if ADMIN_PASSWORD:
    if len(ADMIN_PASSWORD) < 8:
        raise ValueError(
            "ADMIN_PASSWORD must be at least 8 characters long for security reasons."
        )
    if ADMIN_PASSWORD == ADMIN_USERNAME:
        raise ValueError("ADMIN_PASSWORD cannot be the same as ADMIN_USERNAME.")

LOG_TRANSCRIPTIONS = os.getenv("LOG_TRANSCRIPTIONS", "false").lower() == "true"

DB_FILE = "/app/config/devices.json"
VAD_SILENCE_FRAMES = int(os.getenv("VAD_SILENCE_FRAMES", 8))
MAX_FIRMWARE_SIZE = int(os.getenv("MAX_FIRMWARE_SIZE", 10 * 1024 * 1024))
WATCHDOG_TIMEOUT = int(os.getenv("WATCHDOG_TIMEOUT", 30))
STANDBY_TIMEOUT_QUESTION = int(os.getenv("STANDBY_TIMEOUT_QUESTION", 30))
STANDBY_TIMEOUT_STATEMENT = int(os.getenv("STANDBY_TIMEOUT_STATEMENT", 10))
CHAT_ID_TTL = int(
    os.getenv("CHAT_ID_TTL", 604800)
)  # 7-day sliding window — Nanobot context lives a week
THINKING_SOUND_PATH = os.getenv("THINKING_SOUND_PATH", "")

ENERGY_THRESHOLD = float(os.getenv("ENERGY_THRESHOLD", "0.002"))
MIN_SPEECH_RATIO = float(os.getenv("MIN_SPEECH_RATIO", "0.12"))
VAD_ADAPTIVE = os.getenv("VAD_ADAPTIVE", "true").lower() == "true"

from audio_utils import pack_ogg, is_valid_text

HOLD_PHRASES = {
    "подожди",
    "мomento",
    "секундочку",
    "подожди-ка",
    "один момент",
    "мomento",
}

HAS_QUESTION_RE = re.compile(r"[?？]\s*$")
SENTENCE_END_RE = re.compile(r"[.!?…](?:\s|$)|[\n]")
HAS_QUESTION_WORDS_RE = re.compile(
    r"\b(что|как|где|когда|почему|зачем|сколько|кто|какой|какая|какое|какие|чей|чья|чьё|чьи|куда|откуда|уточни|расскажи|напомни|объясни|повтори|скажи|покажи|подожди|помоги|ответь|напиши|сделай|включи|выключи|открой|закрой|дай|можешь|не знаю|не понимаю)\b",
    re.IGNORECASE,
)

SPEAKER_NAME_FILE = "/app/config/speaker_names.json"


def load_speaker_names() -> dict:
    try:
        with open(SPEAKER_NAME_FILE) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading speaker names: {e}")
        return {}


SPEAKER_NAME_MAP = load_speaker_names()

CHAT_ID_CACHE = {}
CHAT_ID_CACHE_FILE = "/app/config/chat_id_cache.json"


def load_chat_id_cache():
    """Loads chat_id cache from disk (survives container rebuild)."""
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
    """Persists chat_id cache to disk."""
    if cache_data is None:
        # Shallow copy to avoid "dictionary changed size during iteration" in bg thread
        cache_data = dict(CHAT_ID_CACHE)
    try:
        with open(CHAT_ID_CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save chat_id cache: {e}")


def get_cached_chat_id(mac: str) -> str | None:
    # Adding a small comment to ensure the tests verify the function in the file
    entry = CHAT_ID_CACHE.get(mac.lower())
    if entry and time.time() - entry["ts"] < CHAT_ID_TTL:
        return entry["chat_id"]
    return None


_chat_id_last_save = 0.0


def set_cached_chat_id(mac: str, chat_id: str):
    global _chat_id_last_save
    CHAT_ID_CACHE[mac.lower()] = {"chat_id": chat_id, "ts": time.time()}
    now = time.time()
    if now - _chat_id_last_save > 5.0:
        _chat_id_last_save = now
        cache_copy = dict(CHAT_ID_CACHE)
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, save_chat_id_cache, cache_copy)
        except RuntimeError:
            save_chat_id_cache(cache_copy)


def make_chat_id(mac: str) -> str:
    """Deterministic chat_id from MAC address in UUID format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx).
    Nanobot reuses the session across reconnects."""
    h = hashlib.sha256(f"{mac.lower()}{NANOBOT_SESSION_SALT}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


load_chat_id_cache()  # Load on startup

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

security = HTTPBasic(auto_error=False)


def verify_auth(
    request: Request, credentials: HTTPBasicCredentials | None = Depends(security)
):
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
    is_user_ok = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    is_pass_ok = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (is_user_ok and is_pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


FIRMWARE_DIR = "/app/config/firmware"
os.makedirs(FIRMWARE_DIR, exist_ok=True)
FIRMWARE_META = "/app/config/firmware.json"
app.mount("/firmware", StaticFiles(directory=FIRMWARE_DIR), name="firmware")

templates = Jinja2Templates(directory="templates")


# ==========================================
# VAD ENGINE (Voice Activity Detection)
# ==========================================
class VadEngine:
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
        logger.info("Loading Silero VAD (ONNX) model...")
        opts = ort.SessionOptions()
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
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self.buffer = np.array([], dtype=np.float32)

    def _run_onnx(
        self, chunk: np.ndarray, state: np.ndarray, context: np.ndarray
    ) -> tuple:
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
                energy_thresh = max(self.energy_threshold, self.rms_noise_floor * 1.2)
                if rms > energy_thresh:
                    speech_energy = True

            speech_detected = speech_onnx or speech_energy

            self.rms_noise_floor = (
                1 - self.rms_alpha
            ) * self.rms_noise_floor + self.rms_alpha * rms

            return speech_detected, rms
        except Exception as e:
            logger.error(f"❌ VAD Error: {e}")
            return False, 0.0


# ==========================================
# UTILS & AUDIO PACKING
# ==========================================
_DB_CACHE = None


def load_db() -> dict:
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
    global _DB_CACHE
    _DB_CACHE = db
    await asyncio.to_thread(_save_db_sync, db)


def _save_db_sync(db: dict):
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=4)


# ==========================================
# HARDWARE CONTROL (MCP Tools)
# ==========================================
active_sessions = {}
session_states = {}
mcp_futures = {}

# Speaker ID lock: prevents duplicate processing when two devices hear the same speaker
_active_speaker_lock: dict | None = None  # {"uid": str, "sid": str, "expires": float}


def _clean_stale_futures():
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
    if req_id is None:
        req_id = int(time.time() * 1000)

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
    tc = d.get("tool_call", {})
    tool_name = tc.get("name", "")
    arguments = tc.get("arguments", {})
    tool_call_id = tc.get("id", "")
    logger.info(f"🔧 [DeviceTool] AI calling ESP32 tool: {tool_name}({arguments})")
    req_id = int(time.time() * 1000)
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    _clean_stale_futures()
    if len(mcp_futures) > 200:
        oldest = min(mcp_futures.keys())
        old_fut = mcp_futures.pop(oldest, None)
        if old_fut and not old_fut.done():
            old_fut.cancel()
    mcp_futures[req_id] = future
    try:
        await send_mcp_cmd(device_ws, state["sid"], tool_name, arguments, req_id)
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
    task = asyncio.create_task(coro, name=name)
    state["tasks"].add(task)
    task.add_done_callback(lambda _: state["tasks"].discard(task))
    return task


async def activity_monitor_task(device_ws: WebSocket, state: dict):
    """Monitors idle: dims screen but does NOT close the connection.
    Persistent mode — WS/context lives while ESP32 is on."""
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
                            f"🔇 [Diag] Wake timeout — no audio received in {int(time_idle)}s after listen:start"
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
    Keeps WS open — MCP tools (temperature monitoring) continue working."""
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


async def watchdog_timeout(device_ws: WebSocket, state: dict):
    logger.warning("⏱ [Watchdog] Upstream AI timed out.")
    state["watchdog_fired"] = True
    state["status"] = "SPEAKING"
    try:
        await generate_and_stream_tts(
            "Прости, я затупила. Повтори пожалуйста.", device_ws, state["sid"], state
        )
    except Exception as e:
        logger.error(f"❌ [Watchdog] TTS failed: {e}")

    logger.info("🎤 [Watchdog] Keeping mic open for repeat.")
    state.update(
        {"status": "LISTENING", "frames": [], "silence": 0, "has_speech": False}
    )
    state["last_activity"] = time.time()  # Reset timer
    state["vad"].reset()
    state["tts_cooldown_until"] = time.time() + 0.3  # Short cooldown after apology
    await send_mcp_cmd(
        device_ws, state["sid"], "self.screen.set_brightness", {"brightness": 100}
    )


# ==========================================
# ASYNC PIPELINE (STT & SPEAKER ID)
# ==========================================
async def fetch_speaker_id(audio: bytes, sess: aiohttp.ClientSession) -> str:
    try:
        form = aiohttp.FormData()
        form.add_field("file", audio, filename="audio.ogg", content_type="audio/ogg")
        async with sess.post(SPEAKER_ID_URL, data=form, timeout=10) as r:
            if r.status == 200:
                json_resp = await r.json()
                uid, conf = json_resp.get("user_id", "unknown"), json_resp.get(
                    "confidence", 0.0
                )
                if uid != "unknown" and conf > 0.1:
                    logger.info(f"✅ [SpeakerID] Recognized: {uid} ({conf:.2f})")
                    return uid
                else:
                    logger.debug(f"👤 [SpeakerID] Rejected: {uid} ({conf:.2f})")
    except Exception as e:
        logger.warning(f"⚠️ [SpeakerID] Request failed: {e}")
    return "unknown"


async def fetch_transcription(audio: bytes, sess: aiohttp.ClientSession) -> str:
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
    """Calculate RMS energy from PCM16 audio data."""
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
    """Decode Opus frames to PCM and return (combined_pcm, rms_list, vad_results)."""
    all_pcm = bytearray()
    rms_list = []
    vad_results = []

    for frame in frames:
        try:
            pcm = decoder.decode(frame, 960)
            all_pcm.extend(pcm)

            rms = calculate_rms(pcm)
            rms_list.append(rms)

            is_sp, _ = await vad.is_speech(pcm, rms)
            vad_results.append(is_sp)
        except Exception:
            rms_list.append(0.0)
            vad_results.append(False)

    return bytes(all_pcm), rms_list, vad_results


async def _process_audio_metrics_and_gates(
    frames: list, state: dict, decoder: opuslib.Decoder = None
) -> tuple[bool, float, float, opuslib.Decoder]:
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
    state["_rejected_count"] = 0
    _t_transcribed = time.time()
    state["_stt_time"] = _t_transcribed
    if LOG_TRANSCRIPTIONS:
        logger.info(f"🗣 [User: {uid}] Transcribed: '{txt}'")
    else:
        logger.info(f"🗣 [User: {uid}] Transcribed: [REDACTED] ({len(txt)} chars)")
    state["last_activity"] = _t_transcribed
    state["last_text"] = txt

    if state.get("nanobot_chat_id"):
        set_cached_chat_id(state["mac"].lower(), state["nanobot_chat_id"])

    if any(p in txt.lower() for p in HOLD_PHRASES):
        logger.info(f"🛑 [Hold] Detected hold phrase, extending listening")
        state["last_activity"] = time.time()

    try:
        await device_ws.send_json(
            {"type": "stt", "text": txt, "session_id": state["sid"]}
        )
    except Exception:
        return

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


async def _handle_rejected_transcription(
    txt: str,
    uid: str,
    state: dict,
    device_ws: WebSocket,
    avg_rms: float,
    speech_ratio: float,
    frames_len: int,
):
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
    _t0 = time.time()
    if time.time() < camera_client.GLOBAL_TTS_UNTIL:
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
    """Synthesize text to MP3 via the TTS API (network-bound, no streaming)."""
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
    """Stream pre-synthesized MP3 to the device. Returns True on success."""
    try:
        if not state or not state.get("tts_started"):
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

        camera_client.GLOBAL_TTS_UNTIL = time.time() + len(pcm_data) / (16000 * 2) + 3.0

        enc = state.setdefault("tts_encoder", opuslib.Encoder(16000, 1, "voip"))
        frame_size = 960
        chunk_size = frame_size * 2
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


async def generate_and_stream_tts(
    text: str, device_ws: WebSocket, session_id: str, state: dict = None
):
    """Compatibility wrapper: synthesize then stream a single utterance."""
    if not state:
        logger.error("❌ [TTS] generate_and_stream_tts called without state")
        return
    mp3_data = await synthesize_tts_mp3(text, state)
    if mp3_data is not None:
        await stream_tts_pcm(mp3_data, device_ws, session_id, state)


async def trigger_emotion(
    emotion: str, device_ws: WebSocket, session_id: str, logger=logger
):
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
EMOTION_REGEX = re.compile(r"\[([a-zA-Z0-9_]+)\]")

class NanobotResponseHandler:
    def __init__(self, device_ws, state):
        self.device_ws = device_ws
        self.state = state
        self.buffer = []
        self.full_response_text = []
        self.timer = None
        self.emotion_regex = EMOTION_REGEX
        self.is_flushing = False
        self._first_chunk_time = None
        self._last_chunk_time = None
        self._chunk_count = 0
        self.tts_audio_queue: asyncio.Queue = asyncio.Queue()
        self.tts_player_task = None
        self._synth_tasks: set = set()

    async def _tts_player(self):
        """Play pre-synthesized segments back-to-back as one continuous
        stream (no stop/start between segments, so audio is gapless)."""
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
        if self.state.get("tts_started"):
            self.state["tts_started"] = False
            try:
                await self.device_ws.send_json(
                    {"type": "tts", "state": "stop", "session_id": self.state["sid"]}
                )
            except Exception:
                pass

    async def _enqueue_tts(self, text: str):
        """Synthesize in background; player picks it up when ready."""
        mp3_data = await synthesize_tts_mp3(text, self.state)
        if mp3_data is not None:
            await self.tts_audio_queue.put(mp3_data)

    def _ensure_tts_player(self):
        if self.tts_player_task is None or self.tts_player_task.done():
            self.tts_player_task = create_tracked_task(self._tts_player(), self.state)

    async def _await_tts_drained(self):
        """Wait until all background synthesis and playback has finished,
        then close the continuous TTS stream with a single stop message."""
        while self._synth_tasks:
            await asyncio.sleep(0.05)
        try:
            await asyncio.wait_for(self.tts_audio_queue.join(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("❌ [TTS] Timed out waiting for playback to finish")
        await self._send_tts_stop()

    def reset_timing(self):
        self._first_chunk_time = None
        self._last_chunk_time = None
        self._chunk_count = 0

    async def handle_chunk(self, chunk: str):
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

    async def flush(self):
        if self.is_flushing or not "".join(self.buffer).strip():
            return
        self.is_flushing = True
        self.state["status"] = "SPEAKING"

        text = "".join(self.buffer)
        self.buffer = []

        # If emotion tag is still incomplete, wait for next chunk
        if text.count("[") > text.count("]"):
            self.buffer = [text]
            self.timer = asyncio.get_event_loop().call_later(
                0.5, lambda: create_tracked_task(self.flush(), self.state)
            )
            self.is_flushing = False
            return

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

        emotions = self.emotion_regex.findall(text)
        for emotion in emotions:
            create_tracked_task(
                trigger_emotion(emotion, self.device_ws, self.state["sid"]), self.state
            )

        clean_text = self.emotion_regex.sub("", text).strip()

        # [disconnect] command — user said "disconnect", Nanobot confirmed
        if "[disconnect]" in clean_text:
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
            self.is_flushing = False
            return

        if clean_text:
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
            task = create_tracked_task(self._enqueue_tts(clean_text), self.state)
            self._synth_tasks.add(task)
            task.add_done_callback(self._synth_tasks.discard)

        # End of flush: if the LLM is still streaming, flush the remaining
        # content shortly after; only finalize (dialogue mode check, return
        # to standby) once the whole response has been produced.
        self.is_flushing = False
        if "".join(self.buffer).strip():
            self.timer = asyncio.get_event_loop().call_later(
                0.15, lambda: create_tracked_task(self.flush(), self.state)
            )
            return

        await self._await_tts_drained()

        clean_for_check = "".join(self.full_response_text).strip().lower()
        has_question = (
            HAS_QUESTION_RE.search(clean_for_check) is not None
            or "повторите пожалуйста" in clean_for_check
            or HAS_QUESTION_WORDS_RE.search(clean_for_check) is not None
        )
        self.state["last_ai_had_question"] = has_question

        if has_question:
            self.state.update(
                {
                    "status": "LISTENING",
                    "frames": [],
                    "silence": 0,
                    "has_speech": False,
                }
            )
            self.state["tts_cooldown_until"] = time.time() + 0.2
            logger.info(f"💡 [Brightness] Dialogue mode — screen 100%")
            await send_mcp_cmd(
                self.device_ws,
                self.state["sid"],
                "self.screen.set_brightness",
                {"brightness": 100},
            )
        else:
            self.state.update(
                {"status": "LISTENING", "frames": [], "silence": 0, "has_speech": False}
            )
            self.state["tts_cooldown_until"] = time.time() + 0.2
            if self.state.get("watchdog"):
                self.state["watchdog"].cancel()
                self.state["watchdog"] = None
        self.state["last_activity"] = time.time()
        self.state["vad"].reset()
        self.full_response_text = []


# ==========================================
# WEB UI
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def web_index(req: Request, username: str = Depends(verify_auth)):
    return templates.TemplateResponse(req, "index.html")


# ==========================================
# MAIN FASTAPI WEBSOCKET GATEWAY
# ==========================================
async def listen_to_nanobot_task(
    device_ws: WebSocket, state: dict, nano_session: aiohttp.ClientSession
):
    nano_ws = state.get("nano_ws")
    handler = NanobotResponseHandler(device_ws, state)
    state["handler"] = handler
    while state["sid"] in session_states:
        if nano_ws is None or nano_ws.closed:
            try:
                mac_key = state["mac"].lower()
                det_chat_id = make_chat_id(mac_key)
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
                f"🛠 [MCP] Tools available: {len(state['available_tools'])} (managed by nanobot in v0.3.0)"
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
        if nano_listener_task is None:
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

    if token != NANOBOT_TOKEN:
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
    raw = os.getenv("CAMERA_STREAMS", "")
    streams = [s.strip() for s in raw.split(",") if s.strip()]
    if not streams:
        logger.info("📷 No CAMERA_STREAMS configured")
        return

    logger.info(f"📷 Starting camera sessions: {streams}")
    http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    go2rtc_host = os.getenv("GO2RTC_HOST", "192.168.22.102")
    go2rtc_port = int(os.getenv("GO2RTC_PORT", "1984"))

    for name in streams:
        try:
            config = CameraConfig(
                stream_name=name,
                go2rtc_host=go2rtc_host,
                go2rtc_port=go2rtc_port,
                http_session=http,
                nanobot_url=NANOBOT_WS_URL,
                nanobot_token=NANOBOT_TOKEN,
                tts_url=TTS_URL,
                tts_api_key=TTS_API_KEY,
                speaker_id_url=SPEAKER_ID_URL,
                vad=VadEngine(
                    energy_fallback=True,
                    energy_threshold=0.005,
                    onnx_threshold=0.02,
                    rms_noise_floor=0.005,
                    rms_alpha=0.0,
                    onnx_gain=8.0,
                ),
            )
            session = CameraSession(config=config)
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
