import asyncio
import json
import os
import time
import struct
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
from fastapi import (
    FastAPI,
    Request,
    Form,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    UploadFile,
    File,
    Depends,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
NANOBOT_WS_URL = os.getenv("NANOBOT_WS_URL", "ws://nanobot:8765/").rstrip("/")
SPEAKER_ID_URL = os.getenv("SPEAKER_ID_URL", "http://192.168.22.102:8001/identify")

WHISPER_URL = os.getenv(
    "WHISPER_URL", "http://192.168.22.111:8000/v1/audio/transcriptions"
)

TTS_URL = os.getenv("TTS_URL", "http://edge_tts:5050/v1/audio/speech")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "ru-RU-SvetlanaNeural")
TTS_API_KEY = os.getenv("TTS_API_KEY", "")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

DB_FILE = "/app/config/devices.json"
VAD_SILENCE_FRAMES = int(os.getenv("VAD_SILENCE_FRAMES", 8))
WATCHDOG_TIMEOUT = 15
STANDBY_TIMEOUT_QUESTION = int(os.getenv("STANDBY_TIMEOUT_QUESTION", 30))
STANDBY_TIMEOUT_STATEMENT = int(os.getenv("STANDBY_TIMEOUT_STATEMENT", 10))
CHAT_ID_TTL = int(
    os.getenv("CHAT_ID_TTL", 604800)
)  # 7-day sliding window — Nanobot context lives a week
THINKING_SOUND_PATH = os.getenv("THINKING_SOUND_PATH", "")

ENERGY_THRESHOLD = float(os.getenv("ENERGY_THRESHOLD", "0.008"))
MIN_SPEECH_RATIO = float(os.getenv("MIN_SPEECH_RATIO", "0.15"))
VAD_ADAPTIVE = os.getenv("VAD_ADAPTIVE", "true").lower() == "true"

WHISPER_HALLUCINATIONS = [
    "субтитры подогнал симон",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "аминь",
    "субтитры создавал",
    "редактор субтитров",
    "thank you",
    "thanks for watching",
    "so",
    "dimatorzok",
    "субтитры сделал",
    "dima torzok",
    "продолжение следует",
    "синкинг",
]

SINGLE_WORD_HALLUCINATIONS = {"о", "а", "и", "кх-кх", "ха-ха", "жизнь", "пьютер", "как"}

HOLD_PHRASES = {
    "подожди",
    "мomento",
    "секундочку",
    "подожди-ка",
    "один момент",
    "мomento",
}

HAS_QUESTION_RE = re.compile(r"[?？]\s*$")

SPEAKER_NAME_FILE = "/app/config/speaker_names.json"


def load_speaker_names() -> dict:
    try:
        with open(SPEAKER_NAME_FILE) as f:
            return json.load(f)
    except Exception:
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
    h = hashlib.sha256(mac.lower().encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def clear_expired_chat_ids():
    now = time.time()
    expired = [
        mac for mac, entry in CHAT_ID_CACHE.items() if now - entry["ts"] >= CHAT_ID_TTL
    ]
    for mac in expired:
        del CHAT_ID_CACHE[mac]
    save_chat_id_cache()


load_chat_id_cache()  # Load on startup

app = FastAPI()

security = HTTPBasic()


def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Authentication not configured",
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
    def __init__(self):
        logger.info("Loading Silero VAD (ONNX) model...")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            "silero_vad.onnx", sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.buffer = np.array([], dtype=np.float32)

        # Adaptive VAD threshold
        self.noise_floor = 0.02
        self.threshold = 0.15
        self.alpha = 0.01  # Adaptation speed
        self.vad_adaptive = VAD_ADAPTIVE

        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self.buffer = np.array([], dtype=np.float32)
        if self.vad_adaptive:
            self.noise_floor = 0.02
            self.threshold = 0.15

    def _run_onnx(self, chunk: np.ndarray, state: np.ndarray) -> tuple:
        return self.session.run(
            None,
            {
                "input": chunk[np.newaxis, :],
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
            self.buffer = np.concatenate((self.buffer, audio_float32))
            speech_detected = False
            current_threshold = self.threshold
            while len(self.buffer) >= 512:
                chunk = self.buffer[:512]
                self.buffer = self.buffer[512:]
                out, self._state = await asyncio.to_thread(
                    self._run_onnx, chunk, self._state
                )
                if out[0][0] > current_threshold:
                    speech_detected = True
            if rms > 0.02 and not speech_detected:
                speech_detected = True
            if self.vad_adaptive and not speech_detected:
                self.noise_floor = (
                    1 - self.alpha
                ) * self.noise_floor + self.alpha * rms
                self.threshold = max(0.15, self.noise_floor * 4)

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


def _build_crc_table():
    table = []
    for i in range(256):
        c = i << 24
        for _ in range(8):
            c = (c << 1) ^ 0x04C11DB7 if c & 0x80000000 else c << 1
        table.append(c & 0xFFFFFFFF)
    return table


_OGG_CRC_TABLE = _build_crc_table()


def pack_ogg(frames: list, sample_rate=16000) -> bytes:
    def ogg_crc(data: bytes) -> int:
        crc = 0
        for b in data:
            crc = ((crc << 8) & 0xFFFFFFFF) ^ _OGG_CRC_TABLE[((crc >> 24) ^ b) & 0xFF]
        return crc

    def page(idx: int, gran: int, ser: int, bos: bool, eos: bool, pkts: list) -> bytes:
        h = struct.pack(
            "<4sBBqIIIB",
            b"OggS",
            0,
            (2 if bos else 0) | (4 if eos else 0),
            gran,
            ser,
            idx,
            0,
            len(pkts),
        )
        p = h + bytearray([len(x) for x in pkts]) + b"".join(pkts)
        crc = ogg_crc(p)
        return p[:22] + struct.pack("<I", crc) + p[26:]

    ser = int(time.time()) & 0xFFFFFFFF
    res = page(
        0,
        0,
        ser,
        True,
        False,
        [struct.pack("<8sBBHIHB", b"OpusHead", 1, 1, 312, sample_rate, 0, 0)],
    )
    res += page(
        1,
        0,
        ser,
        False,
        False,
        [struct.pack("<8sI8sI", b"OpusTags", 8, b"VoiceGW ", 0)],
    )
    for i in range(0, len(frames), 50):
        c = frames[i : i + 50]
        res += page(
            2 + i // 50,
            (i + len(c)) * int(48000 * 0.06),
            ser,
            False,
            (i + 50 >= len(frames)),
            c,
        )
    return res


# ==========================================
# HARDWARE CONTROL (MCP Tools)
# ==========================================
active_sessions = {}
session_states = {}
mcp_futures = {}


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


def is_valid_text(txt: str) -> bool:
    clean = txt.strip(" .,?!-").lower()
    if not clean:
        return False

    if len(clean) >= 10:
        max_run, cur = 1, 1
        for i in range(1, len(clean)):
            cur = cur + 1 if clean[i] == clean[i - 1] else 1
            max_run = max(max_run, cur)
        if max_run / len(clean) > 0.5:
            return False

    words = clean.split()
    if len(words) == 1:
        if words[0] in SINGLE_WORD_HALLUCINATIONS:
            return False
        if len(words[0]) <= 2:
            return False

    for bad in WHISPER_HALLUCINATIONS:
        if bad in clean:
            return False
    return True


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


async def process_audio_and_send(
    frames: list, state: dict, device_ws: WebSocket, decoder: opuslib.Decoder = None
):
    if len(frames) < 25:
        logger.info(
            f"🔇 [Pipeline] Too few frames ({len(frames)}), likely noise — skipping STT"
        )
        state["status"] = "LISTENING"
        return
    try:
        # Decode frames for metrics and gates
        dec = decoder or opuslib.Decoder(16000, 1)
        _, rms_list, vad_results = await decode_opus_frames(frames, dec, state["vad"])

        # Debug metrics logging
        avg_rms = sum(rms_list) / len(rms_list) if rms_list else 0.0
        speech_frames = sum(1 for v in vad_results if v)
        speech_ratio = speech_frames / len(vad_results) if vad_results else 0.0

        logger.info(
            f"📊 [Metrics] RMS={avg_rms:.4f}, speech_ratio={speech_ratio:.2f}, frames={len(frames)}"
        )

        # Energy Gate: reject if overall RMS too low
        if avg_rms < ENERGY_THRESHOLD:
            logger.info(
                f"🔇 [Energy Gate] RMS={avg_rms:.4f} < {ENERGY_THRESHOLD}, skipping Whisper"
            )
            state["status"] = "LISTENING"
            return

        # Speech Ratio Gate: require minimum fraction of VAD-speech frames
        if speech_ratio < MIN_SPEECH_RATIO:
            logger.info(
                f"🔇 [Speech Ratio] {speech_ratio:.1%} < {MIN_SPEECH_RATIO}, skipping Whisper"
            )
            state["status"] = "LISTENING"
            return

        audio = pack_ogg(frames)
        logger.info(
            f"🎙 [Pipeline] Processing {len(audio)} bytes (rms={avg_rms:.4f}, speech_ratio={speech_ratio:.2f})..."
        )
        # Screen already lit by listen:start handler; no need to set brightness here
        create_tracked_task(trigger_emotion("thinking", device_ws, state["sid"]), state)
        sess = state["http_session"]
        uid_task = create_tracked_task(fetch_speaker_id(audio, sess), state)
        stt_task = create_tracked_task(fetch_transcription(audio, sess), state)
        uid, txt = await asyncio.gather(uid_task, stt_task)

        if is_valid_text(txt):
            logger.info(f"🗣 [User: {uid}] Transcribed: '{txt}'")
            state["last_activity"] = time.time()
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
                await nano_ws.send_json(payload)

                # Send tts:start early to prevent device auto-timeout during Nanobot wait
                try:
                    await device_ws.send_json(
                        {"type": "tts", "state": "start", "session_id": state["sid"]}
                    )
                except Exception:
                    return
                state["status"] = "SPEAKING"
                state["tts_started"] = True

                if state.get("watchdog"):
                    state["watchdog"].cancel()
                state["watchdog"] = asyncio.get_event_loop().call_later(
                    WATCHDOG_TIMEOUT,
                    lambda: create_tracked_task(
                        watchdog_timeout(device_ws, state), state
                    ),
                )
            else:
                logger.warning(
                    f"⚠️ [Pipeline] Nanobot not connected, falling back to direct TTS"
                )
                state["status"] = "SPEAKING"
                await generate_and_stream_tts(txt, device_ws, state["sid"], state)
                await reset_to_standby(device_ws, state)
        else:
            logger.warning(
                f"⚠ [Pipeline] Rejected transcription (text='{txt}') from user '{uid}'"
            )
            await reset_to_standby(device_ws, state)
    except Exception as e:
        logger.error(f"❌ Pipeline Error: {e}")
        await reset_to_standby(device_ws, state)


# ==========================================
# TTS & EMOTION
# ==========================================
async def generate_and_stream_tts(
    text: str, device_ws: WebSocket, session_id: str, state: dict = None
):
    logger.info(f"🔊 [TTS] Synthesizing: '{text}'")
    try:
        if not state or not state.get("tts_started"):
            try:
                await device_ws.send_json(
                    {"type": "tts", "state": "start", "session_id": session_id}
                )
            except Exception:
                return
            if state:
                state["tts_started"] = True

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
                mp3_data = await r.read()
                audio_seg = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
                audio_seg = (
                    audio_seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
                )
                pcm_data = audio_seg.raw_data

                enc = state.setdefault("tts_encoder", opuslib.Encoder(24000, 1, "voip"))
                frame_size = 1440
                chunk_size = frame_size * 2

                start_stream = time.perf_counter()
                next_chunk_time = start_stream

                for i in range(0, len(pcm_data), chunk_size):
                    chunk = pcm_data[i : i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk += b"\x00" * (chunk_size - len(chunk))
                    opus_frame = enc.encode(chunk, frame_size)

                    try:
                        await device_ws.send_bytes(opus_frame)
                    except Exception:
                        return

                    next_chunk_time += 0.06
                    sleep_duration = next_chunk_time - time.perf_counter()
                    if sleep_duration > 0:
                        await asyncio.sleep(sleep_duration)

                logger.info("✅ [TTS] Audio stream completed smoothly.")
                # Set TTS cooldown to prevent VAD triggering on our own output
                if state:
                    state["tts_cooldown_until"] = (
                        time.time() + 1.5
                    )  # 1.5s cooldown after TTS
            else:
                logger.error(f"❌ [TTS] API Error: {await r.text()}")
    except Exception as e:
        if "Cannot call" not in str(e):
            logger.error(f"❌ [TTS] Error: {e}")
    finally:
        if state:
            state["tts_started"] = False
        try:
            await device_ws.send_json(
                {"type": "tts", "state": "stop", "session_id": session_id}
            )
        except Exception:
            pass


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
class NanobotResponseHandler:
    def __init__(self, device_ws, state):
        self.device_ws = device_ws
        self.state = state
        self.buffer = ""
        self.full_response_text = ""
        self.timer = None
        self.emotion_regex = re.compile(r"\[([a-zA-Z0-9_]+)\]")
        self.is_flushing = False

    async def handle_chunk(self, chunk: str):
        if self.state.get("watchdog"):
            self.state["watchdog"].cancel()
            self.state["watchdog"] = None
        self.buffer += chunk
        if self.timer:
            self.timer.cancel()
        # Increase delay for longer responses to avoid premature flush
        delay = 2.0 if self.buffer.count("[") > self.buffer.count("]") else 1.5
        self.timer = asyncio.get_event_loop().call_later(
            delay, lambda: create_tracked_task(self.flush(), self.state)
        )

    async def flush(self):
        if self.is_flushing or not self.buffer.strip():
            return
        self.is_flushing = True
        self.state["status"] = "SPEAKING"

        text = self.buffer
        self.buffer = ""

        # If emotion tag is still incomplete, wait for next chunk
        if text.count("[") > text.count("]"):
            self.buffer = text
            self.timer = asyncio.get_event_loop().call_later(
                0.5, lambda: create_tracked_task(self.flush(), self.state)
            )
            self.is_flushing = False
            return

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
                await generate_and_stream_tts(
                    clean_text, self.device_ws, self.state["sid"], self.state
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
            logger.info(
                f"📝 [TTS Input] Sending to TTS: '{clean_text[:100]}...' (len={len(clean_text)})"
            )
            self.full_response_text += clean_text + " "
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

            await generate_and_stream_tts(
                clean_text, self.device_ws, self.state["sid"], self.state
            )

            clean_for_check = self.full_response_text.strip().lower()
            has_question = (HAS_QUESTION_RE.search(clean_for_check) is not None) or (
                "повторите пожалуйста" in clean_for_check
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
                self.state["tts_cooldown_until"] = 0
                logger.info(f"💡 [Brightness] Dialogue mode — screen 100%")
                await send_mcp_cmd(
                    self.device_ws,
                    self.state["sid"],
                    "self.screen.set_brightness",
                    {"brightness": 100},
                )
            else:
                self.state.update(
                    {"status": "IDLE", "frames": [], "silence": 0, "has_speech": False}
                )
                if self.state.get("watchdog"):
                    self.state["watchdog"].cancel()
                    self.state["watchdog"] = None
                logger.info(
                    f"💤 [Info] Statement — idle, screen stays lit for 10s then dims"
                )
            self.state["last_activity"] = time.time()
            self.state["vad"].reset()
            self.full_response_text = ""
        self.is_flushing = False


# ==========================================
# WEB UI
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def web_index(req: Request, username: str = Depends(verify_auth)):
    return templates.TemplateResponse(req, "index.html")


# ==========================================
# MAIN FASTAPI WEBSOCKET GATEWAY
# ==========================================
@app.websocket("/")
async def voice_ws(device_ws: WebSocket):
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
        "nanobot_chat_id": None,
        "last_text": "",
        "last_activity": time.time(),
        "available_tools": [],
        "tts_cooldown_until": 0.0,
        "vad": VadEngine(),
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
                "sample_rate": 24000,
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

    async def listen_to_nanobot():
        nonlocal nano_ws, nano_listener_task
        handler = NanobotResponseHandler(device_ws, state)
        while state["sid"] in session_states:
            if nano_ws is None or nano_ws.closed:
                try:
                    mac_key = state["mac"].lower()
                    det_chat_id = make_chat_id(mac_key)
                    auth_url = f"{NANOBOT_WS_URL}?token=token&chat_id={det_chat_id}"
                    nano_ws = await nano_session.ws_connect(auth_url)
                    state["nano_ws"] = nano_ws
                    handler = NanobotResponseHandler(device_ws, state)
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
                                    f"📤 [Nanobot] Feeding AI tool list: {len(state['available_tools'])} tools"
                                )
                                await nano_ws.send_json(
                                    {
                                        "type": "tools_update",
                                        "chat_id": state["nanobot_chat_id"],
                                        "tools": state["available_tools"],
                                    }
                                )
                        elif d.get("event") == "error":
                            logger.error(f"❌ Nanobot Error: {d.get('detail')}")
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
                            state["vad"].reset()
                            await reset_to_standby(device_ws, state)
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
                                logger.error(
                                    f"❌ Nanobot Error in text: {text_content}"
                                )
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
                                state["vad"].reset()
                                await reset_to_standby(device_ws, state)
                            else:
                                if state.get("watchdog_fired"):
                                    logger.info(
                                        f"⏱️ [Watchdog] Ignoring late Nanobot response"
                                    )
                                    continue
                                await handler.handle_chunk(text_content)
            except Exception as e:
                logger.warning(
                    f"🔁 [Nanobot] Connection lost ({e}), reconnecting in 5s..."
                )

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

                if d.get("type") == "mcp":
                    _clean_stale_futures()
                    payload = d.get("payload", {})
                    req_id = payload.get("id")

                    if (
                        req_id == 999
                        and "result" in payload
                        and "tools" in payload["result"]
                    ):
                        state["available_tools"] = payload["result"]["tools"]
                        tool_names = [t.get("name") for t in state["available_tools"]]
                        logger.info(f"🛠 [MCP] ESP32 returned tools: {tool_names}")
                        nano_ws = state.get("nano_ws")
                        if (
                            nano_ws
                            and not nano_ws.closed
                            and state.get("nanobot_chat_id")
                        ):
                            logger.info(
                                f"📤 [MCP] Forwarding {len(state['available_tools'])} tools to Nanobot"
                            )
                            try:
                                await nano_ws.send_json(
                                    {
                                        "type": "tools_update",
                                        "chat_id": state["nanobot_chat_id"],
                                        "tools": state["available_tools"],
                                    }
                                )
                            except Exception as e:
                                logger.error(
                                    f"❌ [MCP] Failed to forward tools to Nanobot: {e}"
                                )
                        continue

                    if req_id in mcp_futures and not mcp_futures[req_id].done():
                        mcp_futures[req_id].set_result(payload)
                    elif req_id is None and payload.get("method"):
                        nano_ws = state.get("nano_ws")
                        if (
                            nano_ws
                            and not nano_ws.closed
                            and state.get("nanobot_chat_id")
                        ):
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
                            listen_to_nanobot(), state
                        )

                elif d.get("type") == "listen" and d.get("state") == "start":
                    # Skip false wake word immediately after TTS (audio tail)
                    if time.time() < state.get("tts_cooldown_until", 0):
                        logger.info(
                            f"👂 [Wake] Ignoring listen:start during TTS cooldown (false wake)"
                        )
                        continue
                    logger.info(
                        f"👂 [Wake] listen:start — wake word detected, entering LISTENING"
                    )
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
                            process_audio_and_send(
                                frames_to_process, state, device_ws, dec
                            ),
                            state,
                        )

            # --- AUDIO BYTES ---
            byte_data = m.get("bytes")
            if byte_data:
                # In IDLE we don't listen — wait for listen:start from device (wake word)
                if state["status"] == "IDLE":
                    continue
                if state["status"] in ["PROCESSING", "SPEAKING"]:
                    continue
                # Skip audio during TTS cooldown to prevent self-triggering
                if time.time() < state.get("tts_cooldown_until", 0):
                    continue
                # Skip audio during post-wake cooldown (pip noise after beep)
                if time.time() < state.get("post_wake_cooldown_until", 0):
                    continue
                state["status"] = "LISTENING"

                f = (
                    byte_data
                    if state["version"] == 1
                    else (byte_data[16:] if state["version"] == 2 else byte_data[4:])
                )
                state["frames"].append(f)

                try:
                    pcm = dec.decode(f, 960)
                    # Pre-VAD RMS gate: skip VAD for near-silent frames
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
                            process_audio_and_send(
                                frames_to_process, state, device_ws, dec
                            ),
                            state,
                        )
                    else:
                        state["frames"], state["silence"] = [], 0
                        state["vad"].reset()

    except Exception as e:
        logger.error(f"Error in device websocket loop: {e}")
    finally:
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
    data = await req.json()
    req_id = data.get("id")
    if session_id == "latest" and active_sessions:
        session_id = list(active_sessions.keys())[-1]
    if session_id in active_sessions:
        if req_id is not None:
            future = asyncio.get_event_loop().create_future()
            mcp_futures[req_id] = future
            await active_sessions[session_id].send_json(
                {"session_id": session_id, "type": "mcp", "payload": data}
            )
            try:
                return await asyncio.wait_for(future, timeout=10.0)
            except asyncio.TimeoutError:
                del mcp_futures[req_id]
                return {"error": "Timeout"}
        else:
            await active_sessions[session_id].send_json(
                {"session_id": session_id, "type": "mcp", "payload": data}
            )
            return {"status": "sent"}
    return {"error": "Offline"}


_firmware_meta_cache = None


def load_firmware_meta() -> dict:
    global _firmware_meta_cache
    if _firmware_meta_cache is not None:
        return _firmware_meta_cache
    try:
        with open(FIRMWARE_META) as f:
            _firmware_meta_cache = json.load(f)
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
    if not file.filename or not file.filename.endswith(".bin"):
        raise HTTPException(400, "Only .bin files accepted")
    fname = f"firmware_v{version}.bin" if version else file.filename
    fname = os.path.basename(fname)
    fpath = os.path.join(FIRMWARE_DIR, fname)
    content = await file.read()

    def write_sync(path, data):
        with open(path, "wb") as f:
            f.write(data)

    await asyncio.to_thread(write_sync, fpath, content)
    meta = save_firmware_meta(version, fname)
    return {"status": "ok", "meta": meta}


@app.get("/api/firmware")
async def firmware_info(username: str = Depends(verify_auth)):
    meta = load_firmware_meta()
    return meta


@app.api_route("/ota", methods=["GET", "POST"])
async def ota_handler(req: Request):
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
            "access_token": "token",
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


class DeviceUpdate(BaseModel):
    friendly_name: str | None = None
    ws_url: str | None = None
    allowed: bool | None = None


def normalize_mac(mac: str) -> str:
    return mac.strip().upper()


def device_online_status(mac: str) -> str:
    for st in session_states.values():
        if st.get("mac", "").lower() == mac.lower():
            return st["status"]
    return "offline"


def device_list_with_status() -> list[dict]:
    db = load_db()
    result = []
    for mac, cfg in db.items():
        entry = {"mac": mac, **cfg, "status": device_online_status(mac)}
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=18792)
