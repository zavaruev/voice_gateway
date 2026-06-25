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
import numpy as np
import noisereduce as nr
import onnxruntime as ort
import opuslib
from pydub import AudioSegment
from loguru import logger
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
NANOBOT_WS_URL = os.getenv("NANOBOT_WS_URL", "ws://nanobot:8765/").rstrip("/")
SPEAKER_ID_URL = os.getenv("SPEAKER_ID_URL", "http://192.168.22.102:8001/identify")

WHISPER_URL = os.getenv("WHISPER_URL", "http://192.168.22.111:8000/v1/audio/transcriptions")

TTS_URL = os.getenv("TTS_URL", "http://edge_tts:5050/v1/audio/speech")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")  
TTS_VOICE = os.getenv("TTS_VOICE", "ru-RU-SvetlanaNeural")
TTS_API_KEY = os.getenv("TTS_API_KEY", "")

DB_FILE = "/app/config/devices.json"
VAD_SILENCE_FRAMES = int(os.getenv("VAD_SILENCE_FRAMES", 8))  
WATCHDOG_TIMEOUT = 30.0  
STANDBY_TIMEOUT_QUESTION = int(os.getenv("STANDBY_TIMEOUT_QUESTION", 30))
STANDBY_TIMEOUT_STATEMENT = int(os.getenv("STANDBY_TIMEOUT_STATEMENT", 10))
CHAT_ID_TTL = int(os.getenv("CHAT_ID_TTL", 604800))  # 7-day sliding window — Nanobot context lives a week
THINKING_SOUND_PATH = os.getenv("THINKING_SOUND_PATH", "")

ENERGY_THRESHOLD = float(os.getenv("ENERGY_THRESHOLD", "0.008"))
MIN_SPEECH_RATIO = float(os.getenv("MIN_SPEECH_RATIO", "0.15"))
VAD_ADAPTIVE = os.getenv("VAD_ADAPTIVE", "true").lower() == "true"

WHISPER_HALLUCINATIONS = [
    "субтитры подогнал симон", "спасибо за просмотр", "подписывайтесь на канал",  
    "аминь", "субтитры создавал", "редактор субтитров", "thank you", "thanks for watching", "so",
    "dimatorzok", "субтитры сделал", "dima torzok", "продолжение следует"
]

SINGLE_WORD_HALLUCINATIONS = {"о", "а", "и", "кх-кх", "ха-ха", "жизнь", "пьютер"}

HOLD_PHRASES = {"подожди", "мomento", "секундочку", "подожди-ка", "один момент", "мomento"}

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
                mac: entry for mac, entry in data.items()
                if now - entry.get("ts", 0) < CHAT_ID_TTL
            }
        except Exception:
            CHAT_ID_CACHE = {}

def save_chat_id_cache():
    """Persists chat_id cache to disk."""
    try:
        with open(CHAT_ID_CACHE_FILE, "w") as f:
            json.dump(CHAT_ID_CACHE, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save chat_id cache: {e}")

def get_cached_chat_id(mac: str) -> str | None:
    entry = CHAT_ID_CACHE.get(mac.lower())
    if entry and time.time() - entry["ts"] < CHAT_ID_TTL:
        return entry["chat_id"]
    return None

def set_cached_chat_id(mac: str, chat_id: str):
    CHAT_ID_CACHE[mac.lower()] = {"chat_id": chat_id, "ts": time.time()}
    save_chat_id_cache()  # Persist to disk — survives container rebuild

def make_chat_id(mac: str) -> str:
    """Deterministic chat_id from MAC address in UUID format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx).
       Nanobot reuses the session across reconnects."""
    h = hashlib.sha256(mac.lower().encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

def clear_expired_chat_ids():
    now = time.time()
    expired = [mac for mac, entry in CHAT_ID_CACHE.items() if now - entry["ts"] >= CHAT_ID_TTL]
    for mac in expired:
        del CHAT_ID_CACHE[mac]
    save_chat_id_cache()

load_chat_id_cache()  # Load on startup

app = FastAPI()

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
        self.session = ort.InferenceSession("silero_vad.onnx", sess_options=opts, providers=['CPUExecutionProvider'])
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

    def is_speech(self, pcm: bytes) -> tuple[bool, float]:
        try:
            audio_int16 = np.frombuffer(pcm, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(np.square(audio_float32))))
            self.buffer = np.concatenate((self.buffer, audio_float32))
            speech_detected = False
            current_threshold = self.threshold
            while len(self.buffer) >= 512:
                chunk = self.buffer[:512]
                self.buffer = self.buffer[512:]
                out, self._state = self.session.run(None, {
                    'input': chunk[np.newaxis, :],  
                    'state': self._state,  
                    'sr': np.array([16000], dtype=np.int64)
                })
                if out[0][0] > current_threshold:
                    speech_detected = True
            if rms > 0.02:
                speech_detected = True
            
            # Adaptive threshold: learn noise floor during silence
            if self.vad_adaptive and not speech_detected:
                self.noise_floor = (1 - self.alpha) * self.noise_floor + self.alpha * rms
                self.threshold = max(0.15, self.noise_floor * 4)
            
            return speech_detected, rms
        except Exception as e:
            logger.error(f"❌ VAD Error: {e}")
            return False, 0.0

vad = VadEngine()

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
        except Exception: pass
    _DB_CACHE = {}
    return _DB_CACHE

def save_db(db: dict):
    global _DB_CACHE
    _DB_CACHE = db
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    with open(DB_FILE, "w") as f: json.dump(db, f, indent=4)

def pack_ogg(frames: list, sample_rate=16000) -> bytes:
    def ogg_crc(data: bytes) -> int:
        crc, table = 0, []
        for i in range(256):
            c = i << 24
            for _ in range(8): c = (c << 1) ^ 0x04C11DB7 if c & 0x80000000 else c << 1
            table.append(c & 0xFFFFFFFF)
        for b in data: crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ b) & 0xFF]
        return crc

    def page(idx: int, gran: int, ser: int, bos: bool, eos: bool, pkts: list) -> bytes:
        h = struct.pack('<4sBBqIIIB', b'OggS', 0, (2 if bos else 0)|(4 if eos else 0), gran, ser, idx, 0, len(pkts))
        p = h + bytearray([len(x) for x in pkts]) + b"".join(pkts)
        crc = ogg_crc(p)
        return p[:22] + struct.pack('<I', crc) + p[26:]

    ser = int(time.time()) & 0xFFFFFFFF
    res = page(0, 0, ser, True, False, [struct.pack('<8sBBHIHB', b'OpusHead', 1, 1, 312, sample_rate, 0, 0)])
    res += page(1, 0, ser, False, False, [struct.pack('<8sI8sI', b'OpusTags', 8, b'VoiceGW ', 0)])
    for i in range(0, len(frames), 50):
        c = frames[i:i+50]
        res += page(2 + i//50, (i + len(c)) * int(48000 * 0.06), ser, False, (i + 50 >= len(frames)), c)
    return res

# ==========================================
# HARDWARE CONTROL (MCP Tools)
# ==========================================
active_sessions = {}
session_states = {}
mcp_futures = {}

async def send_mcp_cmd(device_ws: WebSocket, session_id: str, tool_name: str, arguments: dict, req_id: int = None):
    if req_id is None:
        req_id = int(time.time() * 1000)
        
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
        "id": req_id
    }
    try:
        await device_ws.send_json({
            "session_id": session_id,
            "type": "mcp",  
            "payload": payload
        })
    except Exception: pass

async def request_mcp_tools(device_ws: WebSocket, session_id: str):
    logger.info("🛠 [MCP] Requesting available tools from ESP32...")
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": { "cursor": "", "withUserTools": True },
        "id": 999  
    }
    try:
        await device_ws.send_json({
            "session_id": session_id,
            "type": "mcp",  
            "payload": payload
        })
    except Exception as e:
        logger.error(f"❌ [MCP] Failed to request tools: {e}")

async def activity_monitor_task(device_ws: WebSocket, state: dict):
    """Monitors idle: dims screen but does NOT close the connection.
       Persistent mode — WS/context lives while ESP32 is on."""
    await send_mcp_cmd(device_ws, state["sid"], "self.audio_speaker.set_volume", {"volume": 100})
    await send_mcp_cmd(device_ws, state["sid"], "self.screen.set_brightness", {"brightness": 100})
    
    try:
        while state["sid"] in session_states:
            now = time.time()
            
            # Freeze timer while the speaker is active (thinking/speaking)
            if state["status"] in ["PROCESSING", "SPEAKING"]:
                state["last_activity"] = now
                
            time_idle = now - state["last_activity"]
            
            # Adaptive standby: 30s after question, 10s after statement
            if state["status"] == "LISTENING":
                timeout = STANDBY_TIMEOUT_QUESTION if state.get("last_ai_had_question") else STANDBY_TIMEOUT_STATEMENT
                if time_idle > timeout:
                    logger.info(f"💤 [Timeout] {int(time_idle)}s idle (limit {timeout}s). Standby.")
                    asyncio.create_task(reset_to_standby(device_ws, state))
                    break
                    
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error in monitor task: {e}")

async def reset_to_standby(device_ws: WebSocket, state: dict):
    """Switches the speaker to standby: dims screen, closes WS.
       chat_id is cached on disk — Nanobot restores context on reconnect."""
    logger.info("💤 Idle — dim screen, close WS. chat_id saved for next connection")
    state["status"] = "IDLE"
    
    await send_mcp_cmd(device_ws, state["sid"], "self.screen.set_brightness", {"brightness": 0})
    await asyncio.sleep(0.5)
    
    state["last_ai_had_question"] = False
    
    if state.get("watchdog"):
        state["watchdog"].cancel()
        state["watchdog"] = None
    try:
        await device_ws.close()
    except Exception: pass

async def watchdog_timeout(device_ws: WebSocket, state: dict):
    logger.warning("⏱ [Watchdog] Upstream AI timed out.")
    state["status"] = "SPEAKING"
    try:
        await generate_and_stream_tts("Простите, я задумалась. Повторите пожалуйста.", device_ws, state["sid"], state)
    except Exception: pass
    
    logger.info("🎤 [Watchdog] Keeping mic open for repeat.")
    state.update({"status": "LISTENING", "frames": [], "silence": 0, "has_speech": False})
    state["last_activity"] = time.time() # Reset timer
    vad.reset()

# ==========================================
# ASYNC PIPELINE (STT & SPEAKER ID)
# ==========================================
async def fetch_speaker_id(audio: bytes, sess: aiohttp.ClientSession) -> str:
    try:
        form = aiohttp.FormData()
        form.add_field('file', audio, filename='audio.ogg', content_type='audio/ogg')
        async with sess.post(SPEAKER_ID_URL, data=form, timeout=10) as r:
            if r.status == 200:
                json_resp = await r.json()
                uid, conf = json_resp.get("user_id", "unknown"), json_resp.get("confidence", 0.0)
                if uid != "unknown" and conf > 0.1:
                    logger.info(f"✅ [SpeakerID] Recognized: {uid} ({conf:.2f})")
                    return uid
                else:
                    logger.debug(f"👤 [SpeakerID] Rejected: {uid} ({conf:.2f})")
    except Exception: pass
    return "unknown"

async def fetch_transcription(audio: bytes, sess: aiohttp.ClientSession) -> str:
    try:
        form = aiohttp.FormData()
        form.add_field('file', audio, filename='a.ogg')
        
        # REQUIRED field for speaches (OpenAI API), otherwise 422 error
        # Exact downloaded model name
        form.add_field('model', 'koekaverna/faster-whisper-podlodka-turbo')
        form.add_field('language', 'ru')
        form.add_field('temperature', '0.0')

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
    if not clean: return False

    if len(clean) >= 10:
        max_run, cur = 1, 1
        for i in range(1, len(clean)):
            cur = cur + 1 if clean[i] == clean[i-1] else 1
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
        if bad in clean: return False
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

def denoise_audio(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Apply noise reduction to PCM16 audio data using noisereduce."""
    try:
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Apply noise reduction
        # stationary=False for non-stationary noise (speech-like noise)
        # prop_decrease=0.8 to reduce noise by 80%
        reduced = nr.reduce_noise(
            y=audio_float32,
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.8
        )
        
        # Convert back to int16
        reduced_int16 = (reduced * 32767).astype(np.int16)
        return reduced_int16.tobytes()
    except Exception as e:
        logger.warning(f"⚠️ [Denoise] Failed: {e}, returning original audio")
        return pcm_data

def decode_opus_frames(frames: list, decoder: opuslib.Decoder) -> tuple[bytes, list[float], list[bool]]:
    """Decode Opus frames to PCM and return (combined_pcm, rms_list, vad_results)."""
    all_pcm = bytearray()
    rms_list = []
    vad_results = []
    
    # Create a temporary decoder for this batch
    temp_dec = opuslib.Decoder(16000, 1)
    temp_vad = VadEngine()
    
    for frame in frames:
        try:
            pcm = temp_dec.decode(frame, 960)
            all_pcm.extend(pcm)
            
            # Calculate RMS
            rms = calculate_rms(pcm)
            rms_list.append(rms)
            
            # VAD check
            is_sp, _ = vad.is_speech(pcm)
            vad_results.append(is_sp)
        except Exception:
            rms_list.append(0.0)
            vad_results.append(False)
    
    return bytes(all_pcm), rms_list, vad_results

async def process_audio_and_send(frames: list, state: dict, device_ws: WebSocket):
    if len(frames) < 25:
        logger.info(f"🔇 [Pipeline] Too few frames ({len(frames)}), likely noise — skipping STT")
        state["status"] = "LISTENING"
        return
    try:
        # Decode frames for metrics and gates
        dec = opuslib.Decoder(16000, 1)
        pcm_data, rms_list, vad_results = decode_opus_frames(frames, dec)
        
        # Apply noise reduction to combined PCM
        pcm_data = denoise_audio(pcm_data, sample_rate=16000)
        
        # Debug metrics logging
        avg_rms = sum(rms_list) / len(rms_list) if rms_list else 0.0
        speech_frames = sum(1 for v in vad_results if v)
        speech_ratio = speech_frames / len(vad_results) if vad_results else 0.0
        
        logger.info(f"📊 [Metrics] RMS={avg_rms:.4f}, speech_ratio={speech_ratio:.2f}, frames={len(frames)}")
        
        # Energy Gate: reject if overall RMS too low
        avg_rms = sum(rms_list) / len(rms_list) if rms_list else 0.0
        if avg_rms < ENERGY_THRESHOLD:
            logger.info(f"🔇 [Energy Gate] RMS={avg_rms:.4f} < {ENERGY_THRESHOLD}, skipping Whisper")
            state["status"] = "LISTENING"
            return
        
        # Speech Ratio Gate: require minimum fraction of VAD-speech frames
        speech_frames = sum(1 for v in vad_results if v)
        speech_ratio = speech_frames / len(vad_results) if vad_results else 0.0
        if speech_ratio < MIN_SPEECH_RATIO:
            logger.info(f"🔇 [Speech Ratio] {speech_ratio:.1%} < {MIN_SPEECH_RATIO}, skipping Whisper")
            state["status"] = "LISTENING"
            return
        
        audio = pack_ogg(frames)
        logger.info(f"🎙 [Pipeline] Processing {len(audio)} bytes (rms={avg_rms:.4f}, speech_ratio={speech_ratio:.2f})...")
        # Light up screen before sending to Nanobot — in case listen:start hasn't arrived yet
        asyncio.create_task(send_mcp_cmd(device_ws, state["sid"],
            "self.screen.set_brightness", {"brightness": 100}))
        asyncio.create_task(trigger_emotion("thinking", device_ws, state["sid"]))
        async with aiohttp.ClientSession() as sess:
            uid_task = asyncio.create_task(fetch_speaker_id(audio, sess))
            stt_task = asyncio.create_task(fetch_transcription(audio, sess))
            uid, txt = await asyncio.gather(uid_task, stt_task)
            
            if is_valid_text(txt):
                logger.info(f"🗣 [User: {uid}] Transcribed: '{txt}'")
                state["last_activity"] = time.time()
                state["last_text"] = txt  
                
                # Extend chat_id cache TTL on each successful STT — 7-day sliding window
                if state.get("nanobot_chat_id"):
                    set_cached_chat_id(state["mac"].lower(), state["nanobot_chat_id"])
                
                # Hold phrase detection - extend listening
                if any(p in txt.lower() for p in HOLD_PHRASES):
                    logger.info(f"🛑 [Hold] Detected hold phrase, extending listening")
                    state["last_activity"] = time.time()
                
                try:
                    await device_ws.send_json({"type": "stt", "text": txt, "session_id": state["sid"]})
                except Exception: return
                
                nano_ws = state.get("nano_ws")
                if nano_ws and not nano_ws.closed:
                    chat_id = state.get("nanobot_chat_id")
                    if not chat_id:
                        logger.warning("⚠️ [Pipeline] Nanobot connected but no chat_id yet, waiting...")
                        state["status"] = "SPEAKING"
                        await generate_and_stream_tts("Система не готова, повторите.", device_ws, state["sid"], state)
                        await reset_to_standby(device_ws, state)  
                        return
                    
                    speaker_name = SPEAKER_NAME_MAP.get(uid, uid)
                    payload = {"type": "message", "chat_id": chat_id, "content": txt, "user_id": uid, "user_name": speaker_name, "voice_reply": True}
                    await nano_ws.send_json(payload)
                    
                    if state.get("watchdog"): state["watchdog"].cancel()
                    state["watchdog"] = asyncio.get_event_loop().call_later(WATCHDOG_TIMEOUT, lambda: asyncio.create_task(watchdog_timeout(device_ws, state)))
                else:
                    logger.warning(f"⚠️ [Pipeline] Nanobot not connected, falling back to direct TTS")
                    state["status"] = "SPEAKING"
                    await generate_and_stream_tts(txt, device_ws, state["sid"], state)
                    await reset_to_standby(device_ws, state)
            else:
                logger.warning(f"⚠ [Pipeline] Rejected transcription (text='{txt}') from user '{uid}', asking for repeat")
                state["status"] = "SPEAKING"
                await generate_and_stream_tts("Скажи ещё раз?", device_ws, state["sid"], state)
                logger.info("🎤 [Pipeline] Returning mic for repeat after rejected.")
                state.update({"status": "LISTENING", "frames": [], "silence": 0, "has_speech": False})
                state["last_activity"] = time.time()
                vad.reset()
    except Exception as e:
        logger.error(f"❌ Pipeline Error: {e}")
        await reset_to_standby(device_ws, state)

# ==========================================
# TTS & EMOTION
# ==========================================
async def generate_and_stream_tts(text: str, device_ws: WebSocket, session_id: str, state: dict = None):
    logger.info(f"🔊 [TTS] Synthesizing: '{text}'")
    try:
        try: await device_ws.send_json({"type": "tts", "state": "start", "session_id": session_id})
        except Exception: return
        
        async with aiohttp.ClientSession() as sess:
            payload = {"model": TTS_MODEL, "input": text, "voice": TTS_VOICE, "response_format": "mp3"}
            headers = {"Content-Type": "application/json"}
            if TTS_API_KEY:
                headers["Authorization"] = f"Bearer {TTS_API_KEY}"
            async with sess.post(TTS_URL, json=payload, headers=headers, timeout=30) as r:
                if r.status == 200:
                    mp3_data = await r.read()
                    audio_seg = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
                    audio_seg = audio_seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
                    pcm_data = audio_seg.raw_data
                    
                    enc = opuslib.Encoder(24000, 1, 'voip')
                    frame_size = 1440  
                    chunk_size = frame_size * 2  
                    
                    start_stream = time.perf_counter()
                    next_chunk_time = start_stream
                    
                    for i in range(0, len(pcm_data), chunk_size):
                        chunk = pcm_data[i:i+chunk_size]
                        if len(chunk) < chunk_size: chunk += b'\x00' * (chunk_size - len(chunk))
                        opus_frame = enc.encode(chunk, frame_size)
                        
                        try: await device_ws.send_bytes(opus_frame)
                        except Exception: return  
                        
                        next_chunk_time += 0.06
                        sleep_duration = next_chunk_time - time.perf_counter()
                        if sleep_duration > 0: await asyncio.sleep(sleep_duration)
                    
                    logger.info("✅ [TTS] Audio stream completed smoothly.")
                    # Set TTS cooldown to prevent VAD triggering on our own output
                    if state:
                        state["tts_cooldown_until"] = time.time() + 1.5  # 1.5s cooldown after TTS
                else:
                    logger.error(f"❌ [TTS] API Error: {await r.text()}")
    except Exception as e:  
        if "Cannot call" not in str(e): logger.error(f"❌ [TTS] Error: {e}")
    finally:
        try: await device_ws.send_json({"type": "tts", "state": "stop", "session_id": session_id})
        except Exception: pass

async def trigger_emotion(emotion: str, device_ws: WebSocket, session_id: str):
    logger.info(f"💡 [Emotion] Setting display face to: '{emotion}'")
    try: await device_ws.send_json({"session_id": session_id, "type": "llm", "emotion": emotion, "text": " "})
    except Exception: pass

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
        self.emotion_regex = re.compile(r'\[([a-zA-Z0-9_]+)\]')
        self.is_flushing = False

    async def handle_chunk(self, chunk: str):
        if self.state.get("watchdog"):
            self.state["watchdog"].cancel()
            self.state["watchdog"] = None
        self.buffer += chunk
        if self.timer: self.timer.cancel()
        # Increase delay for longer responses to avoid premature flush
        delay = 2.0 if self.buffer.count("[") > self.buffer.count("]") else 1.5
        self.timer = asyncio.get_event_loop().call_later(delay, lambda: asyncio.create_task(self.flush()))

    async def flush(self):
        if self.is_flushing or not self.buffer.strip(): return
        self.is_flushing = True
        self.state["status"] = "SPEAKING"  
        
        text = self.buffer
        self.buffer = ""  
        
        # If emotion tag is still incomplete, wait for next chunk
        if text.count("[") > text.count("]"):
            self.buffer = text
            self.timer = asyncio.get_event_loop().call_later(0.5, lambda: asyncio.create_task(self.flush()))
            self.is_flushing = False
            return
        
        emotions = self.emotion_regex.findall(text)
        for emotion in emotions:
            asyncio.create_task(trigger_emotion(emotion, self.device_ws, self.state["sid"]))
        
        clean_text = self.emotion_regex.sub('', text).strip()
        
        # [disconnect] command — user said "disconnect", Nanobot confirmed
        if "[disconnect]" in clean_text:
            clean_text = clean_text.replace("[disconnect]", "").strip()
            if clean_text:
                logger.info(f"📝 [TTS Input] Sending to TTS: '{clean_text[:100]}...' (len={len(clean_text)})")
                try: await self.device_ws.send_json({"type": "tts", "state": "sentence_start", "text": clean_text, "session_id": self.state["sid"]})
                except Exception: pass
                await generate_and_stream_tts(clean_text, self.device_ws, self.state["sid"], self.state)
            # Close session — user explicitly requested disconnect
            logger.info("🔌 [Disconnect] User requested disconnect — closing session")
            try: await self.device_ws.close()
            except Exception: pass
            self.is_flushing = False
            return
        
        if clean_text:
            logger.info(f"📝 [TTS Input] Sending to TTS: '{clean_text[:100]}...' (len={len(clean_text)})")
            self.full_response_text += clean_text + " "
            try: await self.device_ws.send_json({"type": "tts", "state": "sentence_start", "text": clean_text, "session_id": self.state["sid"]})
            except Exception: pass
            
            await generate_and_stream_tts(clean_text, self.device_ws, self.state["sid"], self.state)
            
            clean_for_check = self.full_response_text.strip().lower()
            has_question = (re.search(r'[?？]\s*$', clean_for_check) is not None) or ("повторите пожалуйста" in clean_for_check)
            self.state["last_ai_had_question"] = has_question
            
            self.state.update({"status": "LISTENING", "frames": [], "silence": 0, "has_speech": False})
            self.state["last_activity"] = time.time()
            vad.reset()
            self.full_response_text = ""
        self.is_flushing = False

# ==========================================
# WEB UI
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def web_index(req: Request):
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
        "status": "LISTENING", "frames": [], "silence": 0, "has_speech": False,  
        "sid": session_id, "mac": mac_addr, "version": 1, "watchdog": None,
        "nanobot_chat_id": None, "last_text": "", "last_activity": time.time(),
        "available_tools": [],
        "tts_cooldown_until": 0.0,
    }
    session_states[session_id] = state
    logger.info(f"🔌 [WS] Device connected. Session: {session_id}")
    
    monitor_task = asyncio.create_task(activity_monitor_task(device_ws, state))
    
    await device_ws.send_json({
        "type": "hello", "transport": "websocket", "session_id": session_id,
        "audio_params": {"format": "opus", "sample_rate": 24000, "channels": 1, "frame_duration": 60}
    })
    
    await request_mcp_tools(device_ws, session_id)
    
    dec = opuslib.Decoder(16000, 1)
    vad.reset()
    nano_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    nano_ws = None
    nano_listener_task = None
    
    async def listen_to_nanobot():
        handler = NanobotResponseHandler(device_ws, state)
        async for msg in nano_ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                if state.get("watchdog"):
                    state["watchdog"].cancel()
                    state["watchdog"] = asyncio.get_event_loop().call_later(WATCHDOG_TIMEOUT, lambda: asyncio.create_task(watchdog_timeout(device_ws, state)))
                
                try:
                    d = json.loads(msg.data)
                    if d.get("event") == "ready":
                        nano_chat_id = d.get("chat_id")
                        state["nanobot_chat_id"] = make_chat_id(state["mac"])
                        mac_key = state["mac"].lower()
                        set_cached_chat_id(mac_key, state["nanobot_chat_id"])
                        logger.info(f"💾 [Nanobot] Cached deterministic chat_id for {mac_key}: {state['nanobot_chat_id']} (Nanobot assigned: {nano_chat_id})")
                        if state["available_tools"]:
                            logger.info(f"📤 [Nanobot] Feeding AI tool list: {len(state['available_tools'])} tools")
                            await nano_ws.send_json({
                                "type": "tools_update",
                                "chat_id": state["nanobot_chat_id"],
                                "tools": state["available_tools"]
                            })
                    elif d.get("event") == "error":
                        logger.error(f"❌ Nanobot Error: {d.get('detail')}")
                    elif "text" in d and d.get("type") not in ["stt", "listen"] and d.get("event") != "reasoning_delta":
                        await handler.handle_chunk(d["text"])
                except Exception: pass

    try:
        while True:
            m = await device_ws.receive()
            
            # --- TEXT EVENTS ---
            text_data = m.get("text")
            if text_data:
                state["last_activity"] = time.time()
                d = json.loads(text_data)
                
                if d.get("type") == "mcp":
                    payload = d.get("payload", {})
                    req_id = payload.get("id")
                    
                    if req_id == 999 and "result" in payload and "tools" in payload["result"]:
                        state["available_tools"] = payload["result"]["tools"]
                        tool_names = [t.get("name") for t in state["available_tools"]]
                        logger.info(f"🛠 [MCP] ESP32 returned tools: {tool_names}")
                        continue
                    
                    if req_id in mcp_futures and not mcp_futures[req_id].done():
                        mcp_futures[req_id].set_result(payload)
                
                elif d.get("type") == "hello":
                    state["version"] = d.get("version", 1)
                    logger.info(f"🤝 [Device] Hello received (v{state['version']})")
                    async def connect_upstream():
                        nonlocal nano_ws, nano_listener_task
                        mac_key = state["mac"].lower()
                        det_chat_id = make_chat_id(mac_key)
                        auth_url = f"{NANOBOT_WS_URL}?token=token&chat_id={det_chat_id}"
                        logger.info(f"🔁 [Nanobot] Connecting with deterministic chat_id: {det_chat_id}")
                        for attempt in range(8):
                            try:
                                nano_ws = await nano_session.ws_connect(auth_url)
                                state["nano_ws"] = nano_ws
                                logger.info(f"✅ AI Brain connected (chat_id will follow).")
                                nano_listener_task = asyncio.create_task(listen_to_nanobot())
                                return
                            except Exception as e:
                                logger.warning(f"⚠️ [Nanobot] Connection attempt {attempt+1}/8 failed: {e}")
                                await asyncio.sleep(3)
                        logger.error(f"❌ [Nanobot] All 8 connection attempts failed!")
                    asyncio.create_task(connect_upstream())
                
                elif d.get("type") == "listen" and d.get("state") == "start":
                    state["status"] = "LISTENING"
                    state["frames"] = []
                    state["silence"] = 0
                    state["has_speech"] = False
                    vad.reset()
                    # Wake — light up screen after wake word before audio processing starts
                    try: await device_ws.send_json({"session_id": state["sid"],
                        "type": "mcp", "payload": {
                            "jsonrpc": "2.0", "method": "tools/call",
                            "params": {"name": "self.screen.set_brightness", "arguments": {"brightness": 100}},
                            "id": int(time.time() * 1000)
                        }})
                    except Exception: pass
                
                elif d.get("type") == "listen" and d.get("state") == "stop":
                    if state["status"] == "LISTENING" and state["has_speech"]:
                        state["status"] = "PROCESSING"
                        frames_to_process, state["frames"] = list(state["frames"]), []
                        asyncio.create_task(process_audio_and_send(frames_to_process, state, device_ws))

            # --- AUDIO BYTES ---
            byte_data = m.get("bytes")
            if byte_data:
                # In IDLE we don't listen — wait for listen:start from device (wake word)
                if state["status"] == "IDLE": continue
                if state["status"] in ["PROCESSING", "SPEAKING"]: continue
                # Skip audio during TTS cooldown to prevent self-triggering
                if time.time() < state.get("tts_cooldown_until", 0):
                    continue
                state["status"] = "LISTENING"
                
                f = byte_data if state["version"] == 1 else (byte_data[16:] if state["version"] == 2 else byte_data[4:])
                state["frames"].append(f)
                
                try:
                    pcm = dec.decode(f, 960)
                    is_sp, _ = vad.is_speech(pcm)
                    if is_sp:  
                        state["silence"] = 0
                        state["has_speech"] = True
                    else: state["silence"] += 1
                except Exception: state["silence"] += 1
                
                if state["silence"] > VAD_SILENCE_FRAMES:
                    if state["has_speech"]:
                        logger.info(f"🔪 Server VAD triggered.")
                        state["status"] = "PROCESSING"
                        frames_to_process, state["frames"] = list(state["frames"]), []
                        state["silence"] = 0
                        state["has_speech"] = False
                        asyncio.create_task(process_audio_and_send(frames_to_process, state, device_ws))
                    else:
                        state["frames"], state["silence"] = [], 0
                        vad.reset()

    except Exception: pass
    finally:
        monitor_task.cancel()
        if state.get("watchdog"): state["watchdog"].cancel()
        if session_id in active_sessions: del active_sessions[session_id]
        if session_id in session_states: del session_states[session_id]
        if nano_listener_task: nano_listener_task.cancel()
        if nano_ws: await nano_ws.close()
        await nano_session.close()

# ==========================================
# REST API & OTA
# ==========================================
@app.get("/api/devices")
async def api_get_devices():
    return [{"session_id": sid, "mac": st["mac"], "status": st["status"], "last_text": st["last_text"]} for sid, st in session_states.items()]

@app.post("/mcp/{session_id}")
async def execute_mcp(session_id: str, req: Request):
    data = await req.json()
    req_id = data.get("id")
    if session_id == "latest" and active_sessions: session_id = list(active_sessions.keys())[-1]
    if session_id in active_sessions:
        if req_id is not None:
            future = asyncio.get_event_loop().create_future()
            mcp_futures[req_id] = future
            await active_sessions[session_id].send_json({"session_id": session_id, "type": "mcp", "payload": data})
            try: return await asyncio.wait_for(future, timeout=10.0)
            except asyncio.TimeoutError:
                del mcp_futures[req_id]
                return {"error": "Timeout"}
        else:
            await active_sessions[session_id].send_json({"session_id": session_id, "type": "mcp", "payload": data})
            return {"status": "sent"}
    return {"error": "Offline"}

def load_firmware_meta() -> dict:
    try:
        with open(FIRMWARE_META) as f: return json.load(f)
    except Exception: return {"version": "", "filename": "", "timestamp": 0}

def save_firmware_meta(version: str, filename: str):
    meta = {"version": version, "filename": filename, "timestamp": int(time.time() * 1000)}
    with open(FIRMWARE_META, "w") as f: json.dump(meta, f)
    return meta

@app.post("/api/firmware/upload")
async def firmware_upload(file: UploadFile = File(...), version: str = Form("")):
    if not file.filename or not file.filename.endswith(".bin"):
        raise HTTPException(400, "Only .bin files accepted")
    fname = f"firmware_v{version}.bin" if version else file.filename
    fpath = os.path.join(FIRMWARE_DIR, fname)
    content = await file.read()
    with open(fpath, "wb") as f: f.write(content)
    meta = save_firmware_meta(version, fname)
    return {"status": "ok", "meta": meta}

@app.get("/api/firmware")
async def firmware_info():
    meta = load_firmware_meta()
    return meta

@app.api_route("/ota", methods=["GET", "POST"])
async def ota_handler(req: Request):
    db = load_db()
    meta = load_firmware_meta()
    has_update = bool(meta.get("version"))
    url = f"http://{req.url.hostname}:18792/firmware/{meta['filename']}" if has_update else ""
    return {
        "server_time": {"timestamp": int(time.time() * 1000), "timeZone": "Europe/Moscow", "timezone_offset": 180},
        "protocol": "websocket",
        "websocket": {"url": f"ws://{req.url.hostname}:18792/", "access_token": "token"},
        "firmware": {"has_update": has_update, "version": meta.get("version", ""), "url": url}
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
async def api_get_device_config():
    return device_list_with_status()

@app.post("/api/devices/config")
async def api_create_device(body: DeviceCreate):
    mac = normalize_mac(body.mac)
    if not mac:
        raise HTTPException(400, "MAC address required")
    db = load_db()
    if mac in db:
        raise HTTPException(409, "Device already exists")
    db[mac] = {"friendly_name": body.friendly_name, "ws_url": body.ws_url, "allowed": body.allowed}
    save_db(db)
    return {"mac": mac, **db[mac], "status": "offline"}

@app.put("/api/devices/config/{mac}")
async def api_update_device(mac: str, body: DeviceUpdate):
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
    save_db(db)
    return {"mac": normalized, **entry, "status": device_online_status(normalized)}

@app.delete("/api/devices/config/{mac}")
async def api_delete_device(mac: str):
    normalized = normalize_mac(mac)
    db = load_db()
    if normalized not in db:
        raise HTTPException(404, "Device not found")
    del db[normalized]
    save_db(db)
    return {"status": "deleted", "mac": normalized}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18792)
