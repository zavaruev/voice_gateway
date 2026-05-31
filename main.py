import asyncio
import json
import os
import time
import struct
import uuid
import re
import io
import aiohttp
import numpy as np
import onnxruntime as ort
import opuslib
from pydub import AudioSegment
from loguru import logger
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
NANOBOT_WS_URL = os.getenv("NANOBOT_WS_URL", "ws://nanobot:8765/").rstrip("/")
SPEAKER_ID_URL = os.getenv("SPEAKER_ID_URL", "http://192.168.22.102:8001/identify")

WHISPER_URL = os.getenv("WHISPER_URL", "http://192.168.22.111:8000/v1/audio/transcriptions")

TTS_URL = os.getenv("TTS_URL", "http://edge_tts:5050/v1/audio/speech")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")  
TTS_VOICE = os.getenv("TTS_VOICE", "ru-RU-SvetlanaNeural")

DB_FILE = "/app/config/devices.json"
VAD_SILENCE_FRAMES = int(os.getenv("VAD_SILENCE_FRAMES", 15))  
WATCHDOG_TIMEOUT = 30.0  

WHISPER_HALLUCINATIONS = [
    "субтитры подогнал симон", "спасибо за просмотр", "подписывайтесь на канал",  
    "аминь", "субтитры создавал", "редактор субтитров", "thank you", "thanks for watching", "so",
    "dimatorzok", "субтитры сделал", "dima torzok", "продолжение следует"
]

app = FastAPI()

if os.path.exists("/app/config/firmware"):
    app.mount("/firmware", StaticFiles(directory="/app/config/firmware"), name="firmware")

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
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self.buffer = np.array([], dtype=np.float32)

    def is_speech(self, pcm: bytes) -> tuple[bool, float]:
        try:
            audio_int16 = np.frombuffer(pcm, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(np.square(audio_float32))))
            self.buffer = np.concatenate((self.buffer, audio_float32))
            speech_detected = False
            while len(self.buffer) >= 512:
                chunk = self.buffer[:512]
                self.buffer = self.buffer[512:]
                out, self._state = self.session.run(None, {
                    'input': chunk[np.newaxis, :],  
                    'state': self._state,  
                    'sr': np.array([16000], dtype=np.int64)
                })
                if out[0][0] > 0.15:
                    speech_detected = True
            if rms > 0.02:
                speech_detected = True
            return speech_detected, rms
        except Exception as e:
            logger.error(f"❌ VAD Error: {e}")
            return False, 0.0

vad = VadEngine()

# ==========================================
# UTILS & AUDIO PACKING
# ==========================================
def load_db() -> dict:
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except Exception: pass
    return {}

def save_db(db: dict):
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
    is_dimmed = False
    await send_mcp_cmd(device_ws, state["sid"], "self.audio_speaker.set_volume", {"volume": 100})
    await send_mcp_cmd(device_ws, state["sid"], "self.screen.set_brightness", {"brightness": 100})
    
    try:
        while state["sid"] in session_states:
            now = time.time()
            
            # Замораживаем таймер, пока колонка активна (думает/говорит)
            if state["status"] in ["PROCESSING", "SPEAKING"]:
                state["last_activity"] = now
                
            time_idle = now - state["last_activity"]
            
            # Защита от зависания сессии, если пользователь ушел во время вопроса
            if state["status"] == "LISTENING" and time_idle > 45:
                logger.info("💤 [Timeout] Пользователь не отвечает. Закрываем брошенную сессию.")
                asyncio.create_task(reset_to_standby(device_ws, state))
                break

            if time_idle > 30:
                if not is_dimmed:
                    logger.debug("🌙 [Screen] Dimming to 25% due to inactivity.")
                    await send_mcp_cmd(device_ws, state["sid"], "self.screen.set_brightness", {"brightness": 25})
                    is_dimmed = True
            else:
                if is_dimmed:
                    logger.debug("☀ [Screen] Activity detected. Resetting brightness to 100%.")
                    await send_mcp_cmd(device_ws, state["sid"], "self.screen.set_brightness", {"brightness": 100})
                    is_dimmed = False
                    
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error in monitor task: {e}")

async def reset_to_standby(device_ws: WebSocket, state: dict):
    logger.info("💤 Переводим колонку в Standby. Понижаем яркость и закрываем сокет...")
    state["status"] = "IDLE"
    
    await send_mcp_cmd(device_ws, state["sid"], "self.screen.set_brightness", {"brightness": 25})
    await asyncio.sleep(0.5)
    
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
        await generate_and_stream_tts("Простите, я задумалась. Повторите пожалуйста.", device_ws, state["sid"])
    except Exception: pass
    
    logger.info("🎤 [Watchdog] Оставляем микрофон открытым для повтора.")
    state.update({"status": "LISTENING", "frames": [], "silence": 0, "has_speech": False})
    state["last_activity"] = time.time() # Сброс таймера
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
                if uid != "unknown" and conf > 0.2:
                    logger.info(f"✅ [SpeakerID] Recognized: {uid} ({conf:.2f})")
                    return uid
    except Exception: pass
    return "unknown"

async def fetch_transcription(audio: bytes, sess: aiohttp.ClientSession) -> str:
    try:
        form = aiohttp.FormData()
        form.add_field('file', audio, filename='a.ogg')
        
        # ОБЯЗАТЕЛЬНОЕ ПОЛЕ для speaches (OpenAI API), иначе ошибка 422
        # Точное имя скачанной модели
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
    for bad in WHISPER_HALLUCINATIONS:
        if bad in clean: return False
    return True

async def process_audio_and_send(frames: list, state: dict, device_ws: WebSocket, nano_ws: aiohttp.ClientWebSocketResponse):
    try:
        audio = pack_ogg(frames)
        logger.debug(f"🎙 [Pipeline] Processing {len(audio)} bytes...")
        async with aiohttp.ClientSession() as sess:
            uid_task = asyncio.create_task(fetch_speaker_id(audio, sess))
            stt_task = asyncio.create_task(fetch_transcription(audio, sess))
            uid, txt = await asyncio.gather(uid_task, stt_task)
            
            if is_valid_text(txt):
                logger.info(f"🗣 [User: {uid}] Transcribed: '{txt}'")
                state["last_activity"] = time.time()
                state["last_text"] = txt  
                
                try:
                    await device_ws.send_json({"type": "stt", "text": txt, "session_id": state["sid"]})
                except Exception: return
                
                if nano_ws and not nano_ws.closed:
                    chat_id = state.get("nanobot_chat_id")
                    if not chat_id:
                        state["status"] = "SPEAKING"
                        await generate_and_stream_tts("Система не готова, повторите.", device_ws, state["sid"])
                        await reset_to_standby(device_ws, state)  
                        return
                    
                    payload = {"type": "message", "chat_id": chat_id, "content": txt, "user_id": uid, "voice_reply": True}
                    await nano_ws.send_json(payload)
                    
                    if state.get("watchdog"): state["watchdog"].cancel()
                    state["watchdog"] = asyncio.get_event_loop().call_later(WATCHDOG_TIMEOUT, lambda: asyncio.create_task(watchdog_timeout(device_ws, state)))
            else:
                state["status"] = "SPEAKING"
                await generate_and_stream_tts("Связь с сервером потеряна.", device_ws, state["sid"])
                await reset_to_standby(device_ws, state)
    except Exception as e:
        logger.error(f"❌ Pipeline Error: {e}")
        await reset_to_standby(device_ws, state)

# ==========================================
# TTS & EMOTION
# ==========================================
async def generate_and_stream_tts(text: str, device_ws: WebSocket, session_id: str):
    logger.info(f"🔊 [TTS] Synthesizing: '{text}'")
    try:
        try: await device_ws.send_json({"type": "tts", "state": "start", "session_id": session_id})
        except Exception: return
        
        async with aiohttp.ClientSession() as sess:
            payload = {"model": TTS_MODEL, "input": text, "voice": TTS_VOICE, "response_format": "mp3"}
            headers = {"Authorization": "Bearer sk-dummy-key-12345", "Content-Type": "application/json"}
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
                    await asyncio.sleep(1.0)  
                else: logger.error(f"❌ [TTS] API Error: {await r.text()}")
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
        self.timer = asyncio.get_event_loop().call_later(0.8, lambda: asyncio.create_task(self.flush()))

    async def flush(self):
        if self.is_flushing or not self.buffer.strip(): return
        self.is_flushing = True
        self.state["status"] = "SPEAKING"  
        
        text = self.buffer
        self.buffer = ""  
        
        emotions = self.emotion_regex.findall(text)
        for emotion in emotions:
            asyncio.create_task(trigger_emotion(emotion, self.device_ws, self.state["sid"]))
        
        clean_text = self.emotion_regex.sub('', text).strip()
        if clean_text:
            self.full_response_text += clean_text + " "
            try: await self.device_ws.send_json({"type": "tts", "state": "sentence_start", "text": clean_text, "session_id": self.state["sid"]})
            except Exception: pass
            
            await generate_and_stream_tts(clean_text, self.device_ws, self.state["sid"])
            
            clean_for_check = self.full_response_text.strip().lower()
            has_question = (re.search(r'[?？]\s*$', clean_for_check) is not None) or ("повторите пожалуйста" in clean_for_check)
            
            if not has_question:
                await reset_to_standby(self.device_ws, self.state)
            else:
                logger.info("🎤 [Dialogue] Question detected. Keeping mic open.")
                self.state.update({"status": "LISTENING", "frames": [], "silence": 0, "has_speech": False})
                self.state["last_activity"] = time.time()
                vad.reset()
            self.full_response_text = ""
        self.is_flushing = False

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
        "available_tools": []
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
                        state["nanobot_chat_id"] = d.get("chat_id")
                        if state["available_tools"]:
                            logger.info(f"📤 [Nanobot] Снабжаем AI списком инструментов: {len(state['available_tools'])} шт.")
                            await nano_ws.send_json({
                                "type": "tools_update",
                                "chat_id": state["nanobot_chat_id"],
                                "tools": state["available_tools"]
                            })
                    elif d.get("event") == "error":
                        logger.error(f"❌ Nanobot Error: {d.get('detail')}")
                    elif "text" in d and d.get("type") not in ["stt", "listen"]:
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
                        logger.info(f"🛠 [MCP] ESP32 вернул инструменты: {tool_names}")
                        continue
                    
                    if req_id in mcp_futures and not mcp_futures[req_id].done():
                        mcp_futures[req_id].set_result(payload)
                
                elif d.get("type") == "hello":
                    state["version"] = d.get("version", 1)
                    async def connect_upstream():
                        nonlocal nano_ws, nano_listener_task
                        auth_url = f"{NANOBOT_WS_URL}?token=token" if "?" not in NANOBOT_WS_URL else f"{NANOBOT_WS_URL}&token=token"
                        for attempt in range(8):
                            try:
                                nano_ws = await nano_session.ws_connect(auth_url)
                                logger.info(f"✅ AI Brain connected.")
                                nano_listener_task = asyncio.create_task(listen_to_nanobot())
                                return
                            except Exception: await asyncio.sleep(3)
                    asyncio.create_task(connect_upstream())
                
                elif d.get("type") == "listen" and d.get("state") == "start":
                    state["status"] = "LISTENING"
                    state["frames"] = []
                    state["silence"] = 0
                    state["has_speech"] = False
                    vad.reset()
                
                elif d.get("type") == "listen" and d.get("state") == "stop":
                    if state["status"] == "LISTENING" and state["has_speech"]:
                        state["status"] = "PROCESSING"
                        frames_to_process, state["frames"] = list(state["frames"]), []
                        asyncio.create_task(process_audio_and_send(frames_to_process, state, device_ws, nano_ws))

            # --- AUDIO BYTES ---
            byte_data = m.get("bytes")
            if byte_data:
                if state["status"] in ["PROCESSING", "SPEAKING"]: continue
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
                        asyncio.create_task(process_audio_and_send(frames_to_process, state, device_ws, nano_ws))
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

@app.api_route("/ota", methods=["GET", "POST"])
async def ota_handler(req: Request):
    db = load_db()
    return {
        "server_time": {"timestamp": int(time.time() * 1000), "timeZone": "Europe/Moscow", "timezone_offset": 180},
        "protocol": "websocket",
        "websocket": {"url": f"ws://{req.url.hostname}:18792/", "access_token": "token"},
        "firmware": {"has_update": False, "version": "2.2.6", "url": ""}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18792)
