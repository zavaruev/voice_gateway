#!/bin/bash

# 1. Структура папок
mkdir -p config/firmware

# 2. Зависимости
cat <<EOF > requirements.txt
fastapi
uvicorn
python-multipart
aiohttp
websockets
loguru
numpy
onnxruntime
opuslib
EOF

# 3. Dockerfile
cat <<EOF > Dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    libopus0 curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Качаем легкую модель VAD (всего 3мб)
RUN curl -L -o silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx

COPY main.py .
EXPOSE 18792 8080

CMD ["python", "-u", "main.py"]
EOF

# 4. Основной код шлюза (согласно официальной спецификации XiaoZhi)
cat <<EOF > main.py
import asyncio, json, os, time, struct, aiohttp, numpy as np
import onnxruntime as ort
from loguru import logger
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# --- Конфигурация ---
NANOBOT_URL = os.getenv("NANOBOT_URL", "http://nanobot_prod:18790/voice_message")
SPEAKER_URL = os.getenv("SPEAKER_ID_URL", "http://192.168.22.102:8001/identify")
WHISPER_URL = os.getenv("WHISPER_URL", "http://192.168.22.111:8000/v1/audio/transcriptions")
DB_FILE = "/app/config/devices.json"

app = FastAPI()
if os.path.exists("/app/config/firmware"):
    app.mount("/firmware", StaticFiles(directory="/app/config/firmware"), name="firmware")

# Глобальный реестр активных сессий (чтобы Нанобот мог слать команды на колонку)
active_sessions = {}

# --- VAD Engine (ONNX) ---
class SileroVAD:
    def __init__(self):
        self.session = ort.InferenceSession("silero_vad.onnx", providers=['CPUExecutionProvider'])
        self.reset()
    def reset(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
    def is_speech(self, pcm):
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < 512: return False
        out, self._h, self._c = self.session.run(None, {'input': audio[np.newaxis, :512], 'h': self._h, 'c': self._c, 'sr': np.array([16000], dtype=np.int64)})
        return out[0][0] > 0.5

vad = SileroVAD()

# --- База данных ---
def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_db(db):
    with open(DB_FILE, "w") as f: json.dump(db, f, indent=4)

# --- OGG упаковка ---
import zlib
_BIT_REVERSE_TABLE = bytes(int('{:08b}'.format(i)[::-1], 2) for i in range(256))

def pack_ogg(frames):
    def ogg_crc(data):
        crc = zlib.crc32(data.translate(_BIT_REVERSE_TABLE), 0xFFFFFFFF) ^ 0xFFFFFFFF
        return int.from_bytes(crc.to_bytes(4, 'little').translate(_BIT_REVERSE_TABLE), 'big')
    def page(idx, gran, ser, bos, eos, pkts):
        h = struct.pack('<4sBBqIIIB', b'OggS', 0, (2 if bos else 0)|(4 if eos else 0), gran, ser, idx, 0, len(pkts))
        p = h + bytearray([len(x) for x in pkts]) + b"".join(pkts)
        struct.pack_into('<I', p, 22, ogg_crc(p))
        return p
    ser = int(time.time()) & 0xFFFFFFFF
    res = page(0, 0, ser, True, False, [struct.pack('<8sBBHIHB', b'OpusHead', 1, 1, 312, 16000, 0, 0)])
    res += page(1, 0, ser, False, False, [struct.pack('<8sI8sI', b'OpusTags', 8, b'VoiceGW ', 0)])
    for i in range(0, len(frames), 50):
        c = frames[i:i+50]
        res += page(2+i//50, (i+len(c))*2880, ser, False, (i+50>=len(frames)), c)
    return res

# --- Пайплайн обработки (Voice -> Text -> Nanobot) ---
async def process_voice(frames, sid, ws):
    audio = pack_ogg(frames)
    async with aiohttp.ClientSession() as sess:
        uid, txt = "unknown", ""
        
        # 1. Распознавание говорящего (Speaker ID)
        try:
            f = aiohttp.FormData(); f.add_field('file', audio, filename='a.ogg')
            async with sess.post(SPEAKER_URL, data=f, timeout=5) as r:
                if r.status==200: uid = (await r.json()).get("user_id", "unknown")
        except: pass
        
        # 2. Распознавание текста (Whisper STT)
        try:
            f = aiohttp.FormData(); f.add_field('file', audio, filename='a.ogg')
            f.add_field('model', 'Systran/faster-whisper-large-v3')
            async with sess.post(WHISPER_URL, data=f, timeout=30) as r:
                if r.status==200: txt = (await r.json()).get("text", "").strip()
        except: pass
        
        if txt:
            logger.info(f"Session {sid}: {uid} -> {txt}")
            
            # Отправляем STT статус на экран колонки (согласно доке 4.2.2)
            await ws.send_json({"session_id": sid, "type": "stt", "text": txt})
            
            # 3. Передача в Нанобот
            try:
                payload = {"user_id": uid, "text": txt, "session_id": sid, "channel": "voice_gateway"}
                async with sess.post(NANOBOT_URL, json=payload, timeout=10) as r:
                    pass # Ответ придет через отдельный API или WS
            except Exception as e: logger.error(f"Nanobot notify error: {e}")
        else:
            await ws.send_json({"session_id": sid, "type": "tts", "state": "stop"})

# --- Эндпоинты OTA и Приема команд от Нанобота ---
@app.api_route("/ota", methods=["GET", "POST"])
@app.api_route("/xiaozhi/ota/", methods=["GET", "POST"])
async def ota_handler(req: Request):
    mac = "UNKNOWN"
    try:
        if req.method == "POST": mac = (await req.json()).get("mac", "UNKNOWN")
        else: mac = req.query_params.get("mac") or req.headers.get("Mac") or "UNKNOWN"
    except: pass
    db = load_db()
    if mac not in db:
        db[mac] = {"friendly_name": "", "ws_url": f"ws://{req.url.host}:18792/", "allowed": False}
        save_db(db)
    c = db[mac]
    return {"protocol":"websocket","websocket":{"url":c["ws_url"],"access_token":c.get("access_token","")},"firmware":{"has_update":c.get("has_update",False),"version":c.get("version","2.2.5"),"url":c.get("firmware_url","")}}

# Эндпоинт для Нанобота: отправить MCP или TTS в нужную колонку
@app.post("/send/{session_id}")
async def send_to_device(session_id: str, req: Request):
    if session_id in active_sessions:
        data = await req.json()
        await active_sessions[session_id].send_json(data)
        return {"status": "ok"}
    return {"status": "not_found", "error": "Device not connected"}

# --- Основной WebSocket (Умный парсер) ---
@app.websocket("/")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    import opuslib
    dec = opuslib.Decoder(16000, 1)
    
    # Состояние сессии
    state = {"listening": False, "frames": [], "silence": 0, "sid": "unknown", "version": 1}
    vad.reset()
    
    try:
        while True:
            m = await ws.receive()
            if "text" in m:
                d = json.loads(m["text"])
                msg_type = d.get("type")
                
                if msg_type == "hello":
                    # Сохраняем версию протокола и фичи колонки
                    state["version"] = d.get("version", 1)
                    state["sid"] = d.get("session_id", "unknown")
                    active_sessions[state["sid"]] = ws
                    
                    # Отвечаем правильным handshake (дока 4.2.1)
                    await ws.send_json({
                        "type": "hello",
                        "transport": "websocket",
                        "session_id": state["sid"],
                        "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60}
                    })
                    
                elif msg_type == "listen" and d.get("state") == "start":
                    state.update({"listening": True, "frames": [], "silence": 0, "sid": d.get("session_id", state["sid"])})
                    active_sessions[state["sid"]] = ws
                    vad.reset()
                    
                elif msg_type == "mcp":
                    # Прозрачный проброс MCP от ESP32 в Нанобот (дока 4.1.5)
                    logger.info(f"Received MCP from device: {d}")
                    # В будущем Нанобот может слушать этот эндпоинт
                    pass

            elif "bytes" in m and state["listening"]:
                payload = m["bytes"]
                audio_frame = b""
                
                # ДИНАМИЧЕСКИЙ ПАРСИНГ ЗАГОЛОВКОВ (Раздел 3 в websocket.md)
                try:
                    if state["version"] == 1:
                        audio_frame = payload
                    elif state["version"] == 2 and len(payload) >= 16:
                        # struct BinaryProtocol2: version(2), type(2), reserved(4), timestamp(4), payload_size(4)
                        m_type = struct.unpack('<H', payload[2:4])[0]
                        if m_type == 0: audio_frame = payload[16:]
                    elif state["version"] == 3 and len(payload) >= 4:
                        # struct BinaryProtocol3: type(1), reserved(1), payload_size(2)
                        m_type = payload[0]
                        if m_type == 0: 
                            p_size = struct.unpack('<H', payload[2:4])[0]
                            audio_frame = payload[4:4+p_size]
                except Exception as e:
                    logger.error(f"Binary parse error: {e}")
                    continue

                if not audio_frame: continue

                state["frames"].append(audio_frame)
                pcm = dec.decode(audio_frame, 960)
                
                if vad.is_speech(pcm): state["silence"] = 0
                else: state["silence"] += 1
                
                # 25 кадров = 1.5 секунды тишины. 500 кадров = хардлимит 30 сек.
                if state["silence"] > 25 or len(state["frames"]) > 500:
                    state["listening"] = False
                    asyncio.create_task(process_voice(state["frames"], state["sid"], ws))
    except WebSocketDisconnect:
        if state["sid"] in active_sessions:
            del active_sessions[state["sid"]]
        logger.info(f"Device disconnected: {state['sid']}")
    except Exception as e:
        logger.error(f"WS error: {e}")

# --- Админка (Управление устройствами) ---
@app.get("/", response_class=HTMLResponse)
async def admin_page(selected_mac: str = None):
    db = load_db()
    devices_menu = "".join([f"<a href='/?selected_mac={m}' style='display:block;padding:10px;border:1px solid #ccc;margin-bottom:5px;text-decoration:none;background:{'#0d6efd;color:white' if m==selected_mac else 'white'}'>{d.get('friendly_name') or m}</a>" for m, d in db.items()])
    form = "<h3>Выберите устройство слева</h3>"
    if selected_mac in db:
        c = db[selected_mac]
        form = f"""<form action='/save/{selected_mac}' method='post' style='background:white;padding:20px;'>
            <label>Имя:</label><br><input type='text' name='friendly_name' value='{c.get('friendly_name','')}' style='width:100%'><br><br>
            <label><input type='checkbox' name='allowed' value='1' {'checked' if c.get('allowed') else ''}> Доступ разрешен</label><br><br>
            <label>WebSocket URL:</label><br><input type='text' name='ws_url' value='{c.get('ws_url','')}' style='width:100%'><br><br>
            <button type='submit' style='width:100%;padding:10px;background:green;color:white;'>Сохранить настройки</button>
        </form>"""
    return f"<html><head><meta charset='utf-8'></head><body style='font-family:sans-serif;display:flex;padding:20px;gap:20px;background:#f4f4f4;'><div style='width:300px;'>{devices_menu}</div><div style='flex:1;'>{form}</div></body></html>"

@app.post("/save/{mac}")
async def save_device(mac: str, friendly_name: str = Form(""), ws_url: str = Form(...), allowed: str = Form(None)):
    db = load_db(); db[mac].update({"friendly_name": friendly_name, "ws_url": ws_url, "allowed": bool(allowed)}); save_db(db)
    return RedirectResponse(url=f"/?selected_mac={mac}", status_code=303)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18792)
EOF

# 5. Сборка
docker build -t voice_gateway .

echo "-------------------------------------------------------"
echo "Успешно! Образ 'voice_gateway' собран с учетом спецификаций."
echo "-------------------------------------------------------"
