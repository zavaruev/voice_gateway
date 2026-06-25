# Voice Gateway

WebSocket gateway connecting ESP32 smart speakers to an AI pipeline (Whisper STT → Nanobot LLM → Edge TTS).

## Entrypoint

`main.py` — FastAPI app on port 18792, run via `python -u main.py` (uvicorn inside `__main__`).

## External services (all env-overridable)

| Service | Default URL | Role |
|---|---|---|
| Nanobot | `ws://nanobot:8765/` | AI brain (WebSocket, streamed text) |
| Whisper | `http://192.168.22.111:8000/v1/audio/transcriptions` | STT (OpenAI-compatible) |
| Speaker ID | `http://192.168.22.102:8001/identify` | Speaker recognition |
| Edge TTS | `http://edge_tts:5050/v1/audio/speech` | Text-to-speech (OpenAI-compatible) |

## Build & run

```sh
docker build -t voice_gateway .
docker run -p 18792:18792 voice_gateway
```

No test/lint/CI infrastructure.

## Architecture notes

- **STT pipeline**: Opus frames → `pack_ogg()` → parallel POST to Whisper + Speaker ID → text sent to Nanobot via WS
- **TTS pipeline**: Nanobot text → POST to Edge TTS (MP3) → pydub decode + resample to 24kHz → Opus encode → raw Opus frames to ESP32 (60ms chunks, paced via `asyncio.sleep`)
- **VAD**: Silero ONNX model, server-side (`silero_vad.onnx`), 15 silence frames (~900ms) triggers processing
- **Binary frame versions**: v1 = raw Opus, v2 = 16-byte header, v3 = 4-byte header
- **MCP**: Gateway requests tool list (`tools/list` id=999) on connect; forwards to Nanobot as `tools_update`
- **Dialogue mode**: If AI response ends with `?` or contains "повторите пожалуйста", mic stays open; otherwise returns to standby
- **Watchdog**: 30s timeout → fallback TTS "Простите, я задумалась. Повторите пожалуйста."
- **Activity monitor**: dims screen to 25% after 30s idle; closes abandoned sessions after 45s in LISTENING
- **Emotions**: extracted from Nanobot text via `[emotion_name]` regex

## REST API

| Path | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI (Apple-style dashboard: devices + sessions) |
| `/api/devices` | GET | List active WebSocket sessions |
| `/api/devices/config` | GET | List all registered devices from `devices.json` |
| `/api/devices/config` | POST | Create a new device entry |
| `/api/devices/config/{mac}` | PUT | Update device fields |
| `/api/devices/config/{mac}` | DELETE | Remove device entry |
| `/mcp/{session_id}` | POST | Send MCP command to device (`"latest"` for most recent session) |
| `/ota` | GET/POST | ESP32 OTA handshake; returns WS URL + firmware info |

## Config

`config/devices.json` — MAC-keyed device DB. Lookup is case-insensitive via `device-id` header (fallback: `mac` header). Duplicate MACs with varying case exist in the DB.

## Key env vars

| Variable | Default | Notes |
|---|---|---|
| `NANOBOT_WS_URL` | `ws://nanobot:8765/` | Trailing `/` stripped, `?token=token` appended |
| `VAD_SILENCE_FRAMES` | `15` | ~60ms per frame |
| `TTS_VOICE` | `ru-RU-SvetlanaNeural` | |

## Gotchas

- `silero_vad.onnx` downloaded at Docker build time (must exist at runtime)
- `config/devices.json` auto-created with `{}` if missing
- Whisper form must include `model` field (`koekaverna/faster-whisper-podlodka-turbo`)
- TTS sends auth header if `TTS_API_KEY` is provided (`Bearer $TTS_API_KEY`)
- Dockerfile exposes 8080 alongside 18792 but code never listens on 8080
- `setup_gateway.sh` is an outdated snapshot; not authoritative
- `config/` has its own `.git` (no commits); parent dir not version-controlled
- Code and comments are in English (Russian string literals kept for TTS/STT data)
- Whisper hallucination filter rejects transcriptions containing known garbage strings
