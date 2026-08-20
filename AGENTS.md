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
- **Wake word**: openwakeword custom model `config/computer.onnx` (96-dim LR trained on TTS «компьютер», end-aligned, W=16), threshold 0.5, evaluated on every live 16k chunk (echo-guarded). Raw int16 required — never feed normalized floats (openwakeword casts its buffer with `.astype(np.int16)`; floats quantize to {-1,0,1}). Wake is suppressed for the first 5s after an RTSP audio (re)connect: the mic/ffmpeg startup transient scores 0.7-0.96 at idle.
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
| `WATCHDOG_TIMEOUT` | `30` | Seconds before fallback TTS (Nanobot unresponsive) |
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
- Wake-word model retrains with real corridor mic data (v2 channel-sim, v3 dirty labels, v4 confident real pos/neg) all degraded discrimination (false wakes on idle) — v1 (TTS-only, end-aligned) is the best known; keep `model_stream.npz` weights and re-export with `export_onnx.py` if needed
- Model backups in `config/`: `computer.onnx.bak_v1` (current), `computer.onnx.bak_v4` (failed retrain), `computer.onnx.bak_degenerate` (original 415KB model)
- Quiet speech (live rms < ~0.01) never wakes — mic/AGC sensitivity limit, not the model; normal-volume speech scores 0.7-0.9 live
