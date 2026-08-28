# Voice Gateway

WebSocket gateway connecting ESP32 smart speakers **and OpenIPC IP cameras** (corridor / kitchen / livingroom) to an AI pipeline (openWakeWord → Whisper STT → Nanobot LLM → Edge TTS).

## Entrypoint

`main.py` — FastAPI app on port 18792, run via `python -u main.py` (uvicorn inside `__main__`).

## Core modules

- `main.py` (~2,200 lines) — FastAPI app, ESP32 WebSocket protocol, REST API, OTA
- `camera_client.py` (~2,260 lines) — per-camera session: RTSP audio feed, VAD, wake-word gate cascade, cross-camera arbiter, TTS playback pipeline
- `engine.py` — Silero VAD wrapper + openWakeWord scoring (`check_wakeword()` stores the raw score in `.last_score`)
- `tests/` — pytest suite (`pytest tests/`), ~1,160 lines: engine scoring, camera arbitration, OTA auth, RMS utils. No lint/CI.

## External services (all env-overridable)

| Service | Default URL | Role |
|---|---|---|
| Nanobot | `ws://nanobot:8765/` | AI brain (WebSocket, streamed text) |
| Whisper | `http://192.168.22.111:8000/v1/audio/transcriptions` | STT (OpenAI-compatible) |
| Speaker ID | `http://192.168.22.102:8001/identify` | Speaker recognition |
| Edge TTS | `http://edge_tts:5050/v1/audio/speech` | Text-to-speech (OpenAI-compatible) |
| go2rtc | `http://192.168.22.102:1984` | Camera stream registry (RTSP/WebRTC bridge) |

## Build & run

```sh
docker build -t voice_gateway .
docker run -p 18792:18792 \
  -e CAMERA_STREAMS=corridor,kitchen,livingroom voice_gateway
```

## Architecture notes

- **ESP32 STT pipeline**: Opus frames → `pack_ogg()` → parallel POST to Whisper + Speaker ID → text sent to Nanobot via WS
- **Camera audio**: go2rtc RTSP backchannel (raw L16 16 kHz via ffmpeg) → echo guards → Silero VAD → SpeexDSP NS rescue → adaptive normalisation → Whisper **only inside an active wake window** (ambient utterances never reach STT)
- **Wake word**: openwakeword custom model `config/computer.onnx` (v1: trained on TTS «компьютер», end-aligned). Raw int16 required — never feed normalized floats. Per-room base thresholds: kitchen 0.47, others 0.52. Sliding-window debounce: 2 qualifying chunks out of the last 3 (~80 ms chunks); single chunk ≥0.68 fires immediately. Bidirectional AGC normalises every chunk to a per-room target peak (kitchen 6500, others 4000) before scoring.
- **Wake gate cascade** (in order): own-playback guard → appliance hold (60 s bg median >800 ⇒ need ≥0.92) → quiet-source hold (own level <3000 ⇒ need ≥0.72) → distant-source veto (level <1600 while another room hears ≥1.4× louder ⇒ stand down) → clipping-bang gate (peak >24k & score <0.85) → crest-factor gate (dense impact: rms·2 > peak) → debounce → arbiter.
- **Cross-camera arbiter** (`_arbiter_*` in camera_client.py): first detector becomes interaction owner; others stand down. Proximity steal: a room with ≥5× the owner's loudness level takes over a not-yet-dispatched wake. `_ARB_STATE["cmd_sent"]` blocks steal once the owner dispatched to Nanobot.
- **TTS pipeline**: Nanobot text → sentence splitter → prefetch player (sentence N+1 synthesises while N plays; `_tts_fetch` + `_speak_pcm`) → pydub decode + resample → Opus to ESP32 / PCM queued to camera WebRTC track. Every played clip is registered in the echo-reference ring; confirmed mic echoes extend `_wake_suppress_until`.
- **VAD**: Silero ONNX server-side (`silero_vad.onnx`), 10 silence frames triggers processing, 7 s max-duration cap; VAD state is reset at wake fire so the post-wake command starts clean.
- **Binary frame versions**: v1 = raw Opus, v2 = 16-byte header, v3 = 4-byte header
- **MCP**: Gateway requests tool list (`tools/list` id=999) on connect; forwards to Nanobot as `tools_update`
- **Dialogue mode**: If AI response ends with `?`, follow-up window opens when playback drains; otherwise returns to standby
- **Watchdog**: 30s timeout → fallback TTS "Простите, я задумалась. Повторите пожалуйста."
- **Emotions**: extracted from Nanobot text via `[emotion_name]` regex

## REST API

| Path | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI dashboard |
| `/api/devices` | GET | List active WebSocket sessions |
| `/api/devices/config` | GET/POST | Device DB from `devices.json` |
| `/api/devices/config/{mac}` | PUT/DELETE | Update / remove device |
| `/mcp/{session_id}` | POST | Send MCP command to device (`"latest"` for most recent session) |
| `/api/camera/tts` | POST | Speak a phrase on a camera session |
| `/ota` | GET/POST | ESP32 OTA handshake; returns WS URL + firmware info |

## Config

`config/devices.json` — MAC-keyed device DB. Lookup is case-insensitive via `device-id` header (fallback: `mac` header).

## Key env vars

| Variable | Default | Notes |
|---|---|---|
| `NANOBOT_WS_URL` | `ws://nanobot:8765/` | Trailing `/` stripped, `?token=` appended |
| `CAMERA_STREAMS` | `""` | Comma-separated go2rtc stream names |
| `GO2RTC_HOST` / `GO2RTC_PORT` | `192.168.22.102` / `1984` | go2rtc control API |
| `VAD_SILENCE_FRAMES` | `8` | ESP32 path only; camera path hardcodes 10 |
| `WATCHDOG_TIMEOUT` | `30` | Seconds before fallback TTS |
| `TTS_VOICE` | `ru-RU-SvetlanaNeural` | |

## Gotchas

- `engine.last_score` must be written by `check_wakeword()` — camera_client reads it after every call; a missing write silently kills all wake detection (this exact bug shipped once)
- openwakeword needs raw int16; floats quantize to {-1,0,1} and break the model
- Wake suppressed first 5 s after an RTSP audio (re)connect — ffmpeg startup transient scores 0.7–0.96 at idle
- Confirmed mic echoes extend `_wake_suppress_until` by +20 s each; the RTSP backchannel returns played audio 3–40 s late
- Whisper form must include `model` field (`koekaverna/faster-whisper-podlodka-turbo`)
- Corridor mic clips at close range (peak 32k) — clipped speech mangles oww scores; bang gate ignores peak >24k unless score ≥0.85
- Robot vacuum / hood noise keeps bg median high → appliance hold requires score ≥0.92 in that room
- go2rtc 1.9.2 leaks zombie RTSP sessions under slow links; cameras run `/etc/watchdog_majestic.sh` via crond (restart majestic when :554 dead or send-queues pile up)
- Wake-word retrain attempts v2–v4 all degraded discrimination — keep `model_stream.npz` and `export_onnx.py`; v1 backups: `computer.onnx.bak_v1` (current), `.bak_v4` (failed)
- Dockerfile exposes 8080 but nothing listens on it
- Code and comments are in English (Russian string literals kept for TTS/STT data)
