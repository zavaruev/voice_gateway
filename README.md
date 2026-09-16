# Voice Gateway

> **Version 2.24** — Hermes-mode hardening: no more reconnect loop to Nanobot when `LLM_BACKEND=hermes`, `DISABLE_CAMERAS` is now actually honored, Hermes requests carry device-identifying headers. See Changelog below.

WebSocket gateway bridging [Xiaozhi ESP32](https://github.com/78/xiaozhi-esp32) smart speakers **and WebRTC/IP cameras** to an AI agent (**[Nanobot](https://github.com/HKUDS/nanobot)** or **Hermes**) with real-time speech processing.

```
ESP32 (Opus WS)  ──┐
                   ├──▶ Voice Gateway ──STT text──▶ Nanobot AI Agent
Camera (RTSP/L16) ─┘          │                        │
                        Silero VAD               Edge TTS
                        Wake word (openWakeWord)  emotions
                        Speaker ID                MCP commands
                        noise gate + SpeexDSP     OTA firmware
                        echo guards
                           │                        │
                           ◀──────── Opus / PCM ─────┘
```

## Pipeline

1. **ESP32** sends Opus audio frames over WebSocket; **cameras** stream raw L16 PCM over RTSP (via go2rtc)
2. **Silero VAD** detects speech, energy gates reject noise/artifacts
3. **openWakeWord** listens for the wake word «компьютер» (custom-trained model) on camera streams
4. **Whisper STT** transcribes audio to text (Russian, retries at temperature 0.0/0.5)
5. **Speaker ID** identifies the speaker (parallel with STT)
 6. **LLM backend** (Nanobot *or* Hermes) processes text, returns response with optional emotions `[emotion_name]`
 7. **Edge TTS** synthesizes speech; Opus streamed to ESP32, PCM queued to camera speakers

## Features

- **ESP32 + camera support** — both device classes in one gateway; cameras use WebRTC (go2rtc) with the mic feed taken from the RTSP audio backchannel (`ffmpeg`, raw L16, no μ-law quantisation)
- **Wake word «компьютер»** — custom-trained openWakeWord model (`config/computer.onnx` + `config/embedding_model.onnx` backbone), detected on the live 16 kHz camera feed with per-room thresholds (kitchen 0.47, other rooms 0.52), sliding-window debounce (2 qualifying chunks out of the last 3), single-chunk bypass at score ≥0.68, attention beep
- **Bidirectional AGC** — every oww chunk is normalised to a per-room target peak (kitchen 6500, others 4000) in *both* directions; quiet far-room copies are boosted while loud close-range speech is scaled down, so all rooms feed the model equally loud audio
- **Wake arbitration** — strict first-detector ownership across rooms; a loud room can steal a not-yet-dispatched wake from a far room (proximity steal); muffled through-wall copies are vetoed when another room hears the same sound ≥1.4× louder
- **Noise-rejection gates** — quiet-source hold (faint audio needs a confident score), appliance hold (continuous background like a robot vacuum requires an overwhelming score), crest-factor gate (dense impacts), clipping-bang gate, own-TTS playback guard
- **Whisper on demand only** — STT runs solely inside an active wake window; ambient utterances never reach Whisper
- **Server-side VAD** — Silero ONNX + energy fallback; adaptive background floor
- **Distant-speech tuned** — corridor/lobby coverage: lowered RMS gates, SpeexDSP noise suppression with "NS rescue" (utterance salvaged from the noise floor), adaptive peak normalisation (up to 20x)
- **Echo guards** — global `GLOBAL_TTS_UNTIL` gate (no STT while any TTS plays — kills ESP32↔camera echo cascade) + duration-proportional mic hold (0.3 s beep → ~0.6 s hold, 20 s TTS → capped 15 s)
- **Dialogue mode** — mic stays open after questions (`?` anywhere in the reply, question words, imperative verbs); returns to standby after statements
- **MCP hardware control** — ESP32/camera tools (screen, volume, LEDs) are requested via `tools/list`; in Nanobot mode device events are forwarded to Nanobot, in Hermes mode the tool list is kept for future use (Hermes chat requests currently carry no `tools` payload)
- **Watchdog** — 30 s timeout → fallback TTS «Простите, я задумалась. Повторите пожалуйста.»
- **Activity monitor** — dims screen to 25% after 30 s idle, closes abandoned sessions after 45 s in LISTENING
- **OTA** — ESP32 firmware handshake returning WS URL + `access_token` + firmware info
- **Emotions** — extracted from LLM text via `[emotion_name]` regex (both backends strip them before TTS)
- **Pluggable LLM backend** — switch the AI brain between **Nanobot** (`nanobot`, WebSocket, streaming) and **Hermes** (`hermes`, OpenAI-compatible `/v1/chat/completions`) with a single env var. Both paths feed the same VAD → STT → TTS pipeline; Hermes replies are sentence-split and streamed through the prefetch TTS player exactly like Nanobot delta text.

## Quick Start

```sh
docker build -t voice_gateway .
docker run -p 18792:18792 \
  -e NANOBOT_WS_URL=ws://nanobot:8765/ \
  -e WHISPER_URL=http://whisper:8000/v1/audio/transcriptions \
  -e TTS_URL=http://edge_tts:5050/v1/audio/speech \
  -e SPEAKER_ID_URL=http://speaker-id:8001/identify \
  voice_gateway
```

ESP32 connects to `ws://gateway:18792/?token=<NANOBOT_TOKEN>` with header `device-id: <MAC>`. The token is mandatory (unauthenticated sockets are closed); `/ota` returns both the WS URL and the current `access_token`.

### Using Hermes instead of Nanobot

Set `LLM_BACKEND=hermes` and point `HERMES_API_URL` at the Hermes API server (OpenAI-compatible). The gateway posts to `${HERMES_API_URL}/v1/chat/completions` and streams the reply through the same TTS pipeline:

```sh
docker run -p 18792:18792 \
  -e LLM_BACKEND=hermes \
  -e HERMES_API_URL=http://hermes:8000 \
  -e HERMES_API_KEY=your-key \
  -e WHISPER_URL=http://whisper:8000/v1/audio/transcriptions \
  -e TTS_URL=http://edge_tts:5050/v1/audio/speech \
  voice_gateway
```

The model is fixed in `backends.py` (`HermesBackend` → `model: "omniroute/oc/hy3-free"`); adjust there if your Hermes deployment exposes a different model id.

### Cameras

Set `CAMERA_STREAMS` to a comma-separated list of stream names registered in go2rtc. The gateway pulls each stream's audio from `rtsp://<GO2RTC_HOST>:8554/<name>?audio=copy` and writes replies back to the camera's WebRTC audio track. Set `DISABLE_CAMERAS=true` to shut the whole camera subsystem off (no RTSP, no ffmpeg, no VAD/wake threads).

### Production deployment (docker compose)

Run via the compose project in `ai-prod` — it sets `network_mode: host`, all required env vars, and bind-mounts `main.py`, `backends.py`, `camera_client.py`, `engine.py`, `audio_utils.py` and `config/` into the container:

```sh
docker compose up -d --no-deps --build voice_gateway
```

Do **not** start the container manually with `docker run` on the default bridge network: the ESP32s reach the gateway at `<host>:18792` from the LAN, and an unpublished bridge container is unreachable (this exact misconfiguration caused a total ESP32 outage). `network_mode: host` also keeps `req.url.hostname` in `/ota` responses correct.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NANOBOT_WS_URL` | `ws://nanobot:8765/` | Nanobot AI agent WebSocket URL |
| `NANOBOT_TOKEN` | `token` | Token appended to Nanobot WS URL |
| `LLM_BACKEND` | `nanobot` | AI brain: `nanobot` (default) or `hermes` |
| `HERMES_API_URL` | `http://192.168.22.102:8000` | Hermes OpenAI-compatible base URL (used when `LLM_BACKEND=hermes`) |
| `HERMES_API_KEY` | `""` | Bearer token sent to Hermes if set |
| `WHISPER_URL` | `http://192.168.22.111:8000/v1/audio/transcriptions` | OpenAI-compatible STT endpoint |
| `TTS_URL` | `http://edge_tts:5050/v1/audio/speech` | OpenAI-compatible TTS endpoint |
| `TTS_VOICE` | `ru-RU-SvetlanaNeural` | TTS voice identifier |
| `TTS_API_KEY` | `""` | Sends `Authorization: Bearer` if set |
| `SPEAKER_ID_URL` | `http://192.168.22.102:8001/identify` | Speaker recognition ([speaker-id](https://github.com/zavaruev/speaker-id) container) |
| `CAMERA_STREAMS` | `""` | Comma-separated go2rtc stream names to attach to |
| `DISABLE_CAMERAS` | `""` | Set to `true`/`1`/`yes` to disable all camera sessions entirely (takes precedence over `CAMERA_STREAMS`) |
| `GO2RTC_HOST` / `GO2RTC_PORT` | `192.168.22.102` / `1984` | go2rtc control host |
| `VAD_SILENCE_FRAMES` | `8` | Silence frames before processing (~60 ms each) |
| `WATCHDOG_TIMEOUT` | `30` | AI response timeout before fallback TTS |
| `STANDBY_TIMEOUT_QUESTION` | `30` | Seconds before standby after a question |
| `STANDBY_TIMEOUT_STATEMENT` | `10` | Seconds before standby after a statement |
| `CHAT_ID_TTL` | `604800` | Chat context lifetime in seconds (7 days) |
| `ENERGY_THRESHOLD` | `0.002` | Energy gate threshold for noise reduction |
| `MIN_SPEECH_RATIO` | `0.12` | Minimum speech ratio to trigger processing |
| `VAD_ADAPTIVE` | `true` | Enable adaptive VAD threshold |
| `THINKING_SOUND_PATH` | `""` | Path to thinking indicator sound |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | `""` | HTTP basic auth for the API |
| `LOG_TRANSCRIPTIONS` | `false` | Log transcriptions |

## REST API

| Path | Method | Description |
|---|---|---|
| `/` | GET | Web UI dashboard (devices + sessions) |
| `/` | WebSocket | Main device gateway (`?token=` required) |
| `/api/devices` | GET | Active WebSocket sessions |
| `/api/devices/config` | GET/POST | Registered device list / register |
| `/api/devices/config/{mac}` | PUT/DELETE | Update / remove device |
| `/mcp/{session_id}` | POST | Send MCP command to device (`"latest"` = most recent session) |
| `/api/camera/tts` | POST | Speak a phrase on a camera session |
| `/ota` | GET/POST | ESP32 OTA handshake; returns WS URL + firmware info |

## Architecture

### STT Pipeline
Opus frames → VAD (Silero) → energy gate → noise reduction → Ogg → Whisper STT + Speaker ID (parallel)

### Camera audio pipeline
RTSP (raw L16 16 kHz) → echo guard (waveform + `_is_echo` cross-correlation vs played-audio ring, delays 2–45 s) → VAD + RMS gates → SpeexDSP NS rescue → adaptive normalisation → **(only after an acoustic wake)** Whisper (+ Speaker ID via Ogg) → Nanobot

### Wake-word detector feed
RTSP 16 kHz → own-echo drop → per-room bidirectional AGC (target peak: kitchen 6500, others 4000) → openWakeWord scoring → gate cascade (see *Wake word & arbitration*)

### Wake word & arbitration
openWakeWord `kompyuter` model (custom-trained, ONNX) on the live 16 kHz feed of every camera. Each chunk is AGC-normalised to the room's target peak, scored, and pushed through a gate cascade before a wake fires:

1. **Own-playback guard** — no fire while this camera's speaker is playing (any wake-shaped sound then is our own echo)
2. **Appliance hold** — 60 s background median rms >800 (robot vacuum, hood…) requires a single-chunk score ≥0.92
3. **Quiet-source hold** — own signal level <3000 requires score ≥0.72 (faint TV/muffled speech scores deceptively high on the TTS-trained model)
4. **Distant-source veto** — level <1600 while another camera hears the same sound ≥1.4× louder → the wake belongs to that room
5. **Clipping bang / crest-factor gates** — door slams and dense impacts (peak >24k or rms·2 > peak) are ignored unless overwhelming
6. **Debounce** — 2 qualifying chunks out of the last 3 (~240 ms), or one confident chunk ≥0.68
7. **Cross-camera arbiter** — first detector becomes interaction owner; others stand down. A room ≥5× louder than the owner steals a not-yet-dispatched wake so the answer sounds where the user actually is

After a fire: attention pip, VAD state reset, Whisper listens until the command is dispatched (or the ~60 s window expires).

### TTS Pipeline
Nanobot text → sentence splitter → **prefetch pipeline** (sentence N+1 is synthesised while N plays, hiding Edge-TTS latency) → Edge TTS (MP3) → decode → resample (16/24 kHz) → Opus for ESP32 / PCM for cameras → paced 60 ms chunks. Every played clip is registered in the echo-reference ring and extends wake suppression past the delayed-echo window.

### LLM Backend (Nanobot / Hermes)

The AI brain is selected by `LLM_BACKEND` and implemented as a `BaseLLMBackend` (`backends.py`):

- **`NanobotBackend`** (default) — opens a WebSocket to `NANOBOT_WS_URL?token=…&chat_id=…`, streams `text` deltas, handles `[thinking]` blocks, and pushes sentence-split replies into the response queue.
- **`HermesBackend`** — streams the user message to `${HERMES_API_URL}/v1/chat/completions` (OpenAI-compatible SSE, `stream: true`); each `choices[].delta.content` chunk is accumulated and flushed sentence-by-sentence on `. ! ? …` into the response queue. `[thinking]` reasoning blocks and `[emotion_name]` tags are stripped before TTS, identical to `NanobotBackend`. Streaming keeps first-token latency low so the prefetch TTS player starts before the full reply arrives (avoiding the 30 s watchdog fallback).

Both backends expose the same `generate_response(text, session_id, stream_name, response_queue)` contract. The ESP32 path dispatches via `_dispatch_hermes`/`_hermes_player_task`; camera sessions receive the backend instance at construction and call `_call_backend` → `_nanobot_player_task` (the player is backend-agnostic). Either way replies reach the prefetch TTS player, so sentence-level latency hiding works identically for both brains.

### Dialogue Mode
- AI response contains `?` (or Russian question patterns) → mic stays open
- AI response is a statement → returns to standby
- Hold phrases extend listening

### Binary Frame Formats
| Version | Format |
|---|---|
| v1 | Raw Opus frame |
| v2 | 16-byte header + Opus frame |
| v3 | 4-byte header + Opus frame |

## Device Configuration

Create `config/devices.json` (auto-created as `{}` if missing):

```json
{
  "aa:bb:cc:dd:ee:ff": {
    "friendly_name": "Kitchen Speaker",
    "allowed": true
  }
}
```

Device identifies itself via `device-id` header (fallback: `mac` header). MAC keys are normalized to upper case on lookup — keep a single entry per device.

## Known Issues & Current Problems

- **go2rtc zombie RTSP sessions** — go2rtc 1.9.2 on the streams host leaks sessions when a camera link is slow; 17–23 half-dead connections with 100–290 KB send queues can stall a camera's majestic until it stops serving :554. Mitigated by a per-camera watchdog (`/etc/watchdog_majestic.sh` + crond) that restarts majestic on dead/blocked RTSP, but the real fix is upgrading/restarting go2rtc on `192.168.22.102`.
- **Camera WiFi links** — the kitchen camera's link quality fluctuates (21–34/100 vs 80+ elsewhere); its video stream was reduced to fps 10 / bitrate 1024 to keep the audio backchannel stable.
- **Echo cascade ESP32 ↔ camera is only partially solved** — `GLOBAL_TTS_UNTIL` + duration-proportional mic hold are band-aids. Camera-speaker echo returns via WebRTC with 8–15 s delay; `_is_echo` cross-correlates mic chunks against a reference ring of recently played audio (delays 2–45 s) and confirmed echoes extend wake suppression.
- **TTS-trained wake model prefers muffled audio** — through-wall copies of «компьютер» can out-score close live speech; the distant-source veto and quiet-source hold compensate, but retraining on real in-room recordings (v2–v4 attempts degraded discrimination — keep v1) remains the proper fix.
- **Gates are room-calibrated** — thresholds (wake 0.47/0.52, hold/veto levels, bg-median 800) were tuned against measured score distributions in three specific rooms. They will not generalise to other rooms without recalibration.
- **Whisper retry at temperature 0.5** — empty/`unknown` transcripts trigger a noisier re-transcription; on some engines this doubles STT latency in the worst case.
- **`Dockerfile` exposes 8080 but nothing listens on it** (18792 is the only real port).
- **`setup_gateway.sh` is an outdated snapshot** — not authoritative.
- **Vosk models downloaded but unused** — `config/vosk-model-ru-0.42/` (3.5 GB) candidate for a local low-latency STT fallback; not wired into the pipeline.

## Tests

Host Python usually lacks the runtime deps (`opuslib`, `onnxruntime`, …), and the image has no `pytest` — run the suite inside the running container:

```sh
docker exec voice_gateway pip install -q pytest httpx pytest-asyncio
docker cp tests voice_gateway:/tmp/vg_tests
docker exec voice_gateway python3 -m pytest /tmp/vg_tests -q
docker exec voice_gateway rm -rf /tmp/vg_tests
```

Covers engine wake scoring/gates, camera arbitration helpers, OTA auth, and RMS utilities (~1,160 lines).

## Dependencies

- Python 3.12+
- FFmpeg (RTSP capture, WAV/OGG conversion)
- Silero VAD ONNX model (downloaded at build time)
- openWakeWord + custom `computer.onnx` / `embedding_model.onnx` (baked into the image)
- SpeexDSP noise suppression (`speexdsp-ns`)
- External services: Whisper STT, Nanobot (or Hermes), Edge TTS, Speaker ID, go2rtc (for cameras)

## Changelog

- **2.24** — Hermes-mode hardening + camera kill-switch.
  - The per-device Nanobot listener no longer starts when `LLM_BACKEND=hermes` (previously it retried `localhost:8765` every 10 s forever, spamming the log; the Hermes reply path never used that socket).
  - `DISABLE_CAMERAS=true|1|yes` is now honored by `start_camera_sessions()` (previously the flag existed in compose but was ignored, so cameras ran anyway).
  - `HermesBackend` sends `X-Source: voice_gateway`, `X-Device-MAC` and `X-Stream-Name` headers for per-device routing upstream.
  - Removed the stale `(managed by nanobot in v0.3.0)` suffix from the MCP tools log line; deduped the MAC entry in `config/devices.json`.
- **2.8** — Pluggable LLM backend. Added `backends.py` with `BaseLLMBackend`, `NanobotBackend` (moved out of `main.py`/`camera_client.py`), and `HermesBackend` (OpenAI-compatible `/v1/chat/completions`). Select via `LLM_BACKEND` (`nanobot` | `hermes`); Hermes configured with `HERMES_API_URL` / `HERMES_API_KEY`. Same VAD→STT→TTS pipeline, emotion tags, watchdog and echo guards apply to both backends.
