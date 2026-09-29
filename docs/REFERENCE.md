# Voice Gateway — Reference

> Back to the project overview: [`../README.md`](../README.md). This file is the detailed reference: configuration tables, REST API, architecture notes, known issues and the changelog.

> **Version 2.34** — L2 escalation context: both turns lost in the 28.09 ESP32 dialogue were answered blind (the router sent L2 the bare utterance). The router now hands over the near-miss device hint («кашеварку» → «кофеварка», `resolver.unresolved_hint`) and the satellite's last finished turns (`services/jev-router/history.py`), so a pronoun resolves instead of inventing a device; the agent also gets one bounded retry carrying the recorded tool error whenever the honesty veto fires. Pinned by `tests/test_history.py` plus hint and retry-note cases — 177 tests across 12 files.

WebSocket gateway bridging [Xiaozhi ESP32](https://github.com/78/xiaozhi-esp32) smart speakers **and WebRTC/IP cameras** to an AI backend (**Cascade** 3-level router, **[Nanobot](https://github.com/HKUDS/nanobot)** or **Hermes**) with real-time speech processing.

```
ESP32 (Opus WS)  ──┐
                   ├──▶ Voice Gateway ──STT text──▶ AI backend (Cascade | Nanobot | Hermes)
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
3. **openWakeWord** listens for the wake word «компьютер» on camera streams (library head `computer_20260706_130638.onnx`, custom `computer.onnx` in the kitchen)
4. **Whisper STT** transcribes audio to text (Russian; the camera path retries at temperature 0.0 → 0.5, the ESP32 path sends a single 0.0 attempt)
5. **Speaker ID** identifies the speaker (parallel with STT)
6. **LLM backend** (Cascade *or* Nanobot *or* Hermes) processes text, returns response with optional emotions `[emotion_name]`
7. **Edge TTS** synthesizes speech; Opus streamed to ESP32, PCM queued to camera speakers

## Features

- **ESP32 + camera support** — both device classes in one gateway; cameras use WebRTC (go2rtc) with the mic feed taken from the RTSP audio backchannel (`ffmpeg`, raw L16, no μ-law quantisation)
- **go2rtc self-healer** — after a camera reboot go2rtc can serve video-only SDP and ffmpeg stalls; 3 consecutive stalls (`GO2RTC_HEAL_STALLS`) trigger a stream re-register using `GO2RTC_SOURCE_URL[_<NAME>]`, rate-limited to once per minute
- **Wake word «компьютер»** — openWakeWord on the live 16 kHz camera feed: library head `config/computer_20260706_130638.onnx` (Creator #7074, Classic V3) everywhere except the kitchen, which keeps the custom `config/computer.onnx` (the library head scores its clipped audio at 0.001); per-room base thresholds **kitchen 0.40, others 0.30**, sliding-window debounce (2 qualifying chunks out of the last 3), single-chunk bypass at score ≥0.68, attention pip
- **Bidirectional AGC** — every oww chunk is normalised to a per-room target peak (**kitchen 6500, corridor 9000, others 4000**) in *both* directions; quiet far-room copies are boosted while loud close-range speech is scaled down, so all rooms feed the model equally loud audio
- **Wake arbitration** — strict first-detector ownership across rooms; a loud room can steal a not-yet-dispatched wake from a far room (proximity steal); muffled through-wall copies are vetoed when another room hears the same sound ≥1.4× louder
- **Noise-rejection gates** — quiet-source hold (faint audio needs a confident score), appliance hold (continuous background like a robot vacuum requires a top score ≥0.60), crest-factor gate (dense impacts), clipping-bang gate, own-TTS playback guard, ambiguous-zone STT confirm
- **STT confirm window** — when a gate holds a plausible score instead of a pip, the next VAD utterance inside the 10 s window goes to Whisper; «компьютер» in the transcript fires the real wake, garbage expires silently
- **Auto-greeting** — a bare wake with no command within 5 s (11.5 s from an STT-confirm window) speaks a local «Да?» instead of polling the LLM; a repeat inside 60 s with no transcript is forced through STT confirmation
- **Whisper on demand only** — STT runs solely inside an active wake window or an opened STT-confirm window; ambient utterances never reach Whisper
- **Server-side VAD** — Silero ONNX + energy fallback; adaptive background floor
- **Distant-speech tuned** — corridor/lobby coverage: lowered RMS gates, SpeexDSP noise suppression with "NS rescue" (utterance salvaged from the noise floor), adaptive peak normalisation (up to 20x)
- **Echo guards** — global `GLOBAL_TTS_UNTIL` gate (no STT while any TTS plays — kills ESP32↔camera echo cascade) + duration-proportional mic hold (0.3 s beep → ~0.6 s hold, 20 s TTS → capped 15 s)
- **Dialogue mode** — mic stays open when the reply *ends* with `?` (ASCII or fullwidth), contains the «повторите пожалуйста» apology, or matches a Russian interrogative/imperative anywhere in the text; returns to standby after plain statements
- **MCP hardware control** — ESP32/camera tools (screen, volume, LEDs) are requested via `tools/list`; in Nanobot mode device events are forwarded to Nanobot, in Hermes/Cascade mode the tool list is kept for future use (neither Hermes nor the router sends a `tools` payload)
- **Cross-device speaker lock** — two satellites hearing the same person at once: a 30 s per-`uid` lock lets one session run the pipeline, the duplicate is dropped (no double LLM answer)
- **Watchdog** — `WATCHDOG_TIMEOUT`, default **90 s** → fallback TTS «Простите, я задумалась. Повторите пожалуйста.»
- **Activity monitor** — dims screen to 25 % after **10 s** idle (any active status re-arms `dim_sent`); **never closes the connection** — MCP tools keep working while the ESP32 is on. Adaptive standby opens separately: `STANDBY_TIMEOUT_QUESTION` 30 s after a question, `STANDBY_TIMEOUT_STATEMENT` 10 s after a statement
- **OTA** — ESP32 firmware handshake returning WS URL + `access_token` + firmware info; `POST /api/firmware/upload` stores a `.bin` (capped by `MAX_FIRMWARE_SIZE`, default 10 MB, path-traversal-safe) and refreshes `firmware.json`, `GET /api/firmware` reads it back
- **Emotions** — extracted from LLM text via `[emotion_name]` regex (Nanobot/Hermes strip them before TTS; router output in cascade mode carries none)
- **Pluggable LLM backend** — switch the AI brain between **Nanobot** (`nanobot`, WebSocket, streaming), **Hermes** (`hermes`, OpenAI-compatible `/v1/chat/completions`) and **Cascade** (`cascade`, 3-level router) with a single env var. All paths feed the same VAD → STT → TTS pipeline; replies are sentence-split and streamed through the prefetch TTS player exactly like Nanobot delta text.
- **Cascade AI (3 levels)** — `LLM_BACKEND=cascade` routes every utterance through **jev-router** (semantic classifier on local Ollama embeddings, 5 routes: `easy_action`/`easy_query`/`general_qa`/`expert`/`complex_logic`). Easy routes resolve slots offline and call Home Assistant MCP directly; general chat streams from the OmniRoute combo; expert goes to Hermes; complex logic escalates to the **smolagents-worker** (CodeAgent with HA/memory/expert tools, 120 s cap + progress heartbeats). Every failure or ambiguity escalates to L2 — never a wrong side-effect. Qdrant `voice_turns`/`voice_facts` store dialogue memory (written by the router, read by the L2 tool).

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

### Using Cascade (3-level AI router)

Set `LLM_BACKEND=cascade` (plus `ROUTER_URL`, default `http://localhost:8091`) and run the two companion services — they are part of the `ai-prod` compose project and run with `network_mode: host`:

```sh
docker compose up -d --no-deps --build jev-router smolagents-worker voice_gateway
```

- **L1 `jev-router`** (port 8091) — semantic route classifier (local Ollama embeddings) + offline slot resolver. Easy commands execute directly against Home Assistant (~0.1 s), state queries are answered from the live registry, `general_qa` streams from OmniRoute, `expert` hits Hermes first (OmniRoute failover), weather is built from open-meteo (Hermes on API failure, L2 as last resort); anything ambiguous or failed escalates to L2 with decoded error context **plus two dialogue aids** — a near-miss device hint (`resolver.unresolved_hint`, difflib over the THING dictionary: «кашеварку» → «кофеварка», 0.70 threshold, only on the exact word that made the resolver bail) and the satellite's last 4 finished turns (`history.py`, TTL 10 min) so pronouns like «выключи её» have an antecedent. Neither resolves anything by itself, they only ride along with the escalation.
- **L2 `smolagents-worker`** (port 8092) — smolagents `CodeAgent` (`ha_action`/`ha_read`/`qdrant_search`/`hermes_expert`/`weather_forecast`/`get_datetime`), 120 s cap with progress heartbeats, and programmatic honesty vetoes: an answer claiming success after only failed tool calls is replaced with the recorded truth, and a spoken forecast is checked against the recorded `weather_forecast` output. A veto also triggers **one bounded retry** fed with the raw recorded errors (`honesty.failure_note()`) and the same history — a second veto just speaks the truth, so the retry can never loop.

External dependencies (env, see `services/*/config.py`): Ollama `qwen3-embedding`, Qdrant, OmniRoute combo, Home Assistant MCP.

Rollback to a single backend: `LLM_BACKEND=hermes` (or `nanobot`) + `docker compose up -d --no-deps --build voice_gateway`.

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
| `NANOBOT_TOKEN` | `""` | WS auth token; appended as `?token=`. **Must be non-empty or every socket is rejected** (deployments set it to `token`) |
| `NANOBOT_SESSION_SALT` | `""` | Salt mixed into the deterministic per-MAC `chat_id` so session ids can't be guessed from a MAC |
| `LLM_BACKEND` | `nanobot` | AI brain: `nanobot` (default), `hermes` or `cascade` |
| `HERMES_API_URL` | `http://192.168.22.102:8000` | Hermes OpenAI-compatible base URL (used when `LLM_BACKEND=hermes`) |
| `HERMES_API_KEY` | `""` | Bearer token sent to Hermes if set |
| `ROUTER_URL` | `http://localhost:8091` | jev-router SSE endpoint (used when `LLM_BACKEND=cascade`) |
| `ROUTER_ACK_DELAY` | `3` | Seconds before the fallback ack «Секунду…» when the first sentence is slow (chat/easy routes; complex/expert ack immediately) |
| `WHISPER_URL` | `http://192.168.22.111:8000/v1/audio/transcriptions` | OpenAI-compatible STT endpoint |
| `TTS_URL` | `http://edge_tts:5050/v1/audio/speech` | OpenAI-compatible TTS endpoint |
| `TTS_VOICE` | `ru-RU-SvetlanaNeural` | TTS voice identifier |
| `TTS_MODEL` | `tts-1` | `model` field posted to the TTS endpoint |
| `TTS_API_KEY` | `""` | Sends `Authorization: Bearer` if set |
| `SPEAKER_ID_URL` | `http://192.168.22.102:8001/identify` | Speaker recognition ([speaker-id](https://github.com/zavaruev/speaker-id) container) |
| `CAMERA_STREAMS` | `""` | Comma-separated go2rtc stream names to attach to |
| `DISABLE_CAMERAS` | `""` | Set to `true`/`1`/`yes` to disable all camera sessions entirely (takes precedence over `CAMERA_STREAMS`) |
| `GO2RTC_HOST` / `GO2RTC_PORT` | `192.168.22.102` / `1984` | go2rtc control host |
| `GO2RTC_SOURCE_URL` | `""` | RTSP source URL the self-healer re-registers with (`GO2RTC_SOURCE_URL_<NAME>` per stream). Without it the healer falls back to whatever URL go2rtc still lists |
| `GO2RTC_HEAL_STALLS` | `3` | Consecutive ffmpeg stalls (~20 s read timeout each) before a stream re-register |
| `WAKE_WORD` | `компьютер` | Spoken phrase stripped from the transcript to expose the command |
| `WAKE_WORD_MODEL` | `config/computer_20260706_130638.onnx` | openWakeWord head; bare package names resolve inside the package. `WAKE_WORD_MODEL_<NAME>` overrides per stream (kitchen defaults to `config/computer.onnx`) |
| `VAD_SILENCE_FRAMES` | `8` | Silence frames before processing (~60 ms each; ESP32 path only — the camera path hardcodes 10) |
| `WATCHDOG_TIMEOUT` | `90` | AI response timeout before fallback TTS |
| `STANDBY_TIMEOUT_QUESTION` | `30` | Seconds before standby after a question |
| `STANDBY_TIMEOUT_STATEMENT` | `10` | Seconds before standby after a statement |
| `CHAT_ID_TTL` | `604800` | Chat context lifetime in seconds (7 days) |
| `ENERGY_THRESHOLD` | `0.002` | Energy gate threshold for noise reduction |
| `MIN_SPEECH_RATIO` | `0.12` | Minimum speech ratio to trigger processing |
| `VAD_ADAPTIVE` | `true` | **Declared for compatibility — read but unused** |
| `THINKING_SOUND_PATH` | `""` | **Declared for compatibility — unused in `main.py`** |
| `MAX_FIRMWARE_SIZE` | `10485760` | Byte cap for `/api/firmware/upload` (10 MB) |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | `""` | HTTP basic auth for the API (empty = auth disabled) |
| `LOG_TRANSCRIPTIONS` | `false` | Log transcriptions |

## REST API

| Path | Method | Description |
|---|---|---|
| `/` | GET | Web UI dashboard (devices + sessions) |
| `/health` | GET | Liveness probe, no auth: `{"status":"ok"}` |
| `/` | WebSocket | Main device gateway (`?token=` required) |
| `/api/devices` | GET | Active WebSocket sessions |
| `/api/devices/config` | GET/POST | Registered device list / register |
| `/api/devices/config/{mac}` | PUT/DELETE | Update / remove device |
| `/mcp/{session_id}` | POST | Send MCP command to device (`"latest"` = most recent session) |
| `/api/tts` | POST | Speak `text` on a connected device (`session_id: "latest"` = most recent; non-admin may only target devices they own) |
| `/api/camera/tts` | POST | Speak a phrase on a camera session |
| `/api/firmware/upload` | POST | Upload an ESP32 `.bin` (multipart `file` + `version`, capped by `MAX_FIRMWARE_SIZE`) and refresh `firmware.json` |
| `/api/firmware` | GET | Current firmware metadata (`firmware.json`) |
| `/ota` | GET/POST | ESP32 OTA handshake; returns WS URL + firmware info |

## Architecture

### STT Pipeline
Opus frames → VAD (Silero) → energy gate → noise reduction → Ogg → Whisper STT + Speaker ID (parallel)

### Camera audio pipeline
RTSP (raw L16 16 kHz) → echo guard (waveform + `_is_echo` cross-correlation vs played-audio ring, delays 2–45 s) → VAD + RMS gates → SpeexDSP NS rescue → adaptive normalisation → **(only after an acoustic wake)** Whisper (+ Speaker ID via Ogg) → LLM backend

### Wake-word detector feed
RTSP 16 kHz → own-echo drop → per-room bidirectional AGC (target peak: kitchen 6500, corridor 9000, others 4000) → openWakeWord scoring → gate cascade (see *Wake word & arbitration*)

### Wake word & arbitration
openWakeWord on the live 16 kHz feed of every camera — library head `computer_20260706_130638.onnx` except the kitchen, which keeps the custom `computer.onnx`. Each chunk is AGC-normalised to the room's target peak, scored, and pushed through a gate cascade before a wake fires:

1. **Own-playback guard** — no fire while this camera's speaker is playing (any wake-shaped sound then is our own echo)
2. **Appliance hold** — 60 s background median rms >800 (robot vacuum, hood…) holds the wake unless the top score reaches **0.60**; a held score in **[0.55, 0.60)** opens an STT-confirm window rather than being dropped
3. **Quiet-source hold** — own signal level <3000 requires **0.58 below level 2000 / 0.55 above** (faint TV/muffled speech scores deceptively high on the TTS-trained model); a held score in **[tier−0.05, tier)** opens an STT-confirm window
4. **Distant-source veto** — level <3000 while another camera hears the same sound ≥1.4× louder → the wake belongs to that room; the veto sticks for 5 s
5. **Clipping bang / crest-factor gates** — door slams and dense impacts (peak >24k, or rms·2 > peak above peak 4000) are ignored unless the score reaches 0.85
6. **Debounce** — 2 qualifying chunks out of the last 3 (~240 ms), or one confident chunk ≥0.68
7. **Ambiguous zone** — top score <0.55 (or a repeat within 60 s of an ignored auto-greet) with level <3000 → no pip; the next utterance goes to Whisper for confirmation
8. **Cross-camera arbiter** — first detector becomes interaction owner; others stand down. A room ≥5× louder than the owner steals a not-yet-dispatched wake so the answer sounds where the user actually is

After a fire: attention pip, VAD state reset, Whisper listens until the command is dispatched (or the **15 s** wake window expires — a bare wake with no command within 5 s speaks a local «Да?» instead of polling the LLM).

### TTS Pipeline
Backend text → sentence splitter → **prefetch pipeline** (sentence N+1 is synthesised while N plays, hiding Edge-TTS latency) → Edge TTS (MP3) → decode → resample (16/24 kHz) → Opus for ESP32 / PCM for cameras → paced 60 ms chunks. Every played clip is registered in the echo-reference ring and extends wake suppression past the delayed-echo window.

### LLM Backend (Nanobot / Hermes / Cascade)

The AI brain is selected by `LLM_BACKEND` and implemented as a `BaseLLMBackend` (`backends.py`):

- **`NanobotBackend`** (default) — opens a WebSocket to `NANOBOT_WS_URL?token=…&chat_id=…`, streams `text` deltas, handles `[thinking]` blocks, and pushes sentence-split replies into the response queue.
- **`HermesBackend`** — streams the user message to `${HERMES_API_URL}/v1/chat/completions` (OpenAI-compatible SSE, `stream: true`); each `choices[].delta.content` chunk is accumulated and flushed sentence-by-sentence on `. ! ? …` into the response queue. `[thinking]` reasoning blocks and `[emotion_name]` tags are stripped before TTS, identical to `NanobotBackend`. Streaming keeps first-token latency low so the prefetch TTS player starts before the full reply arrives (avoiding the 90 s watchdog fallback).
- **`CascadeBackend`** — POSTs the text to `${ROUTER_URL}/route` (SSE) and replays router events into the same queue: `sentence`/`progress` are spoken (emotion tags stripped, no monologue gate — the router output is already curated), `route` events drive the ack timer (immediate «Секунду, занимаюсь…» for `complex_logic`/`expert`, `ROUTER_ACK_DELAY`-second fallback for chat/easy), `error`/empty streams produce an immediate apology so the user is never left in silence. The first played audio (usually the ack) cancels the watchdog downstream, which is what makes 48–120 s L2 turns possible.

Both backends expose the same `generate_response(text, session_id, stream_name, response_queue)` contract. The ESP32 path dispatches via `_dispatch_hermes`/`_hermes_player_task`; camera sessions receive the backend instance at construction and call `_call_backend` → `_nanobot_player_task` (the player is backend-agnostic). Either way replies reach the prefetch TTS player, so sentence-level latency hiding works identically for both brains.

### Dialogue Mode
- Reply **ends with `?`** (ASCII or fullwidth), or contains the «повторите пожалуйста» apology → mic stays open when playback drains
- Reply matches a Russian interrogative/imperative anywhere in the text (`HAS_QUESTION_WORDS_RE`: что, как, где, включи, выключи, расскажи, повтори…) → same follow-up window
- Reply is a plain statement → returns to standby (`STANDBY_TIMEOUT_STATEMENT`, 10 s)
- Hold phrases («подожди»…) extend listening without dispatching to the LLM

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
- **Gates are room-calibrated** — thresholds (wake 0.40 kitchen / 0.30 elsewhere, hold/veto levels, bg-median 800) were tuned against measured score distributions in three specific rooms. They will not generalise to other rooms without recalibration.
- **Whisper retry at temperature 0.5 is camera-only** — the camera path (`_fetch_transcription`) retries 0.0 → 0.5 on an empty transcript, doubling STT latency in the worst case; the ESP32 path (`fetch_transcription`) sends a single attempt at 0.0 and never retries.
- **`Dockerfile` exposes 8080 but nothing listens on it** (18792 is the only real port).
- **`setup_gateway.sh` is an outdated snapshot** — not authoritative.
- **Vosk models downloaded but unused** — `config/vosk-model-ru-0.42/` (3.5 GB) candidate for a local low-latency STT fallback; not wired into the pipeline.

## Tests

Host Python usually lacks the runtime deps (`opuslib`, `onnxruntime`, …), and the image has no `pytest` — run the suite inside the running container:

```sh
docker exec voice_gateway pip install -q pytest httpx pytest-asyncio
docker exec voice_gateway rm -rf /tmp/vg_src && docker exec voice_gateway mkdir -p /tmp/vg_src
for f in audio_utils.py backends.py camera_client.py engine.py main.py; do docker cp "$f" voice_gateway:/tmp/vg_src/"$f"; done
docker cp services voice_gateway:/tmp/vg_src/services
docker cp templates voice_gateway:/tmp/vg_src/templates
docker cp tests voice_gateway:/tmp/vg_src/tests
docker exec -w /tmp/vg_src voice_gateway python3 -m pytest tests -q
docker exec voice_gateway rm -rf /tmp/vg_src
```

Tests must run against a copy of the sources, not the live `/app` mounts: the three router tests import `honesty`/`classifier`/`weather` from `services/jev-router`, which is not part of the image (it runs as its own container), so the sources, `services/` and `tests/` have to travel together.

Covers 177 tests across 12 files (2,458 lines): engine wake scoring/gates, camera arbitration, the wake-gate cascade's STT-confirm bands, the ESP32 `main.py` protocol, cascade backend streaming, honesty vetoes + the retry note built from them, router slot resolution and the escalation hint, the short-term dialogue ring, weather, TTS gate, OTA auth and RMS utilities.

## Dependencies

- Python 3.12+
- FFmpeg (RTSP capture, WAV/OGG conversion)
- Silero VAD ONNX model (downloaded at build time)
- openWakeWord + the library head `computer_20260706_130638.onnx` and the custom `computer.onnx` / `embedding_model.onnx` (baked into the image)
- SpeexDSP noise suppression (`speexdsp-ns`)
- External services: Whisper STT, Edge TTS, Speaker ID, go2rtc (for cameras), plus one of the backends — Nanobot; Hermes; or (cascade) Ollama embeddings + Qdrant + OmniRoute + Home Assistant MCP

## Changelog

- **2.34** — L2 escalation context: the two turns lost in the 28.09 ESP32 dialogue were answered blind (the router escalated `resolver_ambiguous` with an EMPTY `context`), the third was a tool failure the model then claimed as success.
  - **What went wrong** — «Выключи кашеварку» (one STT-corrupted word) reached L2 bare, so the model invented `{"name": "кашеварка"}` → `MatchFailedError(NAME)` → honesty veto → «Не нашла такого устройства». «Выключи её» had the model search long-term Qdrant for «устройства женского рода» and call `HassTurnOff(area=спальня, domain=["light"])` — it nearly switched off a bedroom lamp instead of the coffee maker, saved only by the RU-vs-display-name mismatch.
  - **Near-miss device hint (`resolver.unresolved_hint`)** — difflib match of the unknown word against every THING stem *and* exact name (0.70 threshold: «кашеварку» scores 0.875, unrelated household words ≤0.667) → the prompt line `Похожее устройство: «кофеварка». Если это оно, вызывай ha_action`. Four guards keep it off every other escalation path: a command verb present, `_DEVICE_NOUN` absent (the exact test that made the resolver bail), no negation, no candidate the dictionary already knows. The resolver stays conservative — the hint rides with the escalation, it never resolves.
  - **Short-term dialogue ring (`services/jev-router/history.py`)** — last 4 finished turns per satellite, TTL 10 min, 32-stream cold eviction; pushed by `app._handle` after every answer (fast path included) and read BEFORE the turn mutates it, so a Qdrant write that lands after the turn can never be mistaken for an antecedent. Handed to L2 as the «Предыдущие реплики» block.
  - **Bounded retry after the honesty veto (`smolagents-worker app._run_agent`)** — veto fired == the model believed it acted and only the tool disagreed, so ONE more pass gets `honesty.failure_note()` (the RAW recorded errors, not the speakable truth) plus the same history; a second veto speaks the truth (`retry=True`) and stops. Costs ~+4–5 s on failed turns only; successful turns are untouched.
  - **Tests** — new `tests/test_history.py` (8: ring order, per-satellite isolation, cap, TTL, cold eviction), 7 hint cases in `test_router_resolution.py`, 3 `failure_note` cases in `test_honesty.py`; suite 159 → **177 tests across 12 files (2,458 lines)**.
  - **E2E (live router + worker)** — «Выключи кашеварку» → hint visible in the task → `HassTurnOff(name=кофеварка)` confirmed → «Хорошо, выключаю кофеварку» 4.0 s (was a lost turn); «Включи кофеварку» 0.23 s fast path → «Выключи её» → history block present → **«Выключила кофеварку» 3.2 s** (was a bedroom-lamp attempt); «Выключи микроволновку» → model claimed success after `MatchFailedError` → veto → bounded retry → honest «Не нашла такого устройства» 14.2 s.

- **2.33** — Wake-gate fix: the two STT-confirm rescue bands that the threshold recalibration had left unreachable are live again; no other behaviour changed.
  - **Appliance hold** — the guard admits `max < 0.60` under bg median >800, but its rescue branch still tested `top >= 0.85`, a leftover of the pre-recalibration 0.92 bar, so a held 0.55–0.59 «компьютер» was dropped silently. The band is now **[0.55, 0.60)** — everything this guard can hold that is still worth a Whisper arbitration (0.55 is the same bar the ambiguous zone uses; at or above 0.60 the wake already fires).
  - **Quiet-source hold** — the guard admits `max < 0.58/0.55` while its rescue tested `>= 0.60` *after* `_ww_recent.clear()`, dead since `bfc328f` moved the tier down from 0.72/0.85. The band is now **[tier−0.05, tier)**, i.e. `[0.53, 0.58)` below level 2000 and `[0.50, 0.55)` above it; the tier is computed once (`q_tier`) so guard, log line and band cannot drift apart again.
  - **`_open_stt_confirm()` is idempotent** — the appliance hold clears the score ring, so on a quiet-room chunk at bg >800 both gates hit their bands; the second call now returns instead of refreshing the 10 s deadline and stacking a second 11.5 s expiry watcher over the first.
  - **Stale log fixed** — the appliance-hold line printed `needs >=0.95` while the guard it belongs to enforces 0.60.
  - **New `tests/test_wake_gates.py`** — 10 end-to-end tests driving `_vad_process` with the engine, utterance VAD and arbiter stubbed: both rescue bands, the pass-through above each guard, the drop below each band, and the single-window guarantee. Suite 149 → **159 tests across 11 files (2,308 lines)**; the five band tests fail against the pre-fix code, which is what holds the regression.
  - **Docs** — the two Known Issues (unreachable branches, stale log wording) and the AGENTS gotchas derived from them are gone; README's own test recipe was also wrong (`docker cp tests … pytest /tmp/vg_tests` failed with `ModuleNotFoundError` on `honesty`/`classifier`/`weather`) and now copies the sources and `services/` together.

- **2.32** — Documentation audit: README and AGENTS.md re-synced with the code; no code changes.
  - **Corrected stale values (both files)** — wake thresholds `0.47/0.52` → **kitchen 0.40 / others 0.30** (`camera_client.py:570`, clamped back to base in `_call_backend`/`_call_nanobot`); wake model "custom `computer.onnx`" → default is the **library head `computer_20260706_130638.onnx`** with the custom head kept only for kitchen; AGC targets `kitchen 6500, others 4000` → **corridor 9000 added** (`camera_client.py:1377`); gate cascade appliance `≥0.92` → **0.60** (`f38d73e`), quiet-source `≥0.72` → **0.58/0.55** (`bfc328f`), distant-veto `<1600` → **`<3000`** (`03371ea`); `WATCHDOG_TIMEOUT` `30` → **90** (since v2.25); wake window `~60 s` → **15 s** (`CameraConfig.wake_timeout`); activity monitor "dim after 30 s, close sessions after 45 s" → **dim after 10 s, connections are never closed**; dialogue mode "`?` anywhere" → **trailing `?` only** (plus apology/interrogative words anywhere); Whisper retry → **camera path only** (the ESP32 path is a single attempt); tests `~1,160 lines` → **10 files / 2,087 lines / 149 tests**; `NANOBOT_TOKEN` default `token` → `""` (deployments set it).
  - **Added missing REST endpoints** — `/api/tts` (POST, ownership-checked device TTS), `/api/firmware/upload` (POST, `MAX_FIRMWARE_SIZE`-capped), `/api/firmware` (GET), plus `/health` in AGENTS.md.
  - **Added missing env vars** — `TTS_MODEL`, `MAX_FIRMWARE_SIZE`, `NANOBOT_SESSION_SALT`, `WAKE_WORD`, `WAKE_WORD_MODEL[_<NAME>]`, `GO2RTC_SOURCE_URL[_<NAME>]`, `GO2RTC_HEAL_STALLS`; flagged `VAD_ADAPTIVE` and `THINKING_SOUND_PATH` as declared-but-unused instead of functional.
  - **Documented five previously undocumented features** — go2rtc self-healer (`_heal_go2rtc_stream`, stall-triggered stream re-register), the STT-confirm window (`_open_stt_confirm`: a held score routes the next utterance to Whisper instead of pipping), auto-greeting «Да?» on a bare wake, the cross-device speaker lock (`_check_speaker_lock`, 30 s per-uid TTL), and firmware upload.
  - **Recorded two known bugs found during the audit** — two of the three `_open_stt_confirm()` call sites are unreachable (appliance branch tests `≥0.85` behind a `<0.60` guard; quiet-source branch tests `≥0.60` after `_ww_recent.clear()` behind a `<0.55/0.58` guard), and the appliance-hold log still prints `needs >=0.95`. Both are documented as Known Issues; the stale *code* comments (`camera_client.py` lines 556, 1511, 1562, 2169) and the dead branches were deliberately left for a separate code change.
- **2.31** — The assistant keeps the mic open, keeps its own audio protocol consistent, and the code is fully documented in English.
  - **Follow-up window (`main.py`)** — the cascade/hermes path used to end every turn with an unconditional `reset_to_standby()`, so a reply that ENDED in a question went IDLE: the ESP32's post-TTS `listen:start` was rejected by the 1.5 s TTS cooldown or found the session IDLE, and every frame was dropped — the user had to repeat the wake word to answer the assistant's own question. `_reply_has_question()` is now the single source of truth (trailing `?`/`？`, the «повторите пожалуйста» apology, interrogative words) and `_finalize_turn_followup()` is the one post-turn transition shared by both backends: a question opens the 30 s dialogue window (screen 100 %), a statement opens the 10 s window and disarms the pending watchdog. `_wake_audio_received` is finally written (it was read but never set, so the "no audio received" warning fired unconditionally).
  - **One `tts stop` per turn (`main.py`)** — the cascade played every sentence with `send_stop=True`, so the device saw `tts start`/`tts stop` between sentences; each stop re-arms the mic only while the firmware still considers itself LISTENING (it falls back to IDLE after ~10 s without an `stt` reply), and Edge-TTS gaps of 11–16 s meant long replies timed out into IDLE and ignored the FINAL stop too — no mic, follow-up window expired. Now the turn streams as ONE audio unit (`send_stop=False`) and a new idempotent `send_tts_stop()` closes it exactly once from `_dispatch_hermes()`'s `finally` and from the watchdog — covering normal end, backend exception, playback cancel and a failed synthesis (Edge-TTS DNS outage left the satellite stuck in SPEAKING).
  - **`EXPERT_TIMEOUT` 90 → 25 s (`config.py`)** — `hermes_expert` shared the gateway's 90 s watchdog budget, so it always lost the race: measured 2026-09-27, `qdrant_search` 7.3 s + `hermes_expert` 92.5 s on the one-word utterance «связь» → a 104 s turn whose only spoken outcome was the watchdog apology. 25 s still covers a real infrastructure diagnostic and leaves L2 room to speak before the watchdog (env override in `ai-prod/docker-compose.yml` must move with it).
  - **English comment audit (comments/docstrings only, code byte-identical)** — all Russian prose comments translated to English (`camera_client.py`, `main.py`, `setup_gateway.sh` — Russian *string literals* such as prompts, TTS/STT data and UI text deliberately untouched); docstrings added for the last ~40 undocumented symbols (`camera_client.py` session/track/arbiter helpers, `main.py` REST/OTA/device CRUD routes, `engine.py`, `backends.py`, jev-router `classifier`/`ha_client`/`weather`). The `@tool` docstrings in `tools.py` stay Russian on purpose: smolagents feeds them to the LLM verbatim as tool descriptions (see the PROMPT-TEXT WARNING in that file). Verified: AST of every changed file identical with docstrings stripped, `py_compile` + `bash -n` pass, **75 passed** host suite, **149 passed** full Docker suite.
- **2.30** — «Пусть робот уберется на кухне» failed end-to-end; the fix is the right intent, not a prompt.
  - **What happened** — the model called `vacuum__HassVacuumStart(area="кухня")`. In HA **2026.9.3** the `area` slot of `HassVacuumStart` filters by the vacuum's **LOCATION** (entity matching), not by the cleaning target: the robot is assigned to no room in HA (`area_id=None`) → `MatchFailedReason.AREA`, `states=[]`, the robot stayed `docked`. The same step also wrote `final_answer` («отправляю робота-пылесоса») in the code block — and the honesty veto stayed silent because `_RE_CLAIM` had no «отправляю».
  - **The right intent** — `vacuum__HassVacuumCleanArea`: there `area` is a service parameter (`cleaning_area_id`), the entity matches by domain + `CLEAN_AREA` feature only. Verified on the live HA: `CLEAN_AREA=16384` is set in `supported_features=29372`, `area_mapping` is configured (**Kitchen → segment 16**), `last_seen_segments` present, `should_expose=true`.
  - **L2 (`tools.py`)** — deterministic fallback: `Start+area` failing with `MatchFailedReason.AREA` → retry `CleanArea(area)`; `ReturnToBase+area` → retry without `area` (a location filter is meaningless for docking); original behaviour kept when Start+area actually matches. `ha_action` docstring steers the model to CleanArea for a room.
  - **L1 (`resolver.py`)** — a room NAMED in the utterance («на кухне») → `CleanArea`; the stream's default area stays a speaker-location hint and keeps `Start+area` (never silently changes a plain Start into a room clean).
  - **Honesty (`honesty.py`)** — `_RE_CLAIM` learns the vacuum/movement verbs (`отправ\w*|запусти\w*|запуска\w*|начина\w*|убира\w*|убер[её]т\w*`) with a `(?<!не )` guard so an honest status («робот не убирается») is not a claim; `_truth` gets an `AREA` branch — «устройство не привязано к этой комнате» instead of the generic «HA отклонил команду».
  - **Tests** — `test_honesty.py` +3 (field promise replaced, start/clean verbs, negated status untouched), `test_router_resolution.py` +2 (named room → CleanArea, stream default → Start) → suite **75 passed**.
  - **E2E** — «Пусть робот уберется на кухне» → 2.4s → model itself called `CleanArea({"area": "кухня"})` → vacuum `state: cleaning` (20s+ confirmed, segment 16), veto silent **because the claim was backed by the recorded success**.
- **2.29** — L2 gets a weather tool, and the forecast it returns is enforced (field case: «подвоя»).
  - **What happened** — STT garbled «погоду» → «подвоя», the L1 weather regex missed, the resolver escalated to `complex_logic` (correct), but the L2 CodeAgent called `ha_read("погода")` (no such entity), **made up** «солнечно, 10–22°» and its honesty veto stayed silent: `get_datetime`/`qdrant_search` had succeeded, so the «0 successes» condition didn't hold. On retry it did call the new tool, saw the right forecast in the logs — and still wrote `final_answer` **in the same code block, before reading it** (twice out of two; prompt rules ignored).
  - **`weather_forecast` tool** — GET jev-router `/weather` (the same open-meteo chain, single source of WMO/coordinates logic); empty → honest «прогноз недоступен», never a guess. Manifest line added to `TASK_TEMPLATE`.
  - **Weather veto (`honesty.py: vet_weather`)** — recorded tool detail decides: a forecast whose numbers/conditions don't match the recorded output is replaced by it verbatim; a forecast claim after a failed fetch is replaced by the refusal. Skipped when a `ha_action` was attempted this run (must never wipe an action report). Same fail-open philosophy as the action veto, pure stdlib.
  - **Tests** — `test_honesty.py` +7 (lie replaced, faithful passes, numbers-only mismatch, failed fetch ± admission, no events, action skip); suite **70 passed**.
  - **E2E** — «Какая завтра будет подвоя?» → 4.1s → spoken «Завтра ожидается пасмурно: днём до 16°, ночью до 9°» (weather veto logged); fast L1 path 0.38s; non-weather `complex_logic` untouched.
- **2.28** — Weather hybrid chain + expert route really on Hermes («не решается ниже → Гермес» now works).
  - **Root cause** — «Какая завтра будет погода?» was routed `easy_query` → the weather branch streamed the free combo; its polite refusal counted as *success* (`chat_proxy` failed over to Hermes only on transport errors), and the `query_unresolved → complex_logic` escalation was unreachable for weather because any stream sets `emitted=True`. Nothing ever switched to Hermes — by construction, not by accident.
  - **Hybrid weather (L1)** — `services/jev-router/weather.py`: a deterministic forecast from open-meteo (free, no key) using coordinates from HA `/api/config` (cached): current conditions with wind («сейчас»), «завтра»/«послезавтра»/«в пятницу» (7-day window; WMO codes → RU phrases; precip probability spoken only when ≥30%). Instant, no LLM, cannot invent dates (E2E: the free model refused, Hermes named a wrong date). API unreachable → **direct Hermes L3 stream** (`stream_hermes`); Hermes silent too → the existing `query_unresolved → complex_logic` escalation finally fires, where the L2 can call `hermes_expert`.
  - **expert = Hermes first** — `stream_chat(expert=True)` now tries Hermes before OmniRoute (README always promised this; previously the free combo answered expert questions too), OmniRoute stays as failover; a `stream_chat expert=… primary=…` log line shows which brain took the call.
  - **Tests** — `tests/test_weather.py` (target picking, RU phrasing, weekday window, fallbacks); suite 63 passed on host.
- **2.27** — E2E fixes: a promised side effect must be a performed side effect.
  - **Area canonical names (root cause of «пообещало выключить и не выключило»)** — the resolver now sends area-registry *display* names (`Living Room`, `Kitchen`, `Corridor`, `Bedroom`...): RU «гостиная»/«ванная»/«туалет» have no alias in the live HA registry and the intent matcher rejected them with `MatchFailedError INVALID_AREA` (E2E via ESP32, 25.09). RU words remain the lookup keys; `_area_phrase` gained a table so EN names still produce Russian phrases («В гостиной», «На кухне»).
  - **On/off target resolution from the live registry** — `find_action_targets()` + `HAClient.get_entity_areas()` (cached `area_name` template map): the living-room lamp is `switch.living_room_light_swith_relay` while every `light.*` there is an unavailable ESP indicator, so a blind `domain:["light"]+area` match both missed the relay and could no-op on `unavailable` states. The executor now: picks concrete entities (bilingual «свет»→light), calls the intent with the exact friendly name, skips entities not exposed to the voice assistant (`MatchFailedReason.ASSISTANT`) without failing the rest, and answers «В гостиной уже выключено.» when every target is already in the requested state instead of pretending. Result: «Выключи/Включи свет в гостиной» → 0.08–0.11 s, no escalation, state change verified in HA; 0 escalations across the regression batch (state/datetime queries with EN areas intact).
  - **Escalation context + worker honesty** — a failed `easy_action` now escalates to L2 with tool, args, plain-Russian error decoding (`INVALID_AREA`, `ASSISTANT` → «не открыто голосовому ассистенту») and any confirmed partial results (`done`); the worker prompt explicitly forbids promising an action («выключаю/сделал») before the tool confirmed success — the L2 previously answered «Хорошо, выключаю свет в гостиной» right after its own `ha_action` failed (log-verified).
  - **Honesty veto + coffemaker E2E (25.09)** — STT heard «кафеварку» (dropped «о»), the resolver escalated to L2, and the model answered «Кафеварка включена!» after *both* its `ha_read` and `ha_action` failed (`MatchFailedReason.NAME`) — 3rd field lie, prompt rules ignored 3/3. Fixes: THING alias «кафеварк» (STT typo); deterministic «Не нашла такого устройства» straight from L1 when the hint matches nothing in the whole registry (0.06 s, no L2 to lie in); `dedupe_device_facets()` so `switch.coffemaker_child_lock` is not toggled along with the device root; on/off commands matching several devices *without* a room in the utterance escalate instead of guessing; **programmatic honesty veto** (`honesty.py` + `ha_action` outcome recorder in `tools.py`): a success claim backed only by failed side effects is replaced by the recorded truth («Не нашла такого устройства…» / «не открыто голосовому ассистенту» / «комната не найдена»), never by the model's word. E2E: «Включи кафеварку.» → «Включила» 0.19 s, `switch.coffemaker=on`, child lock untouched, 0 escalations.
  - **Tests** — 54 passed on host (`test_router_resolution.py` incl. new area/target/hint/typo/dedupe cases, `test_honesty.py` veto cases, `test_cascade_backend.py`, `test_tts_gate.py`).
- **2.26** — Cascade AI architecture (3 levels) + dialogue memory.
  - **`services/jev-router` (L1, port 8091)** — semantic route classifier: local Ollama `qwen3-embedding:0.6b` (1024-dim cosine, calibrated Sep 2026: conf = clip((score−0.50)/0.40), threshold 0.85 + margin ≥0.05) over 5 routes (`easy_action`, `easy_query`, `general_qa`, `expert`, `complex_logic`), regex fast-paths for imperatives, negation guard («не включи свет» never reaches HA), slot resolver for HA MCP intents (RU/EN room dictionary, stream→room defaults, brightness/color/temp, vacuum, timers, broadcast). SSE `POST /route` emits `route`/`sentence`/`progress`/`done`/`error`. Any ambiguity, missing MCP tool or HA failure escalates to `complex_logic` (self-healing retry via a second `route` event).
  - **`services/smolagents-worker` (L2, port 8092)** — smolagents 1.26 `CodeAgent` (MAX_STEPS=15) over the OmniRoute free combo (`gemma4_31b_free`) with Hermes (`hermes-agent`) failover; tools: `ha_action` (MCP side effects), `ha_read` (GetLiveContext + REST fallback), `qdrant_search` (dialogue memory), `hermes_expert` (L3 delegation), `weather_forecast` (jev-router `/weather` → open-meteo), `get_datetime`. SSE `POST /invoke` with progress heartbeats every 15 s (replace the old watchdog) and a hard 120 s cap; final answer is split into TTS sentences with the same boundary rules as `backends.py`.
  - **`CascadeBackend` in `backends.py`** — third gateway backend (`LLM_BACKEND=cascade`): ack-timer instead of the emotion gate (immediate ack for `complex_logic`/`expert`, `ROUTER_ACK_DELAY`=3 s for chat/easy), progress events spoken as short phrases, immediate apology on error/empty stream. Existing `hermes`/`nanobot` paths untouched — rollback is one env line.
  - **Qdrant memory** — collections `voice_turns` + `voice_facts` (dim 1024, cosine) created idempotently at router startup; every turn is written fire-and-forget and searchable by the L2 `qdrant_search` tool.
  - **Tests** — `tests/test_cascade_backend.py` (8: ack timing, escalation acks, tag stripping, apology paths) + `tests/test_router_resolution.py` (25: resolver slots, escalation cases, regex fast-paths, calibration); `test_tts_gate.py` unchanged, 33 passed on host.
- **2.25** — TTS stutter fixes + monologue gate.
  - Abbreviation-aware sentence splitter (`_sentence_boundary`): a `.`/`!`/`?`/`…` splits only before whitespace + uppercase/digit/quote or end-of-string, so «мм рт. ст.», «т.д.», «16.09» survive as one utterance instead of audible fragments.
  - Hermes player (`_hermes_player_task`) with one-ahead synth prefetch: sentence N+1 synthesizes while N plays (was sequential: 1–4 s of silence between sentences). First flowing audio cancels the watchdog; sentences arriving after the watchdog apology are dropped instead of played stale.
  - Monologue gate: reply must start with `[happy]`/`[neutral]`/`[thinking]`/`[surprised]`/`[sad]`/`[angry]`; pre-tag sentences are held (≤500 chars) and discarded on tag, spoken on release — never silent. `HermesBackend` sends a system prompt enforcing the leading tag and forbidding verbalized tool-talk («Need to execute code…» class of leaks).
  - `tests/test_tts_gate.py` (4 tests: gate, loss-free, fail-open, abbreviations); suite 78 passed.
- **2.24** — Hermes-mode hardening + camera kill-switch.
  - The per-device Nanobot listener no longer starts when `LLM_BACKEND=hermes` (previously it retried `localhost:8765` every 10 s forever, spamming the log; the Hermes reply path never used that socket).
  - `DISABLE_CAMERAS=true|1|yes` is now honored by `start_camera_sessions()` (previously the flag existed in compose but was ignored, so cameras ran anyway).
  - `HermesBackend` sends `X-Source: voice_gateway`, `X-Device-MAC` and `X-Stream-Name` headers for per-device routing upstream.
  - Removed the stale `(managed by nanobot in v0.3.0)` suffix from the MCP tools log line; deduped the MAC entry in `config/devices.json`.
- **2.8** — Pluggable LLM backend. Added `backends.py` with `BaseLLMBackend`, `NanobotBackend` (moved out of `main.py`/`camera_client.py`), and `HermesBackend` (OpenAI-compatible `/v1/chat/completions`). Select via `LLM_BACKEND` (`nanobot` | `hermes`); Hermes configured with `HERMES_API_URL` / `HERMES_API_KEY`. Same VAD→STT→TTS pipeline, emotion tags, watchdog and echo guards apply to both backends.
