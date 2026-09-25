# Voice Gateway

> **Version 2.28** — Cascade AI architecture (jev-router → smolagents CodeAgent → Hermes Expert), hybrid weather chain (open-meteo → Hermes → L2) + E2E honesty fixes for side effects. See Changelog below.

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
3. **openWakeWord** listens for the wake word «компьютер» (custom-trained model) on camera streams
4. **Whisper STT** transcribes audio to text (Russian, retries at temperature 0.0/0.5)
5. **Speaker ID** identifies the speaker (parallel with STT)
6. **LLM backend** (Cascade *or* Nanobot *or* Hermes) processes text, returns response with optional emotions `[emotion_name]`
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
- **MCP hardware control** — ESP32/camera tools (screen, volume, LEDs) are requested via `tools/list`; in Nanobot mode device events are forwarded to Nanobot, in Hermes/Cascade mode the tool list is kept for future use (neither Hermes nor the router sends a `tools` payload)
- **Watchdog** — 30 s timeout → fallback TTS «Простите, я задумалась. Повторите пожалуйста.»
- **Activity monitor** — dims screen to 25% after 30 s idle, closes abandoned sessions after 45 s in LISTENING
- **OTA** — ESP32 firmware handshake returning WS URL + `access_token` + firmware info
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

- **L1 `jev-router`** (port 8091) — semantic route classifier (local Ollama embeddings) + offline slot resolver. Easy commands execute directly against Home Assistant (~0.1 s), state queries are answered from the live registry, `general_qa` streams from OmniRoute, `expert` hits Hermes first (OmniRoute failover), weather is built from open-meteo (Hermes on API failure, L2 as last resort); anything ambiguous or failed escalates to L2 with decoded error context.
- **L2 `smolagents-worker`** (port 8092) — smolagents `CodeAgent` (`ha_action`/`ha_read`/`qdrant_search`/`hermes_expert`/`get_datetime`), 120 s cap with progress heartbeats, and a programmatic honesty veto: an answer claiming success after only failed tool calls is replaced with the recorded truth.

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
| `NANOBOT_TOKEN` | `token` | Token appended to Nanobot WS URL |
| `LLM_BACKEND` | `nanobot` | AI brain: `nanobot` (default), `hermes` or `cascade` |
| `HERMES_API_URL` | `http://192.168.22.102:8000` | Hermes OpenAI-compatible base URL (used when `LLM_BACKEND=hermes`) |
| `HERMES_API_KEY` | `""` | Bearer token sent to Hermes if set |
| `ROUTER_URL` | `http://localhost:8091` | jev-router SSE endpoint (used when `LLM_BACKEND=cascade`) |
| `ROUTER_ACK_DELAY` | `3` | Seconds before the fallback ack «Секунду…» when the first sentence is slow (chat/easy routes; complex/expert ack immediately) |
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
| `/health` | GET | Liveness probe, no auth: `{"status":"ok"}` |
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
RTSP (raw L16 16 kHz) → echo guard (waveform + `_is_echo` cross-correlation vs played-audio ring, delays 2–45 s) → VAD + RMS gates → SpeexDSP NS rescue → adaptive normalisation → **(only after an acoustic wake)** Whisper (+ Speaker ID via Ogg) → LLM backend

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
Backend text → sentence splitter → **prefetch pipeline** (sentence N+1 is synthesised while N plays, hiding Edge-TTS latency) → Edge TTS (MP3) → decode → resample (16/24 kHz) → Opus for ESP32 / PCM for cameras → paced 60 ms chunks. Every played clip is registered in the echo-reference ring and extends wake suppression past the delayed-echo window.

### LLM Backend (Nanobot / Hermes / Cascade)

The AI brain is selected by `LLM_BACKEND` and implemented as a `BaseLLMBackend` (`backends.py`):

- **`NanobotBackend`** (default) — opens a WebSocket to `NANOBOT_WS_URL?token=…&chat_id=…`, streams `text` deltas, handles `[thinking]` blocks, and pushes sentence-split replies into the response queue.
- **`HermesBackend`** — streams the user message to `${HERMES_API_URL}/v1/chat/completions` (OpenAI-compatible SSE, `stream: true`); each `choices[].delta.content` chunk is accumulated and flushed sentence-by-sentence on `. ! ? …` into the response queue. `[thinking]` reasoning blocks and `[emotion_name]` tags are stripped before TTS, identical to `NanobotBackend`. Streaming keeps first-token latency low so the prefetch TTS player starts before the full reply arrives (avoiding the 30 s watchdog fallback).
- **`CascadeBackend`** — POSTs the text to `${ROUTER_URL}/route` (SSE) and replays router events into the same queue: `sentence`/`progress` are spoken (emotion tags stripped, no monologue gate — the router output is already curated), `route` events drive the ack timer (immediate «Секунду, занимаюсь…» for `complex_logic`/`expert`, `ROUTER_ACK_DELAY`-second fallback for chat/easy), `error`/empty streams produce an immediate apology so the user is never left in silence. The first played audio (usually the ack) cancels the watchdog downstream, which is what makes 48–120 s L2 turns possible.

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
- External services: Whisper STT, Edge TTS, Speaker ID, go2rtc (for cameras), plus one of the backends — Nanobot; Hermes; or (cascade) Ollama embeddings + Qdrant + OmniRoute + Home Assistant MCP

## Changelog

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
  - **`services/smolagents-worker` (L2, port 8092)** — smolagents 1.26 `CodeAgent` (MAX_STEPS=15) over the OmniRoute free combo (`gemma4_31b_free`) with Hermes (`hermes-agent`) failover; tools: `ha_action` (MCP side effects), `ha_read` (GetLiveContext + REST fallback), `qdrant_search` (dialogue memory), `hermes_expert` (L3 delegation), `get_datetime`. SSE `POST /invoke` with progress heartbeats every 15 s (replace the old watchdog) and a hard 120 s cap; final answer is split into TTS sentences with the same boundary rules as `backends.py`.
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
