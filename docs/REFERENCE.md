# Voice Gateway — Reference

> Back to the project overview: [`../README.md`](../README.md). This file is the detailed reference: configuration tables, REST API, architecture notes, known issues and the changelog.

> **Version 2.36** — «включи свет в **первом** коридоре» moved BOTH corridor relays (02.10.2026): `resolve_action` dropped the ordinal — the slots are HA matcher slots and a digit is not one — so L1 resolved hint «свет» + area `Corridor`, while L2's resolver and `ha_read` ignored «первом»/«втором» as an unknown word. The ordinal now rides inside the hint («свет 1») and `ha_client.find_action_targets()` keeps only the relay whose id carries the digit; `smolagents-worker/ha_match.py` mirrors the same `ordinal_digit()` for `resolve_onoff_targets()`/`match_states()`. A numbered instance that does not exist returns `[]` (escalate / keep the original error) — never a guessed both-relays toggle. The corridor relays also got HA aliases («первый коридор», «коридор 1», …) so a model-passed `name` matches natively, and the same pass fixed L1's "already" truth: an unexposed Network-led facet sitting in `todo` made «включи свет в прихожей» (already on) escalate instead of «уже включено». The same session exposed the deeper break: pinning the FREINDLY NAME into `name` made HA answer `MatchFailedReason.NAME` for `corridor1_light_switch Relay` and `coffemaker` (while `entrance_light_switch Relay` matched), so every corridor fast path failed and escalated — both L1 and the L2 retry now pin the ENTITY ID instead, `NAME` joins `AREA`/`ASSISTANT` as a fallback trigger (a phrase or a friendly name the matcher cannot use is a PHRASING problem, and an invented word still resolves to nothing), and a name-only call consults the AREA REGISTRY instead of an id substring. Pinned by 13 new cases — **262 tests across 14 files (3,565 lines)**.

> **Version 2.35** — «Значит, прихожей.» could not switch the hallway light (02.10.2026 05:45): the lamp is `switch.entrance_light_switch_relay` and area Entrance holds no `light.*` at all, so L2's blind `domain: ["light"]` intent answered `MatchFailedReason.AREA` twice and the relay was not even exposed to Assist until today. `tools.ha_action` now resolves the concrete entity from raw `/api/states` (the rule L1 already has) and retries with `name`+`domain` pinned, while `ha_match.match_states` ranks the ROOM above the domain hint — «свет в прихожей» used to lose the relay to ten `light.*` status LEDs inside the 10-line cap. Pinned by 14 new cases in `tests/test_ha_match.py` — **249 tests across 14 files (3,399 lines)**.

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
- **Telegram as a third source** — `telegram_client.py` long-polls `getUpdates` (no webhook, no public HTTPS) in one asyncio task; **chat allowlist** (empty = nobody served) and private-chats-only by default. A voice note (Ogg/Opus, or mp3/m4a re-containered) goes through the same Whisper endpoint, is validated with a Telegram-specific filter (deliberate speech: «да»/«ок» must pass — the ambient echo filter would reject them), then the same `llm_backend` answers: streamed text via `editMessageText` (ack phrases like «Секунду…» are a separate status bubble and never enter the reply or the voice), plus an optional voice note (Edge TTS → ffmpeg → Ogg/Opus, `TELEGRAM_REPLY_VOICE`, capped at ~900 chars of text). Commands: `/start`, `/room <name>` (default room for bare «включи свет» — travels as `RouteRequest.room`, the history key stays `tg:<chat_id>`), `/voice on|off`. Per-turn cap `TELEGRAM_TURN_TIMEOUT` 120 s (partial reply or apology on expiry), 2 concurrent turns max, per-chat ordered queue.

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
| `TELEGRAM_BOT_TOKEN` | `""` | Telegram bot token; **empty disables the source**. Lives in `.env.voice_gateway` (compose `env_file`, gitignored) — never in the repo |
| `TELEGRAM_ALLOWED_CHAT_IDS` | `""` | Comma-separated chat ids allowed to talk to the bot; **empty = nobody is served** (fail-closed on a leaked token) |
| `TELEGRAM_REPLY_VOICE` | `true` | Also reply as a voice note (TTS → ffmpeg → Ogg/Opus) |
| `TELEGRAM_ALLOW_GROUPS` | `false` | Permit group chats (also requires the group id in the allowlist) |
| `TELEGRAM_MAX_VOICE_S` | `60` | Refuse longer voice notes (checked before download) |
| `TELEGRAM_TURN_TIMEOUT` | `120` | Per-turn cap; on expiry the partial reply is sent, otherwise «Простите, я задумалась…» |
| `TELEGRAM_COOLDOWN_S` | `1.5` | Minimum spacing between turn starts inside one chat |
| `TELEGRAM_API_BASE` | `https://api.telegram.org` | Bot API base (overridable for a local mock) |

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

Both backends expose the same `generate_response(text, session_id, stream_name, response_queue, room="")` contract. The ESP32 path dispatches via `_dispatch_hermes`/`_hermes_player_task`; camera sessions receive the backend instance at construction and call `_call_backend` → `_nanobot_player_task` (the player is backend-agnostic). Either way replies reach the prefetch TTS player, so sentence-level latency hiding works identically for both brains. `room` is a default-area hint only (Telegram `/room` binding) — only `CascadeBackend` forwards it, as `RouteRequest.room`; it is deliberately NOT folded into `stream_name`, which keys the dialogue history.

### Telegram source

`telegram_client.py` runs one long-polling task (`getUpdates`, 25 s window) started in `on_startup()` and cancelled in `on_shutdown()` — same event loop, no webhook, no extra port. Startup calls `getMe` (logs the bot name, `401/403` stops the source with a clear error) and `deleteWebhook(drop_pending_updates=True)` so messages queued while the container was down are not answered belatedly; the `409 Conflict` recovery path deletes the webhook **without** dropping live updates. Poll errors back off exponentially (429 honours `retry_after`), and `update_id` is acked (`offset+1`) before handling — a turn lost to a crash beats answering twice.

Flow per message: allowlist/group gate (in `dispatch()` *and* `handle_message()`) → per-chat worker queue (ordered within a chat, parallel across chats, cap 3 pending) → command / voice / text. Voice: `getFile` + download (duration refused before the download), `prepare_stt_audio()` in a worker thread (already-Ogg passes through, mp3/m4a → Ogg/Opus at 48 kHz so the upload matches the ESP32's `a.ogg` contract), Whisper via the injected `fetch_transcription()`, then a Telegram-specific transcript validation. A turn runs under a 2-turn semaphore with a `typing` refresh every 4 s: the backend's sentences are flushed to one message via `editMessageText` (≥1 s apart), ack phrases become a separate status bubble, the final text is split at sentence boundaries into ≤4000-char messages, and `TELEGRAM_TURN_TIMEOUT` cancels the producer with a partial reply or an apology. The optional voice reply (first sentences ≤900 chars → `synthesize_tts_mp3` → ffmpeg mp3→Ogg/Opus) is best-effort — the text is already in the chat when it runs.

Per-chat state (`room`, `/voice` toggle) persists in `config/telegram_chats.json`. `stream_name` is always `tg:<chat_id>` so `history.block()`/`memory` keep every chat (and the cameras) on their own ring; the `/room` binding reaches the resolver only through `RouteRequest.room`.

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

## Vacuum map segments (Valetudo)

The Roborock S5 runs Valetudo as `ZealousEverlastingGaur` (`valetudo-zealouseverlastinggaur.local`). Room cleaning reaches it two ways:

- `vacuum__HassVacuumCleanArea` with an HA **area** name — HA resolves the area through the vacuum entity's `area_mapping` option (entity registry), so the gateway only ever sends display names such as `Kitchen` / `Living Room`.
- `mqtt.publish` to `valetudo/ZealousEverlastingGaur/MapSegmentationCapability/clean/set` with a **JSON object**, never a bare array.

**Current map — 6 segments** (re-created 2026-10-01; `sensor.…_map_segments` reports `6`, was `8`):

| Segment id | Valetudo name | HA area id | Area display name |
|---|---|---|---|
| `16` | `Living_room` | `living_room` | Living Room |
| `17` | `Kitchen` | `kitchen` | Kitchen |
| `18` | `Bedroom_Vlada` | `bedroom_vlada` | Bedroom Vlada |
| `19` | `Bedroom` | `bedroom` | Bedroom |
| `20` | `Corridor` | `sorridor` | Corridor |
| `21` | `Entrance` | `entrance` | Entrance |

Source of truth is Valetudo itself — `GET http://valetudo-zealouseverlastinggaur.local/api/v2/robot/capabilities/MapSegmentationCapability`. The old 8-segment map (Kitchen = `16`, plus `22`/`23` and a split `corridor1`/`corridor2`) is gone: ids `22` and `23` no longer exist, the two corridor segments are now the single `20`, and there is **no** bathroom/WC/balcony segment at all — an area with no segment (`wc`, `big_wc`, `bathroom`, `balcony`) cannot be cleaned by segment.

Rules that follow from the map:

- **MQTT payload is an object**: `{"segment_ids":["17"],"iterations":1,"customOrder":false}`. Valetudo 2026.08.0 does `JSON.parse()` then requires `reqSegments.segment_ids`; a bare `["kitchen"]` is rejected with `Missing or empty segment_ids Array in payload`, and each id is validated against `segm.id`, so segment *names* never work either.
- **HA area ids ≠ Valetudo segment names.** `resolver.py` (`ROOMS`, `STREAM_DEFAULT_AREA`), `ha_client.py` (`_AREA_LAT`, `_area_phrase`) and `ha_match.py` (`AREAS`) send HA display names and are deliberately independent of segment ids — a map re-creation must **not** touch them. Note the misspelled area id `sorridor` whose display name is `Corridor`.
- Segment ids live in two places only: this table and the `area_mapping` option in the live HA entity registry (`options.vacuum.area_mapping` on `vacuum.valetudo_zealouseverlastinggaur`).

## Known Issues & Current Problems

- **go2rtc zombie RTSP sessions** — go2rtc 1.9.2 on the streams host leaks sessions when a camera link is slow; 17–23 half-dead connections with 100–290 KB send queues can stall a camera's majestic until it stops serving :554. Mitigated by a per-camera watchdog (`/etc/watchdog_majestic.sh` + crond) that restarts majestic on dead/blocked RTSP, but the real fix is upgrading/restarting go2rtc on `192.168.22.102`.
- **Camera WiFi links** — the kitchen camera's link quality fluctuates (21–34/100 vs 80+ elsewhere); its video stream was reduced to fps 10 / bitrate 1024 to keep the audio backchannel stable.
- **Echo cascade ESP32 ↔ camera is only partially solved** — `GLOBAL_TTS_UNTIL` + duration-proportional mic hold are band-aids. Camera-speaker echo returns via WebRTC with 8–15 s delay; `_is_echo` cross-correlates mic chunks against a reference ring of recently played audio (delays 2–45 s) and confirmed echoes extend wake suppression.
- **TTS-trained wake model prefers muffled audio** — through-wall copies of «компьютер» can out-score close live speech; the distant-source veto and quiet-source hold compensate, but retraining on real in-room recordings (v2–v4 attempts degraded discrimination — keep v1) remains the proper fix.
- **Gates are room-calibrated** — thresholds (wake 0.40 kitchen / 0.30 elsewhere, hold/veto levels, bg-median 800) were tuned against measured score distributions in three specific rooms. They will not generalise to other rooms without recalibration.
- **Whisper retry at temperature 0.5 is camera-only** — the camera path (`_fetch_transcription`) retries 0.0 → 0.5 on an empty transcript, doubling STT latency in the worst case; the ESP32 path (`fetch_transcription`) sends a single attempt at 0.0 and never retries.
- **`Dockerfile` exposes 8080 but nothing listens on it** (18792 is the only real port).
- **`setup_gateway.sh` is an outdated snapshot** — not authoritative.
- **Vosk is installed but unused** — the `vosk` wheel is in `requirements.txt` and `models/vosk_decode.py` is the only importer. The big `config/vosk-model-ru-0.42/` (3.5 GB) was deleted on 2026-09-29; only `config/vosk-model-small-ru-0.22/` (88 MB) is kept as the candidate for a local low-latency STT fallback, never wired into the pipeline.

## Tests

Host Python usually lacks the runtime deps (`opuslib`, `onnxruntime`, …), and the image has no `pytest` — run the suite inside the running container:

```sh
docker exec voice_gateway pip install -q pytest httpx pytest-asyncio
docker exec voice_gateway rm -rf /tmp/vg_src && docker exec voice_gateway mkdir -p /tmp/vg_src
for f in audio_utils.py backends.py camera_client.py engine.py main.py telegram_client.py; do docker cp "$f" voice_gateway:/tmp/vg_src/"$f"; done
docker cp services voice_gateway:/tmp/vg_src/services
docker cp templates voice_gateway:/tmp/vg_src/templates
docker cp tests voice_gateway:/tmp/vg_src/tests
docker exec -w /tmp/vg_src voice_gateway python3 -m pytest tests -q
docker exec voice_gateway rm -rf /tmp/vg_src
```

Tests must run against a copy of the sources, not the live `/app` mounts: the three router tests import `honesty`/`classifier`/`weather` from `services/jev-router`, which is not part of the image (it runs as its own container), so the sources, `services/` and `tests/` have to travel together.

Covers **299 tests across 15 files (4,169 lines)**: router slot resolution and the escalation hint, the ESP32 `main.py` protocol, honesty vetoes (action / data / weather) + the retry note built from them, the HA REST fallback and its "carries data?" detector, the wake-gate cascade's STT-confirm bands, engine wake scoring, the camera audio track plus its echo and SDP guards, cascade backend streaming (including the `room` payload), the Telegram source (`test_telegram.py`: allowlist and group gates, ack phrases kept out of the final text, turn timeout with partial/fallback, the voice turn end to end, `/room` binding persisted to disk), the short-term dialogue ring, weather, TTS gate, OTA auth, RMS utils the RU→latin hint-table sync (`test_hint_sync.py`, which reads both copies out of the source with `ast` — importing the two services in one pytest process collides on the `config` module name) the HA on/off intent target resolver plus the room-first registry ranking (`test_ha_match.py`, `resolve_onoff_targets`/`match_states`) and the ordinal table that selects corridor1/corridor2 (the same file plus `test_hint_sync.py`, which also holds `ORDINAL_STEMS`/`ORDINAL_ENDINGS` together).

**Not covered:** cross-camera arbitration. `test_wake_gates.py` stubs `_arbiter_submit` and resets `_ARB_STATE` so the gate cascade can be driven in isolation — no test exercises `_arbiter_set_owner` / the proximity steal / `_arbiter_sent_recently`. This is the largest untested block in `camera_client.py`.

CI (`.github/workflows/tests.yml`) runs this exact recipe inside the built image on every push to `main`, on PRs and on manual dispatch — the host runner has none of the runtime deps and the image itself ships without pytest.

## Dependencies

- Python 3.12+
- FFmpeg (RTSP capture, WAV/OGG conversion)
- Silero VAD ONNX model (downloaded at build time)
- openWakeWord + the library head `computer_20260706_130638.onnx` and the custom `computer.onnx` / `embedding_model.onnx` (baked into the image)
- SpeexDSP noise suppression (`speexdsp-ns`)
- External services: Whisper STT, Edge TTS, Speaker ID, go2rtc (for cameras), plus one of the backends — Nanobot; Hermes; or (cascade) Ollama embeddings + Qdrant + OmniRoute + Home Assistant MCP

## Changelog

- **2.37** — Telegram becomes the third source of requests alongside the ESP32 satellites and the cameras.
  - **Why** — the same assistant had to be reachable away from the mics: send a voice note or text from the phone, get the answer back as text and (optionally) as a voice note. Zero new dependencies: the Bot API is a handful of JSON endpoints over the `aiohttp` the process already has — no aiogram/python-telegram-bot.
  - **`telegram_client.py`** — long-poll `getUpdates` (25 s, no webhook / no public HTTPS) in one task started from `on_startup()`; `getMe` + `deleteWebhook(drop_pending_updates=True)` at boot (the `409` recovery deletes the webhook *without* dropping live updates), exponential backoff (429 honours `retry_after`, `401/403` stops the source with a clear log), `update_id` acked before handling. Access: **chat allowlist, empty = nobody served** (fail-closed), private chats only unless `TELEGRAM_ALLOW_GROUPS=true`; the gate runs in `dispatch()` *and* `handle_message()`. Per-chat worker queue (ordered inside a chat, parallel across chats, cap 3), 2 turns max globally, `TELEGRAM_TURN_TIMEOUT` 120 s — on expiry the partial reply is sent, or «Простите, я задумалась…».
  - **Same brain, different validation** — a voice note is downloaded (length refused *before* the download), `prepare_stt_audio()` runs in a thread (Ogg passes through, mp3/m4a → Ogg/Opus so the upload matches the `a.ogg` contract `fetch_transcription()` expects) and the text goes to the injected `transcribe`/`llm_backend`. Deliberate speech needs its own filter: `is_valid_voice_text()` keeps Whisper's hallucinations, char-runs and repetition checks but DROPS the device filter's echo gates — «да»/«ок», the likeliest answer to the bot's question, must pass.
  - **Ack vs reply** — `CascadeBackend.ACK_*`/`SORRY` phrases become a separate «Секунду…» status bubble and never enter the streamed text, so the synthesized voice cannot say them after the fact; `SORRY` already closes a turn and is never doubled with a second apology. The reply is streamed through `editMessageText` (≥1 s apart), split at sentence boundaries into ≤4000-char messages, with a `typing` ping every 4 s.
  - **`/room` without lying to the history key** — the trap in the first draft: jev-router keys `history.block()`/`memory` on `stream_name or session_id`, so a bare room name there would merge a Telegram chat's turns into a camera's ring. The key is now always `tg:<chat_id>` and the `/room` binding travels as a new optional `room` field on `RouteRequest` (only `CascadeBackend` forwards it; devices and cameras send nothing and the payload is unchanged) used solely as the resolver's default area. `/voice on|off` toggles the voice reply per chat; both persist in `config/telegram_chats.json`.
  - **Voice out** — TTS on the first sentences (≤900 chars) → `synthesize_tts_mp3` → ffmpeg mp3→Ogg/Opus (`sendVoice`); best-effort: the text is already in the chat if it fails.
  - **Config in ONE place** — `.env.voice_gateway` in the stack root (gitignored, compose `env_file` on the `voice_gateway` service, precedent `.env.nanobot`): `TELEGRAM_BOT_TOKEN` + `TELEGRAM_ALLOWED_CHAT_IDS` plus the tuning vars. Empty token = source off. The token never touches the repo or the tracked compose file.
  - **Tests** — `tests/test_telegram.py` (+35, fake API + scripted backend, no network): allowlist/group gates, per-chat history keys, ack separation, timeout partial/fallback, the voice turn end to end with monkeypatched audio helpers, `/room` persistence across restart, queue cap; `test_cascade_backend.py` +2 for the `room` payload (present when given, absent otherwise). Suite 262 → **299 tests across 15 files (4,169 lines)**; CI's source-copy list and the Dockerfile `COPY` gained `telegram_client.py`.
  - **Requires `LLM_BACKEND=cascade`** for full function: the `/room` default area and HA actions live in the router path; Nanobot mode answers chat but resolves no devices (and neither does `hermes`).

- **2.36** — «Первый коридор» and «второй коридор» are two devices, not «коридор, оба реле».
  - **What happened** — `resolve_action("включи свет в первом коридоре")` returned `args={area: Corridor, domain: [light]}, hint='свет'`: the ordinal was parsed as noise, so L1's `find_action_targets` matched both `corridor1_…_relay` and `corridor2_…_relay` and «выключи свет во втором коридоре» also killed the first one. In L2 `resolve_onoff_targets(area="второй коридор")` returned both, and `match_states` ranked corridor1 first for «втором» too (registry order, not the wording).
  - **The ordinal travels inside the hint** — a digit cannot go into an HA matcher slot (`area` must stay a registry display name), so `resolver.resolve_action` appends it: hint «свет 1» / «свет 2». Plain «в коридоре» keeps «свет» and therefore still moves both relays. `ha_client.ordinal_digit()` reads it back in `find_action_targets()` and skips any entity whose id+name lacks the digit — an empty result is deliberate: escalating is better than toggling the corridor the user did not name.
  - **Two copies, one contract** — `ORDINAL_STEMS` (`перв/втор/трет` → `1/2/3`) and `ORDINAL_ENDINGS` are duplicated in `services/jev-router/ha_client.py` and `services/smolagents-worker/ha_match.py` (two images), and `tests/test_hint_sync.py` reads both out of the source with `ast` — same treatment `HINTS` already gets. The ENDINGS are the false-friend guard: «вторник» starts with «втор» but ends in «ник», so Tuesday never selects digit 2.
  - **L2 mirrors it** — `resolve_onoff_targets()` takes the digit from `name` **or** `area` (the model may hand the whole phrase over either way), filters the pool and returns `[]` when nothing carries it; `match_states()` narrows the reading only when the digit AND the room both match, so a stray `light.wled1` cannot empty the payload — and when nothing carries the digit the unfiltered ranking survives.
  - **HA aliases (entity registry, outside git)** — `switch.corridor1_light_switch_relay` ↔ «первый коридор»/«коридор 1»/«первый свет в коридоре», `switch.corridor2_light_switch_relay` ↔ «второй коридор»/«коридор 2»/«второй свет в коридоре»: a `name` slot that echoes the user's phrase now matches natively (verified with `config/entity_registry/get` — the `list` response omits `aliases`).
  - **L1's "already" truth** — `_execute_action` only said «уже включено» when `todo` was empty; an unexposed `…_network_led_switch` facet in the opposite state is in `todo`, gets `ASSISTANT`-skipped, and the turn escalated as an error. Now: nothing ok, nothing fatal, everything skipped, and at least one target already in the requested state → the truthful «В прихожей уже включено.»
  - **Pin the ENTITY ID, not the friendly name** — the deepest break of the session: `_execute_action` (L1) and `ha_action`'s retry (L2) both sent `name: <friendly_name>`, and HA's matcher accepts that concatenated string only for SOME entities — field check 02.10.2026: `corridor1_light_switch Relay` and `coffemaker` → `MatchFailedReason.NAME`, `entrance_light_switch Relay` → matched (why the hallway fix worked and the corridor one did not). Every corridor command therefore failed on the fast path and escalated. `name` now carries the entity id: unique, always resolved, and still behind `assistant='conversation'`, so exposure keeps being enforced (the unexposed Network-led facet answers `ASSISTANT` exactly as before).
  - **`NAME` joins the fallback trigger** — HA answers `NAME` for two cases the resolver can settle: a whole phrase in `name` («свет в первом коридоре» — HA matches names, not sentences) and that friendly name. The 2.34 honesty contract survives, because an INVENTED word matches nothing in the registry: `resolve_onoff_targets` returns `[]`, the original error stands and «Не нашла такого устройства» is still what gets spoken.
  - **A name-only call consults the AREA REGISTRY** — with no `area` slot the room was checked against an ID SUBSTRING, so area-less `switch.corridor1_detect` (id says «corridor1») entered the pool, crossed `MAX_ONOFF_TARGETS` and `name='первый коридор'` returned `[]` while the same words as `area` resolved the relay. Now the registry area is authoritative — the rule `find_action_targets` already documents — and the exact-room rule applies to name-only calls too. A latin fragment in `name` («corridor1_light_switch Relay») is matched as a DESCRIPTOR: every token must be present, otherwise the bare «…_relay» tail would admit corridor2's relay and the pool would be refused as ambiguous.
  - **Tests** — `tests/test_ha_match.py` +7 (whole-token `ordinal_digit`, the named-corridor target, the digit arriving in `name`, the nonexistent third corridor → `[]`, no-ordinal → both, narrowed reading, the stray-digit guard) and `tests/test_hint_sync.py` +2 (the two tables), then +4 more for the pinning pass (name-only registry rule, the exact room on a name-only call, the descriptor rule for a latin friendly name, the invented word that must still refuse). Suite 249 → **262 tests across 14 files (3,565 lines)**.
  - **Verified live (read-only + a stubbed MCP)** — `resolve_action` → hint «свет 1»/«свет 2»; a recorder patched over `app.ha.call_tool` shows «в первом» pinning ONLY `corridor1_light_switch Relay` (the LED is `ASSISTANT`-skipped), «в коридоре» pinning both, «выключи во втором» returning «На коридоре уже выключено.» with zero calls; `resolve_onoff_targets` → one relay per ordinal, `[]` for «третий коридор»; `ha_read('свет в первом коридоре')` line 1 = `switch.corridor1_light_switch_relay`.

- **2.35** — «Значит, прихожей.» could not switch the hallway light; the fix is a deterministic target resolver, not a prompt.
  - **What happened (02.10.2026 05:45)** — «Выключи свет в коридоре» (05:44:51) was answered by L1's raw-states path with no MCP call at all (both corridor relays already `off`) → «На коридоре уже выключено». The follow-up «Значит, прихожей.» was `complex_logic conf=0.00 reason=ambiguous` → L2 → `ha_action(intent__HassTurnOff, {"area": "прихожая", "domain": ["light"]})`, retried once as `{"area": "Entrance"}` → both `MatchFailedError reason=AREA, states=[]` → «Не удалось выключить свет, я не нашёл устройств в прихожей.»
  - **Two causes, both proven against the live HA** — area `Entrance` (id `entrance`, aliases `прихожая`/`вход` — the model picked the area correctly) has **no `light.*`**, the lamp is the Zigbee `switch.entrance_light_switch_relay`; and until today that relay was `conversation.should_expose=false`, which answers `reason=ASSISTANT` instead. Only 10 entities in the whole instance were open to the voice assistant.
  - **Exposure (HA entity registry, outside git)** — `config/entity_registry/update` set `options.conversation.should_expose=true` on `switch.entrance_light_switch_relay`, `switch.corridor1_light_switch_relay`, `switch.corridor2_light_switch_relay`; MCP `GetLiveContext` now returns them with their spoken names («свет в прихожей», «свет в коридоре»).
  - **L2 `tools.ha_action` — deterministic retry** — on `MatchFailedReason.AREA`/`.ASSISTANT` for `intent__HassTurnOn|HassTurnOff`, resolve the target from raw `/api/states` plus one `area_name` template render (`ha_match.resolve_onoff_targets`, the rule L1 already has in `find_action_targets`) and retry with `name`+`domain` pinned — the `area` slot that just failed is dropped, the entity id is unique already. Refusal guards: no device word given, several devices with no room named (mirrors L1's `ambiguous_no_area`), >5 targets, state not `on`/`off` (offline must escalate, never be reported as done), facets dropped (the network LED next to the relay matches every word the relay does), an exactly-named room beats one whose name merely contains it («Bedroom Vlada» ≠ «спальня»). Zero successes keep the ORIGINAL error — an ambiguity stays an honest refusal; a partial fan-out is recorded as `выполнено N/M`, never as success, and `_action_ok()` requires a non-empty `data.success` because `has_data()` alone would accept an `action_done` whose success list is empty.
  - **`ha_match.match_states` — room first** — the sort before the 10-line cap now takes five keys: room hit, distinct-matchers count, facet, helper domain, readout domain, domain hint. «свет в прихожей» used to return ten `light.*` status LEDs ranked first (their domain IS the hint, and they sat earlier in the registry) while `switch.entrance_light_switch_relay` — matching BOTH «light» and «entrance» — was cut by the cap. `AREAS` also gains the EN display names (`Entrance`, `Corridor`, …, mirrored from the router's `_AREA_LAT`) and `«вход»`: `area_matchers("Entrance")` was EMPTY, which silently switched any room filter off the moment HA answered in English.
  - **Tests** — `tests/test_ha_match.py` +14: room-over-domain ranking, cap survival, facet ranking, the resolver's five refusal branches, the exact-room rule, bilingual `area_matchers`. Suite 235 → **249 tests across 14 files (3,399 lines)**; the `@tool` docstrings (prompt text) are byte-identical, `HINTS` untouched.
  - **Verified live** — the incident call `intent__HassTurnOff {area: прихожая, domain: ["light"]}` → `action_done` on `switch.entrance_light_switch_relay` (the light was already `off`, so nothing moved); `{area: спальня, domain: ["light"]}` still fails with the original `MatchFailedReason.AREA` and touches nothing; `ha_read` returns the relay as line 1 for «свет в прихожей» and both corridor relays for «свет в коридоре».

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
  - **The right intent** — `vacuum__HassVacuumCleanArea`: there `area` is a service parameter (`cleaning_area_id`), the entity matches by domain + `CLEAN_AREA` feature only. Verified on the live HA: `CLEAN_AREA=16384` is set in `supported_features=29372`, `area_mapping` is configured (**Kitchen → segment 16** — *as the map stood on 29.09.2026; it has since been re-created and Kitchen is now segment `17`, see* Vacuum map segments), `last_seen_segments` present, `should_expose=true`.
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
