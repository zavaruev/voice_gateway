# Voice Gateway

![version](https://img.shields.io/badge/version-v2.34-blue)
[![tests](https://github.com/zavaruev/voice_gateway/actions/workflows/tests.yml/badge.svg)](https://github.com/zavaruev/voice_gateway/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![backend](https://img.shields.io/badge/AI%20backend-Cascade-orange)
[![license](https://img.shields.io/badge/license-AGPL--3.0--or--later%20%2B%20commercial-blue)](LICENSE)
[![stars](https://img.shields.io/github/stars/zavaruev/voice_gateway?style=flat&logo=github&color=yellow)](https://github.com/zavaruev/voice_gateway/stargazers)
[![issues](https://img.shields.io/github/issues/zavaruev/voice_gateway)](https://github.com/zavaruev/voice_gateway/issues)

> One process that turns every microphone in the house (ESP32 satellites and OpenIPC cameras) into a single employee, and every AI answer back into sound out of the right speaker.

![architecture](docs/images/architecture.svg)

FastAPI service on port **18792** that terminates the WebSocket of every ESP32 satellite speaker and the RTSP/WebRTC session of every configured camera, runs the audio front half (VAD, wake word, STT, speaker ID), hands the transcript to a pluggable LLM backend, and streams the spoken answer back to the device that asked.

---

## Why this and not something else

Home Assistant's own voice stack (Wyoming + ESPHome satellites) is the right answer if everything you own already lives inside HA. This project exists for the three things it does not cover:

| | HA + ESPHome / Wyoming satellites | **voice_gateway** |
|---|---|---|
| Microphone hardware | a purpose-built satellite board | **every ESP32 you already have, plus the mics inside your OpenIPC cameras** (RTSP backchannel, no firmware change) |
| Who answers | one conversation agent | a 3-level cascade: offline intent router → code agent with honesty vetoes → any OpenAI-compatible LLM |
| Several rooms hearing one utterance | preferred-satellite selection | per-chunk arbitration with a proximity steal (the room 5× louder takes the wake) and per-room wake thresholds |
| Who is speaking | — | CAMP++ speaker embeddings, cross-device speaker lock |
| False-wake protection | per-device threshold | an 8-gate cascade (echo ring, appliance noise, distance, clipping, crest factor, …) with a Whisper confirm-rescue band instead of silence |

Honest limits, so nobody is surprised later: the camera subsystem is implemented and tested but currently **disabled in production** (`DISABLE_CAMERAS=true`) — the ESP32 path is what runs live. See [`docs/REFERENCE.md`](docs/REFERENCE.md) → *Known Issues & Current Problems*.

### Numbers from the production log

| What | Measured |
|---|---|
| Simple command, L1 fast path (after STT) | **0.08–0.23 s** («выключи свет в гостиной» → 0.08–0.11 s) |
| Weather on the deterministic L1 route | **0.38 s** |
| L2 turn with a verified side effect | **3.2–4.0 s** |
| Wake debounce | ~240 ms (2 of 3 chunks); a single chunk ≥ 0.68 fires immediately |
| Wake window / watchdog | 15 s / 90 s |
| Test suite | 177 tests, 12 files, ~2,500 lines |

---

## Quick Start

![what you need to run it](docs/images/topology.svg)

The gateway needs three things to speak: an LLM backend, an STT endpoint and a TTS endpoint. All of them are env-overridable and default to the in-house services, so a minimal run is just:

```sh
docker build -t voice_gateway .
docker run --network host -e NANOBOT_TOKEN=token voice_gateway
```

- **`NANOBOT_TOKEN` must be non-empty** — every satellite WebSocket is authenticated against it with a constant-time compare; an empty token means every socket is rejected. `/ota` hands the same token to the firmware.
- **`--network host` is not optional** — ESP32 satellites reach the gateway at `<host>:18792` from the LAN, and an unpublished bridge container is unreachable. This exact misconfiguration once caused a total satellite outage; host networking also keeps the `req.url.hostname` in `/ota` responses correct.
- **`LLM_BACKEND`** picks the brain: `nanobot` (default), `hermes` or `cascade`. Production runs `cascade` alongside the two companion services, which live in the parent `ai-prod` compose project (not part of this repo):

```sh
docker compose up -d --no-deps --build jev-router smolagents-worker voice_gateway
```

Full environment table, backend variants, REST API and OTA: [`docs/REFERENCE.md`](docs/REFERENCE.md).

---

## Why a separate layer exists

1. **Audio is real-time, the LLM is not.** VAD cut points, TTS prefetch, echo windows and watchdogs all have their own timing. Something has to own that timing — that is this process.
2. **Devices differ, the brain does not.** ESP32 sends Opus frames over WebSocket, a camera streams raw PCM over RTSP. After `audio → text` both paths converge into the same queue and the same backend.
3. **Echo and false wakes are a system-level problem.** A mic hears its speaker, a camera hears the corridor satellite, a room hears its neighbour. The guards are shared and global (`GLOBAL_TTS_UNTIL`), because one device's playback must mute every other device's mic.

---

## What it is made of

| File | Responsibility | Why it is separate |
|---|---|---|
| **`main.py`** (~3,600 lines) | Core: ESP32 WebSocket protocol, REST API, OTA, web UI, camera session startup | Single entry point for everything |
| **`camera_client.py`** (~2,800 lines) | One `CameraSession` per stream: RTSP mic feed, wake word, cross-camera arbitration, replies into the WebRTC track | Cameras have their own transport and their own hard problem — echo arriving 3–40 s late |
| **`engine.py`** (~480 lines) | Silero VAD + openWakeWord scoring — the "science" part, pure ONNX calls | Testable without a network, without a camera |
| **`backends.py`** (~590 lines) | Who answers: `NanobotBackend` / `HermesBackend` / `CascadeBackend` | One interface — the brain is swappable with a single env var |
| **`audio_utils.py`** (~220 lines) | `pack_ogg()` (Opus → Ogg for Whisper) and `is_valid_text()` (drops hallucinations and mic echoes of our own TTS) | Shared by both device paths |
| **`services/jev-router/`** | **L1** — intent classifier, offline slot resolver, direct Home Assistant calls, weather, chat/expert proxy | ~0.1 s for a simple command, no LLM in the loop |
| **`services/smolagents-worker/`** | **L2** — smolagents `CodeAgent` with HA / Qdrant / Hermes / weather tools, plus honesty vetoes | Multi-step tasks that must not lie about side effects |
| **`tests/`** | 177 tests across 12 files | Pins the wake-gate bands, the protocol and the cascade contract |

> Production (`docker-compose.yml`) currently runs `LLM_BACKEND=cascade` with **`DISABLE_CAMERAS=true`** — the camera subsystem is implemented and tested but switched off in the live stack; only the ESP32 path is active.

---

## One turn on an ESP32 satellite

1. **Connect** — `voice_ws()` (`main.py`) compares `?token=` against `NANOBOT_TOKEN` in constant time, allocates a session and its `state` dict, replies `hello` with the audio contract (Opus 16 kHz mono, 60 ms frames) and requests the device's MCP tool list (`tools/list`, id 999).
2. **Wake** — the wake word is detected **on the device**; it sends `listen:start` → status `LISTENING`, buffers cleared, VAD reset.
3. **Capture** — each binary message is one 60 ms Opus frame (v1 raw / v2 16-byte header / v3 4-byte header) → decoded to PCM16@16k → `VadEngine` counts silence.
4. **Cut** — on `listen:stop`, or after ~8 quiet frames following real speech → `PROCESSING` → `process_audio_and_send()`.
5. **Gates, cheapest first so noise costs nothing** — echo guard (`GLOBAL_TTS_UNTIL`: while our TTS plays we do not listen) → minimum 15 frames (~0.9 s) → `ENERGY_THRESHOLD` / `MIN_SPEECH_RATIO` → cross-device **speaker lock** (two satellites hearing the same person answer once).
6. **STT + identity in parallel** — `pack_ogg()` then two independent HTTP calls overlapped with `asyncio.gather`: Whisper and Speaker ID.
7. **Dispatch** — the transcript goes to the selected backend.
8. **Speak back** — sentences land on an `asyncio.Queue` → **prefetch player** (sentence N+1 is synthesised while N plays) → pydub decode/resample → Opus → speaker, bracketed by `tts start` / `tts stop`.
9. **Finalise** — if the reply ends with `?`, or matches a Russian interrogative/imperative, the mic stays open (`STANDBY_TIMEOUT_QUESTION`, 30 s); a plain statement drops to standby after 10 s.

A turn that produces no answer inside `WATCHDOG_TIMEOUT` (90 s) is cut short with the fallback apology instead of leaving the user in silence.

---

## Cameras: always-on microphones

Everything here exists to avoid listening to garbage.

![the wake gate cascade](docs/images/wake-gates.svg)

- **Transport** — `ffmpeg` pulls `rtsp://go2rtc:8554/<stream>?audio=copy` → raw L16 @ 16 kHz (no μ-law, which would degrade recognition).
- **Two echo guards** — a hard one (while TTS frames are queued, the mic is not fed at all) and a correlational one, `_is_echo()`, which cross-checks the incoming mic chunk against a ring buffer of recently played audio: the RTSP backchannel returns our own speech **3–40 s after playback**, so fixed suppression windows never work.
- **Wake-gate cascade** (before any wake fires): own-playback guard → appliance hold (sustained background like a vacuum demands a confident score) → quiet-source hold → distant-source veto (a through-wall copy must not beat the real room) → clipping-bang / crest-factor gates → debounce (2 qualifying chunks out of the last 3) → STT-confirm → **cross-camera arbiter** (the first detector becomes the interaction owner; a room ≥5× louder may steal a not-yet-dispatched wake).
- **Whisper only inside a wake window.** Ambient speech from a TV never reaches STT — this is the main defence against false bills and wasted latency.
- **Answer** goes into the camera's WebRTC sendonly track → go2rtc backbridge → camera speaker.

---

## The brain: three backends, three levels

`LLM_BACKEND` picks a `BaseLLMBackend` implementation. All three honour the same contract (`backends.py`):

```python
async def generate_response(text, session_id, stream_name, response_queue)
# pushes already speakable sentences, terminates with EXACTLY one None,
# never raises
```

Why this contract: the player blocks on `queue.get()`, so a lost `None` hangs the voice turn, and a raised exception means silence instead of an apology. **A backend failure must yield a short reply, never a frozen mic.**

Production runs **Cascade**:

| Level | Service | Does | Why |
|---|---|---|---|
| **L1** | `jev-router` :8091 | Classifies on local Ollama embeddings (5 routes), resolves slots offline, calls Home Assistant directly, builds weather, proxies chat/expert | "Turn off the light" reaches Home Assistant in ~0.1 s with no LLM in the loop |
| **L2** | `smolagents-worker` :8092 | `CodeAgent` over `ha_action` / `ha_read` / `qdrant_search` / `hermes_expert` / `weather_forecast`, 120 s cap with heartbeats | Multi-step tasks that need tools |
| **L3** | Hermes | Expert answers | The expensive brain, called rarely |

Two valves keep L2 honest: the **honesty veto** (a reply claiming success after failed tool calls is replaced by the recorded truth, plus exactly one bounded retry carrying the raw errors) and the **near-miss hint** (`«кашеварку» → «кофеварка»`, difflib over the device dictionary) so escalation arrives with context instead of guessing. The satellite's last 4 finished turns ride along as well, so a pronoun has an antecedent.

---

## Design rules worth remembering

1. **Queue of sentences, not tokens** — TTS is sentence-granular; token-by-token synthesis produces audible gaps.
2. **TTS prefetch hides latency** — the first played audio (usually the ack) cancels the 90 s watchdog, which is what makes 48–120 s L2 turns possible.
3. **`engine.last_score` must be written by `check_wakeword()`** — camera sessions read it after every call; a missing write silently kills all wake detection (this bug has shipped once).
4. **Echo is the main enemy** — RTSP backchannel delays of 3–40 s mean suppression is derived from correlation and extended to `play_end + 45 s`, never a fixed window.
5. **`network_mode: host` is mandatory** — ESP32 satellites cannot reach an unpublished bridge container (this misconfiguration already caused a total satellite outage).
6. **Every gate threshold is room-calibrated** — wake scores, hold levels and background medians were tuned against three specific rooms; a new room needs recalibration or it will get false fires or deafness.

---

## Repository layout

```
voice_gateway/
├── main.py                 # FastAPI app: ESP32 WS, REST, OTA, camera startup
├── camera_client.py        # per-camera session (RTSP in, WebRTC out)
├── engine.py               # Silero VAD + openWakeWord
├── backends.py             # Nanobot / Hermes / Cascade backends
├── audio_utils.py          # pack_ogg(), is_valid_text()
├── config/                 # devices.json, wake-word ONNX heads
├── services/
│   ├── jev-router/         # cascade L1
│   └── smolagents-worker/  # cascade L2
├── tests/                  # 177 tests / 12 files
└── docs/
    └── REFERENCE.md        # full config tables, REST API, known issues, changelog
```

- **Configuration, REST API, environment variables, known issues and the full changelog:** [`docs/REFERENCE.md`](docs/REFERENCE.md)
- **Tests:** 177 tests across 12 files covering engine scoring and wake gates, camera arbitration, the ESP32 protocol, cascade streaming, honesty vetoes, router resolution and dialogue memory. Run them inside the container — see [`docs/REFERENCE.md`](docs/REFERENCE.md#tests), the host Python usually lacks `opuslib` / `onnxruntime`.

---

## License

Dual-licensed — pick one:

| | License | Price | Applies when |
|---|---|---|---|
| **A** | **[GNU AGPL v3 or later](LICENSE)** | free | you accept copyleft: derivative works and network services must stay open source |
| **B** | **[Commercial License](COMMERCIAL-LICENSE.md)** | **paid** | you want it in a **commercial product** — bundled with hardware, shipped closed-source, offered as SaaS, or licensed away from the AGPL |

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: alexander.zavaruev@gmail.com
```

Third-party components (openWakeWord model files, Silero VAD, Vosk models)
keep their own licenses — see [`NOTICE`](NOTICE).

