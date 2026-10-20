# Voice Gateway

![version](https://img.shields.io/badge/version-v2.52-blue)
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

The camera subsystem runs **four rooms** — `livingroom`, `kitchen`, `entrance1` and `camera-corridor1`. The wake word there is detected by **decoding** it with a local vosk model (`vosk_wake.py`, 88 MB, CPU-only, one per process shared by all rooms) rather than by scoring it acoustically: a head trained on 27 recordings of the word learned the envelope instead of the word and fired 114–229 times per hour on the television. Measured on the living room: 5/5 live detections, 0 false accepts over 7 minutes of TV. Set `WAKE_VOSK_MODEL_<NAME>` to enable per room; without it a room keeps the acoustic path.

Two lessons from bringing the rooms up are worth stating here, because both cost a day each and neither is visible in the code:

- **`audio.volume` must be *applied*, not just written.** The OpenIPC web UI writes the value to `/etc/majestic.yaml` and keeps the old one in memory until you press apply. A camera left on `volume: 30` read `rms 4` (−77.8 dB) with `volume: 100` already sitting in the config file, and looked indistinguishable from dead hardware — a 700× gap from a healthy room that announced itself nowhere in the log. Re-reading the file is not evidence that a setting is in force.
- **A room that does not wake and a room with nobody in it are indistinguishable from one measurement.** Every conclusion that a microphone was deaf came from a recording made while the room was empty. `CAMERA_VOSK_DUMP_<NAME>` (below) exists because that is the mistake you cannot catch by trying harder.
- **Two rooms stutter and two do not, on identical hardware, identical firmware and identical config — because of where the access point is.** Every camera here runs on `wlan0`, and the WiFi signal orders the rooms: 80/64 and 77/63 are the two fast rooms, 60/42 the corridor, and the kitchen has been seen at both 30/63 (0.29×) and 1.06× on the same hardware. `majestic` is single-threaded and one link carries both directions — the outgoing stream and the incoming `POST /play_audio` — so when it cannot keep up its output audio channel does not get data in time: `Device 0 is empty`, which is what stuttering *is*. Three days were spent comparing what was inside the cameras before anyone read `/proc/net/wireless`. **"Same hardware, same firmware, same config" is only what you checked.**
- **RETRACTED (17.10.2026): "capture speed predicts the stutter" — it predicts only the capture side.** Measured on the same day: the corridor delivers its stream at 0.96–0.79× while its `/play_audio` runs at ×4.45, with `Send-Q` 0 and no unread peer anywhere. Playback has its own cause and its own signature, and the two have to be measured separately. This sentence was written into 2.40 and it was wrong.

See [`docs/REFERENCE.md`](docs/REFERENCE.md) → *Known Issues & Current Problems*.

### Numbers from the production log

| What | Measured |
|---|---|
| Simple command, L1 fast path (after STT) | **0.08–0.23 s** («выключи свет в гостиной» → 0.08–0.11 s) |
| Weather on the deterministic L1 route | **0.38 s** |
| L2 turn with a verified side effect | **3.2–4.0 s** |
| Wake debounce | ~240 ms (2 of 3 chunks); a single chunk ≥ 0.68 fires immediately |
| Wake window / watchdog | 15 s / 90 s |
| Test suite | 663 tests, 19 files, ~13,194 lines |

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
| **`main.py`** (~4,000 lines) | Core: ESP32 WebSocket protocol, REST API, OTA, web UI, camera session startup | Single entry point for everything |
| **`camera_client.py`** (~5,850 lines) | One `CameraSession` per stream: RTSP mic feed, wake word, cross-camera arbitration, replies over the camera's `/play_audio` | Cameras have their own transport and their own hard problem — the mic hears its own speaker |
| **`engine.py`** (~570 lines) | Silero VAD + openWakeWord scoring — the "science" part, pure ONNX calls | Testable without a network, without a camera |
| **`backends.py`** (~610 lines) | Who answers: `NanobotBackend` / `HermesBackend` / `CascadeBackend` | One interface — the brain is swappable with a single env var |
| **`telegram_client.py`** (~1,190 lines) | Third request source: Telegram long-polling, chat allowlist, voice note → Whisper → the same backend → text reply (+ optional voice note) | No bot framework — three Bot API calls over the `aiohttp` the process already has; all main.py helpers injected (no import cycle) |
| **`audio_utils.py`** (~220 lines) | `pack_ogg()` (Opus → Ogg for Whisper) and `is_valid_text()` (drops hallucinations and mic echoes of our own TTS) | Shared by both device paths |
| **`vosk_wake.py`** (~290 lines) | Wake word by **decoding** it (per-room model, free decoder, no grammar) | Decoding has no ceiling where scoring has one — see *Three instruments* above |
| **`services/jev-router/`** | **L1** — intent classifier, offline slot resolver, direct Home Assistant calls, weather, chat/expert proxy | ~0.1 s for a simple command, no LLM in the loop |
| **`services/smolagents-worker/`** | **L2** — smolagents `CodeAgent` with HA / Qdrant / Hermes / weather tools, plus honesty vetoes | Multi-step tasks that must not lie about side effects |
| **`tests/`** | 645 tests across 19 files | Pins the wake-gate bands, the protocol, the cascade contract, the Telegram source, the utterance endpointing and the side-effect verification |

> Production (`docker-compose.yml`) runs `LLM_BACKEND=cascade` with **four cameras enabled** (`CAMERA_STREAMS=livingroom,kitchen,entrance1,camera-corridor1`, `DISABLE_CAMERAS=false`), all four decoding the wake word with vosk and playing replies over `/play_audio`; the ESP32 path remains the main one.

### Utterance endpointing (off by default, on in three rooms)

The living-room VAD never reports silence — Silero said `speech=True` for a whole
capture (0 `speech=False` in 15 live minutes) — so every command used to wait out
the full 7 s duration cap before reaching Whisper. `_PauseEndpoint` ends an
utterance on a dip in the **level** envelope instead, which happens between words
whatever the VAD thinks. Lowering the cap is not the fix: it is load-bearing for
long commands.

    CAMERA_PAUSE_ENDPOINT[_<NAME>]=true      # default false; true in livingroom, kitchen, camera-corridor1
    CAMERA_PAUSE_RATIO[_<NAME>]              # dip depth, default 0.55
    CAMERA_PAUSE_RUN_FRAMES[_<NAME>]         # dip run before the end, default 6 (0.96 s)
    CAMERA_PAUSE_MIN_SPEECH_FRAMES[_<NAME>]  # speech before a pause may end, default 5

Every end logs `via <reason>` with the `rms/ref/floor/run/speech` numbers behind
it and tallies per boot — `since boot: {'pause': N, 'cap': M}` answers whether the
endpoint is doing anything without an argument. `0` means "unset, use the default",
so a half-filled override cannot silently disable a threshold. Enable it per room
only after reading that room's `UTTERANCE END` lines on real audio;
`bash scripts/verify_endpoint.sh` runs the whole check in one command.

`run_frames` sets the shortest utterance the detector will accept, and the default
of 6 is **below a short command**: the living room runs at 15 (minimum 3.20 s),
because at 6 the detector cut «включи свет» after the wake word and the ~0.96 s gap
before the command, and dispatched «В плечи свет». The direction is always to
under-run rather than over-run — a chopped command is a wrong command, while a slow
one is recoverable.

### Answering a question without the wake word

    CAMERA_FOLLOWUP[_<NAME>]=question   # none | question (default) | all

A camera that asks a question must be able to hear the answer, so the default is
`question`: the microphone stays open for `CAMERA_DIALOGUE_QUESTION_S` (30 s)
after **our own question** and a reply needs no keyword. It is a mode and not a
duration because `0` means "unset, use the built-in default" throughout the
configuration — a window of `0` would produce a 30 s window instead of none — and
an unrecognised value falls back to `question`, so a typo in the environment
cannot close the microphone for a question.

`all` also opens the 10 s statement window and is not the default, on a
measurement: in a room whose television is louder than its occupant a statement
window accepted noise while refusing the user, and **no peak threshold separates
them** (user 4725 / 9124, television and appliances 13197–32522). The wake word is
the only discriminator measured to work. `none` requires the wake word again after
every reply.

### Three instruments that decide whether the wake word is broken

They are the reason a mute room can be told apart from a deaf one:

| | Where | Answers |
|---|---|---|
| `vosk diag` | every 300 s per room | `chunks` advancing, `triggers`/`decodes`, the decoder's own `hyp=`, and `drop[muted=… echo=…]` |
| `CAMERA_WAKE_MIN_PEAK[_<NAME>]` | `0` → 3000 | Refuses a decoded wake word on silence, so a dead mic cannot invent both the word and the command |
| `CAMERA_VOSK_DUMP_<NAME>` | off; per room | Writes the exact bytes handed to the decoder, so a question about the **live** path can be replayed offline |

`vosk diag` is deliberately rare rather than a health line — it is the debug aid
that settles a silent room, not something to read every turn. Two readings matter
more than the rest:

- **`muted` frozen while `echo` climbs** is the signature of room noise being
  counted as our own echo. Nothing was playing, so nothing could have been our
  echo, and yet the correlator kept firing.
- **`chunks` advancing at a fraction of its expected rate** means the camera is
  delivering audio slower than real time. A stretched signal contains no word for
  any decoder, so the wake word "not recognised" and the wake word "refused" look
  identical from the outside.

The gate reads the **loudest chunk of the last 1.9 s**, not the chunk that
triggered it — «компьютер» spans about ten 160 ms chunks and the trigger lands at
an arbitrary point inside the word. `VOSK WAKE` logs both numbers
(`peak=` and `window=<window max>/<threshold>`) so the number in the log is the
number the decision was made on.

`CAMERA_UTTERANCE_DUMP[_<NAME>]` is a fourth instrument, aimed at the STT side
rather than the wake side: it writes **every** utterance to a host directory as a
wav plus one `manifest.jsonl` line carrying `dur`, `raw_rms`, `peak_raw`, `dB`,
`gain`, the transcript and the verdict — and the verdict is written **after** the
transcript is known, because that is what the sample is for. A level threshold for
STT was attempted and withdrawn on 15.10.2026 for want of both classes of sample;
this is what makes obtaining them routine. It must point at a **host** directory:
the in-container `/tmp/utterances/` dump is a tmpfs and is wiped by every restart,
which is exactly why the samples were not there to measure.

`CAMERA_VOSK_DUMP_<NAME>=<dir>` appends the decoder's actual input to
`<dir>/<NAME>.raw`, after any wake-path filter and before `feed()`, capped at
64 MB (~33 min per room) and flushed every chunk so the file can be copied out of
a live room. It exists because **an external recording cannot answer a question
about the live path**: an `ffmpeg` capture of the same go2rtc stream clipped at the
rail (peak exactly 32768) while the gateway received 7448–11201 over the same
seconds. Two different signals, so every offline reproduction built on the capture
was invalid by construction. When you use it, confirm the utterance was actually
spoken — in the log, at the same timestamp — or the file is a recording of an empty
room and it will support any conclusion you put it to.

### The camera watchdog

    */3 * * * * voice_gateway/scripts/camera_watchdog.sh

Five states, because three independent signals are measured: HTTP latency, whether
go2rtc holds a producer with a consumer, and **how fast the audio arrives**.
`audio-starved` is the one that matters in practice — a stream that is alive and
carrying audio at a fraction of real time is invisible to latency and to
`producers/consumers`, and it is not fixable by restarting the gateway, so those
rooms skip that step and go straight to restarting `majestic` on the camera (the
cheapest lever; a power cycle is the fallback, because ONVIF `Reboot` is not
implemented on OpenIPC).

The rate is read from the gateway's own log rather than measured by a second
consumer: a watchdog must not perturb the single-threaded `majestic` it watches.
`DRY_RUN=1` decides and logs what it would do and touches nothing — the ladder's
first step restarts the gateway, so testing it the hard way is what once made it
restart a healthy container seventeen times in a day.

**Known gap — capture speed.** A camera can be healthy on every signal the watchdog
measures and still be ×4.45 on `/play_audio`: HTTP latency, stream presence and
level are normal while the speaker takes four times the audio length. **Capture
speed separates the two classes on the capture side** (measured 17.10.2026, three
12 s captures per room: 1.08 / 1.08 / 1.08 fast, 0.96 / 0.74 / 0.79 degraded) but
it says nothing about playback, which has its own cause. It is **not implemented
yet**, and it would not have caught the corridor.

### Repository gates

    bash scripts/verify_endpoint.sh     # the full check, in one command
    python3 scripts/check_lang.py --self-test && python3 scripts/check_lang.py
    python3 scripts/check_doc_headings.py --self-test
    python3 scripts/check_env_wiring.py --self-test
    python3 scripts/check_doc_numbers.py --tests <N>

Two of these are about the documents and the source rather than the behaviour,
and both exist because of a specific way of being wrong:

- **`check_lang.py`** enforces the rule that prose is English and Cyrillic is
  data. The distinction is what makes it usable: «компьютер», «включи свет» and
  every other Russian string in this repository is payload, while the sentence
  around it is documentation. It removes string literals by AST, so the L2 system
  prompt and the `@tool` descriptions the model reads are never touched — those
  are data too, and translating them would change model behaviour.
- **`check_doc_headings.py`** catches a heading spliced to a copy of itself in
  `AGENTS.md`. Three earlier versions of that check passed *while the defect was
  present*, so the rule every gate here follows is: **`--self-test` first, and a
  check that has never been seen failing proves nothing.**
- **`check_env_wiring.py`** catches an environment variable set in
  `docker-compose.yml` that the code never reads — the quietest failure in this
  stack, because the container starts, the flag looks enabled, and nothing
  happens. It resolves the per-room scheme (`f"CAMERA_PAUSE_ENDPOINT_{name…}"`),
  without which it would report twenty-eight false positives on the first run.
- **`check_doc_numbers.py`** catches README numbers that have drifted from the
  repository. It found a real gap on its first run: `vosk_wake.py` was in the
  repository layout with no row in the component table. The suite size is passed
  in from the pytest run rather than starting a second one.

### One table, and a recorded history

    python3 scripts/gate_status.py [--days 7]

Four gates printing four blocks is readable; eight is not, and the failure is not
cosmetic — **a gate whose result scrolls off the screen is a gate that stops being
run**, and then it protects nothing while still being in the repository. So
`verify_endpoint.sh` ends with one table: every gate, its verdict now, what it
catches, and **when it last caught something**.

The history columns are read from `logs/gates.jsonl`, which every run appends to.
They are deliberately not hand-maintained — a typed-in number is exactly what
`check_doc_numbers.py` exists to stop. Before that history began the column reads
`unrecorded` rather than `never`, because two of these gates had already
demonstrably fired; nothing is backfilled, because an invented row is the kind of
fact this project keeps being corrected for.

**`RUNS` matters more than `FIRES`, and they mean opposite things.** A gate that
never fires because the tree is clean is working. A gate that never **runs**
protects nothing while still looking like protection — and the second is the
failure this whole table exists to catch, so it is marked `NORUN` and exits
non-zero. `NORUN` is only raised when a gate's neighbours *did* run in the window,
so a fresh clone with no history is not reported as four broken gates.

`gate_status.py` also treats **an unregistered `check_*.py` as a failure**. A gate
that exists and is not wired into the runner is indistinguishable from no gate at
all, which is the same shape as the bug in `check_lang.py`'s own first self-test,
where the sample passed on the code it was meant to refute.

All four run inside `verify_endpoint.sh` — the first three as step 1/7, before
anything is built, and the numbers at step 6/7 where pytest has already run. The
summary goes last, once the test count exists.

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
- **Two echo guards** — a hard one (while TTS frames are queued, the mic is not fed at all) and a correlational one, `_is_echo()`, which cross-checks the incoming mic chunk against a ring buffer of recently played audio. The correlator is **bounded by `_ECHO_HORIZON_S`** (8 s after playback ends): the 3–40 s late return that once justified a wide lag sweep belonged to the ONVIF backchannel, which is no longer in the path. Unbounded, it did not just miss echoes — it counted room noise as our own voice and held the wake gate shut for as long as the noise lasted.
- **Wake-gate cascade** (before any wake fires): own-playback guard → appliance hold (sustained background like a vacuum demands a confident score) → quiet-source hold → distant-source veto (a through-wall copy must not beat the real room) → clipping-bang / crest-factor gates → debounce (2 qualifying chunks out of the last 3) → STT-confirm → **cross-camera arbiter** (the first detector becomes the interaction owner; a room ≥5× louder may steal a not-yet-dispatched wake).
- **Whisper only inside a wake window.** Ambient speech from a TV never reaches STT — this is the main defence against false bills and wasted latency.
- **The player does not wait for the upload before the next sentence.** `/play_audio` answers 3–12× late on a weak link, and the player used to sit inside that call for the whole of it, so the gap between sentences was the gap between uploads. The POST runs as a task, bounded by `_MAX_OUTSTANDING_PLAY_POSTS = 2` — a slow link plus unbounded requests is a pile-up of audio the camera is already failing to play — and `_settle_play_posts()` drains them before anything reads `_last_play_ok`, which is what decides whether a failed sentence adds an apology. The **cues keep the waiting behaviour on purpose**: the attention pip is what this class of bug started with.
- **Answer** goes to the camera's `/play_audio` endpoint as raw mono s16le with the rate in the `Content-Type` header (`;rate=48000`). The ONVIF/go2rtc backchannel is a **fallback only** and it is pitch-broken — the camera advertises PCMU/8000 and then plays it at 48 kHz, so replies come out ~6× too fast. Set `CAMERA_WEBRTC[_<NAME>]=false` for a room that has `/play_audio`: the session carries no audio this gateway uses, and every `webrtc/offer` makes go2rtc rebuild the producer, which fills the **camera's** send queue and blocks it.

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
4. **Echo is the main enemy, and a correlation without a playback horizon is worse than no correlator** — suppression is derived from correlation against recent playback, and every correlation is only believed within `_ECHO_HORIZON_S` of playback actually ending. A wide sweep over a ring buffer that is never cleared will eventually claim room noise as our own voice, and then the block it opens can be extended indefinitely. (Shipped, measured, fixed — 07.10.2026.)
5. **`network_mode: host` is mandatory** — ESP32 satellites cannot reach an unpublished bridge container (this misconfiguration already caused a total satellite outage).
6. **Every gate threshold is room-calibrated** — wake scores, hold levels and background medians were tuned against specific rooms; a new room needs recalibration or it will get false fires or deafness.
7. **Arbitrate on level relative to the room's own noise floor, never on raw level** — the four microphones differ by ~4× in sensitivity, so a raw comparison is mostly a measurement of which mic is loudest. A room whose floor cannot be established scores 0 and stands down: another room answers, never the wrong one carrying the command. (Fixed 13.10.2026.)
8. **Bound a cascade instead of trying to detect it perfectly** — after `_FOLLOWUP_CHAIN_MAX = 2` consecutive turns answered from a window opened by our own reply, the next reply opens no window and the wake word is required again. It is one-directional on purpose: it can only cost a wake word, never a wrong action. (Shipped 13.10.2026.)
9. **A verdict needs an instrument behind it, and "I don't know" is a third outcome** — an on/off device is proven by its own power sensor; where none exists the turn is `unproven`, spoken as silence after «Включаю», never as a claim in either direction. (Shipped 16.10.2026.)
10. **Confirm the stimulus reached the sensor before calling the sensor broken** — a "dead microphone" conclusion is invalid until the stimulus is visible in the log at the same timestamp. Four separate false diagnoses of a mute room came from recordings made in an empty one.

---

## Repository layout

```
voice_gateway/
├── main.py                 # FastAPI app: ESP32 WS, REST, OTA, camera startup
├── camera_client.py        # per-camera session (RTSP in, WebRTC out)
├── engine.py               # Silero VAD + openWakeWord
├── backends.py             # Nanobot / Hermes / Cascade backends
├── telegram_client.py      # Telegram source: long polling, allowlist, /room
├── audio_utils.py          # pack_ogg(), is_valid_text()
├── vosk_wake.py            # wake word by DECODING (per-room model, free decoder)
├── config/                 # devices.json, wake-word ONNX heads
├── services/
│   ├── jev-router/         # cascade L1
│   └── smolagents-worker/  # cascade L2
├── tests/                  # 645 tests / 19 files
└── docs/
    └── REFERENCE.md        # full config tables, REST API, known issues, changelog
```

- **Configuration, REST API, environment variables, known issues and the full changelog:** [`docs/REFERENCE.md`](docs/REFERENCE.md)
- **Tests:** 645 tests across 19 files covering engine scoring and wake gates, the camera audio track and its echo guards, the ESP32 protocol, cascade streaming, honesty vetoes, HA entity matching, router resolution, dialogue memory and the Telegram source (allowlist, ack vs final text, turn timeout, voice turns, `/room`). Run them inside the container — see [`docs/REFERENCE.md`](docs/REFERENCE.md#tests), the host Python usually lacks `opuslib` / `onnxruntime`. CI (`.github/workflows/tests.yml`) runs the same recipe on every push and PR.

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

