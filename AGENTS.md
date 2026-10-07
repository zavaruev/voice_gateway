# Voice Gateway

WebSocket gateway connecting ESP32 smart speakers **and OpenIPC IP cameras** (corridor / kitchen / livingroom) to an AI pipeline (openWakeWord → Whisper STT → Nanobot LLM → Edge TTS).

## Entrypoint

`main.py` — FastAPI app on port 18792, run via `python -u main.py` (uvicorn inside `__main__`).

## Core modules

- `main.py` (~3,590 lines) — FastAPI app, ESP32 WebSocket protocol, REST API, OTA, firmware upload
- `camera_client.py` (~2,800 lines) — per-camera session: RTSP audio feed, VAD, wake-word gate cascade, cross-camera arbiter, go2rtc self-healer, TTS playback pipeline
- `engine.py` — Silero VAD wrapper + openWakeWord scoring (`check_wakeword()` stores the raw score in `.last_score`)
- `backends.py` — `BaseLLMBackend` implementations: `NanobotBackend` (WS), `HermesBackend` (OpenAI SSE), `CascadeBackend` (jev-router SSE); all take an optional `room=""` default-area hint (only cascade forwards it to the router)
- `telegram_client.py` — Telegram as a third source: long-poll `getUpdates` loop (no webhook), chat allowlist, voice note → Whisper → same `llm_backend` → streamed text via `editMessageText` (+ optional ogg/opus voice reply), commands `/start` `/room` `/voice`; helpers (`transcribe`/`tts_mp3`/`make_session_id`) are INJECTED from `main.py` — it never imports `main` (cycle). Per-chat history key is `tg:<chat_id>`, the `/room` binding travels as the backend's `room` kwarg (never folded into `stream_name`)
- `tests/` — pytest suite (`pytest tests/`), 16 files / 5 381 lines / 372 tests (media transport + the invented-intent whitelist added 03.10.2026, the `_CORRIDOR_STREAMS` rename guard + the `/play_audio` speaker path added 03.10.2026): router slot resolution + escalation hint + the media fast path (`media__*`, room/pronoun/ambiguity rules, `find_media_targets`), the ESP32 `main.py` protocol, honesty vetoes (action/data/weather) + the retry note, HA REST fallback + data detection, the on/off intent target resolver + room-first registry ranking, the L2 media target resolver + action/value normalisation, the media promise veto + the two new refusal sentences, wake-gate STT-confirm bands, engine wake scoring, camera audio track + echo/SDP guards, cascade backend streaming (incl. the `room` payload), the Telegram source (allowlist/groups gate, ack-vs-final text, turn timeout, voice turn, `/room` persistence), short-term dialogue ring, weather, TTS gate, OTA auth, RMS utils, the Kodi library decision layer (`kodi.py`: title scoring, episode choice incl. the no-substitution rule, box identity by zeroconf name), RU→latin hint sync + the ordinal table + the media service/preference tables (all both copies — the sync test also asserts both levels pick the SAME player for the same command). Cross-camera arbitration is NOT covered — `test_wake_gates.py` stubs the arbiter. CI: `.github/workflows/tests.yml` runs pytest inside the built image; no linter.

## External services (all env-overridable)

| Service | Default URL | Role |
|---|---|---|
| Nanobot | `ws://nanobot:8765/` | AI brain (WebSocket, streamed text) |
| Whisper | `http://192.168.22.111:8000/v1/audio/transcriptions` | STT (OpenAI-compatible) |
| Speaker ID | `http://192.168.22.102:8001/identify` | Speaker recognition |
| Edge TTS | `http://edge_tts:5050/v1/audio/speech` | Text-to-speech (OpenAI-compatible) |
| go2rtc | `http://192.168.22.102:1984` | Camera stream registry (RTSP/WebRTC bridge) |
| Telegram Bot API | `https://api.telegram.org` | Third source (`telegram_client.py`, long polling; container needs outbound 443) |

## Build & run

```sh
docker build -t voice_gateway .
docker run -p 18792:18792 \
  -e CAMERA_STREAMS=livingroom,kitchen voice_gateway
```

`CAMERA_STREAMS` takes go2rtc stream names. The live registry (Frigate's
`config.yaml` → `go2rtc.streams`) is `balcony, corridor1, corridor2, kitchen,
livingroom, pantry` — there is **no plain `corridor`**, it was split into
`corridor1`/`corridor2`, so the room-specific code tests `_CORRIDOR_STREAMS`
rather than the literal name. `DISABLE_CAMERAS=true` is the master off switch.

`GO2RTC_SOURCE_URL_<NAME>` must be the **verbatim** go2rtc config entry,
credentials and `#backchannel=1` included: the cams answer 401 without the
credentials, and the fallback that reads the URL back out of go2rtc's own
listing is lossy (it reports the bare `rtsp://<ip>/stream=0`).

## Architecture notes

- **Camera audio OUT goes over OpenIPC `/play_audio`, not over go2rtc.** Two
  things have to be true or the room is mute, and both look like "the code is
  broken":

  1. **The ONVIF AudioOutput must be enabled on the camera.** `GetCapabilities`
     never lists an audio service and `GetProfiles` has no
     `AudioOutputConfiguration`, so the only evidence it exists is
     `GetAudioOutputs` returning `A_OUT_000`. Until the output is enabled,
     `/play_audio` returns **200 and plays nothing**.
  2. **A go2rtc producer created BEFORE the enablement keeps the old SDP.**
     There is no `sendonly` track until the producer is re-created (heal path:
     DELETE + PUT), and a DESCRIBE without `Require:
     www.onvif.org/ver20/backchannel` hides the backchannel m-line even when it
     is there.

  `/play_audio` takes raw mono s16le with the rate in the Content-Type:
  `curl -u root:PW -X POST --data-binary @x.pcm -H
  'Content-Type: application/octet-stream;rate=48000' http://<cam>/play_audio`
  (OpenIPC wiki, "How to play audio file on camera's speaker over network").
  `TTS_PLAY_RATE = 48000` is that rate, and `_speak_pcm` /
  `_play_attention` / `_play_activation_sound` all route through
  `_play_audio_http()` first.

  **The go2rtc ONVIF backchannel is a fallback and it is pitch-broken**: the
  camera advertises exactly one codec (`m=audio 0 RTP/AVP 0 /
  a=rtpmap:0 PCMU/8000 / a=sendonly / a=control:audio-backchannel`) and then
  plays that 8 kHz stream at 48 kHz, so replies come out ~6x too fast
  ("пищит как бурундук"). Measured 03.10.2026 by playing a 1000 Hz tone and
  FFT-ing the camera's own microphone: **1000 Hz sent, 6000 Hz heard**.
  Pre-stretching by 6 does fix the pitch but makes every reply take 6x longer
  to play, so it is not usable for dialogue — hence `/play_audio`. If you ever
  see `_play_audio_http` fail, expect a squeaky robot voice from the fallback,
  and fix the endpoint rather than tuning the audio.

- **ESP32 STT pipeline**: Opus frames → `pack_ogg()` → parallel POST to Whisper + Speaker ID → text sent to Nanobot via WS
- **Camera audio**: go2rtc RTSP backchannel (raw L16 16 kHz via ffmpeg) → echo guards → Silero VAD → SpeexDSP NS rescue → adaptive normalisation → Whisper **only inside an active wake window** (ambient utterances never reach STT)
- **Wake word**: openWakeWord. Default model is the library head `config/computer_20260706_130638.onnx` (Creator #7074 "Classic V3"; recall 55.6 %, measured positives 0.64 / clean negatives 0.001–0.05) for every room except kitchen, which keeps the custom `config/computer.onnx` (the library head scores clipped kitchen audio 0.001). Override per room with `WAKE_WORD_MODEL_<NAME>`. Raw int16 required — never feed normalized floats. Per-room base thresholds: kitchen 0.40, others 0.30 (they self-clamp back to base after a command). Sliding-window debounce: 2 qualifying chunks out of the last 3 (~80 ms chunks); single chunk ≥0.68 fires immediately. Bidirectional AGC normalises every chunk to a per-room target peak before scoring: kitchen 6500, corridor1/corridor2 9000, others 4000.

- **The wake front-end is a SHARED cost, and the head MLP is the cheap part** (measured 04.10.2026 on 120 s of livingroom audio with the TV on, per 2560-sample chunk = 160 ms, 6.25 chunks/s): openWakeWord `predict()` 13.0 ms of which Google's speech-embedding CNN is 5.9 ms (38 %), the mel spectrogram 2.9 ms, the *buffer rebuild* 3.8 ms, numpy glue in `predict()` 2.6 ms, and the actual wake-word MLP **0.07 ms — 0.5 %**. A bigger or smarter head is therefore essentially free; what costs money is re-deriving the same 96-dim embeddings 12.5x per second per room. Silero VAD adds 3.0 ms (19 %) and ffmpeg RTSP decode ~2.6 % of a core. Per camera that is ~13 % of one core, so 8 cameras ≈ 1.05 cores — fine on the 24-core host, and it is why the wake path scales linearly with room count.

- **openWakeWord's audio buffer is a Python deque and it is rebuilt on every mel update** (the one wake-path bug worth remembering): `AudioFeatures.raw_data_buffer` is a `deque(maxlen=160_000)` and `_streaming_melspectrogram` evaluates `list(self.raw_data_buffer)[-n-480:]` — materialising **160 000 Python ints to use 3 040** of them, on *every* chunk. `engine._use_numpy_ring_buffer()` (called at the end of `initialize_models`) swaps it for an int16 numpy ring, and the mel model gets bit-identical input, so **scores do not move** — measured `IDENTICAL SCORES: True` over 750 chunks, and `tests/test_wake_ring_buffer.py` enforces that against the real ONNX graphs. Worth **2.5 ms/chunk (1.19x, −12.6 % of a core at 8 cameras)**, which is more than the whole `_is_echo` correlation ring (0.0006 ms) costs by four orders of magnitude. Two traps: the replacement methods must be bound with `types.MethodType` (they are instance attributes, so a bare function loses `self`), and they cannot call a closure defined inside the installer — `_ring_buffer_tail` is a module-level function for that reason. Raising the hop from 80 to 160 ms does NOT help: mel is recomputed for the whole 2560-sample window, and batching the CNN across rooms does not amortise either (8-frame batch costs 45 ms vs 6.3 ms for one — the graph is already per-frame and CPU-bound).

- **A wake head trained on 36 bursts learns ENVELOPES, not words — and the CV number will not tell you.** Three attempts, same shape each time: 0.9994 CV, clean offline acceptance, false fires on the TV. The failures were never the classifier and never the hyper-parameters; they were all in the data, and each one has to be fixed by construction, not by tuning:
  1. **Level leaked through the labels.** The ambience negatives were written pre-normalised to peak 4000 while the positives stayed raw at 1964..29490; median rms was 976 vs 2451. A classifier can separate those perfectly without hearing the word, and `diagnose_wake_data.py` reports the separation in sigma so it is visible before training. Fix: EVERY sample of BOTH classes goes through `wake_features.runtime_chunks` (160 ms chunks, per-chunk AGC to peak 4000), so absolute level is identical by construction. Measured crest-factor separation after that: 0.12 sigma.
  2. **Every positive looked identical.** A 1.3 s burst yields exactly ONE 1.28 s window, and it always lands at the END — Whisper's padding in front, the word last. 27 bursts therefore became 27 examples of the shape "silence, …, level jump at the end", and the head learned the jump. `frame_std` (spread of magnitude across the 16 frames) separated the classes by **1.38 sigma** — more than any spectral feature — and on live TV the false fires correlated with it, not with rms (−0.012) or crest factor (+0.018). Fix: `build_wake_dataset.stream_vectors()` drops each burst at a RANDOM offset inside real room audio twice the window length, so the word lands anywhere in the window; windows that miss the word are emitted as negatives from the same stream, which makes the two classes indistinguishable by envelope shape. That turns 27 positives into ~900 each.
  3. **SNR was a label proxy.** Mixing background into negatives only makes "noisy" mean "negative". Both classes get the SAME SNR list (30/20/14/9/5 dB) and the same number of background draws.
  4. **CV over windows is meaningless** when each burst is expanded ~900 ways — the same recording lands in train and test. `train_wake_head_v2.py` uses `GroupKFold` on the burst name. 0.9994 under a random split is not information; the acceptance numbers to trust are the held-out BURSTS and, above all, the live gate below.
  5. **The offline acceptance must be the production code.** `eval_runtime_path.py` replays a raw capture through per-chunk AGC + the real 2-of-3 debounce; `why_it_fires.py` re-derives features from a continuous stream to find class-conditional structure when a model passes everything and still misbehaves.

- **The live gate is the only acceptance that counts, and every trained head failed it** (04.10.2026). Measured on the **held-out 126 s tail** of the TV capture (the first 294 s are in the training set — scoring the whole file measures training data, which is how an earlier run "fired 12 times" and that was partly an artefact of the counter too, see below), counting distinct EVENTS (a new one after 2 s of silence, not per chunk):

  | head | events/hour on TV at 0.85 | max score | verdict |
  |---|---|---|---|
  | library `computer_20260706` | **0.0** | 0.5351 | silent, but 0/36 on the word |
  | `lr_v3` (old features) | 60 | 0.9948 | unusable |
  | `lr_v4` (level-blind, random CV) | 114 | 0.9996 | unusable |
  | `lr_v5` (level-blind, burst-disjoint CV + burst-disjoint acceptance) | 229 | 1.0000 | unusable |

  `lr_v5` is the interesting one because its data is clean: 36 528 windows, 7 297 positives from 27 bursts, 60 groups, crest-factor separation 0.12 sigma, both classes mixed over the same SNR list, `GroupKFold` AUC 0.9948, and on ten **unseen** speech bursts it scored 9 of them below 0.29 — one single burst («Секунду, занимаюсь.», /sʲɪˈkundʊ/, a plausible collision with /kɐmˈpʲjutʃɪr/) at 1.0000. So the head did learn roughly the right thing. It still fires 8 times in two minutes of television, because sustained speech at the same level as the user simply is not separable from 27 examples of one word. **The binding constraint is 27 recordings, not the classifier, the features or the threshold** — do not spend another attempt on tuning.

- **Two measurement bugs that made bad models look good** (both cost a full training cycle each):
  - `eval_runtime_path.fires()` counted every CHUNK whose 2-of-3 debounce was satisfied, so one 1 s episode was reported as ~6 "false fires". Always count events (new onset after ≥2 s of silence) before quoting an FA/hour number.
  - The acceptance set initially re-used the same 42 speech bursts the model trained on, and the trainer happily printed `neg_holdout max 1.0000` next to the recall as if it were generalisation. It was memorisation. Both classes are now split by BURST (`--holdout-fraction`), and `neg_label_leak` is gone: `speech_span()` answers "is there speech here", which on a speech burst is true nearly everywhere, so using it as the wake-word span marked 12 599 windows positive (vs 7 297 real positives) and taught the head that ordinary speech is the wake word. Negative bursts get an EMPTY span.

- **The wake word is DETECTED BY DECODING in rooms with `WAKE_VOSK_MODEL_<NAME>`** (`vosk_wake.py`, live in livingroom since 04.10.2026). No acoustic head can pass here: the honest limit was 27 usable recordings, and every trained head either ignored the word or fired on the TV. A decoder has no such ceiling, because "did it hear the wake word" becomes a lookup instead of a discrimination problem.

- **USE A FREE (UNCONSTRAINED) DECODER — DO NOT "OPTIMISE" IT WITH A GRAMMAR.** A two-stage design (fast grammar trigger + free confirmation) was built, measured as good on isolated bursts, and removed after it was caught failing on live audio. Restricted to «компьютер» + 33 filler words the recogniser produced **one hypothesis across 25 s of loud live speech**, while the free decoder produced 27 on the very same bytes. The grammar had 100 % recall on 1.3 s isolated bursts and **0 %** on the live room: constraining the decoder makes it reject exactly the word the wake word is made of when the speaker is across the room and the television is on. Shipping the free decoder instead: **3/3 detections at 9 dB SNR planted into live room audio, 0 false accepts in 22 s of that same audio**, and 0/99 windows on the held-out television tail. Speed is not worth a detector that cannot hear its own word. `tests/test_vosk_wake.py::test_matcher_uses_a_free_decoder` asserts no grammar is passed, so this cannot be "improved" back into a bug.

- **`AcceptWaveform` IS A STREAMING CALL — FEED IT ONLY THE NEW AUDIO.** Re-feeding the accumulated buffer makes the recogniser hear the same window again on every chunk; by the end of one 3 s window it has consumed ~30 s and its hypothesis collapses to `''`. That presented as `triggers=0 hyp=''` with loud speech in the room, while every offline probe passed (they fed each chunk exactly once). `tests/test_vosk_wake.py::test_feeds_only_the_new_audio` pins the byte count per call.

- **When the wake word does not fire, read `vosk diag:` before anything else.** It reports chunks fed, peak, trigger/decode tallies and `hyp=` — the decoder's own live hypothesis. An empty `hyp` means the decoder is hearing nothing usable, and the next question is whether it was fed anything at all (see the audio-rate watchdog below). `tests/test_vosk_wake.py` also deliberately does **not** `sys.path.insert(0, "/app")`, unlike nothing else in the suite: it did, and it was silently testing the module baked into the image instead of the copy under test.

  Two more rules the numbers forced, both cheap:
  - **FEED RAW AUDIO, NEVER THE AGC-NORMALISED CHUNK.** 94 % recall on raw, **44 %** on the per-chunk peak-normalised stream — openWakeWord's requirement actively destroys a decoder's input. `camera_client` therefore feeds `_vosk_wake` the un-AGC'd `chunk`.
  - **`WAKE_TOKENS` IS AN ALLOWLIST, NOT A "комп..." PREFIX.** A prefix match accepted «компот» and «компания» — a false activation per occurrence. List only the observed spellings (`компьютер`, `комп`, `компютер`, …).

  Cost per room: the decoder runs only while the Silero VAD says someone is speaking, ~1.8 ms per 200 ms chunk (~0.9 % of a core). `get_model()` is a process-wide singleton — 8 rooms share one 88 MB model, one `KaldiRecognizer` each. `max_secs=3.0` bounds the recogniser's context and is the only thing that does: the room's VAD never reports silence (0 `speech=False` in 15 live minutes), so there is no utterance boundary to rely on. **A construction failure logs a warning and falls back to openWakeWord**, which happened twice here for real reasons (`vosk_wake.py` missing from the Dockerfile's COPY list, and a stray `config.` reference inside `_init_audio_processing`, which has no `config` in scope) — so check the log line `vosk wake-word:` after any deploy.

- **Reproduce the PRODUCTION feed path when validating the wake word — an offline probe cannot see the important bugs.** `scripts/repro_production_feed.py` plants a real recorded burst at a known offset in a real room capture and feeds it exactly as `_vad_process` does: `begin()` once, `feed()` chunk after chunk, no utterance boundary, no `flush()`. Three separate defects surfaced only there, and each one looked fine in every fixed-window probe:
  1. stage 2 anchored at the window start rather than at the trigger;
  2. stage 2 running before any audio followed the trigger (`tail_secs`);
  3. the harness itself lying twice — it put its own directory first on `sys.path` and imported a stale `vosk_wake.py` from `/tmp` (0/7 for a build that fires), and it truncated the planted burst to 4000 samples (0.25 s) which no decoder can finalise. When a reproduction reports zero, check the reproduction before touching the code.

- **NEVER RUN BOTH DETECTORS IN THE SAME ROOM.** The openWakeWord block in `_vad_process` is gated on `self._vosk_wake is None`. Running both cost ~13 ms per chunk (~8 % of a core) for nothing in a room that decodes, and left a second, independent source of false wakes — the one thing the livingroom must not have. CPU after the fix: **31 % of one core for the whole gateway** with one camera (was 51 % with both detectors live). `config/computer_livingroom.onnx` — the trained head that fired 229/hour on the television — has been **deleted**, so it cannot be re-enabled by uncommenting an env var; the wake path is vosk-only for this room.

- **`vosk diag` is deliberately rare (every 300 s), not a health line.** It is the debug aid that settled the last two silent-room investigations (chunks fed, peak, tallies, the decoder's own hypothesis). If the room goes mute again, lower the interval and read it before changing anything.

- **Both `_fire_wake` and the vosk branch live in `CameraSession._vad_process`, and the wake block was extracted into `_fire_wake()`** so the two detectors cannot drift apart on arbitration, pip, greeting or the VAD reset. vosk fires on the chunk that confirms and exactly once per utterance — `_fire_wake()` must never be called twice for one wake.

- **THE VAD MUST NOT GATE THE WAKE DECODER, AND THE DECODER'S WINDOW MUST BE COUNTED IN AUDIO — 1 HIT IN 5 WAS TWO BUGS, NOT NOISE.** Both were found by finally capturing the room while the user spoke (84 s, five spoken wake words) instead of reasoning about it:
  1. **The Silero VAD passed 21 of 419 chunks (95 % rejected)** on that recording, where the word was plainly audible. vosk then saw a sparse, discontinuous stream and its hypothesis never developed. The wake decoder is now fed **every** chunk. The VAD still delimits utterances for STT — it just has no authority over audio a streaming decoder needs. This is why every earlier measurement (all of which fed continuous audio) disagreed with the live room.
  2. **`max_secs` was counted in WALL CLOCK, so a sparse feed threw the recogniser away every 3 s** having consumed almost nothing. It is now counted in **audio fed**, and the window is **30 s** — a 3 s window cut through the wake word itself. Measured on that same recording: `max_secs` 3 s → **2 detections**, 8 s / 15 s / 30 s / 60 s → **4 detections**, identical from 8 s up. A bare `reset()` there also left `_rec = None` so every later `feed()` returned False until the caller called `begin()` again; `feed()` now rolls the window over by itself.
  - Net: 1 in 5 → **4 in 5 on the identical recording**, and a fresh live run recorded **5 fires in 14 s for five spoken attempts (5/5)** with **0 false accepts** across 7 minutes of television (294 s training part + 126 s unseen tail, both exactly 0.0/hour). The lesson generalises: *a detector fed a different signal than production is not a measurement*, and this time the difference was not an abstraction but a 95 %-lossy filter sitting in front of it.

- **A FALSE ALARM THAT COST A DEPLOYMENT: DO NOT TRUST AUTOCORRELATION F0 ON A ROOM WITH A TELEVISION.** Mid-investigation, autocorrelation on the livingroom microphone gave a median F0 of 314-390 Hz (normal speech is 85-255) and it was read as "the camera compresses audio 2x". It does not: the recorded wake bursts from the same mic measure **96-125 Hz**, the trained head scores **1.000** on them and **0.000** after decimation, and the live audio carries 13.9 % of its energy in 5-7 kHz and 3.9 % above 7 kHz, which a 2x-compressed signal cannot have. The autocorrelation had locked onto a harmonic of the television's music. A `CAMERA_AUDIO_DECIM` setting was added and then reverted.
  - **How to actually tell whether audio is time-compressed: measure energy ABOVE 7 kHz, or decode it.** A signal squeezed 2x has its 7 kHz content pushed to 14 kHz, so 5-8 kHz is empty. Speech has plenty there. Decoding is even better: `vosk` turned the live audio into coherent Russian («иди сюда напитки»), which a chipmunk cannot produce.
  - **Suspect F0-based conclusions whenever the "voice" is coming from a room with music or a TV**, and confirm on audio you recorded yourself before changing a pipeline.

- **A CAMERA WHOSE AUDIO IS DELIVERED AT A FRACTION OF REAL TIME LOOKS EXACTLY LIKE A BROKEN WAKE WORD.** This is what actually happened on 04.10.2026 and it cost hours: the user said «компьютер», nothing happened, and every offline probe passed. The vosk matcher was fine, the model was fine — the microphone had been dead for over an hour. The chain of evidence, in the order it was found:
  1. The `vosk diag:` line added to `_vad_process` showed `chunks=27` after 90 s. Expected ~560 (6.25 chunks/s at 160 ms) — the stream was at **5 % of real time**.
  2. A standalone `ffmpeg -i rtsp://.../livingroom?audio=copy -t 30` produced **1.1 s of audio in 35 s wall** — so it was not the gateway.
  3. The same against the camera directly (`rtsp://root:2441@192.168.22.241/stream=0`) gave 3-4 %, bypassing go2rtc entirely.
  4. On the camera: `nproc` = **1**, load average 11.5 with only 1 runnable task, `ai0_P0_MAIN` stuck in state D on `CamOsTcondTimedWait`, and go2rtc showing **932 MB of video received vs 10 MB of audio received**.
  5. Turning `rtsp.backchannel` off did nothing (the AO/AI contention theory was wrong). Lowering `audio.volume` 100→50 made it worse (8/s). A camera **reboot** restored it: `speed=1.04x`, 20 s of audio, `rms 397 / peak 2685` — and `audio.volume` must stay at **100**, because at 50 the room is effectively silent (`rms 9`).
  - **Lesson: measure the audio RATE, not just "is audio arriving".** Silence, a low score and a working matcher are indistinguishable without a rate counter. The permanent `speech=True` VAD state that looked like a tuning problem was this same fault seen from the other end.

- **THE STALL HEALERS CANNOT SEE THIS, SO THE RATE WATCHDOG EXISTS.** Every existing recovery path keys off an ffmpeg **timeout** (`asyncio.wait_for(..., timeout=20.0)`), which fires only when delivery stops completely. A wedged mic driver keeps delivering a trickle, so nothing ever fired and the room stayed silently mute. `_rtsp_audio_loop` now accounts bytes against wall-clock over 20 s windows: below `self._min_audio_rate` (0.5) it logs `⚠ AUDIO STARVED`, re-registers the go2rtc stream and restarts ffmpeg; after `self._starve_restarts` (3) failed attempts it logs `🔴 CAMERA AUDIO DEAD` with the measured percentage and says a **reboot is required**, because ONVIF `Reboot` is not implemented on OpenIPC and only a power cycle clears `ai0_P0_MAIN`. Measured healthy ratio on this room is 100-170 % (the stream runs slightly fast), broken was 3-6 %.

- **The openWakeWord path is still what corridor/kitchen use** and is untouched: `wake_vosk_model=""` (the default) leaves the acoustic head exactly as before. Do not "fix" those rooms by copying the livingroom setup blindly — their heads were calibrated separately and are the only thing working there.

- **To diagnose a mute room, run these three measurements before touching the wake word**: (a) `docker logs | grep 'vosk diag'` — is `chunks` advancing at ~6/s? (b) standalone `ffmpeg -i rtsp://root:PW@<cam>/stream=0 -vn -t 30 -f null -` and read `speed=`; (c) on the camera, `nproc`, `cat /proc/loadavg`, and `grep aio_dma /proc/interrupts` twice a few seconds apart. (c) is the decisive one: **41/s is this camera's healthy audio-DMA rate**, and a load average of 11 on a single core with only 1 runnable task means the media threads are wedged, not busy.

- **openwakeword.com/library has no usable Russian «компьютер»** (scanned ids 1..260 on 04.10.2026): the four candidates are `kompuklerr` (recall 53.5 %, **FA 15.75/hour**), `khom_piew_ter` (48.3 %, 9.0/hour), `compewter` (5.6 %, 9.0/hour) and `hey computer` (11.7 %, 4.5/hour) — the best model in the whole library is `BIMO` at recall 65.3 % with 0.0 FA/hour, and it is a different word. **Downloads need an account** (`/api/models/<id>/download` → 401 `Sign in at openwakeword.com to download models`), so nothing can be fetched anonymously. Measured locally on our own audio, every ready head is a dead end in both directions: `alexa`/`hey_jarvis`/`hey_mycroft`/`computer_20260706` all score **0/36** on the confirmed «компьютер» bursts *and* 0.0 false/hour on the TV — silent but deaf. So there is no ready model to swap in, which is why the shipped path decodes instead.
- **Wake gate cascade** (in order): own-playback guard → appliance hold (60 s bg median >800 ⇒ top score must be ≥0.60; a held [0.55, 0.60) opens an STT-confirm window) → quiet-source hold (own level <3000 ⇒ ≥0.58 below level 2000, ≥0.55 above; a held [tier−0.05, tier) opens an STT-confirm window) → distant-source veto (level <3000 while another room hears ≥1.4× louder ⇒ stand down 5 s) → clipping-bang gate (peak >24k & score <0.85) → crest-factor gate (dense impact: rms·2 > peak) → ambiguous-zone STT confirm (top <0.55 or a recent unanswered auto-greet, level <3000 ⇒ no pip, route to Whisper) → debounce → arbiter.
- **Cross-camera arbiter** (`_arbiter_*` in camera_client.py): first detector becomes interaction owner; others stand down. Proximity steal: a room with ≥5× the owner's loudness level takes over a not-yet-dispatched wake. `_ARB_STATE["cmd_sent"]` blocks steal once the owner dispatched to Nanobot.
- **TTS pipeline**: Nanobot text → sentence splitter → prefetch player (sentence N+1 synthesises while N plays; `_tts_fetch` + `_speak_pcm`) → pydub decode + resample → Opus to ESP32 / 48 kHz PCM POSTed to the camera's `/play_audio`. Every played clip is registered in the echo-reference ring (resampled to the 16 kHz mic feed — the rate must track `TTS_PLAY_RATE` or `_is_echo` stops matching); confirmed mic echoes extend `_wake_suppress_until`.
- **DO NOT PLAY TEST TONES THROUGH THE CAMERA AT NIGHT — there is a hamster cage
  under it.** On the night of 04→05.10.2026 I rang 1 kHz tones through `/play_audio`
  at three levels and then four declared sample rates to diagnose crackling
  playback. That is what I did without asking, and it was the wrong call: the
  data I already had (`peak/median` says the room is quiet, THD is unmeasurable
  through a saturated mic) did not justify more sound. Open with
  `/api/camera/tts` at a LOW volume, once, during the day, and let the user judge
  the result by ear — the microphone cannot be the instrument here (see below).
- **The microphone CANNOT measure playback quality: it is saturated.** Measured
  05.10.2026 00:01 by playing pure tones and recording the room: input peaks
  20000, 10000 and 4000 were all captured as **rms ~26 300, peak 32768, ~40%
  of samples pinned at the rail**, with 2nd–5th harmonics within −2 dB of the
  fundamental. The captured level does not follow the input at all. A recording
  taken through this microphone therefore tells you nothing about what the
  speaker produced — an earlier conclusion of mine ("found clipping, 1197
  samples on the rail, that's the crackle") was exactly this mistake and was
  retracted. Judge playback **by ear, from the user**, or through a non-acoustic
  path. The ambient level is unaffected (`feed rms=424 peak=2685`),
  `speed=1.04x` direct from the camera, and vosk reads the room fine — so this
  is a capture-path limit, not a dead mic.
- **Crackling playback (`хрипит`), reported by the user 05.10.2026: cause NOT
  found, and everything on the gateway side is clean.** Three hypotheses were
  measured and all three are dead ends — do not re-run them:
  1. *Clipping in our PCM.* Real `_tts_fetch` output for a spoken sentence:
     peak **17091** (52% of full scale), **0** samples at the rail. Clean.
  2. *Broken resampling.* The TTS mp3 decodes at 24 kHz; `set_frame_rate(48000)`
     gives 141696 → **283391** samples with duration 5.90 s → 5.90 s, i.e. it
     really resamples (the pydub "changes the header only" trap does not apply
     here).
  3. *Sample-rate mismatch.* The same 1 kHz waveform sent with
     `;rate=48000`, `24000`, `16000` and `8000` was heard by the microphone as
     **exactly 1000 Hz in all four cases** — the camera plays a fixed rate, so
     the declared rate does not need to match and 48 kHz is not the problem.
  What is left is the camera's own output stage (speaker/amp overdrive), which
  is exactly what the saturated microphone cannot distinguish — so the cheap
  next step is one quiet daytime phrase at reduced level and the user's ear.

  **The lever now exists**: `CAMERA_TTS_TARGET_PEAK[_<NAME>]` (0 => 20000, i.e.
  61 % of full scale) normalises the reply to a chosen peak per room, and
  `_tts_fetch` logs `peak/target/gain/applied/out_peak` at DEBUG. It applies
  the gain in BOTH directions — the original `gain > 1.2` could only ever
  boost, so a target below the TTS's natural peak produced gain ~0.70, took
  the dead band by accident and did nothing at all, which from the field
  reads as "lowering the level did not fix it". Start at 12000 (37 %).

  Meanwhile the log-only suspect is the SENTENCE PREFETCH: the reply is
  synthesised one sentence at a time and each sentence is a separate
  `/play_audio` POST, so a gap between sentences would stutter rather than
  crackle — worth ruling out before touching gain.
- **The wake path is proven end-to-end, by loopback rather than by asking the
  user to talk.** 05.10.2026 00:57: `POST /api/camera/tts` with the phrase
  «Проверка микрофона. Компьютер, включи свет в гостиной.» → the camera's own
  microphone recorded the playback at **52× the room floor** (rms 30 436 vs a
  median of 468, i.e. unmistakable speech) → vosk decoded it verbatim and
  flagged it: `[WAKE] компьютер включи свет в гостиной`. So speaker → air →
  mic → decoder is intact and the 20:18 outage has not returned.
- **A silent capture window is not a failed wake word — check the levels before
  blaming the detector.** Across five recordings totalling ~7 minutes (some taken
  while the user believed they were speaking) `max/median` never exceeded
  **3.6×** (speech is 5–20×), and vosk produced **0** segments, which is the
  correct answer to a room containing no speech. Meanwhile the real
  `feed rms`/`peak` logs showed one genuine burst at 21:44:04 (rms 3787, peak
  12178) — and `feed rms` is only printed **once every ~3 s**, so a 1-second
  command can fall entirely between two lines and leave no trace at all. That
  sampling is too coarse to prove a command was captured; the live test still
  has to be a real utterance followed by a `grep UTTERANCE END`.
- **VAD**: Silero ONNX server-side (`silero_vad.onnx`), 10 silence frames triggers processing, 7 s max-duration cap; VAD state is reset at wake fire so the post-wake command starts clean.
- **Binary frame versions**: v1 = raw Opus, v2 = 16-byte header, v3 = 4-byte header
- **MCP**: Gateway requests tool list (`tools/list` id=999) on connect; forwards to Nanobot as `tools_update`
- **Dialogue mode**: if the reply ends with `?` (ASCII/fullwidth), contains the «повторите пожалуйста» apology, or matches a Russian interrogative/imperative (`HAS_QUESTION_WORDS_RE`, anywhere in the text) → follow-up window opens when playback drains; otherwise returns to standby
- **Watchdog**: `WATCHDOG_TIMEOUT` (default 90 s) → fallback TTS "Простите, я задумалась. Повторите пожалуйста." — never outlive it (the L2 `EXPERT_TIMEOUT` is 25 s for exactly this reason)
- **Emotions**: extracted from Nanobot text via `[emotion_name]` regex
- **L2 HA action fallbacks** (`tools.ha_action`): a blind intent whose `area`/`domain` filter was rejected (`MatchFailedReason.AREA`/`.ASSISTANT`) re-resolves the target from raw `/api/states` + one `area_name` template render — the rule L1 already has — and retries with `name`+`domain` pinned, dropping the slot that failed. `find_action_targets()`/`resolve_onoff_targets()`/`match_states()` also honour ORDINALS — `resolve_action` appends the digit to the hint («свет 1», a digit is not an HA matcher slot) so «первый коридор» reaches only `corridor1_…_relay`; `ordinal_digit()` + `ORDINAL_STEMS`/`ORDINAL_ENDINGS` live in both services and `test_hint_sync.py` holds them together, and a numbered instance that does not exist yields `[]` (escalate / original error), never a both-relays toggle. The lamps here are `switch.*_relay`, so `domain: ["light"]` can never match them; guards: a device word required, several devices need a named room, ≤5 targets, state must be `on`/`off`, facets (status/network LED) dropped, exact room name beats a containing one, zero successes keep the ORIGINAL error. The vacuum pair (`CleanArea`) has the same shape.
- **Cascade L2 context** (jev-router → smolagents-worker): an escalation carries two extras — `resolver.unresolved_hint()` (difflib over the THING stems/names, threshold 0.70, guarded to fire only on a command verb whose exact word made the resolver bail) and the last 4 finished turns of that satellite from `history.py` (TTL 10 min, read before the turn is pushed). A fired honesty veto triggers exactly one retry fed with `honesty.failure_note()` (raw tool errors) plus the same history; a second veto speaks the truth instead of looping.
- **THE 7 s DEAD WAIT EXISTS BECAUSE THE VAD NEVER REPORTS SILENCE — and the cap
  cannot simply be lowered.** Measured 04.10.2026: wake 19:20:10.1, Whisper
  19:20:17.3 — 7.2 s for «выключи свет», of which the command was the first 1.5.
  An utterance ends on 10 VAD-silent frames (1.6 s) or on the 7 s cap, and in
  this room Silero never said `speech=False` (0 in 15 live minutes; every
  `VAD rms=... speech=True` line sits above `consec=250`), so **every** command
  ran to the cap. Lowering the cap is NOT the fix — it is load-bearing:
  «я просил включить следующую серию черного зеркала» transcribes correctly at
  7.04 s and is chopped at 3.5 s.
  * `_PauseEndpoint` (`camera_client.py`) ends an utterance on a dip in the
    **LEVEL** envelope, which happens between words whatever the VAD thinks.
    Reference = 80th percentile of a trailing 12-frame window: not the max (one
    door slam must not redefine it) and not the min (sustained noise would pull
    it down until everything reads as a pause). No reference => no pause => the
    cap still fires. **Slower is recoverable; a chopped command is a wrong
    command.**
  * It requires `min_speech_frames` of real speech FIRST, so the gap right after
    «компьютер» does not dispatch a bare wake word — the failure behind «Да?». **Two
    different levels, on purpose:** the PAUSE test uses the trailing percentile
    (a responsive question — has the speaker stopped?), while the SPEECH floor is
    **utterance-scoped**: the median of the first 4 frames, then frozen. A windowed
    speech floor collapses to the room floor after ~2 s of silence, the pause test
    stops matching, and every remaining quiet frame is then counted as speech —
    measured on the buggy version: a bare «компьютер» plus a 30-frame pause drove
    the counter to 50 on silence alone, so the guard guarded nothing. Frozen because
    a single door slam must not redefine «speech» for the rest of the sentence; a
    user who starts very quietly is then under-counted, which degrades to the cap.
  * **Off by default**, per room: `CAMERA_PAUSE_ENDPOINT[_<NAME>]=true`. It
    changes WHEN a command is dispatched in a live audio path, so it is enabled
    only after its own log lines have been read on real audio from that room.
  * Tuning is ENV (`CAMERA_PAUSE_RATIO` / `_RUN_FRAMES` / `_MIN_SPEECH_FRAMES`,
    per room), not constants: the logged numbers exist to be tuned from, and a
    rebuild per iteration is not tuning. `0` means "unset" — the detector
    substitutes its default, because a half-filled override that zeroed
    `ratio` would make every pause test `rms < ref * 0` fail and fall back to the
    cap, which is indistinguishable from "the endpoint does not work".
  * Every end now logs `via <reason>` plus the rms/ref numbers, and tallies per
    boot, so «is the endpoint working or is everything still hitting the cap?»
    is one `grep UTTERANCE END` away instead of an argument.
  * **The endpoint is gated on an ACTIVE WAKE WINDOW, not on "an utterance is
    open", and so is the tally.** Measured 04.10.2026 21:01-21:05 on the living
    room: **70 `UTTERANCE END ... via cap 7s` in four minutes, not one of them
    a command.** The VAD there reports `speech=True` continuously, so the
    television opens an utterance every ~7.2 s, fills 7 s of buffer and hits
    the cap, and `_process_utterance` discards it one line later for want of a
    wake window. Endpoints on that audio would have made `pause` in the tally
    mean mostly "the television" — the one number that decides whether the
    endpoint works would have answered the wrong question — and the ambient
    ends put one INFO line every 7 s into the log (~12000 a day). Ambient ends
    still reach `_process_utterance` (dispatch unchanged) and are logged at
    DEBUG with their reason, so nothing is hidden; the line now says
    `awake ends:` and not `since boot:`, because the difference is not
    something anyone should be able to miss.
  * **ENABLED in the living room 04.10.2026 21:37** via
    `CAMERA_PAUSE_ENDPOINT_LIVINGROOM=true` in `docker-compose.yml`, after the
    baseline above was read off a real log. Ship the same way to another room:
    read that room's own `UTTERANCE END` lines first.
  * All three terminators (silence, pause, cap) go through ONE method,
    `_end_utterance()`. They used to inline three copies of the same reset
    block and had already drifted: the cap copy reset `_vad_start_time` and
    the silence copy did not. That variable turned out to be **written three
    times and read nowhere** — dead state, so it was removed instead of
    propagated. A write-only field written from several places is a trap for
    whoever reads it next.
- **A WATCHDOG THAT ONLY RUNS ON SUCCESS CANNOT SEE TOTAL FAILURE.** The
  living room went mute on 04.10.2026 20:18 and the log said only
  `RTSP audio reconnecting in 3s...` — 25+ times, every 3.2 s, and not one
  line said why. Four separate holes, each visible only in that log:
  * **`stderr=PIPE` and nobody ever read it.** Every real cause
    (`Connection refused`, `404 Not Found`, `Invalid data found`) was
    discarded while the loop said "reconnecting". `_read_ffmpeg_stderr()`
    now reads it; the failure line is useless without it.
  * **`_stall_count` needs audio BEFORE it can stall** (20 s of silence
    AFTER bytes flowed). With go2rtc having no producer, ffmpeg delivered
    zero bytes, so `_stall_count` stayed 0 forever and
    `_heal_go2rtc_stream()` was **never called**. `_audio_cycle_done()` counts
    zero-byte connections separately — the case that is a failed connect,
    not a stream that ended.
  * **The rate watchdog lived inside the success path**, so "20 s of wall
    clock, zero bytes" was invisible to it — the exact condition it exists
    to catch. It is now fed on the failure path too (`_account_audio_rate(0)`).
  * **No backoff and no escalation**: 3 s forever. It now declares the room
    dead after the heal budget, names the likely cause, and drops to 30 s.
  **Diagnostic rule worth keeping:** `feed rms=` absent from a log while
  reconnects repeat means NOTHING EVER ARRIVED, and the heal paths are all
  keyed on bytes having flowed. Absence of the watchdog lines was the tell.
* **`aioice` IS NOT A CHILD OF `aiortc`, so silencing one does not silence the
  other — and an ICE startup burst is 80% of the whole log.** Measured
  04.10.2026 21:31: one gateway restart produced **1059** lines of
  `aioice.ice - Check CandidatePair(...) FAILED` inside one minute, against
  1221 lines total for the container; `logging.getLogger("aiortc")` was
  already at WARNING and none of it helped, because `aioice` is a separate
  top-level logger. It is a **burst, not a steady cost** — 0 aioice lines in
  the 14 minutes after ICE connected, and the gateway measured **29% of one
  core** (AGENTS.md's budget is 31%), so do not go looking for a CPU leak
  here. It still fires on every (re)connect, so the fix is
  `logging.getLogger("aioice").setLevel(logging.WARNING)`: 1059 -> 0, total log
  1221 -> 110. Read it as noise that buried the lines that matter.
* **The WebRTC session decodes camera audio and throws it away.** `_connect()`
  opens a `recvonly` transceiver and `_recv_audio` runs Opus decode in Python on
  every frame, with a comment saying the mic actually arrives via the RTSP
  loop. The `sendonly` `AIVoiceOutputTrack` is the ONVIF backchannel, which is
  the pitch-broken path `/play_audio` replaced. So the session is currently
  there for a keepalive that nothing reads. It is NOT worth ripping out for CPU
  (29% of one core total, measured) — but do not assume the recv track feeds
  the VAD, and do not debug wake behaviour through it.
- **go2rtc IS REACHABLE WITHOUT AUTH FROM MOST HOSTS** (`/api/streams`,
  `/api/frame.jpeg`), as is `jev-router` on 8091 — so when shell access is
  unavailable, `fetch` can still drive the stack. That is how this was
  diagnosed. What that showed: `livingroom` had a producer entry carrying
  only `url` — no `type`/`sdp`/`medias`/`recv` — and **zero consumers**, while
  all five other streams had a live producer plus one ffmpeg consumer. That
  shape means configured-but-not-connected.
- **A wake word must get attention the moment it is HEARD.** Three separate
  mechanisms were swallowing it, all found by reading three field dialogues
  (04.10.2026, 19:20 / 19:22 / 19:37). The log shape is unambiguous:
  `VOSK WAKE` … `router: speaking: '…'` … `VOSK wake heard but not fired
  (suppressed=N)` twice … `VOSK WAKE` 28 s later.
  * **The echo tail after playback was 15 s (16 s for the pip).** A reply ending
    at T blocked every «компьютер» until T+18 — the +8 s and +11 s attempts in the
    log. It was standing in for a mechanism that already works: `_feed_audio`
    drops our own audio by CROSS-CORRELATION (`_is_echo`, which extends
    suppression by up to 20 s when and only when a correlation matches), and it
    does so BEFORE the chunk reaches the decoder. Now `_ECHO_TAIL_S = 3.0`, used
    by both the pip and the reply.
  * **`_wake_detected` is NOT a firing guard.** It stays True for the whole
    `_wake_timeout` (60 s), so repeating the word during the dialogue window did
    nothing — while the decoder had HEARD it. What remains is
    `_WAKE_REARM_DEBOUNCE_S = 1.5` for the same breath saying it twice, because a
    second `_fire_wake()` clears `_vad_speech_buf`. The openWakeWord branch
    legitimately keeps its own `_wake_detected` guard (it has a 2-of-3 debounce);
    it is the decoder's guard that was wrong, and the two must not be "aligned".
  * The gate is a method, `_wake_gate_open(now)`, with `_wake_gate_reason(now)`
    for the log line. **Test it by calling it.** Three earlier versions of these
    tests grepped the module source for strings — the wrong instrument, because
    each rule is written IN ITS COMMENT as the mistake that was made, so the
    search finds the very counter-example it is trying to prove absent.
- **A media TITLE is not a missing DEVICE, and refusing blocks escalation.**
  `RE_MEDIA_NOUN` matches «серию»/«эпизод»/«передачу», but `resolve_action` only
  takes the media branch when the thing resolves to a `media_player` — which a
  title never does, so «включи следующую серию черного зеркала» fell through to
  `HassTurnOn` with hint «серию» and the user was told «Не нашла такого
  устройства» (field case 04.10.2026 19:22). L2 has `media_search`/`media_play`
  for exactly this, and the deterministic refusal is what kept it from getting
  there. The refusal is now noun-aware and returns an error L2 can act on.

- **THE "DYING CAMERA" WAS US ALL ALONG — the gateway's own WebRTC churn
  wedged it. Measured 05.10.2026; the hardware-fault story below is WITHDRAWN.**
  It looked exactly like failing silicon: `majestic` restart brought audio back
  ("none" -> 100 % of real time) and then it wedged again within minutes, twice
  over a real reboot. What actually ended it:

  | | before | after `docker restart voice_gateway` |
  |---|---|---|
  | connections on 554 | 3, one with **Send-Q 193712** | **0** |
  | `GET /` | **14.34 s** | **0.021 s** |
  | audio straight off the camera | **none** | **100 % of real time** |

  The `Send-Q 193712` was the whole answer: the camera had written 193 kB into a
  socket that **our side never read**. Its single-threaded majestic then blocked
  on the full send buffer, so the web UI answered in 14 s and RTSP produced
  nothing — a fast-looking box that is completely deaf. `docker restart
  voice_gateway` dropped the reader and it recovered instantly, with no power
  cycle and no service restart on the camera at all.

  **The mechanism, and it is ours:** `_connect()` offers `webrtc/offer` on
  `src=livingroom`, which makes **go2rtc rebuild that stream's producer**. When
  go2rtc accepted the websocket but never answered (7 offers in 12 min, measured)
  the offer was **abandoned**, leaving a producer with nobody consuming it — go2rtc
  stopped reading the camera, and the camera's send queue filled. That is the
  24 s retry loop, and `_run()` re-entered `_connect()` with **no pause at all**,
  so it was permanent churn. Two fixes, both measured:
  1. **Exponential backoff** in `_run()` (15 s -> 300 s cap, reset on an answered
     session). The session carries no audio we use, so retrying slowly costs
     nothing; it did connect within 90 s of a restart.
  2. **`CAMERA_WEBRTC[_<NAME>]=false`** — no offer is made at all. Set on the
     living room, which has `/play_audio`, so it needs the session for nothing.
     Default stays `true` because for a room **without** `/play_audio` the
     sendonly track is the only playback path left.

  **The WebRTC session delivers NOTHING this gateway uses** — `_recv_audio` runs
  Opus decode and discards it (the VAD is fed by the RTSP loop) and replies go
  out over `/play_audio`. It was described earlier in this file as "there for a
  keepalive that nothing reads"; that was right and the cost was much higher than
  CPU.

  **A guard this bug would have exposed silently:** `_speak_pcm` wrote
  `_speaking_until` only `if self._out_track`. That timestamp is the hard mic mute
  in `_feed_audio`, so with the session off the room would have heard **its own
  reply** come back through the decoder. It is now unconditional, and the pre-POST
  block falls back to the clip's own length when there is no track.

- **THE FIRST COMMAND AFTER A RESTART COST 22 s — a warm-up that gave up in
  microseconds and then blocked the user's turn. Measured 05.10.2026.**
  `включи свет` went Whisper-done at 14:52:57.7 to `speaking:` at 14:53:20.8:
  `elapsed=22.35s`. The log said why: `classifier warmed: 60 utterances` was
  printed at 14:53:20.6 **from inside that request**.

  Two independent faults, both of which had to be fixed:

  1. **`warmup()` retried with no delay.** `for attempt in (1, 2)` with no
     `await` between them, so both attempts hit the same unopened socket.
     Ollama and Qdrant are containers on the same host started **in parallel**
     with the router, so at boot "not listening yet" is the NORMAL case, not an
     exception — and indeed the router restarted twice (12:09 and 12:46 UTC),
     logging `Cannot connect to host 192.168.22.102:11434` and
     `qdrant ... 6333` **both times**. `ollama`/`qdrant` then show `Up 2 hours`,
     i.e. they came up after the router gave up. It stayed cold for two hours.
     Now: 6 attempts with a linear backoff (~2 min total, `_WARMUP_*`), one
     warm-up at a time (`_warming`), and `lifespan` starts it in the BACKGROUND
     so the port opens immediately instead of blocking on a dependency that is
     not there yet. Qdrant got the same patience via `_retry_memory()`.
  2. **`classify()` awaited the warm-up.** `if not self.warmed: await
     self.warmup()` — so whoever asked first paid for it. It now calls
     `start_warmup()` (fire-and-forget) and answers from the deterministic
     `_regex_action` path, which is what that path was written for.

  Measured after: the port answers `/health` in **1.6 ms** at startup, and the
  **first command on a deliberately cold classifier takes 0.12 s wall / 0.07 s
  router**, logged as `classifier cold, regex action path: 'включи свет'`. Warm
  commands: 0.18-0.34 s.

- **THE DIALOGUE WINDOW OPENED WHILE THE ROOM WAS STILL TALKING — the deafness
  right after an answer.** `_wait_playback_drain()` returned immediately whenever
  there was no WebRTC output track, and there almost never is one (the session
  mostly fails to get an answer from go2rtc, so `_out_track` stayed `None`). So
  `Follow-up open` was logged **0.07 s after the TTS fetch** while `_speaking_until`
  — the hard mic mute — ran **4.3 s longer**. For those 4.3 s the window was
  "open" and the mic was muted, so anything the user said was discarded before
  it reached Whisper. Measured 05.10.2026; this is the "camera stays deaf after
  it answers" report, and it made the 10 s statement window feel like it never
  arrived.

  Without a track the playback end is still known exactly: `_tts_play_end` is
  written from the clip's own length **after** the POST returns, which is when
  the camera starts playing. The drain now waits for it, and only when it is
  still in the future — a stale value from the previous turn must not park the
  player for the length of a whole reply. Deliberately NOT bounded by
  `_speaking_until`, which would add the 3 s echo tail on top: the window is for
  the moment the answer ends, not 3 s into the quiet after it.

- **A FOLLOW-UP WINDOW IS NOT THE USER — the room answered its own television six
  times in two minutes. Measured 05.10.2026 19:24-19:26, and it was in the field
  log the whole time.**
  Reported as «после неудавшегося диалога херню несет». What the log showed, in
  one session, same room:

  | peak | transcript | who |
  |---|---|---|
  | **32767** | «выключи свет» | user |
  | **32767** | «включи свет в гостиной» | user |
  | 16074 | «музыкант» | television |
  | 12348 | «распаковываешь» | television |
  | 10792 | «Другая, пожалуйста, у тебя был шанс, но ты обожался.» | television |
  | 6021 / 5626 | «Добро пожаловать!» | television |
  | 4625 | «Мама, ты что? Да, ха-ха-ха.» | television |
  | 4130 | «Это вот сейчас и не родит, не хранит стресс.» | television |

  Every one of those cost 4-13 s of L2 and then **spoke**, which is what "talks
  nonsense on its own" is. The separation is not subtle: the user's voice pins the
  rail at 32767, the room never exceeds 16074.

  **The note in this file that said the window was safe because "TV speech has to
  contain «компьютер» to open anything" was WRONG.** That is true of a window
  opened by the wake word, and false of a window opened by our own REPLY: inside
  it, every utterance was dispatched with no keyword at all. It was first "fixed"
  with `CAMERA_FOLLOWUP_MIN_PEAK[_<NAME>]` (default 24000), on the reasoning that
  the voice — not a keyword — identifies the user, because you are close and the
  television is across the room. **That reasoning was measured on 06.10.2026 and it
  is wrong: the user's own voice peaks at 4725 and 9124 while the television and the
  appliances peak at 13197-32522.** The threshold refused the user twice and admitted
  noise twice in twenty minutes, and it cannot be re-tuned, because the two
  distributions overlap — see "THE PEAK CANNOT TELL THE USER FROM THE TELEVISION"
  below. What survives from the old attempt is the mechanism: `self._followup_only`
  marks a window opened by our reply and is cleared by `_fire_wake` and
  `_back_to_wake`, and below a gate the room logs `follow-up ignored, peak=... ` and
  goes back to waiting for the wake word.

- **THE ECHO SUPPRESSION WAS A SLIDING DEADLINE, SO EVERY REPLY DEAFENED THE ROOM FOR 20 s MORE. Measured 06.10.2026 12:40, reported as "both cameras take a long time to react to «компьютер» after a confirmation".** The log said it three times in a row:

  ```
  12:40:24.025 VOSK wake heard but not fired: echo_tail=+19.6s (suppressed=1)
  12:40:26.274 VOSK wake heard but not fired: echo_tail=+17.4s (suppressed=2)
  12:40:28.911 VOSK wake heard but not fired: echo_tail=+14.7s (suppressed=3)
  ```

  The user said «компьютер» three times and was **told the wake word was heard** while it was blocked. The arithmetic was `min(now + 20.0, _tts_play_end + 45.0)`, applied on **every** chunk that correlated with our own playback — so while a reply was sounding, each echo chunk pushed the block 20 s past the *current* moment, and the last one left 20 s of deafness after the reply ended.
  * Now anchored to when playback ends: `max(now + _ECHO_TAIL_S, _tts_play_end + _ECHO_TAIL_S)`. Both call sites — the 48 kHz path and the generic resampler.
  * **The `+20 s` existed for the ONVIF backchannel, which returns our own speech 3-40 s late. That path is gone**: both rooms play over `/play_audio` (`CAMERA_WEBRTC` off) and `_tts_play_end` is written after the POST returns, which is when the camera actually starts playing. So `play_end + tail` covers the tail and **does not move as more echo arrives**.
  * `now + _ECHO_TAIL_S` is a deliberate FLOOR, not an oversight: a stale `_tts_play_end` from a previous turn must not park the block for the length of a whole reply. So the deadline still slides with each echo chunk — bounded by the tail plus however long the echo itself lasted, which is under 6 s over the measured 1.9 s tail. The test asserts that bound and records what the old arithmetic gave, rather than asserting immobility it would be wrong to demand.
  * **A wake must never be reported as heard while it is blocked.** `VOSK wake heard but not fired` is the instrument that showed this, and it is the reason to keep the reason string on that line.
- **THE ROOMS ARE NOT DIFFERENT IN HOW WELL THEY HEAR THE WAKE WORD, AND A ONE-SHOT COMPARISON SAYS THEY ARE. Retracted 06.10.2026; this is how the wrong answer was produced.** The question was real ("does the kitchen hear «компьютер» worse"), the instrument was right, and the conclusion drawn from it was wrong. `scripts/measure_wake_sensitivity.py` plays one identical phrase through each camera's own `/play_audio` at the normal reply level, captures each microphone directly from the camera, computes per-band SNR and runs the production feed path (`begin()` once, `feed()` chunk by chunk).

  One run said the kitchen was **11 dB worse in the speech band** (+4.4 vs +15.4 dB) and never fired, with the decoder resolving «компьютер» as «как театр». **That was a single sample.** Repeating the measurement:

  | repeats | livingroom | kitchen |
  |---|---|---|
  | 4 | **4/4**, mean +13.8 dB | **1/4**, mean +9.9 dB |
  | 5 | **2/5**, mean +13.1 dB | **3/5**, mean +12.8 dB |

  Two runs of the same experiment, minutes apart, disagree about which room is better and by how much. Speech-band SNR converges to **+13.1 vs +12.8 dB — the rooms are equal** — and detection is a coin flip in both. **A single run of this measurement carries no information about the room; `--repeat` is not optional.** (It is also not a proxy for a person across the room: a speaker one metre from its own microphone is far louder than any human voice, and at that level both rooms decode about half the time.)
  * What the search did establish, and it cost three experiments to separate from the noise:
    * **A high-pass is not the answer.** A 4th-order Butterworth at 0/80/120/160/200/300 Hz changes the kitchen from not firing to not firing, at every cutoff.
    * **Gain is not the answer in either direction.** +3/+6/+9/+12 dB does not make the kitchen fire and −3…−18 dB does not either — and at **+6 dB the living room stops firing too**. Neither direction touches detection because **level does not change SNR**, which is the whole lesson and which one number per run will never show.
    * So the instrument is **per-band SNR over repeats**, not any single peak.
  * The kitchen's ambient is genuinely higher — **rms 525-590 against the living room's 388-414** — and that is the only stable difference between the rooms. If the kitchen still feels worse in daily use, the cause is not here: it is the geometry of a human voice at the distance the user actually stands, which this test cannot reproduce, and the log lines to read are `VOSK WAKE ... peak=` on a fire and `VOSK wake heard but not fired: <reason>` on a refusal.
  * **`POST /api/camera/tts` returns 401 without the admin credentials**, which cost one wasted round of the first attempt.

- **A WAKE THE DECODER MISSED LEAVES NO TRACE, AND THAT IS THE MOST EXPENSIVE SILENCE IN THE FILE.** Measured 06.10.2026: a second speaker said «компьютер» in the living room and the room ignored her for twenty minutes. What the log actually said:

  ```
  chunks=112058 suppressed=0 peak_max=32767 triggers=7 decodes=7 hyp=''  drop[muted=395 track=0 echo=69]
  ```

  `suppressed=0` — **the gate refused nothing**; the mic was muted for 395 chunks and 69 were claimed as echo, which is normal over 45 minutes; the wake decoder simply never fired, and `triggers` sat at 7 across four consecutive five-minute diagnostics. Two explanations remain and **both are silent**:
  * the model did not decode the word at all, or
  * it decoded a spelling outside `WAKE_TOKENS` — which allows exactly five (`компьютер`, `комп`, `компютер`, `компъютер`, `компьытер`) because five were OBSERVED — and `transcript_has_wake` dropped it on the floor without a line.
  **A different voice lands outside a list built from one voice's observations, by construction.** The first version of this finding blamed the decoder; the counters do not support that, they only narrow it to these two.
  * **RETRACTED: I read `VAD SPEECH START` as evidence of speech, and it is not evidence of anything.** I wrote that the room "heard speech" on the strength of 412 such events. It is a chatter, not a person: the lines arrive **every 6.55 s to the millisecond** for hours with `rms` pinned at 0.0104-0.0134, which is the room's own ambient (`feed rms=350-460`, i.e. 0.0107 normalised). This is the long-documented behaviour that the VAD in this room **never reports silence**, so every few seconds a level wobble reopens an utterance that is then discarded for want of a wake window. **A counter that fires on noise cannot be used as proof that a human spoke**, and using it that way made a 20-minute gap look like a detection failure when the log never supported that. The only level evidence of a person in that window is chunk peaks (6128), which a television also produces.
  * **What the follow-up run actually showed: both attempts worked.** 06.10.2026 20:49 and 20:51 MSK, `VOSK WAKE trig='компьютер' conf='компьютер'` twice, pip both times, and the commands went through — «Включи свет в гостиной» at peak 28403, then «Выключи свет.» at 13139. **Near misses: 0.** So the decoder heard the second speaker cleanly on this occasion, and the earlier failure is still undiagnosed — but it is no longer an open question whether the word reaches the decoder at all, and the instrument will name the cause if it happens again.
  * `wake_near_miss()` now reports a decoded word that ALMOST matched: a 3-character shared prefix plus a length delta of at most 3, which is cheap enough to run on every hypothesis. It is a **diagnostic and never a match** — `transcript_has_wake` stays exact, because widening it to a prefix admits «компот» and «компания». Counted in `near=` on `vosk diag` and logged outright, since the two explanations need different fixes: heard-and-discarded versus not-heard.
  * **A prefix is a truncation, not a misspelling.** The first version reported every allowed spelling as its own near miss, because «комп» is a prefix of every other entry. Both an exact hit and a prefix relation are excluded.
  * **A stub that mirrors a class has to track the class.** `_StubVosk` in `tests/test_wake_gates.py` raised `AttributeError` on the new counter — inside the live audio path, which is the worst place to find out a test-only shortcut exists. The real class therefore also carries **class-level defaults**, so an instance built with `__new__` (which `tests/test_vosk_wake.py` does to avoid an 88 MB model) cannot miss the attribute.
  * **Next occurrence of a silent wake is now diagnosable instead of arguable.** Read `near=` on `vosk diag` and `vosk: wake NEAR MISS` in the log. `near>0` means the allowlist threw the word away and the fix is one more measured spelling; `near=0` means the model did not hear it and no threshold will help.

- **WHISPER MANGLES «ВЫКЛЮЧИ», AND AN LLM HANDED THE GARBLED TEXT TURNS THE LAMP ON. Measured 06.10.2026, and the damage was a REVERSED ACTION on a real device.** «выключи свет» reached the router as:

  | heard | router | system did |
  |---|---|---|
  | «куча свет» | L2, conf **0.75** | **«Включила свет»** |
  | «куча свет в гостиной» | L2, conf **0.72** | **«Включила свет в гостиной»** |
  | «кучи свет» | L2, conf 0.83 | «Включила свет в гостиной» |
  | «ключи свет» | L2, conf 0.74 | «Не удалось найти свет в комнате ключи…» |
  | «выкл юч свет» | L2, conf 0.86, `resolver_ambiguous` | «В какой комнате выключить свет?» |

  Asked to turn a lamp OFF, the room turned it ON, and the room name in the answer was invented — the router with the correct spelling answers «В гостиной уже выключено», so the entrance-hall report was the LLM guessing, not a resolver default. `RE_OFF` matches none of those spellings, so the deterministic path is skipped entirely.
  * **Fix 1 — repair the transcript where it ENTERS, not inside the resolver.** `normalize_stt_verbs()` maps only the observed spellings (`куч[аеуыи]`, `ключи`, `выкл\s*юч`, bare `выкл`) and **only when a device noun is present**, which is what makes mapping two ordinary Russian words safe: without the gate «куча чая» switches a lamp off. No ON entry exists because nothing has been observed mangling it — the log holds «включи», «Включи», «включи свет» — and inventing one would flip a real lamp on no evidence.
  * **The placement was the whole bug, and the first attempt proved it.** Normalising in `resolve_action` fixed only «выкл юч свет», because that one happened to score **0.86**, just over the 0.85 gate. «куча свет» scored **0.75** and went to L2 **without the resolver ever being called**. Same defect, opposite outcome, decided by how close to the threshold the embedding happened to land. **A transcription repair is not a routing concern and must not sit behind a confidence score.** Verified live after the move: all five variants route `easy_action` at confidence 1.0, resolve to the living room, and no other relay moves.
  * **Fix 2 — the net for mangles nobody has observed yet.** `action_verb_missing(text)` fires when a device noun is present and no verb of any family matches; `_handle` then attaches a note to every `complex_logic` escalation: *do not guess the direction, do not call `ha_action`, ask.* It must be attached for **all** escalations and not only the resolver's, because today's case went out through the classifier's `low_confidence`, which never calls the resolver. It returns **False** for a measured mangle, because after normalisation the verb is present and the turn never escalates — measured mangles are fixed at source, not warned about.
  * **Two of my own errors on the way, both caught by their tests:** `[аеуы]` does not contain `и`, so «кучи» was not covered; and the first version of the test asserted `action_verb_missing("куча свет") is True`, which was my wrong expectation of the semantics rather than a code fault.

- **THE PEAK CANNOT TELL THE USER FROM THE TELEVISION IN THIS ROOM, SO AFTER A REPLY THE WAKE WORD IS REQUIRED AGAIN. Measured, and decided by the user 06.10.2026.** Twenty minutes, both directions, the same window:

  | utterance | peak | who | `followup_min_peak=24000` verdict |
  |---|---|---|---|
  | 'ключ' | **4725** | the user | rejected as "not the user" |
  | 'Выключить свет.' | **9124** | the user, a follow-up | rejected as "not the user" |
  | 'атака' | 32522 | noise | accepted |
  | 'пиздец' | 13197 | noise | accepted |
  | television, earlier session | 4130-16074 | television | accepted |

  **The noise is LOUDER than the user.** 24000 was calibrated from one sample of the user at 32767; today the same user measured 4725 and 9124 — a 3.6x spread — so the threshold sat inside the overlap with nothing to gain on either side. **Do not re-tune `CAMERA_FOLLOWUP_MIN_PEAK`: there is no value that separates these two distributions.** The user's voice is quieter than the television's because the television is a speaker two metres away and the user is a person in the room.
  * `CAMERA_FOLLOWUP[_<NAME>]` (**default OFF**) opens a free-form window after our own reply, so «а теперь выключи» needs no wake word. Off means the wake word is required again, which is the only discriminator measured to work — the decoder, 0 false accepts across 99 windows of held-out television.
  * **It is a boolean and not `dialogue_question_s=0` on purpose.** `0` means "unset, use the built-in default" throughout this file, so a window of 0 would have produced a **30 s window** instead of none — the opposite of the intent. `test_a_zero_window_is_not_the_same_as_no_window` pins that.
  * The dialogue behaviour stays reachable and tested behind the flag: `test_no_followup_window_opens_after_a_reply_by_default` pins the off state, `test_the_window_still_opens_when_it_is_asked_for` the on state.
- **THE WAKE WORD'S OWN AUDIO MUST NOT BE DISPATCHED AS A COMMAND, AND THE TRANSCRIPT IS NOT THE WAY TO ASK.** Measured 06.10.2026 12:39 on the living room: the wake fired at 12:39:56.340, the next utterance was **1.12 s** long and ended **1.85 s after the wake**, and Whisper wrote the wake word out as **«1000 свят»**. There is no «компьютер» in that string for the strip to remove, so the wake word was dispatched as a command and the router answered **«Включила»** to it.
  * `_is_wake_word_alone()` asks by TIME — onset within `_WAKE_ALONE_MAX_ONSET_S` (0.9 s), stopped within `_WAKE_ALONE_MAX_SINCE_S` (2.5 s), shorter than `_WAKE_ALONE_MAX_DUR_S` (1.5 s). Measured on both sides: the wake word alone began 0.73 s after the wake and ran 1.12 s; real commands run **1.68 s** («включи кофеварку») and **1.84 s** («Выключить свет»).
  * **The error can only be made in the safe direction.** A real short command mistaken for the wake word is re-opened, not dispatched — the next pause commits «компьютер <command>» together and the strip removes the wake word, so the command still runs, one pause later. The reverse error is what shipped.
  * **A chained comparison is not two bounds.** `(A <= onset <= B and ...)` bounds one value by both constants instead of bounding `onset` and `since_wake` separately. The first version of this read the wake word as a command, and its own test caught it.
- **A TEST HELPER THAT BUILDS A SESSION BY HAND CONVERTS A MISSING ATTRIBUTE INTO A PLAYBACK BUG.** `_player_session()` uses `CameraSession.__new__` and assigns fields by hand, so a new config attribute simply does not exist there. The player's broad `except Exception` turns that into `TTS sentence failed`, which reads as a TTS fault rather than a helper fault — the failure mode the helper's own comment warns about, walked into the moment the field was added. Every attribute the loop touches belongs in that helper, and the opt-in must be explicit (`s._followup_window = True`), because those tests are about the window's behaviour while the flag itself defaults off.

- **THE FOLLOW-UP DID NOT WORK BECAUSE THREE SEPARATE 3-SECOND BLIND WINDOWS ATE
  THE FIRST WORD. Measured by arithmetic, not by guessing.**
  The user said «компьютер, включи свет», then «а теперь выключи» with no wake
  word, and nothing happened. The log:
  ```
  19:24:22.178 tts pcm            <- reply starts
  19:24:24.340 Follow-up open     <- window opens (my earlier drain fix: correct)
  19:24:33.122 UTTERANCE END dur=5920ms via pause
  19:24:34.431 Whisper empty       <- the follow-up, gone
  ```
  The utterance ran 5.92 s and Whisper got `raw_rms=522` — near-silence. It
  **started at 19:24:27.2**, and the mic mute ran to 19:24:27.3, so the first
  ~1.3 s of the sentence was discarded before STT ever saw it. Three independent
  places added an echo tail on top of the clip:
  1. `_speak_pcm` set `_speaking_until = now + audio_dur + ECHO_TAIL` (1.5 s for a
     question, 3 s otherwise). That timestamp is a hard `return` in `_feed_audio`.
  2. The pre-POST block did the same with `+ 3.0`.
  3. `GLOBAL_TTS_UNTIL = now + audio_dur + echo_tail` — a **cross-camera** guard
     that drops a finished transcript outright with `TTS playback active (echo
     guard)`, so even after the mic unmuted the follow-up was thrown away with a
     good transcript in hand. This one is why the log looked like the user had
     said nothing.

  All three now end with the audio (`_SPEAKER_SETTLE_S = 0.3`, which only covers
  the upload and the duration rounding). **Recognising our own returning voice is
  not a timer's job and a blind window was never the right instrument for it** —
  it cannot tell our voice from the user's. Two layers that CAN do it already
  existed and are untouched: `_is_echo` (cross-correlation against the pcm we
  stored, applied before the chunk reaches the decoder) and `_echo_of_reply` on
  the transcript. The WAKE word keeps its own 3 s tail
  (`_wake_suppress_until`), because a «компьютер» decoded out of our own speaker
  really is a false wake and is a different failure.

- **STATE AS LEFT FOR FIELD OBSERVATION, 06.10.2026 ~01:50.** Measured on the
  living room, not asserted. Read this first when a field report arrives: it is
  the baseline the numbers below are compared against.

  **Working, verified:**
  * *The camera was answering its own pip.* `attention pip` leaked into the mic at
    the rail, the VAD opened an utterance ON it, and the buffer was
    `[1379, 1335, 1596, 32767×4, 2656, 1615, 1614]` — 1.6 s containing **no user
    speech at all**. Whisper transcribed the beep as «пап» and L2 spent 8 s on it.
    `_play_attention` muted for `now + 0.35` against a `duration = 0.4` pip, with
    `now` captured before the POST. Now measured from after playback.
  * Media transport: five commands in a row, each confirmed by the router's own
    `_media_moved()` re-read (`moved after an empty reply`) — `media_pause` at
    21:32 and 21:46, `media_play` at 21:11, 21:48, 21:51. «пауза» ->
    «Поставила на паузу», «Продолжай»/«Продолжаем воспроизведение» -> «Продолжаю».
  * Pause endpoint 6 of 7 ends via `pause`; the pause detector's `floor` tracks
    0.0110-0.0118 with `thresh` ~0.028 against measured speech at 0.037-0.13.
  * Television refused twice on the follow-up gate (`peak=4507`, `5949`, both
    below 24000) — and no TV utterance has reached the router since.
  * Wake: 3-4 triggers in 30 min, `suppressed=0` — no «компьютер» decoded and lost.
  * Drop counters healthy: `drop[muted=130 track=0 echo=3]`. All mutes fall inside
    playback; the 48 kHz and generic `_is_echo` paths are both counted now.
  * A follow-up borrows the previous turn's device and the lamp really moves:
    «включи свет в гостиной» -> «а теперь выключи» -> `HassTurnOff`, 0.13 s, with
    HA read back after every call (off -> on -> off).

  **Open, and what each looks like in the log:**
  1. **Level endpointing cannot separate a voice from a television in this room.**
     With the TV on, the room sits at raw rms 2800-3300 for six straight seconds
     while the user is at 4000-8000 — only three of the user's frames clear 2x the
     television, and `min_speech_frames=5` then blocks the end. Seen as an
     occasional `via cap 7s` followed by `Whisper empty` (3 in 30 min). Harmless
     but it spends ~2 s of Whisper on television. A rolling-percentile floor does
     follow the room (measured: 2940 vs the deployed 2676) and still does not fix
     it; the wake DECODER is what separates them, which is why this room decodes
     its wake word instead of scoring it.
  2. **`floor` is not room-tracking by design** — it only accepts frames it already
     calls noise, so a loud room cannot raise its own reference. It wandered to
     0.0164 (`thresh` 0.0409) once while the TV was loud, then recovered. That is
     the same cause as (1).
  3. **The 7 s cap still costs Whisper a full-length call on television.** Worth
     gating on `raw_rms` before calling STT, but not done: it is a change to when
     audio is discarded in a live path.
  4. **Two crash-time `except Exception: pass` used to hide real failures**
     (the activation cue). Both now log.
  5. **`CAMERA_ATTENTION_PIP_LIVINGROOM=false` is available and NOT set.** The mute
     is fixed, so the pip is now harmless to STT, but it still costs 0.4 s of
     deafness per wake and is 60x the room floor in the detector's reference.

  **Not to be re-derived:** the 97 s gap in the router log at 21:52 is a TELEGRAM
  turn (`stream=tg:543867556`), not a stuck camera utterance — the camera's
  `Post-wake retry` had produced `Whisper empty` and was dropped, never dispatched.

- **MEASURE THE SEND QUEUES BEFORE CALLING A BOX DEAD.** `netstat -ant` on the
  camera, one line per connection, is the instrument that settled this:
  `tcp 0 193712 192.168.22.241:554 192.168.22.102:37544 ESTABLISHED` — 193 kB
  written and unread. It answers three questions at once: who is not reading (the
  peer), how badly (the byte count), and whether the box is at fault (a large
  Send-Q is the box waiting on us, not the box failing). Still on record, all
  measured 05.10.2026:
  * A `majestic` restart does restore audio **completely without a power cycle**
    — audio "none" -> 100 % of real time, uptime unchanged at 8 min. So the
    user's instinct was right that a service hangs rather than the whole system.
    It is also only a band-aid: the room wedged again minutes later.
  * The box is **70-89 % idle** with 32 MB free throughout. Not CPU or memory.
  * `ai0_P0_MAIN` sits in `D` with `wchan=CamOsTcondTimedWait` **whether the audio
    works or not** — present during a 100 %-of-real-time capture. **Not a
    symptom**; the earlier note naming it as the culprit is withdrawn.
  * Restarting `majestic` also kills the go2rtc producer (bare `recv=None`, zero
    consumers), so any camera-side recovery MUST re-register the stream or the
    room stays mute on a perfectly healthy camera.
  * `HTTP latency alone is not a health signal`: the camera measured 0.066 s
    while RTSP was already dead, and 14 s while it was alive-but-blocked. Pair it
    with a producer that has a non-null `recv` and a consumer.
  **Consequence:** `scripts/camera_watchdog.sh` stays as a mitigation, but its
  first step is now the one that actually works — **restart the gateway**, not the
  camera. Nothing in it reboots hardware to fix our own back-pressure.

- **THE CAMERA IS REACHABLE OVER SSH (root@192.168.22.241, dropbear), and two
  "symptoms" in this file are measurement errors, not faults.** Added 05.10.2026 so
  that nobody re-derives them:
  * **`/proc/loadavg` is a constant on this box, not a symptom.** Measured
    11.3–11.6 with **55.5 % idle** and 1 runnable task of 69. Eleven vendor kernel
    threads sit in `D` with `wchan=msleep`/`CamOsTcondTimedWait`; D-state tasks
    count toward loadavg but burn no CPU. **The load average here means nothing** —
    so the earlier "load average 11.5" evidence for a wedged camera is withdrawn.
  * **`aio_dma` in `/proc/interrupts` is the PLAYBACK dma, not capture.** With
    `audio.outputEnabled: true` it sits near zero while audio streams at 100 % of
    real time. The "41/s is this camera's healthy audio-DMA rate" note above is
    **wrong for this firmware** and must not be used as a health signal.
  * **There is no separate audio service.** `/etc/init.d` has only `S70vendor`
    and `S95majestic`; capture and playback both live inside `majestic`, so the
    only cheaper-than-a-reboot lever is `/etc/init.d/S95majestic restart`, and it
    costs the video stream too. `reboot` and `/proc/sysrq-trigger` both exist.
  * **The cheapest reliable health signal is HTTP latency to the camera**:
    7.38 / 7.44 / 7.45 s across three tries while wedged, 0.016–0.050 s when
    healthy, with no overlap. A trivial `GET /` is enough; no stream capture
    needed. Real-time audio was also measured directly at 100 % when healthy.
* **CAMERA WATCHDOG: `scripts/camera_watchdog.sh`, written 05.10.2026, NOT yet
  installed or tested.** One-shot, meant for `*/3 * * * *`. Key-based SSH is
  already in place (`~/.ssh/voice_watchdog_ed25519`, authorised on the camera)
  so no password lives in a script. Design decisions worth keeping:
  * signal = HTTP latency above `WEDGE_S` (2.5 s) **confirmed
    `CONFIRM_N` times `CONFIRM_GAP_S` apart** — one slow sample is a busy CPU or
    a go2rtc re-register, not a wedge;
  * **UNREACHABLE is not WEDGED and is deliberately never acted upon.** No answer
    means the network, and rebooting a camera you cannot reach costs a boot cycle
    and fixes nothing; go2rtc and the gateway's own heal already cover it;
  * recovery is a ladder, cheapest first: restart `majestic` → re-check →
    only then `reboot`, with everything before and after measured and logged,
    because "it recovered" is worthless without knowing WHICH step did it;
  * `MIN_GAP_S` (600) rate-limits attempts so a flapping camera cannot be
    reboot-looped.
  **Status: the healthy path was verified (silent exit 0). The recovery ladder is
  untested, because testing it means deliberately wedging the camera.** It needs
  `bash -n`, the two dry runs below, and then a cron entry — all of which need a
  shell this session no longer has.
- **A FAST TELEGRAM VOICE TURN IS NOT EVIDENCE THAT THE VOICE PATH IS FAST —
  Telegram never touches the camera.** Measured 05.10.2026 09:53:19.447 voice
  note in, 09:53:20.844 voice reply out: **1.397 s for the whole turn** (Whisper
  + L1 + one TTS). Two structural reasons it cannot be compared with a camera
  turn: `telegram_client.py` synthesises **once for the entire reply** (one
  `_tts_mp3(excerpt)` at the end, while the text is edited into the message live
  as it streams), and it has no wake word, no VAD endpoint and no speaker. So a
  voice turn that "feels fast in Telegram" says nothing about the room — the
  comparison removes the camera rather than improving the pipeline. For the
  camera path, measured on the same day: speech start 09:54:39.753 -> utterance
  end 09:54:43.614 -> Whisper 09:54:44.276 (+0.66 s) -> L1 09:54:45.054 (0.17 s)
  -> reply TTS 09:54:46.149 (+1.10 s). **The two big costs are the utterance
  collection and the single Edge-TTS round trip, not the L1 router** (0.17 s).
* **A wedged camera makes confirmation stutter, and its HTTP latency is the
  cheapest tell.** 05.10.2026 10:38: the gateway logged `🔴 CAMERA AUDIO DEAD
  (0% of real time)`, go2rtc reported the producer with `recv=None` and **zero
  consumers**, and a trivial `GET http://192.168.22.241/` took a consistent
  **7.4 s across three tries** (a healthy OpenIPC answers in tens of ms) —
  while an EMPTY `POST /play_audio` took 4.7 s. Capture and playback both go
  through that box, so every confirmation stutters, and the mic stops feeding
  Vosk. Do not chase this in the audio pipeline: check the camera's HTTP latency
  first, and remember that only a power cycle clears it (ONVIF `Reboot` is not
  implemented on OpenIPC, per the watchdog section above).
- **A CONSUMED SENTINEL IS NOT AN END-OF-STREAM TEST — the camera was deaf for
  119 s after EVERY reply, and it is the reason «Да?» happens.** Measured
  05.10.2026 09:54: wake, `UTTERANCE END dur=2080ms via pause`, Whisper OK
  `'включи свет'`, reply TTS at 09:54:46.149 — then **no `VAD SPEECH START` and
  no `UTTERANCE END` at all** until 09:56:45, which is
  `wait_for(player_task, timeout=120.0)` expiring to the second. The user spoke
  again at 09:54:58, the wake fired, and the utterance was never committed, so
  the greeting answered instead: «Да?» is this, not a lost command.

  `_nanobot_player_task` reads **two** queue items per pass, and only the
  `None` at the TOP of the loop counted as end-of-stream. With the queue
  `[sentence, None]` the sentinel was taken as the "next sentence", the
  `if nxt is not None` check correctly skipped the prefetch, `sent = nxt` made
  `sent` None, and the next pass did `sent = await q.get()` on a queue nobody
  would write to again. It blocked forever. The onset into `_vad_has_speech`
  requires `not _processing_utterance`, held for the whole of `_call_backend`,
  so the blocked player took the microphone deaf with it. **A blocked consumer
  behind a broad `except Exception` is silent** — there is no log line between
  the reply's TTS and the expiry, which is what makes this so easy to misread as
  "the wake word is deaf".

  Fixed by flagging the sentinel (`stream_done`), playing the sentence in hand,
  and exiting **afterwards** — breaking at the sentinel drops the reply, because
  `sent` has not been synthesised yet. Three tests, all of which fail on the old
  code on their own 5 s timeout.
- **A camera must listen after an ANSWER too — it is a separate path from the
  satellites, and the asymmetry is real.** `main.py::_finalize_turn_followup()`
  sets a satellite to LISTENING in **both** branches, with
  `STANDBY_TIMEOUT_QUESTION` 30 s and `STANDBY_TIMEOUT_STATEMENT` 10 s. The
  camera opened a window only for a question and called `_back_to_wake()` the
  moment it had answered, so «а теперь выключи» was never heard and every
  follow-up needed the wake word again — reported 05.10.2026 as "the dialogue
  setting does not work for cameras the way it does for the ESP32". Both branches
  now open a window: `CAMERA_DIALOGUE_QUESTION_S[_<NAME>]` 30 s and
  `CAMERA_DIALOGUE_STATEMENT_S[_<NAME>]` 10 s (0 => those defaults), opened when
  the reply FINISHES sounding rather than when it is queued, and the player
  closes the mic from an explicit per-turn `_followup_open` flag — not from
  `last_q`, which is what made the camera deaf, and not from a flag that could
  stay True across turns and hold the mic open forever. The statement window is
  short on purpose: it is for a natural follow-up, not for listening to the
  television. What makes that safe is that the wake word is DECODED — TV speech
  has to contain «компьютер» to open anything — plus `_wake_gate_open()` while
  our own playback is in the air.
- **THE PAUSE ENDPOINT IS PROVEN IN THE FIELD.** 05.10.2026 09:54:41.346 wake →
  09:54:43.614 `VAD UTTERANCE END dur=2080ms via pause (rms=0.0121 ref=0.0699
  floor=0.0067 run=6 speech=8) (awake ends: {'pause': 1})` → 09:54:44.276 Whisper
  OK `'включи свет'`. **2.93 s from wake to transcript, of which 2.08 s is the
  utterance** — against 8.0 s and a 7040 ms cap before. The endpoint was worth
  enabling on its own evidence; note the `ref` is 5.8x the `rms`, i.e. the level
  dip it fires on is real rather than the room floor moving.
- **THE L1 COMMAND PATH IS VERIFIED END-TO-END — measure it before blaming the
  audio.** 05.10.2026, after a full rebuild of `voice_gateway` + `jev-router` +
  `smolagents-worker`, driven straight at `POST :8091/route` with the state read
  back from HA `/api/states` after every call:

  | command | reply | time | state |
  |---|---|---|---|
  | выключи свет | Выключила | 0.17 s | **off** |
  | выключи свет | В гостиной уже выключено. | 0.10 s | off |
  | включи свет | Включила | 0.17 s | **on** |
  | включи свет | В гостиной уже включено. | 0.12 s | on |
  | выключи свет | Выключила | 0.16 s | **off** |

  Two things this establishes, both of which had been argued about: the side
  effect is REAL (the entity id is `switch.living_room_light_swith_relay` — no
  extra `switch` — and it actually flips), and the repeat is an honest refusal
  rather than a second «Выключила». **0.10–0.17 s** is the whole L1 cost, so if
  a voice turn feels slow, the time is upstream of the router, not in it.
  `/route` answers **SSE**, not JSON: read every `data:` line and take the last
  one carrying text, or you will report the `route` event and think the command
  produced no reply.
- **THE ON/OFF FAST PATH NEVER VERIFIED, AND A 15-SECOND PIP LIMIT TURNED ONE LIE INTO A CASCADE. Measured 06.10.2026 18:21 UTC, kitchen, in one report.** The whole chain, in order:

  ```
  18:21:04 VOSK WAKE → 'включи кофеварку' → router conf 0.92 → speaking: 'Включила'
  18:21:16 VOSK WAKE → 🔇 attention pip skipped: 3.2s left of the 15 s limit
  18:21:19 VOSK WAKE → 🔇 attention pip skipped: 0.0s left of the 15 s limit
  18:21:23 VOSK WAKE → 🔔 attention pip
  18:21:26 Whisper OK: 'Почему ты пиздишь, что не включил кофеварку?'  → dispatched
  18:21:51 Whisper OK: 'включи блядь кофеварку уже'  → speaking: 'Включила' again
  18:22:06 VOSK WAKE → 🔔 attention pip
  18:22:08 Whisper OK: 'снова пиздишь'  → dispatched
  ```

  `switch.coffemaker` ended at **`off`, `last_changed` 13:14 UTC** — it never moved, across both claims. The user's two replies were spoken AT the room and dispatched as commands, and on both the router produced **no sentence at all**: a lie, then silence, then silence.

  **1. The verification existed and was wired into the wrong path.** `_touched_a_usable_entity()` was added on 04.10.2026 precisely because «включи свет» answered «Сделала» with the lamp untouched — and it is called only on the **blind** path (line ~558). The per-entity on/off path returned `call.speak_ok` on `ok and not fatal` **without ever re-reading a state**. Same class of lie, one path guarded and one not.
  * `_confirm_on_off_moved(ids, before, want_on)` now gates that return: snapshot the states before the calls, then read back with a short retry (0/0.25/0.5 s — a relay is not instantaneous, and a single instant read would produce a false refusal). Nothing reached the requested state → `unverified_side_effect` escalation carrying the real blocker, never a spoken success.
  * **`reached` means `state == target` and nothing else.** The first version also accepted `before != now` "as evidence it moved", and probing it against the **running** router before trusting it showed both holes: a device left in the opposite state reported `moved`, and an entity **absent from the registry** reported `moved` (it reads as an empty state, which therefore "changed"). **The first version would have accepted the very lie it was written to catch.** A unit test that only exercised "stayed exactly off" passed over both holes — a check that looks like a safeguard and is a no-op is worse than none, and only a live probe found it.
  * Verified live afterwards: reached → `moved`; went the other way → `unmoved`; absent from the registry → `unmoved`; `unavailable`/`unknown` → `unmoved`; already in the requested state → `moved` (the user asked for a state, not an event).
  * **The relay itself was not permanently broken**: after the deploy «включи кофеварку» moved it (`last_changed=18:37:27`, state `on`) and the room said «Включила» in 0.5 s. So the 18:21 failure was real but transient, which is exactly the case a verification exists for.

  **2. A limit on the confirmation is a limit on the user's retries.** Two of three wake words got **no beep**, because `_play_attention` opened with `now - _last_attention < 15.0`. The silence is what made the user repeat, the repeats fired three wakes in nine seconds, and the third utterance collected the complaint. What 15 s was really protecting against — two pips from one utterance — is already blocked upstream by `_WAKE_REARM_DEBOUNCE_S` (3.0 s), at the decoder, where the second wake is *refused* rather than its beep *hidden*. **`_ATTENTION_MIN_GAP_S` is now 3.5 s**, just above that, so every distinct wake gets feedback.
  * A hardcoded `11.1 s` in a test stopped being a skip the moment the limit moved, which is why the test now derives its offset from the constant: **a test that pins a log line must not pin the value behind it.**

- **AN HONEST `ok` IS NOT A DEVICE THAT MOVED.** The MCP intent answers `{"data":{"success":[...],"failed":[...]}}`, and `success` is full of things that are not a working device. Measured 04.10.2026 on «выключи свет» in the living room: the blind call (`domain:["light"]`, `area:"Living Room"`, **no `name`**) returned `success: [{type:area, id:living_room}, {type:entity, id:light.wled_living_room}]`, `failed: []` — HA matched the ROOM plus a WLED strip that has been `unavailable` since 28.09 — so `ok=True`, the gateway said **«Выключила»**, and the lamp (`switch.*_relay`, which `domain:["light"]` cannot match at all) never changed. Three rules now, each with a test:
  1. `get_states()` returns its **previous snapshot** on failure, and that snapshot is `[]` on a cold cache — so an empty list means "HA did not answer", NOT "no such device". An empty registry and an empty target set both **refuse**; on/off never falls through to the blind intent any more.
  2. `_touched_a_usable_entity()` is required before any `speak_ok`: HA must name at least one entity of `type: "entity"` that is **present in the live registry** and not `unavailable`. An `area` entry means a room matched, not a device. (An id ABSENT from `/api/states` counts as unverified — that mistake was in the first version of this check and its own test caught it.)
  3. The router logs `speaking: '…'` — without it a false side effect is invisible. On 04.10.2026 the log showed a clean `easy_action`, no error anywhere, `elapsed=0.18s`, and the only evidence was HA's own `last_changed`, which had not moved.
  **Nothing in the suite imported `services/jev-router/app.py`** (the name `app` belongs to FastAPI in `test_main.py`), which is why this shipped; `tests/test_router_resolution.py` now loads it under the name `jev_router_app`.
- **A «Да?» greeting means the command was NEVER CAPTURED — it is not a failed command.** The onset into `_vad_has_speech` requires `not _processing_utterance`, so speech arriving while a previous turn is in flight is dropped before Whisper sees it, and `_wake_greeting` (which tested `_vad_has_speech`) read the room as silent. Measured: VAD `speech=True consec=54` with **no `VAD SPEECH START` line at all**, then «Да?» at wake+5.0 s. The greeting now asks `_last_speech_at` (wall clock of the newest speech frame) instead. Note also that in this room the VAD never reports silence, so an utterance only ever ends at the **7 s** cap — anything that fires before 7 s is cutting into the command it is waiting for.
- **A wake word must stay detectable at all times — never gate its FEED, only its FIRING.** Gating the vosk `feed()` on `not _wake_detected` (added 04.10.2026) made the room deaf for the whole 60 s post-wake dialogue window: the user said «компьютер» again and nothing happened, which is indistinguishable from a broken detector — the exact confusion `vosk diag` exists to prevent. The decoder now takes every chunk always; a wake heard inside an open window is counted in `suppressed=` and dropped without firing, because a second `_fire_wake()` clears `_vad_speech_buf` and destroys the command being collected.
- **The container logs in UTC while the host is MSK.** Loguru's printed timestamps are 3 h behind the host clock while the epoch is identical, so `docker logs --since 6m` returns lines whose text looks 3 h old and `docker logs | tail` looks like the gateway died. Compare epochs (`date +%s` vs `docker exec … date +%s`), not the printed prefix.

- **Media transport** (Kodi/TV/speaker, added 03.10.2026): HA's MCP server exposes exactly TEN tools and not one of them is a media intent (`tools/list`, verified live), so pause/play/tracks/volume can NEVER go through `ha_action` — the model used to invent `intent__HassMediaPause`, burn two steps on «Tool … not found» and answer «не удалось». Two paths now reach the REST service instead: L1 resolves «пауза коди во владиной комнате» into `media__<service>` + `POST /api/services/media_player/…` (`app._execute_media`, ~0.3 s), L2 has `media_control(action, area, name, value)`. `MEDIA_SERVICES`/`MEDIA_PREFER` live in both services and `test_hint_sync.py` holds them together. Target choice (`find_media_targets` / `resolve_media_targets`): `unavailable` entities and AirPlay/DLNA endpoints are not players; the device word AND the room are intersected (hint «коди» → «le» matches all four LE boxes, so the room must decide); an exact area beats a containing one («спальня» must not reach «Bedroom Vlada»); a room holding no player refuses; a pronoun («поставь ЕГО на паузу») resolves by `MEDIA_PREFER` — the state the request is ABOUT, `paused` included, because repeating «на паузу» on a paused box means that box; a satellite's default area is dropped (speaker location ≠ the device meant); a level-less `volume_set`, an already-reached state («Уже на паузе.») and a service call that changed nothing are honest answers/escalations, never «готово». `tools.MCP_TOOLS` is the whitelist `ha_action` checks BEFORE sending — an unknown tool name is refused in code with the real list instead of being handed to HA.
- **Media side effects are ASYNCHRONOUS** (verified 03.10.2026 on `media_player.le_vlada`): HA answers `POST /api/services/media_player/media_pause` with `[]` INSTANTLY and the entity flips to `paused` ~2 s later, while `media_position_updated_at` may be minutes stale (HA extrapolates the position locally, so a `playing` state with a frozen timestamp is NOT proof a box is really playing). Therefore `changed: []` is not proof of failure — `_media_moved()` (router) and `media_control` (worker) re-read the player for ~2 s and compare `media_fingerprint()` (state + volume_level + is_volume_muted + media_title + media_content_id + source; `media_position` deliberately excluded because it ticks on its own). An attribute-only change (`volume_set`/`volume_up`) never shows up in `changed` at all, and `volume_level` is `None` on an idle Kodi — a relative «громче» cannot move there while `volume_set` can. Skipping the wait made every WORKING pause look like a failure (4-8 s L2 escalation + «не получилось»), and skipping the fingerprint would have called a real volume change a failure.
- **A volume call on an IDLE box is answered, never escalated** (04.10.2026 21:08): `volume_up` is deliberately NOT in `MEDIA_TRANSPORT` ("a muted idle box is still a box whose loudness the user is asking about"), so the «nothing was playing» branch skipped it — and since a volume change never reaches HA's `changed` list and lands late on an idle Kodi (le_spalnya 0.80 → 0.85 arrived AFTER the 2.1 s window), the unmoved fingerprint became a failure: «сделай громче в спальне» escalated and the turn ended 60+ s later (a 29.7 s LLM step plus an httpx retry) for a command HA had already accepted. `media_no_movement_answer()` now lives in `ha_client` (unit-testable without fastapi) and answers «{room} ничего не играет.» for transport AND volume on a box that was idle before the call; only a PLAYING box that stayed put escalates. Two supporting rules: volume gets a longer re-read window than transport (`MEDIA_CONFIRM_DELAYS` `{default: ~2.1 s, volume: ~6.8 s}`, duplicated in `ha_match` and held together by `test_hint_sync.py`), and a box reporting `volume_level: None` (le_kitchen, le_vlada — le_spalnya reports a real 0.85) is not polled at all, since no wait can confirm a level that does not exist. Measured after the fix: 1.0 s «Сделала громче» on the bedroom box, 0.2 s «ничего не играет» on the two boxes with no level, zero escalations.
- **A domain named in a `ha_read` query is a HARD filter** (`query_domains()` in ha_match): «media_player в гостиной» answered with ten living-room SWITCHES (the light relay + six camera switches) and not one player, because `matchers()` ORs the room word with the domain word. The model then planned two steps on that list (field case 03.10.2026 18:21, «Ну так найди её и включи»). The domain token is also REMOVED from the fuzzy set, or «light» + «гостиной» returns lights from all six rooms, and a room the entity names cannot carry (`media_player.le_zal_2`) is resolved from the registry area map — `match_states(..., area_map=)` — as an OR with the name test, never a narrowing.
- **«СНИМИ С ПАУЗЫ» WAS PAUSING, AND A BARE «ПАУЗА» HAD NO ROOM AT ALL. Measured 06.10.2026 20:10, living room, two commands in a row:**

  ```
  20:10:20 'иметь паузу'   -> media_pause on media_player.le_vlada   -> «Поставила на паузу»
  20:10:40 'сними из паузы' -> media_target_ambiguous_or_absent — escalating
  ```

  The user asked the camera to unpause and it paused — then failed, because a pause on an already-paused player has no target to move — and **then asked which room to pause in.** Two independent faults:
  * **Order in `RE_MEDIA` IS the semantics.** `\bпауз\w*` sat FIRST, matched the bare stem inside «паузы», and the scan breaks on the first hit, so every unpause phrasing became a pause. The resume patterns are now before it, and accept `с` **or** `из`: the log contains both «сними из паузы» and «сними с паузы гостиной» and the old pattern allowed only `с`.
  * **The default area was dropped for media, and only for media.** `if area_src == "default" and not media_noun and not thing_stem` sent `args={'service_data': {}}` — no area at all — so `_execute_media` guessed among four boxes and chose `le_vlada`. The lights take the default area in the same breath (`{'domain': ['light'], 'area': 'Living Room'}`), so the asymmetry was in this branch alone. The drop is now scoped to `_MEDIA_RELATIVE` (`volume_up/down/mute/set`): **«пауза» said in a room names exactly one player**, which is precisely the case the original reasoning excluded. Verified live: living room → `le_zal_2`, kitchen → `le_kitchen`, «громче» still drops the room and still follows `MEDIA_PREFER`.
- **`CAMERA_FOLLOWUP` IS A THREE-STATE MODE, AND ITS DEFAULT IS `question`.** The user, 06.10.2026: **when the camera asks a question it must listen to the answer immediately, no wake word** — that is the "asks a question and then ignores the answer" complaint. An answer to OUR OWN question is the one follow-up that needs no keyword.
  * `none` | `question` (default) | `all`. `all` is not the default because a **statement** window is 10 s of open microphone in a room whose television is louder than its occupant, and it was measured accepting noise («атака» 32522, «пиздец» 13197) while refusing the user (4725, 9124). A question window is short and something is expected to be said in it.
  * **It is a mode and not a duration because `0` means "unset, use the default" throughout this file** — a window of 0 would produce a 30 s window, not none. There is no numeric value for "no window at all" here. An unrecognised value falls back to `question`, so a typo in the env var cannot close the microphone for a question.

- **Media on/off is a transport call, not an intent** (03.10.2026 18:21): «включи коди» → `intent__HassTurnOn` answered `MatchFailedReason.INVALID_AREA` for the RU room and then `MatchFailedReason.ASSISTANT` for `name='LE-zal'` — the MCP server has no media intent and the Kodi entities are NOT exposed to Assist, so that path can never reach them. `MEDIA_POWER_ON/OFF` send `media_play`/`media_stop` instead, which ignore Assist exposure; `homeassistant.turn_off` is deliberately NOT used (the box would sleep and no voice command could wake it). The media nouns also cover CONTENT words («серию», «эпизод», «передачу») — that is how the user asks for a player.
- **Every escalation carries the canonical room name** (`resolver.area_of` → `ctx`): «гостиная» is not an HA area, and an intent fed the RU word answers INVALID_AREA, which cost the model a step it then spent guessing device names. The classifier's media verb list, the resolver's `RE_MEDIA` and `_regex_action()` all have to agree — the last one is the SINGLE definition of "action-like", reused by the cold-embedder path.
- **The classifier warm-up is CHUNKED and retried**: embedding all 60 route utterances in one request took **102 s** on this host (≈1.7 s each) against a 30 s client timeout, so the warm-up failed on EVERY start — `classifier warmup failed: ` with an EMPTY message (a timeout, logged without its type). `EMBED_CHUNK = 8` keeps a request near 14 s, `warmup()` retries once for the cold-model case, and a cold embedder no longer kills the pure-regex action path (`_regex_action`), which needs no cosine at all — one cold Ollama used to escalate every voice command in the house.
- **An honest refusal beats an invented one** (03.10.2026): the model answered «не смог связаться с телевизором» when the real blocker was «такого сериала нет в библиотеке» — a connection diagnosis it had no evidence for. Two rules keep that class out: `media_search`/`media_play` REPORT the library state instead of guessing it, and the TASK_TEMPLATE forbids inventing a reason when a tool has already said what is missing. A refusal is still the right answer when a box is offline (`«ящик во владиной комнате не отвечает»`), which is a fact — unlike «не смог связаться», which was not.
- **Kodi library search & play by title** (`kodi.py`, 03.10.2026): HA exposes no media intent and no library search, so «включи серию "Темного зеркала"» was impossible and the model invented a reason («не смог связаться с телевизором»). `media_search(title, area)` returns the whole catalogue (so the MODEL bridges «Симпсоны»→«The Simpsons» — no string matcher can) and `media_play(title, area, episode)` opens a library item. Details that cost real time: the boxes are identified by their OWN zeroconf name (`System.FriendlyName` → "Kodi (LE-zal)" == HA friendly_name), so `KODI_HOSTS` is a seed list, not a room table. CREDENTIALS ARE PER BOX and the seed format is `[user:pass@]host`: three boxes answer to kodi/2441 and the fourth — 192.168.22.176, found by DNS `le-vlada.local` because a plain /24 sweep misses a box that 401s — to kodi/kodi. Two measured failure modes: probing it with the shared password reported «ящик не отвечает» (a wrong diagnosis, not an outage), and a probe row that dropped its seed answered 401 on the next call, so `probe_boxes` carries `seed` in every row; `VideoLibrary.Search` does not exist on Kodi 21.3 (use `GetTvshows`/`GetEpisodes`); `Player.Open` takes `{"item": {"episodeid": N}}` — `{"item_id":…, "video":0}` answers "Too many parameters"; the Jellyfin plugin needs 8-10 s to start, and the verification waits for the STATE to become `playing` rather than for a field to change — re-opening the episode a box is already playing changes nothing and was reported as «не смог открыть файл» for a show that WAS playing (19:55). `media_search` lists the whole catalogue but fetches episodes for the BEST MATCH only: one round trip per show cost 14 of them and 18-22 s per turn. The boxes are probed AND scanned in PARALLEL (`ThreadPoolExecutor`): sequential probing charged every media turn a full timeout for each box that was switched off (measured: probing four healthy boxes 0.04 s parallel), and `probe_one` must wrap its single `friendly_name()` round trip completely — an unguarded second call let one half-dead box raise straight through `boxes()` and break the whole media path. `KodiError.kind` keeps an HTTP 401 («неверный логин или пароль») apart from an outage («не отвечает»): they need opposite fixes, and calling a live box «не отвечает» is what sent the search after the wrong thing today. A library start is therefore 13-18 s end to end, most of it LLM steps; a NAMED episode that is not in the library is refused with its real range instead of a silent substitution; «следующая серия» means the first `playcount == 0`, not the first row. Credentials/hosts live in the gitignored `.env.kodi` (`env_file` on the worker).
- **Assist exposure for media lives in the entity registry**, and HA 2026.9's websocket schema is unusual: `config/entity_registry/update` takes `entity_id` = the ENTITY (`media_player.le_zal_2`, the row id answers "is an invalid entity ID") and `options` ONLY together with `options_domain` ("some but not all values in the same group of inclusion"); the payload is `options: {"should_expose": true}` + `options_domain: "conversation"`, and passing the scope twice stores `{"conversation": {"conversation": …}}`. All four Kodi boxes were opened this way (they answered `MatchFailedReason.ASSISTANT` before), so the intent path works too — but pause/volume/next-track still go through REST, which ignores exposure.
- **A latin room word must be matched FORWARD only** (`area_matchers`): «Living Room» tokenises to «living» + «room», and the Russian-style reverse test (`tok in stem`) made «room» a substring of the stems «bedroom»/«bathroom», so the area became {living, bedroom, bathroom}, the exact-room rule preferred the phantom «Bedroom», and `media_play(area="Living Room")` started the show on the BEDROOM box (field check 03.10.2026 19:32). Two rules then pick the room, in this order and identically in both services: the player's area carrying the MOST named room words («Bedroom Vlada» over «Bedroom»), then an exactly named room over a merely containing one.
- **`_area_phrase` must not fall through to a generic declension** (04.10.2026): the resolver hands over the CANONICAL HA area name («Bedroom»), so «В спальнее» never actually reached the user — but the helper is also called with a Russian display name and with whatever the caller carries, and its fallback turned «спальне» into «В спальнее» silently. `_prep_candidates()` now derives the nominative from the oblique («гостиной»→«гостиная», «улице»→«улица», «прихожей»→«прихожая») and looks THAT up, so an unknown word still falls back instead of inventing a room.
- **Hermes is diagnostics only**: `hermes_expert` is a bare `/v1/chat/completions` (no tools, 900 tokens, 25 s). It answers «почему сломалось», it cannot execute a command — TASK_TEMPLATE says so, so a media/vacuum/light failure is never handed to it (the Kodi turn that started this was answered by the gateway itself, 0.3 s).

- **THE PAUSE ENDPOINT CLOSED AN UTTERANCE ON THE WAKE WORD ALONE, AND THE TELEVISION TOOK THE SLOT THE USER WAS SUPPOSED TO FILL.** Measured 06.10.2026 10:25 UTC, one wake and everything after it a room answering its own TV: `VOSK WAKE 10:25:10.900` → `UTTERANCE END 2560ms via pause (rms=0.0207 floor=0.0110 thresh=0.0274 run=6 speech=7)` → `Whisper OK: 'компьютер'` → then three 7 s caps carrying «Числят.», «Часок на один. Это дождь.» and «Что ты думаешь, что я не ехать ночью?» each dispatched to the router and answered aloud.
  * **Why `min_speech_frames` could not save it.** That guard exists for exactly this case and its comment is right — the gap after «компьютер» is a gap between the wake word and the command, not the end of a sentence — but the rule as implemented CANNOT see the difference: **the wake word IS speech, and it arrives before the endpoint has any reason to suspect anything** (`speech=7` is the word's own frames). Five frames of wake word are exactly five frames of command.
  * **The fix is to not close the utterance, not to raise a threshold.** `_handle_wake_or_command` sets `_wake_only_pending` instead of returning, and `_process_utterance` calls `_reopen_utterance()` — restoring `_vad_has_speech`, the silence counter, the endpoint AND `_preroll`. The wake word's audio is restored with the pre-roll deliberately: without it the utterance resumes at the moment of the reopen and loses the first syllable, which is the bug the pre-roll exists to fix. The next pause then commits wake word and command together, and there is no empty slot for the television to occupy. A threshold here would only have moved the failure; nothing about this is a level problem.
- **A DECODER RUN ON NEAR-SILENCE RETURNS ITS MOST LIKELY PHRASE, AND FOR THIS SYSTEM THAT PHRASE IS «КОМПЬЮТЕР».** Measured 06.10.2026 08:55 on the kitchen, whose mic is dead (rms 3-4, peak 129-388): `VOSK WAKE trig='компьютер' conf='компьютер'` → `attention pip` → `Whisper IN: 1.15s raw_rms=77 peak=388 dB=-52.6 gain=20.0x` → `Whisper OK: 'Выключи.'` → dispatched with no device named, so `resolve_action` had nothing to resolve and nothing moved. **The whole of the «answered unintelligibly, did nothing, and the beep was strange» report is one hallucinated wake followed by one hallucinated command, both out of silence.** The wake word and the command were each invented by a different model on the same dead input; there was no audio to invent anything from.
  * Gate: `CameraConfig.wake_min_peak` / `CAMERA_WAKE_MIN_PEAK[_<NAME>]`, `0` => `_WAKE_MIN_PEAK` = 3000. **Measured on BOTH sides**, which is what makes the number defensible rather than argued:

    | | ambient (60 s, nobody in the room) | real speech (recorded utterances) |
    |---|---|---|
    | per 1 chunk | median 1148, p99 1660, **max 2316** | p10 1180, p25 2912, median 5259 |
    | per 12 chunks (1.9 s) | median 1394, **max 2316** | p10 1579, p25 5024, median **11250** |

    The four ambient decodes the gate refused measured 1106 / 1452 / 2076 / 2324 — **every one of them was the loudest ambient chunk there is.** So 3000 clears the measured ambient ceiling by 30 % and sits well under speech.
  * **It is evaluated ONLY on a chunk the decoder already fired on**, so it cannot suppress a wake — only refuse one — and it costs nothing on ambient audio.
  * **It reads the LOUDEST chunk of the last 1.9 s, not the trigger chunk** (`_WAKE_PEAK_WINDOW`, a bounded deque — deliberately bounded, so a loud moment cannot launder a later silence, which is its own test). «компьютер» spans ~10 chunks at the 160 ms hop, so the trigger chunk is a sample from an arbitrary point inside the word. Measured effect on the same recordings: speech median **5259 -> 11250**, p25 **2912 -> 5024**, while the ambient ceiling **does not move at all (2316)**. A free 2x on the accept side.
  * **The error this is allowed to make is asymmetric, deliberately.** A refused real wake is recoverable — the user says it again. A false wake opens a 60 s window in which the room answers its own television, which is the failure that shipped. Do not lower this toward the ambient ceiling without a new two-sided measurement.
  * **Still unconfirmed: the peak at the trigger moment of a REAL wake word.** Every number above comes either from ambient or from speech not known to contain the word. The first real «компьютер» in each room settles it — `VOSK WAKE` logs `peak=` and `min_peak=`, and a refusal logs `peak=` plus `window=N/max` and the window length. **Read those before changing the value.**
  * This is a *containment*, not a repair. It makes a silent room refuse its own inventions instead of speaking them.
  * It also showed the cascade earns its place: the living room produced **4 ambient decodes in 7 seconds at 14:25** with nobody in it, which before the gate were 4 false wakes, each opening a window that collects the television.
- **HTTP LATENCY IS NOT A HEALTH SIGNAL, AND A WATCHDOG THAT TRUSTS IT WILL RESTART A HEALTHY GATEWAY.** `scripts/camera_watchdog.sh` fired **17 times in a day** on `WEDGE_S=2.5`, which was derived when ONE room was polled and a healthy OpenIPC answered in 0.016-0.050 s. Measured 06.10.2026, 20 samples per camera: living room min 0.201 / median **2.354** / p90 **8.492** / max **11.840** s, kitchen min 0.195 / median 0.475 / p90 0.940 / max 1.114 s. The living room was in the range this file calls wedged **while delivering audio at rms 371 with Send-Q = 0, the box 88.8 % idle, and go2rtc holding a producer with a consumer**. So it was not wedged; the camera does more media work than when the threshold was calibrated and its web UI queues behind the media pipeline. (It reads 0.017-0.033 s again when the gateway is quiet — so even the elevated band is a load artefact, not a fault to chase.) This is this file's own warning ("HTTP latency alone is not a health signal", "it measured 0.066 s while RTSP was dead") applied to the watchdog that read it.
  * **Four states, and only a BROKEN STREAM acts**: `healthy` (fast + streaming), `slow-only` (slow + streaming — logged, never acted on), `dead-stream` (fast + not streaming — our reader), `wedged-slow` (both, which is what a real outage looks like). `WEDGE_S` 2.5 -> 6.0, `CONFIRM_N` 2 -> 3, `CONFIRM_GAP_S` 15 -> 20, so a camera must stay broken for ~a minute before anything disruptive happens.
  * **A restart is not free** — it drops every model, every RTSP loop and any turn in flight — so it must not be reachable by one transient sample. That is why the confirmation counts, and why a false positive now costs two rooms.
  * **`DRY_RUN=1`, because testing this script restarted the gateway.** Its first step IS `docker restart voice_gateway`, so a script whose first action is to restart the thing it is meant to protect cannot be exercised against a fault at all — which is how the 17 false positives stayed undiscovered. It now decides and logs what it *would* do, touching nothing; verified against a dead go2rtc it prints `[dry-run] would: docker restart voice_gateway` and the container's `StartedAt` does not move. `STAMP_DIR` is created rather than assumed, which was the second thing that test found.
  * **A watchdog that can restart a shared container must be tested against a fault before it is armed.** Mine was armed first, and the first thing it ever did was the wrong thing seventeen times.
- **THE KITCHEN MICROPHONE WAS A VOLUME SETTING, NOT A FAULT.** Measured dead for a week (rms 3-4, -77.8 dB, `peak_max` 297 over 3742 chunks, `triggers=0`) and reported as a code fault; it was `audio.volume` at **30**, raised to **100** on 06.10.2026. Measured after, and no `majestic` restart was needed — the volume applies on its own after a reboot:

  | | rms | dB | peak | samples > 50 |
  |---|---|---|---|---|
  | was (30 %) | 4 | -77.4 | 129-353 | 0.06 % |
  | **now (100 %)** | **410** | **-38.0** | **2302** | **90.2 %** |
  | living room, for comparison | 417 | -37.9 | — | — |

  60 s of kitchen background, 100 ms chunks: rms min 318 / median **389** / p90 439 / max 1498, chunk peak median 1282. Spectrum: **60.4 % of the energy is 20-200 Hz** — the permanent appliance hum the `ns_rms_gate` below exists to strip. **A room this size needs `audio.volume` at 100**; the living room's mic reads 50 as `rms 9`, which is silence.
  * **Do not touch `CAMERA_NS_RMS_GATE` on this evidence.** NS applies only while `rms_raw < ns_rms_gate`, and the gate is **400** in both rooms while the measured ambient is 389 (kitchen) and 416-424 (living room) — so on these numbers NS is effectively never applied in either room. That was not true before the volume change: at 30 % the kitchen's ambient was ~117, so a quiet utterance sat under 400 and did get NS. **The regime changed under the gate, not the gate's logic** — but the gate was set to protect a *distant* command, and whether one still arrives at that level is a question only a real kitchen utterance can answer. Watch `Whisper IN: … raw_rms=… ns=0|1` from the first real command and decide from that number, not from this paragraph.
  * `~/.ssh/voice_watchdog_ed25519` is now authorised on **192.168.22.232** as well, so `cam_ssh_ok` passes for the kitchen and the watchdog's camera-side steps (restart `majestic`, then reboot) are reachable there for the first time. `dropbear`'s `authorized_keys` is the file — the key was installed with paramiko over the password `root`/`2441`, since `sshpass` is not on this host.
- **`DRY_RUN` MUST BE SILENT ABOUT NOTHING, OR IT VERIFIES NOTHING.** It logged `would: docker restart` and `would: re-register`, but the two steps that touch the **physical camera** — restart `majestic` and reboot — were called through `[ "$DRY_RUN" = "1" ] ||` and logged nothing. So the single capability worth proving (restarting `majestic` on a room that had just been granted its SSH key) left no evidence in the dry run. Both now go through `act()`. Verified: the full four-step ladder prints, the container's `StartedAt` does not move, and both cameras' uptimes keep climbing (132 and 42 min) instead of resetting — which is the only proof that a reboot did not happen.

## REST API

| Path | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI dashboard |
| `/api/devices` | GET | List active WebSocket sessions |
| `/api/devices/config` | GET/POST | Device DB from `devices.json` |
| `/api/devices/config/{mac}` | PUT/DELETE | Update / remove device |
| `/mcp/{session_id}` | POST | Send MCP command to device (`"latest"` for most recent session) |
| `/api/tts` | POST | Speak `text` on a connected device (`session_id: "latest"` = most recent; ownership-checked) |
| `/api/camera/tts` | POST | Speak a phrase on a camera session |
| `/api/firmware/upload` | POST | Upload an ESP32 `.bin` (multipart, size-capped by `MAX_FIRMWARE_SIZE`) + update `firmware.json` |
| `/api/firmware` | GET | Current firmware metadata |
| `/ota` | GET/POST | ESP32 OTA handshake; returns WS URL + firmware info |
| `/health` | GET | Liveness probe, no auth |

## Config

`config/devices.json` — MAC-keyed device DB. Lookup is case-insensitive via `device-id` header (fallback: `mac` header).

## Key env vars

| Variable | Default | Notes |
|---|---|---|
| `NANOBOT_WS_URL` | `ws://nanobot:8765/` | Trailing `/` stripped, `?token=` appended |
| `CAMERA_STREAMS` | `""` | Comma-separated go2rtc stream names |
| `GO2RTC_HOST` / `GO2RTC_PORT` | `192.168.22.102` / `1984` | go2rtc control API |
| `VAD_SILENCE_FRAMES` | `8` | ESP32 path only; camera path hardcodes 10 |
| `WATCHDOG_TIMEOUT` | `90` | Seconds before fallback TTS |
| `LLM_BACKEND` | `nanobot` | `nanobot` \| `hermes` \| `cascade` |
| `TTS_VOICE` | `ru-RU-SvetlanaNeural` | |
| `TTS_MODEL` | `tts-1` | Model field posted to the TTS endpoint |
| `WAKE_WORD` | `компьютер` | Spoken phrase used by the command stripper |
| `TELEGRAM_BOT_TOKEN` | `""` | Empty = Telegram source off. Token + ids live in `.env.voice_gateway` (compose `env_file`, gitignored) — never in the repo |
| `TELEGRAM_ALLOWED_CHAT_IDS` | `""` | Comma-separated chat ids; **empty = nobody is served** (fail-closed) |
| `TELEGRAM_REPLY_VOICE` | `true` | Also reply as a voice note (TTS → ffmpeg ogg/opus) |
| `TELEGRAM_MAX_VOICE_S` / `TELEGRAM_TURN_TIMEOUT` / `TELEGRAM_COOLDOWN_S` | `60` / `120` / `1.5` | Voice length cap, per-turn cap, spacing between turn starts in one chat |
| `WAKE_VOSK_MODEL_<NAME>` | `""` | Per-room vosk model dir. Set => that room DECODES the wake word and the acoustic head is skipped. Empty => openWakeWord (corridor/kitchen) |
| `CAMERA_FOLLOWUP[_<NAME>]` | `question` | Open a free-form listening window after our OWN reply so a follow-up needs no wake word. **`none` \| `question` (default) \| `all`.** The default listens only after a QUESTION — an answer to our own question is the one follow-up that needs no keyword; `all` also opens a 10 s window after a statement, which was measured admitting television while refusing the user. A mode, not a duration, because `0` means "unset" and would produce a 30 s window instead of none |
| `CAMERA_PAUSE_ENDPOINT[_<NAME>]` | `false` | Per-room level-based utterance endpointing (removes the 7 s wait). **Off by default** — it changes WHEN a command is dispatched; enable only after reading that room's `UTTERANCE END` lines on real audio |
| `CAMERA_WEBRTC[_<NAME>]` | `true` | Open the WebRTC session at all. Set `false` for a room that has `/play_audio`: the session feeds the VAD nothing (audio comes from the RTSP loop) and delivers replies nowhere we read, but every `webrtc/offer` makes go2rtc rebuild the producer, and an abandoned one fills the **camera's** send queue and blocks majestic (measured 05.10.2026). Keep `true` where the sendonly track is the only playback path |
| `CAMERA_WAKE_MIN_PEAK[_<NAME>]` | `0` (=> 3000) | Raw peak a decoded wake word must reach in its own chunk, or the wake is refused as a decoder artefact on silence. Only ever evaluated on a chunk the decoder ALREADY fired on, so it cannot suppress a wake — only refuse one. `VOSK WAKE` logs `peak=`/`min_peak=` so the number is measured, not assumed |
| `CAMERA_TTS_TARGET_PEAK[_<NAME>]` | `0` (=> 20000) | Peak a reply is normalised to before the camera plays it, per room. 0 => built-in 20000 (61 % of full scale). Exists for the **unresolved** crackling playback report of 05.10.2026 — the gateway's own audio is measured clean, so the camera's output stage is the remaining suspect and the level has to be found by ear. Applies in **both** directions (dead band 0.85..1.2), so lowering it is actually audible |

## Gotchas

- `engine.last_score` must be written by `check_wakeword()` — camera_client reads it after every call; a missing write silently kills all wake detection (this exact bug shipped once)
- openwakeword needs raw int16; floats quantize to {-1,0,1} and break the model
- Wake suppressed first 5 s after an RTSP audio (re)connect — ffmpeg startup transient scores 0.7–0.96 at idle
- Confirmed mic echoes extend `_wake_suppress_until` by +20 s each; the RTSP backchannel returns played audio 3–40 s late
- Whisper form must include `model` field (`koekaverna/faster-whisper-podlodka-turbo`)
- Corridor mic clips at close range (peak 32k) — clipped speech mangles oww scores; bang gate ignores peak >24k unless score ≥0.85
- Robot vacuum / hood noise keeps bg median high → appliance hold suppresses wakes below score 0.60 in that room, but a held 0.55–0.59 still reaches STT confirm instead of being dropped
- A gate's STT-rescue band must sit *below* its guard: appliance `[0.55, 0.60)`, quiet-source `[tier−0.05, tier)`. Both bands were killed once by a recalibration that lowered the guard but left the rescue threshold behind (the branches tested `≥0.85` / `≥0.60` under `<0.60` / `<0.55/0.58` guards) — move them together, and mind that `_open_stt_confirm()` now ignores a call while a window is open so two gates hitting their bands on one chunk cannot stack expiry watchers.
- go2rtc 1.9.2 leaks zombie RTSP sessions under slow links; cameras run `/etc/watchdog_majestic.sh` via crond (restart majestic when :554 dead or send-queues pile up)
- Wake-word retrain attempts v2–v4 all degraded discrimination — keep `model_stream.npz` and `export_onnx.py`; v1 backups: `computer.onnx.bak_v1` (current), `.bak_v4` (failed)
- Dockerfile exposes 8080 but nothing listens on it
- HA's matcher accepts the CONCATENATED friendly name only for some entities (`corridor1_light_switch Relay` and `coffemaker` → `MatchFailedReason.NAME`, `entrance_light_switch Relay` → matches): pin the ENTITY ID into the `name` slot, both in `_execute_action` and in `ha_action`'s retry. `NAME` is also a fallback trigger for `ha_action` now — a phrase or a friendly name is a phrasing problem the resolver can settle, while an invented word resolves to nothing and keeps the original error.
- Ordinals pick a device by a digit in its entity id — only devices that HAVE one (corridor1/corridor2). `match_states` narrows only when the digit and the room both match and keeps the unfiltered hits otherwise (a read must not go empty); the ACTION path refuses instead. `ordinal_digit` is exact-token on purpose: the ENDINGS tuple is what stops «вторник» from selecting digit 2.
- Voice exposure is stored in the HA **entity registry** (`options.conversation.should_expose`), not in git: an unexposed relay answers `MatchFailedReason.ASSISTANT` no matter how correct the match is. The three light relays (`switch.entrance_light_switch_relay`, `corridor1`, `corridor2`) were opened on 02.10.2026; `ha_read` only sees exposed entities too.
- Code and comments are in English (Russian string literals kept for TTS/STT data)
