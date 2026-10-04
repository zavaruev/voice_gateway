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
- **Media on/off is a transport call, not an intent** (03.10.2026 18:21): «включи коди» → `intent__HassTurnOn` answered `MatchFailedReason.INVALID_AREA` for the RU room and then `MatchFailedReason.ASSISTANT` for `name='LE-zal'` — the MCP server has no media intent and the Kodi entities are NOT exposed to Assist, so that path can never reach them. `MEDIA_POWER_ON/OFF` send `media_play`/`media_stop` instead, which ignore Assist exposure; `homeassistant.turn_off` is deliberately NOT used (the box would sleep and no voice command could wake it). The media nouns also cover CONTENT words («серию», «эпизод», «передачу») — that is how the user asks for a player.
- **Every escalation carries the canonical room name** (`resolver.area_of` → `ctx`): «гостиная» is not an HA area, and an intent fed the RU word answers INVALID_AREA, which cost the model a step it then spent guessing device names. The classifier's media verb list, the resolver's `RE_MEDIA` and `_regex_action()` all have to agree — the last one is the SINGLE definition of "action-like", reused by the cold-embedder path.
- **The classifier warm-up is CHUNKED and retried**: embedding all 60 route utterances in one request took **102 s** on this host (≈1.7 s each) against a 30 s client timeout, so the warm-up failed on EVERY start — `classifier warmup failed: ` with an EMPTY message (a timeout, logged without its type). `EMBED_CHUNK = 8` keeps a request near 14 s, `warmup()` retries once for the cold-model case, and a cold embedder no longer kills the pure-regex action path (`_regex_action`), which needs no cosine at all — one cold Ollama used to escalate every voice command in the house.
- **An honest refusal beats an invented one** (03.10.2026): the model answered «не смог связаться с телевизором» when the real blocker was «такого сериала нет в библиотеке» — a connection diagnosis it had no evidence for. Two rules keep that class out: `media_search`/`media_play` REPORT the library state instead of guessing it, and the TASK_TEMPLATE forbids inventing a reason when a tool has already said what is missing. A refusal is still the right answer when a box is offline (`«ящик во владиной комнате не отвечает»`), which is a fact — unlike «не смог связаться», which was not.
- **Kodi library search & play by title** (`kodi.py`, 03.10.2026): HA exposes no media intent and no library search, so «включи серию "Темного зеркала"» was impossible and the model invented a reason («не смог связаться с телевизором»). `media_search(title, area)` returns the whole catalogue (so the MODEL bridges «Симпсоны»→«The Simpsons» — no string matcher can) and `media_play(title, area, episode)` opens a library item. Details that cost real time: the boxes are identified by their OWN zeroconf name (`System.FriendlyName` → "Kodi (LE-zal)" == HA friendly_name), so `KODI_HOSTS` is a seed list, not a room table. CREDENTIALS ARE PER BOX and the seed format is `[user:pass@]host`: three boxes answer to kodi/2441 and the fourth — 192.168.22.176, found by DNS `le-vlada.local` because a plain /24 sweep misses a box that 401s — to kodi/kodi. Two measured failure modes: probing it with the shared password reported «ящик не отвечает» (a wrong diagnosis, not an outage), and a probe row that dropped its seed answered 401 on the next call, so `probe_boxes` carries `seed` in every row; `VideoLibrary.Search` does not exist on Kodi 21.3 (use `GetTvshows`/`GetEpisodes`); `Player.Open` takes `{"item": {"episodeid": N}}` — `{"item_id":…, "video":0}` answers "Too many parameters"; the Jellyfin plugin needs 8-10 s to start, and the verification waits for the STATE to become `playing` rather than for a field to change — re-opening the episode a box is already playing changes nothing and was reported as «не смог открыть файл» for a show that WAS playing (19:55). `media_search` lists the whole catalogue but fetches episodes for the BEST MATCH only: one round trip per show cost 14 of them and 18-22 s per turn. The boxes are probed AND scanned in PARALLEL (`ThreadPoolExecutor`): sequential probing charged every media turn a full timeout for each box that was switched off (measured: probing four healthy boxes 0.04 s parallel), and `probe_one` must wrap its single `friendly_name()` round trip completely — an unguarded second call let one half-dead box raise straight through `boxes()` and break the whole media path. `KodiError.kind` keeps an HTTP 401 («неверный логин или пароль») apart from an outage («не отвечает»): they need opposite fixes, and calling a live box «не отвечает» is what sent the search after the wrong thing today. A library start is therefore 13-18 s end to end, most of it LLM steps; a NAMED episode that is not in the library is refused with its real range instead of a silent substitution; «следующая серия» means the first `playcount == 0`, not the first row. Credentials/hosts live in the gitignored `.env.kodi` (`env_file` on the worker).
- **Assist exposure for media lives in the entity registry**, and HA 2026.9's websocket schema is unusual: `config/entity_registry/update` takes `entity_id` = the ENTITY (`media_player.le_zal_2`, the row id answers "is an invalid entity ID") and `options` ONLY together with `options_domain` ("some but not all values in the same group of inclusion"); the payload is `options: {"should_expose": true}` + `options_domain: "conversation"`, and passing the scope twice stores `{"conversation": {"conversation": …}}`. All four Kodi boxes were opened this way (they answered `MatchFailedReason.ASSISTANT` before), so the intent path works too — but pause/volume/next-track still go through REST, which ignores exposure.
- **A latin room word must be matched FORWARD only** (`area_matchers`): «Living Room» tokenises to «living» + «room», and the Russian-style reverse test (`tok in stem`) made «room» a substring of the stems «bedroom»/«bathroom», so the area became {living, bedroom, bathroom}, the exact-room rule preferred the phantom «Bedroom», and `media_play(area="Living Room")` started the show on the BEDROOM box (field check 03.10.2026 19:32). Two rules then pick the room, in this order and identically in both services: the player's area carrying the MOST named room words («Bedroom Vlada» over «Bedroom»), then an exactly named room over a merely containing one.
- **`_area_phrase` must not fall through to a generic declension** (04.10.2026): the resolver hands over the CANONICAL HA area name («Bedroom»), so «В спальнее» never actually reached the user — but the helper is also called with a Russian display name and with whatever the caller carries, and its fallback turned «спальне» into «В спальнее» silently. `_prep_candidates()` now derives the nominative from the oblique («гостиной»→«гостиная», «улице»→«улица», «прихожей»→«прихожая») and looks THAT up, so an unknown word still falls back instead of inventing a room.
- **Hermes is diagnostics only**: `hermes_expert` is a bare `/v1/chat/completions` (no tools, 900 tokens, 25 s). It answers «почему сломалось», it cannot execute a command — TASK_TEMPLATE says so, so a media/vacuum/light failure is never handed to it (the Kodi turn that started this was answered by the gateway itself, 0.3 s).

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
| `CAMERA_PAUSE_ENDPOINT[_<NAME>]` | `false` | Per-room level-based utterance endpointing (removes the 7 s wait). **Off by default** — it changes WHEN a command is dispatched; enable only after reading that room's `UTTERANCE END` lines on real audio |

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
