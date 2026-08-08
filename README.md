# Voice Gateway

WebSocket gateway bridging [Xiaozhi ESP32](https://github.com/78/xiaozhi-esp32) smart speakers **and WebRTC/IP cameras** to the [Nanobot](https://github.com/HKUDS/nanobot) AI agent with real-time speech processing.

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
6. **Nanobot** processes text, returns response with optional emotions `[emotion_name]`
7. **Edge TTS** synthesizes speech; Opus streamed to ESP32, PCM queued to camera speakers

## Features

- **ESP32 + camera support** — both device classes in one gateway; cameras use WebRTC (go2rtc) with the mic feed taken from the RTSP audio backchannel (`ffmpeg`, raw L16, no μ-law quantisation)
- **Wake word «компьютер»** — custom-trained openWakeWord model (`config/computer.onnx` + `config/embedding_model.onnx` backbone), detected on the live 16 kHz camera feed, threshold 0.25, 60 s wake window, attention beep
- **Server-side VAD** — Silero ONNX + energy fallback; adaptive background floor
- **Distant-speech tuned** — corridor/lobby coverage: lowered RMS gates, SpeexDSP noise suppression with "NS rescue" (utterance salvaged from the noise floor), adaptive peak normalisation (up to 20x)
- **Echo guards** — global `GLOBAL_TTS_UNTIL` gate (no STT while any TTS plays — kills ESP32↔camera echo cascade) + duration-proportional mic hold (0.3 s beep → ~0.6 s hold, 20 s TTS → capped 15 s)
- **Dialogue mode** — mic stays open after questions (`?` anywhere in the reply, question words, imperative verbs); returns to standby after statements
- **MCP hardware control** — camera/ESP32 tools (screen, volume, LEDs) forwarded to Nanobot as `tools_update`
- **Watchdog** — 30 s timeout → fallback TTS «Простите, я задумалась. Повторите пожалуйста.»
- **Activity monitor** — dims screen to 25% after 30 s idle, closes abandoned sessions after 45 s in LISTENING
- **OTA** — ESP32 firmware handshake returning WS URL + firmware info
- **Emotions** — extracted from Nanobot text via `[emotion_name]` regex

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

ESP32 connects to `ws://gateway:18792/` with header `device-id: <MAC>`.

### Cameras

Set `CAMERA_STREAMS` to a comma-separated list of stream names registered in go2rtc. The gateway pulls each stream's audio from `rtsp://<GO2RTC_HOST>:8554/<name>?audio=copy` and writes replies back to the camera's WebRTC audio track.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NANOBOT_WS_URL` | `ws://nanobot:8765/` | Nanobot AI agent WebSocket URL |
| `NANOBOT_TOKEN` | `token` | Token appended to Nanobot WS URL |
| `WHISPER_URL` | `http://192.168.22.111:8000/v1/audio/transcriptions` | OpenAI-compatible STT endpoint |
| `TTS_URL` | `http://edge_tts:5050/v1/audio/speech` | OpenAI-compatible TTS endpoint |
| `TTS_VOICE` | `ru-RU-SvetlanaNeural` | TTS voice identifier |
| `TTS_API_KEY` | `""` | Sends `Authorization: Bearer` if set |
| `SPEAKER_ID_URL` | `http://192.168.22.102:8001/identify` | Speaker recognition ([speaker-id](https://github.com/zavaruev/speaker-id) container) |
| `CAMERA_STREAMS` | `""` | Comma-separated go2rtc stream names to attach to |
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
| `/ws` | WebSocket | Main device gateway |
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
RTSP (raw L16 16 kHz) → echo guard → VAD + RMS gates → SpeexDSP NS rescue → adaptive normalisation → Whisper (+ Speaker ID via Ogg) → Nanobot

### Wake word
openWakeWord `kompyuter` model (custom-trained, ONNX) on the live 16 kHz feed, echo-guarded. Score threshold 0.25 → 60 s wake window with attention beep, then follow-up commands without re-triggering.

### TTS Pipeline
Nanobot text → Edge TTS (MP3) → decode → resample (16/24 kHz) → Opus for ESP32 / PCM for cameras → paced 60 ms chunks

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

Device identifies itself via `device-id` header (fallback: `mac` header), case-insensitive. Duplicate MACs with varying case exist in the DB.

## Known Issues & Current Problems

- **Echo cascade ESP32 ↔ camera is only partially solved** — `GLOBAL_TTS_UNTIL` + duration-proportional mic hold are band-aids. Camera speaker echo returns via WebRTC with 8–15 s delay in ~3 s bursts; a bad guess on the hold time either swallows the user's follow-up question or lets echo through. A proper AEC is the real fix.
- **Wake word not validated in the field** — `config/computer.onnx` was trained synthetically (openWakeWord pipeline) but never tested on the real corridor microphone. Threshold 0.25 is a guess; expect false positives/negatives until tuned. `models/kompyuter.onnx` is a 0-byte leftover.
- **Vosk models downloaded but unused** — `config/vosk-model-ru-0.42/` (3.5 GB) and `config/vosk-model-small-ru-0.22/` (88 MB) are not wired into the pipeline (gitignored, not in the image). Candidate for a low-latency local STT fallback.
- **Gates are corridor-calibrated** — RMS/VAD thresholds and NS rescue were tuned for one location (AGC-boosted noise floor ~0.015–0.042, distant speech ~0.03–0.12). They will not generalise to other rooms without recalibration.
- **Whisper retry at temperature 0.5** — empty/`unknown` transcripts trigger a noisier re-transcription; on some engines this doubles STT latency in the worst case.
- **`Dockerfile` exposes 8080 but nothing listens on it** (18792 is the only real port).
- **`setup_gateway.sh` is an outdated snapshot** — not authoritative.
- **`config/` once had its own `.git` (no commits)** — parent repo now version-controls the config files.
- **`main.py.bak-20260807`** — leftover backup in the working tree (gitignored, not committed).

## Dependencies

- Python 3.12+
- FFmpeg (RTSP capture, WAV/OGG conversion)
- Silero VAD ONNX model (downloaded at build time)
- openWakeWord + custom `computer.onnx` / `embedding_model.onnx` (baked into the image)
- SpeexDSP noise suppression (`speexdsp-ns`)
- External services: Whisper STT, Nanobot, Edge TTS, Speaker ID, go2rtc (for cameras)
