# Voice Gateway

WebSocket gateway bridging [Xiaozhi ESP32](https://github.com/78/xiaozhi-esp32) smart speakers to the [Nanobot](https://github.com/HKUDS/nanobot) AI agent with real-time speech processing.

## Pipeline

```
ESP32 ──Opus frames──▶ Voice Gateway ──STT text──▶ Nanobot AI Agent
                         │                             │
                    Silero VAD                    Edge TTS
                    Speaker ID                    emotions
                    noise gate                    MCP commands
                         │                             │
                         ◀─────────Opus frames──────────
```

1. **ESP32** sends Opus audio frames over WebSocket
2. **Silero VAD** detects speech, noise reduction filters background
3. **Whisper STT** transcribes audio to text (Russian)
4. **Speaker ID** identifies the speaker
5. **Nanobot** processes text, returns response with optional emotions
6. **Edge TTS** synthesizes speech, streams Opus back to ESP32

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

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NANOBOT_WS_URL` | `ws://nanobot:8765/` | Nanobot AI agent WebSocket URL |
| `WHISPER_URL` | `http://192.168.22.111:8000/v1/audio/transcriptions` | OpenAI-compatible STT endpoint |
| `TTS_URL` | `http://edge_tts:5050/v1/audio/speech` | OpenAI-compatible TTS endpoint |
| `TTS_VOICE` | `ru-RU-SvetlanaNeural` | TTS voice identifier |
| `SPEAKER_ID_URL` | `http://192.168.22.102:8001/identify` | Speaker recognition ([speaker-id](https://github.com/zavaruev/speaker-id) container) |
| `VAD_SILENCE_FRAMES` | `8` | Silence frames before processing (~60ms each) |
| `STANDBY_TIMEOUT_QUESTION` | `30` | Seconds before standby after a question |
| `STANDBY_TIMEOUT_STATEMENT` | `10` | Seconds before standby after a statement |
| `WATCHDOG_TIMEOUT` | `30` | AI response timeout before fallback TTS |
| `CHAT_ID_TTL` | `604800` | Chat context lifetime in seconds (7 days) |
| `ENERGY_THRESHOLD` | `0.008` | Energy gate threshold for noise reduction |
| `MIN_SPEECH_RATIO` | `0.15` | Minimum speech ratio to trigger processing |
| `VAD_ADAPTIVE` | `true` | Enable adaptive VAD threshold |
| `THINKING_SOUND_PATH` | `""` | Path to thinking indicator sound |

## API

| Path | Method | Description |
|---|---|---|
| `/` | GET | Web UI dashboard |
| `/ws` | WebSocket | Main device gateway |
| `/api/devices` | GET | Active WebSocket sessions |
| `/api/devices/config` | GET | Registered device list |
| `/api/devices/config` | POST | Register a device |
| `/api/devices/config/{mac}` | PUT | Update device settings |
| `/api/devices/config/{mac}` | DELETE | Remove device |
| `/mcp/{session_id}` | POST | Send MCP command to device |
| `/ota` | GET/POST | ESP32 OTA handshake |

## Architecture

### STT Pipeline
Opus frames → VAD (Silero) → energy gate → noise reduction → Ogg → Whisper STT + Speaker ID (parallel)

### TTS Pipeline
Nanobot text → Edge TTS (MP3) → decode → resample 24kHz → Opus encode → stream 60ms chunks

### Dialogue Mode
- AI response ends with `?` → mic stays open
- AI response is a statement → returns to standby
- Hold phrases (`подожди`, `секундочку`) extend listening

### MCP Hardware Control
ESP32 exposes tools (screen, volume, LEDs) via MCP. Gateway forwards them to Nanobot as `tools_update`. Nanobot can call `self.screen.set_brightness`, `self.audio_speaker.set_volume`, etc.

### Activity Monitor
- After 30s idle: dim screen to 25%
- After 45s idle in LISTENING: close abandoned session
- Freezes timer while speaker is thinking/speaking

### Watchdog
If Nanobot doesn't respond within 30s → fallback TTS "Простите, я задумалась. Повторите пожалуйста." → mic reopens.

### Binary Frame Formats
| Version | Format |
|---|---|
| v1 | Raw Opus frame |
| v2 | 16-byte header + Opus frame |
| v3 | 4-byte header + Opus frame |

## Device Configuration

Create `config/devices.json`:

```json
{
  "aa:bb:cc:dd:ee:ff": {
    "friendly_name": "Kitchen Speaker",
    "allowed": true
  }
}
```

Device identifies itself via `device-id` header (fallback: `mac` header).

## Dependencies

- Python 3.12+
- FFmpeg (for audio processing)
- Silero VAD ONNX model (downloaded at build time)
- External services: Whisper STT, Nanobot, Edge TTS, Speaker ID
