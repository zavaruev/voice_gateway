"""Local audio engine: VAD, wake word, utterance state machine, STT.

PURPOSE
    Front half of the voice pipeline. Feeds it raw PCM (int16, 16 kHz,
    mono, one chunk at a time) and it decides: is this speech (Silero VAD
    via ONNX), was it addressed to us (openWakeWord), when does an
    utterance start/end (CameraProcessor state machine), and finally it
    sends the recorded WAV to the STT HTTP endpoint and hands the
    transcript to main.py through callbacks.

ROLE IN THE CASCADE (three-level voice control of the smart home)
    ESP32 satellite -> main.py (FastAPI :6050, audio/VAD/STT/TTS) ->
    jev-router (L1 :8091) -> smolagents-worker (L2 :8092) -> Hermes (L3)
    -> Home Assistant. This module produces the TRANSCRIPT that feeds the
    whole chain; everything after STT (routing, agent, tools) happens
    elsewhere. camera_client.py imports LocalAudioEngine/CameraProcessor/
    AgentState for the always-on camera microphones.

CONTRACTS
    LocalAudioEngine
      initialize_models(wakeword_path)  load ONNX models once at startup
      check_vad(audio_int16) -> bool    per-chunk speech verdict
      check_wakeword(audio_int16, threshold, stream, vad_ok) -> bool
                                         wake verdict + self.last_score
      reset_vad()                       clear state between utterances
    CameraProcessor
      process_chunk(audio_int16)        async entry point, call per chunk
      on_activation_cb / on_command_cb  async callbacks wired by main.py
    STT: POST a WAV (16 kHz mono) to SPEECHES_STT_URL with a faster-whisper
    model + language=ru; any HTTP/transport failure returns "" and the
    chunk is treated as "nothing to do" — STT must never crash the loop.

DESIGN NOTES / WHY
    The two VAD paths keep SEPARATE recurrent state (utterance vs wake)
    because they see different chunk boundaries; the wake path resets its
    state on any feed gap >200 ms and must be fed RAW int16 (no gain, no
    per-chunk AGC) — see the measured numbers in _ww_vad_speech/ 
    check_wakeword, both failure modes were observed in the field.
    `SPEECHES_STT_URL` is a legacy hardcoded default; point deployments at
    an env-configured URL instead of committing new addresses (repo is
    public — no IPs/keys in code).
"""

import asyncio
import io
import logging
import time
import wave
from enum import Enum
import fractions
import aiohttp
import av
import numpy as np
import onnxruntime as ort
from openwakeword.model import Model
from aiortc import MediaStreamTrack


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | engine:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)


# Legacy hardcoded STT endpoint (Speeches-style faster-whisper API).
# Kept as-is for compatibility; deployments should point this at their own
# instance — never commit NEW addresses or credentials here (public repo).
SPEECHES_STT_URL = "http://192.168.22.111:8000/v1/audio/transcriptions"


class AgentState(Enum):
    """Utterance lifecycle of one microphone session.
    LISTENING  waiting for a (wake) trigger;
    RECORDING  buffering speech until end-of-utterance silence;
    PROCESSING STT + LLM dispatch in flight;
    SPEAKING   TTS is playing — input chunks are ignored (echo guard)."""

    LISTENING = 1
    RECORDING = 2
    PROCESSING = 3
    SPEAKING = 4


class AudioStreamTrack(MediaStreamTrack):
    """aiortc audio source backed by an asyncio queue.

    Consumers pull frames via recv(); when nothing is queued for >1 s the
    method synthesizes a silent 20 ms frame instead of blocking — a WebRTC
    receiver must keep seeing media or it tears the track down."""

    kind = "audio"

    def __init__(self):
        """Allocate the empty ONNX handle slots (session/model load in load())."""
        super().__init__()
        self._queue = asyncio.Queue()
        self._pts = 0

    async def recv(self):
        """Next frame, or silence (correct PTS/time_base) after 1 s of
        queue inactivity so the stream clock keeps advancing."""
        try:
            frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            return frame
        except asyncio.TimeoutError:
            frame = av.AudioFrame(format='s16', layout='mono', samples=960)
            frame.sample_rate = 48000
            frame.pts = self._pts
            frame.time_base = fractions.Fraction(1, 48000)
            frame.planes[0].update(b'\x00' * 1920)
            self._pts += 960
            return frame

    async def put_frame(self, frame: av.AudioFrame):
        """Enqueue one encoded frame for recv()."""
        await self._queue.put(frame)


class LocalAudioEngine:
    """Shared inference state for VAD + wake word (one instance per mic).

    Holds the Silero ONNX session, the openWakeWord model and the
    recurrent state of BOTH VAD paths. Not thread-safe by design: main/
    camera loops call it from a single worker thread per stream (the few
    asyncio callers use asyncio.to_thread to stay off the event loop)."""

    def __init__(self, vad_threshold: float = 0.5):
        """Create the ONNX session and default buffers.

        The actual Silero/openWakeWord weights are loaded lazily by
        load(); construction alone must stay cheap enough to call in tests.
        """
        self.vad_session = None
        self.oww_model = None
        self.vad_threshold = vad_threshold
        self._vad_buffer = np.array([], dtype=np.float32)
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._vad_calls = 0
        self._ww_calls = 0
        self.last_score = 0.0
        # Dedicated Silero state for the wake-word path (independent from the
        # utterance VAD above — the two consume different chunk boundaries).
        self._wwv_buffer = np.array([], dtype=np.float32)
        self._wwv_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._wwv_context = np.zeros((1, 64), dtype=np.float32)

    def initialize_models(self, wakeword_path: str = ""):
        """Load Silero VAD (+ optional openWakeWord) once at startup.

        VAD is loaded from `silero_vad.onnx` in the working directory;
        threads are pinned to 1 (the model is tiny — a thread pool would
        cost more than it saves on this box). `wakeword_path` selects a
        custom wake model with the shared embedding model, otherwise the
        built-in "alexa" model is used. Raises on missing model files —
        deliberately, so a bad deploy fails at boot, not mid-utterance."""
        logger.info("Loading Silero VAD (ONNX)...")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.vad_session = ort.InferenceSession(
            "silero_vad.onnx", sess_options=opts, providers=["CPUExecutionProvider"]
        )

        if wakeword_path:
            self.oww_model = Model(
                wakeword_model_paths=[wakeword_path],
                embedding_onnx_model_path="config/embedding_model.onnx",
            )
            logger.info(f"Loading openWakeWord model: {wakeword_path}")
        else:
            self.oww_model = Model()
            logger.info("Loading openWakeWord built-in models (alexa)...")
        logger.info("Local models initialized.")

    def reset_vad(self):
        """Clear utterance VAD buffer + recurrent state (call between
        utterances / on state transitions).

        Note: this also (re)creates `_vad_context`, which __init__ does
        NOT set — so reset_vad() must run before the first check_vad()
        call (CameraProcessor.set_state does it for LISTENING/SPEAKING).
        Stale LSTM state from the previous utterance would mis-score the
        first windows of the next one."""
        self._vad_buffer = np.array([], dtype=np.float32)
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._vad_context = np.zeros((1, 64), dtype=np.float32)

    def check_vad(self, audio_int16: np.ndarray) -> bool:
        """Speech verdict for one chunk (True = speech above threshold).

        The chunk is scaled to float32 (with a 32x gain for quiet mics,
        then hard-clipped to [-1,1] — safe for this utterance head; do NOT
        copy the gain into the wake path, see _ww_vad_speech), appended to
        the buffer and consumed in Silero's 512-sample windows. Returns
        whether the BEST window probability exceeds `vad_threshold`, i.e.
        speech anywhere in the chunk wins; the buffer keeps the ragged
        remainder until the next call.
        """
        self._vad_calls += 1
        gain = 32.0
        audio_float32 = audio_int16.astype(np.float32) * gain / 32768.0
        np.clip(audio_float32, -1.0, 1.0, out=audio_float32)
        self._vad_buffer = np.concatenate((self._vad_buffer, audio_float32))
        max_prob = 0.0
        while len(self._vad_buffer) >= 512:
            # Silero's native window at 16 kHz is 512 samples; the model
            # input is the previous 64-sample context prefixed to it, and
            # only the tail survives as the next context.
            chunk = self._vad_buffer[:512]
            self._vad_buffer = self._vad_buffer[512:]
            full_input = np.concatenate([self._vad_context, chunk[np.newaxis, :]], axis=1)
            out, self._vad_state = self.vad_session.run(
                None,
                {"input": full_input, "state": self._vad_state, "sr": np.array([16000], dtype=np.int64)},
            )
            self._vad_context = np.concatenate([self._vad_context, chunk[np.newaxis, :]], axis=1)[:, -64:]
            max_prob = max(max_prob, out[0][0])
        # Throttled telemetry: per-chunk logging would drown the gateway.
        if self._vad_calls % 200 == 0:
            level = np.sqrt(np.mean(audio_float32**2))
            logger.info(f"VAD calls={self._vad_calls} max_prob={max_prob:.4f} level={level:.6f} (threshold={self.vad_threshold})")
        return max_prob > self.vad_threshold

    def _ww_vad_speech(self, audio_int16: np.ndarray) -> bool:
        """Silero speech-probability for the wake path. Non-speech transients
        (hamster cage clicks, thuds, rustle) historically scored 0.6-0.9 on
        this TTS-trained head while being NOT speech at all; gating the model
        behind a real VAD removes that entire false-positive class."""
        # NO extra gain here: a 32x boost hard-clips loud speech into a
        # square wave and Silero's probability collapses to <0.25 even on
        # real commands (measured: rms 3463 clip -> max_prob 0.24; without
        # gain -> 1.0). Natural [-1,1] scaling keeps the VAD honest.
        # This feeder is NOT called for every chunk of wall-clock audio
        # (suppressed windows / dialogue mode skip whole stretches), so any
        # gap >200ms means the LSTM state no longer matches the stream —
        # measured effect: probabilities stuck at 0.00 for EVERYTHING.
        # Reset the recurrent state on gap; Silero re-warms within ~5 frames.
        now_t = time.time()
        if now_t - getattr(self, "_wwv_last_feed", 0.0) > 0.2:
            self._wwv_buffer = np.array([], dtype=np.float32)
            self._wwv_state = np.zeros((2, 1, 128), dtype=np.float32)
            self._wwv_context = np.zeros((1, 64), dtype=np.float32)
        self._wwv_last_feed = now_t
        f = audio_int16.astype(np.float32) / 32768.0
        self._wwv_buffer = np.concatenate((self._wwv_buffer, f))
        max_prob = 0.0
        while len(self._wwv_buffer) >= 512:
            chunk = self._wwv_buffer[:512]
            self._wwv_buffer = self._wwv_buffer[512:]
            full_input = np.concatenate(
                [self._wwv_context, chunk[np.newaxis, :]], axis=1
            )
            out, self._wwv_state = self.vad_session.run(
                None,
                {"input": full_input, "state": self._wwv_state,
                 "sr": np.array([16000], dtype=np.int64)},
            )
            self._wwv_context = np.concatenate(
                [self._wwv_context, chunk[np.newaxis, :]], axis=1
            )[:, -64:]
            max_prob = max(max_prob, out[0][0])
        self._wwv_calls = getattr(self, "_wwv_calls", 0) + 1
        if self._wwv_calls % 100 == 0:
            lvl = int(np.max(np.abs(audio_int16)))
            logger.info(
                f"[ww-vad] probe #{self._wwv_calls}: "
                f"prob={max_prob:.2f} peak={lvl} -> {'PASS' if max_prob > 0.5 else 'gate'}"
            )
        return max_prob > 0.5

    def check_wakeword(self, audio_int16: np.ndarray, threshold: float = 0.4, stream: str = "", vad_ok: bool | None = None) -> bool:
        """Wake-word verdict for one RAW int16 chunk.

        Params: `threshold` = per-stream sensitivity (distant microphones
        need tuning), `stream` = label used in logs only, `vad_ok` =
        caller's _ww_vad_speech verdict computed BEFORE any AGC — False
        short-circuits (non-speech transients are the dominant false-
        positive source), None keeps the legacy self-gating behaviour.
        Returns True when the best openWakeWord score beats `threshold`;
        the raw score is always mirrored to self.last_score, which
        camera_client's debounce/fire logic reads.
        """
        # NOTE: the caller should run _ww_vad_speech on the RAW chunk BEFORE
        # any AGC normalization and pass the verdict here. Per-chunk peak
        # normalization flattens amplitude dynamics and Silero goes blind on
        # the normalized signal (measured: raw feed -> 69 speech chunks;
        # per-chunk AGC -> 0). vad_ok=None keeps legacy self-gating behavior.
        if vad_ok is False:
            self.last_score = 0.0
            return False
        peak = int(np.max(np.abs(audio_int16)))
        if self._ww_calls % 500 == 0:
            rms = np.sqrt(np.mean(audio_int16.astype(np.float64)**2))
            logger.info(f"[{stream}] WW peek: peak={peak} rms={rms:.1f}")
        self._ww_calls += 1
        # openwakeword's melspec pipeline expects raw int16 samples
        # (its internal buffer is cast with .astype(np.int16)); feeding
        # normalized floats quantized everything to {-1,0,1} and broke
        # every model. The wake-word model was trained on int16 scale.
        prediction = self.oww_model.predict(audio_int16)
        score = float(max(prediction.values())) if prediction else 0.0
        # Expose the raw score for camera_client's debounce/fire logic. This
        # attribute was documented but never written — getattr fallback made
        # every chunk score 0.0 and the oww fire branch was unreachable.
        self.last_score = score
        # Tag scores with the camera so per-room sensitivity can be debugged
        # (kitchen's distant mic needs different treatment than corridor).
        if score > 0.4:
            logger.info(f"[{stream}] Wake word score: {score:.4f} (peak={peak})")
        return score > threshold


class CameraProcessor:
    """Per-microphone utterance state machine on top of LocalAudioEngine.

    Wires the engine to STT and to main.py: `on_activation_cb` fires the
    moment recording starts (play the attention cue / open the mic),
    `on_command_cb` receives the final transcript (returns a coroutine that
    is awaited, so the reply can drive TTS synchronously). End of
    utterance = SILENCE_TIMEOUT (1.5 s) of non-speech after the first
    second of audio. Chunks arriving in SPEAKING are dropped on purpose —
    the mic hears its own TTS otherwise."""

    def __init__(
        self,
        engine: LocalAudioEngine,
        on_command_cb=None,
        on_activation_cb=None,
        stt_url: str = SPEECHES_STT_URL
    ):
        """Bind the shared engine + callbacks; start in LISTENING with a
        1.5 s end-of-utterance silence timeout (the FSM's default).
        """
        self.engine = engine
        self.stt_url = stt_url
        self.on_command_cb = on_command_cb
        self.on_activation_cb = on_activation_cb

        self.state = AgentState.LISTENING
        self.record_buffer = []
        self.silence_start = None
        self.SILENCE_TIMEOUT = 1.5

        self._wake_hold_speech = 0
        self._record_start = 0.0

    def set_state(self, new_state: AgentState):
        """FSM transition + log. Entering LISTENING or SPEAKING resets the
        utterance VAD: a new utterance starts from clean recurrent state,
        and while TTS plays the VAD must not accumulate the echo."""
        logger.info(f"State transition: {self.state} -> {new_state}")
        self.state = new_state
        if new_state in (AgentState.LISTENING, AgentState.SPEAKING):
            self.engine.reset_vad()

    async def _send_to_speeches_stt(self, audio_pcm16: np.ndarray) -> str:
        """PCM16 (16 kHz mono) -> WAV -> multipart POST to the STT service.

        The WAV is built in a worker thread (wave encoding would stall
        the event loop). Returns the transcript, or "" on any HTTP or
        transport failure — empty text is handled by the caller as
        "nothing recognised", never as an error to the user.
        """
        def _write_wav():
            """PCM16 numpy -> WAV bytes (runs inside asyncio.to_thread)."""
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_pcm16.tobytes())
            return wav_io.getvalue()

        wav_bytes = await asyncio.to_thread(_write_wav)

        # OpenAI-style multipart transcription request; model + language
        # are pinned (household speaks Russian only) — changing either
        # changes STT quality for every satellite.
        form = aiohttp.FormData()
        form.add_field('file', wav_bytes, filename='speech.wav', content_type='audio/wav')
        form.add_field('model', 'koekaverna/faster-whisper-podlodka-turbo')
        form.add_field('language', 'ru')

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.stt_url, data=form, timeout=10.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("text", "").strip()
                    else:
                        err_text = await resp.text()
                        logger.error(f"Speeches STT error HTTP {resp.status}: {err_text}")
                        return ""
        except Exception as e:
            logger.error(f"Failed to connect to Speeches STT at {self.stt_url}: {e}")
            return ""

    async def _handle_stt_and_llm(self):
        """Transcribe the finished utterance and dispatch it.

        STATE/PROCESSING: the buffer is taken (and cleared) first so a
        late chunk cannot double-feed this run; then STT, then the
        `on_command_cb` pipeline (awaited — it drives the whole reply),
        or back to LISTENING when the transcript was empty or no callback
        is wired. Empty STT text is normal (noise, hall-effect echo) and
        is NOT reported to the user.
        """
        self.set_state(AgentState.PROCESSING)
        full_audio = np.concatenate(self.record_buffer)
        self.record_buffer.clear()

        logger.info(f"Sending audio to Speeches STT ({self.stt_url}), duration={len(full_audio)/16000:.1f}s")
        text = await self._send_to_speeches_stt(full_audio)
        logger.info(f"STT Result: '{text}'")

        if text and self.on_command_cb:
            await self.on_command_cb(text)
        else:
            self.set_state(AgentState.LISTENING)

    async def process_chunk(self, audio_int16: np.ndarray):
        """Entry point — feed one raw PCM chunk (int16, 16 kHz).

        LISTENING: 3 consecutive speech chunks are required to trigger
        (debounce against a single noisy window); a non-speech chunk
        decays the counter, so an interrupted trigger starts over.
        RECORDING: every chunk is buffered; the first second is exempt
        from the silence timer (a clipped initial word must not end the
        utterance), then 1.5 s of non-speech finishes it and hands the
        job to _handle_stt_and_llm (spawned, not awaited — this loop must
        return immediately or the mic would drop audio).
        SPEAKING: dropped entirely (echo guard). VAD runs in a worker
        thread (ONNX inference is blocking).
        """
        if self.state == AgentState.SPEAKING:
            return

        speech = await asyncio.to_thread(self.engine.check_vad, audio_int16)

        if self.state == AgentState.LISTENING:
            if speech:
                self._wake_hold_speech += 1
                if self._wake_hold_speech >= 3:
                    # Debounce: only SUSTAINED speech wakes us — a single
                    # VAD window can fire on a door click or a cough.
                    logger.info(f"VAD speech detected — waking")
                    logger.warning("VAD WAKE! Recording...")
                    self.set_state(AgentState.RECORDING)
                    self.record_buffer.clear()
                    self.silence_start = None
                    self._record_start = 0.0
                    if self.on_activation_cb:
                        await self.on_activation_cb()
            else:
                self._wake_hold_speech = max(0, self._wake_hold_speech - 1)

        elif self.state == AgentState.RECORDING:
            self.record_buffer.append(audio_int16)
            now = time.time()
            if self._record_start == 0.0:
                self._record_start = now
            elif speech:
                self.silence_start = None  # speech resumed: restart the timer
            else:
                elapsed = now - self._record_start
                if elapsed < 1.0:
                    pass  # grace: don't cut off a clipped first word
                elif self.silence_start is None:
                    self.silence_start = now
                elif now - self.silence_start > self.SILENCE_TIMEOUT:
                    # End of utterance: hand over to STT without awaiting
                    # (see process_chunk docstring).
                    logger.info(f"VAD silence ({len(self.record_buffer)} frames). Processing STT...")
                    asyncio.create_task(self._handle_stt_and_llm())
