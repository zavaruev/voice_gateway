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


SPEECHES_STT_URL = "http://192.168.22.111:8000/v1/audio/transcriptions"


class AgentState(Enum):
    LISTENING = 1
    RECORDING = 2
    PROCESSING = 3
    SPEAKING = 4


class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue()
        self._pts = 0

    async def recv(self):
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
        await self._queue.put(frame)


class LocalAudioEngine:
    def __init__(self, vad_threshold: float = 0.5):
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
            logger.info("Loading openWakeWord custom model...")
        else:
            self.oww_model = Model()
            logger.info("Loading openWakeWord built-in models (alexa)...")
        logger.info("Local models initialized.")

    def reset_vad(self):
        self._vad_buffer = np.array([], dtype=np.float32)
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._vad_context = np.zeros((1, 64), dtype=np.float32)

    def check_vad(self, audio_int16: np.ndarray) -> bool:
        self._vad_calls += 1
        gain = 32.0
        audio_float32 = audio_int16.astype(np.float32) * gain / 32768.0
        np.clip(audio_float32, -1.0, 1.0, out=audio_float32)
        self._vad_buffer = np.concatenate((self._vad_buffer, audio_float32))
        max_prob = 0.0
        while len(self._vad_buffer) >= 512:
            chunk = self._vad_buffer[:512]
            self._vad_buffer = self._vad_buffer[512:]
            full_input = np.concatenate([self._vad_context, chunk[np.newaxis, :]], axis=1)
            out, self._vad_state = self.vad_session.run(
                None,
                {"input": full_input, "state": self._vad_state, "sr": np.array([16000], dtype=np.int64)},
            )
            self._vad_context = np.concatenate([self._vad_context, chunk[np.newaxis, :]], axis=1)[:, -64:]
            max_prob = max(max_prob, out[0][0])
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
        return max_prob > 0.5

    def check_wakeword(self, audio_int16: np.ndarray, threshold: float = 0.4, stream: str = "") -> bool:
        peak = int(np.max(np.abs(audio_int16)))
        if not self._ww_vad_speech(audio_int16):
            # Not speech: never even show it to the wake model.
            self.last_score = 0.0
            return False
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
    def __init__(
        self,
        engine: LocalAudioEngine,
        on_command_cb=None,
        on_activation_cb=None,
        stt_url: str = SPEECHES_STT_URL
    ):
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
        logger.info(f"State transition: {self.state} -> {new_state}")
        self.state = new_state
        if new_state in (AgentState.LISTENING, AgentState.SPEAKING):
            self.engine.reset_vad()

    async def _send_to_speeches_stt(self, audio_pcm16: np.ndarray) -> str:
        def _write_wav():
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_pcm16.tobytes())
            return wav_io.getvalue()

        wav_bytes = await asyncio.to_thread(_write_wav)

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
        if self.state == AgentState.SPEAKING:
            return

        speech = await asyncio.to_thread(self.engine.check_vad, audio_int16)

        if self.state == AgentState.LISTENING:
            if speech:
                self._wake_hold_speech += 1
                if self._wake_hold_speech >= 3:
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
                self.silence_start = None
            else:
                elapsed = now - self._record_start
                if elapsed < 1.0:
                    pass
                elif self.silence_start is None:
                    self.silence_start = now
                elif now - self.silence_start > self.SILENCE_TIMEOUT:
                    logger.info(f"VAD silence ({len(self.record_buffer)} frames). Processing STT...")
                    asyncio.create_task(self._handle_stt_and_llm())
