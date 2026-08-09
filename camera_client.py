"""
Camera client module — bidirectional audio for IP cameras.
Audio from camera via go2rtc WebRTC (OPUS), TTS sent back via WebRTC track (PCMU).
"""

import asyncio
import fractions
import hashlib
import io
import json
import logging
import re
import struct
import time

import aiohttp
import aioice.ice
import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.rtcicetransport import RTCIceCandidate
from pydub import AudioSegment

from audio_utils import is_valid_text
from engine import LocalAudioEngine, CameraProcessor, AgentState

logger = logging.getLogger("camera")
logger.setLevel(logging.INFO)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
_r = logging.getLogger()
if not _r.handlers:
    _rh = logging.StreamHandler()
    _rh.setLevel(logging.DEBUG)
    _rh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _r.addHandler(_rh)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
logger.propagate = False


class AIVoiceOutputTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 8000):
        super().__init__()
        self._sample_rate = sample_rate
        # Буфер на ~10 секунд аудио. Если очередь заполнится, 
        # писатель будет ждать (backpressure), не теряя фреймы.
        self._queue: asyncio.Queue[av.AudioFrame] = asyncio.Queue(maxsize=500)
        self._timestamp = 0
        self._start: float = 0.0
        
        self._frame_samples = max(sample_rate * 20 // 1000, 160)
        silence_pcm = b'\x00\x00' * self._frame_samples
        self._silence_frame = self._pcm_to_frame(silence_pcm, sample_rate)
        self._frame_count = 0
        self._last_real_recv: float = 0.0
        # Duration (seconds) of the last queued playback; the mic echo-guard
        # holds closed only as long as the playback itself (beeps are short,
        # TTS replies long), instead of a blanket 10s that swallowed the
        # question spoken right after the wake beep.
        self._last_play_duration: float = 0.0

    def _pcm_to_frame(self, pcm: bytes, sample_rate: int) -> av.AudioFrame:
        samples = len(pcm) // 2
        frame = av.AudioFrame(
            format="s16",
            layout="mono",
            samples=samples,
        )
        frame.sample_rate = sample_rate
        frame.planes[0].update(pcm)
        return frame

    async def recv(self):
        try:
            frame = self._queue.get_nowait()
            self._frame_count += 1
            self._last_real_recv = time.time()
            if self._frame_count <= 3 or self._frame_count % 500 == 0:
                logger.info(f"AIVoiceOutputTrack: real frame #{self._frame_count}, queue={self._queue.qsize()}")
        except asyncio.QueueEmpty:
            frame = self._silence_frame

        if self._timestamp == 0:
            self._start = time.time()

        # Строгий тайминг выдачи фреймов для WebRTC (RTP clock)
        wait = self._start + (self._timestamp / self._sample_rate) - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._timestamp += frame.samples
        return frame

    async def queue_frame(self, pcm: bytes, sample_rate: int = 8000):
        # Используем await put, чтобы при больших TTS ответах трек не переполнялся 
        # и не дропал слова в середине предложения.
        frame = self._pcm_to_frame(pcm, sample_rate)
        await self._queue.put(frame)

    def queue_seconds(self) -> float:
        return self._queue.qsize() * (self._frame_samples / self._sample_rate)

    def echo_active(self) -> bool:
        if self._queue.qsize() > 0:
            return True
        # WebRTC/go2rtc buffers return the camera's own speaker echo back
        # with 8-15s delay, in ~3s bursts. Hold the mic closed only long
        # enough for the echo of WHAT PLAYED (plus a small tail) to finish
        # arriving: 0.3s beep -> ~0.6s, 20s TTS -> capped 15s.
        hold = min(15.0, self._last_play_duration + 0.3)
        return time.time() - self._last_real_recv < hold

    def stop(self):
        pass


_HAS_QUESTION_RE = re.compile(r"[?？]")

# Global: while any TTS playback is running (ESP32 or camera), neither path
# may send to nanobot — kills echo cascade (camera hears ESP32 TTS, resends).
GLOBAL_TTS_UNTIL = 0.0


class CameraSession:

    def __init__(
        self,
        stream_name: str,
        go2rtc_host: str = "192.168.22.102",
        go2rtc_port: int = 1984,
        http_session: aiohttp.ClientSession | None = None,
        chat_id: str | None = None,
        whisper_url: str = "http://192.168.22.111:8000/v1/audio/transcriptions",
        whisper_model: str = "koekaverna/faster-whisper-podlodka-turbo",
        tts_url: str = "http://edge_tts:5050/v1/audio/speech",
        tts_voice: str = "ru-RU-SvetlanaNeural",
        tts_api_key: str = "",
        nanobot_url: str = "ws://nanobot:8765/",
        nanobot_token: str = "token",
        speaker_id_url: str = "http://192.168.22.102:8001/identify",
        aec_block_ms: int = 1500,
        wake_keyword: str = "компьютер",
        wake_timeout: float = 60.0,
        vad: object = None,
        wakeword_model_path: str = "",
        activation_wav_path: str = "/app/activation.wav",
    ):
        self.stream_name = stream_name
        self.go2rtc_host = go2rtc_host
        self.go2rtc_port = go2rtc_port
        self.chat_id = chat_id or self._make_chat_id(stream_name)
        self.http_session = http_session
        self.whisper_url = whisper_url
        self.whisper_model = whisper_model
        self.tts_url = tts_url
        self.tts_voice = tts_voice
        self.tts_api_key = tts_api_key
        self._nanobot_url = nanobot_url.rstrip("/")
        self._nanobot_token = nanobot_token
        self._speaker_id_url = speaker_id_url
        self.aec_block_ms = aec_block_ms
        self.wake_keyword = wake_keyword
        self._wake_timeout = wake_timeout
        self._wake_detected = False
        self._wake_expires = 0.0
        self._vad = vad
        if self._vad is not None:
            self._vad.energy_threshold = 0.005
            self._vad.rms_noise_floor = 0.005
            self._vad.rms_alpha = 0.0
        self._wakeword_model_path = wakeword_model_path or ""
        self._activation_wav_path = activation_wav_path

        self._pc: RTCPeerConnection | None = None
        self._out_track: AIVoiceOutputTrack | None = None
        self._last_transcript: str = ""
        self._speaking = False
        self._speaking_until = 0.0

        # Статистика и таймеры
        self._last_attention = 0.0
        self._last_feed_log = 0.0
        self._last_rms_log = 0.0
        self._bg_window: list[int] = []
        self._resample_buf = b""

        self._engine = LocalAudioEngine(vad_threshold=0.03)
        self._processor = CameraProcessor(
            self._engine,
            on_command_cb=self._on_user_command,
            on_activation_cb=self._play_activation_sound,
        )

        self._vad_buf = bytearray()
        self._vad_speech_buf = bytearray()
        self._vad_silence_frames = 0
        self._vad_has_speech = False
        self._vad_speech_consecutive = 0
        self._vad_window: list[int] = []
        self._vad_silence_limit = 15
        self._vad_max_duration = 3.0
        self._vad_start_time = 0.0
        self._vad_lock = asyncio.Lock()
        self._processing_utterance = False

        self._tasks: set[asyncio.Task] = set()
        self._stopped = asyncio.Event()

        self._attention_played = False

    @staticmethod
    def _make_chat_id(name: str) -> str:
        h = hashlib.sha256(name.lower().encode()).hexdigest()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    async def start(self):
        self._stopped.clear()
        await self._init_engine()
        task = asyncio.create_task(self._run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        rtsp_task = asyncio.create_task(self._rtsp_audio_loop())
        self._tasks.add(rtsp_task)
        rtsp_task.add_done_callback(self._tasks.discard)
        logger.info(f"[{self.stream_name}] CameraSession started")

    async def _init_engine(self):
        try:
            await asyncio.to_thread(self._engine.initialize_models, self._wakeword_model_path or "config/computer.onnx")
        except Exception as e:
            logger.warning(f"[{self.stream_name}] Failed to init audio engine: {e}")

    async def stop(self):
        self._stopped.set()
        for t in list(self._tasks):
            t.cancel()
        await self._cleanup()
        logger.info(f"[{self.stream_name}] CameraSession stopped")

    async def _cleanup(self):
        pc = self._pc
        self._pc = None
        if pc:
            try:
                await pc.close()
            except Exception:
                pass

    async def _run(self):
        while not self._stopped.is_set():
            try:
                await self._connect()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"[{self.stream_name}] {exc}", exc_info=True)
                await asyncio.sleep(5)

    @staticmethod
    def _filter_sdp(sdp: str, drop_ip: str = "192.168.22.250") -> str:
        """Drop candidates on the macvlan host alias .250.

        go2rtc answers from .250:8555 and picks a host pair on .250 if the
        offer advertises one; packets then traverse the macvlan host path
        and get lost (all-zero windows). Keeping .250 out of the offer makes
        go2rtc send to the peer-reflexive .102 address instead.
        """
        lines = sdp.split("\n")
        out: list[str] = []
        dropped = 0
        for line in lines:
            if line.startswith("a=candidate:") and drop_ip in line:
                logger.debug(f"filter_sdp: dropping {line.strip()[:80]}")
                dropped += 1
            else:
                out.append(line)
        if dropped:
            logger.info(f"filter_sdp: dropped {dropped} candidate(s) on {drop_ip}")
        return "\n".join(out)

    @staticmethod
    def _filter_offer_sdp(sdp: str) -> str:
        """Remove PCMU/PCMA codecs from audio media lines in offer SDP.
        
        Camera exposes two audio media lines: opus/48000/2 (mic) and PCMU/8000
        (backchannel speaker). go2rtc can mismatch them, causing Python to receive
        μ-law data instead of opus. Remove PCMU/PCMA from the offer so go2rtc
        can only negotiate opus for audio.
        """
        lines = sdp.split("\n")
        out: list[str] = []
        in_audio = False
        removed_pts: set[str] = set()
        audio_line_indices: list[int] = []
        for i, line in enumerate(lines):
            if line.startswith("m=audio"):
                in_audio = True
                audio_line_indices.append(len(out))
                out.append(line)
                continue
            if in_audio:
                if line.startswith("m="):
                    in_audio = False
                    out.append(line)
                    continue
                low = line.lower()
                if "pcmu" in low or "pcma" in low or "pcml" in low:
                    # Keep L16 (raw PCM mic, corridor) — dropping it forces
                    # go2rtc onto the PCMU backchannel line which adds
                    # μ-law quantization noise on quiet distant speech.
                    logger.debug(f"filter_offer_sdp: dropping codec line: {line.strip()[:60]}")
                    if "a=rtpmap:" in line:
                        pt = line.split("a=rtpmap:")[1].split(" ")[0]
                        removed_pts.add(pt)
                    continue
            out.append(line)
        if removed_pts:
            for idx in audio_line_indices:
                mline = out[idx]
                parts = mline.split()
                if len(parts) >= 4:
                    pts = [p for p in parts[3:] if p not in removed_pts]
                    out[idx] = " ".join(parts[:3] + pts)
                    logger.info(f"filter_offer_sdp: m=audio cleaned: {out[idx]}")
        return "\n".join(out)

    @staticmethod
    def _parse_candidate(value) -> RTCIceCandidate | None:
        if isinstance(value, dict):
            raw = value.get("candidate", "")
            sdp_mid = value.get("sdpMid") or "0"
            sdp_ml = value.get("sdpMLineIndex")
        else:
            raw = value
            sdp_mid = "0"
            sdp_ml = None
        if sdp_ml is None:
            sdp_ml = 0
        if not raw or not raw.startswith("candidate:"):
            return None
        parts = raw.split()
        if len(parts) < 8:
            return None
        try:
            cand = RTCIceCandidate(
                component=int(parts[1]),
                foundation=parts[0].split(":")[1],
                ip=parts[4],
                port=int(parts[5]),
                priority=int(parts[3]),
                protocol=parts[2],
                type=parts[7],
                sdpMid=sdp_mid,
                sdpMLineIndex=sdp_ml,
            )
            return cand
        except (ValueError, IndexError):
            return None

    @staticmethod
    async def _add_candidate(pc: RTCPeerConnection, value):
        cand = CameraSession._parse_candidate(value)
        if cand is None:
            logger.debug(f"candidate parse failed: {str(value)[:80]}")
            return
        try:
            await pc.addIceCandidate(cand)
        except Exception as e:
            logger.debug(f"candidate add failed: {e}")

    async def _connect(self):
        sig_url = (
            f"ws://{self.go2rtc_host}:{self.go2rtc_port}"
            f"/api/ws?src={self.stream_name}"
        )

        if self._pc:
            try:
                await self._pc.close()
            except Exception:
                pass
        self._pc = RTCPeerConnection()

        self._out_track = AIVoiceOutputTrack(sample_rate=8000)
        self._pc.addTransceiver(self._out_track, direction="sendonly")
        self._pc.addTransceiver("audio", direction="recvonly")
        logger.info(f"[{self.stream_name}] created sendonly track + recvonly transceiver")

        @self._pc.on("track")
        async def on_track(track):
            logger.info(f"[{self.stream_name}] Incoming audio track: {track.kind}")
            await self._recv_audio(track)

        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        offer_sdp = self._filter_sdp(offer.sdp)

        async with aiohttp.ClientSession() as sig:
            async with sig.ws_connect(sig_url) as ws:
                await ws.send_json({
                    "type": "webrtc/offer",
                    "value": offer_sdp,
                })
                answered = False
                announced = False
                ticks = 0
                while not self._stopped.is_set():
                    try:
                        msg = await ws.receive(timeout=3.0)
                    except asyncio.TimeoutError:
                        if not answered:
                            continue
                        st = self._pc.connectionState
                        if st not in ("connected", "connecting"):
                            logger.warning(
                                f"[{self.stream_name}] ICE lost ({st}), reconnecting"
                            )
                            return
                        ticks += 1
                        if ticks % 10 == 0:
                            logger.info(
                                f"[{self.stream_name}] keepalive {ticks * 3}s ICE={st}"
                            )
                        continue

                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        st = self._pc.connectionState if answered else "None"
                        logger.warning(
                            f"[{self.stream_name}] signaling WS closed ({st}), reconnecting"
                        )
                        return

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue

                    data = json.loads(msg.data)
                    t = data.get("type")
                    if t == "webrtc/candidate":
                        await self._add_candidate(self._pc, data["value"])
                    elif t == "webrtc/answer":
                        answer_sdp = data["value"]
                        logger.info(f"[{self.stream_name}] ANSWER SDP: {answer_sdp}")
                        await self._pc.setRemoteDescription(
                            RTCSessionDescription(sdp=answer_sdp, type="answer")
                        )
                        answered = True
                        logger.info(f"[{self.stream_name}] WebRTC ready (bidirectional)")

                    if answered:
                        st = self._pc.connectionState
                        if st == "connected" and not announced:
                            announced = True
                            logger.info(f"[{self.stream_name}] WebRTC connected")
                            asyncio.create_task(self._delayed_attention())
                        elif st == "failed":
                            logger.warning(f"[{self.stream_name}] WebRTC failed, retrying")
                            return

    def _frame_to_16k_mono(self, frame: av.AudioFrame) -> bytes:
        if frame.format.name in ('flt', 'fltp'):
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                mono = arr.mean(axis=0)
            else:
                mono = arr.flatten()
            mono = (mono * 32767).clip(-32768, 32767).astype(np.int16)
        else:
            raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
            n_planes = len(frame.planes)
            if n_planes > 1:
                channels = [np.frombuffer(bytes(p), dtype=np.int16) for p in frame.planes]
                mono = np.mean(channels, axis=0, dtype=np.int16)
            else:
                layout_name = frame.layout.name if hasattr(frame.layout, 'name') else str(frame.layout)
                if layout_name in ('mono', '1'):
                    mono = raw
                else:
                    n_ch = len(frame.layout.channels) if hasattr(frame.layout, 'channels') else 2
                    mono = raw.reshape(-1, n_ch).mean(axis=1, dtype=np.int16)

        return bytes(mono.tobytes())

    @staticmethod
    def _resample_16k(mono_48k: bytes) -> bytes:
        """Resample a 160ms mono block from 48kHz to 16kHz in a worker thread.

        Resampling per 20ms frame was too slow: it held the aiortc track lock
        long enough for the UDP socket buffer to overflow, dropping RTP
        packets and producing all-zero feed windows (35%+ vs 0% for a plain
        reader on the same stream). One polyphase pass per 160ms block is
        ~8x cheaper and runs off the event loop.
        """
        from scipy.signal import resample_poly
        arr = np.frombuffer(mono_48k, dtype=np.int16).astype(np.float32)
        return resample_poly(arr, 1, 3).astype(np.int16).tobytes()

    @staticmethod
    def _resample_generic(pcm: bytes, rate: int) -> bytes:
        from scipy.signal import resample_poly
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        g = np.gcd(16000, rate)
        up, down = 16000 // g, rate // g
        return resample_poly(arr, up, down).astype(np.int16).tobytes()

    async def _recv_audio(self, track):
        frame_count = 0
        while not self._stopped.is_set():
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=10.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.info(f"[{self.stream_name}] _recv_audio ended: {e}")
                break
            
            if not isinstance(frame, av.AudioFrame):
                continue
            
            frame_count += 1
            if frame_count <= 2:
                logger.info(
                    f"[{self.stream_name}] recv frame #{frame_count}: "
                    f"rate={frame.sample_rate} fmt={frame.format.name} "
                    f"layout={frame.layout.name} planes={len(frame.planes)} "
                    f"samples={frame.samples}"
                )
            # Mic audio comes from the RTSP loop (ffmpeg, raw L16, see
            # _rtsp_audio_loop) — the WebRTC audio track is not used for VAD.

    async def _rtsp_audio_loop(self):
        rtsp_url = f"rtsp://{self.go2rtc_host}:8554/{self.stream_name}?audio=copy"
        logger.info(f"[{self.stream_name}] RTSP audio loop starting (ffmpeg): {rtsp_url}")
        while not self._stopped.is_set():
            proc = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-loglevel", "error",
                    "-rtsp_transport", "tcp",
                    "-i", rtsp_url,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "s16le",
                    "pipe:1",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                frame_bytes = 1280 * 2
                while not self._stopped.is_set():
                    chunk = await asyncio.wait_for(
                        proc.stdout.readexactly(frame_bytes), timeout=20.0
                    )
                    if not chunk:
                        break
                    await self._feed_audio(chunk)
            except asyncio.IncompleteReadError:
                logger.info(f"[{self.stream_name}] RTSP ffmpeg stream ended")
            except asyncio.TimeoutError:
                logger.warning(f"[{self.stream_name}] RTSP ffmpeg stalled, restarting")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[{self.stream_name}] RTSP audio error: {e}")
            finally:
                if proc:
                    try:
                        proc.kill()
                        await proc.wait()
                    except Exception:
                        pass
            if not self._stopped.is_set():
                logger.info(f"[{self.stream_name}] RTSP audio reconnecting in 3s...")
                await asyncio.sleep(3)

    async def _feed_audio(self, pcm: bytes, rate: int = 16000):
        if time.time() < self._speaking_until:
            return
        # Hard echo guard: while TTS frames are still queued for playback,
        # the camera speaker is (or will be) sounding — do not feed the mic.
        if self._out_track and self._out_track.echo_active():
            return
        
        if self._wake_detected and time.time() >= self._wake_expires:
            self._back_to_wake()

        if rate == 48000:
            self._resample_buf += pcm
            # 160ms at 48kHz mono = 7680 samples = 15360 bytes
            while len(self._resample_buf) >= 15360:
                block = self._resample_buf[:15360]
                self._resample_buf = self._resample_buf[15360:]
                pcm_16k = await asyncio.to_thread(self._resample_16k, block)
                self._vad_buf.extend(pcm_16k)
                self._drain_vad_buf()
        else:
            pcm_16k = await asyncio.to_thread(self._resample_generic, pcm, rate)
            self._vad_buf.extend(pcm_16k)
            self._drain_vad_buf()

    def _drain_vad_buf(self):
        chunk_bytes = 2560
        while len(self._vad_buf) >= chunk_bytes:
            chunk = bytes(self._vad_buf[:chunk_bytes])
            self._vad_buf = self._vad_buf[chunk_bytes:]

            now = time.time()
            if now - self._last_feed_log >= 3.0:
                self._last_feed_log = now
                samples_f32 = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                rms_int = int(np.sqrt(np.mean(np.square(samples_f32))))
                peak = int(np.max(np.abs(samples_f32)))
                logger.info(f"[{self.stream_name}] feed rms={rms_int} peak={peak}")
                self._bg_window.append(rms_int)
                if len(self._bg_window) > 20:
                    self._bg_window.pop(0)

            asyncio.create_task(self._vad_process(chunk))

    async def _vad_process(self, chunk: bytes):
        async with self._vad_lock:
            s16 = np.frombuffer(chunk, dtype=np.int16)
            orig_rms = float(np.sqrt(np.mean(np.square(s16.astype(np.float64))))) / 32768.0
            speech, _ = await self._vad.is_speech(s16.tobytes())
            now = time.time()
            
            # Only trust VAD if original audio has meaningful energy.
            # Corridor noise floor (mic AGC-boosted) sits at 0.015-0.042
            # normalized (quiet hours 0.022-0.029); ticks/artifacts reach 0.05+.
            # Distant speech (3m) sits ~0.03-0.12. Gate at 0.030 to reject
            # the idle noise floor while keeping distant speech.
            if orig_rms < 0.030:
                speech = False

            if now - self._last_rms_log > 2.0:
                self._last_rms_log = now
                logger.info(f"[{self.stream_name}] VAD rms={orig_rms:.4f} speech={speech} consec={self._vad_speech_consecutive}")

            # Sliding window (6 frames = 480ms): start utterance when >=3 speech frames,
            # tolerate short pauses between syllables instead of requiring 3 consecutive.
            self._vad_window.append(1 if speech else 0)
            if len(self._vad_window) > 6:
                self._vad_window.pop(0)
            win_speech = sum(self._vad_window)

            if speech:
                self._vad_speech_consecutive += 1
                if not self._vad_has_speech:
                    if win_speech >= 3 and not self._processing_utterance:
                        logger.info(f"[{self.stream_name}] VAD SPEECH START rms={orig_rms}")
                        self._vad_has_speech = True
                        self._vad_silence_frames = 0
                        self._vad_start_time = now
                        self._vad_speech_buf.extend(chunk)
                else:
                    self._vad_speech_buf.extend(chunk)
                self._vad_silence_frames = 0
            else:
                if self._vad_has_speech:
                    self._vad_silence_frames += 1
                    if self._vad_silence_frames >= self._vad_silence_limit:
                        dur = len(self._vad_speech_buf) / 32
                        logger.info(f"[{self.stream_name}] VAD UTTERANCE END dur={dur:.0f}ms")
                        buf = bytes(self._vad_speech_buf)
                        self._vad_speech_buf.clear()
                        self._vad_has_speech = False
                        self._vad_speech_consecutive = 0
                        self._vad_silence_frames = 0
                        self._vad_window.clear()
                        asyncio.create_task(self._process_utterance(buf))
                    else:
                        self._vad_speech_buf.extend(chunk)

            # Max duration by accumulated audio length (WebRTC delivers frames in
            # bursts, so wall-clock time is not a reliable cap)
            if self._vad_has_speech and len(self._vad_speech_buf) >= int(self._vad_max_duration * 32000):
                buf = bytes(self._vad_speech_buf)
                self._vad_speech_buf.clear()
                self._vad_has_speech = False
                self._vad_speech_consecutive = 0
                self._vad_silence_frames = 0
                self._vad_window.clear()
                self._vad_start_time = now
                asyncio.create_task(self._process_utterance(buf))

            # openWakeWord detection on the live 16k feed (echo-guarded)
            if (
                not self._wake_detected
                and self._engine.oww_model is not None
                and time.time() >= self._speaking_until
            ):
                wake = await asyncio.to_thread(self._engine.check_wakeword, s16, 0.25)
                if wake:
                    logger.info(f"[{self.stream_name}] 🎯 Wake word (openWakeWord)")
                    self._wake_detected = True
                    self._wake_expires = time.time() + self._wake_timeout
                    if not self._vad_has_speech:
                        asyncio.create_task(self._play_attention("oww"))

    def _apply_ns(self, audio_f: np.ndarray):
        """SpeexDSP noise suppression — strips stationary noise (distant speech)."""
        try:
            from speexdsp_ns import NoiseSuppression
            ns = NoiseSuppression.create(frame_size=256, sample_rate=16000)
            clean = bytearray()
            for i in range(0, len(audio_f) - 255, 256):
                clean += ns.process(audio_f[i:i+256].astype(np.int16).tobytes())
            return np.frombuffer(bytes(clean), dtype=np.int16).astype(np.float32), True
        except Exception as exc:
            logger.warning(f"[{self.stream_name}] NS error: {exc}")
            return audio_f, False

    async def _process_utterance(self, buf: bytes):
        if self._processing_utterance:
            return
        
        self._processing_utterance = True
        try:
            samples = np.frombuffer(buf, dtype=np.int16)
            duration_s = len(samples) / 16000.0
            rms_raw = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
            peak_raw = int(np.max(np.abs(samples)))
            rms_dB = 20 * np.log10(max(rms_raw, 1) / 32768)

            # Minimum duration gate — very short utterances produce garbage
            if duration_s < 0.5:
                logger.info(f"[{self.stream_name}] ⏩ Too short ({duration_s:.2f}s rms={rms_raw:.0f} peak={peak_raw})")
                return

            # Minimum energy gate vs adaptive background floor.
            # Corridor noise sits at rms 500-1400; distant speech (3m) ~950-2500.
            bg = float(np.median(self._bg_window)) if self._bg_window else 0.0
            min_rms = max(350, 1.15 * bg)
            if rms_raw < min_rms:
                # NS rescue: distant speech sits inside the noise floor — try
                # cleaning first. Clean noise is ~100-250 int16, speech survives
                # at 400+.
                cleaned, ns_ok = self._apply_ns(samples.astype(np.float32))
                rms_clean = float(np.sqrt(np.mean(np.square(cleaned))))
                if ns_ok and rms_clean >= 400:
                    logger.info(f"[{self.stream_name}] 🛟 NS rescue: rms {rms_raw:.0f} -> clean {rms_clean:.0f} (bg={bg:.0f})")
                    samples = cleaned.astype(np.int16)
                    rms_raw = rms_clean
                else:
                    logger.info(f"[{self.stream_name}] ⏩ Too quiet (rms={rms_raw:.0f} peak={peak_raw} dB={rms_dB:.1f} bg={bg:.0f} min={min_rms:.0f})")
                    return

            # Speech-energy ratio gate — count chunks with real energy.
            # Background noise ~190-500 int16 (AGC), speech chunks ~800+,
            # distant speech (3m) chunks ~400-900.
            chunk_samples = 1280  # 80ms at 16kHz
            speech_chunks = 0
            total_chunks = 0
            for i in range(0, len(samples) - chunk_samples + 1, chunk_samples):
                chunk_rms = float(np.sqrt(np.mean(np.square(samples[i:i+chunk_samples].astype(np.float32)))))
                total_chunks += 1
                if chunk_rms >= 400:
                    speech_chunks += 1
            speech_ratio = speech_chunks / max(total_chunks, 1)
            if speech_ratio < 0.35:
                logger.info(f"[{self.stream_name}] ⏩ Low speech ratio ({speech_ratio:.0%}={speech_chunks}/{total_chunks} rms={rms_raw:.0f} dB={rms_dB:.1f})")
                return

            # Adaptive peak normalization (max 20x gain for distant speech)
            audio_f = samples.astype(np.float32)

            # Noise suppression — only when speech is near the noise floor
            # (distant 3m). Normal near-field speech (rms well above bg) must
            # pass through untouched so Whisper sees the original spectrum.
            ns_applied = False
            if rms_raw < max(1500, 1.8 * bg) and len(audio_f) >= 256:
                audio_f, ns_applied = self._apply_ns(audio_f)

            peak = np.max(np.abs(audio_f))
            gain_applied = 1.0
            if peak > 0:
                gain = min(32767.0 / peak, 20.0)
                if gain > 1.5:
                    audio_f = np.clip(audio_f * gain, -32768, 32767)
                    gain_applied = gain
            buf = audio_f.astype(np.int16).tobytes()
            buf_raw = samples.astype(np.int16).tobytes()

            wav = await asyncio.to_thread(self._encode_wav, buf)
            try:
                import os
                os.makedirs("/tmp/utterances", exist_ok=True)
                with open(f"/tmp/utterances/u_{int(time.time())}.wav", "wb") as f:
                    f.write(wav)
            except Exception:
                pass
            logger.info(f"[{self.stream_name}] 🎤 Whisper IN: {duration_s:.2f}s raw_rms={rms_raw:.0f} peak={peak_raw} dB={rms_dB:.1f} gain={gain_applied:.1f}x ns={int(ns_applied)} wav={len(wav)}B")
            txt, uid = await asyncio.gather(
                self._fetch_transcription(wav),
                self._fetch_speaker_id(wav),
            )
            if (not txt or not is_valid_text(txt)) and self._wake_detected:
                # Post-wake rescue: NS may have eaten the consonants of a
                # distant command — retry the unprocessed signal.
                logger.info(f"[{self.stream_name}] 🔁 Post-wake retry raw signal (wake={self._wake_detected})")
                txt = await self._fetch_transcription(
                    await asyncio.to_thread(self._encode_wav, buf_raw),
                    temperature="0.7",
                )
            if not txt or not is_valid_text(txt):
                if not txt:
                    logger.info(f"[{self.stream_name}] ❌ Whisper empty for {len(buf)}-byte ({duration_s:.2f}s)")
                else:
                    logger.info(f"[{self.stream_name}] ❌ Rejected: '{txt[:60]}'")
                return
                
            logger.info(f"[{self.stream_name}] ✅ Whisper OK: '{txt[:60]}'")
            norm = txt.lower().strip().strip(".,!? -")
            if norm in ("готов", "да", "го", "начинаем", "начали", "готово", "ок", "окей", "давай"):
                logger.info(f"[{self.stream_name}] ✅ Acknowledge: '{txt[:60]}'")
                await self._play_attention("ack")
                self._back_to_wake()
                return
            # Anti-loop: camera hears its own TTS echo with 15-25s delay;
            # if this transcript matches a recent one, it's the echo.
            if norm and norm == self._last_transcript:
                logger.info(f"[{self.stream_name}] ❌ Echo repeat, ignoring: '{txt[:60]}'")
                return
            self._last_transcript = norm
            if uid != "unknown":
                logger.info(f"[{self.stream_name}] 👤 Speaker: {uid} | '{txt[:60]}'")
            await self._handle_wake_or_command(txt, uid)
            return
        finally:
            self._processing_utterance = False

    async def _handle_wake_or_command(self, txt: str, uid: str = "camera"):
        low = txt.lower()
        m = re.search(r"\b(компьютер|компютер|computer)\b", low)
        if m:
            self._wake_detected = True
            self._wake_expires = time.time() + self._wake_timeout
            asyncio.create_task(self._play_attention("processor_kw"))
            cmd = (txt[: m.start()] + " " + txt[m.end():]).strip()
            if cmd:
                logger.info(f"[{self.stream_name}] 🎯 Wake+cmd: '{txt[:60]}' -> '{cmd[:60]}'")
                await self._call_nanobot(cmd, uid)
            else:
                logger.info(f"[{self.stream_name}] 🎯 Wake word (awaiting command)")
            return
        if not self._wake_detected:
            logger.info(f"[{self.stream_name}] ⏳ No wake word, ignoring: '{txt[:60]}'")
            return
        logger.info(f"[{self.stream_name}] 🗣 Post-wake dialogue: '{txt[:60]}'")
        await self._call_nanobot(txt, uid)

    async def _on_user_command(self, text: str):
        try:
            kw = self.wake_keyword.lower()
            if kw not in text.lower():
                self._processor.set_state(AgentState.LISTENING)
                return
            idx = text.lower().find(kw)
            cmd = text[idx + len(kw):].strip().lstrip(".,!? -")
            if not cmd:
                self._wake_detected = True
                self._wake_expires = time.time() + self._wake_timeout
                asyncio.create_task(self._play_attention("processor_kw"))
                return
            self._processor.set_state(AgentState.SPEAKING)
            await self._call_nanobot(cmd)
        except Exception as e:
            logger.error(f"[{self.stream_name}] Error handling command: {e}")
            self._processor.set_state(AgentState.LISTENING)

    async def _call_nanobot(self, txt: str, uid: str = "camera"):
        if not txt:
            return
        if time.time() < GLOBAL_TTS_UNTIL:
            logger.info(
                f"[{self.stream_name}] Ignoring '{txt[:40]}' — TTS playback active (echo guard)"
            )
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"{self._nanobot_url}?token={self._nanobot_token}&chat_id={self.chat_id}"
                ) as ws:
                    await ws.send_json({
                        "type": "message",
                        "chat_id": self.chat_id,
                        "content": txt,
                        "user_id": uid,
                        "user_name": self.stream_name,
                        "voice_reply": True,
                    })
                    reply = await asyncio.wait_for(self._wait_reply(ws), timeout=30.0)
                    if reply:
                        self._wake_expires = time.time() + self._wake_timeout
                        await asyncio.wait_for(self._speak(reply, reply), timeout=90.0)
        except Exception as e:
            logger.warning(f"[{self.stream_name}] Nanobot error: {e}")

    @staticmethod
    async def _wait_reply(ws) -> str | None:
        parts: list[str] = []
        got_text = False
        try:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                ev = data.get("event")
                if ev in ("done", "error"):
                    break
                if ev == "stream_end":
                    if data.get("resuming") and not got_text:
                        # End of the PREVIOUS stream (session resume), sent
                        # before the new response starts — must not terminate.
                        continue
                    break  # end of the current response
                if ev == "final":
                    break
                if "text" in data and data.get("type") not in ("stt", "listen") and ev not in ("reasoning_delta", "thinking", "ready"):
                    parts.append(str(data["text"]))
                    got_text = True
                elif "parts" in data:
                    parts.append(str(data["parts"]))
                    got_text = True
        except Exception:
            pass
        text = "".join(parts)
        # Nanobot streams text in fragments — strip tags on the JOINED text.
        # Drop explicitly closed [thinking]...[/thinking] blocks, else just
        # remove the bare tags (the model's replies carry [emotion] prefixes).
        while True:
            m = re.search(r"\[thinking\](.*?)\[/thinking\]", text, flags=re.S)
            if not m:
                break
            text = text[: m.start()] + text[m.end():]
        text = re.sub(r"\[/?thinking\]", "", text)
        text = re.sub(r"\[[a-z]{2,30}\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text if text else None

    def _encode_wav(self, pcm_16k: bytes) -> bytes:
        sample_rate = 16000
        bits = 16
        channels = 1
        data_size = len(pcm_16k)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_size, b"WAVE",
            b"fmt ", 16, 1, channels, sample_rate,
            sample_rate * channels * bits // 8,
            channels * bits // 8,
            bits,
            b"data", data_size,
        )
        return header + pcm_16k

    async def _fetch_transcription(self, wav: bytes, temperature: str | None = None) -> str:
        for temp in (temperature, "0.0", "0.5"):
            if not temp:
                continue
            form = aiohttp.FormData()
            form.add_field("file", wav, filename="a.wav")
            form.add_field("model", self.whisper_model)
            form.add_field("language", "ru")
            form.add_field("temperature", temp)
            try:
                async with self.http_session.post(
                    self.whisper_url, data=form,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    if r.status == 200:
                        data = await r.json()
                        text = (data.get("text") or "").strip()
                        if text:
                            return text
            except Exception:
                pass
        return ""

    async def _fetch_speaker_id(self, wav: bytes) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
                "-c:a", "libopus", "-f", "ogg", "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            ogg, _ = await asyncio.wait_for(proc.communicate(wav), timeout=10)
            if not ogg:
                return "unknown"
            form = aiohttp.FormData()
            form.add_field("file", ogg, filename="audio.ogg", content_type="audio/ogg")
            async with self.http_session.post(
                self._speaker_id_url, data=form, timeout=10
            ) as r:
                if r.status == 200:
                    j = await r.json()
                    uid, conf = j.get("user_id", "unknown"), j.get("confidence", 0.0)
                    if uid != "unknown" and conf > 0.1:
                        logger.info(f"[{self.stream_name}] 👤 [SpeakerID] Recognized: {uid} ({conf:.2f})")
                        return uid
                    logger.debug(f"[{self.stream_name}] 👤 [SpeakerID] Rejected: {uid} ({conf:.2f})")
        except Exception as e:
            logger.warning(f"[{self.stream_name}] ⚠️ [SpeakerID] Request failed: {e}")
        return "unknown"

    def _back_to_wake(self):
        self._wake_detected = False

    async def _delayed_attention(self):
        try:
            await asyncio.sleep(5)
            if self._attention_played:
                return
            self._attention_played = True
            await self._play_attention("connect")
        except Exception:
            pass

    async def _play_attention(self, reason: str = ""):
        now = time.time()
        if now - self._last_attention < 60.0:
            return
        self._last_attention = now
        try:
            sr = 8000
            duration = 0.3
            n = int(sr * duration)
            t = np.arange(n) / sr
            tone = np.sin(2 * np.pi * 880 * t) * 0.3
            tone += np.sin(2 * np.pi * 1320 * t) * 0.15
            fade_n = int(sr * 0.03)
            env = np.ones(n)
            env[:fade_n] = np.linspace(0, 1, fade_n)
            env[-fade_n:] = np.linspace(1, 0, fade_n)
            mix = tone * env
            pcm = (mix * 32767).astype(np.int16).tobytes()
            if self._out_track:
                await self._out_track.queue_frame(pcm, sr)
                self._out_track._last_play_duration = duration
            # Only extend speaking_until if not already blocked longer by KWS
            delay = now + 0.35
            if delay > self._speaking_until:
                self._speaking_until = delay
        except Exception as exc:
            logger.warning(f"[{self.stream_name}] attention error: {exc}")

    async def _play_activation_sound(self):
        try:
            seg = AudioSegment.from_file(self._activation_wav_path)
            seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
            pcm = seg.raw_data
            if self._out_track:
                await self._out_track.queue_frame(pcm, 8000)
                self._out_track._last_play_duration = len(pcm) / (2 * 8000)
        except Exception:
            pass

    async def _speak(self, text: str, reply: str = ""):
        self._speaking = True
        logger.info(f"[{self.stream_name}] _speak: input='{text[:60]}'")

        # Block mic immediately: TTS request takes seconds over network,
        # and queued frames may already be playing out of the camera speaker.
        if self._out_track:
            self._speaking_until = time.time() + self._out_track.queue_seconds() + 3.0

        headers = {}
        if self.tts_api_key:
            headers["Authorization"] = f"Bearer {self.tts_api_key}"

        try:
            async with self.http_session.post(
                self.tts_url,
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": self.tts_voice,
                    "response_format": "mp3",
                },
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as r:
                if r.status != 200:
                    self._speaking = False
                    return
                mp3 = await r.read()
        except Exception:
            self._speaking = False
            return

        try:
            seg = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
            sr = 8000
            seg = seg.set_frame_rate(sr).set_channels(1).set_sample_width(2)
            pcm = seg.raw_data
            pcm_arr = np.frombuffer(pcm, dtype=np.int16)
            peak = np.max(np.abs(pcm_arr))
            if peak > 0:
                target = 20000
                gain = min(target / peak, 4.0)
                if gain > 1.2:
                    pcm_arr = np.clip(pcm_arr * gain, -32768, 32767).astype(np.int16)
                    pcm = pcm_arr.tobytes()
            
            # Блокируем микрофон только сейчас, когда звук реально готов пойти в канал
            audio_dur = len(pcm) / (sr * 2)
            if self._out_track:
                self._out_track._last_play_duration = audio_dur
            is_question = bool(_HAS_QUESTION_RE.search(reply.strip()))
            echo_tail = 1.5 if is_question else 3.0

            chunk_size = sr * 20 // 1000 * 2
            for i in range(0, len(pcm), chunk_size):
                c = pcm[i: i + chunk_size]
                if len(c) < chunk_size:
                    c += b"\x00" * (chunk_size - len(c))
                if self._out_track:
                    # Теперь мы просто асинхронно пушим куски, тайминг задается внутри AIVoiceOutputTrack.recv()
                    await self._out_track.queue_frame(c, sr)

            # Set the lock AFTER the push completes: pushing into a full queue
            # takes ~audio_dur of real time (backpressure), so a lock set
            # before the loop would expire while the speaker is still playing.
            if self._out_track:
                self._speaking_until = time.time() + self._out_track.queue_seconds() + echo_tail
            global GLOBAL_TTS_UNTIL
            GLOBAL_TTS_UNTIL = time.time() + audio_dur + 3.0
        except Exception as e:
            logger.warning(f"[{self.stream_name}] playback error: {e}")

        self._speaking = False

        if _HAS_QUESTION_RE.search(reply.strip()):
            self._wake_expires = time.time() + 30
        else:
            self._back_to_wake()
