"""
Camera client module — bidirectional voice assistant for OpenIPC IP cameras.

Per-camera pipeline (one CameraSession per stream, see main.start_camera_sessions):

  MIC (listen):
    go2rtc RTSP (:8554/<stream>?audio=copy)
      -> ffmpeg -> PCM s16le 16k mono
      -> _feed_audio: echo guards (queue + cross-correlation vs TTS ring buffer)
     -> _vad_process: Silero VAD + energy gates
          - utterances end after ~10 VAD-silent frames (~1.44s); the noise
            floor estimate behind the energy gates is floor-biased (median of
            the QUIETEST half of a sliding window) so sustained loud speech
            cannot redefine what counts as "silence"
          - openWakeWord on live chunks (AGC boost to peak=4000), adaptive
            threshold that self-tunes to false fires and decays over time;
            every imminent fire is arbitrated ACROSS cameras by raw mic
            loudness so only the closest device answers (see the arbiter
            section below)
       -> Whisper STT (+ parallel SpeakerID, non-blocking) -> wake/command routing

  SPEAKER (talk back):
    Nanobot WS (streaming) -> sentence flusher -> Edge TTS per sentence
      -> WebRTC sendonly track -> go2rtc backbridge -> camera speaker
      (requires majestic.yaml: rtsp.backchannel=true, audio.srate "8000", codec pcm)

  SELF-HEAL:
    go2rtc sometimes loses the producer or its audio track after a camera reboot
    (serves video-only SDP). After N consecutive ffmpeg stalls the session re-
    registers the stream via go2rtc HTTP API (DELETE ?src=<name> + PUT
    ?name=&src=<url>) which forces a fresh, audio-capable producer.

All identifiers/comments in English; Russian string literals are runtime data.
"""

import asyncio
import base64
import fractions
import hashlib
import io
import json
import logging
import os
import re
import struct
import time
from dataclasses import dataclass
from math import gcd

import aiohttp
import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.rtcicetransport import RTCIceCandidate
from pydub import AudioSegment

from audio_utils import is_valid_text
from engine import LocalAudioEngine, CameraProcessor, AgentState

logger = logging.getLogger("camera")

# go2rtc stream names that count as "the corridor". There is NO plain
# "corridor" entry any more — the second corridor cam split it into
# corridor1/corridor2 (frigate/config/config.yaml) — so every `== "corridor"`
# test below was matching nothing and the corridor-specific mic gain silently
# stopped applying when the rename happened.
_CORRIDOR_STREAMS = frozenset({"corridor", "corridor1", "corridor2"})

# Sample rate of everything we push INTO the camera speaker (AIVoiceOutputTrack
# -> go2rtc -> ONVIF AudioOutput backchannel).
#
# This was 8000 and it was WRONG: the OpenIPC speaker runs its output at
# 48 kHz, so an 8 kHz stream came out six times too fast and an octave too
# high — "пищит как бурундук". 8 kHz is the rate of the ESP32 leg (Opus to a
# tiny speaker codec) and the mic feed is 16 kHz; neither sets the camera's
# playback rate. Note this is the rate of OUR track: go2rtc re-encodes to
# whatever the camera negotiated on its backchannel (PCMU/8000 here).
TTS_PLAY_RATE = 48000
# Follow-up listening windows after a reply, per room via ENV (0 => these
# defaults). They exist because the cameras did NOT listen after an ANSWER,
# only after a question: `_back_to_wake()` closed the window unconditionally
# on any non-question reply. Reported 05.10.2026 as "the dialogue setting does
# not work for cameras the way it does for the ESP32", and it is exactly the
# asymmetry: `_finalize_turn_followup()` in main.py sets status LISTENING in
# BOTH branches for satellites, with STANDBY_TIMEOUT_QUESTION 30 s and
# STANDBY_TIMEOUT_STATEMENT 10 s.
#
# The statement window is the short one on purpose — it exists so a natural
# follow-up ("а теперь выключи") is heard without a second wake word, not so
# the room stays open to the television. Note what makes that safe here and not
# a false-wake risk: the wake word is DECODED, so television speech has to
# actually contain «компьютер» to open the window, and `_wake_gate_open()`
# still refuses while our own playback is in the air.
CAMERA_DIALOGUE_QUESTION_S = 30.0
CAMERA_DIALOGUE_STATEMENT_S = 10.0
# Peak a synthesised reply is normalised to before it is POSTed to the camera
# speaker. 20000 of 32767 is 61 % of full scale. Overridable per room via
# CAMERA_TTS_TARGET_PEAK[_<NAME>] — see CameraConfig.tts_target_peak for why
# that has to be an ENV rather than a constant.
_TTS_TARGET_PEAK_DEFAULT = 20000

# How long the wake decoder stays deaf AFTER a clip we played ourselves.
# Only the direct acoustic return needs a timer: `_is_echo` removes the delayed
# room echo by cross-correlation in `_feed_audio`, before the chunk reaches the
# decoder, and extends suppression itself when a correlation matches. The old
# value was 15 s (16 s for the pip) and it silently ate the user's next wake
# word — measured 04.10.2026, two «компьютер» 8 s and 11 s after the reply were
# decoded and dropped before the third one got through 28 s later.
_ECHO_TAIL_S = 3.0

# Same-breath debounce: «...компьютер, да, компьютер, выключи свет» must fire
# ONCE, because a second `_fire_wake()` clears `_vad_speech_buf` and throws
# away the command being collected. This replaces `_wake_detected` as the
# firing guard — a wake word must always be able to re-arm the interaction,
# which is the entire point of having one.
_WAKE_REARM_DEBOUNCE_S = 1.5

# How loud a frame must be, relative to the utterance anchor, to count as
# SPEECH rather than as background. The pause test uses the windowed percentile
# above (a responsive question: has the speaker stopped?); this one is about
# whether the utterance contains speech at all, and it must survive a long
# silence — hence the utterance-scoped anchor in _PauseEndpoint.
_ANCHOR_FRAC = 0.5

# How we get audio ONTO the camera speaker. Two paths, and the difference is
# not cosmetic:
#
# 1. `/play_audio` (PREFERRED) — a plain HTTP POST of raw mono s16le PCM with
#    the rate in the Content-Type:
#       curl -u root:PW -X POST --data-binary @x.pcm \
#            -H 'Content-Type: application/octet-stream;rate=48000' \
#            http://<cam>/play_audio
#    (OpenIPC wiki, "How to play audio file on camera's speaker over
#    network".) Straight into the speaker at the right rate — measured
#    03.10.2026: a 1000 Hz tone comes back at 1000 Hz and speech F0 239 Hz in
#    / 235 Hz out, in real time.
#
# 2. The go2rtc ONVIF backchannel via the WebRTC sendonly track (FALLBACK) —
#    broken on this firmware. The camera advertises exactly ONE codec:
#       m=audio 0 RTP/AVP 0 / a=rtpmap:0 PCMU/8000 / a=sendonly
#            / a=control:audio-backchannel
#    ...and then plays that 8 kHz stream at 48 kHz, so everything comes out
#    ~6x too fast ("пищит как бурундук"). Measured with a tone: 1000 Hz sent,
#    6000 Hz heard by the camera's own microphone. Pre-stretching by 6 fixes
#    the pitch but makes every reply take 6x longer to play, so it is not
#    usable for dialogue — hence /play_audio above.
#
# Audio OUTPUT also requires the ONVIF AudioOutput to be enabled on the cam;
# until it is, /play_audio returns 200 but nothing is heard.



def _echo_of_reply(norm: str, reply_norm: str) -> bool:
    """True if `norm` is the echoed TTS reply `reply_norm`.

    The camera speaker plays our own reply, so Whisper transcribes it
    ~verbatim. A high token-overlap (Jaccard) means the incoming transcript
    is the echo of what we just said, not a real user command.
    """
    if not norm or not reply_norm:
        return False
    a = set(norm.split())
    b = set(reply_norm.split())
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.5


logger.setLevel(logging.INFO)
logging.getLogger("aiortc").setLevel(logging.WARNING)
# aioice is NOT a child of "aiortc" — it is its own top-level logger, so the
# line above never touched it. Measured 04.10.2026 21:31: one gateway restart
# produced 1059 lines of `aioice.ice - Check CandidatePair(...) FAILED` in a
# single minute, 80% of the container's entire log volume, while the ICE storm
# is only a startup burst (zero aioice lines in the 14 minutes after ICE
# connected). Every (re)connect pays it again, so it is log noise that hides
# the lines that matter, not a steady CPU cost.
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
_r = logging.getLogger()
if not _r.handlers:
    _rh = logging.StreamHandler()
    _rh.setLevel(logging.DEBUG)
    _rh.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _r.addHandler(_rh)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_h)
logger.propagate = False


class AIVoiceOutputTrack(MediaStreamTrack):
    """aiortc sendonly track that plays TTS PCM to the camera speaker.

    recv() pulls the next queued frame (or a pre-built silence frame when
    the queue is dry) and paces it against the RTP clock, so go2rtc sees a
    steady 20 ms stream instead of bursts. Writers use queue_frame(),
    which blocks when the buffer is full rather than dropping audio.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = TTS_PLAY_RATE):
        super().__init__()
        self._sample_rate = sample_rate
        # Buffer for ~10 seconds of audio. If the queue fills up, the
        # writer waits (backpressure) instead of dropping frames.
        self._queue: asyncio.Queue[av.AudioFrame] = asyncio.Queue(maxsize=500)
        self._timestamp = 0
        self._start: float = 0.0

        self._frame_samples = max(sample_rate * 20 // 1000, 160)
        silence_pcm = b"\x00\x00" * self._frame_samples
        self._silence_frame = self._pcm_to_frame(silence_pcm, sample_rate)
        self._frame_count = 0
        self._last_real_recv: float = 0.0
        # Duration (seconds) of the last queued playback; the mic echo-guard
        # holds closed only as long as the playback itself (beeps are short,
        # TTS replies long), instead of a blanket 10s that swallowed the
        # question spoken right after the wake beep.
        self._last_play_duration: float = 0.0

    def _pcm_to_frame(self, pcm: bytes, sample_rate: int) -> av.AudioFrame:
        """Wrap raw s16le mono PCM into a PyAV AudioFrame for the WebRTC track."""
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
                logger.info(
                    f"AIVoiceOutputTrack: real frame #{self._frame_count}, queue={self._queue.qsize()}"
                )
        except asyncio.QueueEmpty:
            frame = self._silence_frame

        if self._timestamp == 0:
            self._start = time.time()

        # Strict frame pacing for WebRTC (RTP clock)
        wait = self._start + (self._timestamp / self._sample_rate) - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._timestamp += frame.samples
        return frame

    async def queue_frame(self, pcm: bytes, sample_rate: int = TTS_PLAY_RATE):
        # We use `await put` so long TTS replies never overflow the track
        # and drop words in the middle of a sentence.
        frame = self._pcm_to_frame(pcm, sample_rate)
        await self._queue.put(frame)

    def queue_seconds(self) -> float:
        return self._queue.qsize() * (self._frame_samples / self._sample_rate)

    def echo_active(self) -> bool:
        if self._queue.qsize() > 0:
            return True
        # WebRTC/go2rtc buffers return the camera's own speaker echo back
        # with 8-15s delay, in ~3s bursts. Hold the mic closed for the full
        # playback duration plus a long tail so the delayed echo of the END
        # of the clip is still suppressed (prevents the wake model from
        # re-triggering on its own TTS and looping).
        hold = min(15.0, self._last_play_duration + 0.3)
        return time.time() - self._last_real_recv < hold

    def stop(self):
        pass


_HAS_QUESTION_RE = re.compile(r"[?？]")

# ---------------------------------------------------------------------------
# Cross-camera arbitration.
#
# Cameras overlap acoustically (a voice in the living room is clearly audible
# in the corridor), so one spoken phrase wakes several sessions at once and
# each would answer over the others. Every session registers a claim carrying
# its RAW mic level (untouched by per-stream AGC, so it is a proximity proxy:
# the nearest microphone clips highest). After a short grace window the
# loudest claim wins; losers stand down silently (no pip, no Nanobot call).
# ---------------------------------------------------------------------------
_ARB_STATE: dict = {"claims": {}, "sent": {}, "cmd_sent": None}
# stream -> (timestamp, mean_raw_peak of last ~2s). Cross-room loudness map
# used to veto "distant source" wakes: a camera that hears the wake word
# much quieter than another room is hearing a muffled through-the-wall
# copy, which this TTS-trained model scores HIGHER than close live speech.
_ROOM_PEAKS: dict = {}
_ARB_LOCK: asyncio.Lock | None = None


def _arbiter_set_owner(stream: str, level: float) -> None:
    """Mark `stream` as the camera currently owning the interaction."""
    _ARB_STATE["owner"] = {"stream": stream, "ts": time.time(), "level": level}
    # A new owner starts a fresh interaction: the previous owner's dispatched
    # command no longer protects it from a proximity steal.
    _ARB_STATE["cmd_sent"] = None


def _arbiter_owner_active(exclude: str, max_age: float = 45.0):
    """Return (other_owner_active, owner_level) if another camera is mid-dialogue.

    Ownership is advisory and time-limited: `max_age` (45s) must span one
    full question->answer exchange, after which a stale entry no longer
    blocks new wakes (e.g. if the owning session died mid-reply without
    calling _arbiter_clear_owner).
    """
    o = _ARB_STATE.get("owner")
    if not o:
        return False, 0.0
    if time.time() - o["ts"] > max_age:
        return False, 0.0
    if o["stream"] == exclude:
        # The owner asking about itself is never "busy elsewhere" — this is
        # what lets the owning camera run its own follow-up dialogue.
        return False, 0.0
    return True, float(o.get("level", 0.0))


def _arbiter_clear_owner(stream: str) -> None:
    """Release the arbitration ownership only if `stream` currently holds it.

    Called when a winner goes back to standby or resets; ownership by a
    DIFFERENT camera is left untouched so a concurrent interaction is not
    silently stolen by a late reset.
    """
    o = _ARB_STATE.get("owner")
    if o and o.get("stream") == stream:
        _ARB_STATE["owner"] = None
        _ARB_STATE["cmd_sent"] = None


def _arbiter_mark_sent(key: tuple, stream: str) -> None:
    """Winner reports that it actually dispatched the command.

    Receipts are keyed by (arbitration bucket, stream) and expire after
    _SENT_TTL seconds; pruning happens here so the dict cannot grow
    without bound over a long uptime.
    """
    now = time.time()
    sent = _ARB_STATE["sent"]
    sent[(key, stream)] = now
    stale = [k for k, ts in sent.items() if now - ts > _SENT_TTL]
    for k in stale:
        del sent[k]


# How long a "winner dispatched" receipt stays meaningful. Must comfortably
# exceed the 2.2s fallback wait in CameraSession._cmd_fallback.
_SENT_TTL = 60.0


def _arbiter_sent_recently(key: tuple, stream: str, within: float = 3.0) -> bool:
    """True if `stream` reported a dispatch within the last `within` seconds.

    Falls back to a stream-only scan because the winner may hold a DIFFERENT
    bucket key than the loser that is asking (their claims landed in adjacent
    3s buckets but were merged into one verdict group). An exact
    (key, stream) lookup alone would miss such dispatches and trigger a
    bogus take-over of a command that was already answered.
    """
    now = time.time()
    ts = _ARB_STATE["sent"].get((key, stream))
    if ts and now - ts <= within:
        return True
    return any(
        s == stream and now - t <= within for (_k, s), t in _ARB_STATE["sent"].items()
    )


def _arb_lock() -> asyncio.Lock:
    """Lazily create the module-level arbitration lock.

    Built on first use because the lock binds to the event loop that
    creates it; constructing at import time would tie the arbiter to a
    loop that may not be the one uvicorn ends up running.
    """
    global _ARB_LOCK
    if _ARB_LOCK is None:
        _ARB_LOCK = asyncio.Lock()
    return _ARB_LOCK


async def _arbiter_submit(
    kind: str,
    stream: str,
    level: float,
    grace: float = 0.7,
    quality: float = 0.0,
    respect_owner: bool = False,
) -> tuple[bool, tuple, str]:
    """Register a claim and report whether `stream` won its group.

    `kind` separates concurrent event types ("wake" vs "cmd"). Claims land in
    coarse 3s time buckets; the verdict merges the bucket plus its predecessor
    and filters by recency, so groups straddling a bucket edge still compete
    together instead of both winning.
    """
    # An active interaction owned by another room wins outright. First
    # detector takes the dialogue; no mid-dialogue takeovers — they produced
    # double pips and duelling answers.
    if respect_owner:
        busy, _olvl = _arbiter_owner_active(stream)
        if busy:
            logger.info(
                f"⚖️ arbiter[{kind}]: '{stream}' stands down — "
                f"interaction owned elsewhere"
            )
            return False, ("owned", 0), "owner"

    now = time.time()
    bucket_key = (kind, int(now // 3))
    async with _arb_lock():
        _ARB_STATE["claims"].setdefault(bucket_key, {})[stream] = (
            now,
            float(level),
            float(quality),
        )
    await asyncio.sleep(grace)
    async with _arb_lock():
        # prune buckets that are entirely stale
        for k in [
            k
            for k, b in _ARB_STATE["claims"].items()
            if all(now - ts > 8 for ts, _, _q in b.values())
        ]:
            del _ARB_STATE["claims"][k]
        # Wake detections across rooms can stagger by seconds (different
        # detectors/debounce), so group them more generously than commands.
        span = 5.0 if kind == "wake" else 3.0
        # key=(stream) -> [max_level, max_quality]
        merged: dict[str, list[float]] = {}
        for bk in (
            (kind, bucket_key[1] - 1),
            (kind, bucket_key[1]),
            (kind, bucket_key[1] + 1),
        ):
            for st, (ts, lvl, q) in _ARB_STATE["claims"].get(bk, {}).items():
                if abs(now - ts) <= span:
                    cell = merged.setdefault(st, [0.0, 0.0])
                    cell[0] = max(cell[0], lvl)
                    cell[1] = max(cell[1], q)
        # Louder wins; on a clipping tie (both 32767) the richer transcript wins.
        winner = (
            max(merged.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
            if merged
            else stream
        )
        pretty = {k: (round(v[0]), round(v[1])) for k, v in merged.items()}
        logger.info(f"⚖️ arbiter[{kind}]: {pretty} -> '{winner}'")
        # Tuple so losers know WHO won (for the take-over-if-silent fallback).
        return winner == stream, bucket_key, winner


# Global: while any TTS playback is running (ESP32 or camera), neither path
# may send to nanobot — kills echo cascade (camera hears ESP32 TTS, resends).
GLOBAL_TTS_UNTIL = 0.0


class _PauseEndpoint:
    """Level-based utterance endpointing, independent of the Silero verdict.

    WHY THIS EXISTS
    An utterance is supposed to end when the speaker stops: Silero reports
    speech=False for a few frames and _vad_silence_limit (10 x 160 ms) commits
    it. In the living room that NEVER HAPPENS — Silero reported speech=True for
    the whole capture (0 speech=False in 15 live minutes, and every
    "VAD rms=... speech=True" line sits above consec=250), so every command ran
    to the 7 s duration cap instead. Measured 04.10.2026: wake at 19:20:10.1,
    Whisper at 19:20:17.3 — 7.2 s, of which the user's «выключи свет» was the
    first 1.5.

    Lowering the cap is NOT the fix: the cap is load-bearing for long commands.
    «я просил включить следующую серию черного зеркала» transcribes correctly at
    7.04 s and is chopped at 3.5 s.

    WHAT IT DOES
    The gap between words is a DROP IN LEVEL, whatever the VAD thinks. So the
    endpoint watches the level envelope against a trailing reference and ends
    the utterance after a run of dip frames — but only once enough speech has
    been seen, so the gap right after «компьютер» does not dispatch a bare
    wake word.

    WHY A TRAILING REFERENCE AND NOT A FLOOR
    With the television on there is no quiet room to measure a floor against,
    and an absolute floor is unreachable there. The reference is a high
    percentile of the trailing window (NOT the max: one door slam must not
    redefine it, and NOT the min: sustained noise would pull it down until
    everything reads as a pause). If the reference cannot be established —
    speech quieter than the background — no pause is ever reported and the
    utterance falls back to the duration cap. That is the safe direction:
    slower, never chopped.
    """

    def __init__(
        self,
        ratio: float = 0.55,
        run_frames: int = 6,
        min_speech_frames: int = 5,
        ref_frames: int = 12,
        anchor_frames: int = 4,
    ):
        # 0 from a config field means "unset" — the env is optional and a
        # half-filled override must not silently zero a threshold.
        self.ratio = ratio if ratio > 0 else 0.55
        self.run_frames = run_frames if run_frames > 0 else 6
        self.min_speech_frames = (
            min_speech_frames if min_speech_frames > 0 else 5
        )
        self.ref_frames = ref_frames if ref_frames > 0 else 12
        self.anchor_frames = anchor_frames if anchor_frames > 0 else 4
        self.reset()

    def reset(self) -> None:
        self._ref: list[float] = []
        self._utt: list[float] = []
        self._anchor = 0.0
        self._run = 0
        self._speech_frames = 0

    def feed(self, rms: float) -> tuple[str, str]:
        """Classify one 160 ms frame.

        Returns ("speech"|"pause"|"end", detail). The detail carries the numbers
        that produced the verdict — they are the only way to tune this from
        field logs instead of from a guess.
        """
        rms = float(rms)
        self._ref.append(rms)
        if len(self._ref) > self.ref_frames:
            self._ref.pop(0)
        # 80th percentile of the trailing window: high enough that ordinary
        # speech sets it, low enough that a single transient cannot. This decides
        # whether the speaker PAUSED — a responsive, windowed question.
        ref = float(np.percentile(self._ref, 80))

        # The speech floor is a SEPARATE, utterance-scoped number: the median of
        # the first few frames, then frozen.
        #
        # It has to be utterance-scoped. A windowed reference collapses to the
        # room floor after ~2 s of silence, the pause test stops matching, and
        # from then on every quiet frame looks like speech — which is the one
        # thing min_speech_frames exists to exclude. Measured failure of the
        # windowed version: a bare «компьютер» (3 speech frames) followed by a
        # pause drove _speech_frames to 50 on silence alone.
        #
        # Frozen after the first few frames so a single transient — a door slam
        # — cannot redefine "speech" for the rest of the sentence. The cost is
        # that a user who starts very quietly and gets louder is under-counted,
        # which degrades to the duration cap: the safe direction.
        if self._anchor == 0.0 and len(self._utt) < self.anchor_frames:
            self._utt.append(rms)
            if len(self._utt) >= self.anchor_frames:
                self._anchor = float(np.median(self._utt))
        floor = self._anchor * _ANCHOR_FRAC

        is_pause = ref > 0.0 and rms < ref * self.ratio
        if is_pause:
            self._run += 1
        else:
            self._run = 0
            if floor > 0.0 and rms >= floor:
                self._speech_frames += 1

        detail = (
            f"rms={rms:.4f} ref={ref:.4f} floor={floor:.4f} "
            f"run={self._run} speech={self._speech_frames}"
        )
        # Enough speech FIRST: the gap right after the wake word is a gap
        # between words, not the end of a sentence.
        if (
            self._run >= self.run_frames
            and self._speech_frames >= self.min_speech_frames
        ):
            self.reset()
            return "end", detail
        return ("pause" if is_pause else "speech"), detail


@dataclass
class CameraConfig:
    """Everything one CameraSession needs: go2rtc endpoints, STT/TTS/Nanobot
    URLs, wake-word tuning and the go2rtc re-register hints. Defaults match
    the in-home services; main.start_camera_sessions() fills them from env.
    """
    stream_name: str
    go2rtc_host: str = "192.168.22.102"
    go2rtc_port: int = 1984
    http_session: aiohttp.ClientSession | None = None
    chat_id: str | None = None
    whisper_url: str = "http://192.168.22.111:8000/v1/audio/transcriptions"
    whisper_model: str = "koekaverna/faster-whisper-podlodka-turbo"
    tts_url: str = "http://edge_tts:5050/v1/audio/speech"
    tts_voice: str = "ru-RU-SvetlanaNeural"
    tts_api_key: str = ""
    nanobot_url: str = "ws://nanobot:8765/"
    nanobot_token: str = "token"
    speaker_id_url: str = "http://192.168.22.102:8001/identify"
    aec_block_ms: int = 1500
    wake_keyword: str = "компьютер"
    # Per-room openWakeWord threshold. 0 => the historical default (kitchen
    # 0.40, everything else 0.30). A room-specific model has its own score
    # distribution, so its operating point has to be measured, not inherited:
    # the livingroom head saturates its positives at 1.000 and still fires on
    # nothing else at 0.70 (36/36 hits, 0/42 false), while the library head it
    # replaced sat at 0.006 median and reached only 2/36 even at 0.10.
    wake_threshold: float = 0.0
    # Path to a vosk model directory. When set, THIS room detects the wake word
    # by DECODING it (vosk_wake.VoskWakeMatcher) instead of scoring it with
    # openWakeWord, and the acoustic head is skipped entirely. Empty = acoustic
    # as before, so corridor/kitchen keep their tuned openWakeWord behaviour.
    # See vosk_wake's module docstring for the measured numbers: 94 % recall and
    # 0 false accepts on a held-out television tail, where an acoustic head
    # trained on this room managed 229 false/hour.
    wake_vosk_model: str = ""
    wake_timeout: float = 15.0
    vad: object = None
    wakeword_model_path: str = ""
    activation_wav_path: str = "/app/activation.wav"
    # Source RTSP URL used to (re)register this stream with go2rtc when its
    # audio track is lost after a camera reboot. If empty, the healer tries to
    # derive it from go2rtc's own /api/streams listing.
    go2rtc_source_url: str = ""
    # OpenIPC /play_audio endpoint (e.g. http://192.168.22.241/play_audio) plus
    # its HTTP basic credentials. When set, TTS is POSTed there as raw 48 kHz
    # mono s16le — the only path that plays at the right pitch on this
    # firmware. Empty => fall back to the (pitch-broken) go2rtc backchannel.
    play_audio_url: str = ""
    play_audio_user: str = ""
    play_audio_password: str = ""
    # Consecutive ffmpeg stalls before triggering a go2rtc stream re-register.
    heal_stalls: int = 3
    # Level-based utterance endpointing for this room. Off by default because
    # it replaces a measured behaviour in a live pipeline; enable per room once
    # its end-of-utterance log lines have been read on real audio. When off, an
    # utterance ends only on VAD silence or the duration cap — see
    # _PauseEndpoint for why that costs 7 s in a room with the TV on.
    pause_endpoint: bool = False
    # Endpoint tuning, per room, as ENV rather than constants: the whole point
    # of logging rms/ref/floor/run/speech is to tune from what the room actually
    # does, and tuning must not cost a rebuild + redeploy per iteration. 0 => the
    # _PauseEndpoint default.
    pause_ratio: float = 0.0
    pause_run_frames: int = 0
    pause_min_speech_frames: int = 0
    # Peak the reply is normalised to before it goes to the camera speaker.
    # 0 => _TTS_TARGET_PEAK_DEFAULT.
    #
    # Exists because of an UNRESOLVED field report of crackling playback
    # (05.10.2026). Everything on this side of the wire was measured clean:
    # the PCM peaks at 17091 (52 % of full scale) with 0 samples at the rail,
    # the 24 kHz -> 48 kHz resample really resamples, and the camera hears
    # exactly 1000 Hz from a 1 kHz tone whatever rate is declared. So the
    # remaining suspect is the camera's own output stage, and the microphone
    # cannot measure it: input peaks of 20000, 10000 and 4000 were all
    # captured as rms ~26 300, peak 32768, ~40 % of samples pinned at the
    # rail — the captured level does not follow the input at all.
    #
    # The default drives a small speaker at 61 % of full scale, which is a
    # plausible way to clip a cheap amplifier. Making it a per-room ENV means
    # the level is found by listening and restarting, with no rebuild per
    # iteration — the same rule the pause endpoint follows.
    tts_target_peak: int = 0
    # Follow-up listening windows, per room, as ENV rather than constants:
    # 0 => CAMERA_DIALOGUE_QUESTION_S / CAMERA_DIALOGUE_STATEMENT_S. Mirrors the
    # satellite's STANDBY_TIMEOUT_QUESTION 30 s / STANDBY_TIMEOUT_STATEMENT 10 s
    # so a camera and an ESP32 answer a follow-up on the same terms.
    dialogue_question_s: float = 0.0
    dialogue_statement_s: float = 0.0


class CameraSession:
    """One bidirectional voice session for a single go2rtc camera stream.

    Owns the full loop: RTSP mic feed -> echo guards -> VAD/wake gate ->
    cross-camera arbiter -> Whisper STT -> LLM backend -> TTS back out over
    the WebRTC sendonly track, plus the self-heal path that re-registers a
    dead go2rtc stream. Lifetime: start() -> _run() reconnect loop -> stop().
    """

    def __init__(self, config: CameraConfig, backend=None):
        if not re.match(r"^[a-zA-Z0-9_-]+$", config.stream_name):
            raise ValueError(f"Invalid stream_name: {config.stream_name}")
        self.stream_name = config.stream_name

        # LLM backend (BaseLLMBackend from backends.py). None => legacy Nanobot path.
        # This USED to be a hard `self.backend = None`, which silently threw
        # away the backend main.start_camera_sessions() builds (Cascade ->
        # jev-router -> Hermes) and sent every camera command down the legacy
        # Nanobot WebSocket. Nanobot is retired, so that path 401s on connect
        # and the camera goes mute after the wake word: observed 03.10.2026 —
        # "Wake word (openWakeWord)" + "Whisper OK: 'компьютер компьютер'"
        # followed by "Nanobot error: Cannot connect to host localhost:8765".
        self.backend = backend
        self.go2rtc_host = config.go2rtc_host
        self.go2rtc_port = config.go2rtc_port
        self.chat_id = config.chat_id or self._make_chat_id(config.stream_name)
        self.http_session = config.http_session
        self.whisper_url = config.whisper_url
        self.whisper_model = config.whisper_model
        self.tts_url = config.tts_url
        self.tts_voice = config.tts_voice
        self.tts_api_key = config.tts_api_key
        self._nanobot_url = config.nanobot_url.rstrip("/")
        self._nanobot_token = config.nanobot_token
        self._speaker_id_url = config.speaker_id_url
        self.aec_block_ms = config.aec_block_ms
        self.wake_keyword = config.wake_keyword
        self._wake_timeout = config.wake_timeout

        self.go2rtc_source_url = config.go2rtc_source_url
        self._heal_stalls = config.heal_stalls
        # Rate-watchdog knobs: below 50% of real time for 20 s counts as starved
        # (measured healthy 100-170%, measured broken 3-6%), and after 3 heal
        # attempts we stop pretending and report a camera reboot is needed.
        self._min_audio_rate = 0.5
        self._starve_restarts = 3
        self._wake_threshold = config.wake_threshold
        # vosk wake-word config, read in __init__ because _init_audio_processing()
        # builds the matcher and has no access to the CameraConfig object.
        self._wake_vosk_model = config.wake_vosk_model
        # Level-based utterance endpointing for this room (see _PauseEndpoint).
        # Read here because _init_audio_processing() has no CameraConfig.
        self._pause_endpoint_on = config.pause_endpoint
        self._pause_ratio = config.pause_ratio
        self._pause_run_frames = config.pause_run_frames
        self._pause_min_speech_frames = config.pause_min_speech_frames
        # 0 => the default, so a half-filled override cannot mute the speaker.
        self._tts_target_peak = config.tts_target_peak or _TTS_TARGET_PEAK_DEFAULT
        # 0 => the default, so a half-filled override cannot close the window
        # the user is trying to lengthen.
        self._dialogue_question_s = config.dialogue_question_s or CAMERA_DIALOGUE_QUESTION_S
        self._dialogue_statement_s = config.dialogue_statement_s or CAMERA_DIALOGUE_STATEMENT_S
        # Always defined so _vad_process() can test it without a hasattr guard,
        # and so tests that build a session via __new__ cannot trip over it.
        self._vosk_wake = None
        # diagnostic counters for the "why did the live room stay silent"
        # question; see the vosk diag log line in _vad_process
        self._vosk_chunks = 0
        self._vosk_suppressed = 0
        self._last_wake_fired_at = 0.0
        self._vosk_last_peak = 0
        self._vosk_last_log = 0.0
        # Audio-rate watchdog. The stall handlers only catch total timeouts; a
        # camera whose mic driver wedges keeps delivering a trickle and the
        # wake word just silently never fires.
        self._rate_bytes = 0
        self._rate_start = 0.0
        self._rate_check = 0.0
        self._rate_ratio = 1.0
        self._rate_starved = 0
        self._play_audio_url = config.play_audio_url
        # Pre-built Authorization header: aiohttp.BasicAuth is deprecated in
        # aiohttp 4, and /play_audio is the only place we do HTTP basic auth.
        self._play_audio_headers = {}
        if config.play_audio_url and config.play_audio_user:
            token = base64.b64encode(
                f"{config.play_audio_user}:{config.play_audio_password or ''}".encode()
            ).decode()
            self._play_audio_headers["Authorization"] = f"Basic {token}"

        self._vad = config.vad
        if self._vad is not None:
            self._vad.energy_threshold = 0.005
            self._vad.rms_noise_floor = 0.005
            self._vad.rms_alpha = 0.0
        self._wakeword_model_path = config.wakeword_model_path or ""
        self._activation_wav_path = config.activation_wav_path

        self._init_state()
        self._init_audio_processing()

    def _init_state(self):
        self._wake_detected = False
        self._wake_expires = 0.0
        self._ww_consec = 0  # consecutive openWakeWord fires (debounce)
        self._ww_recent: list[float] = []  # last 3 oww scores (sliding debounce)
        # Gray-zone wake handling: when gates hold a decent oww score, do not
        # pip — instead let the next utterance reach Whisper, and a live
        # «компьютер» in the transcript confirms the wake (or drops it).
        self._stt_confirm_until = 0.0  # wall clock: STT-confirmation window
        self._confirm_saw_text = False  # a transcript arrived during window
        self._tts_play_end = 0.0  # approx wall clock: speaker finishes
        self._veto_until = 0.0  # wall clock: sticky distant-veto
        self._wake_greeting_task = None
        # True while the AUTO-greeting (bare wake, no command yet) is being
        # spoken: shortens the ownership/follow-up window so other rooms do
        # not stay deaf for half a minute when the user walks away.
        self._auto_greeting = False
        self._last_auto_greet_ts = 0.0
        self._last_tts_reply = ""  # normalized text we last spoke (echo guard)
        self._wake_greeting_delay = 5.0
        # True while a follow-up window WE opened is still live. The player's
        # finally block uses it to decide whether it may close the window, and
        # it must be per-TURN: leftover True from an earlier turn would keep
        # the mic open forever.
        self._followup_open = False
        # Wall clock of the most recent VAD "speech" frame. The greeting
        # asks "is the user talking?", and `_vad_has_speech` answers that
        # wrongly while an utterance is in flight (see _wake_greeting).
        self._last_speech_at = 0.0
        self._audio_epoch = 0.0

        self._pc: RTCPeerConnection | None = None
        self._out_track: AIVoiceOutputTrack | None = None
        self._last_transcript: str = ""
        self._speaking = False
        self._speaking_until = 0.0
        # Suppress the wake *model* for the full delayed-echo window after TTS
        # (the AI's own "компьютер" echoes back 8-15s later and would re-trigger
        # the model). This is separate from _speaking_until (which hard-mutes the
        # mic) so the user's real follow-up still reaches the mic.
        self._wake_suppress_until = 0.0
        self._stall_count = 0
        self._last_heal_ts = 0.0
        # Consecutive ffmpeg connections that delivered ZERO audio. This is a
        # different failure from a stall: a stall needs 20 s of silence AFTER
        # bytes have flowed, so when go2rtc has no producer for the stream the
        # stall counter never moves and the healer never runs. Measured
        # 04.10.2026 20:18-20:19: 25+ reconnects every 3.2 s, no AUDIO STARVED,
        # no heal, no reason logged anywhere.
        self._dead_cycles = 0
        # Same thing but NOT reset by healing: how many zero-byte connections
        # have happened in a row in total. Escalation needs this — resetting
        # `_dead_cycles` after each heal made the "declared dead" branch
        # unreachable, because it can never climb past `_heal_stalls`.
        self._dead_total = 0
        # Reconnect backoff. 3 s normally; raised once the room is declared dead
        # so a hopeless loop does not bury everything else in the log.
        self._audio_sleep = 3.0

        # Stats and timers
        self._last_attention = 0.0
        self._last_feed_log = 0.0
        self._last_rms_log = 0.0

        self._wake_cmd_sent = False
        self._last_uid = None
        self._last_uid_ts = 0.0
        self._last_fire_ts = 0.0
        self._last_decay_ts = 0.0
        self._vad_max_duration = 7.0
        # How each utterance was terminated since boot, so one log line answers
        # "is the endpoint working or is everything still hitting the cap?".
        # Lives here rather than in _init_audio_processing because it is state,
        # not wiring.
        self._end_reason: dict[str, int] = {}
        self._vad_lock = asyncio.Lock()
        self._processing_utterance = False

        self._tasks: set[asyncio.Task] = set()
        self._stopped = asyncio.Event()

        self._attention_played = False

    def _init_audio_processing(self):
        # Echo reference ring buffer (16 kHz float32) — holds recently played
        # TTS audio so the mic feed can be checked for our own echo (the camera
        # speaker returns via the mic 8-15s later). Cross-correlation against
        # this buffer at the expected echo delay lets us drop the echo precisely
        # instead of blind time-suppression that also eats the user's answer.
        self._tts_ring_len = 40 * 16000
        self._tts_ring = np.zeros(self._tts_ring_len, dtype=np.float32)
        self._tts_total = 0  # monotonic samples written
        self._echo_corr_threshold = 0.4
        self._last_echo_log = 0.0

        self._bg_window: list[int] = []
        self._resample_buf = bytearray()

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
        self._vad_silence_limit = 10
        # None unless this room enabled pause endpointing. Every use is a
        # cheap None test, and with it off the utterance is ended by exactly
        # the two terminators that existed before (VAD silence, duration cap).
        self._endpoint = (
            _PauseEndpoint(
                ratio=self._pause_ratio,
                run_frames=self._pause_run_frames,
                min_speech_frames=self._pause_min_speech_frames,
            )
            if self._pause_endpoint_on
            else None
        )
        self._ww_ring: list[bytes] = []
        # Raw (pre-AGC) chunk peaks, proximity proxy for cross-camera arbitration.
        self._recent_peaks: list[int] = []
        self._last_utt_peak = 0
        self._skipped_noise = 0
        # Adaptive wake threshold (false-wake immunity for the noisy corridor).
        #
        # The openWakeWord score of a genuine "компьютер" varies with distance
        # and background level (observed 0.55-0.95), so a fixed threshold is
        # either deaf or jumpy. Instead it self-tunes:
        #   * every fire NOT followed by a command ("Пип был зря") raises the
        #     bar by +0.07 — the environment just taught us it was wrong;
        #   * a successful command resets it to the sensitive base immediately;
        #   * after 2.5 quiet minutes it decays by -0.02/min back to base, so
        #     sensitivity is never lost for long;
        #   * hard floor/base 0.58, hard ceiling 0.70;
        #   * any single-chunk score >=0.88 bypasses everything (a loud,
        #     confident "компьютер" can never be locked out).
        # Kitchen's weak mic link tops out at ~0.55-0.57 on a real "компьютер"
        # (measured live) — its base threshold is lowered so the room stays
        # usable; false-fire protection there still comes from the 2-chunk
        # debounce + cross-camera arbitration.
        self._ww_base_thresh = self._wake_threshold or (
            0.40 if self.stream_name == "kitchen" else 0.30
        )
        self._ww_thresh = self._ww_base_thresh

        # vosk decoder replaces the acoustic head for this room when configured.
        # The model is a process-wide singleton (88 MB); only the recogniser is
        # per stream. A failure here must not take the room down — fall back to
        # the acoustic path, which is deaf for this word but still handles the
        # rest of the pipeline.
        if self._wake_vosk_model:
            try:
                from vosk_wake import VoskWakeMatcher

                self._vosk_wake = VoskWakeMatcher(self._wake_vosk_model)
                logger.info(
                    f"[{self.stream_name}] vosk wake-word: "
                    f"{self._wake_vosk_model} "
                    f"(openWakeWord отключён для этой комнаты)"
                )
            except Exception as e:
                logger.warning(
                    f"[{self.stream_name}] vosk недоступен ({e}), "
                    f"остаюсь на openWakeWord"
                )
                self._vosk_wake = None

    @staticmethod
    def _make_chat_id(name: str) -> str:
        h = hashlib.sha256(name.lower().encode()).hexdigest()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    async def start(self):
        """Boot the engine and spawn the WebRTC + RTSP background tasks.

        Both tasks are registered in `self._tasks` with a discard callback
        so `stop()` can cancel them and re-start does not leak the old ones.
        """
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
        # Startup race: the config bind-mount may not be visible for the
        # first seconds after container start (observed 22:43 — all three
        # cameras initialized with a dead engine and never retried).
        deadline = time.time() + 90.0
        while True:
            try:
                await self._init_engine_once()
                return
            except Exception as e:
                if time.time() >= deadline or self._stopped.is_set():
                    logger.error(
                        f"[{self.stream_name}] engine init failed permanently: {e}"
                    )
                    return
                logger.warning(
                    f"[{self.stream_name}] engine init deferred ({e}); retrying"
                )
                await asyncio.sleep(3.0)

    async def _init_engine_once(self):
        if True:  # exceptions propagate to the retry wrapper on purpose
            model_path = self._wakeword_model_path or "config/computer.onnx"
            if not os.path.exists(model_path):
                # Bare name (e.g. 'hey_jarvis') -> built-in pretrained model
                # shipped inside the openwakeword package.
                import openwakeword

                cand = os.path.join(
                    os.path.dirname(openwakeword.__file__),
                    "resources",
                    "models",
                    model_path,
                )
                if not cand.endswith(".onnx"):
                    cand += ".onnx"
                if os.path.exists(cand):
                    model_path = cand
                else:
                    import glob as _glob

                    hits = sorted(
                        _glob.glob(
                            os.path.join(
                                res_dir := os.path.dirname(cand),
                                os.path.basename(cand)[:-5] + "*.onnx",
                            )
                        )
                    )
                    if hits:
                        model_path = hits[0]
            await asyncio.to_thread(
                self._engine.initialize_models,
                model_path,
            )

    async def stop(self):
        """Set the stop flag, cancel every background task, close the PC."""
        self._stopped.set()
        for t in list(self._tasks):
            t.cancel()
        await self._cleanup()
        logger.info(f"[{self.stream_name}] CameraSession stopped")

    async def _cleanup(self):
        """Close (and forget) the WebRTC peer connection, swallowing errors.

        The PC reference is cleared FIRST so a concurrently running
        _connect() cannot accidentally close a freshly created one.
        """
        pc = self._pc
        self._pc = None
        if pc:
            try:
                await pc.close()
            except Exception:
                pass

    async def _run(self):
        """Reconnect supervisor: keep _connect() alive until stop() is called.

        Any connect failure (go2rtc down, camera rebooting, SDP mismatch)
        is logged and retried after 5 s instead of killing the session.
        """
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
    def _parse_candidate(value) -> RTCIceCandidate | None:
        """Parse an ICE candidate from go2rtc's dict or bare-string form.

        Accepts both shapes the signalling channel may deliver
        (`{"candidate": "candidate:...", "sdpMid": ..., ...}` and the raw
        SDP attribute string). Returns None for malformed input instead of
        raising — a bad candidate must not tear down the whole session.
        """
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
        """Parse `value` and hand it to the PC; log-and-ignore on failure.

        Late/trickle candidates regularly arrive after the PC is closed or
        are rejected by aiortc — both are normal, never fatal.
        """
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

        self._out_track = AIVoiceOutputTrack(sample_rate=TTS_PLAY_RATE)
        self._pc.addTransceiver(self._out_track, direction="sendonly")
        self._pc.addTransceiver("audio", direction="recvonly")
        logger.info(
            f"[{self.stream_name}] created sendonly track + recvonly transceiver"
        )

        @self._pc.on("track")
        async def on_track(track):
            logger.info(f"[{self.stream_name}] Incoming audio track: {track.kind}")
            await self._recv_audio(track)

        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        offer_sdp = self._filter_sdp(offer.sdp)

        async with aiohttp.ClientSession() as sig:
            async with sig.ws_connect(sig_url) as ws:
                await ws.send_json(
                    {
                        "type": "webrtc/offer",
                        "value": offer_sdp,
                    }
                )
                answered = False
                announced = False
                ticks = 0
                wait_ticks = 0
                while not self._stopped.is_set():
                    try:
                        msg = await ws.receive(timeout=3.0)
                    except asyncio.TimeoutError:
                        if not answered:
                            # go2rtc accepted the WS but never answered the
                            # offer (seen after Frigate restarts). Retry with
                            # a fresh peer instead of hanging forever.
                            wait_ticks += 1
                            if wait_ticks >= 8:
                                logger.warning(
                                    f"[{self.stream_name}] no WebRTC answer in 24s, reconnecting"
                                )
                                return
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
                        logger.info(
                            f"[{self.stream_name}] WebRTC ready (bidirectional)"
                        )

                    if answered:
                        st = self._pc.connectionState
                        if st == "connected" and not announced:
                            announced = True
                            logger.info(f"[{self.stream_name}] WebRTC connected")
                            asyncio.create_task(self._delayed_attention())
                        elif st == "failed":
                            logger.warning(
                                f"[{self.stream_name}] WebRTC failed, retrying"
                            )
                            return

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
        """Resample s16le mono PCM from `rate` to the internal 16 kHz.

        Uses scipy's polyphase resampler (band-limited, no audible aliasing
        on speech) with the gcd-reduced up/down factors.
        """
        from scipy.signal import resample_poly

        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        g = np.gcd(16000, rate)
        up, down = 16000 // g, rate // g
        return resample_poly(arr, up, down).astype(np.int16).tobytes()

    async def _account_audio_rate(self, nbytes: int) -> bool:
        """Accumulate delivered audio against wall clock; return True if the
        stream should be restarted.

        Separate from `_rtsp_audio_loop` so the recovery decision is testable
        without driving ffmpeg, and so the "restart" signal cannot be confused
        with task cancellation (which is fatal here, see the call site).
        """
        now = time.time()
        self._rate_bytes += nbytes
        self._rate_start = self._rate_start or now
        if now - self._rate_check < 20.0:
            return False

        span = now - self._rate_start
        ratio = (
            self._rate_bytes / (span * 16000 * 2) if span > 0 else 1.0
        )
        self._rate_ratio = ratio
        self._rate_bytes = 0
        self._rate_start = now
        self._rate_check = now

        if ratio >= self._min_audio_rate:
            if self._rate_starved:
                logger.info(
                    f"[{self.stream_name}] audio rate recovered: "
                    f"{ratio*100:.0f}% of real time"
                )
            self._rate_starved = 0
            return False

        self._rate_starved += 1
        logger.warning(
            f"[{self.stream_name}] ⚠ AUDIO STARVED: {ratio*100:.0f}% of real "
            f"time (min {self._min_audio_rate*100:.0f}%), attempt "
            f"{self._rate_starved}/{self._starve_restarts}"
        )
        if self._rate_starved >= self._starve_restarts:
            # ffmpeg restarts and stream re-registration did not help. Measured
            # root cause on this hardware: the camera's ai0_P0_MAIN thread
            # wedges in CamOsTcondTimedWait and aio_dma drops to ~40/s, and only
            # a CAMERA REBOOT recovers it (ONVIF Reboot is not implemented on
            # OpenIPC). Say so instead of sitting silent.
            logger.error(
                f"[{self.stream_name}] 🔴 CAMERA AUDIO DEAD — ffmpeg restart and "
                f"stream re-register did not help ({ratio*100:.0f}% of real "
                f"time). Reboot {self.stream_name} manually; the mic driver "
                f"wedges and only a power cycle clears it."
            )
            self._rate_starved = 0
            return False

        try:
            await self._heal_go2rtc_stream()
        except Exception as e:
            logger.warning(f"[{self.stream_name}] starve heal error: {e}")
        return True

    async def _recv_audio(self, track):
        """Pull audio frames off the go2rtc WebRTC track until stop().

        A 10 s recv() timeout keeps the loop responsive when the producer
        goes quiet, and a non-audio frame (video keyframe, application
        data) is skipped rather than crashing the consumer.
        """
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

    async def _heal_go2rtc_stream(self) -> bool:
        """Force go2rtc to recreate this stream's producer.

        After a camera reboot go2rtc sometimes reconnects the producer without
        the audio track (serves video-only SDP), which makes ffmpeg stall.
        Re-registering the stream forces a fresh producer that re-negotiates
        audio. Rate-limited to once per minute.
        """
        now = time.time()
        if now - self._last_heal_ts < 60.0:
            return False
        self._last_heal_ts = now

        # Resolve the source URL to (re)register.
        source_url = self.go2rtc_source_url
        derived = False
        if not source_url:
            try:
                base = f"http://{self.go2rtc_host}:{self.go2rtc_port}"
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=8)
                ) as s:
                    async with s.get(f"{base}/api/streams") as r:
                        if r.status == 200:
                            data = await r.json()
                            prods = data.get(self.stream_name, {}).get("producers", [])
                            for p in prods:
                                u = p.get("url", "")
                                if u and u.startswith("rtsp://"):
                                    source_url = u
                                    derived = True
                                    break
            except Exception as e:
                logger.warning(
                    f"[{self.stream_name}] heal: failed to query go2rtc streams: {e}"
                )
            # go2rtc's own listing is a LOSSY copy of the configured source:
            # it strips the credentials and the "#backchannel=1" fragment
            # (rtsp://root:pass@cam/stream=0#backchannel=1 comes back as
            # rtsp://cam/stream=0). Re-registering from it therefore either
            # 401s (no credentials) or, worse, succeeds without the speaker
            # track and this room stops answering out loud from the first
            # heal on. Say so loudly instead of failing quietly later.
            if derived and source_url:
                logger.warning(
                    f"[{self.stream_name}] heal: using the URL go2rtc reports "
                    f"({source_url}), which carries no credentials and no "
                    f"#backchannel=1 — set GO2RTC_SOURCE_URL_{self.stream_name.upper()} "
                    f"to the verbatim go2rtc config entry"
                )

        if not source_url:
            logger.warning(
                f"[{self.stream_name}] heal: no source URL (set GO2RTC_SOURCE_URL), cannot re-register"
            )
            return False

        base = f"http://{self.go2rtc_host}:{self.go2rtc_port}"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as s:
                # Best-effort remove the broken entry so POST creates a fresh one.
                # go2rtc quirk: the DELETE query param is named "src" but holds
                # the stream NAME ("?src=corridor"); "?name=" silently no-ops.
                try:
                    async with s.delete(
                        f"{base}/api/streams", params={"src": self.stream_name}
                    ) as r:
                        logger.info(
                            f"[{self.stream_name}] heal: DELETE stream -> HTTP {r.status}"
                        )
                except Exception as e:
                    logger.info(f"[{self.stream_name}] heal: DELETE skipped ({e})")
                # Re-register with the source URL. go2rtc-under-Frigate rejects
                # POST (400); PUT creates or updates the entry (200).
                async with s.put(
                    f"{base}/api/streams",
                    params={"name": self.stream_name, "src": source_url},
                ) as r:
                    body = (await r.read()).decode(errors="replace")[:200]
                    logger.info(
                        f"[{self.stream_name}] heal: re-registered stream -> HTTP {r.status} {body}"
                    )
                    return r.status in (200, 201)
        except Exception as e:
            logger.warning(f"[{self.stream_name}] heal: re-register failed: {e}")
            return False

    async def _rtsp_audio_loop(self):
        rtsp_url = f"rtsp://{self.go2rtc_host}:8554/{self.stream_name}?audio=copy"
        logger.info(
            f"[{self.stream_name}] RTSP audio loop starting (ffmpeg): {rtsp_url}"
        )
        while not self._stopped.is_set():
            proc = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-rtsp_transport",
                    "tcp",
                    "-i",
                    rtsp_url,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-f",
                    "s16le",
                    "pipe:1",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._audio_epoch = time.time()
                frame_bytes = 1280 * 2
                delivered = 0
                try:
                    while not self._stopped.is_set():
                        chunk = await asyncio.wait_for(
                            proc.stdout.readexactly(frame_bytes), timeout=20.0
                        )
                        if not chunk:
                            break
                        self._stall_count = 0
                        delivered += len(chunk)
                        # Rate accounting: the stall handlers above only fire on
                        # TIMEOUTS, i.e. when ffmpeg stops delivering altogether.
                        # They cannot see a stream that keeps delivering but at a
                        # fraction of real time — which is exactly what happened
                        # on 04.10.2026 and it went unnoticed for over an hour,
                        # with the wake word simply never firing.
                        #
                        # `break` (not an exception) is the recovery signal: it
                        # leaves the inner read loop, the `finally` reaps ffmpeg
                        # and control falls through to the reconnect below.
                        # Raising asyncio.CancelledError here would be caught by
                        # the `except asyncio.CancelledError: break` of THIS
                        # task's own handler and terminate the whole loop for
                        # good — the session would never take audio again even
                        # after the camera recovered, while still looking
                        # healthy (WebRTC connected, keepalive ticking,
                        # /health ok).
                        if await self._account_audio_rate(len(chunk)):
                            break
                        await self._feed_audio(chunk)
                finally:
                    # Runs on EVERY exit from the read loop, including the
                    # IncompleteReadError raised by the very first read — which
                    # is the case that matters. A connection that delivered
                    # nothing never started streaming at all.
                    if not self._stopped.is_set():
                        await self._audio_cycle_done(proc, delivered)
            except asyncio.IncompleteReadError:
                logger.info(f"[{self.stream_name}] RTSP ffmpeg stream ended")
            except asyncio.TimeoutError:
                self._stall_count += 1
                logger.warning(
                    f"[{self.stream_name}] RTSP ffmpeg stalled ({self._stall_count}/"
                    f"{self._heal_stalls}), restarting"
                )
                if self._stall_count >= self._heal_stalls:
                    self._stall_count = 0
                    logger.warning(
                        f"[{self.stream_name}] persistent stalls -> healing go2rtc stream"
                    )
                    try:
                        await self._heal_go2rtc_stream()
                    except Exception as e:
                        logger.warning(f"[{self.stream_name}] heal error: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[{self.stream_name}] RTSP audio error: {e}")
            finally:
                if proc:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    # Bounded reap: asyncio occasionally reports 'exit status
                    # already read' and an unbounded proc.wait() then blocks
                    # FOREVER — the audio loop froze silently and the camera
                    # went deaf until a container restart (18:25 trace).
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except Exception:
                        pass
            if not self._stopped.is_set():
                # Backoff, not a fixed 3 s: once the room is declared dead a
                # 3 s retry loop only buries every other line in the log.
                logger.info(
                    f"[{self.stream_name}] RTSP audio reconnecting in "
                    f"{self._audio_sleep:.0f}s..."
                )
                await asyncio.sleep(self._audio_sleep)

    async def _audio_cycle_done(self, proc, delivered: int) -> None:
        """Judge one ffmpeg connection now that it has ended.

        Two cases, and only the first one existed before:

        * bytes flowed -> a normal end or a stream restart; nothing to do.
        * zero bytes   -> the connection never started streaming. go2rtc had
          no producer for this stream, so ffmpeg was handed nothing and
          exited. Counted SEPARATELY because the stall counter cannot see
          it: a stall needs 20 s of silence AFTER audio has been flowing,
          so with no audio ever arriving _stall_count stays 0 forever and
          _heal_go2rtc_stream() is never called. Measured 04.10.2026 20:18:
          exactly that — 25+ reconnects, no AUDIO STARVED, no heal, no
          explanation anywhere.
        """
        if delivered > 0:
            self._dead_cycles = 0
            self._dead_total = 0
            self._audio_sleep = 3.0
            return

        self._dead_cycles += 1
        self._dead_total += 1
        err = await self._read_ffmpeg_stderr(proc) or "(пусто)"
        logger.warning(
            f"[{self.stream_name}] RTSP audio: 0 байт за соединение "
            f"({self._dead_cycles}) — ffmpeg: {err}"
        )

        # Feed the rate watchdog on the failure path too. It only ever ran
        # inside the success path, so "20 s of wall clock, zero bytes" was
        # invisible to it — the exact condition it exists to catch.
        await self._account_audio_rate(0)

        if self._dead_cycles >= self._heal_stalls:
            self._dead_cycles = 0
            logger.warning(
                f"[{self.stream_name}] нет аудио {self._heal_stalls} "
                f"попыток подряд -> перерегистрирую поток в go2rtc"
            )
            try:
                await self._heal_go2rtc_stream()
            except Exception as e:
                logger.warning(f"[{self.stream_name}] heal error: {e}")

        # `if`, not `elif`: the same cycle can heal AND declare the room dead.
        # Keyed on _dead_total because _dead_cycles was just reset above.
        if self._dead_total >= self._heal_stalls * 4:
            self._audio_sleep = 30.0
            logger.error(
                f"[{self.stream_name}] КАМЕРА БЕЗ АУДИО — соединение "
                f"устанавливается, но не приходит ни байта. Почти всегда "
                f"у go2rtc нет producer для этого потока. "
                f"ffmpeg: {err}. Продолжаю раз в 30 с."
            )

    async def _read_ffmpeg_stderr(self, proc) -> str:
        """Whatever ffmpeg said on its way out.

        stderr is piped and was NEVER read, so every real cause was thrown
        away while the loop logged only "reconnecting in 3s". Measured
        04.10.2026 20:18: 25 consecutive reconnects and not one line in the
        whole log said why.
        """
        if proc is None or getattr(proc, "stderr", None) is None:
            return ""
        try:
            data = await asyncio.wait_for(proc.stderr.read(), timeout=0.5)
        except Exception:
            return ""
        if not data:
            return ""
        return data.decode("utf-8", "replace").strip()[-300:]

    def _store_tts_echo(self, pcm: bytes, rate: int = TTS_PLAY_RATE) -> None:
        """Record played TTS audio into the echo reference ring buffer,
        resampled to the 16 kHz the mic feed runs at.

        `rate` is whatever we queued on the output track (TTS_PLAY_RATE = 48 kHz).
        The resampling has to track it: a hardcoded 8k->16k stretch here would
        leave the reference 3x too long once playback moved to 48 kHz, and
        _is_echo would stop matching the mic's echo at all.
        """
        try:
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            if len(arr) == 0:
                return
            if rate != 16000:
                if rate <= 0 or 16000 % rate:
                    # Non-integer ratio: resample properly rather than stretch.
                    from scipy.signal import resample_poly

                    g = gcd(16000, rate)
                    up = resample_poly(arr, 16000 // g, rate // g).astype(np.float32)
                else:
                    up = np.repeat(arr, 16000 // rate)
            else:
                up = arr
            n = len(up)
            i0 = self._tts_total % self._tts_ring_len
            if i0 + n <= self._tts_ring_len:
                self._tts_ring[i0 : i0 + n] = up
            else:
                first = self._tts_ring_len - i0
                self._tts_ring[i0:] = up[:first]
                self._tts_ring[: n - first] = up[first:]
            self._tts_total += n
        except Exception:
            pass

    def _is_echo(self, pcm_16k: bytes) -> bool:
        """Return True if the 16 kHz mic chunk is our own TTS echoed back."""
        x = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32)
        n = len(x)
        if n == 0:
            return False
        if self._tts_total < n + 2 * 16000:
            return False
        xn = float(np.linalg.norm(x))
        if xn < 2000.0:  # too quiet to be a played clip echo
            return False
        ring = self._tts_ring
        total = self._tts_total
        best = 0.0
        for d in range(2, 46):  # observed RTSP backchannel echo: 3-40s after playback
            ds = d * 16000
            start = total - ds - n
            if start < 0:
                continue
            i0 = int(start) % self._tts_ring_len
            if i0 + n <= self._tts_ring_len:
                seg = ring[i0 : i0 + n]
            else:
                seg = np.concatenate([ring[i0:], ring[: i0 + n - self._tts_ring_len]])
            sn = float(np.linalg.norm(seg))
            if sn < 1e-3:
                continue
            corr = float(np.dot(x, seg) / (xn * sn))
            if corr > best:
                best = corr
        return best >= self._echo_corr_threshold

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
            self._resample_buf.extend(pcm)
            # 160ms at 48kHz mono = 7680 samples = 15360 bytes
            while len(self._resample_buf) >= 15360:
                block = bytes(self._resample_buf[:15360])
                del self._resample_buf[:15360]
                pcm_16k = await asyncio.to_thread(self._resample_16k, block)
                # Drop our own TTS echo (returned via the mic 8-15s later) by
                # cross-correlation with the recently played audio. This is what
                # lets the user's real answer through while silencing the echo.
                if self._is_echo(pcm_16k):
                    now = time.time()
                    if now - self._last_echo_log > 2.0:
                        self._last_echo_log = now
                        logger.info(
                            f"[{self.stream_name}] 🔁 Echo chunk dropped (corr)"
                        )
                    # Confirmed TTS echo in the mic feed: keep the wake model
                    # suppressed while echoes keep arriving. The RTSP
                    # backchannel returns our own speech 3-40s after playback;
                    # without this a fixed window expired mid-burst and the
                    # late tail re-fired the wake. Bounded by play_end+45s:
                    # an unbounded now+N per chunk let the burst push its own
                    # deadline forward indefinitely and muted the mic for
                    # minutes after every dialogue (user: перестала
                    # реагировать на компьютер).
                    bound = getattr(self, "_tts_play_end", 0.0) + 45.0
                    self._wake_suppress_until = max(
                        self._wake_suppress_until,
                        min(now + 20.0, bound),
                    )
                    continue
                self._vad_buf.extend(pcm_16k)
                self._drain_vad_buf()
        else:
            pcm_16k = await asyncio.to_thread(self._resample_generic, pcm, rate)
            if self._is_echo(pcm_16k):
                now = time.time()
                if now - self._last_echo_log > 2.0:
                    self._last_echo_log = now
                    logger.info(f"[{self.stream_name}] 🔁 Echo chunk dropped (corr)")
                bound = getattr(self, "_tts_play_end", 0.0) + 45.0
                self._wake_suppress_until = max(
                    self._wake_suppress_until,
                    min(now + 20.0, bound),
                )
                return
            self._vad_buf.extend(pcm_16k)
            self._drain_vad_buf()

    def _drain_vad_buf(self):
        """Carve _vad_buf into fixed 160 ms chunks and hand them to _vad_process.

        Silero VAD needs a stable frame size, so incoming (jittery) RTSP
        data is buffered here and emitted in exact chunk_bytes units; the
        per-3s RMS/peak log and the background-noise window are updated on
        the chunk boundary.
        """
        chunk_bytes = 2560
        while len(self._vad_buf) >= chunk_bytes:
            chunk = bytes(self._vad_buf[:chunk_bytes])
            del self._vad_buf[:chunk_bytes]

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
                now_d = time.time()

            asyncio.create_task(self._vad_process(chunk))

    def _proximity_level(self) -> float:
        """Mean raw chunk peak of the last ~2s — stable closeness proxy.

        Mean over max: two cameras can both clip at 32767 when the user stands
        between rooms, but the mean still ranks them by how much of the time
        the signal was loud.
        """
        return (
            float(sum(self._recent_peaks) / len(self._recent_peaks))
            if self._recent_peaks
            else 0.0
        )

    async def _vad_process(self, chunk: bytes):
        async with self._vad_lock:
            s16 = np.frombuffer(chunk, dtype=np.int16)
            self._ww_ring.append(chunk)
            if len(self._ww_ring) > 25:
                self._ww_ring.pop(0)
            # raw peak of this chunk — feeds the proximity arbiter
            self._recent_peaks.append(int(np.abs(s16.astype(np.int32)).max()))
            if len(self._recent_peaks) > 12:  # ~2s window
                self._recent_peaks.pop(0)
            global _ROOM_PEAKS
            _ROOM_PEAKS[self.stream_name] = (
                time.time(),
                self._proximity_level(),
            )
            orig_rms = (
                float(np.sqrt(np.mean(np.square(s16.astype(np.float64))))) / 32768.0
            )
            speech, _ = await self._vad.is_speech(s16.tobytes())
            now = time.time()

            # Only trust VAD if original audio has meaningful energy.
            # Corridor noise floor (mic AGC-boosted) sits at 0.015-0.042
            # normalized (quiet hours 0.022-0.029); ticks/artifacts reach 0.05+.
            # Distant speech (3m) sits ~0.03-0.12. Gate at ~0.0015
            # (≈ rms_int 49) to reject the idle noise floor while keeping
            # distant speech.
            if orig_rms < 0.0015:
                speech = False

            if now - self._last_rms_log > 2.0:
                self._last_rms_log = now
                logger.info(
                    f"[{self.stream_name}] VAD rms={orig_rms:.4f} speech={speech} consec={self._vad_speech_consecutive}"
                )

            # Sliding window (6 frames = 480ms): start utterance when >=3 speech frames,
            # tolerate short pauses between syllables instead of requiring 3 consecutive.
            self._vad_window.append(1 if speech else 0)
            if len(self._vad_window) > 6:
                self._vad_window.pop(0)
            win_speech = sum(self._vad_window)

            if speech:
                self._vad_speech_consecutive += 1
                self._last_speech_at = now
                if not self._vad_has_speech:
                    if win_speech >= 3 and not self._processing_utterance:
                        logger.info(
                            f"[{self.stream_name}] VAD SPEECH START rms={orig_rms}"
                        )
                        self._vad_has_speech = True
                        self._vad_silence_frames = 0
                        if getattr(self, "_endpoint", None) is not None:
                            # A new utterance must not inherit the previous
                            # one's dip run, or it can end before it begins.
                            self._endpoint.reset()
                        self._vad_speech_buf.extend(chunk)
                else:
                    self._vad_speech_buf.extend(chunk)
                self._vad_silence_frames = 0
                # ~1.44s of true floor-level audio commits an utterance.
                # Shorter runs chopped sentences at natural inter-word pauses
                # into 1.3s shards that Whisper could not assemble.
            else:
                if self._vad_has_speech:
                    self._vad_silence_frames += 1
                    if self._vad_silence_frames >= self._vad_silence_limit:
                        self._end_utterance("silence")
                    else:
                        self._vad_speech_buf.extend(chunk)

            # Max duration by accumulated audio length (WebRTC delivers frames in
            # bursts, so wall-clock time is not a reliable cap)
            if self._vad_has_speech and len(self._vad_speech_buf) >= int(
                self._vad_max_duration * 32000
            ):
                self._end_utterance(f"cap {self._vad_max_duration:.0f}s")

            # Level-based endpoint: a pause in the LEVEL, not in Silero's
            # verdict. `speech` is deliberately NOT required to be False — in
            # this room it never is, which is the whole reason this block
            # exists.
            #
            # Gated on an ACTIVE WAKE WINDOW, not merely on "an utterance is
            # open": this room's VAD reports speech=True continuously, so the
            # television opens an utterance every 7 s and fills 7 s of buffer
            # for a transcript that _process_utterance discards one line later
            # (measured 04.10.2026 21:01-21:05: 70 ends in 4 minutes, none of
            # them a command). Endpoints on that audio would make the
            # since-boot tally meaningless — `pause` would mostly count the
            # television, and the one number that decides whether the endpoint
            # works would answer the wrong question.
            if (
                getattr(self, "_endpoint", None) is not None
                and self._vad_has_speech
                and self._wake_detected
            ):
                verdict, detail = self._endpoint.feed(orig_rms)
                if verdict == "end":
                    self._end_utterance(f"pause ({detail})")

            # vosk wake word (rooms configured with WAKE_VOSK_MODEL_<NAME>).
            # Fed EVERY chunk, with NO VAD gate: the Silero VAD rejects 95 % of
            # chunks in this room (measured on a real 84 s recording where the
            # word was plainly audible — 21 of 419 chunks passed), and the decoder
            # then saw a sparse, discontinuous stream. That is why it fired once
            # in five attempts while the same file decoded «компьютер» four times
            # when fed continuously. The VAD exists to delimit utterances for
            # STT; it has no business discarding audio a streaming decoder needs.
            #
            # ALWAYS fed — the wake word must stay detectable at all times and
            # the decoder costs ~1.8 ms per 200 ms. Gating the FEED on
            # `_wake_detected` (added 04.10.2026) made the room deaf for the
            # whole 60 s post-wake dialogue window: the user said «компьютер»
            # again and nothing happened, which is indistinguishable from a
            # broken detector — the exact confusion `vosk diag` exists to
            # prevent. The guard belongs on the FIRING, not on the feed.
            if self._vosk_wake is not None:
                if not self._vosk_wake.active:
                    self._vosk_wake.begin()
                self._vosk_chunks += 1
                self._vosk_last_peak = max(
                    self._vosk_last_peak, int(np.abs(s16).max())
                )
                got_wake = await asyncio.to_thread(self._vosk_wake.feed, chunk)
                if got_wake:
                    # A wake word that HEARD always re-arms the interaction,
                    # even in the middle of a dialogue — that is what a wake
                    # word is for, and the old `_wake_detected` guard made the
                    # room ignore the user for the whole 60 s follow-up window
                    # (field case 04.10.2026: two «компьютер» decoded and
                    # dropped). The only thing that must not fire twice is the
                    # same breath saying the word twice, and that is a
                    # sub-second event, not a minute.
                    now_w = time.time()
                    if not self._wake_gate_open(now_w):
                        # Our own playback still in the air, or the same
                        # breath. Roll the decoder over so it cannot re-trigger
                        # on the same audio, and stay silent.
                        self._vosk_wake.reset()
                        self._vosk_suppressed += 1
                        logger.info(
                            f"[{self.stream_name}] VOSK wake heard but not "
                            f"fired: {self._wake_gate_reason(now_w)} "
                            f"(suppressed={self._vosk_suppressed})"
                        )
                        got_wake = False
                if now - self._vosk_last_log > 300.0:
                    # Deliberately rare: this is a debug aid for "the room is
                    # mute again", not a health line. It reports chunks fed,
                    # peak, trigger/decode tallies and the decoder's own live
                    # hypothesis — which is what actually settled the last two
                    # silent-room investigations. Raise it to log a silent room.
                    self._vosk_last_log = now
                    logger.info(
                        f"[{self.stream_name}] vosk diag: "
                        f"chunks={self._vosk_chunks} suppressed={self._vosk_suppressed} "
                        f"peak_max={self._vosk_last_peak} "
                        f"triggers={self._vosk_wake.triggers} "
                        f"decodes={self._vosk_wake.decodes} "
                        f"hyp='{self._vosk_wake.last_partial}'"
                    )
                    self._vosk_last_peak = 0
                if got_wake:
                    logger.info(
                        f"[{self.stream_name}] VOSK WAKE "
                        f"trig='{self._vosk_wake.last_trigger_text}' "
                        f"conf='{self._vosk_wake.last_text}' "
                        f"triggers={self._vosk_wake.triggers} "
                        f"decodes={self._vosk_wake.decodes}"
                    )
                    self._vosk_wake.reset()
                    await self._fire_wake(reason="vosk")
                    return

            # openWakeWord acoustic wake detector on the live 16k feed. This is
            # the reliable wake for corridor/kitchen, where Whisper transcription
            # of the keyword is unreliable (the word simply isn't in the
            # transcript). Tuned to wake ONLY on a genuine "компьютер":
            # noisy-corridor live attempts score 0.55-0.78, quiet ambient
            # 0.30-0.55. Threshold 0.58 (was 0.65 — missed real attempts at 0.62
            # in bg-noise rms~880) + a 2-chunk debounce kills transient false
            # fires, while the AGC boost still lifts a quiet "компьютер"
            # (peak ~350 -> 4000).
            #
            # SKIPPED ENTIRELY when this room decodes instead. Both detectors
            # running at once cost ~13 ms per chunk (~8 % of a core) for nothing
            # — and the acoustic head is a second, independent source of false
            # wakes, which is exactly what the livingroom must not have.
            if (
                self._vosk_wake is None
                and not self._wake_detected
                and self._engine.oww_model is not None
                and time.time() >= self._wake_suppress_until
                and time.time() - self._audio_epoch > 5.0
            ):
                # Corridor mic lost ~5-10x sensitivity after a majestic
                # restart (voice at the camera reads rms 600-1900 vs 7962
                # before; speaker-loop test confirms the capture path works,
                # just quiet). Boost the chunk BEFORE the VAD gate and AGC:
                # fixed gain preserves dynamics (unlike per-chunk peak
                # normalization, which blinds Silero), and his raw peaks
                # <=4k * 5 stay clear of clipping.
                if self.stream_name in _CORRIDOR_STREAMS:
                    # Adaptive: loud speech must NOT be pushed into the rail
                    # (a square-wave chunk blinds both Silero and the model —
                    # 12:04 trace, raw rms 15500 x5 = clipped).
                    pk0 = int(np.max(np.abs(s16)))
                    g = min(5.0, 30000.0 / max(pk0, 1))
                    s16 = np.clip(s16.astype(np.float32) * g, -32768, 32767).astype(
                        np.int16
                    )
                raw_peak = int(np.max(np.abs(s16)))
                s16_w = s16
                # Per-room AGC target for the wake detector: the kitchen cam
                # sits behind a wall / weak WiFi link and its mic hears the
                # user 3-4x quieter than corridor (peak ~2400 vs ~9500). The
                # shared default of 4000 left its boosted chunks still too
                # quiet for openWakeWord, so distant rooms get a higher boost.
                # Normalize BOTH ways: chunks louder than target are scaled
                # DOWN to it. Otherwise a mid-distance speaker (raw peak 5k)
                # reaches the model unamplified while the same sound in a
                # quieter room gets boosted 3x — and the far room's muffled
                # copy out-scores the room the user is actually in (14:03
                # trace: corridor raw 5403 -> no boost, score <0.4; kitchen
                # raw 1973 -> x3.3 boost, score 0.87 and fired).
                # Corridor mic sensitivity dropped after a majestic restart
                # (voice raw peaks fell from ~30k to <4k); compensate with an
                # aggressive per-room target.
                _WW_TARGET_PEAK = (
                    6500
                    if self.stream_name == "kitchen"
                    else 9000 if self.stream_name in _CORRIDOR_STREAMS else 4000
                )
                if 100 <= raw_peak:
                    s16_w = np.clip(
                        s16.astype(np.float32) * (_WW_TARGET_PEAK / raw_peak),
                        -32768,
                        32767,
                    ).astype(np.int16)
                if raw_peak < 100:
                    self._ww_consec = 0
                elif raw_peak < 600:
                    # Below the meaningful-audio floor: boosting this 10-40x
                    # turns mic noise into plausible model input and scores
                    # 0.6-0.8 in empty rooms (16:38/16:53 traces). Real
                    # speech — even through a wall — measures >=1000.
                    self._ww_consec = 0
                    self._ww_recent.clear()
                else:
                    # VAD gate on the RAW chunk (before AGC): per-chunk peak
                    # normalization flattens amplitude dynamics and Silero
                    # cannot see speech in the normalized signal. The verdict
                    # is passed INTO check_wakeword so a non-speech chunk just
                    # scores 0 through the normal not-wake path.
                    vad_ok = await asyncio.to_thread(self._engine._ww_vad_speech, s16)
                    await asyncio.to_thread(
                        self._engine.check_wakeword,
                        s16_w,
                        self._ww_thresh,
                        self.stream_name,
                        vad_ok,
                    )
                    sc = float(getattr(self._engine, "last_score", 0.0))
                    # Clipping bang gate: door slams / dropped objects hit the
                    # 24k+ peak range and transiently score high (observed
                    # 0.80-0.93 at 02:21 on a corridor bang). Speech peaks
                    # stay below ~20k raw; treat clipped non-speech as noise
                    # unless the score is overwhelming.
                    # Speech has a high crest factor (peak >> rms, ratio
                    # typically 4-7x); impacts and thumps are dense waves
                    # (ratio <2x) yet transiently score 0.55-0.6 on this model.
                    rrms = float(np.sqrt((s16.astype(np.int64) ** 2).mean()))
                    if (raw_peak > 24000 and sc < 0.85) or (
                        raw_peak > 4000 and rrms * 2 > raw_peak and sc < 0.85
                    ):
                        # Clipping bang / dense impact: not speech, kill
                        # debounce state.
                        self._ww_consec = 0
                        self._ww_recent.clear()
                        sc = 0.0
                    # A loud, confident detection always passes — the adaptive
                    # threshold must never be able to lock the system out.
                    wake = sc >= self._ww_thresh or sc >= 0.68

                    # --- Debounce counter maintenance -------------------------
                    # Two consecutive qualifying chunks are required before the
                    # wake fires; ANY non-qualifying chunk resets the counter.
                    # This branch ONLY maintains the counter — the actual fire
                    # happens further below once BOTH the debounce and the
                    # cross-camera ownership checks have passed.
                    self._ww_recent.append(sc)
                    if len(self._ww_recent) > 3:
                        self._ww_recent.pop(0)
                    if not wake:
                        # Sliding window: 2 qualifying chunks out of the last
                        # 3 still count as debounced — the word's score spikes
                        # for a single 80ms chunk then decays, and strict
                        # consecutiveness missed real "компьютер" attempts.
                        if sum(1 for s in self._ww_recent if s >= self._ww_thresh) >= 2:
                            wake = True
                            sc = max(self._ww_recent)
                            if self._ww_consec == 0:
                                self._ww_consec = 1
                        else:
                            self._ww_consec = 0
                    elif self._ww_consec >= 1 or sc >= 0.68:
                        # Fire is imminent (this is at least the second
                        # qualifying chunk) or the score bypasses the debounce
                        # entirely: settle cross-camera ownership NOW.
                        #
                        # Arbitration deliberately runs only at this moment,
                        # never per-chunk: _arbiter_submit sleeps through its
                        # grace window while this coroutine holds the session
                        # VAD lock, so earlier calls would stall the whole
                        # audio pipeline.
                        w_ok, _, _ = await _arbiter_submit(
                            "wake", self.stream_name, self._proximity_level()
                        )
                        if not w_ok:
                            # A closer camera claimed the same wake — abandon
                            # our candidacy entirely (no pip, no state change).
                            logger.info(
                                f"[{self.stream_name}] 🤝 wake ceded to a closer camera"
                            )
                            self._ww_consec = 0
                        elif sc >= 0.68:
                            # Loud-confident single chunk counts as fully
                            # debounced (bypass path, see `wake` above).
                            self._ww_consec = 2
                        else:
                            self._ww_consec += 1
                    else:
                        # First qualifying chunk: remember it but do not
                        # arbitrate yet — single-chunk scores are too noisy
                        # to start a cross-camera contest over.
                        self._ww_consec = 1

                    # --- Fire -------------------------------------------------
                    # Fires only when fully debounced AND no other camera owns
                    # this interaction (its pip / reply are already running);
                    # late echo detections must not add a second beep or steal
                    # the dialogue.
                    if wake and self._ww_consec >= 2:
                        ot = self._out_track
                        speaker_active = ot is not None and (
                            ot.queue_seconds() > 0.3
                            or time.time() < self._speaking_until
                        )
                        my_lvl = self._proximity_level()
                        # Quiet-source fire bar: 0.58 below 2000 raw peak,
                        # 0.55 above it. Computed once so the guard, its log
                        # line and the STT rescue band below cannot drift
                        # apart from each other.
                        q_tier = 0.58 if my_lvl < 2000 else 0.55
                        vetoed = False
                        # Appliance gate: a continuously running appliance
                        # (robot vacuum, hood, AC) keeps the 60s background
                        # median high — speech between words always dips back
                        # toward a quiet floor. A loud motor whine also scores
                        # 0.6-0.9 on this TTS-trained model (14:52 trace:
                        # vacuum rms ~850 -> scores 0.64/0.90 -> double fire).
                        # From inside such noise require an overwhelming score.
                        bg_med = (
                            float(np.median(self._bg_window))
                            if self._bg_window
                            else 0.0
                        )
                        if bg_med > 800 and max(sc, max(self._ww_recent or [0])) < 0.60:
                            logger.info(
                                f"[{self.stream_name}] 🧹 appliance hold — bg {bg_med:.0f} "
                                f"sc {sc:.2f} needs >=0.60"
                            )
                            self._ww_consec = 0
                            top = max(sc, max(self._ww_recent or [0]))
                            self._ww_recent.clear()
                            if top >= 0.55:
                                # Gray zone just under the bar. The bar used to
                                # be 0.92 and the rescue band 0.85-0.92: real
                                # speech over appliance noise reached 0.85-0.99
                                # while the whine topped out ~0.90, so only STT
                                # could tell them apart. When Silero VAD started
                                # gating non-speech the bar dropped to 0.60, and
                                # the same near-miss attempts now land at
                                # 0.55-0.59 — the band the rescue forgot to
                                # follow it into. Anything at or above 0.60
                                # already fires; anything below 0.55 is too weak
                                # to be worth a Whisper call.
                                self._open_stt_confirm(top)
                            vetoed = True
                        if speaker_active:
                            # Speaker is actively playing: any wake-shaped
                            # sound right now IS our own TTS/pip leaking back
                            # through the mic. Never fire on it.
                            self._ww_consec = 0
                            self._ww_recent.clear()
                        elif my_lvl < 3000 and max(sc, max(self._ww_recent or [0])) < q_tier:
                            # Quiet-source confirmation gate: a real user even
                            # at mid-distance produces peaks >3k HERE; faint
                            # through-wall copies stay under it while still
                            # scoring 0.53-0.77 on the TTS-trained model.
                            # Applies regardless of other rooms' noise (13:38
                            # trace: kitchen vacuum noise kept the old
                            # "all rooms quiet" check false and let a 0.71
                            # muffled copy fire in the livingroom).
                            logger.info(
                                f"[{self.stream_name}] 🔈 quiet-source hold — "
                                f"lvl {my_lvl:.0f} sc {sc:.2f} needs >={q_tier}"
                            )
                            self._ww_consec = 0
                            top = max(sc, max(self._ww_recent or [0]))
                            self._ww_recent.clear()
                            if top >= q_tier - 0.05:
                                # Rescue band [tier-0.05, tier): the only scores
                                # this gate can hold that are still plausibly
                                # the user — a genuine attempt landing just under
                                # the bar is exactly what a threshold cannot
                                # separate from a muffled copy. Attempts at or
                                # above the tier already pass and fire; well
                                # below the band they are too faint to be worth
                                # a Whisper call. This band used to sit at
                                # 0.60-0.72/0.85 back when the tier did; the
                                # recalibration to 0.55/0.58 left the rescue
                                # threshold behind, so the branch was dead.
                                self._open_stt_confirm(top)
                            vetoed = True
                        elif my_lvl < 3000:
                            # Ceiling matches the quiet-source hold ceiling;
                            # a gap between the two (old 2500 vs 3000) left a
                            # dead zone where neither gate ran and a muffled
                            # through-wall copy fired freely (17:43 trace).
                            # Distant-source veto: our raw signal is very quiet
                            # (<1600) while another room currently hears the
                            # same sound >=2x louder. The user is THERE; the
                            # muffled through-wall copy of "компьютер" scores
                            # deceptively high on this TTS-trained model.
                            now_ts = time.time()
                            louder = [
                                (s, p)
                                for s, (t, p) in _ROOM_PEAKS.items()
                                if s != self.stream_name
                                and now_ts - t < 3.0
                                and p >= 1.4 * max(my_lvl, 1.0)
                            ]
                            if louder:
                                s, p = max(louder, key=lambda x: x[1])
                                logger.info(
                                    f"[{self.stream_name}] 🚫 distant-source "
                                    f"veto — '{s}' hears {p:.0f} vs our "
                                    f"{my_lvl:.0f}; wake belongs there"
                                )
                                self._ww_consec = 0
                                self._ww_recent.clear()
                                # Sticky: our own level oscillates around the
                                # 2500 ceiling chunk-to-chunk; a one-shot veto
                                # was followed 0.7s later by a fire from the
                                # next chunk (16:56 trace).
                                self._veto_until = time.time() + 5.0
                                vetoed = True
                        # Ambiguous zone: score passed the room threshold but
                        # is not overwhelming and the user is not clearly close.
                        # Rustle/cage noise lands exactly here (LR 20:39-20:45:
                        # six pips on hamster activity, lvl bursts >=3000 kept
                        # the quiet-hold from ever running). Route to Whisper
                        # confirmation instead of beeping; a bare word with no
                        # follow-up is caught by the expiry watcher.
                        qual_top = max(sc, max(self._ww_recent or [0]))
                        # A room that was recently auto-greeted but produced
                        # NO command is almost certainly playing TV/noise:
                        # humans do not re-say 'компьютер' seconds after
                        # ignoring a greeting (22:15 trace: TV burst 0.9012
                        # fired 20s after the greeting). Force such repeats
                        # through STT confirmation for a full minute.
                        recent_unanswered_greet = (
                            time.time() - self._last_auto_greet_ts < 60.0
                        )
                        if (
                            not vetoed
                            and (qual_top < 0.55 or recent_unanswered_greet)
                            and my_lvl < 3000
                            and time.time() >= self._stt_confirm_until
                        ):
                            self._ww_consec = 0
                            self._ww_recent.clear()
                            self._open_stt_confirm(qual_top)
                            vetoed = True
                        if (
                            not speaker_active
                            and not vetoed
                            and time.time() >= self._veto_until
                        ):
                            busy, olvl = _arbiter_owner_active(self.stream_name)
                            # Proximity steal: the user is clearly standing in
                            # OUR room (peak >=5x the owner's) and the owner
                            # has not even dispatched a command yet — the wake
                            # was ours, the far camera just caught the sound
                            # first. Take over so the answer sounds where the
                            # user actually is.
                            if (
                                busy
                                and not _ARB_STATE.get("cmd_sent")
                                and my_lvl >= 5.0 * max(olvl, 1.0)
                            ):
                                logger.info(
                                    f"[{self.stream_name}] 🥇 stealing ownership "
                                    f"(lvl {my_lvl:.0f} vs owner {olvl:.0f})"
                                )
                                busy = False
                            if busy:
                                logger.info(
                                    f"[{self.stream_name}] 🤝 standing down — "
                                    f"interaction owned by another room"
                                )
                                self._ww_consec = 0
                            else:
                                logger.info(
                                    f"[{self.stream_name}] 🎯 Wake word (openWakeWord)"
                                )
                                await self._fire_wake(reason="oww")

    def _wake_gate_open(self, now: float) -> bool:
        """May a wake decoded at `now` actually FIRE?

        Two things close the gate, and neither of them is `_wake_detected`:

        * `echo_tail` — we are still hearing our own playback. This is the only
          part a timer can decide, and it is deliberately short.
        * `rearm` — the same breath said the word twice. A second
          `_fire_wake()` would clear `_vad_speech_buf` and throw away the
          command being collected.

        `_wake_detected` is deliberately NOT consulted. It stays True for the
        whole post-wake dialogue window (`_wake_timeout`, 60 s), and using it
        here is what made the room ignore the user after the first trigger:
        measured 04.10.2026, two «компьютер» 8 s and 11 s after the reply were
        decoded and dropped before the third, 28 s later, got through. A wake
        word exists so that saying it gets attention — always.
        """
        if now < self._wake_suppress_until:
            return False
        if now - self._last_wake_fired_at < _WAKE_REARM_DEBOUNCE_S:
            return False
        return True

    def _wake_gate_reason(self, now: float) -> str:
        """Which of the two closed the gate — for the log line, so a silent
        room can be told apart from a deaf one."""
        if now < self._wake_suppress_until:
            return f"echo_tail={self._wake_suppress_until - now:+.1f}s"
        return f"same_breath={now - self._last_wake_fired_at:.1f}s"

    async def _fire_wake(self, reason: str, score: float = 1.0):
        """Enter the post-wake state and announce the wake.

        Shared by both detectors so the arbitration, pip, greeting and VAD reset
        cannot drift apart between them. Extracted verbatim from the old inline
        openWakeWord block — the only change is the `reason` in the log line.
        """
        # Claim global interaction ownership so other rooms' detectors stand
        # down for this exchange.
        _arbiter_set_owner(self.stream_name, self._proximity_level())
        self._wake_cmd_sent = False
        self._last_fire_ts = time.time()
        self._last_wake_fired_at = time.time()
        self._wake_detected = True
        self._wake_expires = time.time() + self._wake_timeout
        self._ww_consec = 0
        self._ww_recent.clear()
        # Reset VAD collection: corridor's permanent noise floor keeps
        # _vad_has_speech=True indefinitely (consec>100 observed), so the
        # post-wake command would otherwise be drowned inside a noise buffer that
        # never hits the silence limit. Start the command utterance from a clean
        # slate.
        self._vad_has_speech = False
        self._vad_speech_buf.clear()
        self._vad_speech_consecutive = 0
        if getattr(self, "_endpoint", None) is not None:
            self._endpoint.reset()
        self._vad_silence_frames = 0
        asyncio.create_task(self._play_attention(reason))
        # Bare 'компьютер' with no follow-up used to end in eternal silence (pip
        # only). After a short pause greet so the user knows they were heard.
        # Cap ownership: if nobody responds to pip + greeting, release other
        # rooms quickly.
        self._wake_expires = min(self._wake_expires, time.time() + 14.0)
        self._schedule_wake_greeting()

    async def _confirm_expiry_watch(self):
        """Confirm window closed with ZERO utterances: that pattern is a bare
        'компьютер' spoken into an otherwise quiet room (the word triggered
        oww, the user then waits). Greet them. Noise sources (hamster cage,
        vacuum) never stay silent — they keep feeding garbage transcripts,
        which drop without a pip."""
        try:
            await asyncio.sleep(11.5)
        except asyncio.CancelledError:
            return
        if (
            time.time() >= self._stt_confirm_until
            and not self._confirm_saw_text
            and not self._wake_detected
        ):
            logger.info(f"[{self.stream_name}] 🔎 confirm window silent — bare wake")
            self._schedule_wake_greeting()

    def _open_stt_confirm(self, sc: float) -> None:
        """Gates held a decent oww score (appliance/quiet-source). Do not pip;
        open a short window during which the next VAD utterance goes to
        Whisper. A live «компьютер» in the transcript then fires the real
        wake; garbage transcripts just expire silently. This rescues strong
        attempts that threshold gates cannot separate from noise (kitchen
        trace: real 0.91 vs vacuum 0.90 — only STT tells them apart)."""
        if time.time() < self._stt_confirm_until:
            # A window is already open. Two gates in the same cascade can hit
            # their rescue bands on one chunk (appliance holds first and clears
            # the score ring, the quiet-source gate then re-reads the bare
            # chunk score) — stacking a second expiry watcher would greet the
            # same silent room twice, and the refreshed deadline would keep
            # Whisper listening longer than a single window should.
            return
        self._stt_confirm_until = time.time() + 10.0
        self._confirm_saw_text = False
        asyncio.create_task(self._confirm_expiry_watch())
        logger.info(
            f"[{self.stream_name}] 🔎 oww {sc:.2f} held — waiting for STT "
            f"confirmation utterance"
        )

    def _apply_ns(self, audio_f: np.ndarray):
        """SpeexDSP noise suppression — strips stationary noise (distant speech)."""
        try:
            from speexdsp_ns import NoiseSuppression

            ns = NoiseSuppression.create(frame_size=256, sample_rate=16000)
            clean = []
            for i in range(0, len(audio_f) - 255, 256):
                clean.append(
                    ns.process(audio_f[i : i + 256].astype(np.int16).tobytes())
                )
            return (
                np.frombuffer(b"".join(clean), dtype=np.int16).astype(np.float32),
                True,
            )
        except Exception as exc:
            logger.warning(f"[{self.stream_name}] NS error: {exc}")
            return audio_f, False

    def _preprocess_audio(
        self, buf: bytes
    ) -> tuple[bytes | None, bytes | None, float, dict]:
        samples = np.frombuffer(buf, dtype=np.int16)
        duration_s = len(samples) / 16000.0
        rms_raw = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
        peak_raw = int(np.max(np.abs(samples)))
        rms_dB = 20 * np.log10(max(rms_raw, 1) / 32768)

        # Minimum duration gate — very short utterances produce garbage
        if duration_s < 0.5:
            logger.info(
                f"[{self.stream_name}] ⏩ Too short ({duration_s:.2f}s rms={rms_raw:.0f} peak={peak_raw})"
            )
            return None, None, duration_s, {}

        # Minimum energy gate vs adaptive background floor.
        # Corridor noise sits at rms 500-1400; distant speech (3m) ~950-2500.
        bg = float(np.median(self._bg_window)) if self._bg_window else 0.0
        min_rms = max(350, 1.15 * bg)
        if rms_raw < min_rms and not self._wake_detected:
            # NS rescue: distant speech sits inside the noise floor — try
            # cleaning first. Only keep the cleaned signal if it is still
            # strong speech (rms >= 400); otherwise the utterance is just
            # noise and must be dropped, not forwarded to Whisper as garbage.
            cleaned, ns_ok = self._apply_ns(samples.astype(np.float32))
            rms_clean = float(np.sqrt(np.mean(np.square(cleaned))))
            if ns_ok and rms_clean >= 400:
                logger.info(
                    f"[{self.stream_name}] 🛟 NS rescue: rms {rms_raw:.0f} -> clean {rms_clean:.0f} (bg={bg:.0f})"
                )
                samples = cleaned.astype(np.int16)
                rms_raw = rms_clean
            else:
                logger.info(
                    f"[{self.stream_name}] ⏩ Too quiet (rms={rms_raw:.0f} peak={peak_raw} dB={rms_dB:.1f} bg={bg:.0f} min={min_rms:.0f})"
                )
                return None, None, duration_s, {}

        # Speech-energy ratio gate — count chunks with real energy.
        # Background noise ~190-500 int16 (AGC), speech chunks ~800+,
        # distant speech (3m) chunks ~400-900.
        chunk_samples = 1280  # 80ms at 16kHz
        total_chunks = len(samples) // chunk_samples
        if total_chunks > 0:
            reshaped_samples = samples[: total_chunks * chunk_samples].reshape(
                total_chunks, chunk_samples
            ).astype(np.float32)
            chunk_rms = np.sqrt(np.mean(np.square(reshaped_samples), axis=1))
            speech_chunks = int(np.sum(chunk_rms >= 400))
        else:
            speech_chunks = 0
        speech_ratio = speech_chunks / max(total_chunks, 1)
        # Post-wake commands must always reach STT: the user is talking to
        # the device and short words (e.g. «свет») yield a low ratio
        # because of the silence tail the VAD appends.
        if speech_ratio < 0.35 and not self._wake_detected:
            logger.info(
                f"[{self.stream_name}] ⏩ Low speech ratio ({speech_ratio:.0%}={speech_chunks}/{total_chunks} rms={rms_raw:.0f} dB={rms_dB:.1f})"
            )
            return None, None, duration_s, {}

        # Adaptive peak normalization (max 20x gain for distant speech)
        audio_f = samples.astype(np.float32)

        # Noise suppression is ONLY for clearly distant/quiet speech. In a
        # busy corridor (other people talking) the noise is non-stationary
        # speech, which SpeexDSP cannot separate — applying NS there just
        # mutilates the user's own voice. So near-field commands pass through
        # untouched and Whisper gets the original (user-dominant) spectrum.
        ns_applied = False
        # Kitchen's permanent appliance hum (fridge/vent, ~600 rms, 30-400Hz)
        # sits above the generic quiet-speech threshold and drowns its distant
        # mic — Whisper returns "взвввв" garbage for that room. Give kitchen
        # NS at a higher rms cutoff so its hum is stripped before STT.
        ns_rms_gate = 800 if self.stream_name == "kitchen" else 400
        if rms_raw < ns_rms_gate and len(audio_f) >= 256:
            audio_f, ns_applied = self._apply_ns(audio_f)

        peak = np.max(np.abs(audio_f))
        gain_applied = 1.0
        if peak > 0:
            gain = min(32767.0 / peak, 20.0)
            if gain > 1.5:
                audio_f = np.clip(audio_f * gain, -32768, 32767)
                gain_applied = gain
        buf_processed = audio_f.astype(np.int16).tobytes()
        buf_raw = samples.astype(np.int16).tobytes()

        stats = {
            "rms_raw": rms_raw,
            "peak_raw": peak_raw,
            "rms_dB": rms_dB,
            "gain_applied": gain_applied,
            "ns_applied": ns_applied,
        }
        return buf_processed, buf_raw, duration_s, stats

    async def _process_stt(
        self, buf_processed: bytes, buf_raw: bytes, duration_s: float, stats: dict
    ) -> tuple[str | None, str | None]:
        wav = await asyncio.to_thread(self._encode_wav, buf_processed)

        def _save_debug_wav(wav_data, ts):
            """Best-effort dump of every utterance to /tmp/utterances/.

            Field debugging aid only: created lazily, failures swallowed —
            a full tmpfs must never abort an STT round-trip.
            """
            import os

            os.makedirs("/tmp/utterances", exist_ok=True)
            with open(f"/tmp/utterances/u_{ts}.wav", "wb") as f:
                f.write(wav_data)

        try:
            await asyncio.to_thread(_save_debug_wav, wav, int(time.time()))
        except Exception:
            pass

        logger.info(
            f"[{self.stream_name}] 🎤 Whisper IN: {duration_s:.2f}s "
            f"raw_rms={stats['rms_raw']:.0f} peak={stats['peak_raw']} "
            f"dB={stats['rms_dB']:.1f} gain={stats['gain_applied']:.1f}x "
            f"ns={int(stats['ns_applied'])} wav={len(wav)}B"
        )
        # Speaker ID runs in PARALLEL but never gates the reply — the service
        # can take 5+s while Whisper needs <1s, and uid is only used for logs.
        stt_task = asyncio.create_task(
            self._fetch_transcription(wav, rms_raw=float(stats.get("rms_raw") or 0.0))
        )
        uid_task = asyncio.create_task(self._fetch_speaker_id(wav))

        def _uid_done(t: "asyncio.Task") -> None:
            """done-callback: log the recognised speaker, never raise.

            Must be exception-safe (a raise inside a callback is only
            logged by asyncio and would look like a SpeakerID bug); the
            result is read defensively because the task may have been
            cancelled along with the session.
            """
            try:
                u = t.result()
            except BaseException:
                return
                self._last_uid = u
                self._last_uid_ts = time.time()
            if u and u != "unknown":
                logger.info(f"[{self.stream_name}] 👤 [SpeakerID] {u}")

        uid_task.add_done_callback(_uid_done)
        txt = await stt_task
        self._last_utt_peak = int(stats.get("peak_raw") or 0)
        uid = (
            uid_task.result() if uid_task.done() and not uid_task.exception() else None
        )
        if (not txt or not is_valid_text(txt)) and self._wake_detected:
            # Post-wake rescue: NS may have eaten the consonants of a
            # distant command — retry the unprocessed signal.
            logger.info(
                f"[{self.stream_name}] 🔁 Post-wake retry raw signal (wake={self._wake_detected})"
            )
            txt = await self._fetch_transcription(
                await asyncio.to_thread(self._encode_wav, buf_raw),
                temperature="0.0",
                rms_raw=float(stats.get("rms_raw") or 0.0),
            )
        if not txt or not is_valid_text(txt):
            if not txt:
                logger.info(
                    f"[{self.stream_name}] ❌ Whisper empty for {len(buf_processed)}-byte ({duration_s:.2f}s)"
                )
            else:
                logger.info(f"[{self.stream_name}] ❌ Rejected: '{txt[:60]}'")
            return None, None

        logger.info(f"[{self.stream_name}] ✅ Whisper OK: '{txt[:60]}'")
        return txt, uid

    async def _handle_stt_result(self, txt: str, uid: str | None) -> None:
        # --- Gray-zone STT confirmation ------------------------------------
        # A gate held a decent oww score and opened a confirmation window.
        # The utterance just transcribed decides: live «компьютер» -> full
        # wake (pip, ownership, command window); anything else -> drop.
        if not self._wake_detected and time.time() < self._stt_confirm_until:
            self._stt_confirm_until = 0.0
            self._confirm_saw_text = True
            # The oww detector ALREADY matched the wake sound (that is why
            # this window is open). The confirming utterance is the COMMAND
            # that follows the word — it does NOT contain «компьютер» again.
            # So confirmation = Whisper produced real speech, not silence/
            # hallucination; the text itself becomes the command.
            # Never confirm on an echo of our own last reply: delayed RTSP
            # return of TTS audio is LOUD (passes any raw-energy gate) and
            # transcribes into valid text -> camera would interview itself.
            norm_c = txt.lower().strip(" .,!?-")
            if self._last_tts_reply and _echo_of_reply(norm_c, self._last_tts_reply):
                logger.info(f"[{self.stream_name}] ❌ stt-confirm: echo of own reply")
                return
            sane = (
                bool(txt.strip())
                and is_valid_text(txt)
                and getattr(self, "_last_utt_rms_raw", 0.0) >= 750.0
            )
            if not sane and txt.strip():
                logger.info(
                    f"[{self.stream_name}] ❌ stt-confirm dropped: raw rms "
                    f"{getattr(self, '_last_utt_rms_raw', 0):.0f} < 750 "
                    f"(amplified ambient noise)"
                )
                return
            if sane:
                # Whisper hallucinates fluent garbage over ambient noise
                # ('и пей девочка, ой, блядь...' passed is_valid_text;
                # 'короче к лицо если придать мне ничего больше не рассыпать'
                # had all-unique words). SpeakerID also reports 0.97 on such
                # amplified noise. The only honest discriminator left: RAW
                # capture energy — real commands measure rms >=1100 even
                # through a wall; hamster rustle boosted 6x sits at ~640.
                words = re.findall(r"[а-яёa-z0-9]+", txt.lower())
                top = max((words.count(w) for w in set(words)), default=0)
                if len(words) < 2 or top > 2:
                    logger.info(
                        f"[{self.stream_name}] ❌ stt-confirm looks like "
                        f"hallucination ({len(words)}w, maxrep {top})"
                    )
                    return
            if sane:
                logger.info(
                    f"[{self.stream_name}] ✅ STT confirmed wake via command: "
                    f"'{txt[:60]}'"
                )
                won, _, winner = await _arbiter_submit(
                    "wake", self.stream_name, self._proximity_level()
                )
                if not won:
                    logger.info(
                        f"[{self.stream_name}] 🤝 stt-confirm ceded to '{winner}'"
                    )
                    return
                _arbiter_set_owner(self.stream_name, self._proximity_level())
                self._wake_cmd_sent = False
                self._last_fire_ts = time.time()
                self._wake_detected = True
                self._wake_expires = time.time() + self._wake_timeout
                asyncio.create_task(self._play_attention("oww"))
                # Confirmed without an inline command? Greet as well.
                self._schedule_wake_greeting()
            else:
                logger.info(
                    f"[{self.stream_name}] ❌ stt-confirm dropped: '{txt[:50]}'"
                )
                return
        norm = txt.lower().strip().strip(".,!? -")
        if norm in (
            "готов",
            "да",
            "го",
            "начинаем",
            "начали",
            "готово",
            "ок",
            "окей",
            "давай",
        ):
            logger.info(f"[{self.stream_name}] ✅ Acknowledge: '{txt[:60]}'")
            await self._play_attention("ack")
            self._back_to_wake()
            return
        # Anti-loop: camera hears its own TTS echo with 15-25s delay;
        # if this transcript matches a recent one, it's the echo.
        if norm and norm == self._last_transcript:
            logger.info(f"[{self.stream_name}] ❌ Echo repeat, ignoring: '{txt[:60]}'")
            return
        # Stronger echo guard: the speaker plays our own reply, so Whisper
        # transcribes it ~verbatim. If this transcript strongly overlaps the
        # text we most recently spoke (and we're in a wake window, where an
        # echo is plausible), treat it as the echo and drop it. This is the
        # backstop for the waveform-based _is_echo in _feed_audio.
        if norm and self._last_tts_reply and self._wake_detected:
            if _echo_of_reply(norm, self._last_tts_reply):
                logger.info(
                    f"[{self.stream_name}] ❌ Echo of TTS reply, ignoring: '{txt[:60]}'"
                )
                return
        self._last_transcript = norm
        if uid and uid != "unknown":
            logger.info(f"[{self.stream_name}] 👤 Speaker: {uid} | '{txt[:60]}'")
        await self._handle_wake_or_command(txt, uid or "camera")

    def _end_utterance(self, reason: str) -> None:
        """Close the current utterance and hand its audio to Whisper.

        ONE path for all three terminators — VAD silence, a level-based pause
        and the duration cap. Each used to inline its own copy of this block,
        which is exactly how a third terminator could be added with a subtly
        different reset set and quietly break the wake path.

        The reason is logged with the numbers that produced it, and tallied per
        boot, so "is the endpoint working or is everything still hitting the
        7 s cap?" is one `grep UTTERANCE END` away.
        """
        buf = bytes(self._vad_speech_buf)
        dur = len(buf) / 32.0
        self._vad_speech_buf.clear()
        self._vad_has_speech = False
        self._vad_speech_consecutive = 0
        self._vad_silence_frames = 0
        self._vad_window.clear()
        if getattr(self, "_endpoint", None) is not None:
            self._endpoint.reset()
        # Tally and log ONLY for utterances a transcript could actually use.
        # This room's VAD reports speech=True continuously, so a
        # television-only utterance opens every ~7 s and is discarded by
        # _process_utterance for want of a wake window. Counting those made the
        # tally useless — measured 04.10.2026 21:01-21:05: 70 "via cap 7s" in
        # four minutes, not one of them a command — and put one INFO line every
        # 7 s into the log, i.e. ~12000 a day.
        if self._wake_detected:
            key = reason.split(" ", 1)[0].split("(", 1)[0]
            self._end_reason[key] = self._end_reason.get(key, 0) + 1
            logger.info(
                f"[{self.stream_name}] VAD UTTERANCE END dur={dur:.0f}ms "
                f"via {reason} (awake ends: {self._end_reason})"
            )
        else:
            logger.debug(
                f"[{self.stream_name}] ambient utterance ended after "
                f"{dur:.0f}ms via {reason} — no wake window, discarded"
            )
        asyncio.create_task(self._process_utterance(buf))

    async def _process_utterance(self, buf: bytes):
        if self._processing_utterance:
            return

        # Whisper listens ONLY after the acoustic wake fired (oww pip / active
        # wake window) or while a gray-zone STT-confirmation window is open
        # (gates held a decent oww score; Whisper arbitrates). Everything else
        # is ambient noise — feeding it to STT just spams Whisper with TV/
        # radio garbage and burns CPU. The oww detector runs continuously on
        # the live feed instead (cheap).
        if not self._wake_detected and time.time() >= self._stt_confirm_until:
            return

        # Trim trailing silence: the corridor noise floor keeps VAD "speech"
        # alive, so buffers routinely run to the max-duration cap with ~1s of
        # dead tail — that tail inflates Whisper time and adds nothing.
        s16 = np.frombuffer(buf, dtype=np.int16)
        if s16.size > 3200:
            loud = np.nonzero(np.abs(s16.astype(np.int32)) > 300)[0]
            if loud.size:
                cut = min(s16.size, int(loud[-1]) + 1600)  # keep 100ms tail
                buf = s16[:cut].tobytes()
        self._processing_utterance = True
        try:
            buf_processed, buf_raw, duration_s, stats = self._preprocess_audio(buf)
            if not buf_processed or not buf_raw:
                return

            txt, uid = await self._process_stt(
                buf_processed, buf_raw, duration_s, stats
            )
            if not txt:
                return

            self._last_utt_rms_raw = float((stats or {}).get("rms_raw", 0.0))
            await self._handle_stt_result(txt, uid)
        finally:
            self._processing_utterance = False

    async def _handle_wake_or_command(self, txt: str, uid: str = "camera"):
        """Handle an utterance that Whisper transcribed AFTER the acoustic
        wake fired. The oww detector is the ONLY wake path now — this method
        never wakes on transcript content, it only extracts and dispatches
        the command that followed a live oww detection."""
        if not self._wake_detected:
            # No active wake window: nothing to do. (Reached only via stale
            # queued utterances — live utterances are gated earlier.)
            return
        self._cancel_wake_greeting()
        # Never forward the wake word itself to nanobot — only the command
        # that follows it. Tolerate Whisper's glued repeats like
        # "компьютеркомпьютер" by matching without word boundaries.
        kw = re.escape(self.wake_keyword.lower())
        m = re.search(
            rf"(?:хей\s+)?({kw}|{kw.replace('джарвис','jarvis')})", txt, re.IGNORECASE
        )
        if not m:
            m = re.search(rf"({kw})", txt, re.IGNORECASE)
        cmd = txt
        if m:
            cmd = (txt[: m.start()] + " " + txt[m.end() :]).strip()
        cmd = cmd.strip(" .,!?-")
        if not cmd or not is_valid_text(cmd):
            logger.info(
                f"[{self.stream_name}] 🎯 Wake-only utterance, awaiting command"
            )
            return
        logger.info(f"[{self.stream_name}] 🗣 Post-wake command: '{cmd[:60]}'")
        # Overlapping wake windows: several cameras hear the same answer —
        # only the closest one talks to Nanobot.
        qual = float(len(cmd)) if is_valid_text(cmd) else 0.0
        won, gk, winner = await _arbiter_submit(
            "cmd",
            self.stream_name,
            float(self._last_utt_peak or 0),
            0.6,
            quality=qual,
            respect_owner=True,
        )
        if not won:
            logger.info(
                f"[{self.stream_name}] 🤝 post-wake command ceded to '{winner}' — "
                f"will take over if it stays silent"
            )
            asyncio.create_task(self._cmd_fallback(cmd, uid, gk, winner))
            return
        _arbiter_mark_sent(gk, self.stream_name)
        await self._call_nanobot(cmd, uid)

    async def _cmd_fallback(self, cmd: str, uid: str, gk: tuple, winner: str):
        """Take over a lost command arbitration if the winner never dispatches.

        Covers the race where BOTH cameras transcribed the same command but
        only one received a usable Whisper result: this loser waits ~2.2s and
        dispatches only if the winner stayed completely silent (nothing was
        ever sent to Nanobot under its name).
        """
        if gk == ("owned", 0) or winner == "owner":
            # We lost to the OWNERSHIP veto, not to a level contest with a
            # peer claim. Another camera is mid-dialogue and WILL answer;
            # taking the command over here would fire a duplicate reply
            # ~2.2s later. No fallback in this case.
            return
        await asyncio.sleep(2.2)
        if _arbiter_sent_recently(gk, winner):
            # Winner dispatched in time — nothing to rescue.
            return
        logger.warning(
            f"[{self.stream_name}] 🥈 winner '{winner}' silent — taking over command"
        )
        await self._call_nanobot(cmd, uid)

    def _cancel_wake_greeting(self):
        task = self._wake_greeting_task
        if task is not None and not task.done():
            task.cancel()
        self._wake_greeting_task = None
        # True while the AUTO-greeting (bare wake, no command yet) is being
        # spoken: shortens the ownership/follow-up window so other rooms do
        # not stay deaf for half a minute when the user walks away.
        self._auto_greeting = False
        self._last_auto_greet_ts = 0.0

    def _schedule_wake_greeting(self):
        self._cancel_wake_greeting()
        self._wake_greeting_task = asyncio.create_task(self._wake_greeting())

    async def _wake_greeting(self):
        try:
            await asyncio.sleep(self._wake_greeting_delay)
        except asyncio.CancelledError:
            return
        if not self._wake_detected:
            return
        # «Is anyone talking?» must mean "has any speech frame arrived lately",
        # NOT "has an utterance been committed". The two differ exactly when it
        # hurts: `_vad_has_speech` stays False while `_processing_utterance` is
        # in flight (the onset requires `not _processing_utterance`), so a
        # command spoken right after the previous turn is dropped before it
        # ever reaches Whisper — measured 04.10.2026: VAD reported
        # `speech=True consec=54` with no SPEECH START at all, and the camera
        # answered «Да?» to «включи свет» because the room looked silent.
        if time.time() - self._last_speech_at < self._wake_greeting_delay:
            try:
                await asyncio.sleep(4.0)
            except asyncio.CancelledError:
                return
            if not self._wake_detected:
                return
        logger.info(f"[{self.stream_name}] 🎯 Wake with no command — sending greeting")
        # NOTE: keep _wake_detected True so a command the user utters right
        # after the greeting is still treated as post-wake dialogue and is not
        # dropped ("No wake word, ignoring"). The 60s _wake_timeout window
        # governs how long follow-ups are accepted.
        self._auto_greeting = True
        self._last_auto_greet_ts = time.time()
        try:
            # Fixed local prompt instead of nanobot 'привет': instant, no
            # chat-history pollution, and it can't be mistaken for a
            # misheard transcript ('камера считает что я сказал привет').
            fb = "Да?"
            pcm = await self._tts_fetch(fb)
            if pcm:
                await self._speak_pcm(pcm, fb)
        finally:
            self._auto_greeting = False
        self._last_auto_greet_ts = 0.0

    async def _on_user_command(self, text: str):
        """Route a transcript that arrived with the wake word present.

        Text after the wake keyword: that is the command -> dispatch to the
        backend. Nothing after it: a bare wake ("компьютер") -> extend the
        wake window and chirp the attention cue so the user knows to speak.
        """
        try:
            kw = self.wake_keyword.lower()
            if kw not in text.lower():
                self._processor.set_state(AgentState.LISTENING)
                return
            idx = text.lower().find(kw)
            cmd = text[idx + len(kw) :].strip().lstrip(".,!? -")
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

    async def _call_backend(self, txt: str, uid: str = "camera"):
        """Universal path: stream LLM reply from self.backend into the speaker
        queue via the same player task used by the Nanobot path."""
        if not txt:
            return
        self._wake_cmd_sent = True
        _ARB_STATE["cmd_sent"] = time.time()
        base_thresh = self._ww_base_thresh
        if self._ww_thresh > base_thresh:
            self._ww_thresh = base_thresh
        if time.time() < GLOBAL_TTS_UNTIL:
            logger.info(f"[{self.stream_name}] Ignoring '{txt[:40]}' — TTS playback active (echo guard)")
            return
        try:
            q: asyncio.Queue = asyncio.Queue()
            player_task = asyncio.create_task(self._nanobot_player_task(q))
            try:
                await self.backend.generate_response(
                    text=txt,
                    session_id=self.chat_id,
                    stream_name=self.stream_name,
                    response_queue=q,
                )
            finally:
                try:
                    await asyncio.wait_for(player_task, timeout=120.0)
                except asyncio.TimeoutError:
                    player_task.cancel()
        except Exception as e:
            logger.warning(f"[{self.stream_name}] Backend error: {e}")

    async def _call_nanobot(self, txt: str, uid: str = "camera"):
        if not txt:
            return
        # Backend dispatch: Hermes (or any BaseLLMBackend) instead of Nanobot
        if self.backend is not None:
            await self._call_backend(txt, uid)
            return
        self._wake_cmd_sent = True
        _ARB_STATE["cmd_sent"] = time.time()
        base_thresh = self._ww_base_thresh
        if self._ww_thresh > base_thresh:
            logger.info(
                f"[{self.stream_name}] ✅ real command — wake threshold reset to {base_thresh}"
            )
            self._ww_thresh = base_thresh
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
                    await ws.send_json(
                        {
                            "type": "message",
                            "chat_id": self.chat_id,
                            "content": txt,
                            "user_id": uid,
                            "user_name": self.stream_name,
                            "voice_reply": True,
                        }
                    )
                    # Stream sentences to the speaker as they arrive instead of
                    # waiting for the full LLM reply — cuts perceived latency
                    # from (full LLM completion + TTS) to (first sentence + TTS).
                    q: asyncio.Queue = asyncio.Queue()
                    player_task = asyncio.create_task(self._nanobot_player_task(q))
                    _n_sent = 0
                    try:
                        _n_sent = await self._nanobot_read_loop(ws, q)
                    finally:
                        try:
                            await asyncio.wait_for(player_task, timeout=120.0)
                        except asyncio.TimeoutError:
                            player_task.cancel()

                        if _n_sent == 0:
                            # Nanobot hung (idle-timeout) or answered empty —
                            # the camera would go silent forever. Speak the
                            # same fallback the ESP32 watchdog uses.
                            logger.warning(
                                f"[{self.stream_name}] 🕐 nanobot silent — fallback TTS"
                            )
                            fb = "Простите, я задумалась. Повторите пожалуйста."
                            try:
                                pcm_fb = await self._tts_fetch(fb)
                                if pcm_fb:
                                    await self._speak_pcm(pcm_fb, fb)
                            except Exception as e:
                                logger.warning(f"fallback TTS failed: {e}")
        except Exception as e:
            logger.warning(f"[{self.stream_name}] Nanobot error: {e}")

    async def _nanobot_read_loop(
        self, ws: aiohttp.ClientWebSocketResponse, q: asyncio.Queue
    ) -> int:
        _t0 = time.time()
        _t_first = None
        _n_sent = 0
        _seen_ev: set = set()

        def speakable(s: str) -> str:
            s = re.sub(r"\[[a-z]{2,30}\]", "", s)
            return re.sub(r"\s+", " ", s).strip()

        buf = ""
        spoke_any = False

        def flush(final: bool) -> None:
            nonlocal buf, spoke_any, _n_sent
            # Hold back an unterminated [thinking block or any
            # dangling "[tag" so its inner text never leaks to TTS.
            m = re.search(r"\[thinking\]", buf)
            if m and not re.search(r"\[/thinking\]", buf[m.start() :]):
                part, holder = (
                    (buf[: m.start()], buf[m.start() :]) if not final else (buf, "")
                )
            else:
                idx = buf.rfind("[")
                if idx != -1 and not re.search(r"\]", buf[idx:]):
                    part, holder = buf[:idx], buf[idx:]
                else:
                    part, holder = buf, ""
            while True:
                mm = re.search(r"\[thinking\](.*?)\[/thinking\]", part, flags=re.S)
                if not mm:
                    break
                part = part[: mm.start()] + part[mm.end() :]
            mm = re.search(r"^(.*[.!?…])([^.!?…]*)$", part, flags=re.S)
            if mm:
                done = [mm.group(1)] if mm.group(1).strip() else []
                tail = mm.group(2)
            else:
                done, tail = [], part
            if final and tail.strip():
                done.append(tail)
                tail = ""
            for s in done:
                s2 = speakable(s)
                if s2:
                    spoke_any = True
                    _n_sent += 1
                    logger.info(
                        f"[{self.stream_name}] ⏱ nanobot sentence #{_n_sent} "
                        f"@ +{time.time()-_t0:.1f}s: '{s2[:40]}'"
                    )
                    q.put_nowait(s2)
            buf = (tail + " " + holder).strip()

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=45.0)
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.stream_name}] Nanobot stream idle 45s")
                    break
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                ev = data.get("event")
                evk = str(data.get("type")) + "/" + str(ev)
                if evk not in _seen_ev:
                    _seen_ev.add(evk)
                    logger.info(
                        f"[{self.stream_name}] ⏱ nanobot ev: {evk} "
                        f"+{time.time()-_t0:.1f}s data_keys={list(data.keys())[:6]}"
                    )
                if ev in ("done", "error", "final"):
                    break
                if ev == "stream_end":
                    if data.get("resuming") and not buf and not spoke_any:
                        continue  # end of a PREVIOUS resumed stream
                    break
                delta = None
                if (
                    "text" in data
                    and data.get("type") not in ("stt", "listen")
                    and ev not in ("reasoning_delta", "thinking", "ready")
                ):
                    delta = str(data["text"])
                elif "parts" in data:
                    delta = str(data["parts"])
                if delta is None:
                    continue
                if _t_first is None:
                    _t_first = time.time()
                    logger.info(
                        f"[{self.stream_name}] ⏱ nanobot first token +{_t_first-_t0:.1f}s"
                    )
                buf += delta
                got = buf and (
                    re.search(r"[.!?…](\s|$)", buf) or "\n" in buf or len(buf) > 400
                )
                if got or len(buf) > 900:
                    flush(final=False)
        finally:
            flush(final=True)
            await q.put(None)
            logger.info(
                f"[{self.stream_name}] ⏱ nanobot done: sentences={_n_sent} "
                f"leftover={buf[:80]!r}"
            )
        return _n_sent

    async def _prefetch_next(self, q: asyncio.Queue):
        """Wait for the NEXT sentence and synthesise it, concurrently with the
        playback of the current one.

        Returns (sentence, pcm), or None at the end-of-stream sentinel.

        This exists because playback used to be gated on the NEXT sentence
        arriving. Measured 05.10.2026 with a producer that emits a sentence
        every 1.5 s (an L2 turn does exactly this): the reply played at
        t=0.00, the next sentence at t=1.50 — a 1.50 s hole — and the
        following one at t=1.50 as well. The gap equalled the LLM's think
        time, because `nxt = await q.get()` sat BEFORE `_speak_pcm`.

        Words that do not exist yet cannot be spoken, so part of that gap is
        irreducible — but the CURRENT sentence never has to wait for the next
        one, and the next one's synthesis can overlap this one's playback
        instead of starting after it. That is what this coroutine buys: the
        text is picked up the moment the model emits it and Edge-TTS runs
        while the speaker is still busy.
        """
        nxt = await q.get()
        if nxt is None:
            return None
        return nxt, await self._tts_fetch(nxt)

    async def _nanobot_player_task(self, q: asyncio.Queue) -> None:
        last_q = False
        # Per turn: a window opened by an EARLIER turn must not make this
        # turn's finally block believe it may leave the mic open forever.
        self._followup_open = False
        pending: asyncio.Task | None = None  # (sentence, pcm) for the NEXT turn
        sent: str | None = None
        pcm: bytes | None = None
        try:
            while True:
                if sent is None:
                    if pending is None:
                        pending = asyncio.create_task(self._prefetch_next(q))
                    got = await pending
                    pending = None
                    if got is None:
                        # End of stream: the sentinel, consumed where it is seen.
                        #
                        # It used to be read as the "next sentence" by a loop
                        # whose only end-of-stream test sat at the TOP, so the
                        # sentinel went into `nxt`, `sent = nxt` made sent None,
                        # and the next pass blocked on `q.get()` for a write
                        # that would never come. `_call_backend` then sat on
                        # `wait_for(player_task, 120.0)` and, because the
                        # utterance onset requires `not _processing_utterance`,
                        # the room was deaf for 120 s after every reply —
                        # measured 09:54:46 -> 09:56:45, and the reason «Да?»
                        # appears at all.
                        break
                    sent, pcm = got
                elif pcm is None:
                    pcm = await self._tts_fetch(sent)

                # Ask the queue for the following sentence BEFORE speaking this
                # one, so its synthesis overlaps this one's playback.
                if pending is None:
                    pending = asyncio.create_task(self._prefetch_next(q))

                try:
                    self._wake_expires = time.time() + self._wake_timeout
                    is_q = (
                        await asyncio.wait_for(
                            self._speak_pcm(pcm, sent), timeout=90.0
                        )
                        if pcm
                        else False
                    )
                    last_q = bool(is_q)
                    # The follow-up window opens when the reply actually
                    # FINISHES sounding, not when it was merely queued — a long
                    # reply played long past the old queue-time expiry.
                    #
                    # BOTH branches open it, and that is the whole point: a
                    # camera that closes the mic the instant it has answered
                    # cannot hear "а теперь выключи", so the user must repeat
                    # the wake word for every follow-up. main.py's
                    # `_finalize_turn_followup()` does exactly this for
                    # satellites (status LISTENING in both branches) and the
                    # windows are its numbers: 30 s after a question, 10 s
                    # after a statement.
                    window = (
                        self._dialogue_question_s
                        if is_q
                        else self._dialogue_statement_s
                    )
                    if window > 0:
                        await self._wait_playback_drain()
                        self._wake_detected = True
                        self._wake_expires = time.time() + window
                        self._followup_open = True
                        logger.info(
                            f"[{self.stream_name}] 💬 Follow-up open "
                            f"({'question' if is_q else 'answer'}) "
                            f"{window:.0f}s until {self._wake_expires:.1f}"
                        )
                except Exception as e:
                    logger.warning(f"[{self.stream_name}] TTS sentence failed: {e}")
                    break
                # Consumed: the next pass must take its sentence from `pending`.
                sent, pcm = None, None
        finally:
            if pending is not None:
                pending.cancel()
            # Close the window only when we did NOT open one. Deciding from
            # `last_q` alone is what made the camera deaf after every answer: a
            # statement set last_q False and the window was torn down even
            # though the utterance it belonged to had just been dispatched.
            if not self._followup_open:
                self._back_to_wake()

    def _encode_wav(self, pcm_16k: bytes) -> bytes:
        """Prepend a 44-byte canonical WAV header to raw s16le 16 kHz mono PCM.

        Hand-rolled (no `wave` module) because this runs in a worker thread
        and the format is fixed — header + payload, nothing else.
        """
        sample_rate = 16000
        bits = 16
        channels = 1
        data_size = len(pcm_16k)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            channels,
            sample_rate,
            sample_rate * channels * bits // 8,
            channels * bits // 8,
            bits,
            b"data",
            data_size,
        )
        return header + pcm_16k

    async def _fetch_transcription(
        self,
        wav: bytes,
        temperature: str | None = None,
        rms_raw: float = 0.0,
    ) -> str:
        for temp in (temperature, "0.0", "0.5"):
            if not temp:
                continue
            form = aiohttp.FormData()
            form.add_field("file", wav, filename="a.wav")
            form.add_field("model", self.whisper_model)
            form.add_field("language", "ru")
            form.add_field("temperature", temp)
            # Domain prompt ONLY for loud captures: it recovers phonemes
            # destroyed by ADC clipping ('включи кофеварку' case), but on
            # quiet ambient noise it actively HALLUCINATES command words
            # straight from the prompt itself ('чайник телевизор' at
            # raw_rms=801 -> pip -> nanobot).
            if rms_raw >= 2000.0:
                form.add_field(
                    "prompt",
                    "Команда умному дому: включи выключи кофеварку стиралку "
                    "свет чайник телевизор музыку пожалуйста",
                )
            try:
                async with self.http_session.post(
                    self.whisper_url,
                    data=form,
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
        """Identify the speaker: WAV -> Opus/Ogg -> SpeakerID POST.

        ffmpeg re-encodes to Ogg/Opus in a pipe (the service rejects raw
        WAV), then the multipart POST returns user_id+confidence. Returns
        "unknown" on any failure or when confidence <= 0.1 — speaker ID
        only annotates logs and must never gate or fail the reply.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-c:a",
                "libopus",
                "-f",
                "ogg",
                "pipe:1",
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
                        logger.info(
                            f"[{self.stream_name}] 👤 [SpeakerID] Recognized: {uid} ({conf:.2f})"
                        )
                        return uid
                    logger.debug(
                        f"[{self.stream_name}] 👤 [SpeakerID] Rejected: {uid} ({conf:.2f})"
                    )
        except Exception as e:
            logger.warning(f"[{self.stream_name}] ⚠️ [SpeakerID] Request failed: {e}")
        return "unknown"

    def _back_to_wake(self):
        self._wake_detected = False
        _arbiter_clear_owner(self.stream_name)
        # No threshold bump on empty wake windows: the old +0.07 "false-fire
        # penalty" compounded after every missed/empty command (Whisper glitch,
        # user silence) and eventually locked wake detection out entirely.
        # Thresholds now come from measured score distributions and stay put.

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
        if now - self._last_attention < 15.0:
            return
        self._last_attention = now
        logger.info(f"[{self.stream_name}] 🔔 attention pip ({reason})")
        try:
            sr = TTS_PLAY_RATE
            duration = 0.4
            n = int(sr * duration)
            t = np.arange(n) / sr
            # Clear two-tone "di-ding" so the wake is unmistakable even in a
            # noisy corridor (a single short 880Hz blip was easy to miss).
            tone = (
                np.where(
                    t < 0.2,
                    np.sin(2 * np.pi * 988 * t),
                    np.sin(2 * np.pi * 1319 * (t - 0.2)),
                )
                * 0.5
            )
            fade_n = int(sr * 0.03)
            env = np.ones(n)
            env[:fade_n] = np.linspace(0, 1, fade_n)
            env[-fade_n:] = np.linspace(1, 0, fade_n)
            mix = tone * env
            pcm = (mix * 32767).astype(np.int16).tobytes()
            # The pip's own echo returns through the speaker seconds later;
            # register it in the reference ring so _is_echo recognizes (and
            # suppresses wake on) the distorted return signal.
            self._store_tts_echo(pcm)
            if self._out_track:
                self._out_track._last_play_duration = duration
            if not await self._play_audio_http(pcm) and self._out_track:
                await self._out_track.queue_frame(pcm, sr)
            # The pip's own acoustic echo returns via the mic seconds later and
            # can re-trigger the decoder — but `_is_echo` already drops it by
            # CROSS-CORRELATION in `_feed_audio`, before the chunk ever reaches
            # vosk, and it is the pip's own pcm that was registered as the
            # reference. A flat 16 s wall-clock ban on top of that only
            # swallowed the user: measured 04.10.2026, «компьютер» spoken 8 s
            # and 11 s after the reply was HEARD and dropped twice
            # (`suppressed=1,2`), and only the third attempt 28 s later
            # answered. Keep a short tail for the direct return, and let the
            # correlation own the delayed one.
            self._wake_suppress_until = max(
                self._wake_suppress_until, time.time() + _ECHO_TAIL_S
            )
            # Only extend speaking_until if not already blocked longer by KWS
            delay = now + 0.35
            if delay > self._speaking_until:
                self._speaking_until = delay
        except Exception as exc:
            logger.warning(f"[{self.stream_name}] attention error: {exc}")

    async def _play_activation_sound(self):
        """Queue the short "I'm listening" cue on the camera speaker.

        The wav is normalised to 48 kHz mono s16le (the track's contract,
        TTS_PLAY_RATE) and its duration is registered so the echo guard stays
        closed for exactly as long as the cue sounds.
        """
        try:
            seg = AudioSegment.from_file(self._activation_wav_path)
            seg = seg.set_frame_rate(TTS_PLAY_RATE).set_channels(1).set_sample_width(2)
            pcm = seg.raw_data
            if self._out_track:
                self._out_track._last_play_duration = len(pcm) / (2 * TTS_PLAY_RATE)
            if not await self._play_audio_http(pcm) and self._out_track:
                await self._out_track.queue_frame(pcm, TTS_PLAY_RATE)
        except Exception:
            pass

    async def _wait_playback_drain(self) -> None:
        """Block until the WebRTC output queue has actually finished playing.

        Used before opening the dialogue follow-up window: the window must
        start when the question STOPS sounding, not when it was queued.
        """
        ot = self._out_track
        if not ot:
            return
        while not self._stopped.is_set():
            if ot.queue_seconds() < 0.1 and time.time() >= self._speaking_until - 1.6:
                return
            await asyncio.sleep(0.2)

    async def _tts_fetch(self, text: str) -> bytes | None:
        """Synthesize one sentence to PCM (48kHz mono s16le, the camera
        speaker's native rate — see TTS_PLAY_RATE). Split out of _speak so the
        player can prefetch the NEXT sentence while the current one is still
        playing — without that overlap the playback queue runs dry between
        sentences and the speaker stutters with silence gaps."""
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
                    return None
                mp3 = await r.read()
        except Exception:
            return None
        try:
            seg = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
            sr = TTS_PLAY_RATE
            seg = seg.set_frame_rate(sr).set_channels(1).set_sample_width(2)
            pcm = seg.raw_data
            pcm_arr = np.frombuffer(pcm, dtype=np.int16)
            peak = float(np.max(np.abs(pcm_arr))) if len(pcm_arr) else 0.0
            target = self._tts_target_peak
            gain = min(target / peak, 4.0) if peak > 0 else 1.0
            # Apply the gain in BOTH directions. The condition used to be
            # `gain > 1.2`, which can only ever BOOST: set the target below
            # the TTS's natural peak and the gain drops to e.g. 0.70, the
            # branch does not run, and the lever does NOTHING — which reads
            # from the field as "lowering the level did not fix the crackle"
            # rather than "the knob is broken". Half a percent of nothing is
            # the worst possible outcome for a diagnostic control.
            #
            # The dead band is 0.85..1.2, so behaviour at the default target is
            # bit-identical to before: measured 05.10.2026 peak 17091,
            # gain 1.17 => untouched, the TTS already sat at 52 % of full scale.
            applied = peak > 0 and (gain > 1.2 or gain < 0.85)
            if applied:
                pcm_arr = np.clip(pcm_arr * gain, -32768, 32767).astype(np.int16)
                pcm = pcm_arr.tobytes()
            # Log the numbers, not the decision: whether the crackle is
            # overdrive has to be settled by ear, so the log's job is to say
            # what was ACTUALLY sent.
            #
            # INFO, not DEBUG, and this used to be DEBUG — which made the whole
            # lever unreadable in the field. The logger runs at INFO (see the
            # `logger.setLevel(logging.INFO)` at the top of this module), so a
            # DEBUG line is not emitted at all. The entire point of an ENV knob
            # is that it is tuned from what the log actually prints; hiding its
            # numbers behind a level nobody enables is the same bug as shipping
            # the knob with no output. One line per synthesised sentence, which
            # is a handful per voice command.
            logger.info(
                f"[{self.stream_name}] tts pcm peak={peak:.0f} target={target} "
                f"gain={gain:.2f} applied={applied} "
                f"out_peak={int(np.max(np.abs(np.frombuffer(pcm, dtype=np.int16)))) if len(pcm) else 0}"
            )
            return pcm
        except Exception as e:
            logger.warning(f"[{self.stream_name}] TTS decode failed: {e}")
            return None

    async def _speak(self, text: str, reply: str = "") -> bool:
        """Synthesise `text` and play it; False when TTS produced nothing."""
        pcm = await self._tts_fetch(text)
        if not pcm:
            return False
        return await self._speak_pcm(pcm, reply)

    async def _play_audio_http(self, pcm: bytes) -> bool:
        """POST raw PCM to the camera's OpenIPC /play_audio endpoint.

        The rate travels in the Content-Type (`;rate=48000`), which is how
        majestic knows how to feed its 48 kHz speaker. Returns False when the
        endpoint is not configured or the POST failed, so the caller can fall
        back to the go2rtc backchannel.
        """
        if not self._play_audio_url:
            return False
        headers = dict(self._play_audio_headers)
        headers["Content-Type"] = f"application/octet-stream;rate={TTS_PLAY_RATE}"
        try:
            async with self.http_session.post(
                self._play_audio_url,
                data=pcm,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                if r.status == 200:
                    return True
                logger.warning(
                    f"[{self.stream_name}] play_audio HTTP {r.status}"
                )
                return False
        except Exception as e:
            logger.warning(f"[{self.stream_name}] play_audio failed: {e}")
            return False

    async def _speak_pcm(self, pcm: bytes, reply: str = "") -> bool:
        sr = TTS_PLAY_RATE
        is_question = bool(_HAS_QUESTION_RE.search((reply or "").strip()))
        self._speaking = True
        # Remember what we said so we can drop the echoed transcript later.
        self._last_tts_reply = reply.lower().strip().strip(".,!? -") if reply else ""

        # Block mic immediately: TTS request takes seconds over network,
        # and queued frames may already be playing out of the camera speaker.
        if self._out_track:
            self._speaking_until = time.time() + self._out_track.queue_seconds() + 3.0
        try:
            # Record the audio we are about to play so the mic feed can later
            # be checked for our own echo (cross-correlation in _is_echo).
            self._store_tts_echo(pcm)

            audio_dur = len(pcm) / (sr * 2)
            echo_tail = 1.5 if is_question else 3.0
            played = False

            if self._play_audio_url:
                # Preferred path: straight to the speaker at the right rate.
                played = await self._play_audio_http(pcm)
                if not played:
                    logger.warning(
                        f"[{self.stream_name}] falling back to the go2rtc "
                        f"backchannel — expect a ~6x pitch shift"
                    )

            if not played:
                # Fallback: the WebRTC sendonly track -> go2rtc -> ONVIF
                # backchannel. Timed as before.
                self._tts_play_end = time.time() + audio_dur
                if self._out_track:
                    self._out_track._last_play_duration = audio_dur
                chunk_size = sr * 20 // 1000 * 2
                for i in range(0, len(pcm), chunk_size):
                    c = pcm[i : i + chunk_size]
                    if len(c) < chunk_size:
                        c += b"\x00" * (chunk_size - len(c))
                    if self._out_track:
                        # We just push chunks asynchronously now; the pacing
                        # is enforced inside AIVoiceOutputTrack.recv().
                        await self._out_track.queue_frame(c, sr)
            else:
                # /play_audio returns as soon as the body is uploaded and the
                # camera then plays it in real time, so the playback window
                # starts AFTER the POST, not before it.
                self._tts_play_end = time.time() + audio_dur
                if self._out_track:
                    self._out_track._last_play_duration = audio_dur

            # Suppress wake detection AND inbound commands only for the direct
            # playback tail. The delayed TTS echo (returns via the mic 8-15s
            # later) is now caught precisely by cross-correlation in _is_echo,
            # so we no longer need a long blind lock that would also swallow
            # the user's answer to a question.
            ECHO_TAIL = echo_tail
            if self._out_track:
                self._speaking_until = time.time() + audio_dur + ECHO_TAIL
            global GLOBAL_TTS_UNTIL
            GLOBAL_TTS_UNTIL = time.time() + audio_dur + ECHO_TAIL
            # Suppress the decoder for the playback plus a SHORT tail. This used to be
            # `audio_dur + 15.0`, which is the direct cause of "after the first
            # trigger the word stops working": a reply finished at T blocked
            # every «компьютер» until T+18, and the user's next two attempts
            # were decoded and thrown away. The delayed room return is not this
            # timer's job — `_is_echo` drops it by cross-correlation in
            # `_feed_audio` (and extends suppression by up to 20 s when, and
            # only when, a correlation actually matches).
            self._wake_suppress_until = time.time() + audio_dur + _ECHO_TAIL_S
        except Exception as e:
            logger.warning(f"[{self.stream_name}] playback error: {e}")

        self._speaking = False

        return is_question
